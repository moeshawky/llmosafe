//! Python bindings for llmosafe using PyO3.
//!
//! Docstrings here appear in help(), IDE tooltips, and PyPI docs.
//! They must be self-contained — no reading Rust source required.

use pyo3::prelude::*;
use pyo3::create_exception;

// ── Custom exceptions ──────────────────────────────────────────

create_exception!(_llmosafe, LLMOSafeError, pyo3::exceptions::PyException,
    "Base exception for all llmosafe errors. All specific exceptions inherit from this."
);
create_exception!(_llmosafe, ResourceExhaustedError, LLMOSafeError,
    "RSS memory has reached or exceeded the configured ceiling.\n\nThis is an enforcement-grade signal — you MUST stop processing.\n\nNote: This monitors RSS memory only, not filesystem capacity.\nRSS pressure often precedes disk exhaustion because processes\nbuffering large writes consume RAM before flushing to disk."
);
create_exception!(_llmosafe, CognitiveInstabilityError, LLMOSafeError,
    "Cognitive entropy has exceeded the stability threshold (PRESSURE_THRESHOLD = 40000).\n\nThis is an enforcement-grade signal — system state is too chaotic\nto continue safely. Entropy range is [0, 65535]."
);
create_exception!(_llmosafe, BiasHaloDetectedError, LLMOSafeError,
    "Bias manipulation patterns detected in input text.\n\nThis is an enforcement-grade signal — the input may be attempting\nto manipulate the system into ignoring safety limits."
);

// ── Imports ────────────────────────────────────────────────────

use ::llmosafe::llmosafe_body::ResourceGuard;
use ::llmosafe::llmosafe_kernel::{KernelError, Synapse};
use ::llmosafe::llmosafe_sifter::sift_text;
use ::llmosafe::llmosafe_body::llmosafe_get_environmental_entropy;
use ::llmosafe::llmosafe_memory::cognitive_memory::process_state_update;

// ── Bias Detection ─────────────────────────────────────────────

/// Calculate the bias entropy score for text via dual-path analysis.
///
/// Routes text through the full dual-path sifter: classifier (adaptive
/// layer, trained on 42K samples) + keyword-bias (innate backstop layer).
/// Returns the combined entropy in `[0, 65535]` — the greater of the
/// two layers' scores.
///
/// **Dual-path architecture:**
/// - **Classifier layer** (adaptive): TF-IDF logistic regression with
///   93.4% accuracy. Detects learned manipulation patterns.
/// - **Keyword layer** (innate): 8 bias categories (authority, social
///   proof, scarcity, urgency, emotional appeal, expertise signaling,
///   semantic traps, template fitting). Negation-aware.
///
/// Either layer can elevate the score. The output is `max(classifier_score,
/// keyword_score)` — no double-counting.
///
/// Args:
///     text: Input text to scan for manipulation patterns.
///
/// Returns:
///     Combined entropy score `[0, 65535]`. 0 = safe, higher = manipulation
///     probability. The classifier sigmoid maps to probability — 32768 ≈ p=0.5
///     (maximum uncertainty).
///
/// Example:
///     >>> calculate_halo("The expert provides an official recommendation")
///     200
///     >>> calculate_halo("A normal sentence with no manipulation")
///     0
#[pyfunction]
fn calculate_halo(text: &str) -> u16 {
    let (sifted, _proof) = sift_text(text);
    sifted.raw_entropy()
}

// ── Resource Management ────────────────────────────────────────

/// Check if current RSS memory is within the specified ceiling.
///
/// **This is an enforcement-grade API** — it raises an exception when
/// the ceiling is breached. You MUST stop processing when this fires.
///
/// **Monitors RSS memory only — does not inspect filesystem capacity.**
/// RSS pressure often precedes disk exhaustion because processes
/// buffering large writes consume RAM before flushing to disk.
/// For direct disk checks, use shutil.disk_usage().
///
/// Args:
///     ceiling_mb: Maximum allowed RSS memory in **megabytes**.
///
/// Returns:
///     0 if OK.
///
/// Raises:
///     ResourceExhaustedError: If RSS ≥ ceiling, or if ceiling_mb == 0.
///     LLMOSafeError: For other internal errors.
///
/// Example:
///     >>> check_resources(1024)  # 1 GB ceiling
///     0
///     >>> check_resources(0)     # zero ceiling → always fails
///     ResourceExhaustedError: Memory ceiling exceeded
#[pyfunction]
fn check_resources(ceiling_mb: u32) -> PyResult<i32> {
    let ceiling_bytes = (ceiling_mb as usize).saturating_mul(1024 * 1024);
    let guard = ResourceGuard::new(ceiling_bytes);
    match guard.check() {
        Ok(_) => Ok(0),
        Err(KernelError::ResourceExhaustion) => {
            Err(ResourceExhaustedError::new_err("Memory ceiling exceeded"))
        }
        Err(e) => Err(LLMOSafeError::new_err(e.to_string())),
    }
}

/// Get current RSS memory as a percentage of the ceiling (0–100).
///
/// **Platform-portable**: works on Linux and Windows.
/// Returns 100 if ceiling_mb == 0.
///
/// **This reflects RSS memory pressure only, not disk usage.**
///
/// Args:
///     ceiling_mb: Memory ceiling in megabytes.
///
/// Returns:
///     Pressure percentage (0–100). 0 = no pressure, 100 = at ceiling.
///
/// Example:
///     >>> get_resource_pressure(1024)  # 1 GB ceiling
///     3  # process using ~30 MB
#[pyfunction]
fn get_resource_pressure(ceiling_mb: u32) -> u8 {
    let ceiling_bytes = (ceiling_mb as usize).saturating_mul(1024 * 1024);
    if ceiling_bytes == 0 {
        return 100;
    }
    let guard = ResourceGuard::new(ceiling_bytes);
    guard.pressure()
}

// ── Stability and Pipeline ─────────────────────────────────────

/// Check if a cognitive state (synapse) is stable.
///
/// The synapse_bits parameter encodes cognitive state in a 64-bit integer.
/// For most usage, only the low 16 bits (raw_entropy, 0–1000) matter.
/// Use make_synapse() to construct values.
///
/// **Bit layout** (lower 64 bits):
///
///     Bits [0:15]  → raw_entropy   (u16, operational range 0–1000)
///     Bits [16:31] → raw_surprise  (u16, 0–65535)
///     Bit  [32]    → has_bias      (0 or 1)
///     Bits [33:44] → position      (u12)
///     Bits [45:60] → timestamp     (u16)
///     Bits [61:68] → cascade_depth (u8)
///
/// **Return codes**:
///
///     0  = stable
///     -1 = DepthExceeded (runaway recursion)
///     -2 = CognitiveInstability (entropy > 1000)
///     -3 = BiasHaloDetected (has_bias bit set)
///     -4 = HallucinationDetected (surprise > threshold)
///     -5 = ResourceExhaustion
///     -6 = SelfMemoryExceeded
///     -7 = DeadlineExceeded
///
/// Args:
///     synapse_bits: 64-bit encoded cognitive state.
///
/// Returns:
///     0 if stable, negative error code otherwise.
///
/// Example:
///     >>> get_stability(400)    # entropy=400, stable
///     0
///     >>> get_stability(1100)   # entropy=1100, unstable
///     -2
#[pyfunction]
fn get_stability(synapse_bits: u64) -> i32 {
    let synapse = Synapse::from_raw_u64(synapse_bits);
    match synapse.validate() {
        Ok(()) => 0,
        Err(KernelError::CognitiveInstability) => -2,
        Err(KernelError::BiasHaloDetected) => -3,
        Err(KernelError::DepthExceeded) => -1,
        Err(KernelError::HallucinationDetected) => -4,
        Err(KernelError::ResourceExhaustion) => -5,
        Err(KernelError::SelfMemoryExceeded) => -6,
        Err(KernelError::DeadlineExceeded) => -7,
    }
}

/// Get the current system CPU load percentage (0–100).
///
/// Uses delta measurement over a 100ms window on Linux.
/// Returns 0 on other platforms.
///
/// ⚠ BLOCKING: Reads /proc/stat twice with a 100ms sleep on Linux.
/// Do NOT call in async hot paths — offload to a thread.
///
/// Returns:
///     CPU load percentage (0–100).
#[pyfunction]
fn get_system_cpu_load() -> u8 {
    ResourceGuard::system_cpu_load()
}

/// Get the environmental entropy score (0–1000).
///
/// This is a **predictive signal** (advisory, not enforcement-grade).
/// Compose it into your own escalation policy.
///
/// **Weighting**:
///
/// | Component       | Weight | What It Measures                              |
/// |-----------------|--------|-----------------------------------------------|
/// | RSS memory      | 50%    | current_rss / ceiling (ceiling = 50% sys RAM) |
/// | IO wait         | 25%    | delta iowait / delta total CPU (100ms window) |
/// | CPU load avg    | 25%    | 1-min loadavg / 10.0, capped at 1.0           |
///
/// **Threshold recommendations**:
///
/// | Range   | Zone      | Action                            |
/// |---------|-----------|-----------------------------------|
/// | 0–400   | Normal    | Proceed                           |
/// | 400–600 | Elevated  | Log, continue                     |
/// | 600–800 | Pressure  | Throttle inputs, IO likely stressed|
/// | 800–1000| Critical  | Halt new work                     |
///
/// **IO wait is your early warning for disk pressure** — it rises
/// before disk fills because the kernel blocks on writes when the
/// IO subsystem saturates.
///
/// ⚠ BLOCKING: Same blocking behavior as get_system_cpu_load().
/// On non-Linux platforms, reflects RSS only (max ~500/1000).
///
/// Returns:
///     Environmental entropy score (0–1000).
#[pyfunction]
fn get_environmental_entropy() -> u16 {
    llmosafe_get_environmental_entropy()
}

/// Process a cognitive state update through the safety pipeline.
///
/// Pipeline: surprise gating → entropy check (via the global WorkingMemory,
/// 64-entry ring buffer, surprise threshold 500). Does NOT run the Rust-side
/// sifter — call `calculate_halo()` to detect manipulation patterns
/// before constructing the synapse.
///
/// **Bit layout** for synapse_bits — same as get_stability():
///
///     Bits [0:15]  → raw_entropy   (u16, 0–1000)
///     Bits [16:31] → raw_surprise  (u16, 0–65535)
///     Bit  [32]    → has_bias      (0 or 1)
///
/// **Return codes** — same as get_stability():
///
///     0  = success
///     -1 = DepthExceeded
///     -2 = CognitiveInstability (entropy > 1000)
///     -3 = BiasHaloDetected
///     -4 = HallucinationDetected (surprise > 500)
///     -5 = ResourceExhaustion
///     -6 = SelfMemoryExceeded
///     -7 = DeadlineExceeded
///
/// Args:
///     synapse_bits: 64-bit encoded cognitive state.
///
/// Returns:
///     0 on success, negative error code on failure.
#[pyfunction]
fn process_synapse(synapse_bits: u64) -> i32 {
    process_state_update(synapse_bits.into())
}

// ── Module registration ────────────────────────────────────────

/// llmosafe — Predictive resource-pressure instrumentation and runtime guardrails.
///
/// Signal classes:
///
///   DIRECT GUARANTEES (enforcement-grade — raise exceptions):
///     check_resources()       → RSS memory ceiling
///     get_stability()         → cognitive entropy threshold
///
///   PREDICTIVE SIGNALS (advisory — compose into your policy):
///     get_environmental_entropy() → weighted composite (RSS 50%, IO 25%, CPU 25%)
///     get_resource_pressure()     → RSS as % of ceiling
///     get_system_cpu_load()       → CPU load %
///
///   BEHAVIORAL SIGNALS:
///     calculate_halo()       → manipulation pattern detection in text
///
/// For disk exhaustion protection, compose llmosafe signals with
/// shutil.disk_usage(). See README for the canonical cookbook.
///
/// Example:
///     >>> from llmosafe import check_resources, get_environmental_entropy
///     >>> check_resources(1024)  # 1GB RSS ceiling
///     0
///     >>> get_environmental_entropy()  # 0-1000, IO wait is key for disk
///     15
#[pymodule]
fn _llmosafe(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_halo, m)?)?;
    m.add_function(wrap_pyfunction!(check_resources, m)?)?;
    m.add_function(wrap_pyfunction!(get_resource_pressure, m)?)?;
    m.add_function(wrap_pyfunction!(get_stability, m)?)?;
    m.add_function(wrap_pyfunction!(get_system_cpu_load, m)?)?;
    m.add_function(wrap_pyfunction!(get_environmental_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(process_synapse, m)?)?;
    m.add("LLMOSafeError", _py.get_type::<LLMOSafeError>())?;
    m.add("ResourceExhaustedError", _py.get_type::<ResourceExhaustedError>())?;
    m.add("CognitiveInstabilityError", _py.get_type::<CognitiveInstabilityError>())?;
    m.add("BiasHaloDetectedError", _py.get_type::<BiasHaloDetectedError>())?;
    Ok(())
}
