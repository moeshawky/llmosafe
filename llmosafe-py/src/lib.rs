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
use ::llmosafe::llmosafe_memory::cognitive_memory::{process_state_update, get_memory_stats};
use ::llmosafe::c_abi::{
    llmosafe_create, llmosafe_sift_and_process,
    llmosafe_get_classifier_score, llmosafe_get_decision, llmosafe_get_pid_state,
    llmosafe_get_memory_stats as c_get_memory_stats,
    llmosafe_get_kernel_output, llmosafe_get_body_pressure, llmosafe_destroy,
    llmosafe_get_entropy, llmosafe_get_surprise, llmosafe_get_detection_flags,
    llmosafe_get_oov_ratio, llmosafe_get_stages_executed, llmosafe_get_step_count,
    llmosafe_process_with_pressure, llmosafe_reset_detectors, llmosafe_reset_full,
    llmosafe_configure,
};

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

/// Get working-memory statistics from the global ring buffer.
///
/// Returns a dict with four keys:
///
/// | Key          | Type  | Description                                      |
/// |--------------|-------|--------------------------------------------------|
/// | ``mean``     | float | Running mean entropy [0, 65535]                  |
/// | ``variance`` | float | Running variance of entropy                      |
/// | ``trend``    | float | Linear regression slope over the buffer window   |
/// | ``drifting`` | bool  | True when |trend| > 10.0                       |
///
/// Returns:
///     dict with keys ``mean``, ``variance``, ``trend``, ``drifting``.
#[pyfunction]
fn memory_stats(py: Python<'_>) -> PyResult<PyObject> {
    let (mean, variance, trend, is_drifting) = get_memory_stats();
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("mean", mean)?;
    dict.set_item("variance", variance)?;
    dict.set_item("trend", trend)?;
    dict.set_item("drifting", is_drifting)?;
    Ok(dict.into())
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

// ── PID State Introspection ─────────────────────────────────────

/// Read the live PID state from a CognitivePipeline instance.
///
/// Returns the dual-rate leaky integrators and step-change tracking
/// field from the PID controller. All values are `f32` clamped to
/// `[0, 1]`, returned as `float`.
///
/// Args:
///     instance_id: Pipeline handle returned by `llmosafe_create()` (0–15).
///
/// Returns:
///     A dict with keys `acute_entropy`, `chronic_entropy`,
///     `prev_pressure_norm`. All values are `float`.
///
/// Raises:
///     LLMOSafeError: If `instance_id` is invalid or the slot is
///     uninitialized.
///
/// Example:
///     >>> get_pid_state(0)
///     {'acute_entropy': 0.0, 'chronic_entropy': 0.0, 'prev_pressure_norm': 0.0}
#[pyfunction]
fn get_pid_state(instance_id: u32) -> PyResult<PyObject> {
    let mut acute: f64 = 0.0;
    let mut chronic: f64 = 0.0;
    let mut pressure: f64 = 0.0;
    let result = ::llmosafe::c_abi::llmosafe_get_pid_state(instance_id, &mut acute, &mut chronic, &mut pressure);
    if result != 0 {
        return Err(LLMOSafeError::new_err(format!(
            "instance {} not found",
            instance_id
        )));
    }
    Python::with_gil(|py| {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("acute_entropy", acute)?;
        dict.set_item("chronic_entropy", chronic)?;
        dict.set_item("prev_pressure_norm", pressure)?;
        Ok(dict.into())
    })
}

// ── Arena-based Classifier Score ───────────────────────────────

/// Get the raw classifier score from the most recent pipeline invocation.
///
/// Queries the arena slot identified by `instance_id` (the handle returned
/// by a prior `llmosafe_create` call).  Returns the raw classifier logit
/// score (f32→f64) from the last `sift_and_process` run.
///
/// Negative values indicate a safe-text classification; positive values
/// indicate manipulation signal.  The score is unbounded.
///
/// Returns -1.0 if the instance_id is invalid, the slot is uninitialized,
/// or no result is available yet.
///
/// Args:
///     instance_id: Pipeline handle (returned by llmosafe_create).
///
/// Returns:
///     Classifier logit score as float, or -1.0 on error.
#[pyfunction]
fn get_classifier_score(instance_id: u32) -> f64 {
    ::llmosafe::c_abi::llmosafe_get_classifier_score(instance_id)
}

// ── Kernel Output Introspection ─────────────────────────────────

/// Read kernel output fields from the last pipeline invocation.
///
/// Queries the arena slot identified by `instance_id`.  Returns a tuple
/// of `(error_kernel, is_stable, depth)` from the kernel output of the
/// most recent `sift_and_process` call.  `error_kernel` is the normalised
/// entropy error `[0.0, 1.0]` (setpoint=0, the raw entropy divided by
/// 65535).  `is_stable` is 1 when mean entropy was below the stability
/// threshold, 0 otherwise.  `depth` is the reasoning step count.
///
/// Args:
///     instance_id: Pipeline handle (returned by llmosafe_create).
///
/// Returns:
///     `(error_kernel: float, is_stable: int, depth: int)`.
///
/// Raises:
///     LLMOSafeError: If `instance_id` is invalid or no kernel output
///     is available (pipeline halted before kernel stage).
#[pyfunction]
fn get_kernel_output(instance_id: u32) -> PyResult<(f32, u8, u32)> {
    let mut error: f32 = 0.0;
    let mut is_stable: u8 = 0;
    let mut depth: u32 = 0;
    let rc = ::llmosafe::c_abi::llmosafe_get_kernel_output(
        instance_id,
        &mut error,
        &mut is_stable,
        &mut depth,
    );
    if rc != 0 {
        return Err(LLMOSafeError::new_err(format!(
            "instance {} has no kernel output",
            instance_id
        )));
    }
    Ok((error, is_stable, depth))
}

/// Read the body pressure from the last pipeline invocation.
///
/// Queries the arena slot identified by `instance_id`.  Returns the
/// RSS memory pressure percentage [0, 100] from the most recent
/// `process_with_pressure` call.  Returns `u32::MAX` (4294967295) on
/// invalid handle.
///
/// Args:
///     instance_id: Pipeline handle (returned by llmosafe_create).
///
/// Returns:
///     Body pressure percentage [0, 100], or 4294967295 on error.
#[pyfunction]
fn get_body_pressure(instance_id: u32) -> u32 {
    ::llmosafe::c_abi::llmosafe_get_body_pressure(instance_id)
}

// ── Pipeline Result Queries ───────────────────────────────────

/// Read the safety decision from the last pipeline invocation.
///
/// Returns a dict with the decision name as a string:
/// "Proceed", "Warn", "Escalate", or "Halt".
///
/// Args:
///     instance_id: Pipeline handle returned by `llmosafe_create()` (0–15).
///
/// Returns:
///     dict with key ``decision`` mapping to a string.
///
/// Raises:
///     LLMOSafeError: If `instance_id` is invalid or no result is available.
#[pyfunction]
fn get_decision(instance_id: u32) -> PyResult<PyObject> {
    let code = llmosafe_get_decision(instance_id as usize);
    match code {
        -9 => Err(LLMOSafeError::new_err(format!(
            "instance {} not found or no result available",
            instance_id
        ))),
        _ => Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            let name: &str = match code {
                0 => "Proceed",
                1 => "Warn",
                2 => "Escalate",
                -8..=-1 => "Halt",
                _ => "Unknown",
            };
            dict.set_item("decision", name)?;
            Ok(dict.into())
        }),
    }
}

/// Read the entropy field from the last pipeline invocation.
///
/// Returns the raw entropy value [0, 65535] from the classified synapse.
///
/// Args:
///     instance_id: Pipeline handle (0–15).
///
/// Returns:
///     Entropy as u32 (0 if instance is invalid or no result available).
#[pyfunction]
fn get_entropy(instance_id: u32) -> u32 {
    llmosafe_get_entropy(instance_id) as u32
}

/// Read the surprise field from the last pipeline invocation.
///
/// Returns the raw surprise value [0, 65535] from the classified synapse.
///
/// Args:
///     instance_id: Pipeline handle (0–15).
///
/// Returns:
///     Surprise as u32 (0 if instance is invalid or no result available).
#[pyfunction]
fn get_surprise(instance_id: u32) -> u32 {
    llmosafe_get_surprise(instance_id) as u32
}

/// Read the detection flags from the last pipeline invocation.
///
/// Returns a bitmask of 5 detection flags (stuck, drifting, low-confidence,
/// decaying, anomaly/adversarial). Bits 0-4 correspond to individual
/// detector outputs.
///
/// Args:
///     instance_id: Pipeline handle (0–15).
///
/// Returns:
///     Detection flags bitmask as u32 (0 if instance is invalid or no result available).
#[pyfunction]
fn get_detection_flags(instance_id: u32) -> u32 {
    llmosafe_get_detection_flags(instance_id) as u32
}

/// Read the OOV (out-of-vocabulary) ratio from the last pipeline invocation.
///
/// Returns the OOV ratio [0, 255] where 0 = 0% OOV, 255 = 100% OOV.
/// High OOV combined with anomaly flags indicates distribution-shift.
///
/// Args:
///     instance_id: Pipeline handle (0–15).
///
/// Returns:
///     OOV ratio as u32 (0 if instance is invalid or no result available).
#[pyfunction]
fn get_oov_ratio(instance_id: u32) -> u32 {
    llmosafe_get_oov_ratio(instance_id) as u32
}

/// Read the stages_executed bitmask from the last pipeline invocation.
///
/// Returns a bitmask indicating which pipeline stages ran:
/// 0x01=SIFT, 0x02=MEMORY, 0x04=KERNEL, 0x08=DETECTION, 0x10=MONITOR.
///
/// Args:
///     instance_id: Pipeline handle (0–15).
///
/// Returns:
///     Stages executed bitmask as u32 (0 if instance is invalid or no result available).
#[pyfunction]
fn get_stages_executed(instance_id: u32) -> u32 {
    llmosafe_get_stages_executed(instance_id) as u32
}

/// Read the reasoning step count from the last pipeline invocation.
///
/// Returns the current step count of the reasoning loop.
///
/// Args:
///     instance_id: Pipeline handle (0–15).
///
/// Returns:
///     Step count as u32 (0 if instance is invalid or no result available).
#[pyfunction]
fn get_step_count(instance_id: u32) -> u32 {
    llmosafe_get_step_count(instance_id) as u32
}

/// Packs OOV ratio and detection flags from a synapse into a single u16.
///
/// **Layout**: `[OOV:8 bits][FLAGS:6 bits]`.  The upper 8 bits carry the
/// out-of-vocabulary ratio (0=0%, 255=100%).  The lower 6 bits carry
/// detection flags (stuck, drifting, low-confidence, decaying, anomaly,
/// adversarial).  Together they encode a 2D risk surface: high OOV +
/// anomaly flag = distribution-shift attack, high OOV + adversarial
/// flag = confirmed adversarial input.
///
/// Args:
///     synapse_bits: 64-bit encoded synapse.
///
/// Returns:
///     Combined risk bits (u16). Mask with `0b111111_11111111` for OOV,
///     mask with `0b111111` for detection flags.
#[pyfunction]
fn combined_risk_bits(synapse_bits: u64) -> u16 {
    let synapse = Synapse::from_raw_u64(synapse_bits);
    synapse.combined_risk_bits()
}

// ── CognitivePipeline pyclass ────────────────────────────────

/// Wraps a C-ABI arena-backed CognitivePipeline into a Python class.
///
/// Each instance holds an arena slot handle (0–15). The pipeline
/// runs the full 5-stage safety analysis (SIFT → MEMORY → KERNEL →
/// 6 detectors → PID → safety overrides) on every `process()` call.
///
/// Drop (or Python `del`) releases the arena slot for reuse.
///
/// Args:
///     dal_level: DesignAssuranceLevel as u8 (0=A, 1=B, 2=C, 3=D, 4=E).
///         Controls runtime escalation behavior. Default: 4 (E).
///     use_detection_gate: Whether to route decisions through the
///         detection-gate path instead of PID. Default: False.
///     memory_depth: Reserved for future use. WorkingMemory is
///         fixed at compile time (64 entries). Default: 10.
///
/// Example:
///     >>> pipeline = CognitivePipeline()
///     >>> result = pipeline.process("some input text")
///     >>> print(result["decision"], result["entropy"])
#[pyclass]
struct CognitivePipeline {
    instance_id: u32,
}

#[pymethods]
impl CognitivePipeline {
    #[new]
    fn new(
        dal_level: Option<u8>,
        use_detection_gate: Option<bool>,
        memory_depth: Option<usize>,
    ) -> PyResult<Self> {
        let objective = b"safety";
        let handle = llmosafe_create(objective.as_ptr(), objective.len());
        if handle == usize::MAX {
            return Err(LLMOSafeError::new_err(
                "Failed to create pipeline — arena full (max 16 instances)"
            ));
        }
        let instance_id = handle as u32;
        let dal = dal_level.unwrap_or(4);
        let gate = if use_detection_gate.unwrap_or(false) { 1u32 } else { 0u32 };
        let mem = memory_depth.unwrap_or(10) as u32;
        llmosafe_configure(instance_id, dal, gate, mem);
        Ok(Self { instance_id })
    }

    /// Process text through the full 5-stage pipeline.
    ///
    /// Returns a dict with all PipelineResult fields:
    /// decision, entropy, surprise, classifier_score, detection_flags,
    /// oov_ratio, stages_executed, step_count, body_pressure,
    /// kernel_error, kernel_is_stable, kernel_depth.
    fn process(&mut self, text: &str) -> PyResult<PyObject> {
        let code = llmosafe_sift_and_process(self.instance_id as usize, text.as_ptr(), text.len());
        if code == -9 {
            return Err(LLMOSafeError::new_err("Invalid pipeline instance"));
        }
        self.build_result_dict(code)
    }

    /// Process text with body pressure fed into the pre-SIFT gate.
    ///
    /// Args:
    ///     text: Input text to analyze.
    ///     entropy: Body entropy [0, 1000].
    ///     pressure: RSS memory pressure [0, 100].
    fn process_with_pressure(&mut self, text: &str, entropy: u16, pressure: u8) -> PyResult<PyObject> {
        let code = llmosafe_process_with_pressure(
            self.instance_id as usize,
            text.as_ptr(),
            text.len(),
            entropy,
            pressure,
        );
        if code == -9 {
            return Err(LLMOSafeError::new_err("Invalid pipeline instance"));
        }
        self.build_result_dict(code)
    }

    /// Process text with resource-gated safe path selection.
    ///
    /// Checks resource availability with a 5-second deadline before
    /// choosing the processing path:
    ///
    /// - **Normal path**: If resources are available, calls `process()`.
    /// - **Pressure path**: If the deadline is exceeded, calls
    ///   `process_with_pressure()` using current environmental entropy
    ///   and RSS memory pressure.
    ///
    /// Args:
    ///     ceiling_mb: RSS memory ceiling in megabytes.
    ///     text: Input text to analyze.
    ///
    /// Returns:
    ///     Result dict (same format as `process()`).
    ///
    /// Raises:
    ///     LLMOSafeError: If `ceiling_mb == 0`, resource exhaustion
    ///     detected immediately, or pipeline instance is invalid.
    fn process_safe(&mut self, _py: Python<'_>, text: &str, ceiling_mb: u64) -> PyResult<PyObject> {
        let ceiling_bytes = (ceiling_mb as usize).saturating_mul(1024 * 1024);
        let guard = ResourceGuard::new(ceiling_bytes);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);

        match guard.check_with_deadline(deadline) {
            Ok(_) => {
                let code = llmosafe_sift_and_process(
                    self.instance_id as usize,
                    text.as_ptr(),
                    text.len(),
                );
                if code == -9 {
                    return Err(LLMOSafeError::new_err("Invalid pipeline instance"));
                }
                self.build_result_dict(code)
            }
            Err(KernelError::DeadlineExceeded | KernelError::ResourceExhaustion) => {
                let entropy = guard.raw_entropy();
                let pressure = guard.pressure();
                let code = llmosafe_process_with_pressure(
                    self.instance_id as usize,
                    text.as_ptr(),
                    text.len(),
                    entropy,
                    pressure,
                );
                if code == -9 {
                    return Err(LLMOSafeError::new_err("Invalid pipeline instance"));
                }
                self.build_result_dict(code)
            }
            Err(e) => Err(LLMOSafeError::new_err(format!(
                "Resource check error: {}",
                e
            ))),
        }
    }

    /// Reset PID state and working memory (full reset).
    fn reset(&mut self) -> PyResult<()> {
        let rc = llmosafe_reset_full(self.instance_id as usize);
        if rc != 0 {
            return Err(LLMOSafeError::new_err("Invalid pipeline instance"));
        }
        Ok(())
    }

    /// Reset only the detectors (not PID/memory).
    fn reset_detectors(&mut self) -> PyResult<()> {
        let rc = llmosafe_reset_detectors(self.instance_id as usize);
        if rc != 0 {
            return Err(LLMOSafeError::new_err("Invalid pipeline instance"));
        }
        Ok(())
    }

    /// Get the raw classifier logit score from the last process().
    fn get_classifier_score(&self) -> f64 {
        llmosafe_get_classifier_score(self.instance_id)
    }

    /// Get live PID state from the pipeline.
    ///
    /// Returns dict with acute_entropy, chronic_entropy, prev_pressure_norm.
    fn get_pid_state(&self) -> PyResult<PyObject> {
        let mut acute: f64 = 0.0;
        let mut chronic: f64 = 0.0;
        let mut pressure: f64 = 0.0;
        let rc = llmosafe_get_pid_state(self.instance_id, &mut acute, &mut chronic, &mut pressure);
        if rc != 0 {
            return Err(LLMOSafeError::new_err(format!(
                "instance {} not found", self.instance_id
            )));
        }
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("acute_entropy", acute)?;
            dict.set_item("chronic_entropy", chronic)?;
            dict.set_item("prev_pressure_norm", pressure)?;
            Ok(dict.into())
        })
    }

    /// Get working-memory statistics from the pipeline.
    ///
    /// Returns dict with mean, variance, trend, drifting.
    fn get_memory_stats(&self) -> PyResult<PyObject> {
        let mut mean: f64 = 0.0;
        let mut variance: f64 = 0.0;
        let mut trend: f64 = 0.0;
        let mut drifting: u32 = 0;
        let rc = c_get_memory_stats(self.instance_id, &mut mean, &mut variance, &mut trend, &mut drifting);
        if rc != 0 {
            return Err(LLMOSafeError::new_err(format!(
                "instance {} not found", self.instance_id
            )));
        }
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("mean", mean)?;
            dict.set_item("variance", variance)?;
            dict.set_item("trend", trend)?;
            dict.set_item("drifting", drifting != 0)?;
            Ok(dict.into())
        })
    }

    /// Get kernel output from the last pipeline invocation.
    ///
    /// Returns tuple (error_kernel: float, is_stable: int, depth: int).
    fn get_kernel_output(&self) -> PyResult<(f32, u8, u32)> {
        let mut error: f32 = 0.0;
        let mut is_stable: u8 = 0;
        let mut depth: u32 = 0;
        let rc = llmosafe_get_kernel_output(self.instance_id, &mut error, &mut is_stable, &mut depth);
        if rc != 0 {
            return Err(LLMOSafeError::new_err(format!(
                "instance {} has no kernel output", self.instance_id
            )));
        }
        Ok((error, is_stable, depth))
    }

    /// Get the body pressure from the last process_with_pressure() call.
    fn get_body_pressure(&self) -> u32 {
        llmosafe_get_body_pressure(self.instance_id)
    }
}

impl CognitivePipeline {
    fn build_result_dict(&self, code: i32) -> PyResult<PyObject> {
        let entropy = llmosafe_get_entropy(self.instance_id);
        let surprise = llmosafe_get_surprise(self.instance_id);
        let detection_flags = llmosafe_get_detection_flags(self.instance_id);
        let oov_ratio = llmosafe_get_oov_ratio(self.instance_id);
        let stages_executed = llmosafe_get_stages_executed(self.instance_id);
        let step_count = llmosafe_get_step_count(self.instance_id);
        let classifier_score = llmosafe_get_classifier_score(self.instance_id);
        let body_pressure = llmosafe_get_body_pressure(self.instance_id);

        let (kernel_error, kernel_is_stable, kernel_depth) = {
            let mut error: f32 = 0.0;
            let mut is_stable: u8 = 0;
            let mut depth: u32 = 0;
            let rc = llmosafe_get_kernel_output(self.instance_id, &mut error, &mut is_stable, &mut depth);
            if rc == 0 {
                (Some(error), Some(is_stable), Some(depth))
            } else {
                (None, None, None)
            }
        };

        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("decision", code)?;
            dict.set_item("entropy", entropy)?;
            dict.set_item("surprise", surprise)?;
            dict.set_item("detection_flags", detection_flags)?;
            dict.set_item("oov_ratio", oov_ratio)?;
            dict.set_item("stages_executed", stages_executed)?;
            dict.set_item("step_count", step_count)?;
            dict.set_item("classifier_score", classifier_score)?;
            dict.set_item("body_pressure", body_pressure)?;
            if let (Some(e), Some(s), Some(d)) = (kernel_error, kernel_is_stable, kernel_depth) {
                dict.set_item("kernel_error", e)?;
                dict.set_item("kernel_is_stable", s != 0)?;
                dict.set_item("kernel_depth", d)?;
            }
            Ok(dict.into())
        })
    }
}

impl Drop for CognitivePipeline {
    fn drop(&mut self) {
        llmosafe_destroy(self.instance_id as usize);
    }
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
    m.add_function(wrap_pyfunction!(memory_stats, m)?)?;
    m.add_function(wrap_pyfunction!(get_classifier_score, m)?)?;
    m.add_function(wrap_pyfunction!(get_pid_state, m)?)?;
    m.add_function(wrap_pyfunction!(get_kernel_output, m)?)?;
    m.add_function(wrap_pyfunction!(get_body_pressure, m)?)?;
    m.add_function(wrap_pyfunction!(combined_risk_bits, m)?)?;
    m.add_function(wrap_pyfunction!(get_decision, m)?)?;
    m.add_function(wrap_pyfunction!(get_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(get_surprise, m)?)?;
    m.add_function(wrap_pyfunction!(get_detection_flags, m)?)?;
    m.add_function(wrap_pyfunction!(get_oov_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(get_stages_executed, m)?)?;
    m.add_function(wrap_pyfunction!(get_step_count, m)?)?;
    m.add("LLMOSafeError", _py.get_type::<LLMOSafeError>())?;
    m.add("ResourceExhaustedError", _py.get_type::<ResourceExhaustedError>())?;
    m.add("CognitiveInstabilityError", _py.get_type::<CognitiveInstabilityError>())?;
    m.add("BiasHaloDetectedError", _py.get_type::<BiasHaloDetectedError>())?;
    m.add_class::<CognitivePipeline>()?;
    Ok(())
}
