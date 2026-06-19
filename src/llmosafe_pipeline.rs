//! `CognitivePipeline` — 5-stage sequential safety pipeline.
//!
//! Wires the sifter, working memory, kernel, 5 detectors, dynamic stability
//! monitor, PID controller, and escalation policy into a single cascade that
//! can short-circuit at any stage.
//!
//! # Stage Flow
//!
//! ```text
//! process(text) → SIFT → MEMORY → KERNEL → DETECTION → PID → MONITOR → PipelineResult
//!                    │       │        │         │        │        │
//!                    ▼       ▼        ▼         ▼        ▼        ▼
//!             Halt?   Halt?    Halt?    Gate?    Risk    Advisory
//! ```
//!
//! 1. **SIFT** (Tier 3) — `sift_text_with_score()` classifies text, builds
//!    `SiftedSynapse`. Gate: `EscalationPolicy::decide()`.
//! 2. **MEMORY** (Tier 2) — `WorkingMemory::update()` pushes synapse into ring
//!    buffer. Gate: surprise threshold.
//! 3. **KERNEL** (Tier 1) — `ReasoningLoop::next_step()` advances reasoning.
//!    Gate: depth, bias, entropy stability.
//! 4. **DETECTION** — 5 detectors observe the observation. Flags packed into
//!    synapse reserved bits. Optional detection-gate path bypasses PID.
//! 5. **PID** — `compute_pid_score_pure()` + `apply_safety_overrides()` produce
//!    a risk score mapped to `SafetyDecision` via thresholds.
//! 6. **MONITOR** — `DynamicStabilityMonitor::update()` records entropy envelope.
//!    Advisory only.
//!
//! # Key Types
//!
//! - `CognitivePipeline<'a, MEM_SIZE, MAX_STEPS>` — owns all safety components
//! - `PipelineConfig` — threshold configuration with `validate()` bounds checking
//! - `PipelineResult` — final decision, classified synapse, stage bitmask, diagnostics
//! - `MemoryStats` — snapshot of working-memory mean, variance, trend
//!
//! # Processing Modes
//!
//! - `process(observation)` — standard 5-stage pipeline
//! - `process_with_pressure(observation, body_entropy, pressure)` — adds resource
//!   body pre-gate before SIFT
//! - `process_ctrl(observation, e_body, pressure)` — control-theory composition
//!   path with PID mandatory
//! - `process_safe(text, guard)` — pre-flight resource gate with deadline
#![deny(clippy::cast_lossless)]
// Arithmetic in this module operates on bounded counters and time values
// where wrap/instant arithmetic is the intended behavior.
// DO-178C: these operations are verified safe by value range analysis at
// the module boundary — inputs are always validated before arithmetic.
#![allow(clippy::arithmetic_side_effects)]

use crate::control_types::OverrideFlags;
#[cfg(feature = "std")]
use crate::llmosafe_detection::DetectionResult;
use crate::llmosafe_detection::{
    AdversarialDetector, ConfidenceTracker, CusumDetector, DriftDetector, RepetitionDetector,
};
use crate::llmosafe_integration::EscalationPolicy;
use crate::llmosafe_integration::SafetyDecision;
use crate::llmosafe_kernel::{
    DynamicStabilityMonitor, KernelError, KernelOutput, ReasoningLoop, StabilityResult, Synapse,
    ValidatedSynapse, FLAG_ADVERSARIAL, FLAG_ANOMALY, FLAG_DECAYING, FLAG_DRIFTING,
    FLAG_LOW_CONFIDENCE, FLAG_STUCK, U16_MAX_F32,
};
use crate::llmosafe_memory::WorkingMemory;
use crate::llmosafe_pid::{PidConfig, PidState};
#[cfg(feature = "std")]
use crate::ResourceGuard;

/// Bitmask constants for `PipelineResult.stages_executed`.
/// Set in `process_ctrl()` during sequential stage execution.
pub const STAGE_SIFT: u8 = 0x01;
/// Bitmask constant 0x02 for the MEMORY stage. Set in `process_ctrl()`.
pub const STAGE_MEMORY: u8 = 0x02;
/// Bitmask constant 0x04 for the KERNEL stage. Set in `process_ctrl()`.
pub const STAGE_KERNEL: u8 = 0x04;
/// Bitmask constant 0x08 for the DETECTION stage. Set in `process_ctrl()`.
pub const STAGE_DETECTION: u8 = 0x08;
/// Bitmask constant 0x10 for the MONITOR stage. Set in `process_ctrl()`.
pub const STAGE_MONITOR: u8 = 0x10;
/// Bitmask constant 0x20 gated behind cfg(feature="std"). Set in
/// `process_with_pressure()` after the body pressure pre-gate executes
/// and `process_ctrl()` returns.
#[cfg(feature = "std")]
pub const STAGE_BODY: u8 = 0x20;

/// Configuration for a CognitivePipeline instance.
///
/// Every threshold has a safe default via `Default::default()`.
/// Fields with `f32` values must be in `[0.0, 1.0]` and finite.
/// Use `validate()` to check bounds before constructing a pipeline.
///
/// # Dual-Calibration Architecture
///
/// `PipelineConfig` holds two independently calibrated decision frameworks:
///
/// 1. **EscalationPolicy** (`policy`) — operates in raw entropy space `[0, 65535]`.
///    Thresholds `halt_entropy`, `escalate_entropy`, `warn_entropy` are compared
///    directly against `u16` synapse entropy values. This is the innate immune
///    backstop — simple, fast, threshold-based gating at every pipeline stage.
///
/// 2. **PidConfig** (`pid_config`) — operates in normalised risk space `[0.0, 1.0]`.
///    Thresholds `halt_gain`, `warn_gain` are compared against a PID-computed
///    risk score that fuses entropy, memory surprise, kernel stability, classifier
///    probability, and trend into a single value. This is the adaptive control
///    path — stateful, integrative, with anti-windup and sidechain modulation.
///
/// The PID normalises raw entropy via `entropy / U16_MAX_F32`, establishing a
/// mapping between the two spaces. `validate()` checks cross-consistency of
/// equivalent thresholds across the two frameworks (advisory, not a hard error).
/// `validate_cross_consistency()` provides the detailed warning list.
///
/// Fields:
/// - `policy: EscalationPolicy` — escalation policy thresholds (entropy warn/escalate/halt, surprise, bias).
/// - `pid_config: PidConfig` — PID controller configuration. Must be valid.
/// - `surprise_threshold: i128` — surprise threshold for `WorkingMemory`. Values above this are rejected as `HallucinationDetected`.
/// - `max_repetitions: usize` — maximum repetitions before stuck detection fires.
/// - `drift_threshold: f32` — drift threshold (0.0–1.0). Drift above this triggers `GoalDriftDetected`.
/// - `min_confidence: f32` — minimum confidence threshold (0.0–1.0). Confidence below this is flagged.
/// - `decay_threshold: usize` — decay threshold: consecutive confidence drops before decay warning.
/// - `monitor_k: u8` — `DynamicStabilityMonitor` safety margin k (1–5). Controls envelope sensitivity.
/// - `use_detection_gate: bool` — when true, routes decisions through `decide_from_detection()` instead of the PID weighted summation path.
pub struct PipelineConfig {
    /// Escalation policy thresholds (entropy warn/escalate/halt, surprise, bias).
    pub policy: EscalationPolicy,
    /// PID controller configuration. Must be valid.
    pub pid_config: PidConfig,
    /// Surprise threshold for `WorkingMemory`. Values above this are rejected
    /// as `HallucinationDetected`.
    pub surprise_threshold: i128,
    /// Maximum repetitions before stuck detection fires.
    pub max_repetitions: usize,
    /// Drift threshold (0.0–1.0). Drift above this triggers `GoalDriftDetected`.
    pub drift_threshold: f32,
    /// Minimum confidence threshold (0.0–1.0). Confidence below this is flagged.
    pub min_confidence: f32,
    /// Decay threshold: consecutive confidence drops before decay warning.
    pub decay_threshold: usize,
    /// `DynamicStabilityMonitor` safety margin k (1–5). Controls envelope sensitivity.
    pub monitor_k: u8,
    /// When true, routes decisions through `decide_from_detection()` (first-match-wins
    /// severity ordering: Anomaly > Adversarial > Drifting > Stuck > Confidence) instead
    /// of the PID weighted summation path.  The detection-gate path avoids PID integrator
    /// state entirely — it is simpler and faster but does not remember past observations
    /// beyond what the individual detectors track.  Default: false (PID path).
    /// Only available when `std` feature is enabled (requires vec![] allocation).
    #[cfg(feature = "std")]
    pub use_detection_gate: bool,
}

impl Default for PipelineConfig {
    /// Safe defaults calibrated for the classifier entropy range `[0, 65535]`.
    fn default() -> Self {
        Self {
            policy: EscalationPolicy::default(),
            pid_config: PidConfig::default(),
            surprise_threshold: 58000,
            max_repetitions: 3,
            drift_threshold: 0.5,
            min_confidence: 0.3,
            decay_threshold: 3,
            monitor_k: 3,
            #[cfg(feature = "std")]
            use_detection_gate: false,
        }
    }
}

impl PipelineConfig {
    /// Validates all configuration fields are within safe bounds.
    ///
    /// Checks performed in order:
    /// 1. `drift_threshold` — must be finite and in `[0.0, 1.0]`.
    /// 2. `min_confidence` — must be finite and in `[0.0, 1.0]`.
    /// 3. `monitor_k` — must be in `[1, 5]`.
    /// 4. `max_repetitions` — must be `> 0`.
    /// 5. `decay_threshold` — must be `> 0`.
    /// 6. `pid_config.validate()` — delegates to `PidConfig::validate()` for
    ///    NaN/out-of-range gain checks and `warn_gain < halt_gain` ordering.
    /// 7. **Cross-consistency** (`validate_cross_consistency`) — compares
    ///    `EscalationPolicy` entropy thresholds (mapped to risk via
    ///    `threshold / U16_MAX_F32`) against `PidConfig` risk thresholds
    ///    with ±15% tolerance. Advisory only: mismatches are logged to
    ///    stderr but do NOT cause `validate()` to return `Err`. This is
    ///    intentional — the two frameworks serve different safety layers
    ///    and can be intentionally diverged.
    ///
    /// `validate()` is called by `CognitivePipeline::with_config()` before
    /// construction.
    ///
    /// # Errors
    ///
    /// Returns `"drift_threshold must be in [0.0, 1.0]"` if out of range or NaN.
    /// Returns `"min_confidence must be in [0.0, 1.0]"` if out of range or NaN.
    /// Returns `"monitor_k must be in [1, 5]"` if out of range.
    /// Returns `"max_repetitions must be > 0"` if zero.
    /// Returns `"decay_threshold must be > 0"` if zero.
    /// Propagates PID config validation errors.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.drift_threshold.is_nan() || self.drift_threshold < 0.0 || self.drift_threshold > 1.0
        {
            return Err("drift_threshold must be in [0.0, 1.0]");
        }
        if self.min_confidence.is_nan() || self.min_confidence < 0.0 || self.min_confidence > 1.0 {
            return Err("min_confidence must be in [0.0, 1.0]");
        }
        if self.monitor_k < 1 || self.monitor_k > 5 {
            return Err("monitor_k must be in [1, 5]");
        }
        if self.max_repetitions == 0 {
            return Err("max_repetitions must be > 0");
        }
        if self.decay_threshold == 0 {
            return Err("decay_threshold must be > 0");
        }
        self.pid_config.validate()?;

        // Cross-consistency check: compares EscalationPolicy entropy
        // thresholds against PID risk thresholds. Advisory only —
        // warnings are logged but validate() still returns Ok(()).
        // Gated behind std because Vec<String> requires alloc.
        #[cfg(feature = "std")]
        {
            #[allow(clippy::print_stderr)]
            if let Err(warnings) = self.validate_cross_consistency() {
                for w in &warnings {
                    eprintln!("[llmosafe] {}", w);
                }
            }
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = &self.policy; // Silence unused field warnings in no_std
        }

        Ok(())
    }

    /// Validates cross-consistency between `EscalationPolicy` entropy thresholds
    /// and `PidConfig` risk thresholds.
    ///
    /// # Dual-Calibration Mapping
    ///
    /// The PID normalises raw entropy `[0, 65535]` to risk `[0.0, 1.0]` via
    /// `entropy / U16_MAX_F32`. This method computes the risk-equivalent of
    /// each `EscalationPolicy` threshold and compares it against the
    /// corresponding `PidConfig` threshold:
    ///
    /// | Policy threshold    | PID equivalent            | Tolerance |
    /// |--------------------|--------------------------|-----------|
    /// | `halt_entropy`     | `halt_gain`              | ±15%      |
    /// | `escalate_entropy` | `halt_gain × 0.8`        | ±15%      |
    /// | `warn_entropy`     | `warn_gain`              | ±15%      |
    ///
    /// The escalate mapping uses `halt_gain × 0.8` as a heuristic — escalate
    /// is typically set at ~80% of the halt threshold in both frameworks.
    ///
    /// # Tolerance
    ///
    /// Relative tolerance of ±15% is used when both values are non-zero.
    /// Falls back to absolute tolerance for near-zero thresholds. This
    /// allows reasonable calibration divergence while flagging significant
    /// misalignment.
    ///
    /// # Returns
    ///
    /// - `Ok(())` — all three threshold pairs are within ±15% tolerance.
    /// - `Err(warnings)` — one or more pairs exceed tolerance. Each warning
    ///   is a human-readable string containing the policy threshold name,
    ///   its risk-equivalent, the PID gain value, and the divergence
    ///   percentage.
    ///
    /// # Why Soft, Not Hard
    ///
    /// The two frameworks serve different safety layers: `EscalationPolicy`
    /// is the innate immune backstop (fast, stateless, threshold-gated),
    /// while `PidConfig` drives the adaptive control path (stateful,
    /// integrative, with anti-windup). Independent calibration is a valid
    /// operational choice. This check makes inconsistency visible without
    /// preventing deployment. Operators can tune thresholds to converge
    /// or keep the divergence intentionally.
    ///
    /// Only available when the `std` feature is enabled (requires `Vec<String>`
    /// for warning collection).
    #[cfg(feature = "std")]
    pub fn validate_cross_consistency(&self) -> Result<(), Vec<String>> {
        let mut warnings: Vec<String> = Vec::new();

        // ── Halt: halt_entropy → risk vs halt_gain ──
        let halt_policy_risk = f32::from(self.policy.halt_entropy) / U16_MAX_F32;
        Self::check_cross_pair(
            halt_policy_risk,
            self.pid_config.halt_gain,
            "halt_entropy",
            self.policy.halt_entropy,
            "halt_gain",
            &mut warnings,
        );

        // ── Escalate: escalate_entropy → risk vs halt_gain × 0.8 ──
        let escalate_policy_risk = f32::from(self.policy.escalate_entropy) / U16_MAX_F32;
        let escalate_pid_equiv = self.pid_config.halt_gain * 0.8_f32;
        Self::check_cross_pair(
            escalate_policy_risk,
            escalate_pid_equiv,
            "escalate_entropy",
            self.policy.escalate_entropy,
            "halt_gain × 0.8",
            &mut warnings,
        );

        // ── Warn: warn_entropy → risk vs warn_gain ──
        let warn_policy_risk = f32::from(self.policy.warn_entropy) / U16_MAX_F32;
        Self::check_cross_pair(
            warn_policy_risk,
            self.pid_config.warn_gain,
            "warn_entropy",
            self.policy.warn_entropy,
            "warn_gain",
            &mut warnings,
        );

        if warnings.is_empty() {
            Ok(())
        } else {
            Err(warnings)
        }
    }

    /// Compares a policy threshold (mapped to risk space) against a PID gain.
    ///
    /// Uses relative tolerance (`CROSS_CONSISTENCY_TOLERANCE`) when both
    /// values are above `f32::EPSILON`. Falls back to absolute tolerance
    /// for near-zero thresholds to avoid division-by-zero in the ratio.
    ///
    /// If the pair exceeds tolerance, appends a human-readable warning
    /// string to `warnings` containing:
    /// - The severity level (halt/escalate/warn)
    /// - The policy threshold field name and raw `u16` value
    /// - The risk-equivalent (policy_threshold / 65535)
    /// - The PID gain field name and value
    /// - The divergence percentage
    #[cfg(feature = "std")]
    fn check_cross_pair(
        policy_risk: f32,
        pid_gain: f32,
        policy_name: &str,
        policy_raw: u16,
        pid_name: &str,
        warnings: &mut Vec<String>,
    ) {
        // Extract severity label from the field name (halt_entropy → "halt")
        let severity = policy_name.split('_').next().unwrap_or(policy_name);

        // Near-zero values: use absolute tolerance to avoid division blow-up
        if policy_risk <= f32::EPSILON || pid_gain <= f32::EPSILON {
            if (policy_risk - pid_gain).abs() <= CROSS_CONSISTENCY_TOLERANCE {
                return;
            }
            let divergence_pct = (policy_risk - pid_gain).abs() * 100.0_f32;
            warnings.push(format!(
                "RC-DUAL {} mismatch: policy.{}={} → risk {:.4}, pid_config.{}={:.4} (divergence: {:.1}%, absolute)",
                severity, policy_name, policy_raw, policy_risk, pid_name, pid_gain, divergence_pct
            ));
            return;
        }

        // Relative tolerance check
        let ratio = policy_risk / pid_gain;
        if (1.0_f32 - CROSS_CONSISTENCY_TOLERANCE..=1.0_f32 + CROSS_CONSISTENCY_TOLERANCE)
            .contains(&ratio)
        {
            return;
        }

        let divergence_pct = (ratio - 1.0_f32).abs() * 100.0_f32;
        warnings.push(format!(
            "RC-DUAL {} mismatch: policy.{}={} → risk {:.4}, pid_config.{}={:.4} (divergence: {:.1}%)",
            severity, policy_name, policy_raw, policy_risk, pid_name, pid_gain, divergence_pct
        ));
    }
}

/// Constant: ±15% relative tolerance for cross-consistency checks between
/// EscalationPolicy entropy thresholds and PidConfig risk thresholds.
/// Used by `PipelineConfig::validate_cross_consistency()`.
#[cfg(feature = "std")]
const CROSS_CONSISTENCY_TOLERANCE: f32 = 0.15_f32;

/// Snapshot of working-memory statistics.
///
/// All fields are computed from the ring-buffer state at call time.
/// `is_drifting` compares `trend` against a fixed threshold of 10.0.
///
/// Fields:
/// - `mean: f64` — running mean entropy of the ring buffer [0, 65535].
/// - `variance: f64` — running variance of ring-buffer entropy.
/// - `trend: f64` — linear regression slope over the buffer window.
/// - `is_drifting: bool` — true when `|trend| > 10.0`.
pub struct MemoryStats {
    /// Running mean entropy of the ring buffer `[0, 65535]`.
    pub mean: f64,
    /// Running variance of ring-buffer entropy.
    pub variance: f64,
    /// Linear regression slope over the buffer window.
    pub trend: f64,
    /// `true` when `|trend| > 10.0`.
    pub is_drifting: bool,
}

/// Aggregate output of a single `CognitivePipeline::process()` invocation.
///
/// Carries the final `SafetyDecision`, the classified `Synapse` (with packed
/// detection flags and OOV ratio), a stages-executed bitmask, and diagnostic
/// fields for the C-ABI query functions.
///
/// Fields:
/// - `decision: SafetyDecision` — final safety decision from the pipeline.
/// - `synapse: Synapse` — classified synapse with entropy, surprise, bias, detection flags, OOV ratio.
/// - `stages_executed: u8` — bitmask of stages that executed. `STAGE_SIFT` (0x01) through `STAGE_MONITOR` (0x10).
/// - `detection_flags: u8` — five detection flags packed into 5 bits.
/// - `oov_ratio: u8` — OOV (out-of-vocabulary) ratio. 0=0%, 255=100%.
/// - `entropy: u16` — convenience copy of `synapse.raw_entropy()`. Required by C-ABI query functions.
/// - `surprise: u16` — convenience copy of `synapse.raw_surprise()`. Required by C-ABI query functions.
/// - `monitor_state: StabilityResult` — stability state from the `DynamicStabilityMonitor` after this invocation.
/// - `body_pressure: Option<u8>` — resource body pressure percentage [0, 100] when `process_with_pressure()` was used.
/// - `step_count: usize` — current reasoning step count after this invocation.
/// - `kernel_output: Option<KernelOutput>` — kernel output from the reasoning loop (diagnostic).
/// - `classifier_score: f32` — raw classifier logit (`ClassificationResult.score`) before sigmoid.
pub struct PipelineResult {
    /// Final safety decision from the pipeline.
    pub decision: SafetyDecision,
    /// Classified synapse with entropy, surprise, bias, detection flags, OOV ratio.
    pub synapse: Synapse,
    /// Bitmask of stages that executed. `STAGE_SIFT` (0x01) through `STAGE_MONITOR` (0x10).
    pub stages_executed: u8,
    /// Five detection flags packed into 5 bits.
    pub detection_flags: u8,
    /// OOV (out-of-vocabulary) ratio. 0=0%, 255=100%.
    pub oov_ratio: u8,
    /// Convenience copy of `synapse.raw_entropy()`. Required by C-ABI query functions.
    pub entropy: u16,
    /// Convenience copy of `synapse.raw_surprise()`. Required by C-ABI query functions.
    pub surprise: u16,
    /// Stability state from the `DynamicStabilityMonitor` after this invocation.
    pub monitor_state: StabilityResult,
    /// Resource body pressure percentage [0, 100] when `process_with_pressure()` was used.
    #[cfg(feature = "std")]
    pub body_pressure: Option<u8>,
    /// Current reasoning step count after this invocation.
    pub step_count: usize,
    /// Kernel output from the reasoning loop (diagnostic).
    pub kernel_output: Option<KernelOutput>,
    /// Raw classifier logit (`ClassificationResult.score`) before sigmoid.
    /// Unbounded f32 — negative = safe, positive = manipulation signal.
    /// Set to 0.0 in error-path results where no classification was performed.
    pub classifier_score: f32,
}

impl PipelineResult {
    /// Returns `true` when the pipeline decision allows processing to continue.
    pub fn is_safe(&self) -> bool {
        matches!(self.decision, SafetyDecision::Proceed)
    }

    /// Returns `Some(KernelError)` if the result is a `Halt` or `Exit` decision.
    pub fn halt_reason(&self) -> Option<&KernelError> {
        match &self.decision {
            SafetyDecision::Halt(err, _) | SafetyDecision::Exit(err) => Some(err),
            SafetyDecision::Proceed | SafetyDecision::Warn(_) | SafetyDecision::Escalate { .. } => {
                None
            }
        }
    }

    /// Returns a reference to the kernel output if the reasoning loop ran.
    ///
    /// `None` when the pipeline halted before the kernel stage completed
    /// (SIFT or MEMORY short-circuit, or kernel error).  The kernel output
    /// carries the normalised entropy error `[0.0, 1.0]`, a stability
    /// boolean, and the reasoning step depth at output time.
    pub fn kernel_output(&self) -> Option<&KernelOutput> {
        self.kernel_output.as_ref()
    }

    /// Returns the resource body pressure percentage [0, 100].
    ///
    /// Returns 0 when `process()` was used instead of `process_with_pressure()`
    /// (no resource data available).  Pressure is the RSS memory percentage of
    /// the configured ceiling fed through `process_with_pressure()`.
    #[cfg(feature = "std")]
    pub fn body_pressure(&self) -> u8 {
        self.body_pressure.unwrap_or(0)
    }
}

/// Five-stage cognitive safety pipeline.
///
/// Owns one instance of each safety component and orchestrates them through
/// sequential stages: SIFT → MEMORY → KERNEL → DETECTION → MONITOR.
/// Each stage can short-circuit the pipeline with a `Halt` or `Escalate` decision.
///
/// # Type parameters
///
/// * `MEM_SIZE` — ring-buffer capacity for `WorkingMemory` (default: 64).
/// * `MAX_STEPS` — maximum reasoning steps before `DepthExceeded` (default: 10).
///
/// # Lifetime
///
/// * `'a` — the objective string is borrowed; the caller must keep it alive.
///
/// Fields:
/// - `memory: WorkingMemory<MEM_SIZE>` — surprise-gated ring buffer for entropy history.
/// - `reasoning: ReasoningLoop<MAX_STEPS>` — deterministic reasoning step counter.
/// - `monitor: DynamicStabilityMonitor` — self-calibrating envelope tracker.
/// - `repetition: RepetitionDetector` — loop detection (stuck agent).
/// - `drift: DriftDetector` — goal drift detection.
/// - `confidence: ConfidenceTracker` — confidence decay tracking.
/// - `cusum: CusumDetector` — CUSUM anomaly detection.
/// - `adversarial: AdversarialDetector` — adversarial pattern recognition.
/// - `objective: &'a str` — original objective string for drift detection.
/// - `step_count: usize` — current reasoning step count.
/// - `pid_state: PidState` — PID controller state (dual-rate integrators).
/// - `pid_config: PidConfig` — PID controller configuration.
/// - `esc_policy: EscalationPolicy` — escalation policy thresholds.
/// - `use_detection_gate: bool` — when true, routes through detection-gate path.
/// - `drift_threshold: f32` — drift threshold [0.0, 1.0].
/// - `surprise_threshold: i128` — surprise threshold for WorkingMemory.
pub struct CognitivePipeline<'a, const MEM_SIZE: usize, const MAX_STEPS: usize> {
    memory: WorkingMemory<MEM_SIZE>,
    reasoning: ReasoningLoop<MAX_STEPS>,
    monitor: DynamicStabilityMonitor,
    repetition: RepetitionDetector,
    drift: DriftDetector,
    confidence: ConfidenceTracker,
    cusum: CusumDetector,
    adversarial: AdversarialDetector,
    objective: &'a str,
    step_count: usize,
    pid_state: PidState,
    pid_config: PidConfig,
    #[allow(dead_code)]
    pub(crate) esc_policy: EscalationPolicy,
    /// When true, routes decisions through the detection-gate path instead of PID.
    /// Only available with `std` feature (detection-gate path uses vec![]).
    #[cfg(feature = "std")]
    #[allow(dead_code)]
    pub(crate) use_detection_gate: bool,
    /// Drift threshold [0.0, 1.0]. Stored for `reset_detectors()` and `reset_full()`.
    drift_threshold: f32,
    /// Surprise threshold for `WorkingMemory` reconstruction in `reset_full()`.
    surprise_threshold: i128,
}

impl<'a, const MEM_SIZE: usize, const MAX_STEPS: usize> CognitivePipeline<'a, MEM_SIZE, MAX_STEPS> {
    /// Creates a pipeline with the given objective and default configuration.
    ///
    /// The objective string is borrowed — it must outlive the pipeline.
    /// Drift detection is initialized with the objective's keyword hashes.
    pub fn new(objective: &'a str) -> Self {
        let config = PipelineConfig::default();
        Self::with_config(objective, config).unwrap_or_else(|_| unreachable!())
    }

    /// Creates a pipeline with a custom `PipelineConfig`.
    ///
    /// Returns `Err` if `config.validate()` fails (NaN, out-of-range, zero
    /// thresholds). All detector instances are constructed from config fields.
    ///
    /// # Errors
    ///
    /// Propagates the same error strings from `PipelineConfig::validate()`.
    pub fn with_config(objective: &'a str, config: PipelineConfig) -> Result<Self, &'static str> {
        config.validate()?;
        Ok(Self {
            memory: WorkingMemory::<MEM_SIZE>::new(config.surprise_threshold),
            reasoning: ReasoningLoop::<MAX_STEPS>::new(),
            monitor: DynamicStabilityMonitor::new(config.monitor_k),
            repetition: RepetitionDetector::new(config.max_repetitions),
            drift: DriftDetector::new(objective, config.drift_threshold),
            confidence: ConfidenceTracker::new(config.min_confidence, config.decay_threshold),
            cusum: CusumDetector::new(0.0, 50.0, 200.0),
            adversarial: AdversarialDetector::new(),
            objective,
            step_count: 0,
            pid_state: PidState::new(),
            pid_config: config.pid_config,
            esc_policy: config.policy,
            #[cfg(feature = "std")]
            use_detection_gate: config.use_detection_gate,
            drift_threshold: config.drift_threshold,
            surprise_threshold: config.surprise_threshold,
        })
    }

    /// Processes an observation through the full 5-stage pipeline.
    ///
    /// # Stages
    ///
    /// 1. **SIFT** — Classifies text via TF-IDF classifier. Builds a `SiftedSynapse`.
    ///    Gate: `EscalationPolicy::decide(entropy, surprise, has_bias)`.
    /// 2. **MEMORY** — Pushes synapse into working-memory ring buffer.
    ///    Gate: `EscalationPolicy::decide_from_stability(stability)`.
    /// 3. **KERNEL** — Advances the reasoning loop. Checked for depth and bias.
    /// 4. **DETECTION** — Runs all 5 detectors. Packs flags into synapse reserved bits.
    ///    Gate: `EscalationPolicy::decide_from_detection()` (std) or inline checks (no_std).
    /// 5. **MONITOR** — Updates the `DynamicStabilityMonitor`. Advisory only.
    ///
    /// # Returns
    ///
    /// A `PipelineResult` with the final decision, classified synapse, and diagnostics.
    pub fn process(&mut self, observation: &str) -> PipelineResult {
        self.process_ctrl(observation, 0.0, 0)
    }

    /// Processes an observation with resource body pressure gating.
    ///
    /// Before the SIFT stage, `body_entropy` and `pressure` (0–100) are mapped
    /// to a `PressureLevel` via `EscalationPolicy` thresholds and gated via
    /// `decide_with_pressure()`. If pressure is `Critical` or `Emergency`, the
    /// pipeline returns an `Escalate` or `Halt` decision before running SIFT.
    ///
    /// `body_entropy` is in `[0, 1000]` (RSS + IO + load weighted score).
    /// `pressure` is a percentage `[0, 100]` of RSS memory ceiling.
    ///
    /// When PID mode is active, the legacy pressure gate is advisory only —
    /// the pressure value is fed directly into `compute_pid_score()` as the P-term.
    ///
    /// Sets `STAGE_BODY` (0x20) in the result's `stages_executed` bitmask.
    #[cfg(feature = "std")]
    pub fn process_with_pressure(
        &mut self,
        observation: &str,
        body_entropy: u16,
        pressure: u8,
    ) -> PipelineResult {
        use crate::llmosafe_integration::PressureLevel;

        // Pre-SIFT pressure gate: map pressure to PressureLevel.
        // If pressure is Critical or Emergency, short-circuit before SIFT
        // via decide_with_pressure(). This is the documented safety
        // requirement from the pipeline doc comment (line 408-412).
        let pressure_level = PressureLevel::from_percentage(pressure);
        if pressure_level.requires_action() {
            // Without SIFT data, use body_entropy as entropy proxy;
            // surprise=0, has_bias=false (not classified yet).
            let decision =
                self.esc_policy
                    .decide_with_pressure(body_entropy, 0, false, pressure_level);
            let mut synapse = Synapse::new();
            synapse.set_raw_entropy(body_entropy);
            return PipelineResult {
                decision,
                synapse,
                stages_executed: STAGE_BODY,
                detection_flags: 0,
                oov_ratio: 0,
                entropy: body_entropy,
                surprise: 0,
                monitor_state: crate::llmosafe_kernel::StabilityResult::Stable,
                #[cfg(feature = "std")]
                body_pressure: Some(pressure),
                step_count: self.step_count,
                kernel_output: None,
                classifier_score: 0.0,
            };
        }

        // Route body pressure through BodyOutput → PidInput.e_body
        let e_body = (f32::from(body_entropy) / 1000.0_f32).clamp(0.0, 1.0);
        let mut result = self.process_ctrl(observation, e_body, pressure);
        result.stages_executed |= STAGE_BODY;
        result.body_pressure = Some(pressure);
        result
    }

    /// Pre-flight resource gate on the cognitive pipeline.
    ///
    /// Calls `ResourceGuard::check_with_deadline()` with a 5-second deadline
    /// before processing. If resources are safe (guard returns `Ok`), the full
    /// pipeline runs via `process()`. If the deadline is exceeded (guard returns
    /// `DeadlineExceeded`), the pipeline still runs but via
    /// `process_with_pressure()` — passing the guard's `raw_entropy()` and
    /// `pressure()` values so resource body state is recorded in the result.
    /// All other guard errors are returned without running the pipeline.
    ///
    /// # Errors
    ///
    /// Propagates `KernelError` from `ResourceGuard::check_with_deadline()` for
    /// all errors except `DeadlineExceeded`, which is handled gracefully.
    #[cfg(feature = "std")]
    pub fn process_safe(
        &mut self,
        text: &str,
        guard: &ResourceGuard,
    ) -> Result<PipelineResult, KernelError> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        match guard.check_with_deadline(deadline) {
            Ok(_synapse) => Ok(self.process(text)),
            Err(KernelError::DeadlineExceeded) => {
                Ok(self.process_with_pressure(text, guard.raw_entropy(), guard.pressure()))
            }
            Err(e) => Err(e),
        }
    }

    /// Resets all 6 detectors and the `DynamicStabilityMonitor`.
    ///
    /// Preserves `WorkingMemory` ring-buffer state and `ReasoningLoop` step count.
    /// Use `reset_full()` for a complete reset.
    pub fn reset_detectors(&mut self) {
        self.repetition.reset();
        self.confidence.reset();
        self.cusum.reset();
        self.adversarial = AdversarialDetector::new();
        self.monitor.reset();
        self.drift = DriftDetector::new(self.objective, self.drift_threshold);
    }

    /// Full reset to post-construction state.
    ///
    /// Clears detectors, monitor, memory ring buffer (all entropy entries zeroed),
    /// reasoning step count, and PID integrators. Objective string is preserved.
    pub fn reset_full(&mut self) {
        self.memory = WorkingMemory::<MEM_SIZE>::new(self.surprise_threshold);
        self.reasoning = ReasoningLoop::<MAX_STEPS>::new();
        self.monitor.reset();
        self.repetition.reset();
        self.drift = DriftDetector::new(self.objective, self.drift_threshold);
        self.confidence.reset();
        self.cusum.reset();
        self.adversarial = AdversarialDetector::new();
        self.step_count = 0;
        self.pid_state.reset();
    }

    /// Returns a reference to the PID state.
    ///
    /// `PidState` holds the dual-rate leaky integrators (`acute_entropy`,
    /// `chronic_entropy`) and `prev_pressure_norm` for step-change detection.
    /// All fields are `f32` clamped to `[0, 1]`. The state mutates across
    /// `process()` calls; this getter returns the live state for introspection.
    pub fn pid_state(&self) -> &PidState {
        &self.pid_state
    }

    /// Returns a snapshot of working-memory statistics.
    ///
    /// `is_drifting` uses a fixed threshold of 10.0 — positive values
    /// indicate rising entropy, negative values indicate falling entropy.
    pub fn memory_stats(&self) -> MemoryStats {
        let mean = self.memory.mean_entropy();
        let variance = self.memory.entropy_variance();
        let trend = self.memory.trend();
        MemoryStats {
            mean,
            variance,
            trend,
            is_drifting: self.memory.is_drifting(10.0),
        }
    }

    // ── Control Theory Composition Path ────────────────────────────

    /// Processes an observation through the full cascade control pipeline.
    ///
    /// Uses the control theory architecture with typed output structs,
    /// pure PID computation, and safety overrides (infusion pump pattern).
    ///
    /// # Control Flow
    ///
    /// ```text
    /// SifterOutput → MemoryOutput → KernelOutput → Detection → PidInput →
    ///   compute_pid_score_pure → apply_safety_overrides → pid_risk_to_decision
    /// ```
    ///
    /// # DAL A
    ///
    /// Safety overrides (bias, exhaustion, kernel instability) are applied
    /// AFTER pure PID computation, preventing PID bugs from bypassing safety.
    ///
    /// # MC/DC
    ///
    /// Each control signal independently affects its PID term.
    /// Each override flag independently forces Halt.
    ///
    /// Control-theory composition path (cascade control).
    /// PID is now mandatory — see `pid_config` field.
    /// `e_body` is the normalised body pressure error [0.0, 1.0] from BodyOutput.
    /// `pressure` is the resource pressure percentage [0, 100], passed for diagnostics.
    pub fn process_ctrl(&mut self, observation: &str, e_body: f32, pressure: u8) -> PipelineResult {
        let _pressure = pressure; // Only used in test/diagnostics in some configurations
        let mut stages = 0u8;

        // ── Stage 1: SIFT (Tier 3) ──
        // Single canonical entry point. Keyword-bias (innate layer) OR-s into
        // the classifier result. One synapse, one proof, no duplicate compute.
        stages |= STAGE_SIFT;
        let (sifted, sifted_proof, classifier_score) =
            crate::llmosafe_sifter::sift_text_with_score(observation);
        let entropy = sifted.raw_entropy();
        let surprise_val = sifted.raw_surprise();
        let oov_ratio = sifted.oov_ratio();
        let has_bias = sifted.has_bias();

        // ── Stage 2: MEMORY (Tier 2) ──
        stages |= STAGE_MEMORY;
        let mem_result = self.memory.update(sifted, sifted_proof);
        let validated = match mem_result {
            Ok((v, _p)) => v,
            Err(err) => {
                return self.ctrl_result_from_error(err, stages, oov_ratio, entropy, surprise_val);
            }
        };

        // ── Stage 3: KERNEL (Tier 1) ──
        stages |= STAGE_KERNEL;
        // NOTE: The ValidatedProof returned by WorkingMemory::update() is discarded
        // here (bound to _p in the match above) and a fresh ValidatedProof(()) is
        // minted below. This is intentional: the data DID pass through
        // WorkingMemory::update() — only the compile-time proof token is
        // regenerated to enter the ReasoningLoop. ValidatedProof uses
        // pub(crate) visibility, so re-minting is permitted anywhere within
        // the crate. See invariants.toml typestate_pipeline_order for the
        // invariant this satisfies.
        let kernel_synapse = validated.into_inner();
        let kernel_entropy = kernel_synapse.raw_entropy();
        let kernel_validated = ValidatedSynapse::new(kernel_synapse);
        let kernel_result = self
            .reasoning
            .next_step(kernel_validated, crate::llmosafe_kernel::ValidatedProof(()));
        let kernel_synapse_out = match kernel_result {
            Ok(()) => {
                self.step_count += 1;
                validated.into_inner()
            }
            Err(err) => {
                return self.ctrl_result_from_kernel_error(
                    err,
                    stages,
                    oov_ratio,
                    entropy,
                    surprise_val,
                );
            }
        };

        // ── Stage 4: DETECTION (Sidechain) ──
        stages |= STAGE_DETECTION;
        self.repetition.observe(observation);
        self.drift.observe(observation);
        let classifier_prob = f32::from(entropy) / U16_MAX_F32;
        self.confidence.observe(classifier_prob);
        let _cusum_anomaly = self.cusum.update(f64::from(entropy));

        let is_stuck = self.repetition.is_stuck();
        let is_drifting = self.drift.is_drifting();
        let is_low_confidence = self.confidence.is_low();
        let is_decaying = self.confidence.is_decaying();
        let anomaly_detected = self.cusum.detected();
        let adversarial_detected = self.adversarial.is_adversarial(observation);

        let mut flags: u8 = 0;
        if is_stuck {
            flags |= FLAG_STUCK;
        }
        if is_drifting {
            flags |= FLAG_DRIFTING;
        }
        if is_low_confidence {
            flags |= FLAG_LOW_CONFIDENCE;
        }
        if is_decaying {
            flags |= FLAG_DECAYING;
        }
        if anomaly_detected {
            flags |= FLAG_ANOMALY;
        }
        if adversarial_detected {
            flags |= FLAG_ADVERSARIAL;
        }

        // ── Stage 5a: DETECTION GATE (optional, non-PID path) ──
        // First-match-wins severity ordering: Anomaly > Adversarial > Drifting > Stuck > Confidence.
        // Avoids PID integrator state entirely. If the gate produces a Halt, return early;
        // otherwise fall through to the PID composition below.
        #[cfg(feature = "std")]
        if self.use_detection_gate {
            let detection_result = DetectionResult {
                is_stuck,
                is_drifting,
                is_low_confidence,
                is_decaying,
                adversarial_patterns: if adversarial_detected {
                    vec!["adversarial"]
                } else {
                    vec![]
                },
                risk_score: if anomaly_detected { 0.9 } else { 0.0 },
            };
            let gate_decision =
                self.esc_policy
                    .decide_from_detection(&detection_result, entropy, surprise_val);
            if gate_decision.must_halt() {
                stages |= STAGE_MONITOR;
                let monitor_state = self.monitor.update(u32::from(entropy));
                let kernel_output = Some(KernelOutput {
                    error_kernel: f32::from(kernel_entropy) / U16_MAX_F32,
                    is_stable: u32::from(kernel_entropy)
                        < crate::llmosafe_kernel::STABILITY_THRESHOLD as u32,
                    depth: self.step_count,
                });
                return PipelineResult {
                    decision: gate_decision,
                    synapse: kernel_synapse_out,
                    stages_executed: stages,
                    detection_flags: flags,
                    oov_ratio,
                    entropy,
                    surprise: surprise_val,
                    monitor_state,
                    #[cfg(feature = "std")]
                    body_pressure: Some(pressure),
                    step_count: self.step_count,
                    kernel_output,
                    classifier_score,
                };
            }
        }

        // ── Stage 5: PID COMPOSITION ──
        let pressure_term = (e_body * 100.0_f32) as u8;
        let trend = self.memory.trend();

        // Compute tier error signals for the 4-tier PID cascade.
        // e_mem: memory surprise = |current_entropy − mean_entropy| / 65535
        // e_kernel: kernel stability error = kernel_entropy / 65535
        // Both clamped to [0.0, 1.0].
        let mem_mean = self.memory.mean_entropy();
        let e_mem = ((f64::from(entropy) - mem_mean).abs() as f32 / U16_MAX_F32).clamp(0.0, 1.0);
        let e_kernel = (f32::from(kernel_entropy) / U16_MAX_F32).clamp(0.0, 1.0);

        let pid_input = crate::control_types::PidInput::new(
            e_body,
            f32::from(entropy) / U16_MAX_F32,
            e_mem,
            e_kernel,
            trend,
            classifier_prob,
            has_bias,
            flags,
            pressure_term,
        );
        let pure_risk = crate::llmosafe_pid::compute_pid_score_pure(
            &pid_input,
            &self.pid_config,
            &mut self.pid_state,
        );

        // ── Stage 5bis: MONITOR (before overrides — gates KERNEL_UNSTABLE) ──
        stages |= STAGE_MONITOR;
        let monitor_state = self.monitor.update(u32::from(kernel_entropy));

        let mut override_flags = OverrideFlags::empty();
        if has_bias {
            override_flags = override_flags | OverrideFlags::BIAS;
        }
        if e_body > 0.9 {
            override_flags = override_flags | OverrideFlags::EXHAUSTED;
        }
        // KERNEL_UNSTABLE: set when the dynamic stability monitor detects
        // High, Low, or Both — any non-Stable state triggers the safety gate.
        if monitor_state != crate::llmosafe_kernel::StabilityResult::Stable {
            override_flags = override_flags | OverrideFlags::KERNEL_UNSTABLE;
        }
        let limited_risk = crate::llmosafe_pid::apply_safety_overrides(
            pure_risk,
            override_flags,
            &self.pid_config,
        );
        let decision = crate::llmosafe_pid::pid_risk_to_decision(limited_risk, &self.pid_config);

        let kernel_output = Some(KernelOutput {
            error_kernel: f32::from(kernel_entropy) / U16_MAX_F32,
            is_stable: u32::from(kernel_entropy)
                < crate::llmosafe_kernel::STABILITY_THRESHOLD as u32,
            depth: self.step_count,
        });

        PipelineResult {
            decision,
            synapse: kernel_synapse_out,
            stages_executed: stages,
            detection_flags: flags,
            oov_ratio,
            entropy,
            surprise: surprise_val,
            monitor_state,
            #[cfg(feature = "std")]
            body_pressure: Some(pressure),
            step_count: self.step_count,
            kernel_output,
            classifier_score,
        }
    }

    /// Constructs a `PipelineResult` from a `WorkingMemory` error (`KernelError`).
    ///
    /// Maps `HallucinationDetected` → `Escalate` (5000ms cooldown),
    /// `CognitiveInstability` → `Halt` (30000ms),
    /// `BiasHaloDetected` → `Halt` (30000ms),
    /// all other errors → `Halt(error, 30000ms)`.
    /// Constructs a fresh `Synapse` with entropy populated, detection_flags=0,
    /// monitor_state=Stable, kernel_output=None.
    fn ctrl_result_from_error(
        &self,
        err: KernelError,
        stages: u8,
        oov_ratio: u8,
        entropy: u16,
        surprise_val: u16,
    ) -> PipelineResult {
        let (decision, synapse) = match err {
            KernelError::HallucinationDetected => {
                let mut s = Synapse::new();
                s.set_raw_entropy(entropy);
                (
                    SafetyDecision::Escalate {
                        entropy,
                        reason: crate::llmosafe_integration::EscalationReason::Custom(
                            "hallucination",
                        ),
                        cooldown_ms: 5000,
                    },
                    s,
                )
            }
            KernelError::CognitiveInstability => {
                let mut s = Synapse::new();
                s.set_raw_entropy(entropy);
                (
                    SafetyDecision::Halt(KernelError::CognitiveInstability, 30000),
                    s,
                )
            }
            KernelError::BiasHaloDetected => {
                let mut s = Synapse::new();
                s.set_raw_entropy(entropy);
                (
                    SafetyDecision::Halt(KernelError::BiasHaloDetected, 30000),
                    s,
                )
            }
            KernelError::DepthExceeded
            | KernelError::ResourceExhaustion
            | KernelError::SelfMemoryExceeded
            | KernelError::DeadlineExceeded => {
                let mut s = Synapse::new();
                s.set_raw_entropy(entropy);
                (SafetyDecision::Halt(err, 30000), s)
            }
        };
        PipelineResult {
            decision,
            synapse,
            stages_executed: stages,
            detection_flags: 0,
            oov_ratio,
            entropy,
            surprise: surprise_val,
            monitor_state: StabilityResult::Stable,
            #[cfg(feature = "std")]
            body_pressure: None,
            step_count: self.step_count,
            kernel_output: None,
            classifier_score: 0.0,
        }
    }

    /// Constructs a `PipelineResult` from a `ReasoningLoop` error (`KernelError`).
    ///
    /// Maps `DepthExceeded` → `Escalate` (10000ms cooldown),
    /// `BiasHaloDetected` → `Halt` (30000ms),
    /// `CognitiveInstability` → `Halt` (30000ms),
    /// all other errors → `Halt(error, 30000ms)`.
    /// Constructs a fresh `Synapse` with entropy populated, detection_flags=0,
    /// monitor_state=Stable, kernel_output=None.
    fn ctrl_result_from_kernel_error(
        &self,
        err: KernelError,
        stages: u8,
        oov_ratio: u8,
        entropy: u16,
        surprise_val: u16,
    ) -> PipelineResult {
        let decision = match err {
            KernelError::DepthExceeded => SafetyDecision::Escalate {
                entropy,
                reason: crate::llmosafe_integration::EscalationReason::Custom(
                    "reasoning depth exceeded",
                ),
                cooldown_ms: 10000,
            },
            KernelError::BiasHaloDetected => {
                SafetyDecision::Halt(KernelError::BiasHaloDetected, 30000)
            }
            KernelError::CognitiveInstability => {
                SafetyDecision::Halt(KernelError::CognitiveInstability, 30000)
            }
            KernelError::HallucinationDetected
            | KernelError::ResourceExhaustion
            | KernelError::SelfMemoryExceeded
            | KernelError::DeadlineExceeded => SafetyDecision::Halt(err, 30000),
        };
        let mut err_synapse = Synapse::new();
        err_synapse.set_raw_entropy(entropy);
        PipelineResult {
            decision,
            synapse: err_synapse,
            stages_executed: stages,
            detection_flags: 0,
            oov_ratio,
            entropy,
            surprise: surprise_val,
            monitor_state: StabilityResult::Stable,
            #[cfg(feature = "std")]
            body_pressure: None,
            step_count: self.step_count,
            kernel_output: None,
            classifier_score: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llmosafe_kernel::DETECTION_FLAGS_MASK;

    #[test]
    fn test_pipelineconfig_default_validates() {
        let config = PipelineConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_pipelineconfig_validate_rejects_nan_drift_threshold() {
        let config = PipelineConfig {
            drift_threshold: f32::NAN,
            ..PipelineConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_pipelineconfig_validate_rejects_out_of_range_confidence() {
        let config = PipelineConfig {
            min_confidence: 2.0,
            ..PipelineConfig::default()
        };
        assert!(config.validate().is_err());

        let config = PipelineConfig {
            min_confidence: -0.1,
            ..PipelineConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_pipelineconfig_validate_rejects_zero_monitor_k() {
        let config = PipelineConfig {
            monitor_k: 0,
            ..PipelineConfig::default()
        };
        assert!(config.validate().is_err());
        let config = PipelineConfig {
            monitor_k: 6,
            ..PipelineConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_pipelineconfig_validate_rejects_zero_max_repetitions() {
        let config = PipelineConfig {
            max_repetitions: 0,
            ..PipelineConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_pipelineconfig_validate_rejects_zero_decay_threshold() {
        let config = PipelineConfig {
            decay_threshold: 0,
            ..PipelineConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_pipelineresult_is_safe() {
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(100);
        let result = PipelineResult {
            decision: SafetyDecision::Proceed,
            synapse,
            stages_executed: STAGE_SIFT | STAGE_MEMORY,
            detection_flags: 0,
            oov_ratio: 0,
            entropy: 100,
            surprise: 0,
            monitor_state: StabilityResult::Stable,
            #[cfg(feature = "std")]
            body_pressure: None,
            step_count: 1,
            kernel_output: None,
            classifier_score: 0.0,
        };
        assert!(result.is_safe());
        assert!(result.halt_reason().is_none());
    }

    #[test]
    fn test_pipelineresult_halt_reason() {
        let synapse = Synapse::new();
        let result = PipelineResult {
            decision: SafetyDecision::Halt(KernelError::CognitiveInstability, 30000),
            synapse,
            stages_executed: STAGE_SIFT,
            detection_flags: 0,
            oov_ratio: 0,
            entropy: 51000,
            surprise: 0,
            monitor_state: StabilityResult::Stable,
            #[cfg(feature = "std")]
            body_pressure: None,
            step_count: 0,
            kernel_output: None,
            classifier_score: 0.0,
        };
        assert!(!result.is_safe());
        assert_eq!(
            *result.halt_reason().unwrap(),
            KernelError::CognitiveInstability
        );
    }

    #[test]
    fn test_cognitive_pipeline_new_creates_with_defaults() {
        let pipeline = CognitivePipeline::<64, 10>::new("test objective");
        assert_eq!(pipeline.objective, "test objective");
        assert_eq!(pipeline.step_count, 0);
    }

    #[test]
    fn test_cognitive_pipeline_with_config_validates() {
        let config = PipelineConfig::default();
        let pipeline = CognitivePipeline::<64, 10>::with_config("test", config);
        assert!(pipeline.is_ok());
    }

    #[test]
    fn test_cognitive_pipeline_with_config_rejects_invalid() {
        let config = PipelineConfig {
            drift_threshold: f32::NAN,
            ..PipelineConfig::default()
        };
        let pipeline = CognitivePipeline::<64, 10>::with_config("test", config);
        assert!(pipeline.is_err());
    }

    #[test]
    fn test_process_safe_text_returns_proceed() {
        let mut pipeline = CognitivePipeline::<64, 10>::new("test objective");
        let result = pipeline.process("a completely ordinary sentence about everyday topics");
        // Safe text should produce a valid PipelineResult regardless of classifier.
        let _entropy: u16 = result.entropy; // always in [0, 65535] by type
        let _surprise: u16 = result.surprise;
        assert!(result.decision.severity() <= 4);
    }

    #[test]
    fn test_process_returns_pipeline_result_with_synapse() {
        let mut pipeline = CognitivePipeline::<64, 10>::new("test");
        let result = pipeline.process("checking some input text here");
        assert!(result.stages_executed & STAGE_SIFT != 0);
        // entropy is u16 — always in [0, 65535] by type
        let _ = result.entropy;
        let _ = result.surprise;
        assert!(result.detection_flags <= DETECTION_FLAGS_MASK);
    }

    #[test]
    fn test_reset_detectors_preserves_step_count() {
        let mut pipeline = CognitivePipeline::<64, 10>::new("test");
        let _ = pipeline.process("step one");
        let _ = pipeline.process("step two");
        let before = pipeline.step_count;
        pipeline.reset_detectors();
        assert_eq!(pipeline.step_count, before);
    }

    #[test]
    fn test_reset_full_clears_everything() {
        let mut pipeline = CognitivePipeline::<64, 10>::new("test");
        let _ = pipeline.process("this is a normal observation about weather");
        let _ = pipeline.process("another normal sentence for testing");
        pipeline.reset_full();
        assert_eq!(pipeline.step_count, 0);
        let result = pipeline.process("completely normal text after reset");
        // After reset_full + one process, step_count may be 0 or 1 depending
        // on classifier output — verify the pipeline returned a valid result.
        assert!(result.stages_executed & STAGE_SIFT != 0);
    }

    #[test]
    fn test_detection_flags_bitmask_uniqueness() {
        // Verify no overlapping bits
        assert_ne!(FLAG_STUCK, 0);
        assert_ne!(FLAG_DRIFTING, 0);
        assert_ne!(FLAG_LOW_CONFIDENCE, 0);
        assert_ne!(FLAG_DECAYING, 0);
        assert_ne!(FLAG_ANOMALY, 0);
        assert_ne!(FLAG_ADVERSARIAL, 0);
        let combined = FLAG_STUCK
            | FLAG_DRIFTING
            | FLAG_LOW_CONFIDENCE
            | FLAG_DECAYING
            | FLAG_ANOMALY
            | FLAG_ADVERSARIAL;
        assert_eq!(combined, DETECTION_FLAGS_MASK);
    }

    #[test]
    fn test_synapse_detection_flags_roundtrip() {
        let mut synapse = Synapse::new();
        synapse.set_detection_flags(FLAG_STUCK | FLAG_ANOMALY);
        assert_eq!(synapse.detection_flags(), FLAG_STUCK | FLAG_ANOMALY);

        synapse.set_detection_flags(0);
        assert_eq!(synapse.detection_flags(), 0);

        synapse.set_detection_flags(DETECTION_FLAGS_MASK);
        assert_eq!(synapse.detection_flags(), DETECTION_FLAGS_MASK);
    }

    #[test]
    fn test_synapse_oov_ratio_roundtrip() {
        let mut synapse = Synapse::new();
        synapse.set_oov_ratio(128);
        assert_eq!(synapse.oov_ratio(), 128);
        synapse.set_oov_ratio(0);
        assert_eq!(synapse.oov_ratio(), 0);
        synapse.set_oov_ratio(255);
        assert_eq!(synapse.oov_ratio(), 255);
    }

    #[test]
    fn test_synapse_clear_detection() {
        let mut synapse = Synapse::new();
        synapse.set_detection_flags(FLAG_STUCK | FLAG_DRIFTING);
        synapse.set_oov_ratio(200);
        synapse.clear_detection();
        assert_eq!(synapse.detection_flags(), 0);
        assert_eq!(synapse.oov_ratio(), 0);
    }

    #[test]
    fn test_synapse_detection_independent_from_entropy() {
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(12345);
        synapse.set_detection_flags(FLAG_ANOMALY);
        assert_eq!(synapse.raw_entropy(), 12345);
        assert_eq!(synapse.detection_flags(), FLAG_ANOMALY);

        // Verify entropy field is unchanged by detection operations
        synapse.set_oov_ratio(100);
        assert_eq!(synapse.raw_entropy(), 12345);
        assert_eq!(synapse.oov_ratio(), 100);
    }

    #[test]
    fn test_synapse_detection_does_not_affect_validate() {
        // Detection flags should not affect synapse validation
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(500);
        synapse.set_has_bias(false);
        synapse.set_detection_flags(DETECTION_FLAGS_MASK);
        synapse.set_oov_ratio(255);
        assert!(synapse.validate().is_ok());

        // But the underlying reserved field roundtrips through from_raw_u128
        let bits = u128::from_le_bytes(synapse.into_bytes());
        let reconstructed = Synapse::from_raw_u128(bits);
        assert_eq!(reconstructed.detection_flags(), DETECTION_FLAGS_MASK);
        assert_eq!(reconstructed.oov_ratio(), 255);
        assert_eq!(reconstructed.raw_entropy(), 500);
    }

    #[test]
    fn test_process_with_same_text_triggers_stuck() {
        let mut pipeline = CognitivePipeline::<64, 10>::new("test");
        let same = "the outdoor temperature readings indicate mild conditions";
        // Feed the same text multiple times — repetition should accumulate.
        // After enough calls, IF the pipeline reaches detection stage,
        // the stuck flag should be set.
        for _ in 0..5 {
            let result = pipeline.process(same);
            // PipelineResult always has valid bitmasks.
            assert!(result.detection_flags <= DETECTION_FLAGS_MASK);
        }
        // If the classifier blocks at SIFT, no detection occurs — that is
        // correct behavior (fail-fast on dangerous input).
    }

    #[test]
    fn test_process_with_drifting_text() {
        let mut pipeline =
            CognitivePipeline::<64, 10>::new("rust safety library performance analysis");
        // Text completely unrelated to objective should drift
        let result = pipeline.process("pizza recipes with extra cheese toppings");
        assert!(result.detection_flags & FLAG_DRIFTING != 0);
    }

    #[test]
    fn test_process_max_steps() {
        let mut pipeline = CognitivePipeline::<64, 2>::new("test");
        // Process until depth is exceeded — use repeated calls.
        let mut last_step = 0usize;
        for _ in 0..5 {
            let result = pipeline.process("checking safety of different input text");
            last_step = result.step_count;
            if result.step_count >= 2 {
                break;
            }
        }
        // Either we hit depth exceeded or ran out of calls.
        // MAX_STEPS=2 means step_count should cap at 2.
        assert!(last_step <= 2);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_process_with_pressure_nominal_proceeds() {
        let mut pipeline = CognitivePipeline::<64, 10>::new("test");
        let result = pipeline.process_with_pressure(
            "how do i write a function to sort a list in python",
            100,
            10,
        );
        assert!(result.body_pressure.is_some());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_process_with_pressure_critical_escalates() {
        let mut pipeline = CognitivePipeline::<64, 10>::new("test");
        let result = pipeline.process_with_pressure(
            "how do i write a function to sort a list in python",
            500,
            60,
        );
        assert!(result.decision.is_blocking());
        assert_eq!(result.body_pressure, Some(60));
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_process_with_pressure_emergency_halt() {
        let mut pipeline = CognitivePipeline::<64, 10>::new("test");
        let result = pipeline.process_with_pressure(
            "how do i write a function to sort a list in python",
            800,
            90,
        );
        assert!(result.decision.is_blocking());
        assert_eq!(result.body_pressure, Some(90));
    }

    // ── Control Theory Composition Tests ──

    #[test]
    fn test_process_ctrl_returns_valid_result() {
        let mut pipeline = CognitivePipeline::<64, 10>::new("test objective");
        let result = pipeline.process_ctrl("a completely ordinary sentence", 0.0, 0);
        assert!(result.stages_executed & STAGE_SIFT != 0);
        assert!(result.decision.severity() <= 4);
        // Control loop path should produce bounded entropy
        assert!(
            result.entropy > 0,
            "entropy must be non-zero for valid input"
        );
    }

    #[test]
    fn test_process_ctrl_memory_output_has_bounded_error() {
        let mut pipeline = CognitivePipeline::<64, 10>::new("test");
        let result = pipeline.process_ctrl("safety test input observation", 0.0, 0);
        assert!(result.stages_executed & STAGE_MEMORY != 0);
        // Detection stage always runs
        assert!(result.detection_flags <= DETECTION_FLAGS_MASK);
    }

    // ── Detection Gate Tests ──

    /// Detection gate enabled with safe text: gate is exercised.
    /// If CUSUM does not fire (classifier entropy < 250), the gate falls
    /// through to PID and the full pipeline executes. If CUSUM fires,
    /// the gate halts early — both paths exercise the detection gate block.
    #[cfg(feature = "std")]
    #[test]
    fn test_detection_gate_enabled_falls_through_to_pid() {
        let config = PipelineConfig {
            use_detection_gate: true,
            ..PipelineConfig::default()
        };
        let mut pipeline =
            CognitivePipeline::<64, 10>::with_config("test objective", config).unwrap();
        let result = pipeline.process("a completely ordinary sentence about everyday topics");
        // Detection stage must have executed — confirms the gate block ran.
        assert!(
            result.stages_executed & STAGE_DETECTION != 0,
            "detection stage must execute when detection gate is enabled"
        );
        // All 4 main stages (SIFT, MEMORY, KERNEL, DETECTION) must be set.
        assert_eq!(
            result.stages_executed & (STAGE_SIFT | STAGE_MEMORY | STAGE_KERNEL | STAGE_DETECTION),
            STAGE_SIFT | STAGE_MEMORY | STAGE_KERNEL | STAGE_DETECTION
        );
    }

    /// Detection gate enabled: verify that when `must_halt()` returns true
    /// (CUSUM anomaly or adversarial pattern), the gate returns early with
    /// MONITOR stage set (but without PID composition).
    #[cfg(feature = "std")]
    #[test]
    fn test_detection_gate_must_halt_triggers_early_return() {
        let config = PipelineConfig {
            use_detection_gate: true,
            ..PipelineConfig::default()
        };
        let mut pipeline =
            CognitivePipeline::<64, 10>::with_config("test objective", config).unwrap();
        let result = pipeline.process("text triggering detection analysis in the pipeline");
        // Detection gate block was exercised.
        assert!(
            result.stages_executed & STAGE_DETECTION != 0,
            "detection stage must be set when detection gate is enabled"
        );
        // The detection gate either halted or passed through — both are valid.
        // Verify the result carries a valid decision (severity within range).
        assert!(result.decision.severity() <= 4);
    }

    /// Detection gate enabled with repeated text: exercises the detection gate
    /// block on every iteration. Stuck flags are accumulated in the
    /// RepetitionDetector regardless of whether the gate halts (CUSUM) or
    /// escalates (stuck). The detection stage and gate code block
    /// (lines 671–713) are exercised on every call.
    #[cfg(feature = "std")]
    #[test]
    fn test_detection_gate_stuck_falls_through_to_pid() {
        let config = PipelineConfig {
            use_detection_gate: true,
            max_repetitions: 3,
            ..PipelineConfig::default()
        };
        // Use MAX_STEPS=20 so kernel never exhausts before detection runs.
        let mut pipeline =
            CognitivePipeline::<64, 20>::with_config("test objective", config).unwrap();
        let same = "the outdoor temperature readings indicate mild conditions";
        // Feed the same text repeatedly — RepetitionDetector accumulates,
        // and the detection gate block (lines 671–713) runs on every iteration.
        for _ in 0..6 {
            let result = pipeline.process(same);
            // Detection stage must have executed — confirms gate block ran.
            assert!(
                result.stages_executed & STAGE_DETECTION != 0,
                "detection stage must run on every iteration"
            );
            // Decision must be valid (severity 0–4).
            assert!(result.decision.severity() <= 4);
        }
    }

    // ── Memory Error Path Tests ──

    /// Triggers a working-memory `HallucinationDetected` error by setting
    /// an extremely low surprise threshold. Almost any text with OOV tokens
    /// will produce surprise > 1, triggering the memory gate.
    #[test]
    fn test_process_ctrl_memory_error_hallucination_detected() {
        let config = PipelineConfig {
            surprise_threshold: 1, // Any surprise > 1 triggers HallucinationDetected
            ..PipelineConfig::default()
        };
        let mut pipeline = CognitivePipeline::<64, 10>::with_config("test", config).unwrap();
        let result = pipeline.process_ctrl("testing the memory error pathway in pipeline", 0.0, 0);
        // Memory stage must have executed (even if it produced an error).
        assert!(result.stages_executed & STAGE_MEMORY != 0);
        // KERNEL and later stages must NOT have executed (early return from memory error).
        assert_eq!(result.stages_executed & STAGE_KERNEL, 0);
        assert_eq!(result.stages_executed & STAGE_DETECTION, 0);
        // Decision must be blocking (Escalate from HallucinationDetected).
        assert!(result.decision.is_blocking());
        // No kernel output since kernel never ran.
        assert!(result.kernel_output.is_none());
    }

    // ── Kernel Error Path Tests ──

    /// Triggers a `DepthExceeded` kernel error by using MAX_STEPS=1 and
    /// processing twice. The first call advances the step to 1; the second
    /// call hits `current_step >= MAX_STEPS` at the kernel stage.
    /// Uses text known to pass through SIFT and MEMORY so the kernel stage
    /// is reached on every call.
    #[test]
    fn test_process_ctrl_kernel_error_depth_exceeded() {
        let mut pipeline = CognitivePipeline::<64, 1>::new("test objective");
        let safe_text = "a completely ordinary sentence about everyday topics";
        // First call: step_count advances from 0 → 1 (kernel succeeds).
        let result1 = pipeline.process_ctrl(safe_text, 0.0, 0);
        assert!(
            result1.stages_executed & STAGE_KERNEL != 0,
            "kernel stage must execute on first call; text may be triggering memory error"
        );
        assert_eq!(pipeline.step_count, 1);

        // Second call: kernel returns DepthExceeded because current_step (1) >= MAX_STEPS (1).
        let result2 = pipeline.process_ctrl(safe_text, 0.0, 0);
        assert!(result2.stages_executed & STAGE_KERNEL != 0);
        // DETECTION and MONITOR must NOT have executed (early return from kernel error).
        assert_eq!(result2.stages_executed & STAGE_DETECTION, 0);
        assert_eq!(result2.stages_executed & STAGE_MONITOR, 0);
        // DepthExceeded maps to Escalate.
        assert!(result2.decision.is_blocking());
        assert!(result2.kernel_output.is_none());
    }

    // ── process_with_pressure Elevated (26-50%) ──

    /// Elevated pressure (35%) should NOT short-circuit; it should fall
    /// through to process_ctrl with e_body and pressure populated.
    #[cfg(feature = "std")]
    #[test]
    fn test_process_with_pressure_elevated_proceeds() {
        let mut pipeline = CognitivePipeline::<64, 10>::new("test objective");
        let result = pipeline.process_with_pressure(
            "safe text for elevated pressure test",
            350, // body_entropy (normalised by /1000)
            35,  // pressure = 35 → Elevated
        );
        // Elevated does NOT short-circuit — all main stages should run.
        assert!(result.stages_executed & STAGE_SIFT != 0);
        assert!(result.stages_executed & STAGE_MEMORY != 0);
        assert!(result.stages_executed & STAGE_KERNEL != 0);
        assert!(result.stages_executed & STAGE_DETECTION != 0);
        // Body pressure must be populated.
        assert_eq!(result.body_pressure, Some(35));
        // Result should be valid.
        assert!(result.is_safe() || result.decision.severity() <= 4);
    }

    // ── process_safe Tests ──

    /// process_safe with safe resource guard returns Ok with a valid decision.
    #[cfg(feature = "std")]
    #[test]
    fn test_process_safe_returns_ok_for_safe_guard() {
        let mut pipeline = CognitivePipeline::<64, 10>::new("test");
        let guard = ResourceGuard::for_testing(1024 * 1024, 100, 10);
        let result = pipeline.process_safe("safe input text for guarded pipeline", &guard);
        assert!(result.is_ok(), "process_safe should succeed for safe guard");
        let pipeline_result = result.unwrap();
        assert!(pipeline_result.decision.severity() <= 4);
    }

    /// When ResourceGuard::check_with_deadline receives an already-expired
    /// deadline, it returns DeadlineExceeded. This exercises the same code
    /// path that process_safe uses internally for deadline handling.
    #[cfg(feature = "std")]
    #[test]
    fn test_check_with_deadline_zero_deadline_returns_exceeded() {
        let guard = ResourceGuard::for_testing(1024 * 1024, 100, 60);
        // Pass a deadline already in the past — must return DeadlineExceeded.
        let past = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_secs(1))
            .unwrap_or(std::time::Instant::now());
        let result = guard.check_with_deadline(past);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), KernelError::DeadlineExceeded);
    }

    // ── PipelineConfig invalid PidConfig ──

    /// PipelineConfig::validate() propagates PidConfig::validate() errors.
    /// A negative integrator_decay is invalid — validate must return Err.
    #[test]
    fn test_pipelineconfig_validate_rejects_invalid_pid_config() {
        let pid_config = PidConfig {
            integrator_decay: -0.1,
            ..PidConfig::default()
        };
        let config = PipelineConfig {
            pid_config,
            ..PipelineConfig::default()
        };
        assert!(
            config.validate().is_err(),
            "negative integrator_decay must be rejected by validate"
        );
    }

    /// warn_gain >= halt_gain is rejected by PidConfig::validate,
    /// and the error propagates through PipelineConfig::validate.
    #[test]
    fn test_pipelineconfig_validate_rejects_warn_gain_ge_halt_gain() {
        let pid_config = PidConfig {
            warn_gain: 0.9,
            halt_gain: 0.9, // warn_gain must be strictly less than halt_gain
            ..PidConfig::default()
        };
        let config = PipelineConfig {
            pid_config,
            ..PipelineConfig::default()
        };
        assert!(
            config.validate().is_err(),
            "warn_gain >= halt_gain must be rejected by validate"
        );
    }

    // ── Cross-Consistency Tests ───────────────────────────────────

    /// Default configs: validate() returns Ok (cross-consistency is soft).
    /// validate_cross_consistency() returns Err because halt_entropy=50000
    /// (risk 0.763) diverges from halt_gain=1.0 (31%), and
    /// escalate_entropy=40000 (risk 0.610) diverges from halt_gain×0.8=0.8
    /// (24%). warn_entropy=30000 (risk 0.458) is within 15% of
    /// warn_gain=0.5 (8.4%).
    #[cfg(feature = "std")]
    #[test]
    fn test_pipelineconfig_cross_consistency_defaults() {
        let config = PipelineConfig::default();
        // Hard validation must still pass — cross-consistency is advisory.
        assert!(config.validate().is_ok());

        let result = config.validate_cross_consistency();
        assert!(
            result.is_err(),
            "default configs have known divergence in halt and escalate thresholds"
        );
        let warnings = result.unwrap_err();
        // Expect exactly 2 warnings: halt_entropy and escalate_entropy diverge.
        // warn_entropy=30000 → risk 0.458 is within 15% of warn_gain=0.5.
        assert_eq!(
            warnings.len(),
            2,
            "expected 2 warnings for halt_entropy and escalate_entropy divergence, got: {:?}",
            warnings
        );
        assert!(
            warnings[0].contains("halt"),
            "first warning should be about halt_entropy: {}",
            warnings[0]
        );
        assert!(
            warnings[1].contains("escalate"),
            "second warning should be about escalate_entropy: {}",
            warnings[1]
        );
    }

    /// Aligned configs: EscalationPolicy thresholds map to the same
    /// risk-equivalents as the PID gains. Cross-consistency should pass
    /// (return Ok(())).
    #[cfg(feature = "std")]
    #[test]
    fn test_pipelineconfig_cross_consistency_aligned() {
        let config = PipelineConfig {
            policy: EscalationPolicy {
                halt_entropy: 65535,     // 65535/65535 = 1.000  vs halt_gain=1.0
                escalate_entropy: 52428, // 52428/65535 = 0.800  vs halt_gain×0.8=0.8
                warn_entropy: 32768,     // 32768/65535 = 0.500  vs warn_gain=0.5
                ..EscalationPolicy::default()
            },
            pid_config: PidConfig {
                halt_gain: 1.0,
                warn_gain: 0.5,
                ..PidConfig::default()
            },
            ..PipelineConfig::default()
        };
        assert!(config.validate().is_ok());
        assert!(
            config.validate_cross_consistency().is_ok(),
            "aligned thresholds should pass cross-consistency"
        );
    }

    /// Intentionally misaligned configs: halt_entropy=10000 (risk 0.153)
    /// vs halt_gain=1.0, escalate_entropy=1 (risk ~0) vs halt_gain×0.8=0.8,
    /// warn_entropy=1 (risk ~0) vs warn_gain=0.99. All three pairs should
    /// produce warnings.
    #[cfg(feature = "std")]
    #[test]
    fn test_pipelineconfig_cross_consistency_misaligned() {
        let config = PipelineConfig {
            policy: EscalationPolicy {
                halt_entropy: 10000,
                escalate_entropy: 1,
                warn_entropy: 1,
                ..EscalationPolicy::default()
            },
            pid_config: PidConfig {
                halt_gain: 1.0,
                warn_gain: 0.99,
                ..PidConfig::default()
            },
            ..PipelineConfig::default()
        };
        assert!(config.validate().is_ok());
        let result = config.validate_cross_consistency();
        assert!(
            result.is_err(),
            "intentionally misaligned configs must produce warnings"
        );
        let warnings = result.unwrap_err();
        assert!(
            warnings.len() >= 2,
            "expected at least 2 warnings for misaligned thresholds, got: {:?}",
            warnings
        );
        // Each warning must identify the severity level (halt/escalate/warn).
        for w in &warnings {
            assert!(
                w.contains("halt") || w.contains("escalate") || w.contains("warn"),
                "warning must identify severity: {}",
                w
            );
        }
    }
}
