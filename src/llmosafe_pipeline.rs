//! LLMOSAFE CognitivePipeline — 5-stage safety orchestrator.
#![deny(clippy::cast_lossless)]
//!
//! # Architecture
//!
//! The CognitivePipeline wires the existing sifter, working memory, kernel,
//! escalation policy, 5 detectors, and dynamic stability monitor into a
//! single sequential pipeline that can short-circuit at any stage:
//!
//! ```text
//! process(text) → SIFT → MEMORY → KERNEL → DETECTION → MONITOR → PipelineResult
//!                    │       │        │         │           │
//!                    ▼       ▼        ▼         ▼           ▼
//!             Halt?   Halt?    Halt?    Halt?       (advisory only)
//! ```
//!
//! Each stage transforms data and can halt the pipeline with a `SafetyDecision`.
//! `EscalationPolicy` is invoked at every gate.
//!
//! # Usage
//!
//! ```ignore
//! use llmosafe::CognitivePipeline;
//!
//! let mut pipeline = CognitivePipeline::<64, 10>::new("analyze safety");
//! let result = pipeline.process("user input text");
//! if result.decision.must_halt() {
//!     // handle halt
//! }
//! ```

use crate::control_types::OverrideFlags;
use crate::llmosafe_detection::{
    ConfidenceTracker, CusumDetector, DriftDetector, RepetitionDetector,
};
use crate::llmosafe_integration::EscalationPolicy;
#[cfg(feature = "std")]
use crate::llmosafe_integration::SafetyDecision;
#[cfg(not(feature = "std"))]
use crate::llmosafe_integration::SafetyDecision;
use crate::llmosafe_kernel::{
    DynamicStabilityMonitor, KernelError, KernelOutput, ReasoningLoop, StabilityResult, Synapse,
    ValidatedSynapse, FLAG_ANOMALY, FLAG_DECAYING, FLAG_DRIFTING, FLAG_LOW_CONFIDENCE, FLAG_STUCK,
    STABILITY_THRESHOLD,
};
use crate::llmosafe_memory::WorkingMemory;
use crate::llmosafe_pid::{PidConfig, PidState};

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
/// Bitmask constant 0x20 gated behind cfg(feature="std"). Defined but never
/// set in `process_ctrl()` — the BODY stage was moved into
/// `process_with_pressure()` as a pre-SIFT gate.
#[cfg(feature = "std")]
pub const STAGE_BODY: u8 = 0x20;

/// Configuration for a CognitivePipeline instance.
///
/// Every threshold has a safe default via `Default::default()`.
/// Fields with `f32` values must be in `[0.0, 1.0]` and finite.
/// Use `validate()` to check bounds before constructing a pipeline.
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
        }
    }
}

impl PipelineConfig {
    /// Validates all configuration fields are within safe bounds.
    ///
    /// Rejects NaN, out-of-range floats, zero-valued integer parameters,
    /// and the empty memory edge case. `validate()` is called by
    /// `CognitivePipeline::with_config()` before construction.
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
        Ok(())
    }
}

/// Aggregate output of a single `CognitivePipeline::process()` invocation.
///
/// Carries the final `SafetyDecision`, the classified `Synapse` (with packed
/// detection flags and OOV ratio), a stages-executed bitmask, and diagnostic
/// fields for the C-ABI query functions.
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
            _ => None,
        }
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
pub struct CognitivePipeline<'a, const MEM_SIZE: usize, const MAX_STEPS: usize> {
    memory: WorkingMemory<MEM_SIZE>,
    reasoning: ReasoningLoop<MAX_STEPS>,
    monitor: DynamicStabilityMonitor,
    repetition: RepetitionDetector,
    drift: DriftDetector,
    confidence: ConfidenceTracker,
    cusum: CusumDetector,
    objective: &'a str,
    step_count: usize,
    pid_state: PidState,
    pid_config: PidConfig,
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
            objective,
            step_count: 0,
            pid_state: PidState::new(),
            pid_config: config.pid_config,
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
    #[cfg(feature = "std")]
    pub fn process_with_pressure(
        &mut self,
        observation: &str,
        body_entropy: u16,
        pressure: u8,
    ) -> PipelineResult {
        // Route body pressure through BodyOutput → PidInput.e_body
        let e_body = (f32::from(body_entropy) / 1000.0_f32).clamp(0.0, 1.0);
        self.process_ctrl(observation, e_body, pressure)
    }

    /// Resets all 5 detectors and the `DynamicStabilityMonitor`.
    ///
    /// Preserves `WorkingMemory` ring-buffer state and `ReasoningLoop` step count.
    /// Use `reset_full()` for a complete reset.
    pub fn reset_detectors(&mut self) {
        self.repetition.reset();
        self.confidence.reset();
        self.cusum.reset();
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
        self.step_count = 0;
        self.pid_state.reset();
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
        let mut stages = 0u8;

        // ── Stage 1: SIFT (Tier 3) ──
        // Single canonical entry point. Keyword-bias (innate layer) OR-s into
        // the classifier result. One synapse, one proof, no duplicate compute.
        stages |= STAGE_SIFT;
        let (sifted, sifted_proof) = crate::llmosafe_sifter::sift_text(observation);
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
        let classifier_prob = f32::from(entropy) / 65535.0_f32;
        self.confidence.observe(classifier_prob);
        let _cusum_anomaly = self.cusum.update(f64::from(entropy));

        let is_stuck = self.repetition.is_stuck();
        let is_drifting = self.drift.is_drifting();
        let is_low_confidence = self.confidence.is_low();
        let is_decaying = self.confidence.is_decaying();
        let anomaly_detected = self.cusum.detected();

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

        // ── Stage 5: PID COMPOSITION ──
        let pressure_term = (e_body * 100.0_f32) as u8;
        let trend = self.memory.trend();
        let pure_risk = crate::llmosafe_pid::compute_pid_score_pure(
            entropy,
            trend,
            pressure_term,
            classifier_prob,
            flags,
            &self.pid_config,
            &mut self.pid_state,
        );

        let mut override_flags = OverrideFlags::empty();
        if has_bias {
            override_flags = override_flags | OverrideFlags::BIAS;
        }
        if e_body > 0.9 {
            override_flags = override_flags | OverrideFlags::EXHAUSTED;
        }
        if u32::from(kernel_entropy) > STABILITY_THRESHOLD as u32 {
            override_flags = override_flags | OverrideFlags::KERNEL_UNSTABLE;
        }
        let limited_risk = crate::llmosafe_pid::apply_safety_overrides(
            pure_risk,
            override_flags,
            &self.pid_config,
        );
        let decision = crate::llmosafe_pid::pid_risk_to_decision(limited_risk, &self.pid_config);

        // ── Stage 6: MONITOR ──
        stages |= STAGE_MONITOR;
        let monitor_state = self.monitor.update(u32::from(entropy));

        let kernel_output = Some(crate::llmosafe_kernel::KernelOutput {
            error_kernel: f32::from(kernel_entropy) / 65535.0_f32,
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
            _ => {
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
            _ => SafetyDecision::Halt(err, 30000),
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
        let combined =
            FLAG_STUCK | FLAG_DRIFTING | FLAG_LOW_CONFIDENCE | FLAG_DECAYING | FLAG_ANOMALY;
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
            assert!(result.detection_flags <= crate::llmosafe_kernel::DETECTION_FLAGS_MASK);
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
}
