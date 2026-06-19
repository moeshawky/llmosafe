// Arithmetic in this module operates on bounded decision/probability values
// [0, 65535] where saturating/wrapping semantics are the intended behavior.
// DO-178C: these operations are verified safe by value range analysis at
// the module boundary — inputs are always validated before arithmetic.
#![allow(clippy::arithmetic_side_effects)]

//! Escalation policy and decision primitives.
//!
//! Maps entropy/surprise/bias/detection signals to typed `SafetyDecision`
//! outcomes. The `EscalationPolicy` is the central threshold engine used by
//! `CognitivePipeline` at every stage gate.
//!
//! # Key Types
//!
//! - `SafetyDecision` — enum: `Proceed`, `Warn`, `Escalate`, `Halt`, `Exit`.
//!   Each variant carries a severity (0–4) and cooldown (0, 5000, or 30000ms).
//! - `EscalationPolicy` — configurable thresholds for entropy, surprise, bias,
//!   and pressure. Builder pattern. Default thresholds calibrated for the
//!   classifier range [0, 65535].
//! - `PressureLevel` — `Nominal`/`Elevated`/`Critical`/`Emergency` from
//!   percentage (0–100). `Critical` (51–75%) triggers escalation.
//! - `EscalationReason` — reason code for `Escalate` variant (entropy, surprise,
//!   bias, resource pressure, stuck agent, drift, adversarial, etc.).
//! - `SafetyContext` — thread-local accumulator for multi-observation decisions
//!   (`std` only). Tracks max entropy, max surprise, any bias across observations.
//!
//! # Default Thresholds
//!
//! | Gauge | Warn | Escalate | Halt |
//! |-------|------|----------|------|
//! | Entropy | 30000 | 40000 | 50000 |
//! | Surprise | 42600 | 55700 | — |
//! | Bias | — | Escalate | — |
//!
//! # DAL Gating
//!
//! `EscalationPolicy::apply_dal_to_decision()` downgrades decisions at runtime
//! based on Design Assurance Level (A–E). DAL A passes all decisions through;
//! DAL E converts everything to Proceed.

use crate::control_types::DesignAssuranceLevel;
use crate::llmosafe_kernel::{CognitiveStability, KernelError};

#[cfg(feature = "std")]
use crate::llmosafe_detection::DetectionResult;

/// Safety decision outcome from the cognitive safety pipeline.
///
/// This enum represents the decision flow for a processed input,
/// from "proceed normally" through escalating severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SafetyDecision {
    /// Input is safe, proceed with normal processing.
    Proceed,
    /// Input has elevated metrics but within tolerance. Log and proceed.
    /// Contains the reason for the warning.
    Warn(&'static str),
    /// Input requires escalation to a higher-level handler.
    /// System should prepare fallback or human review.
    Escalate {
        entropy: u16,
        reason: EscalationReason,
        cooldown_ms: u32,
    },
    /// Input must be rejected. Halt processing immediately.
    /// Contains the kernel error that triggered the halt.
    Halt(KernelError, u32),
    /// Unrecoverable error - must terminate immediately.
    /// Contains the kernel error that triggered the exit.
    Exit(KernelError),
}

impl SafetyDecision {
    /// Returns true if processing can continue.
    ///
    /// # Inputs
    /// * `self`: safety decision variant.
    ///
    /// # Outputs
    /// Returns `true` if decision is `Proceed` or `Warn`. Returns `false` otherwise.
    pub fn can_proceed(&self) -> bool {
        matches!(self, Self::Proceed | Self::Warn(_))
    }

    /// Returns true if processing must stop.
    ///
    /// # Inputs
    /// * `self`: safety decision variant.
    ///
    /// # Outputs
    /// Returns `true` if decision is `Halt`. Returns `false` otherwise.
    pub fn must_halt(&self) -> bool {
        matches!(self, Self::Halt(..))
    }

    /// Returns the severity level (0=Proceed, 1=Warn, 2=Escalate, 3=Halt, 4=Exit).
    ///
    /// # Inputs
    /// * `self`: safety decision variant.
    ///
    /// # Outputs
    /// Returns a `u8` severity indicator in the range `[0, 4]`.
    pub fn severity(&self) -> u8 {
        match self {
            Self::Proceed => 0,
            Self::Warn(_) => 1,
            Self::Escalate { .. } => 2,
            Self::Halt(..) => 3,
            Self::Exit(_) => 4,
        }
    }

    /// Returns true if this decision requires blocking/throttling.
    ///
    /// # Inputs
    /// * `self`: safety decision variant.
    ///
    /// # Outputs
    /// Returns `true` if decision is `Escalate`, `Halt`, or `Exit`. Returns `false` otherwise.
    pub fn is_blocking(&self) -> bool {
        matches!(self, Self::Escalate { .. } | Self::Halt(..) | Self::Exit(_))
    }

    /// Returns true if process must terminate immediately.
    ///
    /// # Inputs
    /// * `self`: safety decision variant.
    ///
    /// # Outputs
    /// Returns `true` if decision is `Exit`. Returns `false` otherwise.
    pub fn should_exit(&self) -> bool {
        matches!(self, Self::Exit(_))
    }

    /// Returns recommended cooldown in milliseconds.
    ///
    /// # Inputs
    /// * `self`: safety decision variant.
    ///
    /// # Outputs
    /// Returns cooldown in milliseconds (`u32`). Defaults to 0 for non-blocking decisions.
    pub fn recommended_cooldown_ms(&self) -> u32 {
        match self {
            Self::Proceed | Self::Warn(_) | Self::Exit(_) => 0,
            Self::Escalate { cooldown_ms, .. } => *cooldown_ms,
            Self::Halt(_, cooldown_ms) => *cooldown_ms,
        }
    }

    /// Returns machine-readable status label.
    ///
    /// # Inputs
    /// * `self`: safety decision variant.
    ///
    /// # Outputs
    /// Returns static string slice: `"safe"`, `"warning"`, `"escalate"`, `"halt"`, or `"exit"`.
    pub fn status_label(&self) -> &'static str {
        match self {
            Self::Proceed => "safe",
            Self::Warn(_) => "warning",
            Self::Escalate { .. } => "escalate",
            Self::Halt(..) => "halt",
            Self::Exit(_) => "exit",
        }
    }
}

/// Reason for escalation when SafetyDecision is Escalate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EscalationReason {
    /// Entropy approaching threshold.
    EntropyApproachingLimit,
    /// Surprise level elevated.
    SurpriseElevated,
    /// Bias signals detected.
    BiasDetected,
    /// Resource pressure elevated.
    ResourcePressure,
    /// Anomalous pattern detected by Cusum.
    AnomalyDetected,
    /// Custom reason.
    Custom(&'static str),
    /// Agent appears stuck in a loop.
    StuckAgent,
    /// Goal has drifted significantly from original objective.
    GoalDriftDetected,
    /// Model confidence is decaying rapidly.
    ConfidenceDecaying,
    /// Adversarial prompt patterns detected.
    AdversarialDetected,
}

/// Resource pressure level mapping.
///
/// Maps raw pressure percentage to semantic levels for decision-making.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PressureLevel {
    /// Pressure 0-25%: System is healthy.
    Nominal,
    /// Pressure 26-50%: Minor concern, monitor.
    Elevated,
    /// Pressure 51-75%: Significant concern, consider action.
    Critical,
    /// Pressure 76-100%: Emergency, must act immediately.
    Emergency,
}

impl PressureLevel {
    /// Convert a percentage (0-100) to a PressureLevel.
    ///
    /// # Inputs
    /// * `pct`: raw percentage value.
    ///
    /// # Outputs
    /// Returns mapped `PressureLevel`.
    ///
    /// # Invariants
    /// * Clamps any values greater than 100 to `Emergency`.
    pub fn from_percentage(pct: u8) -> Self {
        match pct {
            0..=25 => Self::Nominal,
            26..=50 => Self::Elevated,
            51..=75 => Self::Critical,
            76..=100 => Self::Emergency,
            _ => Self::Emergency, // Clamp over 100
        }
    }

    /// Returns true if pressure requires immediate attention.
    ///
    /// # Inputs
    /// * `self`: pressure level.
    ///
    /// # Outputs
    /// Returns `true` if level is `Critical` or `Emergency`. Returns `false` otherwise.
    pub fn requires_action(&self) -> bool {
        matches!(self, Self::Critical | Self::Emergency)
    }
}

impl From<u8> for PressureLevel {
    fn from(pct: u8) -> Self {
        Self::from_percentage(pct)
    }
}

/// Configurable policy for mapping entropy/surprise/bias to decisions.
///
/// The EscalationPolicy defines the thresholds at which inputs transition
/// from Proceed → Warn → Escalate → Halt.
// Internal escalation decision engine used by CognitivePipeline.
// The `dal` field gates decision severity at runtime (DO-178C tiers A–E).
// External consumers should prefer CognitivePipeline which wraps this.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EscalationPolicy {
    /// Entropy threshold for Warn (default: 30000, recalibrated for classifier probability space).
    pub warn_entropy: u16,
    /// Entropy threshold for Escalate (default: 40000).
    pub escalate_entropy: u16,
    /// Entropy threshold for Halt (default: 50000). Overrides Escalate and Warn.
    pub halt_entropy: u16,
    /// Surprise threshold for Warn (default: 300).
    pub warn_surprise: u16,
    /// Surprise threshold for Escalate (default: 500).
    pub escalate_surprise: u16,
    /// Whether bias detection triggers immediate escalation.
    pub bias_escalates: bool,
    /// Pressure level that triggers Escalate.
    pub escalate_pressure: PressureLevel,
    /// Design Assurance Level (DO-178C) controlling runtime escalation behavior.
    ///
    /// This is the output-side gate — it downgrades or suppresses decisions
    /// after the PID control loop has computed a risk score. The `dal` feature
    /// (compile-time) controls `apply_safety_overrides` inside the PID loop.
    /// This field controls `decide_from_detection()` and `decide_with_pressure()`
    /// at runtime.
    ///
    /// | DAL | Effect |
    /// |-----|--------|
    /// | A   | No suppression — Halt/Escalate/Warn/Proceed pass through |
    /// | B   | Halt → Escalate; Escalate/Warn/Proceed pass through |
    /// | C   | Halt → Warn; Escalate → Warn; Warn/Proceed pass through |
    /// | D   | Halt/Escalate → Warn; Warn/Proceed pass through |
    /// | E   | All decisions → Proceed (no enforcement) |
    ///
    /// Default: A (no runtime gating). Both the compile-time `dal` feature AND
    /// a runtime DAL of A must be active for Halt decisions to reach the actuator.
    pub dal: DesignAssuranceLevel,
}

impl Default for EscalationPolicy {
    fn default() -> Self {
        Self {
            warn_entropy: 30000,
            escalate_entropy: 40000,
            halt_entropy: 50000,
            warn_surprise: 42600,
            escalate_surprise: 55700,
            bias_escalates: true,
            escalate_pressure: PressureLevel::Critical,
            dal: DesignAssuranceLevel::A,
        }
    }
}

impl EscalationPolicy {
    /// Create a new policy with default thresholds.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set warn entropy threshold.
    pub const fn with_warn_entropy(mut self, threshold: u16) -> Self {
        self.warn_entropy = threshold;
        self
    }

    /// Builder: set escalate entropy threshold.
    pub const fn with_escalate_entropy(mut self, threshold: u16) -> Self {
        self.escalate_entropy = threshold;
        self
    }

    /// Builder: set halt entropy threshold.
    pub const fn with_halt_entropy(mut self, threshold: u16) -> Self {
        self.halt_entropy = threshold;
        self
    }

    /// Builder: set bias escalation behavior.
    pub const fn with_bias_escalates(mut self, value: bool) -> Self {
        self.bias_escalates = value;
        self
    }

    /// Builder: set Design Assurance Level for runtime decision gating.
    pub const fn with_dal(mut self, dal: DesignAssuranceLevel) -> Self {
        self.dal = dal;
        self
    }

    /// Evaluate entropy, surprise, and bias flags to produce a DAL-gated decision.
    ///
    /// Delegates to `canonical_decision` for threshold checks with universal
    /// [`DesignAssuranceLevel`] gating. All return paths pass through
    /// `apply_dal_to_decision` so DAL is enforced uniformly.
    ///
    /// # Inputs
    /// * `entropy`: `u16` — raw entropy in `[0, 65535]`.
    /// * `surprise`: `u16` — raw surprise in `[0, 65535]`.
    /// * `has_bias`: `bool` — `true` if bias was detected.
    ///
    /// # Outputs
    /// Returns a `SafetyDecision` gated by the configured DAL.
    ///
    /// # Threshold Semantics
    /// All thresholds use inclusive comparison (`>=`) — entropy equal to
    /// `halt_entropy` triggers Halt, entropy equal to `escalate_entropy`
    /// triggers Escalate.
    pub fn decide(&self, entropy: u16, surprise: u16, has_bias: bool) -> SafetyDecision {
        self.canonical_decision(entropy, surprise, has_bias)
    }

    /// Evaluate entropy/surprise/bias with a resource pressure signal.
    ///
    /// Halt-entropy is checked first (inclusive `>=`, consistent with
    /// `canonical_decision`). If not at the Halt threshold, resource
    /// pressure that meets or exceeds `escalate_pressure` triggers an
    /// `Escalate(ResourcePressure)` decision. When neither Halt nor
    /// pressure escalation fires, delegates to `canonical_decision`
    /// for the standard threshold ladder with universal DAL gating.
    ///
    /// # Inputs
    /// * `entropy`: `u16` — raw entropy in `[0, 65535]`.
    /// * `surprise`: `u16` — raw surprise in `[0, 65535]`.
    /// * `has_bias`: `bool` — `true` if bias was detected.
    /// * `pressure`: `PressureLevel` — resource pressure level.
    ///
    /// # Outputs
    /// Returns a `SafetyDecision` gated by the configured DAL.
    ///
    /// # Threshold Semantics
    /// All threshold comparisons use `>=` — entropy ≥ `halt_entropy`
    /// triggers Halt, pressure ≥ `escalate_pressure` triggers Escalate.
    pub fn decide_with_pressure(
        &self,
        entropy: u16,
        surprise: u16,
        has_bias: bool,
        pressure: PressureLevel,
    ) -> SafetyDecision {
        // Halt takes priority over everything — inclusive check
        // matches canonical_decision semantics.
        if entropy >= self.halt_entropy {
            return self.apply_dal_to_decision(SafetyDecision::Halt(
                KernelError::CognitiveInstability,
                30000,
            ));
        }

        // Pressure escalation (only when not at Halt threshold)
        if pressure >= self.escalate_pressure {
            return self.apply_dal_to_decision(SafetyDecision::Escalate {
                entropy,
                reason: EscalationReason::ResourcePressure,
                cooldown_ms: 5000,
            });
        }

        self.canonical_decision(entropy, surprise, has_bias)
    }

    /// Evaluate from CognitiveStability.
    pub fn decide_from_stability(&self, stability: CognitiveStability) -> SafetyDecision {
        match stability {
            CognitiveStability::Stable => SafetyDecision::Proceed,
            CognitiveStability::Pressure => SafetyDecision::Warn("cognitive pressure detected"),
            CognitiveStability::Unstable => {
                SafetyDecision::Halt(KernelError::CognitiveInstability, 30000)
            }
        }
    }

    /// Canonical decision engine — unified threshold ladder with universal DAL gating.
    ///
    /// Every public decision method routes through this single implementation.
    /// Thresholds are evaluated in severity order (Halt → Escalate → Warn →
    /// Proceed) using inclusive `>=` comparison at every boundary.  All return
    /// paths pass through `apply_dal_to_decision` so the configured
    /// [`DesignAssuranceLevel`] gates every decision uniformly — no call path
    /// can bypass DAL.
    ///
    /// # Inputs
    /// * `entropy`: `u16` — raw entropy in `[0, 65535]`.
    /// * `surprise`: `u16` — raw surprise in `[0, 65535]`.
    /// * `has_bias`: `bool` — `true` if bias was detected.
    ///
    /// # Outputs
    /// Returns a `SafetyDecision` gated by the configured DAL.
    ///
    /// # Invariants
    /// * Inclusive halt check: `entropy ≥ halt_entropy` triggers Halt.
    /// * DAL gating is universal: every branch calls `apply_dal_to_decision`.
    /// * Severity ordering is preserved: Halt > Escalate > Warn > Proceed.
    /// * First-match-wins: higher-severity conditions are checked first and
    ///   short-circuit lower-severity checks.
    ///
    /// # DAL Gating Contract
    /// This method is the single point where threshold-based decisions meet
    /// DAL enforcement.  Every return value has passed through
    /// `apply_dal_to_decision`.  Callers that apply additional DAL wrapping
    /// produce harmless double-gating (all DAL levels are idempotent).
    fn canonical_decision(&self, entropy: u16, surprise: u16, has_bias: bool) -> SafetyDecision {
        // Halt checks first (highest severity — must not be overridden).
        if entropy >= self.halt_entropy {
            return self.apply_dal_to_decision(SafetyDecision::Halt(
                KernelError::CognitiveInstability,
                30000,
            ));
        }

        // Escalate checks (medium severity)
        if has_bias && self.bias_escalates {
            return self.apply_dal_to_decision(SafetyDecision::Escalate {
                entropy,
                reason: EscalationReason::BiasDetected,
                cooldown_ms: 5000,
            });
        }
        if entropy >= self.escalate_entropy {
            return self.apply_dal_to_decision(SafetyDecision::Escalate {
                entropy,
                reason: EscalationReason::EntropyApproachingLimit,
                cooldown_ms: 5000,
            });
        }
        if surprise >= self.escalate_surprise {
            return self.apply_dal_to_decision(SafetyDecision::Escalate {
                entropy,
                reason: EscalationReason::SurpriseElevated,
                cooldown_ms: 5000,
            });
        }

        // Warn checks (lowest severity)
        if entropy >= self.warn_entropy {
            return self.apply_dal_to_decision(SafetyDecision::Warn("entropy elevated"));
        }
        if surprise >= self.warn_surprise {
            return self.apply_dal_to_decision(SafetyDecision::Warn("surprise elevated"));
        }

        self.apply_dal_to_decision(SafetyDecision::Proceed)
    }

    /// Apply runtime DAL gating to a raw decision.
    ///
    /// This is the output-side safety gate: after the PID loop or threshold
    /// checks produce a decision, this method downgrades or suppresses it
    /// based on the configured Design Assurance Level.
    ///
    /// | DAL | Halt → | Escalate → | Warn → | Proceed → |
    /// |-----|--------|------------|--------|-----------|
    /// | A   | Halt   | Escalate   | Warn   | Proceed   |
    /// | B   | Escalate | Escalate | Warn   | Proceed   |
    /// | C   | Warn   | Warn      | Warn   | Proceed   |
    /// | D   | Warn   | Warn      | Warn   | Proceed   |
    /// | E   | Proceed | Proceed  | Proceed | Proceed   |
    fn apply_dal_to_decision(&self, decision: SafetyDecision) -> SafetyDecision {
        match self.dal {
            DesignAssuranceLevel::A => decision,
            DesignAssuranceLevel::B => match decision {
                SafetyDecision::Halt(_, cooldown_ms) => SafetyDecision::Escalate {
                    entropy: 0,
                    reason: EscalationReason::Custom("DAL B: Halt downgraded"),
                    cooldown_ms,
                },
                SafetyDecision::Proceed
                | SafetyDecision::Warn(_)
                | SafetyDecision::Escalate { .. }
                | SafetyDecision::Exit(_) => decision,
            },
            DesignAssuranceLevel::C => match decision {
                SafetyDecision::Halt(..) | SafetyDecision::Escalate { .. } => {
                    SafetyDecision::Warn("DAL C: Escalation downgraded")
                }
                SafetyDecision::Proceed | SafetyDecision::Warn(_) | SafetyDecision::Exit(_) => {
                    decision
                }
            },
            DesignAssuranceLevel::D => match decision {
                SafetyDecision::Proceed | SafetyDecision::Warn(_) => decision,
                SafetyDecision::Escalate { .. }
                | SafetyDecision::Halt(..)
                | SafetyDecision::Exit(_) => SafetyDecision::Warn("DAL D: Capped at Warn"),
            },
            DesignAssuranceLevel::E => SafetyDecision::Proceed,
        }
    }

    /// Evaluate detection results and produce a safety decision.
    ///
    /// Maps each detection signal to the appropriate escalation level based
    /// on severity: adversarial patterns and high risk → Halt,
    /// stuck/drifting → Escalate, decaying confidence → Warn.
    ///
    /// Checks are ordered by severity with first-match-wins semantics:
    /// when multiple detection flags are active simultaneously, the
    /// highest-severity result is returned and lower-severity signals
    /// are not aggregated into the decision. Inspect `DetectionResult`
    /// directly to see the full set of active alarms.
    #[cfg(feature = "std")]
    pub fn decide_from_detection(
        &self,
        detection: &DetectionResult,
        entropy: u16,
        surprise: u16,
    ) -> SafetyDecision {
        // Halt conditions: adversarial attack or high composite risk
        if !detection.adversarial_patterns.is_empty() {
            return self
                .apply_dal_to_decision(SafetyDecision::Halt(KernelError::BiasHaloDetected, 30000));
        }
        if detection.risk_score > 0.85 {
            return self.apply_dal_to_decision(SafetyDecision::Halt(
                KernelError::CognitiveInstability,
                30000,
            ));
        }

        // Escalate conditions: stuck agent or goal drift
        if detection.is_stuck {
            return self.apply_dal_to_decision(SafetyDecision::Escalate {
                entropy,
                reason: EscalationReason::StuckAgent,
                cooldown_ms: 5000,
            });
        }
        if detection.is_drifting {
            return self.apply_dal_to_decision(SafetyDecision::Escalate {
                entropy,
                reason: EscalationReason::GoalDriftDetected,
                cooldown_ms: 5000,
            });
        }

        // Warn conditions: decaying or low confidence
        if detection.is_decaying {
            return self.apply_dal_to_decision(SafetyDecision::Warn("Confidence decay detected"));
        }
        if detection.is_low_confidence {
            return self.apply_dal_to_decision(SafetyDecision::Warn("Low model confidence"));
        }

        // Fall through to canonical threshold ladder
        self.canonical_decision(entropy, surprise, false)
    }
}

/// Thread-local context for tracking safety decisions across a request.
///
/// Use this to accumulate safety signals during request processing
/// and make a final decision at the end.
///
/// Fields:
/// - `max_entropy: u16` — maximum observed entropy.
/// - `max_surprise: u16` — maximum observed surprise.
/// - `any_bias: bool` — true if any observation had bias detected.
/// - `decision_count: usize` — number of observations recorded.
/// - `policy: EscalationPolicy` — escalation policy for final decision.
#[cfg(feature = "std")]
pub struct SafetyContext {
    max_entropy: u16,
    max_surprise: u16,
    any_bias: bool,
    decision_count: usize,
    policy: EscalationPolicy,
}

#[cfg(feature = "std")]
impl SafetyContext {
    /// Create a new context with the given policy.
    pub fn new(policy: EscalationPolicy) -> Self {
        Self {
            max_entropy: 0,
            max_surprise: 0,
            any_bias: false,
            decision_count: 0,
            policy,
        }
    }

    /// Create a new context with default policy.
    pub fn default_context() -> Self {
        Self::new(EscalationPolicy::default())
    }

    /// Record an observation (entropy, surprise, bias).
    pub fn observe(&mut self, entropy: u16, surprise: u16, has_bias: bool) {
        self.max_entropy = self.max_entropy.max(entropy);
        self.max_surprise = self.max_surprise.max(surprise);
        self.any_bias = self.any_bias || has_bias;
        self.decision_count += 1;
    }

    /// Get the final decision based on all observations.
    pub fn finalize(&self) -> SafetyDecision {
        self.policy
            .decide(self.max_entropy, self.max_surprise, self.any_bias)
    }

    /// Reset the context for reuse.
    pub fn reset(&mut self) {
        self.max_entropy = 0;
        self.max_surprise = 0;
        self.any_bias = false;
        self.decision_count = 0;
    }

    /// Get the number of observations recorded.
    pub fn observation_count(&self) -> usize {
        self.decision_count
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn test_safety_decision_severity() {
        assert_eq!(SafetyDecision::Proceed.severity(), 0);
        assert_eq!(SafetyDecision::Warn("test").severity(), 1);
        assert_eq!(
            SafetyDecision::Escalate {
                entropy: 500,
                reason: EscalationReason::BiasDetected,
                cooldown_ms: 5000,
            }
            .severity(),
            2
        );
        assert_eq!(
            SafetyDecision::Halt(KernelError::CognitiveInstability, 30000).severity(),
            3
        );
        assert_eq!(
            SafetyDecision::Exit(KernelError::CognitiveInstability).severity(),
            4
        );
    }

    #[test]
    fn test_pressure_level_from_percentage() {
        assert_eq!(PressureLevel::from_percentage(10), PressureLevel::Nominal);
        assert_eq!(PressureLevel::from_percentage(30), PressureLevel::Elevated);
        assert_eq!(PressureLevel::from_percentage(60), PressureLevel::Critical);
        assert_eq!(PressureLevel::from_percentage(90), PressureLevel::Emergency);
        assert_eq!(
            PressureLevel::from_percentage(255),
            PressureLevel::Emergency
        );
    }

    #[test]
    fn test_escalation_policy_default_decide() {
        let policy = EscalationPolicy::default();
        // Safe input (below warn_entropy=30000)
        let decision = policy.decide(400, 100, false);
        assert!(matches!(decision, SafetyDecision::Proceed));
        // Warn entropy
        let decision = policy.decide(31000, 100, false);
        assert!(matches!(decision, SafetyDecision::Warn(_)));
        // Escalate entropy
        let decision = policy.decide(41000, 100, false);
        assert!(matches!(decision, SafetyDecision::Escalate { .. }));
        // Halt entropy
        let decision = policy.decide(51000, 100, false);
        assert!(matches!(decision, SafetyDecision::Halt(..)));
        // Bias escalation
        let decision = policy.decide(400, 100, true);
        assert!(matches!(decision, SafetyDecision::Escalate { .. }));
    }

    #[test]
    fn test_escalation_policy_builder() {
        let policy = EscalationPolicy::new()
            .with_warn_entropy(500)
            .with_escalate_entropy(700)
            .with_halt_entropy(900)
            .with_bias_escalates(false);
        // Bias should not escalate
        let decision = policy.decide(400, 100, true);
        assert!(matches!(decision, SafetyDecision::Proceed));
        // Custom warn threshold
        let decision = policy.decide(550, 100, false);
        assert!(matches!(decision, SafetyDecision::Warn(_)));
    }

    #[test]
    fn test_safety_context_accumulation() {
        let mut ctx = SafetyContext::default_context();
        ctx.observe(300, 100, false);
        ctx.observe(500, 200, false);
        ctx.observe(400, 250, false);
        assert_eq!(ctx.observation_count(), 3);
        let decision = ctx.finalize();
        // Max entropy 500 < warn_threshold 30000, max surprise 250 < warn_surprise 300
        assert!(matches!(decision, SafetyDecision::Proceed));
    }

    #[test]
    fn test_safety_context_with_bias() {
        let mut ctx = SafetyContext::default_context();
        ctx.observe(300, 100, false);
        ctx.observe(400, 100, true); // Bias detected
        ctx.observe(350, 100, false);
        let decision = ctx.finalize();
        assert!(matches!(decision, SafetyDecision::Escalate { .. }));
    }

    #[test]
    fn test_pressure_level_ordering() {
        assert!(PressureLevel::Nominal < PressureLevel::Elevated);
        assert!(PressureLevel::Elevated < PressureLevel::Critical);
        assert!(PressureLevel::Critical < PressureLevel::Emergency);
    }

    #[test]
    fn test_decide_from_stability() {
        let policy = EscalationPolicy::default();
        let decision = policy.decide_from_stability(CognitiveStability::Stable);
        assert!(matches!(decision, SafetyDecision::Proceed));
        let decision = policy.decide_from_stability(CognitiveStability::Pressure);
        assert!(matches!(decision, SafetyDecision::Warn(_)));
        let decision = policy.decide_from_stability(CognitiveStability::Unstable);
        assert!(matches!(decision, SafetyDecision::Halt(..)));
    }

    #[test]
    fn test_safety_context_reset() {
        let mut ctx = SafetyContext::default_context();
        ctx.observe(800, 500, true);
        assert_eq!(ctx.observation_count(), 1);
        ctx.reset();
        assert_eq!(ctx.observation_count(), 0);
        assert_eq!(ctx.max_entropy, 0);
        assert!(!ctx.any_bias);
    }

    #[test]
    fn test_escalation_policy_with_pressure() {
        let policy = EscalationPolicy::default().with_dal(DesignAssuranceLevel::A);
        // Normal case
        let decision = policy.decide_with_pressure(400, 100, false, PressureLevel::Nominal);
        assert!(matches!(decision, SafetyDecision::Proceed));
        // Pressure override
        let decision = policy.decide_with_pressure(400, 100, false, PressureLevel::Critical);
        assert!(matches!(decision, SafetyDecision::Escalate { .. }));
    }

    #[test]
    fn test_escalation_policy_cooldown_values() {
        let policy = EscalationPolicy::default();

        // Test Escalate cooldown (5000ms) for entropy-based escalation
        let decision = policy.decide(41000, 100, false);
        assert!(matches!(
            decision,
            SafetyDecision::Escalate {
                cooldown_ms: 5000,
                ..
            }
        ));

        // Test Escalate cooldown (5000ms) for bias-based escalation
        let decision = policy.decide(400, 100, true);
        assert!(matches!(
            decision,
            SafetyDecision::Escalate {
                cooldown_ms: 5000,
                ..
            }
        ));

        // Test Escalate cooldown (5000ms) for surprise-based escalation
        let decision = policy.decide(400, 55800, false);
        assert!(matches!(
            decision,
            SafetyDecision::Escalate {
                cooldown_ms: 5000,
                ..
            }
        ));

        // Test Halt cooldown (30000ms) for entropy-based halt
        let decision = policy.decide(51000, 100, false);
        assert!(matches!(decision, SafetyDecision::Halt(_, 30000)));
    }

    #[test]
    fn test_escalation_policy_cooldown_non_zero() {
        // Verify that all Escalate and Halt decisions have non-zero cooldowns
        let policy = EscalationPolicy::default();

        // Escalate cases should have 5000ms cooldown
        if let SafetyDecision::Escalate { cooldown_ms, .. } = policy.decide(41000, 100, false) {
            assert_ne!(cooldown_ms, 0, "Escalate cooldown should be non-zero");
            assert_eq!(cooldown_ms, 5000, "Escalate cooldown should be 5000ms");
        } else {
            panic!("Expected Escalate decision");
        }

        // Halt case should have 30000ms cooldown
        if let SafetyDecision::Halt(_, cooldown_ms) = policy.decide(51000, 100, false) {
            assert_ne!(cooldown_ms, 0, "Halt cooldown should be non-zero");
            assert_eq!(cooldown_ms, 30000, "Halt cooldown should be 30000ms");
        } else {
            panic!("Expected Halt decision");
        }
    }

    #[test]
    fn test_safety_decision_should_exit() {
        assert!(!SafetyDecision::Proceed.should_exit());
        assert!(!SafetyDecision::Warn("test").should_exit());
        assert!(!SafetyDecision::Escalate {
            entropy: 500,
            reason: EscalationReason::BiasDetected,
            cooldown_ms: 5000,
        }
        .should_exit());
        assert!(!SafetyDecision::Halt(KernelError::CognitiveInstability, 30000).should_exit());
        assert!(SafetyDecision::Exit(KernelError::ResourceExhaustion).should_exit());
    }

    #[test]
    fn test_pressure_level_requires_action() {
        assert!(!PressureLevel::Nominal.requires_action());
        assert!(!PressureLevel::Elevated.requires_action());
        assert!(PressureLevel::Critical.requires_action());
        assert!(PressureLevel::Emergency.requires_action());
    }

    #[test]
    fn test_must_halt_behavior() {
        // must_halt should return true ONLY for Halt, not for Exit or Escalate.
        assert!(!SafetyDecision::Proceed.must_halt());
        assert!(!SafetyDecision::Warn("test").must_halt());
        assert!(!SafetyDecision::Escalate {
            entropy: 500,
            reason: EscalationReason::BiasDetected,
            cooldown_ms: 5000,
        }
        .must_halt());
        assert!(SafetyDecision::Halt(KernelError::CognitiveInstability, 30000).must_halt());
        assert!(!SafetyDecision::Exit(KernelError::ResourceExhaustion).must_halt());
    }

    #[test]
    fn test_default_context_boundaries() {
        let mut ctx = SafetyContext::default_context();
        // Check conservative/sane defaults
        assert_eq!(ctx.observation_count(), 0);
        assert_eq!(ctx.max_entropy, 0);
        assert_eq!(ctx.max_surprise, 0);
        assert!(!ctx.any_bias);

        // Finalize before any observations
        let decision = ctx.finalize();
        assert!(matches!(decision, SafetyDecision::Proceed));

        // Observation exactly at warn boundary (30000 entropy)
        ctx.observe(30000, 100, false);
        let decision = ctx.finalize();
        assert!(matches!(decision, SafetyDecision::Warn(_)));
        assert_eq!(ctx.observation_count(), 1);

        // Exceeding warn_surprise (42600)
        ctx.observe(30000, 42600, false);
        let decision = ctx.finalize();
        assert!(matches!(decision, SafetyDecision::Warn(_)));
    }

    #[test]
    fn test_decide_from_stability_bypasses_dal() {
        // decide_from_stability maps enums directly and intentionally bypasses apply_dal_to_decision()
        // Here we test that a DAL E configuration (which would otherwise suppress Halt)
        // does not suppress a Halt decision coming from Unstable stability.
        let policy = EscalationPolicy::default().with_dal(DesignAssuranceLevel::E);

        let decision = policy.decide_from_stability(CognitiveStability::Unstable);
        assert!(
            matches!(decision, SafetyDecision::Halt(..)),
            "decide_from_stability should intentionally bypass DAL logic"
        );
    }

    #[test]
    fn test_apply_dal_to_decision_boundaries() {
        let halt_decision = SafetyDecision::Halt(KernelError::CognitiveInstability, 30000);
        let escalate_decision = SafetyDecision::Escalate {
            entropy: 40000,
            reason: EscalationReason::ResourcePressure,
            cooldown_ms: 5000,
        };
        let warn_decision = SafetyDecision::Warn("Test");

        // DAL A: Pass through
        let policy_a = EscalationPolicy::default().with_dal(DesignAssuranceLevel::A);
        assert!(matches!(
            policy_a.apply_dal_to_decision(halt_decision),
            SafetyDecision::Halt(..)
        ));
        assert!(matches!(
            policy_a.apply_dal_to_decision(escalate_decision),
            SafetyDecision::Escalate { .. }
        ));

        // DAL B: Halt downgraded to Escalate
        let policy_b = EscalationPolicy::default().with_dal(DesignAssuranceLevel::B);
        assert!(matches!(
            policy_b.apply_dal_to_decision(halt_decision),
            SafetyDecision::Escalate { .. }
        ));
        assert!(matches!(
            policy_b.apply_dal_to_decision(escalate_decision),
            SafetyDecision::Escalate { .. }
        )); // Escalate passes through

        // DAL C: Halt/Escalate downgraded to Warn
        let policy_c = EscalationPolicy::default().with_dal(DesignAssuranceLevel::C);
        assert!(matches!(
            policy_c.apply_dal_to_decision(halt_decision),
            SafetyDecision::Warn(_)
        ));
        assert!(matches!(
            policy_c.apply_dal_to_decision(escalate_decision),
            SafetyDecision::Warn(_)
        ));
        assert!(matches!(
            policy_c.apply_dal_to_decision(warn_decision),
            SafetyDecision::Warn(_)
        ));

        // DAL D: Halt/Escalate downgraded to Warn, Exit to Warn
        let policy_d = EscalationPolicy::default().with_dal(DesignAssuranceLevel::D);
        assert!(matches!(
            policy_d.apply_dal_to_decision(halt_decision),
            SafetyDecision::Warn(_)
        ));
        assert!(matches!(
            policy_d.apply_dal_to_decision(SafetyDecision::Exit(KernelError::ResourceExhaustion)),
            SafetyDecision::Warn(_)
        ));

        // DAL E: All converted to Proceed
        let policy_e = EscalationPolicy::default().with_dal(DesignAssuranceLevel::E);
        assert!(matches!(
            policy_e.apply_dal_to_decision(halt_decision),
            SafetyDecision::Proceed
        ));
        assert!(matches!(
            policy_e.apply_dal_to_decision(escalate_decision),
            SafetyDecision::Proceed
        ));
        assert!(matches!(
            policy_e.apply_dal_to_decision(warn_decision),
            SafetyDecision::Proceed
        ));
    }

    #[test]
    fn test_decide_boundaries() {
        let policy = EscalationPolicy::default();

        // warn_entropy is 30000. Test below, equal, above
        assert!(matches!(
            policy.decide(29999, 100, false),
            SafetyDecision::Proceed
        ));
        assert!(matches!(
            policy.decide(30000, 100, false),
            SafetyDecision::Warn(_)
        ));
        assert!(matches!(
            policy.decide(30001, 100, false),
            SafetyDecision::Warn(_)
        ));

        // escalate_entropy is 40000. Test below, equal, above
        assert!(matches!(
            policy.decide(39999, 100, false),
            SafetyDecision::Warn(_)
        ));
        assert!(matches!(
            policy.decide(40000, 100, false),
            SafetyDecision::Escalate { .. }
        ));
        assert!(matches!(
            policy.decide(40001, 100, false),
            SafetyDecision::Escalate { .. }
        ));

        // halt_entropy is 50000 (>= inclusive). Test below, equal, above
        assert!(matches!(
            policy.decide(49999, 100, false),
            SafetyDecision::Escalate { .. }
        ));
        assert!(matches!(
            policy.decide(50000, 100, false),
            SafetyDecision::Halt(..)
        ));
        assert!(matches!(
            policy.decide(50001, 100, false),
            SafetyDecision::Halt(..)
        ));

        // warn_surprise is 42600. Test below, equal, above
        assert!(matches!(
            policy.decide(100, 42599, false),
            SafetyDecision::Proceed
        ));
        assert!(matches!(
            policy.decide(100, 42600, false),
            SafetyDecision::Warn(_)
        ));
        assert!(matches!(
            policy.decide(100, 42601, false),
            SafetyDecision::Warn(_)
        ));

        // escalate_surprise is 55700. Test below, equal, above
        assert!(matches!(
            policy.decide(100, 55699, false),
            SafetyDecision::Warn(_)
        ));
        assert!(matches!(
            policy.decide(100, 55700, false),
            SafetyDecision::Escalate { .. }
        ));
        assert!(matches!(
            policy.decide(100, 55701, false),
            SafetyDecision::Escalate { .. }
        ));
    }

    #[test]
    fn test_decision_path_isolation() {
        let policy = EscalationPolicy::default();

        // Bias should escalate even if entropy is low
        assert!(matches!(
            policy.decide(100, 100, true),
            SafetyDecision::Escalate {
                reason: EscalationReason::BiasDetected,
                ..
            }
        ));

        // Pressure should escalate even if entropy and bias are low
        assert!(matches!(
            policy.decide_with_pressure(100, 100, false, PressureLevel::Critical),
            SafetyDecision::Escalate {
                reason: EscalationReason::ResourcePressure,
                ..
            }
        ));

        // High entropy should Escalate for entropy, not bias or pressure
        assert!(matches!(
            policy.decide(41000, 100, false),
            SafetyDecision::Escalate {
                reason: EscalationReason::EntropyApproachingLimit,
                ..
            }
        ));
    }

    // ── decide_from_detection() tests ─────────────────────────────

    #[test]
    fn test_decide_from_detection_adversarial_halt() {
        let policy = EscalationPolicy::default().with_dal(DesignAssuranceLevel::A);
        let detection = DetectionResult {
            is_stuck: false,
            is_drifting: false,
            is_low_confidence: false,
            is_decaying: false,
            adversarial_patterns: vec!["ignore previous"],
            risk_score: 0.1,
        };
        let decision = policy.decide_from_detection(&detection, 100, 100);
        assert!(
            matches!(decision, SafetyDecision::Halt(..)),
            "adversarial patterns must cause Halt"
        );
    }

    #[test]
    fn test_decide_from_detection_high_risk_halt() {
        let policy = EscalationPolicy::default().with_dal(DesignAssuranceLevel::A);
        let detection = DetectionResult {
            is_stuck: false,
            is_drifting: false,
            is_low_confidence: false,
            is_decaying: false,
            adversarial_patterns: vec![],
            risk_score: 0.9, // > 0.85
        };
        let decision = policy.decide_from_detection(&detection, 100, 100);
        assert!(
            matches!(decision, SafetyDecision::Halt(..)),
            "high risk (>0.85) must cause Halt"
        );
    }

    #[test]
    fn test_decide_from_detection_stuck_escalate() {
        let policy = EscalationPolicy::default().with_dal(DesignAssuranceLevel::A);
        let detection = DetectionResult {
            is_stuck: true,
            is_drifting: false,
            is_low_confidence: false,
            is_decaying: false,
            adversarial_patterns: vec![],
            risk_score: 0.1,
        };
        let decision = policy.decide_from_detection(&detection, 40000, 100);
        assert!(
            matches!(
                decision,
                SafetyDecision::Escalate {
                    reason: EscalationReason::StuckAgent,
                    ..
                }
            ),
            "stuck agent must cause Escalate"
        );
    }

    #[test]
    fn test_decide_from_detection_drifting_escalate() {
        let policy = EscalationPolicy::default().with_dal(DesignAssuranceLevel::A);
        let detection = DetectionResult {
            is_stuck: false,
            is_drifting: true,
            is_low_confidence: false,
            is_decaying: false,
            adversarial_patterns: vec![],
            risk_score: 0.1,
        };
        let decision = policy.decide_from_detection(&detection, 100, 100);
        assert!(
            matches!(
                decision,
                SafetyDecision::Escalate {
                    reason: EscalationReason::GoalDriftDetected,
                    ..
                }
            ),
            "goal drift must cause Escalate"
        );
    }

    #[test]
    fn test_decide_from_detection_decaying_warn() {
        let policy = EscalationPolicy::default().with_dal(DesignAssuranceLevel::A);
        let detection = DetectionResult {
            is_stuck: false,
            is_drifting: false,
            is_low_confidence: false,
            is_decaying: true,
            adversarial_patterns: vec![],
            risk_score: 0.1,
        };
        let decision = policy.decide_from_detection(&detection, 100, 100);
        assert!(
            matches!(decision, SafetyDecision::Warn(_)),
            "decaying confidence must cause Warn"
        );
    }

    #[test]
    fn test_decide_from_detection_low_confidence_warn() {
        let policy = EscalationPolicy::default().with_dal(DesignAssuranceLevel::A);
        let detection = DetectionResult {
            is_stuck: false,
            is_drifting: false,
            is_low_confidence: true,
            is_decaying: false,
            adversarial_patterns: vec![],
            risk_score: 0.1,
        };
        let decision = policy.decide_from_detection(&detection, 100, 100);
        assert!(
            matches!(decision, SafetyDecision::Warn(_)),
            "low confidence must cause Warn"
        );
    }

    #[test]
    fn test_decide_from_detection_all_false_falls_through() {
        let policy = EscalationPolicy::default().with_dal(DesignAssuranceLevel::A);
        let detection = DetectionResult {
            is_stuck: false,
            is_drifting: false,
            is_low_confidence: false,
            is_decaying: false,
            adversarial_patterns: vec![],
            risk_score: 0.1,
        };
        // Low entropy/surprise → Proceed via fall-through decide()
        let decision = policy.decide_from_detection(&detection, 100, 100);
        assert!(
            matches!(decision, SafetyDecision::Proceed),
            "all false with low signals must Proceed"
        );
        // High entropy → Escalate via fall-through decide()
        let decision = policy.decide_from_detection(&detection, 41000, 100);
        assert!(
            matches!(decision, SafetyDecision::Escalate { .. }),
            "all false with high entropy must Escalate via decide()"
        );
    }

    // ── decide_with_pressure() edge cases ─────────────────────────

    #[test]
    fn test_decide_with_pressure_elevated_no_halt() {
        // Elevated pressure (35%) with low entropy — falls through to decide()
        let policy = EscalationPolicy::default().with_dal(DesignAssuranceLevel::A);
        let decision = policy.decide_with_pressure(100, 100, false, PressureLevel::Elevated);
        // default escalate_pressure is Critical, so Elevated < Critical → no pressure escalation
        assert!(
            matches!(decision, SafetyDecision::Proceed),
            "Elevated pressure below escalate_pressure must not halt"
        );
    }

    #[test]
    fn test_decide_with_pressure_emergency_halt() {
        // Emergency pressure (90%) — should Escalate via pressure escalation
        let policy = EscalationPolicy::default().with_dal(DesignAssuranceLevel::A);
        let decision = policy.decide_with_pressure(100, 100, false, PressureLevel::Emergency);
        assert!(
            matches!(decision, SafetyDecision::Escalate { .. }),
            "Emergency pressure must escalate (via pressure escalation)"
        );
    }

    #[test]
    fn test_decide_with_pressure_custom_escalate_pressure() {
        // Set escalate_pressure to Elevated directly (no builder exists)
        let policy = EscalationPolicy {
            escalate_pressure: PressureLevel::Elevated,
            ..EscalationPolicy::default()
        }
        .with_dal(DesignAssuranceLevel::A);
        // Elevated pressure (35%) now triggers escalation
        let decision = policy.decide_with_pressure(100, 100, false, PressureLevel::Elevated);
        assert!(
            matches!(decision, SafetyDecision::Escalate { .. }),
            "custom escalate_pressure=Elevated must escalate at Elevated pressure"
        );
        // Nominal pressure still proceeds
        let decision = policy.decide_with_pressure(100, 100, false, PressureLevel::Nominal);
        assert!(
            matches!(decision, SafetyDecision::Proceed),
            "custom escalate_pressure=Elevated must not escalate at Nominal"
        );
    }

    // ── SafetyDecision Exit edge cases ────────────────────────────

    #[test]
    fn test_exit_is_blocking() {
        let exit = SafetyDecision::Exit(KernelError::ResourceExhaustion);
        assert!(exit.is_blocking());
    }

    #[test]
    fn test_exit_status_label() {
        let exit = SafetyDecision::Exit(KernelError::ResourceExhaustion);
        assert_eq!(exit.status_label(), "exit");
    }

    #[test]
    fn test_exit_recommended_cooldown_ms() {
        let exit = SafetyDecision::Exit(KernelError::ResourceExhaustion);
        assert_eq!(exit.recommended_cooldown_ms(), 0);
    }
}
