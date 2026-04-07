//! LLMOSAFE Integration Layer - Composition primitives for external systems
//!
//! This module provides the integration primitives that make llmosafe composable
//! with other Rust ecosystems (tower, tokio, async runtimes) and cross-language
//! FFI consumers.
//!
//! # Architecture
//!
//! The integration layer provides:
//! - SafetyDecision enum for decision flow semantics
//! - PressureLevel enum for resource pressure semantics
//! - EscalationPolicy for configurable response thresholds
//! - SafetyContext for thread-local decision tracking
//!
//! # Example
//!
//! ```
//! use llmosafe::{SafetyDecision, EscalationPolicy};
//!
//! let policy = EscalationPolicy::default();
//! let decision = policy.decide(500, 100, false);
//! assert!(matches!(decision, SafetyDecision::Proceed));
//! ```

use crate::llmosafe_kernel::{CognitiveStability, KernelError};

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
    },
    /// Input must be rejected. Halt processing immediately.
    /// Contains the kernel error that triggered the halt.
    Halt(KernelError),
}

impl SafetyDecision {
    /// Returns true if processing can continue.
    pub fn can_proceed(&self) -> bool {
        matches!(self, Self::Proceed | Self::Warn(_))
    }

    /// Returns true if processing must stop.
    pub fn must_halt(&self) -> bool {
        matches!(self, Self::Halt(_))
    }

    /// Returns the severity level (0=Proceed, 1=Warn, 2=Escalate, 3=Halt).
    pub fn severity(&self) -> u8 {
        match self {
            Self::Proceed => 0,
            Self::Warn(_) => 1,
            Self::Escalate { .. } => 2,
            Self::Halt(_) => 3,
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
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EscalationPolicy {
    /// Entropy threshold for Warn (default: 600).
    pub warn_entropy: u16,
    /// Entropy threshold for Escalate (default: 800).
    pub escalate_entropy: u16,
    /// Entropy threshold for Halt (default: 1000).
    pub halt_entropy: u16,
    /// Surprise threshold for Warn (default: 300).
    pub warn_surprise: u16,
    /// Surprise threshold for Escalate (default: 500).
    pub escalate_surprise: u16,
    /// Whether bias detection triggers immediate escalation.
    pub bias_escalates: bool,
    /// Pressure level that triggers Escalate.
    pub escalate_pressure: PressureLevel,
}

impl Default for EscalationPolicy {
    fn default() -> Self {
        Self {
            warn_entropy: 600,
            escalate_entropy: 800,
            halt_entropy: 1000,
            warn_surprise: 300,
            escalate_surprise: 500,
            bias_escalates: true,
            escalate_pressure: PressureLevel::Critical,
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

    /// Evaluate entropy, surprise, and bias flags to produce a decision.
    pub fn decide(&self, entropy: u16, surprise: u16, has_bias: bool) -> SafetyDecision {
        // Bias check first (if configured)
        if has_bias && self.bias_escalates {
            return SafetyDecision::Escalate {
                entropy,
                reason: EscalationReason::BiasDetected,
            };
        }

        // Entropy-based decisions
        if entropy >= self.halt_entropy {
            return SafetyDecision::Halt(KernelError::CognitiveInstability);
        }
        if entropy >= self.escalate_entropy {
            return SafetyDecision::Escalate {
                entropy,
                reason: EscalationReason::EntropyApproachingLimit,
            };
        }
        if entropy >= self.warn_entropy {
            return SafetyDecision::Warn("entropy elevated");
        }

        // Surprise-based decisions
        if surprise >= self.escalate_surprise {
            return SafetyDecision::Escalate {
                entropy,
                reason: EscalationReason::SurpriseElevated,
            };
        }
        if surprise >= self.warn_surprise {
            return SafetyDecision::Warn("surprise elevated");
        }

        SafetyDecision::Proceed
    }

    /// Evaluate with pressure level.
    pub fn decide_with_pressure(
        &self,
        entropy: u16,
        surprise: u16,
        has_bias: bool,
        pressure: PressureLevel,
    ) -> SafetyDecision {
        // Pressure override
        if pressure >= self.escalate_pressure {
            return SafetyDecision::Escalate {
                entropy,
                reason: EscalationReason::ResourcePressure,
            };
        }
        self.decide(entropy, surprise, has_bias)
    }

    /// Evaluate from CognitiveStability.
    pub fn decide_from_stability(&self, stability: CognitiveStability) -> SafetyDecision {
        match stability {
            CognitiveStability::Stable => SafetyDecision::Proceed,
            CognitiveStability::Pressure => SafetyDecision::Warn("cognitive pressure detected"),
            CognitiveStability::Unstable => SafetyDecision::Halt(KernelError::CognitiveInstability),
        }
    }
}

/// Thread-local context for tracking safety decisions across a request.
/// 
/// Use this to accumulate safety signals during request processing
/// and make a final decision at the end.
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
        self.policy.decide(self.max_entropy, self.max_surprise, self.any_bias)
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
                reason: EscalationReason::BiasDetected
            }
            .severity(),
            2
        );
        assert_eq!(SafetyDecision::Halt(KernelError::CognitiveInstability).severity(), 3);
    }

    #[test]
    fn test_pressure_level_from_percentage() {
        assert_eq!(PressureLevel::from_percentage(10), PressureLevel::Nominal);
        assert_eq!(PressureLevel::from_percentage(30), PressureLevel::Elevated);
        assert_eq!(PressureLevel::from_percentage(60), PressureLevel::Critical);
        assert_eq!(PressureLevel::from_percentage(90), PressureLevel::Emergency);
        assert_eq!(PressureLevel::from_percentage(255), PressureLevel::Emergency);
    }

    #[test]
    fn test_escalation_policy_default_decide() {
        let policy = EscalationPolicy::default();
        // Safe input
        let decision = policy.decide(400, 100, false);
        assert!(matches!(decision, SafetyDecision::Proceed));
        // Warn entropy
        let decision = policy.decide(650, 100, false);
        assert!(matches!(decision, SafetyDecision::Warn(_)));
        // Escalate entropy
        let decision = policy.decide(850, 100, false);
        assert!(matches!(decision, SafetyDecision::Escalate { .. }));
        // Halt entropy
        let decision = policy.decide(1100, 100, false);
        assert!(matches!(decision, SafetyDecision::Halt(_)));
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
        // Max entropy 500 < warn_threshold 600, max surprise 250 < warn_threshold 300
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
        assert!(matches!(decision, SafetyDecision::Halt(_)));
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
        let policy = EscalationPolicy::default();
        // Normal case
        let decision = policy.decide_with_pressure(400, 100, false, PressureLevel::Nominal);
        assert!(matches!(decision, SafetyDecision::Proceed));
        // Pressure override
        let decision = policy.decide_with_pressure(400, 100, false, PressureLevel::Critical);
        assert!(matches!(decision, SafetyDecision::Escalate { .. }));
    }
}
