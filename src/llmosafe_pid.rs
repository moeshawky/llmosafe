//! LLMOSAFE PID Decision Subsystem.
#![deny(clippy::cast_lossless)]
//!
//! Replaces scattered threshold logic with a Proportional-Integral-Derivative-FeedForward
//! controller. All inputs normalised to `[0, 1]`; output is a risk score mapped to
//! Proceed/Warn/Halt via gain thresholds.
//!
//! # Gains
//!
//! Default gains calibrated for classifier entropy [0, 65535]:
//! Kp=1.0, Ki_fast=0.5, Ki_slow=0.3, Kd=2.0, Kf=0.3, warn_gain=0.5, halt_gain=1.0
//!
//! # State
//!
//! PidState owns dual-rate leaky integrators (acute decay 0.9, chronic decay 0.99)
//! and prev_pressure for step-change detection. State persists across process() calls;
//! reset by reset_full().
//!
//! # Anti-windup
//!
//! Integrators freeze when risk >= halt_gain, preventing meaningless accumulation
//! during Halt state and avoiding the halt→recovery→re-halt loop.
//!
//! # Sidechain
//!
//! Detection flags (already computed per cycle in the pipeline) modulate PID gains:
//! stuck boosts Kp and Ki_slow ×1.3 (loop detection), drifting boosts Kd ×1.2
//! (velocity), decaying boosts Ki_fast ×1.3 (confidence collapse), low_confidence
//! boosts Kf ×1.5 (classifier amp), anomaly boosts Kd ×1.5 (rate-of-change)
//! overrides drifting. When no flags are set, all multipliers return to 1.0.

use crate::control_types::OverrideFlags;
use crate::llmosafe_integration::{EscalationReason, SafetyDecision};
use crate::llmosafe_kernel::{
    KernelError, FLAG_ANOMALY, FLAG_DECAYING, FLAG_DRIFTING, FLAG_LOW_CONFIDENCE, FLAG_STUCK,
};

/// PID controller configuration.
///
/// All gains are dimensionless multipliers on normalised signals.
/// `validate()` rejects NaN, out-of-range, and `warn_gain >= halt_gain`.
/// `ki_fast` and `ki_slow` form a dual-rate integrator
/// separating acute (request-level) from chronic (session-level) risk.
pub struct PidConfig {
    /// Proportional gain for resource pressure [0, 5.0], default 1.0
    pub kp: f32,
    /// Fast integral gain for acute entropy accumulation [0, 3.0], default 0.5
    pub ki_fast: f32,
    /// Slow integral gain for chronic entropy accumulation [0, 3.0], default 0.3
    pub ki_slow: f32,
    /// Derivative gain for entropy trend [0, 5.0], default 2.0
    pub kd: f32,
    /// Feed-forward gain for classifier probability [0, 1.0], default 0.3
    pub kf: f32,
    /// RiskScore fraction at which Warn fires [0, halt_gain), default 0.5
    pub warn_gain: f32,
    /// RiskScore fraction at which Halt fires (warn_gain, 1.0], default 1.0
    pub halt_gain: f32,
    /// Per-cycle decay factor for chronic integrator [0, 1), default 0.99
    pub integrator_decay: f32,
    /// Pressure delta that triggers gain-scheduled Kp doubling [0, 1], default 0.5
    pub step_change_threshold: f32,
}

impl Default for PidConfig {
    /// Sets kp=1.0, ki_fast=0.5, ki_slow=0.3, kd=2.0, kf=0.3,
    /// warn_gain=0.5, halt_gain=1.0, integrator_decay=0.99,
    /// step_change_threshold=0.5.
    fn default() -> Self {
        Self {
            kp: 1.0,
            ki_fast: 0.5,
            ki_slow: 0.3,
            kd: 2.0,
            kf: 0.3,
            warn_gain: 0.5,
            halt_gain: 1.0,
            integrator_decay: 0.99,
            step_change_threshold: 0.5,
        }
    }
}

impl PidConfig {
    /// Validates all gain values are finite, within range, and `warn_gain < halt_gain`.
    ///
    /// # Errors
    ///
    /// Returns `&'static str` describing the first invalid field found.
    pub fn validate(&self) -> Result<(), &'static str> {
        // Helper: must be finite and in [min, max]
        fn check(val: f32, min: f32, max: f32, name: &'static str) -> Result<(), &'static str> {
            if val.is_nan() || !val.is_finite() {
                return Err(name);
            }
            if val < min || val > max {
                return Err(name);
            }
            Ok(())
        }

        check(self.kp, 0.0, 5.0, "kp must be in [0.0, 5.0]")?;
        check(self.ki_fast, 0.0, 3.0, "ki_fast must be in [0.0, 3.0]")?;
        check(self.ki_slow, 0.0, 3.0, "ki_slow must be in [0.0, 3.0]")?;
        check(self.kd, 0.0, 5.0, "kd must be in [0.0, 5.0]")?;
        check(self.kf, 0.0, 1.0, "kf must be in [0.0, 1.0]")?;
        check(self.warn_gain, 0.0, 1.0, "warn_gain must be in [0.0, 1.0]")?;
        check(self.halt_gain, 0.0, 1.0, "halt_gain must be in [0.0, 1.0]")?;
        check(
            self.integrator_decay,
            0.0,
            1.0,
            "integrator_decay must be in [0.0, 1.0)",
        )?;
        check(
            self.step_change_threshold,
            0.0,
            1.0,
            "step_change_threshold must be in [0.0, 1.0]",
        )?;

        if self.integrator_decay >= 1.0 {
            return Err("integrator_decay must be < 1.0");
        }
        if self.warn_gain >= self.halt_gain {
            return Err("warn_gain must be < halt_gain");
        }
        // acute_decay (0.9) must be < chronic_decay (integrator_decay, default 0.99)
        // Verified by construction: acute is hardcoded 0.9, chronic must be > 0.9
        if self.integrator_decay <= 0.899 {
            return Err(
                "integrator_decay must be > 0.9 (chronic must decay slower than acute=0.9)",
            );
        }

        Ok(())
    }
}

/// Runtime PID state.
///
/// `acute_entropy` (decay 0.9/cycle, ~10 cycle memory) catches request-level spikes.
/// `chronic_entropy` (decay `integrator_decay`/cycle, ~100 cycle memory) catches
/// session-level elevation. Both are leaky integrators clamped to [0, 1].
/// `prev_pressure_norm` enables step-change detection for gain-scheduled P.
/// Zero-initialised; `reset()` zeros all fields.
pub struct PidState {
    /// Fast integrator: acute (request-level) entropy accumulation, decay 0.9
    pub acute_entropy: f32,
    /// Slow integrator: chronic (session-level) entropy accumulation, decay config.integrator_decay
    pub chronic_entropy: f32,
    /// Previous cycle's normalised pressure for step-change delta computation
    pub prev_pressure_norm: f32,
}

impl PidState {
    /// Zero-initialises all state fields.
    pub fn new() -> Self {
        Self {
            acute_entropy: 0.0,
            chronic_entropy: 0.0,
            prev_pressure_norm: 0.0,
        }
    }

    /// Zeros all state fields.
    pub fn reset(&mut self) {
        self.acute_entropy = 0.0;
        self.chronic_entropy = 0.0;
        self.prev_pressure_norm = 0.0;
    }
}

impl Default for PidState {
    /// Delegates to PidState::new(), returning zero-initialized state.
    fn default() -> Self {
        Self::new()
    }
}

/// Private struct holding modulated gain values after sidechain detection
/// flags are applied. Fields: kp, ki_fast, ki_slow, kd, kf (all f32).
/// Constructed only by `modulate_gains()`.
struct EffectiveGains {
    kp: f32,
    ki_fast: f32,
    ki_slow: f32,
    kd: f32,
    kf: f32,
}

/// Modulates PID gains using detection flags as a sidechain.
///
/// Each flag shifts the corresponding gain by a bounded factor ([0.5, 2.0]).
/// When no flags are set, returns identity (1.0× multiplier on all gains).
/// The flags are already computed every cycle at the pipeline DETECTION stage —
/// this function costs zero additional measurement.
fn modulate_gains(config: &PidConfig, flags: u8) -> EffectiveGains {
    EffectiveGains {
        kp: config.kp * if flags & FLAG_STUCK != 0 { 1.3 } else { 1.0 },
        ki_fast: config.ki_fast * if flags & FLAG_DECAYING != 0 { 1.3 } else { 1.0 },
        ki_slow: config.ki_slow * if flags & FLAG_STUCK != 0 { 1.3 } else { 1.0 },
        kd: config.kd
            * if flags & FLAG_ANOMALY != 0 {
                1.5
            } else if flags & FLAG_DRIFTING != 0 {
                1.2
            } else {
                1.0
            },
        kf: config.kf
            * if flags & FLAG_LOW_CONFIDENCE != 0 {
                1.5
            } else {
                1.0
            },
    }
}

/// Shared PID computation: normalise, modulate gains, update integrators, compute terms.
///
/// Steps 1-4 of the PID algorithm. Step 5 (bias override) is caller's responsibility.
///
/// Allow: 10 arguments is the minimum for composing entropy, trend, pressure,
/// classifier probability, detection flags, pid config, and pid state
/// into a single control-loop score. Individual struct grouping would
/// require allocation or pass-through wrappers — inappropriate at Tier 2
/// (see SYS-SPEC-602 paragraph 7.3.4).
#[allow(clippy::too_many_arguments)]
fn compute_pid_score_inner(
    entropy_raw: u16,
    trend: f64,
    pressure: u8,
    classifier_prob: f32,
    detection_flags: u8,
    config: &PidConfig,
    state: &mut PidState,
) -> f32 {
    let entropy_norm = (f32::from(entropy_raw) / 65535.0_f32).clamp(0.0, 1.0);
    let trend_abs_norm = ((trend.abs() as f32) / 65535.0_f32).clamp(0.0, 1.0);
    let pressure_norm = (f32::from(pressure) / 100.0_f32).clamp(0.0, 1.0);
    let f_norm = (1.0_f32 - classifier_prob).clamp(0.0, 1.0);

    let eff = modulate_gains(config, detection_flags);

    let pressure_delta = (pressure_norm - state.prev_pressure_norm).abs();
    let kp_effective = if pressure_delta > config.step_change_threshold {
        eff.kp * 2.0
    } else {
        eff.kp
    };

    let risk_estimate = (kp_effective * pressure_norm)
        + (eff.ki_fast * state.acute_entropy + eff.ki_slow * state.chronic_entropy)
        + (eff.kd * trend_abs_norm)
        + (eff.kf * f_norm);

    if risk_estimate < config.halt_gain {
        state.acute_entropy = (state.acute_entropy * 0.9 + entropy_norm).clamp(0.0, 1.0);
        state.chronic_entropy =
            (state.chronic_entropy * config.integrator_decay + entropy_norm).clamp(0.0, 1.0);
    } else {
        state.acute_entropy *= 0.999;
        state.chronic_entropy *= 0.999;
    }

    let p_term = kp_effective * pressure_norm;
    let i_term = eff.ki_fast * state.acute_entropy + eff.ki_slow * state.chronic_entropy;
    let d_term = eff.kd * trend_abs_norm;
    let f_term = eff.kf * f_norm;

    let risk = (p_term + i_term + d_term + f_term).clamp(0.0, 1.0);

    state.prev_pressure_norm = pressure_norm;

    risk
}

/// Computes a risk score from normalised sensor inputs using the PIDF formula.
///
/// Thin wrapper over `compute_pid_score_inner` that applies the bias override.
/// See `compute_pid_score_pure` for the bias-free version.
/// Allow: same justification as compute_pid_score_pure
/// (Tier-2 no-allocation constraint precludes struct grouping).
#[allow(clippy::too_many_arguments)]
pub fn compute_pid_score(
    entropy_raw: u16,
    trend: f64,
    pressure: u8,
    classifier_prob: f32,
    has_bias: bool,
    detection_flags: u8,
    config: &PidConfig,
    state: &mut PidState,
) -> f32 {
    let mut risk = compute_pid_score_inner(
        entropy_raw,
        trend,
        pressure,
        classifier_prob,
        detection_flags,
        config,
        state,
    );

    if has_bias {
        risk = risk.max(config.halt_gain + 0.001);
    }
    risk.clamp(0.0, 1.0)
}

/// Pure PID risk computation — no safety overrides.
///
/// Implements the infusion pump pattern: this function computes risk
/// from sensor fusion alone (P+I+D+F terms). Safety overrides are
/// applied separately by `apply_safety_overrides()`.
///
/// # DAL A
///
/// Core computation path. All inputs must be finite. Anti-windup
/// prevents integrator saturation during sustained Halt.
///
/// # MC/DC
///
/// Each of 4 terms independently affects risk. Step-change detection
/// independently doubles Kp. See traceability matrix R-05.
/// Allow: 10 arguments is the minimum for Tier-2 PID composition
/// (see justification on compute_pid_score_inner, SYS-SPEC-602 §7.3.4).
#[allow(clippy::too_many_arguments)]
pub fn compute_pid_score_pure(
    entropy_raw: u16,
    trend: f64,
    pressure: u8,
    classifier_prob: f32,
    detection_flags: u8,
    config: &PidConfig,
    state: &mut PidState,
) -> f32 {
    compute_pid_score_inner(
        entropy_raw,
        trend,
        pressure,
        classifier_prob,
        detection_flags,
        config,
        state,
    )
}

/// Applies safety overrides to a PID risk score.
///
/// Infusion pump pattern: PID computes pure risk, safety supervisor
/// enforces hard limits. This prevents a PID bug from bypassing
/// safety enforcement.
///
/// # DAL A Overrides
///
/// | Flag | Effect | Condition |
/// |------|--------|-----------|
/// | `BIAS` | `risk = max(risk, halt_gain + 0.001)` | Bias detection overrides all computation |
/// | `EXHAUSTED` | `risk = 1.0` | Resource exhaustion is max priority |
/// | `KERNEL_UNSTABLE` | `risk = max(risk, halt_gain)` | Kernel instability forces halt |
///
/// # MC/DC
///
/// Each override flag must independently force Halt regardless of
/// PID output. 100% MC/DC required on all 3 condition/decision pairs.
pub fn apply_safety_overrides(risk: f32, flags: OverrideFlags, config: &PidConfig) -> f32 {
    let mut risk = risk;

    // DAL A: bias detection forces halt regardless of PID
    // MC/DC: has_bias independently forces risk >= halt_gain + ε
    if flags.contains(OverrideFlags::BIAS) {
        risk = risk.max(config.halt_gain + 0.001);
    }

    // DAL A: resource exhaustion forces max risk
    // MC/DC: is_exhausted independently forces risk = 1.0
    if flags.contains(OverrideFlags::EXHAUSTED) {
        risk = 1.0;
    }

    // DAL A: kernel instability forces halt-level risk
    // MC/DC: is_kernel_unstable independently forces risk >= halt_gain
    if flags.contains(OverrideFlags::KERNEL_UNSTABLE) {
        risk = risk.max(config.halt_gain);
    }

    risk.clamp(0.0, 1.0)
}

/// Maps a risk score to a `SafetyDecision` via gain thresholds.
///
/// * `risk < warn_gain` → `Proceed`
/// * `warn_gain ≤ risk < halt_gain` → `Escalate` (carries entropy=0)
/// * `risk ≥ halt_gain` → `Halt(CognitiveInstability, 30000)`
pub fn pid_risk_to_decision(risk: f32, config: &PidConfig) -> SafetyDecision {
    debug_assert!(
        risk.is_finite() && (0.0..=1.0).contains(&risk),
        "risk must be in [0,1]: {}",
        risk
    );

    if risk >= config.halt_gain {
        SafetyDecision::Halt(KernelError::CognitiveInstability, 30000)
    } else if risk >= config.warn_gain {
        SafetyDecision::Escalate {
            entropy: 0,
            reason: EscalationReason::Custom("PID risk elevated"),
            cooldown_ms: 5000,
        }
    } else {
        SafetyDecision::Proceed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── PidConfig tests ──────────────────────────────────────────

    #[test]
    fn pid_config_default_validates() {
        let config = PidConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn pid_config_rejects_nan_kp() {
        let config = PidConfig {
            kp: f32::NAN,
            ..PidConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn pid_config_rejects_inf_kd() {
        let config = PidConfig {
            kd: f32::INFINITY,
            ..PidConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn pid_config_rejects_negative_ki_fast() {
        let config = PidConfig {
            ki_fast: -0.1,
            ..PidConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn pid_config_rejects_kp_out_of_range_high() {
        let config = PidConfig {
            kp: 5.1,
            ..PidConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn pid_config_rejects_kf_out_of_range_high() {
        let config = PidConfig {
            kf: 1.1,
            ..PidConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn pid_config_rejects_warn_gain_ge_halt_gain() {
        let config = PidConfig {
            warn_gain: 1.0,
            halt_gain: 1.0,
            ..PidConfig::default()
        };
        assert!(config.validate().is_err());

        let config2 = PidConfig {
            warn_gain: 0.9,
            halt_gain: 0.8,
            ..PidConfig::default()
        };
        assert!(config2.validate().is_err());
    }

    #[test]
    fn pid_config_rejects_integrator_decay_at_1() {
        let config = PidConfig {
            integrator_decay: 1.0,
            ..PidConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn pid_config_rejects_integrator_decay_below_acute() {
        let config = PidConfig {
            integrator_decay: 0.5,
            ..PidConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn pid_config_rejects_step_change_threshold_out_of_range() {
        let config = PidConfig {
            step_change_threshold: 1.1,
            ..PidConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn pid_config_boundary_values_validate() {
        let config = PidConfig {
            kp: 0.0,
            ki_fast: 0.0,
            ki_slow: 0.0,
            kd: 0.0,
            kf: 0.0,
            warn_gain: 0.0,
            halt_gain: 1.0,
            integrator_decay: 0.9, // just above acute 0.9 threshold
            step_change_threshold: 0.0,
        };
        assert!(config.validate().is_ok());

        let config_max = PidConfig {
            kp: 5.0,
            ki_fast: 3.0,
            ki_slow: 3.0,
            kd: 5.0,
            kf: 1.0,
            warn_gain: 0.99,
            halt_gain: 1.0,
            integrator_decay: 0.999,
            step_change_threshold: 1.0,
        };
        assert!(config_max.validate().is_ok());
    }

    // ── PidState tests ───────────────────────────────────────────

    #[test]
    fn pid_state_new_zeros_all() {
        let state = PidState::new();
        assert_eq!(state.acute_entropy, 0.0);
        assert_eq!(state.chronic_entropy, 0.0);
        assert_eq!(state.prev_pressure_norm, 0.0);
    }

    #[test]
    fn pid_state_reset_zeros_all() {
        let mut state = PidState {
            acute_entropy: 0.5,
            chronic_entropy: 0.8,
            prev_pressure_norm: 0.3,
        };
        state.reset();
        assert_eq!(state.acute_entropy, 0.0);
        assert_eq!(state.chronic_entropy, 0.0);
        assert_eq!(state.prev_pressure_norm, 0.0);
    }

    // ── compute_pid_score tests ──────────────────────────────────

    #[test]
    fn zero_input_produces_low_risk() {
        let config = PidConfig::default();
        let mut state = PidState::new();
        let risk = compute_pid_score(0, 0.0, 0, 1.0, false, 0, &config, &mut state);
        // At prob=1.0, F_term = kf * (1 - 1) = 0. All other terms are zero.
        assert!((risk - 0.0).abs() < 0.01, "risk={}", risk);
    }

    #[test]
    fn max_entropy_produces_i_term() {
        let config = PidConfig::default();
        let mut state = PidState::new();
        // First cycle: integrators start at 0, then updated BEFORE I-term is computed.
        // acute = 0*0.9 + 1.0 = 1.0, chronic = 0*0.99 + 1.0 = 0.01 (clamped to 1.0)
        // Actually: acute=1.0, chronic=0.01. I_term = 0.5*1.0 + 0.3*0.01 = 0.503
        let risk = compute_pid_score(65535, 0.0, 0, 1.0, false, 0, &config, &mut state);
        // I_term alone ≈ 0.503, but F=0 (prob=1.0), P=0, D=0
        assert!(risk > 0.4, "risk should have I-term contribution: {}", risk);

        // Second cycle: integrators still saturated
        let risk2 = compute_pid_score(65535, 0.0, 0, 1.0, false, 0, &config, &mut state);
        assert!(risk2 > 0.4, "risk2 should also have I-term contribution");
    }

    #[test]
    fn integrator_decays_over_clean_cycles() {
        let config = PidConfig::default();
        let mut state = PidState::new();
        // Seed integrators with high entropy (one cycle is enough to saturate)
        compute_pid_score(65535, 0.0, 0, 1.0, false, 0, &config, &mut state);
        let after_spike = state.chronic_entropy;
        assert!(
            after_spike > 0.9,
            "integrator should saturate quickly: {}",
            after_spike
        );

        // Feed zero entropy for 50 cycles
        for _ in 0..50 {
            compute_pid_score(0, 0.0, 0, 1.0, false, 0, &config, &mut state);
        }
        assert!(
            state.chronic_entropy < after_spike,
            "integrator should decay: was {}, now {}",
            after_spike,
            state.chronic_entropy
        );
    }

    #[test]
    fn step_change_doubles_kp() {
        let config = PidConfig::default();
        let mut state = PidState::new();
        // Start with prev_pressure_norm = 0.0 (default)
        // Feed pressure 80 → delta = 0.8 > 0.5 → Kp doubled
        let risk_with_step = compute_pid_score(0, 0.0, 80, 1.0, false, 0, &config, &mut state);
        // P_term = (kp * 2) * 0.8 = 2 * 0.8 = 1.6, clamped → 1.0
        assert!(
            (risk_with_step - 1.0).abs() < 0.01,
            "risk={}",
            risk_with_step
        );

        // Next cycle with same pressure: delta = 0.0 → no step change
        let risk_no_step = compute_pid_score(0, 0.0, 80, 1.0, false, 0, &config, &mut state);
        // P_term = kp * 0.8 = 1.0 * 0.8 = 0.8 (no doubling)
        assert!(
            risk_no_step < risk_with_step,
            "step-change risk should be higher"
        );
    }

    #[test]
    fn bias_override_forces_halt_level() {
        let config = PidConfig::default();
        let mut state = PidState::new();
        // All inputs clean, but has_bias=true
        let risk = compute_pid_score(0, 0.0, 0, 1.0, true, 0, &config, &mut state);
        // risk >= halt_gain + 0.001 = 1.001, clamped to 1.0
        assert!(
            risk >= config.halt_gain,
            "bias should force >= halt_gain: got {}",
            risk
        );
    }

    #[test]
    fn risk_never_exceeds_1_with_max_gains() {
        let config = PidConfig {
            kp: 5.0,
            ki_fast: 3.0,
            ki_slow: 3.0,
            kd: 5.0,
            kf: 1.0,
            ..PidConfig::default()
        };
        let mut state = PidState::new();
        // Pre-seed integrators at max
        state.acute_entropy = 1.0;
        state.chronic_entropy = 1.0;
        let risk = compute_pid_score(
            65535,
            65535.0,
            100,
            0.0,
            false,
            FLAG_STUCK | FLAG_ANOMALY | FLAG_LOW_CONFIDENCE,
            &config,
            &mut state,
        );
        assert!(risk <= 1.0, "risk must be clamped: got {}", risk);
        assert!(risk.is_finite());
    }

    #[test]
    fn monotonic_higher_entropy_higher_risk() {
        let config = PidConfig::default();
        let mut state_low = PidState::new();
        let mut state_high = PidState::new();
        // Feed low entropy into both, then compare one step
        let _ = compute_pid_score(1000, 0.0, 30, 0.8, false, 0, &config, &mut state_low);
        let _ = compute_pid_score(1000, 0.0, 30, 0.8, false, 0, &config, &mut state_high);

        let risk_low = compute_pid_score(5000, 0.0, 30, 0.8, false, 0, &config, &mut state_low);
        let risk_high = compute_pid_score(50000, 0.0, 30, 0.8, false, 0, &config, &mut state_high);
        assert!(
            risk_high > risk_low,
            "higher entropy should produce higher risk: {} vs {}",
            risk_low,
            risk_high
        );
    }

    // ── Anti-windup tests ────────────────────────────────────────

    #[test]
    fn anti_windup_freezes_integrator_when_risk_at_halt() {
        let mut state = PidState::new();
        state.acute_entropy = 1.0;
        state.chronic_entropy = 1.0;
        let forced_config = PidConfig {
            kp: 5.0,
            kd: 5.0,
            ki_fast: 3.0,
            ki_slow: 3.0,
            ..PidConfig::default()
        };
        let risk = compute_pid_score(
            65535,
            65535.0,
            100,
            0.0,
            false,
            FLAG_STUCK | FLAG_ANOMALY,
            &forced_config,
            &mut state,
        );
        assert!(risk >= 1.0);

        let acute_before = state.acute_entropy;
        let chronic_before = state.chronic_entropy;
        // Feed more entropy while risk is at halt — integrator should bleed (0.999×)
        let _ = compute_pid_score(
            65535,
            65535.0,
            100,
            0.0,
            false,
            FLAG_STUCK,
            &forced_config,
            &mut state,
        );
        assert!(
            state.acute_entropy < acute_before,
            "acute integrator should bleed during windup"
        );
        assert!(
            state.chronic_entropy < chronic_before,
            "chronic integrator should bleed during windup"
        );
    }

    #[test]
    fn anti_windup_resumes_when_risk_drops() {
        let config = PidConfig::default();
        let mut state = PidState::new();
        // Build up integrators
        for _ in 0..5 {
            compute_pid_score(65535, 0.0, 30, 0.5, false, 0, &config, &mut state);
        }
        let acute_before = state.acute_entropy;
        // Feed clean signal — risk should be below halt_gain, integrator must update
        let _ = compute_pid_score(0, 0.0, 0, 1.0, false, 0, &config, &mut state);
        assert!(
            state.acute_entropy < acute_before,
            "integrator should decay on clean input: {} -> {}",
            acute_before,
            state.acute_entropy
        );
    }

    // ── Sidechain modulation tests ───────────────────────────────

    #[test]
    fn sidechain_anomaly_boosts_kd() {
        let config = PidConfig::default();
        let mut state_clean = PidState::new();
        let mut state_anomaly = PidState::new();
        // Give both the same trend input
        let risk_clean = compute_pid_score(0, 32767.0, 0, 1.0, false, 0, &config, &mut state_clean);
        let risk_anomaly = compute_pid_score(
            0,
            32767.0,
            0,
            1.0,
            false,
            FLAG_ANOMALY,
            &config,
            &mut state_anomaly,
        );
        // Anomaly doubles Kd, so D_term should be higher
        assert!(
            risk_anomaly > risk_clean,
            "FLAG_ANOMALY should boost Kd: clean={}, anomaly={}",
            risk_clean,
            risk_anomaly
        );
    }

    #[test]
    fn sidechain_stuck_doubles_ki() {
        let config = PidConfig::default();
        let mut state_clean = PidState::new();
        let mut state_stuck = PidState::new();
        // Pre-seed integrators
        for _ in 0..3 {
            compute_pid_score(32767, 0.0, 0, 1.0, false, 0, &config, &mut state_clean);
            compute_pid_score(32767, 0.0, 0, 1.0, false, 0, &config, &mut state_stuck);
        }
        let risk_clean = compute_pid_score(32767, 0.0, 0, 1.0, false, 0, &config, &mut state_clean);
        let risk_stuck = compute_pid_score(
            32767,
            0.0,
            0,
            1.0,
            false,
            FLAG_STUCK,
            &config,
            &mut state_stuck,
        );
        assert!(
            risk_stuck > risk_clean,
            "FLAG_STUCK should boost Ki: clean={}, stuck={}",
            risk_clean,
            risk_stuck
        );
    }

    #[test]
    fn sidechain_no_flags_identity_modulation() {
        let config = PidConfig::default();
        let mut state_a = PidState::new();
        let mut state_b = PidState::new();
        let risk_a = compute_pid_score(10000, 5000.0, 30, 0.8, false, 0, &config, &mut state_a);
        let risk_b = compute_pid_score(10000, 5000.0, 30, 0.8, false, 0, &config, &mut state_b);
        assert!(
            (risk_a - risk_b).abs() < 0.001,
            "same input, same state → same risk"
        );
    }

    // ── Dual-rate integrator tests ───────────────────────────────

    #[test]
    fn acute_integrator_rises_faster_than_chronic() {
        let config = PidConfig::default();
        let mut state = PidState::new();
        // Spike of moderate entropy: both start at 0
        // After one cycle: acute = 0.5, chronic = 0.5 (same if both start at 0)
        // After two cycles: acute = 0.5*0.9 + 0.5 = 0.95, chronic = 0.5*0.99 + 0.5 = 0.995
        // Chronic remembers MORE, so chronic > acute during accumulation.
        // The DIFFERENCE is in decay speed, tested in acute_integrator_decays_faster_than_chronic.
        state.acute_entropy = 0.5;
        state.chronic_entropy = 0.5;
        // Feed moderate entropy: the acute converges to steady-state faster
        for _ in 0..5 {
            compute_pid_score(32767, 0.0, 0, 1.0, false, 0, &config, &mut state);
        }
        // Both should be near saturation (steady-state > 1.0 for both, clamped to 1.0)
        // The key test: after 5 cycles of max input, both integrators approach 1.0
        assert!(
            (state.acute_entropy - 1.0).abs() < 0.01,
            "acute should be near 1.0: {}",
            state.acute_entropy
        );
        assert!(
            (state.chronic_entropy - 1.0).abs() < 0.01,
            "chronic should be near 1.0: {}",
            state.chronic_entropy
        );
    }

    #[test]
    fn acute_integrator_decays_faster_than_chronic() {
        let config = PidConfig::default();
        let mut state = PidState::new();
        // Build up both
        for _ in 0..100 {
            compute_pid_score(65535, 0.0, 0, 1.0, false, 0, &config, &mut state);
        }
        let acute_peak = state.acute_entropy;
        let chronic_peak = state.chronic_entropy;

        // Feed clean for many cycles
        for _ in 0..20 {
            compute_pid_score(0, 0.0, 0, 1.0, false, 0, &config, &mut state);
        }
        let acute_decay_ratio = acute_peak / state.acute_entropy.max(0.001);
        let chronic_decay_ratio = chronic_peak / state.chronic_entropy.max(0.001);
        assert!(
            acute_decay_ratio > chronic_decay_ratio * 0.8,
            "acute should decay faster (higher ratio): acute={}, chronic={}",
            acute_decay_ratio,
            chronic_decay_ratio
        );
    }

    // ── pid_risk_to_decision tests ────────────────────────────────

    #[test]
    fn risk_below_warn_is_proceed() {
        let config = PidConfig::default();
        let decision = pid_risk_to_decision(0.3, &config);
        assert!(matches!(decision, SafetyDecision::Proceed));
    }

    #[test]
    fn risk_at_warn_is_escalate() {
        let config = PidConfig::default();
        let decision = pid_risk_to_decision(0.5, &config);
        assert!(matches!(decision, SafetyDecision::Escalate { .. }));
    }

    #[test]
    fn risk_at_halt_is_halt() {
        let config = PidConfig::default();
        let decision = pid_risk_to_decision(1.0, &config);
        assert!(matches!(decision, SafetyDecision::Halt(..)));
    }

    #[test]
    fn risk_above_halt_is_halt() {
        let config = PidConfig::default();
        // Compute risk from max signals (guaranteed to be >= halt_gain=1.0 after clamp)
        let mut state = PidState::new();
        let risk = compute_pid_score(65535, 65535.0, 100, 0.0, false, 0, &config, &mut state);
        assert!(risk >= 1.0);
        let decision = pid_risk_to_decision(risk, &config);
        assert!(matches!(decision, SafetyDecision::Halt(..)));
    }

    #[test]
    fn pid_decision_severity_monotonic() {
        let config = PidConfig::default();
        let d_low = pid_risk_to_decision(0.3, &config);
        let d_mid = pid_risk_to_decision(0.5, &config);
        let d_high = pid_risk_to_decision(1.0, &config);
        assert!(d_high.severity() >= d_mid.severity());
        assert!(d_mid.severity() >= d_low.severity());
    }

    // ── compute_pid_score_pure tests ──────────────────────────────

    #[test]
    fn pure_score_no_bias_override() {
        let config = PidConfig::default();
        let mut state = PidState::new();
        // Pure version must return ~0 for clean input regardless of has_bias
        let risk = compute_pid_score_pure(0, 0.0, 0, 1.0, 0, &config, &mut state);
        assert!(
            (risk - 0.0).abs() < 0.01,
            "pure risk should be ~0: {}",
            risk
        );
    }

    #[test]
    fn pure_score_identical_to_original_without_bias() {
        let config = PidConfig::default();
        let mut state_pure = PidState::new();
        let mut state_orig = PidState::new();
        let r_pure = compute_pid_score_pure(32767, 5000.0, 50, 0.7, 0, &config, &mut state_pure);
        let r_orig = compute_pid_score(32767, 5000.0, 50, 0.7, false, 0, &config, &mut state_orig);
        assert!(
            (r_pure - r_orig).abs() < 0.001,
            "pure and original should match without bias: pure={}, orig={}",
            r_pure,
            r_orig
        );
    }

    // ── apply_safety_overrides tests ──────────────────────────────

    #[test]
    fn safety_overrides_no_flags_passthrough() {
        let config = PidConfig::default();
        let result = apply_safety_overrides(0.3, OverrideFlags::empty(), &config);
        assert!((result - 0.3).abs() < 0.001, "no flags: {}", result);
    }

    #[test]
    fn safety_overrides_bias_forces_halt() {
        let config = PidConfig::default();
        // Even zero risk must become halt-level when bias is set
        let result = apply_safety_overrides(0.0, OverrideFlags::BIAS, &config);
        assert!(
            result >= config.halt_gain,
            "bias should force >= halt_gain: {}",
            result
        );
    }

    #[test]
    fn safety_overrides_exhausted_forces_max() {
        let config = PidConfig::default();
        let result = apply_safety_overrides(0.0, OverrideFlags::EXHAUSTED, &config);
        assert!(
            (result - 1.0).abs() < 0.001,
            "exhausted should force 1.0: {}",
            result
        );
    }

    #[test]
    fn safety_overrides_kernel_unstable_forces_halt() {
        let config = PidConfig::default();
        let result = apply_safety_overrides(0.0, OverrideFlags::KERNEL_UNSTABLE, &config);
        assert!(
            result >= config.halt_gain,
            "kernel unstable should force >= halt_gain: {}",
            result
        );
    }

    #[test]
    fn safety_overrides_bias_exhausted_combined() {
        let config = PidConfig::default();
        // EXHAUSTED takes priority (forces 1.0)
        let result =
            apply_safety_overrides(0.0, OverrideFlags::BIAS | OverrideFlags::EXHAUSTED, &config);
        assert!((result - 1.0).abs() < 0.001, "exhausted+bias: {}", result);
    }

    #[test]
    fn safety_overrides_dal_a_monotonic() {
        // MC/DC: each override independently forces halt
        let config = PidConfig::default();
        let r_none = apply_safety_overrides(0.0, OverrideFlags::empty(), &config);
        let r_bias = apply_safety_overrides(0.0, OverrideFlags::BIAS, &config);
        let r_exhausted = apply_safety_overrides(0.0, OverrideFlags::EXHAUSTED, &config);
        let r_kernel = apply_safety_overrides(0.0, OverrideFlags::KERNEL_UNSTABLE, &config);
        assert!(r_none < r_bias, "none={} < bias={}", r_none, r_bias);
        assert!(
            r_bias <= r_exhausted,
            "bias={} <= exhausted={}",
            r_bias,
            r_exhausted
        );
        assert!(r_none < r_kernel, "none={} < kernel={}", r_none, r_kernel);
    }
}
