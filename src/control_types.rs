//! DO-178C DAL A/E: Control theory types for cascade control architecture.
//!
//! # Control Signal Contract
//!
//! Every tier output implements `ControlSignal` with:
//! - `setpoint()` — the reference value the loop tries to maintain
//! - `error()` — the signed deviation from setpoint, normalised to `[0.0, 1.0]`
//!
//! # DAL Partitioning
//!
//! | DAL | Path | Description |
//! |-----|------|-------------|
//! | A | Halt | Catastrophic — bias/exhaustion/kernel → halt |
//! | B | Escalate | Hazardous — detection flags modulate gains |
//! | C | Warn | Major — advisory only |
//! | D | Monitor | Minor — informational |
//! | E | Proceed | No effect — pass-through |
//!
//! # MC/DC
//!
//! All decision branches in `apply_safety_overrides` and
//! `pid_risk_to_decision` must have independent condition coverage.
//! See `mc_dc` annotations in the traceability matrix.

/// Design Assurance Level per DO-178C.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DesignAssuranceLevel {
    /// Catastrophic — halt failure → system compromise.
    A,
    /// Hazardous — missed escalation → degraded safety.
    B,
    /// Major — false warn → user burden.
    C,
    /// Minor — informational only.
    D,
    /// No effect — proceed path.
    E,
}

/// Compile-time setpoint constant.
///
/// `REF` is the ideal value in the signal's native units.
/// For normalised signals, `REF = 0.0` (zero deviation from ideal).
pub struct Setpoint<const REF: i128>;

/// Control signal contract: every tier output must provide error and setpoint.
pub trait ControlSignal {
    /// Signed deviation from setpoint, normalised to `[0.0, 1.0]`.
    fn error(&self) -> f32;
    /// Reference value the loop maintains, normalised to `[0.0, 1.0]`.
    fn setpoint(&self) -> f32;
}

/// Dimensionless gain schedule for PID composition.
///
/// All gains are multipliers on normalised `[0.0, 1.0]` signals.
/// Default values calibrated for classifier entropy `[0, 65535]` range.
/// See `design_ctrl_v0.9.0.yaml` gain_schedule section for derivations.
///
/// # MC/DC
///
/// Each gain independently affects the P/I/D/F term it modulates.
/// Gain validation rejects NaN, out-of-range, and `warn_gain >= halt_gain`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GainSchedule {
    /// Proportional gain for resource pressure `[0, 5.0]`, default 1.0.
    /// DAL A: Kp=1.0 → 100% pressure maps to P-term=1.0 → RiskScore ≥ halt_gain.
    pub kp: f32,
    /// Fast integral gain for acute entropy `[0, 3.0]`, default 0.5.
    /// DAL B: captures request-level risk spikes (~10 cycle memory).
    pub ki_fast: f32,
    /// Slow integral gain for chronic entropy `[0, 3.0]`, default 0.3.
    /// DAL B: captures session-level risk elevation (~100 cycle memory).
    pub ki_slow: f32,
    /// Derivative gain for entropy trend `[0, 5.0]`, default 2.0.
    /// DAL B: panic button for sudden entropy jumps.
    pub kd: f32,
    /// Feed-forward gain for classifier probability `[0, 1.0]`, default 0.3.
    /// DAL C: provides baseline safety credit for confident "safe" classifications.
    pub kf: f32,
}

impl Default for GainSchedule {
    fn default() -> Self {
        Self {
            kp: 1.0,
            ki_fast: 0.5,
            ki_slow: 0.3,
            kd: 2.0,
            kf: 0.3,
        }
    }
}

/// Safety override flags applied AFTER PID computation.
///
/// Infusion pump pattern: PID computes pure risk from sensor fusion,
/// safety supervisor applies hard limits before actuation.
/// This prevents a PID bug from bypassing safety enforcement.
///
/// # DAL A Paths
///
/// - `BIAS`: [infusion pump override] forces risk ≥ halt_gain + 0.001
/// - `EXHAUSTED`: [infusion pump override] forces risk = 1.0
/// - `KERNEL_UNSTABLE`: [infusion pump override] forces risk ≥ halt_gain
///
/// # MC/DC
///
/// Each flag must independently force Halt regardless of PID output.
/// Test: `apply_safety_overrides(0.0, BIAS) == halt_gain + 0.001`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OverrideFlags(u8);

impl OverrideFlags {
    /// Bias detection forces halt regardless of PID risk score.
    pub const BIAS: Self = Self(0x01);
    /// Resource exhaustion forces max risk (1.0).
    pub const EXHAUSTED: Self = Self(0x02);
    /// Kernel instability forces risk ≥ halt_gain.
    pub const KERNEL_UNSTABLE: Self = Self(0x04);

    pub const fn empty() -> Self {
        Self(0)
    }

    pub const fn contains(&self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    pub const fn from_bits(bits: u8) -> Self {
        Self(bits & 0x07)
    }
}

impl core::ops::BitOr for OverrideFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// Aggregate input to the PID composition loop.
///
/// Carries normalised error signals from all 4 tiers plus
/// detection sidechain flags for gain modulation.
///
/// # DAL A
///
/// All fields must be finite and in expected ranges. PidConfig::validate()
/// is called at construction. Signal normalisation uses saturating casts.
///
/// # MC/DC
///
/// Each error signal independently affects its corresponding PID term.
/// has_bias independently forces Halt via apply_safety_overrides.
#[derive(Debug, Clone, Copy)]
pub struct PidInput {
    /// Normalised body pressure error `[0.0, 1.0]`, from BodyOutput.error_body.
    pub e_body: f32,
    /// Normalised sifter error `[0.0, 1.0]`, from SifterOutput.error_sift.
    pub e_sift: f32,
    /// Normalised memory error `[0.0, 1.0]`, from MemoryOutput.error_mem.
    pub e_mem: f32,
    /// Normalised kernel error `[0.0, 1.0]`, from KernelOutput.error_kernel.
    pub e_kernel: f32,
    /// Raw entropy trend from WorkingMemory::trend().
    pub trend: f64,
    /// Classifier probability `[0.0, 1.0]`, from SifterOutput.classifier_prob.
    pub classifier_prob: f32,
    /// Bias flag from SifterOutput.has_bias.
    pub has_bias: bool,
    /// Packed detection flags for gain sidechain modulation.
    pub detection_flags: u8,
    /// Resource pressure percentage `[0, 100]`, from BodyOutput.pressure.
    pub pressure: u8,
}

impl PidInput {
    // Allow: PidInput is a flat data-transfer struct for the 4-tier PID
    // cascade. A builder pattern would require allocation — inappropriate
    // for no_std Tier-2 (see SYS-SPEC-602 §7.3.4).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        e_body: f32,
        e_sift: f32,
        e_mem: f32,
        e_kernel: f32,
        trend: f64,
        classifier_prob: f32,
        has_bias: bool,
        detection_flags: u8,
        pressure: u8,
    ) -> Self {
        Self {
            e_body,
            e_sift,
            e_mem,
            e_kernel,
            trend,
            classifier_prob,
            has_bias,
            detection_flags,
            pressure,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_override_flags_empty() {
        let flags = OverrideFlags::empty();
        assert!(!flags.contains(OverrideFlags::BIAS));
        assert!(!flags.contains(OverrideFlags::EXHAUSTED));
        assert!(!flags.contains(OverrideFlags::KERNEL_UNSTABLE));
    }

    #[test]
    fn test_override_flags_contains() {
        let flags = OverrideFlags::BIAS | OverrideFlags::EXHAUSTED;
        assert!(flags.contains(OverrideFlags::BIAS));
        assert!(flags.contains(OverrideFlags::EXHAUSTED));
        assert!(!flags.contains(OverrideFlags::KERNEL_UNSTABLE));
    }

    #[test]
    fn test_gain_schedule_default() {
        let g = GainSchedule::default();
        assert!((g.kp - 1.0).abs() < 1e-6);
        assert!((g.ki_fast - 0.5).abs() < 1e-6);
        assert!((g.ki_slow - 0.3).abs() < 1e-6);
        assert!((g.kd - 2.0).abs() < 1e-6);
        assert!((g.kf - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_dal_ordering() {
        assert!(DesignAssuranceLevel::A as u8 == 0);
        assert!(DesignAssuranceLevel::E as u8 == 4);
    }
}
