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
//! DAL (Design Assurance Level) safety overrides are gated behind the `dal` feature.
//! With `dal` enabled, the following tiers apply:
//!
//! | DAL | Path | Description |
//! |-----|------|-------------|
//! | A | Halt | Catastrophic — bias/exhaustion/kernel → halt |
//! | B | Escalate | Hazardous — detection flags modulate gains |
//! | C | Warn | Major — advisory only |
//! | D | Monitor | Minor — informational |
//! | E | Proceed | No effect — pass-through |
//!
//! Without the `dal` feature, `apply_safety_overrides` is a passthrough — no
//! hard limits are enforced on risk scores.
//!
//! # MC/DC
//!
//! All decision branches in `apply_safety_overrides` and
//! `pid_risk_to_decision` must have independent condition coverage.
//! See `mc_dc` annotations in the traceability matrix.

/// Design Assurance Level per DO-178C.
///
/// # Runtime vs Compile-Time
///
/// Three enforcement layers operate simultaneously:
///
/// 1. **Compile-time** (conditional compilation): The `dal` feature gate controls
///    whether `apply_safety_overrides` applies hard limits at all. Without the
///    feature, overrides are a passthrough — no safety constraints are enforced
///    in PID computation. This is the coarse control.
///
/// 2. **Runtime** (this enum): `EscalationPolicy.dal` gates the output-side
///    escalation decisions (`decide_from_detection`, `decide_with_pressure`)
///    regardless of the compile-time feature. Setting DAL E means "proceed
///    always" even if `dal` feature is active. This is the fine control.
///
/// 3. **Static check** (design review): DAL A/B paths are traceable in the
///    MC/DC matrix. Every path from detection to decision must have independent
///    condition coverage. See `invariants.toml` §4.
///
/// The runtime DAL gates decisions AFTER PID computation. The compile-time
/// feature gates safety overrides DURING PID computation. Both must pass
/// for a Halt decision to reach the actuator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

/// Control signal contract: every tier output must provide error and setpoint.
pub trait ControlSignal {
    /// Signed deviation from setpoint, normalised to `[0.0, 1.0]`.
    fn error(&self) -> f32;
    /// Reference value the loop maintains, normalised to `[0.0, 1.0]`.
    fn setpoint(&self) -> f32;
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

    /// Returns OverrideFlags(0) — a bitfield with no flags set.
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Returns true if all bits set in `other` are also set in `self`, using bitwise AND equality check.
    pub const fn contains(&self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Constructs OverrideFlags from raw u8 bits, masking to lower 3 bits (0x07).
    /// Unknown upper bits are discarded.
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits & 0x07)
    }
}

/// Combines two OverrideFlags via bitwise OR of their inner u8 values.
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
/// # Tier provenance
///
/// | Field            | Tier | Source                           |
/// |------------------|------|----------------------------------|
/// | `e_body`         | 0    | `BodyOutput.error_body`          |
/// | `e_sift`         | 3    | `SifterOutput.error_sift`        |
/// | `e_mem`          | 2    | `MemoryOutput.error_mem`         |
/// | `e_kernel`       | 1    | `KernelOutput.error_kernel`      |
/// | `trend`          | 2    | `WorkingMemory::trend()`         |
/// | `classifier_prob`| 3    | `SifterOutput.classifier_prob`   |
/// | `has_bias`       | 3    | `SifterOutput.has_bias`          |
/// | `detection_flags`| DET  | Packed from 6 detectors          |
/// | `pressure`       | 0    | `BodyOutput.pressure`            |
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
    /// Constructor taking 9 parameters and storing them directly into corresponding
    /// struct fields. No validation or transformation performed — raw field assignment.
    /// Uses #[allow(clippy::too_many_arguments)] because a builder pattern would
    /// require allocation, inappropriate for no_std Tier-2.
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
    fn test_dal_ordering() {
        assert!(DesignAssuranceLevel::A as u8 == 0);
        assert!(DesignAssuranceLevel::E as u8 == 4);
    }
}
