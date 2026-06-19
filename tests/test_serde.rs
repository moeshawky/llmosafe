// Test code uses unwrap for assertions, raw indexing for fixed arrays,
// float comparison for exact-match tests, and arithmetic on controlled
// test inputs — all safe in test context per DO-178C.
#![cfg_attr(test, allow(clippy::unwrap_used))]
#![cfg_attr(test, allow(clippy::float_cmp))]
#![cfg_attr(test, allow(clippy::float_cmp_const))]
#![cfg_attr(test, allow(clippy::arithmetic_side_effects))]
#![cfg_attr(test, allow(clippy::indexing_slicing))]
#![cfg_attr(test, allow(clippy::as_conversions))]
#![cfg_attr(test, allow(clippy::expect_used))]
#![cfg_attr(test, allow(unused_results))]
#![cfg_attr(test, allow(clippy::shadow_reuse))]
#![cfg_attr(test, allow(clippy::shadow_same))]
#![cfg_attr(test, allow(clippy::shadow_unrelated))]
#![cfg_attr(test, allow(clippy::undocumented_unsafe_blocks))]

//! Serde round-trip property tests for types that derive Serialize + Deserialize.
//!
//! Every type in `src/` with `#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]`
//! is tested. Discovery via grep before writing:
//!
//! Types with BOTH Serialize AND Deserialize (9 total):
//!   `control_types::DesignAssuranceLevel`
//!   `llmosafe_integration::SafetyDecision`, `EscalationReason`, `PressureLevel`, `EscalationPolicy`
//!   `llmosafe_kernel::CognitiveEntropy<P,S>`, `CognitiveStability`, `Synapse`, `KernelError`
//!
//! Two types (`SafetyDecision`, `EscalationReason`) have `&'static str` fields
//! (`Warn`, `Custom` variants) without `#[serde(borrow)]`. Their derived `Deserialize`
//! requires `'de: 'static`. The helper `roundtrip_static` leaks the JSON string to `'static`.
//!
//! `EscalationPolicy` lacks `PartialEq`, so its round-trip is verified via JSON equality.

#[cfg(feature = "serde")]
use llmosafe::{
    CognitiveEntropy, CognitiveStability, DesignAssuranceLevel, EscalationPolicy, EscalationReason,
    KernelError, PressureLevel, SafetyDecision, Synapse,
};

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Round-trip for types that implement `PartialEq + DeserializeOwned`.
#[cfg(feature = "serde")]
fn roundtrip<T>(val: &T)
where
    T: serde::Serialize + serde::de::DeserializeOwned + std::fmt::Debug + std::cmp::PartialEq,
{
    let json = serde_json::to_string(val).unwrap_or_else(|e| {
        panic!("serialize failed for {val:?}: {e}");
    });
    let deserialized: T = serde_json::from_str(&json).unwrap_or_else(|e| {
        panic!("deserialize failed for JSON `{json}`: {e}");
    });
    assert_eq!(
        *val, deserialized,
        "round-trip mismatch for {val:?} via JSON `{json}`"
    );
}

/// Round-trip for types that only implement `Deserialize<'static>` (due to `&'static str`
/// fields without `#[serde(borrow)]`). Leaks the JSON buffer for `'static` bound.
#[cfg(feature = "serde")]
fn roundtrip_static<T>(val: &T)
where
    T: serde::Serialize + serde::de::Deserialize<'static> + std::fmt::Debug + std::cmp::PartialEq,
{
    let json = serde_json::to_string(val).unwrap_or_else(|e| {
        panic!("serialize failed for {val:?}: {e}");
    });
    let leaked: &'static str = Box::leak(json.into_boxed_str());
    let deserialized: T = serde_json::from_str(leaked).unwrap_or_else(|e| {
        panic!("deserialize failed for JSON `{leaked}`: {e}");
    });
    assert_eq!(
        *val, deserialized,
        "round-trip mismatch for {val:?} via JSON `{leaked}`"
    );
}

/// Round-trip via JSON equality for types that lack `PartialEq`.
#[cfg(feature = "serde")]
fn roundtrip_json_eq<T>(val: &T)
where
    T: serde::Serialize + serde::de::DeserializeOwned + std::fmt::Debug,
{
    let json = serde_json::to_string(val).unwrap_or_else(|e| {
        panic!("serialize failed for {val:?}: {e}");
    });
    let deserialized: T = serde_json::from_str(&json).unwrap_or_else(|e| {
        panic!("deserialize failed for JSON `{json}`: {e}");
    });
    let json2 = serde_json::to_string(&deserialized).unwrap_or_else(|e| {
        panic!("re-serialize failed for {deserialized:?}: {e}");
    });
    assert_eq!(
        json, json2,
        "round-trip JSON mismatch for {val:?}: first=`{json}` second=`{json2}`"
    );
}

/// Verify serialization succeeds (for types where full round-trip isn't feasible).
#[cfg(feature = "serde")]
fn assert_serialize_only<T>(val: &T)
where
    T: serde::Serialize + std::fmt::Debug,
{
    serde_json::to_string(val).unwrap_or_else(|e| {
        panic!("serialize failed for {val:?}: {e}");
    });
}

// ── DesignAssuranceLevel ────────────────────────────────────────────────────

#[cfg(feature = "serde")]
#[test]
fn design_assurance_level_roundtrip_all() {
    for dal in &[
        DesignAssuranceLevel::A,
        DesignAssuranceLevel::B,
        DesignAssuranceLevel::C,
        DesignAssuranceLevel::D,
        DesignAssuranceLevel::E,
    ] {
        roundtrip(dal);
    }
}

// ── CognitiveStability ─────────────────────────────────────────────────────

#[cfg(feature = "serde")]
#[test]
fn cognitive_stability_roundtrip_all() {
    for cs in &[
        CognitiveStability::Stable,
        CognitiveStability::Pressure,
        CognitiveStability::Unstable,
    ] {
        roundtrip(cs);
    }
}

// ── PressureLevel ──────────────────────────────────────────────────────────

#[cfg(feature = "serde")]
#[test]
fn pressure_level_roundtrip_all() {
    for pl in &[
        PressureLevel::Nominal,
        PressureLevel::Elevated,
        PressureLevel::Critical,
        PressureLevel::Emergency,
    ] {
        roundtrip(pl);
    }
}

// ── KernelError ────────────────────────────────────────────────────────────

#[cfg(feature = "serde")]
#[test]
fn kernel_error_roundtrip_all() {
    for ke in &[
        KernelError::DepthExceeded,
        KernelError::CognitiveInstability,
        KernelError::BiasHaloDetected,
        KernelError::HallucinationDetected,
        KernelError::ResourceExhaustion,
        KernelError::SelfMemoryExceeded,
        KernelError::DeadlineExceeded,
    ] {
        roundtrip(ke);
    }
}

// ── EscalationReason ────────────────────────────────────────────────────────
// Custom(&'static str) binds Deserialize to 'static.

#[cfg(feature = "serde")]
#[test]
fn escalation_reason_unit_variants_roundtrip() {
    for er in &[
        EscalationReason::EntropyApproachingLimit,
        EscalationReason::SurpriseElevated,
        EscalationReason::BiasDetected,
        EscalationReason::ResourcePressure,
        EscalationReason::AnomalyDetected,
        EscalationReason::StuckAgent,
        EscalationReason::GoalDriftDetected,
        EscalationReason::ConfidenceDecaying,
        EscalationReason::AdversarialDetected,
    ] {
        roundtrip_static(er);
    }
}

#[cfg(feature = "serde")]
#[test]
fn escalation_reason_custom_roundtrip() {
    roundtrip_static(&EscalationReason::Custom("test custom reason"));
}

// ── SafetyDecision ─────────────────────────────────────────────────────────
// Warn(&'static str) binds Deserialize to 'static.

#[cfg(feature = "serde")]
#[test]
fn safety_decision_proceed_roundtrip() {
    roundtrip_static(&SafetyDecision::Proceed);
}

#[cfg(feature = "serde")]
#[test]
fn safety_decision_warn_roundtrip() {
    roundtrip_static(&SafetyDecision::Warn("test warning"));
}

#[cfg(feature = "serde")]
#[test]
fn safety_decision_escalate_roundtrip() {
    roundtrip_static(&SafetyDecision::Escalate {
        entropy: 12345,
        reason: EscalationReason::BiasDetected,
        cooldown_ms: 5000,
    });
}

#[cfg(feature = "serde")]
#[test]
fn safety_decision_halt_roundtrip() {
    roundtrip_static(&SafetyDecision::Halt(KernelError::DepthExceeded, 42));
}

#[cfg(feature = "serde")]
#[test]
fn safety_decision_exit_roundtrip() {
    roundtrip_static(&SafetyDecision::Exit(KernelError::HallucinationDetected));
}

// ── EscalationPolicy ───────────────────────────────────────────────────────
// No PartialEq — verified via JSON equality.

#[cfg(feature = "serde")]
#[test]
fn escalation_policy_default_roundtrip() {
    roundtrip_json_eq(&EscalationPolicy::default());
}

#[cfg(feature = "serde")]
#[test]
fn escalation_policy_custom_roundtrip() {
    let policy = EscalationPolicy::default()
        .with_halt_entropy(10000)
        .with_escalate_entropy(8000)
        .with_warn_entropy(5000);
    roundtrip_json_eq(&policy);
}

#[cfg(feature = "serde")]
#[test]
fn escalation_policy_extreme_values_roundtrip() {
    let policy = EscalationPolicy::default()
        .with_halt_entropy(u16::MAX)
        .with_escalate_entropy(u16::MAX)
        .with_warn_entropy(u16::MAX);
    roundtrip_json_eq(&policy);
}

// ── CognitiveEntropy ───────────────────────────────────────────────────────

#[cfg(feature = "serde")]
type TestEntropy = CognitiveEntropy<28, 2>;

#[cfg(feature = "serde")]
#[test]
fn cognitive_entropy_roundtrip() {
    roundtrip(&TestEntropy::new(12345));
}

#[cfg(feature = "serde")]
#[test]
fn cognitive_entropy_zero_roundtrip() {
    roundtrip(&TestEntropy::new(0));
}

#[cfg(feature = "serde")]
#[test]
fn cognitive_entropy_max_roundtrip() {
    roundtrip(&TestEntropy::new(i128::MAX));
}

// ── Synapse ────────────────────────────────────────────────────────────────

#[cfg(feature = "serde")]
#[test]
fn synapse_default_roundtrip() {
    roundtrip(&Synapse::new());
}

#[cfg(feature = "serde")]
#[test]
fn synapse_with_fields_roundtrip() {
    let mut synapse = Synapse::new();
    synapse.set_raw_entropy(30000);
    synapse.set_raw_surprise(500);
    synapse.set_has_bias(true);
    synapse.set_position(100);
    synapse.set_timestamp(42);
    synapse.set_cascade_depth(5);
    synapse.set_anchor_hash(0x1A2B3C); // max 31 bits: 0x7FFFFFFF
    synapse.set_reserved(0);
    roundtrip(&synapse);
}

// ── Smoke: all types at least serialize ────────────────────────────────────

#[cfg(feature = "serde")]
#[test]
fn safety_decision_all_variants_serialize() {
    assert_serialize_only(&SafetyDecision::Proceed);
    assert_serialize_only(&SafetyDecision::Warn("warn reason"));
    assert_serialize_only(&SafetyDecision::Escalate {
        entropy: 0,
        reason: EscalationReason::SurpriseElevated,
        cooldown_ms: 0,
    });
    assert_serialize_only(&SafetyDecision::Halt(KernelError::ResourceExhaustion, 1));
    assert_serialize_only(&SafetyDecision::Exit(KernelError::DeadlineExceeded));
}

#[cfg(feature = "serde")]
#[test]
fn escalation_reason_all_variants_serialize() {
    assert_serialize_only(&EscalationReason::EntropyApproachingLimit);
    assert_serialize_only(&EscalationReason::SurpriseElevated);
    assert_serialize_only(&EscalationReason::BiasDetected);
    assert_serialize_only(&EscalationReason::ResourcePressure);
    assert_serialize_only(&EscalationReason::AnomalyDetected);
    assert_serialize_only(&EscalationReason::Custom("custom reason"));
    assert_serialize_only(&EscalationReason::StuckAgent);
    assert_serialize_only(&EscalationReason::GoalDriftDetected);
    assert_serialize_only(&EscalationReason::ConfidenceDecaying);
    assert_serialize_only(&EscalationReason::AdversarialDetected);
}
