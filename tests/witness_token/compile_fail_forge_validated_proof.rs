fn main() {
    // This MUST fail: proof field is pub(crate)
    let forged = llmosafe::ValidatedProof(());
    let synapse = llmosafe::Synapse::from_raw_u128(0);
    let validated = llmosafe::ValidatedSynapse::new(synapse);  // also pub(crate)
    let mut guard = llmosafe::ReasoningLoop::<10>::new();
    let _ = guard.next_step(validated, forged);
}
