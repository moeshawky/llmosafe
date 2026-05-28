fn main() {
    // This MUST fail: the proof's inner field is pub(crate)
    let forged = llmosafe::SiftedProof(());
    let synapse = llmosafe::Synapse::from_raw_u128(0);
    let sifted = llmosafe::SiftedSynapse::from_synapse(synapse);
    let mut memory = llmosafe::WorkingMemory::<64>::new(500);
    let _ = memory.update(sifted, forged);
}
