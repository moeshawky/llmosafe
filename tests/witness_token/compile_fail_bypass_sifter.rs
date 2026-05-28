fn main() {
    let synapse = llmosafe::Synapse::from_raw_u128(0);
    let sifted = llmosafe::SiftedSynapse::from_synapse(synapse);
    let mut memory = llmosafe::WorkingMemory::<64>::new(500);
    // This MUST fail: update() requires 2 arguments (sifted + proof)
    let _ = memory.update(sifted);
}
