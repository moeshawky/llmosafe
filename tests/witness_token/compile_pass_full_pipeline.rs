fn main() {
    let (sifted, proof) = llmosafe::sift_perceptions(&["valid input"], "test");
    let mut memory = llmosafe::WorkingMemory::<64>::new(500);
    let (validated, vproof) = memory.update(sifted, proof).unwrap();
    let mut guard = llmosafe::ReasoningLoop::<10>::new();
    guard.next_step(validated, vproof).unwrap();
}
