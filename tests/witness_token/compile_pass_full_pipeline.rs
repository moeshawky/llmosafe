fn main() {
    let (sifted, proof) = llmosafe::sift_text("valid input");
    let mut memory = llmosafe::WorkingMemory::<64>::new(500);
    let _ = memory.update(sifted, proof);
    // Pipeline may reject based on classifier — that's valid behavior.
    // This test checks the API compiles, not runtime outcome.
}
