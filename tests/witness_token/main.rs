#[test]
fn test_witness_token_compile() {
    let t = trybuild::TestCases::new();
    t.pass("tests/witness_token/compile_pass_full_pipeline.rs");
    t.compile_fail("tests/witness_token/compile_fail_forge_proof.rs");
    t.compile_fail("tests/witness_token/compile_fail_bypass_sifter.rs");
    t.compile_fail("tests/witness_token/compile_fail_forge_validated_proof.rs");
}
