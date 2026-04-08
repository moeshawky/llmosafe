//! Example: llmosafe as a Tower middleware
//!
//! This example shows how to wrap llmosafe around HTTP/gRPC services
//! using the Tower ecosystem.
//!
//! Run with: cargo run --example tower_middleware --features full

use llmosafe::{
    sift_perceptions, EscalationPolicy, PressureLevel, SafetyContext, SafetyDecision,
    WorkingMemory,
};

/// A mock Tower Service for demonstration.
struct MockService;

impl MockService {
    /// Process a request with cognitive safety checks.
    fn process(&self, request: &str) -> Result<String, llmosafe::KernelError> {
        // Tier 3: Sift the input
        let sifted = sift_perceptions(&[request], "safe response");

        // Tier 2: Validate through working memory
        let mut memory = WorkingMemory::<64>::new(1000);
        let validated = memory.update(sifted)?;

        // Tier 1: Bounded reasoning (simulated)
        let entropy = validated.raw_entropy();
        let surprise = validated.raw_surprise();
        let has_bias = validated.has_bias();

        // Integration: Make safety decision
        let policy = EscalationPolicy::default();
        let decision = policy.decide(entropy, surprise, has_bias);

        match decision {
            SafetyDecision::Proceed => Ok(format!("Processed: {}", request)),
            SafetyDecision::Warn(reason) => {
                eprintln!("Warning: {}", reason);
                Ok(format!("Processed with warning: {}", request))
            }
            SafetyDecision::Escalate { .. } => {
                Err(llmosafe::KernelError::BiasHaloDetected)
            }
            SafetyDecision::Halt(err) => Err(err),
        }
    }
}

fn main() {
    println!("=== llmosafe Tower Middleware Example ===\n");

    let service = MockService;

    // Safe request
    match service.process("Hello, world!") {
        Ok(response) => println!("✓ Safe request: {}", response),
        Err(e) => println!("✗ Error: {}", e),
    }

    // Biased request
    match service.process("The expert provided an official recommendation") {
        Ok(response) => println!("✓ Biased request: {}", response),
        Err(e) => println!("✗ Biased request rejected: {}", e),
    }

    // Demonstrating SafetyContext for request pipelines
    println!("\n=== SafetyContext Demo ===\n");
    let mut ctx = SafetyContext::new(EscalationPolicy::default());

    // Simulate a request pipeline
    ctx.observe(300, 50, false);
    ctx.observe(400, 80, false);
    ctx.observe(500, 120, false);

    println!("Observations: {}", ctx.observation_count());
    println!("Final decision: {:?}", ctx.finalize());

    // Pressure-aware decision
    println!("\n=== Pressure-Aware Decision ===\n");
    let policy = EscalationPolicy::default();
    let decision = policy.decide_with_pressure(400, 100, false, PressureLevel::Critical);
    println!("Decision with critical pressure: {:?}", decision);
}
