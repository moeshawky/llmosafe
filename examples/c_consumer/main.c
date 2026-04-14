/*
 * Example: C consumer of llmosafe FFI
 * 
 * Demonstrates calling llmosafe from C using the C-ABI.
 * 
 * Build with:
 *   cargo build --release --features ffi
 *   gcc -o c_harness examples/c_consumer/main.c -L./target/release -lllmosafe -lpthread -ldl -lm
 *   LD_LIBRARY_PATH=./target/release ./c_harness
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

// C-ABI declarations
extern int32_t llmosafe_process_synapse(uint64_t synapse_bits);
extern uint16_t llmosafe_calculate_halo(const char* text, size_t text_len);
extern int32_t llmosafe_check_resources(uint32_t ceiling_mb);
extern uint8_t llmosafe_get_resource_pressure(uint32_t ceiling_mb);
extern int32_t llmosafe_get_stability(uint64_t synapse_bits);
extern uint8_t llmosafe_get_system_cpu_load();
extern uint16_t llmosafe_get_environmental_entropy();

void print_separator(const char* title) {
    printf("\n=== %s ===\n\n", title);
}

int main() {
    printf("=== llmosafe FFI Demo (C Consumer) ===\n");

    // 1. Halo signal calculation
    print_separator("Halo Signal Calculation");
    
    const char* safe_text = "This is a normal sentence.";
    uint16_t halo1 = llmosafe_calculate_halo(safe_text, strlen(safe_text));
    printf("Safe text: \"%s\"\n", safe_text);
    printf("Halo signal: %u\n\n", halo1);

    const char* biased_text = "The expert provided an official recommendation.";
    uint16_t halo2 = llmosafe_calculate_halo(biased_text, strlen(biased_text));
    printf("Biased text: \"%s\"\n", biased_text);
    printf("Halo signal: %u\n", halo2);
    printf("(Higher = more bias detected)\n");

    // 2. Resource checking
    print_separator("Resource Checking");
    
    uint32_t ceiling_mb = 1024; // 1 GB ceiling
    int32_t check_result = llmosafe_check_resources(ceiling_mb);
    printf("Memory ceiling: %u MB\n", ceiling_mb);
    printf("Check result: %d (0=ok, negative=error)\n\n", check_result);

    uint8_t pressure = llmosafe_get_resource_pressure(ceiling_mb);
    printf("Resource pressure: %u%%\n", pressure);

    // 3. Stability check
    print_separator("Stability Check");
    
    uint64_t stable_synapse = 400;  // Low entropy
    int32_t stability1 = llmosafe_get_stability(stable_synapse);
    printf("Stable synapse (entropy=400): result=%d\n", stability1);

    uint64_t unstable_synapse = 1100;  // High entropy
    int32_t stability2 = llmosafe_get_stability(unstable_synapse);
    printf("Unstable synapse (entropy=1100): result=%d\n", stability2);
    printf("(0=stable, -2=cognitive instability)\n");

    // 4. Synapse processing
    print_separator("Synapse Processing");
    
    uint64_t synapse_bits = 500;  // Valid synapse
    int32_t process_result = llmosafe_process_synapse(synapse_bits);
    printf("Process synapse (bits=500): result=%d\n", process_result);
    printf("(0=success, negative=error code)\n");

    // 5. System metrics
    print_separator("System Metrics");
    
    uint8_t cpu_load = llmosafe_get_system_cpu_load();
    printf("System CPU load: %u%%\n", cpu_load);

    uint16_t env_entropy = llmosafe_get_environmental_entropy();
    printf("Environmental entropy: %u\n", env_entropy);

    // 6. Edge cases
    print_separator("Edge Cases");
    
    // Null pointer handling
    uint16_t null_halo = llmosafe_calculate_halo(NULL, 10);
    printf("Halo with NULL pointer: %u (should be 0)\n", null_halo);

    // Zero ceiling
    int32_t zero_ceiling = llmosafe_check_resources(0);
    printf("Check with zero ceiling: %d (should be -5)\n", zero_ceiling);

    printf("\n=== Demo Complete ===\n");

    return 0;
}
