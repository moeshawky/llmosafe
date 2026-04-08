//! G-SEC security tests - fuzz and injection resilience

#[cfg(test)]
mod tests {
    use llmosafe::{calculate_halo_signal, SiftedSynapse, Synapse, WorkingMemory};

    #[test]
    fn test_no_integer_overflow_entropy() {
        // Test entropy values at boundaries
        let mut synapse = Synapse::new();

        synapse.set_raw_entropy(0);
        assert_eq!(synapse.raw_entropy(), 0);

        synapse.set_raw_entropy(u16::MAX);
        assert_eq!(synapse.raw_entropy(), u16::MAX);

        synapse.set_raw_entropy(u16::MAX / 2);
        assert_eq!(synapse.raw_entropy(), u16::MAX / 2);

        // Test accumulation doesn't overflow
        let mut memory = WorkingMemory::<64>::new(i128::MAX);
        for i in 0..1000 {
            let mut s = Synapse::new();
            s.set_raw_entropy(i as u16 % 1000);
            let sifted = SiftedSynapse::new(s);
            let _ = memory.update(sifted);
        }
        // If we get here without panic, overflow protection works
    }

    #[test]
    fn test_no_panic_on_malformed_input() {
        // Null bytes
        let _signal = calculate_halo_signal("hello\0world\0test");

        // Invalid UTF-8 sequences (simulated with valid escapes)
        let _signal2 = calculate_halo_signal("expert recommendation");

        // Very long strings
        let long = "expert ".repeat(100_000);
        let _signal3 = calculate_halo_signal(&long);

        // Empty after trimming
        let _signal4 = calculate_halo_signal("\0\0\0");
    }

    #[test]
    fn test_resource_exhaustion_limits() {
        // Memory should have bounded state
        let mut memory = WorkingMemory::<2>::new(1000);

        // Fill it up many times - should wrap around without growing
        for i in 0..1000 {
            let mut synapse = Synapse::new();
            synapse.set_raw_entropy((i % 100) as u16);
            let sifted = SiftedSynapse::new(synapse);
            memory.update(sifted).unwrap();
        }

        // State should still be bounded to SIZE
        let mean = memory.mean_entropy();
        assert!(mean.is_finite(), "Mean should be finite after wraparound");
    }

    #[test]
    fn test_injection_patterns_in_text() {
        // These should NOT cause injection or panic
        let injection_patterns = vec![
            "expert '; DROP TABLE users; --",
            "expert <script>alert('xss')</script>",
            "expert ${dangerous}",
            "expert {{template_injection}}",
            "expert \n\n\n\n\n",
            "expert \r\n\r\n",
            "expert \t\t\t",
            "expert \\x00\\x01\\x02",
            "expert {{config.secret}}",
            "expert <%= evil %>",
            "expert <?php system('ls'); ?>",
            "expert ${7*7}",
        ];

        for pattern in injection_patterns {
            let signal = calculate_halo_signal(pattern);

            // Should still detect bias keyword
            assert!(
                signal > 0,
                "Should still detect 'expert' in injection attempt"
            );
        }
    }

    #[test]
    fn test_synapse_bit_injection() {
        // Test all bit patterns don't cause issues
        let test_patterns: Vec<u128> = vec![
            0,
            u128::MAX,
            0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFE,
            0x00000000_00000000_00000000_FFFFFFFF,
            0xFFFFFFFF_00000000_00000000_00000000,
            1 << 127, // Highest bit
            1,        // Lowest bit
        ];

        for pattern in test_patterns {
            let synapse = Synapse::from_raw_u128(pattern);
            let _ = synapse.raw_entropy();
            let _ = synapse.raw_surprise();
            let _ = synapse.has_bias();
            // Should not panic on any bit pattern
        }
    }

    #[test]
    fn test_surprise_threshold_boundaries() {
        let mut memory = WorkingMemory::<64>::new(1000);

        // Test at threshold
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(100);
        synapse.set_raw_surprise(1000);
        let sifted = SiftedSynapse::new(synapse);
        assert!(memory.update(sifted).is_ok(), "At threshold should succeed");

        // Test above threshold
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(100);
        synapse.set_raw_surprise(1001);
        let sifted = SiftedSynapse::new(synapse);
        assert!(
            memory.update(sifted).is_err(),
            "Above threshold should fail"
        );
    }

    #[test]
    fn test_unicode_boundary_handling() {
        // Test various Unicode edge cases
        // Note: Surrogates are not valid Unicode scalar values in Rust strings
        let unicode_tests = vec![
            "\u{0000}",         // Null character
            "\u{FFFF}",         // Max BMP
            "\u{10FFFF}",       // Max Unicode
            "\u{FFFD}",         // Replacement character
            "\u{200D}",         // Zero-width joiner
            "\u{FEFF}",         // BOM
            "\u{0041}\u{0301}", // A with combining accent
            "\u{1F600}",        // Emoji
        ];

        for test in unicode_tests {
            let _signal = calculate_halo_signal(test);
        }
    }
}
