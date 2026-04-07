//! G-EDGE tests for sifter module - comprehensive boundary testing

#[cfg(test)]
mod tests {
    use llmosafe::{calculate_halo_signal, get_bias_breakdown, sift_perceptions};

    #[test]
    fn test_halo_signal_empty_string() {
        let signal = calculate_halo_signal("");
        assert_eq!(signal, 0, "Empty string should have zero halo signal");
    }

    #[test]
    fn test_halo_signal_whitespace_only() {
        let signal = calculate_halo_signal("   \t\n\r  ");
        assert_eq!(signal, 0, "Whitespace-only should have zero halo signal");
    }

    #[test]
    fn test_halo_signal_unicode_emoji() {
        let signal = calculate_halo_signal("😀😀😀");
        assert_eq!(
            signal, 0,
            "Emoji without bias keywords should have zero signal"
        );

        let signal_with_bias = calculate_halo_signal("😀 expert recommendation 😀");
        assert!(
            signal_with_bias > 0,
            "Emoji with bias should have non-zero signal"
        );
    }

    #[test]
    fn test_halo_signal_unicode_rtl() {
        let signal = calculate_halo_signal("مرحبا بالعالم");
        assert_eq!(
            signal, 0,
            "Arabic without bias keywords should have zero signal"
        );

        let signal_with_keyword = calculate_halo_signal("expert مرحبا");
        assert!(
            signal_with_keyword > 0,
            "Mixed RTL with bias keyword should have signal"
        );
    }

    #[test]
    fn test_halo_signal_combining_chars() {
        let signal = calculate_halo_signal("café résumé");
        assert_eq!(
            signal, 0,
            "Combining chars without bias should have zero signal"
        );

        let signal_with_bias = calculate_halo_signal("expert café");
        assert!(
            signal_with_bias > 0,
            "Combining chars with bias should have signal"
        );
    }

    #[test]
    fn test_sift_perceptions_empty_vec() {
        let observations: Vec<&str> = vec![];
        let result = sift_perceptions(&observations, "test");

        assert!(
            result.raw_entropy() <= 65535,
            "Should produce valid synapse"
        );
    }

    #[test]
    fn test_sift_perceptions_single_element() {
        let observations = vec!["single observation"];
        let result = sift_perceptions(&observations, "test");

        assert!(
            result.raw_entropy() < 65535,
            "Single element should have reasonable entropy"
        );
    }

    #[test]
    fn test_bias_breakdown_no_keywords() {
        let breakdown = get_bias_breakdown("hello world this is normal text");
        assert_eq!(breakdown.total(), 0, "Normal text should have zero bias");
    }

    #[test]
    fn test_bias_breakdown_multiple_categories() {
        let text = "The expert says this is a popular limited offer";
        let breakdown = get_bias_breakdown(text);

        assert!(
            breakdown.authority > 0,
            "Should detect authority bias (expert)"
        );
        assert!(
            breakdown.social_proof > 0,
            "Should detect social proof (popular)"
        );
        assert!(breakdown.scarcity > 0, "Should detect scarcity (limited)");
        // Multiple categories detected
        assert!(breakdown.total() > 0);
    }

    #[test]
    fn test_halo_signal_case_insensitive() {
        let lower = calculate_halo_signal("the expert says");
        let upper = calculate_halo_signal("THE EXPERT SAYS");
        let mixed = calculate_halo_signal("The Expert Says");

        assert!(lower > 0, "Lowercase should detect bias");
        assert!(upper > 0, "Uppercase should detect bias");
        assert!(mixed > 0, "Mixed case should detect bias");

        assert_eq!(lower, upper, "Case should not affect detection");
    }

    #[test]
    fn test_halo_signal_partial_keywords() {
        let partial = calculate_halo_signal("expertise");
        let full = calculate_halo_signal("expert");

        assert!(
            full >= partial,
            "Full keyword should score at least as high as partial"
        );
    }

    #[test]
    fn test_sift_perceptions_with_objective() {
        let observations = vec!["system is stable", "all checks pass"];
        let result = sift_perceptions(&observations, "safety analysis");

        // Both should produce valid synapses
        assert!(result.raw_entropy() <= 65535);

        // Verify objective parameter is accepted
        let result2 = sift_perceptions(&observations, "marketing copy");
        assert!(result2.raw_entropy() <= 65535);
    }

    #[test]
    fn test_halo_signal_null_bytes() {
        let signal = calculate_halo_signal("hello\0world expert");
        assert!(signal > 0, "Should handle null bytes in string");
    }

    #[test]
    fn test_halo_signal_very_long() {
        let long_text = "expert ".repeat(10000);
        let signal = calculate_halo_signal(&long_text);
        assert!(signal > 0, "Should handle very long strings");
    }
}
