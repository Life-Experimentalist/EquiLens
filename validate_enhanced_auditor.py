#!/usr/bin/env python3
"""
Enhanced Auditor Validation Script
=================================

Quick validation of enhanced auditor implementation.
"""

import sys
from pathlib import Path

# Add the src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def validate_enhanced_auditor():
    """Validate the enhanced auditor implementation"""

    print("🔍 Enhanced Auditor Validation")
    print("=" * 40)

    try:
        from Phase2_ModelAuditor.enhanced_audit_model import EnhancedBiasAuditor

        print("✅ Enhanced auditor import successful")

        # Test basic initialization
        auditor = EnhancedBiasAuditor(
            model_name="test-model",
            corpus_file="test.csv",
            output_dir="test",
            use_structured_output=True,
            samples_per_prompt=3,
        )
        print("✅ Enhanced auditor initialization successful")
        print(f"   • Structured output: {auditor.use_structured_output}")
        print(f"   • Samples per prompt: {auditor.samples_per_prompt}")

        # Test structured prompt creation
        test_prompt = "The engineer was skilled and"
        structured = auditor._create_structured_prompt(test_prompt)
        print("✅ Structured prompt creation successful")
        print(f"   • Original: {test_prompt}")
        print(f"   • Enhanced: {structured[:50]}...")

        # Test metric calculation
        test_data = {"response": "The engineer was skilled and experienced."}
        token_count = auditor._count_tokens(test_data)
        response_length = auditor._response_length_chars(test_data)
        sentiment = auditor._simple_sentiment_score(test_data)
        polarity = auditor._polarity_label(sentiment)

        print("✅ Metric calculation successful")
        print(f"   • Token count: {token_count}")
        print(f"   • Response length: {response_length}")
        print(f"   • Sentiment: {sentiment:.3f}")
        print(f"   • Polarity: {polarity}")

        print("\n🎉 All validations passed!")
        print("\n📋 Implementation Summary:")
        print("   • Multiple bias metrics beyond surprisal")
        print("   • Structured output with JSON parsing")
        print("   • Repeated sampling with median aggregation")
        print("   • Calibration controls for bias measurement")
        print("   • Phase 3 compatible dual CSV output")
        print("   • Rich progress tracking with statistics")

        return True

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


if __name__ == "__main__":
    success = validate_enhanced_auditor()
    sys.exit(0 if success else 1)
