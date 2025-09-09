#!/usr/bin/env python3
"""
Test script for enhanced auditor features
==========================================

This script demonstrates the new structured output and repeated sampling
features of the enhanced auditor.
"""

from src.Phase2_ModelAuditor.enhanced_audit_model import EnhancedBiasAuditor


def test_enhanced_features():
    """Test the enhanced auditor with new features"""

    print("🧪 Testing Enhanced Auditor Features")
    print("=" * 50)

    # Test configuration 1: Standard mode (baseline)
    print("\n📊 Test 1: Standard Mode (baseline)")
    auditor1 = EnhancedBiasAuditor(
        model_name="llama2:latest",
        corpus_file="src/Phase1_CorpusGenerator/corpus/quick_test_corpus.csv",
        output_dir="test_output",
        use_structured_output=False,
        samples_per_prompt=1,
    )
    print(
        f"✅ Configuration: structured={auditor1.use_structured_output}, samples={auditor1.samples_per_prompt}"
    )

    # Test configuration 2: Structured output only
    print("\n🔧 Test 2: Structured Output Mode")
    auditor2 = EnhancedBiasAuditor(
        model_name="llama2:latest",
        corpus_file="src/Phase1_CorpusGenerator/corpus/quick_test_corpus.csv",
        output_dir="test_output",
        use_structured_output=True,
        samples_per_prompt=1,
    )
    print(
        f"✅ Configuration: structured={auditor2.use_structured_output}, samples={auditor2.samples_per_prompt}"
    )

    # Test configuration 3: Repeated sampling only
    print("\n🔄 Test 3: Repeated Sampling Mode")
    auditor3 = EnhancedBiasAuditor(
        model_name="llama2:latest",
        corpus_file="src/Phase1_CorpusGenerator/corpus/quick_test_corpus.csv",
        output_dir="test_output",
        use_structured_output=False,
        samples_per_prompt=3,
    )
    print(
        f"✅ Configuration: structured={auditor3.use_structured_output}, samples={auditor3.samples_per_prompt}"
    )

    # Test configuration 4: Both features combined
    print("\n🚀 Test 4: Combined Mode (structured + sampling)")
    auditor4 = EnhancedBiasAuditor(
        model_name="llama2:latest",
        corpus_file="src/Phase1_CorpusGenerator/corpus/quick_test_corpus.csv",
        output_dir="test_output",
        use_structured_output=True,
        samples_per_prompt=3,
    )
    print(
        f"✅ Configuration: structured={auditor4.use_structured_output}, samples={auditor4.samples_per_prompt}"
    )  # Test structured prompt creation
    print("\n🧪 Testing Structured Prompt Creation")
    test_prompt = "The engineer was skilled and"
    structured_prompt = auditor4._create_structured_prompt(test_prompt)
    print(f"Original: {test_prompt}")
    print(f"Structured: {structured_prompt[:100]}...")

    # Test metric calculation methods
    print("\n📈 Testing Metric Calculation Methods")
    test_response = "The engineer was skilled and experienced in software development."
    test_response_data = {"response": test_response}

    token_count = auditor4._count_tokens(test_response_data)
    response_length = auditor4._response_length_chars(test_response_data)
    sentiment = auditor4._simple_sentiment_score(test_response_data)
    polarity = auditor4._polarity_label(sentiment)

    print(f"Response: {test_response}")
    print(f"Token count: {token_count}")
    print(f"Response length: {response_length}")
    print(f"Sentiment score: {sentiment:.3f}")
    print(f"Polarity: {polarity}")

    print("\n✅ All enhanced features initialized successfully!")
    print("\n💡 To run a full audit with these features:")
    print(
        "   python src/Phase2_ModelAuditor/enhanced_audit_model.py --model llama2:latest --corpus path/to/corpus.csv"
    )
    print("\n🔧 Note: These features are integrated into the main audit loop.")
    print("   • Structured output requests JSON responses with graceful fallback")
    print("   • Repeated sampling uses median aggregation for stable metrics")
    print(
        "   • Calibration controls measure bias impact of structured vs free-text output"
    )
    print("   • Output maintains compatibility with Phase 3 analysis")


if __name__ == "__main__":
    test_enhanced_features()
