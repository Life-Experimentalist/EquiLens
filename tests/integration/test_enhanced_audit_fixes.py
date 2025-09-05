#!/usr/bin/env python3
"""
Test script to verify enhanced audit model fixes
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.Phase2_ModelAuditor.enhanced_audit_model import EnhancedBiasAuditor


def test_ollama_connection():
    """Test Ollama connection with the fixed logic"""
    print("🧪 Testing Enhanced Audit Model Fixes")
    print("=" * 50)

    # Create a test auditor
    auditor = EnhancedBiasAuditor(
        model_name="llama2:latest", corpus_file="test.csv", output_dir="test_output"
    )

    print("🔍 Testing Ollama service connection...")

    # Test the connection
    if auditor.check_ollama_service():
        print("✅ Ollama connection successful!")
        return True
    else:
        print("❌ Ollama connection failed")
        return False


if __name__ == "__main__":
    test_ollama_connection()
