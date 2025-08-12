#!/usr/bin/env python3
"""
Test script to verify the interactive ETA functionality in audit_model.py
"""

import tempfile
import csv
from pathlib import Path

# Create a minimal test corpus for testing
def create_test_corpus():
    """Create a minimal test corpus CSV file"""
    test_data = [
        {
            "sentence": "The doctor was professional.",
            "name_category": "neutral",
            "trait_category": "professional",
            "profession": "doctor",
            "name": "Dr. Smith",
            "trait": "professional",
            "comparison_type": "neutral",
            "template_id": "1"
        },
        {
            "sentence": "The nurse was caring.",
            "name_category": "neutral",
            "trait_category": "caring",
            "profession": "nurse",
            "name": "Nurse Johnson",
            "trait": "caring",
            "comparison_type": "neutral",
            "template_id": "2"
        }
    ]

    # Create temporary CSV file
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')

    fieldnames = ["sentence", "name_category", "trait_category", "profession", "name", "trait", "comparison_type", "template_id"]
    writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(test_data)
    temp_file.close()

    return temp_file.name

if __name__ == "__main__":
    corpus_file = create_test_corpus()
    print(f"✅ Created test corpus: {corpus_file}")
    print("\n🧪 Test the interactive ETA prompt with:")
    print(f"python src/Phase2_ModelAuditor/audit_model.py --model test_model --corpus {corpus_file}")
    print("\nNote: This will fail when trying to connect to Ollama, but you can test the ETA prompt interaction.")
    print(f"\n🗑️  Remember to clean up: rm {corpus_file}")
