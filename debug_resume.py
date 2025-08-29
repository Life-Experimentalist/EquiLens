#!/usr/bin/env python3
"""
Debug script to test resume functionality directly
"""
import sys
sys.path.insert(0, 'src')

from Phase2_ModelAuditor.enhanced_audit_model import EnhancedBiasAuditor

# Test with the progress file that has 5372 tests
resume_file = "results/llama2_latest_20250810_020300/progress_20250810_020300.json"
print(f"Testing resume with: {resume_file}")

auditor = EnhancedBiasAuditor(
    model_name="llama2:latest",
    corpus_file="src/Phase1_CorpusGenerator/corpus/audit_corpus_gender_bias.csv",
    output_dir="results"
)

print("Starting auditor with resume file...")
success = auditor.run_enhanced_audit(resume_file=resume_file)
print(f"Result: {success}")
