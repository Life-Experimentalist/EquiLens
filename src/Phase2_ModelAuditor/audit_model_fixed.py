#!/usr/bin/env python3
"""
Simple wrapper script for audit_model.py with fixed syntax
This bypasses the broken audit_model.py and uses the enhanced_audit_model directly
"""

import argparse
import sys
import os
from pathlib import Path

# Add the source path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from Phase2_ModelAuditor.enhanced_audit_model import EnhancedBiasAuditor
    USE_ENHANCED = True
except ImportError:
    USE_ENHANCED = False
    print("❌ Enhanced auditor not available")


def main():
    parser = argparse.ArgumentParser(description="EquiLens Model Auditor")
    parser.add_argument("--model", help="Model name to audit")
    parser.add_argument("--corpus", help="Path to corpus CSV file")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--resume", help="Resume from progress file")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--eta-per-test", type=float, help="ETA per test")
    parser.add_argument("--max-workers", type=int, default=1, help="Concurrent workers")

    args = parser.parse_args()

    if args.list_models:
        print("📋 Available models:")
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                for model in models:
                    print(f"  • {model['name']}")
            else:
                print("❌ Could not connect to Ollama service")
        except Exception as e:
            print(f"❌ Error listing models: {e}")
        return

    # Require model and corpus for actual auditing
    if not args.model:
        print("❌ --model is required for auditing")
        return
    if not args.corpus:
        print("❌ --corpus is required for auditing")
        return

    if USE_ENHANCED:
        print("🚀 Using enhanced auditor with dynamic concurrency...")

        # Create enhanced auditor
        auditor = EnhancedBiasAuditor(
            model_name=args.model,
            corpus_file=args.corpus,
            output_dir=args.output_dir,
            eta_per_test=args.eta_per_test
        )

        # Set concurrency if specified
        if args.max_workers > 1:
            auditor.batch_size = args.max_workers
            print(f"🔧 Configured for {args.max_workers} concurrent workers")

        # Run audit
        success = auditor.run_enhanced_audit(resume_file=args.resume)

        if success:
            print("✅ Audit completed successfully!")
        else:
            print("❌ Audit failed!")
            sys.exit(1)
    else:
        print("❌ Enhanced auditor not available and legacy auditor has syntax errors")
        print("🔧 Please check the audit_model.py file for syntax issues")
        sys.exit(1)


if __name__ == "__main__":
    main()
