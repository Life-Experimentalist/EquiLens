"""Run both auditors (stable and enhanced) and produce a comparison manifest

This helper runs the two auditor implementations on the same corpus and
produces a small JSON manifest describing their results files so the Phase3
analysis step can easily locate and compare outputs.

Usage (from project root):

python -m Phase2_ModelAuditor.run_both_auditors --model llama2:latest --corpus path/to/corpus.csv --output results/compare_run

"""

import argparse
import json
import sys
from pathlib import Path

# Add package path when running as module from repository root
sys.path.append(str(Path(__file__).parents[1]))

from Phase2_ModelAuditor import audit_model, enhanced_audit_model


def run_auditor(auditor_cls, model, corpus, output_dir, resume=None, **kwargs):
    """Instantiate auditor class and run its audit method.
    Returns path to results file(s) or None on failure.
    """
    try:
        auditor = auditor_cls(model, corpus, output_dir, **kwargs)

        # Each auditor class exposes a run method with slightly different name
        if hasattr(auditor, "run_audit"):
            ok = auditor.run_audit(resume)
            results_file = getattr(auditor, "results_file", None)
        elif hasattr(auditor, "run_enhanced_audit"):
            ok = auditor.run_enhanced_audit(resume)
            results_file = getattr(auditor, "results_file", None)
        else:
            print("❌ Auditor class does not expose run method")
            return None

        if ok:
            print(f"✅ Auditor finished: results at {results_file}")
            return str(results_file)
        else:
            print("❌ Auditor failed")
            return None

    except Exception as e:
        print(f"❌ Exception while running auditor: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run both auditors for comparison")
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output-dir", default="results/compare_run")
    parser.add_argument("--skip-enhanced", action="store_true")
    parser.add_argument("--skip-stable", action="store_true")
    parser.add_argument(
        "--resume", help="Pass a resume progress file to both auditors if desired"
    )

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    manifest = {
        "model": args.model,
        "corpus": args.corpus,
        "runs": {},
    }

    if not args.skip_stable:
        print("\n--- Running stable auditor ---\n")
        stable_results = run_auditor(
            audit_model.ModelAuditor,
            args.model,
            args.corpus,
            args.output_dir,
            args.resume,
        )
        manifest["runs"]["stable"] = stable_results

    if not args.skip_enhanced:
        print("\n--- Running enhanced auditor ---\n")
        enhanced_results = run_auditor(
            enhanced_audit_model.EnhancedBiasAuditor,
            args.model,
            args.corpus,
            args.output_dir,
            args.resume,
        )
        manifest["runs"]["enhanced"] = enhanced_results

    manifest_path = Path(args.output_dir) / f"compare_manifest_{datetime_now()}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Comparison manifest written to {manifest_path}")


def datetime_now():
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    main()
