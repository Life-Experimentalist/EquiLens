#!/usr/bin/env python3
"""EquiLens Bias Analysis Module"""
import os
import warnings
warnings.filterwarnings("ignore")

def analyze_results(results_file, use_ai=True):
    from .analytics import BiasAnalytics
    print(f"Starting analysis of: {results_file}")
    analyzer = BiasAnalytics(results_file)
    return analyzer.run_complete_analysis(use_ai=use_ai)

analyze_results_enhanced = analyze_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file")
    parser.add_argument("--no-ai", action="store_true")
    args = parser.parse_args()
    analyze_results(args.results_file, use_ai=not args.no_ai)
