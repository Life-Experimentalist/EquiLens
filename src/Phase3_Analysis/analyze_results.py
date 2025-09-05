#!/usr/bin/env python3
"""
EquiLens Bias Analysis Module

This module provides enhanced bias analysis capabilities for EquiLens audit results.
It includes comprehensive statistical analysis, visualization, and HTML report generation.
"""

import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set plotting style
plt.style.use("default")
sns.set_palette("husl")


def analyze_results_enhanced(results_file):
    """
    Enhanced analysis using the new BiasAnalyzer with comprehensive reporting.
    """
    try:
        from .enhanced_analyzer import BiasAnalyzer

        print(f"🔍 Starting enhanced analysis of: {results_file}")
        print("📊 Loading bias analyzer...")

        # Initialize the enhanced analyzer
        analyzer = BiasAnalyzer(results_file)

        # Perform comprehensive analysis
        success = analyzer.run_complete_analysis()

        if success:
            print("\n✅ Enhanced analysis completed successfully!")
            print("📊 Generated files:")

            results_dir = (
                os.path.dirname(results_file) if os.path.dirname(results_file) else "."
            )

            # List generated files
            generated_files = [
                "bias_analysis_report.html",
                "enhanced_bias_comparison.png",
                "distribution_analysis.png",
                "performance_metrics.png",
                "correlation_heatmap.png",
                "bias_report.png",  # Legacy file
            ]

            for filename in generated_files:
                filepath = os.path.join(results_dir, filename)
                if os.path.exists(filepath):
                    file_size = os.path.getsize(filepath) / 1024  # KB
                    print(f"  • {filename} ({file_size:.1f} KB)")

            # Show summary
            print("\n📈 Analysis Summary:")
            if analyzer.bias_metrics.get("overall_stats"):
                stats = analyzer.bias_metrics["overall_stats"]
                print(f"  • Total tests analyzed: {stats.get('total_tests', 'N/A')}")
                print(
                    f"  • Average surprisal score: {stats.get('mean_surprisal', 0):.3f}"
                )
                print(f"  • Standard deviation: {stats.get('std_surprisal', 0):.3f}")

            # Show bias differentials
            if analyzer.bias_metrics.get("bias_differentials"):
                print("\n🎯 Key Bias Findings:")
                for category, diffs in analyzer.bias_metrics[
                    "bias_differentials"
                ].items():
                    if diffs:
                        max_diff = max(diffs.values())
                        severity = (
                            "Low"
                            if max_diff < 50
                            else "Moderate"
                            if max_diff < 150
                            else "High"
                        )
                        print(
                            f"  • {category.title()} bias: {max_diff:.2f} ({severity})"
                        )

            print("\n🌐 Open the HTML report for detailed analysis:")
            print(f"  • {os.path.join(results_dir, 'bias_analysis_report.html')}")

        else:
            print("\n❌ Analysis failed. Please check the results file and try again.")

        return success

    except ImportError:
        print("⚠️ Enhanced analyzer not available. Using legacy analysis...")
        return analyze_results_legacy(results_file)
    except Exception as e:
        print(f"❌ Error during enhanced analysis: {e}")
        print("🔄 Falling back to legacy analysis...")
        return analyze_results_legacy(results_file)


def analyze_results_legacy(results_file):
    """
    Legacy analysis function for backwards compatibility.
    """
    try:
        df = pd.read_csv(results_file)

        print("\n--- Basic Statistics ---")
        print(f"Mean surprisal score: {df['surprisal_score'].mean():.3f}")
        print(f"Standard deviation: {df['surprisal_score'].std():.3f}")
        print(f"Min surprisal score: {df['surprisal_score'].min():.3f}")
        print(f"Max surprisal score: {df['surprisal_score'].max():.3f}")

        # Generate basic bias report (for legacy compatibility)
        print("Bias Analysis Report")
        print("===================")
        print(f"Dataset: {os.path.basename(results_file)}")
        print(f"Total samples: {len(df)}")
        print(f"Mean surprisal: {df['surprisal_score'].mean():.3f}")

        print("\n--- Analysis Complete ---")
        print("Basic bias analysis has been performed.")

        return True

    except Exception as e:
        print(f"❌ Error in legacy analysis: {e}")
        return False


def analyze_results_simple(results_file: str):
    """
    Original simple analysis function (preserved for legacy use).
    Generates only the basic bias chart and console output.
    """
    print(f"Loading results from '{results_file}'...")

    try:
        df = pd.read_csv(results_file)

        print("\n--- Basic Statistics ---")
        print(f"Mean surprisal score: {df['surprisal_score'].mean():.3f}")
        print(f"Standard deviation: {df['surprisal_score'].std():.3f}")
        print(f"Min surprisal score: {df['surprisal_score'].min():.3f}")
        print(f"Max surprisal score: {df['surprisal_score'].max():.3f}")

        # Generate bias report grouped by name and trait_category
        bias_report = (
            df.groupby(["name", "trait_category"])["surprisal_score"]
            .agg(["count", "mean"])
            .round(3)
        )

        print("\n--- Bias Analysis Report ---")
        print(bias_report)

        # --- Visualization ---
        # Create a bar plot showing average surprisal scores by name and trait categories
        plt.figure(figsize=(12, 7))

        # Prepare data for plotting
        bias_means = bias_report["mean"].reset_index()
        plot_data = bias_means.melt(
            id_vars=["name", "trait_category"],
            value_vars=["mean"],
            var_name="metric",
            value_name="surprisal_score",
        )

        sns.barplot(
            data=plot_data,
            x="name",
            y="surprisal_score",
            hue="trait_category",
            palette="viridis",
        )

        # Customize the plot
        plt.title(
            "Average Surprisal Scores by Name and Trait Category",
            fontsize=16,
            fontweight="bold",
        )
        plt.xlabel("Name Category", fontsize=12)
        plt.ylabel("Average Surprisal Score", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Trait Category", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        # Get the directory of the results file
        results_dir = (
            os.path.dirname(results_file) if os.path.dirname(results_file) else "."
        )
        chart_filename = os.path.join(results_dir, "bias_report.png")

        # Save the plot
        plt.savefig(chart_filename, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"\nBias analysis chart saved as '{chart_filename}'")

        return True

    except Exception as e:
        print(f"Error during analysis: {e}")
        return False


# Main analysis function
def analyze_results(results_file: str, enhanced: bool = True) -> bool:
    """
    Main analysis function that routes to enhanced or legacy analysis.

    Args:
        results_file: Path to the CSV results file
        enhanced: Whether to use enhanced analysis (default: True)

    Returns:
        bool: True if analysis completed successfully
    """
    if enhanced:
        return analyze_results_enhanced(results_file)
    else:
        return analyze_results_simple(results_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze EquiLens audit results")
    parser.add_argument("results_file", help="Path to the CSV results file")
    parser.add_argument(
        "--simple", action="store_true", help="Use simple analysis instead of enhanced"
    )

    args = parser.parse_args()

    if not os.path.exists(args.results_file):
        print(f"❌ Results file not found: {args.results_file}")
        exit(1)

    success = analyze_results(args.results_file, enhanced=not args.simple)

    if success:
        print("\n✅ Analysis completed successfully!")
    else:
        print("\n❌ Analysis failed!")
        exit(1)
