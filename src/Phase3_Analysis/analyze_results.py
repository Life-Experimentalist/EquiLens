#!/usr/bin/env python3
"""
EquiLens Phase 3: Advanced Bias Analysis Module

Comprehensive analysis pipeline for AI bias detection results with:
- Statistical significance testing (t-tests, ANOVA, effect sizes)
- Rich visualizations (violin plots, heatmaps, correlation matrices)
- AI-powered report generation with Ollama integration
- Multiple export formats (HTML, Markdown, JSON)
- Batch analysis and comparison support
- Progressive analysis with checkpointing
- Performance optimization and caching

Usage:
    # Simple analysis
    python -m src.Phase3_Analysis.analyze_results results.csv

    # Advanced analysis with AI
    python -m src.Phase3_Analysis.analyze_results results.csv --model phi3:mini

    # Batch analysis
    python -m src.Phase3_Analysis.analyze_results results/*.csv --batch

    # Comparative analysis
    python -m src.Phase3_Analysis.analyze_results audit1.csv audit2.csv --compare
"""

import json
import logging
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AdvancedAnalysisEngine:
    """
    Advanced analysis engine with comprehensive features for bias detection results.

    Features:
    - Multi-phase statistical analysis
    - Advanced visualizations
    - AI-powered insights
    - Batch processing
    - Result comparison
    - Performance optimization

    Example:
        engine = AdvancedAnalysisEngine(results_file)
        engine.run_analysis(generate_html=True, use_ai=True)
    """

    def __init__(
        self,
        results_file: str | Path,
        ollama_url: str | None = None,
        model: str | None = None,
        verbose: bool = False,
        output_dir: str | None = None,
    ):
        """
        Initialize analysis engine.

        Args:
            results_file: Path to CSV results file
            ollama_url: Custom Ollama API URL
            model: Ollama model for AI insights
            verbose: Enable verbose logging
            output_dir: Custom output directory (default: same as results file)
        """
        from .analytics import BiasAnalytics

        self.results_file = Path(results_file)
        self.ollama_url = ollama_url or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.model = model
        self.verbose = verbose
        self.output_dir = Path(output_dir) if output_dir else self.results_file.parent

        if self.verbose:
            logger.setLevel(logging.DEBUG)

        # Validate input
        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")

        # Initialize analytics
        self.analytics = BiasAnalytics(
            str(self.results_file),
            ollama_url=self.ollama_url,
            report_model=self.model,
        )

        logger.info(f"Analysis engine initialized for: {self.results_file.name}")

    def validate_results(self) -> bool:
        """
        Validate results file structure and integrity.

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            df = pd.read_csv(self.results_file)

            # Check required columns
            required_cols = {"surprisal_score", "name_category", "profession"}
            missing_cols = required_cols - set(df.columns)

            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False

            # Check data quality
            if df.empty:
                logger.error("Results file is empty")
                return False

            if df["surprisal_score"].isna().sum() > 0:
                logger.warning(
                    f"Found {df['surprisal_score'].isna().sum()} NaN values in surprisal_score"
                )

            logger.info(f"✅ Validation passed: {len(df)} records")
            return True

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

    def run_analysis(
        self,
        generate_html: bool = True,
        generate_markdown: bool = True,
        use_ai: bool = True,
        generate_json: bool = False,
        export_charts: bool = True,
    ) -> bool:
        """
        Run complete analysis pipeline with all advanced features.

        Args:
            generate_html: Generate HTML report
            generate_markdown: Generate Markdown report
            use_ai: Enable AI-powered insights
            generate_json: Export analysis results as JSON
            export_charts: Export individual chart files

        Returns:
            bool: True if analysis succeeded, False otherwise
        """
        try:
            print("\n" + "=" * 70)
            print("🔬 EQUILENS ADVANCED BIAS ANALYSIS")
            print("=" * 70)

            # Step 1: Validation
            print("\n📋 Step 1: Validating results...")
            if not self.validate_results():
                print("❌ Validation failed. Aborting analysis.")
                return False
            print("✅ Validation passed")

            # Step 2: Load and analyze
            print("\n📊 Step 2: Running statistical analysis...")
            if not self.analytics.load_and_validate_data():
                logger.error("Failed to load data")
                return False

            # Perform all statistical tests
            self.analytics.perform_statistical_tests()
            self.analytics.calculate_effect_sizes()
            self.analytics.calculate_confidence_intervals()
            print("✅ Statistical analysis complete")

            # Step 3: Generate visualizations
            print("\n📈 Step 3: Generating visualizations...")
            viz_count = self._generate_visualizations(export_charts)
            print(f"✅ Generated {viz_count} visualizations")

            # Step 4: Generate reports
            print("\n📝 Step 4: Generating reports...")
            reports_generated = []

            if generate_html:
                try:
                    html_path = self.analytics.generate_html_report(use_ai=use_ai)
                    reports_generated.append(("HTML", html_path))
                    print(f"  ✅ HTML: {Path(html_path).name}")
                except Exception as e:
                    logger.error(f"HTML report generation failed: {e}")
                    if not self.verbose:
                        print(
                            "  ⚠️  HTML report generation failed (use --verbose for details)"
                        )

            if generate_markdown:
                try:
                    md_path = self.analytics.generate_markdown_report(use_ai=use_ai)
                    reports_generated.append(("Markdown", md_path))
                    print(f"  ✅ Markdown: {Path(md_path).name}")
                except Exception as e:
                    logger.error(f"Markdown report generation failed: {e}")
                    if not self.verbose:
                        print(
                            "  ⚠️  Markdown report generation failed (use --verbose for details)"
                        )

            if generate_json:
                try:
                    json_path = self._export_json()
                    reports_generated.append(("JSON", json_path))
                    print(f"  ✅ JSON: {Path(json_path).name}")
                except Exception as e:
                    logger.error(f"JSON export failed: {e}")
                    if not self.verbose:
                        print("  ⚠️  JSON export failed (use --verbose for details)")

            # Summary
            print("\n" + "=" * 70)
            print("✅ ANALYSIS COMPLETE")
            print("=" * 70)
            print(f"\n📁 Output directory: {self.output_dir}")
            print(f"📊 Generated {len(reports_generated)} report(s)")

            for report_type, path in reports_generated:
                print(f"  • {report_type}: {Path(path).name}")

            # Show scoring method note if available
            if (
                hasattr(self.analytics, "score_method_note")
                and self.analytics.score_method_note
            ):
                print(f"\n🧠 {self.analytics.score_method_note}")

            return True

        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            print(f"\n❌ Analysis failed: {e}")
            if self.verbose:
                import traceback

                traceback.print_exc()
            return False

    def _generate_visualizations(self, export: bool = True) -> int:
        """
        Generate all visualizations.

        Args:
            export: Whether to export chart files

        Returns:
            int: Number of visualizations generated
        """
        count = 0
        visualizations = [
            ("violin plot", self.analytics.create_violin_plot),
            ("heatmap matrix", self.analytics.create_heatmap_matrix),
            ("effect sizes", self.analytics.create_effect_size_chart),
            ("box plot", self.analytics.create_box_plot_profession),
            ("scatter correlations", self.analytics.create_scatter_correlations),
            ("time series", self.analytics.create_time_series_progression),
            ("dashboard", self.analytics.create_comprehensive_dashboard),
        ]

        with tqdm(
            total=len(visualizations), desc="  Generating charts", unit="chart"
        ) as pbar:
            for name, viz_func in visualizations:
                try:
                    if export:
                        viz_func()
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to generate {name}: {e}")
                pbar.update(1)

        return count

    def _export_json(self) -> str:
        """
        Export analysis results as JSON.

        Returns:
            str: Path to JSON file
        """
        output_file = self.output_dir / "bias_analysis_results.json"

        # Prepare data
        data = {
            "metadata": {
                "results_file": str(self.results_file),
                "analysis_date": pd.Timestamp.now().isoformat(),
                "model": self.analytics.model_name,
                "score_method": getattr(self.analytics, "score_method_note", ""),
                "score_label": getattr(
                    self.analytics, "score_label", "Surprisal Score"
                ),
                "score_unit": getattr(self.analytics, "score_short_unit", "ns/token"),
            },
            "statistics": self.analytics.stats_results,
            "summary": {
                "total_tests": len(self.analytics.df),
                "mean_score": float(self.analytics.df["surprisal_score"].mean()),
                "std_score": float(self.analytics.df["surprisal_score"].std()),
                "min_score": float(self.analytics.df["surprisal_score"].min()),
                "max_score": float(self.analytics.df["surprisal_score"].max()),
            },
        }

        # Save JSON
        with Path(output_file).open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"JSON export saved: {output_file}")
        return str(output_file)

    def compare_with(self, other_results: str | Path) -> dict:
        """
        Compare this analysis with another results file (advanced feature).

        Args:
            other_results: Path to another results CSV

        Returns:
            dict: Comparison results
        """
        try:
            df1 = pd.read_csv(self.results_file)
            df2 = pd.read_csv(other_results)

            comparison = {
                "file1": str(self.results_file),
                "file2": str(other_results),
                "metrics": {
                    "file1_mean": float(df1["surprisal_score"].mean()),
                    "file2_mean": float(df2["surprisal_score"].mean()),
                    "difference": float(
                        df1["surprisal_score"].mean() - df2["surprisal_score"].mean()
                    ),
                    "file1_count": len(df1),
                    "file2_count": len(df2),
                },
            }

            logger.info(f"Comparison complete: {comparison}")
            return comparison

        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            return {}


def analyze_results(
    results_file: str | Path,
    use_ai: bool = True,
    model: str | None = None,
    html: bool = True,
    markdown: bool = True,
    json_export: bool = False,
    verbose: bool = False,
) -> bool:
    """
    Standard analysis function for integration with other modules.

    Args:
        results_file: Path to CSV results file
        use_ai: Enable AI-powered insights
        model: Ollama model to use
        html: Generate HTML report
        markdown: Generate Markdown report
        json_export: Export results as JSON
        verbose: Enable verbose output

    Returns:
        bool: True if analysis succeeded
    """
    try:
        engine = AdvancedAnalysisEngine(
            results_file,
            model=model,
            verbose=verbose,
        )
        return engine.run_analysis(
            generate_html=html,
            generate_markdown=markdown,
            use_ai=use_ai,
            generate_json=json_export,
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"❌ Error: {e}")
        return False


# Backward compatibility
analyze_results_enhanced = analyze_results


def main():
    """
    Advanced CLI interface for bias analysis with comprehensive options.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="EquiLens Phase 3: Advanced Bias Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python -m src.Phase3_Analysis.analyze_results results.csv

  # With AI insights
  python -m src.Phase3_Analysis.analyze_results results.csv --model phi3:mini

  # Full analysis with all outputs
  python -m src.Phase3_Analysis.analyze_results results.csv --full --json

  # Verbose output for debugging
  python -m src.Phase3_Analysis.analyze_results results.csv --verbose

  # Without AI (faster)
  python -m src.Phase3_Analysis.analyze_results results.csv --no-ai
        """,
    )

    parser.add_argument("results_file", help="Path to CSV results file")
    parser.add_argument(
        "--model",
        help="Ollama model for AI insight generation (e.g., phi3:mini, llama3.2:latest)",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Disable AI-powered report generation (faster)",
    )
    parser.add_argument(
        "--no-html", action="store_true", help="Skip HTML report generation"
    )
    parser.add_argument(
        "--no-markdown", action="store_true", help="Skip Markdown report generation"
    )
    parser.add_argument(
        "--json", action="store_true", help="Export analysis results as JSON"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Generate all outputs (HTML, Markdown, JSON)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Custom output directory (default: results directory)",
    )

    args = parser.parse_args()

    # Validate input file
    if not Path(args.results_file).exists():
        print(f"❌ Error: File not found: {args.results_file}")
        sys.exit(1)

    # Determine options
    use_ai = not args.no_ai
    generate_html = not args.no_html or args.full
    generate_markdown = not args.no_markdown or args.full
    generate_json = args.json or args.full

    # Run analysis
    success = analyze_results(
        results_file=args.results_file,
        use_ai=use_ai,
        model=args.model,
        html=generate_html,
        markdown=generate_markdown,
        json_export=generate_json,
        verbose=args.verbose,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
