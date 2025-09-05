"""
Enhanced Bias Analysis Module for EquiLens

This module provides comprehensive bias analysis including:
- Statistical significance testing
- Multi-dimensional bias analysis
- Performance metrics analysis
- Detailed HTML report generation
- Comparative analysis capabilities
"""

import argparse
import json
import os
from datetime import datetime
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.container import BarContainer


class BiasAnalyzer:
    """Comprehensive bias analysis class for EquiLens audit results."""

    def __init__(self, results_file: str):
        """Initialize the analyzer with a results file."""
        self.results_file = results_file
        self.df: pd.DataFrame | None = None
        self.model_name = ""
        self.session_info = {}
        self.bias_metrics = {}
        self.performance_metrics = {}
        self.statistical_results = {}

    def load_data(self) -> bool:
        """Load and validate the results data."""
        print(f"Loading results from '{self.results_file}'...")
        try:
            self.df = pd.read_csv(self.results_file)
        except FileNotFoundError:
            print(f"Error: Results file not found at '{self.results_file}'")
            print("Please run the Phase 2 audit script first.")
            return False

        # Load session information if available
        results_dir = os.path.dirname(self.results_file)
        session_files = [
            f
            for f in os.listdir(results_dir)
            if f.startswith("summary_") and f.endswith(".json")
        ]
        if session_files:
            summary_file = os.path.join(results_dir, session_files[0])
            try:
                with open(summary_file) as f:
                    self.session_info = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load session info: {e}")

        # Extract model name
        self.model_name = (
            os.path.basename(self.results_file)
            .replace("results_", "")
            .replace(".csv", "")
        )

        return True

    def clean_data(self) -> bool:
        """Clean and validate the data."""
        if self.df is None:
            return False

        initial_count = len(self.df)

        # Drop rows where surprisal score could not be calculated
        self.df.dropna(subset=["surprisal_score"], inplace=True)

        # Ensure the surprisal column is numeric
        self.df["surprisal_score"] = pd.to_numeric(
            self.df["surprisal_score"], errors="coerce"
        )
        self.df.dropna(subset=["surprisal_score"], inplace=True)

        # Clean response time data
        if "response_time" in self.df.columns:
            self.df["response_time"] = pd.to_numeric(
                self.df["response_time"], errors="coerce"
            )

        final_count = len(self.df)

        if self.df.empty:
            print("No valid results found in the file. Cannot perform analysis.")
            return False

        print(
            f"Data cleaning: {initial_count} → {final_count} valid results ({final_count / initial_count * 100:.1f}% retained)"
        )
        return True

    def calculate_bias_metrics(self) -> dict[str, Any]:
        """Calculate comprehensive bias metrics."""
        if self.df is None:
            return {}

        print("Calculating bias metrics...")

        # Basic bias report (existing functionality)
        bias_report = (
            self.df.groupby(["name_category", "trait_category"])["surprisal_score"]
            .agg(["mean", "std", "count"])
            .round(4)
        )

        # Calculate bias differentials
        bias_differentials = {}

        # Gender bias analysis
        if "name_category" in self.df.columns:
            gender_categories = self.df["name_category"].unique()
            if len(gender_categories) >= 2:
                bias_differentials["gender"] = self._calculate_category_bias(
                    "name_category"
                )

        # Trait bias analysis
        if "trait_category" in self.df.columns:
            trait_categories = self.df["trait_category"].unique()
            if len(trait_categories) >= 2:
                bias_differentials["trait"] = self._calculate_category_bias(
                    "trait_category"
                )

        # Professional bias analysis
        if "profession" in self.df.columns:
            profession_stats = (
                self.df.groupby("profession")["surprisal_score"]
                .agg(["mean", "std", "count"])
                .round(4)
            )
        else:
            profession_stats = pd.DataFrame()

        self.bias_metrics = {
            "bias_report": bias_report,
            "bias_differentials": bias_differentials,
            "profession_stats": profession_stats,
            "overall_stats": {
                "mean_surprisal": self.df["surprisal_score"].mean(),
                "std_surprisal": self.df["surprisal_score"].std(),
                "min_surprisal": self.df["surprisal_score"].min(),
                "max_surprisal": self.df["surprisal_score"].max(),
                "total_tests": len(self.df),
            },
        }

        return self.bias_metrics

    def _calculate_category_bias(self, category_col: str) -> dict[str, float]:
        """Calculate bias differential for a specific category."""
        if self.df is None:
            return {}

        categories = self.df[category_col].unique()
        if len(categories) < 2:
            return {}

        category_means = self.df.groupby(category_col)["surprisal_score"].mean()

        # Calculate pairwise differences
        bias_diffs = {}
        categories_list = list(categories)
        for i in range(len(categories_list)):
            for j in range(i + 1, len(categories_list)):
                cat1, cat2 = categories_list[i], categories_list[j]
                diff = abs(category_means[cat1] - category_means[cat2])
                bias_diffs[f"{cat1}_vs_{cat2}"] = round(diff, 4)

        return bias_diffs

    def calculate_performance_metrics(self) -> dict[str, Any]:
        """Calculate performance and efficiency metrics."""
        if self.df is None:
            return {}

        print("Calculating performance metrics...")

        performance = {}

        # Response time analysis
        if "response_time" in self.df.columns:
            response_times = self.df["response_time"].dropna()
            if not response_times.empty:
                performance["response_time"] = {
                    "mean": response_times.mean(),
                    "median": response_times.median(),
                    "std": response_times.std(),
                    "min": response_times.min(),
                    "max": response_times.max(),
                    "percentile_95": response_times.quantile(0.95),
                }

        # GPU utilization analysis
        if "gpu_utilization" in self.df.columns:
            gpu_util = self.df["gpu_utilization"].dropna()
            if not gpu_util.empty:
                performance["gpu_utilization"] = {
                    "mean": gpu_util.mean(),
                    "median": gpu_util.median(),
                    "max": gpu_util.max(),
                }

        # Evaluation metrics
        if "eval_duration" in self.df.columns:
            eval_duration = self.df["eval_duration"].dropna()
            if not eval_duration.empty:
                performance["eval_duration"] = {
                    "mean_ms": eval_duration.mean()
                    / 1_000_000,  # Convert nanoseconds to ms
                    "total_hours": eval_duration.sum() / 3.6e12,  # Convert to hours
                }

        # Session metrics from summary
        if self.session_info:
            performance["session"] = {
                "total_tests": self.session_info.get("total_tests", 0),
                "completed_tests": self.session_info.get("completed_tests", 0),
                "failed_tests": self.session_info.get("failed_tests", 0),
                "success_rate": (
                    self.session_info.get("completed_tests", 0)
                    / max(self.session_info.get("total_tests", 1), 1)
                    * 100
                ),
                "total_duration_minutes": self.session_info.get("total_duration", 0)
                / 60,
            }

        self.performance_metrics = performance
        return performance

    def perform_statistical_analysis(self) -> dict[str, Any]:
        """Perform statistical significance testing."""
        if self.df is None:
            return {}

        print("Performing statistical analysis...")

        results = {}

        # T-test for gender bias (if applicable)
        if "name_category" in self.df.columns:
            gender_categories = self.df["name_category"].unique()
            if len(gender_categories) == 2:
                cat1, cat2 = gender_categories
                group1 = self.df[self.df["name_category"] == cat1]["surprisal_score"]
                group2 = self.df[self.df["name_category"] == cat2]["surprisal_score"]

                # Perform t-test (using basic statistics since scipy not available)
                results["gender_ttest"] = self._simple_ttest(group1, group2, cat1, cat2)

        # Effect size calculations
        if "name_category" in self.df.columns and "trait_category" in self.df.columns:
            results["effect_sizes"] = self._calculate_effect_sizes()

        self.statistical_results = results
        return results

    def _simple_ttest(
        self, group1: pd.Series, group2: pd.Series, label1: str, label2: str
    ) -> dict[str, Any]:
        """Perform a simple t-test using basic statistics."""
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return {"error": "Insufficient data for t-test"}

        mean1, mean2 = float(group1.mean()), float(group2.mean())

        # Use numpy for variance calculation to avoid type issues
        var1 = float(np.var(group1, ddof=1))
        var2 = float(np.var(group2, ddof=1))

        # Pooled standard error
        pooled_se = ((var1 / n1) + (var2 / n2)) ** 0.5

        # T-statistic
        t_stat = (mean1 - mean2) / pooled_se if pooled_se != 0 else 0.0

        # Degrees of freedom (approximation)
        df = n1 + n2 - 2

        return {
            "groups": f"{label1} vs {label2}",
            "group1_mean": round(mean1, 4),
            "group2_mean": round(mean2, 4),
            "difference": round(abs(mean1 - mean2), 4),
            "relative_difference_pct": round(
                abs(mean1 - mean2) / max(mean1, mean2) * 100, 2
            ),
            "t_statistic": round(float(t_stat), 4),
            "degrees_of_freedom": df,
            "sample_sizes": f"{n1}, {n2}",
        }

    def _calculate_effect_sizes(self) -> dict[str, float]:
        """Calculate Cohen's d effect sizes for bias comparisons."""
        if self.df is None:
            return {}

        effect_sizes = {}

        # Gender effect size
        if "name_category" in self.df.columns:
            gender_categories = self.df["name_category"].unique()
            if len(gender_categories) == 2:
                cat1, cat2 = gender_categories
                group1 = self.df[self.df["name_category"] == cat1]["surprisal_score"]
                group2 = self.df[self.df["name_category"] == cat2]["surprisal_score"]

                # Cohen's d using numpy for variance calculation
                var1 = float(np.var(group1, ddof=1))
                var2 = float(np.var(group2, ddof=1))
                pooled_std = ((var1 + var2) / 2) ** 0.5
                if pooled_std != 0:
                    cohens_d = (
                        abs(float(group1.mean()) - float(group2.mean())) / pooled_std
                    )
                    effect_sizes["gender_cohens_d"] = round(cohens_d, 4)

        return effect_sizes

    def create_visualizations(self, output_dir: str) -> list[str]:
        """Create comprehensive visualizations."""
        if self.df is None:
            return []

        print("Creating visualizations...")

        # Set style
        plt.style.use("default")
        sns.set_palette("viridis")

        visualization_files = []

        # 1. Main bias comparison chart (enhanced version of original)
        fig1_path = self._create_bias_comparison_chart(output_dir)
        if fig1_path:
            visualization_files.append(fig1_path)

        # 2. Distribution analysis
        fig2_path = self._create_distribution_analysis(output_dir)
        if fig2_path:
            visualization_files.append(fig2_path)

        # 3. Performance metrics visualization
        fig3_path = self._create_performance_chart(output_dir)
        if fig3_path:
            visualization_files.append(fig3_path)

        # 4. Correlation heatmap
        fig4_path = self._create_correlation_heatmap(output_dir)
        if fig4_path:
            visualization_files.append(fig4_path)

        return visualization_files

    def _create_bias_comparison_chart(self, output_dir: str) -> str:
        """Create enhanced bias comparison chart."""
        if (
            self.df is None
            or "name_category" not in self.df.columns
            or "trait_category" not in self.df.columns
        ):
            return ""

        # Reshape the data for plotting
        bias_report = self.bias_metrics["bias_report"]
        plot_data = (
            bias_report["mean"]
            .reset_index()
            .melt(
                id_vars="name_category",
                var_name="trait_category",
                value_name="average_surprisal",
            )
        )

        # Add standard deviations for error bars
        std_data = (
            bias_report["std"]
            .reset_index()
            .melt(
                id_vars="name_category",
                var_name="trait_category",
                value_name="std_surprisal",
            )
        )
        plot_data = plot_data.merge(std_data, on=["name_category", "trait_category"])

        plt.figure(figsize=(14, 8))
        ax = sns.barplot(
            data=plot_data,
            x="name_category",
            y="average_surprisal",
            hue="trait_category",
            palette="viridis",
        )

        # Add error bars and labels
        for i, container in enumerate(ax.containers):
            if hasattr(container, "datavalues"):
                # Get corresponding standard deviations
                trait_cat = plot_data["trait_category"].unique()[i]
                std_values = plot_data[plot_data["trait_category"] == trait_cat][
                    "std_surprisal"
                ].values
                ax.errorbar(
                    [bar.get_x() + bar.get_width() / 2 for bar in container],
                    [bar.get_height() for bar in container],
                    yerr=std_values,
                    fmt="none",
                    ecolor="black",
                    capsize=5,
                    alpha=0.7,
                )
                ax.bar_label(cast(BarContainer, container), fmt="%.2f")

        plt.title(
            f'Enhanced Bias Analysis for Model: "{self.model_name}"',
            fontsize=16,
            fontweight="bold",
        )
        plt.ylabel("Average Surprisal Score (Lower = More Plausible)", fontsize=12)
        plt.xlabel("Name Category", fontsize=12)
        plt.xticks(rotation=0)
        plt.legend(title="Trait Category", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        chart_path = os.path.join(output_dir, "enhanced_bias_comparison.png")
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()

        return chart_path

    def _create_distribution_analysis(self, output_dir: str) -> str:
        """Create distribution analysis visualization."""
        if self.df is None:
            return ""

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            f"Distribution Analysis - {self.model_name}", fontsize=16, fontweight="bold"
        )

        # 1. Overall surprisal distribution
        axes[0, 0].hist(
            self.df["surprisal_score"], bins=30, alpha=0.7, edgecolor="black"
        )
        axes[0, 0].set_title("Overall Surprisal Score Distribution")
        axes[0, 0].set_xlabel("Surprisal Score")
        axes[0, 0].set_ylabel("Frequency")

        # 2. Box plot by name category
        if "name_category" in self.df.columns:
            sns.boxplot(
                data=self.df, x="name_category", y="surprisal_score", ax=axes[0, 1]
            )
            axes[0, 1].set_title("Surprisal Distribution by Name Category")
            axes[0, 1].tick_params(axis="x", rotation=45)

        # 3. Box plot by trait category
        if "trait_category" in self.df.columns:
            sns.boxplot(
                data=self.df, x="trait_category", y="surprisal_score", ax=axes[1, 0]
            )
            axes[1, 0].set_title("Surprisal Distribution by Trait Category")
            axes[1, 0].tick_params(axis="x", rotation=45)

        # 4. Response time distribution (if available)
        if (
            "response_time" in self.df.columns
            and self.df["response_time"].notna().any()
        ):
            response_times = self.df["response_time"].dropna()
            axes[1, 1].hist(response_times, bins=30, alpha=0.7, edgecolor="black")
            axes[1, 1].set_title("Response Time Distribution")
            axes[1, 1].set_xlabel("Response Time (seconds)")
            axes[1, 1].set_ylabel("Frequency")
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "Response Time\nData Not Available",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )
            axes[1, 1].set_title("Response Time Distribution")

        plt.tight_layout()

        chart_path = os.path.join(output_dir, "distribution_analysis.png")
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()

        return chart_path

    def _create_performance_chart(self, output_dir: str) -> str:
        """Create performance metrics visualization."""
        if not self.performance_metrics:
            return ""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            f"Performance Metrics - {self.model_name}", fontsize=16, fontweight="bold"
        )

        # 1. Response time metrics
        if "response_time" in self.performance_metrics:
            rt_metrics = self.performance_metrics["response_time"]
            metrics = ["mean", "median", "percentile_95"]
            values = [rt_metrics.get(m, 0) for m in metrics]
            axes[0, 0].bar(metrics, values, color="skyblue", alpha=0.7)
            axes[0, 0].set_title("Response Time Metrics (seconds)")
            axes[0, 0].set_ylabel("Time (seconds)")
            for i, v in enumerate(values):
                axes[0, 0].text(
                    i, v + max(values) * 0.01, f"{v:.2f}", ha="center", va="bottom"
                )
        else:
            axes[0, 0].text(
                0.5,
                0.5,
                "Response Time\nMetrics Not Available",
                ha="center",
                va="center",
                transform=axes[0, 0].transAxes,
            )
            axes[0, 0].set_title("Response Time Metrics")

        # 2. Session success metrics
        if "session" in self.performance_metrics:
            session = self.performance_metrics["session"]
            completed = session.get("completed_tests", 0)
            failed = session.get("failed_tests", 0)
            total = session.get("total_tests", completed + failed)

            labels = ["Completed", "Failed"]
            sizes = [completed, failed]
            colors = ["lightgreen", "lightcoral"]

            if sum(sizes) > 0:
                wedges, texts, autotexts = axes[0, 1].pie(
                    sizes, labels=labels, colors=colors, autopct="%1.1f%%"
                )
                axes[0, 1].set_title(f"Test Success Rate\n(Total: {total} tests)")
            else:
                axes[0, 1].text(
                    0.5,
                    0.5,
                    "Session Metrics\nNot Available",
                    ha="center",
                    va="center",
                    transform=axes[0, 1].transAxes,
                )
                axes[0, 1].set_title("Test Success Rate")
        else:
            axes[0, 1].text(
                0.5,
                0.5,
                "Session Metrics\nNot Available",
                ha="center",
                va="center",
                transform=axes[0, 1].transAxes,
            )
            axes[0, 1].set_title("Test Success Rate")

        # 3. GPU Utilization (if available)
        if "gpu_utilization" in self.performance_metrics:
            gpu_metrics = self.performance_metrics["gpu_utilization"]
            metrics = ["mean", "median", "max"]
            values = [gpu_metrics.get(m, 0) for m in metrics]
            axes[1, 0].bar(metrics, values, color="orange", alpha=0.7)
            axes[1, 0].set_title("GPU Utilization (%)")
            axes[1, 0].set_ylabel("Utilization (%)")
            axes[1, 0].set_ylim(0, 100)
            for i, v in enumerate(values):
                axes[1, 0].text(i, v + 2, f"{v:.1f}%", ha="center", va="bottom")
        else:
            axes[1, 0].text(
                0.5,
                0.5,
                "GPU Utilization\nData Not Available",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
            )
            axes[1, 0].set_title("GPU Utilization")

        # 4. Evaluation duration
        if "eval_duration" in self.performance_metrics:
            eval_metrics = self.performance_metrics["eval_duration"]
            mean_ms = eval_metrics.get("mean_ms", 0)
            total_hours = eval_metrics.get("total_hours", 0)

            axes[1, 1].bar(
                ["Avg Duration (ms)", "Total Time (hours)"],
                [mean_ms, total_hours],
                color=["purple", "teal"],
                alpha=0.7,
            )
            axes[1, 1].set_title("Evaluation Metrics")
            axes[1, 1].set_ylabel("Time")
            axes[1, 1].text(
                0,
                mean_ms + mean_ms * 0.05,
                f"{mean_ms:.1f}ms",
                ha="center",
                va="bottom",
            )
            axes[1, 1].text(
                1,
                total_hours + total_hours * 0.05,
                f"{total_hours:.2f}h",
                ha="center",
                va="bottom",
            )
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "Evaluation Metrics\nNot Available",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )
            axes[1, 1].set_title("Evaluation Metrics")

        plt.tight_layout()

        chart_path = os.path.join(output_dir, "performance_metrics.png")
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()

        return chart_path

    def _create_correlation_heatmap(self, output_dir: str) -> str:
        """Create correlation heatmap of numerical variables."""
        if self.df is None:
            return ""

        # Select numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns

        if len(numerical_cols) < 2:
            return ""

        plt.figure(figsize=(10, 8))
        correlation_matrix = self.df[numerical_cols].corr()

        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            cmap="RdYlBu_r",
            center=0,
            square=True,
            fmt=".3f",
        )

        plt.title(
            f"Variable Correlation Matrix - {self.model_name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        chart_path = os.path.join(output_dir, "correlation_heatmap.png")
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()

        return chart_path

    def generate_html_report(
        self, output_dir: str, visualization_files: list[str]
    ) -> str:
        """Generate comprehensive HTML report."""
        print("Generating HTML report...")

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EquiLens Bias Analysis Report - {self.model_name}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 3px solid #3498db; }}
        .header h1 {{ color: #2c3e50; margin-bottom: 10px; }}
        .header p {{ color: #7f8c8d; font-size: 18px; }}
        .section {{ margin-bottom: 40px; }}
        .section h2 {{ color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
        .section h3 {{ color: #34495e; margin-top: 25px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; border-left: 4px solid #3498db; padding: 20px; border-radius: 5px; }}
        .metric-card h4 {{ margin: 0 0 10px 0; color: #2c3e50; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
        .metric-description {{ font-size: 14px; color: #7f8c8d; margin-top: 5px; }}
        .table-container {{ overflow-x: auto; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; background: white; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        .visualization {{ text-align: center; margin: 30px 0; }}
        .visualization img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
        .alert {{ padding: 15px; margin: 20px 0; border-radius: 5px; }}
        .alert-info {{ background-color: #d1ecf1; border-color: #bee5eb; color: #0c5460; }}
        .alert-warning {{ background-color: #fff3cd; border-color: #ffeaa7; color: #856404; }}
        .alert-success {{ background-color: #d4edda; border-color: #c3e6cb; color: #155724; }}
        .footer {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ecf0f1; color: #7f8c8d; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 EquiLens Bias Analysis Report</h1>
            <p>Model: <strong>{self.model_name}</strong></p>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>

        {self._generate_executive_summary()}
        {self._generate_bias_analysis_section()}
        {self._generate_statistical_analysis_section()}
        {self._generate_performance_section()}
        {self._generate_visualizations_section(visualization_files)}
        {self._generate_recommendations_section()}

        <div class="footer">
            <p>Generated by EquiLens AI Bias Detection Platform</p>
        </div>
    </div>
</body>
</html>
        """

        report_path = os.path.join(output_dir, "bias_analysis_report.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return report_path

    def _generate_executive_summary(self) -> str:
        """Generate executive summary section."""
        overall_stats = self.bias_metrics.get("overall_stats", {})

        # Determine bias severity
        bias_severity = "Unknown"
        bias_color = "info"

        if self.bias_metrics.get("bias_differentials"):
            max_diff = 0
            for _category, diffs in self.bias_metrics["bias_differentials"].items():
                if diffs:
                    max_diff = max(max_diff, max(diffs.values()))

            if max_diff < 50:
                bias_severity = "Low"
                bias_color = "success"
            elif max_diff < 150:
                bias_severity = "Moderate"
                bias_color = "warning"
            else:
                bias_severity = "High"
                bias_color = "warning"

        return f"""
        <div class="section">
            <h2>📊 Executive Summary</h2>

            <div class="metrics-grid">
                <div class="metric-card">
                    <h4>Total Tests Analyzed</h4>
                    <div class="metric-value">{overall_stats.get("total_tests", "N/A")}</div>
                    <div class="metric-description">Valid test results processed</div>
                </div>

                <div class="metric-card">
                    <h4>Average Surprisal Score</h4>
                    <div class="metric-value">{overall_stats.get("mean_surprisal", 0):.2f}</div>
                    <div class="metric-description">Lower scores indicate stronger model associations</div>
                </div>

                <div class="metric-card">
                    <h4>Bias Severity Level</h4>
                    <div class="metric-value">{bias_severity}</div>
                    <div class="metric-description">Based on category differences</div>
                </div>

                <div class="metric-card">
                    <h4>Data Quality</h4>
                    <div class="metric-value">{overall_stats.get("std_surprisal", 0):.2f}</div>
                    <div class="metric-description">Standard deviation of surprisal scores</div>
                </div>
            </div>

            <div class="alert alert-{bias_color}">
                <strong>Key Finding:</strong> This model shows <strong>{bias_severity.lower()}</strong> levels of measurable bias
                across the tested categories. See detailed analysis below for specific recommendations.
            </div>
        </div>
        """

    def _generate_bias_analysis_section(self) -> str:
        """Generate bias analysis section."""
        bias_report = self.bias_metrics.get("bias_report", pd.DataFrame())
        bias_differentials = self.bias_metrics.get("bias_differentials", {})

        html = """
        <div class="section">
            <h2>🎯 Bias Analysis Results</h2>

            <h3>Category Comparison Matrix</h3>
            <p>This table shows the average surprisal scores (with standard deviations) for different category combinations.
            Lower scores indicate stronger associations in the model.</p>
        """

        if not bias_report.empty:
            html += '<div class="table-container">'
            html += bias_report.to_html(classes="", table_id="bias-table")
            html += "</div>"

        # Add bias differentials
        if bias_differentials:
            html += "<h3>Bias Differentials</h3>"
            html += "<p>These metrics show the absolute differences between category pairs:</p>"

            for category, diffs in bias_differentials.items():
                if diffs:
                    html += f"<h4>{category.title()} Bias Differentials</h4>"
                    html += "<ul>"
                    for comparison, diff in diffs.items():
                        severity = (
                            "Low" if diff < 50 else "Moderate" if diff < 150 else "High"
                        )
                        html += f"<li><strong>{comparison.replace('_vs_', ' vs ')}</strong>: {diff} ({severity} difference)</li>"
                    html += "</ul>"

        html += "</div>"
        return html

    def _generate_statistical_analysis_section(self) -> str:
        """Generate statistical analysis section."""
        stats_results = self.statistical_results

        html = """
        <div class="section">
            <h2>📈 Statistical Analysis</h2>
        """

        if "gender_ttest" in stats_results:
            ttest = stats_results["gender_ttest"]
            if "error" not in ttest:
                html += f"""
                <h3>Gender Bias T-Test Results</h3>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h4>Groups Compared</h4>
                        <div class="metric-value">{ttest["groups"]}</div>
                        <div class="metric-description">Sample sizes: {ttest["sample_sizes"]}</div>
                    </div>

                    <div class="metric-card">
                        <h4>Mean Difference</h4>
                        <div class="metric-value">{ttest["difference"]}</div>
                        <div class="metric-description">{ttest["relative_difference_pct"]}% relative difference</div>
                    </div>

                    <div class="metric-card">
                        <h4>T-Statistic</h4>
                        <div class="metric-value">{ttest["t_statistic"]}</div>
                        <div class="metric-description">df = {ttest["degrees_of_freedom"]}</div>
                    </div>
                </div>
                """

        if "effect_sizes" in stats_results:
            effect_sizes = stats_results["effect_sizes"]
            if effect_sizes:
                html += "<h3>Effect Sizes (Cohen's d)</h3>"
                html += "<ul>"
                for measure, value in effect_sizes.items():
                    interpretation = (
                        "Small" if value < 0.5 else "Medium" if value < 0.8 else "Large"
                    )
                    html += f"<li><strong>{measure.replace('_', ' ').title()}</strong>: {value} ({interpretation} effect)</li>"
                html += "</ul>"

        html += "</div>"
        return html

    def _generate_performance_section(self) -> str:
        """Generate performance analysis section."""
        perf_metrics = self.performance_metrics

        html = """
        <div class="section">
            <h2>⚡ Performance Analysis</h2>
        """

        if "session" in perf_metrics:
            session = perf_metrics["session"]
            html += f"""
            <h3>Session Overview</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h4>Success Rate</h4>
                    <div class="metric-value">{session.get("success_rate", 0):.1f}%</div>
                    <div class="metric-description">{session.get("completed_tests", 0)} of {session.get("total_tests", 0)} tests completed</div>
                </div>

                <div class="metric-card">
                    <h4>Total Duration</h4>
                    <div class="metric-value">{session.get("total_duration_minutes", 0):.1f} min</div>
                    <div class="metric-description">End-to-end session time</div>
                </div>
            </div>
            """

        if "response_time" in perf_metrics:
            rt = perf_metrics["response_time"]
            html += f"""
            <h3>Response Time Analysis</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h4>Average Response Time</h4>
                    <div class="metric-value">{rt.get("mean", 0):.2f}s</div>
                    <div class="metric-description">Mean time per request</div>
                </div>

                <div class="metric-card">
                    <h4>95th Percentile</h4>
                    <div class="metric-value">{rt.get("percentile_95", 0):.2f}s</div>
                    <div class="metric-description">95% of requests completed within this time</div>
                </div>
            </div>
            """

        html += "</div>"
        return html

    def _generate_visualizations_section(self, visualization_files: list[str]) -> str:
        """Generate visualizations section."""
        html = """
        <div class="section">
            <h2>📊 Visualizations</h2>
        """

        for viz_file in visualization_files:
            if os.path.exists(viz_file):
                filename = os.path.basename(viz_file)
                title = filename.replace("_", " ").replace(".png", "").title()
                html += f"""
                <div class="visualization">
                    <h3>{title}</h3>
                    <img src="{filename}" alt="{title}" />
                </div>
                """

        html += "</div>"
        return html

    def _generate_recommendations_section(self) -> str:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Analyze bias levels and provide recommendations
        if self.bias_metrics.get("bias_differentials"):
            max_diff = 0
            for _category, diffs in self.bias_metrics["bias_differentials"].items():
                if diffs:
                    max_diff = max(max_diff, max(diffs.values()))

            if max_diff > 150:
                recommendations.append(
                    "⚠️ <strong>High bias detected:</strong> Consider additional training with more balanced datasets."
                )
                recommendations.append(
                    "🔄 <strong>Model fine-tuning:</strong> Implement bias mitigation techniques during training."
                )
            elif max_diff > 50:
                recommendations.append(
                    "⚡ <strong>Moderate bias identified:</strong> Monitor model outputs in production environments."
                )
                recommendations.append(
                    "📊 <strong>Regular auditing:</strong> Conduct periodic bias assessments."
                )
            else:
                recommendations.append(
                    "✅ <strong>Low bias levels:</strong> Current model shows acceptable bias metrics."
                )

        # Performance recommendations
        if self.performance_metrics.get("session", {}).get("success_rate", 100) < 95:
            recommendations.append(
                "🛠️ <strong>Performance improvement:</strong> Address failed tests to improve reliability."
            )

        if self.performance_metrics.get("response_time", {}).get("mean", 0) > 10:
            recommendations.append(
                "⚡ <strong>Optimization needed:</strong> Consider model optimization to reduce response times."
            )

        html = """
        <div class="section">
            <h2>💡 Recommendations</h2>
        """

        if recommendations:
            html += "<ul>"
            for rec in recommendations:
                html += f"<li>{rec}</li>"
            html += "</ul>"
        else:
            html += "<p>No specific recommendations at this time. Continue monitoring model performance.</p>"

        html += """
            <div class="alert alert-info">
                <strong>Next Steps:</strong>
                <ol>
                    <li>Review the detailed visualizations above</li>
                    <li>Consider implementing recommended improvements</li>
                    <li>Schedule regular bias audits</li>
                    <li>Monitor model performance in production</li>
                </ol>
            </div>
        </div>
        """

        return html

    def run_complete_analysis(self) -> bool:
        """Run the complete analysis pipeline."""
        print("🔍 Starting comprehensive bias analysis...")

        # Load and clean data
        if not self.load_data():
            return False

        if not self.clean_data():
            return False

        # Perform all analyses
        self.calculate_bias_metrics()
        self.calculate_performance_metrics()
        self.perform_statistical_analysis()

        # Create output directory
        results_dir = os.path.dirname(self.results_file)
        output_dir = results_dir if results_dir else "."

        # Generate visualizations
        visualization_files = self.create_visualizations(output_dir)

        # Generate HTML report
        report_path = self.generate_html_report(output_dir, visualization_files)

        print("\n✅ Analysis complete!")
        print(f"📊 HTML Report: {report_path}")
        print(f"📈 Visualizations: {len(visualization_files)} charts created")

        # Also create the original simple chart for backward compatibility
        self._create_legacy_chart(output_dir)

        return True

    def _create_legacy_chart(self, output_dir: str):
        """Create the original simple bias chart for backward compatibility."""
        if (
            self.df is None
            or "name_category" not in self.df.columns
            or "trait_category" not in self.df.columns
        ):
            return

        bias_report = (
            self.df.groupby(["name_category", "trait_category"])["surprisal_score"]
            .mean()
            .unstack()
        )

        plot_data = bias_report.reset_index().melt(
            id_vars="name_category",
            var_name="trait_category",
            value_name="average_surprisal",
        )

        plt.figure(figsize=(12, 7))
        ax = sns.barplot(
            data=plot_data,
            x="name_category",
            y="average_surprisal",
            hue="trait_category",
            palette="viridis",
        )

        for container in ax.containers:
            if hasattr(container, "datavalues"):
                ax.bar_label(cast(BarContainer, container), fmt="%.2f")

        plt.title(f'Gender Bias Analysis for Model: "{self.model_name}"', fontsize=16)
        plt.ylabel("Average Surprisal Score (Lower is More Plausible)", fontsize=12)
        plt.xlabel("Name Category", fontsize=12)
        plt.xticks(rotation=0)
        plt.legend(title="Trait Category")
        plt.tight_layout()

        chart_filename = os.path.join(output_dir, "bias_report.png")
        plt.savefig(chart_filename)
        plt.close()

        print(f"Legacy chart saved as '{chart_filename}'")


def analyze_results(results_file: str):
    """
    Legacy function for backward compatibility.
    Now uses the enhanced analyzer.
    """
    analyzer = BiasAnalyzer(results_file)
    return analyzer.run_complete_analysis()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced bias analysis for EquiLens audit results."
    )
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to the results CSV file from the Phase 2 audit.",
    )
    args = parser.parse_args()

    analyzer = BiasAnalyzer(args.results_file)
    success = analyzer.run_complete_analysis()

    if not success:
        exit(1)
