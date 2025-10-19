#!/usr/bin/env python3
"""
EquiLens Bias Analytics Module

Comprehensive statistical analysis and visualization suite for bias detection.
Generates professional HTML and Markdown reports using AI-powered content generation.

Features:
- Statistical analysis (t-tests, effect sizes, confidence intervals)
- Rich visualizations (violin plots, heatmaps, regression analysis)
- HTML report generation with Jinja2 templates
- AI-powered report content generation via Ollama models
- Professional presentation-ready outputs
"""

import base64
import json
import os
import time
import warnings
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from jinja2 import Template
from scipy import stats

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set professional plotting style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
sns.set_context("notebook", font_scale=1.2)


class BiasAnalytics:
    """
    Comprehensive bias analysis with advanced statistics, visualizations, and AI-powered reporting.
    """

    def __init__(
        self,
        results_file: str,
        ollama_url: str | None = None,
        report_model: str | None = None,
        ai_num_predict: int = 512,
    ):
        """
        Initialize analytics engine with results file.

        Args:
            results_file: Path to CSV results file
            ollama_url: Ollama API endpoint for AI report generation (defaults to environment variable or host.docker.internal)
            report_model: Model to use for AI-generated reports (optional)
            ai_num_predict: Number of tokens to predict for AI-generated content (default: 512)
        """
        self.results_file = results_file
        self.results_dir = Path(os.path.dirname(results_file) or ".")
        self.ollama_url = ollama_url or os.getenv(
            "OLLAMA_BASE_URL", "http://host.docker.internal:11434"
        )
        self.report_model = report_model
        self.ai_num_predict = ai_num_predict

        self.df: pd.DataFrame = pd.DataFrame()
        self.model_name = self._extract_model_name()
        self.stats_results: dict[str, Any] = {}
        self.viz_files: list[str] = []
        self.viz_data: dict[str, str] = {}  # Store base64 encoded images

        # Validate Ollama URL reachability
        if self.ollama_url:
            try:
                resp = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
                if resp.status_code != 200:
                    print(
                        f"⚠️  Warning: Ollama service at {self.ollama_url} is not responding as expected (status {resp.status_code})."
                    )
            except Exception as e:
                print(
                    f"⚠️  Warning: Could not reach Ollama service at {self.ollama_url}: {e}"
                )

    def _extract_model_name(self) -> str:
        """Extract model name from filename."""
        basename = os.path.basename(self.results_file)
        name = basename.replace("results_", "").replace(".csv", "")
        name = name.replace("_responses", "")
        # Remove timestamp patterns
        import re
        name = re.sub(r'_\d{8}_\d{6}$', '', name)
        return name

    # ========================
    # DATA LOADING & VALIDATION
    # ========================

    def load_and_validate_data(self) -> bool:
        """
        Load and validate results data.

        Returns:
            bool: True if data loaded successfully
        """
        print(f"📂 Loading results from: {os.path.basename(self.results_file)}")

        try:
            self.df = pd.read_csv(self.results_file)
        except FileNotFoundError:
            print(f"❌ File not found: {self.results_file}")
            return False
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return False

        # Clean column names and values
        self.df.columns = self.df.columns.str.strip()
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].str.strip()

        # Validate required columns
        required_cols = ["surprisal_score", "name_category", "trait_category"]
        missing_cols = [col for col in required_cols if col not in self.df.columns]

        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            # Try alternate file
            if self.results_file.endswith("_responses.csv"):
                base_file = self.results_file.replace("_responses.csv", ".csv")
                if os.path.exists(base_file):
                    print(f"ℹ️  Switching to: {os.path.basename(base_file)}")
                    self.results_file = base_file
                    return self.load_and_validate_data()
            return False

        # Clean data
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=["surprisal_score"])
        self.df["surprisal_score"] = pd.to_numeric(
            self.df["surprisal_score"], errors="coerce"
        )
        self.df = self.df.dropna(subset=["surprisal_score"])

        clean_pct = len(self.df) / initial_count * 100 if initial_count > 0 else 0
        print(f"✅ Loaded {len(self.df)} valid results ({clean_pct:.1f}% clean)")
        print(f"📊 Model: {self.model_name}")
        print(f"📁 Output directory: {self.results_dir}")

        # Auto-detect corpus structure
        self._detect_corpus_structure()

        return True

    def _detect_corpus_structure(self):
        """
        Auto-detect the structure of the corpus for flexible analysis.

        Detects:
        - Comparison type (gender_bias, nationality_bias, etc.)
        - Available categories in name_category
        - Available groupings (profession, trait_category, etc.)
        """
        # Detect comparison type
        if "comparison_type" in self.df.columns:
            self.comparison_type = self.df["comparison_type"].iloc[0]
        else:
            self.comparison_type = "bias_detection"

        # Detect name categories (e.g., Male/Female, Western/Eastern, etc.)
        self.name_categories = sorted(self.df["name_category"].unique())

        # Detect trait categories
        self.trait_categories = sorted(self.df["trait_category"].unique())

        # Detect if profession column exists
        self.has_profession = "profession" in self.df.columns

        # Detect grouping fields (for flexible analysis)
        self.grouping_fields = []
        potential_fields = ["profession", "trait", "trait_category", "template_id"]
        for field in potential_fields:
            if field in self.df.columns and self.df[field].nunique() > 1:
                self.grouping_fields.append(field)

        print("\n📋 Detected corpus structure:")
        print(f"   • Comparison: {self.comparison_type}")
        print(
            f"   • Categories ({len(self.name_categories)}): {', '.join(self.name_categories)}"
        )
        print(f"   • Trait types: {', '.join(self.trait_categories)}")
        print(f"   • Grouping fields: {', '.join(self.grouping_fields)}")

        # Store category labels for flexible reporting (support N categories)
        self.category_label_1 = (
            self.name_categories[0] if len(self.name_categories) > 0 else "Category 1"
        )
        self.category_label_2 = (
            self.name_categories[1] if len(self.name_categories) > 1 else "Category 2"
        )

        # For N-way comparisons (3+ categories)
        self.is_multi_category = len(self.name_categories) > 2

    # ========================
    # STATISTICAL ANALYSIS
    # ========================

    def calculate_effect_sizes(self) -> dict[str, Any]:
        """Calculate Cohen's d effect sizes for category comparison."""
        print("\n🔬 Calculating effect sizes (Cohen's d)...")

        effect_sizes = {}

        if "profession" in self.df.columns and len(self.name_categories) >= 2:
            for profession in self.df["profession"].unique():
                prof_data = self.df[self.df["profession"] == profession]

                cat1_data = prof_data[
                    prof_data["name_category"] == self.category_label_1
                ]["surprisal_score"]
                cat2_data = prof_data[
                    prof_data["name_category"] == self.category_label_2
                ]["surprisal_score"]

                if len(cat1_data) > 0 and len(cat2_data) > 0:
                    cohens_d = self._cohens_d(cat1_data, cat2_data)
                    effect_sizes[profession] = {
                        "cohens_d": cohens_d,
                        "interpretation": self._interpret_cohens_d(cohens_d),
                        f"{self.category_label_1.lower()}_mean": float(
                            cat1_data.mean()
                        ),
                        f"{self.category_label_2.lower()}_mean": float(
                            cat2_data.mean()
                        ),
                        f"{self.category_label_1.lower()}_std": float(cat1_data.std()),
                        f"{self.category_label_2.lower()}_std": float(cat2_data.std()),
                        f"{self.category_label_1.lower()}_n": int(len(cat1_data)),
                        f"{self.category_label_2.lower()}_n": int(len(cat2_data)),
                    }

        self.stats_results["effect_sizes"] = effect_sizes
        return effect_sizes

    def _cohens_d(self, group1: pd.Series, group2: pd.Series) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)

        # Explicitly convert to float to avoid type checker issues
        var1 = float(group1.var(ddof=1))  # type: ignore[arg-type]
        var2 = float(group2.var(ddof=1))  # type: ignore[arg-type]

        # Pooled standard deviation
        pooled_std = float(np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)))

        if pooled_std == 0:
            return 0.0

        mean1 = float(group1.mean())  # type: ignore[arg-type]
        mean2 = float(group2.mean())  # type: ignore[arg-type]
        return float((mean1 - mean2) / pooled_std)

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "Negligible"
        elif abs_d < 0.5:
            return "Small"
        elif abs_d < 0.8:
            return "Medium"
        else:
            return "Large"

    def perform_statistical_tests(self) -> dict[str, Any]:
        """Perform t-tests (2 categories) or ANOVA (3+ categories)."""
        print("\n📈 Performing statistical tests...")

        test_results = {}

        # For multi-category comparisons (3+), use ANOVA
        if self.is_multi_category:
            print(
                f"   ℹ️  Detected {len(self.name_categories)} categories, using ANOVA..."
            )

            # Prepare data for ANOVA
            groups = [
                self.df[self.df["name_category"] == cat]["surprisal_score"].values
                for cat in self.name_categories
            ]

            # Filter out empty groups
            groups = [g for g in groups if len(g) > 0]

            if len(groups) >= 2:
                f_stat, p_value = stats.f_oneway(*groups)
                test_results["overall_comparison"] = {
                    "test_type": "ANOVA",
                    "f_statistic": float(f_stat),
                    "p_value": float(p_value),
                    "significant": bool(p_value < 0.05),
                    "num_categories": len(self.name_categories),
                    "categories": self.name_categories,
                }

                # Add mean for each category
                for cat in self.name_categories:
                    cat_scores = self.df[self.df["name_category"] == cat][
                        "surprisal_score"
                    ]
                    if len(cat_scores) > 0:
                        test_results["overall_comparison"][f"{cat.lower()}_mean"] = (
                            float(cat_scores.mean())
                        )

        # For 2-category comparisons, use t-test
        elif len(self.name_categories) >= 2:
            cat1_scores = self.df[self.df["name_category"] == self.category_label_1][
                "surprisal_score"
            ]
            cat2_scores = self.df[self.df["name_category"] == self.category_label_2][
                "surprisal_score"
            ]

            if len(cat1_scores) > 0 and len(cat2_scores) > 0:
                t_stat, p_value = stats.ttest_ind(cat1_scores, cat2_scores)
                test_results["overall_comparison"] = {
                    "test_type": "t-test",
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": bool(p_value < 0.05),
                    f"{self.category_label_1.lower()}_mean": float(cat1_scores.mean()),
                    f"{self.category_label_2.lower()}_mean": float(cat2_scores.mean()),
                    "difference": float(cat2_scores.mean() - cat1_scores.mean()),
                }

        # Profession-specific tests (if profession field exists)
        if "profession" in self.df.columns and len(self.name_categories) >= 2:
            profession_tests = {}
            for profession in self.df["profession"].unique():
                prof_data = self.df[self.df["profession"] == profession]

                if self.is_multi_category:
                    # ANOVA for multi-category
                    groups = [
                        prof_data[prof_data["name_category"] == cat][
                            "surprisal_score"
                        ].values
                        for cat in self.name_categories
                    ]
                    groups = [g for g in groups if len(g) > 1]

                    if len(groups) >= 2:
                        f_stat, p_value = stats.f_oneway(*groups)
                        profession_tests[profession] = {
                            "test_type": "ANOVA",
                            "f_statistic": float(f_stat),
                            "p_value": float(p_value),
                            "significant": bool(p_value < 0.05),
                        }
                else:
                    # T-test for 2 categories
                    cat1_prof = prof_data[
                        prof_data["name_category"] == self.category_label_1
                    ]["surprisal_score"]
                    cat2_prof = prof_data[
                        prof_data["name_category"] == self.category_label_2
                    ]["surprisal_score"]

                    if len(cat1_prof) > 1 and len(cat2_prof) > 1:
                        t_stat, p_value = stats.ttest_ind(cat1_prof, cat2_prof)
                        profession_tests[profession] = {
                            "test_type": "t-test",
                            "t_statistic": float(t_stat),
                            "p_value": float(p_value),
                            "significant": bool(p_value < 0.05),
                            f"{self.category_label_1.lower()}_mean": float(
                                cat1_prof.mean()
                            ),
                            f"{self.category_label_2.lower()}_mean": float(
                                cat2_prof.mean()
                            ),
                            "difference": float(cat2_prof.mean() - cat1_prof.mean()),
                        }

            test_results["by_profession"] = profession_tests

        self.stats_results["statistical_tests"] = test_results
        return test_results

    def calculate_confidence_intervals(
        self, confidence: float = 0.95
    ) -> dict[str, Any]:
        """Calculate confidence intervals for mean surprisal scores."""
        print("\n📊 Calculating confidence intervals...")

        ci_results = {}
        z_score = stats.norm.ppf((1 + confidence) / 2)

        for category in self.df["name_category"].unique():
            cat_data = self.df[self.df["name_category"] == category]["surprisal_score"]
            mean = float(cat_data.mean())
            std = float(cat_data.std())
            n = len(cat_data)
            margin = float(z_score * (std / np.sqrt(n)))

            ci_results[category] = {
                "mean": mean,
                "std": std,
                "n": n,
                "ci_lower": mean - margin,
                "ci_upper": mean + margin,
                "margin": margin,
            }

        self.stats_results["confidence_intervals"] = ci_results
        return ci_results

    # ========================
    # VISUALIZATIONS
    # ========================

    def create_violin_plot(self) -> str:
        """Create violin plot comparing distributions."""
        print("\n🎻 Creating violin plot...")

        fig, ax = plt.subplots(figsize=(12, 7))

        sns.violinplot(
            data=self.df,
            x="name_category",
            y="surprisal_score",
            hue="trait_category",
            split=False,
            inner="box",
            palette="Set2",
            ax=ax
        )

        ax.set_title(
            f"Distribution of Surprisal Scores - {self.model_name}",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Gender Category", fontsize=12)
        ax.set_ylabel("Surprisal Score (ns/token)", fontsize=12)
        ax.legend(title="Trait Category", loc="upper right")
        plt.tight_layout()

        return self._save_plot("violin_plot.png")

    def create_heatmap_matrix(self) -> str:
        """Create correlation heatmap."""
        print("\n🔥 Creating correlation heatmap...")

        # Pivot data for heatmap
        pivot_data = self.df.pivot_table(
            values="surprisal_score",
            index="name_category",
            columns="trait_category",
            aggfunc="mean"
        )

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn_r",
            center=pivot_data.values.mean(),
            square=True,
            linewidths=1,
            cbar_kws={"label": "Mean Surprisal Score"},
            ax=ax
        )

        ax.set_title(
            f"Bias Heatmap - {self.model_name}",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.tight_layout()

        return self._save_plot("heatmap_matrix.png")

    def create_effect_size_chart(self) -> str:
        """Create effect size visualization."""
        if not self.stats_results.get("effect_sizes"):
            return ""

        print("\n📊 Creating effect size chart...")

        effect_sizes = self.stats_results["effect_sizes"]
        professions = list(effect_sizes.keys())
        cohens_d = [effect_sizes[p]["cohens_d"] for p in professions]

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = ["#e74c3c" if abs(d) > 0.8 else "#f39c12" if abs(d) > 0.5 else "#27ae60"
                  for d in cohens_d]

        ax.barh(professions, cohens_d, color=colors, edgecolor="black")

        # Add reference lines
        ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
        ax.axvline(-0.2, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.axvline(0.2, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.axvline(-0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.axvline(0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

        ax.set_xlabel("Cohen's d", fontsize=12)
        ax.set_title(
            f"Effect Sizes by Profession - {self.model_name}",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.text(
            0.95,
            0.95,
            "Green: Small | Orange: Medium | Red: Large",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        plt.tight_layout()
        return self._save_plot("effect_sizes.png")

    def create_box_plot_profession(self) -> str:
        """Create box plot comparing professions."""
        print("\n📦 Creating profession box plot...")

        fig, ax = plt.subplots(figsize=(14, 8))

        # Prepare data for box plot
        professions = self.df["name_category"].unique()
        data_by_profession = [
            self.df[self.df["name_category"] == prof]["surprisal_score"].values
            for prof in professions
        ]

        bp = ax.boxplot(
            data_by_profession,
            labels=professions,  # type: ignore
            patch_artist=True,
            notch=True,
            showmeans=True,
        )

        # Color boxes
        colors = plt.cm.get_cmap("Set3")(range(len(professions)))
        for patch, color in zip(bp["boxes"], colors, strict=False):  # type: ignore
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel("Surprisal Score (ns/token)", fontsize=12)
        ax.set_xlabel("Profession", fontsize=12)
        ax.set_title(
            f"Distribution of Surprisal Scores by Profession - {self.model_name}",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        return self._save_plot("box_plot_profession.png")

    def create_scatter_correlations(self) -> str:
        """Create scatter plot showing correlations."""
        print("\n🔸 Creating correlation scatter plot...")

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create scatter plot by profession
        professions = self.df["name_category"].unique()
        colors = plt.cm.get_cmap("Set2")(range(len(professions)))

        for prof, color in zip(professions, colors, strict=False):
            prof_data = self.df[self.df["name_category"] == prof]
            ax.scatter(
                range(len(prof_data)),
                prof_data["surprisal_score"],
                label=prof,
                alpha=0.6,
                s=100,
                c=[color],
                edgecolors="black",
                linewidth=0.5,
            )

        ax.set_xlabel("Sample Index", fontsize=12)
        ax.set_ylabel("Surprisal Score (ns/token)", fontsize=12)
        ax.set_title(
            f"Surprisal Score Distribution by Profession - {self.model_name}",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        return self._save_plot("scatter_correlations.png")

    def create_time_series_progression(self) -> str:
        """Create time series-style progression chart."""
        print("\n📈 Creating progression chart...")

        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot running average
        window = min(5, len(self.df) // 10 + 1)
        rolling_mean = (
            self.df["surprisal_score"].rolling(window=window, center=True).mean()
        )

        ax.plot(
            range(len(self.df)),
            self.df["surprisal_score"],
            "o-",
            alpha=0.3,
            color="steelblue",
            label="Individual Scores",
        )
        ax.plot(
            range(len(self.df)),
            rolling_mean,
            "-",
            color="darkblue",
            linewidth=2,
            label=f"Rolling Average (window={window})",
        )

        # Add mean line
        mean_val = self.df["surprisal_score"].mean()
        ax.axhline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Overall Mean ({mean_val:.2f})",
        )

        ax.set_xlabel("Test Index", fontsize=12)
        ax.set_ylabel("Surprisal Score (ns/token)", fontsize=12)
        ax.set_title(
            f"Surprisal Score Progression - {self.model_name}",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        return self._save_plot("time_series_progression.png")

    def create_comprehensive_dashboard(self) -> str:
        """Create comprehensive multi-panel dashboard (flexible for N categories)."""
        print("\n🎨 Creating comprehensive dashboard...")

        try:
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

            # Panel 1: Overall distribution (violin plot) - supports N categories
            ax1 = fig.add_subplot(gs[0, :2])

            if len(self.name_categories) >= 2:
                # Prepare data for all categories
                category_data = [
                    self.df[self.df["name_category"] == cat]["surprisal_score"]
                    for cat in self.name_categories
                ]

                positions = list(range(1, len(self.name_categories) + 1))

                parts = ax1.violinplot(
                    category_data,
                    positions=positions,
                    showmeans=True,
                    showmedians=True,
                )

                # Color the violin plots
                colors = plt.cm.get_cmap("tab10")(range(len(self.name_categories)))
                for pc, color in zip(parts["bodies"], colors, strict=False):  # type: ignore
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)

                ax1.set_xticks(positions)
                ax1.set_xticklabels(
                    [f"{cat}" for cat in self.name_categories],
                    rotation=45 if len(self.name_categories) > 3 else 0,
                )
                ax1.set_ylabel("Surprisal Score (ns/token)")
                ax1.set_title(
                    f"Overall {self.comparison_type.replace('_', ' ').title()} Distribution",
                    fontweight="bold",
                )
                ax1.grid(axis="y", alpha=0.3)

            # Panel 2: Summary statistics - supports N categories
            ax2 = fig.add_subplot(gs[0, 2])
            ax2.axis("off")

            stats_text = f"""
Summary Statistics

Total Tests: {len(self.df)}
Mean: {self.df["surprisal_score"].mean():.2f}
Std Dev: {self.df["surprisal_score"].std():.2f}
Min: {self.df["surprisal_score"].min():.2f}
Max: {self.df["surprisal_score"].max():.2f}

Category Means:
"""
            # Add all category means
            for cat in self.name_categories:
                cat_data = self.df[self.df["name_category"] == cat]["surprisal_score"]
                if len(cat_data) > 0:
                    stats_text += f"  {cat}: {cat_data.mean():.2f}\n"

            ax2.text(
                0.1,
                0.5,
                stats_text,
                transform=ax2.transAxes,
                fontsize=11 if len(self.name_categories) <= 4 else 9,
                verticalalignment="center",
                family="monospace",
            )

            # Panel 3: Box plot by category - flexible
            ax3 = fig.add_subplot(gs[1, :])
            categories = self.df["name_category"].unique()
            data_by_category = [
                self.df[self.df["name_category"] == cat]["surprisal_score"].values
                for cat in categories
            ]

            bp = ax3.boxplot(data_by_category, labels=categories, patch_artist=True)  # type: ignore
            colors = plt.cm.get_cmap("tab10")(range(len(categories)))
            for patch, color in zip(bp["boxes"], colors, strict=False):  # type: ignore
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax3.set_ylabel("Surprisal Score (ns/token)")
            ax3.set_title("Distribution by Category", fontweight="bold")
            ax3.grid(axis="y", alpha=0.3)
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")

            # Panel 4: Effect sizes - flexible
            ax4 = fig.add_subplot(gs[2, :2])
            if self.stats_results.get("effect_sizes"):
                effect_sizes = self.stats_results["effect_sizes"]
                profs = list(effect_sizes.keys())
                cohens_d = [effect_sizes[p]["cohens_d"] for p in profs]

                colors_list = [
                    "#e74c3c"
                    if abs(d) > 0.8
                    else "#f39c12"
                    if abs(d) > 0.5
                    else "#27ae60"
                    for d in cohens_d
                ]

                ax4.barh(
                    profs, cohens_d, color=colors_list, edgecolor="black", alpha=0.8
                )
                ax4.axvline(0, color="black", linewidth=1)
                ax4.axvline(
                    -0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5
                )
                ax4.axvline(0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
                ax4.set_xlabel("Cohen's d")
                ax4.set_title(
                    f"Effect Sizes ({self.comparison_type.replace('_', ' ').title()})",
                    fontweight="bold",
                )
                ax4.grid(axis="x", alpha=0.3)

            # Panel 5: Statistical significance - flexible
            ax5 = fig.add_subplot(gs[2, 2])
            ax5.axis("off")
            if self.stats_results.get("statistical_tests", {}).get(
                "overall_comparison"
            ):
                test = self.stats_results["statistical_tests"]["overall_comparison"]
                sig_text = f"""
Statistical Tests

t-statistic: {test["t_statistic"]:.4f}
p-value: {test["p_value"]:.6f}

Significant: {"✅ Yes" if test["significant"] else "❌ No"}
(α = 0.05)

Interpretation:
{"Differences are" if test["significant"] else "No significant"}
{"statistically significant" if test["significant"] else "differences found"}
                """
                ax5.text(
                    0.1,
                    0.5,
                    sig_text,
                    transform=ax5.transAxes,
                    fontsize=10,
                    verticalalignment="center",
                    family="monospace",
                )

            fig.suptitle(
                f"EquiLens Comprehensive Bias Analysis Dashboard - {self.model_name}",
                fontsize=18,
                fontweight="bold",
                y=0.995,
            )

            return self._save_plot("comprehensive_dashboard.png")

        except Exception as e:
            print(f"  ⚠️  Dashboard creation failed: {type(e).__name__}: {e}")
            print("  ℹ️  Other visualizations were created successfully")
            return ""

    def _save_plot(self, filename: str) -> str:
        """Save plot to file and store base64 encoding."""
        filepath = self.results_dir / filename

        # Save to file
        plt.savefig(filepath, dpi=300, bbox_inches="tight")

        # Store base64 encoding for HTML embedding
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        self.viz_data[filename] = image_base64
        buffer.close()

        plt.close()

        self.viz_files.append(str(filepath))
        print(f"  ✅ Saved: {filename}")
        return str(filepath)

    # ========================
    # AI-POWERED REPORT GENERATION
    # ========================

    def _get_available_models(self, max_retries: int = 2) -> list[str]:
        """
        Get list of available Ollama models with retry logic.

        Args:
            max_retries: Maximum number of retry attempts (default: 2)

        Returns:
            List of available model names
        """
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    return [model["name"] for model in models]
                elif response.status_code == 404:
                    print(f"⚠️  Ollama API endpoint not found at {self.ollama_url}")
                    return []

            except requests.exceptions.ConnectionError:
                if attempt < max_retries - 1:
                    print("   🔌 Connection failed, retrying...")
                    time.sleep(1)
                else:
                    print(f"⚠️  Cannot connect to Ollama at {self.ollama_url}")

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print("   ⏱️  Timeout, retrying...")
                    time.sleep(1)
                else:
                    print("⚠️  Ollama request timed out")

            except Exception as e:
                print(f"⚠️  Could not fetch models: {type(e).__name__}: {e}")
                break

        return []

    def _generate_ai_content(
        self,
        prompt: str,
        model: str | None = None,
        max_retries: int = 3,
        timeout: int = 60,
    ) -> str:
        """
        Generate content using AI model with retry logic and comprehensive error handling.

        Args:
            prompt: The prompt to send to the AI model
            model: Optional specific model to use
            max_retries: Maximum number of retry attempts (default: 3)
            timeout: Request timeout in seconds (default: 60)

        Returns:
            Generated content string or error message
        """
        target_model = model or self.report_model

        if not target_model:
            # Try to find a suitable model
            available = self._get_available_models()
            # Prefer smaller, faster models for report generation
            preferred = ["llama3.2:latest", "llama3.1:latest", "llama2:latest", "mistral:latest"]
            for pref in preferred:
                if pref in available:
                    target_model = pref
                    break
            if not target_model and available:
                target_model = available[0]

        if not target_model:
            return "⚠️ AI content generation unavailable - no models found. Please install an Ollama model."

        # Retry logic with exponential backoff
        last_error = None
        for attempt in range(max_retries):
            try:
                # Calculate exponential backoff delay
                if attempt > 0:
                    # compute a delay using base_delay and max_delay if available
                    base = getattr(self, "base_delay", 0.5)
                    maxd = getattr(self, "max_delay", 30.0)
                    delay = min(base * (2**attempt), maxd)
                    time.sleep(delay)

                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": target_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": 512,  # Optimized for analytics - balance between detail and timeout prevention
                            "top_p": 0.9,
                        },
                    },
                    timeout=timeout,
                )

                if response.status_code == 200:
                    result = response.json().get("response", "")
                    if result and len(result.strip()) > 0:
                        return result
                    else:
                        last_error = "Empty response from AI model"
                        continue

                elif response.status_code == 404:
                    return f"⚠️ Model '{target_model}' not found. Please pull the model: ollama pull {target_model}"

                elif response.status_code == 500:
                    last_error = (
                        "Ollama internal server error (model may be overloaded)"
                    )
                    if attempt < max_retries - 1:
                        print("   ⚠️  Ollama returned 500 error, retrying...")
                        # Increase timeout significantly for next attempt
                        timeout = min(timeout * 2, 180)
                        continue
                    else:
                        # Give up after max retries
                        print(
                            "   ⚠️  Ollama consistently returning 500 errors, skipping AI generation"
                        )
                        return "⚠️ AI generation failed - Ollama server errors"

                elif response.status_code == 503:
                    last_error = "Ollama service unavailable"
                    if attempt < max_retries - 1:
                        print("   ⚠️  Ollama service busy, retrying...")
                        continue

                else:
                    last_error = f"HTTP {response.status_code}: {response.text[:100]}"

            except requests.exceptions.Timeout:
                last_error = f"Request timed out after {timeout}s"
                if attempt < max_retries - 1:
                    print(f"   ⏱️  Timeout after {timeout}s, retrying...")
                    # Don't increase timeout - if it's taking this long, something is wrong
                    # timeout = int(min(timeout * 1.5, 180))  # Don't increase
                else:
                    print("   ⚠️  Consistent timeouts, skipping AI generation")
                    return "⚠️ AI generation timed out - model may be too large or busy"

            except requests.exceptions.ConnectionError:
                last_error = "Cannot connect to Ollama service"
                if attempt < max_retries - 1:
                    print("   🔌 Connection failed, retrying...")
                else:
                    return f"⚠️ Cannot connect to Ollama at {self.ollama_url}. Is Ollama running?"

            except requests.exceptions.RequestException as e:
                last_error = f"Request error: {str(e)}"

            except json.JSONDecodeError:
                last_error = "Invalid JSON response from Ollama"

            except Exception as e:
                last_error = f"Unexpected error: {type(e).__name__}: {str(e)}"

        # All retries exhausted
        error_msg = f"⚠️ AI generation failed after {max_retries} attempts. Last error: {last_error}"
        print(f"   {error_msg}")
        return f"AI content generation failed. {last_error}"

    def generate_ai_insights(self) -> dict[str, str]:
        """
        Generate AI-powered insights for the report with comprehensive error handling.

        Returns:
            Dictionary with AI-generated insights or fallback messages
        """
        print("\n🤖 Generating AI-powered insights...")

        insights = {}

        # Default fallback messages
        default_summary = f"""**Bias Analysis Summary**

Model tested: {self.model_name}
Total tests conducted: {len(self.df)}
Mean surprisal score: {self.df["surprisal_score"].mean():.2f} ns/token

Statistical analysis completed. Review visualizations and statistical tests for detailed findings."""

        default_recommendations = """<strong>General Recommendations:</strong>

- Review the effect sizes (Cohen's d) to identify professions with significant gender bias <br>
- Investigate categories where |Cohen's d| > 0.5 (medium effect) or > 0.8 (large effect) <br>
- Compare male vs female surprisal scores across different professional contexts <br>
- Consider retraining or fine-tuning the model if consistent bias patterns are detected <br>
- Document findings and share with stakeholders for transparency"""

        # Prepare data summary for AI
        data_summary = {
            "model_name": self.model_name,
            "total_tests": len(self.df),
            "mean_surprisal": float(self.df["surprisal_score"].mean()),
            "stats": self.stats_results,
        }

        # Check if Ollama is available
        try:
            test_response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if test_response.status_code != 200:
                print("⚠️  Ollama service not available. Using default insights.")
                return {
                    "executive_summary": default_summary
                    + "\n\n*Note: AI insights unavailable - Ollama service not responding.*",
                    "recommendations": default_recommendations,
                }
        except Exception as e:
            print(f"⚠️  Cannot connect to Ollama: {e}")
            print("   📝 Using default insights instead.")
            return {
                "executive_summary": default_summary
                + f"\n\n*Note: AI insights unavailable - {str(e)}*",
                "recommendations": default_recommendations,
            }

        # Generate executive summary with timeout protection
        print("   📝 Generating executive summary...")
        try:
            # Simplified prompt to reduce generation time
            summary_prompt = f"""Write a brief executive summary for this bias audit:
Model: {data_summary["model_name"]}
Tests: {data_summary["total_tests"]}
Mean surprisal: {data_summary["mean_surprisal"]:.2f} ns/token

Summarize the bias assessment in 2-3 sentences."""

            # Reduced retries and timeout - fail fast if Ollama is having issues
            summary = self._generate_ai_content(
                summary_prompt, max_retries=1, timeout=30
            )

            # Check if summary generation failed
            if (
                summary.startswith("⚠️")
                or "failed" in summary.lower()
                or "timed out" in summary.lower()
            ):
                print("   ⚠️  AI summary generation failed, using default")
                insights["executive_summary"] = (
                    default_summary + "\n\n*Note: AI generation unavailable*"
                )
            else:
                insights["executive_summary"] = summary
        except Exception as e:
            print(f"   ⚠️  Summary generation error: {e}")
            insights["executive_summary"] = default_summary

        # Generate recommendations with timeout protection
        print("   💡 Generating recommendations...")
        try:
            # Simplified prompt to reduce generation time
            rec_prompt = f"""List 3-4 recommendations to reduce bias in this model:
Model: {data_summary["model_name"]}
Tests: {data_summary["total_tests"]}

Format as bullet points."""

            # Reduced retries and timeout - fail fast if Ollama is having issues
            recommendations = self._generate_ai_content(
                rec_prompt, max_retries=1, timeout=30
            )
        except Exception as e:
            print(f"   ⚠️  Recommendations generation error: {e}")
            recommendations = default_recommendations

        # Check if recommendations generation failed
        if (
            recommendations.startswith("⚠️")
            or "failed" in recommendations.lower()
            or "timed out" in recommendations.lower()
        ):
            print("   ⚠️  AI recommendations generation failed, using default")
            insights["recommendations"] = (
                default_recommendations + "\n\n*Note: AI generation unavailable*"
            )
        else:
            insights["recommendations"] = recommendations

        print("   ✅ Insights generation completed (with fallback if needed)")
        return insights

    # ========================
    # HTML REPORT GENERATION
    # ========================

    def generate_html_report(self, use_ai: bool = True) -> str:
        """Generate comprehensive HTML report."""
        print("\n📝 Generating HTML report...")

        # Generate AI insights if requested
        ai_insights = {}
        if use_ai:
            ai_insights = self.generate_ai_insights()

        # Prepare template data
        template_data = {
            "model_name": self.model_name,
            "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_tests": len(self.df),
            "mean_surprisal": float(self.df["surprisal_score"].mean()),
            "std_surprisal": float(self.df["surprisal_score"].std()),
            "stats": self.stats_results,
            "visualizations": self.viz_data,
            "ai_insights": ai_insights,
        }

        # HTML template using Jinja2
        template_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EquiLens Bias Analysis Report - {{ model_name }}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        .header {
            text-align: center;
            padding: 30px 0;
            border-bottom: 3px solid #667eea;
            margin-bottom: 40px;
        }
        .header h1 {
            font-size: 2.5em;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        .header .subtitle {
            font-size: 1.2em;
            color: #7f8c8d;
            margin: 10px 0;
        }
        .section {
            margin: 40px 0;
            padding: 30px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        .section h2 {
            color: #2c3e50;
            font-size: 1.8em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }
        .section h3 {
            color: #34495e;
            font-size: 1.4em;
            margin: 20px 0 10px 0;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 25px 0;
        }
        .metric-card {
            background: white;
            border-left: 5px solid #667eea;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }
        .metric-card h4 {
            color: #7f8c8d;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }
        .metric-description {
            font-size: 0.9em;
            color: #95a5a6;
        }
        .visualization {
            text-align: center;
            margin: 30px 0;
            padding: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .visualization img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.1);
        }
        .visualization h3 {
            margin-bottom: 20px;
            color: #2c3e50;
        }
        .stats-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .stats-table th {
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }
        .stats-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #ecf0f1;
        }
        .stats-table tr:hover {
            background: #f8f9fa;
        }
        .alert {
            padding: 20px;
            margin: 25px 0;
            border-radius: 8px;
            border-left: 5px solid;
        }
        .alert-info {
            background: #d1ecf1;
            border-color: #0c5460;
            color: #0c5460;
        }
        .alert-warning {
            background: #fff3cd;
            border-color: #856404;
            color: #856404;
        }
        .alert-success {
            background: #d4edda;
            border-color: #155724;
            color: #155724;
        }
        .ai-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin: 25px 0;
        }
        .ai-content h3 {
            color: white;
            margin-bottom: 15px;
        }
        .footer {
            text-align: center;
            margin-top: 50px;
            padding-top: 30px;
            border-top: 2px solid #ecf0f1;
            color: #7f8c8d;
        }
        .badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin: 0 5px;
        }
        .badge-success { background: #27ae60; color: white; }
        .badge-warning { background: #f39c12; color: white; }
        .badge-danger { background: #e74c3c; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 EquiLens Bias Analysis Report</h1>
            <p class="subtitle">Model: <strong>{{ model_name }}</strong></p>
            <p class="subtitle">Generated: {{ generated_date }}</p>
        </div>

        <!-- Executive Summary -->
        {% if ai_insights.executive_summary %}
        <div class="section ai-content">
            <h2>🤖 Executive Summary (AI-Generated)</h2>
            <div>{{ ai_insights.executive_summary }}</div>
        </div>
        {% endif %}

        <!-- Key Metrics -->
        <div class="section">
            <h2>📊 Key Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h4>Total Tests</h4>
                    <div class="metric-value">{{ total_tests }}</div>
                    <div class="metric-description">Valid test results analyzed</div>
                </div>
                <div class="metric-card">
                    <h4>Mean Surprisal</h4>
                    <div class="metric-value">{{ "%.2f"|format(mean_surprisal) }}</div>
                    <div class="metric-description">ns/token (lower = stronger bias)</div>
                </div>
                <div class="metric-card">
                    <h4>Std Deviation</h4>
                    <div class="metric-value">{{ "%.2f"|format(std_surprisal) }}</div>
                    <div class="metric-description">Measure of variance</div>
                </div>
                <div class="metric-card">
                    <h4>Bias Level</h4>
                    <div class="metric-value">
                        {% if stats.statistical_tests and stats.statistical_tests.overall_gender %}
                            {% if stats.statistical_tests.overall_gender.significant %}
                                <span class="badge badge-warning">Detected</span>
                            {% else %}
                                <span class="badge badge-success">Low</span>
                            {% endif %}
                        {% else %}
                            <span class="badge badge-info">Unknown</span>
                        {% endif %}
                    </div>
                    <div class="metric-description">Statistical significance</div>
                </div>
            </div>
        </div>

        <!-- Statistical Analysis -->
        {% if stats.statistical_tests %}
        <div class="section">
            <h2>📈 Statistical Analysis</h2>

            {% if stats.statistical_tests.overall_gender %}
            <h3>Overall Gender Comparison</h3>
            {% set test = stats.statistical_tests.overall_gender %}
            <table class="stats-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Male Mean</td>
                    <td>{{ "%.2f"|format(test.male_mean) }} ns/token</td>
                </tr>
                <tr>
                    <td>Female Mean</td>
                    <td>{{ "%.2f"|format(test.female_mean) }} ns/token</td>
                </tr>
                <tr>
                    <td>Difference</td>
                    <td>{{ "%.2f"|format(test.difference) }} ns/token</td>
                </tr>
                <tr>
                    <td>t-statistic</td>
                    <td>{{ "%.4f"|format(test.t_statistic) }}</td>
                </tr>
                <tr>
                    <td>p-value</td>
                    <td>{{ "%.6f"|format(test.p_value) }}</td>
                </tr>
                <tr>
                    <td>Significant (α=0.05)</td>
                    <td>
                        {% if test.significant %}
                            <span class="badge badge-warning">Yes</span>
                        {% else %}
                            <span class="badge badge-success">No</span>
                        {% endif %}
                    </td>
                </tr>
            </table>
            {% endif %}
        </div>
        {% endif %}

        <!-- Effect Sizes -->
        {% if stats.effect_sizes %}
        <div class="section">
            <h2>🎯 Effect Sizes (Cohen's d)</h2>
            <table class="stats-table">
                <thead>
                    <tr>
                        <th>Profession</th>
                        <th>Cohen's d</th>
                        <th>Interpretation</th>
                        <th>Male Mean</th>
                        <th>Female Mean</th>
                    </tr>
                </thead>
                <tbody>
                {% for profession, data in stats.effect_sizes.items() %}
                    <tr>
                        <td>{{ profession }}</td>
                        <td>{{ "%.3f"|format(data.cohens_d) }}</td>
                        <td>
                            {% if data.interpretation == "Large" %}
                                <span class="badge badge-danger">{{ data.interpretation }}</span>
                            {% elif data.interpretation == "Medium" %}
                                <span class="badge badge-warning">{{ data.interpretation }}</span>
                            {% else %}
                                <span class="badge badge-success">{{ data.interpretation }}</span>
                            {% endif %}
                        </td>
                        <td>{{ "%.1f"|format(data.male_mean) }}</td>
                        <td>{{ "%.1f"|format(data.female_mean) }}</td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}

        <!-- Visualizations -->
        <div class="section">
            <h2>📊 Visualizations</h2>

            {% for filename, image_data in visualizations.items() %}
            <div class="visualization">
                <h3>{{ filename.replace('_', ' ').replace('.png', '').title() }}</h3>
                <img src="data:image/png;base64,{{ image_data }}" alt="{{ filename }}">
            </div>
            {% endfor %}
        </div>

        <!-- AI Recommendations -->
        {% if ai_insights.recommendations %}
        <div class="section ai-content">
            <h2>💡 Recommendations (AI-Generated)</h2>
            <div>{{ ai_insights.recommendations }}</div>
        </div>
        {% endif %}

        <div class="footer">
            <p><strong>EquiLens</strong> - AI Bias Detection Platform</p>
            <p>Generated with ❤️ by the EquiLens Team</p>
        </div>
    </div>
</body>
</html>
        """

        # Render template
        template = Template(template_html)
        html_content = template.render(**template_data)

        # Save report
        report_path = self.results_dir / "bias_analysis_report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"  ✅ Saved: {report_path.name}")
        return str(report_path)

    def generate_markdown_report(self, use_ai: bool = True) -> str:
        """Generate markdown report with embedded images (optionally AI-enhanced)."""
        print("\n📝 Generating Markdown report...")

        lines = [
            "# EquiLens Bias Analysis Report",
            "",
            f"**Model**: {self.model_name}  ",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Total Tests**: {len(self.df)}  ",
            f"**Comparison Type**: {self.comparison_type.replace('_', ' ').title()}  ",
            f"**Categories**: {', '.join(self.name_categories)}  ",
            "",
            "---",
            "",
        ]

        # AI-generated executive summary
        if use_ai:
            ai_insights = self.generate_ai_insights()
            if ai_insights.get("executive_summary"):
                lines.extend(
                    [
                        "## 📋 Executive Summary (AI-Generated)",
                        "",
                        ai_insights["executive_summary"],
                        "",
                    ]
                )

        # Statistics by Category
        lines.extend(
            [
                "## 📊 Overall Statistics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Mean Surprisal | {self.df['surprisal_score'].mean():.2f} ns/token |",
                f"| Std Deviation | {self.df['surprisal_score'].std():.2f} ns/token |",
                f"| Min Surprisal | {self.df['surprisal_score'].min():.2f} ns/token |",
                f"| Max Surprisal | {self.df['surprisal_score'].max():.2f} ns/token |",
                "",
            ]
        )

        # Category-specific statistics
        lines.extend(
            [
                "### Statistics by Category",
                "",
                "| Category | Mean | Std Dev | Count |",
                "|----------|------|---------|-------|",
            ]
        )

        for cat in self.name_categories:
            cat_data = self.df[self.df["name_category"] == cat]["surprisal_score"]
            if len(cat_data) > 0:
                lines.append(
                    f"| {cat} | {cat_data.mean():.2f} | {cat_data.std():.2f} | {len(cat_data)} |"
                )
        lines.append("")

        # Statistical tests
        if self.stats_results.get("statistical_tests", {}).get("overall_comparison"):
            test = self.stats_results["statistical_tests"]["overall_comparison"]
            test_type = test.get("test_type", "t-test")

            lines.extend(
                [
                    "## 📈 Statistical Significance",
                    "",
                    f"**Test Type**: {test_type}  ",
                    "",
                    "| Metric | Value |",
                    "|--------|-------|",
                ]
            )

            if test_type == "ANOVA":
                lines.extend(
                    [
                        f"| F-statistic | {test.get('f_statistic', 0):.4f} |",
                        f"| p-value | {test.get('p_value', 1):.6f} |",
                        f"| Significant (α=0.05) | {'✅ Yes' if test.get('significant', False) else '❌ No'} |",
                        f"| Number of Categories | {test.get('num_categories', len(self.name_categories))} |",
                    ]
                )
            else:
                lines.extend(
                    [
                        f"| t-statistic | {test.get('t_statistic', 0):.4f} |",
                        f"| p-value | {test.get('p_value', 1):.6f} |",
                        f"| Significant (α=0.05) | {'✅ Yes' if test.get('significant', False) else '❌ No'} |",
                    ]
                )

            lines.append("")

        # Effect sizes
        if self.stats_results.get("effect_sizes"):
            lines.extend(
                [
                    "## 📊 Effect Sizes (Cohen's d)",
                    "",
                    "| Profession | Cohen's d | Interpretation |",
                    "|------------|-----------|----------------|",
                ]
            )
            for prof, data in self.stats_results["effect_sizes"].items():
                lines.append(
                    f"| {prof} | {data['cohens_d']:.3f} | {data['interpretation']} |"
                )
            lines.append("")

        # Embedded Visualizations
        lines.extend(
            [
                "## 📊 Visualizations",
                "",
            ]
        )

        # List of chart files to embed
        chart_files = [
            ("violin_plot.png", "Distribution Violin Plot"),
            ("heatmap_matrix.png", "Correlation Heatmap"),
            ("effect_sizes.png", "Effect Sizes by Profession"),
            ("box_plot_profession.png", "Box Plot by Profession"),
            ("scatter_correlations.png", "Scatter Plot Correlations"),
            ("time_series_progression.png", "Time Series Progression"),
            ("comprehensive_dashboard.png", "Comprehensive Dashboard"),
        ]

        for filename, title in chart_files:
            chart_path = self.results_dir / filename
            if chart_path.exists():
                lines.extend(
                    [
                        f"### {title}",
                        "",
                        f"![{title}]({filename})",
                        "",
                    ]
                )

        # AI recommendations
        if use_ai:
            if ai_insights.get("recommendations"):
                lines.extend(
                    [
                        "## 💡 Recommendations (AI-Generated)",
                        "",
                        ai_insights["recommendations"],
                        "",
                    ]
                )

        # Footer
        lines.extend(
            [
                "---",
                "",
                "*Report generated by EquiLens - AI Bias Detection Platform*  ",
                "*For more information, visit: [EquiLens Documentation](https://github.com/Life-Experimentalist/EquiLens)*",
                "",
            ]
        )

        # Save report
        report_path = self.results_dir / "bias_analysis_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"  ✅ Saved: {report_path.name}")
        return str(report_path)

    # ========================
    # MAIN EXECUTION
    # ========================

    def run_complete_analysis(
        self, generate_html: bool = True, generate_ai_insights: bool = True
    ) -> bool:
        """
        Run complete analysis pipeline.

        Args:
            generate_html: Whether to generate HTML report
            generate_ai_insights: Whether to use AI for insights in reports

        Returns:
            bool: True if analysis completed successfully
        """
        print("\n" + "=" * 70)
        print("🔬 EQUILENS BIAS ANALYSIS")
        print("=" * 70)

        # Load data
        if not self.load_and_validate_data():
            return False

        # Perform statistical analysis
        self.perform_statistical_tests()
        self.calculate_effect_sizes()
        self.calculate_confidence_intervals()

        # Generate visualizations
        print("\n" + "=" * 70)
        print("📊 GENERATING VISUALIZATIONS")
        print("=" * 70)

        # Core visualizations (always generate)
        self.create_violin_plot()
        self.create_heatmap_matrix()
        self.create_effect_size_chart()

        # Additional visualizations (for comprehensive analysis)
        if generate_html:  # Generate all charts for HTML report
            self.create_box_plot_profession()
            self.create_scatter_correlations()
            self.create_time_series_progression()
            self.create_comprehensive_dashboard()

        # Generate reports
        print("\n" + "=" * 70)
        print("📝 GENERATING REPORTS")
        print("=" * 70)

        html_report = None
        md_report = None

        # Generate HTML report with error handling
        if generate_html:
            try:
                html_report = self.generate_html_report(use_ai=generate_ai_insights)
            except Exception as e:
                print(f"  ⚠️  HTML report generation failed: {type(e).__name__}: {e}")
                print("  ℹ️  Continuing with markdown report...")

        # Generate Markdown report with error handling
        try:
            md_report = self.generate_markdown_report(use_ai=generate_ai_insights)
        except Exception as e:
            print(f"  ⚠️  Markdown report generation failed: {type(e).__name__}: {e}")

        # Summary
        print("\n" + "=" * 70)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"\n📁 Output directory: {self.results_dir}")
        print("\n📊 Generated Reports:")
        if html_report:
            print(f"  • HTML: {os.path.basename(html_report)}")
        if md_report:
            print(f"  • Markdown: {os.path.basename(md_report)}")
        print(f"\n📈 Visualizations: {len(self.viz_files)} charts created")

        return True


# ========================
# CLI INTERFACE
# ========================


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="EquiLens Bias Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analytics.py results.csv
  python analytics.py results/model_results.csv --no-ai
  python analytics.py results.csv --model llama3.2:latest
        """,
    )

    parser.add_argument("results_file", help="Path to CSV results file")
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Disable AI-powered report generation"
    )
    parser.add_argument(
        "--model",
        help="Ollama model to use for AI report generation"
    )
    args = parser.parse_args()

    if not os.path.exists(args.results_file):
        print(f"❌ Error: File not found: {args.results_file}")
        return 1

    # Run analysis
    analyzer = BiasAnalytics(
        results_file=args.results_file,
        ollama_url=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
        report_model=args.model,
        ai_num_predict=512,  # You can make this configurable via CLI if desired
    )

    success = analyzer.run_complete_analysis(
        generate_html=True,
        generate_ai_insights=not args.no_ai
    )
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
