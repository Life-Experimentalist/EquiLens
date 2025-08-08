import argparse
import os
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.container import BarContainer


def analyze_results(results_file: str):
    """
    Analyzes the audit results to generate a bias report and chart.
    """
    print(f"Loading results from '{results_file}'...")
    try:
        df = pd.read_csv(results_file)
    except FileNotFoundError:
        print(f"Error: Results file not found at '{results_file}'")
        print("Please run the Phase 2 audit script first.")
        return

    # --- Data Cleaning ---
    # Drop rows where surprisal score could not be calculated (e.g., incomplete runs)
    df.dropna(subset=["surprisal_score"], inplace=True)
    # Ensure the surprisal column is numeric, coercing errors to NaN and then dropping them
    df["surprisal_score"] = pd.to_numeric(df["surprisal_score"], errors="coerce")
    df.dropna(subset=["surprisal_score"], inplace=True)

    if df.empty:
        print("No valid results found in the file. Cannot perform analysis.")
        return

    print(f"Analyzing {len(df)} valid results...")

    # --- Bias Calculation ---
    # Group by the categories of interest and calculate the mean surprisal score.
    bias_report = (
        df.groupby(["name_category", "trait_category"])["surprisal_score"]
        .mean()
        .unstack()
    )

    print("\n--- Bias Report (Average Surprisal Score) ---")
    print(
        "Lower scores indicate a stronger association (more 'plausible' to the model)."
    )
    print(bias_report)
    print("-" * 55)

    # --- Visualization ---
    # Reshape the data for easier plotting with seaborn
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

    # Add labels to the bars
    for container in ax.containers:
        if hasattr(container, "datavalues"):  # Check if it's a BarContainer
            ax.bar_label(cast(BarContainer, container), fmt="%.2f")

    model_name = (
        os.path.basename(results_file).replace("results_", "").replace(".csv", "")
    )
    plt.title(f'Gender Bias Analysis for Model: "{model_name}"', fontsize=16)
    plt.ylabel("Average Surprisal Score (Lower is More Plausible)", fontsize=12)
    plt.xlabel("Name Category", fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title="Trait Category")
    plt.tight_layout()

    # Save the chart - use the same directory structure as the results file
    results_dir = os.path.dirname(results_file)
    if os.path.basename(results_dir) == "results":
        # If results file is directly in results/, create model subdirectory
        chart_filename = f"results/{model_name}/bias_report.png"
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(chart_filename), exist_ok=True)
    else:
        # If results file is in a session directory, save chart there too
        chart_filename = os.path.join(results_dir, "bias_report.png")

    plt.savefig(chart_filename)

    print(f"\nChart saved as '{chart_filename}'")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze bias audit results from an EquiLens run."
    )
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to the results CSV file from the Phase 2 audit.",
    )
    args = parser.parse_args()

    analyze_results(args.results_file)
