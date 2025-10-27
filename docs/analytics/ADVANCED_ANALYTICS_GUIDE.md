# EquiLens Advanced Analytics Guide

## Overview

EquiLens now features **comprehensive advanced analytics** with professional-grade statistical analysis and visualizations perfect for research presentations and faculty reviews.

---

## Quick Start

### Standard Analysis (Quick)
```bash
uv run equilens analyze
# or
uv run equilens analyze --results path/to/results.csv
```

**Output**: Basic bias_report.png + console statistics

### Advanced Analysis (Comprehensive)
```bash
uv run equilens analyze --advanced
# or
uv run equilens analyze --advanced --results path/to/results.csv
```

**Output**: 8+ charts + statistical report + comprehensive dashboard

---

## What's Included in Advanced Analytics

### 📊 8 Professional Visualizations

1. **Comprehensive Dashboard** (`comprehensive_dashboard.png`)
   - Multi-panel overview with all key metrics
   - Gender comparison bar charts
   - Distribution density plots
   - Profession and trait breakdowns
   - Statistical summary panel
   - **Perfect for presentations!**

2. **Violin Plot Distribution** (`violin_plot_distribution.png`)
   - Shows full distribution shape of surprisal scores
   - Side-by-side gender comparison
   - Split violin plots by trait category
   - Reveals outliers and data spread

3. **Box Plot with Statistics** (`box_plot_profession.png`)
   - Box plots by profession and gender
   - Shows median, quartiles, outliers
   - Mean markers (diamond shapes)
   - Easy comparison across professions

4. **Heatmap Bias Matrix** (`heatmap_bias_matrix.png`)
   - Color-coded profession-gender matrix
   - Shows mean surprisal scores
   - Highlights bias differentials (Female - Male)
   - Quickly identify problematic professions

5. **Scatter Correlations** (`scatter_correlations.png`)
   - Token count vs evaluation duration
   - Response time vs surprisal score
   - Reveals performance patterns
   - Gender-coded scatter points

6. **Effect Sizes (Cohen's d)** (`effect_sizes_cohens_d.png`)
   - Professional horizontal bar chart
   - Color-coded by effect size magnitude
   - Reference lines (Small: 0.2, Medium: 0.5, Large: 0.8)
   - Shows direction and strength of bias

7. **Time-Series Progression** (`time_series_progression.png`)
   - Surprisal scores over test sequence
   - 5-test moving average smoothing
   - Detects performance degradation
   - Shows temporal patterns in bias

8. **Statistical Report** (`statistical_report.md`)
   - Comprehensive Markdown report
   - T-test results with p-values
   - Effect sizes for each profession
   - Confidence intervals (95%)
   - Key findings summary
   - Ready for publication!

---

## Statistical Analysis Features

### ✅ Statistical Tests Performed

1. **Independent T-Tests**
   - Overall gender comparison
   - Profession-specific comparisons
   - Trait-category comparisons
   - Reports t-statistic and p-value
   - Significance threshold: α = 0.05

2. **Effect Size Calculations (Cohen's d)**
   - Quantifies magnitude of bias
   - Interpretations:
     * |d| < 0.2: Negligible
     * 0.2 ≤ |d| < 0.5: Small
     * 0.5 ≤ |d| < 0.8: Medium
     * |d| ≥ 0.8: Large
   - Sign indicates direction (+/- bias)

3. **Confidence Intervals**
   - 95% confidence intervals for all means
   - Overall, male-specific, female-specific
   - Shows margin of error
   - Helps assess reliability

4. **Summary Statistics**
   - Mean, median, standard deviation
   - Min/max values
   - Sample sizes
   - Data quality metrics

---

## Example Output (Llama2 Results)

### Key Findings from Statistical Report

```markdown
## Statistical Summary
- Total Tests: 40
- Mean Surprisal: 293,832,927 ns/token
- Gender Difference: 11,227,031 ns/token (Female higher)
- Statistical Significance: ❌ No (p = 0.43)

## Effect Sizes by Profession
| Profession | Cohen's d | Interpretation |
|------------|-----------|----------------|
| Manager    | -1.423    | Large          |
| Social Worker | -1.335 | Large          |
| Engineer   | -0.916    | Large          |
| Counselor  | -0.829    | Large          |
| Teacher    | +0.824    | Large (reversed) |
| Therapist  | -0.805    | Large          |
```

### Interpretation

- **Negative d**: Higher surprisal for Female names (bias against women)
- **Positive d**: Higher surprisal for Male names (bias against men)
- **Large effects** found in: Manager, Social Worker, Engineer, Counselor, Therapist
- **Teacher** showed reversed bias (Male nurses more surprising to AI)

---

## Use Cases

### 📚 For Academic Presentations
- Use `comprehensive_dashboard.png` as main slide
- Show `effect_sizes_cohens_d.png` for statistical rigor
- Reference `statistical_report.md` for methodology

### 🎓 For Faculty Reviews
- Provide full analysis folder with all 8 outputs
- Highlight statistical significance testing
- Demonstrate research-grade analysis

### 📊 For Research Papers
- Export visualizations at 300 DPI (publication quality)
- Use statistical report for methods section
- Cite p-values and effect sizes from report

### 🏢 For Industry Reports
- Professional dashboard for stakeholders
- Heatmaps for quick identification of issues
- Time-series for monitoring over versions

---

## Advanced Features

### Data Cleaning
- Automatically removes invalid entries
- Strips whitespace from column names
- Handles missing values gracefully
- Reports data quality percentage

### Auto-Detection
- Switches from `_responses.csv` to results file
- Finds correct output directory
- Lists all generated files with sizes

### Statistical Robustness
- Paired comparisons within professions
- Multiple testing considerations
- Confidence interval reporting
- Effect size quantification

---

## Command Options

### Full CLI Syntax

```bash
uv run equilens analyze [OPTIONS]

Options:
  -r, --results PATH     Path to results CSV file (auto-detect if omitted)
  -a, --advanced         Use comprehensive analytics (8+ charts)
  -s, --silent           Suppress output (avoid Unicode errors)
  --help                 Show help message
```

### Examples

```bash
# Auto-detect latest results, advanced mode
uv run equilens analyze --advanced

# Specific file, standard mode
uv run equilens analyze --results results/llama2_results.csv

# Advanced + silent (for automation)
uv run equilens analyze --advanced --silent

# Interactive selection
uv run equilens analyze
```

---

## File Outputs Location

All files generated in the **same directory as results file**:

```
results/
└── llama2_latest_20251016_011024/
    ├── results_llama2_latest_20251016_011024.csv      # Original results
    ├── comprehensive_dashboard.png                     # Main overview
    ├── violin_plot_distribution.png                   # Distributions
    ├── box_plot_profession.png                        # By profession
    ├── heatmap_bias_matrix.png                        # Bias matrix
    ├── scatter_correlations.png                       # Correlations
    ├── effect_sizes_cohens_d.png                      # Effect sizes
    ├── time_series_progression.png                    # Time trends
    └── statistical_report.md                          # Full report
```

---

## Comparing Standard vs Advanced

| Feature | Standard | Advanced |
|---------|----------|----------|
| **Charts** | 1 (basic bar) | 7 (professional) |
| **Statistics** | Mean, SD | T-tests, Cohen's d, CI |
| **Report** | Console only | Markdown + Console |
| **Time** | ~5 seconds | ~10 seconds |
| **File Size** | ~50 KB | ~2.5 MB total |
| **Best For** | Quick checks | Presentations, research |

---

## Tips for Best Results

### 1. Use Sufficient Data
- Minimum 20 tests recommended
- 40+ tests for robust statistics
- Balanced male/female samples

### 2. Check Data Quality
- Review "clean %" in output
- 100% is ideal, >90% acceptable
- Investigate if <80%

### 3. Interpret Effect Sizes Carefully
- Small p-value ≠ large effect
- Cohen's d shows practical significance
- Context matters (profession norms)

### 4. Share Complete Analysis
- Include statistical_report.md
- Provide comprehensive_dashboard.png
- Explain methodology from report

---

## Troubleshooting

### "Missing required columns" Error
**Solution**: Use base results file, not `_responses.csv`
```bash
# ❌ Wrong
uv run equilens analyze --results results_responses.csv

# ✅ Correct
uv run equilens analyze --results results.csv
```

### Unicode/Emoji Display Issues
**Solution**: Use `--silent` flag
```bash
uv run equilens analyze --advanced --silent
```

### "No module named scipy" Error
**Solution**: Install dependencies
```bash
uv pip install scipy numpy pandas matplotlib seaborn
```

### Charts Look Cluttered
**Solution**: Use fewer professions or larger figure size
- Edit `advanced_analytics.py` figure sizes
- Or filter results CSV to specific professions

---

## Future Enhancements (Planned)

- [ ] Interactive HTML dashboard with Plotly
- [ ] ANOVA for multi-group comparisons
- [ ] Correlation matrix heatmaps
- [ ] Automated bias severity classification
- [ ] Export to PDF report
- [ ] Comparative analysis (multi-model)

---

## Technical Details

### Dependencies
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **matplotlib**: Static visualizations
- **seaborn**: Statistical plotting
- **scipy**: Statistical tests

### Performance
- **Speed**: ~10 seconds for 40 tests
- **Memory**: ~100 MB peak
- **Disk**: ~2.5 MB output files

### File Format
- **Input**: CSV with surprisal_score, name_category, trait_category
- **Output**: PNG (300 DPI) + Markdown

---

## Citation

If using EquiLens advanced analytics in research:

```bibtex
@software{equilens_analytics_2025,
  title = {EquiLens Advanced Analytics},
  author = {Life-Experimentalists},
  year = {2025},
  url = {https://github.com/Life-Experimentalists/EquiLens},
  version = {2.0}
}
```

---

## Support

- **Documentation**: See `docs/` folder
- **Issues**: GitHub issue tracker
- **Examples**: Check `results/` folder samples

---

**Ready to analyze your bias audit results with research-grade statistics!** 🎉

Run: `uv run equilens analyze --advanced`
