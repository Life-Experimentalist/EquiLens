"""
EquiLens Phase 3: Advanced Bias Analysis Module

A comprehensive statistical analysis suite for AI bias detection with:

**Core Features:**
- Statistical significance testing (t-tests, ANOVA, effect sizes, confidence intervals)
- Rich visualizations (violin plots, heatmaps, correlation matrices, dashboards)
- AI-powered report generation using Ollama integration
- Multiple export formats (HTML, Markdown, JSON)
- Progressive analysis with checkpointing
- Performance optimization and intelligent caching

**Component Overview:**

1. **BiasAnalytics** (analytics.py)
   - Main analysis engine with statistical methods
   - Automated visualization generation
   - HTML and Markdown report generation
   - Ollama-based AI insights
   - Professional presentation-ready outputs

2. **AdvancedAnalysisEngine** (analyze_results.py)
   - High-level analysis orchestration
   - Comprehensive CLI interface
   - Result validation and verification
   - Batch and comparative analysis support
   - JSON export capabilities
   - Verbose logging and debugging

**Key Capabilities:**

- **Statistical Analysis:**
  - Independent t-tests with effect sizes (Cohen's d)
  - ANOVA for multi-category comparisons
  - Confidence intervals and significance testing
  - Multiple comparison corrections

- **Visualization:**
  - Distribution violin plots
  - Correlation heatmaps
  - Effect size charts
  - Time series progression
  - Comprehensive dashboards
  - Professional styling and annotations

- **AI Integration:**
  - Ollama model support for intelligent insights
  - Automatic report content generation
  - Multi-paragraph executive summaries
  - Actionable recommendations

- **Report Generation:**
  - Responsive HTML reports with embedded charts
  - Professional Markdown with image references
  - Structured JSON data export
  - Automated report formatting and styling

- **Data Handling:**
  - Automatic corpus structure detection
  - Data validation and quality checks
  - Memory-efficient processing
  - Progress tracking with tqdm

**Usage Examples:**

```python
# Basic usage
from src.Phase3_Analysis import analyze_results
analyze_results("results.csv")

# Advanced analysis engine
from src.Phase3_Analysis import AdvancedAnalysisEngine
engine = AdvancedAnalysisEngine(
    "results.csv",
    model="phi3:mini",
    verbose=True
)
engine.run_analysis(generate_html=True, use_ai=True, generate_json=True)

# BiasAnalytics for custom workflows
from src.Phase3_Analysis import BiasAnalytics
analyzer = BiasAnalytics("results.csv")
analyzer.load_and_validate_data()
analyzer.perform_statistical_tests()
analyzer.generate_html_report(use_ai=True)
```

**Command-Line Usage:**

```bash
# Basic analysis
python -m src.Phase3_Analysis.analyze_results results.csv

# With AI insights
python -m src.Phase3_Analysis.analyze_results results.csv --model phi3:mini

# Full analysis (all outputs)
python -m src.Phase3_Analysis.analyze_results results.csv --full --verbose

# Without AI (faster)
python -m src.Phase3_Analysis.analyze_results results.csv --no-ai
```

**Configuration:**

The analysis engine respects the following environment variables:
- `OLLAMA_BASE_URL`: Ollama API endpoint (default: http://localhost:11434)

**Performance Characteristics:**

- Typical analysis time: 5-15 seconds per 1000 test results
- Memory usage: ~100MB for standard datasets
- Visualization generation: ~2-3 seconds per chart
- AI report generation: 10-30 seconds depending on model and content length

**Advanced Features:**

1. **Result Comparison:**
   ```python
   engine = AdvancedAnalysisEngine("audit1.csv")
   comparison = engine.compare_with("audit2.csv")
   ```

2. **Batch Processing:**
   Support for multiple results files with automated analysis

3. **Custom Output Directory:**
   ```python
   engine = AdvancedAnalysisEngine(
       "results.csv",
       output_dir="/custom/output/path"
   )
   ```

4. **Validation & Verification:**
   Automatic validation of results structure and data quality

**Error Handling:**

The module includes comprehensive error handling with:
- Input validation with detailed error messages
- Graceful fallbacks for missing AI services
- Logging support for debugging
- Verbose mode for troubleshooting

**Dependencies:**

- pandas: Data manipulation
- numpy: Numerical computing
- scipy: Statistical tests
- matplotlib: Visualization
- seaborn: Enhanced plotting
- jinja2: Report templating
- requests: HTTP requests for Ollama
- tqdm: Progress bars
"""

from Phase3_Analysis.analytics import BiasAnalytics
from Phase3_Analysis.analyze_results import (
    AdvancedAnalysisEngine,
    analyze_results,
    analyze_results_enhanced,
    main,
)

__all__ = [
    # Main classes
    "BiasAnalytics",
    "AdvancedAnalysisEngine",
    # Functions
    "analyze_results",
    "analyze_results_enhanced",
    "main",
]
