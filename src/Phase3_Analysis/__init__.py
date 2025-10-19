"""
EquiLens Phase 3: Analysis Module

This module provides comprehensive bias analysis capabilities for EquiLens audit results.

Main Components:
- analytics.py: Unified BiasAnalytics class with AI-powered reporting
- analyze_results.py: Main analysis entry point

Key Features:
- Statistical significance testing (t-tests, effect sizes, confidence intervals)
- Rich visualizations (violin plots, heatmaps, effect size charts)
- AI-powered report generation using Ollama models
- Professional HTML and Markdown reports
- Jinja2-based templating for reports
"""

from .analyze_results import analyze_results
from .analytics import BiasAnalytics

__all__ = [
    "analyze_results",
    "BiasAnalytics",
]

