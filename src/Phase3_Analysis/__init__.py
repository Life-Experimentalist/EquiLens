"""
EquiLens Phase 3: Analysis Module

This module provides comprehensive bias analysis capabilities for EquiLens audit results.

Main Components:
- enhanced_analyzer.py: Comprehensive BiasAnalyzer class with statistical analysis
- analyze_results.py: Main analysis functions with enhanced and legacy modes

Key Features:
- Statistical significance testing
- Multi-dimensional bias analysis
- Performance metrics analysis
- HTML report generation
- Comprehensive visualizations
"""

from .analyze_results import (
    analyze_results,
    analyze_results_enhanced,
    analyze_results_legacy,
)
from .enhanced_analyzer import BiasAnalyzer

__all__ = [
    "analyze_results",
    "analyze_results_enhanced",
    "analyze_results_legacy",
    "BiasAnalyzer",
]
