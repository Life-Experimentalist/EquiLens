"""
EquiLens - AI Bias Detection Platform

A comprehensive platform for detecting and analyzing bias in AI language models.
"""

__version__ = "0.1.0"
__author__ = "VKrishna04"
__email__ = "your.email@example.com"

from .core.manager import EquiLensManager
from .core.gpu import GPUManager
from .core.docker import DockerManager

__all__ = [
    "EquiLensManager",
    "GPUManager",
    "DockerManager",
]
