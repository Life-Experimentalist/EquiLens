"""
EquiLens - AI Bias Detection Platform

A comprehensive platform for detecting and analyzing bias in AI language models.
"""

__version__ = "0.1.0"
__author__ = "VKrishna04"
__email__ = "equilens@vkrishna04.me"

from equilens.core.docker import DockerManager
from equilens.core.gpu import GPUManager
from equilens.core.manager import EquiLensManager

__all__ = [
    "EquiLensManager",
    "GPUManager",
    "DockerManager",
]
