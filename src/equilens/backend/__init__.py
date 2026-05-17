"""
EquiLens Backend API

FastAPI backend for managing EquiLens operations with persistent job tracking.
"""

from .api import app
from .database import init_db

__all__ = ["app", "init_db"]
