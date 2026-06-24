"""
Shared pytest fixtures for the test suite.

Provides standard market parameters used across multiple test files to avoid re-declaring S, K, T, r, sigma, q every time.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is importable as `src.*` regardless of where pytest is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def standard_params():
    """Standard ATM option parameters used across most pricing tests."""
    return dict(S=100.0, K=100.0, T=1, r=0.05, sigma=0.20, q=0)


@pytest.fixture
def dividend_params():
    """Standard parameters with a non-zero continuous dividend yield."""
    return dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.02)


@pytest.fixture
def rng():
    """Fixed-seed random generator for reproducible test runs."""
    return np.random.default_rng(926)