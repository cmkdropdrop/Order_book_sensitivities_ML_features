"""
Order‑Book Liquidity Feature Engineering 🕳📈

Install development version:

    pip install -e .

Run analysis:

    shap-analysis --symbol BTCUSDT ...

The public API is re‑exported here for convenience.
"""
from importlib.metadata import version

from .data_loader import load_price_series, load_depth_raw
from .depth_features import calc_features_per_snapshot
from .triple_barrier import ewm_vol, brute_triple_barrier
from .shap_runner import run_shap_analysis

__all__ = [
    "load_price_series",
    "load_depth_raw",
    "calc_features_per_snapshot",
    "ewm_vol",
    "brute_triple_barrier",
    "run_shap_analysis",
]

__version__ = version(__name__)
