"""
Unit tests for the per‑snapshot liquidity metrics.

Only numpy / pandas are required – no database or XGBoost dependency.
"""

import numpy as np
import pandas as pd
import pytest

from orderbook_liquidity.depth_features import (
    bid_ask_slopes,
    intersection,
    proxy_real_spread,
    price_drift_sensitivity,
    slope_after_price_shift,
    volume_to_move_price_drift,
)


@pytest.fixture(scope="module")
def df_linear() -> pd.DataFrame:
    """
    Create an *ideal* symmetric depth curve:

        bids :  depth = -10 * percentage        (percentage < 0)
        asks :  depth =  10 * percentage        (percentage > 0)

    → bid slope  = −10  
      ask slope  = +10  
      intersection at (0, 0)
    """
    pct = np.concatenate([np.arange(-5, 0), np.arange(1, 6)])  # −5 … −1, 1 … 5
    depth = 10 * np.abs(pct)                                   # 10,20,…
    return pd.DataFrame({"percentage": pct, "depth": depth})


def test_bid_ask_slopes(df_linear):
    slopes = bid_ask_slopes(df_linear)
    assert slopes["bid_slope"] < 0
    assert slopes["ask_slope"] > 0
    # absolute values should match in this symmetric fixture
    np.testing.assert_allclose(abs(slopes["bid_slope"]), slopes["ask_slope"], rtol=1e-6)


def test_intersection(df_linear):
    x_star, y_star = intersection(df_linear)
    assert abs(x_star) < 1e-6
    assert abs(y_star) < 1e-6


def test_proxy_real_spread(df_linear):
    spread = proxy_real_spread(df_linear)
    # Perfect symmetry ⇒ spread ≈ 0
    assert spread < 1e-6


def test_price_drift_sensitivity(df_linear):
    sens = price_drift_sensitivity(df_linear)
    expected_keys = {"sensi_depth_vs_price_increase", "sensi_depth_vs_price_decrease"}
    assert expected_keys <= sens.keys()
    # Values should be finite (not NaN / inf)
    for v in sens.values():
        assert np.isfinite(v)


def test_slope_after_price_shift(df_linear):
    # After removing bids inside −1 %, slope should remain ≈ original (−10)
    slope_bid_after = slope_after_price_shift(df_linear, -1.0, "bid")
    np.testing.assert_allclose(slope_bid_after, -10, rtol=1e-6)


def test_volume_to_move_price_drift(df_linear):
    vols = volume_to_move_price_drift(df_linear, xa_pct=0.2)
    assert vols["sensi_price_shift_sensi_vs_AUC_buy"] >= 0
    assert vols["sensi_price_shift_sensi_vs_AUC_sell"] >= 0