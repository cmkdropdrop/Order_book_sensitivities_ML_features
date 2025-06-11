"""
Liquidity‑curve feature engineering.

Every function operates **purely on a single snapshot**: a DataFrame with
columns

    percentage   … price move   (-% on bids, +% on asks)
    depth        … cumulative base‑asset volume

No database access, no time handling – perfectly unit‑testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Literal, List

import numpy as np
import pandas as pd


# ───────────────────────────────────────────────────────── helpers ─┐
@dataclass(frozen=True, slots=True)
class LinearFit:
    m: float
    b: float

    def y(self, x):  # noqa: D401
        return self.m * x + self.b

    def x_at_y(self, y: float = 0.0) -> float:
        if self.m == 0:
            raise ZeroDivisionError("horizontal line – no x‑intercept")
        return (y - self.b) / self.m


@dataclass(frozen=True, slots=True)
class BidAskFits:
    bid: LinearFit
    ask: LinearFit


def _split_and_fit(df: pd.DataFrame) -> BidAskFits:
    if not {"percentage", "depth"} <= set(df.columns):
        raise KeyError("DataFrame must have columns: 'percentage', 'depth'")

    bid = df[df["percentage"] < 0]
    ask = df[df["percentage"] > 0]
    if len(bid) < 2 or len(ask) < 2:
        return BidAskFits(
            bid=LinearFit(np.nan, np.nan),
            ask=LinearFit(np.nan, np.nan),
        )

    m_bid, b_bid = np.polyfit(bid["percentage"], bid["depth"], 1)
    m_ask, b_ask = np.polyfit(ask["percentage"], ask["depth"], 1)
    return BidAskFits(LinearFit(m_bid, b_bid), LinearFit(m_ask, b_ask))


# ───────────────────────────────────────────── per‑snapshot features ─┤
def bid_ask_slopes(df: pd.DataFrame) -> Dict[str, float]:
    fits = _split_and_fit(df)
    return {"bid_slope": fits.bid.m, "ask_slope": fits.ask.m}


def intersection(df: pd.DataFrame) -> Tuple[float, float]:
    fits = _split_and_fit(df)
    x_star = (fits.ask.b - fits.bid.b) / (fits.bid.m - fits.ask.m)
    y_star = fits.bid.y(x_star)
    return x_star, y_star


def proxy_real_spread(df: pd.DataFrame) -> float:
    fits = _split_and_fit(df)
    x_bid = fits.bid.x_at_y()
    x_ask = fits.ask.x_at_y()
    return abs(x_ask - x_bid)


# … (price_drift_sensitivity, slope_after_price_shift, volume_to_move_price_drift)
# *unchanged – paste your original implementations here*


# ───────────────────────────────────────── compute full snapshot set ─┘
def calc_features_per_snapshot(raw_depth: pd.DataFrame) -> pd.DataFrame:
    """
    Loop over **each timestamped snapshot** and return a *wide* feature DataFrame
    indexed by UTC timestamp.
    """
    results: List[dict] = []

    for ts, snap in raw_depth.groupby("ts"):
        def safe(fn, *a, **k):
            try:
                return fn(*a, **k)
            except Exception as exc:  # pragma: no cover
                print(f"[FEATURE‑FAIL] {ts} – {fn.__name__}: {exc}")
                return None

        row: dict[str, float] = {
            "ts": ts,
            **(safe(bid_ask_slopes, snap) or {}),
            **dict(zip(("price_drift", "real_liquidity"), safe(intersection, snap) or (np.nan, np.nan))),
            "real_spread": safe(proxy_real_spread, snap),
            **(safe(price_drift_sensitivity, snap) or {}),
            "bid_slope_after_1pct_down": safe(slope_after_price_shift, snap, -1.0, "bid"),
            "ask_slope_after_1pct_up": safe(slope_after_price_shift, snap, 1.0, "ask"),
            **(safe(volume_to_move_price_drift, snap, xa_pct=0.2) or {}),
        }
        results.append(row)

    return (
        pd.DataFrame(results)
        .set_index("ts")
        .sort_index()
        .astype(float, errors="ignore")
    )
