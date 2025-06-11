"""EWM volatility and a **pure‑Python** Triple‑Barrier Method labeller."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ewm_vol(returns: pd.Series, span: int = 100) -> pd.Series:
    """Exponential‑weighted σ."""
    return returns.ewm(span=span, adjust=False).std()


def brute_triple_barrier(
    close: pd.Series,
    vol: pd.Series,
    pt_mult: float,
    sl_mult: float,
    max_hold: int,
) -> pd.Series:
    """
    Very small & dependency‑free variant of López de Prado’s TBM.

    Returns a Series of **+1 / 0 / –1** labels aligned with *close*.
    """
    labels = pd.Series(index=close.index, dtype="int8")
    n = len(close)

    for i in range(n):
        price0, vol0 = close.iat[i], vol.iat[i]
        if np.isnan(vol0) or vol0 == 0:
            labels.iat[i] = 0
            continue

        pt = price0 * (1 + pt_mult * vol0)
        sl = price0 * (1 - sl_mult * vol0)

        vt_idx = min(i + max_hold, n - 1)
        window = close.iloc[i : vt_idx + 1]

        hit = window[(window >= pt) | (window <= sl)]
        if hit.empty:
            labels.iat[i] = 0
        else:
            first_touch_price = hit.iloc[0]
            labels.iat[i] = 1 if first_touch_price >= pt else -1
    return labels
