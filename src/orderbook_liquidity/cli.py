"""High‑level orchestration – this is what the console script calls."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data_loader import load_price_series, load_depth_raw
from .triple_barrier import ewm_vol, brute_triple_barrier
from .depth_features import calc_features_per_snapshot
from .shap_runner import run_shap_analysis

# TBM hyper‑params (configurable via CLI if you wish)
PT_CONST = 0.87
SL_CONST = 0.87
MAX_HOLD_BARS = 29


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SHAP analysis of order‑book liquidity features")
    p.add_argument("--symbol", default="AAVEUSDT", help="Asset symbol")
    p.add_argument("--lookback-days", type=int, default=180)
    p.add_argument("--freq-min", type=int, default=15)
    p.add_argument("--depth-db", type=Path, default=Path("binance_swaps.sqlite"))
    p.add_argument("--ohlcv-db", type=Path, default=Path("ohlcv.sqlite"))
    p.add_argument("--out-dir", type=Path, default=Path("docs/images"))
    return p.parse_args(argv)


def main(argv=None) -> None:  # noqa: D401
    args = _parse_args(argv)

    # 1️⃣  Prices & labels
    print("Loading price series …")
    close = load_price_series(args.ohlcv_db, args.symbol, args.lookback_days, args.freq_min)
    vol = ewm_vol(close.pct_change().fillna(0.0), span=100).bfill()
    labels = brute_triple_barrier(close, vol, PT_CONST, SL_CONST, MAX_HOLD_BARS)

    # 2️⃣  Depth snapshots
    start_ms = int(close.index.min().timestamp() * 1_000)
    end_ms = int(close.index.max().timestamp() * 1_000)
    print("Loading depth snapshots …")
    raw_depth = load_depth_raw(args.depth_db, args.symbol, start_ms, end_ms)

    # 3️⃣  Features & alignment
    print("Computing features …")
    feat_snap = calc_features_per_snapshot(raw_depth)

    print("Merging snapshots to bars …")
    labels_df = labels.to_frame("label").sort_index()
    aligned = pd.merge_asof(
        labels_df,
        feat_snap.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
        tolerance=pd.Timedelta(minutes=args.freq_min),
    )

    aligned = (
        aligned.dropna(axis=1, how="all")
        .dropna(subset=["label", "bid_slope", "ask_slope"])
        .ffill()
        .bfill()
    )
    if aligned.shape[1] < 5:
        raise RuntimeError("Fewer than five usable features – aborting.")

    X = aligned.drop(columns=["label"])
    y = aligned["label"]
    print(f"Dataset: {len(aligned)} bars × {X.shape[1]} features")

    print("=== NA counts per feature ===")
    print(X.isna().sum())

    # 4️⃣  SHAP
    run_shap_analysis(X, y, args.out_dir)


if __name__ == "__main__":
    main()
