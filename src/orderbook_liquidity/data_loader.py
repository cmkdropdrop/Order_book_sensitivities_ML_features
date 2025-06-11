"""SQLite helpers for OHLCV bars and raw depth snapshots."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


def load_price_series(
    db_path: Path | str,
    symbol: str,
    lookback_days: int,
    freq_minutes: int,
) -> pd.Series:
    """Return *close* prices resampled to `freq_minutes` for the desired span."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT MAX(open_time) FROM ohlcv WHERE symbol = ?", (symbol,))
    latest_ms = cur.fetchone()[0]
    if latest_ms is None:
        raise RuntimeError(f"No OHLCV data for {symbol!r}")

    latest_ts = pd.to_datetime(latest_ms, unit="ms", utc=True)
    start_ts = latest_ts - pd.Timedelta(days=lookback_days)
    start_ms = int(start_ts.timestamp() * 1_000)

    cur.execute(
        """
        SELECT open_time, close
        FROM   ohlcv
        WHERE  symbol = ? AND open_time >= ?
        ORDER BY open_time ASC
        """,
        (symbol, start_ms),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        raise RuntimeError("Price query yielded no rows")

    df = pd.DataFrame(rows, columns=["open_time", "close"])
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    closes = (
        df.set_index("ts")
        .sort_index()
        .resample(f"{freq_minutes}T")
        .last()
        .dropna()["close"]
        .astype(float)
    )
    return closes


def load_depth_raw(
    db_path: Path | str,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    """Depth curve snapshots from `book_depth` (*percentage*, *depth*)."""
    conn = sqlite3.connect(db_path)
    sql = """
        SELECT timestamp, percentage, depth
        FROM   book_depth
        WHERE  symbol = ? AND timestamp BETWEEN ? AND ?
    """
    df = pd.read_sql_query(sql, conn, params=[symbol, start_ms, end_ms])
    conn.close()

    if df.empty:
        raise RuntimeError("Depth query yielded no rows")

    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df
