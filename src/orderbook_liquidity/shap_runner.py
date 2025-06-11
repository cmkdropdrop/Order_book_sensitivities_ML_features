"""Train an XGBoost model and save SHAP plots per class."""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb


def _normalise_shap_values(raw, n_feat: int, n_class: int) -> List[np.ndarray]:
    """Massage SHAP outputs (old list or new ndarray) to `[array(class0), …]`."""
    def _trim_bias(arr):
        arr = np.asarray(arr)
        return arr[:, :-1] if arr.ndim == 2 and arr.shape[1] == n_feat + 1 else arr

    if isinstance(raw, list):
        return [_trim_bias(v) for v in raw]

    raw = np.asarray(raw)

    if raw.ndim == 3:                    # (samples, class, feature) or (s, f, c)
        if raw.shape[1] == n_class:
            return [_trim_bias(raw[:, k, :]) for k in range(n_class)]
        if raw.shape[2] == n_class:
            return [_trim_bias(raw[:, :, k]) for k in range(n_class)]
    elif raw.ndim == 2:                  # binary
        return [_trim_bias(raw)]

    raise ValueError(f"Unrecognised SHAP shape {raw.shape}")


def run_shap_analysis(X: pd.DataFrame, y: pd.Series, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    lbls = y.sort_values().unique()
    enc = {lbl: i for i, lbl in enumerate(lbls)}
    y_enc = y.map(enc).astype(int)

    dtrain = xgb.DMatrix(X, label=y_enc)
    params = dict(
        objective="multi:softprob",
        num_class=len(lbls),
        eval_metric="mlogloss",
        max_depth=4,
        eta=0.25,
        subsample=0.8,
        colsample_bytree=0.8,
        seed=2025,
    )
    bst = xgb.train(params, dtrain, num_boost_round=300)

    explainer = shap.TreeExplainer(bst)
    shap_vals = _normalise_shap_values(
        explainer.shap_values(X), X.shape[1], len(lbls)
    )

    for lbl, sv in zip(lbls, shap_vals):
        shap.summary_plot(
            sv,
            X,
            feature_names=X.columns,
            max_display=len(X.columns),
            show=False,
            plot_size=(12, 8),
        )
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_summary_{lbl}.png", dpi=150)
        plt.close()

        shap.summary_plot(
            sv,
            X,
            feature_names=X.columns,
            plot_type="bar",
            max_display=len(X.columns),
            show=False,
            plot_size=(10, 6),
        )
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_importance_{lbl}.png", dpi=150)
        plt.close()

    print("✅  SHAP plots saved to", out_dir.resolve())
