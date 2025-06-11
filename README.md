# Order Book Liquidity Machine-Learning Features

This repository provides utilities to engineer depth-based features from order-book snapshots and evaluate their relevance to short-horizon price moves.

## Feature families
Features are derived from individual snapshots of the book and then "as‑of" merged to bar intervals so that predictors and labels refer to the same market state. The metrics fall into four broad groups:

1. **Immediate depth** – `bid_slope`, `ask_slope`, and `real_spread` describe how thick each side of the book is and reveal its true tightness.
2. **Balance / pressure** – `price_drift` and `real_liquidity` capture where the fitted bid/ask curves intersect, signalling directional bias and the inventory required to neutralise it.
3. **Elasticity** – `sensi_depth_vs_*` along with `bid_slope_after_1pct_down` and `ask_slope_after_1pct_up` measure how the book shape reacts when depth is pulled or consumed.
4. **Impact cost** – `sensi_price_shift_sensi_vs_AUC_buy` and `sensi_price_shift_sensi_vs_AUC_sell` estimate the market volume needed to displace the equilibrium by a small amount.

A detailed description of each feature is available in [`docs/features_cheatsheet.md`](docs/features_cheatsheet.md).

## Determining relevance
The `shap-analysis` command (see [`src/orderbook_liquidity`](src/orderbook_liquidity)) trains an XGBoost model on labelled bars and computes SHAP values to attribute importance. Features with larger mean absolute SHAP values are considered more informative. The code produces bar and summary plots for each target class under `docs/images/`. Below is an example output:

![SHAP bar plot](docs/images/shap_summary_-1.0.png)

For a complete walkthrough check [`docs/EXAMPLE_RESULTS.md`](docs/EXAMPLE_RESULTS.md).

## Repository layout
- `src/orderbook_liquidity` – feature engineering, labelling utilities and the SHAP runner.
- `scripts/` – thin wrapper around the CLI.
- `docs/` – feature cheat sheet and example SHAP outputs.
- `tests/` – unit tests for the snapshot metrics.
