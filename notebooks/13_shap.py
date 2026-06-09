"""
SHAP feature importance for LightGBM fill-probability models.

Produces beeswarm and bar plots per ticker at L2, showing which features
drive fill probability and in which direction — more interpretable than
standardised logistic coefficients.

Run from project root:
    ./run.sh notebooks/13_shap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = Path("results")
PLOTS_DIR   = Path("results/shap")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED    = 42
N_CV_SPLITS = 4
_DAY_START  = 34_500.0
_DAY_END    = 57_300.0

FEATURES = [
    "spread_at_entry_ticks",
    "book_imbalance_at_entry",
    "queue_position_at_entry",
    "queue_granularity_at_entry",
    "time_frac",
    "side_bid",
    "ofi_10s",
    "ofi_30s",
]

FEATURE_LABELS = {
    "spread_at_entry_ticks":    "Spread (ticks)",
    "book_imbalance_at_entry":  "Book imbalance",
    "queue_position_at_entry":  "Queue position (shares)",
    "queue_granularity_at_entry": "Granularity (K/Q)",
    "time_frac":                "Time of day",
    "side_bid":                 "Side = bid",
    "ofi_10s":                  "OFI 10s",
    "ofi_30s":                  "OFI 30s",
}

TICKERS = ["AAPL", "INTC", "MSFT"]
SEP = "=" * 65


def load_pooled(level: int) -> pl.DataFrame:
    path = RESULTS_DIR / f"experiment_all_L{level}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    df = pl.read_parquet(path)
    return df.with_columns(
        pl.col("side").eq("bid").cast(pl.Int8).alias("side_bid"),
        ((pl.col("entry_timestamp") - _DAY_START) / (_DAY_END - _DAY_START))
        .clip(0.0, 1.0).alias("time_frac"),
    )


def get_xy(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    avail = [f for f in FEATURES if f in df.columns]
    X = df.select(avail).to_numpy().astype(float)
    y = df["filled"].cast(pl.Int8).to_numpy()
    mask = ~np.isnan(X).any(axis=1)
    return X[mask], y[mask], avail


def train_lgb_last_fold(
    X: np.ndarray, y: np.ndarray, feat_names: list[str]
) -> tuple[lgb.LGBMClassifier, np.ndarray, np.ndarray]:
    """Train LGB on the largest walk-forward fold (last split) and return
    the fitted model plus the held-out X and y for SHAP explanation."""
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]  # largest training set
    model = lgb.LGBMClassifier(
        objective="binary", metric="auc", verbosity=-1,
        n_estimators=300, learning_rate=0.05,
        num_leaves=31, random_state=RNG_SEED,
        feature_name=feat_names,
    )
    model.fit(X[train_idx], y[train_idx], feature_name=feat_names)
    return model, X[test_idx], y[test_idx]


print(f"\n{SEP}")
print("SHAP FEATURE IMPORTANCE — L2 FILL PROBABILITY")
print(SEP)

df_l2 = load_pooled(2)

# ── per-ticker beeswarm plots ─────────────────────────────────────────────────
fig_bee, axes_bee = plt.subplots(1, len(TICKERS), figsize=(15, 5))

for ax, ticker in zip(axes_bee, TICKERS):
    sub = df_l2.filter(pl.col("ticker") == ticker)
    X, y, feat_names = get_xy(sub)
    n = len(X)

    model, X_test, y_test = train_lgb_last_fold(X, y, feat_names)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # Use the SHAP values for class=1 (filled)
    sv = shap_values[..., 1] if shap_values.values.ndim == 3 else shap_values

    # Pretty feature names
    sv.feature_names = [FEATURE_LABELS.get(f, f) for f in feat_names]

    plt.sca(ax)
    shap.plots.beeswarm(sv, max_display=8, show=False, color_bar=False)
    ax.set_title(f"{ticker} L2  (n={n:,})", fontsize=10)
    ax.set_xlabel("SHAP value (impact on fill probability)", fontsize=8)

plt.suptitle("SHAP beeswarm — L2 fill probability by ticker", fontsize=11, y=1.01)
plt.tight_layout()
fig_bee.savefig(PLOTS_DIR / "shap_beeswarm_l2.png", dpi=150, bbox_inches="tight")
plt.close(fig_bee)
print(f"\n  Saved: {PLOTS_DIR}/shap_beeswarm_l2.png")

# ── mean |SHAP| bar chart — all tickers side by side ─────────────────────────
mean_shap_rows = []
for ticker in TICKERS:
    sub = df_l2.filter(pl.col("ticker") == ticker)
    X, y, feat_names = get_xy(sub)
    model, X_test, _ = train_lgb_last_fold(X, y, feat_names)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    sv = shap_values[..., 1] if shap_values.values.ndim == 3 else shap_values
    mean_abs = np.abs(sv.values).mean(axis=0)
    mean_shap_rows.append((ticker, feat_names, mean_abs))
    print(f"\n  {ticker} mean |SHAP| (L2, last CV fold test set):")
    order = np.argsort(mean_abs)[::-1]
    for i in order:
        label = FEATURE_LABELS.get(feat_names[i], feat_names[i])
        print(f"    {label:<30s}  {mean_abs[i]:.4f}")

# bar chart
fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
colors = {"AAPL": "#4a90d9", "INTC": "#e07b3a", "MSFT": "#5cb85c"}
x_labels = [FEATURE_LABELS.get(f, f) for f in mean_shap_rows[0][1]]
x = np.arange(len(x_labels))
w = 0.25

for i, (ticker, feat_names, mean_abs) in enumerate(mean_shap_rows):
    # reorder to match x_labels
    label_to_val = dict(zip([FEATURE_LABELS.get(f, f) for f in feat_names], mean_abs))
    vals = [label_to_val.get(lbl, 0.0) for lbl in x_labels]
    ax_bar.bar(x + i * w, vals, w, label=ticker, color=colors[ticker], alpha=0.85)

ax_bar.set_xticks(x + w)
ax_bar.set_xticklabels(x_labels, rotation=25, ha="right", fontsize=8)
ax_bar.set_ylabel("Mean |SHAP value|")
ax_bar.set_title("Feature importance (mean |SHAP|) — L2 fill probability by ticker")
ax_bar.legend()
plt.tight_layout()
fig_bar.savefig(PLOTS_DIR / "shap_bar_l2.png", dpi=150, bbox_inches="tight")
plt.close(fig_bar)
print(f"\n  Saved: {PLOTS_DIR}/shap_bar_l2.png")

print(f"\n{SEP}")
print("SHAP ANALYSIS COMPLETE")
print(SEP)
