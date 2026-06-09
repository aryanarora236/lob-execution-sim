"""
Cross-date generalization: train on one session, test on another.

For each ticker, fits a logistic regression on each date and evaluates it on
every other date. If AUC on a held-out date matches in-sample AUC, the learned
patterns are durable. If AUC collapses, the model was overfitting to that
session's regime.

Run from project root:
    ./run.sh notebooks/14_cross_date.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = Path("results")
PLOTS_DIR   = Path("results/cross_date")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED   = 42
_DAY_START = 34_500.0
_DAY_END   = 57_300.0

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


def get_xy(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    avail = [f for f in FEATURES if f in df.columns]
    X = df.select(avail).to_numpy().astype(float)
    y = df["filled"].cast(pl.Int8).to_numpy()
    mask = ~np.isnan(X).any(axis=1)
    return X[mask], y[mask]


def cross_date_auc(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
) -> float | None:
    X_tr, y_tr = get_xy(train_df)
    X_te, y_te = get_xy(test_df)
    if len(np.unique(y_te)) < 2 or len(X_tr) < 50:
        return None
    sc = StandardScaler().fit(X_tr)
    m  = LogisticRegression(max_iter=1_000, random_state=RNG_SEED)
    m.fit(sc.transform(X_tr), y_tr)
    prob = m.predict_proba(sc.transform(X_te))[:, 1]
    return roc_auc_score(y_te, prob)


print(f"\n{SEP}")
print("CROSS-DATE GENERALIZATION — L2 FILL PROBABILITY")
print(SEP)

df_l2 = load_pooled(2)
dates  = sorted(df_l2["date"].unique().to_list())

all_results: dict[str, dict] = {}

for ticker in TICKERS:
    sub   = df_l2.filter(pl.col("ticker") == ticker)
    t_dates = sorted(sub["date"].unique().to_list())
    matrix: dict[tuple[str, str], float | None] = {}

    print(f"\n  {ticker}  (dates: {', '.join(t_dates)})")
    print(f"  {'Train \\ Test':<14}  " + "  ".join(f"{d:>12}" for d in t_dates))

    for train_date in t_dates:
        row_vals = []
        line = f"  {train_date:<14}  "
        for test_date in t_dates:
            train_df = sub.filter(pl.col("date") == train_date)
            test_df  = sub.filter(pl.col("date") == test_date)
            auc = cross_date_auc(train_df, test_df)
            matrix[(train_date, test_date)] = auc
            if auc is None:
                cell = "    n/a     "
            elif train_date == test_date:
                cell = f"  [{auc:.3f}]   "   # in-sample diagonal
            else:
                cell = f"   {auc:.3f}    "
            row_vals.append(auc)
            line += cell
        print(line)

    all_results[ticker] = {"dates": t_dates, "matrix": matrix}

    # diagonal (in-sample) vs off-diagonal (OOS) summary
    in_sample  = [matrix[(d, d)] for d in t_dates if matrix.get((d, d)) is not None]
    oos_vals   = [v for (tr, te), v in matrix.items() if tr != te and v is not None]
    if in_sample and oos_vals:
        print(
            f"\n  Mean in-sample AUC : {np.mean(in_sample):.3f}"
            f"\n  Mean OOS AUC       : {np.mean(oos_vals):.3f}"
            f"  (drop = {np.mean(in_sample) - np.mean(oos_vals):+.3f})"
        )

# ── heatmap of cross-date AUC matrices ───────────────────────────────────────
fig, axes = plt.subplots(1, len(TICKERS), figsize=(13, 4))

for ax, ticker in zip(axes, TICKERS):
    res   = all_results[ticker]
    t_dates = res["dates"]
    matrix  = res["matrix"]
    n = len(t_dates)
    grid = np.full((n, n), np.nan)
    for i, tr in enumerate(t_dates):
        for j, te in enumerate(t_dates):
            v = matrix.get((tr, te))
            if v is not None:
                grid[i, j] = v

    im = ax.imshow(grid, vmin=0.45, vmax=0.65, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    short = [d[5:] for d in t_dates]  # MM-DD
    ax.set_xticklabels(short, fontsize=7, rotation=30)
    ax.set_yticklabels(short, fontsize=7)
    ax.set_xlabel("Test date", fontsize=8)
    ax.set_ylabel("Train date", fontsize=8)
    ax.set_title(f"{ticker} L2 cross-date AUC", fontsize=9)

    for i in range(n):
        for j in range(n):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=7, color="black",
                        fontweight="bold" if i == j else "normal")

plt.colorbar(im, ax=axes[-1], shrink=0.8, label="AUC")
plt.suptitle("Cross-date generalization — L2 logistic regression AUC\n"
             "(rows=train date, cols=test date; diagonal=in-sample)", fontsize=10)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "cross_date_auc_l2.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {PLOTS_DIR}/cross_date_auc_l2.png")

print(f"\n{SEP}")
print("CROSS-DATE ANALYSIS COMPLETE")
print(SEP)
