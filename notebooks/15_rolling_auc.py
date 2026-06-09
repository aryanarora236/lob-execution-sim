"""
Rolling AUC — intraday stationarity analysis.

Fits a logistic regression on each ticker's first 50% of the day (by
entry_timestamp) and evaluates AUC in rolling 30-minute windows across the
full day. A flat AUC near the in-sample value = stationary. A declining or
inverting AUC = non-stationarity (the model learned patterns specific to the
morning regime).

Also plots per-date rolling AUC for MSFT to pinpoint which sessions drive
the non-stationarity.

Run from project root:
    ./run.sh notebooks/15_rolling_auc.py
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

RESULTS_DIR  = Path("results")
PLOTS_DIR    = Path("results/rolling_auc")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED     = 42
_DAY_START   = 34_500.0   # 09:35
_DAY_END     = 57_300.0   # 15:55
WINDOW_SEC   = 1_800.0    # 30-minute rolling window
STEP_SEC     = 600.0      # 10-minute step
MIN_WINDOW_N = 50         # minimum orders in a window to compute AUC

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


def rolling_auc_series(
    df: pl.DataFrame,
    train_frac: float = 0.5,
) -> tuple[list[float], list[float]]:
    """
    Train on the first train_frac of the day, evaluate in rolling windows.
    Returns (window_midpoints_hours, auc_values).
    """
    df = df.sort("entry_timestamp")
    ts  = df["entry_timestamp"].to_numpy()
    n   = len(ts)
    split = int(n * train_frac)
    if split < MIN_WINDOW_N:
        return [], []

    X, y = get_xy(df)
    sc = StandardScaler().fit(X[:split])
    m  = LogisticRegression(max_iter=1_000, random_state=RNG_SEED)
    m.fit(sc.transform(X[:split]), y[:split])

    midpoints = []
    aucs      = []

    t_start = ts[0]
    t_end   = ts[-1]
    cursor  = t_start + WINDOW_SEC / 2

    while cursor - WINDOW_SEC / 2 < t_end:
        lo = cursor - WINDOW_SEC / 2
        hi = cursor + WINDOW_SEC / 2
        mask = (ts >= lo) & (ts < hi)
        if mask.sum() >= MIN_WINDOW_N and len(np.unique(y[mask])) == 2:
            prob = m.predict_proba(sc.transform(X[mask]))[:, 1]
            aucs.append(roc_auc_score(y[mask], prob))
            midpoints.append(cursor / 3600)   # convert to hours
        cursor += STEP_SEC

    return midpoints, aucs


print(f"\n{SEP}")
print("ROLLING AUC — INTRADAY STATIONARITY")
print(SEP)

df_l2 = load_pooled(2)

# ── pooled cross-ticker plot ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(TICKERS), figsize=(14, 4), sharey=True)
colors = {"AAPL": "#4a90d9", "INTC": "#e07b3a", "MSFT": "#5cb85c"}

for ax, ticker in zip(axes, TICKERS):
    sub = df_l2.filter(pl.col("ticker") == ticker).sort("entry_timestamp")
    mids, aucs = rolling_auc_series(sub)
    if not mids:
        ax.set_title(f"{ticker} — insufficient data")
        continue
    ax.plot(mids, aucs, color=colors[ticker], linewidth=1.5, label="pooled")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="chance")
    ax.axhline(np.mean(aucs), color=colors[ticker], linestyle=":", linewidth=0.8,
               label=f"mean={np.mean(aucs):.3f}")
    # shade the training region
    train_cutoff = sub["entry_timestamp"][int(len(sub) * 0.5)] / 3600
    ax.axvspan(sub["entry_timestamp"][0] / 3600, train_cutoff,
               alpha=0.08, color=colors[ticker], label="train region")
    ax.set_title(f"{ticker} L2 rolling AUC (30-min window)", fontsize=9)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("AUC")
    ax.legend(fontsize=7)
    ax.set_ylim(0.35, 0.75)
    print(f"  {ticker}  mean AUC={np.mean(aucs):.3f}  min={min(aucs):.3f}  max={max(aucs):.3f}")

plt.suptitle("Rolling AUC — model trained on first 50% of pooled day\n"
             "(shaded = training region; dashed = chance)", fontsize=10)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "rolling_auc_by_ticker.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {PLOTS_DIR}/rolling_auc_by_ticker.png")

# ── per-date breakdown for MSFT (to identify which session drives non-stationarity)
print(f"\n  MSFT per-date rolling AUC:")
msft = df_l2.filter(pl.col("ticker") == "MSFT")
msft_dates = sorted(msft["date"].unique().to_list())

fig2, axes2 = plt.subplots(1, len(msft_dates), figsize=(13, 4), sharey=True)
date_colors = ["#4a90d9", "#e07b3a", "#5cb85c"]

for ax, date, dc in zip(axes2, msft_dates, date_colors):
    sub = msft.filter(pl.col("date") == date).sort("entry_timestamp")
    mids, aucs = rolling_auc_series(sub)
    if not mids:
        ax.set_title(f"MSFT {date} — insufficient data")
        continue
    ax.plot(mids, aucs, color=dc, linewidth=1.5)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.axhline(np.mean(aucs), color=dc, linestyle=":", linewidth=0.8)
    train_cutoff = sub["entry_timestamp"][int(len(sub) * 0.5)] / 3600
    ax.axvspan(sub["entry_timestamp"][0] / 3600, train_cutoff,
               alpha=0.10, color=dc)
    ax.set_title(f"MSFT {date}\nmean={np.mean(aucs):.3f}", fontsize=9)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.3, 0.8)
    min_auc  = min(aucs)
    min_hour = mids[aucs.index(min_auc)]
    print(f"    {date}  mean={np.mean(aucs):.3f}  min={min_auc:.3f} at {min_hour:.1f}h  max={max(aucs):.3f}")

plt.suptitle("MSFT rolling AUC per date — locating non-stationarity\n"
             "(shaded = training region; dotted = session mean)", fontsize=10)
plt.tight_layout()
fig2.savefig(PLOTS_DIR / "rolling_auc_msft_per_date.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"\n  Saved: {PLOTS_DIR}/rolling_auc_msft_per_date.png")

print(f"\n{SEP}")
print("ROLLING AUC COMPLETE")
print(SEP)
