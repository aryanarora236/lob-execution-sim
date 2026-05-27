"""
Multi-ticker cross-sectional analysis.

Compares fill probability predictability across tickers (AAPL, MSFT, INTC)
and dates (2019-12-30 holiday vs 2019-08-30 normal session).

Questions
---------
1. Does the touch/depth AUC gap (L1 ≈ 0.49 vs L2 ≈ 0.54) replicate in MSFT and INTC?
2. Does granularity (K/Q) gain predictive power in wider-spread tickers?
3. Does the side_bid and imbalance effect at L2 hold out-of-sample (different date)?

Run from project root:
    uv run python notebooks/11_multi_ticker.py
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
PLOTS_DIR   = Path("results/multi_ticker")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 0
N_BOOT   = 2_000

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

_DAY_START = 34_500.0  # 09:35
_DAY_END   = 57_300.0  # 15:55


# ── helpers ───────────────────────────────────────────────────────────────────

def load_pooled(depth_level: int) -> pl.DataFrame:
    """Load pooled parquet if it exists, otherwise concat individual files."""
    pool_path = RESULTS_DIR / f"experiment_all_L{depth_level}.parquet"
    if pool_path.exists():
        return pl.read_parquet(pool_path)
    files = sorted(RESULTS_DIR.glob(f"experiment_*_L{depth_level}.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No L{depth_level} parquets found in {RESULTS_DIR}. "
            "Run experiment_runner.py first."
        )
    return pl.concat([pl.read_parquet(f) for f in files])


def prep_features(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = df.with_columns(
        pl.col("side").eq("bid").cast(pl.Int8).alias("side_bid"),
        ((pl.col("entry_timestamp") - _DAY_START) / (_DAY_END - _DAY_START))
        .clip(0.0, 1.0).alias("time_frac"),
    )
    available = [f for f in FEATURES if f in df.columns]
    X = df.select(available).to_numpy().astype(float)
    y = df["filled"].cast(pl.Int8).to_numpy()
    nan_mask = ~np.isnan(X).any(axis=1)
    return X[nan_mask], y[nan_mask], available


def time_split_auc(df: pl.DataFrame, test_frac: float = 0.2) -> tuple[float, float, float]:
    """Logistic regression AUC with chronological train/test split + bootstrap CI."""
    X, y, _ = prep_features(df)
    n = len(X)
    split = int(n * (1 - test_frac))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = LogisticRegression(max_iter=1_000, random_state=RNG_SEED)
    model.fit(X_tr_s, y_tr)
    proba = model.predict_proba(X_te_s)[:, 1]
    point = roc_auc_score(y_te, proba)

    rng = np.random.default_rng(RNG_SEED)
    boot = [
        roc_auc_score(y_te[idx := rng.integers(0, len(y_te), len(y_te))], proba[idx])
        for _ in range(N_BOOT)
    ]
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return point, lo, hi


# ── main ──────────────────────────────────────────────────────────────────────

print("=" * 65)
print("MULTI-TICKER ANALYSIS")
print("=" * 65)

# ── Section 1: per-ticker fill rate and AUC comparison ────────────────────────
print("\n── Fill rate and AUC by ticker / depth level ──────────────────")

rows = []
for level in (1, 2):
    try:
        df = load_pooled(level)
    except FileNotFoundError as e:
        print(f"  L{level}: {e}")
        continue

    df = df.with_columns(pl.col("side").eq("bid").cast(pl.Int8).alias("side_bid"))

    for ticker in df["ticker"].unique().sort().to_list():
        sub = df.filter(pl.col("ticker") == ticker)
        fill_rate = float(sub["filled"].mean())
        n = len(sub)
        auc, lo, hi = time_split_auc(sub)
        rows.append({
            "ticker": ticker,
            "date": sub["date"][0],
            "level": level,
            "n": n,
            "fill_rate": fill_rate,
            "auc": auc,
            "auc_lo": lo,
            "auc_hi": hi,
        })
        print(
            f"  {ticker:5s} L{level}  n={n:,}  fill={fill_rate:.1%}"
            f"  AUC={auc:.3f} [{lo:.3f}, {hi:.3f}]"
        )

if not rows:
    print("  No data found. Run experiment_runner.py first.")
    sys.exit(0)

summary = pl.DataFrame(rows)

# ── Section 2: AUC gap plot (L1 vs L2 per ticker) ────────────────────────────
tickers = summary["ticker"].unique().sort().to_list()
l1 = summary.filter(pl.col("level") == 1)
l2 = summary.filter(pl.col("level") == 2)

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(tickers))
w = 0.35

for i, (ldf, label, color) in enumerate([(l1, "L1 (touch)", "#4a90d9"), (l2, "L2 (depth)", "#e07b3a")]):
    aucs = [ldf.filter(pl.col("ticker") == t)["auc"][0] if t in ldf["ticker"].to_list() else float("nan") for t in tickers]
    los  = [ldf.filter(pl.col("ticker") == t)["auc_lo"][0] if t in ldf["ticker"].to_list() else float("nan") for t in tickers]
    his  = [ldf.filter(pl.col("ticker") == t)["auc_hi"][0] if t in ldf["ticker"].to_list() else float("nan") for t in tickers]
    errs = [[a - l for a, l in zip(aucs, los)], [h - a for h, a in zip(his, aucs)]]
    ax.bar(x + i * w, aucs, w, label=label, color=color, alpha=0.85)
    ax.errorbar(x + i * w, aucs, yerr=errs, fmt="none", color="black", capsize=3, linewidth=1)

ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance")
ax.set_xticks(x + w / 2)
ax.set_xticklabels(tickers)
ax.set_ylabel("OOS AUC (logistic)")
ax.set_title("Touch vs Depth Fill Predictability — Cross-Ticker")
ax.legend()
ax.set_ylim(0.42, 0.65)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "auc_by_ticker_level.png", dpi=150)
plt.close(fig)
print(f"\n  Saved: {PLOTS_DIR}/auc_by_ticker_level.png")

# ── Section 3: granularity effect by ticker ───────────────────────────────────
print("\n── Granularity spread (K/Q) by ticker ─────────────────────────")
for level in (1, 2):
    try:
        df = load_pooled(level)
    except FileNotFoundError:
        continue
    for ticker in df["ticker"].unique().sort().to_list():
        sub = df.filter(pl.col("ticker") == ticker)
        kq  = sub["queue_granularity_at_entry"]
        print(
            f"  {ticker:5s} L{level}  median={kq.median():.4f}"
            f"  p25={kq.quantile(0.25):.4f}  p75={kq.quantile(0.75):.4f}"
            f"  IQR={kq.quantile(0.75) - kq.quantile(0.25):.4f}"
        )

# ── Section 4: logistic coefficients by ticker at L2 ─────────────────────────
print("\n── L2 logistic coefficients by ticker ─────────────────────────")
try:
    df_l2 = load_pooled(2)
    df_l2 = df_l2.with_columns(pl.col("side").eq("bid").cast(pl.Int8).alias("side_bid"))
    available_features = [f for f in FEATURES if f in df_l2.columns]

    for ticker in df_l2["ticker"].unique().sort().to_list():
        sub = df_l2.filter(pl.col("ticker") == ticker)
        X, y, feat_names = prep_features(sub)
        n = len(X)
        split = int(n * 0.8)
        scaler = StandardScaler().fit(X[:split])
        model  = LogisticRegression(max_iter=1_000, random_state=RNG_SEED)
        model.fit(scaler.transform(X[:split]), y[:split])
        print(f"\n  {ticker} (n={n:,}):")
        for feat, coef in sorted(zip(feat_names, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
            print(f"    {feat:<35s}  {coef:+.3f}")
except FileNotFoundError:
    pass

print("\n" + "=" * 65)
print("MULTI-TICKER ANALYSIS COMPLETE")
print("=" * 65)
