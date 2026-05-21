"""
Phase 5: Queue granularity as a predictor of fill probability.

Sections
--------
1.  Load and inspect
2.  Descriptive statistics with bootstrap CIs
3.  Feature engineering and time-based train/test split   [CP2]
4.  Logistic regression                                   [CP3]
5.  LightGBM                                              [CP4]
6.  Partial dependence plots and model comparison         [CP5]

Run from project root:
    uv run python notebooks/05_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")   # headless — save to file, don't pop a window
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PARQUET   = Path("results/experiment_AAPL_2019-12-30.parquet")
PLOTS_DIR = Path("results/analysis")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED  = 0
N_BOOT    = 4_000   # bootstrap resamples for CIs

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Load and inspect
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 68)
print("SECTION 1 — Load and inspect")
print("=" * 68)

df = pl.read_parquet(PARQUET)
n  = len(df)
print(f"\nRows : {n:,}")
print(f"Cols : {df.width}")
print(f"\nSchema:\n{df.schema}")
print(f"\nStatus breakdown:\n{df['status'].value_counts().sort('count', descending=True)}")
print(f"\nSide split:\n{df['side'].value_counts()}")
print(f"\nSpread unique values (ticks): {sorted(df['spread_at_entry_ticks'].unique().to_list())}")

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Descriptive statistics with bootstrap CIs
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 68)
print("SECTION 2 — Descriptive statistics")
print("=" * 68)

rng = np.random.default_rng(RNG_SEED)


def bootstrap_ci(
    arr: np.ndarray,
    stat_fn,
    n_boot: int = N_BOOT,
    alpha: float = 0.05,
    rng: np.random.Generator = rng,
) -> tuple[float, float]:
    """Percentile bootstrap CI for stat_fn applied to arr."""
    boots = [
        stat_fn(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ]
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


filled    = df["filled"].to_numpy().astype(float)
fill_rate = filled.mean()
lo, hi    = bootstrap_ci(filled, np.mean)
print(f"\nOverall fill rate : {fill_rate:.1%}  95% CI [{lo:.1%}, {hi:.1%}]")

# Fill rate by side
for side in ("bid", "ask"):
    mask = (df["side"] == side).to_numpy()
    sr   = filled[mask].mean()
    lo2, hi2 = bootstrap_ci(filled[mask], np.mean)
    print(f"  Fill rate ({side:3s}) : {sr:.1%}  95% CI [{lo2:.1%}, {hi2:.1%}]  (n={mask.sum():,})")

# Time-to-fill stats (filled orders only)
ttf_mask = df["filled"].to_numpy()
ttf      = df["time_to_first_fill"].to_numpy()[ttf_mask]
lo3, hi3 = bootstrap_ci(ttf, np.median)
print(f"\nMedian time-to-fill: {np.median(ttf):.2f}s  95% CI [{lo3:.2f}s, {hi3:.2f}s]  (n={ttf_mask.sum():,})")
lo4, hi4 = bootstrap_ci(ttf, np.mean)
print(f"Mean  time-to-fill : {np.mean(ttf):.2f}s  95% CI [{lo4:.2f}s, {hi4:.2f}s]")

# Fill rate by spread quartile
spread     = df["spread_at_entry_ticks"].to_numpy()
q25, q75   = np.percentile(spread, [25, 75])
print(f"\nSpread distribution: min={spread.min()}  p25={q25:.0f}  med={np.median(spread):.0f}  p75={q75:.0f}  max={spread.max()}")
print("\nFill rate by spread (ticks):")
for s in sorted(np.unique(spread)):
    mask = spread == s
    if mask.sum() < 10:
        continue
    sr   = filled[mask].mean()
    lo5, hi5 = bootstrap_ci(filled[mask], np.mean)
    print(f"  {s:2d} ticks : {sr:.1%}  [{lo5:.1%}, {hi5:.1%}]  n={mask.sum():,}")

# Fill rate by granularity — value thresholds (75% of obs = exactly 0.01,
# so quartile bins are degenerate)
gran  = df["queue_granularity_at_entry"].to_numpy()
edges = np.percentile(gran, [0, 25, 50, 75, 100])
print("\nFill rate by granularity bin:")
gran_bins   = [0, 0.005, 0.01, 0.015, np.inf]
gran_labels = ["<0.005", "0.005–0.01", "0.01–0.015", ">0.015"]
for lo_e, hi_e, lbl in zip(gran_bins[:-1], gran_bins[1:], gran_labels):
    mask = (gran >= lo_e) & (gran < hi_e)
    if mask.sum() < 5:
        continue
    sr = filled[mask].mean()
    lo6, hi6 = bootstrap_ci(filled[mask], np.mean)
    print(f"  {lbl}: fill={sr:.1%}  [{lo6:.1%}, {hi6:.1%}]  n={mask.sum():,}")

# Fill rate by queue-depth quartile
qs    = df["queue_position_at_entry"].to_numpy()
qedge = np.percentile(qs, [0, 25, 50, 75, 100])
labels = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
print("\nFill rate by queue_shares quartile:")
for i, (lo_e, hi_e, lbl) in enumerate(zip(qedge[:-1], qedge[1:], labels)):
    if i == 3:
        mask = qs >= lo_e
    else:
        mask = (qs >= lo_e) & (qs < hi_e)
    if mask.sum() < 5:
        continue
    sr = filled[mask].mean()
    lo7, hi7 = bootstrap_ci(filled[mask], np.mean)
    print(f"  {lbl}: shares=[{lo_e:.0f}, {hi_e:.0f}]  fill={sr:.1%}  [{lo7:.1%}, {hi7:.1%}]  n={mask.sum():,}")

# ── descriptive plots ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# Plot 1: fill rate vs spread
spread_vals_plot, means, lows, highs = [], [], [], []
for s in sorted(np.unique(spread)):
    mask = spread == s
    if mask.sum() < 5:
        continue
    m = filled[mask].mean()
    l, h = bootstrap_ci(filled[mask], np.mean)
    spread_vals_plot.append(s); means.append(m); lows.append(l); highs.append(h)
ax = axes[0]
ax.bar(spread_vals_plot, means, color="steelblue", alpha=0.8)
ax.errorbar(spread_vals_plot, means,
            yerr=[np.array(means)-np.array(lows), np.array(highs)-np.array(means)],
            fmt="none", color="black", capsize=4)
ax.set_xlabel("Spread at entry (ticks)")
ax.set_ylabel("Fill rate within 60s")
ax.set_title("Fill rate by spread")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

# Plot 2: fill rate vs granularity bins
qmeans, qlows, qhighs, qlbl_used = [], [], [], []
for lo_e, hi_e, lbl in zip(gran_bins[:-1], gran_bins[1:], gran_labels):
    mask = (gran >= lo_e) & (gran < hi_e)
    if mask.sum() < 5:
        continue
    m = filled[mask].mean()
    l, h = bootstrap_ci(filled[mask], np.mean)
    qmeans.append(m); qlows.append(l); qhighs.append(h); qlbl_used.append(lbl)
ax = axes[1]
ax.bar(range(len(qmeans)), qmeans, color="tomato", alpha=0.8, tick_label=qlbl_used)
ax.errorbar(range(len(qmeans)), qmeans,
            yerr=[np.array(qmeans)-np.array(qlows), np.array(qhighs)-np.array(qmeans)],
            fmt="none", color="black", capsize=4)
ax.set_xlabel("Granularity (K/Q) bin")
ax.set_ylabel("Fill rate within 60s")
ax.set_title("Fill rate by queue granularity")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

# Plot 3: granularity distribution
ax = axes[2]
ax.hist(gran, bins=40, color="mediumpurple", alpha=0.8, edgecolor="white")
ax.set_xlabel("Queue granularity (K/Q)")
ax.set_ylabel("Count")
ax.set_title("Granularity distribution")

plt.tight_layout()
p = PLOTS_DIR / "01_descriptive.png"
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"\nPlot saved: {p}")

print("\n[CP1 complete]\n")
