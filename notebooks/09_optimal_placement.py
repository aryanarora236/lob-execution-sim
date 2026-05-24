"""
Phase 5 extension: Optimal passive placement — L1 vs L2.

Framework
---------
You need to buy (or sell) 100 shares passively. You choose to post at L1
(best price) or L2 (one tick behind best). If the order doesn't fill within
60 s you're forced to take via market order.

Expected implementation shortfall per share (cents, lower = better):

    E[IS | L1] = p1 × (−S + AS1) + (1 − p1) × C
    E[IS | L2] = p2 × (−S − 1 + AS2) + (1 − p2) × C

where
    p1, p2  = fill probability at L1, L2
    S       = half-spread in cents (price improvement from posting passively)
    AS1, AS2= mean adverse selection in cents (mid-price move against us 1 s after fill)
    C       = unfill penalty in cents (cost of the market-order fallback)
    −1      = extra 1-cent price improvement from posting one tick deeper at L2

Positive IS = money left on the table. Negative IS = better than mid.
L2 is preferred when E[IS | L2] < E[IS | L1].

Research questions
------------------
Q1. What unfill penalty makes L1 and L2 indifferent at average conditions?
Q2. Does spread level shift the crossover?
Q3. How do imbalance and time-of-day interact with the optimal choice?

Run from project root:
    uv run python notebooks/09_optimal_placement.py
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
import matplotlib.ticker as mticker
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PARQUET_L1 = Path("results/experiment_AAPL_2019-12-30_L1.parquet")
PARQUET_L2 = Path("results/experiment_AAPL_2019-12-30_L2.parquet")
PLOTS_DIR  = Path("results/analysis")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TICK = 1.0          # 1 tick = 1 cent for AAPL
RAW_TO_CENTS = 0.01 # spread_at_entry_ticks is already in ticks; adv already in cents

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Load and compute per-order IS at different unfill penalties
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 68)
print("SECTION 1 — Load data and compute per-order IS")
print("=" * 68)

df1 = pl.read_parquet(PARQUET_L1).with_columns(
    (pl.col("adverse_selection_1s") * RAW_TO_CENTS).alias("adv_cents")
)
df2 = pl.read_parquet(PARQUET_L2).with_columns(
    (pl.col("adverse_selection_1s") * RAW_TO_CENTS).alias("adv_cents")
)

# Half-spread in cents (spread_at_entry_ticks × 0.5 × 1¢/tick)
df1 = df1.with_columns((pl.col("spread_at_entry_ticks") * 0.5).alias("half_spread"))
df2 = df2.with_columns((pl.col("spread_at_entry_ticks") * 0.5).alias("half_spread"))

# Overall empirical stats
p1_bar   = df1["filled"].mean()
p2_bar   = df2["filled"].mean()
as1_bar  = df1.filter(pl.col("filled") & pl.col("adv_cents").is_not_null())["adv_cents"].mean()
as2_bar  = df2.filter(pl.col("filled") & pl.col("adv_cents").is_not_null())["adv_cents"].mean()
s_bar    = df1["half_spread"].mean()   # average half-spread (both levels same spread)

print(f"\nOverall stats:")
print(f"  Fill rate   L1={p1_bar:.3f}   L2={p2_bar:.3f}")
print(f"  Adv sel     L1={as1_bar:+.3f}¢  L2={as2_bar:+.3f}¢")
print(f"  Half-spread mean={s_bar:.3f}¢")

# Crossover unfill penalty at mean conditions:
# E[IS|L1] = E[IS|L2]
# p1(-S+AS1) + (1-p1)C = p2(-S-1+AS2) + (1-p2)C
# (1-p1)C - (1-p2)C = p2(-S-1+AS2) - p1(-S+AS1)
# C(p2-p1) = p2(-S-1+AS2) - p1(-S+AS1)
lhs_l1 = p1_bar * (-s_bar + as1_bar)
lhs_l2 = p2_bar * (-s_bar - TICK + as2_bar)
C_cross = (lhs_l2 - lhs_l1) / (p2_bar - p1_bar)   # crossover penalty
print(f"\nCrossover unfill penalty at mean conditions: C* = {C_cross:.2f}¢/share")
print(f"  (Below C*: prefer L2 — extra tick improvement offsets lower fill rate)")
print(f"  (Above C*: prefer L1 — not filling is too expensive to risk)")

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — IS as a function of unfill penalty
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 68)
print("SECTION 2 — IS vs unfill penalty (sensitivity analysis)")
print("=" * 68)

penalties = np.linspace(0, 15, 300)

is_l1 = p1_bar * (-s_bar + as1_bar) + (1 - p1_bar) * penalties
is_l2 = p2_bar * (-s_bar - TICK + as2_bar) + (1 - p2_bar) * penalties

# By spread level
for label_spread, spread_lo, spread_hi in [("1-tick spread", 0.9, 1.1),
                                            ("2-tick spread", 1.9, 2.1),
                                            ("3-tick spread", 2.9, 3.1)]:
    sub1 = df1.filter((pl.col("spread_at_entry_ticks") >= spread_lo) &
                      (pl.col("spread_at_entry_ticks") <= spread_hi))
    sub2 = df2.filter((pl.col("spread_at_entry_ticks") >= spread_lo) &
                      (pl.col("spread_at_entry_ticks") <= spread_hi))
    if len(sub1) < 20 or len(sub2) < 20:
        continue
    p1_s  = sub1["filled"].mean()
    p2_s  = sub2["filled"].mean()
    as1_s = sub1.filter(pl.col("filled") & pl.col("adv_cents").is_not_null())["adv_cents"].mean()
    as2_s = sub2.filter(pl.col("filled") & pl.col("adv_cents").is_not_null())["adv_cents"].mean()
    s_s   = sub1["half_spread"].mean()
    if abs(p1_s - p2_s) < 1e-6:
        continue
    C_s = (p2_s * (-s_s - TICK + as2_s) - p1_s * (-s_s + as1_s)) / (p2_s - p1_s)
    print(f"  {label_spread}: p1={p1_s:.3f} p2={p2_s:.3f}  C*={C_s:.2f}¢  "
          f"n1={len(sub1):,} n2={len(sub2):,}")

# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Conditional IS using logistic model predictions
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 68)
print("SECTION 3 — Conditional optimal placement (logistic predictions)")
print("=" * 68)

FEATURES = ["spread_ticks", "imbalance", "side_bid", "time_frac",
            "log_queue_shares", "granularity"]

def build_X(df: pl.DataFrame) -> np.ndarray:
    ts = df["entry_timestamp"].to_numpy()
    day_lo, day_hi = ts.min(), ts.max()
    return np.column_stack([
        df["spread_at_entry_ticks"].to_numpy().astype(float),
        df["book_imbalance_at_entry"].to_numpy(),
        (df["side"] == "bid").to_numpy().astype(float),
        (ts - day_lo) / (day_hi - day_lo),
        np.log1p(df["queue_position_at_entry"].to_numpy()),
        df["queue_granularity_at_entry"].to_numpy(),
    ])

# Fit logistic models on full dataset (we want predictions, not OOS AUC here)
X1 = build_X(df1); y1 = df1["filled"].to_numpy().astype(int)
X2 = build_X(df2); y2 = df2["filled"].to_numpy().astype(int)

sc1 = StandardScaler(); sc2 = StandardScaler()
clf1 = LogisticRegression(max_iter=1000); clf1.fit(sc1.fit_transform(X1), y1)
clf2 = LogisticRegression(max_iter=1000); clf2.fit(sc2.fit_transform(X2), y2)

# Build a prediction grid: spread (1–3 ticks) × imbalance (−1 to 1)
# Fix side=bid, time_frac=0.5, log_queue=log1p(200), granularity=0.01
GRID_N = 60
spread_vals  = np.linspace(1, 3, GRID_N)
imbalance_vals = np.linspace(-0.8, 0.8, GRID_N)
SS, II = np.meshgrid(spread_vals, imbalance_vals)

# Reference conditions
ref_side = 0.5          # 50/50 bid/ask
ref_time = 0.5          # midday
ref_logq = np.log1p(200)
ref_gran = 0.01

def make_grid_X(spread_2d, imb_2d, scaler):
    n = spread_2d.size
    X = np.column_stack([
        spread_2d.ravel(),
        imb_2d.ravel(),
        np.full(n, ref_side),
        np.full(n, ref_time),
        np.full(n, ref_logq),
        np.full(n, ref_gran),
    ])
    return scaler.transform(X)

p1_grid = clf1.predict_proba(make_grid_X(SS, II, sc1))[:, 1].reshape(GRID_N, GRID_N)
p2_grid = clf2.predict_proba(make_grid_X(SS, II, sc2))[:, 1].reshape(GRID_N, GRID_N)

# Half-spread on the grid (cents)
hs_grid = SS * 0.5

# IS at each grid point for a range of penalties
for C in [2.0, 5.0, 10.0]:
    is1 = p1_grid * (-hs_grid + as1_bar) + (1 - p1_grid) * C
    is2 = p2_grid * (-hs_grid - TICK + as2_bar) + (1 - p2_grid) * C
    pct_prefer_l2 = (is2 < is1).mean() * 100
    print(f"  Unfill penalty={C:.0f}¢: {pct_prefer_l2:.1f}% of grid favours L2")

# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Plots
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 68)
print("SECTION 4 — Plots")
print("=" * 68)

# ── Plot A: IS vs unfill penalty (mean conditions) ────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(penalties, is_l1, color="steelblue", linewidth=2, label="L1 (touch)")
ax.plot(penalties, is_l2, color="tomato",    linewidth=2, label="L2 (depth)")
ax.axvline(C_cross, color="black", linestyle="--", linewidth=1.2,
           label=f"Indifference at C={C_cross:.1f}¢")
ax.axhline(0, color="gray", linewidth=0.8)
ax.fill_between(penalties, is_l1, is_l2,
                where=(is_l2 < is_l1), alpha=0.12, color="tomato",  label="L2 better")
ax.fill_between(penalties, is_l1, is_l2,
                where=(is_l1 < is_l2), alpha=0.12, color="steelblue", label="L1 better")
ax.set_xlabel("Unfill penalty C (¢/share) — cost of fallback market order")
ax.set_ylabel("Expected implementation shortfall (¢/share)")
ax.set_title("Optimal passive placement: L1 vs L2\n"
             f"(mean conditions: p1={p1_bar:.2f}, p2={p2_bar:.2f}, "
             f"AS1={as1_bar:.2f}¢, AS2={as2_bar:.2f}¢, S={s_bar:.2f}¢)")
ax.legend(fontsize=9)
ax.set_xlim(0, 15)

plt.tight_layout()
p = PLOTS_DIR / "19_is_vs_penalty.png"
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"Saved: {p}")

# ── Plot B: IS vs penalty by spread level ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

spread_configs = [
    ("1-tick spread", 0.9, 1.1, "steelblue", "tomato"),
    ("2-tick spread", 1.9, 2.1, "cornflowerblue", "salmon"),
    ("3-tick spread", 2.9, 3.1, "lightsteelblue", "lightsalmon"),
]
for label_spread, spread_lo, spread_hi, c1, c2 in spread_configs:
    sub1 = df1.filter((pl.col("spread_at_entry_ticks") >= spread_lo) &
                      (pl.col("spread_at_entry_ticks") <= spread_hi))
    sub2 = df2.filter((pl.col("spread_at_entry_ticks") >= spread_lo) &
                      (pl.col("spread_at_entry_ticks") <= spread_hi))
    if len(sub1) < 20 or len(sub2) < 20:
        continue
    p1_s  = sub1["filled"].mean()
    p2_s  = sub2["filled"].mean()
    as1_s = sub1.filter(pl.col("filled") & pl.col("adv_cents").is_not_null())["adv_cents"].mean()
    as2_s = sub2.filter(pl.col("filled") & pl.col("adv_cents").is_not_null())["adv_cents"].mean()
    s_s   = sub1["half_spread"].mean()
    is1_s = p1_s * (-s_s + as1_s) + (1 - p1_s) * penalties
    is2_s = p2_s * (-s_s - TICK + as2_s) + (1 - p2_s) * penalties
    ax.plot(penalties, is1_s, color=c1, linewidth=2,   label=f"L1 {label_spread}")
    ax.plot(penalties, is2_s, color=c2, linewidth=2, linestyle="--",
            label=f"L2 {label_spread}")

ax.axhline(0, color="gray", linewidth=0.8)
ax.set_xlabel("Unfill penalty C (¢/share)")
ax.set_ylabel("Expected implementation shortfall (¢/share)")
ax.set_title("IS vs unfill penalty by spread level\n(solid=L1, dashed=L2)")
ax.legend(fontsize=8)
ax.set_xlim(0, 15)

plt.tight_layout()
p = PLOTS_DIR / "20_is_by_spread.png"
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"Saved: {p}")

# ── Plot C: Optimal placement heatmap (spread × imbalance) ───────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, C in zip(axes, [2.0, 5.0, 10.0]):
    is1 = p1_grid * (-hs_grid + as1_bar) + (1 - p1_grid) * C
    is2 = p2_grid * (-hs_grid - TICK + as2_bar) + (1 - p2_grid) * C
    advantage = is1 - is2   # positive → L2 better (lower IS)

    im = ax.contourf(spread_vals, imbalance_vals, advantage,
                     levels=20, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    ax.contour(spread_vals, imbalance_vals, advantage,
               levels=[0], colors="black", linewidths=1.5)
    plt.colorbar(im, ax=ax, label="IS(L1) − IS(L2)  [¢/share]")
    ax.set_xlabel("Spread at entry (ticks)")
    ax.set_ylabel("Book imbalance")
    ax.set_title(f"Unfill penalty = {C:.0f}¢\n"
                 f"Blue = L2 better, Red = L1 better\n"
                 f"(black line = indifference)")

plt.suptitle("Optimal placement region: spread × imbalance\n"
             "(midday, 50/50 bid/ask, 200-share queue)", y=1.01, fontsize=11)
plt.tight_layout()
p = PLOTS_DIR / "21_placement_heatmap.png"
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"Saved: {p}")

# ── Plot D: IS advantage of L2 as a function of time of day ──────────────────
fig, ax = plt.subplots(figsize=(9, 5))

time_vals = np.linspace(0, 1, 100)
# Fix spread=1 tick, imbalance=0, side=0.5, granularity=0.01
ref_spread = np.full(100, 1.0)
ref_imb    = np.full(100, 0.0)
ref_hs     = ref_spread * 0.5

def make_time_X(time_arr, spread_arr, imb_arr, scaler):
    X = np.column_stack([
        spread_arr, imb_arr,
        np.full(len(time_arr), ref_side),
        time_arr,
        np.full(len(time_arr), ref_logq),
        np.full(len(time_arr), ref_gran),
    ])
    return scaler.transform(X)

p1_time = clf1.predict_proba(make_time_X(time_vals, ref_spread, ref_imb, sc1))[:, 1]
p2_time = clf2.predict_proba(make_time_X(time_vals, ref_spread, ref_imb, sc2))[:, 1]

for C, color in [(2.0, "steelblue"), (5.0, "black"), (10.0, "tomato")]:
    is1_t = p1_time * (-ref_hs + as1_bar) + (1 - p1_time) * C
    is2_t = p2_time * (-ref_hs - TICK + as2_bar) + (1 - p2_time) * C
    advantage_t = is1_t - is2_t
    hours = 9.5 + time_vals * 6.5   # map [0,1] → [09:30, 16:00]
    ax.plot(hours, advantage_t, color=color, linewidth=2, label=f"C={C:.0f}¢")

ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_xlabel("Time of day")
ax.set_ylabel("IS(L1) − IS(L2)  (¢/share)\npositive = L2 better")
ax.set_title("L2 advantage over L1 through the trading day\n"
             "(1-tick spread, zero imbalance, 50/50 side)")
ax.set_xticks([9.5, 10, 11, 12, 13, 14, 15, 16])
ax.set_xticklabels(["09:30","10:00","11:00","12:00","13:00","14:00","15:00","16:00"])
ax.legend()

plt.tight_layout()
p = PLOTS_DIR / "22_l2_advantage_by_time.png"
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"Saved: {p}")

print("\nAll plots saved to results/analysis/")
print("\n[Optimal placement analysis complete]")
