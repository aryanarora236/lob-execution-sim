"""
Phase 5 extension: Adverse selection analysis.

Research question
-----------------
Does observable market state at injection time predict execution quality
(adverse selection) for passive limit order fills in AAPL?

Adverse selection definition (already computed in parquet)
----------------------------------------------------------
For a passive BID that fills: adv = mid_at_fill - mid_1s_after
  Positive → mid fell after we filled → we overpaid (adversely selected)
For a passive ASK that fills: adv = mid_1s_after - mid_at_fill
  Positive → mid rose after we filled → we undersold (adversely selected)

In both cases: positive = we got picked off by informed flow.
Units: raw price ($ × 10_000). We convert to cents (/ 100).

Hypotheses tested
-----------------
H1: High book imbalance on the opposite side predicts worse adverse selection
    (positive imbalance + we're a bid → fill is against the order flow trend)
H2: Depth-2 fills have worse adverse selection than touch fills — they occur
    only after L1 is cleared, meaning a decisive price move already happened
H3: Adverse selection is worse at the open and close (Admati-Pfleiderer 1988:
    informed traders concentrate around news events)
H4: Wider spread at entry predicts worse adverse selection (compensation for
    adverse selection risk is already priced into wider spreads)

Run from project root:
    uv run python notebooks/06_adverse_selection.py
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
from scipy.stats import spearmanr, ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

PARQUET_L1 = Path("results/experiment_AAPL_2019-12-30.parquet")
PARQUET_L2 = Path("results/experiment_AAPL_2019-12-30_L2.parquet")
PLOTS_DIR  = Path("results/analysis")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 0
N_BOOT   = 4_000

rng = np.random.default_rng(RNG_SEED)

RAW_TO_CENTS = 1 / 100   # raw units → cents (1 tick = 100 raw = $0.01 = 1 cent)

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Load and filter to filled orders with adverse selection data
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 68)
print("SECTION 1 — Load data")
print("=" * 68)

def load_filled(path: Path, label: str) -> pl.DataFrame:
    df = pl.read_parquet(path)
    df = df.filter(pl.col("filled") & pl.col("adverse_selection_1s").is_not_null())
    df = df.with_columns(
        (pl.col("adverse_selection_1s") * RAW_TO_CENTS).alias("adv_cents"),
        (pl.col("entry_timestamp") / 3600).alias("hour_of_day"),
    )
    print(f"\n{label}: {len(df):,} filled orders with adverse selection data")
    return df

df1 = load_filled(PARQUET_L1, "L1 (touch)")
df2 = load_filled(PARQUET_L2, "L2 (depth)")

def bootstrap_ci(arr, stat_fn, n_boot=N_BOOT, alpha=0.05):
    boots = [stat_fn(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(boots, 100*alpha/2)), float(np.percentile(boots, 100*(1-alpha/2)))

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Descriptive statistics
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 68)
print("SECTION 2 — Descriptive statistics")
print("=" * 68)

for label, df in [("L1", df1), ("L2", df2)]:
    adv = df["adv_cents"].to_numpy()
    lo, hi = bootstrap_ci(adv, np.mean)
    pct_pos = (adv > 0).mean()
    print(f"\n{label} adverse selection (cents/share):")
    print(f"  Mean   : {adv.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  Median : {np.median(adv):+.4f}")
    print(f"  Std    : {adv.std():.4f}")
    print(f"  % positive (adversely selected): {pct_pos:.1%}")

# H2 test: is L2 adverse selection worse than L1?
adv1 = df1["adv_cents"].to_numpy()
adv2 = df2["adv_cents"].to_numpy()
t_stat, p_val = ttest_ind(adv2, adv1, alternative="greater")
print(f"\nH2 test (L2 > L1 adverse selection): t={t_stat:.3f}  p={p_val:.4f}")
lo_diff, hi_diff = bootstrap_ci(adv2 - adv2.mean() + adv1.mean(), np.mean)

# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Regression analysis
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 68)
print("SECTION 3 — Regression: what predicts adverse selection?")
print("=" * 68)

FEATURE_NAMES = [
    "imbalance",
    "spread_ticks",
    "time_frac",
    "side_bid",
    "log_queue_shares",
    "granularity",
]

results = {}

for label, df in [("L1", df1), ("L2", df2)]:
    ts     = df["entry_timestamp"].to_numpy()
    day_lo = ts.min(); day_hi = ts.max()

    X = np.column_stack([
        df["book_imbalance_at_entry"].to_numpy(),
        df["spread_at_entry_ticks"].to_numpy().astype(float),
        (ts - day_lo) / (day_hi - day_lo),
        (df["side"] == "bid").to_numpy().astype(float),
        np.log1p(df["queue_position_at_entry"].to_numpy()),
        df["queue_granularity_at_entry"].to_numpy(),
    ])
    y = df["adv_cents"].to_numpy()

    # Time-based OOS split
    cutoff     = day_lo + 0.8 * (day_hi - day_lo)
    tr         = ts <= cutoff; te = ts > cutoff
    scaler     = StandardScaler()
    X_tr_s     = scaler.fit_transform(X[tr])
    X_te_s     = scaler.transform(X[te])

    reg = LinearRegression()
    reg.fit(X_tr_s, y[tr])
    pred_te = reg.predict(X_te_s)

    r2_tr = r2_score(y[tr], reg.predict(X_tr_s))
    r2_te = r2_score(y[te], pred_te)

    # Bootstrap CIs on coefficients
    boot_coefs = np.zeros((N_BOOT, len(FEATURE_NAMES)))
    for b in range(N_BOOT):
        idx = rng.integers(0, X_tr_s.shape[0], X_tr_s.shape[0])
        reg_b = LinearRegression()
        reg_b.fit(X_tr_s[idx], y[tr][idx])
        boot_coefs[b] = reg_b.coef_

    c_lo = np.percentile(boot_coefs, 2.5,  axis=0)
    c_hi = np.percentile(boot_coefs, 97.5, axis=0)

    # Spearman correlations
    print(f"\n{label} regression  R² train={r2_tr:.4f}  R² test={r2_te:.4f}")
    print(f"  {'Feature':<22s}  {'Coef':>8s}  {'95% CI':<22s}  Sig")
    print(f"  {'-'*22}  {'-'*8}  {'-'*22}  ---")
    coef_order = np.argsort(np.abs(reg.coef_))[::-1]
    for i in coef_order:
        sig = "**" if not (c_lo[i] < 0 < c_hi[i]) else ""
        print(f"  {FEATURE_NAMES[i]:<22s}  {reg.coef_[i]:>+8.4f}"
              f"  [{c_lo[i]:>+.4f}, {c_hi[i]:>+.4f}]  {sig}")

    results[label] = dict(reg=reg, scaler=scaler, X=X, y=y,
                          c_lo=c_lo, c_hi=c_hi, coef_order=coef_order,
                          r2_tr=r2_tr, r2_te=r2_te, df=df)

# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Intraday adverse selection patterns
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 68)
print("SECTION 4 — Intraday patterns")
print("=" * 68)

# 30-minute buckets: 09:35–16:00
bucket_edges = [9.583, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16]
bucket_labels = [f"{int(h):02d}:{int((h%1)*60):02d}" for h in bucket_edges[:-1]]

for label, df in [("L1", df1), ("L2", df2)]:
    hours = df["hour_of_day"].to_numpy()
    adv   = df["adv_cents"].to_numpy()
    print(f"\n{label} — mean adverse selection by 30-min bucket (cents/share):")
    for lo_e, hi_e, lbl in zip(bucket_edges[:-1], bucket_edges[1:], bucket_labels):
        mask = (hours >= lo_e) & (hours < hi_e)
        if mask.sum() < 5:
            continue
        m = adv[mask].mean()
        lo_c, hi_c = bootstrap_ci(adv[mask], np.mean)
        bar = "+" * max(0, int(m * 20)) if m > 0 else "-" * max(0, int(-m * 20))
        print(f"  {lbl}  {m:>+7.4f}  [{lo_c:>+.4f}, {hi_c:>+.4f}]  n={mask.sum():3d}  {bar}")

# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Plots
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 68)
print("SECTION 5 — Plots")
print("=" * 68)

# ── Plot A: Distribution of adverse selection (L1 vs L2) ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
clip = 5   # cents — clip extreme values for readability
ax.hist(np.clip(adv1, -clip, clip), bins=60, density=True,
        alpha=0.6, color="steelblue", label="L1 (touch)",  edgecolor="white")
ax.hist(np.clip(adv2, -clip, clip), bins=60, density=True,
        alpha=0.6, color="tomato",    label="L2 (depth-2)", edgecolor="white")
ax.axvline(adv1.mean(), color="steelblue", linestyle="--", linewidth=1.5,
           label=f"L1 mean={adv1.mean():+.3f}¢")
ax.axvline(adv2.mean(), color="tomato",    linestyle="--", linewidth=1.5,
           label=f"L2 mean={adv2.mean():+.3f}¢")
ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
ax.set_xlabel("Adverse selection (cents/share, clipped at ±5¢)")
ax.set_ylabel("Density")
ax.set_title("Distribution of adverse selection: L1 vs L2")
ax.legend(fontsize=8)

# ── Plot B: Mean adverse selection — L1 vs L2 with CI ────────────────────────
ax = axes[1]
labels_b = ["L1\n(touch)", "L2\n(depth-2)"]
means_b  = [adv1.mean(), adv2.mean()]
lo1b, hi1b = bootstrap_ci(adv1, np.mean)
lo2b, hi2b = bootstrap_ci(adv2, np.mean)
errs_b = [[means_b[0]-lo1b, means_b[1]-lo2b],
          [hi1b-means_b[0], hi2b-means_b[1]]]
colors_b = ["steelblue", "tomato"]
ax.bar(labels_b, means_b, color=colors_b, alpha=0.8)
ax.errorbar(range(2), means_b, yerr=errs_b, fmt="none", color="black", capsize=6)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Mean adverse selection (cents/share)")
ax.set_title(f"L1 vs L2 mean adverse selection\n(p={p_val:.4f}, t-test L2>L1)")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f¢"))

plt.tight_layout()
p = PLOTS_DIR / "06_adv_distribution.png"
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"Saved: {p}")

# ── Plot C: Imbalance vs adverse selection scatter ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, label, df_plot, color in [
    (axes[0], "L1 (touch)",   df1, "steelblue"),
    (axes[1], "L2 (depth-2)", df2, "tomato"),
]:
    imb  = df_plot["book_imbalance_at_entry"].to_numpy()
    adv  = np.clip(df_plot["adv_cents"].to_numpy(), -5, 5)

    # Bin imbalance into 10 equal buckets, plot mean ± bootstrap CI
    edges = np.linspace(-1, 1, 11)
    mids  = (edges[:-1] + edges[1:]) / 2
    bin_means, bin_lo, bin_hi, bin_n = [], [], [], []
    for lo_e, hi_e in zip(edges[:-1], edges[1:]):
        mask = (imb >= lo_e) & (imb < hi_e)
        if mask.sum() < 5:
            bin_means.append(np.nan); bin_lo.append(np.nan)
            bin_hi.append(np.nan);   bin_n.append(0)
            continue
        m    = adv[mask].mean()
        l, h = bootstrap_ci(adv[mask], np.mean)
        bin_means.append(m); bin_lo.append(l); bin_hi.append(h); bin_n.append(mask.sum())

    bm   = np.array(bin_means)
    bl   = np.array(bin_lo)
    bh   = np.array(bin_hi)
    valid = ~np.isnan(bm)

    ax.scatter(imb, adv, alpha=0.03, s=5, color=color)
    ax.plot(mids[valid], bm[valid], "o-", color="black", linewidth=2,
            markersize=5, label="Bin mean")
    ax.fill_between(mids[valid], bl[valid], bh[valid], alpha=0.25, color="black")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="gray",  linewidth=0.6, linestyle=":")

    r, p_r = spearmanr(imb, df_plot["adv_cents"].to_numpy())
    ax.set_xlabel("Book imbalance at entry")
    ax.set_ylabel("Adverse selection (cents, clipped ±5¢)")
    ax.set_title(f"{label}\nSpearman r={r:+.3f}  p={p_r:.3f}")
    ax.legend(fontsize=8)

plt.tight_layout()
p = PLOTS_DIR / "07_imbalance_vs_adv.png"
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"Saved: {p}")

# ── Plot D: Intraday adverse selection by 30-min bucket ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

for ax, label, df_plot, color in [
    (axes[0], "L1 (touch)",   df1, "steelblue"),
    (axes[1], "L2 (depth-2)", df2, "tomato"),
]:
    hours = df_plot["hour_of_day"].to_numpy()
    adv   = df_plot["adv_cents"].to_numpy()
    bkt_means, bkt_lo, bkt_hi, bkt_lbl_used = [], [], [], []
    for lo_e, hi_e, lbl in zip(bucket_edges[:-1], bucket_edges[1:], bucket_labels):
        mask = (hours >= lo_e) & (hours < hi_e)
        if mask.sum() < 5:
            continue
        m = adv[mask].mean()
        l, h = bootstrap_ci(adv[mask], np.mean)
        bkt_means.append(m); bkt_lo.append(l); bkt_hi.append(h); bkt_lbl_used.append(lbl)
    bm = np.array(bkt_means)
    bl = np.array(bkt_lo); bh = np.array(bkt_hi)
    x  = range(len(bm))
    bar_colors = [("tomato" if m > 0 else "steelblue") for m in bm]
    ax.bar(x, bm, color=bar_colors, alpha=0.8)
    ax.errorbar(x, bm, yerr=[bm-bl, bh-bm], fmt="none", color="black", capsize=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x)); ax.set_xticklabels(bkt_lbl_used, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean adverse selection (cents/share)")
    ax.set_title(f"Intraday adverse selection — {label}")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f¢"))

plt.tight_layout()
p = PLOTS_DIR / "08_intraday_adv.png"
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"Saved: {p}")

# ── Plot E: Regression coefficients L1 vs L2 ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for label, ax in [("L1", axes[0]), ("L2", axes[1])]:
    r   = results[label]
    reg = r["reg"]
    lo  = r["c_lo"]; hi = r["c_hi"]
    sig = [not (lo[i] < 0 < hi[i]) for i in range(len(FEATURE_NAMES))]
    colors_r = ["tomato" if s else "lightgray" for s in sig]
    ax.barh(FEATURE_NAMES, reg.coef_, color=colors_r, alpha=0.9)
    ax.errorbar(reg.coef_, range(len(FEATURE_NAMES)),
                xerr=[reg.coef_-lo, hi-reg.coef_],
                fmt="none", color="black", capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (standardised, cents/share per SD)")
    ax.set_title(f"{label} adverse selection predictors\n"
                 f"R² test={r['r2_te']:.4f}  (red = significant)")

plt.tight_layout()
p = PLOTS_DIR / "09_adv_coefs.png"
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"Saved: {p}")

# ── Plot F: Adverse selection by spread level ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, label, df_plot, color in [
    (axes[0], "L1 (touch)",   df1, "steelblue"),
    (axes[1], "L2 (depth-2)", df2, "tomato"),
]:
    spreads = df_plot["spread_at_entry_ticks"].to_numpy()
    adv     = df_plot["adv_cents"].to_numpy()
    s_vals, s_means, s_lo, s_hi = [], [], [], []
    for s in sorted(np.unique(spreads)):
        mask = spreads == s
        if mask.sum() < 10:
            continue
        m = adv[mask].mean()
        l, h = bootstrap_ci(adv[mask], np.mean)
        s_vals.append(s); s_means.append(m); s_lo.append(l); s_hi.append(h)
    sm = np.array(s_means)
    ax.bar(s_vals, sm, color=color, alpha=0.8)
    ax.errorbar(s_vals, sm,
                yerr=[sm-np.array(s_lo), np.array(s_hi)-sm],
                fmt="none", color="black", capsize=4)
    ax.axhline(0, color="black", linewidth=0.8)

    r_s, p_s = spearmanr(spreads, adv)
    ax.set_xlabel("Spread at entry (ticks)")
    ax.set_ylabel("Mean adverse selection (cents/share)")
    ax.set_title(f"{label} — adverse selection by spread\nr={r_s:+.3f}  p={p_s:.3f}")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f¢"))

plt.tight_layout()
p = PLOTS_DIR / "10_adv_by_spread.png"
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"Saved: {p}")

print("\nAll plots saved to results/analysis/")
print("\n[adverse selection analysis complete]\n")
