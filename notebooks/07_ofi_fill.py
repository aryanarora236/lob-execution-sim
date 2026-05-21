"""
Phase 5 extension: Order Flow Imbalance as a fill-probability predictor.

Research question
-----------------
Does OFI (net signed order-book pressure in the 10 s / 30 s before injection)
improve out-of-sample fill-probability prediction at L1 and L2?

Hypotheses
----------
H1: OFI_10s is a significant predictor of fill probability (logistic regression
    bootstrap CI excludes 0) — OFI captures short-term queue drain speed.

H2: OFI sign × level interaction — for a passive ask (bid) at L1, positive
    (negative) OFI means the ask (bid) side is being consumed, raising fill
    probability; the relationship may flip or weaken at L2.

H3: Adding OFI improves OOS AUC at both levels relative to the base model
    (spread, imbalance, side, time_frac, log_queue_shares, granularity).

Run from project root:
    uv run python notebooks/07_ofi_fill.py
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
from sklearn.metrics import roc_auc_score

PARQUET_L1 = Path("results/experiment_AAPL_2019-12-30_L1.parquet")
PARQUET_L2 = Path("results/experiment_AAPL_2019-12-30_L2.parquet")
PLOTS_DIR  = Path("results/analysis")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 0
N_BOOT   = 4_000
rng = np.random.default_rng(RNG_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Load
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 68)
print("SECTION 1 — Load")
print("=" * 68)

df1 = pl.read_parquet(PARQUET_L1)
df2 = pl.read_parquet(PARQUET_L2)

print(f"L1: {len(df1):,} orders  fill rate {df1['filled'].mean():.1%}")
print(f"L2: {len(df2):,} orders  fill rate {df2['filled'].mean():.1%}")
print(f"OFI columns present: {[c for c in df1.columns if 'ofi' in c]}")

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — OFI descriptives & sign check
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 68)
print("SECTION 2 — OFI descriptives")
print("=" * 68)

for label, df in [("L1", df1), ("L2", df2)]:
    filled   = df.filter(pl.col("filled"))["ofi_10s"].to_numpy()
    unfilled = df.filter(~pl.col("filled"))["ofi_10s"].to_numpy()
    print(f"\n{label} OFI_10s:")
    print(f"  Overall   mean={df['ofi_10s'].mean():>+9.1f}  std={df['ofi_10s'].std():>8.1f}")
    print(f"  Filled    mean={filled.mean():>+9.1f}  n={len(filled):,}")
    print(f"  Unfilled  mean={unfilled.mean():>+9.1f}  n={len(unfilled):,}")
    diff = filled.mean() - unfilled.mean()
    print(f"  Δ (filled − unfilled) = {diff:>+.1f} shares")

# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Logistic regression: base vs OFI-augmented
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 68)
print("SECTION 3 — Logistic regression: base vs OFI-augmented")
print("=" * 68)

BASE_FEATURES = ["spread_ticks", "imbalance", "side_bid", "time_frac",
                 "log_queue_shares", "granularity"]
OFI_FEATURES  = BASE_FEATURES + ["ofi_10s", "ofi_30s"]

def build_X(df: pl.DataFrame, feature_names: list[str]) -> np.ndarray:
    ts     = df["entry_timestamp"].to_numpy()
    day_lo = ts.min(); day_hi = ts.max()
    cols = {
        "spread_ticks":    df["spread_at_entry_ticks"].to_numpy().astype(float),
        "imbalance":       df["book_imbalance_at_entry"].to_numpy(),
        "side_bid":        (df["side"] == "bid").to_numpy().astype(float),
        "time_frac":       (ts - day_lo) / (day_hi - day_lo),
        "log_queue_shares":np.log1p(df["queue_position_at_entry"].to_numpy()),
        "granularity":     df["queue_granularity_at_entry"].to_numpy(),
        "ofi_10s":         df["ofi_10s"].to_numpy(),
        "ofi_30s":         df["ofi_30s"].to_numpy(),
    }
    return np.column_stack([cols[f] for f in feature_names])


def bootstrap_auc(y: np.ndarray, proba: np.ndarray, n: int = N_BOOT) -> tuple[float, float]:
    aucs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() in (0, len(idx)):
            aucs[i] = 0.5
            continue
        aucs[i] = roc_auc_score(y[idx], proba[idx])
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


results = {}

for label, df in [("L1", df1), ("L2", df2)]:
    ts     = df["entry_timestamp"].to_numpy()
    cutoff = ts.min() + 0.8 * (ts.max() - ts.min())
    tr     = ts <= cutoff
    te     = ts > cutoff
    y      = df["filled"].to_numpy().astype(int)

    row = {}
    for model_name, feats in [("base", BASE_FEATURES), ("ofi", OFI_FEATURES)]:
        X       = build_X(df, feats)
        scaler  = StandardScaler()
        X_tr_s  = scaler.fit_transform(X[tr])
        X_te_s  = scaler.transform(X[te])

        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X_tr_s, y[tr])

        proba_te = clf.predict_proba(X_te_s)[:, 1]
        auc_te   = roc_auc_score(y[te], proba_te)
        lo, hi   = bootstrap_auc(y[te], proba_te)

        print(f"\n{label} {model_name:4s}  OOS AUC={auc_te:.4f}  95% CI [{lo:.3f}, {hi:.3f}]")

        # Bootstrap CIs on coefficients
        boot_coefs = np.zeros((N_BOOT, len(feats)))
        for b in range(N_BOOT):
            idx = rng.integers(0, X_tr_s.shape[0], X_tr_s.shape[0])
            clf_b = LogisticRegression(max_iter=500, solver="lbfgs")
            clf_b.fit(X_tr_s[idx], y[tr][idx])
            boot_coefs[b] = clf_b.coef_[0]
        c_lo = np.percentile(boot_coefs, 2.5,  axis=0)
        c_hi = np.percentile(boot_coefs, 97.5, axis=0)

        if model_name == "ofi":
            print(f"  {'Feature':<20s}  {'Coef':>8s}  {'95% CI':<22s}  Sig")
            print(f"  {'-'*20}  {'-'*8}  {'-'*22}  ---")
            order = np.argsort(np.abs(clf.coef_[0]))[::-1]
            for i in order:
                sig = "**" if not (c_lo[i] < 0 < c_hi[i]) else ""
                print(f"  {feats[i]:<20s}  {clf.coef_[0][i]:>+8.4f}"
                      f"  [{c_lo[i]:>+.4f}, {c_hi[i]:>+.4f}]  {sig}")

        row[model_name] = dict(
            clf=clf, scaler=scaler, feats=feats,
            auc=auc_te, lo=lo, hi=hi,
            coef=clf.coef_[0], c_lo=c_lo, c_hi=c_hi,
        )

    results[label] = row

# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — OFI × side interaction
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 68)
print("SECTION 4 — OFI × side: does sign matter differently for bids vs asks?")
print("=" * 68)

for label, df in [("L1", df1), ("L2", df2)]:
    for side in ("bid", "ask"):
        sub = df.filter(pl.col("side") == side)
        ofi = sub["ofi_10s"].to_numpy()
        filled = sub["filled"].to_numpy().astype(int)

        # Bin OFI into quintiles and compute fill rate per bin
        edges = np.percentile(ofi, [0, 20, 40, 60, 80, 100])
        bin_means, bin_fills = [], []
        for lo_e, hi_e in zip(edges[:-1], edges[1:]):
            mask = (ofi >= lo_e) & (ofi <= hi_e)
            if mask.sum() < 5:
                continue
            bin_means.append(ofi[mask].mean())
            bin_fills.append(filled[mask].mean())

        direction = "pos" if bin_fills[-1] > bin_fills[0] else "neg"
        print(f"  {label} {side:3s}: fill rate low→high OFI quintile "
              f"{bin_fills[0]:.3f}→{bin_fills[-1]:.3f}  ({direction} slope)")

# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Plots
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 68)
print("SECTION 5 — Plots")
print("=" * 68)

CLIP = 5_000   # clip extreme OFI for readability in plots

# ── Plot A: AUC comparison — base vs OFI-augmented, L1 and L2 ────────────────
fig, ax = plt.subplots(figsize=(8, 4))

x_pos  = np.array([0, 1, 3, 4])
labels_bar = ["L1 base", "L1 + OFI", "L2 base", "L2 + OFI"]
aucs   = [
    results["L1"]["base"]["auc"], results["L1"]["ofi"]["auc"],
    results["L2"]["base"]["auc"], results["L2"]["ofi"]["auc"],
]
los    = [
    results["L1"]["base"]["lo"], results["L1"]["ofi"]["lo"],
    results["L2"]["base"]["lo"], results["L2"]["ofi"]["lo"],
]
his    = [
    results["L1"]["base"]["hi"], results["L1"]["ofi"]["hi"],
    results["L2"]["base"]["hi"], results["L2"]["ofi"]["hi"],
]
colors_bar = ["steelblue", "navy", "tomato", "darkred"]

bars = ax.bar(x_pos, aucs, color=colors_bar, alpha=0.8, width=0.7)
ax.errorbar(x_pos, aucs,
            yerr=[np.array(aucs) - np.array(los),
                  np.array(his) - np.array(aucs)],
            fmt="none", color="black", capsize=5)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="random (0.5)")
ax.set_xticks(x_pos)
ax.set_xticklabels(labels_bar)
ax.set_ylabel("OOS AUC")
ax.set_title("Fill probability AUC: base features vs OFI-augmented\n"
             "(error bars = bootstrap 95% CI)")
ax.set_ylim(0.45, 0.65)
ax.legend()

for bar, auc in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width() / 2, auc + 0.003,
            f"{auc:.3f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
p = PLOTS_DIR / "11_ofi_auc_comparison.png"
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"Saved: {p}")

# ── Plot B: OFI distribution by fill outcome ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, label, df in [(axes[0], "L1 (touch)", df1), (axes[1], "L2 (depth-2)", df2)]:
    filled_ofi   = np.clip(df.filter(pl.col("filled"))["ofi_10s"].to_numpy(), -CLIP, CLIP)
    unfilled_ofi = np.clip(df.filter(~pl.col("filled"))["ofi_10s"].to_numpy(), -CLIP, CLIP)

    ax.hist(unfilled_ofi, bins=50, density=True, alpha=0.5, color="gray",
            label=f"Unfilled (mean={unfilled_ofi.mean():>+.0f})")
    ax.hist(filled_ofi,   bins=50, density=True, alpha=0.5, color="steelblue",
            label=f"Filled   (mean={filled_ofi.mean():>+.0f})")
    ax.axvline(filled_ofi.mean(),   color="steelblue", linestyle="--", linewidth=1.5)
    ax.axvline(unfilled_ofi.mean(), color="gray",      linestyle="--", linewidth=1.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"OFI_10s (shares, clipped ±{CLIP:,})")
    ax.set_ylabel("Density")
    ax.set_title(f"{label}: OFI by fill outcome")
    ax.legend(fontsize=8)

plt.tight_layout()
p = PLOTS_DIR / "12_ofi_by_fill.png"
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"Saved: {p}")

# ── Plot C: Fill rate by OFI quintile, bids vs asks ──────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for row_i, (label, df) in enumerate([("L1 (touch)", df1), ("L2 (depth-2)", df2)]):
    for col_i, side in enumerate(("bid", "ask")):
        ax  = axes[row_i][col_i]
        sub = df.filter(pl.col("side") == side)
        ofi = sub["ofi_10s"].to_numpy()
        fil = sub["filled"].to_numpy().astype(float)

        edges = np.percentile(ofi, np.linspace(0, 100, 11))   # deciles
        bin_mid, bin_fill, bin_n = [], [], []
        for lo_e, hi_e in zip(edges[:-1], edges[1:]):
            mask = (ofi >= lo_e) & (ofi <= hi_e)
            if mask.sum() < 5:
                continue
            bin_mid.append(ofi[mask].mean())
            bin_fill.append(fil[mask].mean())
            bin_n.append(mask.sum())

        color = "steelblue" if side == "bid" else "tomato"
        ax.plot(bin_mid, bin_fill, "o-", color=color, linewidth=1.5, markersize=5)
        ax.axhline(fil.mean(), color="gray", linestyle="--", linewidth=1,
                   label=f"Overall mean={fil.mean():.3f}")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Mean OFI_10s (shares) per decile")
        ax.set_ylabel("Fill rate")
        ax.set_title(f"{label} — {side} orders")
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

plt.tight_layout()
p = PLOTS_DIR / "13_ofi_fill_by_side.png"
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"Saved: {p}")

# ── Plot D: Coefficient plot for OFI features ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, label in [(axes[0], "L1"), (axes[1], "L2")]:
    r     = results[label]["ofi"]
    feats = r["feats"]
    coefs = r["coef"]
    lo    = r["c_lo"]
    hi    = r["c_hi"]
    sig   = [not (lo[i] < 0 < hi[i]) for i in range(len(feats))]

    order = np.argsort(coefs)
    feats_o = [feats[i] for i in order]
    coefs_o = coefs[order]
    lo_o    = lo[order]
    hi_o    = hi[order]
    sig_o   = [sig[i] for i in order]
    colors_c = ["tomato" if s else "lightgray" for s in sig_o]

    y_pos = np.arange(len(feats_o))
    ax.barh(y_pos, coefs_o, color=colors_c, alpha=0.9)
    ax.errorbar(coefs_o, y_pos,
                xerr=[coefs_o - lo_o, hi_o - coefs_o],
                fmt="none", color="black", capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats_o)
    ax.set_xlabel("Logistic coef (standardised)")
    ax.set_title(f"{label} — OFI-augmented model\n"
                 f"OOS AUC={r['auc']:.4f}  (red = 95% CI excludes 0)")

plt.tight_layout()
p = PLOTS_DIR / "14_ofi_coefs.png"
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"Saved: {p}")

print("\nAll plots saved to results/analysis/")
print("\n[OFI fill-probability analysis complete]")
