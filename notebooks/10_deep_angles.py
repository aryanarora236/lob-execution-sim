"""
Two deeper analyses:

A. Fill speed → adverse selection
   Does a faster fill predict worse execution quality?
   Intuition: orders filled in seconds were hit by urgent, informed flow;
   orders filled in ~60s were hit by patient, uninformed flow.

B. Time-of-day decomposition of LightGBM's AUC edge
   LightGBM OOS AUC is 0.548 vs logistic 0.491 (+0.057). time_frac is
   the #1 feature. How much AUC survives when time_frac is removed? If the
   gap collapses, the tree was learning a structural temporal pattern,
   not a genuine book-state signal.

Run from project root:
    uv run python notebooks/10_deep_angles.py
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
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

PARQUET_L1 = Path("results/experiment_AAPL_2019-12-30_L1.parquet")
PARQUET_L2 = Path("results/experiment_AAPL_2019-12-30_L2.parquet")
PLOTS_DIR  = Path("results/analysis")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RAW_TO_CENTS = 0.01
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

df1 = pl.read_parquet(PARQUET_L1)
df2 = pl.read_parquet(PARQUET_L2)

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS A — Fill speed → adverse selection
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 68)
print("ANALYSIS A — Does faster fill predict worse adverse selection?")
print("=" * 68)

def filled_with_adv(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(pl.col("filled") & pl.col("adverse_selection_1s").is_not_null())
          .with_columns((pl.col("adverse_selection_1s") * RAW_TO_CENTS).alias("adv_cents"))
    )

f1 = filled_with_adv(df1)
f2 = filled_with_adv(df2)

for label, f in [("L1", f1), ("L2", f2)]:
    ttf = f["time_to_first_fill"].to_numpy()
    adv = f["adv_cents"].to_numpy()

    rho, p = spearmanr(ttf, adv)
    print(f"\n{label}  n={len(f):,}  Spearman ρ={rho:+.4f}  p={p:.4f}")

    # OLS: adv_cents ~ time_to_first_fill (controlled for side, spread, time_frac)
    ts     = f["entry_timestamp"].to_numpy()
    day_lo, day_hi = ts.min(), ts.max()
    X = np.column_stack([
        (ttf - ttf.mean()) / ttf.std(),
        f["spread_at_entry_ticks"].to_numpy().astype(float),
        (f["side"] == "bid").to_numpy().astype(float),
        (ts - day_lo) / (day_hi - day_lo),
    ])
    X_s = StandardScaler().fit_transform(X)
    reg = LinearRegression().fit(X_s, adv)
    names = ["time_to_first_fill", "spread_ticks", "side_bid", "time_frac"]
    print(f"  OLS (controlled)  R²={reg.score(X_s, adv):.4f}")
    for n, c in zip(names, reg.coef_):
        print(f"    {n:<22s}  {c:>+.4f}¢")

    # Decile bins
    deciles = np.percentile(ttf, np.linspace(0, 100, 11))
    bin_mid, bin_mean, bin_n = [], [], []
    for lo, hi in zip(deciles[:-1], deciles[1:]):
        mask = (ttf >= lo) & (ttf <= hi)
        if mask.sum() < 5:
            continue
        bin_mid.append(ttf[mask].mean())
        bin_mean.append(adv[mask].mean())
        bin_n.append(mask.sum())
    print(f"  Fill-time deciles: slowest→fastest adv "
          f"{bin_mean[-1]:+.3f}¢ → {bin_mean[0]:+.3f}¢")

print()

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS B — Time-of-day decomposition of LightGBM AUC
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 68)
print("ANALYSIS B — How much AUC survives without time_frac?")
print("=" * 68)

ALL_FEATURES  = ["spread_ticks", "imbalance", "side_bid", "time_frac",
                 "log_queue_shares", "granularity"]
NO_TIME_FEATURES = [f for f in ALL_FEATURES if f != "time_frac"]

def build_X(df: pl.DataFrame, features: list[str]) -> np.ndarray:
    ts = df["entry_timestamp"].to_numpy()
    day_lo, day_hi = ts.min(), ts.max()
    col_map = {
        "spread_ticks":    df["spread_at_entry_ticks"].to_numpy().astype(float),
        "imbalance":       df["book_imbalance_at_entry"].to_numpy(),
        "side_bid":        (df["side"] == "bid").to_numpy().astype(float),
        "time_frac":       (ts - day_lo) / (day_hi - day_lo),
        "log_queue_shares":np.log1p(df["queue_position_at_entry"].to_numpy()),
        "granularity":     df["queue_granularity_at_entry"].to_numpy(),
    }
    return np.column_stack([col_map[f] for f in features])

N_BOOT = 2_000

def bootstrap_auc_ci(y, proba):
    aucs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() in (0, len(idx)):
            aucs.append(0.5)
            continue
        aucs.append(roc_auc_score(y[idx], proba[idx]))
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

auc_table = {}

for label, df in [("L1", df1), ("L2", df2)]:
    ts     = df["entry_timestamp"].to_numpy()
    cutoff = ts.min() + 0.8 * (ts.max() - ts.min())
    tr     = ts <= cutoff
    te     = ts > cutoff
    y      = df["filled"].to_numpy().astype(int)

    row = {}
    for feat_label, feats in [("full", ALL_FEATURES), ("no_time", NO_TIME_FEATURES)]:
        X = build_X(df, feats)

        # Logistic
        sc  = StandardScaler()
        clf = LogisticRegression(max_iter=1000)
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        p_log = clf.predict_proba(sc.transform(X[te]))[:, 1]
        auc_log = roc_auc_score(y[te], p_log)

        # LightGBM
        lgb_tr = lgb.Dataset(X[tr], y[tr])
        params = dict(objective="binary", metric="auc", verbosity=-1,
                      num_leaves=31, learning_rate=0.05, n_estimators=200,
                      random_state=RNG_SEED)
        model = lgb.train(params, lgb_tr, num_boost_round=200,
                          valid_sets=[lgb.Dataset(X[te], y[te])],
                          callbacks=[lgb.early_stopping(20, verbose=False),
                                     lgb.log_evaluation(period=-1)])
        p_lgb = model.predict(X[te])
        auc_lgb = roc_auc_score(y[te], p_lgb)
        lo_lgb, hi_lgb = bootstrap_auc_ci(y[te], p_lgb)

        row[feat_label] = dict(log=auc_log, lgb=auc_lgb, lo=lo_lgb, hi=hi_lgb,
                               model=model, feats=feats)

        print(f"\n{label} [{feat_label:8s}]  logistic={auc_log:.4f}  "
              f"LightGBM={auc_lgb:.4f}  [{lo_lgb:.3f}, {hi_lgb:.3f}]")

    delta_lgb = row["full"]["lgb"] - row["no_time"]["lgb"]
    delta_log = row["full"]["log"] - row["no_time"]["log"]
    print(f"  → LightGBM AUC drop when time_frac removed: {delta_lgb:+.4f}")
    print(f"  → Logistic  AUC drop when time_frac removed: {delta_log:+.4f}")
    pct = delta_lgb / (row["full"]["lgb"] - 0.5) * 100 if row["full"]["lgb"] > 0.5 else 0
    print(f"  → time_frac accounts for ~{pct:.0f}% of LightGBM's edge over chance")
    auc_table[label] = row

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 68)
print("PLOTS")
print("=" * 68)

# ── Plot A: Fill time vs adverse selection (decile bins, L1 and L2) ───────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, label, f in [(axes[0], "L1 (touch)", f1), (axes[1], "L2 (depth-2)", f2)]:
    ttf = f["time_to_first_fill"].to_numpy()
    adv = f["adv_cents"].to_numpy()
    rho, p_val = spearmanr(ttf, adv)

    deciles = np.percentile(ttf, np.linspace(0, 100, 11))
    bin_mid, bin_mean, bin_lo, bin_hi = [], [], [], []
    for lo, hi in zip(deciles[:-1], deciles[1:]):
        mask = (ttf >= lo) & (ttf <= hi)
        if mask.sum() < 5:
            continue
        sub = adv[mask]
        boot = [rng.choice(sub, len(sub)).mean() for _ in range(1000)]
        bin_mid.append(ttf[mask].mean())
        bin_mean.append(sub.mean())
        bin_lo.append(np.percentile(boot, 2.5))
        bin_hi.append(np.percentile(boot, 97.5))

    bm = np.array(bin_mean)
    bl = np.array(bin_lo)
    bh = np.array(bin_hi)
    ax.plot(bin_mid, bm, "o-", color="steelblue", linewidth=2, markersize=6)
    ax.fill_between(bin_mid, bl, bh, alpha=0.2, color="steelblue")
    ax.axhline(adv.mean(), color="gray", linestyle="--", linewidth=1,
               label=f"Overall mean={adv.mean():+.3f}¢")
    ax.set_xlabel("Time to fill (seconds)")
    ax.set_ylabel("Mean adverse selection (¢/share)")
    ax.set_title(f"{label}: fill speed vs adverse selection\n"
                 f"Spearman ρ={rho:+.4f}  p={p_val:.4f}")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f¢"))

plt.tight_layout()
p = PLOTS_DIR / "23_fillspeed_vs_adv.png"
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"Saved: {p}")

# ── Plot B: AUC comparison — full vs no-time_frac ────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

x_pos     = np.array([0, 1, 2, 3, 5, 6, 7, 8])
bar_labels = ["L1 log\nfull", "L1 log\nno time", "L1 lgbm\nfull", "L1 lgbm\nno time",
              "L2 log\nfull", "L2 log\nno time", "L2 lgbm\nfull", "L2 lgbm\nno time"]
aucs = [
    auc_table["L1"]["full"]["log"],  auc_table["L1"]["no_time"]["log"],
    auc_table["L1"]["full"]["lgb"],  auc_table["L1"]["no_time"]["lgb"],
    auc_table["L2"]["full"]["log"],  auc_table["L2"]["no_time"]["log"],
    auc_table["L2"]["full"]["lgb"],  auc_table["L2"]["no_time"]["lgb"],
]
los = [0]*2 + [auc_table["L1"]["full"]["lo"], auc_table["L1"]["no_time"]["lo"]] + \
      [0]*2 + [auc_table["L2"]["full"]["lo"], auc_table["L2"]["no_time"]["lo"]]
his = [0]*2 + [auc_table["L1"]["full"]["hi"], auc_table["L1"]["no_time"]["hi"]] + \
      [0]*2 + [auc_table["L2"]["full"]["hi"], auc_table["L2"]["no_time"]["hi"]]
colors_b = ["steelblue", "lightsteelblue", "navy", "cornflowerblue",
            "tomato", "lightsalmon", "darkred", "salmon"]

bars = ax.bar(x_pos, aucs, color=colors_b, alpha=0.85, width=0.7)
# Error bars only for LightGBM (indices 2,3,6,7)
for i in [2, 3, 6, 7]:
    if los[i] > 0:
        ax.errorbar(x_pos[i], aucs[i],
                    yerr=[[aucs[i]-los[i]], [his[i]-aucs[i]]],
                    fmt="none", color="black", capsize=4)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
ax.set_xticks(x_pos)
ax.set_xticklabels(bar_labels, fontsize=8)
ax.set_ylabel("OOS AUC")
ax.set_ylim(0.46, 0.62)
ax.set_title("AUC: full feature set vs without time_frac\n"
             "(how much of LightGBM's edge is just time-of-day?)")
ax.legend()
for bar, auc in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, auc + 0.002,
            f"{auc:.3f}", ha="center", va="bottom", fontsize=7)

plt.tight_layout()
p = PLOTS_DIR / "24_auc_time_decomposition.png"
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"Saved: {p}")

# ── Plot C: LightGBM feature importances — full vs no-time ───────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for row_i, label in enumerate(["L1", "L2"]):
    for col_i, feat_label in enumerate(["full", "no_time"]):
        ax    = axes[row_i][col_i]
        model = auc_table[label][feat_label]["model"]
        feats = auc_table[label][feat_label]["feats"]
        imp   = model.feature_importance(importance_type="gain")
        order = np.argsort(imp)
        colors_imp = ["tomato" if feats[i] == "time_frac" else "steelblue" for i in order]
        ax.barh(range(len(feats)), imp[order], color=colors_imp, alpha=0.85)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels([feats[i] for i in order])
        ax.set_xlabel("Feature importance (gain)")
        auc_v = auc_table[label][feat_label]["lgb"]
        ax.set_title(f"{label} LightGBM — {feat_label}  (AUC={auc_v:.4f})\n"
                     f"{'red = time_frac' if feat_label=='full' else 'time_frac removed'}")

plt.tight_layout()
p = PLOTS_DIR / "25_feature_importance_decomp.png"
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"Saved: {p}")

print("\nAll plots saved to results/analysis/")
print("\n[Deep angles analysis complete]")
