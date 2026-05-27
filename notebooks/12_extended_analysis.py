"""
Extended multi-ticker analysis — all seven analyses.

Sections
--------
1. Bootstrap CIs on L2 logistic coefficients (all tickers)
2. LightGBM + time_frac ablation per ticker (L1 and L2)
3. Adverse selection: mean, fraction, L1 vs L2 gap, intraday pattern
4. Optimal placement crossover (C*) per ticker
5. Survival analysis: Cox PH hazard ratios per ticker
6. Panel regression with ticker fixed effects
7. OFI predictive power per ticker

Run from project root:
    ./run.sh notebooks/12_extended_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, ttest_ind
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import lightgbm as lgb

RESULTS_DIR = Path("results")
PLOTS_DIR   = Path("results/extended")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED     = 42
N_BOOT       = 2_000
LIFETIME     = 60.0
_DAY_START   = 34_500.0
_DAY_END     = 57_300.0
RAW_TO_CENTS = 0.01   # adverse_selection_1s units → cents

FEATURES_BASE = [
    "spread_at_entry_ticks",
    "book_imbalance_at_entry",
    "queue_position_at_entry",
    "queue_granularity_at_entry",
    "time_frac",
    "side_bid",
    "ofi_10s",
    "ofi_30s",
]

rng = np.random.default_rng(RNG_SEED)


# ── shared helpers ────────────────────────────────────────────────────────────

def add_derived(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("side").eq("bid").cast(pl.Int8).alias("side_bid"),
        ((pl.col("entry_timestamp") - _DAY_START) / (_DAY_END - _DAY_START))
        .clip(0.0, 1.0).alias("time_frac"),
        (pl.col("adverse_selection_1s") * RAW_TO_CENTS).alias("adv_cents"),
    )


def load_pooled(level: int) -> pl.DataFrame:
    path = RESULTS_DIR / f"experiment_all_L{level}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run experiment_runner.py first")
    return add_derived(pl.read_parquet(path))


def get_xy(df: pl.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    avail = [f for f in features if f in df.columns]
    X = df.select(avail).to_numpy().astype(float)
    y = df["filled"].cast(pl.Int8).to_numpy()
    mask = ~np.isnan(X).any(axis=1)
    return X[mask], y[mask]


def chrono_auc(
    df: pl.DataFrame,
    features: list[str],
    test_frac: float = 0.2,
) -> tuple[float, float, float]:
    X, y = get_xy(df, features)
    n = len(X); split = int(n * (1 - test_frac))
    sc = StandardScaler().fit(X[:split])
    m  = LogisticRegression(max_iter=1_000, random_state=RNG_SEED)
    m.fit(sc.transform(X[:split]), y[:split])
    prob = m.predict_proba(sc.transform(X[split:]))[:, 1]
    pt   = roc_auc_score(y[split:], prob)
    boots = [
        roc_auc_score(
            y[split:][idx := rng.integers(0, len(y[split:]), len(y[split:]))],
            prob[idx],
        )
        for _ in range(N_BOOT)
    ]
    return pt, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def lgb_auc(
    df: pl.DataFrame,
    features: list[str],
    test_frac: float = 0.2,
) -> float:
    X, y = get_xy(df, features)
    n = len(X); split = int(n * (1 - test_frac))
    params = dict(
        objective="binary", metric="auc", verbosity=-1,
        n_estimators=200, learning_rate=0.05,
        num_leaves=31, random_state=RNG_SEED,
    )
    m = lgb.LGBMClassifier(**params)
    m.fit(X[:split], y[:split])
    prob = m.predict_proba(X[split:])[:, 1]
    return roc_auc_score(y[split:], prob)


TICKERS = ["AAPL", "INTC", "MSFT"]
SEP = "=" * 65


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Bootstrap CIs on L2 logistic coefficients
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{SEP}")
print("SECTION 1 — Bootstrap CIs on L2 logistic coefficients")
print(SEP)

df_l2 = load_pooled(2)
FEAT_NO_OFI = [f for f in FEATURES_BASE if f not in ("ofi_10s", "ofi_30s")]

for ticker in TICKERS:
    sub = df_l2.filter(pl.col("ticker") == ticker)
    avail = [f for f in FEAT_NO_OFI if f in sub.columns]
    X, y = get_xy(sub, avail)
    n = len(X)

    # fit on full data for coefficient bootstrap (in-sample to get stable estimates)
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    m  = LogisticRegression(max_iter=1_000, random_state=RNG_SEED).fit(Xs, y)
    point_coefs = m.coef_[0]

    boot_coefs = np.zeros((N_BOOT, len(avail)))
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        mb  = LogisticRegression(max_iter=500, random_state=b)
        mb.fit(Xs[idx], y[idx])
        boot_coefs[b] = mb.coef_[0]

    lo = np.percentile(boot_coefs, 2.5, axis=0)
    hi = np.percentile(boot_coefs, 97.5, axis=0)
    sig = (lo > 0) | (hi < 0)

    print(f"\n  {ticker} L2  (n={n:,}) — standardised logistic coefs, 95% bootstrap CI")
    print(f"  {'Feature':<35s}  {'Coef':>7s}  {'95% CI':<22s}  Sig")
    print(f"  {'-'*35}  {'-'*7}  {'-'*22}  ---")
    order = np.argsort(np.abs(point_coefs))[::-1]
    for i in order:
        star = "***" if sig[i] else "   "
        print(
            f"  {avail[i]:<35s}  {point_coefs[i]:+.3f}  "
            f"[{lo[i]:+.3f}, {hi[i]:+.3f}]  {star}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LightGBM + time_frac ablation per ticker (L1 and L2)
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{SEP}")
print("SECTION 2 — LightGBM vs logistic, and time_frac ablation")
print(SEP)

FEAT_NO_TIME = [f for f in FEATURES_BASE if f != "time_frac"]

rows_abl = []
for level in (1, 2):
    df = load_pooled(level)
    for ticker in TICKERS:
        sub = df.filter(pl.col("ticker") == ticker)
        lr_auc, lr_lo, lr_hi = chrono_auc(sub, FEATURES_BASE)
        lgb_full  = lgb_auc(sub, FEATURES_BASE)
        lgb_notf  = lgb_auc(sub, FEAT_NO_TIME)
        drop      = lgb_full - lgb_notf
        pct_edge  = drop / max(lgb_full - 0.5, 1e-6) * 100
        rows_abl.append(dict(
            ticker=ticker, level=level,
            lr_auc=lr_auc, lr_lo=lr_lo, lr_hi=lr_hi,
            lgb_full=lgb_full, lgb_no_tf=lgb_notf,
            drop=drop, pct_edge=pct_edge,
        ))
        print(
            f"  {ticker} L{level}  LR={lr_auc:.3f} [{lr_lo:.3f},{lr_hi:.3f}]"
            f"  LGB={lgb_full:.3f}  LGB−tf={lgb_notf:.3f}"
            f"  drop={drop:+.3f}  ({pct_edge:.0f}% of edge)"
        )

# bar chart: LGB full vs LGB without time_frac at L2
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
for ax, level in zip(axes, (1, 2)):
    level_rows = [r for r in rows_abl if r["level"] == level]
    x   = np.arange(len(level_rows))
    w   = 0.28
    ax.bar(x - w, [r["lr_auc"]    for r in level_rows], w, label="Logistic",         color="#5b9bd5")
    ax.bar(x,     [r["lgb_full"]  for r in level_rows], w, label="LGB full",          color="#ed7d31")
    ax.bar(x + w, [r["lgb_no_tf"] for r in level_rows], w, label="LGB w/o time_frac", color="#a9d18e")
    ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels([r["ticker"] for r in level_rows])
    ax.set_title(f"L{level} AUC comparison")
    ax.set_ylabel("OOS AUC"); ax.legend(fontsize=7)
    ax.set_ylim(0.40, 0.65)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "lgb_ablation_by_ticker.png", dpi=150)
plt.close(fig)
print(f"\n  Saved: {PLOTS_DIR}/lgb_ablation_by_ticker.png")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Adverse selection: mean, fraction, L1 vs L2, intraday
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{SEP}")
print("SECTION 3 — Adverse selection by ticker")
print(SEP)

adv_rows = []
for level in (1, 2):
    df = load_pooled(level)
    for ticker in TICKERS:
        sub  = df.filter(pl.col("ticker") == ticker)
        fills = sub.filter(pl.col("filled") & pl.col("adv_cents").is_not_null())
        if len(fills) < 10:
            continue
        adv = fills["adv_cents"].to_numpy()
        mean_adv  = adv.mean()
        frac_adv  = (adv > 0).mean()
        # bootstrap CI on mean
        boots = [rng.choice(adv, len(adv), replace=True).mean() for _ in range(N_BOOT)]
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        adv_rows.append(dict(
            ticker=ticker, level=level, n=len(fills),
            mean_adv=mean_adv, lo=lo, hi=hi, frac_adv=frac_adv,
        ))
        print(
            f"  {ticker} L{level}  n={len(fills):,}"
            f"  mean={mean_adv:+.3f}¢ [{lo:+.3f}, {hi:+.3f}]"
            f"  frac_adversely_selected={frac_adv:.1%}"
        )

# L1 vs L2 t-test per ticker
print()
for ticker in TICKERS:
    l1 = next((r for r in adv_rows if r["ticker"] == ticker and r["level"] == 1), None)
    l2 = next((r for r in adv_rows if r["ticker"] == ticker and r["level"] == 2), None)
    if not l1 or not l2:
        continue
    df1_f = load_pooled(1).filter((pl.col("ticker") == ticker) & pl.col("filled") & pl.col("adv_cents").is_not_null())
    df2_f = load_pooled(2).filter((pl.col("ticker") == ticker) & pl.col("filled") & pl.col("adv_cents").is_not_null())
    stat, p = ttest_ind(df1_f["adv_cents"].to_numpy(), df2_f["adv_cents"].to_numpy(), equal_var=False)
    direction = "L1 > L2" if l1["mean_adv"] > l2["mean_adv"] else "L2 > L1"
    print(f"  {ticker}  {direction}  ({l1['mean_adv']:+.3f}¢ vs {l2['mean_adv']:+.3f}¢)  t={stat:.2f}  p={p:.4f}")

# intraday adverse selection — 1-hour buckets
print("\n  Intraday adverse selection (hourly mean, L2):")
df_l2_adv = load_pooled(2).with_columns(
    ((pl.col("entry_timestamp") // 3600).cast(pl.Int32)).alias("hour")
).filter(pl.col("filled") & pl.col("adv_cents").is_not_null())

fig, axes = plt.subplots(1, len(TICKERS), figsize=(12, 3.5), sharey=False)
for ax, ticker in zip(axes, TICKERS):
    sub = df_l2_adv.filter(pl.col("ticker") == ticker)
    hourly = (
        sub.group_by("hour").agg(pl.col("adv_cents").mean().alias("mean_adv"), pl.len().alias("n"))
        .sort("hour")
    )
    ax.bar(hourly["hour"].to_list(), hourly["mean_adv"].to_list(), color="#4a90d9", alpha=0.8)
    ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
    ax.set_title(f"{ticker} L2 intraday adv. sel.")
    ax.set_xlabel("Hour (UTC-5)"); ax.set_ylabel("Adverse selection (¢)")
plt.tight_layout()
fig.savefig(PLOTS_DIR / "adverse_selection_intraday.png", dpi=150)
plt.close(fig)
print(f"  Saved: {PLOTS_DIR}/adverse_selection_intraday.png")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Optimal placement crossover C* per ticker
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{SEP}")
print("SECTION 4 — Optimal placement crossover C* per ticker")
print(SEP)

TICK_CENTS = 1.0   # 1 tick = 1 cent for all three tickers on this day

print(f"\n  {'Ticker':<6}  {'p1':>5}  {'p2':>5}  {'AS1':>7}  {'AS2':>7}  {'C*':>7}  Prefer at C=10¢")
print(f"  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*15}")

c_star_rows = []
for ticker in TICKERS:
    df1 = load_pooled(1).filter(pl.col("ticker") == ticker)
    df2 = load_pooled(2).filter(pl.col("ticker") == ticker)

    p1 = float(df1["filled"].mean())
    p2 = float(df2["filled"].mean())

    f1 = df1.filter(pl.col("filled") & pl.col("adv_cents").is_not_null())
    f2 = df2.filter(pl.col("filled") & pl.col("adv_cents").is_not_null())

    if len(f1) < 5 or len(f2) < 5:
        continue

    as1 = float(f1["adv_cents"].mean())
    as2 = float(f2["adv_cents"].mean())
    s   = float(df1["spread_at_entry_ticks"].mean()) * 0.5   # half-spread in cents

    # E[IS|L1] = p1*(-S+AS1) + (1-p1)*C
    # E[IS|L2] = p2*(-S-1+AS2) + (1-p2)*C
    # crossover: (p2-p1)*C = p2*(-S-1+AS2) - p1*(-S+AS1)
    num = p2 * (-s - TICK_CENTS + as2) - p1 * (-s + as1)
    den = p2 - p1
    c_star = num / den if abs(den) > 1e-6 else float("nan")

    prefer_at_10 = "L1" if (10.0 > c_star) else "L2"
    c_star_rows.append(dict(ticker=ticker, p1=p1, p2=p2, as1=as1, as2=as2, c_star=c_star))
    print(f"  {ticker:<6}  {p1:.3f}  {p2:.3f}  {as1:+.3f}¢  {as2:+.3f}¢  {c_star:7.2f}¢  {prefer_at_10}")

# crossover curve plot
C_range = np.linspace(0, 30, 300)
fig, ax = plt.subplots(figsize=(8, 4))
colors = {"AAPL": "#4a90d9", "INTC": "#e07b3a", "MSFT": "#5cb85c"}
for row in c_star_rows:
    ticker = row["ticker"]; p1 = row["p1"]; p2 = row["p2"]
    as1 = row["as1"]; as2 = row["as2"]
    s_avg = float(load_pooled(1).filter(pl.col("ticker") == ticker)["spread_at_entry_ticks"].mean()) * 0.5
    is_l1 = p1 * (-s_avg + as1) + (1 - p1) * C_range
    is_l2 = p2 * (-s_avg - TICK_CENTS + as2) + (1 - p2) * C_range
    ax.plot(C_range, is_l1 - is_l2, label=ticker, color=colors.get(ticker, "grey"))
    ax.axvline(row["c_star"], color=colors.get(ticker, "grey"), linestyle=":", linewidth=0.8)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Unfill penalty C (¢/share)")
ax.set_ylabel("E[IS|L1] − E[IS|L2]  (positive → prefer L2)")
ax.set_title("Optimal placement crossover by ticker\n(above 0 = L2 better; below 0 = L1 better)")
ax.legend(); plt.tight_layout()
fig.savefig(PLOTS_DIR / "optimal_placement_crossover.png", dpi=150)
plt.close(fig)
print(f"\n  Saved: {PLOTS_DIR}/optimal_placement_crossover.png")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Survival analysis: Cox PH per ticker at L2
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{SEP}")
print("SECTION 5 — Survival analysis (Cox PH) by ticker, L2")
print(SEP)

COX_FEATURES = [
    "spread_at_entry_ticks", "book_imbalance_at_entry",
    "queue_position_at_entry", "queue_granularity_at_entry",
    "time_frac", "side_bid",
]

df_l2 = load_pooled(2)

fig, axes = plt.subplots(1, len(TICKERS), figsize=(13, 4), sharey=False)

for ax, ticker in zip(axes, TICKERS):
    sub = df_l2.filter(pl.col("ticker") == ticker)
    ts  = sub["entry_timestamp"].to_numpy()

    duration = np.where(sub["filled"].to_numpy(), sub["time_to_first_fill"].to_numpy(), LIFETIME)
    event    = sub["filled"].to_numpy().astype(int)

    avail = [f for f in COX_FEATURES if f in sub.columns]
    cov   = sub.select(avail).to_numpy().astype(float)
    sc    = StandardScaler().fit(cov)
    cov_s = sc.transform(cov)

    surv_df = pd.DataFrame(cov_s, columns=avail)
    surv_df["duration"] = duration
    surv_df["event"]    = event

    # drop rows with NaN
    surv_df = surv_df.dropna()

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(surv_df, duration_col="duration", event_col="event")

    print(f"\n  {ticker} L2  (n={len(surv_df):,}  events={event.sum():,})")
    print(f"  C-index (in-sample): {cph.concordance_index_:.3f}")
    summary = cph.summary[["coef", "exp(coef)", "p"]].copy()
    summary["sig"] = summary["p"].apply(lambda p: "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else "")))
    for feat, row2 in summary.iterrows():
        print(f"    {feat:<35s}  HR={row2['exp(coef)']:.3f}  p={row2['p']:.4f}  {row2['sig']}")

    # KM curve by spread tertile
    spread_vals = sub["spread_at_entry_ticks"].to_numpy()
    t33, t67   = np.percentile(spread_vals, 33), np.percentile(spread_vals, 67)
    for label, mask, color in [
        ("narrow spread", spread_vals <= t33, "#4a90d9"),
        ("wide spread",   spread_vals >= t67, "#e07b3a"),
    ]:
        kmf = KaplanMeierFitter()
        dur_m = duration[mask]; ev_m = event[mask]
        kmf.fit(dur_m, event_observed=ev_m, label=label)
        kmf.plot_survival_function(ax=ax, color=color, ci_show=False)

    # log-rank test
    mask_narrow = spread_vals <= t33
    mask_wide   = spread_vals >= t67
    lr = logrank_test(
        duration[mask_narrow], duration[mask_wide],
        event_observed_A=event[mask_narrow], event_observed_B=event[mask_wide],
    )
    ax.set_title(f"{ticker} L2 KM (log-rank p={lr.p_value:.3f})")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("P(not yet filled)")

plt.tight_layout()
fig.savefig(PLOTS_DIR / "survival_km_by_ticker.png", dpi=150)
plt.close(fig)
print(f"\n  Saved: {PLOTS_DIR}/survival_km_by_ticker.png")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Panel regression with ticker fixed effects
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{SEP}")
print("SECTION 6 — Panel logistic regression with ticker fixed effects")
print(SEP)

df_l2 = load_pooled(2)

# add ticker dummies (AAPL as reference)
df_l2 = df_l2.with_columns(
    pl.col("ticker").eq("INTC").cast(pl.Int8).alias("ticker_INTC"),
    pl.col("ticker").eq("MSFT").cast(pl.Int8).alias("ticker_MSFT"),
)

PANEL_FEATURES = FEATURES_BASE + ["ticker_INTC", "ticker_MSFT"]
avail_panel = [f for f in PANEL_FEATURES if f in df_l2.columns]

X_all, y_all = get_xy(df_l2, avail_panel)
n_all = len(X_all)
split = int(n_all * 0.8)

# sort chronologically for the split (already ordered in pooled parquet)
sc_panel = StandardScaler().fit(X_all[:split])
m_panel  = LogisticRegression(max_iter=1_000, random_state=RNG_SEED)
m_panel.fit(sc_panel.transform(X_all[:split]), y_all[:split])
prob_panel = m_panel.predict_proba(sc_panel.transform(X_all[split:]))[:, 1]
panel_auc  = roc_auc_score(y_all[split:], prob_panel)

# bootstrap CIs on panel coefficients (full data)
sc_full = StandardScaler().fit(X_all)
Xs_full = sc_full.transform(X_all)
m_full  = LogisticRegression(max_iter=1_000, random_state=RNG_SEED).fit(Xs_full, y_all)
point   = m_full.coef_[0]

boot_panel = np.zeros((N_BOOT, len(avail_panel)))
for b in range(N_BOOT):
    idx = rng.integers(0, n_all, n_all)
    mb  = LogisticRegression(max_iter=500, random_state=b)
    mb.fit(Xs_full[idx], y_all[idx])
    boot_panel[b] = mb.coef_[0]

lo_p = np.percentile(boot_panel, 2.5, axis=0)
hi_p = np.percentile(boot_panel, 97.5, axis=0)
sig_p = (lo_p > 0) | (hi_p < 0)

print(f"\n  Panel model OOS AUC: {panel_auc:.3f}  (n={n_all:,}, test n={n_all - split:,})")
print(f"  Reference ticker: AAPL\n")
print(f"  {'Feature':<35s}  {'Coef':>7s}  {'95% CI':<22s}  Sig")
print(f"  {'-'*35}  {'-'*7}  {'-'*22}  ---")
order = np.argsort(np.abs(point))[::-1]
for i in order:
    star = "***" if sig_p[i] else "   "
    print(
        f"  {avail_panel[i]:<35s}  {point[i]:+.3f}  "
        f"[{lo_p[i]:+.3f}, {hi_p[i]:+.3f}]  {star}"
    )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — OFI predictive power per ticker
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{SEP}")
print("SECTION 7 — OFI predictive power by ticker (L2)")
print(SEP)

FEAT_WITH_OFI    = FEATURES_BASE
FEAT_WITHOUT_OFI = [f for f in FEATURES_BASE if f not in ("ofi_10s", "ofi_30s")]

df_l2 = load_pooled(2)

print(f"\n  {'Ticker':<6}  {'AUC w/ OFI':>10}  {'AUC w/o OFI':>12}  {'ΔAUC':>7}  OFI adds signal?")
print(f"  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*7}  {'-'*15}")

for ticker in TICKERS:
    sub = df_l2.filter(pl.col("ticker") == ticker)
    auc_with,    lo_w, hi_w = chrono_auc(sub, FEAT_WITH_OFI)
    auc_without, lo_wo, hi_wo = chrono_auc(sub, FEAT_WITHOUT_OFI)
    delta = auc_with - auc_without
    # OFI adds signal if delta > 0 and lower CI of (with OFI) > upper CI of (without OFI)
    signal = "YES" if (delta > 0 and lo_w > hi_wo) else ("marginal" if delta > 0.003 else "no")
    print(f"  {ticker:<6}  {auc_with:.3f} [{lo_w:.3f},{hi_w:.3f}]  {auc_without:.3f} [{lo_wo:.3f},{hi_wo:.3f}]  {delta:+.3f}  {signal}")

# bootstrap CI on OFI_30s coefficient per ticker (L2)
print(f"\n  Bootstrap 95% CI on OFI_30s coefficient at L2 (standardised logistic):")
avail_ofi = [f for f in FEAT_WITH_OFI if f in df_l2.columns]
ofi_idx   = avail_ofi.index("ofi_30s") if "ofi_30s" in avail_ofi else None

if ofi_idx is not None:
    for ticker in TICKERS:
        sub  = df_l2.filter(pl.col("ticker") == ticker)
        X, y = get_xy(sub, avail_ofi)
        n    = len(X)
        sc   = StandardScaler().fit(X)
        Xs   = sc.transform(X)
        boots_ofi = np.zeros(N_BOOT)
        for b in range(N_BOOT):
            idx = rng.integers(0, n, n)
            mb  = LogisticRegression(max_iter=500, random_state=b)
            mb.fit(Xs[idx], y[idx])
            boots_ofi[b] = mb.coef_[0][ofi_idx]
        lo_o = float(np.percentile(boots_ofi, 2.5))
        hi_o = float(np.percentile(boots_ofi, 97.5))
        sig_o = (lo_o > 0) or (hi_o < 0)
        m_pt  = LogisticRegression(max_iter=1_000, random_state=RNG_SEED).fit(Xs, y)
        pt_o  = m_pt.coef_[0][ofi_idx]
        print(f"  {ticker:<6}  OFI_30s coef={pt_o:+.3f}  CI=[{lo_o:+.3f}, {hi_o:+.3f}]  {'SIGNIFICANT ***' if sig_o else 'not significant'}")


print(f"\n{SEP}")
print("EXTENDED ANALYSIS COMPLETE")
print(SEP)
