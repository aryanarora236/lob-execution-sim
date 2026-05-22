"""
Phase 5 extension: Survival analysis of time-to-fill.

Motivation
----------
The binary "filled within 60 s" outcome discards fill-timing information.
A Cox proportional hazards model treats expired/unfilled orders as
right-censored and models the *hazard rate* — the instantaneous probability
of filling at time t given the order is still alive.

This answers a different, richer question: does a feature like OFI or spread
predict how *quickly* an order fills, not just whether it fills?

Key concepts
------------
- Duration (T): time_to_first_fill for filled orders; lifetime (60 s) for
  unfilled orders (right-censored at expiry).
- Event indicator (E): 1 = filled, 0 = right-censored (expired unfilled).
- Hazard ratio (HR): exp(coef). HR > 1 → faster filling; HR < 1 → slower.
- C-index: concordance between predicted risk and actual fill order. Analogous
  to AUC for survival models (0.5 = random, 1.0 = perfect).

Run from project root:
    uv run python notebooks/08_survival.py
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
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import pandas as pd

PARQUET_L1 = Path("results/experiment_AAPL_2019-12-30_L1.parquet")
PARQUET_L2 = Path("results/experiment_AAPL_2019-12-30_L2.parquet")
PLOTS_DIR  = Path("results/analysis")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

LIFETIME = 60.0   # order expiry in seconds

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Prepare survival data
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 68)
print("SECTION 1 — Prepare survival data")
print("=" * 68)

def make_survival_df(df: pl.DataFrame) -> pd.DataFrame:
    """
    Build a pandas DataFrame suitable for lifelines.

    Columns
    -------
    duration : seconds to fill (filled orders) or LIFETIME (censored)
    event    : 1 = filled, 0 = censored
    + covariates scaled to mean=0 std=1
    """
    ts     = df["entry_timestamp"].to_numpy()
    day_lo = ts.min(); day_hi = ts.max()

    duration = np.where(
        df["filled"].to_numpy(),
        df["time_to_first_fill"].to_numpy(),
        LIFETIME,
    )
    event = df["filled"].to_numpy().astype(int)

    raw = {
        "duration":         duration,
        "event":            event,
        "spread_ticks":     df["spread_at_entry_ticks"].to_numpy().astype(float),
        "imbalance":        df["book_imbalance_at_entry"].to_numpy(),
        "side_bid":         (df["side"] == "bid").to_numpy().astype(float),
        "time_frac":        (ts - day_lo) / (day_hi - day_lo),
        "log_queue_shares": np.log1p(df["queue_position_at_entry"].to_numpy()),
        "granularity":      df["queue_granularity_at_entry"].to_numpy(),
        "ofi_10s":          df["ofi_10s"].to_numpy(),
    }

    pdf = pd.DataFrame(raw)

    # Standardise covariates (exclude duration and event)
    covs = [c for c in pdf.columns if c not in ("duration", "event")]
    pdf[covs] = (pdf[covs] - pdf[covs].mean()) / pdf[covs].std()

    return pdf

pdf1 = make_survival_df(df1 := pl.read_parquet(PARQUET_L1))
pdf2 = make_survival_df(df2 := pl.read_parquet(PARQUET_L2))

for label, pdf in [("L1", pdf1), ("L2", pdf2)]:
    n_fill = int(pdf["event"].sum())
    n_cens = len(pdf) - n_fill
    med_t  = pdf.loc[pdf["event"] == 1, "duration"].median()
    print(f"{label}: {len(pdf):,} orders  filled={n_fill:,}  censored={n_cens:,}  "
          f"median fill time={med_t:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Kaplan-Meier survival curves
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 68)
print("SECTION 2 — Kaplan-Meier curves")
print("=" * 68)

# Split by spread level (1-tick vs wide) and by side for each level
for label, pdf in [("L1", pdf1), ("L2", pdf2)]:
    narrow = pdf[pdf["spread_ticks"] <= pdf["spread_ticks"].median()]
    wide   = pdf[pdf["spread_ticks"] >  pdf["spread_ticks"].median()]
    lr     = logrank_test(
        narrow["duration"], wide["duration"],
        narrow["event"],    wide["event"],
    )
    print(f"\n{label} log-rank test (narrow vs wide spread): "
          f"p={lr.p_value:.4f}  test_stat={lr.test_statistic:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Cox proportional hazards model
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 68)
print("SECTION 3 — Cox PH model")
print("=" * 68)

COVARIATES = ["spread_ticks", "imbalance", "side_bid", "time_frac",
              "log_queue_shares", "granularity", "ofi_10s"]

cox_results = {}

for label, pdf in [("L1", pdf1), ("L2", pdf2)]:
    cph = CoxPHFitter()
    cph.fit(
        pdf[["duration", "event"] + COVARIATES],
        duration_col="duration",
        event_col="event",
        show_progress=False,
    )
    c_idx = cph.concordance_index_
    print(f"\n{label} Cox model  C-index={c_idx:.4f}")
    print(cph.summary[["coef", "exp(coef)", "coef lower 95%", "coef upper 95%",
                         "p"]].to_string())
    cox_results[label] = cph

# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Time-based OOS C-index
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 68)
print("SECTION 4 — OOS C-index (time-based split)")
print("=" * 68)

for label, pdf in [("L1", pdf1), ("L2", pdf2)]:
    # time_frac is standardised so we use the raw split on position
    n = len(pdf)
    split = int(n * 0.8)
    # Sort by original row order (already time-sorted from experiment)
    train = pdf.iloc[:split].copy()
    test  = pdf.iloc[split:].copy()

    cph_oos = CoxPHFitter()
    cph_oos.fit(
        train[["duration", "event"] + COVARIATES],
        duration_col="duration",
        event_col="event",
        show_progress=False,
    )
    c_train = cph_oos.concordance_index_
    c_test  = cph_oos.score(
        test[["duration", "event"] + COVARIATES],
        scoring_method="concordance_index",
    )
    print(f"{label}  C-index  train={c_train:.4f}  test={c_test:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Plots
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 68)
print("SECTION 5 — Plots")
print("=" * 68)

# ── Plot A: Kaplan-Meier L1 vs L2 ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

for label, pdf, color in [("L1 (touch)", pdf1, "steelblue"),
                           ("L2 (depth-2)", pdf2, "tomato")]:
    kmf = KaplanMeierFitter()
    kmf.fit(pdf["duration"], pdf["event"], label=label)
    kmf.plot_survival_function(ax=ax, color=color, ci_show=True, ci_alpha=0.15)

ax.set_xlabel("Time since injection (seconds)")
ax.set_ylabel("P(not yet filled)")
ax.set_title("Kaplan-Meier survival: L1 vs L2\n(shaded = 95% CI)")
ax.set_xlim(0, LIFETIME)
ax.legend()

plt.tight_layout()
p = PLOTS_DIR / "15_km_l1_vs_l2.png"
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"Saved: {p}")

# ── Plot B: KM by spread quartile (L1 and L2 side by side) ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, label, pdf in [(axes[0], "L1 (touch)", pdf1),
                       (axes[1], "L2 (depth-2)", pdf2)]:
    q25, q75 = pdf["spread_ticks"].quantile(0.25), pdf["spread_ticks"].quantile(0.75)
    groups = {
        "Narrow spread (bot 25%)": pdf[pdf["spread_ticks"] <= q25],
        "Mid spread":              pdf[(pdf["spread_ticks"] > q25) & (pdf["spread_ticks"] < q75)],
        "Wide spread (top 25%)":   pdf[pdf["spread_ticks"] >= q75],
    }
    colors_km = ["green", "steelblue", "tomato"]
    for (grp_label, grp), color in zip(groups.items(), colors_km):
        if len(grp) < 10:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(grp["duration"], grp["event"], label=f"{grp_label} (n={len(grp):,})")
        kmf.plot_survival_function(ax=ax, color=color, ci_show=False)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("P(not yet filled)")
    ax.set_title(f"{label}: KM by spread")
    ax.set_xlim(0, LIFETIME)
    ax.legend(fontsize=7)

plt.tight_layout()
p = PLOTS_DIR / "16_km_by_spread.png"
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"Saved: {p}")

# ── Plot C: Cox hazard ratios (forest plot) ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, label in [(axes[0], "L1"), (axes[1], "L2")]:
    cph   = cox_results[label]
    summ  = cph.summary
    hr    = summ["exp(coef)"].values
    lo    = summ["exp(coef) lower 95%"].values
    hi    = summ["exp(coef) upper 95%"].values
    names = list(summ.index)
    p_val = summ["p"].values
    sig   = p_val < 0.05

    order = np.argsort(hr)
    y_pos = np.arange(len(names))

    colors_hr = ["tomato" if sig[i] else "lightgray" for i in order]
    ax.barh(y_pos, hr[order] - 1, left=1, color=colors_hr, alpha=0.85)
    ax.errorbar(hr[order], y_pos,
                xerr=[hr[order] - lo[order], hi[order] - hr[order]],
                fmt="none", color="black", capsize=4)
    ax.axvline(1.0, color="black", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([names[i] for i in order])
    ax.set_xlabel("Hazard ratio  (HR > 1 → faster fill)")
    ax.set_title(f"{label} Cox hazard ratios\n"
                 f"C-index={cox_results[label].concordance_index_:.4f}  "
                 f"(red = p < 0.05)")

plt.tight_layout()
p = PLOTS_DIR / "17_cox_hr.png"
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"Saved: {p}")

# ── Plot D: KM by OFI sign ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, label, pdf in [(axes[0], "L1 (touch)", pdf1),
                       (axes[1], "L2 (depth-2)", pdf2)]:
    # OFI is standardised; split at 0 (= raw mean)
    pos_ofi = pdf[pdf["ofi_10s"] >= 0]
    neg_ofi = pdf[pdf["ofi_10s"] <  0]

    lr = logrank_test(
        pos_ofi["duration"], neg_ofi["duration"],
        pos_ofi["event"],    neg_ofi["event"],
    )

    for grp, color, grp_label in [
        (pos_ofi, "steelblue", f"OFI ≥ mean (n={len(pos_ofi):,})"),
        (neg_ofi, "tomato",    f"OFI < mean (n={len(neg_ofi):,})"),
    ]:
        kmf = KaplanMeierFitter()
        kmf.fit(grp["duration"], grp["event"], label=grp_label)
        kmf.plot_survival_function(ax=ax, color=color, ci_show=False)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("P(not yet filled)")
    ax.set_title(f"{label}: KM by OFI_10s sign\n(log-rank p={lr.p_value:.4f})")
    ax.set_xlim(0, LIFETIME)
    ax.legend(fontsize=8)

plt.tight_layout()
p = PLOTS_DIR / "18_km_by_ofi.png"
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"Saved: {p}")

print("\nAll plots saved to results/analysis/")
print("\n[Survival analysis complete]")
