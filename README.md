# LOB Execution Simulator

A queue-position-aware limit order book simulator built on NASDAQ LOBSTER data.
Injects hypothetical passive limit orders into a replayed historical order book
and measures fill probability, fill speed, and adverse selection as functions of
observable market state.

---

## Motivation

Passive limit orders are cheaper than market orders but uncertain — you might not fill.
The key questions are:

1. **Fill probability:** can observable state (spread, book imbalance, queue depth, time of day) predict whether a passive order fills within its lifetime?
2. **Touch vs depth:** does predictability differ between posting at the best price (L1) and one tick behind it (L2)?
3. **Queue granularity:** does a fragmented queue (many small orders) drain more slowly than a concentrated one (few large orders)?
4. **Adverse selection:** when your order fills, how much does the mid-price move against you in the next second?
5. **Fill speed:** does the same state that predicts *whether* you fill also predict *how fast* you fill?

---

## Data

**NASDAQ LOBSTER** format for **AAPL**, **December 30, 2019** (holiday-week low-volume session).
1.6 million order-book events covering 09:35–15:55 ET.

> Data not included in the repository. Place LOBSTER message and order-book CSVs in `data/raw/`.

---

## Method

**Passive-shadow model.** Hypothetical orders are injected into the historical event stream
without affecting prices or displacing existing resting orders. Each order is placed at
either the best price (L1) or the second-best price (L2) and tracked through the replayed
book until filled or expired (60 s lifetime).

**Stratified injection.** The trading day is divided into N equal windows; one injection
timestamp is drawn uniformly at random within each window, giving uniform temporal coverage.

**OFI.** Order Flow Imbalance (Cont, Kukanov, Stoikov 2014) is computed from consecutive
best-bid/ask snapshots during replay and recorded at 10 s and 30 s look-back windows per injection.

---

## Key findings

### 1. Touch/depth asymmetry in fill predictability

| | L1 (touch) | L2 (depth) |
|---|---|---|
| Orders injected | 4,343 | 4,701 |
| Fill rate (60 s) | **63.2%** | **55.1%** |
| Logistic OOS AUC | 0.491 (≈ chance) | 0.494 (≈ chance) |
| LightGBM OOS AUC | 0.548 | 0.542 |
| Significant linear predictors | 1 (`spread_ticks`) | 4 |

At the touch, fill outcomes are essentially unpredictable — consistent with HFT market-makers
pricing the L1 queue efficiently. At depth, four features become significant:

| Feature | L2 coef (std.) | HR (Cox) | Interpretation |
|---|---|---|---|
| `side_bid` | −0.191 | 0.899 | Bids fill slower/less (upward drift this day) |
| `spread_ticks` | −0.119 | 0.915 | Wider spread → lower fill rate and slower fill |
| `time_frac` | −0.105 | 0.954 | Earlier in day → more likely and faster to fill |
| `imbalance` | −0.063 | 0.957 | Higher imbalance → slightly slower fill |

### 2. Queue granularity — null result

Median K/Q (orders ahead / shares ahead) = 0.01 at both levels with near-zero variance.
AAPL's visible queue is structurally coarse — ~2 large orders sharing ~200 shares on a
typical level. The hypothesis cannot be tested without a wider-spread stock.

### 3. Adverse selection

Both levels average ~+1.7¢/share adverse selection (80% of fills are adversely selected).
Only two features are significant predictors: `side_bid` (+0.29¢) and `time_frac` (−0.17¢).
The open is worst — the 09:34 bucket averages +2.5¢ vs ~+1.5¢ the rest of the day.

Counter-intuitively, L2 fills are *not* worse than L1 (p = 0.97): touch fills are hit by
the most aggressive informed takers; by the time flow reaches L2, the damaging move has
already occurred.

### 4. OFI — null result for fill prediction

Adding 10 s and 30 s OFI leaves OOS AUC unchanged at both levels. Short-window OFI is
fully arbitraged in AAPL before a passive resting order can benefit.

### 5. Survival analysis (Cox PH)

Median time-to-fill: L1 = **9.1 s**, L2 = **12.2 s**. The Cox model recovers the same four
predictors as logistic regression at L2, confirming they predict fill *speed* as well as
fill *probability*. OOS C-index degrades from 0.56 to 0.50 — same overfitting pattern as AUC.

---

## Repo structure

```
src/lob_sim/
  orderbook.py          Price-time priority LOB (FIFO, SortedDict)
  injection.py          Hypothetical order dataclass + injection simulator
  experiment.py         run_experiment(): stratified injection, outcome collection, OFI
  ofi.py                Vectorised OFI computation + windowed cumulative-sum lookup

notebooks/
  experiment_runner.py      Run full 5,000-injection experiment (set DEPTH_LEVEL)
  05_analysis.py            Fill probability: logistic, LightGBM, PDPs, L1 vs L2
  06_adverse_selection.py   Adverse selection regression + intraday patterns
  07_ofi_fill.py            OFI as fill predictor (AUC comparison)
  08_survival.py            Kaplan-Meier + Cox PH survival analysis

results/
  experiment_AAPL_2019-12-30_L1.parquet   4,343 orders, 23 columns incl. ofi_10s/30s
  experiment_AAPL_2019-12-30_L2.parquet   4,701 orders, 23 columns incl. ofi_10s/30s
  analysis/                               18 publication-quality plots

tests/                  ~40 unit and integration tests (pytest)
FINDING.md              Full quantitative findings with tables and limitations
```

---

## How to run

```bash
# Install dependencies (requires uv)
uv sync

# Re-run experiment (edit DEPTH_LEVEL in the script for L1 or L2)
uv run python notebooks/experiment_runner.py

# Analyses
uv run python notebooks/05_analysis.py
uv run python notebooks/06_adverse_selection.py
uv run python notebooks/07_ofi_fill.py
uv run python notebooks/08_survival.py

# Tests
uv run pytest
```

---

## Limitations

1. **Single ticker, single day.** AAPL on Dec 30 2019 — a low-volume holiday session. Findings may not generalise.
2. **Passive-shadow model.** Injected orders don't affect prices or attract strategic responses.
3. **Granularity variance near-zero in AAPL.** The hypothesis requires a different stock or regime.
4. **OOS split is one contiguous block.** Time-series k-fold would give more stable estimates.
5. **OFI is fully arbitraged at AAPL.** A slower-trading stock is the right testbed.

---

## Stack

Python 3.13 · Polars · NumPy · scikit-learn · LightGBM · lifelines · matplotlib
