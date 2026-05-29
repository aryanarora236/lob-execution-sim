# LOB Execution Simulator

A queue-position-aware limit order book simulator built on real NASDAQ market
data. Injects hypothetical passive limit orders into a replayed historical order
book and measures fill probability, fill speed, and adverse selection as
functions of observable market state across multiple tickers and sessions.

---

## Motivation

Passive limit orders are cheaper than market orders but uncertain — you might
not fill. The key questions:

1. **Fill probability:** can observable state (spread, imbalance, queue depth, time of day, OFI) predict whether a passive order fills within its lifetime?
2. **Touch vs depth:** does predictability differ between posting at the best price (L1) and one tick behind it (L2)?
3. **Cross-ticker generalisability:** do the same features matter in a liquid HFT-dominated stock (AAPL) and a wider-spread, less-efficient stock (INTC)?
4. **Adverse selection:** when your order fills, how much does the mid-price move against you in the next second?
5. **Optimal placement:** given empirical fill rates and adverse selection, when should you post at L1 vs L2?

---

## Data

Four ticker-date combinations. Raw data is **not** included in the repo —
place LOBSTER CSVs in `data/raw/` or run the ITCH parser to generate them.

| Ticker | Date | Session | Events |
|--------|------|---------|--------|
| AAPL | 2019-12-30 | Holiday week (low volume) | 1,609,148 |
| AAPL | 2019-08-30 | Normal | 1,283,342 |
| MSFT | 2019-08-30 | Normal | 1,642,017 |
| INTC | 2019-08-30 | Normal | 1,022,326 |

**LOBSTER format:** AAPL 2019-12-30 downloaded from lobsterdata.com (free
sample, Level 10). All others extracted from the public NASDAQ TotalView-ITCH
5.0 feed using the included parser.

---

## Method

**Passive-shadow model.** Hypothetical orders are injected into the historical
event stream without affecting prices or displacing existing resting orders.
Each order is placed at L1 or L2, tracked through the replayed book, and
marked filled when real executions at its price level clear through its queue
position. No market impact, no strategic response — pure counterfactual.

**Stratified injection.** The trading day (09:35–15:55) is divided into N
equal windows; one timestamp is drawn uniformly within each window, giving
uniform temporal coverage with ~5,000 orders per ticker-date.

**OFI.** Order Flow Imbalance (Cont, Kukanov, Stoikov 2014) computed from
consecutive best-bid/ask snapshots at 10 s and 30 s look-back windows.

---

## Key findings

Full quantitative results with tables and limitations are in [`FINDING.md`](FINDING.md).

### 1. Fill predictability is regime-dependent

| Ticker | L1 fill rate | L2 fill rate | L1 AUC | L2 AUC |
|--------|-------------|-------------|--------|--------|
| AAPL (pooled) | 65.1% | 57.7% | 0.519 | 0.529 |
| INTC | 55.7% | **31.7%** | 0.531 | 0.498 |
| MSFT | 68.4% | 57.9% | 0.471 | 0.460 |

In AAPL, only spread is a significant L2 predictor. In INTC, queue position
(coef −0.110 ***) and time_frac (coef −0.367 ***) dominate, and Cox C-index
reaches 0.613 vs 0.536 for AAPL. Fill timing in wider-spread stocks is
substantially more predictable than in liquid HFT-dominated ones.

### 2. OFI is arbitraged in AAPL but not in INTC or MSFT

| Ticker | OFI_30s coef | 95% CI | Significant? |
|--------|-------------|--------|--------------|
| AAPL | +0.009 | [−0.046, +0.063] | No |
| INTC | +0.202 | [+0.120, +0.280] | **Yes** |
| MSFT | +0.168 | [+0.080, +0.258] | **Yes** |

Short-window OFI is fully absorbed by HFTs in AAPL before resting passive
orders benefit. In less liquid stocks, buying pressure propagates to L2 within
the 60 s order lifetime.

### 3. Optimal placement crossover varies dramatically by ticker

| Ticker | C\* (crossover) | Prefer at C = 10¢ |
|--------|-----------------|-------------------|
| AAPL | 8.96¢ | L1 |
| MSFT | 6.03¢ | L1 |
| INTC | **1.64¢** | L1 |

INTC's 24pp fill-rate gap between L1 and L2 makes L2 posting worthwhile only
at unrealistically low urgency. AAPL's "L2 almost always preferred" conclusion
does not generalise.

### 4. Queue granularity — null result (holds across all tickers)

Median K/Q = 0.01 at both levels for all three tickers with near-zero variance.
The hypothesis that fragmented queues fill differently could not be tested
— there is insufficient spread in the independent variable.

### 5. Panel regression identifies robust cross-ticker predictors

Significant across all 18,629 pooled L2 orders (bootstrap CIs):
`queue_position` (−0.190 ***), `spread` (−0.159 ***), `OFI_30s` (+0.138 ***),
`time_frac` (−0.094 ***), `side_bid` (+0.067 ***). Granularity not significant.

---

## Repo structure

```
src/lob_sim/
  orderbook.py          Price-time priority LOB (FIFO queue per level)
  injection.py          Hypothetical order dataclass + injection simulator
  experiment.py         run_experiment() / run_all_experiments()
  explore.py            LOBSTER file loading and discovery
  itch_parser.py        NASDAQ TotalView-ITCH 5.0 binary → LOBSTER CSV
  ofi.py                Vectorised OFI + windowed cumulative-sum lookup
  features.py           Feature engineering helpers
  replay.py             Event replay engine
  validate.py           LOB reconstruction validation

notebooks/
  experiment_runner.py      Run experiments for all files in data/raw/
  05_analysis.py            Fill probability: logistic, LightGBM, PDPs
  06_adverse_selection.py   Adverse selection regression + intraday pattern
  07_ofi_fill.py            OFI as fill predictor (AUC comparison)
  08_survival.py            Kaplan-Meier + Cox PH survival analysis
  09_optimal_placement.py   L1 vs L2 implementation shortfall framework
  10_deep_angles.py         Fill speed vs adverse selection; LGB time_frac ablation
  11_multi_ticker.py        Cross-ticker AUC and coefficient comparison
  12_extended_analysis.py   Bootstrap CIs, LGB ablation, adverse selection,
                            optimal placement, survival, panel regression, OFI
                            — all analyses run cross-ticker

results/
  experiment_<TICKER>_<DATE>_L{1,2}.parquet   Per-file experiment output
  experiment_all_L{1,2}.parquet               Pooled across all ticker-dates
  extended/                                   Plots from 12_extended_analysis.py
  multi_ticker/                               Plots from 11_multi_ticker.py

tests/                  Unit and integration tests (pytest)
FINDING.md              Full quantitative findings, tables, and limitations
run.sh                  Wrapper script (sets PYTHONPATH, uses project venv)
```

---

## How to run

```bash
# Install dependencies
uv sync

# Extract tickers from a downloaded ITCH file (keep source for multiple tickers)
./run.sh -m lob_sim.itch_parser data/raw/08302019.NASDAQ_ITCH50.gz AAPL data/raw
./run.sh -m lob_sim.itch_parser data/raw/08302019.NASDAQ_ITCH50.gz MSFT data/raw
./run.sh -m lob_sim.itch_parser data/raw/08302019.NASDAQ_ITCH50.gz INTC data/raw --delete

# Run experiments for all files in data/raw/ (set DEPTH_LEVEL=1 for touch)
DEPTH_LEVEL=2 ./run.sh notebooks/experiment_runner.py

# Analyses (single-ticker AAPL)
./run.sh notebooks/05_analysis.py
./run.sh notebooks/06_adverse_selection.py
./run.sh notebooks/07_ofi_fill.py
./run.sh notebooks/08_survival.py
./run.sh notebooks/09_optimal_placement.py
./run.sh notebooks/10_deep_angles.py

# Multi-ticker analyses
./run.sh notebooks/11_multi_ticker.py
./run.sh notebooks/12_extended_analysis.py

# Tests
uv run pytest
```

---

## Limitations

1. **Passive-shadow model.** Injected orders don't affect prices or attract strategic responses.
2. **Three tickers, two dates.** MSFT showed severe intraday non-stationarity on 2019-08-30. A larger panel is needed to separate persistent regime effects from session noise.
3. **Granularity variance near-zero.** All three tickers have K/Q clustered at 0.01. Testing the granularity hypothesis requires a smaller-cap or pre-decimalization stock.
4. **OOS split is one contiguous block.** Time-series k-fold cross-validation would give more stable AUC estimates.
5. **Adverse selection measured at 1 s only.** A longer horizon might surface stronger signal in INTC where OFI decays more slowly.

---

## Stack

Python 3.13 · Polars · NumPy · scikit-learn · LightGBM · lifelines · matplotlib
