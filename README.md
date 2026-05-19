# LOB Execution Simulator

Queue-position-aware limit order book execution simulator built on NASDAQ TotalView-ITCH 5.0 data. Reconstructs the full order book event-by-event, simulates placing hypothetical limit orders into historical book state, tracks queue position and fill outcomes, and produces an empirical finding about queue dynamics or adverse selection.

**Target finding:** "queue position dominates spread for fill economics when book imbalance > X"

## Motivation

Most retail and academic microstructure work treats execution as a black box. This project goes one level deeper — reconstructing the actual queue and asking: given where you sit in line, how does that interact with book imbalance and spread to determine whether you fill?

## Stack

- Python 3.13, [uv](https://github.com/astral-sh/uv) for package management
- `polars` for fast tabular processing
- `lightgbm` / logistic regression for interpretable modeling
- `pytest` for unit tests, `matplotlib` for visualization

## Data

NASDAQ TotalView-ITCH 5.0 — the raw binary exchange feed, parsed directly from `emi.nasdaq.com`. Filtered to AAPL (December 30, 2019). 1.6M order events covering extended hours (03:09–20:00 ET).

Data files are gitignored. To reproduce, download a `.NASDAQ_ITCH50.gz` file and run:

```bash
uv run python src/lob_sim/itch_parser.py <file.gz> AAPL data/raw 10
```

## Project Structure

```
src/lob_sim/
├── itch_parser.py   # ITCH 5.0 binary parser → LOBSTER-format CSVs
├── explore.py       # data exploration and sanity checks
├── reconstruct.py   # event-by-event LOB reconstruction
├── simulate.py      # hypothetical order placement + queue tracking
├── features.py      # book imbalance, spread, depth features
└── validate.py      # snapshot consistency checks

tests/               # unit tests for every public function
notebooks/           # analysis notebooks
results/             # figures and tables
```

## Phases

1. **Data acquisition** — parse ITCH feed, explore raw data ✓
2. **LOB reconstruction** — replay message file, validate against snapshots
3. **Order simulation** — place hypothetical orders, track queue position
4. **Feature engineering** — imbalance, spread, depth, time-of-day
5. **Analysis** — LightGBM + SHAP, one defensible empirical result
6. **Writeup** — reproducible notebook + results artifacts

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run pytest
```

## Key Data Notes

- Prices in ITCH are integers in units of $0.0001 (divide by 10,000 for dollars)
- Message types: 1=new limit, 2=partial cancel, 3=full cancel, 4=execution, 7=halt
- Direction: 1=buy, -1=sell
- Regular session: 34,200s–57,600s since midnight (09:30–16:00 ET)
- Stub orders at extreme prices ($400k+) exist in raw data — filter by realistic price range
