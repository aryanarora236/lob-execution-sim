# LOB Execution Simulator — Project Context

## What This Is
Queue-position-aware limit order book execution simulator. Reconstruct event-by-event NASDAQ LOBs from public LOBSTER data, simulate placing hypothetical orders into the historical book, track queue position and fill outcomes, and produce one defensible empirical finding about queue dynamics or adverse selection.

Example target finding: "queue position dominates spread for fill economics when book imbalance > X"

## Stack
- Python with **polars** (not pandas)
- **uv** for package management
- numpy, polars, matplotlib, scikit-learn, lightgbm, jupyter, pytest
- Type hints required on all functions
- No global state
- Docstrings + unit tests on every public function

## Workflow Rules (non-negotiable)
1. Propose plan and wait for approval before writing code for a new phase
2. Validate LOB reconstruction against LOBSTER ground-truth snapshots before anything downstream — everything is worthless if reconstruction is broken
3. Show data (schemas, distributions, sanity checks) before building on top of it
4. Push back if corners are being cut on validation or complexity is being chased over clarity

## Out of Scope
- Live trading, real money, broker integration
- HFT-style latency optimization (Python is fine; correctness is the point)
- Options, futures, FX — equities only
- Deep learning models (LightGBM or logistic regression — interpretable beats fancy)

## Project Phases (planned)
1. **Setup** — project scaffold, uv env, .gitignore
2. **LOB Reconstruction** — parse LOBSTER message + orderbook files, replay event-by-event, validate against snapshots
3. **Order Simulation** — place hypothetical limit orders into historical book, track queue position
4. **Fill Outcome Tracking** — determine if/when/at what price simulated orders fill
5. **Feature Engineering** — book imbalance, spread, queue depth, time-of-day, etc.
6. **Analysis / Finding** — LightGBM or logistic regression, SHAP values, one defensible empirical result
7. **Writeup** — notebook + results/ artifacts suitable for resume/interview discussion

## Data
LOBSTER format (https://lobsterdata.com/):
- `*_message_*.csv` — order events (timestamp, type, order_id, size, price, direction)
- `*_orderbook_*.csv` — top-N LOB snapshots after each event (ground-truth for validation)

Data lives in `data/raw/` (gitignored). Processed outputs go in `data/processed/`.

## Key Invariants
- Price levels in LOBSTER are in integer cents (divide by 10000 for dollars)
- Message types: 1=new limit, 2=partial cancel, 3=full cancel, 4=exec visible, 5=exec hidden, 7=trading halt
- Direction: -1=sell, 1=buy
- Book is always sorted: bids descending, asks ascending
