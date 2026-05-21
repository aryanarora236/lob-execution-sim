"""
Full experiment run: 5 000 hypothetical orders injected across the AAPL day.

Run from project root:
    uv run python notebooks/experiment_runner.py

Output: results/experiment_AAPL_<date>.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl

from lob_sim.experiment import run_experiment

N_INJECTIONS  = 5_000
MIN_QUEUE     = 100     # skip levels with < 100 shares at touch
ORDER_SIZE    = 100     # shares per hypothetical order
LIFETIME      = 60.0    # seconds before expiry
SEED          = 42


def main() -> None:
    df = run_experiment(
        data_dir=Path("data/raw"),
        n_injections=N_INJECTIONS,
        min_queue_shares=MIN_QUEUE,
        order_size=ORDER_SIZE,
        lifetime=LIFETIME,
        seed=SEED,
        out_dir=Path("results"),
    )

    filled   = df.filter(pl.col("filled"))
    unfilled = df.filter(~pl.col("filled"))

    print("\n── Summary ─────────────────────────────────────────────────")
    print(f"  Total injected      : {len(df):,}")
    print(f"  Filled              : {len(filled):,}  ({len(filled)/len(df):.1%})")
    print(f"  Expired / live      : {len(unfilled):,}")
    print(f"  Side split (bid)    : {(df['side'] == 'bid').sum():,} / {len(df):,}")

    print("\n── Queue granularity (all injections) ──────────────────────")
    print(df["queue_granularity_at_entry"].describe())

    print("\n── Time to first fill (filled orders only) ─────────────────")
    print(filled["time_to_first_fill"].describe())

    print("\n── Granularity by fill outcome ─────────────────────────────")
    summary = (
        df.group_by("filled")
        .agg(
            pl.col("queue_granularity_at_entry").mean().alias("mean_granularity"),
            pl.col("queue_position_at_entry").mean().alias("mean_queue_shares"),
            pl.col("orders_ahead_at_entry").mean().alias("mean_orders_ahead"),
            pl.len().alias("n"),
        )
        .sort("filled")
    )
    print(summary)


if __name__ == "__main__":
    main()
