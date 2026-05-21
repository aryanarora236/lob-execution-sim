"""
Experiment: queue granularity as a predictor of time-to-fill.

Hypothesis
----------
At injection, Q shares are resting ahead of us, spread across K distinct
orders. The ratio K/Q ("granularity") measures how fragmented the queue is.
A high-granularity queue (many small orders) should drain more slowly because
each execution event consumes fewer shares. A low-granularity queue (a few
large orders) can clear in a handful of events.

Prediction: higher K/Q → longer time_to_first_fill, conditional on Q.

Sampling scheme
---------------
The trading day [09:35, 15:55] is divided into N equal-width windows.
One injection target timestamp is picked uniformly at random within each
window (fixed seed). At each target we inject one order at the touch —
alternating bid/ask — provided the top-of-book level has at least
`min_queue_shares` shares ahead. Thin levels are skipped.

Output
------
Parquet at results/experiment_<ticker>_<date>.parquet, one row per injected
order. Primary regression columns: queue_granularity_at_entry,
queue_position_at_entry, time_to_first_fill, filled.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from lob_sim.explore import find_lobster_pair, load_messages, parse_filename_meta
from lob_sim.injection import HypotheticalOrder, InjectionSimulator
from lob_sim.ofi import compute_ofi, windowed_ofi
from lob_sim.orderbook import LimitOrderBook

# Trading-day boundaries (seconds since midnight)
# 09:35 start gives 5-min warm-up after open; 15:55 stops 5 min before close.
_DAY_START = 34_500   # 09:35
_DAY_END   = 57_300   # 15:55


def run_experiment(
    data_dir: Path | str = Path("data/raw"),
    n_injections: int = 5_000,
    min_queue_shares: int = 100,
    order_size: int = 100,
    lifetime: float = 60.0,
    seed: int = 42,
    out_dir: Path | str = Path("results"),
    depth_level: int = 1,
) -> pl.DataFrame:
    """
    Replay one day's LOB, inject n_injections hypothetical orders, and return
    a flat DataFrame of per-order outcomes ready for regression analysis.

    Parameters
    ----------
    data_dir        : directory containing the LOBSTER message CSV
    n_injections    : target number of injected orders (some may be skipped
                      if the target level has depth < min_queue_shares)
    min_queue_shares: minimum shares at the injection level to accept an order
    order_size      : size of each hypothetical order (shares)
    lifetime        : max seconds before an order expires unfilled
    seed            : RNG seed for target-timestamp generation
    out_dir         : directory for parquet output
    depth_level     : 1 = inject at best price (touch); 2 = second-best price.
                      Higher levels have smaller, more fragmented queues and
                      lower fill rates — useful for testing granularity signal.

    Returns
    -------
    polars.DataFrame with one row per injected order
    """
    if depth_level not in (1, 2):
        raise ValueError(f"depth_level must be 1 or 2, got {depth_level}")
    data_dir = Path(data_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load messages ─────────────────────────────────────────────────────────
    msg_path, _ = find_lobster_pair(data_dir)
    meta     = parse_filename_meta(msg_path)
    messages = load_messages(msg_path)
    print(f"Loaded {len(messages):,} events from {msg_path.name}")

    # ── generate stratified injection targets ─────────────────────────────────
    # Divide day into n_injections equal windows; pick one timestamp per window.
    rng    = random.Random(seed)
    span   = _DAY_END - _DAY_START
    window = span / n_injections
    targets: list[tuple[int, float]] = []   # (injection_id, target_time)
    for i in range(n_injections):
        lo = _DAY_START + i * window
        targets.append((i, rng.uniform(lo, lo + window)))
    # targets are already in order by construction; sort defensively
    targets.sort(key=lambda x: x[1])

    # ── replay ────────────────────────────────────────────────────────────────
    book = LimitOrderBook()
    sim  = InjectionSimulator(book)

    injected_orders: list[HypotheticalOrder] = []
    skipped = 0
    ptr = 0   # index into targets

    # Accumulate best-bid/ask state after each event for OFI computation.
    _snap_ts:  list[float] = []
    _snap_bp:  list[int]   = []
    _snap_bs:  list[int]   = []
    _snap_ap:  list[int]   = []
    _snap_as:  list[int]   = []

    for event in messages.iter_rows(named=True):
        ts = float(event["time"])

        # Inject any targets whose time has arrived
        while ptr < len(targets):
            inj_id, target_ts = targets[ptr]
            if ts < target_ts:
                break

            # Alternate sides: even id → bid, odd id → ask
            side       = "bid" if inj_id % 2 == 0 else "ask"
            bp, bs, ap, as_ = book.top_of_book()

            # Skip if book not yet populated
            if bp == -1 or ap == -1:
                ptr += 1
                skipped += 1
                continue

            # Resolve injection price and available depth for the chosen level
            if depth_level == 1:
                price       = bp if side == "bid" else ap
                level_depth = bs if side == "bid" else as_
            else:
                snap = book.snapshot(levels=2)
                if side == "bid":
                    price       = snap["bid_price_2"]
                    level_depth = snap["bid_size_2"]
                else:
                    price       = snap["ask_price_2"]
                    level_depth = snap["ask_size_2"]
                # Level 2 may not exist (sentinel = -1)
                if price == -1:
                    ptr += 1
                    skipped += 1
                    continue

            # Skip if the injection level is too thin
            if level_depth < min_queue_shares:
                ptr += 1
                skipped += 1
                continue
            order = HypotheticalOrder(
                order_id=inj_id,
                side=side,
                price=price,
                size=order_size,
                entry_timestamp=ts,
                max_lifetime=lifetime,
            )
            sim.inject(order)
            injected_orders.append(order)
            ptr += 1

        sim.process_event(event)

        # Record best-bid/ask after this event for OFI series.
        _bp, _bs, _ap, _as = book.top_of_book()
        _snap_ts.append(ts)
        _snap_bp.append(_bp)
        _snap_bs.append(_bs)
        _snap_ap.append(_ap)
        _snap_as.append(_as)

    # ── compute OFI series ────────────────────────────────────────────────────
    snap_ts  = np.array(_snap_ts,  dtype=float)
    ofi_vals = compute_ofi(
        np.array(_snap_bp, dtype=float),
        np.array(_snap_bs, dtype=float),
        np.array(_snap_ap, dtype=float),
        np.array(_snap_as, dtype=float),
    )
    ofi_cum = np.cumsum(ofi_vals)

    # ── collect outcomes ──────────────────────────────────────────────────────
    rows: list[dict[str, Any]] = []
    for order in injected_orders:
        out = sim.compute_outcomes(order)
        rows.append({
            "injection_id":    order.order_id,
            "ticker":          meta["ticker"],
            "date":            meta["date"],
            "entry_timestamp": order.entry_timestamp,
            "side":            order.side,
            "price":           order.price,
            "order_size":      order.size,
            "status":          order.status,
            "ofi_10s":         windowed_ofi(snap_ts, ofi_cum, order.entry_timestamp, 10.0),
            "ofi_30s":         windowed_ofi(snap_ts, ofi_cum, order.entry_timestamp, 30.0),
            **out,
        })

    df = pl.DataFrame(rows)

    # ── save ──────────────────────────────────────────────────────────────────
    fname = f"experiment_{meta['ticker']}_{meta['date']}_L{depth_level}.parquet"
    fpath = out_dir / fname
    df.write_parquet(fpath)

    n_total  = len(df)
    n_filled = int(df["filled"].sum())
    print(
        f"\nExperiment complete"
        f"\n  injected : {n_total:,}  (skipped {skipped:,} thin levels)"
        f"\n  filled   : {n_filled:,} ({n_filled / n_total:.1%})"
        f"\n  seed     : {seed}"
        f"\n  output   : {fpath}"
    )
    return df
