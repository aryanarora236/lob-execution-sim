"""
LOB reconstruction validator.

Replays the message file event-by-event through LimitOrderBook and checks
that each resulting snapshot matches the corresponding row in the orderbook
file. This is the acceptance gate for Phase 2 — nothing downstream is
trustworthy if reconstruction disagrees with the ground truth.

Usage:
    uv run python src/lob_sim/replay.py data/raw
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from lob_sim.explore import find_lobster_pair, load_messages, load_orderbook, parse_filename_meta
from lob_sim.orderbook import LimitOrderBook

LEVELS = 10

# Column order in the orderbook file (matches itch_parser output)
_OB_COLS: list[str] = [
    col
    for lvl in range(1, LEVELS + 1)
    for col in (f"ask_price_{lvl}", f"ask_size_{lvl}", f"bid_price_{lvl}", f"bid_size_{lvl}")
]


@dataclass
class MismatchDetail:
    """One mismatch record for diagnostic output."""

    event_idx: int
    event_type: int
    order_id: int
    expected: list[int]  # ground-truth flat snapshot
    actual: list[int]    # reconstructor flat snapshot
    differing_cols: list[str]  # column names where values differ


@dataclass
class ReplayResult:
    """Full validation report."""

    ticker: str
    date: str
    total_events: int
    total_mismatches: int
    agreement_rate: float
    mismatches_by_type: dict[int, int]        # event_type → mismatch count
    mismatches_early: int                      # first 10 % of events
    mismatches_late: int                       # last 10 % of events
    first_mismatches: list[MismatchDetail] = field(default_factory=list)  # up to 20

    def print_report(self) -> None:
        """Print a human-readable validation report."""
        print("=" * 65)
        print("LOB RECONSTRUCTION VALIDATION REPORT")
        print("=" * 65)
        print(f"  Ticker : {self.ticker}")
        print(f"  Date   : {self.date}")
        print(f"  Events : {self.total_events:,}")
        print(f"  Mismatches : {self.total_mismatches:,}")
        print(f"  Agreement  : {self.agreement_rate * 100:.4f}%")

        thresh = 99.9
        status = "PASS ✓" if self.agreement_rate * 100 >= thresh else f"FAIL ✗ (need ≥{thresh}%)"
        print(f"  Status     : {status}")

        print(f"\n{'─'*65}")
        print("MISMATCHES BY EVENT TYPE")
        print(f"{'─'*65}")
        for etype, count in sorted(self.mismatches_by_type.items()):
            pct = count / self.total_mismatches * 100 if self.total_mismatches else 0
            print(f"  type {etype}: {count:>6,}  ({pct:.1f}% of mismatches)")

        cutoff10 = self.total_events // 10
        print(f"\n{'─'*65}")
        print("MISMATCH DISTRIBUTION (early vs. late)")
        print(f"{'─'*65}")
        print(f"  First 10% (events 0–{cutoff10:,})  : {self.mismatches_early:,} mismatches")
        print(f"  Last  10% (events {self.total_events - cutoff10:,}–end): {self.mismatches_late:,} mismatches")

        if self.first_mismatches:
            print(f"\n{'─'*65}")
            print(f"FIRST {len(self.first_mismatches)} MISMATCH DETAILS")
            print(f"{'─'*65}")
            for m in self.first_mismatches:
                print(f"\n  Event #{m.event_idx}  type={m.event_type}  order_id={m.order_id}")
                print(f"  Differing columns: {m.differing_cols}")
                for col in m.differing_cols[:4]:
                    idx = _OB_COLS.index(col)
                    print(f"    {col}: expected={m.expected[idx]}  actual={m.actual[idx]}")

        print(f"\n{'='*65}")


def validate_reconstruction(
    data_dir: Path = Path("data/raw"),
    levels: int = LEVELS,
) -> ReplayResult:
    """
    Replay the message file through LimitOrderBook and compare every snapshot
    to the ground-truth orderbook file.

    Parameters
    ----------
    data_dir:
        Directory containing the LOBSTER-format CSVs.
    levels:
        Number of book levels to validate (must match the orderbook file).

    Returns
    -------
    ReplayResult with full diagnostic information.
    """
    msg_path, ob_path = find_lobster_pair(data_dir)
    meta = parse_filename_meta(msg_path)

    print(f"Loading {msg_path.name} ...", end=" ", flush=True)
    messages = load_messages(msg_path)
    print(f"{len(messages):,} events")

    print(f"Loading {ob_path.name} ...", end=" ", flush=True)
    orderbook = load_orderbook(ob_path, levels)
    print(f"{len(orderbook):,} snapshots")

    assert len(messages) == len(orderbook), (
        f"Row count mismatch: {len(messages)} messages vs {len(orderbook)} snapshots"
    )

    # Convert ground-truth to a flat numpy array for fast row access
    ob_array = orderbook.select(_OB_COLS[:levels * 4]).to_numpy()

    book = LimitOrderBook()
    n = len(messages)
    cutoff10 = n // 10

    total_mismatches = 0
    mismatches_by_type: dict[int, int] = {}
    mismatches_early = 0
    mismatches_late = 0
    first_mismatches: list[MismatchDetail] = []

    print(f"Replaying {n:,} events...", flush=True)
    t0 = time.time()

    for i, event in enumerate(messages.iter_rows(named=True)):
        book.apply_event(event)
        actual = book.snapshot_flat(levels)
        expected = ob_array[i].tolist()

        if actual != expected:
            total_mismatches += 1
            etype = int(event["event_type"])
            mismatches_by_type[etype] = mismatches_by_type.get(etype, 0) + 1

            if i < cutoff10:
                mismatches_early += 1
            if i >= n - cutoff10:
                mismatches_late += 1

            if len(first_mismatches) < 20:
                differing = [
                    _OB_COLS[j]
                    for j in range(levels * 4)
                    if actual[j] != expected[j]
                ]
                first_mismatches.append(MismatchDetail(
                    event_idx=i,
                    event_type=etype,
                    order_id=int(event["order_id"]),
                    expected=expected,
                    actual=actual,
                    differing_cols=differing,
                ))

        if (i + 1) % 200_000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  {i+1:,} / {n:,}  ({rate:,.0f} ev/s)  mismatches so far: {total_mismatches}", flush=True)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s  ({n/elapsed:,.0f} events/s)")

    agreement = (n - total_mismatches) / n

    return ReplayResult(
        ticker=meta["ticker"],
        date=meta["date"],
        total_events=n,
        total_mismatches=total_mismatches,
        agreement_rate=agreement,
        mismatches_by_type=mismatches_by_type,
        mismatches_early=mismatches_early,
        mismatches_late=mismatches_late,
        first_mismatches=first_mismatches,
    )


if __name__ == "__main__":
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
    result = validate_reconstruction(data_dir)
    result.print_report()
