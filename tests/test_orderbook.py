"""Unit tests for LimitOrderBook."""

from __future__ import annotations

import pytest

from lob_sim.orderbook import LimitOrderBook


def _ev(event_type: int, order_id: int, size: int, price: int, direction: int) -> dict:
    return {
        "event_type": event_type,
        "order_id": order_id,
        "size": size,
        "price": price,
        "direction": direction,
        "time": 34200.0,
    }


# ── snapshot format ───────────────────────────────────────────────────────────


def test_snapshot_column_layout() -> None:
    """Snapshot dict has the exact LOBSTER column layout."""
    book = LimitOrderBook()
    snap = book.snapshot(levels=3)
    expected_keys = [
        "ask_price_1", "ask_size_1", "bid_price_1", "bid_size_1",
        "ask_price_2", "ask_size_2", "bid_price_2", "bid_size_2",
        "ask_price_3", "ask_size_3", "bid_price_3", "bid_size_3",
    ]
    assert list(snap.keys()) == expected_keys


def test_empty_book_snapshot_sentinels() -> None:
    """Empty book fills all levels with price=-1, size=0."""
    book = LimitOrderBook()
    snap = book.snapshot(levels=2)
    for lvl in (1, 2):
        assert snap[f"ask_price_{lvl}"] == -1
        assert snap[f"ask_size_{lvl}"] == 0
        assert snap[f"bid_price_{lvl}"] == -1
        assert snap[f"bid_size_{lvl}"] == 0


def test_snapshot_flat_matches_dict() -> None:
    """snapshot_flat and snapshot return consistent data."""
    book = LimitOrderBook()
    book.apply_event(_ev(1, 1, 100, 1_000_000, 1))
    book.apply_event(_ev(1, 2, 50, 1_010_000, -1))
    snap_d = book.snapshot(levels=2)
    snap_f = book.snapshot_flat(levels=2)

    cols = [
        "ask_price_1", "ask_size_1", "bid_price_1", "bid_size_1",
        "ask_price_2", "ask_size_2", "bid_price_2", "bid_size_2",
    ]
    assert snap_f == [snap_d[c] for c in cols]


# ── synthetic 10-event sequence with hand-computed snapshots ──────────────────


def test_synthetic_sequence() -> None:
    """
    10-event sequence verified by hand.

    Prices are in raw LOBSTER units ($ × 10_000):
      1_000_000 = $100.00
      1_010_000 = $101.00
      1_005_000 = $100.50
        995_000 = $ 99.50
    """
    book = LimitOrderBook()

    # 1. Add bid $100.00 size=100 order=1
    book.apply_event(_ev(1, 1, 100, 1_000_000, 1))
    s = book.snapshot(1)
    assert s["bid_price_1"] == 1_000_000
    assert s["bid_size_1"] == 100
    assert s["ask_price_1"] == -1

    # 2. Add ask $101.00 size=50 order=2
    book.apply_event(_ev(1, 2, 50, 1_010_000, -1))
    s = book.snapshot(1)
    assert s["ask_price_1"] == 1_010_000
    assert s["ask_size_1"] == 50

    # 3. Add bid $100.00 size=200 order=3 — same level as order 1
    book.apply_event(_ev(1, 3, 200, 1_000_000, 1))
    s = book.snapshot(1)
    assert s["bid_price_1"] == 1_000_000
    assert s["bid_size_1"] == 300   # 100 + 200

    # 4. Partial cancel order 1 by 30 shares (size field = shares removed)
    book.apply_event(_ev(2, 1, 30, 1_000_000, 1))
    s = book.snapshot(1)
    assert s["bid_size_1"] == 270   # (100 − 30) + 200 = 70 + 200

    # 5. Full cancel order 2 (ask level disappears)
    book.apply_event(_ev(3, 2, 50, 1_010_000, -1))
    s = book.snapshot(1)
    assert s["ask_price_1"] == -1
    assert s["ask_size_1"] == 0

    # 6. Add ask $100.50 size=75 order=4
    book.apply_event(_ev(1, 4, 75, 1_005_000, -1))
    s = book.snapshot(1)
    assert s["ask_price_1"] == 1_005_000
    assert s["ask_size_1"] == 75

    # 7. Visible execution of order 4 by 75 (fully consumed; ask level disappears)
    book.apply_event(_ev(4, 4, 75, 1_005_000, -1))
    s = book.snapshot(1)
    assert s["ask_price_1"] == -1

    # 8. Execution of order 1 by its remaining 70 shares
    book.apply_event(_ev(4, 1, 70, 1_000_000, 1))
    s = book.snapshot(1)
    assert s["bid_size_1"] == 200   # only order 3 remains

    # 9. Full cancel order 3 (bid side empties)
    book.apply_event(_ev(3, 3, 200, 1_000_000, 1))
    s = book.snapshot(1)
    assert s["bid_price_1"] == -1
    assert len(book) == 0

    # 10. Rebuild two-level book — verify multi-level snapshot
    book.apply_event(_ev(1, 5, 25, 995_000, 1))    # bid level 1
    book.apply_event(_ev(1, 6, 10, 990_000, 1))    # bid level 2
    book.apply_event(_ev(1, 7, 30, 1_000_000, -1)) # ask level 1
    book.apply_event(_ev(1, 8, 40, 1_010_000, -1)) # ask level 2
    s = book.snapshot(3)
    assert s["bid_price_1"] == 995_000    # highest bid first
    assert s["bid_size_1"] == 25
    assert s["bid_price_2"] == 990_000
    assert s["bid_size_2"] == 10
    assert s["bid_price_3"] == -1         # only 2 levels exist
    assert s["ask_price_1"] == 1_000_000  # lowest ask first
    assert s["ask_size_1"] == 30
    assert s["ask_price_2"] == 1_010_000
    assert s["ask_size_2"] == 40
    assert s["ask_price_3"] == -1


# ── hidden execution ──────────────────────────────────────────────────────────


def test_hidden_execution_does_not_touch_visible_book() -> None:
    """Type-5 events must not change the visible book state."""
    book = LimitOrderBook()
    book.apply_event(_ev(1, 1, 100, 1_000_000, 1))
    book.apply_event(_ev(1, 2, 50, 1_010_000, -1))
    snap_before = book.snapshot()

    # Fire a type-5 event referencing order 2 (hidden execution)
    book.apply_event(_ev(5, 2, 30, 1_010_000, -1))

    snap_after = book.snapshot()
    assert snap_after == snap_before  # book unchanged


# ── partial cancel ────────────────────────────────────────────────────────────


def test_partial_cancel_removes_shares_not_sets_size() -> None:
    """Type-2 size field is shares REMOVED, not the new remaining size."""
    book = LimitOrderBook()
    book.apply_event(_ev(1, 1, 200, 1_000_000, 1))

    # Cancel 80 shares — order should have 120 remaining
    book.apply_event(_ev(2, 1, 80, 1_000_000, 1))
    s = book.snapshot(1)
    assert s["bid_size_1"] == 120

    # Cancel another 50 — order should have 70 remaining
    book.apply_event(_ev(2, 1, 50, 1_000_000, 1))
    s = book.snapshot(1)
    assert s["bid_size_1"] == 70

    assert len(book) == 1  # order still alive


def test_partial_cancel_to_zero_removes_order() -> None:
    """Cancelling all remaining shares should fully remove the order."""
    book = LimitOrderBook()
    book.apply_event(_ev(1, 1, 100, 1_000_000, 1))
    book.apply_event(_ev(2, 1, 100, 1_000_000, 1))
    assert len(book) == 0
    s = book.snapshot(1)
    assert s["bid_price_1"] == -1


# ── price-time priority ───────────────────────────────────────────────────────


def test_price_time_priority_queue_order() -> None:
    """Same-price orders are queued in arrival order; best price wins across levels."""
    book = LimitOrderBook()
    book.apply_event(_ev(1, 1, 100, 1_010_000, -1))  # ask $101, first arrival
    book.apply_event(_ev(1, 2, 50, 1_010_000, -1))   # ask $101, second arrival
    book.apply_event(_ev(1, 3, 75, 1_000_000, -1))   # ask $100, best price

    s = book.snapshot(2)
    assert s["ask_price_1"] == 1_000_000  # best ask first
    assert s["ask_size_1"] == 75
    assert s["ask_price_2"] == 1_010_000  # second level
    assert s["ask_size_2"] == 150         # order 1 + order 2

    # Cancel order 1 (first at 101); order 2 still rests there
    book.apply_event(_ev(3, 1, 100, 1_010_000, -1))
    s = book.snapshot(2)
    assert s["ask_size_2"] == 50


# ── halt and cross events ─────────────────────────────────────────────────────


def test_halt_and_cross_do_not_modify_book() -> None:
    """Types 6 and 7 must leave the book unchanged."""
    book = LimitOrderBook()
    book.apply_event(_ev(1, 1, 100, 1_000_000, 1))
    snap_before = book.snapshot()

    book.apply_event(_ev(7, 0, 0, 0, 0))  # halt
    book.apply_event(_ev(6, 0, 0, 0, 0))  # cross

    assert book.snapshot() == snap_before


# ── len ───────────────────────────────────────────────────────────────────────


def test_len_counts_orders_not_levels() -> None:
    """__len__ returns number of individual resting orders."""
    book = LimitOrderBook()
    assert len(book) == 0
    book.apply_event(_ev(1, 1, 100, 1_000_000, 1))
    book.apply_event(_ev(1, 2, 100, 1_000_000, 1))  # same price, different order
    book.apply_event(_ev(1, 3, 50, 1_010_000, -1))
    assert len(book) == 3
    book.apply_event(_ev(3, 1, 100, 1_000_000, 1))  # cancel one
    assert len(book) == 2
