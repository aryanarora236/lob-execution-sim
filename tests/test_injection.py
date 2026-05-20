"""Unit tests for InjectionSimulator and HypotheticalOrder."""

from __future__ import annotations

from lob_sim.injection import HypotheticalOrder, InjectionSimulator
from lob_sim.orderbook import LimitOrderBook


# ── helpers ───────────────────────────────────────────────────────────────────


def _ev(event_type: int, order_id: int, size: int, price: int, direction: int,
        time: float = 34200.0) -> dict:
    return {
        "event_type": event_type, "order_id": order_id,
        "size": size, "price": price, "direction": direction, "time": time,
    }


def _make_sim() -> tuple[LimitOrderBook, InjectionSimulator]:
    book = LimitOrderBook()
    sim = InjectionSimulator(book)
    return book, sim


# ── queue granularity at entry ────────────────────────────────────────────────


def test_queue_granularity_multiple_orders() -> None:
    """Three orders totalling 200 shares → granularity = 3/200."""
    book, sim = _make_sim()

    sim.process_event(_ev(1, 1, 100, 1_000_000, 1, time=34200.0))
    sim.process_event(_ev(1, 2,  50, 1_000_000, 1, time=34200.1))
    sim.process_event(_ev(1, 3,  50, 1_000_000, 1, time=34200.2))

    order = HypotheticalOrder(
        order_id=-1, side="bid", price=1_000_000,
        size=10, entry_timestamp=34200.3,
    )
    sim.inject(order)

    assert order.orders_ahead_at_entry == 3
    assert order.queue_position_at_entry == 200
    assert abs(order.queue_granularity_at_entry - 3 / 200) < 1e-9


def test_queue_granularity_single_large_order() -> None:
    """One order of 200 shares → granularity = 1/200."""
    book, sim = _make_sim()

    sim.process_event(_ev(1, 1, 200, 1_000_000, 1, time=34200.0))

    order = HypotheticalOrder(
        order_id=-1, side="bid", price=1_000_000,
        size=10, entry_timestamp=34200.1,
    )
    sim.inject(order)

    assert order.orders_ahead_at_entry == 1
    assert abs(order.queue_granularity_at_entry - 1 / 200) < 1e-9


def test_queue_granularity_empty_level() -> None:
    """Empty queue → granularity is 0.0 (not a division by zero)."""
    book, sim = _make_sim()

    order = HypotheticalOrder(
        order_id=-1, side="bid", price=1_000_000,
        size=10, entry_timestamp=34200.0,
    )
    sim.inject(order)

    assert order.orders_ahead_at_entry == 0
    assert order.queue_granularity_at_entry == 0.0


def test_compute_outcomes_includes_granularity() -> None:
    """compute_outcomes must return orders_ahead_at_entry and queue_granularity_at_entry."""
    book, sim = _make_sim()

    sim.process_event(_ev(1, 1, 100, 1_000_000, 1, time=34200.0))
    sim.process_event(_ev(1, 2,  50, 1_000_000, 1, time=34200.1))

    order = HypotheticalOrder(
        order_id=-1, side="bid", price=1_000_000,
        size=10, entry_timestamp=34200.2, max_lifetime=1.0,
    )
    sim.inject(order)
    sim.process_event(_ev(1, 99, 5, 999_000, 1, time=34201.3))  # expire it

    out = sim.compute_outcomes(order)
    assert out["orders_ahead_at_entry"] == 2
    assert out["queue_position_at_entry"] == 150
    assert abs(out["queue_granularity_at_entry"] - 2 / 150) < 1e-9


# ── queue position at entry ───────────────────────────────────────────────────


def test_queue_position_at_entry_equals_existing_bid_size() -> None:
    """
    Inject a buy order at the current best bid.
    queue_position_at_entry must equal the total visible size at that level.
    """
    book, sim = _make_sim()

    # Two existing bids at 1_000_000 totalling 300 shares
    sim.process_event(_ev(1, 1, 100, 1_000_000, 1, time=34200.0))
    sim.process_event(_ev(1, 2, 200, 1_000_000, 1, time=34200.1))

    order = HypotheticalOrder(
        order_id=-1, side="bid", price=1_000_000,
        size=50, entry_timestamp=34200.2,
    )
    sim.inject(order)

    assert order.queue_position_at_entry == 300
    assert order.queue_position_current == 300


def test_queue_position_zero_when_level_empty() -> None:
    """Injecting at an empty level means we're immediately at the front."""
    book, sim = _make_sim()

    order = HypotheticalOrder(
        order_id=-1, side="bid", price=1_000_000,
        size=50, entry_timestamp=34200.0,
    )
    sim.inject(order)

    assert order.queue_position_at_entry == 0
    assert order.queue_position_current == 0


# ── queue decrements on execution ─────────────────────────────────────────────


def test_execution_decrements_queue_position() -> None:
    """
    Synthesize an execution that partially consumes the order ahead of us.
    queue_position_current should decrement by the executed size.
    """
    book, sim = _make_sim()

    sim.process_event(_ev(1, 1, 200, 1_000_000, 1, time=34200.0))  # 200 shares ahead

    order = HypotheticalOrder(
        order_id=-1, side="bid", price=1_000_000,
        size=100, entry_timestamp=34200.1,
    )
    sim.inject(order)
    assert order.queue_position_current == 200

    # Execute 80 shares off the front — still 120 ahead of us
    sim.process_event(_ev(4, 1, 80, 1_000_000, 1, time=34200.2))
    assert order.queue_position_current == 120
    assert order.status == "live"
    assert order.filled_size == 0


def test_cancel_ahead_decrements_queue_position() -> None:
    """Cancelling an order that was ahead of us reduces our queue position."""
    book, sim = _make_sim()

    sim.process_event(_ev(1, 1, 150, 1_000_000, 1, time=34200.0))

    order = HypotheticalOrder(
        order_id=-1, side="bid", price=1_000_000,
        size=100, entry_timestamp=34200.1,
    )
    sim.inject(order)
    assert order.queue_position_current == 150

    # Full cancel of order 1
    sim.process_event(_ev(3, 1, 150, 1_000_000, 1, time=34200.2))
    assert order.queue_position_current == 0


def test_cancel_behind_does_not_affect_queue_position() -> None:
    """Cancelling an order that arrived AFTER us must not change our position."""
    book, sim = _make_sim()

    sim.process_event(_ev(1, 1, 100, 1_000_000, 1, time=34200.0))  # ahead

    order = HypotheticalOrder(
        order_id=-1, side="bid", price=1_000_000,
        size=50, entry_timestamp=34200.1,
    )
    sim.inject(order)

    # Order 2 arrives AFTER us — behind in queue
    sim.process_event(_ev(1, 2, 999, 1_000_000, 1, time=34200.2))
    sim.process_event(_ev(3, 2, 999, 1_000_000, 1, time=34200.3))  # cancel order 2

    assert order.queue_position_current == 100  # unchanged


def test_partial_cancel_ahead_reduces_queue() -> None:
    """Partial cancel of an order ahead of us reduces by the cancelled amount."""
    book, sim = _make_sim()

    sim.process_event(_ev(1, 1, 200, 1_000_000, 1, time=34200.0))

    order = HypotheticalOrder(
        order_id=-1, side="bid", price=1_000_000,
        size=50, entry_timestamp=34200.1,
    )
    sim.inject(order)
    assert order.queue_position_current == 200

    # Partial cancel: 80 shares removed from order 1
    sim.process_event(_ev(2, 1, 80, 1_000_000, 1, time=34200.2))
    assert order.queue_position_current == 120


# ── fill mechanics ────────────────────────────────────────────────────────────


def test_execution_fills_our_order_when_at_front() -> None:
    """
    Once the queue ahead is exhausted, execution should fill our order.
    status becomes fully_filled and the fill is recorded.
    """
    book, sim = _make_sim()

    sim.process_event(_ev(1, 1, 50, 1_000_000, 1, time=34200.0))  # 50 ahead

    order = HypotheticalOrder(
        order_id=-1, side="bid", price=1_000_000,
        size=100, entry_timestamp=34200.1,
    )
    sim.inject(order)

    # Execution of 150 shares: first 50 drain the queue, next 100 fill us
    sim.process_event(_ev(4, 1, 150, 1_000_000, 1, time=34200.2))

    assert order.status == "fully_filled"
    assert order.filled_size == 100
    assert len(order.fills) == 1
    ts, size, mid = order.fills[0]
    assert size == 100


def test_partial_fill_then_remaining_fill() -> None:
    """Large execution partly fills us; second execution completes the fill."""
    book, sim = _make_sim()

    order = HypotheticalOrder(
        order_id=-1, side="bid", price=1_000_000,
        size=100, entry_timestamp=34200.0,
    )
    sim.inject(order)  # queue empty — at front immediately

    # First execution fills 60 of our 100
    sim.process_event(_ev(4, 99, 60, 1_000_000, 1, time=34200.1))
    assert order.filled_size == 60
    assert order.status == "live"

    # Second execution fills remaining 40
    sim.process_event(_ev(4, 99, 40, 1_000_000, 1, time=34200.2))
    assert order.filled_size == 100
    assert order.status == "fully_filled"


def test_order_expires_when_lifetime_exceeded() -> None:
    """Order status becomes 'expired' when max_lifetime is exceeded."""
    book, sim = _make_sim()

    order = HypotheticalOrder(
        order_id=-1, side="bid", price=1_000_000,
        size=100, entry_timestamp=34200.0, max_lifetime=5.0,
    )
    sim.inject(order)

    # Event arrives after lifetime
    sim.process_event(_ev(1, 99, 10, 999_000, 1, time=34205.1))
    assert order.status == "expired"


# ── hidden execution ──────────────────────────────────────────────────────────


def test_hidden_execution_does_not_affect_queue_position() -> None:
    """
    Type-5 (hidden) executions must not touch the visible book or queue position.
    The passive-shadow model only tracks visible orders.
    """
    book, sim = _make_sim()

    sim.process_event(_ev(1, 1, 200, 1_000_000, 1, time=34200.0))

    order = HypotheticalOrder(
        order_id=-1, side="bid", price=1_000_000,
        size=50, entry_timestamp=34200.1,
    )
    sim.inject(order)
    assert order.queue_position_current == 200

    # Hidden execution at our level — must be ignored
    sim.process_event(_ev(5, 1, 100, 1_000_000, 1, time=34200.2))

    assert order.queue_position_current == 200  # unchanged
    assert order.filled_size == 0
    assert order.status == "live"


# ── compute_outcomes ──────────────────────────────────────────────────────────


def test_compute_outcomes_unfilled() -> None:
    """Unfilled order has filled=False and None fill times."""
    book, sim = _make_sim()

    order = HypotheticalOrder(
        order_id=-1, side="bid", price=1_000_000,
        size=100, entry_timestamp=34200.0, max_lifetime=1.0,
    )
    sim.inject(order)
    sim.process_event(_ev(1, 99, 10, 999_000, 1, time=34201.1))  # expire it

    out = sim.compute_outcomes(order)
    assert out["filled"] is False
    assert out["fully_filled"] is False
    assert out["time_to_first_fill"] is None
    assert out["time_to_full_fill"] is None


def test_compute_outcomes_filled() -> None:
    """Filled order reports correct times and fill prices."""
    book, sim = _make_sim()

    order = HypotheticalOrder(
        order_id=-1, side="bid", price=1_000_000,
        size=50, entry_timestamp=34200.0,
    )
    # Add a matching ask so top_of_book has a valid mid
    sim.process_event(_ev(1, 1, 50, 1_010_000, -1, time=34200.0))
    sim.inject(order)

    sim.process_event(_ev(4, 99, 50, 1_000_000, 1, time=34201.0))

    out = sim.compute_outcomes(order)
    assert out["filled"] is True
    assert out["fully_filled"] is True
    assert out["time_to_first_fill"] == pytest.approx(1.0, abs=1e-6)
    assert out["queue_position_at_entry"] == 0


import pytest  # noqa: E402  (imported here to avoid circular at module level)
