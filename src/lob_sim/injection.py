"""
Hypothetical order injection and queue-position tracking.

Conceptual model
----------------
The historical event stream is treated as ground truth — "the market."
A HypotheticalOrder is a counterfactual: "what would have happened if I had
posted this order at time T?" We inject it into the book as a shadow order
that sits at the back of the queue at its price level, then replay subsequent
events to see whether it fills.

Simplifying assumption (passive-shadow model)
---------------------------------------------
Our order does NOT influence the historical event stream. Real orders affect
the market through price impact, queue displacement, and information signaling.
This model ignores all of that and is appropriate only for small order sizes
relative to top-of-book depth — the standard first-cut in microstructure
research. Results will overestimate fill rates in thin markets.

Queue position mechanics
------------------------
At injection time we snapshot the set of order_ids already resting at our
price level. Those orders are "ahead of us" in the time-priority queue.

- Executions (type 4) are FIFO and always consume from the front, so they
  always reduce our queue position regardless of which specific order_id was
  hit.

- Cancels (types 2 and 3) can come from anywhere in the queue. We check
  whether the cancelled order_id is in our ahead-set before reducing queue
  position. New orders arriving after us go behind us and are ignored.

- Hidden executions (type 5) do not touch the visible book, so they do not
  affect our queue position.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from lob_sim.orderbook import LimitOrderBook

# 1 tick for US equities = $0.01 = 100 raw price units
TICK_SIZE = 100


@dataclass
class HypotheticalOrder:
    """
    A counterfactual limit order injected into the historical replay.

    Attributes set at construction time:
        order_id        — caller-assigned identifier
        side            — "bid" (buy) or "ask" (sell)
        price           — raw integer price ($ × 10_000)
        size            — total shares to fill
        entry_timestamp — seconds-since-midnight when order is injected
        max_lifetime    — seconds before order expires unfilled (default 60s)

    Attributes populated by InjectionSimulator.inject():
        queue_position_at_entry   — total visible size ahead of us at injection
        orders_ahead_at_entry     — count of distinct orders ahead (K)
        queue_granularity_at_entry — K / queue_position_at_entry (orders per share);
                                     0.0 when queue is empty
        mid_at_entry              — mid-price at injection (raw units)
        book_imbalance_at_entry   — (bid_sz − ask_sz) / (bid_sz + ask_sz) at TOB
        spread_at_entry_ticks     — best ask − best bid in ticks

    Attributes updated during replay:
        queue_position_current    — volume still ahead of us (property)
        filled_size               — shares filled so far
        fills                     — list of (timestamp, size, mid_at_fill)
        status                    — "live" | "fully_filled" | "expired"
    """

    order_id: int
    side: str          # "bid" or "ask"
    price: int         # raw integer ($ × 10_000)
    size: int          # total shares
    entry_timestamp: float
    max_lifetime: float = 60.0

    # Populated at inject() time
    queue_position_at_entry: int = field(default=0, init=False)
    orders_ahead_at_entry: int = field(default=0, init=False)
    queue_granularity_at_entry: float = field(default=0.0, init=False)
    mid_at_entry: float = field(default=0.0, init=False)
    book_imbalance_at_entry: float = field(default=0.0, init=False)
    spread_at_entry_ticks: int = field(default=0, init=False)

    # Internal queue tracking (not part of public interface)
    _volume_ahead: int = field(default=0, init=False, repr=False)
    _orders_ahead: set[int] = field(default_factory=set, init=False, repr=False)

    # Fill tracking
    filled_size: int = field(default=0, init=False)
    fills: list[tuple[float, int, float]] = field(default_factory=list, init=False)
    status: str = field(default="live", init=False)  # "live"|"fully_filled"|"expired"

    @property
    def queue_position_current(self) -> int:
        """Volume of orders still ahead of us in the queue."""
        return self._volume_ahead

    @property
    def remaining_size(self) -> int:
        return self.size - self.filled_size


class InjectionSimulator:
    """
    Wraps a LimitOrderBook to support hypothetical order injection and tracking.

    Usage pattern::

        book = LimitOrderBook()
        sim  = InjectionSimulator(book)

        for event in events:
            if should_inject_now(event):
                order = HypotheticalOrder(...)
                sim.inject(order)
            sim.process_event(event)

        outcomes = sim.compute_outcomes(order)
    """

    def __init__(self, book: LimitOrderBook) -> None:
        self._book = book
        self._orders: list[HypotheticalOrder] = []
        # Sparse time-series of (timestamp, mid_price) for adverse-selection lookup
        self._mid_times: list[float] = []
        self._mid_values: list[float] = []

    # ── public API ────────────────────────────────────────────────────────────

    def inject(self, order: HypotheticalOrder) -> None:
        """
        Register a hypothetical order and snapshot its queue position.

        Must be called BEFORE process_event() for the event that occurs at or
        after entry_timestamp.
        """
        direction = 1 if order.side == "bid" else -1
        level_orders = self._book.orders_at_level(order.price, direction)

        order._orders_ahead = {oid for oid, _ in level_orders}
        order._volume_ahead = sum(sz for _, sz in level_orders)
        order.queue_position_at_entry = order._volume_ahead
        order.orders_ahead_at_entry = len(order._orders_ahead)
        order.queue_granularity_at_entry = (
            order.orders_ahead_at_entry / order.queue_position_at_entry
            if order.queue_position_at_entry > 0 else 0.0
        )

        bp, bs, ap, as_ = self._book.top_of_book()
        if bp != -1 and ap != -1:
            order.mid_at_entry = (bp + ap) / 2.0
            total = bs + as_
            order.book_imbalance_at_entry = (bs - as_) / total if total > 0 else 0.0
            order.spread_at_entry_ticks = (ap - bp) // TICK_SIZE
        self._orders.append(order)

    def process_event(self, event: dict) -> None:
        """
        Apply one historical event: update all live hypothetical orders first,
        then advance the underlying LimitOrderBook.

        Mid-price is sampled before the event so fill-price mid reflects
        pre-execution book state.
        """
        ts = float(event["time"])
        pre_mid = self._current_mid()

        for order in self._orders:
            if order.status != "live":
                continue
            if ts > order.entry_timestamp + order.max_lifetime:
                order.status = "expired"
                continue
            self._update_order(order, event, ts, pre_mid)

        self._book.apply_event(event)

        mid = self._current_mid()
        if mid is not None:
            self._mid_times.append(ts)
            self._mid_values.append(mid)

    def compute_outcomes(
        self,
        order: HypotheticalOrder,
        lookback_after_fill: float = 1.0,
    ) -> dict:
        """
        Summarise the outcome for one hypothetical order after full replay.

        Parameters
        ----------
        order:
            A HypotheticalOrder that has been through process_event() calls.
        lookback_after_fill:
            Seconds after first fill at which to sample mid for adverse
            selection (default 1.0 s).

        Returns
        -------
        dict with keys:
            filled, fully_filled, time_to_first_fill, time_to_full_fill,
            mid_at_entry, mid_at_first_fill, mid_at_full_fill,
            adverse_selection_1s, queue_position_at_entry,
            book_imbalance_at_entry, spread_at_entry_ticks
        """
        first_fill = order.fills[0] if order.fills else None
        last_fill = order.fills[-1] if order.fills else None

        time_to_first = (
            first_fill[0] - order.entry_timestamp if first_fill else None
        )
        time_to_full = (
            last_fill[0] - order.entry_timestamp
            if order.status == "fully_filled" and last_fill
            else None
        )

        mid_at_first = first_fill[2] if first_fill else None
        mid_at_full = last_fill[2] if last_fill and order.status == "fully_filled" else None

        # Adverse selection: signed mid move in the direction that hurts us
        # Positive = we got hit by informed flow (mid moved against position)
        adverse_1s: float | None = None
        if first_fill is not None and self._mid_times:
            target_ts = first_fill[0] + lookback_after_fill
            idx = bisect.bisect_left(self._mid_times, target_ts)
            if idx < len(self._mid_values):
                mid_1s_after = self._mid_values[idx]
                if order.side == "bid":
                    # Bought: adverse if mid falls (we overpaid)
                    adverse_1s = (mid_at_first or 0.0) - mid_1s_after
                else:
                    # Sold: adverse if mid rises (we undersold)
                    adverse_1s = mid_1s_after - (mid_at_first or 0.0)

        return {
            "filled": bool(order.fills),
            "fully_filled": order.status == "fully_filled",
            "time_to_first_fill": time_to_first,
            "time_to_full_fill": time_to_full,
            "mid_at_entry": order.mid_at_entry,
            "mid_at_first_fill": mid_at_first,
            "mid_at_full_fill": mid_at_full,
            "adverse_selection_1s": adverse_1s,
            "queue_position_at_entry": order.queue_position_at_entry,
            "orders_ahead_at_entry": order.orders_ahead_at_entry,
            "queue_granularity_at_entry": order.queue_granularity_at_entry,
            "book_imbalance_at_entry": order.book_imbalance_at_entry,
            "spread_at_entry_ticks": order.spread_at_entry_ticks,
        }

    # ── private helpers ───────────────────────────────────────────────────────

    def _current_mid(self) -> float | None:
        bp, _, ap, _ = self._book.top_of_book()
        if bp == -1 or ap == -1:
            return None
        return (bp + ap) / 2.0

    def _update_order(
        self,
        order: HypotheticalOrder,
        event: dict,
        ts: float,
        pre_mid: float | None,
    ) -> None:
        etype = int(event["event_type"])
        price = int(event["price"])
        direction = int(event["direction"])
        size = int(event["size"])
        oid = int(event["order_id"])

        order_direction = 1 if order.side == "bid" else -1

        # Only events at our exact price level on our side matter
        if price != order.price or direction != order_direction:
            return

        if etype == 2:  # partial cancel
            if oid in order._orders_ahead:
                # size = shares removed; order still exists (stays in ahead set)
                order._volume_ahead = max(0, order._volume_ahead - size)

        elif etype == 3:  # full cancel
            if oid in order._orders_ahead:
                # size = remaining shares being removed
                order._volume_ahead = max(0, order._volume_ahead - size)
                order._orders_ahead.discard(oid)

        elif etype == 4:  # visible execution — FIFO, always from front
            # First drain whatever is still ahead of us
            consumed_ahead = min(size, order._volume_ahead)
            order._volume_ahead = max(0, order._volume_ahead - consumed_ahead)
            remaining_exec = size - consumed_ahead

            # Execution reaches us only if volume_ahead is now 0
            if remaining_exec > 0 and order.status == "live":
                fill_size = min(remaining_exec, order.remaining_size)
                mid = pre_mid if pre_mid is not None else 0.0
                order.fills.append((ts, fill_size, mid))
                order.filled_size += fill_size
                if order.remaining_size == 0:
                    order.status = "fully_filled"

        # type 1 (new order): goes behind us — no effect on queue position
        # type 5 (hidden exec): does not affect visible book or queue position
        # types 6, 7: skip
