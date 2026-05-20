"""
Limit order book with price-time priority.

Internal price representation: raw ITCH/LOBSTER integers ($ × 10_000).
Never divide here — callers that want dollars divide by 10_000.

Sentinel for empty levels in snapshots: price = -1, size = 0.
This matches what itch_parser._LimitOrderBook emits and what our
orderbook CSV contains.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from sortedcontainers import SortedDict

# Snapshot sentinel values (match itch_parser output)
_EMPTY_PRICE = -1
_EMPTY_SIZE = 0

# Key function that makes SortedDict iterate in descending price order (best bid first)
_NEG = lambda k: -k  # noqa: E731


@dataclass
class _PriceLevel:
    """One price level: ordered queue of order IDs and a fast-lookup size map."""

    queue: deque[int] = field(default_factory=deque)   # order_ids in arrival order
    sizes: dict[int, int] = field(default_factory=dict)  # order_id → remaining size
    total: int = 0  # cached aggregate size; kept consistent with sizes

    def add(self, order_id: int, size: int) -> None:
        """Append a new order to the back of the queue."""
        self.queue.append(order_id)
        self.sizes[order_id] = size
        self.total += size

    def reduce(self, order_id: int, by: int) -> bool:
        """
        Reduce an order's size by `by` shares.
        Returns True if the order was fully consumed and removed.
        """
        if order_id not in self.sizes:
            return False
        self.sizes[order_id] -= by
        self.total -= by
        if self.sizes[order_id] <= 0:
            del self.sizes[order_id]
            self.queue.remove(order_id)  # O(n) but levels are small in practice
            return True
        return False

    def remove(self, order_id: int) -> bool:
        """
        Fully remove an order regardless of remaining size.
        Returns True if the order was found.
        """
        if order_id not in self.sizes:
            return False
        self.total -= self.sizes[order_id]
        del self.sizes[order_id]
        self.queue.remove(order_id)
        return True

    def is_empty(self) -> bool:
        return not self.sizes


class LimitOrderBook:
    """
    Price-time priority limit order book.

    Bids are stored in a SortedDict with a negated-key comparator so that
    the highest bid price is always first. Asks use natural ascending order
    so the lowest ask is first.

    Each price level holds a _PriceLevel with:
      - a deque of order IDs in arrival order (time priority)
      - a dict mapping order_id → remaining size (O(1) lookup)
      - a cached total size (O(1) aggregate)

    apply_event() handles all LOBSTER event types. The caller should pass
    a dict (or any mapping) with keys: event_type, order_id, size, price,
    direction.
    """

    def __init__(self) -> None:
        # Bids: highest price first (negated-key SortedDict)
        self._bids: SortedDict = SortedDict(_NEG)
        # Asks: lowest price first (natural SortedDict)
        self._asks: SortedDict = SortedDict()
        # Fast route from order_id to its location: ('bid'|'ask', price)
        self._order_index: dict[int, tuple[str, int]] = {}

    # ── public interface ──────────────────────────────────────────────────────

    def apply_event(self, event: dict) -> None:
        """
        Apply one LOBSTER message event to the book.

        Supported event types:
          1 — new limit order submission
          2 — partial cancellation (size field = shares removed)
          3 — full cancellation / deletion
          4 — visible execution (may fully consume the order)
          5 — hidden execution (do NOT touch visible book)
          6 — cross / auction (skip)
          7 — trading halt (skip)
        """
        etype = int(event["event_type"])
        oid = int(event["order_id"])

        if etype == 1:
            self._add(oid, int(event["size"]), int(event["price"]), int(event["direction"]))
        elif etype == 2:
            self._reduce(oid, int(event["size"]))
        elif etype == 3:
            self._remove(oid)
        elif etype == 4:
            self._reduce(oid, int(event["size"]))
        # types 5, 6, 7: no change to visible book

    def top_of_book(self) -> tuple[int, int, int, int]:
        """
        Return (best_bid_price, best_bid_size, best_ask_price, best_ask_size).
        Missing sides return (_EMPTY_PRICE, _EMPTY_SIZE).
        """
        if self._bids:
            bp = next(iter(self._bids))
            bs = self._bids[bp].total
        else:
            bp, bs = _EMPTY_PRICE, _EMPTY_SIZE

        if self._asks:
            ap = next(iter(self._asks))
            as_ = self._asks[ap].total
        else:
            ap, as_ = _EMPTY_PRICE, _EMPTY_SIZE

        return bp, bs, ap, as_

    def snapshot(self, levels: int = 10) -> dict[str, int]:
        """
        Return a dict with the same column layout as the LOBSTER orderbook file:
          ask_price_1, ask_size_1, bid_price_1, bid_size_1, ..., to level N.

        Empty levels are filled with price = -1, size = 0.
        """
        ask_prices = list(self._asks.islice(0, levels))
        bid_prices = list(self._bids.islice(0, levels))

        result: dict[str, int] = {}
        for i in range(levels):
            lvl = i + 1
            if i < len(ask_prices):
                p = ask_prices[i]
                result[f"ask_price_{lvl}"] = p
                result[f"ask_size_{lvl}"] = self._asks[p].total
            else:
                result[f"ask_price_{lvl}"] = _EMPTY_PRICE
                result[f"ask_size_{lvl}"] = _EMPTY_SIZE

            if i < len(bid_prices):
                p = bid_prices[i]
                result[f"bid_price_{lvl}"] = p
                result[f"bid_size_{lvl}"] = self._bids[p].total
            else:
                result[f"bid_price_{lvl}"] = _EMPTY_PRICE
                result[f"bid_size_{lvl}"] = _EMPTY_SIZE

        return result

    def snapshot_flat(self, levels: int = 10) -> list[int]:
        """
        Same as snapshot() but returns a flat list in column order:
        [ask_p1, ask_s1, bid_p1, bid_s1, ask_p2, ask_s2, bid_p2, bid_s2, ...]

        Faster than snapshot() for bulk comparison (avoids dict construction).
        """
        ask_prices = list(self._asks.islice(0, levels))
        bid_prices = list(self._bids.islice(0, levels))

        row: list[int] = []
        for i in range(levels):
            if i < len(ask_prices):
                p = ask_prices[i]
                row.append(p)
                row.append(self._asks[p].total)
            else:
                row.append(_EMPTY_PRICE)
                row.append(_EMPTY_SIZE)

            if i < len(bid_prices):
                p = bid_prices[i]
                row.append(p)
                row.append(self._bids[p].total)
            else:
                row.append(_EMPTY_PRICE)
                row.append(_EMPTY_SIZE)

        return row

    def __len__(self) -> int:
        """Total number of resting orders (not price levels)."""
        return len(self._order_index)

    # ── private helpers ───────────────────────────────────────────────────────

    def _side(self, direction: int) -> tuple[SortedDict, str]:
        if direction == 1:
            return self._bids, "bid"
        return self._asks, "ask"

    def _add(self, order_id: int, size: int, price: int, direction: int) -> None:
        sd, side_name = self._side(direction)
        if price not in sd:
            sd[price] = _PriceLevel()
        sd[price].add(order_id, size)
        self._order_index[order_id] = (side_name, price)

    def _reduce(self, order_id: int, by: int) -> None:
        loc = self._order_index.get(order_id)
        if loc is None:
            return
        side_name, price = loc
        sd = self._bids if side_name == "bid" else self._asks
        level = sd.get(price)
        if level is None:
            return
        fully_consumed = level.reduce(order_id, by)
        if fully_consumed:
            del self._order_index[order_id]
            if level.is_empty():
                del sd[price]

    def _remove(self, order_id: int) -> None:
        loc = self._order_index.get(order_id)
        if loc is None:
            return
        side_name, price = loc
        sd = self._bids if side_name == "bid" else self._asks
        del self._order_index[order_id]
        level = sd.get(price)
        if level is None:
            return
        level.remove(order_id)
        if level.is_empty():
            del sd[price]
