"""
Injection demo: 5 hypothetical orders across the AAPL trading day.

Run from project root:
    uv run python notebooks/injection_demo.py

Orders injected (price decided on-the-fly from live book state):
  1. 09:35 — buy at best bid,        60s lifetime
  2. 10:00 — sell at best ask,       60s lifetime
  3. 11:00 — buy 1 tick behind bid,  60s lifetime  (deeper queue, harder fill)
  4. 13:00 — buy at best bid,         5s lifetime  (almost certain to expire)
  5. 15:00 — sell at best ask,       60s lifetime
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl

from lob_sim.explore import find_lobster_pair, load_messages, parse_filename_meta
from lob_sim.injection import HypotheticalOrder, InjectionSimulator
from lob_sim.orderbook import LimitOrderBook

TICK = 100  # 1 tick = $0.01 = 100 raw price units

# ── injection specs (times are seconds since midnight) ────────────────────────
INJECTION_SPECS = [
    {"id": 1, "target_time": 34_500,  "side": "bid", "tick_offset": 0,  "size": 100, "lifetime": 60},
    {"id": 2, "target_time": 36_000,  "side": "ask", "tick_offset": 0,  "size": 100, "lifetime": 60},
    {"id": 3, "target_time": 39_600,  "side": "bid", "tick_offset": -1, "size": 100, "lifetime": 60},
    {"id": 4, "target_time": 46_800,  "side": "bid", "tick_offset": 0,  "size": 100, "lifetime": 5},
    {"id": 5, "target_time": 54_000,  "side": "ask", "tick_offset": 0,  "size": 100, "lifetime": 60},
]


def hms(s: float) -> str:
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def main() -> None:
    data_dir = Path("data/raw")
    msg_path, _ = find_lobster_pair(data_dir)
    meta = parse_filename_meta(msg_path)

    print(f"Loading {msg_path.name}...")
    messages = load_messages(msg_path)
    print(f"  {len(messages):,} events\n")

    book = LimitOrderBook()
    sim = InjectionSimulator(book)

    orders: dict[int, HypotheticalOrder] = {}
    injected: set[int] = set()

    print("Replaying events and injecting orders...\n")

    for event in messages.iter_rows(named=True):
        ts = float(event["time"])

        # Inject any orders whose target time has arrived
        for spec in INJECTION_SPECS:
            sid = spec["id"]
            if sid in injected or ts < spec["target_time"]:
                continue

            bp, bs, ap, as_ = book.top_of_book()
            if bp == -1 or ap == -1:
                continue  # book not yet populated

            if spec["side"] == "bid":
                price = bp + spec["tick_offset"] * TICK
            else:
                price = ap + spec["tick_offset"] * TICK

            order = HypotheticalOrder(
                order_id=sid,
                side=spec["side"],
                price=price,
                size=spec["size"],
                entry_timestamp=ts,
                max_lifetime=float(spec["lifetime"]),
            )
            sim.inject(order)
            orders[sid] = order
            injected.add(sid)

            print(
                f"  → Injected order {sid}: {spec['side'].upper()} "
                f"{spec['size']} @ ${price/10000:.2f} "
                f"at {hms(ts)} | "
                f"queue ahead: {order.queue_position_at_entry:,} shares | "
                f"spread: {order.spread_at_entry_ticks} ticks | "
                f"imbalance: {order.book_imbalance_at_entry:+.3f}"
            )

        sim.process_event(event)

    # ── print outcomes ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("OUTCOMES")
    print("=" * 70)

    for sid, order in orders.items():
        out = sim.compute_outcomes(order)
        spec = next(s for s in INJECTION_SPECS if s["id"] == sid)

        print(f"\nOrder {sid}: {order.side.upper()} {order.size} @ ${order.price/10000:.2f}")
        print(f"  Injected at   : {hms(order.entry_timestamp)}")
        print(f"  Status        : {order.status}")
        print(f"  Filled        : {order.filled_size} / {order.size} shares")
        print(f"  Queue at entry: {out['queue_position_at_entry']:,} shares ahead")
        print(f"  Spread        : {out['spread_at_entry_ticks']} ticks")
        print(f"  Imbalance     : {out['book_imbalance_at_entry']:+.3f}")

        if out["filled"]:
            print(f"  Time to fill  : {out['time_to_first_fill']:.2f}s")
            mid_e = out["mid_at_entry"] / 10000
            mid_f = (out["mid_at_first_fill"] or 0) / 10000
            print(f"  Mid at entry  : ${mid_e:.4f}")
            print(f"  Mid at fill   : ${mid_f:.4f}")
            if out["adverse_selection_1s"] is not None:
                adv = out["adverse_selection_1s"] / 10000 * 100  # cents
                print(f"  Adverse sel.  : {adv:+.3f} cents/share (1s window)")
        else:
            print(f"  (no fill within {spec['lifetime']}s lifetime)")

    print()


if __name__ == "__main__":
    main()
