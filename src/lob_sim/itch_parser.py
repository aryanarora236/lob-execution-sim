"""
NASDAQ TotalView-ITCH 5.0 parser.

Streams the compressed binary feed, filters to a single ticker, and writes
LOBSTER-compatible message and orderbook CSVs to an output directory.

Usage:
    # Extract one ticker (keep source for additional extractions):
    uv run python -m lob_sim.itch_parser data/raw/10302019.NASDAQ_ITCH50.gz AAPL data/raw
    uv run python -m lob_sim.itch_parser data/raw/10302019.NASDAQ_ITCH50.gz MSFT data/raw
    uv run python -m lob_sim.itch_parser data/raw/10302019.NASDAQ_ITCH50.gz INTC data/raw --delete

Spec: https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHspecification.pdf
"""

from __future__ import annotations

import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── ITCH message format structs ───────────────────────────────────────────────
# Format strings are big-endian (>). Field widths from ITCH 5.0 spec.
# Each entry: (format_string, field_names)

# LOBSTER event type codes
_LOB_NEW = 1
_LOB_PARTIAL_CANCEL = 2
_LOB_FULL_CANCEL = 3
_LOB_EXEC = 4
_LOB_HALT = 7

# Price tick in ITCH: 1 unit = $0.0001 — same as LOBSTER, no conversion needed.
# Timestamp in ITCH: nanoseconds since midnight. LOBSTER uses seconds (float).
_NS_TO_S = 1e-9


# ── order book ────────────────────────────────────────────────────────────────


@dataclass
class _Order:
    """Live resting order tracked in the book."""
    price: int
    size: int
    direction: int  # 1=buy, -1=sell


@dataclass
class _LimitOrderBook:
    """Minimal price-level book for snapshot generation."""
    bids: dict[int, int] = field(default_factory=dict)  # price → agg size
    asks: dict[int, int] = field(default_factory=dict)

    def add(self, price: int, size: int, direction: int) -> None:
        """Add shares to a price level."""
        side = self.bids if direction == 1 else self.asks
        side[price] = side.get(price, 0) + size

    def reduce(self, price: int, size: int, direction: int) -> None:
        """Remove shares from a price level, pruning empty levels."""
        side = self.bids if direction == 1 else self.asks
        if price in side:
            side[price] -= size
            if side[price] <= 0:
                del side[price]

    def snapshot(self, levels: int) -> list[int]:
        """
        Return flat list of [ask_p1, ask_s1, bid_p1, bid_s1, ...] for top-N levels.
        Missing levels are filled with -1 / 0 (LOBSTER convention).
        """
        asks_sorted = sorted(self.asks.items())[:levels]
        bids_sorted = sorted(self.bids.items(), reverse=True)[:levels]
        row: list[int] = []
        for i in range(levels):
            if i < len(asks_sorted):
                row.extend(asks_sorted[i])
            else:
                row.extend([-1, 0])
            if i < len(bids_sorted):
                row.extend(bids_sorted[i])
            else:
                row.extend([-1, 0])
        return row


# ── binary reading helpers ────────────────────────────────────────────────────


def _read_msg(f) -> tuple[bytes, bytes] | None:
    """Read one length-prefixed ITCH message. Returns (msg_type, raw_body) or None at EOF."""
    hdr = f.read(2)
    if not hdr or len(hdr) < 2:
        return None
    length = struct.unpack(">H", hdr)[0]
    body = f.read(length)
    if len(body) < length:
        return None
    return body[0:1], body


def _timestamp_ns(timestamp_hi: bytes, timestamp_lo: int) -> int:
    """Combine 2-byte high + 4-byte low into 6-byte nanosecond timestamp."""
    hi = struct.unpack(">H", timestamp_hi)[0]
    return (hi << 32) | timestamp_lo


# ── main parser ───────────────────────────────────────────────────────────────


def parse_itch(
    gz_path: Path,
    ticker: str,
    out_dir: Path,
    levels: int = 10,
) -> tuple[Path, Path]:
    """
    Stream-parse a NASDAQ ITCH 5.0 .gz file, extract one ticker, and write
    LOBSTER-compatible message and orderbook CSVs.

    Parameters
    ----------
    gz_path:
        Path to the compressed ITCH file.
    ticker:
        Ticker symbol to extract (e.g. "AAPL").
    out_dir:
        Directory to write output CSVs.
    levels:
        Number of book levels to record in the orderbook file.

    Returns
    -------
    (message_path, orderbook_path)
    """
    ticker_bytes = ticker.ljust(8).encode("ascii")
    out_dir.mkdir(parents=True, exist_ok=True)

    # derive date string from filename e.g. 12302019 → 2019-12-30
    stem = gz_path.stem  # "12302019.NASDAQ_ITCH50" → need just date part
    date_raw = stem.split(".")[0]  # "12302019"
    date_str = f"{date_raw[4:]}-{date_raw[:2]}-{date_raw[2:4]}"

    msg_path = out_dir / f"{ticker}_{date_str}_34200000_57600000_message_{levels}.csv"
    ob_path = out_dir / f"{ticker}_{date_str}_34200000_57600000_orderbook_{levels}.csv"

    # pass 1 metadata
    locate_to_ticker: dict[int, bytes] = {}  # stock_locate → ticker bytes
    target_locate: int | None = None

    # order tracking
    orders: dict[int, _Order] = {}  # order_ref → Order
    book = _LimitOrderBook()

    msg_count = 0
    ob_count = 0

    print(f"Parsing {gz_path.name} → filtering for {ticker}...")

    # Use system gzip to decompress — handles multi-stream concatenated .gz files
    # that Python's gzip module fails on at stream boundaries.
    proc = subprocess.Popen(
        ["gzip", "-d", "-c", str(gz_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    f = proc.stdout
    with (
        open(msg_path, "w") as msg_f,
        open(ob_path, "w") as ob_f,
    ):
        # No header — LOBSTER convention has none

        def emit(ts_ns: int, event_type: int, order_id: int, size: int, price: int, direction: int) -> None:
            nonlocal msg_count, ob_count
            ts_s = ts_ns * _NS_TO_S
            msg_f.write(f"{ts_s:.9f},{event_type},{order_id},{size},{price},{direction}\n")
            snap = book.snapshot(levels)
            ob_f.write(",".join(map(str, snap)) + "\n")
            msg_count += 1
            ob_count += 1

        while True:
            result = _read_msg(f)
            if result is None:
                break
            msg_type, body = result

            # ── Stock Directory: build locate → ticker map ─────────────────
            if msg_type == b"R":
                # body[1:3]=locate, [7:9]=timestamp_hi, [9:13]=timestamp_lo, [13:21]=stock
                locate = struct.unpack(">H", body[1:3])[0]
                stock = body[11:19]  # 8 bytes at offset 11 (after type+locate+tracking+ts_hi)
                locate_to_ticker[locate] = stock
                if stock == ticker_bytes:
                    target_locate = locate
                continue

            if target_locate is None:
                continue

            # check locate code before unpacking (fast filter)
            if len(body) < 3:
                continue
            locate = struct.unpack(">H", body[1:3])[0]
            if locate != target_locate and msg_type not in (b"A", b"F", b"U"):
                # A/F/U carry stock name; others use locate
                pass

            # ── Add Order (A and F share same fields we need) ──────────────
            if msg_type in (b"A", b"F"):
                # A: type(1) locate(2) tracking(2) ts_hi(2) ts_lo(4) order_ref(8) buy_sell(1) shares(4) stock(8) price(4)
                # F: same + mpid(4)
                if len(body) < 34:
                    continue
                locate = struct.unpack(">H", body[1:3])[0]
                ts_hi = body[5:7]
                ts_lo = struct.unpack(">I", body[7:11])[0]
                order_ref = struct.unpack(">Q", body[11:19])[0]
                buy_sell = body[19:20]
                shares = struct.unpack(">I", body[20:24])[0]
                stock = body[24:32]
                price = struct.unpack(">I", body[32:36])[0]

                if stock != ticker_bytes:
                    continue

                direction = 1 if buy_sell == b"B" else -1
                ts_ns = _timestamp_ns(ts_hi, ts_lo)
                orders[order_ref] = _Order(price=price, size=shares, direction=direction)
                book.add(price, shares, direction)
                emit(ts_ns, _LOB_NEW, order_ref, shares, price, direction)

            # ── Order Executed ─────────────────────────────────────────────
            elif msg_type == b"E":
                if locate != target_locate:
                    continue
                # type(1) locate(2) tracking(2) ts_hi(2) ts_lo(4) order_ref(8) exec_shares(4) match(8)
                if len(body) < 29:
                    continue
                ts_hi = body[5:7]
                ts_lo = struct.unpack(">I", body[7:11])[0]
                order_ref = struct.unpack(">Q", body[11:19])[0]
                exec_shares = struct.unpack(">I", body[19:23])[0]

                if order_ref not in orders:
                    continue
                order = orders[order_ref]
                ts_ns = _timestamp_ns(ts_hi, ts_lo)
                book.reduce(order.price, exec_shares, order.direction)
                order.size -= exec_shares
                if order.size <= 0:
                    del orders[order_ref]
                emit(ts_ns, _LOB_EXEC, order_ref, exec_shares, order.price, order.direction)

            # ── Order Executed with Price ──────────────────────────────────
            elif msg_type == b"C":
                if locate != target_locate:
                    continue
                if len(body) < 34:
                    continue
                ts_hi = body[5:7]
                ts_lo = struct.unpack(">I", body[7:11])[0]
                order_ref = struct.unpack(">Q", body[11:19])[0]
                exec_shares = struct.unpack(">I", body[19:23])[0]
                exec_price = struct.unpack(">I", body[30:34])[0]

                if order_ref not in orders:
                    continue
                order = orders[order_ref]
                ts_ns = _timestamp_ns(ts_hi, ts_lo)
                # execution price may differ (hidden order); book side uses original resting price
                book.reduce(order.price, exec_shares, order.direction)
                order.size -= exec_shares
                if order.size <= 0:
                    del orders[order_ref]
                emit(ts_ns, _LOB_EXEC, order_ref, exec_shares, exec_price, order.direction)

            # ── Order Cancel (partial) ─────────────────────────────────────
            elif msg_type == b"X":
                if locate != target_locate:
                    continue
                if len(body) < 23:
                    continue
                ts_hi = body[5:7]
                ts_lo = struct.unpack(">I", body[7:11])[0]
                order_ref = struct.unpack(">Q", body[11:19])[0]
                cancelled = struct.unpack(">I", body[19:23])[0]

                if order_ref not in orders:
                    continue
                order = orders[order_ref]
                ts_ns = _timestamp_ns(ts_hi, ts_lo)
                book.reduce(order.price, cancelled, order.direction)
                order.size -= cancelled
                emit(ts_ns, _LOB_PARTIAL_CANCEL, order_ref, cancelled, order.price, order.direction)

            # ── Order Delete (full) ────────────────────────────────────────
            elif msg_type == b"D":
                if locate != target_locate:
                    continue
                if len(body) < 19:
                    continue
                ts_hi = body[5:7]
                ts_lo = struct.unpack(">I", body[7:11])[0]
                order_ref = struct.unpack(">Q", body[11:19])[0]

                if order_ref not in orders:
                    continue
                order = orders[order_ref]
                ts_ns = _timestamp_ns(ts_hi, ts_lo)
                book.reduce(order.price, order.size, order.direction)
                emit(ts_ns, _LOB_FULL_CANCEL, order_ref, order.size, order.price, order.direction)
                del orders[order_ref]

            # ── Order Replace ──────────────────────────────────────────────
            elif msg_type == b"U":
                if locate != target_locate:
                    continue
                if len(body) < 35:
                    continue
                ts_hi = body[5:7]
                ts_lo = struct.unpack(">I", body[7:11])[0]
                orig_ref = struct.unpack(">Q", body[11:19])[0]
                new_ref = struct.unpack(">Q", body[19:27])[0]
                new_shares = struct.unpack(">I", body[27:31])[0]
                new_price = struct.unpack(">I", body[31:35])[0]

                if orig_ref not in orders:
                    continue
                old = orders.pop(orig_ref)
                ts_ns = _timestamp_ns(ts_hi, ts_lo)

                # emit as delete of old + new limit (LOBSTER convention)
                book.reduce(old.price, old.size, old.direction)
                emit(ts_ns, _LOB_FULL_CANCEL, orig_ref, old.size, old.price, old.direction)

                orders[new_ref] = _Order(price=new_price, size=new_shares, direction=old.direction)
                book.add(new_price, new_shares, old.direction)
                emit(ts_ns, _LOB_NEW, new_ref, new_shares, new_price, old.direction)

            # ── Trading Halt ───────────────────────────────────────────────
            elif msg_type == b"H":
                if locate != target_locate:
                    continue
                if len(body) < 25:
                    continue
                ts_hi = body[5:7]
                ts_lo = struct.unpack(">I", body[7:11])[0]
                ts_ns = _timestamp_ns(ts_hi, ts_lo)
                emit(ts_ns, _LOB_HALT, 0, 0, 0, 0)

            if msg_count % 100_000 == 0 and msg_count > 0:
                print(f"  {msg_count:,} {ticker} events written...", end="\r")

    print(f"\nDone. {msg_count:,} message rows, {ob_count:,} orderbook snapshots.")
    print(f"  Message file : {msg_path}")
    print(f"  Orderbook    : {ob_path}")
    return msg_path, ob_path


# ── entry point ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m lob_sim.itch_parser <gz_file> <ticker> [out_dir] [levels] [--delete]")
        print("  --delete  remove the source .gz after parsing (use only after all tickers extracted)")
        sys.exit(1)

    gz = Path(sys.argv[1])
    tkr = sys.argv[2].upper()
    args = sys.argv[3:]
    delete_source = "--delete" in args
    positional = [a for a in args if not a.startswith("--")]
    out = Path(positional[0]) if len(positional) > 0 else gz.parent
    lvl = int(positional[1]) if len(positional) > 1 else 10

    msg_p, ob_p = parse_itch(gz, tkr, out, lvl)

    if delete_source:
        print(f"\nDeleting source file ({gz.stat().st_size / 1e9:.2f} GB)...")
        gz.unlink()
        print("Deleted.")
    else:
        print(f"\nSource file kept at {gz} ({gz.stat().st_size / 1e9:.2f} GB).")
        print("Re-run with --delete after all tickers are extracted.")
