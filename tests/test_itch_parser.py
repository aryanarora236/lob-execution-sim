"""Unit tests for the ITCH 5.0 binary parser."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

from lob_sim.itch_parser import _LimitOrderBook, _timestamp_ns, parse_itch

# ── helpers to build synthetic ITCH binary messages ──────────────────────────


def _stock_dir_msg(locate: int, stock: str) -> bytes:
    """Minimal Stock Directory (R) message for locate→ticker mapping."""
    stock_b = stock.ljust(8).encode("ascii")
    # type(1) locate(2) tracking(2) ts_hi(2) ts_lo(4) stock(8) + padding to fill spec
    payload = (
        b"R"
        + struct.pack(">H", locate)   # locate
        + b"\x00\x00"                 # tracking
        + b"\x00\x00"                 # timestamp_hi
        + struct.pack(">I", 0)        # timestamp_lo
        + stock_b                     # stock (8 bytes)
        + b"\x00" * 20               # remaining R fields (we don't use them)
    )
    return struct.pack(">H", len(payload)) + payload


def _add_order_msg(
    locate: int,
    ts_ns: int,
    order_ref: int,
    side: str,
    shares: int,
    stock: str,
    price: int,
) -> bytes:
    """Add Order (A) message."""
    ts_hi = (ts_ns >> 32) & 0xFFFF
    ts_lo = ts_ns & 0xFFFFFFFF
    stock_b = stock.ljust(8).encode("ascii")
    side_b = b"B" if side == "buy" else b"S"
    payload = (
        b"A"
        + struct.pack(">H", locate)
        + b"\x00\x00"
        + struct.pack(">H", ts_hi)
        + struct.pack(">I", ts_lo)
        + struct.pack(">Q", order_ref)
        + side_b
        + struct.pack(">I", shares)
        + stock_b
        + struct.pack(">I", price)
    )
    return struct.pack(">H", len(payload)) + payload


def _delete_order_msg(locate: int, ts_ns: int, order_ref: int) -> bytes:
    """Order Delete (D) message."""
    ts_hi = (ts_ns >> 32) & 0xFFFF
    ts_lo = ts_ns & 0xFFFFFFFF
    payload = (
        b"D"
        + struct.pack(">H", locate)
        + b"\x00\x00"
        + struct.pack(">H", ts_hi)
        + struct.pack(">I", ts_lo)
        + struct.pack(">Q", order_ref)
    )
    return struct.pack(">H", len(payload)) + payload


def _execute_msg(locate: int, ts_ns: int, order_ref: int, exec_shares: int) -> bytes:
    """Order Executed (E) message."""
    ts_hi = (ts_ns >> 32) & 0xFFFF
    ts_lo = ts_ns & 0xFFFFFFFF
    payload = (
        b"E"
        + struct.pack(">H", locate)
        + b"\x00\x00"
        + struct.pack(">H", ts_hi)
        + struct.pack(">I", ts_lo)
        + struct.pack(">Q", order_ref)
        + struct.pack(">I", exec_shares)
        + struct.pack(">Q", 999)  # match number
    )
    return struct.pack(">H", len(payload)) + payload


def _replace_order_msg(
    locate: int, ts_ns: int, orig_ref: int, new_ref: int, shares: int, price: int
) -> bytes:
    """Order Replace (U) message."""
    ts_hi = (ts_ns >> 32) & 0xFFFF
    ts_lo = ts_ns & 0xFFFFFFFF
    payload = (
        b"U"
        + struct.pack(">H", locate)
        + b"\x00\x00"
        + struct.pack(">H", ts_hi)
        + struct.pack(">I", ts_lo)
        + struct.pack(">Q", orig_ref)
        + struct.pack(">Q", new_ref)
        + struct.pack(">I", shares)
        + struct.pack(">I", price)
    )
    return struct.pack(">H", len(payload)) + payload


def _make_itch_gz(messages: list[bytes], path: Path) -> None:
    """Write a list of raw ITCH message bytes into a gzip file."""
    with gzip.open(path, "wb") as f:
        for msg in messages:
            f.write(msg)


# ── _LimitOrderBook unit tests ────────────────────────────────────────────────


def test_book_add_and_snapshot() -> None:
    book = _LimitOrderBook()
    book.add(price=1000, size=100, direction=1)   # bid
    book.add(price=1001, size=50, direction=-1)   # ask
    snap = book.snapshot(levels=1)
    # [ask_p1, ask_s1, bid_p1, bid_s1]
    assert snap == [1001, 50, 1000, 100]


def test_book_reduce_removes_empty_level() -> None:
    book = _LimitOrderBook()
    book.add(price=500, size=100, direction=1)
    book.reduce(price=500, size=100, direction=1)
    assert 500 not in book.bids


def test_book_snapshot_fills_missing_levels() -> None:
    book = _LimitOrderBook()
    book.add(price=1000, size=100, direction=1)
    snap = book.snapshot(levels=2)
    # level 1: ask missing → [-1, 0], bid = [1000, 100]
    # level 2: both missing → [-1, 0, -1, 0]
    assert snap == [-1, 0, 1000, 100, -1, 0, -1, 0]


def test_book_aggregates_same_price() -> None:
    book = _LimitOrderBook()
    book.add(price=1000, size=100, direction=1)
    book.add(price=1000, size=200, direction=1)
    snap = book.snapshot(levels=1)
    assert snap[3] == 300  # bid size at level 1


# ── _timestamp_ns ─────────────────────────────────────────────────────────────


def test_timestamp_reconstruction() -> None:
    ns = 34_200_000_000_000  # 9:30 AM in ns
    ts_hi_bytes = struct.pack(">H", (ns >> 32) & 0xFFFF)
    ts_lo = ns & 0xFFFFFFFF
    result = _timestamp_ns(ts_hi_bytes, ts_lo)
    assert result == ns


# ── parse_itch integration tests ──────────────────────────────────────────────


def test_parse_add_and_delete(tmp_path: Path) -> None:
    """Add one bid order then delete it — should produce 2 message rows."""
    gz = tmp_path / "01012020.NASDAQ_ITCH50.gz"
    msgs = [
        _stock_dir_msg(locate=1, stock="AAPL"),
        _add_order_msg(1, ts_ns=34_200_000_000_000, order_ref=1,
                       side="buy", shares=100, stock="AAPL", price=1_500_000),
        _delete_order_msg(1, ts_ns=34_200_001_000_000, order_ref=1),
    ]
    _make_itch_gz(msgs, gz)

    msg_p, ob_p = parse_itch(gz, "AAPL", tmp_path, levels=1)
    lines = msg_p.read_text().strip().split("\n")
    # header stripped (no header), 2 data rows
    assert len(lines) == 2
    # first row: event_type=1 (new limit)
    assert lines[0].split(",")[1] == "1"
    # second row: event_type=3 (full cancel)
    assert lines[1].split(",")[1] == "3"


def test_parse_replace_emits_delete_then_add(tmp_path: Path) -> None:
    """Replace should emit a full cancel + new limit."""
    gz = tmp_path / "01012020.NASDAQ_ITCH50.gz"
    msgs = [
        _stock_dir_msg(locate=1, stock="AAPL"),
        _add_order_msg(1, 34_200_000_000_000, 1, "buy", 100, "AAPL", 1_500_000),
        _replace_order_msg(1, 34_200_001_000_000, orig_ref=1, new_ref=2,
                           shares=200, price=1_499_000),
    ]
    _make_itch_gz(msgs, gz)

    msg_p, ob_p = parse_itch(gz, "AAPL", tmp_path, levels=1)
    lines = msg_p.read_text().strip().split("\n")
    assert len(lines) == 3
    event_types = [l.split(",")[1] for l in lines]
    assert event_types == ["1", "3", "1"]  # add, delete-old, add-new


def test_parse_execution_reduces_book(tmp_path: Path) -> None:
    """Partial execution should reduce book size, full execution should remove order."""
    gz = tmp_path / "01012020.NASDAQ_ITCH50.gz"
    msgs = [
        _stock_dir_msg(locate=1, stock="AAPL"),
        _add_order_msg(1, 34_200_000_000_000, 1, "sell", 100, "AAPL", 1_510_000),
        _execute_msg(1, 34_200_001_000_000, order_ref=1, exec_shares=100),
    ]
    _make_itch_gz(msgs, gz)

    msg_p, ob_p = parse_itch(gz, "AAPL", tmp_path, levels=1)
    lines = msg_p.read_text().strip().split("\n")
    assert len(lines) == 2
    assert lines[1].split(",")[1] == "4"  # exec_visible

    ob_lines = ob_p.read_text().strip().split("\n")
    # row 0 = snapshot after add, row 1 = snapshot after exec
    snap_after_exec = ob_lines[1].split(",")
    assert snap_after_exec[0] == "-1"  # ask_price_1 gone


def test_parse_ignores_other_tickers(tmp_path: Path) -> None:
    """Events for MSFT should not appear in AAPL output."""
    gz = tmp_path / "01012020.NASDAQ_ITCH50.gz"
    msgs = [
        _stock_dir_msg(locate=1, stock="AAPL"),
        _stock_dir_msg(locate=2, stock="MSFT"),
        _add_order_msg(1, 34_200_000_000_000, 1, "buy", 100, "AAPL", 1_500_000),
        _add_order_msg(2, 34_200_001_000_000, 2, "buy", 200, "MSFT", 500_000),
    ]
    _make_itch_gz(msgs, gz)

    msg_p, ob_p = parse_itch(gz, "AAPL", tmp_path, levels=1)
    lines = msg_p.read_text().strip().split("\n")
    assert len(lines) == 1
    assert lines[0].split(",")[2] == "1"  # order_ref=1 (AAPL only)


def test_timestamp_in_seconds(tmp_path: Path) -> None:
    """Timestamps in output should be seconds since midnight, not nanoseconds."""
    gz = tmp_path / "01012020.NASDAQ_ITCH50.gz"
    ns = 34_200_000_000_000  # exactly 9:30:00.000 AM = 34200 seconds
    msgs = [
        _stock_dir_msg(locate=1, stock="AAPL"),
        _add_order_msg(1, ns, 1, "buy", 100, "AAPL", 1_500_000),
    ]
    _make_itch_gz(msgs, gz)

    msg_p, _ = parse_itch(gz, "AAPL", tmp_path, levels=1)
    ts_str = msg_p.read_text().strip().split("\n")[0].split(",")[0]
    ts = float(ts_str)
    assert abs(ts - 34200.0) < 1e-3
