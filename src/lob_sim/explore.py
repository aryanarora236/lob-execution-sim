"""
Exploratory analysis of raw LOBSTER data files.

Run directly: uv run python -m lob_sim.explore
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

# ── column definitions ────────────────────────────────────────────────────────

MESSAGE_COLS = ["time", "event_type", "order_id", "size", "price", "direction"]
MESSAGE_DTYPES = {
    "time": pl.Float64,
    "event_type": pl.Int32,
    "order_id": pl.Int64,
    "size": pl.Int64,
    "price": pl.Int64,
    "direction": pl.Int32,
}

EVENT_LABELS = {
    1: "new_limit",
    2: "partial_cancel",
    3: "full_cancel",
    4: "exec_visible",
    5: "exec_hidden",
    6: "cross_trade",
    7: "trading_halt",
}

PRICE_SCALE = 10_000  # LOBSTER stores price as integer; divide by 10000 for dollars


# ── file discovery ────────────────────────────────────────────────────────────


def find_lobster_pair(data_dir: Path) -> tuple[Path, Path]:
    """Return (message_file, orderbook_file) for the first LOBSTER pair found."""
    pairs = find_all_lobster_pairs(data_dir)
    if not pairs:
        raise FileNotFoundError(
            f"No LOBSTER message files found in {data_dir}.\n"
            "Expected pattern: TICKER_DATE_START_END_message_LEVELS.csv"
        )
    return pairs[0]


def find_all_lobster_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    """Return all (message_file, orderbook_file) pairs found in data_dir."""
    msg_files = sorted(data_dir.glob("*_message_*.csv"))
    pairs = []
    for msg in msg_files:
        ob = Path(str(msg).replace("_message_", "_orderbook_"))
        if ob.exists():
            pairs.append((msg, ob))
    return pairs


def parse_filename_meta(path: Path) -> dict[str, str]:
    """Extract ticker, date, and levels from a LOBSTER filename."""
    m = re.match(
        r"(?P<ticker>[A-Z]+)_(?P<date>\d{4}-\d{2}-\d{2})_\d+_\d+_message_(?P<levels>\d+)\.csv",
        path.name,
    )
    if not m:
        return {"ticker": "UNKNOWN", "date": "UNKNOWN", "levels": "UNKNOWN"}
    return m.groupdict()


# ── loaders ───────────────────────────────────────────────────────────────────


def load_messages(path: Path) -> pl.DataFrame:
    """Load LOBSTER message file into a polars DataFrame."""
    return pl.read_csv(
        path,
        has_header=False,
        new_columns=MESSAGE_COLS,
        schema_overrides=MESSAGE_DTYPES,
    )


def load_orderbook(path: Path, levels: int) -> pl.DataFrame:
    """Load LOBSTER orderbook snapshot file into a polars DataFrame."""
    cols: list[str] = []
    for lvl in range(1, levels + 1):
        cols += [f"ask_price_{lvl}", f"ask_size_{lvl}", f"bid_price_{lvl}", f"bid_size_{lvl}"]
    return pl.read_csv(path, has_header=False, new_columns=cols)


# ── analysis helpers ──────────────────────────────────────────────────────────


def seconds_to_hms(sec: float) -> str:
    """Convert seconds-since-midnight to HH:MM:SS string."""
    h = int(sec) // 3600
    m = (int(sec) % 3600) // 60
    s = int(sec) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def event_type_distribution(messages: pl.DataFrame) -> pl.DataFrame:
    """Return event type counts and percentages, sorted by type."""
    total = len(messages)
    return (
        messages.group_by("event_type")
        .agg(pl.len().alias("count"))
        .with_columns(
            (pl.col("count") / total * 100).round(2).alias("pct"),
            pl.col("event_type").replace_strict(
                list(EVENT_LABELS.keys()),
                list(EVENT_LABELS.values()),
                default="unknown",
            ).alias("label"),
        )
        .sort("event_type")
    )


def compute_top_of_book_stats(messages: pl.DataFrame, orderbook: pl.DataFrame) -> dict[str, float]:
    """Compute mean spread and mean size at best bid/ask."""
    best_ask = orderbook["ask_price_1"].cast(pl.Float64) / PRICE_SCALE
    best_bid = orderbook["bid_price_1"].cast(pl.Float64) / PRICE_SCALE
    spread = best_ask - best_bid
    mid = (best_ask + best_bid) / 2

    ask_size = orderbook["ask_size_1"].cast(pl.Float64)
    bid_size = orderbook["bid_size_1"].cast(pl.Float64)

    # fraction of events that change best bid or ask
    ask_changed = (orderbook["ask_price_1"].diff().fill_null(0) != 0) | (
        orderbook["ask_size_1"].diff().fill_null(0) != 0
    )
    bid_changed = (orderbook["bid_price_1"].diff().fill_null(0) != 0) | (
        orderbook["bid_size_1"].diff().fill_null(0) != 0
    )
    top_changed = (ask_changed | bid_changed).cast(pl.Float64)

    return {
        "mean_spread_cents": float(spread.mean() * 100),
        "median_spread_cents": float(spread.median() * 100),
        "mean_ask_size": float(ask_size.mean()),
        "mean_bid_size": float(bid_size.mean()),
        "mean_mid_price": float(mid.mean()),
        "frac_events_change_top": float(top_changed.mean()),
    }


def detect_anomalies(messages: pl.DataFrame, orderbook: pl.DataFrame) -> list[str]:
    """Run basic sanity checks and return a list of warning strings."""
    warnings: list[str] = []

    unknown_types = messages.filter(~pl.col("event_type").is_in(list(EVENT_LABELS.keys())))
    if len(unknown_types) > 0:
        warnings.append(f"  ! {len(unknown_types)} events with unknown event_type: "
                        f"{unknown_types['event_type'].unique().to_list()}")

    crossed = orderbook.filter(pl.col("bid_price_1") >= pl.col("ask_price_1"))
    if len(crossed) > 0:
        warnings.append(f"  ! {len(crossed)} snapshots with crossed/locked book (bid >= ask)")

    neg_spread = orderbook.filter(pl.col("ask_price_1") - pl.col("bid_price_1") < 0)
    if len(neg_spread) > 0:
        warnings.append(f"  ! {len(neg_spread)} snapshots with negative spread")

    zero_prices = orderbook.filter(
        (pl.col("ask_price_1") == 0) | (pl.col("bid_price_1") == 0)
    )
    if len(zero_prices) > 0:
        warnings.append(f"  ! {len(zero_prices)} snapshots with zero price at best level "
                        "(likely trading halts — check event_type=7 rows)")

    bad_direction = messages.filter(~pl.col("direction").is_in([-1, 1]))
    if len(bad_direction) > 0:
        warnings.append(f"  ! {len(bad_direction)} messages with unexpected direction "
                        f"(not -1 or 1): {bad_direction['direction'].unique().to_list()}")

    if len(messages) != len(orderbook):
        warnings.append(
            f"  ! Row count mismatch: {len(messages)} messages vs {len(orderbook)} orderbook rows"
        )

    return warnings


# ── plots ─────────────────────────────────────────────────────────────────────


def plot_event_type_histogram(dist: pl.DataFrame, out_path: Path, meta: dict[str, str]) -> None:
    """Save bar chart of event type distribution."""
    labels = [f"{row['label']}\n({row['event_type']})" for row in dist.iter_rows(named=True)]
    counts = dist["count"].to_list()

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(labels, counts, color="#4a90d9", edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=8)
    ax.set_title(f"{meta['ticker']} {meta['date']} — Event Type Distribution")
    ax.set_ylabel("Count")
    ax.set_xlabel("Event Type")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_mid_price(messages: pl.DataFrame, orderbook: pl.DataFrame, out_path: Path, meta: dict[str, str]) -> None:
    """Save time-series plot of mid-price over the trading day."""
    times = messages["time"].to_numpy()
    best_ask = orderbook["ask_price_1"].to_numpy() / PRICE_SCALE
    best_bid = orderbook["bid_price_1"].to_numpy() / PRICE_SCALE
    mid = (best_ask + best_bid) / 2

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times / 3600, mid, linewidth=0.4, color="#2c7bb6", alpha=0.8)
    ax.set_title(f"{meta['ticker']} {meta['date']} — Mid-Price Over Day")
    ax.set_xlabel("Time (hours since midnight)")
    ax.set_ylabel("Mid-Price ($)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):02d}:{int((x%1)*60):02d}"))
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────


def run(data_dir: Path = Path("data/raw"), results_dir: Path = Path("results/exploration")) -> None:
    """Run full exploratory analysis and print report."""
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("LOBSTER DATA EXPLORATION")
    print("=" * 65)

    # locate files
    msg_path, ob_path = find_lobster_pair(data_dir)
    meta = parse_filename_meta(msg_path)
    levels = int(meta["levels"]) if meta["levels"].isdigit() else 10

    print(f"\nTicker : {meta['ticker']}")
    print(f"Date   : {meta['date']}")
    print(f"Levels : {levels}")
    print(f"Files  : {msg_path.name}")
    print(f"         {ob_path.name}")

    # load
    print("\nLoading files...", end=" ", flush=True)
    messages = load_messages(msg_path)
    orderbook = load_orderbook(ob_path, levels)
    print("done.")

    # ── basic stats ───────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("ROW COUNTS")
    print(f"{'─'*65}")
    print(f"  Message rows  : {len(messages):,}")
    print(f"  Orderbook rows: {len(orderbook):,}")
    if len(messages) != len(orderbook):
        print("  WARNING: counts differ!")

    print(f"\n{'─'*65}")
    print("TIME RANGE")
    print(f"{'─'*65}")
    t_min = messages["time"].min()
    t_max = messages["time"].max()
    duration = t_max - t_min
    print(f"  Start  : {seconds_to_hms(t_min)}  ({t_min:.3f}s since midnight)")
    print(f"  End    : {seconds_to_hms(t_max)}  ({t_max:.3f}s since midnight)")
    print(f"  Duration: {duration/3600:.2f} hours  ({duration:.1f}s)")

    print(f"\n{'─'*65}")
    print("PRICE RANGE (message file, dollars)")
    print(f"{'─'*65}")
    prices = messages["price"].cast(pl.Float64) / PRICE_SCALE
    print(f"  Min  : ${prices.min():.4f}")
    print(f"  Max  : ${prices.max():.4f}")
    print(f"  Mean : ${prices.mean():.4f}")

    print(f"\n{'─'*65}")
    print("EVENT TYPE DISTRIBUTION")
    print(f"{'─'*65}")
    dist = event_type_distribution(messages)
    for row in dist.iter_rows(named=True):
        label = EVENT_LABELS.get(row["event_type"], "unknown")
        print(f"  type {row['event_type']} ({label:<15}): {row['count']:>8,}  ({row['pct']:5.1f}%)")

    print(f"\n{'─'*65}")
    print("TOP-OF-BOOK STATS")
    print(f"{'─'*65}")
    tob = compute_top_of_book_stats(messages, orderbook)
    print(f"  Mean spread      : {tob['mean_spread_cents']:.3f} cents")
    print(f"  Median spread    : {tob['median_spread_cents']:.3f} cents")
    print(f"  Mean ask size    : {tob['mean_ask_size']:.1f} shares")
    print(f"  Mean bid size    : {tob['mean_bid_size']:.1f} shares")
    print(f"  Mean mid-price   : ${tob['mean_mid_price']:.4f}")
    print(f"  Events that change top-of-book: {tob['frac_events_change_top']*100:.1f}%")

    # ── anomaly checks ────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("SANITY CHECKS")
    print(f"{'─'*65}")
    warnings = detect_anomalies(messages, orderbook)
    if warnings:
        for w in warnings:
            print(w)
    else:
        print("  All checks passed.")

    # ── 20-row samples ────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("MESSAGE FILE — 20-ROW SAMPLE")
    print(f"{'─'*65}")
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=120):
        print(messages.head(20))

    print(f"\n{'─'*65}")
    print("ORDERBOOK FILE — 20-ROW SAMPLE (first 4 levels)")
    print(f"{'─'*65}")
    ob_preview_cols = [c for c in orderbook.columns if int(c.split("_")[-1]) <= 4]
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=120):
        print(orderbook.select(ob_preview_cols).head(20))

    # ── plots ─────────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("PLOTS")
    print(f"{'─'*65}")
    plot_event_type_histogram(dist, results_dir / "event_type_histogram.png", meta)
    plot_mid_price(messages, orderbook, results_dir / "mid_price.png", meta)

    print(f"\n{'='*65}")
    print("EXPLORATION COMPLETE")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
    run(data_dir=data_dir)
