"""Unit tests for run_experiment."""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl
import pytest

from lob_sim.experiment import _DAY_END, _DAY_START, run_experiment
from lob_sim.explore import find_lobster_pair


# ── helpers ───────────────────────────────────────────────────────────────────

DATA_DIR = Path("data/raw")


def _has_data() -> bool:
    try:
        find_lobster_pair(DATA_DIR)
        return True
    except (FileNotFoundError, StopIteration, Exception):
        return False


requires_data = pytest.mark.skipif(
    not _has_data(), reason="LOBSTER data not present in data/raw"
)


# ── structural tests (no data required) ──────────────────────────────────────


def test_day_boundaries_sensible() -> None:
    """DAY_START < DAY_END and both fall within regular US equity hours."""
    assert _DAY_START < _DAY_END
    assert 34_200 <= _DAY_START   # 09:30 or later
    assert _DAY_END <= 57_600     # 16:00 or earlier


# ── integration tests (require data/raw) ─────────────────────────────────────


@requires_data
def test_experiment_returns_dataframe() -> None:
    """run_experiment returns a non-empty polars DataFrame."""
    with tempfile.TemporaryDirectory() as tmp:
        df = run_experiment(n_injections=50, out_dir=tmp)
    assert isinstance(df, pl.DataFrame)
    assert len(df) > 0


@requires_data
def test_experiment_output_columns() -> None:
    """Output DataFrame has all required columns for Hypothesis 2 regression."""
    required = {
        "injection_id", "entry_timestamp", "side", "price",
        "queue_position_at_entry", "orders_ahead_at_entry",
        "queue_granularity_at_entry", "spread_at_entry_ticks",
        "book_imbalance_at_entry", "filled", "time_to_first_fill", "status",
    }
    with tempfile.TemporaryDirectory() as tmp:
        df = run_experiment(n_injections=50, out_dir=tmp)
    assert required.issubset(set(df.columns))


@requires_data
def test_experiment_sides_alternate() -> None:
    """Injected orders alternate bid/ask (even id → bid, odd id → ask)."""
    with tempfile.TemporaryDirectory() as tmp:
        df = run_experiment(n_injections=100, out_dir=tmp)
    for row in df.iter_rows(named=True):
        expected = "bid" if row["injection_id"] % 2 == 0 else "ask"
        assert row["side"] == expected, (
            f"injection_id {row['injection_id']} expected {expected}, got {row['side']}"
        )


@requires_data
def test_experiment_granularity_nonnegative() -> None:
    """queue_granularity_at_entry is always >= 0."""
    with tempfile.TemporaryDirectory() as tmp:
        df = run_experiment(n_injections=100, out_dir=tmp)
    assert (df["queue_granularity_at_entry"] >= 0).all()


@requires_data
def test_experiment_granularity_consistent_with_queue() -> None:
    """
    When queue_position_at_entry > 0, granularity == orders_ahead / shares_ahead.
    When empty (queue_position_at_entry == 0), granularity must be 0.0.
    """
    with tempfile.TemporaryDirectory() as tmp:
        df = run_experiment(n_injections=200, out_dir=tmp)

    for row in df.iter_rows(named=True):
        q = row["queue_position_at_entry"]
        k = row["orders_ahead_at_entry"]
        g = row["queue_granularity_at_entry"]
        if q > 0:
            assert abs(g - k / q) < 1e-9, f"granularity mismatch: {g} != {k}/{q}"
        else:
            assert g == 0.0


@requires_data
def test_experiment_min_queue_filter_respected() -> None:
    """Every injected order has queue_position_at_entry >= min_queue_shares."""
    min_q = 100
    with tempfile.TemporaryDirectory() as tmp:
        df = run_experiment(n_injections=200, min_queue_shares=min_q, out_dir=tmp)
    assert (df["queue_position_at_entry"] >= min_q).all()


@requires_data
def test_experiment_reproducible() -> None:
    """Same seed produces identical output."""
    with tempfile.TemporaryDirectory() as tmp:
        df1 = run_experiment(n_injections=50, seed=7, out_dir=tmp)
        df2 = run_experiment(n_injections=50, seed=7, out_dir=tmp)
    assert df1.equals(df2)


@requires_data
def test_experiment_different_seeds_differ() -> None:
    """Different seeds produce different injection timestamps."""
    with tempfile.TemporaryDirectory() as tmp:
        df1 = run_experiment(n_injections=50, seed=1, out_dir=tmp)
        df2 = run_experiment(n_injections=50, seed=2, out_dir=tmp)
    assert not df1["entry_timestamp"].equals(df2["entry_timestamp"])


@requires_data
def test_experiment_parquet_written() -> None:
    """A parquet file is written to out_dir after the run, filename includes depth level."""
    with tempfile.TemporaryDirectory() as tmp:
        run_experiment(n_injections=50, depth_level=1, out_dir=tmp)
        parquet_files = list(Path(tmp).glob("*_L1.parquet"))
        assert len(parquet_files) == 1
        assert parquet_files[0].stat().st_size > 0


@requires_data
def test_experiment_timestamps_within_day() -> None:
    """All injection timestamps fall within the sampling window."""
    with tempfile.TemporaryDirectory() as tmp:
        df = run_experiment(n_injections=100, out_dir=tmp)
    assert (df["entry_timestamp"] >= _DAY_START).all()
    assert (df["entry_timestamp"] <= _DAY_END + 10).all()  # +10s tolerance for last event


@requires_data
def test_depth_level_2_lower_fill_rate() -> None:
    """
    Level-2 orders are deeper in the queue than level-1 — fill rate must be
    strictly lower (level 2 is further from the touch).
    """
    with tempfile.TemporaryDirectory() as tmp:
        df1 = run_experiment(n_injections=200, depth_level=1, seed=42, out_dir=tmp)
        df2 = run_experiment(n_injections=200, depth_level=2, seed=42, out_dir=tmp)
    fill1 = df1["filled"].mean()
    fill2 = df2["filled"].mean()
    assert fill2 < fill1, (
        f"Expected depth-2 fill rate ({fill2:.1%}) < depth-1 ({fill1:.1%})"
    )


@requires_data
def test_depth_level_2_separate_parquet() -> None:
    """depth_level=1 and depth_level=2 write to different parquet files."""
    with tempfile.TemporaryDirectory() as tmp:
        run_experiment(n_injections=50, depth_level=1, out_dir=tmp)
        run_experiment(n_injections=50, depth_level=2, out_dir=tmp)
        parquet_files = list(Path(tmp).glob("*.parquet"))
    assert len(parquet_files) == 2


def test_depth_level_invalid() -> None:
    """depth_level values other than 1 or 2 raise ValueError before any data loading."""
    with pytest.raises(ValueError, match="depth_level"):
        run_experiment(depth_level=3)
