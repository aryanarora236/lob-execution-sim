"""Unit tests for lob_sim.ofi."""

from __future__ import annotations

import numpy as np
import pytest

from lob_sim.ofi import compute_ofi, windowed_ofi


# ── compute_ofi ───────────────────────────────────────────────────────────────

def test_ofi_first_element_is_zero():
    bp = np.array([1000, 1001, 1001])
    bs = np.array([100,  200,  200])
    ap = np.array([1002, 1002, 1003])
    as_ = np.array([100, 100,  50])
    ofi = compute_ofi(bp, bs, ap, as_)
    assert ofi[0] == 0.0


def test_ofi_bid_price_rises():
    """Best bid price rises → full new bid size is positive bid contribution."""
    bp  = np.array([1000, 1001])
    bs  = np.array([100,  150])
    ap  = np.array([1002, 1002])
    as_ = np.array([100,  100])
    ofi = compute_ofi(bp, bs, ap, as_)
    # bid_c = +150 (new level), ask_c = 0 (unchanged price, unchanged size)
    assert ofi[1] == pytest.approx(150.0)


def test_ofi_ask_price_falls():
    """Best ask price falls → full new ask size is positive ask contribution → negative OFI."""
    bp  = np.array([1000, 1000])
    bs  = np.array([100,  100])
    ap  = np.array([1002, 1001])
    as_ = np.array([100,  200])
    ofi = compute_ofi(bp, bs, ap, as_)
    # bid_c = 0 (unchanged), ask_c = +200 → OFI = 0 - 200 = -200
    assert ofi[1] == pytest.approx(-200.0)


def test_ofi_unchanged_prices_size_increase():
    """Both sides unchanged price, bid grows, ask shrinks → positive OFI."""
    bp  = np.array([1000, 1000])
    bs  = np.array([100,  300])
    ap  = np.array([1002, 1002])
    as_ = np.array([200,  50])
    ofi = compute_ofi(bp, bs, ap, as_)
    # bid_c = 300 - 100 = +200, ask_c = 200 - 50 = +150 → OFI = 200 - 150 = 50
    assert ofi[1] == pytest.approx(50.0)


def test_ofi_sentinel_zeroed():
    """Transitions involving sentinel (-1) produce OFI = 0."""
    bp  = np.array([-1, 1000])
    bs  = np.array([0,  100])
    ap  = np.array([-1, 1002])
    as_ = np.array([0,  100])
    ofi = compute_ofi(bp, bs, ap, as_)
    assert ofi[1] == 0.0


def test_ofi_length_matches_input():
    n = 50
    rng = np.random.default_rng(0)
    bp  = np.sort(rng.integers(990, 1010, n))
    bs  = rng.integers(100, 500, n)
    ap  = bp + rng.integers(1, 5, n)
    as_ = rng.integers(100, 500, n)
    ofi = compute_ofi(bp, bs, ap, as_)
    assert len(ofi) == n


def test_ofi_single_event():
    ofi = compute_ofi(
        np.array([1000]), np.array([100]),
        np.array([1002]), np.array([100]),
    )
    assert len(ofi) == 1
    assert ofi[0] == 0.0


# ── windowed_ofi ──────────────────────────────────────────────────────────────

def _make_series():
    """Simple deterministic OFI series for window tests."""
    # 10 events, 1 second apart, OFI = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ts  = np.arange(10, dtype=float)        # 0..9 s
    ofi = np.arange(10, dtype=float)        # per-event OFI
    cum = np.cumsum(ofi)                    # [0,1,3,6,10,15,21,28,36,45]
    return ts, cum


def test_windowed_ofi_full_range():
    ts, cum = _make_series()
    # query at t=10 (past end), window=10 → sum(0..9) = 45
    result = windowed_ofi(ts, cum, 10.0, 10.0)
    assert result == pytest.approx(45.0)


def test_windowed_ofi_partial_window():
    ts, cum = _make_series()
    # query at t=5, window=3 → events at t=2,3,4 → OFI = 2+3+4 = 9
    result = windowed_ofi(ts, cum, 5.0, 3.0)
    assert result == pytest.approx(9.0)


def test_windowed_ofi_no_events_in_window():
    ts, cum = _make_series()
    # query before any events
    result = windowed_ofi(ts, cum, -1.0, 5.0)
    assert result == 0.0


def test_windowed_ofi_zero_window():
    ts, cum = _make_series()
    result = windowed_ofi(ts, cum, 5.0, 0.0)
    assert result == pytest.approx(0.0)
