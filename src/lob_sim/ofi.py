"""
Order Flow Imbalance (Cont, Kukanov, Stoikov 2014).

OFI measures net order-book pressure at the best bid/ask across consecutive
events.  For each event transition (t-1 → t):

  bid_contribution =
      +best_bid_size[t]               if best_bid_price rose     (new best bid appeared)
      +best_bid_size[t] - [t-1]       if best_bid_price unchanged (size change)
      -best_bid_size[t-1]             if best_bid_price fell      (best bid swept away)

  ask_contribution = symmetric (sign convention: ask pressure cancels bid pressure)
      +best_ask_size[t]               if best_ask_price fell      (new aggressive ask)
      +best_ask_size[t-1] - [t]       if best_ask_price unchanged
      -best_ask_size[t-1]             if best_ask_price rose      (asks pulled)

  OFI[t] = bid_contribution - ask_contribution

OFI > 0 → net bid-side pressure → upward price signal.
OFI < 0 → net ask-side pressure → downward price signal.

Units: shares (raw; normalise by window volume or use StandardScaler downstream).
"""

from __future__ import annotations

import numpy as np


def compute_ofi(
    bid_prices: np.ndarray,
    bid_sizes:  np.ndarray,
    ask_prices: np.ndarray,
    ask_sizes:  np.ndarray,
) -> np.ndarray:
    """
    Vectorised per-event OFI from arrays of best-bid/ask price and size.

    Parameters (all length N, integer dtype)
    ----------
    bid_prices, bid_sizes : best bid price and size after each event
    ask_prices, ask_sizes : best ask price and size after each event
                            Sentinel -1 means side is empty.

    Returns
    -------
    ofi : np.ndarray, shape (N,), float64
        First element is always 0 (no prior state).
        Events where either side was empty (sentinel) are set to 0.
    """
    n = len(bid_prices)
    if n < 2:
        return np.zeros(n, dtype=float)

    bp_p = bid_prices[:-1].astype(float)
    bs_p = bid_sizes[:-1].astype(float)
    ap_p = ask_prices[:-1].astype(float)
    as_p = ask_sizes[:-1].astype(float)

    bp_c = bid_prices[1:].astype(float)
    bs_c = bid_sizes[1:].astype(float)
    ap_c = ask_prices[1:].astype(float)
    as_c = ask_sizes[1:].astype(float)

    bid_contribution = np.where(
        bp_c > bp_p, bs_c,
        np.where(bp_c == bp_p, bs_c - bs_p, -bs_p)
    )

    ask_contribution = np.where(
        ap_c < ap_p, as_c,
        np.where(ap_c == ap_p, as_p - as_c, -as_p)
    )

    ofi_inner = bid_contribution - ask_contribution

    # Zero out transitions where either side was empty (sentinel = -1)
    valid = (bp_p > 0) & (ap_p > 0) & (bp_c > 0) & (ap_c > 0)
    ofi_inner = np.where(valid, ofi_inner, 0.0)

    return np.concatenate([[0.0], ofi_inner])


def windowed_ofi(
    timestamps:      np.ndarray,
    cumulative_ofi:  np.ndarray,
    query_ts:        float,
    window_s:        float,
) -> float:
    """
    Sum of OFI in the half-open interval [query_ts - window_s, query_ts).

    Parameters
    ----------
    timestamps     : sorted float array of event timestamps (seconds since midnight)
    cumulative_ofi : np.cumsum of the per-event OFI array (same length)
    query_ts       : injection timestamp
    window_s       : look-back window in seconds

    Returns
    -------
    float : windowed OFI in shares; 0.0 if no events in window
    """
    hi = int(np.searchsorted(timestamps, query_ts,            side="left"))
    lo = int(np.searchsorted(timestamps, query_ts - window_s, side="left"))
    if hi == 0:
        return 0.0
    hi_val = cumulative_ofi[hi - 1]
    lo_val = cumulative_ofi[lo - 1] if lo > 0 else 0.0
    return float(hi_val - lo_val)
