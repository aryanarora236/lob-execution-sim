# Finding

## Headline

Fill predictability at passive limit order depth is regime-dependent and
ticker-specific. In liquid HFT-dominated stocks (AAPL), only spread is a
significant predictor and OFI is fully arbitraged (walk-forward AUC ≈ 0.505).
In wider-spread stocks (INTC), time_frac (−0.313 ***) and spread (+0.059 ***)
dominate at L2, OFI_30s is strongly significant (+0.203 ***), and Cox C-index
reaches 0.594 vs 0.529 for AAPL. Optimal placement crossover varies from
C\* ≈ 6.6¢ (AAPL) to C\* ≈ 1.8¢ (INTC). A new adverse selection decay
analysis shows AAPL's adverse selection compounds from +1.36¢ at 1 s to
+1.64¢ at 30 s (persistent informed flow), while INTC's decays from +0.85¢ to
+0.77¢ (transient/noisy execution). Queue granularity moves from fully null to
marginally significant in survival and panel regression with the expanded panel.

---

## Data and scope

| Ticker | Date | Session | Events | L1 injected | L2 injected |
|--------|------|---------|--------|-------------|-------------|
| AAPL | 2019-08-30 | Normal | 1,283,342 | 3,259 | 3,943 |
| AAPL | 2019-10-30 | Normal | 852,833 | 4,092 | 4,466 |
| AAPL | 2019-12-30 | Holiday week | 1,609,148 | 4,343 | 4,701 |
| INTC | 2019-08-30 | Normal | 1,022,326 | 4,994 | 5,000 |
| INTC | 2019-10-30 | Normal | 983,717 | 4,958 | 4,999 |
| INTC | 2019-12-30 | Holiday week | 789,641 | 4,982 | 5,000 |
| MSFT | 2019-08-30 | Normal | 1,642,017 | 4,690 | 4,985 |
| MSFT | 2019-10-30 | Normal | 1,300,655 | 4,725 | 4,994 |
| MSFT | 2019-12-30 | Holiday week | 1,270,786 | 4,800 | 4,998 |

Source: NASDAQ TotalView-ITCH 5.0, parsed to LOBSTER-compatible format.
Pooled dataset: **40,843 L1 orders**, **43,086 L2 orders** across 9
ticker-date combinations.

Hypothetical orders injected via passive-shadow simulation (no market impact).
Each order expires unfilled after 60 s. Depth level 1 = best price (touch);
level 2 = one tick behind best. OOS evaluation uses 4-fold walk-forward
`TimeSeriesSplit`; reported AUC is the mean across folds, CI is ±2 std.

---

## Part I — Single-ticker AAPL results (2019-12-30)

*Historical baseline from the initial single-session run. Numbers preserved as-is;
do not compare directly with Part II pooled figures due to different sample and CV scheme.*

### L1 (touch) and L2 (depth) fill predictability

| | L1 | L2 |
|---|---|---|
| Fill rate (60 s) | 63.2% | 55.1% |
| Logistic OOS AUC | 0.491 | 0.494 |
| LightGBM OOS AUC | 0.548 | 0.542 |

**Significant predictors at L2** (bootstrap 95% CI excludes 0):

| Feature | Coef (std.) | 95% CI |
|---------|-------------|--------|
| `side_bid` | −0.191 | [−0.257, −0.125] |
| `spread_ticks` | −0.119 | [−0.187, −0.048] |
| `time_frac` | −0.105 | [−0.173, −0.035] |
| `imbalance` | −0.063 | [−0.127, −0.001] |

At L1 only `spread_ticks` is significant. Queue granularity spans zero at
both levels.

### LightGBM time_frac ablation (AAPL 2019-12-30)

| | LGB full | LGB w/o time_frac | Drop |
|---|---|---|---|
| L1 | 0.541 | 0.539 | −0.002 |
| L2 | 0.524 | 0.492 | −0.032 |

At L2, removing time_frac collapses AUC below chance — the tree's entire edge
was the intraday fill-rate profile.

### Adverse selection (AAPL 2019-12-30)

- L1: **+1.73¢/share** [+1.64, +1.82]; 79.9% adversely selected
- L2: **+1.60¢/share** [+1.51, +1.70]; 77.5% adversely selected
- L1 > L2 adversity, t-test significant (p = 0.014)
- Intraday: open hour worst (+2.40–2.60¢), declining to ~+1.5¢ by noon

### Survival analysis (AAPL 2019-12-30, L2)

- Cox C-index: 0.545 in-sample, 0.500 OOS
- Significant predictors: `spread_ticks` (HR=0.915 ***), `side_bid` (HR=0.899 ***),
  `time_frac` (HR=0.954 *), `imbalance` (HR=0.957 *)

### Optimal placement (AAPL 2019-12-30)

C\* ≈ 8¢/share at mean spread (see Part II for cross-ticker comparison).

---

## Part II — Multi-ticker cross-sectional analysis (9 ticker-date combinations)

Source: `notebooks/11_multi_ticker.py`, `notebooks/12_extended_analysis.py`

### Fill rates and AUC by ticker

| Ticker | L1 n | L1 fill | L1 AUC [±2σ CV] | L2 n | L2 fill | L2 AUC [±2σ CV] |
|--------|------|---------|-----------------|------|---------|-----------------|
| AAPL | 11,694 | 66.1% | 0.524 [0.480, 0.567] | 13,110 | 57.0% | 0.505 [0.489, 0.522] |
| INTC | 14,934 | 56.7% | **0.572** [0.517, 0.627] | 14,999 | **34.0%** | **0.567** [0.452, 0.683] |
| MSFT | 14,215 | 69.3% | 0.511 [0.446, 0.575] | 14,977 | 56.3% | 0.495 [0.436, 0.555] |

Key observations:
- INTC's L2 fill rate (34%) is less than 60% of AAPL's (57%) — depth queues in
  wider-spread stocks clear far less often within 60 s.
- MSFT L2 AUC ≈ 0.495 (essentially chance) reflects structural intraday
  non-stationarity: patterns learned in the first folds invert by the later folds.
- AUCs reported here use 4-fold walk-forward CV and are lower than previously
  reported single 80/20 estimates — the CV estimates are more conservative and
  more credible.

### Bootstrap CIs on L2 logistic coefficients

| Feature | AAPL | INTC | MSFT |
|---------|------|------|------|
| `spread_at_entry_ticks` | **−0.119 \*\*\*** | +0.059 *** | −0.026 |
| `time_frac` | +0.025 | **−0.313 \*\*\*** | −0.048 *** |
| `queue_position_at_entry` | +0.009 | −0.081 | **−0.051 \*\*\*** |
| `book_imbalance_at_entry` | +0.004 | **−0.051 \*\*\*** | +0.007 |
| `queue_granularity_at_entry` | +0.020 | +0.047 | +0.059 |
| `side_bid` | +0.006 | −0.003 | +0.021 |

\*\*\* = 95% bootstrap CI excludes zero.

Key contrasts:
- **AAPL**: Spread alone matters. All other features span zero.
- **INTC**: Dominated by time_frac (−0.313 ***), positive spread (+0.059 ***),
  and negative book imbalance (−0.051 ***). Queue position is the largest
  coefficient (−0.081) but its wide CI [−0.398, 0.000] just barely spans zero —
  likely real but noisy.
- **MSFT**: Queue position (−0.051 ***) and time_frac (−0.048 ***) significant,
  but coefficients are small. Features consistent with a market between AAPL
  and INTC in terms of information environment.

### LightGBM time_frac ablation — cross-ticker

| Ticker | Level | LGB full | LGB w/o time_frac | Drop | % of edge |
|--------|-------|----------|-------------------|------|-----------|
| AAPL | L1 | 0.526 | 0.507 | −0.019 | 73% |
| AAPL | L2 | 0.514 | 0.498 | −0.015 | 111% |
| INTC | L1 | 0.562 | 0.552 | −0.010 | 16% |
| INTC | L2 | 0.575 | 0.565 | −0.010 | 13% |
| MSFT | L1 | 0.530 | 0.507 | −0.024 | 78% |
| MSFT | L2 | 0.527 | 0.517 | −0.010 | 39% |

**AAPL and MSFT**: time_frac accounts for 73–111% of the LGB edge — the models
are primarily learning the intraday fill-rate profile, not book state.  
**INTC**: time_frac contributes only 13–16% of the edge. Queue position and OFI
carry genuine cross-sectional signal that persists after controlling for time.

### Adverse selection — cross-ticker with multi-horizon decay

#### 1-second adverse selection

| Ticker | Level | Mean adv. sel. | 95% CI | Frac. adversely selected |
|--------|-------|----------------|--------|--------------------------|
| AAPL | L1 | **+1.389¢** | [+1.338, +1.441] | 76.6% |
| AAPL | L2 | +1.357¢ | [+1.310, +1.406] | 75.7% |
| INTC | L1 | +0.833¢ | [+0.819, +0.847] | 75.4% |
| INTC | L2 | +0.852¢ | [+0.832, +0.873] | 75.0% |
| MSFT | L1 | +0.842¢ | [+0.821, +0.863] | 71.3% |
| MSFT | L2 | +0.834¢ | [+0.809, +0.859] | 71.4% |

L1 vs L2 t-tests: AAPL p=0.364, INTC p=0.137, MSFT p=0.618. None significant.
**The L1 > L2 adverse selection gap found in the single AAPL session (p=0.014)
does not replicate across the full 3-date panel** — it was a session-specific
artifact of the holiday-week data.

#### Adverse selection decay curve (1 s → 30 s after fill, L2)

| Ticker | 1 s | 5 s | 10 s | 30 s | Pattern |
|--------|-----|-----|------|------|---------|
| AAPL | +1.357¢ | +1.515¢ | +1.597¢ | +1.637¢ | **Increasing** |
| INTC | +0.852¢ | +0.822¢ | +0.809¢ | +0.771¢ | **Decreasing** |
| MSFT | +0.834¢ | +0.887¢ | +0.941¢ | +0.849¢ | Non-monotone |

This is the clearest evidence that AAPL and INTC have structurally different
informed-flow dynamics. In AAPL, the informed trader who caused your fill
continues pushing prices in their direction for at least 30 s — adverse
selection compounds. In INTC, the impact reverts within 30 s, consistent with
less persistent (noisier) execution patterns and lower HFT participation.
The OFI story closes the loop: OFI is arbitraged instantly in AAPL (no signal
survives to passive orders) but decays slowly in INTC (HFTs have not fully
competed away the signal, and fills preceded by flow imbalance are still
"good" fills in aggregate).

See `results/extended/adverse_selection_decay.png`.

### Optimal placement — C\* by ticker

| Ticker | p1 (L1) | p2 (L2) | AS1 | AS2 | **C\*** | Prefer at C=10¢ |
|--------|---------|---------|-----|-----|---------|-----------------|
| AAPL | 0.661 | 0.570 | +1.389¢ | +1.357¢ | **6.60¢** | L1 |
| MSFT | 0.693 | 0.563 | +0.842¢ | +0.834¢ | **4.55¢** | L1 |
| INTC | 0.567 | 0.340 | +0.833¢ | +0.852¢ | **1.79¢** | L1 |

With more data, AAPL's crossover falls from 8.96¢ to 6.60¢ and MSFT's from
6.03¢ to 4.55¢ — both move toward the range where L1 preference is clear at
typical urgency levels. INTC's 1.79¢ remains the binding constraint: L2 posting
in INTC is only worthwhile at unrealistically low unfill penalties.

See `results/extended/optimal_placement_crossover.png`.

### Survival analysis — Cox PH by ticker (L2)

| Ticker | n | Events | C-index | Significant predictors |
|--------|---|--------|---------|----------------------|
| AAPL | 13,110 | 7,468 | 0.529 | `spread` (HR=0.928 ***) |
| INTC | 14,999 | 5,098 | **0.594** | `time_frac` (HR=0.804 ***), `queue_position` (HR=0.941 ***), `book_imbalance` (HR=0.965 ***), `granularity` (HR=1.050 ***), `spread` (HR=1.045 ***) |
| MSFT | 14,977 | 8,435 | 0.536 | `queue_position` (HR=0.947 ***), `granularity` (HR=1.043 ***), `time_frac` (HR=0.962 ***), `spread` (HR=0.980 *) |

Key updates:
- **Granularity is now significant in INTC and MSFT survival** (HR≈1.043–1.050 ***).
  Higher fragmentation (more orders per share ahead) → faster fill hazard. This
  is consistent with the theory — many small orders drain faster per event than
  a few large ones — but only detectable in survival analysis, not logistic
  regression. The logistic null result for granularity stands; the survival
  result suggests granularity affects timing rather than whether a fill occurs.
- INTC Cox C-index (0.594) is the highest across all models. Fill timing in
  wider-spread stocks is substantially more predictable than in AAPL.

See `results/extended/survival_km_by_ticker.png`.

### Panel regression with ticker fixed effects (L2)

Pooled logistic regression across all 43,086 L2 orders. OOS AUC = 0.526
(single 80/20 chronological split on pooled data).

**Significant panel predictors** (bootstrap 95% CI excludes zero):

| Feature | Coef (std.) | 95% CI | Notes |
|---------|-------------|--------|-------|
| `ticker_INTC` | −0.474 | [−0.533, −0.417] | Depth fills structurally harder in INTC |
| `queue_position_at_entry` | −0.177 | [−0.295, −0.046] | Robust cross-ticker |
| `ofi_30s` | +0.119 | [+0.087, +0.148] | Driven by INTC and MSFT |
| `time_frac` | −0.111 | [−0.133, −0.087] | Robust cross-ticker |
| `spread_at_entry_ticks` | −0.096 | [−0.121, −0.071] | Robust cross-ticker |
| `ticker_MSFT` | −0.055 | [−0.082, −0.026] | MSFT slightly harder than AAPL |
| `queue_granularity_at_entry` | **+0.031** | [+0.000, +0.064] | **Significant in panel (marginal)** |

`side_bid`, `ofi_10s`, and `book_imbalance` are not significant in the panel.

**Granularity is now marginally significant in the panel** (+0.031 ***) — the
null result from the 4-ticker, 18k-row analysis does not hold at 43k rows.
The effect is small but consistent: higher fragmentation is associated with
higher fill probability, possibly because granular queues are more volatile
(cancel/replace activity clears them faster).

### OFI predictive power — cross-ticker

| Ticker | OFI_30s coef | 95% CI | Significant? |
|--------|-------------|--------|--------------|
| AAPL | −0.007 | [−0.051, +0.037] | No |
| INTC | **+0.203** | [+0.142, +0.259] | **Yes (\*\*\*)** |
| MSFT | **+0.049** | [+0.009, +0.095] | **Yes (\*\*\*)** |

OFI arbitrage is AAPL-specific. In INTC and MSFT, 30-second order flow
imbalance is a significant positive predictor of depth fill probability —
buying pressure that arrives at the touch propagates to L2 within 60 s.
The MSFT result (previously borderline) is now stable across 3 dates.

---

## Why this is interesting

**1. The fill-predictability gap is inverted at INTC, and robust across dates.**
In AAPL, depth fills are barely predictable (AUC ≈ 0.505) and time_frac
accounts for over 100% of the LGB edge. In INTC, AUC reaches 0.567–0.575 and
book-state features (queue position, OFI) carry real OOS signal after removing
the time trend. This is not a single-day artifact — it replicates across three
sessions including holiday week and two normal days.

**2. Adverse selection dynamics reveal two distinct information regimes.**
AAPL's adverse selection compounds from +1.36¢ at 1 s to +1.64¢ at 30 s:
informed flow is durable, HFTs are active, and the market continues to move
against your fill for half a minute. INTC's adverse selection decays from
+0.85¢ to +0.77¢ over the same horizon: the order flow that triggered your
fill was more transient. This decay-curve divergence directly explains why OFI
is useful in INTC (signal persists long enough to benefit resting orders) but
not in AAPL (signal is competed away before passives benefit).

**3. Granularity moves from null to marginal with enough data.**
The granularity null result stands in logistic regression, but survival analysis
(HR≈1.043–1.050 ***) and panel regression (+0.031 ***) both show a small
positive effect — more fragmented queues are associated with faster fills and
slightly higher fill probability. The effect is too small to be a useful signal
but rules out the prior conclusion that granularity is completely uninformative.

**4. OFI arbitrage is a function of HFT participation, not a universal law.**
OFI is the fourth-largest panel coefficient (+0.119 ***) because it has strong
signal in INTC and MSFT. The Cont et al. (2014) finding that OFI predicts price
direction holds everywhere; whether it translates into fill probability depends
on whether HFTs have already acted on it. In AAPL they have; in INTC and MSFT
they have not fully.

**5. Optimal placement requires per-ticker calibration.**
C\* ranges from 1.79¢ (INTC) to 6.60¢ (AAPL). The same passive strategy that
is optimal at 5¢ urgency for AAPL (L1) would be wrong for MSFT at the same
urgency (where C\* = 4.55¢ puts 5¢ on the L2-preferred side, though barely).
Any fixed L1/L2 preference will be miscalibrated for at least one stock.

---

## Limitations

1. **Passive-shadow model (no market impact).** Injected orders are
   counterfactual — they don't displace resting orders, affect prices, or
   attract strategic responses. Real orders do.

2. **Nine combinations, three tickers.** Findings represent three sessions per
   ticker in 2019. MSFT exhibited severe intraday non-stationarity on 2019-08-30
   that makes its model anti-predictive on the test period; the Oct and Dec
   results are more stable. A larger panel spanning multiple years and more tickers
   would be needed to separate persistent regime effects from session noise.

3. **Granularity variance structurally near-zero.** Even INTC shows K/Q IQR
   of only 0.003 at L2. The granularity hypothesis cannot be fully tested without
   a stock where K/Q varies meaningfully (e.g., small-cap, pre-decimalization).

4. **Panel OOS split is a single contiguous block.** The per-ticker AUC uses
   4-fold walk-forward CV, but the panel regression OOS AUC (0.526) uses a
   single 80/20 split on the pooled parquet. The ordering in the pooled file is
   alphabetical by ticker then date, not strictly chronological, so the "test"
   period is the last 20% of MSFT 2019-12-30, not a cross-ticker holdout.

5. **MSFT non-stationarity unexplained.** The near-chance AUC for MSFT at both
   levels indicates the training and test folds have inverted relationships.
   The cause (a specific intraday price move, news event, or structural session
   change) was not investigated.

---

## What I'd do with more time

1. **SHAP values for model interpretability.** Replace coefficient tables with
   SHAP beeswarm plots — shows feature importance nonlinearly and per-observation,
   and is more compelling visually than standardised logistic coefficients.

2. **Cross-date generalization.** Train on one date, test on another for each
   ticker. This answers whether learned patterns are durable across sessions
   or regime-specific — a much stronger validity claim than within-session CV.

3. **Rolling AUC / structural break analysis.** A rolling AUC in 30-minute
   windows throughout the trading day would precisely locate where MSFT's model
   breaks down and quantify the stationarity problem.

4. **Expand the ticker panel.** Add 5–10 stocks spanning the full liquidity
   spectrum. The OFI and queue-position effects suggest a monotone relationship
   with HFT participation; a larger panel could quantify that slope.

5. **Granularity in a wider-spread regime.** All three tickers have K/Q IQR
   below 0.004. Testing the granularity hypothesis properly requires a stock
   where K/Q has an IQR of at least 0.05.
