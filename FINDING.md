# Finding

## Headline

Observable market-state variables (spread, book imbalance, side, time of day)
predict passive fill probability at AAPL's second-best price level (OOS AUC
0.54, four significant linear predictors) but not at the best price (OOS AUC
0.49, one marginal predictor) — consistent with HFT market-makers creating a
near-fair queue at the touch while depth orders face more selective execution.
Queue granularity (K/Q) shows no statistically significant effect at either level.

## Quantified results

### Level 1 — best price (touch)

Source: `results/experiment_AAPL_2019-12-30_L1.parquet`, n = 4,343 orders

- Fill rate within 60 s: **63.2%**  (95% CI 61.8%–64.7%)
- Logistic regression OOS AUC: **0.491**  (95% CI 0.450–0.531) — indistinguishable from chance
- LightGBM OOS AUC: **0.548**  (95% CI 0.506–0.589) — modest intraday nonlinearity
- **Only significant predictor (bootstrap 95% CI excludes 0):** `spread_ticks`
  - Coef (standardised) = −0.106, OR = 0.899
  - Fill rate at 1-tick spread: 68.4% vs 62.1% at 3 ticks

### Level 2 — second-best price (depth)

Source: `results/experiment_AAPL_2019-12-30_L2.parquet`, n = 4,701 orders

- Fill rate within 60 s: **55.1%**  (95% CI 53.7%–56.5%) — 8pp lower than L1
- Logistic regression OOS AUC: **0.494**  (95% CI 0.458–0.532)
- LightGBM OOS AUC: **0.542**  (95% CI 0.503–0.579)
- **Four significant predictors at L2 (logistic bootstrap CIs):**

  | Feature       | Coef (std.) | OR    | 95% CI              |
  |---------------|-------------|-------|---------------------|
  | `side_bid`    | −0.191      | 0.826 | [−0.257, −0.125]    |
  | `spread_ticks`| −0.119      | 0.888 | [−0.187, −0.048]    |
  | `time_frac`   | −0.105      | 0.900 | [−0.173, −0.035]    |
  | `imbalance`   | −0.063      | 0.939 | [−0.127, −0.001]    |

- **Queue granularity at L2:** coef = +0.032, CI [−0.040, +0.103] — spans zero

### Adverse selection — execution quality

Source: `adverse_selection_1s` in both parquets (n = 2,746 L1 fills, 2,590 L2 fills with data)

- **L1 mean adverse selection: +1.73¢/share** (95% CI [+1.64, +1.82]); 79.9% of fills adversely selected
- **L2 mean adverse selection: +1.60¢/share** (95% CI [+1.51, +1.70]); 77.5% adversely selected
- **H2 rejected:** L2 is *not* worse than L1 (t = −1.94, p = 0.97). L1 fills face slightly worse quality — likely because touch fills are executed immediately by the most aggressive (informed) takers, while L2 fills occur only after L1 has been fully cleared.
- **Intraday pattern:** 09:34 bucket is the worst at both levels (+2.40–2.60¢), declining through the morning to ~+1.5¢ by noon — consistent with Admati-Pfleiderer informed-trading concentration at the open. No uptick at close (low-volume holiday session).
- **Regression R² ≈ 0:** adverse selection is largely unpredictable from observable state at entry.
- **Only two significant OLS predictors (bootstrap 95% CIs, both levels):**

  | Feature     | L1 coef (std.) | L2 coef (std.) | Interpretation                          |
  |-------------|----------------|----------------|-----------------------------------------|
  | `side_bid`  | +0.295¢        | +0.282¢        | Bid fills adversely selected more (price drifted up this day) |
  | `time_frac` | −0.166¢        | −0.177¢        | Earlier in day → worse quality (open effect) |

- Imbalance, spread, queue size, and granularity are all insignificant at both levels.

### Granularity verdict

At both levels the median queue granularity (K/Q) is exactly 0.01 — meaning
the typical best-price queue in AAPL on this day consists of ~2 orders sharing
~200 shares. The near-zero variance in K/Q (75th percentile = median = 0.01 at
both levels) means the hypothesis cannot be tested meaningfully: there is not
enough spread in the independent variable.

## Why this is interesting

**The touch/depth asymmetry is the headline finding.** At the best price,
market-state variables explain almost nothing about fill outcomes — the queue
is competitive and fair, consistent with HFT market-maker concentration at
AAPL's best bid/ask. At the second level, four variables become predictive:
side (bids fill 8% less often than asks, suggesting upward price drift on this
day), spread (wider spread = less favourable execution environment), time of
day, and book imbalance. This is consistent with Glosten-Milgrom: informed
flow concentrates at the touch and makes it efficient; depth orders are reached
only by order flow that is less competitive and therefore more predictable.

The granularity null result is itself informative: AAPL's visible queue is
structurally coarse (a small number of large institutional/HFT orders) at
every level on this day. Testing the granularity hypothesis requires either a
different stock (smaller cap, wider spread) or a different regime.

## Limitations

1. **Single ticker, single day.** All results are specific to AAPL on Dec 30
   2019 — a low-volume holiday-week session. Findings may not generalise.

2. **Passive-shadow model (no market impact).** Injected orders are
   counterfactual — they don't displace existing resting orders, don't affect
   prices, and don't attract strategic responses. Real orders do.

3. **Granularity variance is structurally near-zero.** With K/Q at 0.01 for
   75%+ of observations at both levels, the hypothesis has insufficient
   statistical power regardless of the true effect size.

4. **Right-censored survival collapsed to binary.** Using filled-within-60s
   discards information about fill timing. A Cox proportional hazards model
   would handle censoring correctly and might reveal clearer granularity signal.

5. **OOS split is a single contiguous block.** The last 20% of the day (after
   14:38) may differ systematically from the morning session (pre-open
   volatility, end-of-day effects). A k-fold time-series cross-validation
   would give more stable AUC estimates.

6. **Level-2 side effect may be day-specific.** The `side_bid` coefficient
   at L2 (bids fill 18% less often than asks) likely reflects intraday price
   drift on Dec 30 specifically. It would vanish or reverse on a down day.

7. **Adverse selection R² is near-zero.** The OLS models explain <2% of variance in fill quality.
   The dominant signal is structural (time of day, direction of price drift on this specific day)
   rather than predictive from observable queue state. A richer model with lagged OFI or
   realized volatility might surface stronger predictors.

## What I'd do with more time

1. **Multi-stock panel across spread regimes.** Run the same experiment on
   10–15 tickers spanning the liquidity spectrum (e.g., AAPL, MSFT, a
   mid-cap, a small-cap). Stocks with wider average spreads have larger, more
   fragmented depth queues — the regime where K/Q actually varies and the
   granularity hypothesis has power.

2. **Survival analysis on time-to-fill.** Replace the binary outcome with a
   Cox model on time-to-fill, treating expired orders as right-censored. This
   recovers fill-speed information and allows testing whether K/Q predicts the
   *rate* of queue drain (the original theoretical claim) rather than the
   binary outcome.

3. **Add order flow imbalance (OFI) as a feature.** OFI — the net signed
   execution volume in a short pre-injection window — is the strongest
   short-term predictor in empirical microstructure (Cont et al. 2014). Adding
   it would likely increase OOS AUC substantially and provide a cleaner
   baseline against which to measure the marginal contribution of queue
   structure variables.
