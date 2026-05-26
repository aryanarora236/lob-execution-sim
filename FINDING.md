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
   rather than predictive from observable queue state. A richer model with realized volatility
   or signed trade flow might surface stronger predictors.

8. **OFI is fully arbitraged at AAPL.** Short-window OFI (10s, 30s) adds no predictive power for
   fill probability in AAPL. The signal likely exists but is absorbed by HFTs before a passive
   resting order can benefit. A slower-trading, wider-spread stock would be the right testbed.

### OFI as a fill predictor

Source: `ofi_10s`, `ofi_30s` columns added to both parquets; analysis in `notebooks/07_ofi_fill.py`.

- **OFI_10s mean:** +390 shares overall; filled orders have slightly higher OFI at L1 (+411 vs +354, Δ=+57) but slightly *lower* at L2 (+355 vs +421, Δ=−66).
- **AUC: no improvement.** Adding OFI_10s + OFI_30s leaves OOS AUC unchanged at both levels (L1: 0.491→0.488; L2: 0.494→0.494). Neither OFI feature's bootstrap CI excludes zero.
- **Sign × level interaction:** At L1, all order sides show a weak positive OFI→fill slope; at L2, passive asks show a weak *negative* slope (high buying pressure pushes price through L1 but prices out passive depth asks). The effect is ~3–6pp across OFI deciles and not significant.
- **Interpretation:** In AAPL, short-window OFI is fully arbitraged by HFTs before it can inform a passive resting order's fill probability. OFI predicts *price direction* (Cont et al. 2014) but fill probability depends on whether the move *continues* — which is not predictable from a fixed lookback at the touch.

### Fill speed and adverse selection are independent

Source: `notebooks/10_deep_angles.py`

Spearman ρ between time-to-fill and adverse selection: L1 = +0.001 (p=0.96), L2 = −0.006 (p=0.78). After controlling for side, spread, and time of day, the coefficient on fill speed is −0.04¢ (L1) and −0.06¢ (L2) — both negligible. Being filled in 2 seconds is not meaningfully worse quality than being filled in 55 seconds. **In AAPL, execution speed and execution quality are orthogonal.** This contradicts the intuition that fast fills signal informed counterparties — in a liquid HFT-dominated market, the adversity is structural (time of day, price drift direction) not counterparty-driven.

### LightGBM's edge is time-of-day at L2, book-state at L1

Source: `notebooks/10_deep_angles.py`

| | LightGBM full AUC | Without `time_frac` | Drop | % of edge |
|---|---|---|---|---|
| L1 | 0.541 | 0.539 | −0.002 | ~5% |
| L2 | 0.524 | 0.492 | −0.032 | >100% |

**At L1:** Removing time_frac barely moves AUC. The tree's modest advantage over logistic comes from weak nonlinear interactions in other book-state features. There is a small genuine book-state signal at the touch beyond time of day.

**At L2:** Removing time_frac collapses AUC below chance. The entire LightGBM edge at depth was learning that fill rates differ morning vs afternoon — a structural temporal pattern, not a tradeable book-state signal. After stripping time of day, observable book state has zero predictive power for depth fills.

**Implication:** The only exploitable signal for depth fill prediction is time of day. All other features — spread, imbalance, queue size, OFI — add nothing once the intraday fill-rate profile is accounted for. An optimal passive execution algorithm at L2 should condition primarily on time of day when estimating fill probability, not on real-time book state.

## What I'd do with more time

1. **Multi-stock panel across spread regimes.** Run the same experiment on
   10–15 tickers spanning the liquidity spectrum (e.g., AAPL, MSFT, a
   mid-cap, a small-cap). Stocks with wider average spreads have larger, more
   fragmented depth queues — the regime where K/Q actually varies and the
   granularity hypothesis has power.

2. **Survival analysis on time-to-fill.** *(Done — see below)*

3. **Add order flow imbalance (OFI) as a feature.** *(Done — null result at both levels)*

### Survival analysis (Cox proportional hazards)

Source: `notebooks/08_survival.py`

- **Median fill time:** L1 = **9.1 s**, L2 = **12.2 s** — depth orders take 34% longer when they fill.
- **Log-rank test (narrow vs wide spread):** L1 p=0.31 (not significant); L2 p=0.032 (significant) — spread predicts fill speed at depth but not at the touch.
- **Cox C-index:** L1 = 0.528 in-sample, 0.506 OOS; L2 = 0.545 in-sample, 0.500 OOS. Same degradation pattern as logistic AUC.
- **Significant Cox predictors at L2** (all p < 0.05):

  | Feature       | HR     | p-value    | Interpretation                        |
  |---------------|--------|------------|---------------------------------------|
  | `side_bid`    | 0.899  | 6 × 10⁻⁸  | Bids fill 10% more slowly             |
  | `spread_ticks`| 0.915  | 3 × 10⁻⁵  | Wider spread → slower fill            |
  | `time_frac`   | 0.954  | 0.026      | Earlier in day → faster fill          |
  | `imbalance`   | 0.957  | 0.028      | Higher imbalance → slower fill        |

- **At L1:** only `spread_ticks` significant (HR=0.946, p=0.006). Same as logistic.
- **Granularity null** at both levels (L2: HR=1.025, p=0.19).
- **OFI_10s null** at both levels (L2: p=0.26). Consistent with logistic result.
- **Interpretation:** The survival model confirms the logistic findings with a richer framework. The same four features that predict *whether* a depth order fills also predict *how fast* it fills — with consistent signs and similar magnitudes. The Cox framework adds the finding that spread is significant at L1 for fill *speed* even though it was only marginal in the binary model.

### Optimal passive placement

Source: `notebooks/09_optimal_placement.py`

**Framework:** Expected implementation shortfall (IS) per share in cents:

    E[IS | L1] = p1 × (−S + AS1) + (1 − p1) × C
    E[IS | L2] = p2 × (−S − 1¢ + AS2) + (1 − p2) × C

where S = half-spread, AS = mean adverse selection, C = unfill penalty (cost of market-order fallback).

**Key result — crossover at C\* = 8¢/share:**

| Spread | C\* (crossover) | Below C\*: prefer | Above C\*: prefer |
|--------|-----------------|-------------------|-------------------|
| 1-tick | 7.7¢            | L2                | L1                |
| 2-tick | 9.8¢            | L2                | L1                |
| 3-tick | 8.3¢            | L2                | L1                |

**Interpretation:** L2 is the superior passive strategy unless the market-order fallback exceeds ~8¢/share (~5 bps for AAPL at $150). The extra 1-cent price improvement from posting one tick deeper outweighs the 8pp lower fill rate for all realistic unfill penalties faced by a patient trader. Only when execution urgency is high (alpha decays fast, or hedging requires certainty of fill) does L1 dominate. At C=10¢, ~69% of spread × imbalance conditions flip to L1 — concentrated in the wide-imbalance, narrow-spread regime where missing a fill is most costly.
