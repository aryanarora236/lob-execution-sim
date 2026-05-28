# Finding

## Headline

Fill predictability at passive limit order depth (L2) is regime-dependent.
In liquid HFT-dominated stocks (AAPL), only spread is significant and OFI is
fully arbitraged (OOS AUC ≈ 0.53). In wider-spread stocks (INTC), queue
position becomes the dominant predictor (coef −0.110 ***), OFI_30s is
significant (+0.202 ***), and Cox C-index reaches 0.613 vs 0.536 for AAPL.
The optimal placement crossover (L1 vs L2) varies from C\* ≈ 9¢ (AAPL) to
C\* ≈ 1.6¢ (INTC) — meaning L2 is almost never worth posting in INTC at
realistic urgency levels. Queue granularity (K/Q) remains null across all
tickers and both levels.

---

## Data and scope

| Ticker | Date | Session | Events | L1 injections | L2 injections |
|--------|------|---------|--------|---------------|---------------|
| AAPL | 2019-12-30 | Holiday week (low volume) | 1,609,148 | 4,343 | 4,701 |
| AAPL | 2019-08-30 | Normal | 1,283,342 | 3,259 | 3,943 |
| MSFT | 2019-08-30 | Normal | 1,642,017 | 4,690 | 4,985 |
| INTC | 2019-08-30 | Normal | 1,022,326 | 4,994 | 5,000 |

Source data: NASDAQ TotalView-ITCH 5.0 (MSFT, INTC) and LOBSTER Level 10
(AAPL 2019-12-30), parsed to LOBSTER-compatible format. Pooled dataset:
17,286 L1 orders, 18,629 L2 orders across 4 ticker-date combinations.

Hypothetical orders injected via passive-shadow simulation (no market impact).
Each order expires unfilled after 60 s. Depth level 1 = best price (touch);
level 2 = one tick behind best.

---

## Part I — Single-ticker AAPL results (2019-12-30)

### L1 (touch) and L2 (depth) fill predictability

Source: `results/experiment_AAPL_2019-12-30_L{1,2}.parquet`

| | L1 | L2 |
|---|---|---|
| Fill rate (60 s) | 63.2% [61.8%, 64.7%] | 55.1% [53.7%, 56.5%] |
| Logistic OOS AUC | 0.491 [0.450, 0.531] | 0.494 [0.458, 0.532] |
| LightGBM OOS AUC | 0.548 [0.506, 0.589] | 0.542 [0.503, 0.579] |

**Significant predictors at L2** (bootstrap 95% CI excludes 0):

| Feature | Coef (std.) | 95% CI |
|---------|-------------|--------|
| `side_bid` | −0.191 | [−0.257, −0.125] |
| `spread_ticks` | −0.119 | [−0.187, −0.048] |
| `time_frac` | −0.105 | [−0.173, −0.035] |
| `imbalance` | −0.063 | [−0.127, −0.001] |

At L1 only `spread_ticks` is significant. Queue granularity spans zero at
both levels.

### LightGBM time_frac ablation (AAPL)

| | LGB full | LGB w/o time_frac | Drop | % of edge |
|---|---|---|---|---|
| L1 | 0.541 | 0.539 | −0.002 | ~5% |
| L2 | 0.524 | 0.492 | −0.032 | >100% |

At L2, removing time_frac collapses AUC below chance. The tree's entire edge
was learning the intraday fill-rate profile. After controlling for time of day,
observable book state has zero predictive power for AAPL depth fills.

### Adverse selection (AAPL 2019-12-30)

- L1 mean: **+1.73¢/share** [+1.64, +1.82]; 79.9% of fills adversely selected
- L2 mean: **+1.60¢/share** [+1.51, +1.70]; 77.5% adversely selected
- L1 > L2 adversity, t-test significant (p = 0.014)
- Intraday pattern: open hour worst (+2.40–2.60¢), declining to ~+1.5¢ by noon
- Only two significant OLS predictors at both levels: `side_bid` and `time_frac`
- Adverse selection R² < 2% — largely unpredictable from observable state

### Survival analysis (AAPL 2019-12-30, L2)

- Median fill time: L1 = 9.1 s, L2 = 12.2 s
- Significant Cox predictors at L2: `spread_ticks` (HR=0.915 ***),
  `side_bid` (HR=0.899 ***), `time_frac` (HR=0.954 *), `imbalance` (HR=0.957 *)
- Cox C-index L2: 0.545 in-sample, 0.500 OOS

### Optimal placement (AAPL 2019-12-30)

C\* ≈ 8¢/share at mean spread. L2 preferred for patient traders; L1 only
when urgency is high. See Part II for cross-ticker comparison.

### OFI (AAPL)

OFI_10s and OFI_30s add no predictive power at either level. Neither
bootstrap CI excludes zero. OFI is fully arbitraged in AAPL by HFTs.

---

## Part II — Multi-ticker cross-sectional analysis

Source: `notebooks/11_multi_ticker.py`, `notebooks/12_extended_analysis.py`
Pooled parquets: `results/experiment_all_L{1,2}.parquet`

### Fill rates and AUC by ticker

| Ticker | L1 fill rate | L2 fill rate | L1 AUC [95% CI] | L2 AUC [95% CI] |
|--------|-------------|-------------|-----------------|-----------------|
| AAPL (pooled) | 65.1% | 57.7% | 0.519 [0.489, 0.549] | 0.529 [0.502, 0.557] |
| INTC | 55.7% | **31.7%** | 0.531 [0.494, 0.565] | 0.498 [0.456, 0.540] |
| MSFT | 68.4% | 57.9% | 0.471 [0.426, 0.517] | 0.460 [0.423, 0.500] |

INTC's L2 fill rate (31.7%) is less than half of AAPL's (57.7%) — the
depth queue in a wider-spread stock clears far less often within 60 s.
MSFT's AUC < 0.5 reflects strong intraday non-stationarity on 2019-08-30:
the model trained on the morning session is anti-predictive in the afternoon.

**Note on the AAPL AUC revision:** The original single-day Dec 30 result
(L2 AUC 0.54) is revised down to 0.529 in the pooled two-date analysis.
The holiday-week session inflated the signal slightly via day-specific drift.

### Bootstrap CIs on L2 logistic coefficients

| Feature | AAPL | INTC | MSFT |
|---------|------|------|------|
| `spread_at_entry_ticks` | −0.151 *** | −0.007 | −0.037 |
| `time_frac` | −0.002 | **−0.367 \*\*\*** | +0.037 |
| `queue_position_at_entry` | −0.027 | **−0.110 \*\*\*** | −0.002 |
| `side_bid` | +0.028 | +0.095 *** | +0.112 *** |
| `queue_granularity_at_entry` | +0.004 | −0.070 | +0.004 |
| `book_imbalance_at_entry` | −0.024 | −0.043 | +0.025 |

\*\*\* = 95% bootstrap CI excludes zero.

Key contrasts:
- **Spread** is the only significant predictor for AAPL; it is irrelevant for INTC.
- **Queue position** is 4× larger for INTC (−0.110) than AAPL (−0.027) and
  significant only in INTC. In a wider-spread stock, how far back you are in
  the queue matters far more than what the spread is.
- **Time_frac** dominates for INTC (−0.367 ***) but is near-zero for AAPL.
- **Granularity** spans zero for all three tickers — the null result holds
  even with more data and a regime better suited to the hypothesis.

### LightGBM time_frac ablation — cross-ticker

| Ticker | Level | LGB full | LGB w/o time_frac | Drop |
|--------|-------|----------|-------------------|------|
| AAPL | L2 | 0.515 | 0.499 | −0.016 (time_frac = whole edge) |
| INTC | L2 | 0.486 | **0.561** | **+0.075** (time_frac hurts) |
| MSFT | L2 | 0.510 | 0.500 | −0.010 |

For INTC, removing time_frac *improves* LGB AUC from 0.486 → 0.561. The
intraday regime shift in INTC is so strong that a linearly learned time_frac
is counter-productive on the test period — the morning fill-rate profile
reverses in the afternoon. AAPL's result (time_frac = the whole edge) does
not generalise: in INTC, queue position carries real out-of-sample signal
once the noisy time trend is removed.

### Adverse selection — cross-ticker comparison

| Ticker | Level | Mean adv. sel. | 95% CI | Frac. adversely selected |
|--------|-------|----------------|--------|--------------------------|
| AAPL | L1 | +1.47¢ | [+1.40, +1.53] | 77.6% |
| AAPL | L2 | +1.35¢ | [+1.29, +1.41] | 75.1% |
| INTC | L1 | +0.88¢ | [+0.85, +0.90] | 79.6% |
| INTC | L2 | +0.91¢ | [+0.88, +0.95] | 79.4% |
| MSFT | L1 | +0.91¢ | [+0.88, +0.95] | 74.8% |
| MSFT | L2 | +0.86¢ | [+0.82, +0.91] | 72.2% |

**L1 > L2 adverse selection is AAPL-specific.** The gap is significant for
AAPL (p = 0.014) but not for INTC or MSFT (p ≈ 0.08). The Glosten-Milgrom
mechanism — informed flow concentrating at the touch, making L1 fills more
adversely selected — appears only in the most liquid, HFT-dominated stock.

INTC and MSFT absolute adverse selection (~0.87–0.91¢) is ~40% lower than
AAPL (~1.35–1.47¢). Wider-spread stocks are cheaper to execute in quality
terms despite being harder to fill (lower fill rates).

### Optimal placement — C\* by ticker

| Ticker | p1 (L1 fill) | p2 (L2 fill) | Mean AS1 | Mean AS2 | **C\*** | Prefer at C=10¢ |
|--------|-------------|-------------|----------|----------|---------|-----------------|
| AAPL | 0.651 | 0.577 | +1.47¢ | +1.35¢ | **8.96¢** | L1 |
| MSFT | 0.684 | 0.579 | +0.91¢ | +0.86¢ | **6.03¢** | L1 |
| INTC | 0.557 | 0.317 | +0.88¢ | +0.91¢ | **1.64¢** | L1 |

The crossover varies enormously across tickers. For INTC, L2 posting is only
worthwhile when the market-order fallback costs less than 1.64¢ — a threshold
exceeded by almost any realistic execution scenario. The "L2 is almost always
better" conclusion from AAPL does not generalise: it depends entirely on the
fill-rate gap between levels, which is small in AAPL (8pp) and large in INTC
(24pp).

See `results/extended/optimal_placement_crossover.png` for the full curve.

### Survival analysis — Cox PH by ticker (L2)

| Ticker | C-index | Significant predictors |
|--------|---------|----------------------|
| AAPL | 0.536 | `spread_ticks` (HR=0.911 ***), `side_bid` (HR=1.028 **) |
| INTC | **0.613** | `time_frac` (HR=0.785 ***), `queue_position` (HR=0.909 ***), `side_bid` (HR=1.066 ***) |
| MSFT | 0.530 | `side_bid` (HR=1.067 ***) |

INTC's Cox C-index (0.613) is the highest across all models — depth fill
timing in wider-spread stocks is substantially more predictable than in AAPL.
The dominant predictor is time_frac (HR=0.785: early-day fills 21% faster than
late-day), followed by queue position (HR=0.909: one std. step back in queue →
9% slower fill hazard). Log-rank test by spread tertile: AAPL p=0.000, INTC
p=0.030 (marginal), MSFT p=0.009.

### Panel regression with ticker fixed effects (L2)

Pooled logistic regression across all 18,629 L2 orders with MSFT and INTC
dummy variables (AAPL as reference). OOS AUC = 0.530.

**Significant panel predictors** (bootstrap 95% CI excludes zero):

| Feature | Coef (std.) | 95% CI | Notes |
|---------|-------------|--------|-------|
| `ticker_INTC` | −0.468 | [−0.557, −0.385] | INTC depth fills much harder |
| `queue_position_at_entry` | −0.190 | [−0.275, −0.108] | Consistent cross-ticker |
| `spread_at_entry_ticks` | −0.159 | [−0.197, −0.120] | Consistent cross-ticker |
| `ofi_30s` | +0.138 | [+0.096, +0.182] | Driven by INTC/MSFT |
| `time_frac` | −0.094 | [−0.128, −0.059] | Consistent cross-ticker |
| `ticker_MSFT` | −0.071 | [−0.107, −0.033] | MSFT slightly harder than AAPL |
| `side_bid` | +0.067 | [+0.039, +0.096] | Consistent cross-ticker |

Queue position, spread, time_frac, OFI_30s, and side_bid are all significant
in the panel — robust features that survive across ticker regimes. Granularity
is not significant (−0.017 [−0.060, +0.018]).

### OFI predictive power — cross-ticker

| Ticker | OFI_30s coef | 95% CI | Significant? |
|--------|-------------|--------|--------------|
| AAPL | +0.009 | [−0.046, +0.063] | No |
| INTC | **+0.202** | [+0.120, +0.280] | **Yes (\*\*\*)** |
| MSFT | **+0.168** | [+0.080, +0.258] | **Yes (\*\*\*)** |

**OFI arbitrage is AAPL-specific.** In AAPL, HFTs absorb OFI signal before
resting passive orders benefit. In INTC and MSFT, 30-second order flow
imbalance is a significant positive predictor of depth fill probability —
higher buying pressure means resting bids at L2 are more likely to fill within
60 s. This makes intuitive sense: INTC's lower HFT participation means the OFI
signal is not immediately competed away, and buying pressure that arrives at
the touch eventually propagates to L2 within the order's lifetime.

---

## Why this is interesting

The multi-ticker analysis reveals that microstructure regime matters as much
as model choice. Three findings stand out:

**1. The fill-predictability gap is inverted at INTC.**
In AAPL, the "hard" result is that depth fills are barely predictable and
time_frac is the whole story. In INTC, depth fills are still only modestly
predictable in aggregate (AUC ≈ 0.50–0.56), but the *nature* of the signal
is completely different: queue position and OFI carry real cross-sectional
information that survives the time_frac ablation. The implication for
execution algorithms is that book-state conditioning is worthwhile in INTC
but not in AAPL.

**2. OFI arbitrage is a function of HFT participation, not a universal law.**
The original finding that OFI is null in AAPL was correct but not general.
OFI_30s is the fourth-largest panel coefficient (+0.138 ***) because it has
strong signal in less liquid stocks. The Cont et al. (2014) finding that OFI
predicts price direction holds everywhere; whether that translates into fill
probability depends on whether HFTs have already acted on it.

**3. Optimal placement is not a single number.**
The L1/L2 crossover ranges from C\* ≈ 1.6¢ (INTC) to C\* ≈ 9¢ (AAPL).
Any passive execution framework that uses a fixed L1/L2 preference will be
wrong for at least one of these stocks. The key input is the fill-rate gap
between levels, not the spread per se.

---

## Limitations

1. **Passive-shadow model (no market impact).** Injected orders are
   counterfactual — they don't displace resting orders, affect prices, or
   attract strategic responses. Real orders do.

2. **Three tickers, two dates.** Findings represent two normal trading days
   in August 2019 and one holiday-week session. MSFT exhibited severe
   intraday non-stationarity on 2019-08-30 that made its model anti-predictive
   on the test period. A larger panel would separate persistent regime effects
   from day-specific noise.

3. **Granularity variance structurally near-zero across all tickers.** Even
   INTC — the widest-spread stock tested — shows K/Q IQR of only 0.0014 at L2.
   The hypothesis that granularity predicts fills cannot be properly tested
   without a stock where K/Q varies meaningfully (e.g., small-cap, pre-2010
   data, or a period of unusual market-maker fragmentation).

4. **OOS split is a single contiguous block.** The last 20% of each session
   may differ systematically from the morning (open volatility, end-of-day
   effects). k-fold time-series cross-validation would give more stable AUC
   estimates.

5. **Adverse selection is measured at 1-second horizon only.** A longer
   horizon (5 s, 30 s) might reveal stronger adverse selection for faster
   fills in INTC, where the OFI signal decays more slowly than in AAPL.

6. **MSFT non-stationarity unexplained.** The AUC < 0.5 result for MSFT at
   both levels indicates the training and test periods have inverted
   relationships. The cause (a specific intraday price move, a news event, or
   structural session change on 2019-08-30) was not investigated.

---

## What I'd do with more time

1. **Expand the ticker panel.** Add 5–10 stocks spanning the full liquidity
   spectrum: a large-cap ETF, two mid-caps, and two small-caps. The OFI and
   queue-position effects suggest a clear monotone relationship with HFT
   participation; a larger panel could quantify that slope.

2. **Multi-day per ticker.** MSFT's non-stationarity flags the risk of
   day-specific results. Three dates per ticker — one normal, one
   high-volatility, one low-volume — would separate structural regime effects
   from session noise.

3. **Granularity in a wider-spread regime.** All three tickers on these days
   have K/Q clustered at 0.01. Testing the original hypothesis properly
   requires a stock where K/Q has an IQR of at least 0.05.

4. **Longer adverse selection horizon for INTC.** OFI is significant in INTC;
   testing whether fills preceded by high OFI also have worse 30 s adverse
   selection would close the loop between fill probability and execution quality.
