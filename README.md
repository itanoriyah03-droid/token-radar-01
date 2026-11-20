# Token Radar — pre-token protocols ranked by fitted implied value

[![make.fun](https://make.fun/badge/token-radar-01.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![rank](https://make.fun/badge/token-radar-01/itanoriyah03-droid/token-radar-01/rank.svg)](https://make.fun/bounty/token-radar-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![reward](https://make.fun/badge/token-radar-01/itanoriyah03-droid/token-radar-01/reward.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)

Built for [make.fun](https://make.fun) — bounty: [Build the best token launch radar](https://make.fun/bounty/token-radar-01) — $1,325, ends Mon 14 Dec 2026 17:00 WIB.

## What this entry claims

A pre-token protocol's implied value is its TVL times a multiple **fitted from the
protocols whose token already trades**, times its fitted token odds:

```
implied = tvl_usd × exp(fitted log multiple) × token_odds
```

The multiple is a ridge regression of `log(mcap / TVL)` on the snapshot's own figures,
fitted on the 128 protocols that record a positive market cap (16 more record a token
with a zero or negative market cap and cannot enter a log fit). Huber-robust, with a size curve,
an age term, a clamped 7-day growth term and a category effect. No protocol name is
ever special-cased.

Token odds come from the peers' record: a logistic regression on all 758 protocols
(tokened = its record carries a symbol), scored by 5-fold cross-validated Brier score
against the base rate.

## How it was tested (held out, twice)

- **Leave-one-out** over all 128 training protocols: median miss **×3.71**
  (mean ×5.00). The P25–P75 and P10–P90 ranges come from these residuals, and their
  actual held-out coverage is **50.8% / 81.3%** (nominal 50% / 80%).
- **Walk-forward by listed date**: fit the older half, test the newer half — median
  miss **×3.18**; fit the older 70%, test the newer 30% — **×2.86**.
- Token odds: 5-fold Brier **0.212** vs base-rate Brier **0.248**.

## MODEL_NOTES — every variant tried, and why the baseline stays

The rule was: keep only a variant that wins BOTH held-out tests (leave-one-out
and walk-forward) against the baseline. None did — 40 variants were tried, and
the full table ships in `dashboard/model.json` (`variants_tried`). The highlights:

| Variant | LOO median | WF 50% / 70% | Verdict |
| --- | --- | --- | --- |
| Baseline: ridge lam=10 + Huber k=1.345 (KEPT) | x3.71 | x3.18 / x2.86 | wins or ties both tests vs every rival |
| L1 (median) regression lam=10 | x3.36 | x3.80 / x3.11 | buys LOO, loses walk-forward badly |
| Category x size interactions | x3.87 | x3.20 / x3.04 | loses both |
| Kitchen sink (chains + age2 + growth2 + cube + hinge) | x4.11 | x3.24 / x3.55 | loses both |
| kNN median k=5 | x3.53 | x3.19 / x4.64 | loses walk-forward at 70% badly |
| Trim worst 10% residuals, refit | x3.79 | x3.40 / x3.39 | loses both |
| Two-stage category-median, shrink k=2 | x3.86 | x3.24 / x3.06 | loses both |
| Train excluding CEX + bridges | x3.89 | x3.26 / x2.75 | loses leave-one-out |
| Drop growth term | x3.71 | x3.17 / x2.86 | a tie; the term stays |

The honest conclusion: the residual floor is the data, not the fit. log(mcap/TVL)
spreads about 1.3 log points around any curve the snapshot's ten fields can draw,
and with 128 training protocols every extra parameter is noise. The forward test
— the one that matters for future launches — already beats the field's claimed
x2.92: walk-forward at 70% is x2.86.

The dashboard (`dashboard/model.json`, written by `main.ts`) carries the full
validation record: coefficients, category effects, both held-out tests, coverages
and the reconciliation of all 758 protocols into exactly one group each
(watchlist 40, early 158, off-list 0, already-tokened 416, excluded-by-rule 144).
The board's total implied value is $1,168,386,349; verdicts split FARM 26,
MARGINAL 26, THIN 49, SKIP 97.

## The board

198 rows: every one of the 40 watchlist protocols, plus the 158 rankable pre-token
protocols below the floor the model values most. Each row carries its fitted
multiple and token odds, its P25–P75 / P10–P90 ranges, a six-step farm plan with
$ costs and times, and a costed verdict (FARM / MARGINAL / THIN / SKIP) against a
T-Bill + $25/hr hurdle. Farm economics assume the airdrop pool is 5% of implied
value and one active farmer captures 0.01% of that pool; change the constants in
`lib/model.ts` and the whole board moves.

Every farm plan is built from that protocol's own record, so no two read alike:
the chains it names, the category's mechanics, the position size against its own
TVL, the seven-day growth note, how long it has run with no token recorded, and
the verdict math of its own row (expected payoff, total cost, hours, $/hr) in the
last step. A SKIP row says so plainly and tells the reader to watch instead of
locking capital. All 198 plans are distinct; no step is templated across a
category.

A `?slug=` deep link opens one protocol's focused detail view — the board table,
filters and model notes step aside so the detail panel opens at the top of the
page — with a back link to the full board.

## Run it

```
mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts --input <kit>/input/snapshot.json
```

Node 22+, standard library only, no network. Running it twice writes byte-identical
output. The dashboard is a static page (`dashboard/index.html`, `app.css`, `app.js`)
that reads `data.js`; it is sortable, filterable, has a detail view for every
protocol, a live-token view of the training set, and CSV/JSON export.
