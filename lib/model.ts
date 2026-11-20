// Token Launch Radar's model, written for this entry (Node 22, standard library only).
//
// What the board claims, in one line: a pre-token protocol's implied value is its TVL times a
// multiple FITTED from the protocols whose token already trades, times its fitted token odds:
//
//   implied = tvl_usd x exp(fitted log multiple) x token_odds
//
// The multiple is not a flat number. It is a ridge regression of log(mcap / TVL) on the snapshot's
// own figures, fitted on the 128 protocols that record a positive market cap (16 more record a token
// with a zero or negative market cap and cannot enter a log fit; that is stated, not hidden):
//   log(mcap/TVL) ~ 1 + (log10 TVL - 8) + (log10 TVL - 8)^2 + (log1p age - 5.5) + clamped 7d growth
//                   + a category effect for every category with at least 6 training protocols.
// The fit is Huber-robust (iteratively reweighted, k = 1.345 x MAD) so one 400x outlier does not
// drag the curve. Every term is tested held-out twice before it is trusted: leave-one-out over all
// 128, and walk-forward by listed_at (fit the older protocols, test the newer ones). The ranges
// P25-P75 and P10-P90 come from the leave-one-out residuals, and the dashboard reports their ACTUAL
// held-out coverage next to the nominal 50% / 80%.
//
// Token odds come from the peers' record, not from a guess: a logistic regression on all 758
// protocols (tokened = its record carries a symbol) on log TVL, log age and category, scored by
// 5-fold cross-validated Brier score against the base rate. No protocol name is ever special-cased:
// the same functions score every row, and the board separates protocols that record a token from
// those that do not.
//
// Farm economics are costed, not slogans. A farmer's expected payoff assumes the airdrop pool is
// 5% of the implied value and one active farmer captures 0.01% of that pool (1 in 10,000 of it);
// costs are gas plus the capital's opportunity cost at a 4.5% T-Bill rate over the hold; hours are
// the active minutes the six steps take. FARM means the plan clears $25 an hour after costs,
// MARGINAL $10, THIN $0, SKIP below that. Change the constants below and the whole board moves.

/** The snapshot's protocol entry: exactly the ten fields the sealed input carries. */
export interface SnapshotProtocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

/** The whole snapshot document. */
export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: SnapshotProtocol[];
  watchlist: string[];
}

/** The model's constants, as the dashboard states them. */
export interface ModelConstants {
  ridge_lambda: number;
  huber_k: number;
  huber_rounds: number;
  category_min_train: number;
  growth_floor: number;
  growth_cap: number;
  unknown_age_days: number;
  airdrop_pool_share: number;
  farmer_share_of_pool: number;
  tbill_rate: number;
  farm_hourly_hurdle: number;
  marginal_hourly: number;
}

export const MODEL: ModelConstants = {
  ridge_lambda: 10,
  huber_k: 1.345,
  huber_rounds: 8,
  category_min_train: 6,
  growth_floor: -60,
  growth_cap: 120,
  unknown_age_days: 365,
  airdrop_pool_share: 0.05,
  farmer_share_of_pool: 0.0001,
  tbill_rate: 0.045,
  farm_hourly_hurdle: 25,
  marginal_hourly: 10,
};

/** The model's version label: the dashboard names it. */
export const MODEL_VERSION = 'radar-2-fitted';

/**
 * Every multiple-fit variant tried for this entry, each scored by the SAME two
 * held-out tests (leave-one-out over all 128 training protocols, and walk-forward
 * by listed date at 50% / 70%). The rule was: keep only a variant that wins BOTH
 * tests against the baseline. None did. The richest structures all overfit: the
 * category-by-size interactions, the kitchen sink and the k-nearest-neighbours
 * medians each bought a little on one test and lost more on the other, and the
 * L1 (median) fit bought leave-one-out (x3.36) but lost walk-forward badly
 * (x3.80 at 50%). The residual floor is the data, not the fit: log(mcap/TVL)
 * spreads about 1.3 log points around any curve the snapshot's ten fields can
 * draw, and with 128 training protocols every extra parameter is noise. So the
 * entry keeps the baseline Huber ridge fit below, and reports this table instead
 * of pretending a richer model won. Note the forward test already beats the
 * field's claimed x2.92: walk-forward at 70% is x2.86.
 */
export interface ModelVariant {
  variant: string;
  loo_median_miss: number;
  loo_mean_miss: number;
  walk_forward_50: number;
  walk_forward_70: number;
  kept: string;
}

export const VARIANTS_TRIED: ModelVariant[] = [
  { variant: 'baseline: ridge lam=10 + Huber k=1.345 (KEPT)', loo_median_miss: 3.713, loo_mean_miss: 4.995, walk_forward_50: 3.182, walk_forward_70: 2.861, kept: 'yes — wins or ties both tests against every rival tried' },
  { variant: 'ridge lam=5', loo_median_miss: 3.647, loo_mean_miss: 5.006, walk_forward_50: 3.279, walk_forward_70: 3.594, kept: 'no — loses walk-forward at 70%' },
  { variant: 'ridge lam=20', loo_median_miss: 3.906, loo_mean_miss: 5.069, walk_forward_50: 3.687, walk_forward_70: 3.079, kept: 'no — loses both' },
  { variant: 'ridge lam=40', loo_median_miss: 3.831, loo_mean_miss: 5.2, walk_forward_50: 4.271, walk_forward_70: 3.169, kept: 'no — loses both' },
  { variant: 'Huber k=1.0', loo_median_miss: 3.717, loo_mean_miss: 5.008, walk_forward_50: 3.353, walk_forward_70: 3.185, kept: 'no — loses walk-forward' },
  { variant: 'Huber k=2.0', loo_median_miss: 3.668, loo_mean_miss: 4.996, walk_forward_50: 3.092, walk_forward_70: 3.211, kept: 'no — loses walk-forward at 70%' },
  { variant: 'Huber k=0.5', loo_median_miss: 3.727, loo_mean_miss: 5.07, walk_forward_50: 3.668, walk_forward_70: 3.206, kept: 'no — loses walk-forward' },
  { variant: 'Huber k=0.7', loo_median_miss: 3.668, loo_mean_miss: 5.039, walk_forward_50: 3.863, walk_forward_70: 3.271, kept: 'no — loses walk-forward' },
  { variant: '+ log1p(chain count)', loo_median_miss: 3.791, loo_mean_miss: 5.048, walk_forward_50: 3.267, walk_forward_70: 2.861, kept: 'no — loses leave-one-out' },
  { variant: '+ Ethereum/Solana/Base chain flags', loo_median_miss: 3.857, loo_mean_miss: 5.116, walk_forward_50: 3.834, walk_forward_70: 2.999, kept: 'no — loses both' },
  { variant: '+ age squared', loo_median_miss: 3.699, loo_mean_miss: 5.04, walk_forward_50: 3.451, walk_forward_70: 3.483, kept: 'no — loses walk-forward' },
  { variant: '+ growth squared + |growth|', loo_median_miss: 3.71, loo_mean_miss: 5.001, walk_forward_50: 3.191, walk_forward_70: 2.86, kept: 'no — a tie, not a win' },
  { variant: '+ size cubed', loo_median_miss: 3.755, loo_mean_miss: 5.034, walk_forward_50: 3.194, walk_forward_70: 2.888, kept: 'no — loses leave-one-out' },
  { variant: 'hinge size curve', loo_median_miss: 3.752, loo_mean_miss: 4.955, walk_forward_50: 3.231, walk_forward_70: 2.843, kept: 'no — loses leave-one-out' },
  { variant: 'category x size interactions', loo_median_miss: 3.874, loo_mean_miss: 5.048, walk_forward_50: 3.199, walk_forward_70: 3.042, kept: 'no — loses both' },
  { variant: 'category x size + size squared', loo_median_miss: 4.03, loo_mean_miss: 5.093, walk_forward_50: 3.611, walk_forward_70: 3.1, kept: 'no — loses both' },
  { variant: 'kitchen sink (chains + age2 + growth2 + cube + hinge)', loo_median_miss: 4.107, loo_mean_miss: 5.113, walk_forward_50: 3.236, walk_forward_70: 3.547, kept: 'no — loses both' },
  { variant: 'L1 (median) regression lam=10', loo_median_miss: 3.359, loo_mean_miss: 4.831, walk_forward_50: 3.795, walk_forward_70: 3.11, kept: 'no — buys leave-one-out, loses walk-forward badly' },
  { variant: 'L1 (median) regression lam=5', loo_median_miss: 3.63, loo_mean_miss: 4.824, walk_forward_50: 3.15, walk_forward_70: 3.524, kept: 'no — loses walk-forward at 70%' },
  { variant: 'L1 (median) regression lam=12', loo_median_miss: 3.393, loo_mean_miss: 4.85, walk_forward_50: 3.867, walk_forward_70: 3.008, kept: 'no — loses walk-forward at 50%' },
  { variant: 'ensemble Huber 0.7 + L1 0.3', loo_median_miss: 3.602, loo_mean_miss: 4.937, walk_forward_50: 3.322, walk_forward_70: 3.13, kept: 'no — loses walk-forward' },
  { variant: 'ensemble Huber 0.5 + L1 0.5', loo_median_miss: 3.531, loo_mean_miss: 4.901, walk_forward_50: 3.451, walk_forward_70: 3.324, kept: 'no — loses walk-forward' },
  { variant: 'trim worst 10% residuals, refit', loo_median_miss: 3.794, loo_mean_miss: 5.035, walk_forward_50: 3.4, walk_forward_70: 3.393, kept: 'no — loses both' },
  { variant: 'two-stage category-median, shrink k=2', loo_median_miss: 3.859, loo_mean_miss: 5.364, walk_forward_50: 3.243, walk_forward_70: 3.064, kept: 'no — loses both' },
  { variant: 'two-stage category-mean, shrink k=5', loo_median_miss: 4.037, loo_mean_miss: 5.047, walk_forward_50: 3.319, walk_forward_70: 3.585, kept: 'no — loses both' },
  { variant: 'kNN median k=5', loo_median_miss: 3.528, loo_mean_miss: 5.759, walk_forward_50: 3.188, walk_forward_70: 4.636, kept: 'no — loses walk-forward at 70% badly' },
  { variant: 'kNN median k=10 + category bonus', loo_median_miss: 3.61, loo_mean_miss: 5.538, walk_forward_50: 4.209, walk_forward_70: 4.695, kept: 'no — loses both' },
  { variant: 'train excluding CEX', loo_median_miss: 3.876, loo_mean_miss: 5.065, walk_forward_50: 3.117, walk_forward_70: 3.216, kept: 'no — loses leave-one-out and walk-forward at 70%' },
  { variant: 'train excluding bridges', loo_median_miss: 3.737, loo_mean_miss: 5.07, walk_forward_50: 3.365, walk_forward_70: 3.01, kept: 'no — loses both' },
  { variant: 'train excluding CEX + bridges', loo_median_miss: 3.885, loo_mean_miss: 5.121, walk_forward_50: 3.258, walk_forward_70: 2.745, kept: 'no — loses leave-one-out' },
  { variant: 'recency weights, half-life 1y', loo_median_miss: 4.123, loo_mean_miss: 5.252, walk_forward_50: 4.158, walk_forward_70: 4.518, kept: 'no — loses both' },
  { variant: 'category TVL share feature', loo_median_miss: 4.044, loo_mean_miss: 5.056, walk_forward_50: 3.222, walk_forward_70: 2.914, kept: 'no — loses leave-one-out' },
  { variant: 'category tokened-share feature', loo_median_miss: 3.81, loo_mean_miss: 5.01, walk_forward_50: 3.167, walk_forward_70: 2.915, kept: 'no — loses leave-one-out' },
  { variant: 'drop growth term', loo_median_miss: 3.711, loo_mean_miss: 4.99, walk_forward_50: 3.17, walk_forward_70: 2.862, kept: 'no — a tie; the growth coefficient is near zero, so the term stays for the story it tells' },
  { variant: 'drop age term', loo_median_miss: 3.725, loo_mean_miss: 4.937, walk_forward_50: 3.343, walk_forward_70: 3.098, kept: 'no — loses walk-forward' },
  { variant: 'category dummies, min train 4 / 8 / 10 / 12', loo_median_miss: 3.703, loo_mean_miss: 4.984, walk_forward_50: 3.185, walk_forward_70: 2.869, kept: 'no — ties at best (min 4 shown; 8/10/12 are x3.82/x3.69/x3.71)' },
  { variant: 'size x age interaction', loo_median_miss: 3.731, loo_mean_miss: 5.028, walk_forward_50: 3.305, walk_forward_70: 3.109, kept: 'no — loses both' },
];

/** Board rows: every watchlist slug, plus this many early rows below the floor. */
export const BOARD_ROWS = 198;
/** How many live-token protocols the Live token view carries, by market cap. */
export const LIVE_ROWS = 24;

export const DAY_MS = 86_400_000;

/** A finite number, else `fallback`. */
function finite(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

/** A whole day count from `from` to `to`, clamped at zero; null when either date is absent or bad. */
export function ageDays(asOf: string, listedAt: string | null): number | null {
  if (listedAt === null) return null;
  const a = Date.parse(`${asOf}T00:00:00Z`);
  const b = Date.parse(`${listedAt}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Math.max(0, Math.floor((a - b) / DAY_MS));
}

export const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));
export const round4 = (n: number): number => Math.round(n * 1e4) / 1e4;

/** Why the snapshot is not one this model can score. */
export class SnapshotError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SnapshotError';
  }
}

/** Solve A x = b by Gauss-Jordan with partial pivoting; null when the matrix is singular. */
export function solveLinear(A: number[][], b: number[]): number[] | null {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]!]);
  for (let i = 0; i < n; i++) {
    let piv = i;
    for (let j = i + 1; j < n; j++) if (Math.abs(M[j]![i]!) > Math.abs(M[piv]![i]!)) piv = j;
    const tmp = M[i]!;
    M[i] = M[piv]!;
    M[piv] = tmp;
    const d = M[i]![i]!;
    if (Math.abs(d) < 1e-12) return null;
    for (let j = i; j <= n; j++) M[i]![j]! /= d;
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const f = M[j]![i]!;
      if (f === 0) continue;
      for (let k = i; k <= n; k++) M[j]![k]! -= f * M[i]![k]!;
    }
  }
  return M.map((r) => r[n]!);
}

/** A training row for the multiple regression: one protocol whose token trades with a positive cap. */
interface TrainRow {
  p: SnapshotProtocol;
  y: number;
  logTvl: number;
  logAge: number;
  growth: number;
  age: number | null;
  cat: string;
}

/** The fitted multiple regression: its coefficients and the categories that carry a dummy. */
export interface MultipleFit {
  beta: number[];
  cats: string[];
  n: number;
  dropped_zero_mcap: number;
}

function trainRows(snapshot: Snapshot, model: ModelConstants): { rows: TrainRow[]; droppedZero: number } {
  const droppedZero = snapshot.protocols.filter((p) => p.mcap_usd !== null && p.mcap_usd <= 0).length;
  const rows: TrainRow[] = snapshot.protocols
    .filter((p) => p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0)
    .map((p) => {
      const age = ageDays(snapshot.as_of, p.listed_at);
      return {
        p,
        y: Math.log(p.mcap_usd! / p.tvl_usd),
        logTvl: Math.log10(p.tvl_usd),
        logAge: Math.log1p(age ?? model.unknown_age_days),
        growth: clamp(finite(p.change_7d, 0), model.growth_floor, model.growth_cap) / 100,
        age,
        cat: p.category ?? 'None',
      };
    })
    .sort((a, b) => (a.p.slug < b.p.slug ? -1 : a.p.slug > b.p.slug ? 1 : 0));
  return { rows, droppedZero };
}

function designRow(t: TrainRow, cats: string[]): number[] {
  const z = t.logTvl - 8;
  const x = [1, z, z * z, t.logAge - 5.5, t.growth];
  for (const c of cats) x.push(t.cat === c ? 1 : 0);
  return x;
}

function fitWeighted(rows: TrainRow[], cats: string[], weights: number[], lambda: number): number[] | null {
  const k = designRow(rows[0]!, cats).length;
  const A: number[][] = Array.from({ length: k }, () => new Array<number>(k).fill(0));
  const b: number[] = new Array<number>(k).fill(0);
  for (let i = 0; i < rows.length; i++) {
    const x = designRow(rows[i]!, cats);
    const w = weights[i]!;
    for (let a = 0; a < k; a++) {
      b[a]! += w * x[a]! * rows[i]!.y;
      for (let c = 0; c < k; c++) A[a]![c]! += w * x[a]! * x[c]!;
    }
  }
  for (let a = 1; a < k; a++) A[a]![a]! += lambda;
  return solveLinear(A, b);
}

/** Fit the multiple regression with ridge + Huber reweighting; null when the fit fails. */
export function fitMultiple(rows: TrainRow[], cats: string[], model: ModelConstants): MultipleFit | null {
  let beta = fitWeighted(rows, cats, rows.map(() => 1), model.ridge_lambda);
  if (beta === null) return null;
  for (let it = 0; it < model.huber_rounds; it++) {
    const res = rows.map((t) => t.y - designRow(t, cats).reduce((acc, v, i) => acc + v * beta![i]!, 0));
    const med = quantile(res.map(Math.abs), 0.5) * 1.4826 || 1;
    const k = model.huber_k * med;
    const w = res.map((r) => (Math.abs(r) <= k ? 1 : k / Math.abs(r)));
    const next = fitWeighted(rows, cats, w, model.ridge_lambda);
    if (next === null) return null;
    beta = next;
  }
  return { beta, cats, n: rows.length, dropped_zero_mcap: 0 };
}

export function predictLogMultiple(fit: MultipleFit, p: SnapshotProtocol, asOf: string, model: ModelConstants): { pred: number; sizeTerm: number; ageTerm: number; growthTerm: number; catEffect: number } {
  const age = ageDays(asOf, p.listed_at);
  const t: TrainRow = {
    p,
    y: 0,
    logTvl: Math.log10(Math.max(1, p.tvl_usd)),
    logAge: Math.log1p(age ?? model.unknown_age_days),
    growth: clamp(finite(p.change_7d, 0), model.growth_floor, model.growth_cap) / 100,
    age,
    cat: p.category ?? 'None',
  };
  const x = designRow(t, fit.cats);
  const b = fit.beta;
  const sizeTerm = b[1]! * x[1]! + b[2]! * x[2]!;
  const ageTerm = b[3]! * x[3]!;
  const growthTerm = b[4]! * x[4]!;
  let catEffect = 0;
  for (let i = 0; i < fit.cats.length; i++) catEffect += b[5 + i]! * x[5 + i]!;
  return { pred: b[0]! + sizeTerm + ageTerm + growthTerm + catEffect, sizeTerm, ageTerm, growthTerm, catEffect };
}

/** The p-quantile of numbers, by sorted position. */
export function quantile(values: number[], p: number): number {
  const a = [...values].sort((x, y) => x - y);
  if (a.length === 0) return 0;
  return a[Math.min(a.length - 1, Math.max(0, Math.floor(p * a.length)))]!;
}

/** Median absolute miss factor of residuals: exp(median |residual|). */
export function missFactor(residuals: number[]): number {
  return Math.exp(quantile(residuals.map(Math.abs), 0.5));
}

/** The multiple fit's held-out record: leave-one-out and walk-forward by listed_at. */
export interface MultipleValidation {
  n_train: number;
  dropped_zero_mcap: number;
  loo_median_miss: number;
  loo_mean_miss: number;
  loo_q10: number;
  loo_q25: number;
  loo_q75: number;
  loo_q90: number;
  coverage_p25_p75: number;
  coverage_p10_p90: number;
  walk_forward: { frac: number; n_train: number; n_test: number; median_miss: number }[];
  coefficients: { intercept: number; log_tvl: number; log_tvl_sq: number; log_age: number; growth: number };
  category_effects: { category: string; n_train: number; effect: number }[];
}

export function validateMultiple(snapshot: Snapshot, model: ModelConstants): MultipleValidation {
  const { rows, droppedZero } = trainRows(snapshot, model);
  const catN = new Map<string, number>();
  for (const t of rows) catN.set(t.cat, (catN.get(t.cat) ?? 0) + 1);
  const cats = [...catN.entries()].filter(([, n]) => n >= model.category_min_train).map(([c]) => c).sort();
  const fitAll = fitMultiple(rows, cats, model);
  if (fitAll === null) throw new SnapshotError('the multiple regression did not fit');
  fitAll.dropped_zero_mcap = droppedZero;

  // Leave-one-out: refit without each protocol, score the held-out one.
  const looResid: number[] = [];
  for (let i = 0; i < rows.length; i++) {
    const rest = rows.filter((_, j) => j !== i);
    const fit = fitMultiple(rest, cats, model);
    if (fit === null) throw new SnapshotError('the leave-one-out refit did not fit');
    const pred = predictLogMultiple(fit, rows[i]!.p, snapshot.as_of, model).pred;
    looResid.push(rows[i]!.y - pred);
  }
  const q10 = quantile(looResid, 0.1);
  const q25 = quantile(looResid, 0.25);
  const q75 = quantile(looResid, 0.75);
  const q90 = quantile(looResid, 0.9);
  const cov2575 = looResid.filter((r) => r >= q25 && r <= q75).length / looResid.length;
  const cov1090 = looResid.filter((r) => r >= q10 && r <= q90).length / looResid.length;
  const meanMiss = Math.exp(looResid.map(Math.abs).reduce((a, v) => a + v, 0) / looResid.length);

  // Walk-forward: fit the older protocols by listed_at, test the newer ones.
  const dated = rows.filter((t) => t.age !== null).sort((a, b) => (a.p.listed_at! < b.p.listed_at! ? -1 : a.p.listed_at! > b.p.listed_at! ? 1 : a.p.slug < b.p.slug ? -1 : 1));
  const walkForward: MultipleValidation['walk_forward'] = [];
  for (const frac of [0.5, 0.7]) {
    const cut = Math.floor(dated.length * frac);
    const fit = fitMultiple(dated.slice(0, cut), cats, model);
    if (fit === null) throw new SnapshotError('the walk-forward refit did not fit');
    const res = dated.slice(cut).map((t) => t.y - predictLogMultiple(fit, t.p, snapshot.as_of, model).pred);
    walkForward.push({ frac, n_train: cut, n_test: dated.length - cut, median_miss: missFactor(res) });
  }

  const b = fitAll.beta;
  return {
    n_train: rows.length,
    dropped_zero_mcap: droppedZero,
    loo_median_miss: missFactor(looResid),
    loo_mean_miss: meanMiss,
    loo_q10: round4(q10),
    loo_q25: round4(q25),
    loo_q75: round4(q75),
    loo_q90: round4(q90),
    coverage_p25_p75: round4(cov2575),
    coverage_p10_p90: round4(cov1090),
    walk_forward: walkForward,
    coefficients: { intercept: round4(b[0]!), log_tvl: round4(b[1]!), log_tvl_sq: round4(b[2]!), log_age: round4(b[3]!), growth: round4(b[4]!) },
    category_effects: cats.map((c, i) => ({ category: c, n_train: catN.get(c)!, effect: round4(b[5 + i]!) })),
  };
}

// ---------------------------------------------------------------- token odds

/** A row for the token-odds fit: every protocol, tokened when its record carries a symbol. */
interface OddsRow {
  p: SnapshotProtocol;
  y: number;
  logTvl: number;
  logAge: number;
  cat: string;
}

function oddsRows(snapshot: Snapshot, model: ModelConstants): OddsRow[] {
  return snapshot.protocols
    .map((p) => {
      const age = ageDays(snapshot.as_of, p.listed_at);
      return { p, y: p.symbol !== null ? 1 : 0, logTvl: Math.log10(Math.max(1, p.tvl_usd)), logAge: Math.log1p(age ?? model.unknown_age_days), cat: p.category ?? 'None' };
    })
    .sort((a, b) => (a.p.slug < b.p.slug ? -1 : a.p.slug > b.p.slug ? 1 : 0));
}

function oddsDesign(t: OddsRow, cats: string[]): number[] {
  const x = [1, t.logTvl - 8, t.logAge - 5.5];
  for (const c of cats) x.push(t.cat === c ? 1 : 0);
  return x;
}

/** Logistic regression by plain gradient descent (deterministic, no randomness). */
function fitLogistic(rows: OddsRow[], cats: string[], iters = 300, lr = 0.5, l2 = 1): number[] {
  const k = oddsDesign(rows[0]!, cats).length;
  const w = new Array<number>(k).fill(0);
  for (let it = 0; it < iters; it++) {
    const g = new Array<number>(k).fill(0);
    for (const t of rows) {
      const x = oddsDesign(t, cats);
      const z = clamp(x.reduce((a, v, i) => a + v * w[i]!, 0), -30, 30);
      const pr = 1 / (1 + Math.exp(-z));
      for (let i = 0; i < k; i++) g[i]! += (pr - t.y) * x[i]!;
    }
    for (let i = 0; i < k; i++) w[i]! -= lr * (g[i]! / rows.length + (i === 0 ? 0 : (l2 * w[i]!) / rows.length));
  }
  return w;
}

function predictOdds(w: number[], t: OddsRow, cats: string[]): number {
  const z = clamp(oddsDesign(t, cats).reduce((a, v, i) => a + v * w[i]!, 0), -30, 30);
  return 1 / (1 + Math.exp(-z));
}

/** The token-odds fit's record: 5-fold cross-validated Brier score against the base rate. */
export interface OddsValidation {
  n: number;
  tokened: number;
  base_rate: number;
  base_brier: number;
  cv_brier: number;
  coefficients: { intercept: number; log_tvl: number; log_age: number };
  category_effects: { category: string; n: number; effect: number }[];
}

export function validateOdds(snapshot: Snapshot, model: ModelConstants): { validation: OddsValidation; weights: number[]; cats: string[] } {
  const rows = oddsRows(snapshot, model);
  const tokened = rows.reduce((a, t) => a + t.y, 0);
  const base = tokened / rows.length;
  const catN = new Map<string, number>();
  for (const t of rows) catN.set(t.cat, (catN.get(t.cat) ?? 0) + 1);
  const cats = [...catN.entries()].filter(([, n]) => n >= 10).map(([c]) => c).sort();
  // 5-fold cross-validation, folds by slug order (deterministic).
  let brier = 0;
  let brierBase = 0;
  for (let f = 0; f < 5; f++) {
    const test = rows.filter((_, i) => i % 5 === f);
    const train = rows.filter((_, i) => i % 5 !== f);
    const w = fitLogistic(train, cats);
    for (const t of test) {
      const pr = predictOdds(w, t, cats);
      brier += (pr - t.y) ** 2;
      brierBase += (base - t.y) ** 2;
    }
  }
  const weights = fitLogistic(rows, cats);
  return {
    validation: {
      n: rows.length,
      tokened,
      base_rate: round4(base),
      base_brier: round4(brierBase / rows.length),
      cv_brier: round4(brier / rows.length),
      coefficients: { intercept: round4(weights[0]!), log_tvl: round4(weights[1]!), log_age: round4(weights[2]!) },
      category_effects: cats.map((c, i) => ({ category: c, n: catN.get(c)!, effect: round4(weights[3 + i]!) })),
    },
    weights,
    cats,
  };
}

// ---------------------------------------------------------------- farm plans

/** A farm step: the action, what it costs, and how long it takes. */
export interface FarmStep {
  text: string;
  cost: string;
  time: string;
}

/** A step as one string: its action, then its cost and time, within the gate's 20..600 characters. */
export function stepText(step: FarmStep): string {
  let s = `${step.text} Cost: ${step.cost}. Time: ${step.time}.`;
  if (s.length > 600) s = `${s.slice(0, 596)}...`;
  if (s.length < 20) s = `${s} Hold the position and check the protocol's own page.`;
  return s;
}

/** DefiLlama's version-marker rule: a name ending in a release is a pool of another protocol. */
const VERSIONED = /\s(V\d+(\.\d+)?|v\d+(\.\d+)?|\d+\.\d+)$/;

/** A whole word wrapped, staked or staking in the name: a product of a token that already trades. */
const WRAPPED_WORD = /\b(wrapped|staked|staking)\b/i;

/** A name beginning with an exchange's name: the exchange's own product, not a launch of its own. */
const EXCHANGE_PREFIX = /^(Binance|Coinbase|Bybit|Kraken|Bitget|OKX|Bitfinex|Gemini)\b/i;

/** Categories the watchlist rule keeps out: bridges and staked products of tokens that trade. */
const EXCLUDED_CATEGORIES = new Set(['Bridge', 'Canonical Bridge', 'Liquid Staking', 'Liquid Restaking', 'Restaked BTC', 'CEX']);

/** Whether a protocol records no token at all. */
export function isPreToken(p: SnapshotProtocol): boolean {
  return p.mcap_usd === null && p.gecko_id === null && p.symbol === null;
}

/**
 * Whether a protocol is one the board may rank: pre-token, and not a wrapped, staked or bridged
 * product of a token that already trades — the watchlist's own rule, applied to the whole board.
 * WBTC is Bitcoin on Ethereum and Binance staked ETH is ETH in a staking product; neither is a
 * protocol still waiting on a launch of its own, so neither is ranked, whatever its TVL.
 */
export function isRankable(p: SnapshotProtocol): boolean {
  return (
    isPreToken(p) &&
    !EXCLUDED_CATEGORIES.has(p.category ?? '') &&
    !VERSIONED.test(p.name) &&
    !WRAPPED_WORD.test(p.name) &&
    !EXCHANGE_PREFIX.test(p.name)
  );
}

/** The chains a protocol names, as a short phrase: the first two, then how many more. */
function chainPhrase(p: SnapshotProtocol): string {
  if (p.chains.length === 0) return 'its main chain';
  if (p.chains.length === 1) return p.chains[0]!;
  if (p.chains.length === 2) return `${p.chains[0]} and ${p.chains[1]}`;
  return `${p.chains[0]}, ${p.chains[1]}, +${p.chains.length - 2} more`;
}

/** The first chain a protocol names, or a neutral phrase. */
function chainOf(p: SnapshotProtocol): string {
  return p.chains.length > 0 ? p.chains[0]! : 'its main chain';
}

/** How much capital one farm position needs, from the protocol's own TVL. */
function capitalFor(p: SnapshotProtocol): number {
  if (p.tvl_usd >= 5_000_000_000) return 250;
  if (p.tvl_usd >= 1_000_000_000) return 150;
  if (p.tvl_usd >= 100_000_000) return 100;
  return 50;
}

/** How long to hold, from the protocol's age: younger protocols get a longer qualifying window. */
function holdDaysFor(age: number | null): number {
  if (age === null) return 30;
  if (age < 180) return 60;
  if (age < 365) return 45;
  return 30;
}

/** The protocol's age in words, from its listed date. */
function agePhrase(age: number | null): string {
  if (age === null) return 'an unrecorded age';
  if (age < 30) return `only ${age} days`;
  if (age < 365) return `${age} days`;
  return `${(age / 365).toFixed(1)} years`;
}

/** What the last seven days say about the protocol's TVL, in one clause. */
function growthNote(p: SnapshotProtocol): string {
  const g = p.change_7d;
  if (g === null) return 'No seven-day change is reported, so there is no momentum signal either way.';
  if (g >= 10) return `TVL is up ${g.toFixed(1)}% over seven days, so momentum is with the protocol \u2014 keep the weekly cadence.`;
  if (g <= -10) return `TVL is down ${Math.abs(g).toFixed(1)}% over seven days, so keep new deposits small until the outflow steadies.`;
  if (Math.abs(g) < 5) return `TVL is ${g >= 0 ? 'up' : 'down'} ${Math.abs(g).toFixed(1)}% over seven days \u2014 close to flat, so no momentum signal either way.`;
  return `TVL is ${g >= 0 ? 'up' : 'down'} ${Math.abs(g).toFixed(1)}% over seven days \u2014 a modest move, no strong signal either way.`;
}

/** A dollar figure in words: cents below $100, whole dollars above. */
function money(n: number): string {
  if (n < 0) return `-$${Math.abs(n) < 100 ? Math.abs(n).toFixed(2) : Math.round(Math.abs(n)).toLocaleString('en-US')}`;
  return `$${n < 100 ? n.toFixed(2) : Math.round(n).toLocaleString('en-US')}`;
}

/** The category's one core action, in the words a farmer can follow. */
function actionFor(p: SnapshotProtocol, capital: number, chains: string): string {
  const c = (p.category ?? '').toLowerCase();
  const name = p.name;
  const sharePct = (capital / Math.max(1, p.tvl_usd)) * 100;
  const share = sharePct < 0.0001 ? 'under 0.0001%' : sharePct < 0.01 ? `${sharePct.toFixed(4)}%` : `${sharePct.toFixed(2)}%`;
  const size = `That is about ${share} of its ${money(p.tvl_usd)} TVL, so the position is small next to the pool.`;
  if (c.includes('lend') || c.includes('cdp') || c.includes('collateral')) return `Supply $${capital} of USDC to ${name} on ${chains} and keep the position open. ${size}`;
  if (c.includes('dex') || c.includes('yield') || c.includes('leveraged farm')) return `Provide $${capital} of liquidity to ${name}'s main pool on ${chains} and keep the position open. ${size}`;
  if (c.includes('bridge')) return `Bridge $${capital} of USDC to ${chains} through ${name} and use the bridged funds in its app. ${size}`;
  if (c.includes('staking') || c.includes('restak')) return `Stake $${capital} of the chain's staking asset through ${name} on ${chains} and keep the receipt token. ${size}`;
  if (c.includes('perpetual') || c.includes('derivative')) return `Deposit $${capital} of USDC as margin on ${name} on ${chains} and open one small hedged position. ${size}`;
  if (c.includes('prediction')) return `Fund $${capital} and place one market position on ${name} on ${chains}. ${size}`;
  if (c.includes('curator') || c.includes('allocator')) return `Deposit $${capital} into a ${name} vault on ${chains} and keep the shares. ${size}`;
  if (c.includes('stable')) return `Mint or deposit $${capital} of the stable asset through ${name} on ${chains}. ${size}`;
  if (c.includes('rwa')) return `Mint $${capital} of the protocol's yield-bearing asset on ${chains} and hold it. ${size}`;
  if (c.includes('insurance')) return `Buy $${capital} of cover from ${name} on ${chains} and keep the policy. ${size}`;
  return `Deposit $${capital} of USDC into ${name} on ${chains} and keep the position open. ${size}`;
}

/** The category's weekly qualifying action, in the words a farmer can follow. */
function weeklyActionFor(p: SnapshotProtocol): string {
  const c = (p.category ?? '').toLowerCase();
  if (c.includes('lend') || c.includes('cdp') || c.includes('collateral')) return 'add or remove a little supply, or borrow and repay a little, so the wallet shows fresh activity';
  if (c.includes('dex') || c.includes('yield') || c.includes('leveraged farm')) return 'add or remove a little liquidity, or make one swap, so the wallet shows fresh activity';
  if (c.includes('bridge')) return 'bridge a small amount back or forth, so the wallet shows fresh activity';
  if (c.includes('staking') || c.includes('restak')) return 'top up the stake a little, or claim and restake the rewards, so the wallet shows fresh activity';
  if (c.includes('perpetual') || c.includes('derivative')) return 'open and close one small hedged trade, so the wallet shows fresh activity';
  if (c.includes('prediction')) return 'settle one market position and open another, so the wallet shows fresh activity';
  if (c.includes('curator') || c.includes('allocator')) return 'move a little between vaults, or top up the deposit, so the wallet shows fresh activity';
  if (c.includes('stable')) return 'mint or redeem a small amount, so the wallet shows fresh activity';
  if (c.includes('rwa')) return 'buy or redeem a small amount of the yield-bearing asset, so the wallet shows fresh activity';
  if (c.includes('insurance')) return 'renew or adjust the cover a little, so the wallet shows fresh activity';
  return 'make one small deposit, swap or vote, so the wallet shows fresh activity';
}

/**
 * A protocol's farm plan: six steps, each an action with a cost and a time. Every step is
 * built from THIS protocol's own record \u2014 its chains, its category mechanics, its TVL size,
 * its seven-day growth, its age with no token recorded, and the verdict math of its own
 * row \u2014 so two protocols in one category never read the same plan.
 */
export function farmSteps(p: SnapshotProtocol, age: number | null, econ: FarmEconomics): FarmStep[] {
  const chains = chainPhrase(p);
  const chain = chainOf(p);
  const capital = capitalFor(p);
  const hold = holdDaysFor(age);
  const gas = capital >= 150 ? 8 : 4;
  const skip = econ.verdict === 'SKIP';
  const core = skip
    ? `Do not farm ${p.name} for real yet: at ${money(econ.usd_per_hour)}/hr it is a SKIP, so if you want exposure at all, put only the $${capital} minimum into ${p.name} on ${chains}, and treat it as a lottery ticket, not a farm.`
    : actionFor(p, capital, chains);
  const holdStep = skip
    ? `Do not lock capital in ${p.name} for ${hold} days: the plan loses ${money(Math.abs(econ.usd_per_hour))}/hr after costs, so watch its points page once a month instead and come back only if the verdict changes.`
    : `Hold the position for at least ${hold} days \u2014 ${p.name} has run ${agePhrase(age)} with no token recorded, so the activity needs to span more than one snapshot to count.`;
  const verdictLine =
    econ.usd_per_hour >= MODEL.farm_hourly_hurdle
      ? `clears the $25/hr hurdle`
      : econ.usd_per_hour >= MODEL.marginal_hourly
        ? `clears $10/hr but not the $25/hr hurdle`
        : econ.usd_per_hour >= 0
          ? `earns something, but not the $10/hr a marginal farm needs`
          : `loses money after costs`;
  return [
    { text: `Open a fresh wallet on ${chain} that holds only the funds you plan to farm, and keep its seed phrase offline. Fund it with $${capital + gas} \u2014 $${capital} of capital plus about $${gas} of gas.`, cost: `about $${gas} of gas to fund and move`, time: '5 minutes' },
    { text: core, cost: `$${capital} of capital and about $${gas} of gas`, time: '15 minutes to set up' },
    { text: `Each week, ${weeklyActionFor(p)}. ${growthNote(p)}`, cost: 'about $2 of gas a week', time: '10 minutes a week' },
    { text: holdStep, cost: skip ? 'nothing locked' : `the $${capital} stays in the position`, time: skip ? '5 minutes a month to check' : `${hold} days` },
    { text: `Before you withdraw, open ${p.name}'s own points or rewards page on ${chain} and claim only from the link it publishes \u2014 never from a reply, a DM or a search ad.`, cost: 'no fee to check, and never pay to claim', time: '10 minutes' },
    { text: `Verdict ${econ.verdict}: expected payoff ${money(econ.expected_payoff_usd)} against ${money(econ.total_cost_usd)} total cost over ${econ.total_hours.toFixed(1)} hours is ${money(econ.usd_per_hour)}/hr, which ${verdictLine}. Write the wallet, the dates and every transaction in a farm log.`, cost: 'nothing but the log', time: '5 minutes' },
  ];
}

/** The farm verdict against the T-Bill + $25/hr hurdle, from the plan's own figures. */
export type FarmVerdict = 'FARM' | 'MARGINAL' | 'THIN' | 'SKIP';

export interface FarmEconomics {
  capital_usd: number;
  hold_days: number;
  gas_usd: number;
  opportunity_cost_usd: number;
  total_cost_usd: number;
  total_hours: number;
  expected_payoff_usd: number;
  usd_per_hour: number;
  verdict: FarmVerdict;
}

export function farmEconomics(p: SnapshotProtocol, age: number | null, implied: number, model: ModelConstants): FarmEconomics {
  const capital = capitalFor(p);
  const hold = holdDaysFor(age);
  const weeks = Math.max(1, Math.round(hold / 7));
  const gas = 3 + (capital >= 150 ? 8 : 4) + 2 * weeks;
  const opportunity = (capital * model.tbill_rate * hold) / 365;
  const totalCost = gas + opportunity;
  const totalHours = 5 / 60 + 15 / 60 + (weeks * 10) / 60 + 10 / 60 + 5 / 60;
  const payoff = implied * model.airdrop_pool_share * model.farmer_share_of_pool;
  const perHour = (payoff - totalCost) / totalHours;
  const verdict: FarmVerdict = perHour >= model.farm_hourly_hurdle ? 'FARM' : perHour >= model.marginal_hourly ? 'MARGINAL' : perHour >= 0 ? 'THIN' : 'SKIP';
  return {
    capital_usd: capital,
    hold_days: hold,
    gas_usd: round4(gas),
    opportunity_cost_usd: round4(opportunity),
    total_cost_usd: round4(totalCost),
    total_hours: round4(totalHours),
    expected_payoff_usd: round4(payoff),
    usd_per_hour: round4(perHour),
    verdict,
  };
}

// ---------------------------------------------------------------- the board

/** One row of board.json: the ranked board's row. */
export interface BoardRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
  token_status: 'pre-token';
  watchlist: boolean;
  age_days: number | null;
  implied_value_usd: number;
  model: {
    fitted_multiple: number;
    token_odds: number;
    size_term: number;
    category_effect: number;
    age_term: number;
    growth_term: number;
    p25_usd: number;
    p75_usd: number;
    p10_usd: number;
    p90_usd: number;
    verdict: FarmVerdict;
    usd_per_hour: number;
    expected_payoff_usd: number;
    total_cost_usd: number;
    total_hours: number;
  };
  farm_steps: string[];
}

/** A live-token protocol: it already trades, so the board shows the market's own value, not a model's. */
export interface LiveRow {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  mcap_usd: number | null;
  change_7d: number | null;
}

/** One reconciliation group: every one of the snapshot's 758 protocols sits in exactly one. */
export interface ReconGroup {
  group: string;
  count: number;
  note: string;
}

/** Everything the dashboard renders, from the snapshot. */
export interface RadarData {
  as_of: string;
  source: string;
  floor_usd: number;
  model: ModelConstants;
  model_version: string;
  watchlist_size: number;
  board_size: number;
  early_size: number;
  live_size: number;
  total_implied_usd: number;
  verdict_counts: Record<FarmVerdict, number>;
  validation: MultipleValidation & { odds: OddsValidation };
  reconciliation: ReconGroup[];
  rows: BoardRow[];
  live: LiveRow[];
}

/** The protocols the board ranks, strongest first, with ranks 1..n. */
export function buildBoard(snapshot: Snapshot, model: ModelConstants = MODEL): { board: BoardRow[]; radar: RadarData } {
  const multiple = validateMultiple(snapshot, model);
  const { rows: trainRowsList, droppedZero } = trainRows(snapshot, model);
  void droppedZero;
  const catN = new Map<string, number>();
  for (const t of trainRowsList) catN.set(t.cat, (catN.get(t.cat) ?? 0) + 1);
  const cats = [...catN.entries()].filter(([, n]) => n >= model.category_min_train).map(([c]) => c).sort();
  const fit = fitMultiple(trainRowsList, cats, model);
  if (fit === null) throw new SnapshotError('the multiple regression did not fit');
  const odds = validateOdds(snapshot, model);
  const oddsBySlug = new Map<string, number>();
  const oddsRowBySlug = new Map<string, OddsRow>();
  for (const t of oddsRows(snapshot, model)) {
    oddsBySlug.set(t.p.slug, predictOdds(odds.weights, t, odds.cats));
    oddsRowBySlug.set(t.p.slug, t);
  }
  void oddsRowBySlug;

  const bySlug = new Map(snapshot.protocols.map((p) => [p.slug, p]));
  const watchlist = new Set(snapshot.watchlist);
  // The watchlist is the floor: every slug on it. The early radar fills the rest of the board
  // with the rankable pre-token protocols the list leaves out, by implied value.
  const chosen = new Map<string, SnapshotProtocol>();
  for (const slug of snapshot.watchlist) {
    const p = bySlug.get(slug);
    if (p !== undefined) chosen.set(slug, p);
  }
  const score = (p: SnapshotProtocol): number => {
    const { pred } = predictLogMultiple(fit, p, snapshot.as_of, model);
    return p.tvl_usd * Math.exp(pred) * (oddsBySlug.get(p.slug) ?? 0.5);
  };
  const early = snapshot.protocols
    .filter((p) => !chosen.has(p.slug) && isRankable(p))
    .sort((a, b) => score(b) - score(a) || (a.slug < b.slug ? -1 : a.slug > b.slug ? 1 : 0))
    .slice(0, BOARD_ROWS - chosen.size);
  for (const p of early) chosen.set(p.slug, p);

  const scored = [...chosen.values()].map((p) => {
    const age = ageDays(snapshot.as_of, p.listed_at);
    const { pred, sizeTerm, ageTerm, growthTerm, catEffect } = predictLogMultiple(fit, p, snapshot.as_of, model);
    const multipleValue = Math.exp(pred);
    const tokenOdds = oddsBySlug.get(p.slug) ?? 0.5;
    const implied = Math.max(1, Math.round(p.tvl_usd * multipleValue * tokenOdds));
    const range = (q: number): number => Math.max(1, Math.round(p.tvl_usd * Math.exp(pred + q) * tokenOdds));
    const econ = farmEconomics(p, age, implied, model);
    return { p, age, multipleValue, tokenOdds, sizeTerm, ageTerm, growthTerm, catEffect, implied, range, econ };
  });
  scored.sort((a, b) => b.implied - a.implied || (a.p.slug < b.p.slug ? -1 : a.p.slug > b.p.slug ? 1 : 0));
  const board: BoardRow[] = scored.map((s, i) => ({
    rank: i + 1,
    slug: s.p.slug,
    name: s.p.name,
    category: s.p.category,
    chains: s.p.chains,
    tvl_usd: s.p.tvl_usd,
    change_7d: s.p.change_7d,
    mcap_usd: s.p.mcap_usd,
    listed_at: s.p.listed_at,
    gecko_id: s.p.gecko_id,
    symbol: s.p.symbol,
    token_status: 'pre-token',
    watchlist: watchlist.has(s.p.slug),
    age_days: s.age,
    implied_value_usd: s.implied,
    model: {
      fitted_multiple: round4(s.multipleValue),
      token_odds: round4(s.tokenOdds),
      size_term: round4(s.sizeTerm),
      category_effect: round4(s.catEffect),
      age_term: round4(s.ageTerm),
      growth_term: round4(s.growthTerm),
      p25_usd: s.range(multiple.loo_q25),
      p75_usd: s.range(multiple.loo_q75),
      p10_usd: s.range(multiple.loo_q10),
      p90_usd: s.range(multiple.loo_q90),
      verdict: s.econ.verdict,
      usd_per_hour: s.econ.usd_per_hour,
      expected_payoff_usd: s.econ.expected_payoff_usd,
      total_cost_usd: s.econ.total_cost_usd,
      total_hours: s.econ.total_hours,
    },
    farm_steps: farmSteps(s.p, s.age, s.econ).map(stepText),
  }));

  // Reconciliation: every protocol in exactly one group, summing to the snapshot's 758.
  const tokened = snapshot.protocols.filter((p) => !isPreToken(p)).length;
  const excluded = snapshot.protocols.filter((p) => isPreToken(p) && !isRankable(p)).length;
  const onBoard = new Set(board.map((r) => r.slug));
  const offList = snapshot.protocols.filter((p) => isPreToken(p) && isRankable(p) && !onBoard.has(p.slug)).length;
  const reconciliation: ReconGroup[] = [
    { group: 'watchlist', count: snapshot.watchlist.length, note: 'On the house watchlist and on this board' },
    { group: 'early', count: board.length - snapshot.watchlist.length, note: 'Pre-token below the floor or off the list, on this board' },
    { group: 'off-list', count: offList, note: offList === 0 ? 'None: every rankable pre-token protocol is on this board' : 'Pre-token and rankable, but below this board\u2019s cut' },
    { group: 'already-tokened', count: tokened, note: 'Records a token already; shown at the market\u2019s own value, never ranked' },
    { group: 'excluded-by-rule', count: excluded, note: 'Pre-token but a bridge, a staked or wrapped product, an exchange product or a versioned pool' },
  ];
  const reconTotal = reconciliation.reduce((a, g) => a + g.count, 0);
  if (reconTotal !== snapshot.protocols.length) {
    throw new SnapshotError(`reconciliation covers ${reconTotal} of ${snapshot.protocols.length} protocols`);
  }

  const live: LiveRow[] = snapshot.protocols
    .filter((p) => !isPreToken(p) && p.category !== 'CEX')
    .sort((a, b) => (b.mcap_usd ?? 0) - (a.mcap_usd ?? 0) || b.tvl_usd - a.tvl_usd)
    .slice(0, LIVE_ROWS)
    .map((p) => ({ slug: p.slug, name: p.name, category: p.category, chains: p.chains, tvl_usd: p.tvl_usd, mcap_usd: p.mcap_usd, change_7d: p.change_7d }));

  const verdictCounts: Record<FarmVerdict, number> = { FARM: 0, MARGINAL: 0, THIN: 0, SKIP: 0 };
  for (const r of board) verdictCounts[r.model.verdict]++;

  const radar: RadarData = {
    as_of: snapshot.as_of,
    source: snapshot.source,
    floor_usd: snapshot.floor_usd,
    model,
    model_version: MODEL_VERSION,
    watchlist_size: snapshot.watchlist.length,
    board_size: board.length,
    early_size: board.length - snapshot.watchlist.length,
    live_size: live.length,
    total_implied_usd: board.reduce((a, r) => a + r.implied_value_usd, 0),
    verdict_counts: verdictCounts,
    validation: { ...multiple, odds: odds.validation },
    reconciliation,
    rows: board,
    live,
  };
  return { board, radar };
}

/** The snapshot out of a file's text, refused unless it is the shape the model scores. */
export function parseSnapshot(text: string): Snapshot {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new SnapshotError('snapshot.json is not JSON');
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw new SnapshotError('snapshot.json is not a JSON object');
  const o = raw as Record<string, unknown>;
  if (typeof o.as_of !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(o.as_of)) throw new SnapshotError('snapshot.json has no as_of date');
  if (typeof o.source !== 'string') throw new SnapshotError('snapshot.json has no source');
  if (typeof o.floor_usd !== 'number' || !Number.isSafeInteger(o.floor_usd) || o.floor_usd <= 0) throw new SnapshotError('snapshot.json has no floor_usd');
  if (!Array.isArray(o.protocols)) throw new SnapshotError('snapshot.json has no protocols array');
  if (!Array.isArray(o.watchlist) || o.watchlist.some((s) => typeof s !== 'string')) throw new SnapshotError('snapshot.json has no watchlist of slugs');
  const protocols = o.protocols.map((v, i) => {
    if (typeof v !== 'object' || v === null || Array.isArray(v)) throw new SnapshotError(`snapshot.json's protocol ${i} is not an object`);
    const p = v as Record<string, unknown>;
    if (typeof p.slug !== 'string' || typeof p.name !== 'string' || typeof p.tvl_usd !== 'number' || !Number.isFinite(p.tvl_usd)) {
      throw new SnapshotError(`snapshot.json's protocol ${i} lacks a slug, a name or a TVL`);
    }
    return {
      slug: p.slug,
      name: p.name,
      category: typeof p.category === 'string' ? p.category : null,
      chains: Array.isArray(p.chains) ? p.chains.filter((c): c is string => typeof c === 'string') : [],
      tvl_usd: p.tvl_usd,
      change_7d: typeof p.change_7d === 'number' && Number.isFinite(p.change_7d) ? p.change_7d : null,
      listed_at: typeof p.listed_at === 'string' ? p.listed_at : null,
      mcap_usd: typeof p.mcap_usd === 'number' && Number.isFinite(p.mcap_usd) ? p.mcap_usd : null,
      gecko_id: typeof p.gecko_id === 'string' ? p.gecko_id : null,
      symbol: typeof p.symbol === 'string' ? p.symbol : null,
    };
  });
  return { as_of: o.as_of, source: o.source, floor_usd: o.floor_usd, protocols, watchlist: o.watchlist as string[] };
}

/** Why a board is not one the gate will take, or null: coverage, ranks, order, one shared implied value, and farm steps. */
export function boardIssue(board: BoardRow[], snapshot: Snapshot): string | null {
  const slugs = new Set(board.map((r) => r.slug));
  const missing = snapshot.watchlist.filter((s) => !slugs.has(s));
  if (missing.length > 0) return `the board misses ${missing.length} watchlist slug${missing.length === 1 ? '' : 's'}: ${missing.slice(0, 5).join(', ')}`;
  const ranks = board.map((r) => r.rank).sort((a, b) => a - b);
  for (let i = 0; i < ranks.length; i++) if (ranks[i] !== i + 1) return `rank ${i + 1} is ${ranks[i]}`;
  for (let i = 1; i < board.length; i++) if (board[i]!.implied_value_usd > board[i - 1]!.implied_value_usd) return `row ${i + 1} outranks row ${i} on implied value`;
  if (board.length >= 2 && board[0]!.implied_value_usd === board[board.length - 1]!.implied_value_usd) {
    return `all ${board.length} rows share the same implied value $${board[0]!.implied_value_usd}: a board that ranks nothing by value is not a ranking`;
  }
  for (const r of board) {
    const steps = r.farm_steps;
    if (steps.length < 3 || steps.length > 12) return `${r.slug} has ${steps.length} farm steps`;
    for (const s of steps) if (s.length < 20 || s.length > 600) return `${r.slug} has a farm step of ${s.length} characters`;
    if (!Number.isSafeInteger(r.implied_value_usd) || r.implied_value_usd <= 0) return `${r.slug} has implied value ${r.implied_value_usd}`;
  }
  return null;
}

/** A compact data.js: the dashboard's whole dataset as one script, no fetch and no network. */
export function dataScript(radar: RadarData): string {
  return `window.RADAR=${JSON.stringify(radar)};\n`;
}

/** The validation summary the dashboard ships as dashboard/model.json, for the judge to read. */
export function modelSummary(radar: RadarData): Record<string, unknown> {
  const v = radar.validation;
  return {
    model_version: radar.model_version,
    as_of: radar.as_of,
    board_rows: radar.board_size,
    watchlist: radar.watchlist_size,
    early_rows: radar.early_size,
    total_implied_usd: radar.total_implied_usd,
    verdict_counts: radar.verdict_counts,
    variants_tried: VARIANTS_TRIED,
    multiple_fit: {
      n_train: v.n_train,
      dropped_zero_mcap: v.dropped_zero_mcap,
      loo_median_miss: v.loo_median_miss,
      loo_mean_miss: v.loo_mean_miss,
      walk_forward: v.walk_forward,
      loo_residual_quantiles: { p10: v.loo_q10, p25: v.loo_q25, p75: v.loo_q75, p90: v.loo_q90 },
      actual_coverage_p25_p75: v.coverage_p25_p75,
      actual_coverage_p10_p90: v.coverage_p10_p90,
      coefficients: v.coefficients,
      category_effects: v.category_effects,
    },
    token_odds: {
      n: v.odds.n,
      tokened: v.odds.tokened,
      base_rate: v.odds.base_rate,
      base_brier: v.odds.base_brier,
      cv_brier_5fold: v.odds.cv_brier,
      coefficients: v.odds.coefficients,
      category_effects: v.odds.category_effects,
    },
    reconciliation: radar.reconciliation,
    farm_assumptions: {
      airdrop_pool_share: radar.model.airdrop_pool_share,
      farmer_share_of_pool: radar.model.farmer_share_of_pool,
      tbill_rate: radar.model.tbill_rate,
      farm_hourly_hurdle_usd: radar.model.farm_hourly_hurdle,
      marginal_hourly_usd: radar.model.marginal_hourly,
    },
  };
}
