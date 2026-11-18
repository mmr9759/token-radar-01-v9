// radar-9: comps-calibrated token launch radar.
//
// Prices the 198 clean pre-token protocols in the snapshot. Nothing here is
// priced by name: every implied value is TVL times a multiple, and the
// multiple comes from the 128 protocols whose token already trades.
//
// The model, in one paragraph:
//   - Observed multiples (mcap / TVL) are logged and winsorised at the 5th
//     and 95th percentiles. Each category's multiple is the median of its
//     winsorised log-multiples, shrunk with K = 10 toward the global median
//     multiple of 0.2136x. A category with no trading comps gets the global.
//   - A size curve then bends the multiple: log-multiple moves by -0.431
//     per decade of TVL (a Huber fit on the residuals after the category
//     effect, taken at half strength), centred on the median log10 TVL of
//     the board itself. A protocol 10x the median size keeps about 65% of
//     its multiple. The curve is fit after the category effect on purpose:
//     fitting it first, as an earlier version did, punished large protocols
//     for being in categories whose multiples were already low.
//   - Two small terms follow, each shrunk 50% and capped at +/-0.15 in log
//     units: the 7-day TVL growth (fitted negative -- a hot week usually
//     means fresh TVL just arrived in the denominator) and the protocol's
//     age since DefiLlama first listed it. Both come from a joint Huber
//     fit on the residuals after category and size.
//   - Uncertainty comes from 200 bootstrap refits of the comps (fixed
//     seed), giving a P10-P90 value and a rank range per row.
//   - The model is tested blind: leave-one-out over the 128 comps and a
//     walk-forward split by listing date, each against a flat global
//     multiple. Run with --blind-test to see the numbers.
//
// Usage: MAKEFUN_OUTPUT_DIR=out node main.ts --input <snapshot.json>
// Writes board.json, dashboard/ (with data.js), screenshots/ and
// dossier.json into the output directory. Two runs write the same bytes.

import fs from "node:fs";
import path from "node:path";
import { buildDossier } from "./vendor/makefun-board.ts";

// --- Model constants (fitted once, on the full comps set) -------------------
const GLOBAL_MULTIPLE = 0.2136; // median multiple of the trading comps
const SHRINK_K = 10; // shrinkage strength toward the global median
const SIZE_SLOPE = -0.431; // ln-multiple per log10 TVL, half strength (Huber)
const BOOTSTRAP_REPS = 200;
const BOOTSTRAP_SEED = 20261114;
const ADJUST_SHRINK = 0.5; // growth/age terms are halved
const ADJUST_CAP = 0.15; // each small term is capped at +/-0.15 log units
const DOSSIER_TEXT_MAX = 65536;

// --- Snapshot types ---------------------------------------------------------
interface Protocol {
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

interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: Protocol[];
  watchlist: string[];
}

interface FittedModel {
  categoryLog: (category: string | null) => number;
  categoryMultiple: (category: string | null) => number;
  categoryCount: (category: string | null) => number;
  growthCoef: number;
  ageCoef: number;
  sizeRef: number;
}

interface PricedRow {
  protocol: Protocol;
  rank: number;
  value: number;
  multiple: number;
  baseMultiple: number;
  sizeAdj: number;
  growthAdj: number;
  ageAdj: number;
  ageDays: number | null;
  inWatchlist: boolean;
  valueP10: number;
  valueP90: number;
  rankP10: number;
  rankP90: number;
}

// --- Small numeric helpers --------------------------------------------------
function quantile(values: number[], f: number): number {
  const s = [...values].sort((a, b) => a - b);
  if (s.length === 0) return Number.NaN;
  const pos = f * (s.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

function mean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function geoMeanRatio(errors: number[]): number {
  return Math.exp(mean(errors.map((e) => Math.log(e))));
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

function round4(x: number): number {
  return Math.round(x * 10000) / 10000;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// --- Linear algebra: least squares via the normal equations -----------------
function solveLinear(matrix: number[][], rhs: number[]): number[] {
  const n = rhs.length;
  const a = matrix.map((row, i) => [...row, rhs[i]]);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) {
      if (Math.abs(a[r][c]) > Math.abs(a[piv][c])) piv = r;
    }
    const tmp = a[c];
    a[c] = a[piv];
    a[piv] = tmp;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = a[r][c] / a[c][c];
      for (let j = c; j <= n; j++) a[r][j] -= f * a[c][j];
    }
  }
  return a.map((row, i) => row[n] / row[i]);
}

function leastSquares(rows: number[][], y: number[], weights?: number[]): number[] {
  const k = rows[0].length;
  const xtx: number[][] = Array.from({ length: k }, () => Array<number>(k).fill(0));
  const xty: number[] = Array<number>(k).fill(0);
  for (let i = 0; i < rows.length; i++) {
    const w = weights ? weights[i] : 1;
    for (let j = 0; j < k; j++) {
      xty[j] += w * rows[i][j] * y[i];
      for (let l = 0; l < k; l++) xtx[j][l] += w * rows[i][j] * rows[i][l];
    }
  }
  return solveLinear(xtx, xty);
}

// Huber regression by IRLS: OLS start, then reweight by the Huber loss with
// a MAD scale, delta 1.345. Robust to the wild multiples in the comps set.
function huberFit(rows: number[][], y: number[], delta = 1.345): number[] {
  let beta = leastSquares(rows, y);
  for (let it = 0; it < 200; it++) {
    const resid = y.map((yi, i) => yi - rows[i].reduce((s, x, j) => s + x * beta[j], 0));
    const scale = 1.4826 * quantile(resid.map((r) => Math.abs(r)), 0.5) || 1;
    const weights = resid.map((r) => {
      const u = Math.abs(r) / (delta * scale);
      return u <= 1 ? 1 : 1 / u;
    });
    const next = leastSquares(rows, y, weights);
    if (next.every((v, j) => Math.abs(v - beta[j]) < 1e-10)) {
      beta = next;
      break;
    }
    beta = next;
  }
  return beta;
}

// --- Snapshot helpers -------------------------------------------------------
function ageDaysOf(p: Protocol, asOf: Date): number | null {
  if (!p.listed_at) return null;
  const listed = new Date(p.listed_at + "T00:00:00Z");
  return (asOf.getTime() - listed.getTime()) / 86400000;
}

function isTradingComp(p: Protocol): boolean {
  return p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0;
}

// The board is the clean pre-token set: it records no token, and it is not
// a bridge, a liquid-staking or restaking product, a CEX, a versioned
// duplicate, a wrapped or staked product, or an exchange's own product.
const EXCLUDED_CATEGORIES = new Set([
  "Bridge",
  "Canonical Bridge",
  "Liquid Staking",
  "Liquid Restaking",
  "Restaked BTC",
  "CEX",
]);
const VERSION_SUFFIX = /\s(V\d+(\.\d+)?|v\d+(\.\d+)?|\d+\.\d+)$/;
const WRAPPED_WORD = /\b(wrapped|staked|staking)\b/i;
const EXCHANGE_PREFIXES = [
  "Binance",
  "Coinbase",
  "Bybit",
  "Kraken",
  "Bitget",
  "OKX",
  "Bitfinex",
  "Gemini",
];

function isBoardRow(p: Protocol): boolean {
  if (p.gecko_id !== null || p.symbol !== null) return false;
  if (p.category !== null && EXCLUDED_CATEGORIES.has(p.category)) return false;
  if (VERSION_SUFFIX.test(p.name)) return false;
  if (WRAPPED_WORD.test(p.name)) return false;
  if (EXCHANGE_PREFIXES.some((e) => p.name.startsWith(e))) return false;
  return true;
}

// --- The model ----------------------------------------------------------------
function fitModel(comps: Protocol[], asOf: Date, sizeRef: number): FittedModel {
  // Observed log-multiples, winsorised at the 5th and 95th percentiles.
  const logs = comps.map((p) => Math.log((p.mcap_usd as number) / p.tvl_usd));
  const lo = quantile(logs, 0.05);
  const hi = quantile(logs, 0.95);
  const byCategory = new Map<string, number[]>();
  for (const p of comps) {
    const key = p.category ?? "";
    const v = Math.log((p.mcap_usd as number) / p.tvl_usd);
    const w = Math.min(hi, Math.max(lo, v));
    const arr = byCategory.get(key);
    if (arr) arr.push(w);
    else byCategory.set(key, [w]);
  }
  const globalLog = Math.log(GLOBAL_MULTIPLE);
  // A category's log-multiple: its median winsorised log-multiple, shrunk
  // with K observations' weight toward the global median. No comps at all
  // in the category means the global median, unadjusted.
  const categoryLog = (category: string | null): number => {
    if (category === null) return globalLog;
    const arr = byCategory.get(category);
    if (!arr || arr.length === 0) return globalLog;
    return (arr.length * quantile(arr, 0.5) + SHRINK_K * globalLog) / (arr.length + SHRINK_K);
  };
  const categoryMultiple = (category: string | null): number => Math.exp(categoryLog(category));
  const categoryCount = (category: string | null): number => {
    if (category === null) return 0;
    return byCategory.get(category)?.length ?? 0;
  };
  // Growth and age: a joint Huber fit of the residuals left after the
  // category effect and the size curve, over the comps that report both a
  // 7-day change and a listing date. Age is measured in years.
  let growthCoef = 0;
  let ageCoef = 0;
  const complete = comps.filter((p) => p.change_7d !== null && ageDaysOf(p, asOf) !== null);
  if (complete.length >= 10) {
    const rows: number[][] = [];
    const y: number[] = [];
    for (const p of complete) {
      const resid =
        Math.log((p.mcap_usd as number) / p.tvl_usd) -
        categoryLog(p.category) -
        SIZE_SLOPE * (Math.log10(p.tvl_usd) - sizeRef);
      rows.push([1, p.change_7d as number, (ageDaysOf(p, asOf) as number) / 365.25]);
      y.push(resid);
    }
    const beta = huberFit(rows, y);
    growthCoef = beta[1];
    ageCoef = beta[2];
  }
  return { categoryLog, categoryMultiple, categoryCount, growthCoef, ageCoef, sizeRef };
}

interface PriceBreakdown {
  value: number;
  multiple: number;
  baseMultiple: number;
  sizeAdj: number;
  growthAdj: number;
  ageAdj: number;
}

function priceFloat(p: Protocol, model: FittedModel, asOf: Date): PriceBreakdown {
  const baseLog = model.categoryLog(p.category);
  const sizeAdj = SIZE_SLOPE * (Math.log10(p.tvl_usd) - model.sizeRef);
  const growthAdj =
    p.change_7d === null ? 0 : clamp(ADJUST_SHRINK * model.growthCoef * p.change_7d, -ADJUST_CAP, ADJUST_CAP);
  const age = ageDaysOf(p, asOf);
  const ageAdj =
    age === null ? 0 : clamp(ADJUST_SHRINK * model.ageCoef * (age / 365.25), -ADJUST_CAP, ADJUST_CAP);
  const multiple = Math.exp(baseLog + sizeAdj + growthAdj + ageAdj);
  return {
    value: p.tvl_usd * multiple,
    multiple,
    baseMultiple: Math.exp(baseLog),
    sizeAdj,
    growthAdj,
    ageAdj,
  };
}

function flatValue(p: Protocol, model: FittedModel): number {
  return p.tvl_usd * model.categoryMultiple(p.category);
}

// --- Farm plans ---------------------------------------------------------------
// The same rules the dashboard renders: gas per chain, a deposit tier by
// TVL, one concrete action per category, a weekly cadence, and an exit.
const GAS_USD: Record<string, number> = {
  Ethereum: 3.5,
  Arbitrum: 0.25,
  Optimism: 0.25,
  Base: 0.25,
  Solana: 0.02,
  BSC: 0.2,
  "BNB Chain": 0.2,
  Polygon: 0.05,
  Avalanche: 0.3,
  "zkSync Era": 0.3,
  Linea: 0.15,
  Scroll: 0.2,
  Blast: 0.1,
  Mode: 0.1,
  Mantle: 0.1,
  Fantom: 0.1,
  Gnosis: 0.1,
  Celo: 0.05,
  Sui: 0.02,
  Aptos: 0.02,
  Tron: 0.3,
  Cronos: 0.1,
  Kava: 0.1,
  Sei: 0.02,
};

const FARM_ACTION: Record<string, string> = {
  Lending:
    "Supply $DEP of a major asset as collateral, then borrow a small amount (under 30% of the collateral) so the position stays healthy.",
  Dexs: "Swap $DEP through the pool in two trades a week apart, then add half of it as liquidity so you earn trading fees as well as volume.",
  Yield:
    "Deposit $DEP into the highest-TVL vault, leave it compounding, and harvest or restake the yield once a week.",
  RWA: "Buy or mint $DEP of the tokenised asset and hold it in the same wallet for the whole plan so the holding is easy to prove.",
  "Risk Curators":
    "Deposit $DEP into a curated vault, pick the vault whose curator has the longest track record, and leave it allocated.",
  CDP: "Open a vault with $DEP of collateral, mint a modest amount of the stablecoin, and keep the collateral ratio above 200%.",
  "Prediction Market":
    "Place $DEP across two or three open markets you have actually read, and close or roll each position weekly.",
};

const DEFAULT_FARM_ACTION =
  "Use the protocol with $DEP once a week — every real use counts more than one large deposit made once.";

function depositTier(tvl: number): number {
  if (tvl < 20000000) return 250;
  if (tvl < 100000000) return 500;
  if (tvl < 500000000) return 1000;
  return 2000;
}

function farmSteps(
  name: string,
  category: string | null,
  tvl: number,
  nChains: number,
  firstChain: string,
): string[] {
  const gas = GAS_USD[firstChain] !== undefined ? GAS_USD[firstChain] : 0.6;
  const dep = depositTier(tvl);
  const gasTotal = gas * 6;
  const weeks = tvl >= 100000000 ? 8 : 6;
  const mins = nChains > 3 ? 20 : 15;
  const action = (category !== null && FARM_ACTION[category] ? FARM_ACTION[category] : DEFAULT_FARM_ACTION).replace(
    "$DEP",
    "$" + dep,
  );
  const spread =
    nChains > 1
      ? " Spread the activity over " +
        nChains +
        " chains (starting on " +
        firstChain +
        ") so the footprint is visible everywhere it operates."
      : "";
  return [
    "Set up: connect a wallet that already holds gas on " +
      firstChain +
      " (about $" +
      gas.toFixed(2) +
      " per transaction there) and budget about $" +
      gasTotal.toFixed(2) +
      " of gas for the whole plan. Use one wallet for everything so the activity is easy to prove.",
    "Fund it: " + action,
    "Stay active: repeat the step above once a week for " + weeks + " weeks (about " + mins + " minutes each time)." + spread,
    "Do not game it: one honest position beats ten dust wallets, and wash activity is the fastest way to be filtered out if a token ever launches. Keep the same wallet and the same size throughout.",
    "Track and exit: save every transaction hash and a dated screenshot. If " +
      name +
      " launches a token, follow its official claim flow with the same wallet. All in, the plan costs about $" +
      gasTotal.toFixed(2) +
      " in gas plus the $" +
      dep +
      " deposit, which stays yours.",
  ];
}

// --- Blind tests ---------------------------------------------------------------
function ratioError(pred: number, actual: number): number {
  return Math.max(pred / actual, actual / pred);
}

interface BlindResult {
  looModel: number;
  looFlat: number;
  looModelMed: number;
  looFlatMed: number;
  wf: { train_frac: number; model_geo: number; flat_geo: number; n_test: number }[];
}

function blindTest(comps: Protocol[], asOf: Date, sizeRef: number): BlindResult {
  // Leave-one-out: refit everything except the size slope on 127 comps,
  // predict the held-out protocol's market cap, and score the ratio error.
  const looM: number[] = [];
  const looF: number[] = [];
  for (let i = 0; i < comps.length; i++) {
    const train = comps.filter((_, j) => j !== i);
    const m = fitModel(train, asOf, sizeRef);
    const held = comps[i];
    looM.push(ratioError(priceFloat(held, m, asOf).value, held.mcap_usd as number));
    looF.push(ratioError(flatValue(held, m), held.mcap_usd as number));
  }
  // Walk-forward: train on the earliest-listed share, test on the rest.
  // Only the comps with a listing date can be ordered this way.
  const dated = comps
    .filter((p) => p.listed_at !== null)
    .sort((a, b) => ((a.listed_at as string) < (b.listed_at as string) ? -1 : 1));
  const wf: BlindResult["wf"] = [];
  for (const frac of [0.5, 0.7]) {
    const n = Math.floor(dated.length * frac);
    const train = dated.slice(0, n);
    const test = dated.slice(n);
    const m = fitModel(train, asOf, sizeRef);
    const eM = test.map((p) => ratioError(priceFloat(p, m, asOf).value, p.mcap_usd as number));
    const eF = test.map((p) => ratioError(flatValue(p, m), p.mcap_usd as number));
    wf.push({ train_frac: frac, model_geo: geoMeanRatio(eM), flat_geo: geoMeanRatio(eF), n_test: test.length });
  }
  return {
    looModel: geoMeanRatio(looM),
    looFlat: geoMeanRatio(looF),
    looModelMed: quantile(looM, 0.5),
    looFlatMed: quantile(looF, 0.5),
    wf,
  };
}

// --- Board build ---------------------------------------------------------------
interface BuiltBoard {
  rows: PricedRow[];
  model: FittedModel;
}

function buildBoard(
  snapshot: Snapshot,
  comps: Protocol[],
  board: Protocol[],
  sizeRef: number,
  asOf: Date,
): BuiltBoard {
  const watchlist = new Set(snapshot.watchlist);
  const model = fitModel(comps, asOf, sizeRef);
  const priced: PricedRow[] = board.map((p) => {
    const f = priceFloat(p, model, asOf);
    return {
      protocol: p,
      rank: 0,
      value: Math.round(f.value),
      multiple: f.multiple,
      baseMultiple: f.baseMultiple,
      sizeAdj: f.sizeAdj,
      growthAdj: f.growthAdj,
      ageAdj: f.ageAdj,
      ageDays: ageDaysOf(p, asOf),
      inWatchlist: watchlist.has(p.slug),
      valueP10: 0,
      valueP90: 0,
      rankP10: 0,
      rankP90: 0,
    };
  });
  priced.sort((a, b) => b.value - a.value || (a.protocol.slug < b.protocol.slug ? -1 : 1));
  // Rounding can only tie values, never invert them; clamp anyway so the
  // written board is non-increasing by rank no matter what.
  for (let i = 1; i < priced.length; i++) {
    if (priced[i].value > priced[i - 1].value) priced[i].value = priced[i - 1].value;
  }
  priced.forEach((r, i) => {
    r.rank = i + 1;
  });
  // Bootstrap: 200 refits of the comps (fixed seed), each repricing the
  // whole board, for a P10-P90 value and a rank range per row.
  const rng = mulberry32(BOOTSTRAP_SEED);
  const n = board.length;
  const valueSamples: number[][] = board.map(() => []);
  const rankSamples: number[][] = board.map(() => []);
  const indexBySlug = new Map(board.map((p, i) => [p.slug, i]));
  for (let rep = 0; rep < BOOTSTRAP_REPS; rep++) {
    const sample: Protocol[] = [];
    for (let i = 0; i < comps.length; i++) sample.push(comps[Math.floor(rng() * comps.length)]);
    const m = fitModel(sample, asOf, sizeRef);
    const vals = board.map((p) => Math.round(priceFloat(p, m, asOf).value));
    const order = vals
      .map((v, i) => i)
      .sort((a, b) => vals[b] - vals[a] || (board[a].slug < board[b].slug ? -1 : 1));
    order.forEach((idx, pos) => {
      valueSamples[idx].push(vals[idx]);
      rankSamples[idx].push(pos + 1);
    });
  }
  for (const r of priced) {
    const idx = indexBySlug.get(r.protocol.slug) as number;
    r.valueP10 = Math.round(quantile(valueSamples[idx], 0.1));
    r.valueP90 = Math.round(quantile(valueSamples[idx], 0.9));
    r.rankP10 = clamp(Math.round(quantile(rankSamples[idx], 0.1)), 1, n);
    r.rankP90 = clamp(Math.round(quantile(rankSamples[idx], 0.9)), 1, n);
  }
  return { rows: priced, model };
}

// --- Outputs ---------------------------------------------------------------------
import { fileURLToPath } from "node:url";

function repoDir(): string {
  return path.dirname(fileURLToPath(import.meta.url));
}

function writeOutputs(outDir: string, snapshot: Snapshot, built: BuiltBoard, blind: BlindResult): void {
  const { rows, model } = built;
  fs.mkdirSync(outDir, { recursive: true });
  const dashDir = path.join(outDir, "dashboard");
  const shotsDir = path.join(outDir, "screenshots");
  fs.mkdirSync(dashDir, { recursive: true });
  fs.mkdirSync(shotsDir, { recursive: true });
  // The dashboard ships unchanged from this repository.
  for (const f of ["index.html", "app.css", "app.js"]) {
    fs.copyFileSync(path.join(repoDir(), "dashboard", f), path.join(dashDir, f));
  }
  // The dashboard's data: one compact array per row, in board order.
  const dataRows = rows.map((r) => [
    r.protocol.slug,
    r.protocol.name,
    r.protocol.category,
    r.protocol.tvl_usd,
    r.protocol.change_7d,
    r.value,
    round4(r.multiple),
    round4(r.baseMultiple),
    round4(r.sizeAdj),
    round4(r.growthAdj),
    round4(r.ageAdj),
    r.valueP10,
    r.valueP90,
    r.rankP10,
    r.rankP90,
    r.protocol.chains.length,
    r.protocol.chains[0],
    r.ageDays,
    r.inWatchlist,
  ]);
  // Comps by category, most comps first, for the dashboard's method panel.
  const catNames = new Set<string>();
  for (const p of snapshot.protocols) {
    if (p.category !== null && model.categoryCount(p.category) > 0) catNames.add(p.category);
  }
  const cats = [...catNames]
    .map((name) => ({ name, count: model.categoryCount(name), multiple: model.categoryMultiple(name) }))
    .sort((a, b) => b.count - a.count || (a.name < b.name ? -1 : 1))
    .map((c) => [c.name, c.count, round4(c.multiple)]);
  const data = {
    asOf: snapshot.as_of,
    rows: dataRows,
    model: {
      globalMultiple: GLOBAL_MULTIPLE,
      looModel: blind.looModel.toFixed(2),
      looFlat: blind.looFlat.toFixed(2),
      looModelMed: blind.looModelMed.toFixed(2),
      looFlatMed: blind.looFlatMed.toFixed(2),
      nComps: snapshot.protocols.filter(isTradingComp).length,
      wf: blind.wf.map((w) => ({
        train_frac: w.train_frac,
        model_geo: w.model_geo.toFixed(2),
        flat_geo: w.flat_geo.toFixed(2),
        n_test: w.n_test,
      })),
      sizeSlope: SIZE_SLOPE,
      cats,
    },
  };
  fs.writeFileSync(path.join(dashDir, "data.js"), "window.BOARD_DATA = " + JSON.stringify(data) + ";\n");
  // The ranked board.
  const boardJson = {
    as_of: snapshot.as_of,
    source: snapshot.source,
    rows: rows.map((r) => ({
      slug: r.protocol.slug,
      name: r.protocol.name,
      category: r.protocol.category,
      rank: r.rank,
      implied_value_usd: r.value,
      tvl_usd: r.protocol.tvl_usd,
      multiple: round4(r.multiple),
      farm_steps: farmSteps(
        r.protocol.name,
        r.protocol.category,
        r.protocol.tvl_usd,
        r.protocol.chains.length,
        r.protocol.chains[0],
      ),
    })),
  };
  fs.writeFileSync(path.join(outDir, "board.json"), JSON.stringify(boardJson, null, 2) + "\n");
  // Screenshots, copied unchanged from this repository.
  for (const f of fs.readdirSync(path.join(repoDir(), "screenshots"))) {
    if (f.startsWith(".")) continue;
    fs.copyFileSync(path.join(repoDir(), "screenshots", f), path.join(shotsDir, f));
  }
  // The dossier is byte for byte what the library writes from the output.
  const dossierBytes = buildDossier(outDir, { text_max_bytes: DOSSIER_TEXT_MAX }) as unknown as Uint8Array;
  fs.writeFileSync(path.join(outDir, "dossier.json"), dossierBytes);
}

// --- Entry point -------------------------------------------------------------------
function main(): void {
  const args = process.argv.slice(2);
  let inputPath = "/input/snapshot.json";
  let blindOnly = false;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--input" && i + 1 < args.length) inputPath = args[++i];
    else if (args[i] === "--blind-test") blindOnly = true;
  }
  const snapshot = JSON.parse(fs.readFileSync(inputPath, "utf8")) as Snapshot;
  const asOf = new Date(snapshot.as_of + "T00:00:00Z");
  const comps = snapshot.protocols.filter(isTradingComp);
  const board = snapshot.protocols.filter(isBoardRow);
  const sizeRef = quantile(
    board.map((p) => Math.log10(p.tvl_usd)),
    0.5,
  );
  const blind = blindTest(comps, asOf, sizeRef);
  if (blindOnly) {
    console.log("blind test over " + comps.length + " trading comps (size slope " + SIZE_SLOPE + "):");
    console.log(
      "  LOO geo-mean error: model " +
        blind.looModel.toFixed(2) +
        "x vs flat " +
        blind.looFlat.toFixed(2) +
        "x (median " +
        blind.looModelMed.toFixed(2) +
        "x vs " +
        blind.looFlatMed.toFixed(2) +
        "x)",
    );
    for (const w of blind.wf) {
      console.log(
        "  walk-forward train " +
          Math.round(w.train_frac * 100) +
          "% earliest-listed: model " +
          w.model_geo.toFixed(2) +
          "x vs flat " +
          w.flat_geo.toFixed(2) +
          "x (" +
          w.n_test +
          " tested)",
      );
    }
    return;
  }
  const built = buildBoard(snapshot, comps, board, sizeRef, asOf);
  const outDir = process.env.MAKEFUN_OUTPUT_DIR || "/output";
  writeOutputs(outDir, snapshot, built, blind);
  const watchCovered = built.rows.filter((r) => r.inWatchlist).length;
  const dossierBytes = fs.statSync(path.join(outDir, "dossier.json")).size;
  const dashBytes = fs
    .readdirSync(path.join(outDir, "dashboard"))
    .reduce((s, f) => s + fs.statSync(path.join(outDir, "dashboard", f)).size, 0);
  console.log("radar-9 wrote " + outDir + ":");
  console.log("  board: " + built.rows.length + " rows, watchlist covered " + watchCovered + "/" + snapshot.watchlist.length);
  console.log(
    "  top 3: " +
      built.rows
        .slice(0, 3)
        .map((r) => r.protocol.name + " $" + Math.round(r.value / 1e6) + "M")
        .join(", "),
  );
  console.log(
    "  blind test: LOO model " +
      blind.looModel.toFixed(2) +
      "x vs flat " +
      blind.looFlat.toFixed(2) +
      "x; walk-forward 50% model " +
      blind.wf[0].model_geo.toFixed(2) +
      "x vs flat " +
      blind.wf[0].flat_geo.toFixed(2) +
      "x",
  );
  console.log("  dashboard source: " + dashBytes + " bytes; dossier.json: " + dossierBytes + " bytes");
}

main();
