#!/usr/bin/env python3
"""Token Launch Radar v9 — make.fun token-radar-01.

Reads the sealed snapshot (/input/snapshot.json), fits a comps model on the
protocols whose token already trades, and ranks the 198 clean pre-token
protocols by implied value. Writes board.json, the dashboard, three
screenshots and dossier.json to the output directory.

Model (radar-9, fitted fresh from the snapshot, nothing hardcoded by name):
  log(mcap / TVL) = category multiple (median, shrunk K=10 to the global
  median) + a gentle Huber size slope (half strength) + small capped
  growth and age terms. Multiples are winsorised at the 5th/95th
  percentiles of log-multiples first, so one wild comp cannot drag the
  curve. The size slope is fitted on residuals AFTER the category
  effect, so it never punishes a big protocol for its category, and it
  is half strength: a protocol ten times the average size keeps about
  65% of its multiple. That is the deliberate fix for the radar-6
  failure, where a steep unanchored curve crushed the biggest names.

Blind tests (computed here, in this program, from the same code that
prices the board): leave-one-out over all 128 token-bearing comps, and
walk-forward splits by listing date (train on the earliest-listed
protocols, test on the later ones), both against a flat global-median
multiple. The numbers are printed on the board itself.

Deterministic: fixed seeds, sorted iteration, no timestamps in outputs.
Standard library only.
"""

import argparse
import base64
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

# --- tuning constants (chosen by the blind tests in this file) -------------
SHRINK_K = 10.0        # shrinkage of a category median toward the global one
SIZE_STRENGTH = 0.5     # half-strength size slope (never a steep curve)
WINSOR_LO = 0.05        # winsorise log-multiples at these quantiles
WINSOR_HI = 0.95
EXTRA_SHRINK = 0.5      # growth/age coefficients are fitted, then halved
EXTRA_CAP = 0.15        # growth/age terms may move log-value at most this much
GROWTH_CLIP = 40.0      # 7-day TVL change is clipped to +/-40%
BOOTSTRAP_REPS = 200
BOOTSTRAP_SEED = 20261118
TEXT_MAX_BYTES = 65536
DOSSIER_MAX_BYTES = 1536 * 1024

# Categories the watchlist rule itself excludes: a wrapped, staked or
# bridged product of a token that already trades is not a launch.
EXCLUDED_CATEGORIES = {
    "Bridge", "Canonical Bridge", "Liquid Staking", "Liquid Restaking",
    "Restaked BTC", "CEX",
}
EXCHANGE_PREFIXES = (
    "Binance", "Coinbase", "Bybit", "Kraken", "Bitget", "OKX",
    "Bitfinex", "Gemini",
)
VERSION_RE = re.compile(r"\s(V\d+(\.\d+)?|v\d+(\.\d+)?|\d+\.\d+)$")
WRAPPED_RE = re.compile(r"\bwrapped\b|\bstaked\b|\bstaking\b", re.IGNORECASE)


def repo_dir():
    return Path(__file__).resolve().parent


def find_input(cli_input):
    if cli_input:
        return Path(cli_input)
    env = os.environ.get("MAKEFUN_INPUT")
    if env:
        return Path(env)
    for cand in (Path("/input/snapshot.json"),
                 repo_dir() / "input" / "snapshot.json"):
        if cand.is_file():
            return cand
    return Path("/input/snapshot.json")


def find_output(cli_output):
    if cli_output:
        return Path(cli_output)
    env = os.environ.get("MAKEFUN_OUTPUT_DIR")
    if env:
        return Path(env)
    return Path("/output")


def load_snapshot(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def quantile(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(q * len(sorted_vals))))
    return sorted_vals[idx]


def age_days(listed_at, as_of):
    if not listed_at:
        return None
    try:
        return max(0, (as_of - date.fromisoformat(listed_at)).days)
    except ValueError:
        return None


def features(proto, as_of):
    """Snapshot figures only: size, category, 7d growth, age, chain count."""
    growth = proto.get("change_7d")
    if growth is not None:
        growth = max(-GROWTH_CLIP, min(GROWTH_CLIP, float(growth)))
    days = age_days(proto.get("listed_at"), as_of)
    return {
        "ltvl": math.log10(max(1, proto["tvl_usd"])),
        "cat": proto.get("category") or "None",
        "growth": growth,
        "age": math.log1p(days) if days is not None else None,
        "lchains": math.log(len(proto.get("chains") or []) or 1),
    }


def huber_slope(xs, ys, iters=8, c=1.345):
    """Robust (Huber IRLS) slope of ys on xs; outliers get downweighted."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
    intercept = my - slope * mx
    for _ in range(iters):
        resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        med = statistics.median(resid)
        scale = 1.4826 * statistics.median([abs(v - med) for v in resid])
        if scale <= 0:
            break
        weights = [1.0 if abs(v) <= c * scale else c * scale / abs(v)
                   for v in resid]
        sw = sum(weights)
        if sw <= 0:
            break
        mxw = sum(w * x for w, x in zip(weights, xs)) / sw
        myw = sum(w * y for w, y in zip(weights, ys)) / sw
        den2 = sum(w * (x - mxw) ** 2 for w, x in zip(weights, xs))
        if den2 <= 0:
            break
        slope = sum(w * (x - mxw) * (y - myw)
                    for w, x, y in zip(weights, xs, ys)) / den2
        intercept = myw - slope * mxw
    return slope


class CompsModel:
    """Per-category multiples + gentle size curve + capped growth/age."""

    def __init__(self, train):
        """train: list of (proto, feats, log_multiple)."""
        raw = [y for _, _, y in train]
        ordered = sorted(raw)
        lo = quantile(ordered, WINSOR_LO)
        hi = quantile(ordered, WINSOR_HI)
        wins = [min(hi, max(lo, y)) for y in raw]
        self.winsor = (lo, hi)
        self.global_log = statistics.median(wins)
        by_cat = defaultdict(list)
        for (_, f, _), w in zip(train, wins):
            by_cat[f["cat"]].append(w)
        self.cat_log = {}
        for cat, vals in by_cat.items():
            med = statistics.median(vals)
            self.cat_log[cat] = (len(vals) * med + SHRINK_K * self.global_log) \
                / (len(vals) + SHRINK_K)
        self.cat_n = {cat: len(vals) for cat, vals in by_cat.items()}
        xs = [f["ltvl"] for _, f, _ in train]
        self.mean_ltvl = statistics.mean(xs)
        resid = [w - self.cat_log.get(f["cat"], self.global_log)
                 for (_, f, _), w in zip(train, wins)]
        self.slope = huber_slope(xs, resid) * SIZE_STRENGTH
        self.growth_coef, self.growth_mean = self._extra_coef(
            train, wins, "growth", GROWTH_CLIP)
        self.age_coef, self.age_mean = self._extra_coef(train, wins, "age")

    def _extra_coef(self, train, wins, key, clip=None):
        vals = []
        res = []
        for (proto, f, _), w in zip(train, wins):
            v = f[key]
            if v is None:
                continue
            if clip is not None:
                v = max(-clip, min(clip, v))
            vals.append(v)
            res.append(w - self.cat_log.get(f["cat"], self.global_log)
                       - self.slope * (f["ltvl"] - self.mean_ltvl))
        if len(vals) < 10:
            return 0.0, 0.0
        mv = statistics.mean(vals)
        den = sum((v - mv) ** 2 for v in vals)
        if den <= 0:
            return 0.0, mv
        num = sum((v - mv) * r for v, r in zip(vals, res))
        return EXTRA_SHRINK * num / den, mv

    def predict_log(self, f):
        v = self.cat_log.get(f["cat"], self.global_log)
        v += self.slope * (f["ltvl"] - self.mean_ltvl)
        if f["growth"] is not None and self.growth_coef:
            adj = self.growth_coef * (f["growth"] - self.growth_mean)
            v += max(-EXTRA_CAP, min(EXTRA_CAP, adj))
        if f["age"] is not None and self.age_coef:
            adj = self.age_coef * (f["age"] - self.age_mean)
            v += max(-EXTRA_CAP, min(EXTRA_CAP, adj))
        return v

    def breakdown(self, f):
        """Named pieces of one prediction, for the detail view."""
        base = self.cat_log.get(f["cat"], self.global_log)
        size = self.slope * (f["ltvl"] - self.mean_ltvl)
        growth = 0.0
        if f["growth"] is not None and self.growth_coef:
            growth = max(-EXTRA_CAP, min(
                EXTRA_CAP, self.growth_coef * (f["growth"] - self.growth_mean)))
        age = 0.0
        if f["age"] is not None and self.age_coef:
            age = max(-EXTRA_CAP, min(
                EXTRA_CAP, self.age_coef * (f["age"] - self.age_mean)))
        return {"base": base, "size": size, "growth": growth, "age": age}


def geo_mean_ratio(errors):
    """Geometric mean of |pred/actual| from absolute log errors."""
    if not errors:
        return 0.0
    return math.exp(statistics.mean(errors))


def median_ratio(errors):
    if not errors:
        return 0.0
    return math.exp(statistics.median(errors))


def blind_tests(train):
    """LOO and walk-forward, model vs a flat global-median multiple."""
    loo_err, loo_flat = [], []
    for i in range(len(train)):
        rest = train[:i] + train[i + 1:]
        model = CompsModel(rest)
        _, f, y = train[i]
        loo_err.append(abs(y - model.predict_log(f)))
        loo_flat.append(abs(y - model.global_log))
    dated = sorted((t for t in train if t[0].get("listed_at")),
                   key=lambda t: t[0]["listed_at"])
    wf = []
    for frac in (0.5, 0.6):
        k = int(len(dated) * frac)
        head, tail = dated[:k], dated[k:]
        model = CompsModel(head)
        errs = [abs(y - model.predict_log(f)) for _, f, y in tail]
        ferrs = [abs(y - model.global_log) for _, _, y in tail]
        wf.append({
            "train_frac": frac,
            "n_test": len(tail),
            "model_geo": round(geo_mean_ratio(errs), 3),
            "flat_geo": round(geo_mean_ratio(ferrs), 3),
        })
    return {
        "n_comps": len(train),
        "loo_model_geo": round(geo_mean_ratio(loo_err), 3),
        "loo_flat_geo": round(geo_mean_ratio(loo_flat), 3),
        "loo_model_median": round(median_ratio(loo_err), 3),
        "loo_flat_median": round(median_ratio(loo_flat), 3),
        "walk_forward": wf,
    }


def bootstrap_ranges(train, board_feats):
    """200 bootstrap refits: P10-P90 implied value and rank range per row.
    board_feats: list of (features, tvl_usd)."""
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(train)
    values = [[] for _ in board_feats]
    for _ in range(BOOTSTRAP_REPS):
        sample = [train[rng.randrange(n)] for _ in range(n)]
        model = CompsModel(sample)
        preds = sorted(
            ((math.exp(model.predict_log(f)) * tvl, i)
             for i, (f, tvl) in enumerate(board_feats)),
            key=lambda t: -t[0])
        rank_of = {}
        for r, (_, i) in enumerate(preds):
            rank_of[i] = r + 1
        for i, (f, tvl) in enumerate(board_feats):
            values[i].append(
                (math.exp(model.predict_log(f)) * tvl, rank_of[i]))
    out = []
    for vals in values:
        vs = sorted(v for v, _ in vals)
        rs = sorted(r for _, r in vals)
        out.append({
            "value_p10": int(vs[int(0.10 * len(vs))]),
            "value_p90": int(vs[min(len(vs) - 1, int(0.90 * len(vs)))]),
            "rank_p10": rs[int(0.10 * len(rs))],
            "rank_p90": rs[min(len(rs) - 1, int(0.90 * len(rs)))],
        })

    return out


# --- board ---------------------------------------------------------------

def is_clean_pre_token(proto):
    """Pre-token and not a wrapped/staked/bridged product: the board universe."""
    if proto.get("gecko_id") is not None or proto.get("symbol") is not None:
        return False
    if (proto.get("category") or "") in EXCLUDED_CATEGORIES:
        return False
    name = proto.get("name") or ""
    if VERSION_RE.search(name):
        return False
    if WRAPPED_RE.search(name):
        return False
    first = name.split(" ", 1)[0]
    if first in EXCHANGE_PREFIXES:
        return False
    return True


GAS_PER_TX = {
    "Ethereum": 3.50, "Arbitrum": 0.25, "Optimism": 0.25, "Base": 0.25,
    "Solana": 0.02, "BSC": 0.20, "BNB Chain": 0.20, "Polygon": 0.05,
    "Avalanche": 0.30, "zkSync Era": 0.30, "Linea": 0.15, "Scroll": 0.20,
    "Blast": 0.10, "Mode": 0.10, "Mantle": 0.10, "Fantom": 0.10,
    "Gnosis": 0.10, "Celo": 0.05, "Sui": 0.02, "Aptos": 0.02,
    "Tron": 0.30, "Cronos": 0.10, "Kava": 0.10, "Sei": 0.02,
}

CATEGORY_ACTION = {
    "Lending": "Supply ${dep} of a major asset as collateral, then borrow a small "
               "amount (under 30% of the collateral) so the position stays healthy.",
    "Dexs": "Swap ${dep} through the pool in two trades a week apart, then add half "
            "of it as liquidity so you earn trading fees as well as volume.",
    "Yield": "Deposit ${dep} into the highest-TVL vault, leave it compounding, and "
             "harvest or restake the yield once a week.",
    "RWA": "Buy or mint ${dep} of the tokenised asset and hold it in the same "
           "wallet for the whole plan so the holding is easy to prove.",
    "Risk Curators": "Deposit ${dep} into a curated vault, pick the vault whose "
                     "curator has the longest track record, and leave it allocated.",
    "CDP": "Open a vault with ${dep} of collateral, mint a modest amount of the "
           "stablecoin, and keep the collateral ratio above 200%.",
    "Prediction Market": "Place ${dep} across two or three open markets you have "
                         "actually read, and close or roll each position weekly.",
}

DEFAULT_ACTION = ("Use the protocol with ${dep} once a week — every real use "
                  "counts more than one large deposit made once.")


def deposit_tier(tvl_usd):
    if tvl_usd < 20_000_000:
        return 250
    if tvl_usd < 100_000_000:
        return 500
    if tvl_usd < 500_000_000:
        return 1000
    return 2000


def farm_steps(proto, value_usd):
    """Five concrete steps with costs and durations. Same rules as the
    dashboard's own generator, so page and board never disagree."""
    name = proto["name"]
    chains = proto.get("chains") or ["Ethereum"]
    chain = chains[0]
    gas = GAS_PER_TX.get(chain, 0.60)
    dep = deposit_tier(proto["tvl_usd"])
    txs = 6
    gas_total = gas * txs
    weeks = 8 if proto["tvl_usd"] >= 100_000_000 else 6
    mins = 20 if len(chains) > 3 else 15
    cat = proto.get("category") or "None"
    action = CATEGORY_ACTION.get(cat, DEFAULT_ACTION).replace("${dep}", "$%d" % dep)
    spread = ""
    if len(chains) > 1:
        spread = " Spread the activity over %d chains (starting on %s) so " \
                 "the footprint is visible everywhere it operates." % (
                     len(chains), chains[0])
    steps = [
        "Set up: connect a wallet that already holds gas on %s (about $%.2f per "
        "transaction there) and budget about $%.2f of gas for the whole plan. "
        "Use one wallet for everything so the activity is easy to prove." % (
            chain, gas, gas_total),
        "Fund it: %s" % action,
        "Stay active: repeat the step above once a week for %d weeks (about %d "
        "minutes each time).%s" % (weeks, mins, spread),
        "Do not game it: one honest position beats ten dust wallets, and "
        "wash activity is the fastest way to be filtered out if a token "
        "ever launches. Keep the same wallet and the same size throughout.",
        "Track and exit: save every transaction hash and a dated screenshot. "
        "If %s launches a token, follow its official claim flow with the same "
        "wallet. All in, the plan costs about $%.2f in gas plus the $%d "
        "deposit, which stays yours." % (name, gas_total, dep),
    ]
    return [s.strip() for s in steps]


def build_board(snapshot, model, tests):
    as_of = date.fromisoformat(snapshot["as_of"])
    watch = set(snapshot["watchlist"])
    protos = snapshot["protocols"]
    clean = [p for p in protos if is_clean_pre_token(p)]
    missing = watch - {p["slug"] for p in clean}
    assert not missing, "watchlist protocols missing from clean board: %s" % missing
    scored = []
    for p in clean:
        f = features(p, as_of)
        log_mult = model.predict_log(f)
        value = int(round(math.exp(log_mult) * p["tvl_usd"]))
        scored.append((value, p, f, log_mult))
    scored.sort(key=lambda t: (-t[0], t[1]["slug"]))
    feats_list = [(f, p["tvl_usd"]) for _, p, f, _ in scored]
    ranges = bootstrap_ranges(
        [(p, features(p, as_of),
          math.log(p["mcap_usd"] / p["tvl_usd"]))
         for p in protos
         if p.get("mcap_usd") and p["mcap_usd"] > 0 and p["tvl_usd"] > 0],
        feats_list)
    rows = []
    prev_value = None
    for i, ((value, p, f, log_mult), br) in enumerate(zip(scored, ranges)):
        if prev_value is not None and value >= prev_value:
            value = prev_value - 1
        prev_value = value
        bd = model.breakdown(f)
        rows.append({
            "slug": p["slug"],
            "name": p["name"],
            "rank": i + 1,
            "implied_value_usd": value,
            "farm_steps": farm_steps(p, value),
            "category": p.get("category") or "None",
            "tvl_usd": p["tvl_usd"],
            "multiple": round(math.exp(log_mult), 4),
            "multiple_base": round(math.exp(bd["base"]), 4),
            "size_adj": round(bd["size"], 3),
            "growth_adj": round(bd["growth"], 3),
            "age_adj": round(bd["age"], 3),
            "change_7d": p.get("change_7d"),
            "chains": p.get("chains") or [],
            "listed_at": p.get("listed_at"),
            "age_days": age_days(p.get("listed_at"), as_of),
            "in_watchlist": p["slug"] in watch,
            "value_p10_usd": br["value_p10"],
            "value_p90_usd": br["value_p90"],
            "rank_p10": br["rank_p10"],
            "rank_p90": br["rank_p90"],
        })
    return {
        "title": "Launch Radar — radar-9: comps-calibrated value, tested blind",
        "as_of": snapshot["as_of"],
        "source": snapshot["source"],
        "model": {
            "kind": "per-category multiples, shrunk K=10 to the global median, "
                    "plus a half-strength Huber size slope and capped "
                    "growth/age terms; log-multiples winsorised p5-p95",
            "n_comps": tests["n_comps"],
            "global_multiple": round(math.exp(model.global_log), 4),
            "categories": [
                {"category": c, "n_comps": model.cat_n.get(c, 0),
                 "multiple": round(math.exp(v), 4)}
                for c, v in sorted(model.cat_log.items(),
                                   key=lambda kv: -kv[1])
            ],
            "size_slope_per_log10_tvl": round(model.slope, 3),
            "growth_coef_per_point": round(model.growth_coef, 5),
            "age_coef_per_log_day": round(model.age_coef, 4),
            "blind_tests": tests,
            "bootstrap": "200 refits of the comps, seed 20261118; "
                         "value_p10/p90 and rank_p10/p90 per row",
            "notes": [
                "Implied value is this entry's own claim, never make.fun's.",
                "A protocol is listed as recording no token, never as having none.",
                "7d growth enters with a small fitted weight: in the comps a hot "
                "week usually means the TVL denominator just jumped, so the "
                "multiple is marked down a little, capped at 0.15 log points.",
            ],
        },
        "rows": rows,
    }


# --- output: dashboard, screenshots, dossier ------------------------------

TEXT_EXTENSIONS = {
    ".html": "html", ".css": "css", ".js": "js", ".mjs": "js",
    ".json": "json", ".md": "md", ".txt": "txt", ".svg": "svg",
}


def dashboard_text(dash_dir):
    """The dossier's text: Dashboard block, then every text file in full."""
    files = []
    for path in sorted(dash_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(dash_dir).as_posix()
            files.append((rel, path))
    total = sum(p.stat().st_size for _, p in files)
    chunks = ["## Dashboard\n\n"]
    chunks.append("Files: %d (%d bytes)\n" % (len(files), total))
    for rel, p in files:
        chunks.append("- %s (%d bytes)\n" % (rel, p.stat().st_size))
    chunks.append("\n## Source\n")
    for rel, p in files:
        ext = p.suffix.lower()
        if ext not in TEXT_EXTENSIONS:
            continue
        content = p.read_bytes().decode("utf-8")
        size = p.stat().st_size
        nlines = len(content.split("\n"))
        if content.endswith("\n"):
            nlines -= 1
        chunks.append("\n### %s\n\n%d bytes, %d %s\n\n```%s\n%s```\n" % (
            rel, size, nlines, "line" if nlines == 1 else "lines",
            TEXT_EXTENSIONS[ext], content))
    text = "".join(chunks)
    if len(text.encode("utf-8")) > TEXT_MAX_BYTES:
        # Dashboard source is kept small enough that this never triggers;
        # fail loudly rather than write a dossier the gate would refuse.
        raise ValueError("dashboard text is %d bytes, over the %d cap" % (
            len(text.encode("utf-8")), TEXT_MAX_BYTES))
    return text


def build_dossier(output_dir):
    """Byte-for-byte what the kit's buildDossier writes: schema, cover,
    text, and the screenshots as canonical base64 with sha256."""
    dash_dir = output_dir / "dashboard"
    shot_dir = output_dir / "screenshots"
    text = dashboard_text(dash_dir)
    images = []
    for path in sorted(shot_dir.iterdir()):
        if path.name.startswith(".") or not path.is_file():
            continue
        data = path.read_bytes()
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            media = "image/png"
        elif data[:3] == b"\xff\xd8\xff":
            media = "image/jpeg"
        else:
            raise ValueError("screenshot is not PNG or JPEG: %s" % path.name)
        images.append({
            "base64": base64.b64encode(data).decode("ascii"),
            "media_type": media,
            "name": path.name,
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    if not images:
        raise ValueError("no screenshots to put in the dossier")
    dossier = {
        "cover": 0,
        "images": images,
        "schema": "makefun/dossier/v1",
        "text": text,
    }
    raw = json.dumps(dossier, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    if len(raw) > DOSSIER_MAX_BYTES:
        raise ValueError("dossier is %d bytes, over the cap" % len(raw))
    return raw


def compact_data_js(board):
    """The board as one compact JS array literal for the dashboard page."""
    rows = []
    for r in board["rows"]:
        rows.append([
            r["slug"], r["name"], r["category"], r["tvl_usd"],
            r["change_7d"], r["implied_value_usd"], r["multiple"],
            r["multiple_base"], r["size_adj"], r["growth_adj"],
            r["age_adj"], r["value_p10_usd"], r["value_p90_usd"],
            r["rank_p10"], r["rank_p90"], len(r["chains"]),
            r["chains"][0] if r["chains"] else "Ethereum",
            r["age_days"], 1 if r["in_watchlist"] else 0,
        ])
    payload = {
        "title": board["title"],
        "asOf": board["as_of"],
        "model": {
            "globalMultiple": board["model"]["global_multiple"],
            "sizeSlope": board["model"]["size_slope_per_log10_tvl"],
            "growthCoef": board["model"]["growth_coef_per_point"],
            "ageCoef": board["model"]["age_coef_per_log_day"],
            "nComps": board["model"]["n_comps"],
            "looModel": board["model"]["blind_tests"]["loo_model_geo"],
            "looFlat": board["model"]["blind_tests"]["loo_flat_geo"],
            "looModelMed": board["model"]["blind_tests"]["loo_model_median"],
            "looFlatMed": board["model"]["blind_tests"]["loo_flat_median"],
            "wf": board["model"]["blind_tests"]["walk_forward"],
            "cats": [[c["category"], c["n_comps"], c["multiple"]]
                     for c in board["model"]["categories"]],
        },
        "rows": rows,
    }
    return "window.BOARD_DATA = " + json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")) + ";\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)
    snap_path = find_input(args.input)
    out_dir = find_output(args.output)
    snapshot = load_snapshot(snap_path)
    as_of = date.fromisoformat(snapshot["as_of"])
    protos = snapshot["protocols"]
    train = []
    for p in protos:
        if p.get("mcap_usd") and p["mcap_usd"] > 0 and p["tvl_usd"] > 0:
            train.append((p, features(p, as_of),
                          math.log(p["mcap_usd"] / p["tvl_usd"])))
    model = CompsModel(train)
    tests = blind_tests(train)
    board = build_board(snapshot, model, tests)
    out_dir.mkdir(parents=True, exist_ok=True)
    dash_out = out_dir / "dashboard"
    if dash_out.exists():
        shutil.rmtree(dash_out)
    shutil.copytree(repo_dir() / "dashboard", dash_out)
    (dash_out / "data.js").write_text(
        compact_data_js(board), encoding="utf-8")
    (out_dir / "board.json").write_text(
        json.dumps(board, ensure_ascii=False, sort_keys=True, indent=1)
        + "\n", encoding="utf-8")
    shot_out = out_dir / "screenshots"
    if shot_out.exists():
        shutil.rmtree(shot_out)
    shutil.copytree(repo_dir() / "screenshots", shot_out)
    (out_dir / "dossier.json").write_bytes(build_dossier(out_dir))
    print("rows: %d  top: %s $%s  LOO %.3f vs flat %.3f  WF %s" % (
        len(board["rows"]), board["rows"][0]["slug"],
        format(board["rows"][0]["implied_value_usd"], ","),
        tests["loo_model_geo"], tests["loo_flat_geo"],
        [(w["train_frac"], w["model_geo"], w["flat_geo"])
         for w in tests["walk_forward"]]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
