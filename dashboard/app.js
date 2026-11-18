/* Launch Radar — dashboard logic. Data comes from data.js (window.BOARD_DATA).
   Row layout: [slug, name, category, tvl, change7d, value, multiple,
   multipleBase, sizeAdj, growthAdj, ageAdj, valueP10, valueP90,
   rankP10, rankP90, nChains, firstChain, ageDays, inWatchlist] */
(function () {
  var D = window.BOARD_DATA;
  var rows = D.rows.map(function (r, i) { return { r: r, rank: i + 1 }; });
  var M = D.model;

  function money(n) {
    if (n >= 1e9) return "$" + (n / 1e9).toFixed(2) + "B";
    if (n >= 1e6) return "$" + (n / 1e6).toFixed(1) + "M";
    if (n >= 1e3) return "$" + (n / 1e3).toFixed(0) + "K";
    return "$" + n;
  }
  function moneyFull(n) { return "$" + Math.round(n).toLocaleString("en-US"); }
  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }
  function fmtGrowth(g) {
    if (g === null || g === undefined) return "—";
    var cls = g >= 0 ? "up" : "down";
    return '<span class="' + cls + '">' + (g >= 0 ? "+" : "") + g.toFixed(1) + "%</span>";
  }

  /* Header chips */
  document.getElementById("chips").innerHTML =
    '<span class="chip">Snapshot <b>' + esc(D.asOf) + '</b></span>' +
    '<span class="chip"><b>' + rows.length + '</b> pre-token protocols</span>' +
    '<span class="chip">Global multiple <b>' + M.globalMultiple.toFixed(4) + "×</b></span>" +
    '<span class="chip">LOO <b class="good">' + M.looModel + "×</b> vs flat " + M.looFlat + "×</span>";

  /* Blind-test panel */
  var wf = M.wf.map(function (w) {
    return "train " + Math.round(w.train_frac * 100) + "% earliest-listed → " +
      "model <b>" + w.model_geo + "×</b> vs flat " + w.flat_geo + "× (" + w.n_test + " tested)";
  }).join(" · ");
  document.getElementById("tests").innerHTML =
    '<div class="test">Leave-one-out over <b>' + M.nComps + '</b> token-bearing comps: ' +
    'model <b class="good">' + M.looModel + '×</b> vs a flat global multiple ' +
    M.looFlat + '× (median ' + M.looModelMed + '× vs ' + M.looFlatMed + '×)</div>' +
    '<div class="test">Walk-forward by listing date: ' + wf + "</div>" +
    '<div class="test">Size slope <b>' + M.sizeSlope + '</b> per log10 TVL (half strength, Huber) — ' +
    "a protocol 10× the average size keeps about 65% of its multiple</div>";

  /* Category comps table */
  document.querySelector("#cats tbody").innerHTML = D.model.cats.map(function (c) {
    return "<tr><td>" + esc(c[0]) + "</td><td class='num'>" + c[1] +
      "</td><td class='num'>" + c[2].toFixed(4) + "×</td></tr>";
  }).join("");

  /* Category filter options */
  var cats = [];
  rows.forEach(function (x) { if (cats.indexOf(x.r[2]) < 0) cats.push(x.r[2]); });
  cats.sort();
  document.getElementById("cat").innerHTML =
    '<option value="">All categories</option>' +
    cats.map(function (c) { return '<option value="' + esc(c) + '">' + esc(c) + "</option>"; }).join("");

  /* Farm plan generator — the same rules as the board builder. */
  var GAS = { Ethereum: 3.50, Arbitrum: 0.25, Optimism: 0.25, Base: 0.25,
    Solana: 0.02, BSC: 0.20, "BNB Chain": 0.20, Polygon: 0.05, Avalanche: 0.30,
    "zkSync Era": 0.30, Linea: 0.15, Scroll: 0.20, Blast: 0.10, Mode: 0.10,
    Mantle: 0.10, Fantom: 0.10, Gnosis: 0.10, Celo: 0.05, Sui: 0.02,
    Aptos: 0.02, Tron: 0.30, Cronos: 0.10, Kava: 0.10, Sei: 0.02 };
  var ACTION = {
    Lending: "Supply $DEP of a major asset as collateral, then borrow a small amount " +
      "(under 30% of the collateral) so the position stays healthy.",
    Dexs: "Swap $DEP through the pool in two trades a week apart, then add half of it " +
      "as liquidity so you earn trading fees as well as volume.",
    Yield: "Deposit $DEP into the highest-TVL vault, leave it compounding, and harvest " +
      "or restake the yield once a week.",
    RWA: "Buy or mint $DEP of the tokenised asset and hold it in the same wallet for " +
      "the whole plan so the holding is easy to prove.",
    "Risk Curators": "Deposit $DEP into a curated vault, pick the vault whose curator has " +
      "the longest track record, and leave it allocated.",
    CDP: "Open a vault with $DEP of collateral, mint a modest amount of the stablecoin, " +
      "and keep the collateral ratio above 200%.",
    "Prediction Market": "Place $DEP across two or three open markets you have actually read, " +
      "and close or roll each position weekly."
  };
  var DEFAULT_ACTION = "Use the protocol with $DEP once a week — every real use counts " +
    "more than one large deposit made once.";
  function depositTier(tvl) {
    if (tvl < 20000000) return 250;
    if (tvl < 100000000) return 500;
    if (tvl < 500000000) return 1000;
    return 2000;
  }
  function farmSteps(row) {
    var name = row[1], chain = row[16], tvl = row[3];
    var gas = (GAS[chain] !== undefined) ? GAS[chain] : 0.60;
    var dep = depositTier(tvl);
    var gasTotal = gas * 6;
    var weeks = tvl >= 100000000 ? 8 : 6;
    var mins = row[15] > 3 ? 20 : 15;
    var action = (ACTION[row[2]] || DEFAULT_ACTION).replace("$DEP", "$" + dep);
    var spread = "";
    if (row[15] > 1) {
      spread = " Spread the activity over " + row[15] + " chains (starting on " + chain +
        ") so the footprint is visible everywhere it operates.";
    }
    return [
      "Set up: connect a wallet that already holds gas on " + chain +
        " (about $" + gas.toFixed(2) + " per transaction there) and budget about $" +
        gasTotal.toFixed(2) + " of gas for the whole plan. Use one wallet for " +
        "everything so the activity is easy to prove.",
      "Fund it: " + action,
      "Stay active: repeat the step above once a week for " + weeks + " weeks (about " +
        mins + " minutes each time)." + spread,
      "Do not game it: one honest position beats ten dust wallets, and wash activity " +
        "is the fastest way to be filtered out if a token ever launches. Keep the same " +
        "wallet and the same size throughout.",
      "Track and exit: save every transaction hash and a dated screenshot. If " + name +
        " launches a token, follow its official claim flow with the same wallet. All in, " +
        "the plan costs about $" + gasTotal.toFixed(2) + " in gas plus the $" + dep +
        " deposit, which stays yours."
    ];
  }

  /* Board rendering */
  var tbody = document.querySelector("#board tbody");
  var state = { q: "", cat: "", sort: "value", watch: false, sel: null };
  function filtered() {
    var list = rows.filter(function (x) {
      var r = x.r;
      if (state.watch && !r[18]) return false;
      if (state.cat && r[2] !== state.cat) return false;
      if (state.q) {
        var q = state.q.toLowerCase();
        if (r[0].toLowerCase().indexOf(q) < 0 && r[1].toLowerCase().indexOf(q) < 0) return false;
      }
      return true;
    });
    var key = { value: 5, tvl: 3, mult: 6, growth: 4 }[state.sort];
    list.sort(function (a, b) {
      var va = a.r[key], vb = b.r[key];
      if (va === null) return 1;
      if (vb === null) return -1;
      return vb - va;
    });
    return list;
  }
  function render() {
    var list = filtered();
    tbody.innerHTML = list.slice(0, 60).map(function (x) {
      var r = x.r;
      return '<tr data-slug="' + esc(r[0]) + '"' + (state.sel === r[0] ? ' class="sel"' : "") + ">" +
        '<td class="rank">' + x.rank + "</td>" +
        '<td class="name">' + esc(r[1]) + (r[18] ? " ★" : "") + "</td>" +
        '<td class="cat">' + esc(r[2]) + "</td>" +
        '<td class="num">' + money(r[3]) + "</td>" +
        "<td class='num'>" + fmtGrowth(r[4]) + "</td>" +
        '<td class="num">' + r[6].toFixed(3) + "×</td>" +
        '<td class="num"><b>' + money(r[5]) + "</b></td>" +
        '<td class="num">' + money(r[11]) + "–" + money(r[12]) + "</td>" +
        '<td class="num">#' + r[13] + "–" + r[14] + "</td></tr>";
    }).join("");
    document.getElementById("count").textContent =
      list.length + " protocols shown (top 60 of the filtered set) · ★ = snapshot watchlist";
  }
  function barRow(label, val, scale) {
    var pct = Math.min(100, Math.abs(val) / scale * 100);
    var cls = val < 0 ? "fill neg" : "fill";
    var shown = (val >= 0 ? "+" : "") + val.toFixed(3);
    return '<div class="brow"><span>' + esc(label) + '</span><span class="bar"><span class="' +
      cls + '" style="display:block;width:' + pct.toFixed(1) + '%"></span></span>' +
      '<span class="v">' + shown + "</span></div>";
  }
  function showDetail(slug) {
    var x = null;
    rows.forEach(function (y) { if (y.r[0] === slug) x = y; });
    if (!x) return;
    var r = x.r;
    state.sel = slug;
    document.getElementById("d-name").textContent = "#" + x.rank + " " + r[1];
    document.getElementById("d-sub").textContent =
      r[2] + " · TVL " + moneyFull(r[3]) + " · " + r[15] + " chains · first seen " +
      (r[17] === null ? "date unknown" : r[17] + " days ago") +
      " · implied value " + moneyFull(r[5]) + " (P10–P90 " + moneyFull(r[11]) +
      "–" + moneyFull(r[12]) + ", rank #" + r[13] + "–" + r[14] + " over 200 bootstrap refits)";
    document.getElementById("d-break").innerHTML =
      '<div class="brow"><span>Category multiple</span><span class="bar"><span class="fill" style="display:block;width:' +
      Math.min(100, r[7] / 0.8 * 100).toFixed(1) + '%"></span></span>' +
      '<span class="v">' + r[7].toFixed(3) + "×</span></div>" +
      barRow("Size adjustment", r[8], 1.0) +
      barRow("Growth adjustment", r[9], 0.15) +
      barRow("Age adjustment", r[10], 0.15) +
      '<div class="brow"><span><b>Multiple used</b></span><span></span>' +
      '<span class="v"><b>' + r[6].toFixed(3) + "×</b></span></div>";
    document.getElementById("d-steps").innerHTML = farmSteps(r).map(function (s) {
      return "<li>" + esc(s) + "</li>";
    }).join("");
    render();
    document.getElementById("detail").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
  tbody.addEventListener("click", function (ev) {
    var tr = ev.target.closest("tr[data-slug]");
    if (tr) showDetail(tr.getAttribute("data-slug"));
  });
  document.getElementById("q").addEventListener("input", function (ev) {
    state.q = ev.target.value; render();
  });
  document.getElementById("cat").addEventListener("change", function (ev) {
    state.cat = ev.target.value; render();
  });
  document.getElementById("sort").addEventListener("change", function (ev) {
    state.sort = ev.target.value; render();
  });
  document.getElementById("watch").addEventListener("change", function (ev) {
    state.watch = ev.target.checked; render();
  });
  window.addEventListener("hashchange", route);
  function route() {
    var h = location.hash || "";
    var m = h.match(/#detail=([\w-]+)/);
    if (m) { showDetail(m[1]); return; }
    m = h.match(/#filter=(.+)/);
    if (m) {
      state.cat = decodeURIComponent(m[1]);
      document.getElementById("cat").value = state.cat;
    }
    render();
  }
  route();
  if (!location.hash) showDetail(rows[0].r[0]);
  window.__farmSteps = farmSteps;
})();
