/* What each driver has actually earned.

   Two views that must not be confused. The live table is what the
   strategies have done in this account's own record. The bench table is
   how the roster was chosen — each strategy run alone over one fixed
   window, which is a different question and a weaker claim, because the
   choice was made by looking at those same numbers. */

import { loadAll, fmt, sign, initTheme } from "./app.js";
import { renderNav } from "./nav.js";

initTheme();
renderNav();

const el = (id) => document.getElementById(id);

const CLASS_LABEL = {
  crypto_perp: "Crypto futures", crypto_spot: "Crypto spot",
  equity: "Stocks", etf: "ETFs", futures: "Futures", unknown: "Unclassified",
};

/* The solo bench, and then the rosters run together. Kept here rather
   than fetched because these are a record of a decision, not live state:
   they are what the roster was chosen on and should not silently
   change. */
const BENCH = [
  { name: "clenow",     ret: 14.91, n: 39,  r: 0.581,  dd: 1.33, win: 69.2, pf: 5.58, kept: "kept" },
  { name: "turtle",     ret: 10.77, n: 69,  r: 0.255,  dd: 1.29, win: 60.9, pf: 2.09, kept: "kept" },
  { name: "lorentzian", ret: 3.80,  n: 50,  r: -0.002, dd: 4.02, win: 48.0, pf: 1.37, kept: "dropped" },
  { name: "supertrend", ret: -1.17, n: 214, r: -0.070, dd: 5.58, win: 42.1, pf: 1.01, kept: "dropped" },
  { name: "nwenvelope", ret: -3.23, n: 19,  r: -0.308, dd: 3.25, win: 31.6, pf: 0.20, kept: "dropped" },
  { name: "holygrail",  ret: -3.26, n: 55,  r: -0.206, dd: 5.14, win: 30.9, pf: 0.76, kept: "dropped" },
  { name: "smc",        ret: -6.45, n: 162, r: -0.321, dd: 9.40, win: 30.9, pf: 0.65, kept: "dropped" },
];

loadAll(["strategies", "daily", "meta"]).then((d) => {
  if (d.strategies) el("asof").textContent = `updated ${fmt.ago(d.strategies.as_of)}`;
  live(d.strategies, d.meta);
  classes(d.daily);
  bench(d.meta);
  symbols(d.daily);
});

function live(data, meta) {
  const stats = data?.strategies || {};
  const enabled = new Set(meta?.strategies || []);
  const names = Object.keys(stats).sort(
    (a, b) => (stats[b].pnl ?? 0) - (stats[a].pnl ?? 0));

  el("count").textContent = enabled.size ? `${enabled.size} enabled` : "";
  if (!names.length) {
    el("live").innerHTML = "";
    el("live-empty").innerHTML =
      `<div class="empty">No strategy history in this account yet.</div>`;
    return;
  }
  el("live-empty").innerHTML = "";
  el("live").innerHTML = `
    <thead><tr><th>Strategy</th><th class="num">Trades</th>
      <th class="num">P&amp;L</th><th class="num">Expect R</th>
      <th class="num">Win%</th><th>State</th></tr></thead>
    <tbody>${names.map((n) => {
      const s = stats[n];
      const r = s.avg_r ?? s.expectancy_r ?? 0;
      return `<tr>
        <td>${n}</td>
        <td class="num">${s.trades ?? 0}</td>
        <td class="num ${sign(s.pnl)}">${fmt.money(s.pnl ?? 0)}</td>
        <td class="num ${sign(r)}">${fmt.num(r, 3)}</td>
        <td class="num">${fmt.num(s.win_rate ?? 0, 1)}</td>
        <td>${enabled.has(n)
          ? `<span class="tag up">enabled</span>`
          : `<span class="tag muted">off</span>`}</td>
      </tr>`;
    }).join("")}</tbody>`;
}

function classes(daily) {
  const rows = daily?.by_asset_class || {};
  const names = Object.keys(rows);
  if (!names.length) {
    el("classes").outerHTML =
      `<div class="empty" id="classes">No closed trades yet.</div>`;
    return;
  }
  el("classes").innerHTML = `
    <thead><tr><th>Class</th><th class="num">Trades</th>
      <th class="num">P&amp;L</th><th class="num">Expect R</th>
      <th class="num">Real?</th></tr></thead>
    <tbody>${names.map((n) => {
      const c = rows[n];
      return `<tr>
        <td>${CLASS_LABEL[n] || n}</td>
        <td class="num">${c.trades}</td>
        <td class="num ${sign(c.pnl)}">${fmt.money(c.pnl)}</td>
        <td class="num ${sign(c.expectancy_r)}">${fmt.num(c.expectancy_r, 3)}</td>
        <td class="num ${c.significant ? "up" : "muted"}">${
          c.significant ? "yes" : "no"}</td>
      </tr>`;
    }).join("")}</tbody>`;
}

function bench(meta) {
  const enabled = new Set(meta?.strategies || []);
  el("bench").innerHTML = `
    <thead><tr><th>Strategy</th><th class="num">Return</th>
      <th class="num">Trades</th><th class="num">Expectancy</th>
      <th class="num">Max DD</th><th class="num">Win%</th>
      <th class="num">PF</th><th>Verdict</th></tr></thead>
    <tbody>${BENCH.map((b) => `
      <tr>
        <td>${b.name}</td>
        <td class="num ${sign(b.ret)}">${fmt.pct(b.ret)}</td>
        <td class="num">${b.n}</td>
        <td class="num ${sign(b.r)}">${fmt.num(b.r, 3)}R</td>
        <td class="num">${fmt.num(b.dd)}%</td>
        <td class="num">${fmt.num(b.win, 1)}</td>
        <td class="num">${fmt.num(b.pf, 2)}</td>
        <td class="${enabled.has(b.name) ? "up" : "muted"}">${
          enabled.has(b.name) ? b.kept : "dropped"}</td>
      </tr>`).join("")}</tbody>`;

  el("bench-note").innerHTML = `
    Each strategy run alone over the same 90 days of crypto history, on one
    shared download, so the only thing differing between runs was the
    strategy. The four dropped ones traded about 450 times between them to
    lose money. Read this as a reason for the roster, not as a forecast:
    the choice was made by looking at these same numbers, which is exactly
    the kind of selection a backtest flatters.`;
}

function symbols(daily) {
  const rows = daily?.by_symbol || {};
  const names = Object.keys(rows).sort(
    (a, b) => (rows[b].pnl ?? 0) - (rows[a].pnl ?? 0));
  if (!names.length) {
    el("symbols").outerHTML =
      `<div class="empty" id="symbols">No closed trades yet.</div>`;
    return;
  }
  el("symbols").innerHTML = `
    <thead><tr><th>Market</th><th>Class</th><th class="num">Trades</th>
      <th class="num">P&amp;L</th><th class="num">Win%</th>
      <th class="num">Expect R</th></tr></thead>
    <tbody>${names.map((s) => {
      const r = rows[s];
      return `<tr>
        <td><span class="sym"><span class="badge">${s.slice(0, 3)}</span>${s}</span></td>
        <td class="small muted">${CLASS_LABEL[r.asset_class] || r.asset_class || "—"}</td>
        <td class="num">${r.trades}</td>
        <td class="num ${sign(r.pnl)}">${fmt.money(r.pnl)}</td>
        <td class="num">${fmt.num(r.win_rate, 1)}</td>
        <td class="num ${sign(r.expectancy_r)}">${fmt.num(r.expectancy_r, 3)}</td>
      </tr>`;
    }).join("")}</tbody>`;
}
