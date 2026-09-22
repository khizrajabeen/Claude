/* The dashboard.

   Everything on this page comes from the JSON the bot publishes. There is
   no mock data and no placeholder: if a file is missing the page says so
   rather than showing a plausible-looking number, because a dashboard
   that invents figures is worse than one that admits it is empty. */

import {
  loadAll, fmt, sign, el, sparkline, donut, PALETTE, initTheme,
} from "./app.js";

initTheme();

const CLASS_LABEL = {
  crypto_perp: "Crypto futures",
  crypto_spot: "Crypto spot",
  equity: "Stocks",
  etf: "ETFs",
  futures: "Futures",
  unknown: "Unclassified",
};

let timer = null;

async function render() {
  const d = await loadAll([
    "portfolio", "positions", "trades", "daily", "strategies", "meta", "screen",
  ]);

  if (!d.portfolio) {
    document.getElementById("banner").innerHTML = `
      <div class="err" style="margin-bottom:16px">
        <strong>No published data.</strong> This dashboard reads files the bot
        writes. Run <code>python main.py publish</code> (add
        <code>--replay-journal</code> to show a backtest) and redeploy, or
        point it at a server that publishes them.
      </div>`;
    setStatus("off", "no data");
    return;
  }
  document.getElementById("banner").innerHTML = "";

  tiles(d.portfolio);
  status(d.portfolio, d.meta);
  curve(d.daily);
  holdings(d.positions, d.portfolio);
  byClass(d.daily);
  allocation(d.positions, d.portfolio);
  trades(d.trades);
  strategies(d.strategies);
  screen(d.screen);
}

/* ── Header ───────────────────────────────────────────── */

function setStatus(kind, text) {
  document.getElementById("status-dot").className = `dot ${kind}`;
  document.getElementById("status-text").textContent = text;
}

function status(p, meta) {
  const age = (Date.now() - new Date(p.as_of).getTime()) / 1000;
  // A dashboard whose data quietly went stale is the failure worth
  // shouting about: it looks exactly like one that is working.
  const kind = p.halted_reason ? "off" : age > 7200 ? "stale" : "";
  setStatus(kind,
    p.halted_reason ? `halted — ${p.halted_reason}`
      : `updated ${fmt.ago(p.as_of)}`);

  if (meta) {
    const names = (meta.strategies || []).join(", ") || "none";
    document.getElementById("mode-pill").textContent =
      `${meta.mode} · ${meta.days_recorded} days · ${names}`;
  }
}

function tiles(p) {
  const rows = [
    ["Portfolio value", fmt.money(p.equity),
      `${fmt.pct(p.daily_return_pct)} today`, sign(p.daily_return_pct)],
    ["Daily P&L", fmt.money(p.daily_pnl), "realised + open",
      sign(p.daily_pnl)],
    ["Total return", fmt.pct(p.total_return_pct),
      `from ${fmt.money(p.starting_equity)}`, sign(p.total_return_pct)],
    ["Max drawdown", `${fmt.num(p.max_drawdown_pct)}%`, "peak to trough",
      p.max_drawdown_pct > 10 ? "down" : "muted"],
    ["Expectancy", `${fmt.num(p.expectancy_r, 3)}R`,
      `${p.trades} trades · ${fmt.num(p.win_rate, 1)}% win`,
      sign(p.expectancy_r)],
  ];
  document.getElementById("tiles").innerHTML = rows.map(
    ([label, value, delta, cls]) => `
      <div class="card stat">
        <div class="label">${label}</div>
        <div class="value mono ${cls}">${value}</div>
        <div class="delta muted">${delta}</div>
      </div>`).join("");
}

/* ── Panels ───────────────────────────────────────────── */

function curve(daily) {
  const host = document.getElementById("curve");
  const points = (daily?.curve || []).filter((r) => r.equity > 0);
  if (points.length < 2) {
    host.innerHTML = `<div class="empty">Not enough history to plot yet.</div>`;
    return;
  }
  host.innerHTML = sparkline(points.map((r) => r.equity));
  document.getElementById("curve-range").textContent =
    `${points.length} days`;
  document.getElementById("curve-start").textContent =
    `${fmt.day(points[0].day)} · ${fmt.money(points[0].equity)}`;
  document.getElementById("curve-end").textContent =
    `${fmt.day(points.at(-1).day)} · ${fmt.money(points.at(-1).equity)}`;
}

function holdings(positions, p) {
  const rows = positions?.positions || [];
  document.getElementById("pos-count").textContent =
    `${rows.length} of ${p.open_positions ?? rows.length}`;
  const host = document.getElementById("holdings");
  if (!rows.length) {
    host.innerHTML = `<div class="empty">Flat — no positions open.</div>`;
    return;
  }
  host.innerHTML = rows.map((r) => {
    const notional = (r.quantity || 0) * (r.entry_price || 0);
    const share = p.equity ? Math.min(100, (notional / p.equity) * 100) : 0;
    return `
      <div style="margin-bottom:14px">
        <div class="row" style="justify-content:space-between;margin-bottom:6px">
          <span class="sym">
            <span class="badge">${(r.symbol || "?").slice(0, 3)}</span>
            <span><div>${r.symbol}</div>
              <div class="sub">${r.side} · ${r.strategy || "blend"}${
                r.units > 1 ? ` · ${r.units} units` : ""}${
                r.leverage > 1 ? ` · ${fmt.num(r.leverage, 1)}x` : ""}</div>
            </span>
          </span>
          <span class="mono small">${fmt.compact(notional)}</span>
        </div>
        <div class="bar"><i style="width:${share.toFixed(1)}%"></i></div>
      </div>`;
  }).join("");
}

function byClass(daily) {
  const classes = daily?.by_asset_class || {};
  const names = Object.keys(classes);
  const host = document.getElementById("by-class");
  if (!names.length) {
    host.innerHTML = `<div class="empty">No closed trades yet.</div>`;
    return;
  }
  host.innerHTML = `
    <div style="overflow-x:auto"><table>
      <thead><tr><th>Class</th><th class="num">Trades</th>
        <th class="num">P&amp;L</th><th class="num">Win%</th>
        <th class="num">Expect R</th><th class="num">Real?</th></tr></thead>
      <tbody>${names.map((n) => {
        const c = classes[n];
        return `<tr>
          <td>${CLASS_LABEL[n] || n}</td>
          <td class="num">${c.trades}</td>
          <td class="num ${sign(c.pnl)}">${fmt.money(c.pnl)}</td>
          <td class="num">${fmt.num(c.win_rate, 1)}</td>
          <td class="num ${sign(c.expectancy_r)}">${fmt.num(c.expectancy_r, 3)}</td>
          <td class="num ${c.significant ? "up" : "muted"}">${
            c.significant ? "yes" : "no"}</td>
        </tr>`;
      }).join("")}</tbody>
    </table></div>
    <p class="small muted" style="margin-top:12px">
      "Real?" is whether the expectancy clears the critical t-value for its
      own sample size. A fine average over a handful of trades is not a
      finding.</p>`;
}

function allocation(positions, p) {
  const rows = positions?.positions || [];
  const byClassName = {};
  rows.forEach((r) => {
    const key = CLASS_LABEL[r.asset_class] || r.asset_class || "Unclassified";
    byClassName[key] = (byClassName[key] || 0) + (r.quantity || 0) * (r.entry_price || 0);
  });
  const invested = Object.values(byClassName).reduce((a, b) => a + b, 0);
  const cash = Math.max(0, (p.cash ?? 0));
  if (cash > 0) byClassName["Cash"] = cash;

  const slices = Object.entries(byClassName)
    .filter(([, v]) => v > 0)
    .map(([label, value], i) => ({
      label, value,
      color: label === "Cash" ? "var(--ink-3)" : PALETTE[i % PALETTE.length],
    }));

  document.getElementById("donut").innerHTML = donut(slices);
  const total = slices.reduce((a, s) => a + s.value, 0) || 1;
  document.getElementById("alloc-legend").innerHTML = slices.map((s) => `
    <div><span class="swatch" style="background:${s.color}"></span>
      <span>${s.label}</span>
      <span class="val">${((s.value / total) * 100).toFixed(0)}%</span></div>`
  ).join("") || `<div class="empty">Nothing allocated.</div>`;
  void invested;
}

function trades(data) {
  const rows = (data?.trades || []).slice(0, 12);
  const table = document.getElementById("trade-table");
  if (!rows.length) {
    table.outerHTML = `<div class="empty" id="trade-table">No closed trades yet.</div>`;
    return;
  }
  table.innerHTML = `
    <thead><tr><th>Market</th><th>Side</th><th class="num">P&amp;L</th>
      <th class="num">R</th><th>Exit</th><th>Closed</th></tr></thead>
    <tbody>${rows.map((t) => `
      <tr>
        <td><span class="sym"><span class="badge">${(t.symbol || "?").slice(0, 3)}</span>
          <span><div>${t.symbol}</div>
          <div class="sub">${t.strategy || "blend"}</div></span></span></td>
        <td class="${t.side === "long" ? "up" : "down"}">${t.side}</td>
        <td class="num ${sign(t.pnl)}">${fmt.money(t.pnl)}</td>
        <td class="num ${sign(t.r_multiple)}">${fmt.num(t.r_multiple, 2)}</td>
        <td class="muted small">${(t.exit_reason || "").replace(/_/g, " ")}</td>
        <td class="muted small">${fmt.day(t.closed_on_day)}</td>
      </tr>`).join("")}</tbody>`;
}

function strategies(data) {
  const stats = data?.strategies || {};
  const names = Object.keys(stats);
  const table = document.getElementById("strategy-table");
  if (!names.length) {
    table.outerHTML = `<div class="empty" id="strategy-table">No strategy history yet.</div>`;
    return;
  }
  const expectancy = (s) => s.avg_r ?? s.expectancy_r ?? s.expectancy ?? 0;
  const rowOf = (n) => {
    const s = stats[n] || {};
    const exp = expectancy(s);
    return `<tr>
      <td>${n}</td>
      <td class="num">${s.trades ?? s.sample ?? 0}</td>
      <td class="num ${sign(s.pnl ?? 0)}">${fmt.money(s.pnl ?? 0)}</td>
      <td class="num ${sign(exp)}">${fmt.num(exp, 3)}</td>
      <td class="num">${fmt.num(s.win_rate ?? 0, 1)}</td>
    </tr>`;
  };
  // Best first, so what is working is the first thing read.
  const ordered = names.slice().sort(
    (a, b) => (stats[b].pnl ?? 0) - (stats[a].pnl ?? 0));
  table.innerHTML = `
    <thead><tr><th>Strategy</th><th class="num">Trades</th>
      <th class="num">P&amp;L</th><th class="num">Expect R</th>
      <th class="num">Win%</th></tr></thead>
    <tbody>${ordered.map(rowOf).join("")}</tbody>`;
}

function screen(data) {
  const rows = (data?.markets || []).slice(0, 12);
  const table = document.getElementById("screen-table");
  if (!rows.length) {
    document.getElementById("screen-note").textContent =
      "run `python main.py publish --screen` to fill this in";
    table.outerHTML = `<div class="empty" id="screen-table">No screen published.</div>`;
    return;
  }
  table.innerHTML = `
    <thead><tr><th>#</th><th>Market</th><th class="num">24h volume</th>
      <th class="num">24h</th><th class="num">Age</th>
      <th class="num">News</th><th>Note</th></tr></thead>
    <tbody>${rows.map((m) => `
      <tr>
        <td class="muted">${m.rank}</td>
        <td><span class="sym"><span class="badge">${(m.base || "?").slice(0, 3)}</span>
          ${m.symbol}</span></td>
        <td class="num">${fmt.compact(m.quote_volume)}</td>
        <td class="num ${sign(m.change_24h_pct)}">${fmt.pct(m.change_24h_pct, 1)}</td>
        <td class="num muted">${m.age_days == null ? "—" : Math.round(m.age_days) + "d"}</td>
        <td class="num ${sign(m.news_score)}">${fmt.num(m.news_score, 2)}</td>
        <td class="small muted">${(m.notes || []).join("; ") || (m.is_new ? "new listing" : "")}</td>
      </tr>`).join("")}</tbody>`;
}

/* ── Lifecycle ────────────────────────────────────────── */

document.getElementById("refresh").addEventListener("click", () => render());
render();
// The bot publishes on its own schedule; polling keeps a left-open tab
// from drifting without anyone noticing.
timer = setInterval(render, 60_000);
window.addEventListener("beforeunload", () => clearInterval(timer));
