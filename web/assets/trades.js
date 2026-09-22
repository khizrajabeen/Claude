/* Every closed trade, filterable.

   The dashboard shows the last twelve. This is the whole record, because
   the twelve most recent trades are the least useful sample for judging
   anything — they are the ones most likely to be a streak. */

import { load, fmt, sign, initTheme } from "./app.js";
import { renderNav } from "./nav.js";

initTheme();
renderNav();

let ROWS = [];
let sortKey = "closed_on_day";
let sortDir = "desc";

const el = (id) => document.getElementById(id);

const COLUMNS = [
  { key: "symbol",        label: "Market" },
  { key: "strategy",      label: "Strategy" },
  { key: "side",          label: "Side" },
  { key: "entry_price",   label: "Entry",  num: true },
  { key: "exit_price",    label: "Exit",   num: true },
  { key: "pnl",           label: "P&L",    num: true },
  { key: "r_multiple",    label: "R",      num: true },
  { key: "exit_reason",   label: "Why" },
  { key: "opened_on_day", label: "Opened" },
  { key: "closed_on_day", label: "Closed" },
];

load("trades").then((data) => {
  ROWS = data.trades || [];
  el("asof").textContent = `updated ${fmt.ago(data.as_of)}`;
  populateStrategies();
  draw();
}).catch(() => {
  el("banner").innerHTML = `<div class="err">No trade record published.
    Run <code>python main.py publish</code>.</div>`;
});

function populateStrategies() {
  const names = [...new Set(ROWS.map((r) => r.strategy).filter(Boolean))].sort();
  el("strategy").innerHTML = `<option value="">All strategies</option>` +
    names.map((n) => `<option value="${n}">${n}</option>`).join("");
}

function filtered() {
  const q = el("q").value.trim().toLowerCase();
  const side = el("side").value;
  const outcome = el("outcome").value;
  const strategy = el("strategy").value;

  return ROWS.filter((r) => {
    if (side && r.side !== side) return false;
    if (strategy && r.strategy !== strategy) return false;
    if (outcome === "win" && !(r.pnl > 0)) return false;
    if (outcome === "loss" && !(r.pnl <= 0)) return false;
    if (q) {
      const hay = `${r.symbol} ${r.strategy} ${r.exit_reason}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
}

function sorted(rows) {
  const dir = sortDir === "asc" ? 1 : -1;
  return rows.slice().sort((a, b) => {
    const x = a[sortKey], y = b[sortKey];
    if (x == null) return 1;
    if (y == null) return -1;
    if (typeof x === "number" && typeof y === "number") return (x - y) * dir;
    return String(x).localeCompare(String(y)) * dir;
  });
}

function tiles(rows) {
  // Recomputed from whatever is on screen, so the summary always
  // describes the filter rather than the whole file.
  const pnl = rows.reduce((a, r) => a + (r.pnl || 0), 0);
  const wins = rows.filter((r) => r.pnl > 0);
  const rs = rows.map((r) => r.r_multiple).filter((v) => v != null);
  const expectancy = rs.length ? rs.reduce((a, b) => a + b, 0) / rs.length : 0;
  const grossWin = wins.reduce((a, r) => a + r.pnl, 0);
  const grossLoss = Math.abs(rows.filter((r) => r.pnl < 0)
    .reduce((a, r) => a + r.pnl, 0));

  el("tiles").innerHTML = [
    ["Trades shown", String(rows.length), `of ${ROWS.length} recorded`, ""],
    ["Net P&L", fmt.money(pnl), "on this selection", sign(pnl)],
    ["Win rate", rows.length ? `${(wins.length / rows.length * 100).toFixed(1)}%` : "—",
      `${wins.length}W / ${rows.length - wins.length}L`, ""],
    ["Expectancy", `${fmt.num(expectancy, 3)}R`, "per trade", sign(expectancy)],
    ["Profit factor", grossLoss ? fmt.num(grossWin / grossLoss, 2) : "—",
      "gross win ÷ gross loss", ""],
  ].map(([label, value, delta, cls]) => `
    <div class="card stat"><div class="label">${label}</div>
      <div class="value mono ${cls}">${value}</div>
      <div class="delta muted">${delta}</div></div>`).join("");
}

function draw() {
  const rows = sorted(filtered());
  tiles(rows);
  el("count").textContent = `${rows.length} shown`;

  if (!rows.length) {
    el("table").innerHTML = "";
    el("empty").innerHTML = `<div class="empty">Nothing matches that filter.</div>`;
    return;
  }
  el("empty").innerHTML = "";
  el("table").innerHTML = `
    <thead><tr>${COLUMNS.map((c) => `
      <th class="sortable ${c.num ? "num" : ""}" data-key="${c.key}"
          ${c.key === sortKey ? `data-dir="${sortDir}"` : ""}>${c.label}</th>`
    ).join("")}</tr></thead>
    <tbody>${rows.map(row).join("")}</tbody>`;

  el("table").querySelectorAll("th.sortable").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.key;
      sortDir = key === sortKey && sortDir === "desc" ? "asc" : "desc";
      sortKey = key;
      draw();
    });
  });
}

function row(t) {
  return `<tr>
    <td><span class="sym"><span class="badge">${(t.symbol || "?").slice(0, 3)}</span>
      ${t.symbol}</span></td>
    <td><span class="tag">${t.strategy || "blend"}</span></td>
    <td class="${t.side === "long" ? "up" : "down"}">${t.side}</td>
    <td class="num muted">${fmt.price(t.entry_price)}</td>
    <td class="num muted">${fmt.price(t.exit_price)}</td>
    <td class="num ${sign(t.pnl)}">${fmt.money(t.pnl)}</td>
    <td class="num ${sign(t.r_multiple)}">${fmt.num(t.r_multiple, 2)}</td>
    <td class="small muted">${(t.exit_reason || "").replace(/_/g, " ")}</td>
    <td class="small muted">${fmt.day(t.opened_on_day)}</td>
    <td class="small muted">${fmt.day(t.closed_on_day)}</td>
  </tr>`;
}

["q", "side", "outcome", "strategy"].forEach((id) => {
  el(id).addEventListener("input", draw);
});
