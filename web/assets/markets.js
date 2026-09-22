/* The whole venue, ranked.

   The dashboard shows the top twelve. This is all four hundred, with the
   reason each one was disqualified rather than a silent omission — a
   market missing from a screen and a market rejected by it look identical
   otherwise, and only one of them is a decision. */

import { load, fmt, sign, initTheme } from "./app.js";
import { renderNav } from "./nav.js";

initTheme();
renderNav();

let ROWS = [];
let HELD = new Set();
let sortKey = "quote_volume";
let sortDir = "desc";

const el = (id) => document.getElementById(id);

const COLUMNS = [
  { key: "rank",           label: "#",          num: true },
  { key: "symbol",         label: "Market" },
  { key: "quote_volume",   label: "24h volume", num: true },
  { key: "change_24h_pct", label: "24h",        num: true },
  { key: "price",          label: "Price",      num: true },
  { key: "age_days",       label: "Age",        num: true },
  { key: "news_score",     label: "News",       num: true },
  { key: "beta",           label: "β to BTC",   num: true },
  { key: "notes",          label: "Status" },
];

Promise.allSettled([load("screen"), load("meta")]).then(([screen, meta]) => {
  if (screen.status !== "fulfilled") {
    el("banner").innerHTML = `<div class="err">No market screen published.
      Run <code>python main.py publish --screen</code>.</div>`;
    return;
  }
  ROWS = screen.value.markets || [];
  el("asof").textContent = `updated ${fmt.ago(screen.value.as_of)}`;

  if (meta.status === "fulfilled") {
    HELD = new Set(Object.values(meta.value.universe || {}).flat());
  }
  draw();
});

function filtered() {
  const q = el("q").value.trim().toLowerCase();
  const tradable = el("only-tradable").getAttribute("aria-pressed") === "true";
  const isNew = el("only-new").getAttribute("aria-pressed") === "true";
  const held = el("only-held").getAttribute("aria-pressed") === "true";

  return ROWS.filter((m) => {
    if (tradable && (m.notes || []).length) return false;
    if (isNew && !m.is_new) return false;
    if (held && !HELD.has(m.symbol)) return false;
    if (q && !`${m.symbol} ${m.base}`.toLowerCase().includes(q)) return false;
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
  const tradable = ROWS.filter((m) => !(m.notes || []).length);
  const fresh = ROWS.filter((m) => m.is_new);
  const volume = rows.reduce((a, m) => a + (m.quote_volume || 0), 0);
  const movers = rows.filter((m) => Math.abs(m.change_24h_pct || 0) >= 5);

  el("tiles").innerHTML = [
    ["Markets scanned", String(ROWS.length), "on the configured venue"],
    ["Clear the filters", String(tradable.length),
      `${ROWS.length - tradable.length} disqualified`],
    ["Newly listed", String(fresh.length), "too young to measure"],
    ["Turnover shown", fmt.compact(volume), "24h, this selection"],
    ["Moved 5%+", String(movers.length), "in the last day"],
  ].map(([label, value, delta]) => `
    <div class="card stat"><div class="label">${label}</div>
      <div class="value mono">${value}</div>
      <div class="delta muted">${delta}</div></div>`).join("");
}

function draw() {
  const rows = sorted(filtered());
  tiles(rows);
  el("count").textContent = `${rows.length} of ${ROWS.length}`;

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
    <tbody>${rows.slice(0, 200).map(row).join("")}</tbody>`;

  el("table").querySelectorAll("th.sortable").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.key;
      sortDir = key === sortKey && sortDir === "desc" ? "asc" : "desc";
      sortKey = key;
      draw();
    });
  });
}

function row(m) {
  const status = (m.notes || []).length
    ? `<span class="small down">${m.notes.join("; ")}</span>`
    : m.is_new ? `<span class="tag">new listing</span>`
    : HELD.has(m.symbol) ? `<span class="tag up">in universe</span>`
    : `<span class="small muted">eligible</span>`;

  return `<tr>
    <td class="num muted">${m.rank}</td>
    <td><span class="sym"><span class="badge">${(m.base || "?").slice(0, 3)}</span>
      ${m.symbol}</span></td>
    <td class="num">${fmt.compact(m.quote_volume)}</td>
    <td class="num ${sign(m.change_24h_pct)}">${fmt.pct(m.change_24h_pct, 1)}</td>
    <td class="num muted">${fmt.price(m.price)}</td>
    <td class="num muted">${m.age_days == null ? "—" : Math.round(m.age_days) + "d"}</td>
    <td class="num ${sign(m.news_score)}">${m.news_score ? fmt.num(m.news_score, 2) : "—"}</td>
    <td class="num muted">${m.beta == null ? "—" : fmt.num(m.beta, 2)}</td>
    <td>${status}</td>
  </tr>`;
}

el("q").addEventListener("input", draw);
["only-tradable", "only-new", "only-held"].forEach((id) => {
  el(id).addEventListener("click", () => {
    const node = el(id);
    node.setAttribute("aria-pressed",
      node.getAttribute("aria-pressed") === "true" ? "false" : "true");
    draw();
  });
});
