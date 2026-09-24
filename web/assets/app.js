/* Shared runtime: theme, formatting, data loading, and the app shell.
 *
 * The navigation is defined once here and rendered into every page.
 * Repeating a rail across six HTML files guarantees drift — a page
 * gets added, five files get updated, and the sixth quietly points
 * at nothing.
 */

/* ── Theme ──────────────────────────────────────────────────
 * Three states, not two. "Auto" is the honest default: most people
 * have already told their OS what they want, and asking again is
 * a worse answer than listening.
 */
export const THEMES = [
  { id: "light", glyph: "☀", label: "Light" },
  { id: "dark",  glyph: "☾", label: "Dark" },
  { id: "auto",  glyph: "◐", label: "Match system" },
];

const safeGet = (k) => { try { return localStorage.getItem(k); } catch { return null; } };
const safeSet = (k, v) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } };

export const currentTheme = () => safeGet("theme") || "auto";

export function applyTheme(id) {
  document.documentElement.dataset.theme = id;
  safeSet("theme", id);
  for (const b of document.querySelectorAll(".theme-switch button"))
    b.setAttribute("aria-pressed", String(b.dataset.theme === id));
}

export function initTheme() {
  const now = currentTheme();
  for (const host of document.querySelectorAll("[data-toggle-theme]")) {
    const box = document.createElement("div");
    box.className = "theme-switch";
    box.setAttribute("role", "group");
    box.setAttribute("aria-label", "Colour theme");
    for (const t of THEMES) {
      const b = document.createElement("button");
      b.type = "button";
      b.dataset.theme = t.id;
      b.setAttribute("aria-pressed", String(t.id === now));
      b.innerHTML = `<span aria-hidden="true">${t.glyph}</span><span class="vh">${t.label}</span>`;
      b.title = t.label;
      b.onclick = () => applyTheme(t.id);
      box.appendChild(b);
    }
    host.replaceChildren(box);
  }
  applyTheme(now);
  // An "auto" page must follow the OS while it is open, not only on load.
  matchMedia("(prefers-color-scheme: dark)")
    .addEventListener("change", () => { if (currentTheme() === "auto") applyTheme("auto"); });
}

/* ── Formatting ─────────────────────────────────────────────
 * One place, so $1,234.50 never renders three different ways.
 */
export const usd = (v, dp = 2) =>
  v == null || Number.isNaN(v) ? "—"
  : (v < 0 ? "-$" : "$") + Math.abs(v).toLocaleString("en-US",
      { minimumFractionDigits: dp, maximumFractionDigits: dp });

export const compact = (v) =>
  v == null ? "—"
  : Math.abs(v) >= 1e9 ? "$" + (v / 1e9).toFixed(2) + "B"
  : Math.abs(v) >= 1e6 ? "$" + (v / 1e6).toFixed(2) + "M"
  : Math.abs(v) >= 1e3 ? "$" + (v / 1e3).toFixed(1) + "K"
  : usd(v);

export const pct = (v, dp = 2) =>
  v == null || Number.isNaN(v) ? "—" : (v >= 0 ? "+" : "") + v.toFixed(dp) + "%";

export const num = (v, dp = 2) =>
  v == null || Number.isNaN(v) ? "—"
  : v.toLocaleString("en-US", { minimumFractionDigits: dp, maximumFractionDigits: dp });

/* Prices span eight orders of magnitude here — $86,000 BTC and
 * $0.00002 shiba. A fixed 2dp renders half the screen as $0.00. */
export const price = (v) =>
  v == null ? "—"
  : v >= 1000 ? usd(v, 2) : v >= 1 ? usd(v, 3) : v >= 0.01 ? usd(v, 4) : usd(v, 7);

export const cls = (v) => (v > 0 ? "up" : v < 0 ? "down" : "muted");
export const arrow = (v) => (v > 0 ? "▲" : v < 0 ? "▼" : "•");

export function ago(iso) {
  if (!iso) return "never";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 90) return "just now";
  if (s < 5400) return Math.round(s / 60) + "m ago";
  if (s < 172800) return Math.round(s / 3600) + "h ago";
  return Math.round(s / 86400) + "d ago";
}

/* ── Data ───────────────────────────────────────────────────
 * A missing file is a state worth explaining, not a stack trace
 * and not a zero. A dashboard that invents a plausible number is
 * worse than one that admits it has none.
 */
export async function load(name) {
  const res = await fetch(`data/${name}.json`, { cache: "no-store" });
  if (!res.ok) throw new Error(`data/${name}.json — HTTP ${res.status}`);
  return res.json();
}

export function explain(host, err) {
  host.innerHTML =
    `<div class="err"><p>${err.message}</p>
     <p class="muted" style="margin-top:8px">Generate it with
     <code>python main.py publish --screen</code>, then reload.</p></div>`;
}

/* ── Freshness ──────────────────────────────────────────────
 * Data that quietly went stale looks exactly like data that is
 * working, which is the failure mode worth designing against.
 */
export function freshness(iso) {
  if (!iso) return { state: "off", text: "no data" };
  const hours = (Date.now() - new Date(iso).getTime()) / 3.6e6;
  if (hours > 26) return { state: "off", text: `stale — ${ago(iso)}` };
  if (hours > 2)  return { state: "stale", text: `${ago(iso)}` };
  return { state: "", text: `updated ${ago(iso)}` };
}

/* ── Shell ──────────────────────────────────────────────────
 * Icons are inline so a rail never renders as six empty squares
 * while a font request is in flight.
 */
const I = {
  home:      'M3 11l9-8 9 8M5 9.5V21h14V9.5',
  dashboard: 'M3 3h7v9H3zM14 3h7v5h-7zM14 12h7v9h-7zM3 16h7v5H3z',
  trades:    'M3 17l6-6 4 4 8-8M15 7h6v6',
  markets:   'M3 3v18h18M7 15l4-5 3 3 5-7',
  strategies:'M12 3l9 5-9 5-9-5 9-5zM3 13l9 5 9-5M3 17l9 5 9-5',
  settings:  'M12 15a3 3 0 100-6 3 3 0 000 6zM19.4 15a1.6 1.6 0 00.3 1.8l.1.1a2 2 0 11-2.8 2.8l-.1-.1a1.6 1.6 0 00-1.8-.3 1.6 1.6 0 00-1 1.5V21a2 2 0 11-4 0v-.1A1.6 1.6 0 007 19.4a1.6 1.6 0 00-1.8.3l-.1.1a2 2 0 11-2.8-2.8l.1-.1a1.6 1.6 0 00.3-1.8 1.6 1.6 0 00-1.5-1H1a2 2 0 110-4h.1A1.6 1.6 0 002.6 7a1.6 1.6 0 00-.3-1.8l-.1-.1a2 2 0 112.8-2.8l.1.1a1.6 1.6 0 001.8.3H7a1.6 1.6 0 001-1.5V1a2 2 0 114 0v.1a1.6 1.6 0 001 1.5 1.6 1.6 0 001.8-.3l.1-.1a2 2 0 112.8 2.8l-.1.1a1.6 1.6 0 00-.3 1.8V7a1.6 1.6 0 001.5 1H21a2 2 0 110 4h-.1a1.6 1.6 0 00-1.5 1z',
};
const icon = (k) =>
  `<svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor"
     stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
     <path d="${I[k]}"/></svg>`;

export const PAGES = [
  { id: "index",      href: "index.html",      icon: "home",       label: "Overview" },
  { id: "dashboard",  href: "dashboard.html",  icon: "dashboard",  label: "Dashboard" },
  { id: "trades",     href: "trades.html",     icon: "trades",     label: "Trades" },
  { id: "markets",    href: "markets.html",    icon: "markets",    label: "Markets" },
  { id: "strategies", href: "strategies.html", icon: "strategies", label: "Strategies" },
  { id: "settings",   href: "settings.html",   icon: "settings",   label: "Settings" },
];

export function shell(active) {
  const rail = document.querySelector(".rail");
  if (rail) {
    rail.innerHTML =
      `<a class="mark-glyph" href="index.html" aria-label="Meridian home"></a>` +
      PAGES.filter((p) => p.id !== "index").map((p) =>
        `<a href="${p.href}"${p.id === active ? ' aria-current="page"' : ""}>
           ${icon(p.icon)}<span class="tip">${p.label}</span>
           <span class="vh">${p.label}</span></a>`).join("") +
      `<span class="spacer"></span>
       <a href="index.html">${icon("home")}<span class="tip">Landing page</span>
        <span class="vh">Landing page</span></a>`;
  }
  const tabs = document.querySelector(".tabs");
  if (tabs) {
    tabs.innerHTML = PAGES.filter((p) => !["index", "settings"].includes(p.id))
      .map((p) => `<a href="${p.href}"${p.id === active ? ' imsy-current="page"' : ""}>${p.label}</a>`)
      .join("").replaceAll("imsy-current", "aria-current");
  }
  initTheme();
}

/* Deterministic colour for a symbol chip, so BTC is the same
 * colour on every page and across reloads. */
export function chipColour(sym) {
  let h = 0;
  for (const c of sym) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return `hsl(${h % 360} 58% 42%)`;
}
export const base = (sym) => (sym || "").split("/")[0].replace(/:.*/, "");
