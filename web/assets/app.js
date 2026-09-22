/* Shared helpers: loading the bot's published data, formatting it, and
   the handful of chart primitives the dashboard needs.

   No framework and no build step. The page is served from GitHub Pages as
   plain files, so anything that needed compiling would need a toolchain
   between the bot and the screen — and the point of publishing JSON is
   that there is nothing in between. */

export const DATA_DIR = "./data";

/* ── Loading ──────────────────────────────────────────── */

export async function load(name) {
  // Cache-bust: Pages serves these with a long max-age, and a dashboard
  // showing yesterday's equity because of a cached file is worse than
  // one that says it cannot reach the data.
  const res = await fetch(`${DATA_DIR}/${name}.json?t=${Date.now()}`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`${name}: HTTP ${res.status}`);
  return res.json();
}

export async function loadAll(names) {
  const out = {};
  const results = await Promise.allSettled(names.map((n) => load(n)));
  results.forEach((r, i) => {
    out[names[i]] = r.status === "fulfilled" ? r.value : null;
  });
  return out;
}

/* ── Formatting ───────────────────────────────────────── */

export const fmt = {
  money(v, dp = 2) {
    if (v == null || Number.isNaN(v)) return "—";
    return v.toLocaleString("en-US", {
      style: "currency", currency: "USD",
      minimumFractionDigits: dp, maximumFractionDigits: dp,
    });
  },
  compact(v) {
    if (v == null) return "—";
    const abs = Math.abs(v);
    if (abs >= 1e9) return `$${(v / 1e9).toFixed(2)}B`;
    if (abs >= 1e6) return `$${(v / 1e6).toFixed(1)}M`;
    if (abs >= 1e3) return `$${(v / 1e3).toFixed(1)}K`;
    return `$${v.toFixed(0)}`;
  },
  pct(v, dp = 2) {
    if (v == null || Number.isNaN(v)) return "—";
    return `${v >= 0 ? "+" : ""}${v.toFixed(dp)}%`;
  },
  num(v, dp = 2) {
    if (v == null || Number.isNaN(v)) return "—";
    return v.toFixed(dp);
  },
  // Prices span BTC at 85,000 and PEPE at 0.000012, so a fixed number of
  // decimals is wrong for one of them whichever is chosen.
  price(v) {
    if (v == null) return "—";
    const abs = Math.abs(v);
    const dp = abs >= 1000 ? 2 : abs >= 1 ? 4 : abs >= 0.01 ? 6 : 8;
    return v.toLocaleString("en-US", {
      minimumFractionDigits: dp, maximumFractionDigits: dp,
    });
  },
  ago(iso) {
    if (!iso) return "never";
    const secs = (Date.now() - new Date(iso).getTime()) / 1000;
    if (secs < 90) return "just now";
    if (secs < 3600) return `${Math.round(secs / 60)}m ago`;
    if (secs < 86400) return `${Math.round(secs / 3600)}h ago`;
    return `${Math.round(secs / 86400)}d ago`;
  },
  day(d) {
    if (!d) return "—";
    return new Date(d + "T00:00:00Z").toLocaleDateString("en-US", {
      month: "short", day: "numeric", timeZone: "UTC",
    });
  },
};

export const sign = (v) => (v > 0 ? "up" : v < 0 ? "down" : "muted");

export function el(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}

export function setText(id, text, cls) {
  const node = document.getElementById(id);
  if (!node) return;
  node.textContent = text;
  if (cls) node.className = node.className.replace(/\b(up|down|muted)\b/g, "") + " " + cls;
}

/* ── Charts ───────────────────────────────────────────── */
/* Hand-drawn SVG rather than a charting library: three shapes are needed
   and a library is 200KB to draw them. */

export function sparkline(values, { w = 640, h = 190, fill = true } = {}) {
  if (!values || values.length < 2) {
    return `<div class="empty">Not enough history to plot yet.</div>`;
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const pad = 6;
  const x = (i) => (i / (values.length - 1)) * (w - pad * 2) + pad;
  const y = (v) => h - pad - ((v - min) / span) * (h - pad * 2);

  const line = values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const area = `${line} L${x(values.length - 1).toFixed(1)},${h} L${x(0).toFixed(1)},${h} Z`;
  const rising = values[values.length - 1] >= values[0];
  const stroke = rising ? "var(--up)" : "var(--down)";

  return `
    <svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"
         style="width:100%;height:${h}px;display:block" role="img"
         aria-label="Equity curve">
      <defs>
        <linearGradient id="sparkFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="${stroke}" stop-opacity=".26"/>
          <stop offset="100%" stop-color="${stroke}" stop-opacity="0"/>
        </linearGradient>
      </defs>
      ${fill ? `<path d="${area}" fill="url(#sparkFill)"/>` : ""}
      <path d="${line}" fill="none" stroke="${stroke}" stroke-width="2"
            stroke-linejoin="round" stroke-linecap="round"/>
    </svg>`;
}

export function donut(slices, { size = 168, thickness = 22 } = {}) {
  const total = slices.reduce((a, s) => a + Math.abs(s.value), 0);
  if (!total) return `<div class="empty">Nothing allocated yet.</div>`;

  const r = (size - thickness) / 2;
  const c = size / 2;
  const circumference = 2 * Math.PI * r;
  let offset = 0;

  const rings = slices.map((s) => {
    const share = Math.abs(s.value) / total;
    const len = share * circumference;
    const seg = `<circle cx="${c}" cy="${c}" r="${r}" fill="none"
        stroke="${s.color}" stroke-width="${thickness}"
        stroke-dasharray="${len.toFixed(2)} ${(circumference - len).toFixed(2)}"
        stroke-dashoffset="${(-offset).toFixed(2)}"
        transform="rotate(-90 ${c} ${c})"><title>${s.label}</title></circle>`;
    offset += len;
    return seg;
  }).join("");

  return `<svg viewBox="0 0 ${size} ${size}" style="width:${size}px;height:${size}px">
    ${rings}
  </svg>`;
}

export const PALETTE = [
  "#e458a8", "#8b3fd4", "#ff8ec7", "#34d399", "#fbbf24",
  "#60a5fa", "#f472b6", "#a78bfa",
];

/* ── Theme ────────────────────────────────────────────── */

export const THEMES = [
  { id: "light", glyph: "☀", label: "Light" },
  { id: "dark",  glyph: "☾", label: "Dark" },
  { id: "auto",  glyph: "◐", label: "Match system" },
];

export function currentTheme() {
  return safeGet("theme") || "auto";
}

export function applyTheme(id) {
  document.documentElement.dataset.theme = id;
  safeSet("theme", id);
  document.querySelectorAll(".theme-switch button").forEach((b) => {
    b.setAttribute("aria-pressed", String(b.dataset.theme === id));
  });
}

/* Renders the switch into every [data-toggle-theme] host and wires it up.

   Three options rather than two. "Auto" is the default because a person
   who has set their machine to light at their desk and dark at night has
   already answered this question, and a dashboard that ignores that
   answer is one more thing to fix twice a day. */
export function initTheme() {
  applyTheme(currentTheme());

  document.querySelectorAll("[data-toggle-theme]").forEach((host) => {
    const control = document.createElement("div");
    control.className = "theme-switch";
    control.setAttribute("role", "group");
    control.setAttribute("aria-label", "Colour theme");
    control.innerHTML = THEMES.map((t) => `
      <button type="button" data-theme="${t.id}" title="${t.label}"
              aria-pressed="${t.id === currentTheme()}">${t.glyph}<span>${t.label}</span></button>`
    ).join("");
    control.addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-theme]");
      if (btn) applyTheme(btn.dataset.theme);
    });
    host.replaceWith(control);
  });

  // Following the system means following it as it changes, not only as it
  // was when the page loaded.
  const media = window.matchMedia?.("(prefers-color-scheme: light)");
  media?.addEventListener?.("change", () => {
    if (currentTheme() === "auto") applyTheme("auto");
  });
}

/* localStorage throws in a private window and in some embedded views, and
   a dashboard that refuses to render because it could not remember a
   colour scheme is worse than one that forgets. */
export function safeGet(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}
export function safeSet(key, value) {
  try { localStorage.setItem(key, value); return true; } catch { return false; }
}
export function safeRemove(key) {
  try { localStorage.removeItem(key); } catch { /* nothing to do */ }
}
