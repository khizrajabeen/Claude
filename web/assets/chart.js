/* Charts, drawn as plain SVG.
 *
 * No chart library. Not out of purity — a bundled library is 60-200KB
 * before it draws a pixel, needs a build step to tree-shake, and styles
 * itself in colours that ignore the theme. These are four chart types
 * on a static page; the SVG is smaller than the loader would be, and it
 * inherits the theme because it uses the same CSS variables everything
 * else does.
 *
 * Everything scales by viewBox, so one drawing works at any width.
 */

const NS = "http://www.w3.org/2000/svg";
const el = (n, a = {}) => {
  const e = document.createElementNS(NS, n);
  for (const [k, v] of Object.entries(a)) e.setAttribute(k, v);
  return e;
};

/* Axis ticks a human would choose: 1, 2, 2.5, 5 x 10^n. A raw
 * range/5 gives ticks like 3,417 and makes a chart unreadable. */
function niceTicks(lo, hi, count = 4) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(v);
  return out;
}

function tooltip(host) {
  let box = host.querySelector(".tip-box");
  if (!box) {
    box = document.createElement("div");
    box.className = "tip-box";
    host.appendChild(box);
  }
  return {
    show(html, x, y) {
      box.innerHTML = html;
      box.style.left = x + "px";
      box.style.top = (y - 10) + "px";
      box.style.opacity = "1";
    },
    hide() { box.style.opacity = "0"; },
  };
}

/* ── Area / line chart with a hover readout ───────────────────
 * series: [{ name, values: [{x: label, y: number}], color, dashed }]
 * The first series owns the hover, since a shared x means one
 * cursor answers for all of them.
 */
export function lineChart(host, series, opts = {}) {
  const { height = 260, fill = false, fmt = (v) => v.toFixed(2),
          yZero = false, pad = { t: 12, r: 14, b: 26, l: 52 } } = opts;
  const W = 900, H = height;
  host.innerHTML = "";
  host.classList.add("chart");

  const live = series.filter((s) => s.values && s.values.length);
  if (!live.length) { host.innerHTML = '<p class="empty">No data yet.</p>'; return; }

  const n = Math.max(...live.map((s) => s.values.length));
  const ys = live.flatMap((s) => s.values.map((p) => p.y));
  let lo = Math.min(...ys), hi = Math.max(...ys);
  if (yZero) lo = Math.min(0, lo);
  if (lo === hi) { lo -= 1; hi += 1; }
  const padY = (hi - lo) * 0.08;
  lo -= padY; hi += padY;

  const X = (i) => pad.l + (i / Math.max(1, n - 1)) * (W - pad.l - pad.r);
  const Y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (H - pad.t - pad.b);

  const svg = el("svg", {
    viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none",
    role: "img", "aria-label": opts.label || "chart",
  });
  svg.style.height = height + "px";

  // Gridlines and y labels
  for (const t of niceTicks(lo, hi, 4)) {
    svg.appendChild(el("line", { class: "gridline", x1: pad.l, x2: W - pad.r, y1: Y(t), y2: Y(t) }));
    const lab = el("text", { class: "axis", x: pad.l - 8, y: Y(t) + 4, "text-anchor": "end" });
    lab.textContent = opts.yFmt ? opts.yFmt(t) : fmt(t);
    svg.appendChild(lab);
  }

  // x labels — a handful, evenly spaced, never overlapping
  const base = live[0].values;
  const every = Math.max(1, Math.round(n / 6));
  base.forEach((p, i) => {
    if (i % every && i !== n - 1) return;
    const t = el("text", { class: "axis", x: X(i), y: H - 6, "text-anchor": "middle" });
    t.textContent = p.x;
    svg.appendChild(t);
  });

  live.forEach((s, si) => {
    const colour = s.color || `var(--s${(si % 6) + 1})`;
    const pts = s.values.map((p, i) => [X(i), Y(p.y)]);
    const d = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");

    if (fill && si === 0) {
      const gid = `g${Math.random().toString(36).slice(2, 8)}`;
      const grad = el("linearGradient", { id: gid, x1: 0, y1: 0, x2: 0, y2: 1 });
      grad.appendChild(el("stop", { offset: "0%", "stop-color": colour, "stop-opacity": ".26" }));
      grad.appendChild(el("stop", { offset: "100%", "stop-color": colour, "stop-opacity": "0" }));
      const defs = el("defs"); defs.appendChild(grad); svg.appendChild(defs);
      svg.appendChild(el("path", {
        d: `${d} L${pts[pts.length - 1][0]},${H - pad.b} L${pts[0][0]},${H - pad.b} Z`,
        fill: `url(#${gid})`, stroke: "none",
      }));
    }
    svg.appendChild(el("path", {
      d, fill: "none", stroke: colour, "stroke-width": s.width || 2,
      "stroke-linejoin": "round", "stroke-linecap": "round",
      ...(s.dashed ? { "stroke-dasharray": "4 4" } : {}),
    }));
  });

  // Hover: a dashed cursor plus a readout of every series at that x.
  const cursor = el("line", { class: "cursor", y1: pad.t, y2: H - pad.b, opacity: 0 });
  svg.appendChild(cursor);
  const dots = live.map((s, si) => {
    const c = el("circle", {
      r: 4, fill: "var(--surface)", "stroke-width": 2, opacity: 0,
      stroke: s.color || `var(--s${(si % 6) + 1})`,
    });
    svg.appendChild(c); return c;
  });
  const hit = el("rect", { class: "hit", x: pad.l, y: 0, width: W - pad.l - pad.r, height: H });
  svg.appendChild(hit);
  host.appendChild(svg);

  const tip = tooltip(host);
  const move = (ev) => {
    const box = svg.getBoundingClientRect();
    const rel = (ev.clientX - box.left) / box.width * W;
    const i = Math.max(0, Math.min(n - 1, Math.round((rel - pad.l) / (W - pad.l - pad.r) * (n - 1))));
    cursor.setAttribute("x1", X(i)); cursor.setAttribute("x2", X(i));
    cursor.setAttribute("opacity", 1);
    const lines = live.map((s, si) => {
      const p = s.values[i];
      if (!p) { dots[si].setAttribute("opacity", 0); return ""; }
      dots[si].setAttribute("cx", X(i)); dots[si].setAttribute("cy", Y(p.y));
      dots[si].setAttribute("opacity", 1);
      return `<div>${live.length > 1 ? s.name + " " : ""}<b>${fmt(p.y)}</b></div>`;
    }).join("");
    tip.show(`<div class="muted" style="margin-bottom:3px">${base[i]?.x ?? ""}</div>${lines}`,
             X(i) / W * box.width, Y(live[0].values[i]?.y ?? lo) / H * box.height);
  };
  hit.addEventListener("mousemove", move);
  hit.addEventListener("mouseleave", () => {
    cursor.setAttribute("opacity", 0);
    dots.forEach((d) => d.setAttribute("opacity", 0));
    tip.hide();
  });
}

/* ── Donut ────────────────────────────────────────────────── */
export function donut(host, slices, opts = {}) {
  const { size = 190, thickness = 26, centreLabel = "", centreValue = "" } = opts;
  host.innerHTML = "";
  host.classList.add("chart");
  const total = slices.reduce((a, s) => a + s.value, 0);
  if (!total) { host.innerHTML = '<p class="empty">Nothing allocated.</p>'; return; }

  const R = size / 2, r = R - thickness;
  const svg = el("svg", { viewBox: `0 0 ${size} ${size}`, role: "img",
                          "aria-label": opts.label || "allocation" });
  svg.style.maxWidth = size + "px";
  svg.style.margin = "0 auto";

  let a0 = -Math.PI / 2;
  slices.forEach((s, i) => {
    const a1 = a0 + (s.value / total) * Math.PI * 2;
    const big = a1 - a0 > Math.PI ? 1 : 0;
    const P = (ang, rad) => [R + rad * Math.cos(ang), R + rad * Math.sin(ang)];
    const [x0, y0] = P(a0, R), [x1, y1] = P(a1, R);
    const [x2, y2] = P(a1, r), [x3, y3] = P(a0, r);
    const path = el("path", {
      d: `M${x0},${y0} A${R},${R} 0 ${big} 1 ${x1},${y1} L${x2},${y2} A${r},${r} 0 ${big} 0 ${x3},${y3} Z`,
      fill: s.color || `var(--s${(i % 6) + 1})`,
      stroke: "var(--surface)", "stroke-width": 2,
    });
    const t = el("title");
    t.textContent = `${s.name}: ${(s.value / total * 100).toFixed(1)}%`;
    path.appendChild(t);
    svg.appendChild(path);
    a0 = a1;
  });

  if (centreValue) {
    const v = el("text", { x: R, y: R + 2, "text-anchor": "middle",
                           fill: "var(--ink)", "font-size": 21, "font-weight": 660 });
    v.textContent = centreValue;
    svg.appendChild(v);
    const l = el("text", { x: R, y: R + 20, "text-anchor": "middle",
                           fill: "var(--ink-3)", "font-size": 11 });
    l.textContent = centreLabel;
    svg.appendChild(l);
  }
  host.appendChild(svg);
}

/* ── Horizontal bars ──────────────────────────────────────── */
export function barList(host, items, opts = {}) {
  host.innerHTML = "";
  const max = Math.max(...items.map((i) => Math.abs(i.value)), 1e-9);
  const wrap = document.createElement("div");
  for (const it of items) {
    const row = document.createElement("div");
    row.style.cssText = "display:flex;align-items:center;gap:12px;padding:8px 0";
    const neg = it.value < 0;
    row.innerHTML =
      `<span style="width:104px;font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${it.name}</span>
       <span style="flex:1;height:8px;border-radius:4px;background:var(--line);overflow:hidden">
         <i style="display:block;height:100%;width:${(Math.abs(it.value) / max * 100).toFixed(1)}%;
            border-radius:4px;background:${it.color || (neg ? "var(--down)" : "var(--up)")}"></i>
       </span>
       <span class="num ${neg ? "down" : "up"}" style="width:82px;text-align:right;font-size:12.5px">
         ${opts.fmt ? opts.fmt(it.value) : it.value.toFixed(2)}</span>`;
    wrap.appendChild(row);
  }
  host.appendChild(wrap);
}

/* ── Sparkline, returned as a string for table cells ──────── */
export function sparkline(values, opts = {}) {
  const { w = 88, h = 26 } = opts;
  if (!values || values.length < 2) return "";
  const lo = Math.min(...values), hi = Math.max(...values);
  const span = hi - lo || 1;
  const pts = values.map((v, i) =>
    `${(i / (values.length - 1) * w).toFixed(1)},${(h - 2 - ((v - lo) / span) * (h - 4)).toFixed(1)}`);
  const rising = values[values.length - 1] >= values[0];
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true">
    <polyline points="${pts.join(" ")}" fill="none" stroke="${rising ? "var(--up)" : "var(--down)"}"
      stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/></svg>`;
}
