import { shell, load, explain, usd, pct, num, cls, freshness , gated } from "./app.js";
import { lineChart, barList } from "./chart.js";

await gated();
shell("journal");
const $ = (id) => document.getElementById(id);
const CLASS_NAME = { crypto_spot: "Crypto spot", crypto_perp: "Perpetuals",
                     equity: "Equities", etf: "ETFs" };
const EXIT = { stop_loss: "Stopped out", take_profit: "Target hit",
               trailing_stop: "Trailing stop", breakeven_stop: "Breakeven stop",
               time_stop: "Time stop", max_hold: "Max hold", forced_flat: "Flattened" };

try {
  const [d, p, t] = await Promise.all([load("daily"), load("portfolio"), load("trades")]);
  const f = freshness(d.as_of);
  $("fresh").innerHTML = `<span class="dot-live ${f.state}"></span><span>${f.text}</span>`;
  $("period").textContent = `${d.period.from} → ${d.period.to}`;

  const curve = d.curve || [];
  const best = [...curve].sort((a, b) => b.realized_pnl - a.realized_pnl)[0];
  const worst = [...curve].sort((a, b) => a.realized_pnl - b.realized_pnl)[0];
  const tile = (l, v, sub, tone) => `<div class="card tile"><div class="label">${l}</div>
    <div class="value">${v}</div><div class="delta ${tone || "muted"}">${sub}</div></div>`;
  $("tiles").innerHTML =
    tile("Days recorded", curve.length,
         `${d.winning_days}W / ${d.losing_days}L / ${d.flat_days} flat`) +
    tile("Best day", best ? usd(best.realized_pnl) : "—", best?.day || "", "up") +
    tile("Worst day", worst ? usd(worst.realized_pnl) : "—", worst?.day || "", "down") +
    tile("Hit rate", curve.length
         ? num(d.winning_days / (d.winning_days + d.losing_days || 1) * 100, 1) + "%"
         : "—", "of days that traded");

  // Daily realised P&L, as a line rather than 90 bars: at this density
  // bars become a picket fence and the shape is what matters.
  lineChart($("bars"),
    [{ name: "Realised", values: curve.map((r) => ({ x: r.day.slice(5), y: r.realized_pnl })) }],
    { height: 260, yZero: true, fmt: (v) => usd(v), yFmt: (v) => usd(v, 0),
      label: "Realised profit and loss per day" });

  const exits = {};
  for (const r of (t.trades || t)) exits[r.exit_reason] = (exits[r.exit_reason] || 0) + 1;
  barList($("exits"),
    Object.entries(exits).map(([k, v]) => ({
      name: EXIT[k] || k, value: v,
      color: k === "take_profit" ? "var(--up)" : k === "stop_loss" ? "var(--down)" : "var(--s2)",
    })).sort((a, b) => b.value - a.value),
    { fmt: (v) => `${v} trades` });

  $("tbl").tBodies[0].innerHTML = [...curve].reverse().map((r) => `<tr>
      <td>${r.day}</td>
      <td class="r num ${cls(r.realized_pnl)}">${usd(r.realized_pnl)}</td>
      <td class="r num ${cls(r.return_pct)}">${pct(r.return_pct)}</td>
      <td class="r num">${usd(r.equity)}</td>
      <td class="muted" style="font-size:12.5px">${
        Object.entries(r.by_class || {}).map(([k, v]) =>
          `${CLASS_NAME[k] || k} <span class="num ${cls(v)}">${usd(v, 0)}</span>`).join(" · ") || "—"
      }</td></tr>`).join("");
} catch (e) {
  explain(document.querySelector(".content"), e);
}
