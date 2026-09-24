import { shell, load, explain, usd, num, cls, freshness } from "./app.js";
import { barList } from "./chart.js";

shell("strategies");
const $ = (id) => document.getElementById(id);
const CLASS_NAME = { crypto_spot: "Crypto spot", crypto_perp: "Perpetuals",
                     equity: "Equities", etf: "ETFs" };

try {
  const [s, m, d] = await Promise.all([load("strategies"), load("meta"), load("daily")]);
  const f = freshness(s.as_of);
  $("fresh").innerHTML = `<span class="dot-live ${f.state}"></span><span>${f.text}</span>`;
  $("period").textContent = `${d.period.from} → ${d.period.to}`;

  const live = new Set(m.strategies || []);
  const entries = Object.entries(s.strategies)
    .sort((a, b) => (live.has(b[0]) - live.has(a[0])) || b[1].avg_r - a[1].avg_r);

  $("cards").innerHTML = entries.map(([name, v]) => `
    <section class="card">
      <div class="card-head"><h2 style="text-transform:capitalize">${name}</h2>
        <span class="grow"></span>
        <span class="pill ${live.has(name) ? "up" : ""}">${live.has(name) ? "enabled" : "benched"}</span>
      </div>
      <div class="card-body">
        <div class="grid cols-2" style="gap:14px">
          <div><div class="label muted" style="font-size:12px">Expectancy</div>
            <div class="value num ${cls(v.avg_r)}" style="font-size:22px">${num(v.avg_r, 3)}R</div></div>
          <div><div class="label muted" style="font-size:12px">P&amp;L</div>
            <div class="value num ${cls(v.pnl)}" style="font-size:22px">${usd(v.pnl, 0)}</div></div>
        </div>
        <div class="legend" style="margin-top:14px">
          <span>${v.trades} trades</span><span>${num(v.win_rate, 1)}% win</span>
          <span>${num(v.r_sum, 1)}R total</span>
        </div>
      </div>
    </section>`).join("");

  $("tbl").tBodies[0].innerHTML = Object.entries(d.by_asset_class).map(([k, v]) => {
    // Trade count is the sum of how the trades ended — the only
    // place the file records it per class.
    const trades = Object.values(v.exit_reasons || {}).reduce((a, b) => a + b, 0);
    return `<tr>
      <td>${CLASS_NAME[k] || k}</td>
      <td class="r num">${trades}</td>
      <td class="r num ${cls(v.pnl)}">${usd(v.pnl, 0)}</td>
      <td class="r num ${cls(v.expectancy_r)}">${num(v.expectancy_r, 3)}R</td>
      <td class="r num muted">±${num(v.se_r, 3)}</td>
      <td class="r num">${num(v.t_stat, 2)}</td>
      <td class="r num">${num(v.profit_factor, 2)}</td>
      <td>${v.significant
            ? `<span class="pill up">yes, |t| &gt; ${num(v.t_critical, 2)}</span>`
            : `<span class="pill">no — |t| &lt; ${num(v.t_critical, 2)}</span>`}</td></tr>`;
  }).join("");
} catch (e) { explain(document.querySelector(".content"), e); }
