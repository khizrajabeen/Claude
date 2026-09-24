import { shell, load, explain, usd, pct, num, cls, price, freshness, chipColour, base } from "./app.js";

shell("trades");
const $ = (id) => document.getElementById(id);
const CLASS_NAME = { crypto_spot: "Crypto spot", crypto_perp: "Perpetuals",
                     equity: "Equities", etf: "ETFs" };
const EXIT = { stop_loss: "Stop", take_profit: "Target", trailing_stop: "Trail",
               breakeven_stop: "Breakeven", time_stop: "Time", max_hold: "Max hold",
               forced_flat: "Flattened" };

let all = [], sort = { k: "closed_on_day", dir: -1 };

/* The tiles describe the FILTERED set, not the whole file. A header
 * that ignores the filter under it is worse than no header. */
function tiles(rows) {
  const pnl = rows.reduce((a, r) => a + r.pnl, 0);
  const wins = rows.filter((r) => r.pnl > 0);
  const gross = wins.reduce((a, r) => a + r.pnl, 0);
  const loss = Math.abs(rows.filter((r) => r.pnl <= 0).reduce((a, r) => a + r.pnl, 0));
  const avgR = rows.length ? rows.reduce((a, r) => a + (r.r_multiple || 0), 0) / rows.length : 0;
  const t = (l, v, tone, sub) => `<div class="card tile"><div class="label">${l}</div>
    <div class="value ${tone || ""}">${v}</div>${sub ? `<div class="delta muted">${sub}</div>` : ""}</div>`;
  $("tiles").innerHTML =
    t("Net P&amp;L", usd(pnl), cls(pnl), `${rows.length} of ${all.length} trades`) +
    t("Win rate", rows.length ? num(wins.length / rows.length * 100, 1) + "%" : "—", "",
      `${wins.length}W / ${rows.length - wins.length}L`) +
    t("Expectancy", num(avgR, 3) + "R", cls(avgR), "per trade, in risk units") +
    t("Profit factor", loss ? num(gross / loss, 2) : "—", "", `${usd(gross, 0)} won / ${usd(loss, 0)} lost`);
}

function render() {
  const fc = $("f-class").value, fs = $("f-strat").value, fe = $("f-exit").value;
  let rows = all.filter((r) =>
    (!fc || r.asset_class === fc) && (!fs || r.strategy === fs) && (!fe || r.exit_reason === fe));
  rows.sort((a, b) => {
    const x = a[sort.k], y = b[sort.k];
    return (typeof x === "number" ? x - y : String(x).localeCompare(String(y))) * sort.dir;
  });
  tiles(rows);
  $("tbl").tBodies[0].innerHTML = rows.length ? rows.map((r) => `<tr>
    <td class="muted">${r.closed_on_day}</td>
    <td><span class="asset"><span class="coin" style="background:${chipColour(r.symbol)}">${base(r.symbol).slice(0,3)}</span>
      <span><span class="nm">${r.symbol}</span><span class="tk">${CLASS_NAME[r.asset_class] || r.asset_class}</span></span></span></td>
    <td>${r.strategy}</td>
    <td><span class="pill ${r.side === "long" ? "up" : "down"}">${r.side}</span></td>
    <td class="r num">${price(r.entry_price)}</td>
    <td class="r num">${price(r.exit_price)}</td>
    <td class="r num ${cls(r.r_multiple)}">${num(r.r_multiple, 2)}</td>
    <td class="r num ${cls(r.pnl)}">${usd(r.pnl)}</td>
    <td class="muted">${EXIT[r.exit_reason] || r.exit_reason}</td></tr>`).join("")
    : `<tr><td colspan="9" class="empty">No trades match this filter.</td></tr>`;
}

function fill(sel, key, label) {
  const opts = [...new Set(all.map((r) => r[key]))].sort();
  sel.innerHTML = `<option value="">All ${label}</option>` +
    opts.map((o) => `<option value="${o}">${CLASS_NAME[o] || EXIT[o] || o}</option>`).join("");
}

try {
  const [t, p] = await Promise.all([load("trades"), load("portfolio")]);
  all = t.trades || t;
  const f = freshness(p.as_of);
  $("fresh").innerHTML = `<span class="dot-live ${f.state}"></span><span>${f.text}</span>`;

  fill($("f-class"), "asset_class", "classes");
  fill($("f-strat"), "strategy", "strategies");
  fill($("f-exit"), "exit_reason", "exits");
  for (const el of [$("f-class"), $("f-strat"), $("f-exit")]) el.onchange = render;
  $("reset").onclick = () => {
    for (const el of [$("f-class"), $("f-strat"), $("f-exit")]) el.value = "";
    render();
  };
  $("tbl").tHead.onclick = (e) => {
    const th = e.target.closest("th.sortable");
    if (!th) return;
    sort = { k: th.dataset.k, dir: sort.k === th.dataset.k ? -sort.dir : -1 };
    render();
  };
  render();
} catch (e) { explain(document.querySelector(".content"), e); }
