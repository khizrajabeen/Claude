import {
  shell, load, explain, usd, pct, num, cls, arrow, price, compact,
  freshness, chipColour, base,
} from "./app.js";
import { lineChart, donut } from "./chart.js";

shell("dashboard");

const CLASS_NAME = {
  crypto_spot: "Crypto spot", crypto_perp: "Perpetuals",
  equity: "Equities", etf: "ETFs",
};
const $ = (id) => document.getElementById(id);

let daily;   // kept for the range buttons to redraw without refetching

function tile(label, value, delta, tone) {
  return `<div class="card tile">
    <div class="label">${label}</div>
    <div class="value">${value}</div>
    ${delta ? `<div class="delta ${tone || "muted"}">${delta}</div>` : ""}
  </div>`;
}

function drawCurve(days) {
  const rows = daily.curve.slice(-days);
  lineChart($("curve"),
    [{ name: "Equity", values: rows.map((r) => ({ x: r.day.slice(5), y: r.equity })) }],
    { fill: true, height: 280, fmt: (v) => usd(v),
      yFmt: (v) => compact(v),
      label: `Account equity over the last ${rows.length} recorded days` });
}

try {
  const [p, m, pos, d, strat, screen] = await Promise.all([
    load("portfolio"), load("meta"), load("positions"),
    load("daily"), load("strategies"), load("screen").catch(() => null),
  ]);
  daily = d;

  /* ── Header chrome ───────────────────────────────────── */
  const f = freshness(p.as_of);
  $("fresh").innerHTML = `<span class="dot-live ${f.state}"></span><span>${f.text}</span>`;
  $("mode").textContent = m.mode;
  $("avatar").textContent = m.mode === "live" ? "LV" : "PA";

  /* ── Stat tiles ──────────────────────────────────────── */
  $("tiles").innerHTML = [
    tile("Portfolio value", usd(p.equity),
         `${arrow(p.total_return_pct)} ${pct(p.total_return_pct)} from ${usd(p.starting_equity, 0)}`,
         cls(p.total_return_pct)),
    tile("Last session P&amp;L", usd(p.daily_pnl),
         `${arrow(p.daily_pnl)} ${pct(p.daily_return_pct)} on the day`, cls(p.daily_pnl)),
    tile("Max drawdown", num(p.max_drawdown_pct, 2) + "%",
         `peak ${usd(p.peak_equity, 0)}`, "muted"),
    tile("Win rate", num(p.win_rate, 1) + "%",
         `${p.trades} trades · PF ${num(p.profit_factor, 2)}`, "muted"),
  ].join("");

  /* ── Performance ─────────────────────────────────────── */
  $("badges").innerHTML =
    `Sharpe ${num(p.sharpe_daily, 2)} · expectancy ${num(p.expectancy_r, 3)}R`;
  $("legend").innerHTML =
    `<span><i style="background:var(--s1)"></i>Equity</span>
     <span class="muted">${d.period.from} → ${d.period.to}</span>
     <span class="muted">${d.winning_days}W / ${d.losing_days}L / ${d.flat_days} flat</span>`;
  drawCurve(90);

  // A Sharpe this high over ninety days is not what a real edge
  // produces; it is what a lucky window looks like, and the roster was
  // chosen by looking at this window. Saying so next to the number is
  // the difference between a report and a sales pitch.
  const n = daily.curve.length;
  $("caveat").textContent =
    `Measured over ${n} recorded days on ${p.trades} trades — a sample this ` +
    `short cannot separate edge from luck, and the strategy roster was ` +
    `chosen by looking at this same window. Treat the ratios as a ` +
    `description of what happened, not a forecast.`;
  $("range").onclick = (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    for (const x of $("range").children) x.setAttribute("aria-pressed", String(x === b));
    drawCurve(Number(b.dataset.days));
  };

  /* ── Open positions ──────────────────────────────────── */
  const rows = pos.positions || [];
  $("poscount").textContent = `${rows.length} of ${m.risk.max_open_positions}`;
  $("positions").innerHTML = rows.length ? rows.map((r) => {
    // Risk per unit is entry-to-stop in the losing direction. If that
    // comes out negative the stop sits on the wrong side of the entry,
    // which is a broken position, not a 68R winner. Say so instead of
    // dividing by it.
    const risk = r.side === "long" ? r.entry_price - r.stop_price
                                   : r.stop_price - r.entry_price;
    const broken = !(risk > 0);
    const target = broken ? null
      : Math.abs((r.take_profit - r.entry_price) / risk);
    return `<div class="row">
      <span class="asset">
        <span class="coin" style="background:${chipColour(r.symbol)}">${base(r.symbol).slice(0, 3)}</span>
        <span><span class="nm">${r.symbol}</span>
          <span class="tk">${CLASS_NAME[r.asset_class] || r.asset_class} · ${r.strategy}</span></span>
      </span>
      <span class="grow"></span>
      <span class="col"><div class="k num">${price(r.entry_price)}</div>
        <div class="v">${r.quantity} @ ${r.leverage}×</div></span>
      <span class="pill ${r.side === "long" ? "up" : "down"}">${r.side}</span>
      ${broken
        ? `<span class="pill warn" title="Stop is on the wrong side of the entry">stop inverted</span>`
        : `<span class="col"><div class="k num">${usd(r.risk_usd)}</div>
             <div class="v">risk · ${num(target, 1)}R target</div></span>`}
    </div>`;
  }).join("") : `<p class="empty">Flat — no positions open.</p>`;

  /* ── Allocation by risk factor ───────────────────────── */
  const byClass = Object.entries(d.by_asset_class);
  const risked = rows.reduce((a, r) => a + r.risk_usd, 0);
  // Slices are the SIZE of each class's contribution, signed in the
  // legend. Labelling |P&L| as "allocation" would be a lie with a
  // picture attached.
  donut($("donut"),
    byClass.map(([k, v]) => ({ name: CLASS_NAME[k] || k, value: Math.abs(v.pnl) })),
    { centreValue: usd(risked, 0), centreLabel: "at risk now",
      label: "Share of profit and loss by asset class" });
  $("donut-legend").innerHTML = byClass.map(([k, v], i) =>
    `<span style="display:flex;align-items:center;width:100%">
       <i style="background:var(--s${(i % 6) + 1})"></i>${CLASS_NAME[k] || k}
       <span class="grow" style="flex:1"></span>
       <span class="num ${cls(v.pnl)}">${usd(v.pnl, 0)}</span></span>`).join("");
  $("riskstats").innerHTML = [
    ["Heat cap", num(m.risk.max_portfolio_heat_pct, 1) + "%"],
    ["Per trade", num(m.risk.risk_per_trade_pct, 2) + "%"],
    ["Daily stop", "−" + num(m.risk.max_daily_loss_pct, 1) + "%"],
  ].map(([k, v]) => `<span><div class="tk muted">${k}</div>
     <div class="num" style="font-size:14px;margin-top:3px">${v}</div></span>`).join("");

  /* ── Top movers, from the screen ─────────────────────── */
  const mk = (screen?.markets || []).filter((x) => x.change_24h_pct != null);
  const sorted = [...mk].sort((a, b) => b.change_24h_pct - a.change_24h_pct);
  const col = (title, list, tone) =>
    `<div style="border-right:1px solid var(--line-soft)">
      <div class="tk muted" style="padding:12px 16px 6px;font-weight:640;
           letter-spacing:.06em;text-transform:uppercase;font-size:10.5px;color:var(--${tone})">${title}</div>
      ${list.map((r) => `<div class="row" style="padding:9px 16px">
        <span class="asset"><span class="coin" style="background:${chipColour(r.symbol)};
          width:24px;height:24px;font-size:9px">${base(r.symbol).slice(0, 3)}</span>
          <span class="nm" style="font-size:12.5px">${base(r.symbol)}</span></span>
        <span class="grow"></span>
        <span class="num ${cls(r.change_24h_pct)}" style="font-size:12.5px">${pct(r.change_24h_pct, 1)}</span>
      </div>`).join("")}
    </div>`;
  $("movers").innerHTML = mk.length
    ? col("Gainers", sorted.slice(0, 5), "up") +
      col("Losers", sorted.slice(-5).reverse(), "down").replace("border-right", "border-left")
    : `<p class="empty">Run the screener to populate movers.</p>`;

  /* ── Market intelligence ─────────────────────────────
   * News when there is news. When there is none, the honest
   * fallback is what the screen DID measure — breadth, how tightly
   * the market is following BTC, and what is newly listed. An empty
   * card is a wasted third of the row; an invented headline is worse. */
  const scored = (screen?.markets || [])
    .filter((r) => r.news_articles > 0)
    .sort((a, b) => Math.abs(b.news_score) - Math.abs(a.news_score))
    .slice(0, 6);

  if (scored.length) {
    $("newscount").textContent = `${scored.length} in the news`;
    $("news").innerHTML = scored.map((r) => `
      <div class="news">
        <div class="tag">${base(r.symbol)} · rank ${r.rank}${r.is_new ? " · new listing" : ""}</div>
        <p>${r.news_articles} article${r.news_articles === 1 ? "" : "s"} scored
           <b class="${cls(r.news_score)}">${num(r.news_score, 2)}</b>,
           beta ${num(r.beta, 2)} to BTC, 24h ${pct(r.change_24h_pct, 1)}.</p>
      </div>`).join("");
  } else if (mk.length) {
    const up = mk.filter((r) => r.change_24h_pct > 0).length;
    const tight = mk.filter((r) => r.beta_r2 > 0.5);
    const highBeta = [...tight].sort((a, b) => b.beta - a.beta).slice(0, 3);
    const fresh = mk.filter((r) => r.is_new).length;
    const btc = mk.find((r) => r.symbol.startsWith("BTC/"));
    $("newscount").textContent = "from the screen";
    $("news").innerHTML = `
      <div class="news"><div class="tag">Breadth</div>
        <p><b class="${cls(up * 2 - mk.length)}">${up} of ${mk.length}</b> screened markets
           are up over 24h${btc ? `, with BTC ${pct(btc.change_24h_pct, 1)}` : ""}.
           Breadth this wide means the move is the market, not the pick.</p></div>
      <div class="news"><div class="tag">Correlation to BTC</div>
        <p>${tight.length} markets track BTC closely enough to measure
           (r² &gt; 0.5). The highest beta:
           ${highBeta.map((r) => `<b>${base(r.symbol)} ${num(r.beta, 2)}</b>`).join(", ")}.
           Holding several of those is one bet in different wrappers.</p></div>
      <div class="news"><div class="tag">New listings</div>
        <p>${fresh || "No"} market${fresh === 1 ? "" : "s"} below the minimum age
           threshold${fresh ? " — excluded from entry until they have history to measure" : "."}</p></div>
      <div class="news"><div class="tag">News coverage</div>
        <p>No scored articles in the last screen. Add a news key on the
           settings page to widen per-coin coverage.</p></div>`;
  } else {
    $("news").innerHTML = `<p class="empty">Run the screener to populate this.</p>`;
  }
} catch (e) {
  explain(document.querySelector(".content"), e);
}
