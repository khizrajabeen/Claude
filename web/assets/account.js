import { shell, load, explain, usd, pct, num, cls, price, freshness, chipColour, base, ago , gated } from "./app.js";

/* The venue's own view, published beside the bot's ledger.
 *
 * Both are shown because a disagreement between them is the single most
 * useful thing this dashboard can report: it means something happened
 * that the bot does not know about — a manual trade, a partial fill, a
 * corporate action. A dashboard that silently shows only one of the two
 * cannot tell you that.
 */

await gated();
shell("account");
const $ = (id) => document.getElementById(id);

try {
  const [a, p] = await Promise.all([load("account"), load("portfolio").catch(() => null)]);
  const f = freshness(a.as_of);
  $("fresh").innerHTML = `<span class="dot-live ${f.state}"></span><span>${f.text}</span>`;
  $("market").textContent = a.market_open
    ? "US market open" : `US market closed${a.next_open ? ` — opens ${new Date(a.next_open).toUTCString().slice(5, 22)} UTC` : ""}`;

  const dayPnl = a.equity - (a.last_equity || a.equity);
  const tile = (l, v, sub, tone) => `<div class="card tile"><div class="label">${l}</div>
    <div class="value">${v}</div><div class="delta ${tone || "muted"}">${sub}</div></div>`;
  $("tiles").innerHTML =
    tile("Account equity", usd(a.equity),
         `${a.mode} · ${a.account_number}`, "muted") +
    tile("Since yesterday", usd(dayPnl),
         a.last_equity ? pct(dayPnl / a.last_equity * 100) : "—", cls(dayPnl)) +
    tile("Cash", usd(a.cash), `buying power ${usd(a.buying_power, 0)}`, "muted") +
    tile("Status", a.trading_blocked ? "BLOCKED" : a.status,
         a.shorting_enabled ? "shorting enabled" : "long only",
         a.trading_blocked ? "down" : "up");

  const pos = a.positions || [];
  $("poscount").textContent = `${pos.length} held`;
  $("pos").tBodies[0].innerHTML = pos.length ? pos.map((r) => `<tr>
      <td><span class="asset">
        <span class="coin" style="background:${chipColour(r.symbol)}">${base(r.symbol).slice(0, 3)}</span>
        <span><span class="nm">${r.symbol}</span><span class="tk">${r.asset_class}</span></span></span></td>
      <td class="r num">${num(r.qty, 6)}</td>
      <td class="r num">${price(r.avg_entry_price)}</td>
      <td class="r num">${price(r.current_price)}</td>
      <td class="r num">${usd(r.market_value)}</td>
      <td class="r num ${cls(r.unrealized_pl)}">${usd(r.unrealized_pl)}
        <span class="tk">${pct(r.unrealized_plpc, 2)}</span></td></tr>`).join("")
    : `<tr><td colspan="6" class="empty">Flat — the account holds nothing.</td></tr>`;

  const drift = a.drift || [];
  $("drift").innerHTML = drift.length
    ? `<div class="note warn"><b>The ledger and the account disagree.</b>
         Something moved outside the bot. Until this clears, its risk model is
         sizing against a book that is not the real one.</div>
       <ul style="margin:14px 0 0;padding-left:20px;font-size:13px;line-height:1.7">
         ${drift.map((d) => `<li class="num">${d}</li>`).join("")}</ul>`
    : `<p class="muted" style="font-size:13.5px">Every position the bot believes it
         holds is a position the account actually holds, in the same size.</p>
       ${p ? `<div style="margin-top:16px;display:flex;justify-content:space-between;
              padding-top:14px;border-top:1px solid var(--line-soft)">
              <span><div class="tk muted">Bot's ledger</div>
                <div class="num" style="font-size:15px;margin-top:3px">${usd(p.equity)}</div></span>
              <span style="text-align:right"><div class="tk muted">Venue</div>
                <div class="num" style="font-size:15px;margin-top:3px">${usd(a.equity)}</div></span>
              </div>` : ""}`;

  const ord = a.orders || [];
  $("ord").tBodies[0].innerHTML = ord.length ? ord.map((o) => `<tr>
      <td class="muted">${o.submitted_at ? ago(o.submitted_at) : "—"}</td>
      <td>${o.symbol}</td>
      <td><span class="pill ${o.side === "buy" ? "up" : "down"}">${o.side}</span></td>
      <td class="r num">${o.qty ?? "—"}</td>
      <td class="muted">${o.type || "—"}</td>
      <td><span class="pill">${o.status}</span></td></tr>`).join("")
    : `<tr><td colspan="6" class="empty">No orders working.</td></tr>`;
} catch (e) {
  explain(document.querySelector(".content"), e);
}
