import { shell, load, usd, pct, num, cls, price, compact, base, chipColour, ago , gated } from "./app.js";
import { subscribe, ticker } from "./feed.js";

/* The one page that is genuinely live.
 *
 * Everywhere else reads a file the bot wrote. Here the browser polls a
 * public exchange feed itself. Stocks cannot work that way — every
 * equity feed needs a key, and a key on a public page is a published
 * key — so their price comes from quotes.json and is labelled with its
 * age rather than dressed up as real time.
 */

await gated();
shell("live");
const $ = (id) => document.getElementById(id);

let universe = [], published = {}, publishedAt = null, live = {}, prev = {};

function render() {
  const q = $("q").value.trim().toUpperCase();
  const rows = universe.filter((u) => !q || u.symbol.toUpperCase().includes(q));

  const withLive = rows.filter((u) => live[ticker(u.symbol)]);
  const up = withLive.filter((u) => live[ticker(u.symbol)].change > 0).length;
  const vol = withLive.reduce((a, u) => a + (live[ticker(u.symbol)].volume || 0), 0);
  const moves = withLive.map((u) => live[ticker(u.symbol)].change).sort((a, b) => b - a);

  const tile = (l, v, sub, tone) => `<div class="card tile"><div class="label">${l}</div>
    <div class="value ${tone || ""}">${v}</div><div class="delta muted">${sub}</div></div>`;
  $("tiles").innerHTML =
    tile("Markets live", `${withLive.length}/${rows.length}`,
         withLive.length ? "streaming now" : "waiting for a feed") +
    tile("Breadth", withLive.length ? `${up}/${withLive.length}` : "—",
         "up over 24h", cls(up * 2 - withLive.length)) +
    tile("Best 24h", moves.length ? pct(moves[0], 1) : "—", "across the universe", "up") +
    tile("24h volume", vol ? compact(vol) : "—", "summed across the feed");

  $("tbl").tBodies[0].innerHTML = rows.map((u) => {
    const t = ticker(u.symbol);
    const l = live[t];
    const was = published[u.symbol]?.price;
    const moved = l && prev[t] != null && l.price !== prev[t];
    return `<tr>
      <td><span class="asset">
        <span class="coin" style="background:${chipColour(u.symbol)}">${t.slice(0, 3)}</span>
        <span><span class="nm">${u.symbol}</span><span class="tk">${u.venue}</span></span></span></td>
      <td class="r num" ${moved ? `style="color:${l.price > prev[t] ? "var(--up)" : "var(--down)"}"` : ""}>
        ${l ? price(l.price) : '<span class="muted">—</span>'}</td>
      <td class="r num ${l ? cls(l.change) : "muted"}">${l ? pct(l.change, 2) : "—"}</td>
      <td class="r num muted">${l?.volume ? compact(l.volume) : "—"}</td>
      <td class="r num muted">${was ? price(was) : "—"}</td>
      <td class="r num ${u.round_trip > 90 ? "down" : ""}">${u.round_trip ? u.round_trip.toFixed(0) + " bps" : "—"}</td>
    </tr>`;
  }).join("") || `<tr><td colspan="6" class="empty">Nothing matches.</td></tr>`;

  for (const [t, v] of Object.entries(live)) prev[t] = v.price;
}

try {
  const [meta, quotes] = await Promise.all([load("meta"), load("quotes").catch(() => null)]);
  published = quotes?.quotes || {};
  publishedAt = quotes?.as_of || null;

  // The universe, with each instrument's measured cost to trade — the
  // number that decides whether a signal on it is worth acting on.
  const costs = meta.instrument_costs || {};
  universe = Object.entries(published).length
    ? Object.entries(published).map(([symbol, v]) => ({
        symbol, venue: v.venue, round_trip: costs[symbol] || 0 }))
    : [...(meta.universe?.crypto_spot || [])].map((symbol) => ({ symbol, venue: "alpaca" }));

  $("fresh").innerHTML = `<span class="dot-live"></span><span>live feed</span>`;
  $("q").oninput = render;
  render();

  subscribe(({ prices, source, at, error }) => {
    live = prices || {};
    $("tick").textContent = at ? `${source} · ${ago(new Date(at).toISOString())}` : "connecting…";
    $("source").innerHTML = error
      ? `<b>${error}</b> Crypto venues are blocked on some networks and in some
         countries. The bot's own prices, taken on the server, are still shown
         in the last-price column${publishedAt ? ` — ${ago(publishedAt)}` : ""}.`
      : `Prices stream from <b>${source}</b> straight to your browser, refreshed every
         10 seconds. Stock prices cannot work this way — every equity feed needs an
         API key, and a key on a public page is a published key — so those come from
         the bot's own reading${publishedAt ? `, taken ${ago(publishedAt)}` : ""}.`;
    render();
  });
} catch (e) {
  $("source").innerHTML = `<b>${e.message}</b>`;
}
