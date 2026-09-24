/* Live prices, straight from the browser.
 *
 * The constraint that shapes this file: the dashboard is a static page
 * on a public host. It has no server, so it cannot hold a secret, so it
 * cannot call any API that needs a key. An Alpaca key pasted in here
 * would be readable by anyone who opened the network tab.
 *
 * Crypto is fine — the major venues serve public, keyless, CORS-enabled
 * ticker endpoints, and this polls them directly. Stocks are not: every
 * equity feed worth using requires a key, so equity prices come from
 * `data/quotes.json`, written by the machine that holds the key, and are
 * stamped with their age rather than presented as live.
 *
 * Providers are tried in order. Binance is first because it is the
 * deepest book, and it returns 451 in several countries — which is not
 * an exception to log and swallow but the ordinary case for a lot of
 * readers, so the failover is the design rather than a fallback.
 */

/* Only two venues qualify, and which two was decided by measurement
 * rather than by reputation.
 *
 * A browser on a public page can only use an endpoint that is keyless,
 * bulk (one request, not thirty), and sends an
 * Access-Control-Allow-Origin header. That last one eliminates most of
 * the obvious candidates, silently: the request goes out and the
 * browser refuses the response.
 *
 *   Binance   no CORS header, and 451 from several countries
 *   KuCoin    no CORS header
 *   Coinbase  has no bulk ticker endpoint at all; /products/tickers
 *             is a 404, and per-product would be thirty requests
 *   OKX       CORS, 406 USDT pairs in one request
 *   Kraken    CORS, 1,479 pairs in one request
 *
 * Checked against the live endpoints, not against documentation.
 */
const PROVIDERS = [
  {
    name: "OKX",
    url: "https://www.okx.com/api/v5/market/tickers?instType=SPOT",
    parse: (body) => {
      const out = {};
      for (const r of body.data || []) {
        const [base, quote] = r.instId.split("-");
        if (quote !== "USDT") continue;
        const open = +r.open24h;
        out[base] = {
          price: +r.last,
          change: open ? (+r.last - open) / open * 100 : 0,
          volume: +r.volCcy24h,
        };
      }
      return out;
    },
  },
  {
    name: "Kraken",
    url: "https://api.kraken.com/0/public/Ticker",
    parse: (body) => {
      const out = {};
      for (const [pair, r] of Object.entries(body.result || {})) {
        if (!pair.endsWith("USD")) continue;
        // Kraken's own spelling is historical: XXBTZUSD is BTC/USD, and
        // the X/Z prefixes mark assets listed before 2018. Strip them,
        // and map XBT to the name every other venue uses.
        let base = pair.slice(0, -3).replace(/Z$/, "");
        if (base.length > 3 && base.startsWith("X")) base = base.slice(1);
        if (base === "XBT") base = "BTC";
        const last = +r.c?.[0];
        const open = +r.o;
        if (!last) continue;
        out[base] = {
          price: last,
          change: open ? (last - open) / open * 100 : 0,
          volume: +r.v?.[1] * last,      // 24h base volume, in dollars
        };
      }
      return out;
    },
  },
];

const listeners = new Set();
let state = { prices: {}, source: null, at: null, error: null };
let timer = null;

export const snapshot = () => state;

async function fetchFrom(p) {
  const res = await fetch(p.url, { cache: "no-store", mode: "cors" });
  if (!res.ok) throw new Error(`${p.name} HTTP ${res.status}`);
  const prices = p.parse(await res.json());
  if (!Object.keys(prices).length) throw new Error(`${p.name} returned nothing`);
  return prices;
}

async function poll() {
  // Try whichever provider worked last, first. A venue geo-blocked for
  // this reader stays blocked, and re-testing it every ten seconds costs
  // a visible stall on every tick.
  const ordered = state.source
    ? [...PROVIDERS].sort((a, b) => (b.name === state.source) - (a.name === state.source))
    : PROVIDERS;

  for (const p of ordered) {
    try {
      const prices = await fetchFrom(p);
      state = { prices, source: p.name, at: Date.now(), error: null };
      listeners.forEach((fn) => fn(state));
      return;
    } catch {
      if (p === ordered[ordered.length - 1]) {
        state = { ...state, error: "No public price feed is reachable from this network." };
        listeners.forEach((fn) => fn(state));
      }
    }
  }
}

/** Subscribe to live prices. Returns an unsubscribe function. */
export function subscribe(fn, { intervalMs = 10_000 } = {}) {
  listeners.add(fn);
  if (state.at) fn(state);
  if (!timer) {
    poll();
    timer = setInterval(poll, intervalMs);
    // Polling a hidden tab burns the reader's battery and the venue's
    // rate limit to update pixels nobody is looking at.
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) { clearInterval(timer); timer = null; }
      else if (!timer) { poll(); timer = setInterval(poll, intervalMs); }
    });
  }
  return () => {
    listeners.delete(fn);
    if (!listeners.size && timer) { clearInterval(timer); timer = null; }
  };
}

/** The base ticker of one of the bot's symbols: BTC/USD -> BTC. */
export const ticker = (symbol) => (symbol || "").split("/")[0].replace(/:.*/, "");
