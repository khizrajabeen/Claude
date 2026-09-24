"""Build the tradable universe from what Alpaca actually charges.

The universe used to be a hand-written list of fifteen pairs copied from
a crypto exchange's volume ranking. Two things were wrong with that. The
pairs were quoted against USDT on OKX while the account trades USD on
Alpaca, and the cost of trading each one was assumed to be the same 5.5
basis points.

Neither survives contact with measurement. Alpaca lists 33 non-stablecoin
USD pairs, and their spreads run from 1.6 bps on BTC to 65 bps on PAXG —
a fortyfold range. Add the 25 bps taker fee on each side and the round
trip costs between 52 and 115 bps depending only on which coin was
picked.

That range is the whole argument about how many coins to trade. The
measured edge is about +0.275R per trade, and with a 2.5-ATR stop on
crypto 1R is roughly 300 bps of notional, so a winning trade grosses
something like 80 bps. A coin whose round trip costs 110 bps cannot pay
for itself no matter how good the signal is; a coin that costs 52 bps
keeps most of the edge.

So this module does not pick a number of coins. It measures each one and
hands the real cost to the instrument, and the bot's existing edge gate
then refuses individual trades that cannot clear their own cost. Widening
the universe is safe precisely because the cost gate is real.
"""

from __future__ import annotations

import logging
import os
import urllib.parse

logger = logging.getLogger(__name__)

QUOTES_URL = "https://data.alpaca.markets/v1beta3/crypto/us/latest/quotes"
BARS_URL = "https://data.alpaca.markets/v1beta3/crypto/us/bars"

# Not coins. Holding a dollar against a dollar has no trend to trade.
STABLECOINS = {"USDC", "USDG", "USDT", "DAI", "PYUSD"}

# Alpaca's published crypto commission, per side, at the base tier.
TAKER_BPS = 25.0
MAKER_BPS = 15.0


def _session(broker):
    return broker.session()


def measure(broker, min_depth_usd: float = 50.0,
            max_round_trip_bps: float = 160.0) -> list[dict]:
    """Every USD crypto pair Alpaca will trade, with its measured cost.

    Cost is measured from the spread, not from depth at the touch. Depth
    looked like the obvious liquidity filter and is actively misleading:
    the top of book is quoted in coin units, so BTC shows about $83
    resting at a 1.6 bps spread while BAT shows $1 at 59 bps. A $250
    depth floor excluded BTC, ETH and SOL — the three most liquid assets
    on the venue — and kept nothing useful. `min_depth_usd` is now only
    a guard against a quote that is plainly broken.

    `max_round_trip_bps` is a backstop against pairs no strategy could
    pay for. It is deliberately loose, because the per-trade edge gate is
    the real filter: it compares each signal's expected move against that
    instrument's own cost, which is exactly the comparison a fixed
    universe cut cannot make.
    """
    symbols = sorted(
        s for s in broker.tradable()
        if s.endswith("/USD") and s.split("/")[0] not in STABLECOINS
    )
    if not symbols:
        return []

    s = _session(broker)
    quotes = s.get(QUOTES_URL,
                   params={"symbols": ",".join(symbols)},
                   timeout=40).json().get("quotes", {})

    out: list[dict] = []
    for symbol, q in quotes.items():
        ask, bid = float(q.get("ap") or 0), float(q.get("bp") or 0)
        if not (ask > 0 and bid > 0 and ask >= bid):
            continue
        mid = (ask + bid) / 2
        spread_bps = (ask - bid) / mid * 10_000
        depth = min(float(q.get("as") or 0) * ask, float(q.get("bs") or 0) * bid)

        # Crossing the spread costs half of it per side, and the fee is
        # charged on both. That sum is what a round trip actually costs.
        half_spread = spread_bps / 2
        round_trip = spread_bps + TAKER_BPS * 2

        if depth < min_depth_usd or round_trip > max_round_trip_bps:
            logger.debug("skipping %s: spread %.1f bps, $%.0f at touch",
                         symbol, spread_bps, depth)
            continue

        out.append({
            "symbol": symbol,
            "asset_class": "crypto_spot",
            "venue": "alpaca",
            "fee_bps": TAKER_BPS,
            "slippage_bps": round(half_spread, 2),
            "price": round(mid, 8),
            "spread_bps": round(spread_bps, 2),
            "depth_usd": round(depth, 2),
            "round_trip_bps": round(round_trip, 2),
        })

    out.sort(key=lambda r: r["round_trip_bps"])
    return out


def to_config_block(rows: list[dict]) -> str:
    """The measured universe as YAML, ready to paste into config."""
    lines = []
    for r in rows:
        lines.append(
            f'    - {{symbol: "{r["symbol"]}", asset_class: "crypto_spot", '
            f'venue: "alpaca", fee_bps: {r["fee_bps"]}, '
            f'slippage_bps: {r["slippage_bps"]}}}'
            f'   # {r["round_trip_bps"]:.0f} bps round trip'
        )
    return "\n".join(lines)


def render(rows: list[dict]) -> str:
    if not rows:
        return "No tradable pairs measured."
    w = max(len(r["symbol"]) for r in rows)
    head = f"{'pair'.ljust(w)}  {'spread':>8} {'round trip':>11} {'$ at touch':>12}"
    body = "\n".join(
        f"{r['symbol'].ljust(w)}  {r['spread_bps']:7.1f}  {r['round_trip_bps']:10.0f}  "
        f"{r['depth_usd']:12,.0f}"
        for r in rows
    )
    return f"{head}\n{'-' * len(head)}\n{body}"
