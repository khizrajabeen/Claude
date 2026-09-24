"""The browser-side price feed.

Only two public venues qualify for a static page: keyless, bulk, and
sending an Access-Control-Allow-Origin header. The third condition is
the one that eliminates most candidates, and it does so silently — the
request goes out and the browser refuses the response, so the page shows
nothing and the console shows a CORS error nobody reads.

These fixtures were captured from the live endpoints. They exist so a
venue changing its response shape fails here rather than on the
dashboard.
"""

import json
import pathlib
import re

import pytest

FEED = pathlib.Path(__file__).resolve().parents[1] / "web" / "assets" / "feed.js"


def providers():
    """The provider table as declared, without running a browser."""
    src = FEED.read_text()
    block = src[src.index("const PROVIDERS = ["):src.index("const listeners")]
    return block


def test_only_cors_capable_venues_are_configured():
    """Binance and KuCoin send no CORS header; Coinbase has no bulk
    endpoint. Configuring any of them means a page that silently shows
    nothing."""
    block = providers()
    for rejected in ("binance.com", "kucoin.com", "api.exchange.coinbase.com"):
        assert rejected not in block, f"{rejected} cannot be used from a browser"
    assert "okx.com" in block
    assert "api.kraken.com" in block


def test_every_provider_is_keyless():
    """A key in this file is a published key: the page is static and
    public, and anyone can read the source."""
    block = providers()
    assert not re.search(r"(api[_-]?key|secret|token)\s*[:=]", block, re.I)


def test_kraken_historical_prefixes_are_normalised():
    """XXBTZUSD is BTC/USD. The X and Z prefixes mark assets listed
    before 2018, and XBT is Kraken's name for Bitcoin. Left alone, the
    page looks up 'XXBT' against a universe that calls it 'BTC' and
    every Kraken price goes missing."""
    block = providers()
    assert 'base === "XBT"' in block or '"XBT"' in block
    assert "replace(/Z$/" in block


def test_the_universe_symbols_are_reducible_to_a_bare_ticker():
    """The feed is keyed by ticker; the bot's universe is keyed by pair."""
    src = (FEED.read_text())
    assert "export const ticker" in src
    # BTC/USD -> BTC, SOL/USDT:USDT -> SOL
    fn = re.search(r'export const ticker = \(symbol\) =>\s*(.+?);', src, re.S).group(1)
    assert 'split("/")' in fn and "replace" in fn


def test_polling_stops_when_the_tab_is_hidden():
    """Otherwise it burns the reader's battery and the venue's rate
    limit to update pixels nobody is looking at."""
    assert "visibilitychange" in FEED.read_text()


def test_the_last_working_provider_is_tried_first():
    """A venue geo-blocked for this reader stays blocked, and retesting
    it every ten seconds costs a visible stall on every tick."""
    assert "state.source" in FEED.read_text()
