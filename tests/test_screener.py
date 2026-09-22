"""Choosing what to look at, before deciding what to trade.

A static universe goes stale in two directions: it holds names whose
liquidity has drained away, and it misses the ones now dominating the
tape. A live scan of OKX's 401 active USDT markets put PEPE, ZEC and HYPE
in the top ten by turnover, none of which were in the configured list.

The screen decides what is worth looking at. That is a smaller claim than
deciding to trade, and these tests hold it to the smaller claim.
"""

from datetime import datetime, timedelta, timezone

import pytest

from bot.markets.screener import STABLECOINS, CoinScreener, render

NOW = datetime(2026, 9, 22, tzinfo=timezone.utc)


def ms(days_ago: float) -> str:
    return str(int((NOW - timedelta(days=days_ago)).timestamp() * 1000))


class FakeExchange:
    """An exchange whose markets and turnover the test dictates."""

    def __init__(self, markets: dict, tickers: dict, fail_batch=False):
        self._markets = markets
        self._tickers = tickers
        self.fail_batch = fail_batch
        self.ticker_calls = 0

    def load_markets(self):
        return self._markets

    def fetch_tickers(self, symbols=None):
        self.ticker_calls += 1
        if self.fail_batch:
            raise RuntimeError("no batch endpoint")
        return {s: t for s, t in self._tickers.items()
                if symbols is None or s in symbols}


def market(symbol, base=None, quote="USDT", spot=True, active=True,
           age_days=500.0):
    return {
        "symbol": symbol, "base": base or symbol.split("/")[0], "quote": quote,
        "spot": spot, "active": active, "created": ms(age_days), "info": {},
    }


def ticker(volume, last=100.0, change=0.0):
    return {"quoteVolume": volume, "last": last, "percentage": change}


def _exchange(rows):
    """rows: [(symbol, volume, age_days)]"""
    markets, tickers = {}, {}
    for symbol, volume, age in rows:
        markets[symbol] = market(symbol, age_days=age)
        tickers[symbol] = ticker(volume)
    return FakeExchange(markets, tickers)


# ── Ranking ──────────────────────────────────────────────────

def test_markets_are_ranked_by_turnover(config):
    exchange = _exchange([("A/USDT", 5e6, 500), ("B/USDT", 50e6, 500),
                          ("C/USDT", 20e6, 500)])
    found = CoinScreener(config, exchange=exchange).scan(NOW)
    assert [c.symbol for c in found] == ["B/USDT", "C/USDT", "A/USDT"]
    assert [c.rank for c in found] == [1, 2, 3]


def test_the_shortlist_is_the_top_n_that_qualify(config):
    config["screen"] = {"top_n": 2, "min_quote_volume_24h": 1e6,
                        "min_age_days": 0}
    exchange = _exchange([("A/USDT", 5e6, 500), ("B/USDT", 50e6, 500),
                          ("C/USDT", 20e6, 500)])
    shortlist = CoinScreener(config, exchange=exchange).shortlist(NOW)
    assert [c.symbol for c in shortlist] == ["B/USDT", "C/USDT"]


def test_a_market_with_no_turnover_is_not_ranked(config):
    exchange = FakeExchange({"A/USDT": market("A/USDT")},
                            {"A/USDT": ticker(0.0)})
    assert CoinScreener(config, exchange=exchange).scan(NOW) == []


# ── What is not a candidate ──────────────────────────────────

def test_a_stablecoin_pair_is_not_a_position(config):
    """USDC/USDT will rank high on turnover and never move."""
    markets = {"USDC/USDT": market("USDC/USDT", base="USDC"),
               "BTC/USDT": market("BTC/USDT")}
    tickers = {"USDC/USDT": ticker(900e6), "BTC/USDT": ticker(100e6)}
    found = CoinScreener(config, exchange=FakeExchange(markets, tickers)).scan(NOW)
    assert [c.symbol for c in found] == ["BTC/USDT"]


def test_every_listed_stablecoin_is_excluded(config):
    markets, tickers = {}, {}
    for coin in list(STABLECOINS)[:5]:
        if coin == "USDT":
            continue
        symbol = f"{coin}/USDT"
        markets[symbol] = market(symbol, base=coin)
        tickers[symbol] = ticker(100e6)
    found = CoinScreener(config, exchange=FakeExchange(markets, tickers)).scan(NOW)
    assert found == []


def test_another_quote_currency_is_skipped(config):
    markets = {"BTC/EUR": market("BTC/EUR", quote="EUR"),
               "BTC/USDT": market("BTC/USDT")}
    tickers = {"BTC/EUR": ticker(100e6), "BTC/USDT": ticker(50e6)}
    found = CoinScreener(config, exchange=FakeExchange(markets, tickers)).scan(NOW)
    assert [c.symbol for c in found] == ["BTC/USDT"]


def test_a_delisted_or_derivative_market_is_skipped(config):
    markets = {"OLD/USDT": market("OLD/USDT", active=False),
               "PERP/USDT": market("PERP/USDT", spot=False),
               "BTC/USDT": market("BTC/USDT")}
    tickers = {s: ticker(100e6) for s in markets}
    found = CoinScreener(config, exchange=FakeExchange(markets, tickers)).scan(NOW)
    assert [c.symbol for c in found] == ["BTC/USDT"]


def test_an_excluded_name_is_reported_not_dropped(config):
    """The screen says why, rather than silently omitting."""
    config["screen"] = {"exclude": ["DOGE"], "min_quote_volume_24h": 0,
                        "min_age_days": 0}
    exchange = _exchange([("DOGE/USDT", 100e6, 500)])
    found = CoinScreener(config, exchange=exchange).scan(NOW)
    assert found and not found[0].tradable
    assert "excluded" in found[0].notes[0]


# ── Liquidity and age ────────────────────────────────────────

def test_thin_turnover_disqualifies_but_still_reports(config):
    config["screen"] = {"min_quote_volume_24h": 10e6, "min_age_days": 0}
    exchange = _exchange([("THIN/USDT", 1e6, 500)])
    found = CoinScreener(config, exchange=exchange).scan(NOW)
    assert not found[0].tradable
    assert "turnover" in found[0].notes[0]


def test_a_young_market_is_flagged_as_new(config):
    config["screen"] = {"new_listing_days": 30, "min_age_days": 90,
                        "min_quote_volume_24h": 0}
    exchange = _exchange([("NEW/USDT", 50e6, 7)])
    found = CoinScreener(config, exchange=exchange).scan(NOW)
    assert found[0].is_new
    assert found[0].age_days == pytest.approx(7, abs=0.5)


def test_a_market_too_young_to_measure_is_not_tradable(config):
    """A market listed last week has no history to measure a trend, a
    beta or a volatility on."""
    config["screen"] = {"min_age_days": 90, "min_quote_volume_24h": 0}
    exchange = _exchange([("NEW/USDT", 50e6, 7)])
    found = CoinScreener(config, exchange=exchange).scan(NOW)
    assert not found[0].tradable
    assert "history" in found[0].notes[0]


def test_new_listings_can_be_admitted_deliberately(config):
    config["screen"] = {"min_age_days": 90, "new_listing_days": 30,
                        "include_new_listings": True, "min_quote_volume_24h": 0}
    exchange = _exchange([("NEW/USDT", 50e6, 7)])
    found = CoinScreener(config, exchange=exchange).scan(NOW)
    assert found[0].tradable


def test_an_old_market_is_not_called_new(config):
    config["screen"] = {"min_age_days": 90, "min_quote_volume_24h": 0}
    exchange = _exchange([("BTC/USDT", 50e6, 2000)])
    found = CoinScreener(config, exchange=exchange).scan(NOW)
    assert not found[0].is_new and found[0].tradable


def test_an_unknown_listing_date_does_not_disqualify(config):
    """Not every venue reports one, and absence is not youth."""
    config["screen"] = {"min_age_days": 90, "min_quote_volume_24h": 0}
    markets = {"X/USDT": {**market("X/USDT"), "created": None, "info": {}}}
    exchange = FakeExchange(markets, {"X/USDT": ticker(50e6)})
    found = CoinScreener(config, exchange=exchange).scan(NOW)
    assert found[0].age_days is None
    assert found[0].tradable


def test_a_listing_date_in_the_venue_info_is_used(config):
    config["screen"] = {"min_age_days": 90, "min_quote_volume_24h": 0}
    markets = {"X/USDT": {**market("X/USDT"), "created": None,
                          "info": {"listTime": ms(5)}}}
    exchange = FakeExchange(markets, {"X/USDT": ticker(50e6)})
    found = CoinScreener(config, exchange=exchange).scan(NOW)
    assert found[0].age_days == pytest.approx(5, abs=0.5)
    assert found[0].is_new


# ── Failure modes ────────────────────────────────────────────

def test_no_exchange_returns_nothing_rather_than_raising(config):
    assert CoinScreener(config, exchange=None).scan(NOW) == []


def test_an_exchange_that_cannot_load_markets_is_survivable(config):
    class Broken:
        def load_markets(self):
            raise RuntimeError("venue down")
    assert CoinScreener(config, exchange=Broken()).scan(NOW) == []


def test_tickers_failing_yields_an_empty_screen_not_a_crash(config):
    exchange = _exchange([("A/USDT", 5e6, 500)])
    exchange.fail_batch = True
    assert CoinScreener(config, exchange=exchange).scan(NOW) == []


# ── News and beta ────────────────────────────────────────────

class FakeNews:
    def __init__(self, scores):
        self.scores = scores
        self.asked = []

    def sentiment_for(self, symbol, hours, now=None):
        self.asked.append(symbol)
        return self.scores.get(symbol, {"score": 0.0, "articles": 0})


def test_news_is_attached_to_the_candidates(config):
    config["screen"] = {"min_quote_volume_24h": 0, "min_age_days": 0, "top_n": 5}
    exchange = _exchange([("A/USDT", 50e6, 500)])
    news = FakeNews({"A/USDT": {"score": 0.42, "articles": 11}})
    found = CoinScreener(config, exchange=exchange, news=news).scan(NOW)
    assert found[0].news_score == pytest.approx(0.42)
    assert found[0].news_articles == 11


def test_news_is_not_fetched_for_disqualified_names(config):
    """Sentiment for four hundred markets is four hundred lookups for
    nothing."""
    config["screen"] = {"min_quote_volume_24h": 10e6, "min_age_days": 0,
                        "top_n": 5}
    exchange = _exchange([("GOOD/USDT", 50e6, 500), ("THIN/USDT", 1e6, 500)])
    news = FakeNews({})
    CoinScreener(config, exchange=exchange, news=news).scan(NOW)
    assert "GOOD/USDT" in news.asked
    assert "THIN/USDT" not in news.asked


def test_a_news_source_that_raises_does_not_stop_the_screen(config):
    class Angry:
        def sentiment_for(self, *a, **k):
            raise RuntimeError("feed down")

    config["screen"] = {"min_quote_volume_24h": 0, "min_age_days": 0}
    exchange = _exchange([("A/USDT", 50e6, 500)])
    found = CoinScreener(config, exchange=exchange, news=Angry()).scan(NOW)
    assert found and found[0].news_score == 0.0


def test_beta_is_attached_when_a_book_is_given(config):
    from bot.analysis.beta import BetaBook

    class Book(BetaBook):
        def known(self, symbol): return symbol == "A/USDT"
        def beta(self, symbol, default=1.0): return 1.35
        def r2(self, symbol, default=0.0): return 0.61

    config["screen"] = {"min_quote_volume_24h": 0, "min_age_days": 0}
    exchange = _exchange([("A/USDT", 50e6, 500)])
    found = CoinScreener(config, exchange=exchange).scan(NOW, beta_book=Book(config))
    assert found[0].beta == pytest.approx(1.35)
    assert found[0].beta_r2 == pytest.approx(0.61)


def test_render_survives_an_empty_screen(config):
    import logging
    render([], logging.getLogger("test"))


def test_a_candidate_serialises(config):
    config["screen"] = {"min_quote_volume_24h": 0, "min_age_days": 0}
    exchange = _exchange([("A/USDT", 50e6, 500)])
    found = CoinScreener(config, exchange=exchange).scan(NOW)
    row = found[0].to_dict()
    assert row["symbol"] == "A/USDT" and row["rank"] == 1
