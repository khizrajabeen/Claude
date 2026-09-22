"""Per-ticker news, and the strategy that trades on it.

A US stock prints one bar a day and gets one entry slot per session, so
over a 90-day replay the equity leg managed nine trades — not because the
strategies disliked the setups but because a daily chart barely produces
any. News arrives continuously and is per-company, so for equities the
story is the signal and the chart is the filter.

The failure mode this guards against is a headline-chasing machine: one
excitable story opening a position, or good news bought in a downtrend.
"""

import pytest

from bot.analysis.news_sentiment import (
    EQUITY_FEEDS,
    TICKER_FEED_TEMPLATES,
    TICKER_KEYWORDS,
    NewsSentimentAnalyzer,
    detect_coins,
)
from bot.daily.briefing import SymbolRead
from bot.strategies.base import MarketContext
from bot.strategies.newsdriven import NewsDriven


# ── Matching stories to instruments ──────────────────────────

def test_a_company_is_matched_by_name_not_only_by_ticker():
    """A headline says "Nvidia" far more often than it says "NVDA"."""
    found = detect_coins("Nvidia beats earnings expectations",
                         {"NVDA": ["nvidia", "nvda"]})
    assert found.get("NVDA")


def test_a_ticker_alone_still_matches():
    found = detect_coins("NVDA upgraded to buy", {"NVDA": ["nvidia", "nvda"]})
    assert found.get("NVDA")


def test_an_untracked_company_is_not_matched():
    assert not detect_coins("Nvidia beats earnings", {})


def test_matching_respects_word_boundaries():
    """"amd" must not fire on "amduat" or "named"."""
    assert not detect_coins("The scheme was named after him",
                            {"AMD": ["amd"]}).get("AMD")


def test_a_market_wide_equity_story_maps_to_the_index():
    """A story about "the stock market" is a story about every stock in
    the book; dropping it would throw away most of the macro tape."""
    found = detect_coins("Wall Street rallies as the Federal Reserve holds", {})
    assert found.get("SPY")


def test_a_market_wide_crypto_story_still_maps_to_bitcoin():
    found = detect_coins("Crypto markets rally on ETF inflows", {})
    assert found.get("BTC")


def test_a_named_company_beats_the_market_wide_fallback():
    found = detect_coins("Nvidia leads the stock market higher",
                         {"NVDA": ["nvidia"]})
    assert found.get("NVDA")
    assert "SPY" not in found, "a named story is not a market-wide one"


# ── Tracking and feeds ───────────────────────────────────────

def test_nothing_is_fetched_until_a_universe_asks(config):
    """A per-ticker feed is one request per symbol per refresh."""
    analyzer = NewsSentimentAnalyzer(config)
    assert analyzer.tickers == {}
    crypto_only = len(analyzer.feeds)

    analyzer.track_tickers(["NVDA", "AAPL"])
    assert len(analyzer.feeds) > crypto_only


def test_tracking_adds_a_feed_per_ticker_per_template(config):
    analyzer = NewsSentimentAnalyzer(config)
    before = len(analyzer.feeds)
    analyzer.track_tickers(["NVDA", "AAPL", "MSFT"])
    added = len(analyzer.feeds) - before
    assert added == 3 * len(TICKER_FEED_TEMPLATES) + len(EQUITY_FEEDS)


def test_a_tracked_ticker_picks_up_its_company_name(config):
    analyzer = NewsSentimentAnalyzer(config)
    analyzer.track_tickers(["NVDA"])
    assert "nvidia" in analyzer.tickers["NVDA"]
    assert "nvda" in analyzer.tickers["NVDA"]


def test_an_unknown_ticker_is_matched_on_the_ticker_alone(config):
    analyzer = NewsSentimentAnalyzer(config)
    analyzer.track_tickers(["ZZZZ"])
    assert analyzer.tickers["ZZZZ"] == ["zzzz"]


def test_extra_keywords_can_be_supplied(config):
    analyzer = NewsSentimentAnalyzer(config)
    analyzer.track_tickers({"ZZZZ": ["Zeta Corp"]})
    assert "zeta corp" in analyzer.tickers["ZZZZ"]


def test_tracking_a_universe_skips_the_crypto(config):
    from bot.markets.instrument import build_instrument

    analyzer = NewsSentimentAnalyzer(config)
    analyzer.track_universe([
        build_instrument({"symbol": "BTC/USDT", "asset_class": "crypto_spot"}),
        build_instrument({"symbol": "BTC/USDT:USDT", "asset_class": "crypto_perp"}),
        build_instrument({"symbol": "NVDA", "asset_class": "equity"}),
        build_instrument({"symbol": "SPY", "asset_class": "etf"}),
    ])
    assert set(analyzer.tickers) == {"NVDA", "SPY"}


def test_an_empty_universe_tracks_nothing(config):
    analyzer = NewsSentimentAnalyzer(config)
    analyzer.track_universe([])
    assert analyzer.tickers == {}


# ── The strategy ─────────────────────────────────────────────

def _read(symbol="NVDA", asset_class="equity", score=0.4, articles=20,
          strength=0.4, short_score=0.4, primary=0):
    return SymbolRead(
        symbol=symbol, price=100.0, atr=2.0, atr_pct=2.0, adx=25.0,
        regime="trending_up", tradable=True, quote_volume_24h=1e9,
        asset_class=asset_class, news_score=score, news_articles=articles,
        news_strength=strength, news_short_score=short_score,
        primary_trend=primary,
    )


def _ctx(*reads):
    import pandas as pd
    reads = {r.symbol: r for r in reads}
    idx = pd.date_range("2026-09-01", periods=40, freq="1D", tz="UTC")
    frames = {s: pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                               "close": 100.0, "volume": 1.0}, index=idx)
              for s in reads}
    return MarketContext(day="2026-09-21", reads=reads, frames=frames,
                         funding={}, market_tone=0.0, equity=10_000.0,
                         timeframe="1d")


def test_strong_well_covered_good_news_opens_a_long(config):
    strategy = NewsDriven(config)
    signals = strategy.generate(_ctx(_read(score=0.5, articles=25, strength=0.5)))
    assert len(signals) == 1
    assert signals[0].direction == 1
    assert signals[0].symbol == "NVDA"


def test_strong_bad_news_opens_a_short(config):
    strategy = NewsDriven(config)
    signals = strategy.generate(
        _ctx(_read(score=-0.5, short_score=-0.5, articles=25, strength=0.5)))
    assert signals and signals[0].direction == -1


def test_one_excitable_headline_is_not_a_signal(config):
    """A +0.9 score drawn from two stories is a strong opinion weakly
    held."""
    strategy = NewsDriven(config)
    assert not strategy.generate(
        _ctx(_read(score=0.9, short_score=0.9, articles=2, strength=0.9)))


def test_broad_coverage_with_no_consensus_is_not_a_signal(config):
    strategy = NewsDriven(config)
    assert not strategy.generate(
        _ctx(_read(score=0.02, short_score=0.02, articles=40, strength=0.4)))


def test_wide_coverage_but_no_strength_is_refused(config):
    strategy = NewsDriven(config)
    assert not strategy.generate(
        _ctx(_read(score=0.5, short_score=0.5, articles=40, strength=0.0)))


def test_good_news_is_not_bought_in_a_primary_downtrend(config):
    strategy = NewsDriven(config)
    assert not strategy.generate(
        _ctx(_read(score=0.5, short_score=0.5, primary=-1)))


def test_good_news_is_bought_when_the_trend_agrees(config):
    strategy = NewsDriven(config)
    assert strategy.generate(_ctx(_read(score=0.5, short_score=0.5, primary=1)))


def test_the_trend_veto_can_be_switched_off(config):
    config.setdefault("strategies", {})["news"] = {"require_trend_agreement": False}
    strategy = NewsDriven(config)
    assert strategy.generate(
        _ctx(_read(score=0.5, short_score=0.5, primary=-1)))


def test_crypto_is_left_to_the_chart(config):
    """Its tape is fast enough to signal on, and crypto headlines are
    mostly about the whole market rather than one coin."""
    strategy = NewsDriven(config)
    assert not strategy.generate(
        _ctx(_read(symbol="BTC/USDT", asset_class="crypto_spot",
                   score=0.6, short_score=0.6, articles=40, strength=0.6)))


def test_an_etf_is_traded_on_news(config):
    strategy = NewsDriven(config)
    assert strategy.generate(
        _ctx(_read(symbol="SPY", asset_class="etf", score=0.5, short_score=0.5)))


def test_a_stale_story_signals_more_weakly_than_a_fresh_one(config):
    """One the tape has had all day to digest has been priced."""
    strategy = NewsDriven(config)
    fresh = strategy.generate(_ctx(_read(score=0.5, short_score=0.5)))[0]
    stale = strategy.generate(_ctx(_read(score=0.5, short_score=0.0)))[0]
    assert fresh.strength > stale.strength


def test_the_signal_names_its_evidence(config):
    strategy = NewsDriven(config)
    signal = strategy.generate(_ctx(_read(score=0.5, short_score=0.5)))[0]
    assert "stories" in signal.reason
    assert signal.meta["news_articles"] == 20


def test_a_disabled_strategy_says_nothing(config):
    config.setdefault("strategies", {})["news"] = {"enabled": False}
    assert not NewsDriven(config).generate(_ctx(_read()))


def test_an_untradable_read_is_skipped(config):
    read = _read()
    read.tradable = False
    assert not NewsDriven(config).generate(_ctx(read))
