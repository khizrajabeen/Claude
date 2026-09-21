"""News scoring and windowing.

The headline bug this file guards against: a fetch cache that ignored the
requested window, so 1h, 4h and 24h sentiment were three copies of one
number.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from bot.analysis.news_sentiment import (
    NewsItem, NewsSentimentAnalyzer, _dedup_key, detect_coins, score_text,
)

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


def analyzer(**news_config):
    return NewsSentimentAnalyzer({"news": news_config})


def seed(engine, items):
    """Put items straight into the store, bypassing the network."""
    engine._items = list(items)
    engine._last_fetch = 9e18  # never refresh during a test
    return engine


def item(title, hours_ago=1.0, weight=1.0, summary=""):
    node = NewsItem(
        title=title, summary=summary, source="test", url="",
        published=NOW - timedelta(hours=hours_ago), source_weight=weight,
    )
    node.sentiment = score_text(f"{title}. {title}. {summary}")
    node.coins = detect_coins(f"{title} {summary}")
    return node


# ── Scoring ──────────────────────────────────────────────────

def test_obvious_headlines_score_in_the_right_direction():
    assert score_text("Bitcoin ETF approved, BTC surges to all-time high") > 0.5
    assert score_text("Major exchange hacked, $200M stolen in exploit") < -0.5
    assert score_text("Quarterly conference schedule announced") == 0.0


def test_a_denial_is_not_the_opposite_claim():
    """'Exchange denies hack' must not score as bullish as a real rally."""
    denial = score_text("Exchange denies hack rumors, says funds are safe")
    rally = score_text("Bitcoin surges to all-time high on institutional inflows")
    hack = score_text("Exchange hacked, funds stolen")

    assert hack < 0
    assert denial > hack, "negation must blunt the term"
    assert denial < rally, "a denial is a non-event, not good news"


def test_coin_detection_uses_word_boundaries():
    assert "SOL" in detect_coins("Solana network upgrade goes live")
    assert "SOL" not in detect_coins("The team found a solid solution")
    assert "BTC" in detect_coins("Crypto market rallies broadly")


def test_syndicated_copies_collapse_to_one_key():
    assert _dedup_key("SEC Approves Bitcoin ETF") == _dedup_key("Bitcoin ETF approves SEC")
    assert _dedup_key("Ethereum upgrade ships") != _dedup_key("Solana outage continues")


# ── Windows ──────────────────────────────────────────────────

def test_windows_are_genuinely_different():
    """The regression test for the cache bug."""
    engine = seed(analyzer(), [
        item("Bitcoin surges to all-time high on ETF approval", hours_ago=0.5),
        item("Bitcoin crashes as exchange hacked and funds stolen", hours_ago=20),
    ])

    recent = engine.sentiment_for("BTC/USDT", hours=1, now=NOW)
    full_day = engine.sentiment_for("BTC/USDT", hours=24, now=NOW)

    assert recent["articles"] == 1
    assert full_day["articles"] == 2
    assert recent["score"] != full_day["score"]
    assert recent["score"] > 0 > full_day["score"] or recent["score"] > full_day["score"]


def test_features_expose_distinct_windows():
    engine = seed(analyzer(), [
        item("Bitcoin rallies on record institutional inflows", hours_ago=0.25),
        item("Bitcoin plunges after regulatory crackdown", hours_ago=18),
    ])
    features = engine.features("BTC/USDT", now=NOW)

    assert features["news_sentiment_1h"] != features["news_sentiment_24h"]
    assert features["news_volume_1h"] < features["news_volume_24h"]
    assert features["news_momentum"] == pytest.approx(
        features["news_sentiment_1h"] - features["news_sentiment_24h"]
    )


def test_recent_news_outweighs_stale_news():
    fresh = seed(analyzer(half_life_hours=6),
                 [item("Bitcoin surges on ETF approval", hours_ago=0.5)])
    stale = seed(analyzer(half_life_hours=6),
                 [item("Bitcoin surges on ETF approval", hours_ago=30)])

    assert fresh.sentiment_for("BTC/USDT", 48, NOW)["score"] == pytest.approx(
        stale.sentiment_for("BTC/USDT", 48, NOW)["score"]
    ), "a lone article sets the average regardless of age"

    # But mixed with an opposing older story, recency decides the sign.
    mixed = seed(analyzer(half_life_hours=3), [
        item("Bitcoin surges on ETF approval", hours_ago=0.5),
        item("Bitcoin crashes as exchange hacked", hours_ago=24),
    ])
    assert mixed.sentiment_for("BTC/USDT", 48, NOW)["score"] > 0


def test_empty_store_is_neutral_not_an_error():
    engine = seed(analyzer(), [])
    sentiment = engine.sentiment_for("BTC/USDT", 24, NOW)
    assert sentiment == {
        "symbol": "BTC/USDT", "window_hours": 24, "score": 0.0, "articles": 0,
        "bullish": 0, "bearish": 0, "neutral": 0, "consensus": 0.0,
        "signal_strength": 0.0, "headlines": [],
    }
    assert engine.tilt_for("BTC/USDT", NOW) == 0.0


# ── Tilt ─────────────────────────────────────────────────────

def test_one_loud_headline_does_not_move_the_tilt():
    engine = seed(analyzer(min_articles=2), [
        item("Bitcoin surges to all-time high on ETF approval", hours_ago=1),
    ])
    assert engine.tilt_for("BTC/USDT", NOW) == 0.0


def test_consensus_produces_a_tilt():
    engine = seed(analyzer(min_articles=2, signal_threshold=0.2, tilt_window_hours=8), [
        item("Bitcoin ETF approved, inflows surge", hours_ago=1),
        item("Institutional accumulation drives bitcoin rally", hours_ago=2),
        item("Bitcoin adoption milestone as partnership announced", hours_ago=3),
    ])
    assert engine.tilt_for("BTC/USDT", NOW) > 0.1


def test_tilt_is_bounded():
    engine = seed(analyzer(min_articles=1, signal_threshold=0.0), [
        item("Bitcoin ETF approved surge rally all-time high inflows", hours_ago=0.1)
        for _ in range(40)
    ])
    assert -1.0 <= engine.tilt_for("BTC/USDT", NOW) <= 1.0


def test_market_tone_summarises_everything():
    engine = seed(analyzer(), [
        item("Crypto market rallies on rate cut hopes", hours_ago=1),
        item("Ethereum upgrade drives adoption", hours_ago=2),
    ])
    tone = engine.market_bias(hours=12, now=NOW)
    assert tone["articles"] == 2
    assert tone["tone"] in ("bullish", "bearish", "neutral")


def test_editorial_sources_outweigh_social():
    heavy = seed(analyzer(), [item("Bitcoin surges on ETF approval", 1, weight=1.0),
                              item("Bitcoin crashes after hack", 1, weight=0.1)])
    light = seed(analyzer(), [item("Bitcoin surges on ETF approval", 1, weight=0.1),
                              item("Bitcoin crashes after hack", 1, weight=1.0)])
    assert heavy.sentiment_for("BTC/USDT", 24, NOW)["score"] > \
        light.sentiment_for("BTC/USDT", 24, NOW)["score"]
