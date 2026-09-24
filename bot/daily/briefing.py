"""The morning read: what happened overnight and what the tape looks like.

Run once at the start of each trading day, before any order is placed. For
every symbol in the universe it assembles:

  * price structure on the trading timeframe and the higher timeframe
  * volatility (ATR, annualised realised vol) — the input to position size
  * trend strength (ADX/DI) and mean-reversion stretch (z-score, RSI)
  * liquidity (24h quote volume, spread) — the input to the impact model
  * perpetual funding, when the venue publishes it
  * news sentiment for the coin, plus the market-wide tone

It also loads yesterday's record, so the day starts knowing the running
equity, the open positions it inherited and how the recent trades went.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import pandas as pd

from bot.analysis import indicators as ind
from bot.analysis.news_sentiment import NewsSentimentAnalyzer
from bot.analysis.regime import MarketRegime, RegimeDetector
from bot.markets.timeframes import (
    HIGHER_OF,
    bars_per_day,
    choose_timeframe,
    fits_in_session,
)

logger = logging.getLogger("trading_bot")


@dataclass
class SymbolRead:
    """One symbol's state at the open."""

    symbol: str
    price: float = 0.0
    atr: float = 0.0
    atr_pct: float = 0.0
    annualized_vol: float = 0.0
    adx: float = 0.0
    plus_di: float = 0.0
    minus_di: float = 0.0
    rsi: float = 50.0
    zscore: float = 0.0
    donchian: float = 0.5
    macd_hist: float = 0.0
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    htf_trend: int = 0
    # The major trend, read two rungs above the decision frame — daily for
    # crypto, weekly for equities. Separate from htf_trend, which is a fast
    # EMA cross on the 4h bar and cannot see past about nine days.
    primary_trend: int = 0
    primary_strength: float = 0.0
    primary_timeframe: str = ""
    # Where this instrument sits against its own asset class on
    # risk-adjusted momentum: 1.0 is the strongest name in the class,
    # 0.0 the weakest. Absolute strength says an instrument is rising;
    # relative strength says whether it is the one worth owning.
    momentum_score: float = 0.0
    momentum_rank: float = 0.5
    momentum_peers: int = 0
    # Measured, not tabulated. Beta says how much of the market's move
    # this instrument takes; r-squared says how much of *its* move is the
    # market's. A coin at beta 1.0 with r-squared 0.2 is a different bet
    # from one at beta 1.0 with r-squared 0.8.
    beta: float = 1.0
    beta_r2: float = 0.0
    beta_benchmark: str = ""
    htf_strength: float = 0.0
    overnight_return_pct: float = 0.0
    regime: str = "ranging"
    regime_confidence: float = 0.0
    quote_volume_24h: float = 0.0
    spread_bps: float = 0.0
    book_imbalance: float = 0.0
    funding_rate: float = 0.0
    news_score: float = 0.0
    news_articles: int = 0
    news_tilt: float = 0.0
    # Consensus x coverage x magnitude, in [0,1]. Distinct from the score:
    # a +0.9 score from two stories is a strong opinion weakly held, and
    # the strength is what says so.
    news_strength: float = 0.0
    news_short_score: float = 0.0
    news_headlines: list = field(default_factory=list)
    bars: int = 0
    tradable: bool = True
    skip_reason: str = ""
    # Multi-asset context: which market this is, when it is open, and what
    # timeframe it was read on. The strategies do not care, but the session
    # and the P&L attribution do.
    asset_class: str = "crypto_spot"
    venue: str = ""
    timeframe: str = "1h"
    timeframe_reason: str = ""
    market_open: bool = True
    minutes_to_close: float | None = None
    round_trip_bps: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Briefing:
    """The whole morning read for one trading day."""

    day: str
    generated_at: str
    equity: float
    cash: float
    carried_positions: list = field(default_factory=list)
    symbols: dict = field(default_factory=dict)  # symbol -> SymbolRead
    market_tone: dict = field(default_factory=dict)
    rolling_stats: dict = field(default_factory=dict)
    yesterday: dict | None = None
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "day": self.day,
            "generated_at": self.generated_at,
            "equity": round(self.equity, 2),
            "cash": round(self.cash, 2),
            "carried_positions": self.carried_positions,
            "symbols": {k: v.to_dict() for k, v in self.symbols.items()},
            "market_tone": self.market_tone,
            "rolling_stats": self.rolling_stats,
            "yesterday": self.yesterday,
            "notes": self.notes,
        }


class BriefingBuilder:
    """Assembles the morning briefing."""

    def __init__(self, config: dict, router, news: NewsSentimentAnalyzer | None = None,
                 universe: list | None = None):
        self.config = config
        self.router = router
        self.news = news if news is not None else NewsSentimentAnalyzer(config)
        self.regime_detector = RegimeDetector(config)

        from bot.markets import build_universe
        self.universe = universe if universe is not None else build_universe(config)
        self.by_symbol = {i.symbol: i for i in self.universe}
        # Per-ticker news is one HTTP request per symbol per refresh, so
        # nothing is fetched until a universe says which names matter.
        try:
            self.news.track_universe(self.universe)
        except AttributeError:
            pass        # a test double without ticker support

        data = config.get("data", {})
        self.history_bars = int(data.get("history_bars", 500))
        self.adaptive_timeframes = bool(data.get("adaptive_timeframes", True))
        self.atr_period = int(config.get("stops", {}).get("atr_period", 14))

        filters = config.get("filters", {})
        self.min_quote_volume = float(filters.get("min_quote_volume_24h", 5_000_000))
        # Venues where reported volume actually measures available
        # liquidity. See _apply_filters for why that is not universal.
        self.volume_is_liquidity = set(
            filters.get("volume_is_liquidity",
                        ["binance", "okx", "kraken", "kucoin", "coinbase"]))
        self.max_spread_bps = float(filters.get("max_spread_bps", 25))
        self.min_atr_pct = float(filters.get("min_atr_pct", 0.15))
        self.max_atr_pct = float(filters.get("max_atr_pct", 12.0))
        self.min_bars = int(filters.get("min_bars", 120))
        self.book_depth = int(config.get("data", {}).get("orderbook_depth", 20))
        self.book_levels = int(config.get("data", {}).get("orderbook_levels", 10))

        signals = config.get("signals", {})
        self.primary_period = int(signals.get("primary_trend_period", 200))
        self.primary_slope_bars = int(signals.get("primary_trend_slope_bars", 20))
        # Below this the moving average is still finding its level and its
        # slope means nothing, so the read stays neutral rather than guessing.
        self.primary_min_bars = int(signals.get("primary_trend_min_bars",
                                                self.primary_period + 20))
        self.momentum_lookback = int(signals.get("momentum_lookback", 60))

        from bot.analysis.beta import BetaBook
        self.beta_book = BetaBook(config)

    def build(
        self,
        symbols: list[str] | None,
        day: str,
        equity: float,
        cash: float,
        carried_positions: list | None = None,
        rolling_stats: dict | None = None,
        yesterday: dict | None = None,
        now: datetime | None = None,
    ) -> tuple[Briefing, dict[str, pd.DataFrame]]:
        """Build the briefing. Also returns the OHLCV frames it fetched, so
        the planner does not have to re-download them."""
        now = now or datetime.now(timezone.utc)
        logger.info("─" * 62)
        logger.info("  MORNING BRIEFING — %s", day)
        logger.info("─" * 62)

        if self.config.get("news", {}).get("enabled", True):
            self.news.refresh(force=True)
            tone = self.news.market_bias(
                hours=float(self.config.get("news", {}).get("tone_window_hours", 12)),
                now=now,
            )
        else:
            tone = {"score": 0.0, "articles": 0, "tone": "neutral"}

        briefing = Briefing(
            day=day,
            generated_at=now.isoformat(),
            equity=equity,
            cash=cash,
            carried_positions=[p.to_dict() for p in (carried_positions or [])],
            market_tone=tone,
            rolling_stats=rolling_stats or {},
            yesterday=yesterday,
        )

        # Read only the markets that are actually open. An equity is not
        # "tradable but unattractive" at 03:00 UTC — it is shut, and
        # planning around a price no venue is quoting is how a backtest
        # ends up with fills nobody could have got.
        requested = set(symbols) if symbols else None
        instruments = [i for i in self.universe
                       if requested is None or i.symbol in requested]

        frames: dict[str, pd.DataFrame] = {}
        for instrument in instruments:
            symbol = instrument.symbol
            if not instrument.is_open(now):
                next_open = instrument.calendar.next_open(now)
                briefing.symbols[symbol] = SymbolRead(
                    symbol=symbol, tradable=False,
                    skip_reason=f"{instrument.calendar.name} shut until "
                                f"{next_open:%a %H:%M} UTC",
                    asset_class=instrument.asset_class.value,
                    venue=instrument.venue, market_open=False,
                    round_trip_bps=instrument.round_trip_bps,
                )
                continue
            try:
                read, df = self._read_symbol(instrument, now)
            except Exception as e:
                logger.warning("Briefing failed for %s: %s", symbol, e)
                briefing.symbols[symbol] = SymbolRead(
                    symbol=symbol, tradable=False, skip_reason=f"error: {e}"[:120],
                    asset_class=instrument.asset_class.value,
                    venue=instrument.venue,
                    round_trip_bps=instrument.round_trip_bps,
                )
                continue
            briefing.symbols[symbol] = read
            if df is not None and not df.empty:
                frames[symbol] = df

        self._rank_cross_section(briefing, instruments)
        self._measure_betas(briefing, instruments)

        if tone["articles"]:
            logger.info(
                "  Market tone: %s (%.3f from %d stories)",
                tone["tone"], tone["score"], tone["articles"],
            )
        if carried_positions:
            briefing.notes.append(
                f"Carried {len(carried_positions)} position(s) into the day"
            )
        if yesterday:
            briefing.notes.append(
                f"Yesterday closed at ${yesterday.get('ending_equity', 0):,.2f} "
                f"({yesterday.get('return_pct', 0):+.2f}%)"
            )
        stats = briefing.rolling_stats
        if stats.get("sample"):
            briefing.notes.append(
                f"Last {stats['sample']} trades: {stats['win_rate']*100:.0f}% win rate, "
                f"expectancy {stats['expectancy_r']:+.2f}R"
            )
        for note in briefing.notes:
            logger.info("  • %s", note)

        return briefing, frames

    # ── Per-symbol read ───────────────────────────────────────

    def _read_symbol(self, instrument, now: datetime
                     ) -> tuple[SymbolRead, pd.DataFrame | None]:
        symbol = instrument.symbol
        read = SymbolRead(
            symbol=symbol,
            asset_class=instrument.asset_class.value,
            venue=instrument.venue,
            round_trip_bps=instrument.round_trip_bps,
            market_open=True,
            minutes_to_close=instrument.calendar.minutes_until_close(now),
        )

        # Pick the timeframe from the instrument's own volatility, measured
        # on its configured bar, then re-read on the chosen one. A second
        # fetch on a fast market is worth not trading a 15-minute signal off
        # a 4-hour picture.
        timeframe = instrument.timeframe
        higher = instrument.higher_timeframe
        reason = "configured"
        df = self.router.bars(instrument, timeframe, self.history_bars)

        if self.adaptive_timeframes and not df.empty and len(df) > self.atr_period * 2:
            probe_price = float(df["close"].iloc[-1])
            probe_atr = ind.last_value(ind.atr(df, self.atr_period))
            atr_pct = probe_atr / probe_price * 100 if probe_price else None
            timeframe, higher, reason = choose_timeframe(instrument, atr_pct, now)
            if timeframe != instrument.timeframe:
                df = self.router.bars(instrument, timeframe, self.history_bars)

        read.timeframe = timeframe
        read.timeframe_reason = reason
        read.bars = len(df)

        if df.empty:
            read.tradable = False
            read.skip_reason = "no data from provider"
            return read, df

        if not fits_in_session(instrument, timeframe, now):
            read.tradable = False
            read.skip_reason = f"too close to the {instrument.calendar.name} close"
            return read, df

        if len(df) < self.min_bars:
            read.tradable = False
            read.skip_reason = f"only {len(df)} bars (need {self.min_bars})"
            return read, df

        close = df["close"]
        read.price = float(close.iloc[-1])

        atr_series = ind.atr(df, self.atr_period)
        read.atr = ind.last_value(atr_series)
        read.atr_pct = read.atr / read.price * 100 if read.price else 0.0
        read.annualized_vol = ind.last_value(
            ind.realized_vol(close, 24, timeframe), default=0.6
        )

        adx_s, plus_di, minus_di = ind.adx(df, 14)
        read.adx = ind.last_value(adx_s, 20.0)
        read.plus_di = ind.last_value(plus_di)
        read.minus_di = ind.last_value(minus_di)
        read.rsi = ind.last_value(ind.rsi(close), 50.0)
        read.zscore = ind.last_value(ind.zscore(close, 20))
        read.donchian = ind.last_value(ind.donchian_position(df, 20), 0.5)
        read.macd_hist = ind.last_value(ind.macd_histogram(close))
        read.ema_fast = ind.last_value(ind.ema(close, 21), read.price)
        read.ema_slow = ind.last_value(ind.ema(close, 55), read.price)

        # Move since the previous session, which frames whether a signal is
        # early or already extended. One bar for a daily instrument, a full
        # day's worth for an intraday one.
        lookback = max(1, bars_per_day(timeframe))
        if len(close) > lookback:
            ref = float(close.iloc[-lookback - 1])
            read.overnight_return_pct = (read.price - ref) / ref * 100 if ref else 0.0

        try:
            htf_df = self.router.bars(instrument, higher, 200)
            if len(htf_df) >= 60:
                fast = ind.last_value(ind.ema(htf_df["close"], 21))
                slow = ind.last_value(ind.ema(htf_df["close"], 55))
                if fast and slow:
                    read.htf_trend = 1 if fast > slow else -1
                    read.htf_strength = (fast - slow) / slow
        except Exception as e:
            logger.debug("HTF unavailable for %s: %s", symbol, e)

        self._read_primary_trend(read, instrument, higher)

        regime = self.regime_detector.detect(df)
        read.regime = regime["regime"].value if isinstance(regime["regime"], MarketRegime) \
            else str(regime["regime"])
        read.regime_confidence = float(regime["confidence"])

        read.quote_volume_24h = self.router.quote_volume(instrument) or 0.0
        read.funding_rate = self.router.funding_rate(instrument) or 0.0

        # One order book read gives both the spread and the resting-liquidity
        # imbalance. Venues that publish no book (equities here) fall back to
        # whatever spread the provider can report.
        book = self.router.order_book(instrument, self.book_depth)
        if book:
            measured = ind.orderbook_imbalance(book, levels=self.book_levels)
            read.book_imbalance = measured["imbalance"]
            read.spread_bps = measured["spread_bps"]
        else:
            read.spread_bps = self.router.spread_bps(instrument) or 0.0

        if self.config.get("news", {}).get("enabled", True):
            sent = self.news.sentiment_for(symbol, 24, now)
            read.news_score = sent["score"]
            read.news_articles = sent["articles"]
            read.news_strength = sent["signal_strength"]
            read.news_tilt = self.news.tilt_for(symbol, now)
            read.news_headlines = sent["headlines"][:3]
            # A shorter window as well, so a strategy can tell a story
            # that broke this morning from one the tape has had all day
            # to price.
            fresh = self.news.sentiment_for(
                symbol, float(self.config["news"].get("fresh_window_hours", 6)), now)
            read.news_short_score = fresh["score"]

        self._apply_filters(read)

        logger.info(
            "  %-14s %-11s %12.4f | %-3s | ATR %5.2f%% | ADX %5.1f | %-13s | %s",
            symbol, read.asset_class, read.price, read.timeframe,
            read.atr_pct, read.adx, read.regime,
            "ok" if read.tradable else read.skip_reason,
        )
        return read, df

    def _measure_betas(self, briefing: "Briefing", instruments: list) -> None:
        """Beta and r-squared against each class's own benchmark.

        Computed on the daily frame rather than the decision frame: an
        hourly beta is mostly microstructure, and the exposure question
        this answers — how much Bitcoin am I really holding — is a
        multi-day one.
        """
        daily: dict[str, pd.DataFrame] = {}
        for instrument in instruments:
            read = briefing.symbols.get(instrument.symbol)
            if read is None or not read.tradable:
                continue
            frame = read.primary_timeframe or instrument.higher_timeframe
            try:
                df = self.router.bars(instrument, frame, self.beta_book.window + 20)
            except Exception:
                continue
            if df is not None and not df.empty:
                daily[instrument.symbol] = df

        self.beta_book.update(daily, [
            i for i in instruments if i.symbol in daily
        ])
        for symbol, read in briefing.symbols.items():
            if self.beta_book.known(symbol):
                read.beta = round(self.beta_book.beta(symbol), 3)
                read.beta_r2 = round(self.beta_book.r2(symbol), 3)
                read.beta_benchmark = self.beta_book.benchmark(symbol) or ""

    def _rank_cross_section(self, briefing: "Briefing", instruments: list) -> None:
        """Rank each instrument against its own asset class.

        Absolute momentum says an instrument is rising; relative momentum
        says whether it is the one worth owning. In a market where
        everything rose 42-77% over three months, "BTC is trending up" is
        true of the whole universe and picks nothing. The rank is what
        distinguishes the leaders from the laggards, and buying leaders is
        the oldest documented effect in the literature.

        Ranked within the asset class, not across it: a stock's 90-day
        return is not comparable to a perp's, and pooling them would rank
        crypto above equities every time volatility is high rather than
        when it is actually leading.
        """
        by_class: dict[str, list[SymbolRead]] = {}
        for instrument in instruments:
            read = briefing.symbols.get(instrument.symbol)
            if read is None or not read.tradable:
                continue
            score = self._momentum_score(instrument, read)
            if score is None:
                continue
            read.momentum_score = round(score, 4)
            by_class.setdefault(instrument.asset_class.value, []).append(read)

        for klass, reads in by_class.items():
            reads.sort(key=lambda r: r.momentum_score)
            n = len(reads)
            for position, read in enumerate(reads):
                read.momentum_peers = n
                # Midpoint of the rank so a single instrument is 0.5 —
                # neutral — rather than being called both best and worst.
                read.momentum_rank = round((position + 0.5) / n, 4) if n else 0.5

    def _momentum_score(self, instrument, read: "SymbolRead") -> float | None:
        """Risk-adjusted momentum on the primary frame.

        Dividing by realised volatility is what makes two instruments
        comparable: a 40% move in something that swings 5% a day is a
        smaller achievement than a 20% move in something that swings 1%.
        """
        try:
            df = self.router.bars(instrument, read.primary_timeframe
                                  or instrument.higher_timeframe, 200)
        except Exception:
            return None
        if df is None or len(df) < self.momentum_lookback + 5:
            return None

        close = df["close"]
        past = float(close.iloc[-1 - self.momentum_lookback])
        if not past:
            return None
        total_return = float(close.iloc[-1]) / past - 1.0

        returns = close.pct_change().dropna().tail(self.momentum_lookback)
        vol = float(returns.std())
        if vol <= 0:
            return None
        return total_return / (vol * (self.momentum_lookback ** 0.5))

    def _read_primary_trend(self, read: "SymbolRead", instrument, higher: str) -> None:
        """The major trend, on a frame slow enough to see one.

        `htf_trend` is a 21/55 EMA cross on the 4-hour bar: 55 bars is nine
        days, and in a pullback it flips. Over a 90-day replay in which
        crypto rose between 42% and 77%, the bot took 120 shorts against 67
        longs and the shorts averaged -0.078R. Nothing in the read could
        see the move it was fighting.

        So this reads two rungs up — daily for crypto, weekly for equities
        — and reports a direction only when price and the slope of the
        long moving average agree. When they disagree the answer is zero,
        which places no constraint at all: an ambiguous long-term picture
        is not a reason to overrule the strategies, only an unambiguous one
        is.
        """
        primary = HIGHER_OF.get(higher, higher)
        read.primary_timeframe = primary
        try:
            df = self.router.bars(instrument, primary, 400)
        except Exception as e:
            logger.debug("Primary trend unavailable for %s: %s", read.symbol, e)
            return
        if df is None or len(df) < self.primary_min_bars:
            return

        close = df["close"]
        trend_ma = ind.ema(close, self.primary_period)
        now = ind.last_value(trend_ma)
        if not now:
            return
        # Slope over a window, not bar to bar: a single flat bar is noise.
        back = min(self.primary_slope_bars, len(trend_ma) - 1)
        earlier = float(trend_ma.iloc[-1 - back]) if back > 0 else now
        if not earlier or earlier != earlier:
            return

        price = float(close.iloc[-1])
        above = price > now
        rising = now > earlier
        if above and rising:
            read.primary_trend = 1
        elif not above and not rising:
            read.primary_trend = -1
        else:
            read.primary_trend = 0
        read.primary_strength = (price - now) / now if now else 0.0

    def _apply_filters(self, read: SymbolRead) -> None:
        """Liquidity and volatility gates — a signal in an untradeable market
        is not an opportunity."""
        venue = getattr(read, "venue", "") or ""
        # Reported volume measures liquidity only on a venue that matches
        # orders in its own book. Alpaca routes crypto to external market
        # makers and reports only what crossed its tape, so BTC shows
        # about $330k a day there against hundreds of millions globally —
        # while quoting a 2 bps spread, which is as tight as the asset
        # trades anywhere. Applying the $5M floor to that rejected all
        # thirty coins, BTC included, and the backtest returned no trades
        # at all rather than saying why.
        #
        # Where volume does not measure liquidity, spread does, and the
        # spread gate below is left to do the work alone.
        volume_meaningful = venue in self.volume_is_liquidity
        if (volume_meaningful and read.quote_volume_24h
                and read.quote_volume_24h < self.min_quote_volume):
            read.tradable = False
            read.skip_reason = f"thin: ${read.quote_volume_24h:,.0f} < ${self.min_quote_volume:,.0f}"
        elif read.spread_bps and read.spread_bps > self.max_spread_bps:
            read.tradable = False
            read.skip_reason = f"wide spread: {read.spread_bps:.1f}bps"
        elif read.atr_pct < self.min_atr_pct:
            read.tradable = False
            read.skip_reason = f"too quiet: ATR {read.atr_pct:.2f}%"
        elif read.atr_pct > self.max_atr_pct:
            read.tradable = False
            read.skip_reason = f"too wild: ATR {read.atr_pct:.2f}%"
