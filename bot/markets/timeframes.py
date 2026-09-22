"""Choosing the timeframe to trade an instrument on.

The default per asset class is the starting point, not the answer. Two
adjustments on top, both about matching the decision rate to the market
rather than to a config value:

  **Volatility.** An instrument whose ATR is a large share of price gives
  a 15-minute bar real information; one grinding sideways does not, and
  trading it fast just pays the spread more often. So a quiet crypto market
  steps up a timeframe and a wild one steps down.

  **Time left in the session.** An equity with forty minutes to the bell
  cannot be given a daily decision, and a crypto instrument an hour from
  its funding settlement should not be opened on a 4-hour view that will
  cross it. The slot has to fit the trade.

Equities stay on daily. Their session is 6.5 hours, so an hourly bar is one
of seven and anything faster is microstructure this bot has no edge in.
"""

from __future__ import annotations

import logging
from datetime import datetime

from bot.analysis.indicators import TIMEFRAME_SECONDS
from bot.markets.instrument import AssetClass, Instrument

logger = logging.getLogger("trading_bot")

# The ladder a crypto instrument can move along, fastest first.
CRYPTO_LADDER = ["15m", "1h", "4h"]
HIGHER_OF = {"15m": "1h", "1h": "4h", "4h": "1d", "1d": "1w", "1w": "1w"}

# ATR as a percentage of price, measured on the 1h bar. Below the floor the
# market is too quiet to pay 15-minute costs; above the ceiling a fast bar
# is mostly noise.
QUIET_ATR_PCT = 0.35
WILD_ATR_PCT = 1.60


def choose_timeframe(instrument: Instrument, atr_pct: float | None = None,
                     now: datetime | None = None) -> tuple[str, str, str]:
    """Return (timeframe, higher_timeframe, reason)."""
    if not instrument.asset_class.is_crypto:
        # Equities and ETFs: daily, always. See the module docstring.
        return instrument.timeframe, instrument.higher_timeframe, "daily asset class"

    base = instrument.timeframe if instrument.timeframe in CRYPTO_LADDER else "1h"
    index = CRYPTO_LADDER.index(base)
    reason = "configured"

    if atr_pct is not None and atr_pct > 0:
        if atr_pct < QUIET_ATR_PCT and index < len(CRYPTO_LADDER) - 1:
            index += 1
            reason = f"quiet ({atr_pct:.2f}% ATR) — slower bar"
        elif atr_pct > WILD_ATR_PCT and index > 0:
            index -= 1
            reason = f"volatile ({atr_pct:.2f}% ATR) — faster bar"

    timeframe = CRYPTO_LADDER[index]
    return timeframe, HIGHER_OF.get(timeframe, "4h"), reason


def fits_in_session(instrument: Instrument, timeframe: str,
                    now: datetime, min_bars: int = 4) -> bool:
    """Is there room to manage a trade on this timeframe before the close?

    Opening a position that the session will force flat two bars later
    pays a full round trip for a fraction of the intended holding period.
    """
    minutes_left = instrument.calendar.minutes_until_close(now)
    if minutes_left is None:
        return False                      # market shut
    if instrument.asset_class.is_crypto:
        return True                       # no bell to beat

    bar_minutes = TIMEFRAME_SECONDS.get(timeframe, 3600) / 60
    if bar_minutes >= 1440:
        # A daily bar has no intraday management to fit; it only needs the
        # session to be open.
        return minutes_left > 0
    return minutes_left >= bar_minutes * min_bars


def bars_per_day(timeframe: str) -> int:
    seconds = TIMEFRAME_SECONDS.get(timeframe, 3600)
    return max(1, int(86_400 / seconds))


# The frames a replay must carry for an instrument: every rung the adaptive
# chooser can land on, plus the higher timeframe each of those rungs asks
# for. Downloading less means a day where the chooser steps to 15m finds no
# bars and the instrument goes silently untradable — which looks in a report
# exactly like having had no opportunity.
def ladder_for(instrument: Instrument) -> list[str]:
    """Every timeframe this instrument might be asked for, coarsest last."""
    if instrument.asset_class.is_crypto:
        rungs = list(CRYPTO_LADDER)
    else:
        rungs = [instrument.timeframe]

    frames = set(rungs)
    frames.add(instrument.timeframe)
    frames.add(instrument.higher_timeframe)
    for rung in rungs:
        frames.add(HIGHER_OF.get(rung, rung))
    return sorted(frames, key=lambda tf: TIMEFRAME_SECONDS.get(tf, 3600))
