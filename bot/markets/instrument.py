"""What the bot is trading, and the rules that come with it.

An instrument bundles the things that differ by asset class and would
otherwise be hardcoded per call site: which venue quotes it, when that
venue is open, what a round trip costs, whether it pays funding, and what
timeframe it is sensibly traded on.

The timeframe point is the one worth spelling out. Crypto trades round the
clock, so an hourly bar is one twenty-fourth of a day and a 15-minute bar
is a real decision point. A US equity trades 6.5 hours, so an hourly bar is
one of six or seven in the whole session and anything faster is mostly
market microstructure the bot has no edge in. Giving every instrument the
same timeframe therefore either starves the crypto book of decisions or
drowns the equity book in them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from bot.markets.calendar import TradingCalendar, calendar_for


class AssetClass(str, Enum):
    CRYPTO_SPOT = "crypto_spot"
    CRYPTO_PERP = "crypto_perp"
    EQUITY = "equity"
    ETF = "etf"
    FUTURES = "futures"

    @property
    def is_crypto(self) -> bool:
        return self in (AssetClass.CRYPTO_SPOT, AssetClass.CRYPTO_PERP)

    @property
    def pays_funding(self) -> bool:
        """Only perpetuals have a funding leg."""
        return self is AssetClass.CRYPTO_PERP

    @property
    def can_short(self) -> bool:
        """Spot crypto cannot be shorted without a margin facility this bot
        does not model, so a short signal on spot is dropped rather than
        silently treated as a sellable position."""
        return self is not AssetClass.CRYPTO_SPOT

    @property
    def calendar_name(self) -> str:
        if self.is_crypto:
            return "crypto"
        if self is AssetClass.FUTURES:
            return "us_futures"
        return "us_equity"


# Default cost model per class, in basis points of notional per side.
# Crypto taker fees dwarf equity commissions; equity spreads on liquid
# names are tighter than crypto's.
DEFAULT_COSTS: dict[AssetClass, dict[str, float]] = {
    AssetClass.CRYPTO_SPOT: {"fee_bps": 5.5, "slippage_bps": 3.0},
    AssetClass.CRYPTO_PERP: {"fee_bps": 5.0, "slippage_bps": 2.5},
    AssetClass.EQUITY: {"fee_bps": 1.0, "slippage_bps": 2.0},
    AssetClass.ETF: {"fee_bps": 1.0, "slippage_bps": 1.0},
    AssetClass.FUTURES: {"fee_bps": 1.0, "slippage_bps": 1.5},
}

# Timeframe by class: fast enough to matter, slow enough to be signal.
DEFAULT_TIMEFRAMES: dict[AssetClass, tuple[str, str]] = {
    AssetClass.CRYPTO_SPOT: ("1h", "4h"),
    AssetClass.CRYPTO_PERP: ("1h", "4h"),
    AssetClass.EQUITY: ("1d", "1w"),
    AssetClass.ETF: ("1d", "1w"),
    AssetClass.FUTURES: ("1h", "4h"),
}


@dataclass(frozen=True)
class Instrument:
    """One tradeable thing, with its venue, session and cost model."""

    symbol: str                     # what the bot calls it, e.g. "BTC/USDT"
    asset_class: AssetClass
    venue: str                      # "okx", "kraken", "nasdaq"
    timeframe: str = "1h"
    higher_timeframe: str = "4h"
    fee_bps: float = 5.5
    slippage_bps: float = 3.0
    provider_symbol: str = ""       # what the venue calls it, if different
    tags: dict = field(default_factory=dict)

    @property
    def calendar(self) -> TradingCalendar:
        return calendar_for(self.asset_class.calendar_name)

    @property
    def key(self) -> str:
        """Stable identity across venues — two venues' BTC are one bet."""
        return f"{self.asset_class.value}:{self.symbol}"

    @property
    def feed_symbol(self) -> str:
        return self.provider_symbol or self.symbol

    @property
    def round_trip_bps(self) -> float:
        return (self.fee_bps + self.slippage_bps) * 2

    def is_open(self, moment) -> bool:
        return self.calendar.is_open(moment)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "asset_class": self.asset_class.value,
            "venue": self.venue,
            "timeframe": self.timeframe,
            "round_trip_bps": round(self.round_trip_bps, 2),
        }


def build_instrument(spec: dict | str, defaults: dict | None = None) -> Instrument:
    """Build an instrument from config.

    A bare string is taken as crypto spot, which keeps the simple
    single-asset config from before working unchanged.
    """
    defaults = defaults or {}
    if isinstance(spec, str):
        spec = {"symbol": spec}

    asset_class = AssetClass(spec.get("asset_class", defaults.get("asset_class",
                                                                  "crypto_spot")))
    costs = DEFAULT_COSTS[asset_class]
    timeframe, higher = DEFAULT_TIMEFRAMES[asset_class]

    return Instrument(
        symbol=spec["symbol"],
        asset_class=asset_class,
        venue=spec.get("venue", defaults.get("venue", _default_venue(asset_class))),
        timeframe=spec.get("timeframe", defaults.get("timeframe", timeframe)),
        higher_timeframe=spec.get("higher_timeframe",
                                  defaults.get("higher_timeframe", higher)),
        fee_bps=float(spec.get("fee_bps", costs["fee_bps"])),
        slippage_bps=float(spec.get("slippage_bps", costs["slippage_bps"])),
        provider_symbol=spec.get("provider_symbol", ""),
        tags=spec.get("tags", {}) or {},
    )


def _default_venue(asset_class: AssetClass) -> str:
    if asset_class.is_crypto:
        return "okx"
    return "nasdaq"


def build_universe(config: dict) -> list[Instrument]:
    """Every instrument this config asks for, across asset classes."""
    data = config.get("data", {})
    defaults = data.get("instrument_defaults", {})

    specs: list[dict | str] = []
    # New-style: an explicit list of instrument specs.
    for spec in data.get("instruments", []) or []:
        specs.append(spec)
    # Old-style: a bare symbol list, treated as crypto spot.
    for symbol in data.get("symbols", []) or []:
        if not any(_symbol_of(s) == symbol for s in specs):
            specs.append(symbol)

    universe, seen = [], set()
    for spec in specs:
        instrument = build_instrument(spec, defaults)
        if instrument.key in seen:
            continue
        seen.add(instrument.key)
        universe.append(instrument)
    return universe


def _symbol_of(spec: dict | str) -> str:
    return spec if isinstance(spec, str) else spec.get("symbol", "")
