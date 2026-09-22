"""Risk budgeting: how big, and may we trade at all.

The sizing rule is volatility targeting rather than a fixed percentage of
equity. Every trade risks the same dollar amount at its stop, and the stop
is placed a multiple of ATR away, so:

    stop distance = k * ATR
    quantity      = risk_dollars / stop distance

When volatility expands the stop widens and the position shrinks, which
keeps dollar risk constant across assets and regimes. That is what stops
losses from clustering in exactly the periods that hurt most.

On top of per-trade risk sit the portfolio limits:

  * portfolio heat  — the sum of open risk if every stop were hit at once
  * correlation cap — crypto majors trade as one factor, so N same-side
                      positions are closer to one big position than to N
                      independent ones
  * daily loss stop, daily profit lock, total drawdown halt, loss cooldown
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from bot.markets.instrument import trading_profile

logger = logging.getLogger("trading_bot")

# Symbols that move as one factor with the market. Anything not listed is
# still crypto, so it gets the default beta rather than a free pass.
DEFAULT_BETA = 0.8
# Beta is measured against each group's own factor — crypto against BTC,
# equities against SPY — because a single market beta across both would
# claim a long in NVDA hedges a short in ETH, which it does not.
SYMBOL_BETA = {
    # Crypto, vs BTC
    "BTC": 1.0, "ETH": 1.05, "SOL": 1.3, "BNB": 0.9, "XRP": 1.0,
    "ADA": 1.1, "DOGE": 1.4, "AVAX": 1.25, "LINK": 1.15, "MATIC": 1.2,
    "USDT": 0.0, "USDC": 0.0, "DAI": 0.0,
    # US equities and ETFs, vs SPY
    "SPY": 1.0, "QQQ": 1.15, "IWM": 1.15, "DIA": 0.95,
    "GLD": 0.15, "TLT": -0.2, "SLV": 0.3,
    "NVDA": 1.75, "AMD": 1.9, "TSLA": 1.85, "COIN": 2.4, "MSTR": 3.0,
    "AAPL": 1.15, "MSFT": 1.0, "GOOGL": 1.05, "AMZN": 1.2, "META": 1.25,
}

# Which positions share a risk factor. Spot BTC and the BTC perp are the
# same bet; a stock and an index ETF move together through the index. Two
# instruments in different groups are treated as independent, so the
# crowding caps apply within a group rather than across the whole book.
CORRELATION_GROUPS = {
    "crypto_spot": "crypto", "crypto_perp": "crypto",
    "equity": "us_equity", "etf": "us_equity",
    "futures": "futures",
}


@dataclass
class RiskDecision:
    """Whether a proposed trade clears the risk gates."""

    allowed: bool
    reason: str = ""
    details: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.allowed


@dataclass
class SizedOrder:
    """A trade sized to a fixed dollar risk at its stop."""

    symbol: str
    side: str
    quantity: float
    entry_price: float
    stop_price: float
    take_profit: float
    risk_usd: float
    notional: float
    leverage: float
    atr: float
    r_distance: float
    caps_applied: list[str] = field(default_factory=list)
    asset_class: str = ""

    @property
    def valid(self) -> bool:
        return self.quantity > 0 and self.r_distance > 0

    @property
    def group(self) -> str:
        return correlation_group(self.asset_class, self.symbol)


def base_symbol(symbol: str) -> str:
    return symbol.split("/")[0].split(":")[0].upper()


def symbol_beta(symbol: str, book=None) -> float:
    """Beta to the instrument's own market factor.

    A measured beta is preferred over the table whenever one exists. The
    table had XRP at 1.00 against a measured 1.24 and ADA at 1.10 against
    1.40, and no entry at all for half the pairs the bot now trades — they
    silently took the default. It survives only as the fallback for an
    instrument with too little history to measure.
    """
    if book is not None and book.known(symbol):
        return book.beta(symbol)
    return SYMBOL_BETA.get(base_symbol(symbol), DEFAULT_BETA)


def _position_group(position) -> str:
    return correlation_group(getattr(position, "asset_class", ""), position.symbol)


def correlation_group(asset_class, symbol: str = "") -> str:
    """The risk factor a position loads on.

    Falling back on the symbol keeps records written before asset classes
    existed grouped with the crypto they were.
    """
    name = getattr(asset_class, "value", asset_class)
    if name:
        return CORRELATION_GROUPS.get(str(name), str(name))
    return "crypto" if "/" in symbol else "us_equity"


class RiskBudget:
    """Sizes trades and enforces the portfolio-level limits."""

    def __init__(self, config: dict):
        self.config = config
        self.risk = config.get("risk", {})
        self.stops = config.get("stops", {})
        self.sizing = config.get("sizing", {})
        # Set each day from the bars the briefing fetched. None means fall
        # back to the static table.
        self.beta_book = None

    # ── Per-trade sizing ──────────────────────────────────────

    def risk_per_trade_pct(self, rolling: dict | None = None,
                           drawdown_pct: float = 0.0,
                           consecutive_losses: int = 0) -> float:
        """Risk budget for the next trade, adapted to recent results.

        Starts from the configured base and scales it by realised edge,
        current drawdown and loss streak. Nothing here can scale risk *up*
        past the configured ceiling.
        """
        base = float(self.risk.get("risk_per_trade_pct", 0.75))
        mult = 1.0
        applied = []

        # Fractional Kelly on the realised R distribution, once there is
        # enough of a sample for it to mean anything.
        if rolling and rolling.get("sample", 0) >= int(self.sizing.get("kelly_min_trades", 30)):
            kelly = float(rolling.get("kelly", 0.0))
            fraction = float(self.sizing.get("kelly_fraction", 0.25))
            if kelly <= 0:
                mult *= 0.5
                applied.append("negative_kelly")
            else:
                # Kelly caps risk; it never inflates it past the base.
                kelly_cap = max(0.25, min(1.0, kelly * fraction / (base / 100)))
                mult *= kelly_cap
                applied.append(f"kelly={kelly_cap:.2f}")

        # De-risk into drawdown: linear down to 1/4 size at the halt level.
        max_dd = float(self.risk.get("max_total_drawdown_pct", 15.0))
        if drawdown_pct > 0 and max_dd > 0:
            dd_mult = max(0.25, 1.0 - drawdown_pct / max_dd)
            mult *= dd_mult
            applied.append(f"dd={dd_mult:.2f}")

        # Loss streaks usually mean the regime moved, not that we are due.
        streak_start = int(self.risk.get("derisk_after_losses", 3))
        if consecutive_losses >= streak_start:
            streak_mult = max(0.3, 1.0 - 0.2 * (consecutive_losses - streak_start + 1))
            mult *= streak_mult
            applied.append(f"streak={streak_mult:.2f}")

        final = base * mult
        floor = float(self.risk.get("min_risk_per_trade_pct", 0.1))
        final = max(floor, min(final, base))
        if applied:
            logger.debug("Risk/trade %.3f%% (base %.3f%%) %s", final, base, applied)
        return final

    def contrarian_haircut(self, contrarian_share: float) -> float:
        """Risk multiplier for a view that is fading the move.

        A fade enters earlier than a confirmation signal and is right more
        often about the turn, but it is betting against whatever is
        currently working — so when it is wrong, it is wrong into a move
        that is still running. Same dollar risk at the stop, but the stop
        is likelier to gap through it, so the position is cut.
        """
        share = max(0.0, min(1.0, float(contrarian_share)))
        floor = float(self.risk.get("contrarian_risk_factor", 0.7))
        return 1.0 - (1.0 - floor) * share

    def size_order(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        atr: float,
        equity: float,
        risk_pct: float,
        annualized_vol: float | None = None,
        min_qty: float = 0.0,
        qty_step: float = 0.0,
        min_notional: float = 0.0,
        asset_class: str = "",
    ) -> SizedOrder:
        """Size a position so a stop-out costs exactly `risk_pct` of equity.

        The stop width comes from the asset class's trading profile rather
        than one global multiple. A perpetual takes a stop twice as wide
        as spot, because leverage makes the margin for it affordable and
        the measured problem is stop-outs on moves that later reversed —
        65% of all exits over a 90-day replay were stops.
        """
        caps: list[str] = []
        profile = trading_profile(asset_class, self.config)
        k = float(profile.get("atr_stop_mult")
                  or self.stops.get("atr_stop_mult", 2.0))
        stop_distance = k * atr

        # A degenerate ATR (flat or missing data) would size the position at
        # infinity, so fall back to a floor expressed in basis points.
        min_stop_bps = float(self.stops.get("min_stop_bps", 30))
        floor_distance = entry_price * min_stop_bps / 10_000
        if not math.isfinite(stop_distance) or stop_distance < floor_distance:
            stop_distance = floor_distance
            caps.append("min_stop_floor")

        max_stop_bps = float(self.stops.get("max_stop_bps", 1200))
        ceiling = entry_price * max_stop_bps / 10_000
        if stop_distance > ceiling:
            stop_distance = ceiling
            caps.append("max_stop_cap")

        direction = 1 if side == "long" else -1
        stop_price = entry_price - direction * stop_distance
        r_mult = float(self.stops.get("target_r_multiple", 2.0))
        take_profit = entry_price + direction * stop_distance * r_mult

        risk_usd = equity * risk_pct / 100.0
        quantity = risk_usd / stop_distance if stop_distance > 0 else 0.0
        notional = quantity * entry_price

        # Cap 1: notional as a share of equity.
        max_notional = equity * float(self.sizing.get("max_position_pct", 20.0)) / 100.0
        if notional > max_notional:
            quantity = max_notional / entry_price
            notional = max_notional
            caps.append("max_position_pct")

        # Cap 2: volatility target — hold each position's annualised vol
        # contribution under the per-slot budget.
        if annualized_vol and annualized_vol > 0:
            target = float(self.sizing.get("vol_target_annual_pct", 40.0)) / 100.0
            slots = max(1, int(self.risk.get("max_open_positions", 5)))
            budget = equity * target / math.sqrt(slots)
            vol_capped_notional = budget / annualized_vol
            if vol_capped_notional < notional:
                quantity = vol_capped_notional / entry_price
                notional = vol_capped_notional
                caps.append("vol_target")

        # Exchange constraints.
        if qty_step and qty_step > 0:
            quantity = math.floor(quantity / qty_step) * qty_step
            notional = quantity * entry_price
        if min_qty and quantity < min_qty:
            quantity = 0.0
            notional = 0.0
            caps.append("below_min_qty")
        floor_notional = max(min_notional, float(self.sizing.get("min_position_usd", 25.0)))
        if quantity > 0 and notional < floor_notional:
            quantity = 0.0
            notional = 0.0
            caps.append("below_min_notional")

        # Leverage does not change what the trade risks — the stop is
        # where it is, and a stop-out costs the same dollars either way.
        # What it changes is how much cash the position ties up: at 3x the
        # margin is a third of the notional, so the account can carry the
        # position and still have cash for the rest of the book. That was
        # a real constraint, not a theoretical one — an equity entry was
        # refused for want of $1,300 while the crypto book sat on the cash.
        #
        # Safety check: the stop must sit well inside the liquidation
        # price, or leverage converts a normal loss into a total one. At
        # 3x, liquidation is roughly 33% away and a 4xATR stop on a 1% ATR
        # instrument is 4% away, so there is an order of magnitude of
        # room. Where that is not true, the leverage is reduced until it
        # is.
        max_leverage = float(profile.get("max_leverage", 1.0) or 1.0)
        leverage = 1.0
        if max_leverage > 1.0 and entry_price > 0 and stop_distance > 0:
            leverage = max_leverage
            stop_fraction = stop_distance / entry_price
            maintenance = float(self.risk.get("maintenance_margin_rate", 0.005))
            buffer = float(self.risk.get("liquidation_buffer", 3.0))
            # Liquidation sits at about (1/leverage - maintenance) away.
            # Require the stop to be `buffer` times closer than that.
            while leverage > 1.0 and \
                    (1.0 / leverage - maintenance) < stop_fraction * buffer:
                leverage -= 0.5
            leverage = max(1.0, round(leverage, 2))
            if leverage < max_leverage:
                caps.append("liquidation_buffer")

        return SizedOrder(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            stop_price=stop_price,
            take_profit=take_profit,
            # Recompute from the final quantity: caps mean actual risk is
            # often below the budget, and the heat check must see the truth.
            risk_usd=quantity * stop_distance,
            notional=notional,
            leverage=round(leverage, 2),
            atr=atr,
            r_distance=stop_distance,
            caps_applied=caps,
            asset_class=asset_class,
        )

    # ── Portfolio gates ───────────────────────────────────────

    def portfolio_heat(self, positions: list, equity: float) -> float:
        """Open risk as a percentage of equity, if every stop were hit."""
        if equity <= 0:
            return 100.0
        at_risk = 0.0
        for p in positions:
            risk_per_unit = abs(p.entry_price - p.stop_price)
            at_risk += risk_per_unit * p.quantity
        return at_risk / equity * 100.0

    def net_beta_exposure(self, positions: list, equity: float,
                          group: str | None = None) -> float:
        """Signed market exposure in beta-weighted units of equity.

        With `group`, only positions loading on that risk factor count.
        Each group's beta is measured against its own factor (crypto vs
        BTC, equities vs SPY), so summing across groups would add numbers
        that are not in the same units.
        """
        if equity <= 0:
            return 0.0
        total = 0.0
        for p in positions:
            if group is not None and _position_group(p) != group:
                continue
            total += (p.direction * symbol_beta(p.symbol, self.beta_book)
                      * p.entry_price * p.quantity)
        return total / equity

    def check_new_trade(
        self,
        order: SizedOrder,
        positions: list,
        equity: float,
        day_start_equity: float,
        peak_equity: float,
        consecutive_losses: int = 0,
        cooldown_until: datetime | None = None,
        now: datetime | None = None,
    ) -> RiskDecision:
        """Run every portfolio gate against a proposed trade."""
        now = now or datetime.now(timezone.utc)
        details: dict = {}

        if not order.valid:
            return RiskDecision(False, "order_invalid", {"caps": order.caps_applied})

        # Total drawdown halt.
        if peak_equity > 0:
            dd = (peak_equity - equity) / peak_equity * 100.0
            details["drawdown_pct"] = round(dd, 3)
            max_dd = float(self.risk.get("max_total_drawdown_pct", 15.0))
            if dd >= max_dd:
                return RiskDecision(False, "max_drawdown_halt", details)

        # Daily loss circuit breaker, measured on equity so unrealised
        # losses count too.
        if day_start_equity > 0:
            day_pnl_pct = (equity - day_start_equity) / day_start_equity * 100.0
            details["day_pnl_pct"] = round(day_pnl_pct, 3)
            max_daily_loss = float(self.risk.get("max_daily_loss_pct", 2.0))
            if day_pnl_pct <= -abs(max_daily_loss):
                return RiskDecision(False, "daily_loss_limit", details)
            profit_lock = float(self.risk.get("daily_profit_lock_pct", 0) or 0)
            if profit_lock > 0 and day_pnl_pct >= profit_lock:
                return RiskDecision(False, "daily_profit_lock", details)

        # Loss cooldown.
        if cooldown_until and now < cooldown_until:
            details["cooldown_until"] = cooldown_until.isoformat()
            return RiskDecision(False, "cooldown", details)

        # Position count.
        max_open = int(self.risk.get("max_open_positions", 5))
        if len(positions) >= max_open:
            details["open_positions"] = len(positions)
            return RiskDecision(False, "max_open_positions", details)

        # One position per symbol; adding to a loser is how accounts die.
        if any(p.symbol == order.symbol for p in positions):
            return RiskDecision(False, "symbol_already_open", details)

        # Portfolio heat.
        current_heat = self.portfolio_heat(positions, equity)
        new_heat = current_heat + (order.risk_usd / equity * 100.0 if equity > 0 else 100.0)
        details["heat_pct"] = round(current_heat, 3)
        details["heat_after_pct"] = round(new_heat, 3)
        max_heat = float(self.risk.get("max_portfolio_heat_pct", 4.0))
        if new_heat > max_heat:
            return RiskDecision(False, "max_portfolio_heat", details)

        # Crowding, within the risk factor the order loads on. Counting
        # every same-side position in the book instead would call three
        # long stocks and a long perp four correlated bets; they are two
        # factors of two, and blocking the fourth denies the book the
        # diversification that motivated holding both classes.
        group = order.group
        same_side = [p for p in positions
                     if p.side == order.side and _position_group(p) == group]
        details["group"] = group
        details["same_side_in_group"] = len(same_side)
        max_corr = int(self.risk.get("max_correlated_positions", 3))
        if len(same_side) >= max_corr:
            return RiskDecision(False, "max_correlated_positions", details)

        # Counting positions is only honest when they are independent.
        # Altcoins run a beta of 0.85-1.44 to Bitcoin with an average
        # pairwise correlation of 0.62, so five long alts is about 1.7
        # independent bets — and a book that thinks it holds five has
        # understated its concentration fivefold.
        min_effective = float(self.risk.get("min_effective_positions", 0.0) or 0.0)
        if min_effective > 0 and self.beta_book is not None and same_side:
            symbols = [p.symbol for p in same_side] + [order.symbol]
            effective = self.beta_book.effective_positions(symbols)
            details["positions"] = len(symbols)
            details["effective_positions"] = round(effective, 2)
            # Require each additional position to buy some genuine
            # diversification rather than more of the same trade.
            if effective < min_effective * len(symbols):
                return RiskDecision(False, "too_correlated", details)

        # Net beta exposure cap, also per factor.
        beta_now = self.net_beta_exposure(positions, equity, group=group)
        order_beta = (1 if order.side == "long" else -1) \
            * symbol_beta(order.symbol, self.beta_book) \
            * order.notional / equity if equity > 0 else 0.0
        beta_after = beta_now + order_beta
        details["net_beta"] = round(beta_now, 3)
        details["net_beta_after"] = round(beta_after, 3)
        max_beta = float(self.risk.get("max_net_beta", 1.5))
        if abs(beta_after) > max_beta:
            return RiskDecision(False, "max_net_beta", details)

        return RiskDecision(True, "ok", details)

    # ── Cost gate ─────────────────────────────────────────────

    def clears_costs(self, order: "SizedOrder", conviction: float,
                     round_trip_bps: float) -> tuple[bool, dict]:
        """Does the expected move cover the cost of making it?

        Two gates, because they catch different mistakes.

        **Cost in units of risk.** The round trip is a toll on notional;
        R is the distance to the stop. So the toll as a fraction of what
        the trade risks is

            cost_R = round_trip_bps / (10,000 * atr_stop_mult * atr_pct)

        and it depends on nothing but the fee schedule, the stop width and
        the instrument's own volatility. A market whose ATR is 0.5% of
        price costs twice as much per unit of risk as one at 1.0%, for an
        identical signal. This is the gate that matters: over a 90-day
        replay the strategies found +0.024R per trade gross and paid
        0.043R per trade in costs, so execution took the entire edge and
        then some. Refusing trades whose toll is a large share of their
        risk is the direct answer, and unlike a conviction threshold it
        needs no estimate of anything.

        **Expected move against the toll.** The older gate, kept because
        it catches a different case — a stop so tight the target is inside
        the noise. Its expected move assumes the target is reached, which
        it was on 19% of trades, so it is deliberately the looser of the
        two and is not relied on alone.
        """
        if order.entry_price <= 0 or order.r_distance <= 0:
            return False, {"reason": "no price or stop distance"}

        # What the round trip costs, measured in R.
        cost_r = (round_trip_bps / 10_000.0) * order.entry_price / order.r_distance

        target_r = float(self.stops.get("target_r_multiple", 2.0))
        # Expected gross move, scaled by how convinced the book is: a
        # marginal signal should not be credited with the full target.
        expected_move = order.r_distance * target_r * max(0.0, min(1.0, conviction))
        expected_bps = expected_move / order.entry_price * 10_000

        required = float(self.risk.get("min_edge_cost_ratio", 3.0))
        ratio = expected_bps / round_trip_bps if round_trip_bps > 0 else float("inf")
        max_cost_r = float(self.risk.get("max_cost_r", 0.0) or 0.0)

        details = {
            "expected_bps": round(expected_bps, 2),
            "cost_bps": round(round_trip_bps, 2),
            "ratio": round(ratio, 2),
            "required": required,
            "cost_r": round(cost_r, 4),
            "max_cost_r": max_cost_r,
        }

        if max_cost_r > 0 and cost_r > max_cost_r:
            details["reason"] = (f"round trip is {cost_r:.3f}R "
                                 f"(cap {max_cost_r:.3f}R)")
            return False, details
        if ratio < required:
            details["reason"] = f"edge {ratio:.1f}x costs (needs {required:.1f}x)"
            return False, details
        return True, details

    # ── Exit rules ────────────────────────────────────────────

    def update_stop(self, position, price: float, atr: float) -> tuple[float, str]:
        """Ratchet a stop forward. Returns (new_stop, reason) — a stop only
        ever moves in the direction of profit."""
        stop = position.stop_price
        reason = ""
        direction = position.direction
        r = position.unrealized_r(price)

        breakeven_at = float(self.stops.get("breakeven_at_r", 1.0))
        if breakeven_at > 0 and r >= breakeven_at and not position.moved_to_breakeven:
            # Park the stop just past entry so the trade cannot turn into a
            # loser, allowing for the round-trip fee.
            pad = position.entry_price * float(self.stops.get("breakeven_pad_bps", 8)) / 10_000
            candidate = position.entry_price + direction * pad
            if (direction == 1 and candidate > stop) or (direction == -1 and candidate < stop):
                stop = candidate
                reason = "breakeven"

        trail_after = float(self.stops.get("trail_after_r", 1.0))
        trail_mult = float(self.stops.get("trail_atr_mult", 2.5))
        if trail_mult > 0 and atr > 0 and r >= trail_after:
            anchor = position.best_price if direction == 1 else position.worst_price
            candidate = anchor - direction * trail_mult * atr
            if (direction == 1 and candidate > stop) or (direction == -1 and candidate < stop):
                stop = candidate
                reason = "atr_trail"

        return stop, reason

    def time_stop_hit(self, position, bars_held: int) -> bool:
        limit = int(self.stops.get("time_stop_bars", 0) or 0)
        return limit > 0 and bars_held >= limit

    def max_hold_exceeded(self, position, now: datetime, max_days: int) -> bool:
        if max_days <= 0:
            return False
        return now - position.opened_at >= timedelta(days=max_days)
