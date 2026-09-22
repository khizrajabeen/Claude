"""Paper broker — simulated execution with honest accounting.

What the old engine got wrong and this one does not:

  * Margin is actually reserved on open, so two positions cannot both spend
    the same dollar.
  * Equity is cash plus unrealised PnL, so the daily-loss and drawdown
    breakers see a losing open position instead of only realised damage.
  * Slippage is charged on notional (plus a size-vs-liquidity impact term),
    not on a base-unit quantity, so 0.1 BTC and 0.1 SOL are not treated as
    the same order.
  * Perpetual funding accrues against held positions at 00:00/08:00/16:00
    UTC, which is a real cost of carrying a position overnight.
  * MAE/MFE are tracked in R, which is what tells you whether stops are too
    tight or targets too far.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from bot.daily.schedule import funding_events_between
from bot.markets.instrument import AssetClass
from bot.risk.budget import SizedOrder
from bot.trading.models import Position, Trade

logger = logging.getLogger("trading_bot")


class InsufficientFunds(Exception):
    """Raised when an order cannot be funded — callers skip the trade."""


def _pays_funding(asset_class: str) -> bool:
    """Does this instrument have a perpetual funding leg?

    Unknown or missing classes are treated as perpetuals, which is what
    every record written before asset classes existed was.
    """
    try:
        return AssetClass(str(asset_class or "crypto_perp")).pays_funding
    except ValueError:
        return True


class PaperBroker:
    """Simulated execution venue with cash, margin and fee accounting."""

    def __init__(self, config: dict, cash: float | None = None,
                 positions: list[Position] | None = None,
                 trade_counter: int = 0):
        self.config = config
        paper = config.get("paper", {})
        self.maker_bps = float(paper.get("fee_maker_bps", 2.0))
        self.taker_bps = float(paper.get("fee_taker_bps", 5.5))
        self.base_slippage_bps = float(paper.get("slippage_bps", 3.0))
        self.impact_coefficient = float(paper.get("impact_coefficient", 0.5))
        self.reference_daily_vol_bps = float(paper.get("reference_daily_vol_bps", 300.0))
        self.max_slippage_bps = float(paper.get("max_slippage_bps", 100.0))
        self.funding_enabled = bool(paper.get("funding_enabled", True))
        self.default_funding_bps = float(paper.get("default_funding_bps", 1.0))

        initial = float(paper.get("initial_balance", 10000.0))
        self.cash = initial if cash is None else float(cash)
        self.positions: list[Position] = list(positions or [])
        self.trade_counter = int(trade_counter)
        self.closed_today: list[Trade] = []
        self.fees_paid = 0.0
        self.funding_paid = 0.0
        self.slippage_cost = 0.0

    # ── Valuation ─────────────────────────────────────────────

    def reserved_margin(self) -> float:
        return sum(p.margin for p in self.positions)

    def unrealized_pnl(self, prices: dict[str, float]) -> float:
        total = 0.0
        for p in self.positions:
            price = prices.get(p.symbol)
            if price is not None:
                total += p.unrealized_pnl(price)
        return total

    def equity(self, prices: dict[str, float]) -> float:
        """Cash + reserved margin + open PnL. This is what the risk gates
        are measured against."""
        return self.cash + self.reserved_margin() + self.unrealized_pnl(prices)

    def free_cash(self) -> float:
        return self.cash

    # ── Execution ─────────────────────────────────────────────

    def _fill_price(self, price: float, side: str, notional: float,
                    is_entry: bool, adv_notional: float | None,
                    spread_bps: float | None,
                    daily_vol_bps: float | None = None) -> tuple[float, float]:
        """Apply slippage. Returns (fill_price, slippage_cost_usd).

        Impact follows the square-root law: cost is proportional to the
        asset's own volatility times the square root of the fraction of
        daily volume the order represents. Expressing it as raw basis
        points instead — as an earlier version did — charged ~27bps on a
        $1,500 order in a $50M book, which ate a quarter of every stop.
        """
        bps = self.base_slippage_bps
        if spread_bps is not None and spread_bps > 0:
            # Crossing the book costs half the spread on top of drift.
            bps += spread_bps / 2.0

        if adv_notional and adv_notional > 0 and notional > 0:
            participation = min(1.0, notional / adv_notional)
            vol_bps = daily_vol_bps if daily_vol_bps else self.reference_daily_vol_bps
            bps += self.impact_coefficient * vol_bps * math.sqrt(participation)

        bps = min(bps, self.max_slippage_bps)

        # Slippage always works against the trader: entries fill worse,
        # exits fill worse.
        if is_entry:
            adverse = 1 if side == "long" else -1
        else:
            adverse = -1 if side == "long" else 1

        fill = price * (1 + adverse * bps / 10_000)
        cost = abs(fill - price) * (notional / price) if price > 0 else 0.0
        return fill, cost

    def open(
        self,
        order: SizedOrder,
        now: datetime | None = None,
        day: str = "",
        atr: float | None = None,
        adv_notional: float | None = None,
        spread_bps: float | None = None,
        maker: bool = False,
        daily_vol_bps: float | None = None,
        strategy: str = "",
        entry_reason: str = "",
        asset_class: str = "crypto_spot",
        venue: str = "",
        timeframe: str = "",
        tags: dict | None = None,
    ) -> Position:
        """Open a position, charging fees and reserving margin."""
        if not order.valid:
            raise InsufficientFunds(f"invalid order for {order.symbol}")

        now = now or datetime.now(timezone.utc)
        notional = order.quantity * order.entry_price
        fill, slip = self._fill_price(
            order.entry_price, order.side, notional, True, adv_notional,
            spread_bps, daily_vol_bps,
        )

        fill_notional = order.quantity * fill
        fee_bps = self.maker_bps if maker else self.taker_bps
        fee = fill_notional * fee_bps / 10_000
        leverage = max(1.0, float(order.leverage or 1.0))
        margin = fill_notional / leverage

        if margin + fee > self.cash:
            raise InsufficientFunds(
                f"{order.symbol}: need ${margin + fee:,.2f}, free cash ${self.cash:,.2f}"
            )

        self.cash -= margin + fee
        self.fees_paid += fee
        self.slippage_cost += slip
        self.trade_counter += 1

        # The stop keeps its distance from the *fill*, not the quote we
        # sized against — otherwise slippage silently changes the risk.
        direction = 1 if order.side == "long" else -1
        stop_price = fill - direction * order.r_distance
        take_profit = fill + direction * order.r_distance * float(
            self.config.get("stops", {}).get("target_r_multiple", 2.0)
        )

        position = Position(
            id=f"t{self.trade_counter:06d}",
            symbol=order.symbol,
            side=order.side,
            entry_price=fill,
            quantity=order.quantity,
            leverage=leverage,
            opened_at=now,
            stop_price=stop_price,
            take_profit=take_profit,
            initial_stop=stop_price,
            risk_usd=order.quantity * order.r_distance,
            margin=margin,
            entry_fee=fee,
            atr_at_entry=atr if atr is not None else order.atr,
            best_price=fill,
            worst_price=fill,
            opened_on_day=day,
            strategy=strategy,
            entry_reason=entry_reason,
            asset_class=asset_class,
            venue=venue,
            timeframe=timeframe,
            tags=tags or {},
        )
        self.positions.append(position)

        logger.info(
            "OPEN  %-5s %-10s qty=%.6f @ %.6f | stop %.6f | tp %.6f | risk $%.2f | fee $%.2f",
            position.side.upper(), position.symbol, position.quantity,
            position.entry_price, position.stop_price, position.take_profit,
            position.risk_usd, fee,
        )
        return position

    def close(
        self,
        position: Position,
        price: float,
        reason: str = "signal",
        now: datetime | None = None,
        day: str = "",
        adv_notional: float | None = None,
        spread_bps: float | None = None,
        maker: bool = False,
        daily_vol_bps: float | None = None,
    ) -> Trade:
        """Close a position, release margin and book the round trip."""
        now = now or datetime.now(timezone.utc)
        notional = position.quantity * price
        fill, slip = self._fill_price(
            price, position.side, notional, False, adv_notional,
            spread_bps, daily_vol_bps,
        )

        gross = (fill - position.entry_price) * position.quantity * position.direction
        exit_notional = position.quantity * fill
        fee_bps = self.maker_bps if maker else self.taker_bps
        exit_fee = exit_notional * fee_bps / 10_000
        total_fees = position.entry_fee + exit_fee
        net = gross - exit_fee - position.funding_paid

        self.cash += position.margin + net
        self.fees_paid += exit_fee
        self.slippage_cost += slip

        risk_per_unit = abs(position.entry_price - position.initial_stop)
        r_multiple = (net / (risk_per_unit * position.quantity)) if risk_per_unit > 0 and position.quantity > 0 else 0.0

        if position.direction == 1:
            mfe = (position.best_price - position.entry_price)
            mae = (position.worst_price - position.entry_price)
        else:
            mfe = (position.entry_price - position.worst_price)
            mae = (position.entry_price - position.best_price)
        mfe_r = mfe / risk_per_unit if risk_per_unit > 0 else 0.0
        mae_r = mae / risk_per_unit if risk_per_unit > 0 else 0.0

        denom = position.entry_price * position.quantity
        trade = Trade(
            id=position.id,
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=fill,
            quantity=position.quantity,
            leverage=position.leverage,
            opened_at=position.opened_at,
            closed_at=now,
            pnl=net,
            pnl_pct=(net / denom * 100.0) if denom > 0 else 0.0,
            r_multiple=r_multiple,
            fees=total_fees,
            funding=position.funding_paid,
            slippage_cost=slip,
            exit_reason=reason,
            risk_usd=position.risk_usd,
            opened_on_day=position.opened_on_day,
            closed_on_day=day,
            strategy=position.strategy,
            entry_reason=position.entry_reason,
            asset_class=position.asset_class,
            venue=position.venue,
            timeframe=position.timeframe,
            mae_r=round(mae_r, 4),
            mfe_r=round(mfe_r, 4),
        )

        self.positions.remove(position)
        self.closed_today.append(trade)

        logger.info(
            "CLOSE %-5s %-10s @ %.6f | %s | PnL $%+.2f (%+.2fR) | cash $%.2f",
            position.side.upper(), position.symbol, fill, reason,
            net, r_multiple, self.cash,
        )
        return trade

    # ── Marking and exits ─────────────────────────────────────

    def mark(self, symbol: str, price: float) -> None:
        """Record the excursion extremes used by trailing stops and MAE/MFE."""
        for position in self.positions:
            if position.symbol != symbol:
                continue
            # "Best" means best for the position, so it is the high for a
            # long and the low for a short.
            if position.direction == 1:
                position.best_price = max(position.best_price, price)
                position.worst_price = min(position.worst_price, price)
            else:
                position.best_price = min(position.best_price, price)
                position.worst_price = max(position.worst_price, price)

    def stop_or_target_hit(self, position: Position, high: float, low: float) -> str | None:
        """Which barrier a bar touched.

        When a single bar spans both the stop and the target we assume the
        stop — the pessimistic assumption is the only honest one without
        tick data, and the optimistic one is how backtests lie.
        """
        if position.direction == 1:
            hit_stop = low <= position.stop_price
            hit_target = position.take_profit > 0 and high >= position.take_profit
        else:
            hit_stop = high >= position.stop_price
            hit_target = position.take_profit > 0 and low <= position.take_profit

        if hit_stop:
            return "stop_loss"
        if hit_target:
            return "take_profit"
        return None

    def liquidation_price(self, position: Position,
                          maintenance_margin_rate: float = 0.005) -> float | None:
        """Price at which margin is exhausted, for leveraged positions."""
        if position.leverage <= 1:
            return None
        move = position.entry_price * (1.0 / position.leverage - maintenance_margin_rate)
        return position.entry_price - position.direction * move

    # ── Funding ───────────────────────────────────────────────

    def accrue_funding(self, now: datetime, until: datetime,
                       rates: dict[str, float] | None = None,
                       prices: dict[str, float] | None = None) -> float:
        """Charge perpetual funding for settlements in (now, until].

        `rates` are per-settlement rates as decimals (Binance-style, e.g.
        0.0001 = 1bp per 8h). Longs pay a positive rate, shorts receive it.

        Only perpetuals pay it. A stock or an ETF has no funding leg at
        all, and charging one a crypto rate every eight hours quietly
        taxes the equity half of the book for a cost it never incurs —
        which would show up in a cross-asset comparison as equities
        underperforming.
        """
        if not self.funding_enabled:
            return 0.0
        events = funding_events_between(now, until)
        if not events:
            return 0.0

        rates = rates or {}
        prices = prices or {}
        total = 0.0
        for position in self.positions:
            if not _pays_funding(position.asset_class):
                continue
            rate = rates.get(position.symbol)
            if rate is None:
                rate = self.default_funding_bps / 10_000
            price = prices.get(position.symbol, position.entry_price)
            notional = position.quantity * price
            for _ in events:
                # Positive rate: longs pay, shorts are paid.
                charge = notional * rate * position.direction
                position.funding_paid += charge
                total += charge
        self.funding_paid += total
        if total:
            logger.info("Funding accrued over %d settlement(s): $%+.4f", len(events), -total)
        return total

    # ── Reporting ─────────────────────────────────────────────

    def positions_for(self, symbol: str) -> list[Position]:
        return [p for p in self.positions if p.symbol == symbol]

    def summary(self, prices: dict[str, float]) -> dict:
        return {
            "cash": round(self.cash, 2),
            "margin": round(self.reserved_margin(), 2),
            "unrealized": round(self.unrealized_pnl(prices), 2),
            "equity": round(self.equity(prices), 2),
            "open_positions": len(self.positions),
            "fees_paid": round(self.fees_paid, 2),
            "funding_paid": round(self.funding_paid, 4),
            "slippage_cost": round(self.slippage_cost, 2),
        }
