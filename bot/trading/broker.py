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

    def scale_out(
        self,
        position: Position,
        price: float,
        fraction: float,
        reason: str = "take_partial",
        now: datetime | None = None,
        adv_notional: float | None = None,
        spread_bps: float | None = None,
        daily_vol_bps: float | None = None,
    ) -> float:
        """Sell part of a position and bank the profit. Returns cash realised.

        A trend system's usual complaint is that it gives back most of an
        open gain waiting for the trailing stop. Taking a slice off at a
        set multiple of risk converts some of that paper profit into cash
        while leaving the rest to run — the position keeps its stop, its
        target and its identity, so nothing downstream has to know it
        happened beyond the smaller size.

        The remaining position keeps the *original* entry price and stop,
        so its R multiple still measures against what was actually risked
        at the outset rather than against a re-based cost.
        """
        fraction = max(0.0, min(1.0, float(fraction)))
        quantity = position.quantity * fraction
        if quantity <= 0 or fraction <= 0:
            return 0.0
        now = now or datetime.now(timezone.utc)

        notional = quantity * price
        fill, slip = self._fill_price(
            price, position.side, notional, False, adv_notional,
            spread_bps, daily_vol_bps,
        )
        gross = (fill - position.entry_price) * quantity * position.direction
        fee = quantity * fill * self.taker_bps / 10_000
        net = gross - fee

        # Release the margin this slice was holding, and the cash it made.
        released = position.margin * fraction
        position.margin -= released
        self.cash += released + net
        self.fees_paid += fee
        self.slippage_cost += slip

        position.quantity -= quantity
        position.realized_pnl += net
        position.realized_fees += fee
        risk_per_unit = abs(position.entry_price - position.initial_stop)
        at_r = ((fill - position.entry_price) * position.direction / risk_per_unit
                if risk_per_unit > 0 else 0.0)
        position.scaled_out_at.append(round(at_r, 3))

        logger.info(
            "  Booked %.0f%% of %s %s at %.2fR — $%+.2f realised, %.4f left",
            fraction * 100, position.side, position.symbol, at_r, net,
            position.quantity,
        )
        return net

    def add_to(
        self,
        position: Position,
        order: "SizedOrder",
        now: datetime | None = None,
        adv_notional: float | None = None,
        spread_bps: float | None = None,
        daily_vol_bps: float | None = None,
    ) -> bool:
        """Add a unit to a position that is working. Returns whether it filled.

        This is the entry-side counterpart of a trailing stop: the trail
        protects a winner, and this one presses it. The combined position
        is carried at a weighted-average entry, and the stop is pulled up
        so that the whole position still risks what the original unit
        risked.

        That last part was the bug that made this strategy lose money.
        The old code left the stop where it was, on the stated grounds
        that moving it would "quietly increase the risk it was sized
        for". The opposite is true: leaving it still adds a second unit
        with its own full stop distance, so the risk grows with the
        size. Measured on a 10-unit position risking $40, one half-size
        add took the real risk to $80 — exactly double — and the
        stop-out lost $83.60, or 1.57R.

        Across 87 replayed trades that showed up as an average loss of
        1.43R against an average win of 0.77R, with 92% of losses
        exceeding the 1R the position was supposedly sized to. It also
        produced the contradiction of a "breakeven" stop losing 1.25R:
        the stop was parked at the ORIGINAL entry while the averaged
        entry had moved above it, so breakeven was a loss by
        construction.

        The stop now solves for the distance that keeps total risk at
        the original budget: new_stop = avg_entry -/+ risk_usd / qty.
        Pyramiding stays what it is meant to be — more size on a winner,
        not more risk.
        """
        now = now or datetime.now(timezone.utc)
        quantity = float(order.quantity)
        if quantity <= 0:
            return False

        notional = quantity * order.entry_price
        fill, slip = self._fill_price(
            order.entry_price, position.side, notional, True, adv_notional,
            spread_bps, daily_vol_bps,
        )
        fill_notional = quantity * fill
        leverage = max(1.0, float(position.leverage))
        margin = fill_notional / leverage
        fee = fill_notional * self.taker_bps / 10_000
        if margin + fee > self.cash:
            logger.info("  No cash to add to %s (need $%.2f, have $%.2f)",
                        position.symbol, margin + fee, self.cash)
            return False

        self.cash -= margin + fee
        self.fees_paid += fee
        self.slippage_cost += slip

        total = position.quantity + quantity
        position.entry_price = (
            (position.entry_price * position.quantity + fill * quantity) / total
        )
        position.quantity = total
        position.margin += margin
        position.entry_fee += fee
        position.units += 1

        # Hold total risk at the original budget. Without this the added
        # unit carries its own full stop distance and the position risks
        # a multiple of what it was sized for.
        direction = position.direction
        budget = position.risk_usd
        if budget > 0 and total > 0:
            required = budget / total
            tightened = position.entry_price - direction * required
            # Only ever pull the stop closer. If the trail has already
            # moved it past this point, the trail is the tighter of the
            # two and stays.
            if ((direction == 1 and tightened > position.stop_price)
                    or (direction == -1 and tightened < position.stop_price)):
                logger.info("  %s stop tightened for the added unit: "
                            "%.6f → %.6f (risk held at $%.2f)",
                            position.symbol, position.stop_price, tightened, budget)
                position.stop_price = tightened

        logger.info(
            "  Added unit %d to %s %s at %.4f — size now %.4f, risk $%.2f",
            position.units, position.side, position.symbol, fill, total,
            abs(position.entry_price - position.stop_price) * total,
        )
        return True

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
        total_fees = position.entry_fee + position.realized_fees + exit_fee
        # Two different numbers, and conflating them is an accounting bug
        # that made every trade look better than it was.
        #
        # `cash_net` is what moves the balance at this moment. It excludes
        # the entry fee because that already left cash when the position
        # opened; subtracting it again here would charge it twice.
        #
        # `net` is the round trip as a TRADE — what the whole thing made,
        # entry fee included. The old code used the cash figure for both,
        # so `pnl` omitted the entry fee while `fees` reported it. The
        # equity curve was right and every per-trade P&L and R was too
        # generous, by about 0.13R at these sizes, on all 43 trades of
        # the last replay. A reviewer found the gap by hand on one ETH
        # trade before the ledger could show it.
        cash_net = gross - exit_fee - position.funding_paid

        self.cash += position.margin + cash_net
        net = cash_net - position.entry_fee
        self.fees_paid += exit_fee
        self.slippage_cost += slip

        # Profit already banked by scaling out belongs to this trade. A
        # position that sold half at 2R and then stopped at breakeven made
        # money; reporting only the final leg would show it as flat and
        # make every scale-out look like a wasted trade in the record.
        net += position.realized_pnl

        # R is measured against what was originally risked, not against
        # whatever quantity happens to be left after scaling out.
        risk_per_unit = abs(position.entry_price - position.initial_stop)
        sized_quantity = position.original_quantity or position.quantity
        initial_risk = position.risk_usd
        if initial_risk <= 0:
            raise ValueError("Position has no frozen initial dollar risk")
        r_multiple = net / initial_risk

        if position.direction == 1:
            mfe = (position.best_price - position.entry_price)
            mae = (position.worst_price - position.entry_price)
        else:
            mfe = (position.entry_price - position.worst_price)
            mae = (position.entry_price - position.best_price)
        mfe_r = mfe / risk_per_unit if risk_per_unit > 0 else 0.0
        mae_r = mae / risk_per_unit if risk_per_unit > 0 else 0.0

        denom = position.entry_price * (position.original_quantity
                                        or position.quantity)
        trade = Trade(
            id=position.id,
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=fill,
            # The size the trade was sized at, so the record describes the
            # position that was taken rather than its final remnant.
            quantity=position.original_quantity or position.quantity,
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
            units=position.units,
            original_quantity=round(sized_quantity, 10),
            initial_stop=round(position.initial_stop, 10),
            final_stop=round(position.stop_price, 10),
            realized_before_exit=round(position.realized_pnl, 10),
            exit_quantity=round(position.quantity, 10),
            scale_out_fees=position.realized_fees,
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

        A stop that has been trailed past the entry is a different event
        from the one the position was opened with: it banks a profit. Both
        reported as "stop_loss" makes the exit table unreadable — a
        ten-day replay showed stop_loss exits at +0.54R, +0.80R and +0.47R
        sitting alongside real losses, so the column said nothing about
        how trades actually ended.
        """
        if position.direction == 1:
            hit_stop = low <= position.stop_price
            hit_target = position.take_profit > 0 and high >= position.take_profit
            in_profit = position.stop_price > position.entry_price
        else:
            hit_stop = high >= position.stop_price
            hit_target = position.take_profit > 0 and low <= position.take_profit
            in_profit = position.stop_price < position.entry_price

        if hit_stop:
            if in_profit:
                return "trailing_stop"
            if position.moved_to_breakeven:
                return "breakeven_stop"
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
