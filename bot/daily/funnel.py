"""Where signals are lost, stage by stage.

"No strategy has a view" is an outcome, not a diagnosis. It fired 3,022
times in one replay and could equally have meant a genuine absence of
setups, an indicator that never warmed up, a strategy that was not
loaded, or a filter that silently ate everything. The distinction
matters: the first is the market, the rest are bugs.

So the funnel counts every stage, and — this is the part that ordinary
rejection logging gets wrong — it records EVERY condition a symbol
failed, not just the first. A planner that `continue`s on the first
failure reports only the outermost reason, which hides the fact that a
signal would have died three stages later anyway. Fixing the reported
reason then changes nothing and looks inexplicable.

It also separates opportunities from evaluations. The same symbol
examined at six entry slots in one day is one opportunity and six
evaluations, and reporting the larger number makes a quiet system look
busy.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

# In the order a signal must survive them.
STAGES = [
    "examined",           # symbol looked at, at some entry slot
    "tradable",           # passed the liquidity and volatility gates
    "has_view",           # at least one strategy produced a signal
    "survived_tilts",     # news and regime did not flip or mute it
    "direction_allowed",  # shortable if short; with the primary trend
    "clears_cost",        # expected move beats this instrument's round trip
    "sized",              # risk budget produced a valid order
    "accepted",           # passed portfolio heat, correlation, daily caps
]


@dataclass
class Funnel:
    """Counts per stage, plus why things died at each one."""

    stage: Counter = field(default_factory=Counter)
    reasons: defaultdict = field(default_factory=lambda: defaultdict(Counter))
    # Unique symbol-days, so repeated slot evaluations do not inflate it.
    opportunities: set = field(default_factory=set)
    evaluations: int = 0

    def examined(self, symbol: str, day: str) -> None:
        self.evaluations += 1
        self.opportunities.add((symbol, day))
        self.stage["examined"] += 1

    def passed(self, stage: str) -> None:
        self.stage[stage] += 1

    def died(self, stage: str, reason: str) -> None:
        self.reasons[stage][reason] += 1

    def merge(self, other: "Funnel") -> None:
        self.stage.update(other.stage)
        for k, v in other.reasons.items():
            self.reasons[k].update(v)
        self.opportunities |= other.opportunities
        self.evaluations += other.evaluations

    def render(self) -> str:
        top = self.stage.get("examined", 0)
        if not top:
            return "Nothing was examined — the universe or the schedule is empty."

        out = [
            f"{self.evaluations} evaluations of "
            f"{len(self.opportunities)} unique symbol-days",
            "",
            f"  {'stage':<20}{'reached':>9}{'of top':>9}   lost here",
        ]
        prev = top
        for stage in STAGES:
            n = self.stage.get(stage, 0)
            lost = prev - n if stage != "examined" else 0
            share = f"{n / top * 100:5.1f}%"
            out.append(f"  {stage:<20}{n:>9}{share:>9}   {lost if lost > 0 else ''}")
            for reason, count in Counter(self.reasons.get(stage, {})).most_common(4):
                out.append(f"  {'':<20}{'':>9}{'':>9}     {count:>5}  {reason}")
            prev = n
        return "\n".join(out)
