"""FX operations: currency holiday calendars (USD bank holidays, TARGET2, UK, Tokyo), spot and forward value dates
(T+2 with the currency-centre rule, modified following, end-end), forward pricing by covered interest parity on the
overnight rates, the hedge book (sell the non-dollar equity exposure forward, rebalanced weekly, rolled at
maturity), settlement instructions per value date and their CLS-style multilateral netting, and confirmations."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np

from . import book as B, valuation as V
from .xops_bridge import cal

ONE = dt.timedelta(days=1)


# ---- currency calendars ---------------------------------------------------------------------------------------------------
def _target2(year: int) -> set[dt.date]:
    e = cal.easter(year)
    return {dt.date(year, 1, 1), e - dt.timedelta(days=2), e + ONE, dt.date(year, 5, 1), dt.date(year, 12, 25), dt.date(year, 12, 26)}


def _usd_bank(year: int) -> set[dt.date]:
    h = set(cal.nyse_holidays(year))
    h.add(cal.nth_weekday(year, 10, 0, 2))                # Columbus Day: banks closed, NYSE open
    v = dt.date(year, 11, 11); h.add(cal.observed_us(v) or v)   # Veterans Day
    return h


def _jpy(year: int) -> set[dt.date]:
    fixed = [(1, 1), (1, 2), (1, 3), (2, 11), (2, 23), (4, 29), (5, 3), (5, 4), (5, 5), (8, 11), (11, 3), (11, 23), (12, 31)]
    h = {dt.date(year, m, d) for m, d in fixed}
    h |= {cal.nth_weekday(year, 1, 0, 2), cal.nth_weekday(year, 7, 0, 3), cal.nth_weekday(year, 9, 0, 3), cal.nth_weekday(year, 10, 0, 2)}
    h.add(dt.date(year, 3, 20)); h.add(dt.date(year, 9, 23))      # equinox days (20 or 21 March, 22 or 23 September; the approximation is stated)
    # substitute holiday: a holiday on a Sunday moves to the next non-holiday weekday
    out = set(h)
    for d in sorted(h):
        if d.weekday() == 6:
            s = d + ONE
            while s in out or s.weekday() >= 5:
                s += ONE
            out.add(s)
    return out


HOLIDAYS = {"USD": _usd_bank, "EUR": _target2, "GBP": lambda y: set(cal.lse_holidays(y)), "JPY": _jpy}


def is_business_day(ccy: str, d: dt.date) -> bool:
    return d.weekday() < 5 and d not in HOLIDAYS[ccy](d.year)


def good_day(ccys: tuple, d: dt.date) -> bool:
    return all(is_business_day(c, d) for c in ccys)


def spot_date(pair: str, d: dt.date) -> dt.date:
    """T+2: the intermediate day must be a business day in the non-dollar currency (a USD holiday on T+1 does not
    count), the value date must be a business day in both currencies"""
    base, quote = B.CCY_OF_PAIR[pair]; non_usd = tuple(c for c in (base, quote) if c != "USD")
    t1 = d + ONE
    while not good_day(non_usd, t1):
        t1 += ONE
    v = t1 + ONE
    while not good_day((base, quote), v):
        v += ONE
    return v


def forward_date(pair: str, spot: dt.date, months: int) -> dt.date:
    """spot plus months, modified following on both calendars, with the end-end rule"""
    base, quote = B.CCY_OF_PAIR[pair]; ccys = (base, quote)
    import calendar as _c
    nxt = spot + ONE
    while not good_day(ccys, nxt):
        nxt += ONE
    if nxt.month != spot.month:                       # spot is the last business day of its month: end-end
        v = B.month_add(spot, months); v = dt.date(v.year, v.month, _c.monthrange(v.year, v.month)[1])
        while not good_day(ccys, v):
            v -= ONE
        return v
    v = B.month_add(spot, months); m = v.month
    while not good_day(ccys, v):
        v += ONE
    if v.month != m:                                  # modified following: do not cross the month end
        v = B.month_add(spot, months)
        while not good_day(ccys, v):
            v -= ONE
    return v


# ---- the hedge book ----------------------------------------------------------------------------------------------------------
@dataclass
class Forward:
    inst: B.Instrument; qty: float          # base-currency notional, negative = sold base


@dataclass
class DealerQuote:
    counterparty: str; forward: float; points_bp: float


def dealer_quotes(pair: str, spot: float, rates: dict, days: int, side: int, counterparties: tuple, rng: np.random.Generator) -> list[DealerQuote]:
    """four dealers around the parity forward; the spread is a few tenths of a pip on the majors (simulated)"""
    f = V.forward_rate(pair, spot, rates, days); pip = 0.0001 if pair != "USDJPY" else 0.01
    out = []
    for c in counterparties:
        half = pip * float(rng.uniform(0.3, 0.9)); q = f + side * half + pip * float(rng.normal(0, 0.15))
        out.append(DealerQuote(c, q, (q - f) / spot * 1e4))
    return out


def best_quote(quotes: list[DealerQuote], side: int) -> DealerQuote:
    return min(quotes, key=lambda q: side * q.forward)


def settlement_instructions(fwd: Forward, value_date: dt.date) -> list[dict]:
    """what settles: the base notional one way and the quote notional at the strike the other way"""
    base, quote = B.CCY_OF_PAIR[fwd.inst.pair]
    return [{"value_date": value_date, "trade_date": fwd.inst.trade_date, "pair": fwd.inst.pair, "counterparty": fwd.inst.counterparty, "ccy": base, "amount": fwd.qty},
            {"value_date": value_date, "trade_date": fwd.inst.trade_date, "pair": fwd.inst.pair, "counterparty": fwd.inst.counterparty, "ccy": quote, "amount": -fwd.qty * fwd.inst.strike}]


def net_settlements(instr: list[dict]) -> dict:
    """gross versus bilateral (per counterparty and currency) versus multilateral (per currency, CLS-style) settlement"""
    gross = sum(abs(i["amount_usd"]) for i in instr)
    bil = {}
    for i in instr:
        k = (i["counterparty"], i["ccy"]); bil[k] = bil.get(k, 0.0) + i["amount_usd"]
    bilateral = sum(abs(v) for v in bil.values())
    mul = {}
    for i in instr:
        mul[i["ccy"]] = mul.get(i["ccy"], 0.0) + i["amount_usd"]
    multilateral = sum(abs(v) for v in mul.values())
    return {"gross_usd": gross, "bilateral_usd": bilateral, "multilateral_usd": multilateral, "instructions": len(instr), "by_ccy_net": mul}
