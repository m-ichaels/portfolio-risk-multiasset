"""Marks to USD for every asset class, and the FX-forward pricing the FX layer and the marks share.

  equity       qty x price x price_scale x USD-per-currency (London prices are in pence: price_scale 0.01)
  future       notional = qty x price x multiplier (USD contracts); no cash value, P&L is settlement variation
  fx forward   value = qty_base x (F_mkt(value date) - K) x USD-per-quote, with F from covered interest parity on the
               overnight rates (SOFR, euro STR act/360; SONIA, JGB act/365); the quote-leg discounting is omitted
  bond         par x (clean + accrued) / 100
"""
from __future__ import annotations

import datetime as dt

from . import book as B

DAYCOUNT = {"USD": 360, "EUR": 360, "GBP": 365, "JPY": 365}
OIS = {"USD": "SOFR", "EUR": "ESTR", "GBP": "SONIA", "JPY": "JGB1Y"}


def usd_per(ccy: str, fx: dict) -> float:
    """USD per one unit of ccy from spot pairs {EURUSD, GBPUSD, USDJPY}"""
    if ccy == "USD":
        return 1.0
    if ccy == "EUR":
        return fx["EURUSD"]
    if ccy == "GBP":
        return fx["GBPUSD"]
    if ccy == "JPY":
        return 1.0 / fx["USDJPY"]
    raise KeyError(ccy)


def forward_rate(pair: str, spot: float, rates: dict, days: int) -> float:
    """F = S (1 + r_quote tau_q) / (1 + r_base tau_b), rates in percent, simple compounding on each currency's day count"""
    base, quote = B.CCY_OF_PAIR[pair]
    rb = rates.get(OIS[base], 0.0) / 100.0; rq = rates.get(OIS[quote], 0.0) / 100.0
    return spot * (1 + rq * days / DAYCOUNT[quote]) / (1 + rb * days / DAYCOUNT[base])


def forward_value_usd(inst: B.Instrument, qty: float, d: dt.date, fx: dict, rates: dict) -> float:
    """MTM of a forward: base notional qty (signed, + = bought base) against strike K, settled in the quote currency"""
    days = max((inst.value_date - d).days, 0); f = forward_rate(inst.pair, fx[inst.pair], rates, days)
    base, quote = B.CCY_OF_PAIR[inst.pair]
    return qty * (f - inst.strike) * usd_per(quote, fx)


def mark(inst: B.Instrument, qty: float, px: float, d: dt.date, fx: dict, rates: dict | None = None) -> tuple[float, float]:
    """(market value in USD that sits in the NAV, exposure notional in USD) for one position"""
    if inst.asset_class == B.EQUITY:
        v = qty * px * inst.price_scale * usd_per(inst.currency, fx); return v, v
    if inst.asset_class == B.FUTURE:
        return 0.0, qty * px * inst.multiplier * usd_per(inst.currency, fx)
    if inst.asset_class == B.FX_FORWARD:
        base, _ = B.CCY_OF_PAIR[inst.pair]
        return forward_value_usd(inst, qty, d, fx, rates or {}), qty * usd_per(base, fx)
    if inst.asset_class == B.BOND:
        v = qty * (px + B.bond_accrued(inst, d)) / 100.0; return v, v
    raise ValueError(inst.asset_class)


def bond_dv01(inst: B.Instrument, par: float, px: float) -> float:
    """dollars per basis point, positive for a long"""
    return par * px / 100.0 * inst.duration / 1e4
