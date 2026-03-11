"""P&L attribution, daily and intraday, by strategy, asset class and component, exact by construction: every
component is defined and the last one (idiosyncratic / residual) is the remainder, so the sum equals the change in
value the marks show.

  equity      market (beta x region market), sector and style (the day's cross-sectional factor returns), idio,
              fx translation (the change in the currency on the local value), execution (fills marked to the close)
  future      settlement variation on the positions held, execution (roll crossing and fees; the calendar spread
              itself is not a cash flow and is reported as the roll cost, not as P&L)
  fx forward  spot (the spot move on the base notional), forward points (the rest of the MTM change), execution
              (the dealer's price against parity on new trades), settled (the final fixing on maturing trades)
  bond        carry (accrual and coupons), rates (-DV01 x the matched par-yield change), credit (-DV01 x the
              bucket's spread change), idio (the rest), execution (RFQ fills marked to the close)
  financing   cash interest, stock borrow
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import book as B, valuation as V
from .riskmodel import RiskModel


def _add(rows: dict, strat: str, cls: str, comp: str, v: float):
    if not np.isfinite(v):
        raise ValueError(f"non-finite P&L for {strat} {cls} {comp}")
    if abs(v) > 1e-12:
        rows[(strat, cls, comp)] = rows.get((strat, cls, comp), 0.0) + float(v)


def attribute(d: dt.date, positions0: dict, fills: list[dict], insts: dict, marks0: dict, marks1: dict, fx0: dict, fx1: dict, rates0: dict, rates1: dict, rm: RiskModel, betas: pd.DataFrame,
              factor_returns: pd.Series | None = None, market_ret: dict | None = None, spread_chg: dict | None = None, financing: dict | None = None, settled_forwards: list | None = None, dv01_fut: dict | None = None, t_frac: float = 1.0, d0: dt.date | None = None) -> dict:
    """Returns {(strategy, asset_class, component): usd}.  marks0/marks1 are local prices at the two ends; fills are
    {'strategy','instrument_id','qty','px','kind'} executed between them; factor_returns / market_ret are the
    equity factor returns over the interval (the caller runs the cross-section), spread_chg the credit factor."""
    rows: dict = {}; dv01_fut = dv01_fut or {}; d0 = d0 or d - dt.timedelta(days=1)      # forwards at the start are valued as of the previous close
    # equity cross-section over the interval: local log returns of the names held
    eq_syms = sorted({iid for (_, iid) in positions0 if iid in insts and insts[iid].asset_class == B.EQUITY and iid in marks0 and iid in marks1 and marks0[iid] > 0})
    r = pd.Series({s: np.log(marks1[s] / marks0[s]) for s in eq_syms})
    if factor_returns is None and len(r) >= 20:
        mk = {reg: float(r[[s for s in r.index if insts[s].region == reg]].mean()) if any(insts[s].region == reg for s in r.index) else 0.0 for reg in ("US", "UK", "EU")}
        f, idio = rm.cross_section(d, r, betas=betas, market_ret=mk)
    else:
        f = factor_returns if factor_returns is not None else pd.Series(dtype=float); mk = market_ret or {}
    sty_day = rm.styles(rm.adj.loc[:d - dt.timedelta(days=1)].index[-1]) if len(rm.adj.loc[:d - dt.timedelta(days=1)]) else None
    for (strat, iid), q in positions0.items():
        inst = insts.get(iid)
        if inst is None or abs(q) < 1e-12:
            continue
        if inst.asset_class == B.EQUITY:
            if iid not in marks0 or iid not in marks1:
                continue
            p0, p1 = marks0[iid], marks1[iid]; u0, u1 = V.usd_per(inst.currency, fx0), V.usd_per(inst.currency, fx1)
            mv0 = q * p0 * inst.price_scale * u0; total_local = q * (p1 - p0) * inst.price_scale * u0
            fx_tr = q * p1 * inst.price_scale * (u1 - u0)
            b = float(betas["beta_mkt"].get(iid, 1.0)) if betas is not None else 1.0
            market = mv0 * b * mk.get(inst.region, 0.0)
            sec = mv0 * float(f.get(f"SEC_{inst.sector}", 0.0))
            sty = 0.0
            if sty_day is not None and iid in sty_day.index:
                sty = mv0 * sum(float(sty_day.at[iid, s]) * float(f.get(s, 0.0)) for s in ("SIZE", "MOM", "VOL"))
            _add(rows, strat, "equity", "market", market); _add(rows, strat, "equity", "sector", sec); _add(rows, strat, "equity", "style", sty)
            _add(rows, strat, "equity", "idio", total_local - market - sec - sty); _add(rows, strat, "equity", "fx_translation", fx_tr)
        elif inst.asset_class == B.FUTURE:
            if iid in marks0 and iid in marks1:
                _add(rows, strat, "future", "settlement_variation", q * (marks1[iid] - marks0[iid]) * inst.multiplier)
        elif inst.asset_class == B.FX_FORWARD:
            v0 = V.forward_value_usd(inst, q, d0, fx0, rates0); v1 = V.forward_value_usd(inst, q, d, fx1, rates1)
            base, quote = B.CCY_OF_PAIR[inst.pair]
            spot = q * (fx1[inst.pair] - fx0[inst.pair]) * V.usd_per(quote, fx1)
            _add(rows, strat, "fx_forward", "spot", spot); _add(rows, strat, "fx_forward", "forward_points", v1 - v0 - spot)
        elif inst.asset_class == B.BOND:
            if iid not in marks0 or iid not in marks1:
                continue
            p0, p1 = marks0[iid], marks1[iid]; a0 = B.bond_accrued(inst, d0); a1 = B.bond_accrued(inst, d)
            coupon = inst.coupon / 2.0 if a1 < a0 - 1e-9 else 0.0
            carry = q * (a1 - a0 + coupon) / 100.0 * t_frac; dv = V.bond_dv01(inst, q, p0); ten = rm.bond_tenor(inst, d)
            rates_pnl = -dv * (rates1.get(ten, 0.0) - rates0.get(ten, 0.0)) * 100.0
            credit = -dv * float((spread_chg or {}).get(inst.rating_bucket, 0.0))
            total = q * (p1 - p0) / 100.0 + carry
            _add(rows, strat, "bond", "carry", carry); _add(rows, strat, "bond", "rates", rates_pnl); _add(rows, strat, "bond", "credit", credit); _add(rows, strat, "bond", "idio", total - carry - rates_pnl - credit)
    # fills: marked to the end of the interval
    for fl in fills:
        inst = insts.get(fl["instrument_id"])
        if inst is None:
            continue
        if inst.asset_class == B.EQUITY and fl["instrument_id"] in marks1:
            _add(rows, fl["strategy"], "equity", "execution", fl["qty"] * (marks1[fl["instrument_id"]] - fl["px"]) * inst.price_scale * V.usd_per(inst.currency, fx1))
        elif inst.asset_class == B.FUTURE:
            if fl["instrument_id"] in marks1:
                _add(rows, fl["strategy"], "future", "settlement_variation", fl["qty"] * (marks1[fl["instrument_id"]] - fl["px"]) * inst.multiplier)
            _add(rows, fl["strategy"], "future", "execution", fl.get("cost", 0.0))
        elif inst.asset_class == B.FX_FORWARD:
            q = fl["qty"]; v1 = V.forward_value_usd(inst, q, d, fx1, rates1)
            _add(rows, fl["strategy"], "fx_forward", "execution", v1)          # opened at the dealer's strike: its MTM against parity is the execution cost
        elif inst.asset_class == B.BOND and fl["instrument_id"] in marks1:
            _add(rows, fl["strategy"], "bond", "execution", fl["qty"] * (marks1[fl["instrument_id"]] - fl["px"]) / 100.0)
    for s in settled_forwards or []:
        _add(rows, s["strategy"], "fx_forward", "settled", s["pnl"])
    if financing:
        _add(rows, "FUND", "financing", "interest", financing.get("interest", 0.0)); _add(rows, "FUND", "financing", "borrow", financing.get("borrow", 0.0))
    return rows


def total(rows: dict) -> float:
    return float(sum(rows.values()))


def by(rows: dict, key: int) -> dict:
    out = {}
    for k, v in rows.items():
        out[k[key]] = out.get(k[key], 0.0) + v
    return out
