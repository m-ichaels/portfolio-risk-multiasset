"""Futures operations: which contract the book should hold (execution-ops' first-notice and last-trade rules, shifted
by the roll lead), settlement prices from the committed data (the listed contract, or Yahoo's continuous series
while the contract is the front month), the roll itself (close the old month, open the new one at the day's
settlements, pay the calendar spread plus a tick a leg and the fees) and its measured cost per contract, and the
daily settlement variation."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import book as B
from .xops_bridge import cal, fi, futures_dates

ACTIVE_MONTHS = {"quarterly": (3, 6, 9, 12), "monthly": tuple(range(1, 13)), "gold": (2, 4, 6, 8, 12)}   # gold: the active COMEX months (October is listed but thin)


class Settlements:
    def __init__(self, futures: pd.DataFrame, specs: pd.DataFrame):
        self.specs = specs
        self.px = futures.pivot_table(index="date", columns="contract", values="close").sort_index()
        self.listed = {c for c in self.px.columns if not c.endswith("=F")}

    def front(self, root: str, d: dt.date) -> str:
        """the exchange's front month on d: nearest active month still trading and before first notice"""
        months = ACTIVE_MONTHS[self.specs.loc[root, "months"]]; y, m = d.year, d.month
        for _ in range(40):
            if m in months:
                fd = futures_dates(root, y, m)
                if fd["last_trade"] >= d and (fd.get("first_notice") is None or fd["first_notice"] > d):
                    return fi.contract_code(root, y, m)
            m += 1
            if m == 13:
                m = 1; y += 1
        raise RuntimeError(root)

    def should_hold(self, root: str, d: dt.date, roll_days_before: int) -> str:
        """the contract a book that rolls `roll_days_before` CME days ahead of first notice / last trade holds on d"""
        return self.front(root, cal.next_trading_day("CME", d, roll_days_before))

    def settle(self, contract: str, d: dt.date) -> tuple[float, str]:
        """(price, source) on d: 'listed' when the contract's own history has the day, 'continuous' when the
        contract is the front month and only Yahoo's =F series has it, else (nan, 'missing')"""
        if contract in self.listed and d in self.px.index and pd.notna(self.px.at[d, contract]):
            return float(self.px.at[d, contract]), "listed"
        root = fi.parse_contract(contract)[0]
        if self.front(root, d) == contract and root + "=F" in self.px.columns and d in self.px.index and pd.notna(self.px.at[d, root + "=F"]):
            return float(self.px.at[d, root + "=F"]), "continuous"
        return float("nan"), "missing"

    def continuous(self, root: str, d: dt.date) -> float:
        """Yahoo's =F close on or before d (the stand-in for a month the data does not list)"""
        col = root + "=F"
        if col not in self.px.columns:
            return float("nan")
        h = self.px[col].loc[:d].dropna()
        return float(h.iloc[-1]) if len(h) else float("nan")

    def last_settle(self, contract: str, d: dt.date) -> tuple[float, dt.date | None]:
        """the latest settlement on or before d (weekends and holidays fall back to the previous session)"""
        for k in range(0, 7):
            dd = d - dt.timedelta(days=k); p, src = self.settle(contract, dd)
            if np.isfinite(p):
                return p, dd
        return float("nan"), None

    def continuous_vs_front(self, start: dt.date, end: dt.date) -> dict:
        """how often Yahoo's =F equals the listed front month where both exist (the check behind `settle`)"""
        out = {}
        for root in self.specs.index:
            eq = tot = 0; diffs = []
            for d in self.px.loc[start:end].index:
                c = self.front(root, d)
                if c in self.listed and pd.notna(self.px.at[d, c]) and pd.notna(self.px.at[d, root + "=F"]):
                    tot += 1; diff = float(abs(self.px.at[d, c] - self.px.at[d, root + "=F"])); diffs.append(diff); eq += int(diff < 1e-9)
            out[root] = {"days_both": tot, "equal": eq, "max_abs_diff": float(max(diffs)) if diffs else None}
        return out


def roll_needed(positions: dict, insts: dict, sett: Settlements, d: dt.date, roll_days_before: int) -> list[tuple[str, str, str, float]]:
    """(strategy, held contract, target contract, lots) for every futures position not in the contract it should hold"""
    out = []
    for (strat, iid), q in list(positions.items()):
        inst = insts.get(iid)
        if inst is None or inst.asset_class != B.FUTURE or abs(q) < 1e-9:
            continue
        tgt = sett.should_hold(inst.root, d, roll_days_before)
        if tgt != iid:
            out.append((strat, iid, tgt, q))
    return out


def execute_roll(strat: str, old: str, new: str, lots: float, d: dt.date, sett: Settlements, specs: pd.DataFrame, insts: dict) -> dict:
    """close `old`, open `new` at the day's settlements; the cost is the calendar spread paid on the way in, a tick
    a leg for crossing, and the exchange and broker fees.  Returns the roll row; the caller books the positions."""
    root = fi.parse_contract(old)[0]; s = specs.loc[root]; mult = float(s["multiplier"]); tick = float(s["tick_size"]); fee = float(s["fee_per_side_usd"])
    p_old, src_old = sett.settle(old, d); p_new, src_new = sett.settle(new, d)
    observed = np.isfinite(p_old) and np.isfinite(p_new)
    if not observed:
        p_old = p_old if np.isfinite(p_old) else sett.last_settle(old, d)[0]
        p_new = p_new if np.isfinite(p_new) else p_old
    sign = 1.0 if lots > 0 else -1.0
    spread_ticks = (p_new - p_old) / tick
    spread_cost = -sign * (p_new - p_old) * mult * abs(lots)          # a long rolling into a dearer month pays the spread
    crossing = -2 * tick * mult * abs(lots); fees = -2 * fee * abs(lots)
    if new not in insts:
        insts[new] = B.futures_instrument(new, specs)
    return {"date": d, "strategy": strat, "root": root, "from_contract": old, "to_contract": new, "lots": lots, "front_px": p_old, "next_px": p_new, "spread_ticks": float(spread_ticks), "spread_cost_usd": spread_cost, "crossing_usd": crossing, "fees_usd": fees,
            "cost_usd": spread_cost + crossing + fees, "cost_per_contract_usd": (spread_cost + crossing + fees) / abs(lots), "observed": bool(observed), "sources": f"{src_old}/{src_new}"}


def settlement_variation(qty: float, inst: B.Instrument, p_prev: float, p_now: float) -> float:
    return qty * (p_now - p_prev) * inst.multiplier


def roll_calendar(sett: Settlements, roots: list[str], start: dt.date, end: dt.date, roll_days_before: int) -> list[dict]:
    """every roll a book following the rule makes between start and end"""
    out = []
    for root in roots:
        cur = None
        for d in cal.trading_days("CME", start, end):
            c = sett.should_hold(root, d, roll_days_before)
            if cur is not None and c != cur:
                rt, y, m = fi.parse_contract(cur); fd = futures_dates(rt, y, m)
                out.append({"root": root, "date": d, "from_contract": cur, "to_contract": c, "first_notice": fd.get("first_notice"), "last_trade": fd.get("last_trade")})
            cur = c
    return out
