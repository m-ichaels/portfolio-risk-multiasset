"""The intraday market the monitor watches.  Every daily mark is real (equity open and close, futures settlements,
ECB FX fixes, Treasury par yields); what happens between two consecutive daily marks is a Brownian bridge in log
price at five-minute steps with the instrument's own trailing volatility, so that the intraday paths are simulated
and the closes are not.  Equities move inside their exchange session (before the open they sit at the previous
close), futures from the start of the monitoring window to their 16:00 New York settlement, FX and rates across the
whole window.  Fills execute on a U-shaped volume curve at the path plus half a spread."""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import book as B
from .xops_bridge import cal

NY = ZoneInfo("America/New_York")
STEP = 300                       # seconds
WINDOW = ("03:00", "16:30")      # New York time: the London open to after the US close


def ny_time(d: dt.date, hhmm: str) -> float:
    return dt.datetime.combine(d, dt.time.fromisoformat(hhmm), tzinfo=NY).timestamp()


def session(exchange: str, d: dt.date) -> tuple[float, float]:
    o, c, tz = cal.SESSION[exchange]
    if cal.is_early_close(exchange, d):
        c = cal.EARLY_CLOSE_TIME[exchange]
    z = ZoneInfo(tz)
    return dt.datetime.combine(d, dt.time.fromisoformat(o), tzinfo=z).timestamp(), dt.datetime.combine(d, dt.time.fromisoformat(c), tzinfo=z).timestamp()


def bridge(p0: float, p1: float, n: int, sigma_day: float, rng: np.random.Generator) -> np.ndarray:
    """n+1 points from p0 to p1: log-price bridge with daily volatility sigma_day spread over the n steps"""
    if n <= 0 or not (np.isfinite(p0) and np.isfinite(p1)) or p0 <= 0 or p1 <= 0:
        return np.array([p0] + [p1] * max(n, 0))
    w = np.concatenate([[0.0], np.cumsum(rng.normal(0.0, sigma_day / np.sqrt(n), n))])
    k = np.arange(n + 1) / n
    return np.exp(np.log(p0) + k * (np.log(p1) - np.log(p0)) + (w - k * w[-1]))


def u_curve(n: int) -> np.ndarray:
    x = (np.arange(n) + 0.5) / n; w = 1.0 + 3.0 * (x - 0.5) ** 2 * 4; return w / w.sum()


class DayMarket:
    def __init__(self, d: dt.date, insts: dict, eq_today: pd.DataFrame, eq_prev_close: pd.Series, eq_vol: pd.Series, eq_adv: pd.Series, eq_spread_bp: pd.Series,
                 fut_prev: dict, fut_now: dict, fut_vol: dict, fx_prev: dict, fx_now: dict, fx_vol: dict, rates_prev: dict, rates_now: dict, bond_prev: dict, bond_now: dict, rng: np.random.Generator):
        self.d = d; self.rng = rng; self.insts = insts
        self.t0 = ny_time(d, WINDOW[0]); self.t1 = ny_time(d, WINDOW[1]); self.n = int((self.t1 - self.t0) // STEP)
        self.grid = self.t0 + STEP * np.arange(self.n + 1)
        self.t_settle = ny_time(d, "16:00")
        # equities: per symbol session and bridge
        self.eq: dict[str, dict] = {}
        self.eq_adv = eq_adv; self.eq_spread_bp = eq_spread_bp; self.eq_vol = eq_vol
        for r in eq_today.itertuples():
            inst = insts.get(r.symbol)
            if inst is None or not cal.is_trading_day(inst.exchange, d):
                continue
            to, tc = session(inst.exchange, d); k0 = int(np.searchsorted(self.grid, to)); k1 = int(np.searchsorted(self.grid, tc)); n = max(k1 - k0, 1)
            path = bridge(float(r.open), float(r.close), n, float(eq_vol.get(r.symbol, 0.015)), rng)
            self.eq[r.symbol] = {"t_open": to, "t_close": tc, "k0": k0, "k1": k1, "path": path, "prev_close": float(eq_prev_close.get(r.symbol, r.open)), "close": float(r.close), "open": float(r.open), "volume": float(r.volume), "curve": u_curve(n)}
        for s, pc in eq_prev_close.items():        # names without bars today (their exchange is closed): flat at the previous close
            if s not in self.eq and s in insts:
                self.eq[s] = {"t_open": None, "t_close": None, "k0": 0, "k1": 0, "path": np.array([pc]), "prev_close": float(pc), "close": float(pc), "open": float(pc), "volume": 0.0, "curve": np.array([1.0])}
        # futures, FX, rates, bonds: bridges over the window (futures settle at 16:00)
        ks = int(np.searchsorted(self.grid, self.t_settle))
        self.fut = {c: bridge(fut_prev[c], fut_now.get(c, fut_prev[c]), ks, fut_vol.get(c, 0.01), rng) for c in fut_prev}
        self.fut_ks = ks
        self.fxp = {p: bridge(fx_prev[p], fx_now.get(p, fx_prev[p]), self.n, fx_vol.get(p, 0.005), rng) for p in fx_prev}
        self.rates = {k: np.linspace(rates_prev.get(k, v), v, self.n + 1) for k, v in rates_now.items()}
        self.bond = {c: np.linspace(bond_prev.get(c, v), v, self.n + 1) for c, v in bond_now.items()}

    # ---- lookups ----------------------------------------------------------------------------------------------------------
    def step_index(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.grid, t, side="right") - 1, 0, self.n))

    def equity_px(self, sym: str, t: float) -> float | None:
        e = self.eq.get(sym)
        if e is None:
            return None
        if e["t_open"] is None or t < e["t_open"]:
            return e["prev_close"]
        if t >= e["t_close"]:
            return e["close"]
        k = self.step_index(t) - e["k0"]; return float(e["path"][int(np.clip(k, 0, len(e["path"]) - 1))])

    def equity_open(self, sym: str, t: float) -> bool:
        e = self.eq.get(sym); return e is not None and e["t_open"] is not None and e["t_open"] <= t < e["t_close"]

    def futures_px(self, contract: str, t: float) -> float | None:
        p = self.fut.get(contract)
        if p is None:
            return None
        return float(p[int(np.clip(self.step_index(t), 0, len(p) - 1))])

    def fx_at(self, t: float) -> dict:
        k = self.step_index(t); return {p: float(v[k]) for p, v in self.fxp.items()}

    def rates_at(self, t: float) -> dict:
        k = self.step_index(t); return {n: float(v[k]) for n, v in self.rates.items()}

    def bond_px(self, cusip: str, t: float) -> float | None:
        p = self.bond.get(cusip)
        return None if p is None else float(p[self.step_index(t)])

    def marks_at(self, t: float, held: list[str]) -> dict:
        out = {}
        for iid in held:
            inst = self.insts.get(iid)
            if inst is None:
                continue
            if inst.asset_class == B.EQUITY:
                p = self.equity_px(iid, t)
            elif inst.asset_class == B.FUTURE:
                p = self.futures_px(iid, t)
            elif inst.asset_class == B.BOND:
                p = self.bond_px(iid, t)
            else:
                p = 1.0
            if p is not None:
                out[iid] = p
        return out

    # ---- execution --------------------------------------------------------------------------------------------------------
    def equity_slices(self, sym: str, qty: float) -> list[tuple[float, float, float]]:
        """(time, qty, price) child fills for a day order along the U-curve at the path plus half the spread"""
        e = self.eq.get(sym)
        if e is None or e["t_open"] is None or abs(qty) < 1:
            return []
        n = len(e["curve"]); sp = float(self.eq_spread_bp.get(sym, 5.0)) / 1e4 / 2; sign = 1 if qty > 0 else -1
        out = []; done = 0.0
        for k in range(n):
            q = np.floor(abs(qty) * e["curve"][:k + 1].sum()) - done
            if q >= 1:
                t = self.grid[min(e["k0"] + k, self.n)] + STEP / 2; px = float(e["path"][min(k, len(e["path"]) - 1)]) * (1 + sign * sp)
                out.append((t, sign * q, px)); done += q
        rem = abs(qty) - done
        if rem >= 1:
            out.append((e["t_close"] - 1, sign * rem, e["close"] * (1 + sign * sp)))
        return out
