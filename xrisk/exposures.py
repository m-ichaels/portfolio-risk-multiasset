"""Positions to exposures: gross and net by asset class, strategy, sector and currency; beta-dollars; DV01 and CS01;
the factor-exposure vector for the risk model; parametric and historical one-day 99 % VaR; liquidity; the largest
positions.  One function, called at start of day, every monitor step and at the close, on whatever marks the feed
shows (so a bad feed produces bad exposures, which is what the monitor has to catch)."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import book as B, valuation as V
from .riskmodel import RiskModel, RATE_TENORS

Z99 = 2.3263


@dataclass
class Exposures:
    date: dt.date; time: float
    nav: float = 0.0
    by_class: dict = field(default_factory=dict)        # asset class -> {"gross", "net"}
    by_strategy: dict = field(default_factory=dict)     # strategy -> {"gross", "net"}
    by_sector: dict = field(default_factory=dict)       # sector -> net (equities and bonds)
    by_currency: dict = field(default_factory=dict)     # ccy -> net USD exposure to the currency (non-USD only)
    beta_dollars: float = 0.0
    beta_by_strategy: dict = field(default_factory=dict)
    dv01: float = 0.0; dv01_by_strategy: dict = field(default_factory=dict); dv01_by_tenor: dict = field(default_factory=dict)
    cs01: dict = field(default_factory=dict)            # IG / HY
    factor_exposure: pd.Series | None = None
    idio_var: float = 0.0
    var_param: float = 0.0; var_hist: float = 0.0; sigma: float = 0.0
    positions: pd.DataFrame | None = None               # one row per (strategy, instrument): qty, px, mv_usd, notional_usd, ...
    top_names: list = field(default_factory=list)
    days_to_liquidate: float = 0.0; illiquid_share: float = 0.0
    futures_by_root: dict = field(default_factory=dict)

    @property
    def gross(self) -> float:
        return sum(v["gross"] for v in self.by_class.values())

    @property
    def net(self) -> float:
        return sum(v["net"] for v in self.by_class.values())

    def rows(self) -> list[tuple]:
        r = [(self.date, self.time, "nav", "nav", self.nav), (self.date, self.time, "gross", "all", self.gross), (self.date, self.time, "net", "all", self.net),
             (self.date, self.time, "beta_dollars", "all", self.beta_dollars), (self.date, self.time, "dv01", "all", self.dv01), (self.date, self.time, "var", "param99", self.var_param), (self.date, self.time, "var", "hist99", self.var_hist),
             (self.date, self.time, "liquidity", "days_to_liquidate", self.days_to_liquidate)]
        r += [(self.date, self.time, "class_" + k2, k, v[k2]) for k, v in self.by_class.items() for k2 in ("gross", "net")]
        r += [(self.date, self.time, "strategy_" + k2, k, v[k2]) for k, v in self.by_strategy.items() for k2 in ("gross", "net")]
        r += [(self.date, self.time, "sector_net", k, v) for k, v in self.by_sector.items()]
        r += [(self.date, self.time, "currency_net", k, v) for k, v in self.by_currency.items()]
        r += [(self.date, self.time, "cs01", k, v) for k, v in self.cs01.items()]
        r += [(self.date, self.time, "dv01_tenor", k, v) for k, v in self.dv01_by_tenor.items()]
        r += [(self.date, self.time, "futures_notional", k, v) for k, v in self.futures_by_root.items()]
        return r


def compute(d: dt.date, t: float, positions: dict, insts: dict, marks: dict, fx: dict, rates: dict, rm: RiskModel, nav: float, betas: pd.DataFrame, adv: pd.Series | None = None,
            with_var: bool = True, fut_beta: dict | None = None, factor_hist: pd.DataFrame | None = None, idio_vol: pd.Series | None = None, cov: pd.DataFrame | None = None) -> Exposures:
    ex = Exposures(d, t, nav=nav); fut_beta = fut_beta or {}
    rows = []
    for (strat, iid), qty in positions.items():
        if abs(qty) < 1e-9 or iid not in insts:
            continue
        inst = insts[iid]; px = marks.get(iid)
        if px is None or not np.isfinite(px):
            continue
        mv, notional = V.mark(inst, qty, px, d, fx, rates)
        rows.append({"strategy": strat, "instrument_id": iid, "asset_class": inst.asset_class, "qty": qty, "px": px, "mv_usd": mv, "notional_usd": notional, "sector": inst.sector, "currency": inst.currency, "root": inst.root, "pair": inst.pair, "symbol": inst.symbol})
    if not rows:
        ex.positions = pd.DataFrame(columns=["strategy", "instrument_id", "asset_class", "qty", "px", "mv_usd", "notional_usd", "sector", "currency", "root", "pair", "symbol"]); return ex
    p = pd.DataFrame(rows); ex.positions = p
    # forwards to the same pair offset each other (a hedge adjusted by outright trades is one position per pair), so
    # their gross is the net per strategy and pair, not the sum of the trades
    fwd_net = p[p["asset_class"] == B.FX_FORWARD].groupby(["strategy", "pair"])["notional_usd"].sum()
    other = p[p["asset_class"] != B.FX_FORWARD]
    for k, g in other.groupby("asset_class"):
        ex.by_class[k] = {"gross": float(g["notional_usd"].abs().sum()), "net": float(g["notional_usd"].sum())}
    if len(fwd_net):
        ex.by_class[B.FX_FORWARD] = {"gross": float(fwd_net.abs().sum()), "net": float(fwd_net.sum())}
    for k, g in other.groupby("strategy"):
        ex.by_strategy[k] = {"gross": float(g["notional_usd"].abs().sum()), "net": float(g["notional_usd"].sum())}
    for (k, pair), v in fwd_net.items():
        e = ex.by_strategy.setdefault(k, {"gross": 0.0, "net": 0.0}); e["gross"] += abs(float(v)); e["net"] += float(v)
    eq = p[p["asset_class"] == B.EQUITY]; bd = p[p["asset_class"] == B.BOND]; fu = p[p["asset_class"] == B.FUTURE]; fw = p[p["asset_class"] == B.FX_FORWARD]
    for k, g in pd.concat([eq, bd]).groupby("sector"):
        ex.by_sector[k] = float(g["notional_usd"].sum())
    # currency exposure: physical non-USD assets plus FX futures and forwards (base-currency notional), in USD
    cur = {}
    for r in eq.itertuples():
        if r.currency != "USD":
            cur[r.currency] = cur.get(r.currency, 0.0) + r.notional_usd
    for r in fw.itertuples():
        base, quote = B.CCY_OF_PAIR[r.pair]
        if base != "USD":
            cur[base] = cur.get(base, 0.0) + r.notional_usd
        else:
            cur[quote] = cur.get(quote, 0.0) - r.notional_usd
    for r in fu.itertuples():
        if r.root in ("6E", "6B", "6J"):
            c = {"6E": "EUR", "6B": "GBP", "6J": "JPY"}[r.root]; cur[c] = cur.get(c, 0.0) + r.notional_usd
    ex.by_currency = cur
    # beta-dollars: equities to SPY, index futures with their measured beta
    bspy = betas["beta_spy"] if betas is not None else pd.Series(dtype=float)
    eqb = eq.assign(beta=eq["instrument_id"].map(bspy).fillna(1.0)); eqb["bd"] = eqb["notional_usd"] * eqb["beta"]
    fub = fu[fu["root"].isin(("ES", "NQ", "RTY"))].copy(); fub["beta"] = [fut_beta.get(r, 1.0) for r in fub["root"]]; fub["bd"] = fub["notional_usd"].astype(float) * fub["beta"].astype(float)
    ex.beta_dollars = float(eqb["bd"].sum() + fub["bd"].sum())
    for k, g in pd.concat([eqb, fub]).groupby("strategy"):
        ex.beta_by_strategy[k] = float(g["bd"].sum())
    ex.futures_by_root = {k: float(g["notional_usd"].sum()) for k, g in fu.groupby("root")}
    # DV01 and CS01
    dv = {}; dvt = {}; cs = {"IG": 0.0, "HY": 0.0}
    for r in fu.itertuples():
        inst = insts[r.instrument_id]
        if inst.rate_tenor:
            v = r.qty * rm.treasury_dv01(inst.root, d); dv[r.strategy] = dv.get(r.strategy, 0.0) + v; dvt[inst.rate_tenor] = dvt.get(inst.rate_tenor, 0.0) + v
    for r in bd.itertuples():
        inst = insts[r.instrument_id]; v = V.bond_dv01(inst, r.qty, r.px); ten = rm.bond_tenor(inst, d)
        dv[r.strategy] = dv.get(r.strategy, 0.0) + v; dvt[ten] = dvt.get(ten, 0.0) + v; cs[inst.rating_bucket] = cs.get(inst.rating_bucket, 0.0) + v
    ex.dv01_by_strategy = dv; ex.dv01_by_tenor = dvt; ex.dv01 = float(sum(dv.values())); ex.cs01 = cs
    # liquidity (equities): days to liquidate at 20 % of ADV
    if adv is not None and len(eq):
        dtl = (eq.groupby("instrument_id")["qty"].sum().abs() / (0.2 * adv.reindex(eq["instrument_id"].unique()).fillna(1.0))).replace([np.inf], np.nan).fillna(99.0)
        ex.days_to_liquidate = float(dtl.max()); mv_by = eq.groupby("instrument_id")["notional_usd"].sum().abs()
        ex.illiquid_share = float(mv_by[dtl > 5].sum() / max(mv_by.sum(), 1.0))
    top = eq.groupby("instrument_id")["notional_usd"].sum(); top = top.reindex(top.abs().sort_values(ascending=False).index)
    ex.top_names = [(k, float(v)) for k, v in top.head(10).items()]
    if with_var:
        _var(ex, d, insts, rm, betas, fut_beta, factor_hist, idio_vol, cov)
    return ex


def factor_vector(ex: Exposures, d: dt.date, insts: dict, rm: RiskModel, betas: pd.DataFrame, fut_beta: dict, factors: list[str]) -> pd.Series:
    """the book's exposure to every factor: dollars for return factors, dollars per bp (sign: long loses when
    yields or spreads rise) for the rate and credit factors"""
    x = pd.Series(0.0, index=factors); p = ex.positions
    if p is None or p.empty:
        return x
    sty = rm.styles(rm.adj.loc[:d - dt.timedelta(days=1)].index[-1]) if len(rm.adj.loc[:d - dt.timedelta(days=1)]) else None
    for r in p.itertuples():
        inst = insts[r.instrument_id]; n = r.notional_usd
        if inst.asset_class == B.EQUITY:
            b = float(betas["beta_mkt"].get(inst.symbol, 1.0)) if betas is not None else 1.0
            x[f"MKT_{inst.region}"] += n * b; sec = f"SEC_{inst.sector}"
            if sec in x.index:
                x[sec] += n
            if sty is not None and inst.symbol in sty.index:
                for s in ("SIZE", "MOM", "VOL"):
                    x[s] += n * float(sty.at[inst.symbol, s])
            if inst.currency == "EUR":
                x["FX_EURUSD"] += n
            elif inst.currency == "GBP":
                x["FX_GBPUSD"] += n
        elif inst.asset_class == B.FUTURE:
            if inst.rate_tenor:
                x[inst.rate_tenor] += -r.qty * rm.treasury_dv01(inst.root, d)
            elif inst.root == "6J":
                x["FX_USDJPY"] += -n
            elif inst.root == "6E":
                x["FX_EURUSD"] += n
            elif inst.root == "6B":
                x["FX_GBPUSD"] += n
            else:
                x[f"FUT_{inst.root}"] += n
        elif inst.asset_class == B.FX_FORWARD:
            x[f"FX_{inst.pair}"] += n            # long the base currency (USD for USDJPY) gains when the pair rises
        elif inst.asset_class == B.BOND:
            dv = V.bond_dv01(inst, r.qty, r.px); x[rm.bond_tenor(inst, d)] += -dv; x[f"CS_{inst.rating_bucket}"] += -dv
    return x


def _var(ex: Exposures, d: dt.date, insts: dict, rm: RiskModel, betas, fut_beta, factor_hist, idio_vol, cov):
    fh = factor_hist if factor_hist is not None else rm.factor_history(d)
    cv = cov if cov is not None else fh.cov()
    x = factor_vector(ex, d, insts, rm, betas, fut_beta, list(fh.columns)); ex.factor_exposure = x
    iv = idio_vol if idio_vol is not None else rm.idio_vol(d)
    eq = ex.positions[ex.positions["asset_class"] == B.EQUITY].groupby("instrument_id")["notional_usd"].sum()
    ex.idio_var = float(((eq * iv.reindex(eq.index).fillna(iv.median() if len(iv) else 0.02)) ** 2).sum())
    var_f = float(x.values @ cv.values @ x.values)
    ex.sigma = float(np.sqrt(max(var_f + ex.idio_var, 0.0))); ex.var_param = Z99 * ex.sigma
    # historical: actual instrument returns where they exist, factors for the rest
    p = ex.positions; pnl = pd.Series(0.0, index=fh.index)
    if len(eq):
        r = rm.ret.reindex(index=fh.index, columns=eq.index).fillna(0.0)
        pnl += r.values @ eq.values
        fxn = {}
        for iid, n in eq.items():
            c = insts[iid].currency
            if c != "USD":
                fxn[c] = fxn.get(c, 0.0) + n
        for c, n in fxn.items():
            pnl += n * fh[f"FX_{'EURUSD' if c == 'EUR' else 'GBPUSD'}"].values
    for r in p.itertuples():
        inst = insts[r.instrument_id]
        if inst.asset_class == B.FUTURE:
            pnl += r.notional_usd * fh[f"FUT_{inst.root}"].values
        elif inst.asset_class == B.FX_FORWARD:
            pnl += r.notional_usd * fh[f"FX_{inst.pair}"].values
        elif inst.asset_class == B.BOND:
            dv = V.bond_dv01(inst, r.qty, r.px); pnl += -dv * (fh[rm.bond_tenor(inst, d)].values + fh[f"CS_{inst.rating_bucket}"].values)
    ex.var_hist = float(-np.quantile(pnl.values, 0.01)) if len(pnl) > 20 else ex.var_param
