"""Margin on the multi-asset book.

Futures: a SPAN-style calculation.  For each product the price scan range is calibrated on the data (the 99th
percentile of the two-day dollar move per contract over the last 250 days, rounded up to a tick multiple); the
sixteen SPAN scenarios (price at 0, +-1/3, +-2/3, +-1 of the range with volatility up and down, and two extreme
moves at +-2 ranges covered 35 %) are evaluated per combined commodity with the delta-one payoff, the scan risk is
the worst scenario loss over the net position across delivery months; an intra-commodity (calendar) spread charge
is added per spread lot, calibrated the same way on the front-next spread; a delivery charge doubles the scan range
for contracts inside the first-notice window; an inter-commodity credit is subtracted for the equity-index and the
Treasury groups on the smaller leg's scan risk.  The credit rates are this model's parameters, not CME's.

Equities: prime-broker style, 15 % of long market value and 20 % of short, plus 10 % on names above 4 % of the
equity book.  Bonds: repo haircuts, 5 % IG and 15 % HY.  FX forwards: variation margin on the mark-to-market netted per counterparty (one CSA each); no initial
margin, physically settled forwards being exempt under the uncleared-margin rules.

Collateral is the fund's cash (NAV less the market value of physical positions and forward MTM); a margin call is a
requirement above the collateral.  Financing: the debit balance costs SOFR + 50 bp, free cash earns SOFR - 25 bp,
short stock borrow costs 40 bp on the short market value."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import book as B
from .xops_bridge import fi, futures_dates

SPAN_SCENARIOS = [(0.0, +1), (0.0, -1), (1 / 3, +1), (1 / 3, -1), (-1 / 3, +1), (-1 / 3, -1), (2 / 3, +1), (2 / 3, -1), (-2 / 3, +1), (-2 / 3, -1), (1.0, +1), (1.0, -1), (-1.0, +1), (-1.0, -1), (2.0, 0), (-2.0, 0)]
EXTREME_COVER = 0.35
INTER_CREDIT = {"equity_index": 0.50, "treasury": 0.60, "fx": 0.30}
DELIVERY_MULT = 2.0


@dataclass
class MarginParams:
    equity_long: float = 0.15; equity_short: float = 0.20; concentration_addon: float = 0.10; concentration_at: float = 0.04
    bond_haircut: dict = field(default_factory=lambda: {"IG": 0.05, "HY": 0.15})
    debit_spread_bp: float = 50.0; credit_spread_bp: float = -25.0; borrow_bp: float = 40.0
    maintenance_ratio: float = 0.9


def scan_ranges(futures: pd.DataFrame, specs: pd.DataFrame, d: dt.date, window: int = 250) -> dict[str, dict]:
    """per root: price scan range (dollars per contract) and the calendar-spread charge, from the data before d"""
    out = {}
    cont = futures[futures["contract"].str.endswith("=F")]
    for root in specs.index:
        c = cont[(cont["root"] == root) & (cont["date"] < d)].sort_values("date").tail(window + 2)
        mult = float(specs.loc[root, "multiplier"]); tick = float(specs.loc[root, "tick_size"])
        if len(c) < 40:
            out[root] = {"scan": 0.0, "spread": 0.0}; continue
        mv2 = (c["close"].diff(2).abs() * mult).dropna()
        scan = float(np.quantile(mv2, 0.99)); scan = np.ceil(scan / (tick * mult)) * tick * mult
        # calendar spread charge: front-next spread two-day moves, where two listed months overlap before d
        listed = futures[(futures["root"] == root) & (~futures["contract"].str.endswith("=F")) & (futures["date"] < d)]
        spread_chg = []
        if len(listed):
            piv = listed.pivot(index="date", columns="contract", values="close").sort_index()
            cols = sorted(piv.columns, key=lambda c: fi.parse_contract(c)[1:] )
            for a, b in zip(cols[:-1], cols[1:]):
                s = (piv[b] - piv[a]).dropna().tail(window)
                if len(s) > 40:
                    spread_chg += list((s.diff(2).abs() * mult).dropna())
        sp = float(np.quantile(spread_chg, 0.99)) if len(spread_chg) > 40 else 0.1 * scan
        out[root] = {"scan": scan, "spread": float(np.ceil(sp / (tick * mult)) * tick * mult), "n_days": int(len(mv2))}
    return out


def span_scenarios(lots_by_contract: dict[str, float], scan: float) -> list[float]:
    """P&L of the net delta-one position under the sixteen scenarios (losses negative)"""
    net = sum(lots_by_contract.values()); out = []
    for frac, _vol in SPAN_SCENARIOS:
        move = frac * scan
        pnl = net * move
        if abs(frac) > 1.0:
            pnl *= EXTREME_COVER
        out.append(pnl)
    return out


def futures_margin(positions: dict, insts: dict, ranges: dict, d: dt.date, specs: pd.DataFrame, roll_days_before: int = 2) -> dict:
    """SPAN-style requirement over every futures position in the book (all strategies net at the clearing account)"""
    by_root: dict[str, dict[str, float]] = {}
    for (strat, iid), q in positions.items():
        inst = insts.get(iid)
        if inst is None or inst.asset_class != B.FUTURE or abs(q) < 1e-9:
            continue
        by_root.setdefault(inst.root, {}); by_root[inst.root][iid] = by_root[inst.root].get(iid, 0.0) + q
    detail = {}; total_scan = 0.0; total_spread = 0.0; total_delivery = 0.0
    for root, lots in by_root.items():
        r = ranges.get(root, {"scan": 0.0, "spread": 0.0}); scen = span_scenarios(lots, r["scan"]); scan_risk = max(0.0, -min(scen))
        longs = sum(q for q in lots.values() if q > 0); shorts = -sum(q for q in lots.values() if q < 0)
        spread_lots = min(longs, shorts); spread_charge = spread_lots * r["spread"]
        # delivery charge: months inside the first-notice / last-trade window
        deliv = 0.0
        for c, q in lots.items():
            rt, y, m = fi.parse_contract(c); fd = futures_dates(rt, y, m); fn = fd.get("first_notice") or fd.get("last_trade")
            if fn is not None and 0 <= (fn - d).days <= roll_days_before + 3:
                deliv += abs(q) * r["scan"] * (DELIVERY_MULT - 1.0)
        detail[root] = {"net_lots": sum(lots.values()), "gross_lots": longs + shorts, "scan_range": r["scan"], "scan_risk": scan_risk, "spread_lots": spread_lots, "spread_charge": spread_charge, "delivery_charge": deliv, "scenarios": scen, "group": specs.loc[root, "group"]}
        total_scan += scan_risk; total_spread += spread_charge; total_delivery += deliv
    # inter-commodity credits within a group: pair the largest scan risks, credit on the smaller leg
    credit = 0.0
    for grp, rate in INTER_CREDIT.items():
        legs = sorted([(v["scan_risk"], k) for k, v in detail.items() if v["group"] == grp and v["scan_risk"] > 0], reverse=True)
        # a spread needs opposite signs
        pos = [(s, k) for s, k in legs if detail[k]["net_lots"] > 0]; neg = [(s, k) for s, k in legs if detail[k]["net_lots"] < 0]
        while pos and neg:
            (sa, ka), (sb, kb) = pos[0], neg[0]; c = rate * min(sa, sb); credit += c
            if sa > sb:
                pos[0] = (sa - sb, ka); neg.pop(0)
            else:
                neg[0] = (sb - sa, kb); pos.pop(0)
    initial = total_scan + total_spread + total_delivery - credit
    return {"initial": initial, "maintenance": 0.9 * initial, "scan": total_scan, "spread_charge": total_spread, "delivery_charge": total_delivery, "inter_commodity_credit": credit, "sum_of_outrights": sum(v["gross_lots"] * v["scan_range"] for v in detail.values()), "by_root": detail}


def requirement(ex_positions: pd.DataFrame, insts: dict, fut: dict, params: MarginParams) -> dict:
    """the whole requirement from an exposures table (mv_usd per position) plus the futures result"""
    eq = ex_positions[ex_positions["asset_class"] == B.EQUITY]; bd = ex_positions[ex_positions["asset_class"] == B.BOND]; fw = ex_positions[ex_positions["asset_class"] == B.FX_FORWARD]
    by_name = eq.groupby("instrument_id")["mv_usd"].sum()
    long_mv = float(by_name[by_name > 0].sum()); short_mv = float(-by_name[by_name < 0].sum()); gross = long_mv + short_mv
    conc = float(by_name.abs()[by_name.abs() > params.concentration_at * max(gross, 1.0)].sum()) if gross > 0 else 0.0
    eq_req = params.equity_long * long_mv + params.equity_short * short_mv + params.concentration_addon * conc
    bd_req = float(sum(abs(r.mv_usd) * params.bond_haircut[insts[r.instrument_id].rating_bucket] for r in bd.itertuples()))
    cp = fw["instrument_id"].map(lambda x: x.split(":")[3] if x.count(":") >= 3 else x)          # FWD:pair:value date:counterparty:trade date
    fx_vm = float(-fw.assign(cp=cp).groupby("cp")["mv_usd"].sum().clip(upper=0).sum()) if len(fw) else 0.0     # variation margin posted per counterparty (one CSA each), on the netted MTM
    return {"equity": eq_req, "equity_long_mv": long_mv, "equity_short_mv": short_mv, "bonds": bd_req, "futures_initial": fut["initial"], "fx_variation": fx_vm, "total": eq_req + bd_req + fut["initial"] + fx_vm}


def collateral(nav: float, ex_positions: pd.DataFrame) -> dict:
    physical = float(ex_positions.loc[ex_positions["asset_class"].isin((B.EQUITY, B.BOND)), "mv_usd"].sum()); fwd = float(ex_positions.loc[ex_positions["asset_class"] == B.FX_FORWARD, "mv_usd"].sum())
    cash = nav - physical - fwd
    return {"cash": cash, "physical_mv": physical, "forward_mtm": fwd}


def financing_cost(req: dict, coll: dict, sofr_pct: float, params: MarginParams, days: int = 1) -> dict:
    """one day's financing in dollars (negative = cost)"""
    cash = coll["cash"]; debit = max(0.0, -cash); free = max(0.0, cash)
    r_debit = (sofr_pct + params.debit_spread_bp / 100.0) / 100.0; r_free = (sofr_pct + params.credit_spread_bp / 100.0) / 100.0
    interest = (free * r_free - debit * r_debit) * days / 360.0
    borrow = -req["equity_short_mv"] * params.borrow_bp / 1e4 * days / 360.0
    return {"interest": interest, "borrow": borrow, "total": interest + borrow}
