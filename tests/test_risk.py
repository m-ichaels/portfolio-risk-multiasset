"""Exposures, limits, margin, P&L attribution and the risk model on constructed cases."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from xrisk import book as B, exposures as X, limits as L, margin as MG, pnl as PL, valuation as V


@pytest.fixture(scope="module")
def ctx():
    from xrisk import run as R
    return R.build_context()


def test_marks_by_asset_class():
    fx = {"EURUSD": 1.10, "GBPUSD": 1.30, "USDJPY": 150.0}; rates = {"SOFR": 4.0, "ESTR": 2.0, "SONIA": 4.0, "JGB1Y": 0.5}
    eq = B.Instrument("SHEL.L", B.EQUITY, "SHEL.L", "GBP", price_scale=0.01)
    mv, notional = V.mark(eq, 1000, 2500.0, dt.date(2026, 7, 1), fx)
    assert abs(mv - 1000 * 25.0 * 1.30) < 1e-9 and mv == notional
    es = B.Instrument("ESU26", B.FUTURE, "ESU26", "USD", multiplier=50.0, root="ES")
    mv, notional = V.mark(es, -10, 6000.0, dt.date(2026, 7, 1), fx)
    assert mv == 0.0 and abs(notional + 10 * 6000 * 50) < 1e-9
    fwd = B.forward_instrument("EURUSD", dt.date(2026, 8, 6), 1.1015, "DLR1", dt.date(2026, 7, 1))
    f = V.forward_rate("EURUSD", 1.10, rates, 36)
    assert f > 1.10                                                       # dollar rates above euro rates: the euro trades at a forward premium
    mv, notional = V.mark(fwd, -1e6, 1.0, dt.date(2026, 7, 1), fx, rates)
    assert abs(mv - (-1e6) * (f - 1.1015)) < 1e-6 and abs(notional + 1e6 * 1.10) < 1e-6
    bond = B.Instrument("X", B.BOND, "X", "USD", coupon=5.0, maturity=dt.date(2030, 3, 1), duration=4.0, rating_bucket="IG")
    acc = B.bond_accrued(bond, dt.date(2026, 4, 1))
    assert abs(acc - 5.0 * 30 / 360) < 1e-9                              # one month after the 1 March coupon, 30/360
    assert abs(V.bond_dv01(bond, 1e6, 95.0) - 1e6 * 0.95 * 4.0 / 1e4) < 1e-9


def test_limits_utilisation_and_status():
    ex = X.Exposures(dt.date(2026, 7, 1), 0.0, nav=2e9); ex.by_class = {"equity": {"gross": 6e9, "net": 1e8}}; ex.beta_dollars = 6e8; ex.by_sector = {"Energy": 4.5e8}; ex.top_names = [("AAPL", 1.1e8)]
    rows = L.evaluate(L.LimitSet(), ex)
    by = {(r["limit_name"], r["scope"]): r for r in rows}
    assert by[("gross", "all")]["status"] == "warn" and abs(by[("gross", "all")]["utilisation"] - 6e9 / 7e9) < 1e-9
    assert by[("beta_dollars", "all")]["status"] == "warn"
    assert by[("sector_net", "Energy")]["status"] == "breach"
    assert by[("single_name", "AAPL")]["status"] == "breach"


def test_span_scan_and_spread_credit():
    specs = pd.DataFrame({"group": {"ES": "equity_index", "NQ": "equity_index", "ZN": "treasury"}, "multiplier": {"ES": 50, "NQ": 20, "ZN": 1000}, "tick_size": {"ES": 0.25, "NQ": 0.25, "ZN": 0.015625}})
    insts = {c: B.Instrument(c, B.FUTURE, c, "USD", multiplier=float(specs.loc[c[:2] if c[:2] != "ZN" else "ZN", "multiplier"]), root=c[:2] if c[:2] != "ZN" else "ZN") for c in ("ESU26", "ESZ26", "NQU26", "ZNU26")}
    ranges = {"ES": {"scan": 10000.0, "spread": 400.0}, "NQ": {"scan": 20000.0, "spread": 800.0}, "ZN": {"scan": 1000.0, "spread": 50.0}}
    d = dt.date(2026, 7, 15)
    # an outright long is charged the scan range per lot; the sixteen scenarios' worst loss is the full range
    m = MG.futures_margin({("A", "ESU26"): 10}, insts, ranges, d, specs)
    assert abs(m["scan"] - 10 * 10000) < 1e-9 and m["inter_commodity_credit"] == 0 and min(m["by_root"]["ES"]["scenarios"]) == -10 * 10000
    # a calendar spread nets the scan and pays the spread charge
    m = MG.futures_margin({("A", "ESU26"): 10, ("A", "ESZ26"): -10}, insts, ranges, d, specs)
    assert m["scan"] == 0 and abs(m["spread_charge"] - 10 * 400) < 1e-9
    # long ES against short NQ earns the inter-commodity credit on the smaller leg
    m = MG.futures_margin({("A", "ESU26"): 10, ("B", "NQU26"): -3}, insts, ranges, d, specs)
    assert abs(m["inter_commodity_credit"] - 0.5 * min(10 * 10000, 3 * 20000)) < 1e-9
    assert m["initial"] < m["sum_of_outrights"]
    # the delivery charge inside the notice window
    m = MG.futures_margin({("A", "ZNU26"): 5}, insts, ranges, dt.date(2026, 8, 28), specs)
    assert m["delivery_charge"] > 0


def test_requirement_and_collateral():
    p = pd.DataFrame([{"asset_class": "equity", "instrument_id": "A", "mv_usd": 100.0}, {"asset_class": "equity", "instrument_id": "B", "mv_usd": -50.0}, {"asset_class": "bond", "instrument_id": "C", "mv_usd": 40.0}, {"asset_class": "fx_forward", "instrument_id": "F", "mv_usd": -3.0}])
    insts = {"C": B.Instrument("C", B.BOND, "C", "USD", rating_bucket="HY")}
    req = MG.requirement(p, insts, {"initial": 7.0}, MG.MarginParams())
    assert abs(req["equity"] - (0.15 * 100 + 0.20 * 50 + 0.10 * 150)) < 1e-9     # both names above 4 % of the book: the add-on applies
    assert abs(req["bonds"] - 0.15 * 40) < 1e-9 and req["fx_variation"] == 3.0 and abs(req["total"] - (req["equity"] + 6.0 + 7.0 + 3.0)) < 1e-9
    coll = MG.collateral(1000.0, p)
    assert abs(coll["cash"] - (1000.0 - 90.0 + 3.0)) < 1e-9


def test_attribution_identity(ctx):
    """the components sum to the change in value the marks show, with fills in between"""
    rm = ctx.rm; d = dt.date(2026, 7, 2); d0 = dt.date(2026, 7, 1); betas = rm.betas(d)
    syms = [s for s in rm.close.columns if pd.notna(rm.close.at[d0, s]) and pd.notna(rm.close.at[d, s])][:60]
    insts = {s: ctx.insts[s] for s in syms}; insts["ESU26"] = B.futures_instrument("ESU26", ctx.specs)
    cusip = next(k for k, v in ctx.insts.items() if v.asset_class == B.BOND); insts[cusip] = ctx.insts[cusip]
    pos = {("MOM", s): 1000.0 * (1 if i % 2 else -1) for i, s in enumerate(syms)}; pos[("HEDGE", "ESU26")] = -20.0; pos[("CREDIT", cusip)] = 5e6
    m0 = {s: float(rm.close.at[d0, s]) for s in syms}; m1 = {s: float(rm.close.at[d, s]) for s in syms}
    m0["ESU26"], m1["ESU26"] = 6300.0, 6350.0; m0[cusip], m1[cusip] = 98.0, 98.4
    fx0 = ctx.fx.loc[d0].to_dict(); fx1 = ctx.fx.loc[d].to_dict(); r0 = ctx.rates.loc[d0].dropna().to_dict(); r1 = ctx.rates.loc[d].dropna().to_dict()
    fills = [{"strategy": "MOM", "instrument_id": syms[0], "qty": 500.0, "px": m0[syms[0]] * 1.001, "kind": "equity", "cost": 0.0}, {"strategy": "HEDGE", "instrument_id": "ESU26", "qty": 5.0, "px": 6320.0, "kind": "sizing", "cost": -7.5}, {"strategy": "CREDIT", "instrument_id": cusip, "qty": 1e6, "px": 98.2, "kind": "rfq", "cost": 0.0}]
    rows = PL.attribute(d, pos, fills, insts, m0, m1, fx0, fx1, r0, r1, rm, betas, spread_chg={"IG": 1.0, "HY": 2.0}, d0=d0)
    # the change in value: equities and bond mv with fills, futures settlement variation and costs
    expected = 0.0
    for (strat, iid), q in pos.items():
        inst = insts[iid]
        if inst.asset_class == B.EQUITY:
            expected += q * (m1[iid] * V.usd_per(inst.currency, fx1) - m0[iid] * V.usd_per(inst.currency, fx0)) * inst.price_scale
        elif inst.asset_class == B.FUTURE:
            expected += q * (m1[iid] - m0[iid]) * inst.multiplier
        else:
            expected += q * (m1[iid] - m0[iid]) / 100.0 + q * (B.bond_accrued(inst, d) - B.bond_accrued(inst, d0)) / 100.0
    for f in fills:
        inst = insts[f["instrument_id"]]
        if inst.asset_class == B.EQUITY:
            expected += f["qty"] * (m1[f["instrument_id"]] - f["px"]) * inst.price_scale * V.usd_per(inst.currency, fx1)
        elif inst.asset_class == B.FUTURE:
            expected += f["qty"] * (m1[f["instrument_id"]] - f["px"]) * inst.multiplier + f["cost"]
        else:
            expected += f["qty"] * (m1[f["instrument_id"]] - f["px"]) / 100.0
    assert abs(PL.total(rows) - expected) < 1e-6 * max(abs(expected), 1.0)
    comps = {k[2] for k in rows}
    assert {"market", "idio", "execution", "settlement_variation", "carry", "rates", "credit"} <= comps


def test_exposures_and_var(ctx):
    d = dt.date(2026, 7, 2); rm = ctx.rm; betas = rm.betas(d); fh = rm.factor_history(d)
    syms = [s for s in rm.close.columns if pd.notna(rm.close.at[d, s])][:30]
    insts = {s: ctx.insts[s] for s in syms}; insts["ZNU26"] = B.futures_instrument("ZNU26", ctx.specs); insts["6EU26"] = B.futures_instrument("6EU26", ctx.specs)
    pos = {("MOM", s): 2000.0 for s in syms}; pos[("RATES", "ZNU26")] = -100.0; pos[("MACRO", "6EU26")] = 50.0
    marks = {s: float(rm.close.at[d, s]) for s in syms}; marks["ZNU26"] = 110.0; marks["6EU26"] = 1.15
    fx = ctx.fx.loc[d].to_dict(); rates = ctx.rates.loc[d].dropna().to_dict()
    ex = X.compute(d, 0.0, pos, insts, marks, fx, rates, rm, 2e9, betas, factor_hist=fh, idio_vol=rm.idio_vol(d), cov=fh.cov(), fut_beta={"ES": 1.0})
    assert ex.by_class["equity"]["net"] > 0 and ex.by_class["future"]["gross"] > 0
    assert ex.dv01 < 0 and abs(ex.dv01 + 100 * rm.treasury_dv01("ZN", d)) < 1e-6
    eur_equity = float(ex.positions.loc[(ex.positions["asset_class"] == "equity") & (ex.positions["currency"] == "EUR"), "notional_usd"].sum())
    assert abs(ex.by_currency["EUR"] - (50 * 1.15 * 125000 + eur_equity)) < 1e-6
    assert ex.var_param > 0 and ex.var_hist > 0 and ex.factor_exposure["UST7Y"] > 0     # short ZN: gains when yields rise
    # doubling every position doubles the parametric VaR
    ex2 = X.compute(d, 0.0, {k: 2 * v for k, v in pos.items()}, insts, marks, fx, rates, rm, 2e9, betas, factor_hist=fh, idio_vol=rm.idio_vol(d), cov=fh.cov(), fut_beta={"ES": 1.0})
    assert abs(ex2.var_param / ex.var_param - 2.0) < 1e-6


def test_risk_model_dv01_and_factors(ctx):
    rm = ctx.rm; d = dt.date(2026, 7, 1)
    dv = {r: rm.treasury_dv01(r, d) for r in ("ZT", "ZF", "ZN", "ZB")}
    assert dv["ZT"] < dv["ZF"] < dv["ZN"] < dv["ZB"] and 20 < dv["ZT"] < 60 and 80 < dv["ZB"] < 250
    fh = rm.factor_history(d)
    assert len(fh) == 250 and fh.isna().sum().sum() == 0
    assert 0.5 < rm.betas(d)["beta_mkt"].median() < 1.5
    f, idio = rm.cross_section(d, rm.ret.loc[d].dropna(), betas=rm.betas(d))
    assert abs(idio.mean()) < 5e-3 and np.isfinite(f).all()
