"""Futures rolls and settlements, FX value dates, netting and parity, the credit post-trade, the fault scorer, and one
engine day on the committed data with faults injected."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from xrisk import book as B, credit as CR, data, faults as F, futures as FU, fx as FX, margin as MG, valuation as V
from xrisk.xops_bridge import cal


@pytest.fixture(scope="module")
def sett():
    return FU.Settlements(data.load_futures(), data.load_futures_specs())


def test_front_and_should_hold(sett):
    assert sett.front("ES", dt.date(2026, 7, 1)) == "ESU26" and sett.front("ES", dt.date(2026, 9, 19)) == "ESZ26"
    assert sett.front("ZN", dt.date(2026, 8, 28)) == "ZNU26" and sett.front("ZN", dt.date(2026, 8, 31)) == "ZNZ26"        # first notice 31 August
    assert sett.should_hold("ZN", dt.date(2026, 8, 27), 2) == "ZNZ26"                                                    # two CME days ahead of it
    assert sett.front("CL", dt.date(2026, 7, 1)) == "CLQ26" and sett.front("GC", dt.date(2026, 8, 1)) == "GCZ26"        # gold skips the thin October
    assert sett.front("6B", dt.date(2026, 9, 11)) == "6BU26" and sett.front("6B", dt.date(2026, 9, 15)) == "6BZ26"


def test_settlement_sources_and_continuous_check(sett):
    p, src = sett.settle("ESU26", dt.date(2026, 7, 1)); assert src == "listed" and p > 1000
    p, src = sett.settle("CLQ26", dt.date(2026, 7, 1)); assert src == "continuous" and p > 10        # expired month, Yahoo's =F was the front then
    p, src = sett.settle("CLU26", dt.date(2026, 7, 1)); assert src == "missing" and not np.isfinite(p)
    chk = sett.continuous_vs_front(dt.date(2026, 6, 1), dt.date(2026, 9, 18))
    for root in ("ES", "NQ", "RTY", "CL"):
        assert chk[root]["days_both"] > 10 and chk[root]["equal"] == chk[root]["days_both"]


def test_roll_execution_cost(sett):
    specs = data.load_futures_specs(); insts = {}
    row = FU.execute_roll("HEDGE", "ZNU26", "ZNZ26", -100, dt.date(2026, 8, 27), sett, specs, insts)
    assert row["observed"] and "ZNZ26" in insts
    # a short rolling into a cheaper month pays the spread (it sells the dear month and buys the cheap one: gains); sign checked against the prices
    expected = -(-1.0) * (row["next_px"] - row["front_px"]) * 1000 * 100
    assert abs(row["spread_cost_usd"] - expected) < 1e-6
    assert abs(row["crossing_usd"] + 2 * 0.015625 * 1000 * 100) < 1e-6 and abs(row["fees_usd"] + 2 * 1.0 * 100) < 1e-6
    assert abs(row["cost_usd"] - (row["spread_cost_usd"] + row["crossing_usd"] + row["fees_usd"])) < 1e-6


def test_fx_value_dates_and_parity():
    assert FX.spot_date("EURUSD", dt.date(2026, 7, 1)) == dt.date(2026, 7, 6)          # 3 July is the observed US holiday
    assert FX.spot_date("GBPUSD", dt.date(2026, 8, 28)) == dt.date(2026, 9, 2)         # 31 August is a UK bank holiday
    assert FX.spot_date("USDJPY", dt.date(2026, 12, 30)) > dt.date(2027, 1, 3)         # Tokyo closed 31 December to 3 January
    s = FX.spot_date("EURUSD", dt.date(2026, 7, 29)); assert s == dt.date(2026, 7, 31)
    assert FX.forward_date("EURUSD", s, 1) == dt.date(2026, 8, 31)                     # end-end: spot on the last business day
    s2 = FX.spot_date("EURUSD", dt.date(2026, 9, 28)); f2 = FX.forward_date("EURUSD", s2, 1)
    assert f2.month == 10 and FX.good_day(("EUR", "USD"), f2)                          # modified following stays in the month
    rates = {"SOFR": 4.0, "ESTR": 2.0, "SONIA": 4.0, "JGB1Y": 0.5}
    f = V.forward_rate("EURUSD", 1.10, rates, 90); assert abs(f - 1.10 * (1 + 0.04 * 90 / 360) / (1 + 0.02 * 90 / 360)) < 1e-12
    f = V.forward_rate("USDJPY", 150.0, rates, 90); assert f < 150.0                    # yen at a forward premium: the dollar buys fewer yen forward


def test_settlement_netting():
    instr = [{"counterparty": "D1", "ccy": "EUR", "amount_usd": 100.0}, {"counterparty": "D1", "ccy": "USD", "amount_usd": -110.0}, {"counterparty": "D2", "ccy": "EUR", "amount_usd": -60.0}, {"counterparty": "D2", "ccy": "USD", "amount_usd": 66.0}, {"counterparty": "D1", "ccy": "EUR", "amount_usd": -30.0}, {"counterparty": "D1", "ccy": "USD", "amount_usd": 33.0}]
    n = FX.net_settlements(instr)
    assert n["gross_usd"] == 399.0 and n["bilateral_usd"] == 70 + 77 + 60 + 66 and abs(n["multilateral_usd"] - (10 + 11)) < 1e-9
    assert n["multilateral_usd"] <= n["bilateral_usd"] <= n["gross_usd"]


def test_credit_post_trade():
    inst = B.Instrument("X", B.BOND, "X", "USD", coupon=5.0, maturity=dt.date(2032, 1, 1), duration=5.0, rating_bucket="IG")
    rng = np.random.default_rng(3); prints = pd.DataFrame([{"cusip": "X", "date": dt.date(2026, 7, 1), "time": 1000.0 + k * 60, "price": 98.0 + 0.1 * (k % 3), "size": 1e6, "capped": False, "side": "B"} for k in range(5)])
    tr = CR.rfq(inst, +1, 2e6, 98.0, 1100.0, ("D1", "D2", "D3", "D4"), rng); tr["id"] = "RFQ1"
    assert tr["price"] > 97.9 and tr["winner"] in ("D1", "D2", "D3", "D4") and tr["cover"] >= 0   # the best of four quotes sits near the mark, occasionally through it
    pt = CR.post_trade(tr, inst, prints, 120.0)
    assert pt["trace_prints"] == 5 and pt["flags"] == "" and abs(pt["markup_bp_yield"] - 1e4 * pt["markup_pts"] / (5.0 * tr["price"])) < 1e-9
    bad = CR.post_trade(dict(tr, price=tr["price"] + 3.0), inst, prints, 2000.0)
    assert set(bad["flags"].split(",")) == {"off_market_vs_trace", "off_market_vs_mark", "late_trace_report"}
    none = CR.post_trade(tr, inst, prints.iloc[:0], 10.0); assert none["flags"] == "no_trace_prints"


def test_fault_scoring():
    f = [F.Fault("stale_price", 100.0, "AAPL"), F.Fault("margin_shortfall", 200.0, "account"), F.Fault("spiked_mark", 300.0, "MSFT")]
    for x in f:
        x.fired_at = x.t
    alerts = [{"time": 160.0, "rule": "STALE_PRICE", "scope": "AAPL"}, {"time": 150.0, "rule": "STALE_PRICE", "scope": "MSFT"}, {"time": 260.0, "rule": "MARGIN_CALL", "scope": "account"}, {"time": 5000.0, "rule": "PRICE_SPIKE", "scope": "MSFT"}]
    dets = F.score(f, alerts)
    assert dets[0].detected and dets[0].ttd == 60.0 and dets[1].detected and not dets[2].detected      # the MSFT spike alert is outside the window
    s = F.summarize(dets, clean_alerts=[{"rule": "LIMIT_WARN", "scope": "x", "date": 1}], n_clean_days=2)
    assert s["injected"] == 3 and s["detected"] == 2 and s["false_alerts_per_clean_day"] == 0.5


def test_scan_ranges_from_data():
    r = MG.scan_ranges(data.load_futures(), data.load_futures_specs(), dt.date(2026, 7, 1))
    assert r["ES"]["scan"] > r["ZN"]["scan"] > 0 and r["ES"]["spread"] < r["ES"]["scan"]
    assert r["ES"]["scan"] % (0.25 * 50) == 0                                           # a whole number of ticks


def test_engine_day_with_faults():
    """one fault day on the committed data: every planted fault caught inside its window, the P&L identity exact"""
    from xrisk import run as R, engine as E
    ctx = R.build_context(); bonds = data.load_bonds(); start = dt.date(2026, 8, 26)
    ctx.bond_paths = R.bond_mark_paths(ctx, bonds, start, dt.date(2026, 8, 27)); ctx.ranges = MG.scan_ranges(data.load_futures(), ctx.specs, start)
    state = R.initial_state(ctx, start)
    mix = {"stale_price": 1, "spiked_mark": 1, "fx_inverted": 1, "phantom_position": 1, "missing_fill": 1, "wrong_multiplier": 0, "margin_shortfall": 1, "bond_mark_stale": 1, "doubled_load": 1, "settlement_missing": 1}
    # two instructions value today: one is confirmed, the other is the fault
    for ccy, amt in (("EUR", 5e6), ("USD", -5.7e6)):
        state.settlements.append({"value_date": start, "trade_date": start - dt.timedelta(days=30), "pair": "EURUSD", "counterparty": "DLR1", "ccy": ccy, "amount": amt, "key": f"test:{ccy}"})
    r = E.run_day(start, ctx, state, fault_mix=mix, seed=3)
    assert abs(r.identity_gap) < 1.0
    assert len(r.settlements) == 2 and sum(s["confirmed"] for s in r.settlements) == 1 and r.netting["instructions"] == 2
    assert len(r.detections) >= 6 and all(x.detected for x in r.detections), [(x.type, x.detected) for x in r.detections]
    assert r.ex_eod.gross > 0 and r.margin_eod["requirement"]["total"] > 0 and len(r.rfqs) == ctx.cfg.rfq_trades_per_day
    assert any(f["kind"] == "sizing" for f in r.fills) and len(r.fx_trades) >= 1
    # a clean day after it raises no fault-type alert
    r2 = E.run_day(cal.next_trading_day("XNYS", start, 1), ctx, state, fault_mix=None, seed=3)
    assert abs(r2.identity_gap) < 1.0 and not any(a["rule"] in ("POSITION_BREAK", "MASTER_CHANGE", "FX_SPIKE", "STALE_MARK") for a in r2.alerts)
