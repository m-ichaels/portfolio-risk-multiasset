"""Multi-day pipeline: build the risk model, the settlements, the bond mark paths and the book; run the days with
fault days alternating with clean days; write the DuckDB tables and results/*.json."""
from __future__ import annotations

import datetime as dt
import os
import time

import numpy as np
import pandas as pd

from . import book as B, checks as CK, data, engine as E, faults as F, futures as FU, limits as L, margin as MG, pnl as PL, valuation as V
from .monitor import Thresholds
from .riskmodel import RiskModel
from .xops_bridge import SOURCE, cal

BOND_IDIO_BP = {"IG": 1.5, "HY": 4.0}      # daily idiosyncratic yield noise in the modelled bond mark paths (simulated)


def build_context(cfg: B.BookConfig | None = None, limits: L.LimitSet | None = None) -> E.Context:
    cfg = cfg or B.BookConfig(); uni = data.load_universe(); prices = data.load_equity_prices(); fut = data.load_futures(); fx = data.load_fx(); rates = data.load_rates(); etf = data.load_etf_prices(); bonds = data.load_bonds(); specs = data.load_futures_specs()
    rm = RiskModel(prices, uni, fut, fx, rates, etf, bonds, specs)
    insts = B.equity_instruments(uni)
    binst, par = B.bond_instruments(bonds, cfg); insts.update(binst)
    sett = FU.Settlements(fut, specs)
    ctx = E.Context(insts=insts, rm=rm, sett=sett, specs=specs, ranges={}, targets=data.load_equity_targets(), prices=prices, uni=uni, fx=fx, rates=rates, bond_paths=pd.DataFrame(), cfg=cfg, limits=limits or L.LimitSet(), mp=MG.MarginParams(), th=Thresholds())
    ctx.bond_par = par
    return ctx


def bond_mark_paths(ctx: E.Context, bonds: pd.DataFrame, start: dt.date, end: dt.date, seed: int = 1) -> pd.DataFrame:
    """daily clean prices per CUSIP anchored on iShares' evaluated price at its as-of date and driven backwards by
    the matched Treasury par-yield change and the bucket's ETF-implied spread change, plus idiosyncratic noise"""
    rm = ctx.rm; rng = np.random.default_rng(seed); anchor = bonds["asof"].max()
    days = [d for d in rm.rate_chg.index if start - dt.timedelta(days=10) <= d <= max(end, anchor)]
    cols = {}
    b = bonds.set_index("cusip")
    for cusip, inst in ctx.insts.items():
        if inst.asset_class != B.BOND:
            continue
        p_anchor = float(b.loc[cusip, "price"]) if cusip in b.index else 100.0
        ten = rm.bond_tenor(inst, anchor); dur = inst.duration
        dy = rm.rate_chg[ten].reindex(days).fillna(0.0).values; ds = rm.cs[f"CS_{inst.rating_bucket}"].reindex(days).fillna(0.0).values
        eps = rng.normal(0.0, BOND_IDIO_BP[inst.rating_bucket], len(days))
        r = -dur * (dy + ds + eps) / 1e4                                   # price return on each day
        path = np.empty(len(days)); k_anchor = max(i for i, d in enumerate(days) if d <= anchor)
        path[k_anchor] = p_anchor
        for k in range(k_anchor - 1, -1, -1):
            path[k] = path[k + 1] / (1.0 + r[k + 1])
        for k in range(k_anchor + 1, len(days)):
            path[k] = path[k - 1] * (1.0 + r[k])
        cols[cusip] = path
    return pd.DataFrame(cols, index=days)


def initial_state(ctx: E.Context, start: dt.date) -> E.BookState:
    """the book on the day before the window: equities at execution-ops' targets, bonds at par x the SOD mark,
    nothing else yet (the hedges, the steepener, the macro book and the FX hedge trade on the first day)"""
    cfg = ctx.cfg; rm = ctx.rm; d0 = cal.next_trading_day("XNYS", start, -1)
    tg = ctx.targets[ctx.targets["date"] <= d0]; last = tg["date"].max(); tg = tg[tg["date"] == last]
    st = E.BookState(nav=cfg.aum)
    for r in tg.itertuples():
        if r.symbol in ctx.insts:
            st.positions[(r.strategy, r.symbol)] = float(r.target)
    for cusip, par in ctx.bond_par.items():
        st.positions[("CREDIT", cusip)] = par
    close = rm.close.loc[:d0].ffill().iloc[-1]; fxr = ctx.fx.loc[:d0].iloc[-1].to_dict()
    bp = ctx.bond_paths; bprev = bp.loc[[x for x in bp.index if x <= d0][-1]]
    mv = 0.0
    for (s, iid), q in st.positions.items():
        inst = ctx.insts[iid]
        px = float(close.get(iid, np.nan)) if inst.asset_class == B.EQUITY else float(bprev[iid])
        if np.isfinite(px):
            mv += V.mark(inst, q, px, d0, fxr, {})[0]
    st.cash = cfg.aum - mv
    return st


def write_day(con, r: E.DayResult, insts: dict):
    d = r.date
    for tbl in ("positions", "exposures", "limits", "pnl", "margin", "alerts", "faults", "rolls", "checks"):
        con.execute(f"delete from {tbl} where date = ?", [d])
    con.execute("delete from rfq_trades where date = ?", [d]); con.execute("delete from fx_settlements where value_date = ?", [d])
    rows = []
    for src, ex in (("SOD", r.ex_sod), ("EOD", r.ex_eod)):
        for p in ex.positions.itertuples():
            rows.append((d, "book", src, p.strategy, p.instrument_id, p.qty, p.px, 1.0, p.notional_usd))
    con.executemany("insert into positions values (?,?,?,?,?,?,?,?,?)", rows)
    ex_rows = r.ex_sod.rows() + r.ex_eod.rows()
    for ex in r.exposure_log[::6]:
        ex_rows += [(d, ex.time, "nav", "nav", ex.nav), (d, ex.time, "gross", "all", ex.gross), (d, ex.time, "beta_dollars", "all", ex.beta_dollars), (d, ex.time, "dv01", "all", ex.dv01), (d, ex.time, "var", "param99", ex.var_param)]
    con.executemany("insert into exposures values (?,?,?,?,?)", ex_rows)
    con.executemany("insert into limits values (?,?,?,?,?,?,?,?)", [(d, x["time"], x["limit_name"], x["scope"], x["value"], x["threshold"], x["utilisation"], x["status"]) for x in r.limits_sod + r.limits_eod])
    con.executemany("insert into pnl values (?,?,?,?,?)", [(d, k[0], k[1], k[2], v) for k, v in r.pnl_rows.items()])
    m = r.margin_eod
    mrows = [(d, "PB", k, v) for k, v in m["requirement"].items()] + [(d, "PB", "collateral_cash", m["collateral"]["cash"]), (d, "PB", "utilisation", m["utilisation"])]
    mrows += [(d, "CME", k, v) for k, v in m["futures"].items() if k != "by_root"] + [(d, "CME:" + k, "scan_risk", v["scan_risk"]) for k, v in m["futures"]["by_root"].items()] + [(d, "financing", k, v) for k, v in r.financing.items()]
    con.executemany("insert into margin values (?,?,?,?)", mrows)
    if r.alerts:
        con.executemany("insert into alerts values (?,?,?,?,?,?)", [(d, a["time"], a["rule"], a["severity"], a["scope"], a["detail"]) for a in r.alerts])
    if r.detections:
        con.executemany("insert into faults values (?,?,?,?,?,?)", [(d, x.t_fault, x.type, x.scope, x.t_alert, x.rule) for x in r.detections])
    if r.rolls:
        con.executemany("insert into rolls values (?,?,?,?,?,?,?,?,?,?)", [(d, x["root"], x["from_contract"], x["to_contract"], x["lots"], x["front_px"], x["next_px"], x["spread_ticks"], x["cost_usd"], x["observed"]) for x in r.rolls])
    if r.settlements:
        by_ccy = r.netting.get("by_ccy_net", {})
        con.executemany("insert into fx_settlements values (?,?,?,?,?,?,?,?)", [(s["value_date"], s["trade_date"], s["pair"], s["counterparty"], s["ccy"], s["amount"], by_ccy.get(s["ccy"], 0.0), bool(s.get("confirmed"))) for s in r.settlements])
    if r.rfqs:
        con.executemany("insert into rfq_trades values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [(d, x["time"], x["cusip"], x["side"], x["par"], x["price"], x["mark"], x["winner"], x["cover"], x["n_quotes"], x["markup_pts"], x["markup_bp_yield"], x["trace_vwap"], x["trace_prints"], x["trace_dev_pts"], x["report_delay_s"], x["flags"]) for x in r.rfqs])
    if r.checks:
        con.executemany("insert into checks values (?,?,?,?,?)", [(c["date"], c["phase"], c["check_name"], c["status"], c["detail"]) for c in r.checks])


def run_range(start: dt.date, end: dt.date, *, fault_days: str = "alternate", seed: int = 1, cfg: B.BookConfig | None = None, db_path: str | None = None, verbose: bool = True, results_dir: str | None = None) -> dict:
    t0 = time.monotonic(); ctx = build_context(cfg); cfg = ctx.cfg; bonds = data.load_bonds()
    ctx.bond_paths = bond_mark_paths(ctx, bonds, start, end, seed)
    ctx.ranges = MG.scan_ranges(data.load_futures(), ctx.specs, start)
    state = initial_state(ctx, start)
    con = data.connect(db_path or data.DB_PATH)
    con.execute("delete from instruments")
    days = cal.trading_days("XNYS", start, end); per_day = []; all_dets = []; clean_alerts = []; n_clean = 0; results: list[E.DayResult] = []
    for i, d in enumerate(days):
        fault = (i % 2 == 1) if fault_days == "alternate" else (fault_days == "all")
        mix = F.DEFAULT_MIX if fault else None

        def checks_fn(dd, phase, ex, lim, mon, st, rolls, settle, gap=None, rfqs=None):
            return CK.run(dd, phase, ex, lim, mon, st, rolls, settle, ctx, gap=gap, rfqs=rfqs)
        r = E.run_day(d, ctx, state, fault_mix=mix, seed=seed, verbose=verbose, checks_fn=checks_fn)
        if r is None:
            continue
        write_day(con, r, ctx.insts); results.append(r)
        if not fault:
            clean_alerts += r.alerts; n_clean += 1
        all_dets += r.detections
        per_day.append({"date": d.isoformat(), "fault_day": fault, "nav_sod": r.nav_sod, "nav_eod": r.nav_eod, "pnl": r.nav_eod - r.nav_sod, "identity_gap": r.identity_gap, "fills": len(r.fills), "alerts": len(r.alerts), "incidents": len({(a["rule"], a["scope"]) for a in r.alerts}),
                        "faults_injected": len(r.detections), "faults_detected": sum(x.detected for x in r.detections), "gross": r.ex_eod.gross, "net": r.ex_eod.net, "beta_dollars": r.ex_eod.beta_dollars, "dv01": r.ex_eod.dv01, "var_param": r.ex_eod.var_param, "var_hist": r.ex_eod.var_hist,
                        "margin_util": r.margin_eod["utilisation"], "margin_req": r.margin_eod["requirement"]["total"], "futures_im": r.margin_eod["futures"]["initial"], "futures_im_outrights": r.margin_eod["futures"]["sum_of_outrights"], "financing": r.financing["total"],
                        "limits_breach": sum(x["status"] == "breach" for x in r.limits_eod), "limits_warn": sum(x["status"] == "warn" for x in r.limits_eod), "rolls": len(r.rolls), "fx_trades": len(r.fx_trades), "settlements": len(r.settlements), "netting": r.netting, "rfqs": len(r.rfqs),
                        "checks_warn": sum(c["status"] == "warn" for c in r.checks), "checks_fail": sum(c["status"] == "fail" for c in r.checks)})
    con.executemany("insert or replace into instruments values (?,?,?,?,?,?,?,?,?)", [v.as_row() for v in ctx.insts.values()])
    # ---- summaries ---------------------------------------------------------------------------------------------------------------
    monitor = F.summarize(all_dets, clean_alerts, n_clean); monitor["fault_days"] = sum(p["fault_day"] for p in per_day); monitor["clean_days"] = n_clean
    pnl = con.execute("select strategy, asset_class, component, sum(value) v from pnl where date between ? and ? group by 1,2,3 order by 1,2,3", [start, end]).df()
    pnl_summary = {"by_component": pnl.groupby(["asset_class", "component"])["v"].sum().to_dict(), "by_strategy": pnl.groupby("strategy")["v"].sum().to_dict(), "total": float(pnl["v"].sum()), "identity_gap_max": max(abs(p["identity_gap"]) for p in per_day)}
    pnl_summary["by_component"] = {f"{k[0]}:{k[1]}": float(v) for k, v in pnl_summary["by_component"].items()}
    rolls = con.execute("select * from rolls where date between ? and ?", [start, end]).df()
    roll_summary = {}
    for root, g in rolls.groupby("root"):
        roll_summary[root] = {"rolls": int(len(g)), "lots": float(g["lots"].abs().sum()), "observed": int(g["observed"].sum()), "spread_ticks_mean": float(g.loc[g["observed"], "spread_ticks"].mean()) if g["observed"].any() else None, "cost_per_contract_usd": float((g.loc[g["observed"], "cost_usd"] / g.loc[g["observed"], "lots"].abs()).mean()) if g["observed"].any() else None, "total_cost_usd": float(g["cost_usd"].sum())}
    fxs = con.execute("select * from fx_settlements where value_date between ? and ?", [start, end]).df()
    fx_summary = {"trades": sum(p["fx_trades"] for p in per_day), "instructions": int(len(fxs)), "gross_usd": sum(p["netting"]["gross_usd"] for p in per_day), "bilateral_usd": sum(p["netting"]["bilateral_usd"] for p in per_day), "multilateral_usd": sum(p["netting"]["multilateral_usd"] for p in per_day), "unconfirmed": int((~fxs["confirmed"]).sum()) if len(fxs) else 0}
    fx_summary["netting_reduction"] = 1 - fx_summary["multilateral_usd"] / fx_summary["gross_usd"] if fx_summary["gross_usd"] else None
    rq = con.execute("select * from rfq_trades where date between ? and ?", [start, end]).df()
    credit_summary = {"trades": int(len(rq)), "markup_pts_mean": float(rq["markup_pts"].mean()) if len(rq) else None, "markup_bp_mean": float(rq["markup_bp_yield"].mean()) if len(rq) else None, "cover_pts_mean": float(rq["cover"].mean()) if len(rq) else None,
                      "trace_dev_pts_mean": float(rq["trace_dev_pts"].mean()) if len(rq) else None, "flagged": int((rq["flags"] != "").sum()) if len(rq) else 0, "flags": rq["flags"].value_counts().to_dict() if len(rq) else {}, "winners": rq["winner"].value_counts().to_dict() if len(rq) else {}, "late_reports": int((rq["report_delay_s"] > 900).sum()) if len(rq) else 0}
    lim = con.execute("select limit_name, status, count(*) n from limits where date between ? and ? group by 1,2 order by 1,2", [start, end]).df().to_dict("records")
    checks = con.execute("select phase, check_name, status, count(*) n from checks where date between ? and ? group by 1,2,3 order by 1,2,3", [start, end]).df().to_dict("records")
    margin_summary = {"utilisation_mean": float(np.mean([p["margin_util"] for p in per_day])), "utilisation_max": float(max(p["margin_util"] for p in per_day)), "requirement_mean": float(np.mean([p["margin_req"] for p in per_day])), "futures_im_mean": float(np.mean([p["futures_im"] for p in per_day])),
                      "futures_im_outrights_mean": float(np.mean([p["futures_im_outrights"] for p in per_day])), "financing_total": float(sum(p["financing"] for p in per_day)), "scan_ranges": ctx.ranges}
    fut_check = ctx.sett.continuous_vs_front(start - dt.timedelta(days=400), end)
    out = {"from": start.isoformat(), "to": end.isoformat(), "days": per_day, "monitor": monitor, "pnl": pnl_summary, "rolls": roll_summary, "fx": fx_summary, "credit": credit_summary, "margin": margin_summary, "limits": lim, "checks": checks, "futures_continuous_check": fut_check,
           "config": {"aum": cfg.aum, "seed": seed, "hedge_ratio": cfg.hedge_ratio, "steepener_dv01": cfg.steepener_dv01, "macro_lots": cfg.macro_lots, "credit_par": {"IG": cfg.credit_par_ig, "HY": cfg.credit_par_hy}, "roll_days_before": cfg.roll_days_before, "xops_source": SOURCE, "limits": vars(ctx.limits)}, "runtime_s": time.monotonic() - t0}
    R = results_dir or data.RESULTS
    data.to_json(out, os.path.join(R, "run.json")); data.to_json(monitor, os.path.join(R, "monitor.json"))
    con.close()
    return out
