#!/usr/bin/env python3
"""README.md from scripts/README.template.md, results/run.json and the store.   python scripts/fill_readme.py"""
import json
import os
import re

import duckdb
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
run = json.load(open(os.path.join(ROOT, "results", "run.json"), encoding="utf-8"))
con = duckdb.connect(os.path.join(ROOT, "data", "derived", "xrisk.duckdb"), read_only=True)
days = pd.DataFrame(run["days"]); m = run["monitor"]; pn = run["pnl"]; mg = run["margin"]; fx = run["fx"]; cr = run["credit"]
bonds = pd.read_parquet(os.path.join(ROOT, "data", "derived", "bonds.parquet"))


def mins(x):
    return "-" if x is None else (f"{x:.0f} s" if x < 90 else f"{x / 60:.0f} min")


def table(cols, rows):
    return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)


rep = {}
rep["FAULT_DAYS"] = m["fault_days"]; rep["DET"] = m["detected"]; rep["INJ"] = m["injected"]; rep["DET_PCT"] = f"{100 * m['detection_rate']:.1f}"; rep["MED_TTD"] = ("the same five-minute step" if (m["median_ttd_s"] or 0) < 1 else mins(m["median_ttd_s"])) + f" (p90 {mins(m['p90_ttd_s'])})"; rep["CLEAN_DAYS"] = m["clean_days"]
rep["FA"] = f"{m['false_alerts_per_clean_day']:.1f}"
fa = sorted(m["false_alerts_by_rule"].items(), key=lambda kv: -kv[1]); tot = sum(v for _, v in fa)
rep["FA_DESC"] = ", ".join(f"{k} {v}" for k, v in fa[:5]) + (f" of {tot}" if len(fa) > 5 else "") + " (limit incidents the book itself causes, and the P&L rule on large days)"
pos = con.execute("select i.asset_class, sum(abs(p.notional_usd)) g from positions p join instruments i using (instrument_id) where p.phase='EOD' group by 1").df().set_index("asset_class")["g"] / days.shape[0]
rep["EQ_GROSS"] = f"{pos.get('equity', 0) / 1e9:.1f}"; rep["GROSS_X"] = f"{days['gross'].mean() / days['nav_eod'].mean():.2f}"
rep["VAR_PCT"] = f"{100 * days['var_param'].mean() / days['nav_eod'].mean():.2f}"; rep["VARH_PCT"] = f"{100 * days['var_hist'].mean() / days['nav_eod'].mean():.2f}"
rep["VAR_EXC"] = int((days["pnl"] < -days["var_param"]).sum()); rep["NDAYS"] = len(days); rep["GAP"] = f"{pn['identity_gap_max']:.4f}"
rl = con.execute("select * from rolls order by date").df(); rs = run["rolls"]
rep["N_ROLLS"] = len(rl)
obs = rl[rl["observed"]]
parts = []
for root, g in obs.groupby("root"):
    cpc = (g["cost_usd"] / g["lots"].abs()).mean(); parts.append(f"{root} {cpc:+,.0f} ({g['spread_ticks'].mean():+.0f} ticks)")
rep["ROLL_DESC"] = ("was measured on " + ", ".join(parts) + f"; {int((~rl['observed']).sum())} roll's next month was not listed on Yahoo, so its spread is unobserved") if len(obs) else "could not be observed"
rep["IM"] = f"{mg['futures_im_mean'] / 1e6:.1f}"; rep["IM_OUT"] = f"{mg['futures_im_outrights_mean'] / 1e6:.1f}"
rep["FX_TRADES"] = fx["trades"]; rep["FX_INSTR"] = fx["instructions"]; rep["FX_DATES"] = int(con.execute("select count(distinct value_date) from fx_settlements").fetchone()[0])
rep["FX_MULTI"] = f"{fx['multilateral_usd'] / 1e6:,.0f}"; rep["FX_GROSS"] = f"{fx['gross_usd'] / 1e6:,.0f}"; rep["FX_RED"] = f"{100 * (fx['netting_reduction'] or 0):.0f}"; rep["FX_UNCONF"] = fx["unconfirmed"]
rq = con.execute("select * from rfq_trades").df().merge(bonds[["cusip", "fund"]].drop_duplicates("cusip"), on="cusip", how="left")
ig = rq[rq["fund"] == "LQD"]; hy = rq[rq["fund"] == "HYG"]
rep["RFQ_N"] = len(rq); rep["MK_IG"] = f"{ig['markup_pts'].mean():.2f}"; rep["MK_IG_BP"] = f"{ig['markup_bp_yield'].mean():.1f}"; rep["MK_HY"] = f"{hy['markup_pts'].mean():.2f}"; rep["MK_HY_BP"] = f"{hy['markup_bp_yield'].mean():.1f}"; rep["COVER"] = f"{rq['cover'].mean():.2f}"
rep["RFQ_FLAG"] = int((rq["flags"] != "").sum()); rep["RFQ_FAULTS"] = int(sum(v["injected"] for k, v in m["by_type"].items() if k in ("late_trace_report", "off_market_rfq")))
rep["MARGIN_UTIL"] = f"{100 * days.loc[~days['fault_day'], 'margin_util'].mean():.0f}"; rep["MARGIN_MAX"] = f"{100 * days.loc[~days['fault_day'], 'margin_util'].max():.0f}"; rep["FIN"] = f"{mg['financing_total'] / 1e6:+.2f}"
rep["N_TESTS"] = len(re.findall(r"^def test_", open(os.path.join(ROOT, "tests", "test_risk.py")).read() + open(os.path.join(ROOT, "tests", "test_ops.py")).read(), re.M))
rep["BOND_ASOF"] = str(bonds["asof"].max())
# tables
p2 = con.execute("select p.strategy, i.asset_class, count(*) n, sum(abs(p.notional_usd)) gross, sum(p.notional_usd) net from positions p join instruments i using (instrument_id) where p.phase='EOD' and p.date=(select max(date) from positions) group by 1,2 order by 1,2").df()
rep["BOOK_TABLE"] = table(["strategy", "asset class", "positions", "gross $m", "net $m"], [(r.strategy, r.asset_class, r.n, f"{r.gross / 1e6:,.0f}", f"{r.net / 1e6:+,.0f}") for r in p2.itertuples()]) + f"\n\nAt the last close. Over the run: gross ${days['gross'].mean() / 1e9:.2f}bn, net ${days['net'].mean() / 1e6:+,.0f}m, beta-dollars ${days['beta_dollars'].mean() / 1e6:+,.0f}m after the hedge, DV01 ${days['dv01'].mean():+,.0f}/bp; daily P&L mean ${days['pnl'].mean() / 1e6:+.2f}m, sd ${days['pnl'].std() / 1e6:.2f}m, worst ${days['pnl'].min() / 1e6:+.1f}m."
rows = []
for t, s in sorted(m["by_type"].items(), key=lambda kv: -kv[1]["injected"]):
    rules = con.execute("select detected_rule, count(*) n from faults where type = ? and detected_rule is not null group by 1 order by 2 desc", [t]).fetchall()
    rows.append((t, s["injected"], s["detected"], mins(s["median_ttd_s"]), ", ".join(f"{r} ({n})" for r, n in rules)))
rep["MONITOR_TABLE"] = table(["fault", "injected", "caught", "median time to detect", "the monitor sees"], rows)
comp = sorted(pn["by_component"].items())
rep["PNL_TABLE"] = table(["component", "$m"], [(k, f"{v / 1e6:+.2f}") for k, v in comp]) + "\n\n" + table(["strategy", "$m"], [(k, f"{v / 1e6:+.2f}") for k, v in sorted(pn["by_strategy"].items())]) + f"\n\nTotal ${pn['total'] / 1e6:+.1f}m; the largest gap between the NAV change and the attributed total on any day is {pn['identity_gap_max']:.4f} USD."
sr = mg["scan_ranges"]
cme = con.execute("select component, avg(value) v from margin where account = 'CME' group by 1").df().set_index("component")["v"]
rep["MARGIN_DESC"] = f"Requirement ${mg['requirement_mean'] / 1e6:,.0f}m on average against the cash collateral ({rep['MARGIN_UTIL']} % utilisation on clean days, {rep['MARGIN_MAX']} % at worst); the margin-shortfall fault (the prime broker doubling its haircuts to 60 %) takes it above 100 % on fault days and the monitor calls it. Futures: SPAN-style initial margin ${mg['futures_im_mean'] / 1e6:.1f}m a day on average = scan risk ${cme.get('scan', 0) / 1e6:.1f}m + calendar-spread charges ${cme.get('spread_charge', 0) / 1e6:.2f}m (the roll days) + delivery charges ${cme.get('delivery_charge', 0) / 1e6:.2f}m (contracts inside their notice window before the roll) - inter-commodity credit ${cme.get('inter_commodity_credit', 0) / 1e6:.2f}m (the steepener's ZT against ZN; the index hedge's ES and NQ are the same way round and earn none), against ${mg['futures_im_outrights_mean'] / 1e6:.1f}m as the plain sum of outright scan charges. Scan ranges per contract from the data: " + ", ".join(f"{k} ${v['scan']:,.0f} (spread ${v['spread']:,.0f})" for k, v in sr.items()) + f". Financing ${mg['financing_total'] / 1e6:+.2f}m over the run (cash interest less stock borrow)."
rep["ROLL_TABLE"] = table(["date", "roll", "lots", "front", "next", "spread (ticks)", "roll P&L $/contract", "spread observed"], [(r.date, f"{r.from_contract}->{r.to_contract}", f"{r.lots:+.0f}", f"{r.front_px:.4f}", f"{r.next_px:.4f}", f"{r.spread_ticks:+.1f}", f"{r.cost_usd / abs(r.lots):+,.0f}", "yes" if r.observed else "no") for r in rl.itertuples()]) if len(rl) else "no roll in the window"
chk = run["futures_continuous_check"]
rep["FUT_CHECK"] = "Yahoo's continuous series against the listed front month on the days both exist: " + "; ".join(f"{k} {v['equal']}/{v['days_both']}" for k, v in chk.items()) + ". The substitution of the continuous series for an expired front month is used where that ratio is one (ES, NQ, RTY, CL); the Treasuries' listed months cover the window; the FX contracts' and gold's earlier front months are the continuous series on the evidence of the spread's smooth decay to the next month, stated as unverified."
fxs = con.execute("select value_date, ccy, count(*) n, sum(abs(amount)) gross, abs(sum(amount)) net from fx_settlements group by 1,2 order by 1,2").df()
rep["FX_DESC"] = f"{fx['trades']} forward trades, {fx['instructions']} settlement instructions on {rep['FX_DATES']} value dates (T+2 and one month on the USD, TARGET2, UK and Tokyo calendars, modified following, end-end). Gross ${fx['gross_usd'] / 1e6:,.0f}m; bilateral payment netting ${fx['bilateral_usd'] / 1e6:,.0f}m; multilateral netting per currency ${fx['multilateral_usd'] / 1e6:,.0f}m, a {rep['FX_RED']} % reduction. {fx['unconfirmed']} instructions unconfirmed at the 12:00 cutoff, every one the injected fault.\n\n" + table(["value date", "ccy", "instructions", "gross", "net"], [(r.value_date, r.ccy, r.n, f"{r.gross:,.0f}", f"{r.net:,.0f}") for r in fxs.itertuples()])
rows = []
for fund, g in rq.groupby("fund"):
    rows.append((f"{fund} ({'IG' if fund == 'LQD' else 'HY'})", len(g), f"{g['markup_pts'].mean():.3f}", f"{g['markup_pts'].median():.3f}", f"{g['markup_bp_yield'].mean():.1f}", f"{g['cover'].mean():.3f}", f"{g['trace_dev_pts'].mean():.3f}", int((g['trace_prints'] > 0).sum()), int((g['flags'] != '').sum())))
rep["CREDIT_TABLE"] = table(["bucket", "trades", "mean markup (pts)", "median", "mean markup (bp yield)", "mean cover (pts)", "mean dev. vs TRACE VWAP", "with prints", "flagged"], rows) + f"\n\nFlags: {cr['flags']}. Winning dealers: {cr['winners']} (the dealers differ in skill by construction, and the best quote wins). Late TRACE reports: {cr['late_reports']}, all the injected fault."
p = os.path.join(ROOT, "README.md"); s = open(os.path.join(ROOT, "scripts", "README.template.md"), encoding="utf-8").read()
for k, v in rep.items():
    s = s.replace(f"__{k}__", str(v))
left = re.findall(r"__[A-Z_]+__", s)
open(p, "w", encoding="utf-8", newline="\n").write(s); print("filled; left:", left)
