#!/usr/bin/env python3
"""Figures from the store and results/run.json -> results/figures/*.png.   python scripts/plots.py [results]"""
import datetime as dt
import json
import os
import sys

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results")
FIG = os.path.join(R, "figures"); os.makedirs(FIG, exist_ok=True)
DB = os.path.join(ROOT, "data", "derived", "xrisk.duckdb")
run = json.load(open(os.path.join(R, "run.json"), encoding="utf-8"))
con = duckdb.connect(DB, read_only=True)
days = pd.DataFrame(run["days"]); days["date"] = pd.to_datetime(days["date"])
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "figure.dpi": 130})
C = {"fault": "#c0392b", "clean": "#2c7fb8", "grey": "#888", "a": "#1b9e77", "b": "#d95f02", "c": "#7570b3", "d": "#e7298a", "e": "#66a61e", "f": "#e6ab02"}


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name)); plt.close(fig); print("wrote", name)


# ---- book -------------------------------------------------------------------------------------------------------------------
def book():
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
    pos = con.execute("select date, strategy, instrument_id, notional_usd from positions where phase = 'EOD'").df(); pos["date"] = pd.to_datetime(pos["date"])
    inst = con.execute("select instrument_id, asset_class from instruments").df(); pos = pos.merge(inst, on="instrument_id", how="left")
    last = pos[pos["date"] == pos["date"].max()]
    g = last.groupby(["strategy", "asset_class"])["notional_usd"].agg(lambda x: x.abs().sum()).unstack().fillna(0) / 1e6
    g.plot(kind="bar", stacked=True, ax=ax[0, 0], color=[C["a"], C["b"], C["c"], C["d"]][:g.shape[1]]); ax[0, 0].set_title(f"gross exposure by strategy and asset class on {pos['date'].max().date()} ($m)"); ax[0, 0].set_xlabel(""); ax[0, 0].tick_params(axis="x", rotation=30)
    ax[0, 1].plot(days["date"], days["gross"] / 1e6, label="gross", color=C["a"]); ax[0, 1].plot(days["date"], days["net"] / 1e6, label="net", color=C["b"]); ax[0, 1].plot(days["date"], days["beta_dollars"] / 1e6, label="beta-dollars", color=C["c"])
    ax[0, 1].axhline(0, color=C["grey"], lw=0.5); ax[0, 1].set_title("exposure at the close ($m)"); ax[0, 1].legend(fontsize=7)
    ax[1, 0].plot(days["date"], days["var_param"] / 1e6, label="parametric 99%", color=C["a"]); ax[1, 0].plot(days["date"], days["var_hist"] / 1e6, label="historical 99%", color=C["b"])
    ax[1, 0].plot(days["date"], -days["pnl"] / 1e6, ".", color=C["grey"], ms=4, label="loss on the day"); ax[1, 0].set_title("one-day VaR against the realised daily loss ($m)"); ax[1, 0].legend(fontsize=7)
    ex = con.execute("select date, name, value from exposures where bucket = 'sector_net' and time = (select max(time) from exposures e2 where e2.date = exposures.date and e2.bucket = 'sector_net')").df(); ex["date"] = pd.to_datetime(ex["date"])
    piv = ex.pivot_table(index="date", columns="name", values="value").fillna(0) / 1e6
    piv.plot(ax=ax[1, 1], lw=0.9, legend=False); ax[1, 1].axhline(0, color=C["grey"], lw=0.5); ax[1, 1].set_title("net sector exposure at the close ($m), equities and bonds"); ax[1, 1].legend(fontsize=5, ncol=3, loc="lower left")
    save(fig, "book.png")


# ---- monitor ----------------------------------------------------------------------------------------------------------------
def monitor():
    m = run["monitor"]; by = m["by_type"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
    types = sorted(by, key=lambda k: -by[k]["injected"]); inj = [by[k]["injected"] for k in types]; det = [by[k]["detected"] for k in types]
    y = np.arange(len(types)); ax[0, 0].barh(y, inj, color="#ddd", label="injected"); ax[0, 0].barh(y, det, color=C["fault"], label="detected"); ax[0, 0].set_yticks(y); ax[0, 0].set_yticklabels(types, fontsize=7); ax[0, 0].invert_yaxis(); ax[0, 0].legend(fontsize=7); ax[0, 0].set_title("faults injected and caught, by type")
    f = con.execute("select type, detected_time - time ttd from faults where detected_time is not null").df()
    med = f.groupby("type")["ttd"].median().reindex(types) / 60; p90 = f.groupby("type")["ttd"].quantile(0.9).reindex(types) / 60
    ax[0, 1].barh(y, p90.fillna(0), color="#ddd", label="p90"); ax[0, 1].barh(y, med.fillna(0), color=C["clean"], label="median"); ax[0, 1].set_yticks(y); ax[0, 1].set_yticklabels(types, fontsize=7); ax[0, 1].invert_yaxis(); ax[0, 1].set_title("time to detect (minutes)"); ax[0, 1].legend(fontsize=7)
    cols = [C["fault"] if fd else C["clean"] for fd in days["fault_day"]]
    ax[1, 0].bar(days["date"], days["incidents"], color=cols, width=0.8); ax[1, 0].set_title("alert incidents per day (red: fault days, blue: clean days)")
    a = con.execute("select a.rule, count(*) n from alerts a join (select date, fault_day from (values " + ",".join(f"('{d}', {int(fd)})" for d, fd in zip(days['date'].dt.date, days['fault_day'])) + ") t(date, fault_day)) t on a.date = cast(t.date as date) where t.fault_day = 0 group by 1 order by 2 desc").df()
    ax[1, 1].barh(np.arange(len(a)), a["n"], color=C["clean"]); ax[1, 1].set_yticks(np.arange(len(a))); ax[1, 1].set_yticklabels(a["rule"], fontsize=7); ax[1, 1].invert_yaxis(); ax[1, 1].set_title(f"alerts on the {m['clean_days']} clean days, by rule")
    save(fig, "monitor.png")


# ---- one fault day --------------------------------------------------------------------------------------------------------------
def day():
    fd = days[days["fault_day"]]; d = fd.iloc[len(fd) // 2]["date"].date()
    nav = con.execute("select time, value from exposures where date = ? and bucket = 'nav' order by time", [d]).df()
    al = con.execute("select time, rule, scope from alerts where date = ? order by time", [d]).df(); fl = con.execute("select time, type, scope, detected_time from faults where date = ? order by time", [d]).df()
    from zoneinfo import ZoneInfo
    NY = ZoneInfo("America/New_York")
    t0 = nav["time"].min()
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), gridspec_kw={"height_ratios": [2, 3]})
    ax[0].plot((nav["time"] - t0) / 3600, (nav["value"] - nav["value"].iloc[0]) / 1e6, color=C["clean"]); ax[0].axhline(0, color=C["grey"], lw=0.5)
    ax[0].set_title(f"{d}: P&L since the open ($m, 30-minute samples), faults (x) and alerts (|)"); ax[0].set_ylabel("$m")
    for r in fl.itertuples():
        ax[0].axvline((r.time - t0) / 3600, color=C["fault"], lw=0.6, alpha=0.6)
    for r in al.itertuples():
        ax[0].axvline((r.time - t0) / 3600, color=C["a"], lw=0.4, alpha=0.5, ymin=0.9)
    rules = sorted(set(al["rule"]) | set(fl["type"])); ymap = {r: i for i, r in enumerate(rules)}
    for r in fl.itertuples():
        ax[1].plot((r.time - t0) / 3600, ymap[r.type], "x", color=C["fault"], ms=7)
        if pd.notna(r.detected_time):
            ax[1].plot([(r.time - t0) / 3600, (r.detected_time - t0) / 3600], [ymap[r.type]] * 2, color=C["fault"], lw=1)
    for r in al.itertuples():
        ax[1].plot((r.time - t0) / 3600, ymap[r.rule], "|", color=C["a"], ms=9)
    ax[1].set_yticks(range(len(rules))); ax[1].set_yticklabels(rules, fontsize=7); ax[1].set_xlabel("hours after 03:00 New York (the London open); US session 6.5 h to 13 h")
    ax[1].set_title("faults by type (x, with the line to their detection) and alerts by rule (|)")
    save(fig, "day.png")


# ---- P&L ------------------------------------------------------------------------------------------------------------------------
def pnl():
    p = con.execute("select date, strategy, asset_class, component, value from pnl").df(); p["date"] = pd.to_datetime(p["date"])
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
    comp = p.assign(k=p["asset_class"] + ":" + p["component"]).pivot_table(index="date", columns="k", values="value", aggfunc="sum").fillna(0) / 1e6
    order = comp.abs().sum().sort_values(ascending=False).index[:12]; small = comp.drop(columns=order).sum(axis=1)
    comp2 = comp[order].copy(); comp2["other"] = small
    comp2.plot(kind="bar", stacked=True, ax=ax[0, 0], width=0.9, legend=False); ax[0, 0].set_xticks(range(0, len(comp2), 10)); ax[0, 0].set_xticklabels([str(x.date()) for x in comp2.index[::10]], rotation=0, fontsize=6); ax[0, 0].set_title("daily P&L by component ($m)"); ax[0, 0].legend(fontsize=5, ncol=2)
    strat = p.pivot_table(index="date", columns="strategy", values="value", aggfunc="sum").fillna(0).cumsum() / 1e6
    strat.plot(ax=ax[0, 1]); ax[0, 1].axhline(0, color=C["grey"], lw=0.5); ax[0, 1].set_title("cumulative P&L by strategy ($m)"); ax[0, 1].legend(fontsize=6, ncol=2)
    eq = p[p["asset_class"] == "equity"].pivot_table(index="date", columns="component", values="value", aggfunc="sum").fillna(0).cumsum() / 1e6
    eq.plot(ax=ax[1, 0]); ax[1, 0].axhline(0, color=C["grey"], lw=0.5); ax[1, 0].set_title("equities: cumulative attribution ($m)"); ax[1, 0].legend(fontsize=6)
    ax[1, 1].bar(days["date"], days["identity_gap"], color=C["clean"]); ax[1, 1].set_title("NAV change minus attributed P&L, per day ($)"); ax[1, 1].set_ylim(-1, 1)
    save(fig, "pnl.png")


# ---- margin -----------------------------------------------------------------------------------------------------------------------
def margin():
    m = con.execute("select date, account, component, value from margin").df(); m["date"] = pd.to_datetime(m["date"])
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
    pb = m[(m["account"] == "PB") & (m["component"].isin(["equity", "bonds", "futures_initial", "fx_variation"]))].pivot_table(index="date", columns="component", values="value").fillna(0) / 1e6
    pb.plot(kind="area", ax=ax[0, 0], stacked=True, alpha=0.8, lw=0); coll = m[(m["account"] == "PB") & (m["component"] == "collateral_cash")].set_index("date")["value"] / 1e6
    ax[0, 0].plot(coll.index, coll.values, color="k", lw=1.2, label="collateral (cash)"); ax[0, 0].set_title("margin requirement by component against collateral ($m)"); ax[0, 0].legend(fontsize=6)
    cme = m[(m["account"] == "CME") & (m["component"].isin(["scan", "spread_charge", "delivery_charge", "inter_commodity_credit", "sum_of_outrights", "initial"]))].pivot_table(index="date", columns="component", values="value").fillna(0) / 1e6
    ax[0, 1].plot(cme.index, cme["sum_of_outrights"], color=C["grey"], label="sum of outright scan charges"); ax[0, 1].plot(cme.index, cme["initial"], color=C["a"], label="SPAN-style initial"); ax[0, 1].plot(cme.index, cme["inter_commodity_credit"], color=C["b"], label="inter-commodity credit"); ax[0, 1].plot(cme.index, cme["spread_charge"] + cme["delivery_charge"], color=C["c"], label="spread + delivery charges")
    ax[0, 1].set_title("futures margin ($m)"); ax[0, 1].legend(fontsize=6)
    sr = run["margin"]["scan_ranges"]; roots = list(sr); ax[1, 0].bar(roots, [sr[r]["scan"] for r in roots], color=C["a"], label="price scan range"); ax[1, 0].bar(roots, [sr[r]["spread"] for r in roots], color=C["b"], label="calendar spread charge")
    ax[1, 0].set_yscale("log"); ax[1, 0].set_title("scan ranges calibrated on the data ($ per contract, 99th pct of two-day moves)"); ax[1, 0].legend(fontsize=6)
    fin = m[(m["account"] == "financing")].pivot_table(index="date", columns="component", values="value").fillna(0).cumsum() / 1e6
    fin.plot(ax=ax[1, 1]); ax[1, 1].set_title("cumulative financing ($m): cash interest, stock borrow"); ax[1, 1].legend(fontsize=6)
    save(fig, "margin.png")


# ---- futures ------------------------------------------------------------------------------------------------------------------------
def futures():
    rl = con.execute("select * from rolls order by date").df(); rl["date"] = pd.to_datetime(rl["date"])
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    if len(rl):
        rl["cpc"] = rl["cost_usd"] / rl["lots"].abs(); lab = rl["from_contract"] + "->" + rl["to_contract"]
        cols = [C["a"] if o else C["grey"] for o in rl["observed"]]
        ax[0].barh(np.arange(len(rl)), rl["cpc"], color=cols); ax[0].set_yticks(np.arange(len(rl))); ax[0].set_yticklabels(lab, fontsize=6); ax[0].invert_yaxis(); ax[0].set_title("roll cost per contract ($; grey: spread not observable)"); ax[0].axvline(0, color="k", lw=0.5)
        ax[1].barh(np.arange(len(rl)), rl["spread_ticks"], color=cols); ax[1].set_yticks(np.arange(len(rl))); ax[1].set_yticklabels(lab, fontsize=6); ax[1].invert_yaxis(); ax[1].set_title("calendar spread paid on the roll day (ticks, next minus front)"); ax[1].axvline(0, color="k", lw=0.5)
    chk = run["futures_continuous_check"]; roots = list(chk)
    ax[2].bar(roots, [chk[r]["days_both"] for r in roots], color="#ddd", label="days both exist"); ax[2].bar(roots, [chk[r]["equal"] for r in roots], color=C["a"], label="continuous = listed front")
    ax[2].set_title("Yahoo's =F against the listed front month"); ax[2].legend(fontsize=6); ax[2].tick_params(axis="x", labelsize=7)
    save(fig, "futures.png")


# ---- FX --------------------------------------------------------------------------------------------------------------------------------
def fx():
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    cur = con.execute("select date, name, value from exposures where bucket = 'currency_net' and time = (select max(time) from exposures e2 where e2.date = exposures.date and e2.bucket = 'currency_net')").df(); cur["date"] = pd.to_datetime(cur["date"])
    piv = cur.pivot_table(index="date", columns="name", values="value").fillna(0) / 1e6; piv.plot(ax=ax[0]); ax[0].axhline(0, color=C["grey"], lw=0.5); ax[0].set_title("net currency exposure after the forward hedge ($m)")
    n = run["fx"]; ax[1].bar(["gross", "bilateral", "multilateral (CLS-style)"], [n["gross_usd"] / 1e6, n["bilateral_usd"] / 1e6, n["multilateral_usd"] / 1e6], color=[C["grey"], C["b"], C["a"]]); ax[1].set_title("settlement amounts over the run ($m)")
    s = con.execute("select value_date, ccy, sum(abs(amount)) gross, abs(sum(amount)) net from fx_settlements group by 1,2 order by 1").df()
    if len(s):
        s["value_date"] = pd.to_datetime(s["value_date"]); w = s.pivot_table(index="value_date", columns="ccy", values="gross").fillna(0) / 1e6
        w.plot(kind="bar", stacked=True, ax=ax[2], width=0.8); ax[2].set_xticklabels([str(x.date()) for x in w.index], rotation=45, fontsize=6); ax[2].set_title("settlement instructions by value date and currency (gross, millions of the currency)")
    save(fig, "fx.png")


# ---- credit ----------------------------------------------------------------------------------------------------------------------------
def credit():
    q = con.execute("select r.*, i.sector from rfq_trades r left join instruments i on i.instrument_id = r.cusip").df()
    b = pd.read_parquet(os.path.join(ROOT, "data", "derived", "bonds.parquet"))[["cusip", "fund"]].drop_duplicates("cusip"); q = q.merge(b, on="cusip", how="left")
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
    for fund, col in (("LQD", C["a"]), ("HYG", C["b"])):
        g = q[q["fund"] == fund]
        ax[0].hist(g["markup_pts"], bins=25, alpha=0.6, color=col, label=f"{fund} ({'IG' if fund == 'LQD' else 'HY'})"); ax[1].hist(g["markup_bp_yield"], bins=25, alpha=0.6, color=col, label=fund)
    ax[0].set_title("markup against the evaluated mark (points)"); ax[0].legend(fontsize=6); ax[1].set_title("markup in yield (bp)"); ax[1].legend(fontsize=6)
    ax[2].scatter(q["cover"], q["markup_pts"], s=8, c=[C["a"] if f == "LQD" else C["b"] for f in q["fund"]]); ax[2].set_xlabel("cover (points to the second-best quote)"); ax[2].set_ylabel("markup (points)"); ax[2].set_title("winner's markup against cover")
    ok = q[q["trace_prints"] > 0]; ax[3].hist(ok["trace_dev_pts"].dropna(), bins=25, color=C["c"]); ax[3].set_title(f"fill against the TRACE VWAP of +-60 min ({len(ok)} of {len(q)} with prints)")
    save(fig, "credit.png")


# ---- checks ------------------------------------------------------------------------------------------------------------------------------
def checks():
    c = con.execute("select phase, check_name, status, count(*) n from checks group by 1,2,3").df()
    piv = c.pivot_table(index=["phase", "check_name"], columns="status", values="n", aggfunc="sum").fillna(0)
    fig, ax = plt.subplots(figsize=(11, 4)); piv.reindex(columns=["ok", "warn", "fail"]).fillna(0).plot(kind="barh", stacked=True, ax=ax, color=[C["a"], C["f"], C["fault"]]); ax.set_title("start-of-day and end-of-day check outcomes over the run (days)"); ax.set_ylabel("")
    save(fig, "checks.png")


if __name__ == "__main__":
    for fn in (book, monitor, day, pnl, margin, futures, fx, credit, checks):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print("figure failed:", fn.__name__, repr(e))
