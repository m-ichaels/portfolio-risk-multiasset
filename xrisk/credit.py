"""The corporate-bond book's trading and post-trade.  Each day the CREDIT desk sends a few RFQs (buy or sell a bond
of the book, 0.5-5 million par) to four dealers; the best quote wins.  Post-trade, every fill is checked against
(a) the evaluated mark the desk priced from (iShares' end-of-day price for the CUSIP, real for the anchor date,
modelled on the other days as the pnl module says), (b) TRACE prints of the same CUSIP inside +-60 minutes
(simulated around the mark with TRACE's dissemination caps - IG prints above 5 million show as 5MM+, high yield
above 1 million as 1MM+ - unless a real download through the FINRA API is present), and (c) the 15-minute TRACE
reporting rule on our own report.  Markups are in price points and in yield basis points; cover is the distance to
the second-best quote."""
from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

from . import book as B, data

HALF_SPREAD = {"IG": 0.20, "HY": 0.50}          # dealer half spread in points, base level (simulated)
PRINT_SIGMA = {"IG": 0.25, "HY": 0.60}          # dispersion of TRACE prints around the mark (simulated)
PRINTS_PER_DAY = {"IG": 6.0, "HY": 4.0}
CAP = {"IG": 5e6, "HY": 1e6}
OFF_MARKET_PTS = {"IG": 1.0, "HY": 2.0}
REPORT_LIMIT_S = 900.0


def yield_bp(inst: B.Instrument, dpx: float, px: float) -> float:
    """price points to yield basis points through modified duration: dy = -dP / (D P/100)"""
    return -1e4 * dpx / max(inst.duration * px, 1e-6)


def load_real_prints() -> pd.DataFrame | None:
    p = os.path.join(data.DER, "trace_prints.parquet")
    if not os.path.exists(p):
        return None
    t = pd.read_parquet(p)
    cols = {c.lower(): c for c in t.columns}
    need = {"cusip": "cusip", "tradereportdate": "date", "tradeexecutiontime": "time", "lastsaleprice": "price", "quantity": "size"}
    if not all(k in cols for k in need):
        return None
    out = t.rename(columns={cols[k]: v for k, v in need.items()})[list(need.values())]
    out["date"] = pd.to_datetime(out["date"]).dt.date; return out


def simulate_prints(insts: dict, marks: dict, d: dt.date, t_open: float, t_close: float, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for iid, inst in insts.items():
        if inst.asset_class != B.BOND or iid not in marks:
            continue
        b = inst.rating_bucket; n = int(rng.poisson(PRINTS_PER_DAY[b]))
        for _ in range(n):
            side = 1 if rng.random() < 0.5 else -1
            size = float(np.exp(rng.normal(np.log(8e5 if b == "IG" else 4e5), 1.0)))
            px = marks[iid] + side * HALF_SPREAD[b] * float(rng.uniform(0.3, 1.0)) + float(rng.normal(0, PRINT_SIGMA[b]))
            rows.append({"cusip": iid, "date": d, "time": float(rng.uniform(t_open, t_close)), "price": px, "size": min(size, CAP[b]), "capped": size >= CAP[b], "side": "B" if side > 0 else "S"})
    return pd.DataFrame(rows, columns=["cusip", "date", "time", "price", "size", "capped", "side"])


def rfq(inst: B.Instrument, side: int, par: float, mark: float, t: float, dealers: tuple, rng: np.random.Generator) -> dict:
    """side +1 = we buy; dealers quote the mark plus a half spread that widens with size and differs by dealer"""
    b = inst.rating_bucket; size_mult = 1.0 + 0.25 * par / (5e6 if b == "IG" else 2e6)
    quotes = []
    for k, dl in enumerate(dealers):
        skill = 0.8 + 0.15 * k
        q = mark + side * HALF_SPREAD[b] * size_mult * skill * float(rng.uniform(0.6, 1.4)) + float(rng.normal(0, 0.05))
        quotes.append((dl, q))
    ranked = sorted(quotes, key=lambda x: side * x[1]); winner, px = ranked[0]; cover = abs(ranked[1][1] - px) if len(ranked) > 1 else 0.0
    return {"time": t, "cusip": inst.instrument_id, "side": "B" if side > 0 else "S", "par": par, "price": px, "mark": mark, "winner": winner, "cover": cover, "n_quotes": len(quotes), "quotes": quotes}


def post_trade(trade: dict, inst: B.Instrument, prints: pd.DataFrame, report_delay_s: float, window_s: float = 3600.0) -> dict:
    side = 1 if trade["side"] == "B" else -1; b = inst.rating_bucket
    markup = side * (trade["price"] - trade["mark"])                    # positive = we paid above the mark (bought) or sold below it
    p = prints[(prints["cusip"] == inst.instrument_id) & ((prints["time"] - trade["time"]).abs() <= window_s)] if len(prints) else prints
    flags = []
    if len(p):
        vwap = float((p["price"] * p["size"]).sum() / p["size"].sum()); dev = side * (trade["price"] - vwap)
        if abs(dev) > OFF_MARKET_PTS[b]:
            flags.append("off_market_vs_trace")
    else:
        vwap, dev = float("nan"), float("nan"); flags.append("no_trace_prints")
    if abs(markup) > OFF_MARKET_PTS[b]:
        flags.append("off_market_vs_mark")
    if report_delay_s > REPORT_LIMIT_S:
        flags.append("late_trace_report")
    bp = 1e4 * markup / max(inst.duration * trade["price"], 1e-6)              # dy = dP / (D P): the yield given up in bp, positive = cost
    return dict(trade, markup_pts=markup, markup_bp_yield=bp, trace_vwap=vwap, trace_prints=int(len(p)), trace_dev_pts=dev, report_delay_s=report_delay_s, flags=",".join(flags), capped_print=("capped" in p.columns and bool(p["capped"].any())) if len(p) else False)
