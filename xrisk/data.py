"""Loaders for the committed data (equity book and prices from execution-ops, futures settlements, FX, rates, bonds,
sectors) and the DuckDB store the pipeline writes: positions, exposures, limits, P&L attribution, margin, alerts,
faults, rolls, FX settlements, RFQ post-trade, checks."""
from __future__ import annotations

import datetime as dt
import os

import duckdb
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DER = os.path.join(ROOT, "data", "derived")
REF = os.path.join(ROOT, "data", "reference")
RESULTS = os.path.join(ROOT, "results")
DB_PATH = os.path.join(DER, "xrisk.duckdb")


def _pq(name: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(DER, name))


def load_universe() -> pd.DataFrame:
    u = pd.read_csv(os.path.join(ROOT, "data", "universe.csv"))
    s = pd.read_csv(os.path.join(REF, "sectors.csv"))
    u = u.merge(s, on="symbol", how="left").fillna({"sector": "Unknown", "industry": ""})
    u.loc[u["currency"] == "GBp", "region"] = "UK"           # execution-ops files London under EU; the risk model wants its own market factor
    u["ccy"] = u["currency"].str.upper().replace({"GBP": "GBP"})   # GBp (pence) -> GBP, with price_scale 0.01 doing the unit
    return u


def load_equity_prices() -> pd.DataFrame:
    p = _pq("equity_prices.parquet"); p["date"] = pd.to_datetime(p["date"]).dt.date; return p


def load_equity_targets() -> pd.DataFrame:
    p = _pq("equity_positions.parquet"); p["date"] = pd.to_datetime(p["date"]).dt.date; return p


def load_futures() -> pd.DataFrame:
    f = _pq("futures_settles.parquet"); f["date"] = pd.to_datetime(f["date"]).dt.date; return f


def load_fx(source: str = "ecb") -> pd.DataFrame:
    """wide: date x pair (EURUSD, GBPUSD, USDJPY)"""
    f = _pq("fx_rates.parquet"); f = f[f["source"] == source]; f["date"] = pd.to_datetime(f["date"]).dt.date
    return f.pivot(index="date", columns="pair", values="rate").sort_index()


def load_rates() -> pd.DataFrame:
    """wide: date x name (SOFR, ESTR, SONIA, JGB1Y, UST1M..UST30Y), forward-filled across the union of dates"""
    r = _pq("rates.parquet"); r["date"] = pd.to_datetime(r["date"]).dt.date
    return r.pivot(index="date", columns="name", values="value").sort_index().ffill()


def load_bonds() -> pd.DataFrame:
    b = _pq("bonds.parquet"); b["maturity"] = pd.to_datetime(b["maturity"]).dt.date; return b


def load_etf_prices() -> pd.DataFrame:
    e = _pq("etf_prices.parquet"); e["date"] = pd.to_datetime(e["date"]).dt.date
    return e.pivot(index="date", columns="symbol", values="close").sort_index()


def load_futures_specs() -> pd.DataFrame:
    return pd.read_csv(os.path.join(REF, "futures_specs.csv")).set_index("root")


SCHEMA = """
create table if not exists instruments (instrument_id varchar primary key, asset_class varchar, symbol varchar, currency varchar, multiplier double, exchange varchar, sector varchar, region varchar, price_scale double);
create table if not exists positions (date date, source varchar, phase varchar, strategy varchar, instrument_id varchar, qty double, price double, fx double, notional_usd double);
create table if not exists exposures (date date, time double, bucket varchar, name varchar, value double);
create table if not exists limits (date date, time double, limit_name varchar, scope varchar, value double, threshold double, utilisation double, status varchar);
create table if not exists pnl (date date, strategy varchar, asset_class varchar, component varchar, value double);
create table if not exists margin (date date, account varchar, component varchar, value double);
create table if not exists alerts (date date, time double, rule varchar, severity varchar, scope varchar, detail varchar);
create table if not exists faults (date date, time double, type varchar, scope varchar, detected_time double, detected_rule varchar);
create table if not exists rolls (date date, root varchar, from_contract varchar, to_contract varchar, lots double, front_px double, next_px double, spread_ticks double, cost_usd double, observed boolean);
create table if not exists fx_settlements (value_date date, trade_date date, pair varchar, counterparty varchar, ccy varchar, amount double, netted_amount double, confirmed boolean);
create table if not exists rfq_trades (date date, time double, cusip varchar, side varchar, par double, price double, mark double, winner varchar, cover double, n_quotes integer, markup_pts double, markup_bp_yield double, trace_vwap double, trace_prints integer, trace_dev_pts double, report_delay_s double, flags varchar);
create table if not exists checks (date date, phase varchar, check_name varchar, status varchar, detail varchar);
"""


def connect(path: str = DB_PATH, fresh: bool = False) -> duckdb.DuckDBPyConnection:
    if fresh and os.path.exists(path):
        os.remove(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = duckdb.connect(path)
    for stmt in SCHEMA.strip().split(";"):
        if stmt.strip():
            con.execute(stmt)
    return con


def to_json(obj, path: str):
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (dt.date, dt.datetime)):
            return o.isoformat()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        if isinstance(o, float) and np.isnan(o):
            return None
        return str(o)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=default)
