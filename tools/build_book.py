#!/usr/bin/env python3
"""The equity book comes from execution-ops (Project E): its three strategies (12-1 momentum, 5-day reversal,
inverse-volatility) on its 190-name universe and $2bn.  This runs them day by day from the sibling checkout
../ProjectE (its `xops` package and committed prices) and writes what the risk layer needs:

  data/derived/equity_positions.parquet   date, strategy, symbol, target shares (start-of-day targets, the book the
                                          strategies want to hold; the engine executes the day's delta intraday).
                                          Two deliberate differences from execution-ops' own sizing: shares are
                                          the dollar weight over the price in dollars (London pence through
                                          price_scale, then the ECB fix), and each strategy's weights are
                                          normalised to its stated gross (momentum 2, reversal 1, low-vol 1)
                                          rather than gross 2 per region
  data/derived/equity_prices.parquet      daily bars for the universe over the window plus the risk model's lookback

Usage: python tools/build_book.py [--from 2026-06-01] [--to 2026-09-17] [--lookback-days 400]"""
from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIBLING = os.path.join(os.path.dirname(ROOT), "ProjectE")
sys.path.insert(0, SIBLING)
from xops import calendar as cal, data as xdata, strategies as S  # noqa: E402


GROSS = {"MOM": 2.0, "REV": 1.0, "LOWVOL": 1.0}     # gross exposure per unit of strategy capital


def arg(name, default):
    return next((a.split("=", 1)[1] if "=" in a else sys.argv[sys.argv.index(a) + 1] for a in sys.argv if a.startswith(name)), default)


def main():
    start = dt.date.fromisoformat(arg("--from", "2026-06-01")); end = dt.date.fromisoformat(arg("--to", "2026-09-17")); lookback = int(arg("--lookback-days", "400"))
    uni = xdata.load_universe(); prices = xdata.load_prices(start=(start - dt.timedelta(days=lookback)).isoformat(), end=end.isoformat())
    adj = S.panel(prices, "adjclose").ffill(); close = S.panel(prices, "close").ffill(); region = uni.set_index("symbol")["region"]   # a closed exchange carries its last close, so a strategy never drops a region on the other region's holiday
    days = sorted({d for ex in ("XNYS", "XLON", "XETR") for d in cal.trading_days(ex, start, end)})
    scale = uni.set_index("symbol")["price_scale"]; aum = 2e9
    fxr = pd.read_parquet(os.path.join(ROOT, "data", "derived", "fx_rates.parquet")); fxr = fxr[fxr["source"] == "ecb"].pivot(index="date", columns="pair", values="rate").sort_index(); fxr.index = pd.to_datetime(fxr.index)
    ccy = uni.set_index("symbol")["currency"].str.upper()
    prev = {}; rows = []
    for d in days:
        ts = pd.Timestamp(d); prev_day = adj.loc[:ts - pd.Timedelta(days=1)]
        if prev_day.empty:
            continue
        dprev = prev_day.index[-1]; fx_d = fxr.loc[:dprev].iloc[-1]
        usd_per = ccy.reindex(close.columns).map({"USD": 1.0, "EUR": float(fx_d["EURUSD"]), "GBP": float(fx_d["GBPUSD"])}).fillna(1.0)
        px = close.loc[dprev] * scale.reindex(close.columns).fillna(1.0) * usd_per          # the local price in dollars, so a weight is a dollar amount
        tg = {}
        for name, spec in S.STRATEGIES.items():
            if not S.is_rebalance_day(spec["rebalance"], d) and prev and name in prev:
                tg[name] = prev[name]; continue
            w = spec["fn"](adj, dprev, region); cap = aum * spec["capital_share"]
            if len(w) and w.abs().sum() > 0:
                w = w * (GROSS[name] / w.abs().sum())          # execution-ops' momentum weights sum to gross 2 per region; here each strategy runs at its stated gross
            tg[name] = (w * cap / px.reindex(w.index)).replace([np.inf, -np.inf], np.nan).dropna().round()
        prev = tg
        for strat, s in tg.items():
            for sym, q in s.items():
                if abs(q) >= 1:
                    rows.append((d, strat, sym, float(q)))
    pos = pd.DataFrame(rows, columns=["date", "strategy", "symbol", "target"])
    os.makedirs(os.path.join(ROOT, "data", "derived"), exist_ok=True)
    pos.to_parquet(os.path.join(ROOT, "data", "derived", "equity_positions.parquet"), index=False)
    p = prices.copy(); p["date"] = pd.to_datetime(p["date"]).dt.date
    p.to_parquet(os.path.join(ROOT, "data", "derived", "equity_prices.parquet"), index=False)
    uni.drop(columns=["lot_size"], errors="ignore").to_csv(os.path.join(ROOT, "data", "universe.csv"), index=False)
    g = pos.groupby("date")["symbol"].nunique()
    print(f"equity book: {len(days)} days, {g.mean():.0f} names a day, {pos['target'].abs().mul(pos.merge(p[['symbol', 'date', 'close']], on=['symbol', 'date'], how='left')['close']).sum() / len(days) / 1e9:.2f}bn gross a day (local prices); prices {len(p)} rows")


if __name__ == "__main__":
    main()
