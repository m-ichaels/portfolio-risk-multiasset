"""The limit set and its evaluation.  Every limit is a number in one dataclass (dollars, or a fraction of NAV where
the limit scales with the fund); utilisation above `warn_at` is a warning, above 1 a breach.  The evaluation runs on
an Exposures object, so it is the same at start of day, intraday and at the close."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from .exposures import Exposures


@dataclass
class LimitSet:
    gross_nav: float = 3.5              # gross exposure / NAV (the book runs at about 2.5)
    net_equity_nav: float = 0.50        # net equity (cash equities) / NAV
    beta_dollars_nav: float = 0.35      # |beta-dollars after the futures hedge| / NAV (the hedge covers 40 % of the equity beta)
    sector_net_nav: float = 0.20        # |net per sector| / NAV
    single_name_nav: float = 0.05       # |gross per name| / NAV (momentum's European names reach 2.6 %)
    currency_net_nav: float = 0.05      # |net exposure per non-USD currency after hedges| / NAV
    dv01_usd: float = 40_000.0          # |total DV01| in dollars per bp
    dv01_strategy_usd: float = 25_000.0
    cs01_usd: float = 200_000.0         # |CS01| per bucket
    var_nav: float = 0.02               # parametric one-day 99 % VaR / NAV
    futures_root_nav: float = 0.30      # |notional per root| / NAV
    margin_utilisation: float = 0.80    # requirement / collateral
    days_to_liquidate: float = 10.0     # worst equity name at 20 % of ADV
    intraday_loss_nav: float = 0.015    # loss since the open / NAV
    warn_at: float = 0.80


def evaluate(ls: LimitSet, ex: Exposures, margin_util: float | None = None, intraday_pnl: float | None = None) -> list[dict]:
    nav = max(ex.nav, 1.0); out = []

    def add(name, scope, value, threshold):
        u = abs(value) / threshold if threshold > 0 else 0.0
        out.append({"date": ex.date, "time": ex.time, "limit_name": name, "scope": scope, "value": float(value), "threshold": float(threshold), "utilisation": float(u), "status": "breach" if u >= 1.0 else "warn" if u >= ls.warn_at else "ok"})

    add("gross", "all", ex.gross, ls.gross_nav * nav)
    add("net_equity", "all", ex.by_class.get("equity", {}).get("net", 0.0), ls.net_equity_nav * nav)
    add("beta_dollars", "all", ex.beta_dollars, ls.beta_dollars_nav * nav)
    for k, v in ex.by_sector.items():
        add("sector_net", k, v, ls.sector_net_nav * nav)
    for k, v in ex.top_names:
        add("single_name", k, v, ls.single_name_nav * nav)
    for k, v in ex.by_currency.items():
        add("currency_net", k, v, ls.currency_net_nav * nav)
    add("dv01", "all", ex.dv01, ls.dv01_usd)
    for k, v in ex.dv01_by_strategy.items():
        add("dv01_strategy", k, v, ls.dv01_strategy_usd)
    for k, v in ex.cs01.items():
        add("cs01", k, v, ls.cs01_usd)
    add("var", "param99", ex.var_param, ls.var_nav * nav)
    for k, v in ex.futures_by_root.items():
        add("futures_root", k, v, ls.futures_root_nav * nav)
    if margin_util is not None:
        add("margin", "all", margin_util, ls.margin_utilisation)
    add("liquidity", "days_to_liquidate", ex.days_to_liquidate, ls.days_to_liquidate)
    if intraday_pnl is not None:
        add("intraday_loss", "all", max(-intraday_pnl, 0.0), ls.intraday_loss_nav * nav)
    return out


def worst(rows: list[dict]) -> dict:
    return {"breach": [r for r in rows if r["status"] == "breach"], "warn": [r for r in rows if r["status"] == "warn"]}
