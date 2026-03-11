"""The stated multi-asset book and its instruments.  Equities come from execution-ops' three strategies on $2bn
(MOM, REV, LOWVOL); on top of them the fund runs an index-futures hedge (HEDGE), a Treasury-futures steepener
(RATES), a commodity and yen macro book (MACRO), a corporate-bond book hedged with Treasury futures (CREDIT), and
an FX-forward hedge of the non-dollar equity exposure (FXHEDGE).  Every position is (strategy, instrument, qty);
marks are in local currency and converted at the FX rate.  Sizes are defined in risk terms where that is what a
desk would do (the steepener in DV01, the hedge in beta-dollars, the FX hedge in currency exposure)."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data
from .xops_bridge import fi

EQUITY, FUTURE, FX_FORWARD, BOND = "equity", "future", "fx_forward", "bond"
CCY_OF_PAIR = {"EURUSD": ("EUR", "USD"), "GBPUSD": ("GBP", "USD"), "USDJPY": ("USD", "JPY")}


@dataclass
class Instrument:
    instrument_id: str; asset_class: str; symbol: str; currency: str; multiplier: float = 1.0; exchange: str = ""; sector: str = ""; region: str = ""; price_scale: float = 1.0
    root: str = ""; contract: str = ""; tick: float = 0.0; group: str = ""; rate_tenor: str = ""
    pair: str = ""; value_date: dt.date | None = None; strike: float = 0.0; counterparty: str = ""; trade_date: dt.date | None = None
    cusip: str = ""; coupon: float = 0.0; maturity: dt.date | None = None; duration: float = 0.0; rating_bucket: str = ""; accrual_date: dt.date | None = None; fund: str = ""

    def as_row(self) -> tuple:
        return (self.instrument_id, self.asset_class, self.symbol, self.currency, self.multiplier, self.exchange, self.sector, self.region, self.price_scale)


@dataclass
class BookConfig:
    aum: float = 2e9
    hedge_ratio: float = 0.40                 # HEDGE: short index futures against this share of the equity book's beta-dollars
    hedge_split: dict = field(default_factory=lambda: {"ES": 0.7, "NQ": 0.3})
    steepener_dv01: float = 15_000.0          # RATES: long ZT / short ZN, each leg this DV01 in dollars per bp
    macro_lots: dict = field(default_factory=lambda: {"CL": 150, "GC": 60, "6J": 100})
    credit_par_ig: float = 150e6; credit_par_hy: float = 50e6; credit_n_ig: int = 40; credit_n_hy: int = 20
    credit_hedge: bool = True                 # CREDIT: short ZN against the bonds' DV01
    fx_hedge_ratio: float = 1.0               # FXHEDGE: sell EUR and GBP 1M forward against the non-USD equity value
    fx_tenor_months: int = 1
    fx_rebalance_usd: float = 20e6            # trade the hedge when it is off by more than this
    fx_counterparties: tuple = ("DLR1", "DLR2", "DLR3", "DLR4")
    roll_days_before: int = 2                 # roll futures this many CME days before first notice / last trade
    rfq_trades_per_day: int = 4
    seed: int = 11


def equity_instruments(uni: pd.DataFrame) -> dict[str, Instrument]:
    out = {}
    for r in uni.itertuples():
        out[r.symbol] = Instrument(r.symbol, EQUITY, r.symbol, r.ccy, 1.0, r.exchange, r.sector, r.region, float(r.price_scale))
    return out


def futures_instrument(contract: str, specs: pd.DataFrame) -> Instrument:
    root, y, m = fi.parse_contract(contract); s = specs.loc[root]
    return Instrument(contract, FUTURE, contract, s["currency"], float(s["multiplier"]), s["exchange"], s["group"], "US", 1.0, root=root, contract=contract, tick=float(s["tick_size"]), group=s["group"], rate_tenor=s["rate_tenor"] if isinstance(s["rate_tenor"], str) else "")


def forward_instrument(pair: str, value_date: dt.date, strike: float, counterparty: str, trade_date: dt.date) -> Instrument:
    base, quote = CCY_OF_PAIR[pair]
    iid = f"FWD:{pair}:{value_date.isoformat()}:{counterparty}:{trade_date.isoformat()}"
    return Instrument(iid, FX_FORWARD, pair, base, 1.0, "OTC", "fx", "G10", 1.0, pair=pair, value_date=value_date, strike=strike, counterparty=counterparty, trade_date=trade_date)


def bond_instruments(bonds: pd.DataFrame, cfg: BookConfig) -> tuple[dict[str, Instrument], dict[str, float]]:
    """the CREDIT book: the largest LQD and HYG holdings by market value, par allocated pro rata to the ETF's weights"""
    out, par = {}, {}
    for fund, total, n, bucket in (("LQD", cfg.credit_par_ig, cfg.credit_n_ig, "IG"), ("HYG", cfg.credit_par_hy, cfg.credit_n_hy, "HY")):
        b = bonds[(bonds["fund"] == fund) & (bonds["duration"] > 0) & (bonds["price"] > 20)].sort_values("market_value", ascending=False).drop_duplicates("cusip").head(n)
        w = b["market_value"] / b["market_value"].sum()
        for r, wi in zip(b.itertuples(), w):
            out[r.cusip] = Instrument(r.cusip, BOND, r.name, "USD", 1.0, "OTC", r.sector, r.country, 1.0, cusip=r.cusip, coupon=float(r.coupon), maturity=r.maturity, duration=float(r.mod_duration if pd.notna(r.mod_duration) else r.duration), rating_bucket=bucket, accrual_date=r.accrual_date, fund=fund)
            par[r.cusip] = float(np.round(total * wi / 1000) * 1000)
    return out, par


def bond_accrued(inst: Instrument, d: dt.date) -> float:
    """accrued interest in price points, 30/360 from the last coupon date (semi-annual coupons)"""
    if inst.maturity is None or inst.coupon <= 0:
        return 0.0
    # last coupon date: step back from maturity in six-month steps
    m = inst.maturity; last = m
    while last > d:
        mm = last.month - 6; yy = last.year
        if mm <= 0:
            mm += 12; yy -= 1
        last = dt.date(yy, mm, min(last.day, 28))
    days = 360 * (d.year - last.year) + 30 * (d.month - last.month) + (min(d.day, 30) - min(last.day, 30))
    return inst.coupon * days / 360.0


def month_add(d: dt.date, months: int) -> dt.date:
    m = d.month - 1 + months; y = d.year + m // 12; m = m % 12 + 1
    import calendar as _c
    return dt.date(y, m, min(d.day, _c.monthrange(y, m)[1]))
