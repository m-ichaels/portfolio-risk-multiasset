"""A simple multi-asset risk model estimated from the daily data, point in time.

Equities: two stages.  (1) A regional market factor (equal-weighted return of the universe's names in the region:
US, UK, EU) and each stock's 250-day beta to it.  (2) Every day the residuals are regressed cross-sectionally on
sector dummies and three style z-scores (size = log dollar ADV, momentum = 12-1 return, volatility = 60-day),
which gives sector and style factor returns; what is left is idiosyncratic.  Exposures are beta to the region
market, 1 to the sector, and the z-scores.  A separate 250-day beta to SPY gives beta-dollars for the hedge and the
limit.

Futures: own return factors for the equity-index, energy, metals and FX contracts (continuous series, roll days
winsorised); Treasury futures load on the matched par-yield change with a DV01 per contract *estimated* by
regressing the contract's daily dollar move on the yield change.  FX forwards load on spot.  Bonds load on the
nearest par-yield tenor and on a credit-spread factor per bucket (IG, HY) backed out of LQD / HYG returns with the
funds' own holdings-weighted duration, DV01 on both.

Covariance is the 250-day sample covariance of the factor returns; VaR is parametric (2.326 sigma) and historical
(the last 250 days' actual instrument returns applied to today's book, bonds through their factors)."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import book as B

REGIONS = ("US", "UK", "EU")
STYLES = ("SIZE", "MOM", "VOL")
FUT_FACTORS = ("ES", "NQ", "RTY", "CL", "GC", "6E", "6B", "6J")
RATE_TENORS = {"UST2Y": 2, "UST5Y": 5, "UST7Y": 7, "UST10Y": 10, "UST20Y": 20, "UST30Y": 30}
FX_PAIRS = ("EURUSD", "GBPUSD", "USDJPY")
CREDIT = {"IG": "LQD", "HY": "HYG"}


def _z(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan)
    z = (s - s.mean()) / (s.std() + 1e-12)
    return z.clip(-3, 3).fillna(0.0)


class RiskModel:
    def __init__(self, prices: pd.DataFrame, uni: pd.DataFrame, futures: pd.DataFrame, fx: pd.DataFrame, rates: pd.DataFrame, etf: pd.DataFrame, bonds: pd.DataFrame, specs: pd.DataFrame, window: int = 250):
        self.window = window; self.uni = uni.set_index("symbol"); self.specs = specs
        self.adj = prices.pivot(index="date", columns="symbol", values="adjclose").sort_index()
        self.close = prices.pivot(index="date", columns="symbol", values="close").sort_index()
        self.volume = prices.pivot(index="date", columns="symbol", values="volume").sort_index()
        self.ret = np.log(self.adj).diff()
        self.region = self.uni["region"].reindex(self.adj.columns).fillna("US"); self.sector = self.uni["sector"].reindex(self.adj.columns).fillna("Unknown")
        self.sectors = sorted(self.sector.unique())
        # market factors per region
        self.mkt = pd.DataFrame({f"MKT_{r}": self.ret[[c for c in self.ret.columns if self.region[c] == r]].mean(axis=1) for r in REGIONS}).fillna(0.0)   # a region's holiday is a zero market return
        # futures continuous returns (winsorised: roll days and gaps)
        cont = futures[futures["contract"].str.endswith("=F")].pivot(index="date", columns="root", values="close").sort_index()
        self.fut_px = cont
        fr = np.log(cont).diff()
        self.fut_ret = fr.clip(lower=fr.quantile(0.005), upper=fr.quantile(0.995), axis=1)
        # rates in bp changes, FX log returns, credit spreads from the ETFs
        self.rates = rates; self.rate_chg = rates[list(RATE_TENORS)].diff() * 100.0
        self.fx = fx; self.fx_ret = np.log(fx[list(FX_PAIRS)]).diff()
        self.etf = etf
        self.etf_dur = {b: float((bonds.loc[bonds["fund"] == f, "mod_duration"] * bonds.loc[bonds["fund"] == f, "market_value"]).sum() / bonds.loc[bonds["fund"] == f, "market_value"].sum()) for b, f in CREDIT.items()}
        self.etf_tenor = {b: self._nearest_tenor(self.etf_dur[b]) for b in CREDIT}
        cs = {}
        for b, f in CREDIT.items():
            r = np.log(etf[f]).diff(); dy = self.rate_chg[self.etf_tenor[b]].reindex(r.index)
            cs[f"CS_{b}"] = (-1e4 * r / self.etf_dur[b] - dy).clip(-60, 60)
        self.cs = pd.DataFrame(cs)
        self._beta_cache: dict = {}; self._style_cache: dict = {}; self._cs_rows: dict = {}; self._dv01_cache: dict = {}

    # ---- factor return history ----------------------------------------------------------------------------------------
    @staticmethod
    def _nearest_tenor(dur: float) -> str:
        return min(RATE_TENORS, key=lambda k: abs(RATE_TENORS[k] - dur))

    def bond_tenor(self, inst: B.Instrument, d: dt.date) -> str:
        yrs = max((inst.maturity - d).days / 365.25, 0.1) if inst.maturity else inst.duration
        return self._nearest_tenor(min(yrs, inst.duration * 1.6 if inst.duration else yrs))

    def factor_history(self, d: dt.date) -> pd.DataFrame:
        """all factor returns on days strictly before d, last `window` rows; style/sector factors from the daily
        cross-sectional regressions"""
        end = d - dt.timedelta(days=1)
        secstyle = self._sector_style_history(d)
        parts = [self.mkt.loc[:end], secstyle.loc[:end], self.fut_ret.loc[:end].rename(columns=lambda c: f"FUT_{c}"), self.rate_chg.loc[:end], self.fx_ret.loc[:end].rename(columns=lambda c: f"FX_{c}"), self.cs.loc[:end]]
        f = pd.concat(parts, axis=1).sort_index()
        f = f.loc[self.mkt.loc[:end].index]        # equity trading days as the grid
        return f.tail(self.window).fillna(0.0)

    def _sector_style_history(self, d: dt.date) -> pd.DataFrame:
        """one cross-section per day, cached by day (exposures as of the first date that needed the row)"""
        end = d - dt.timedelta(days=1); days = self.ret.loc[:end].index[-self.window:]
        missing = [t for t in days if t not in self._cs_rows]
        if missing:
            b = self.betas(d)
            for t in missing:
                self._cs_rows[t] = self.cross_section(t, self.ret.loc[t], betas=b)[0]
        return pd.DataFrame({t: self._cs_rows[t] for t in days}).T

    # ---- exposures per stock -------------------------------------------------------------------------------------------
    def betas(self, d: dt.date) -> pd.DataFrame:
        """250-day betas to the region market and to SPY, computed from data before d"""
        if d in self._beta_cache:
            return self._beta_cache[d]
        end = d - dt.timedelta(days=1); r = self.ret.loc[:end].tail(self.window)
        spy = np.log(self.etf["SPY"]).diff().reindex(r.index).fillna(0.0)
        out = pd.DataFrame(index=r.columns, columns=["beta_mkt", "beta_spy"], dtype=float)
        n_ok = r.notna().sum()
        for reg in REGIONS:
            cols = [c for c in r.columns if self.region[c] == reg]
            if not cols:
                continue
            x = self.mkt[f"MKT_{reg}"].reindex(r.index).fillna(0.0); rr = r[cols].fillna(0.0)
            xc = x - x.mean(); bm = ((rr - rr.mean()).mul(xc, axis=0)).mean() / (xc.var(ddof=0) + 1e-12)
            sc = spy - spy.mean(); bs = ((rr - rr.mean()).mul(sc, axis=0)).mean() / (sc.var(ddof=0) + 1e-12)
            out.loc[cols, "beta_mkt"] = bm.clip(0.2, 2.5).values; out.loc[cols, "beta_spy"] = bs.clip(-0.5, 3.0).values
        out.loc[n_ok < 60, ["beta_mkt", "beta_spy"]] = 1.0
        out = out.fillna(1.0)
        self._beta_cache = {d: out}
        return out

    def styles(self, t) -> pd.DataFrame:
        """style z-scores from data up to and including t (used as exposures for the following day)"""
        if t in self._style_cache:
            return self._style_cache[t]
        h = self.adj.loc[:t]; c = self.close.loc[:t]; v = self.volume.loc[:t]
        scale = self.uni["price_scale"].reindex(h.columns).fillna(1.0)
        size = np.log((c.tail(20) * v.tail(20)).mean() * scale + 1.0)
        mom = h.iloc[-22] / h.iloc[-252] - 1.0 if len(h) >= 252 else pd.Series(0.0, index=h.columns)
        vol = np.log(h.tail(61)).diff().std()
        df = pd.DataFrame({"SIZE": _z(size), "MOM": _z(mom), "VOL": _z(vol)})
        self._style_cache = {t: df}
        return df

    def exposure_matrix(self, t, symbols: list[str]) -> pd.DataFrame:
        """sector dummies and style z-scores for the cross-sectional regression on day t (styles from t-1)"""
        prev = self.adj.loc[:t].index[-2] if len(self.adj.loc[:t]) >= 2 else t
        st = self.styles(prev).reindex(symbols).fillna(0.0)
        X = pd.DataFrame(0.0, index=symbols, columns=[f"SEC_{s}" for s in self.sectors] + list(STYLES))
        for s in symbols:
            X.at[s, f"SEC_{self.sector.get(s, 'Unknown')}"] = 1.0
        X[list(STYLES)] = st[list(STYLES)].values
        return X

    def cross_section(self, t, r: pd.Series, betas: pd.DataFrame | None = None, market_ret: dict | None = None) -> tuple[pd.Series, pd.Series]:
        """regress residual returns (after the region market) on sectors and styles; returns (factor returns, idio).
        `market_ret` gives the region market returns over the same interval as r (the daily history's when absent)."""
        r = r.replace([np.inf, -np.inf], np.nan).dropna(); syms = list(r.index)
        cols = [f"SEC_{s}" for s in self.sectors] + list(STYLES)
        if len(syms) < 20:
            return pd.Series(0.0, index=cols), r * 0
        b = (betas if betas is not None else self.betas(t if isinstance(t, dt.date) else t.date()))["beta_mkt"].reindex(syms).fillna(1.0)
        if market_ret is None:
            market_ret = {reg: float(self.mkt.at[t, f"MKT_{reg}"]) if t in self.mkt.index else 0.0 for reg in REGIONS}
        mk = np.array([market_ret.get(self.region[s], 0.0) for s in syms])
        e = r.values - b.values * mk
        X = self.exposure_matrix(t, syms).fillna(0.0)
        coef, *_ = np.linalg.lstsq(X.values, e, rcond=None)
        coef = np.nan_to_num(coef)
        f = pd.Series(coef, index=X.columns); idio = pd.Series(e - X.values @ coef, index=syms)
        return f, idio

    def treasury_dv01(self, root: str, d: dt.date) -> float:
        """dollars per basis point per contract, estimated over the last 250 days before d from the continuous series
        against the matched par-yield change (negative slope -> positive DV01)"""
        key = (root, d)
        if key in self._dv01_cache:
            return self._dv01_cache[key]
        end = d - dt.timedelta(days=1); tenor = self.specs.loc[root, "rate_tenor"]; mult = float(self.specs.loc[root, "multiplier"])
        dp = self.fut_px[root].diff().loc[:end].tail(self.window) * mult; dy = self.rate_chg[tenor].reindex(dp.index)
        ok = dp.notna() & dy.notna() & (dy != 0)
        slope = float(np.cov(dp[ok], dy[ok])[0, 1] / (dy[ok].var() + 1e-12)) if ok.sum() > 40 else -mult * 0.07
        v = max(-slope, 1.0); self._dv01_cache[key] = v
        return v

    # ---- covariance -----------------------------------------------------------------------------------------------------
    def covariance(self, d: dt.date) -> tuple[pd.DataFrame, pd.DataFrame]:
        f = self.factor_history(d)
        return f.cov(), f

    def idio_vol(self, d: dt.date) -> pd.Series:
        """daily idiosyncratic volatility per stock from the last 250 cross-sections"""
        end = d - dt.timedelta(days=1); days = self.ret.loc[:end].index[-60:]
        res = pd.DataFrame({t: self.cross_section(t, self.ret.loc[t])[1] for t in days}).T
        return res.std().fillna(res.std().median())
