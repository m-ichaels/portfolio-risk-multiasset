#!/usr/bin/env python3
"""Free data for the multi-asset book, all without a key:

  futures   Yahoo daily bars for every listed CME/CBOT/NYMEX/COMEX contract of the roots the book holds (Yahoo serves
            listed contracts only, expired ones are gone, so what is committed here is the record) and the continuous
            front-month series (=F) two years back
  fx        ECB euro reference rates since 1999 (USD, GBP, JPY against EUR -> EURUSD, GBPUSD, USDJPY) and Yahoo's
            daily pairs for a cross-check
  rates     SOFR (New York Fed), euro STR (ECB), SONIA (Bank of England), JGB par yields (Japan MoF), the US Treasury
            par curve (Treasury.gov)
  bonds     iShares LQD and HYG holdings with each bond's evaluated price, duration, yield and sector (latest file)
  sectors   Yahoo's sector and industry for the equity universe
  trace     FINRA TRACE prints through the FINRA Query API: needs FINRA_API_CLIENT_ID / FINRA_API_CLIENT_SECRET
            (registration is free); without them this step is skipped and the run says so

Usage: python tools/download.py [futures] [fx] [rates] [bonds] [sectors] [trace]   (default: all but trace)"""
from __future__ import annotations

import base64
import csv
import datetime as dt
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import zipfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw"); DER = os.path.join(ROOT, "data", "derived"); REF = os.path.join(ROOT, "data", "reference")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/128 Safari/537.36", "Accept": "*/*"}
MONTH_CODES = "FGHJKMNQUVXZ"
# root -> (Yahoo exchange suffix, listed months); the front-month rule lives in xops.fi
ROOTS = {"ES": (".CME", (3, 6, 9, 12)), "NQ": (".CME", (3, 6, 9, 12)), "RTY": (".CME", (3, 6, 9, 12)),
         "ZT": (".CBT", (3, 6, 9, 12)), "ZF": (".CBT", (3, 6, 9, 12)), "ZN": (".CBT", (3, 6, 9, 12)), "ZB": (".CBT", (3, 6, 9, 12)),
         "6E": (".CME", (3, 6, 9, 12)), "6B": (".CME", (3, 6, 9, 12)), "6J": (".CME", (3, 6, 9, 12)),
         "CL": (".NYM", tuple(range(1, 13))), "GC": (".CMX", (2, 4, 6, 8, 10, 12))}


def get(url: str, retries: int = 3, timeout: int = 60, headers: dict | None = None) -> bytes | None:
    h = dict(UA); h.update(headers or {})
    for k in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if k == retries - 1:
                print("  failed", url, e); return None
        except Exception as e:  # noqa: BLE001
            if k == retries - 1:
                print("  failed", url, e); return None
        time.sleep(1.5 * (k + 1))
    return None


def yahoo_chart(sym: str, rng: str = "2y", interval: str = "1d") -> list[dict]:
    raw = get(f"https://query2.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(sym)}?range={rng}&interval={interval}", retries=2, timeout=30)
    if raw is None:
        return []
    try:
        r = json.loads(raw)["chart"]["result"][0]
    except (KeyError, IndexError, TypeError, ValueError):
        return []
    q = r["indicators"]["quote"][0]; ts = r.get("timestamp") or []
    out = []
    for t, o, h, lo, c, v in zip(ts, q.get("open", []), q.get("high", []), q.get("low", []), q.get("close", []), q.get("volume", [])):
        if c is None:
            continue
        out.append({"symbol": sym, "date": dt.datetime.fromtimestamp(t, dt.timezone.utc).date().isoformat(), "open": o if o is not None else c, "high": h if h is not None else c, "low": lo if lo is not None else c, "close": c, "volume": v or 0})
    return out


# ---- futures ----------------------------------------------------------------------------------------------------------
def futures():
    os.makedirs(os.path.join(RAW, "yahoo"), exist_ok=True)
    today = dt.date.today(); rows = []; listed = []
    for root, (suffix, months) in ROOTS.items():
        n = 0
        for k in range(0, 21):
            y, m = today.year + (today.month - 1 + k) // 12, (today.month - 1 + k) % 12 + 1
            if m not in months:
                continue
            code = f"{root}{MONTH_CODES[m - 1]}{y % 100:02d}"; sym = code + suffix
            bars = yahoo_chart(sym, "2y")
            if not bars:
                continue
            for b in bars:
                b["contract"] = code; b["root"] = root
            rows += bars; n += 1; listed.append({"contract": code, "root": root, "yahoo": sym, "first": bars[0]["date"], "last": bars[-1]["date"], "n": len(bars)})
            time.sleep(0.15)
        cont = yahoo_chart(root + "=F", "2y")
        for b in cont:
            b["contract"] = root + "=F"; b["root"] = root
        rows += cont
        print(f"  {root}: {n} listed contracts, continuous {len(cont)} days", flush=True)
    df = pd.DataFrame(rows); df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[["root", "contract", "date", "open", "high", "low", "close", "volume"]].sort_values(["root", "contract", "date"])
    df.to_parquet(os.path.join(DER, "futures_settles.parquet"), index=False)
    pd.DataFrame(listed).to_csv(os.path.join(DER, "futures_listed.csv"), index=False)
    print(f"  futures: {len(df)} rows, {len(listed)} contracts")


# ---- FX ---------------------------------------------------------------------------------------------------------------
def fx():
    raw = get("https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip")
    z = zipfile.ZipFile(io.BytesIO(raw)); name = [n for n in z.namelist() if n.endswith(".csv")][0]
    ecb = pd.read_csv(z.open(name)); ecb.columns = [c.strip() for c in ecb.columns]
    ecb = ecb[["Date", "USD", "GBP", "JPY"]].rename(columns={"Date": "date"}); ecb["date"] = pd.to_datetime(ecb["date"]).dt.date
    for c in ("USD", "GBP", "JPY"):
        ecb[c] = pd.to_numeric(ecb[c], errors="coerce")
    ecb = ecb.dropna().sort_values("date")
    out = pd.DataFrame({"date": ecb["date"], "EURUSD": ecb["USD"], "GBPUSD": ecb["USD"] / ecb["GBP"], "USDJPY": ecb["JPY"] / ecb["USD"]})
    long = out.melt(id_vars="date", var_name="pair", value_name="rate"); long["source"] = "ecb"
    ys = []
    for pair, sym in (("EURUSD", "EURUSD=X"), ("GBPUSD", "GBPUSD=X"), ("USDJPY", "JPY=X")):
        for b in yahoo_chart(sym, "2y"):
            ys.append({"date": dt.date.fromisoformat(b["date"]), "pair": pair, "rate": b["close"], "source": "yahoo"})
    allfx = pd.concat([long, pd.DataFrame(ys)], ignore_index=True).sort_values(["source", "pair", "date"])
    allfx.to_parquet(os.path.join(DER, "fx_rates.parquet"), index=False)
    print(f"  fx: ECB {len(out)} days from {out['date'].min()}, Yahoo {len(ys)} rows")


# ---- rates ------------------------------------------------------------------------------------------------------------
def rates():
    rows = []
    start = "2015-01-01"
    raw = get(f"https://markets.newyorkfed.org/api/rates/secured/sofr/search.json?startDate={start}&endDate={dt.date.today().isoformat()}")
    for r in json.loads(raw)["refRates"]:
        rows.append({"date": r["effectiveDate"], "name": "SOFR", "value": r["percentRate"]})
    raw = get(f"https://data-api.ecb.europa.eu/service/data/EST/B.EU000A2X2A25.WT?format=csvdata&startPeriod={start}")
    e = pd.read_csv(io.BytesIO(raw))
    for r in e.itertuples():
        rows.append({"date": r.TIME_PERIOD, "name": "ESTR", "value": r.OBS_VALUE})
    d0 = dt.date.fromisoformat(start).strftime("%d/%b/%Y"); d1 = dt.date.today().strftime("%d/%b/%Y")
    raw = get(f"https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes&Datefrom={d0}&Dateto={d1}&SeriesCodes=IUDSOIA&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N")
    b = pd.read_csv(io.BytesIO(raw))
    for r in b.itertuples():
        rows.append({"date": pd.to_datetime(r.DATE, format="%d %b %Y").date().isoformat(), "name": "SONIA", "value": float(r.IUDSOIA)})
    raw = get("https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv")
    j = pd.read_csv(io.BytesIO(raw), skiprows=1)
    j = j[pd.to_datetime(j["Date"], format="%Y/%m/%d", errors="coerce") >= pd.Timestamp(start)]
    for r in j.itertuples():
        v = pd.to_numeric(r._2, errors="coerce")   # 1Y
        if pd.notna(v):
            rows.append({"date": pd.to_datetime(r.Date, format="%Y/%m/%d").date().isoformat(), "name": "JGB1Y", "value": float(v)})
    for y in range(2024, dt.date.today().year + 1):
        raw = get(f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{y}/all?type=daily_treasury_yield_curve&field_tdr_date_value={y}&page&_format=csv")
        t = pd.read_csv(io.BytesIO(raw))
        for r in t.to_dict("records"):
            d = pd.to_datetime(r["Date"]).date().isoformat()
            for col, nm in (("1 Mo", "UST1M"), ("3 Mo", "UST3M"), ("6 Mo", "UST6M"), ("1 Yr", "UST1Y"), ("2 Yr", "UST2Y"), ("3 Yr", "UST3Y"), ("5 Yr", "UST5Y"), ("7 Yr", "UST7Y"), ("10 Yr", "UST10Y"), ("20 Yr", "UST20Y"), ("30 Yr", "UST30Y")):
                v = pd.to_numeric(r.get(col), errors="coerce")
                if pd.notna(v):
                    rows.append({"date": d, "name": nm, "value": float(v)})
    df = pd.DataFrame(rows); df["date"] = pd.to_datetime(df["date"]).dt.date; df = df.sort_values(["name", "date"]).drop_duplicates(["name", "date"])
    df.to_parquet(os.path.join(DER, "rates.parquet"), index=False)
    print("  rates:", df.groupby("name")["date"].agg(["count", "min", "max"]).to_string())


# ---- bonds ------------------------------------------------------------------------------------------------------------
ISHARES = {"LQD": "https://www.ishares.com/us/products/239566/ishares-iboxx-investment-grade-corporate-bond-etf/latest-holdings.csv",
           "HYG": "https://www.ishares.com/us/products/239565/ishares-iboxx-high-yield-corporate-bond-etf/latest-holdings.csv"}


def bonds():
    frames = []
    for fund, url in ISHARES.items():
        raw = get(url); txt = raw.decode("utf-8-sig", "replace")
        lines = txt.splitlines(); hdr = next(i for i, l in enumerate(lines) if l.startswith("Name,"))
        asof = next((l.split(",", 1)[1].strip('" ') for l in lines[:hdr] if l.startswith("Fund Holdings as of")), "")
        t = pd.read_csv(io.StringIO("\n".join(lines[hdr:])), thousands=",")
        t = t[t["Asset Class"] == "Fixed Income"].copy()
        t["fund"] = fund; t["asof"] = pd.to_datetime(asof).date()
        t = t.rename(columns={"Name": "name", "Sector": "sector", "Market Value": "market_value", "Weight (%)": "weight_pct", "Par Value": "par", "CUSIP": "cusip", "ISIN": "isin", "Price": "price", "Location": "country",
                              "Duration": "duration", "YTM (%)": "ytm", "Maturity": "maturity", "Coupon (%)": "coupon", "Mod. Duration": "mod_duration", "Yield to Worst (%)": "ytw", "Accrual Date": "accrual_date"})
        t["maturity"] = pd.to_datetime(t["maturity"], errors="coerce").dt.date; t["accrual_date"] = pd.to_datetime(t["accrual_date"], errors="coerce").dt.date
        for c in ("market_value", "weight_pct", "par", "price", "duration", "ytm", "coupon", "mod_duration", "ytw"):
            t[c] = pd.to_numeric(t[c], errors="coerce")
        frames.append(t[["fund", "asof", "cusip", "isin", "name", "sector", "country", "price", "par", "market_value", "weight_pct", "duration", "mod_duration", "ytm", "ytw", "coupon", "maturity", "accrual_date"]])
        print(f"  {fund}: {len(t)} bonds as of {asof}")
    df = pd.concat(frames, ignore_index=True).dropna(subset=["price", "duration", "maturity"])
    df.to_parquet(os.path.join(DER, "bonds.parquet"), index=False)
    for fund, sym in (("LQD", "LQD"), ("HYG", "HYG")):
        pass
    etf = []
    for sym in ("LQD", "HYG", "SPY", "IWM", "QQQ"):
        etf += yahoo_chart(sym, "2y")
    e = pd.DataFrame(etf); e["date"] = pd.to_datetime(e["date"]).dt.date; e.to_parquet(os.path.join(DER, "etf_prices.parquet"), index=False)
    print(f"  bonds: {len(df)} rows; etf prices {len(e)} rows")


# ---- sectors ----------------------------------------------------------------------------------------------------------
def sectors():
    uni = pd.read_csv(os.path.join(ROOT, "data", "universe.csv")); rows = []
    for s in uni["symbol"]:
        raw = get(f"https://query2.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(s)}&quotesCount=3&newsCount=0", retries=2, timeout=20)
        sec = ind = ""
        try:
            for q in json.loads(raw)["quotes"]:
                if q.get("symbol") == s:
                    sec, ind = q.get("sector", ""), q.get("industry", ""); break
        except (TypeError, ValueError, KeyError):
            pass
        rows.append({"symbol": s, "sector": sec or "Unknown", "industry": ind}); time.sleep(0.15)
    pd.DataFrame(rows).to_csv(os.path.join(REF, "sectors.csv"), index=False)
    print("  sectors:", pd.DataFrame(rows)["sector"].value_counts().to_dict())


# ---- TRACE (FINRA Query API, free registration) -------------------------------------------------------------------------
def trace():
    cid, sec = os.environ.get("FINRA_API_CLIENT_ID"), os.environ.get("FINRA_API_CLIENT_SECRET")
    if not cid or not sec:
        print("  trace: FINRA_API_CLIENT_ID / FINRA_API_CLIENT_SECRET not set; skipped (the credit post-trade runs on simulated prints and says so)"); return
    tok = base64.b64encode(f"{cid}:{sec}".encode()).decode()
    req = urllib.request.Request("https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token?grant_type=client_credentials", method="POST", headers={"Authorization": "Basic " + tok})
    with urllib.request.urlopen(req, timeout=60) as r:
        access = json.loads(r.read())["access_token"]
    b = pd.read_parquet(os.path.join(DER, "bonds.parquet")); cusips = sorted(set(b["cusip"]))
    rows = []
    for i in range(0, len(cusips), 50):
        body = json.dumps({"limit": 5000, "compareFilters": [{"fieldName": "tradeReportDate", "compareType": "GTE", "fieldValue": "2026-07-01"}], "domainFilters": [{"fieldName": "cusip", "values": cusips[i:i + 50]}]}).encode()
        req = urllib.request.Request("https://api.finra.org/data/group/fixedIncomeMarket/name/corporateBondTrades", data=body, method="POST", headers={"Authorization": "Bearer " + access, "Content-Type": "application/json", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                rows += json.loads(r.read())
        except urllib.error.HTTPError as e:
            print("  trace chunk failed", e); continue
    if rows:
        pd.DataFrame(rows).to_parquet(os.path.join(DER, "trace_prints.parquet"), index=False); print(f"  trace: {len(rows)} prints")


def main():
    os.makedirs(DER, exist_ok=True); os.makedirs(REF, exist_ok=True)
    steps = [a for a in sys.argv[1:] if not a.startswith("-")] or ["futures", "fx", "rates", "bonds", "sectors"]
    for s in steps:
        print(s, flush=True); globals()[s]()


if __name__ == "__main__":
    main()
