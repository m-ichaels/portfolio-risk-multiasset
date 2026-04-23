"""Fault injection for the risk layer: the catalogue of what goes wrong between a fund and its feeds, brokers and
back office (frozen or spiked prices, an inverted FX rate, a phantom or missing position, a doubled position load,
a fill booked the wrong way, an instrument-master change, a rogue fill through a limit, a roll that does not
happen, a margin call, a settlement that is not confirmed, evaluated marks that are not refreshed, a late or
off-market bond trade), the schedule generator that plants them on a fault day, and the scorer that turns the
monitor's alerts into detection rates and times to detect."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# fault type -> alert rules that count as a detection
RULES_FOR = {
    "stale_price": ["STALE_PRICE"], "spiked_mark": ["PRICE_SPIKE", "PNL_JUMP"], "fx_stale": ["FX_STALE"], "fx_inverted": ["FX_SPIKE", "PNL_JUMP"],
    "phantom_position": ["POSITION_BREAK"], "missing_fill": ["POSITION_BREAK"], "doubled_load": ["POSITION_BREAK"], "sign_flip": ["POSITION_BREAK"],
    "wrong_multiplier": ["MASTER_CHANGE"], "sector_mislabel": ["MASTER_CHANGE"], "rogue_fill": ["LIMIT_BREACH", "PNL_JUMP"],
    "contract_not_rolled": ["ROLL_OVERDUE", "HELD_PAST_NOTICE"], "margin_shortfall": ["MARGIN_CALL"], "settlement_missing": ["SETTLEMENT_UNCONFIRMED"],
    "bond_mark_stale": ["STALE_MARK"], "late_trace_report": ["LATE_TRACE_REPORT"], "off_market_rfq": ["RFQ_OFF_MARKET"],
}
SCOPED = {"stale_price", "spiked_mark", "fx_stale", "fx_inverted", "phantom_position", "missing_fill", "sign_flip", "wrong_multiplier", "sector_mislabel", "contract_not_rolled", "settlement_missing", "late_trace_report", "off_market_rfq"}
DEFAULT_MIX = {"stale_price": 2, "spiked_mark": 2, "fx_stale": 1, "fx_inverted": 1, "phantom_position": 2, "missing_fill": 2, "doubled_load": 1, "sign_flip": 1, "wrong_multiplier": 1, "sector_mislabel": 1, "rogue_fill": 1,
               "contract_not_rolled": 1, "margin_shortfall": 1, "settlement_missing": 1, "bond_mark_stale": 1, "late_trace_report": 1, "off_market_rfq": 1}
WINDOW_S = {"contract_not_rolled": 4 * 3600.0, "bond_mark_stale": 8 * 3600.0}


@dataclass
class Fault:
    type: str; t: float; scope: str = ""; params: dict = field(default_factory=dict); armed: bool = True; fired_at: float | None = None; applicable: bool = True


def generate_schedule(day, rng: np.random.Generator, mix: dict | None = None) -> list[Fault]:
    """`day` carries what the generator needs: the monitoring window, held equities and futures in session, the
    day's equity slices, rolls due, settlements due, RFQs.  One fault per scope."""
    mix = dict(mix or DEFAULT_MIX); out = []; used: set = set()
    t0, t1 = day["t_open_us"], day["t_close_us"]

    def key(p):
        return p.get("instrument_id", p["id"]) if isinstance(p, dict) else p      # one fault per name or per RFQ, so scopes never overlap

    def pick(pool):
        pool = [p for p in pool if key(p) not in used]
        if not pool:
            return None
        p = pool[int(rng.integers(0, len(pool)))]; used.add(key(p)); return p

    def when(lo=t0 + 1800, hi=t1 - 3600):
        return float(rng.uniform(lo, max(lo + 60, hi)))

    for ftype, n in mix.items():
        for _ in range(n):
            if ftype == "stale_price":
                s = pick(day["liquid_equities"]);  out.append(Fault(ftype, when(), s or "", {"minutes": 45}, applicable=s is not None))
            elif ftype == "spiked_mark":
                s = pick(day["liquid_equities"]);  out.append(Fault(ftype, when(), s or "", {"pct": 0.06}, applicable=s is not None))
            elif ftype == "fx_stale":
                out.append(Fault(ftype, when(), "GBPUSD", {"minutes": 60}))
            elif ftype == "fx_inverted":
                out.append(Fault(ftype, when(), "EURUSD", {"minutes": 15}))
            elif ftype == "phantom_position":
                s = pick(day["not_held_equities"]);  out.append(Fault(ftype, when(), s or "", {"qty": 50_000}, applicable=s is not None))
            elif ftype == "missing_fill":
                f = pick(day["big_slices"]);  out.append(Fault(ftype, f["time"] if f else t0, f["instrument_id"] if f else "", {"fill_id": f["id"] if f else None}, applicable=f is not None))
            elif ftype == "doubled_load":
                out.append(Fault(ftype, when(), "book", {}))
            elif ftype == "sign_flip":
                f = pick(day["big_slices"]);  out.append(Fault(ftype, f["time"] if f else t0, f["instrument_id"] if f else "", {"fill_id": f["id"] if f else None}, applicable=f is not None))
            elif ftype == "wrong_multiplier":
                c = pick(day["held_futures"]);  out.append(Fault(ftype, when(), c or "", {"factor": 10.0}, applicable=c is not None))
            elif ftype == "sector_mislabel":
                s = pick(day["big_equities"]);  out.append(Fault(ftype, when(), s or "", {"sector": "Utilities" if day["sectors"].get(s) != "Utilities" else "Real Estate"}, applicable=s is not None))
            elif ftype == "rogue_fill":
                s = pick(day["big_equities"]);  out.append(Fault(ftype, when(), s or "", {"nav_frac": 0.045}, applicable=s is not None))
            elif ftype == "contract_not_rolled":
                r = pick(day["rolls_due"]);  out.append(Fault(ftype, day["t_roll"], r or "", {}, applicable=r is not None))
            elif ftype == "margin_shortfall":
                out.append(Fault(ftype, when(), "account", {"equity_long": 0.60, "equity_short": 0.60}))      # the prime broker doubles its haircuts (a stress add-on)
            elif ftype == "settlement_missing":
                s = pick(day["settlements_due"]);  out.append(Fault(ftype, day["t_settle_conf"], s or "", {}, applicable=s is not None))
            elif ftype == "bond_mark_stale":
                out.append(Fault(ftype, day["t_sod"], "bonds", {}))
            elif ftype == "late_trace_report":
                r = pick(day["rfqs"]);  out.append(Fault(ftype, r["time"] if r else t0, r["id"] if r else "", {"delay_s": 2400.0}, applicable=r is not None))
            elif ftype == "off_market_rfq":
                r = pick(day["rfqs"]);  out.append(Fault(ftype, r["time"] if r else t0, r["id"] if r else "", {"points": 3.0}, applicable=r is not None))
    return [f for f in out if f.applicable]


@dataclass
class Detection:
    type: str; scope: str; t_fault: float; t_alert: float | None; rule: str | None

    @property
    def detected(self) -> bool:
        return self.t_alert is not None

    @property
    def ttd(self) -> float | None:
        return None if self.t_alert is None else self.t_alert - self.t_fault


def score(faults: list[Fault], alerts: list[dict], window: float = 3600.0) -> list[Detection]:
    out = []
    for f in faults:
        if f.fired_at is None:
            continue
        rules = RULES_FOR.get(f.type, []); w = WINDOW_S.get(f.type, window); best = None
        for a in alerts:
            if a["rule"] not in rules or a["time"] < f.fired_at - 1 or a["time"] > f.fired_at + w:
                continue
            if f.type in SCOPED and f.scope and f.scope not in a["scope"] and a["scope"] not in f.scope:
                continue
            if best is None or a["time"] < best["time"]:
                best = a
        out.append(Detection(f.type, f.scope, f.fired_at, best["time"] if best else None, best["rule"] if best else None))
    return out


def summarize(dets: list[Detection], clean_alerts: list | None = None, n_clean_days: int = 0) -> dict:
    by = {}
    for d in dets:
        s = by.setdefault(d.type, {"injected": 0, "detected": 0, "ttd": []})
        s["injected"] += 1
        if d.detected:
            s["detected"] += 1; s["ttd"].append(d.ttd)
    for k, s in by.items():
        s["median_ttd_s"] = float(np.median(s["ttd"])) if s["ttd"] else None; s["p90_ttd_s"] = float(np.quantile(s["ttd"], 0.9)) if s["ttd"] else None; s.pop("ttd")
    inj = sum(s["injected"] for s in by.values()); det = sum(s["detected"] for s in by.values()); all_ttd = [d.ttd for d in dets if d.detected]
    out = {"by_type": by, "injected": inj, "detected": det, "detection_rate": det / inj if inj else None, "median_ttd_s": float(np.median(all_ttd)) if all_ttd else None, "p90_ttd_s": float(np.quantile(all_ttd, 0.9)) if all_ttd else None}
    if clean_alerts is not None:
        rules = {}
        for a in clean_alerts:
            rules[a["rule"]] = rules.get(a["rule"], 0) + 1
        out["false_alerts_per_clean_day"] = len(clean_alerts) / n_clean_days if n_clean_days else None; out["false_alerts_by_rule"] = rules
        out["clean_incidents_per_day"] = len({(a["rule"], a["scope"], a["date"]) for a in clean_alerts}) / n_clean_days if n_clean_days else None
    return out
