"""The risk monitor.  It reads what a desk reads: the price and FX feeds, the OMS fill stream, the prime broker's
intraday position snapshots, the live instrument master, the margin parameters and the settlement confirmations;
it keeps its own positions from start-of-day plus fills, recomputes exposures, limits, P&L since the open and
margin every five minutes, and evaluates the rules below.  Two kinds of alert: events (a spike, a P&L jump, an
off-market fill) with a cooling window per scope, and conditions (a stale price, a break, a breach, a margin call,
an overdue roll, an unconfirmed settlement) raised once when they appear and again only after they clear.  The
whole log is scored by the fault harness."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

from . import book as B, exposures as X, limits as L, margin as MG, pnl as PL


@dataclass
class Thresholds:
    stale_min: float = 15.0            # a price in session unchanged for this long
    fx_stale_min: float = 30.0
    spike_sigma: float = 6.0           # one-step move in units of the instrument's per-step volatility
    spike_floor: float = 0.03          # ... and at least this fraction
    fx_spike_sigma: float = 8.0; fx_spike_floor: float = 0.01
    position_tolerance: float = 0.0    # OMS vs prime-broker snapshot, in units
    pnl_jump_sigma: float = 5.0        # one-step P&L in units of the book's per-step sigma (opening steps excluded)
    roll_overdue_after: str = "14:30"  # New York time on the roll day
    settlement_cutoff: str = "12:00"
    trace_report_s: float = 900.0
    cooldown_s: float = 1800.0         # events
    book_break_names: int = 5          # more simultaneous breaks than this is one book-wide break, not many


SEVERITY = {"STALE_PRICE": "medium", "PRICE_SPIKE": "high", "FX_STALE": "medium", "FX_SPIKE": "high", "POSITION_BREAK": "high", "MASTER_CHANGE": "high", "LIMIT_BREACH": "critical", "LIMIT_WARN": "low",
            "PNL_JUMP": "high", "MARGIN_CALL": "critical", "ROLL_OVERDUE": "high", "HELD_PAST_NOTICE": "critical", "SETTLEMENT_UNCONFIRMED": "high", "STALE_MARK": "medium", "UNKNOWN_INSTRUMENT": "high",
            "LATE_TRACE_REPORT": "medium", "RFQ_OFF_MARKET": "high"}


class Monitor:
    def __init__(self, d: dt.date, clock, ctx, positions_sod: dict, master: dict, marks_sod: dict, fx_sod: dict, rates_sod: dict, nav_sod: float, th: Thresholds | None = None, limits: L.LimitSet | None = None, margin_params: MG.MarginParams | None = None):
        self.d, self.clock, self.ctx, self.th = d, clock, ctx, th or Thresholds(); self.limits = limits or L.LimitSet(); self.mp = margin_params or MG.MarginParams()
        self.positions = dict(positions_sod)                              # start of day plus the OMS fills
        self.positions_sod = dict(positions_sod); self.master_sod = {k: _copy(v) for k, v in master.items()}; self.master = master
        self.marks = dict(marks_sod); self.marks_sod = dict(marks_sod); self.fx = dict(fx_sod); self.rates = dict(rates_sod); self.nav_sod = nav_sod
        self.last_change: dict[str, float] = {}; self.last_px: dict[str, float] = {}; self.last_px_time: dict[str, float] = {}; self.was_live: dict[str, bool] = {}; self.fx_last_change: dict[str, float] = {}; self.fx_last: dict[str, float] = {}
        self.alerts: list[dict] = []; self._last_event: dict[tuple, float] = {}; self.active: dict[tuple, str] = {}; self.break_amount: dict[str, float] = {}
        self.fills: list[dict] = []; self.pb_snapshots: list[tuple[float, dict]] = []
        self.pending_rolls: dict[str, str] = {}; self.rolled: set = set(); self.settlements: list[dict] = []; self.rfq_reports: dict[str, float | None] = {}; self.rfq_exec: dict[str, dict] = {}
        self.ex: X.Exposures | None = None; self.pnl_since_sod = 0.0; self.pnl_path: list[tuple[float, float]] = []; self.last_pnl = 0.0; self.realised = 0.0
        self.margin: dict = {}; self.limit_rows: list[dict] = []; self.sigma_step = 1.0; self.step_sigma: dict[str, float] = {}
        self.exposure_log: list[X.Exposures] = []; self._var_last: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # ---- alerts ------------------------------------------------------------------------------------------------------------
    def _emit(self, rule: str, scope: str, detail: str):
        self.alerts.append({"date": self.d, "time": self.clock(), "rule": rule, "severity": SEVERITY.get(rule, "low"), "scope": scope, "detail": detail[:300]})

    def event(self, rule: str, scope: str = "", detail: str = ""):
        now = self.clock(); key = (rule, scope)
        if now - self._last_event.get(key, -1e12) < self.th.cooldown_s:
            return
        self._last_event[key] = now; self._emit(rule, scope, detail)

    def condition(self, rule: str, scope: str, active: bool, detail: str = ""):
        key = (rule, scope)
        if active and key not in self.active:
            self.active[key] = detail; self._emit(rule, scope, detail)
        elif not active and key in self.active:
            del self.active[key]

    # ---- inputs ----------------------------------------------------------------------------------------------------------------
    def on_fill(self, f: dict):
        if f["instrument_id"] not in self.master:
            self.event("UNKNOWN_INSTRUMENT", f["instrument_id"], f"fill on an instrument missing from the master: {f['instrument_id']}"); return
        k = (f["strategy"], f["instrument_id"]); self.positions[k] = self.positions.get(k, 0.0) + f["qty"]; self.fills.append(f)

    def on_pb_snapshot(self, t: float, snap: dict):
        """the prime broker's positions per instrument (all strategies) against start-of-day plus fills"""
        self.pb_snapshots.append((t, snap))
        mine: dict[str, float] = {}
        for (s, iid), q in self.positions.items():
            mine[iid] = mine.get(iid, 0.0) + q
        tol = self.th.position_tolerance + 1e-9
        breaks = {iid: (mine.get(iid, 0.0), snap.get(iid, 0.0)) for iid in set(mine) | set(snap) if abs(mine.get(iid, 0.0) - snap.get(iid, 0.0)) > tol}
        book_wide = len(breaks) > self.th.book_break_names
        factor = 1.0
        if book_wide:
            # a position load scales the whole file (a doubled load shows every name at twice the OMS); the residual
            # after that scaling is what a single bad fill looks like, and is still raised per name
            ratios = [b / a for a, b in breaks.values() if abs(a) > 0 and abs(b) > 0]
            factor = float(np.median(ratios)) if ratios else 1.0
        self.condition("POSITION_BREAK", "book", book_wide, f"{len(breaks)} instruments differ between the OMS and the prime broker's snapshot (e.g. {', '.join(list(breaks)[:4])}; snapshot = {factor:.2f} x OMS): a position load, not a fill")
        residual = {iid: (a, b) for iid, (a, b) in breaks.items() if abs(b - factor * a) > tol}
        for iid, (a, b) in residual.items():
            diff = b - factor * a
            if abs(diff - self.break_amount.get(iid, 0.0)) > tol:       # a break that changes size is a new incident on the same name
                self.active.pop(("POSITION_BREAK", iid), None)
            self.break_amount[iid] = diff
            self.condition("POSITION_BREAK", iid, True, f"OMS {a:,.0f} vs prime broker {b:,.0f} in {iid} ({a - b:+,.0f})" + (f" after the {factor:.0f}x load" if book_wide else ""))
        for key in [k for k in self.active if k[0] == "POSITION_BREAK" and k[1] != "book" and k[1] not in residual]:
            del self.active[key]; self.break_amount.pop(key[1], None)

    def on_settlement_confirmed(self, key: str):
        for s in self.settlements:
            if s["key"] == key:
                s["confirmed"] = True

    def on_rfq(self, trade: dict):
        self.rfq_exec[trade["id"]] = trade; self.rfq_reports[trade["id"]] = None
        if "off_market" in trade.get("flags", ""):
            self.event("RFQ_OFF_MARKET", trade["id"], f"{trade['id']} {trade['side']} {trade['par']:,.0f} {trade['cusip']} at {trade['price']:.3f} vs mark {trade['mark']:.3f} (TRACE vwap {trade.get('trace_vwap', float('nan')):.3f}): {trade['flags']}")

    def on_rfq_report(self, trade_id: str, t: float):
        self.rfq_reports[trade_id] = t

    # ---- the step ----------------------------------------------------------------------------------------------------------------
    def step(self, t: float, feed_marks: dict, feed_fx: dict, feed_rates: dict, market, pb_snapshot: dict | None = None, margin_params: MG.MarginParams | None = None, with_var: bool = True, gap_step: bool = False):
        th = self.th; ctx = self.ctx
        # price feed: staleness and spikes (equities in session, futures before settlement)
        for iid, p in feed_marks.items():
            inst = self.master.get(iid)
            if inst is None:
                continue
            live = (inst.asset_class == B.EQUITY and market.equity_open(iid, t)) or (inst.asset_class == B.FUTURE and t < market.t_settle)
            prev = self.last_px.get(iid)
            if prev is None or abs(p - prev) > 1e-12 or (live and not self.was_live.get(iid, False)):      # the clock starts at the open
                self.last_change[iid] = t
            self.was_live[iid] = live
            self.condition("STALE_PRICE", iid, live and t - self.last_change.get(iid, t) >= th.stale_min * 60, f"{iid} unchanged at {p:.4g} for {(t - self.last_change.get(iid, t)) / 60:.0f} min while trading")
            t_open = market.eq[iid]["t_open"] if inst.asset_class == B.EQUITY and iid in market.eq else market.t0
            if prev is not None and prev > 0 and p > 0 and live and self.last_px_time.get(iid, -1.0) >= (t_open or 0.0):      # the first tick after the open is the overnight gap, not a spike
                mv = abs(np.log(p / prev)); s = self.step_sigma.get(iid, 0.01)
                if mv > max(th.spike_sigma * s, th.spike_floor):
                    self.event("PRICE_SPIKE", iid, f"{iid} moved {100 * (p / prev - 1):+.1f}% in one step ({mv / max(s, 1e-9):.0f} sigma)")
            self.last_px[iid] = p; self.last_px_time[iid] = t
        self.marks.update(feed_marks)
        for pair, r in feed_fx.items():
            prev = self.fx_last.get(pair)
            if prev is None or abs(r - prev) > 1e-12:
                self.fx_last_change[pair] = t
            self.condition("FX_STALE", pair, t - self.fx_last_change.get(pair, t) >= th.fx_stale_min * 60, f"{pair} unchanged at {r:.5f} for {(t - self.fx_last_change.get(pair, t)) / 60:.0f} min")
            if prev is not None and abs(np.log(r / prev)) > max(th.fx_spike_sigma * self.step_sigma.get(pair, 0.001), th.fx_spike_floor):
                self.event("FX_SPIKE", pair, f"{pair} {prev:.5f} -> {r:.5f} in one step")
            self.fx_last[pair] = r
        self.fx.update(feed_fx); self.rates.update(feed_rates)
        if pb_snapshot is not None:
            self.on_pb_snapshot(t, pb_snapshot)
        # exposures on the live master; a master that changed since the open is itself an alert
        ex = X.compute(self.d, t, self.positions, self.master, self.marks, self.fx, self.rates, ctx.rm, self.nav_sod + self.pnl_since_sod, ctx.betas, ctx.adv, with_var=with_var, fut_beta=ctx.fut_beta, factor_hist=ctx.factor_hist, idio_vol=ctx.idio_vol, cov=ctx.cov)
        if with_var:
            self._var_last = (ex.var_param, ex.var_hist, ex.sigma)
        else:
            ex.var_param, ex.var_hist, ex.sigma = self._var_last
        self.ex = ex
        for iid, inst in self.master.items():
            b = self.master_sod.get(iid)
            changed = b is not None and (inst.multiplier != b.multiplier or inst.sector != b.sector or inst.price_scale != b.price_scale)
            if changed or ("MASTER_CHANGE", iid) in self.active:
                self.condition("MASTER_CHANGE", iid, changed, f"instrument master changed intraday for {iid}: multiplier {b.multiplier:g}->{inst.multiplier:g}, sector {b.sector}->{inst.sector}, scale {b.price_scale:g}->{inst.price_scale:g}" if changed else "")
        # P&L since the open on the same identity as the close
        rows = PL.attribute(self.d, self.positions_sod, self.fills, self.master, self.marks_sod, self.marks, ctx.fx_sod, self.fx, ctx.rates_sod, self.rates, ctx.rm, ctx.betas, spread_chg={"IG": 0.0, "HY": 0.0}, t_frac=0.0, d0=ctx.d_prev)
        pnl = PL.total(rows) + self.realised; d_step = pnl - self.last_pnl; self.last_pnl = pnl; self.pnl_since_sod = pnl; self.pnl_path.append((t, pnl))
        if not gap_step and self.sigma_step > 0 and abs(d_step) > th.pnl_jump_sigma * self.sigma_step:
            self.event("PNL_JUMP", "book", f"P&L moved {d_step / 1e6:+.1f}m in one step ({abs(d_step) / self.sigma_step:.0f} sigma of a five-minute step)")
        # margin
        mp = margin_params or self.mp
        fut = MG.futures_margin(self.positions, self.master, ctx.ranges, self.d, ctx.specs, ctx.cfg.roll_days_before)
        req = MG.requirement(ex.positions, self.master, fut, mp); coll = MG.collateral(self.nav_sod + pnl, ex.positions)
        util = req["total"] / max(coll["cash"], 1.0) if coll["cash"] > 0 else 9.99
        self.margin = {"requirement": req, "collateral": coll, "utilisation": util, "futures": fut}
        self.condition("MARGIN_CALL", "account", req["total"] > coll["cash"], f"requirement {req['total'] / 1e6:,.0f}m above collateral {coll['cash'] / 1e6:,.0f}m (equity {req['equity'] / 1e6:,.0f}m, futures {req['futures_initial'] / 1e6:,.0f}m, bonds {req['bonds'] / 1e6:,.0f}m, FX VM {req['fx_variation'] / 1e6:,.0f}m)")
        # limits: a breach or a warning is a condition per limit and scope
        self.limit_rows = L.evaluate(self.limits, ex, margin_util=util, intraday_pnl=pnl)
        seen = set()
        for r in self.limit_rows:
            scope = f"{r['limit_name']}:{r['scope']}"; seen.add(scope)
            money = r["threshold"] > 100
            self.condition("LIMIT_BREACH", scope, r["status"] == "breach", f"{r['limit_name']} {r['scope']}: {r['value'] / 1e6:,.1f}m vs limit {r['threshold'] / 1e6:,.1f}m ({100 * r['utilisation']:.0f}%)" if money else f"{r['limit_name']} {r['scope']}: {r['value']:.2f} vs {r['threshold']:.2f}")
            self.condition("LIMIT_WARN", scope, r["status"] == "warn", f"{r['limit_name']} {r['scope']} at {100 * r['utilisation']:.0f}% of the limit")
        for key in [k for k in self.active if k[0] in ("LIMIT_BREACH", "LIMIT_WARN") and k[1] not in seen]:
            del self.active[key]
        # rolls, notice dates, settlements, TRACE reports
        held = {k[1] for k, q in self.positions.items() if abs(q) > 0}
        after_roll = t >= _ny(self.d, th.roll_overdue_after)
        for iid, tgt in self.pending_rolls.items():
            self.condition("ROLL_OVERDUE", iid, after_roll and iid not in self.rolled and iid in held, f"{iid} should have rolled to {tgt} by {th.roll_overdue_after} and is still held")
        for iid, fn in ctx.notice_dates.items():
            self.condition("HELD_PAST_NOTICE", iid, fn is not None and self.d >= fn and iid in held, f"{iid} held on or after its first notice / last trading day {fn}")
        after_cut = t >= _ny(self.d, th.settlement_cutoff)
        for s in self.settlements:
            self.condition("SETTLEMENT_UNCONFIRMED", s["key"], after_cut and not s["confirmed"], f"{s['ccy']} {s['amount']:,.0f} with {s['counterparty']} value {s['value_date']} not confirmed by {th.settlement_cutoff}")
        for tid, rep in self.rfq_reports.items():
            ex_t = self.rfq_exec[tid]["time"]; late = (rep is None and t - ex_t > th.trace_report_s) or (rep is not None and rep - ex_t > th.trace_report_s)
            self.condition("LATE_TRACE_REPORT", tid, late, f"RFQ {tid} executed at {_hhmm(ex_t)} reported {'not yet' if rep is None else _hhmm(rep)} (> {th.trace_report_s / 60:.0f} min)")
        self.exposure_log.append(ex)

    def check_marks_refreshed(self, bond_marks_today: dict, bond_marks_prev: dict):
        same = [c for c, p in bond_marks_today.items() if c in bond_marks_prev and abs(p - bond_marks_prev[c]) < 1e-12]
        if len(same) >= 5:
            self.event("STALE_MARK", "bonds", f"{len(same)} of {len(bond_marks_today)} bond marks identical to the previous day's evaluated prices")


def _copy(inst: B.Instrument) -> B.Instrument:
    import copy
    return copy.copy(inst)


def _ny(d: dt.date, hhmm: str) -> float:
    from .market import ny_time
    return ny_time(d, hhmm)


def _hhmm(t: float) -> str:
    from .market import NY
    return dt.datetime.fromtimestamp(t, NY).strftime("%H:%M")
