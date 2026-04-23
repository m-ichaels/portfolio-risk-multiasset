"""The day runner.  Start of day: marks, the roll and hedge calendar, the FX hedge, the day's equity deltas from
execution-ops' targets, the credit desk's RFQs, the prime broker's snapshot schedule, the fault schedule.  Intraday:
a five-minute virtual clock over the London-to-New-York window, fills executing on the U-curve, the feeds (with
faults applied) into the monitor.  End of day: closes and settlements, the cash ledger, P&L attribution checked
against the NAV change, margin and financing, checks, fault scoring."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import book as B, credit as CR, exposures as X, faults as F, futures as FU, fx as FX, limits as L, margin as MG, market as MK, pnl as PL, valuation as V
from .market import ny_time
from .monitor import Monitor, Thresholds
from .xops_bridge import cal, fi, futures_dates


@dataclass
class Context:
    insts: dict; rm: object; sett: FU.Settlements; specs: pd.DataFrame; ranges: dict; targets: pd.DataFrame; prices: pd.DataFrame; uni: pd.DataFrame; fx: pd.DataFrame; rates: pd.DataFrame
    bond_paths: pd.DataFrame; cfg: B.BookConfig; limits: L.LimitSet; mp: MG.MarginParams; th: Thresholds
    # per-day fields set by run_day
    betas: pd.DataFrame | None = None; adv: pd.Series | None = None; fut_beta: dict = field(default_factory=dict); factor_hist: pd.DataFrame | None = None; idio_vol: pd.Series | None = None; cov: pd.DataFrame | None = None
    fx_sod: dict = field(default_factory=dict); rates_sod: dict = field(default_factory=dict); notice_dates: dict = field(default_factory=dict); d_prev: dt.date | None = None


@dataclass
class BookState:
    positions: dict = field(default_factory=dict)          # (strategy, instrument_id) -> qty
    nav: float = 0.0; cash: float = 0.0
    forwards: dict = field(default_factory=dict)           # instrument_id -> qty (base notional, signed)
    settlements: list = field(default_factory=list)        # pending settlement instructions
    bond_marks_prev: dict = field(default_factory=dict)
    last_fx_rebalance: dt.date | None = None; last_sizing: dt.date | None = None
    rfq_seq: int = 0; fill_seq: int = 0; cash_bond_accrued_today: float = 0.0; cash_fx_today: float = 0.0
    eod_marks: dict = field(default_factory=dict); eod_fx: dict = field(default_factory=dict); eod_rates: dict = field(default_factory=dict)   # the marks the NAV was struck on; the next day's start


@dataclass
class DayResult:
    date: dt.date; fault_day: bool; positions_sod: dict; positions_eod: dict; fills: list; alerts: list; faults: list; detections: list
    pnl_rows: dict; nav_sod: float; nav_eod: float; identity_gap: float; ex_sod: X.Exposures; ex_eod: X.Exposures; margin_eod: dict; financing: dict; limits_sod: list; limits_eod: list
    rolls: list; fx_trades: list; settlements: list; netting: dict; rfqs: list; checks: list; monitor: Monitor; pnl_path: list; exposure_log: list


class Clock:
    def __init__(self, t0: float):
        self.t = t0

    def __call__(self) -> float:
        return self.t


def _prev_row(df: pd.DataFrame, d: dt.date) -> pd.Series:
    h = df.loc[:d - dt.timedelta(days=1)]
    return h.iloc[-1]


def _row_or_prev(df: pd.DataFrame, d: dt.date) -> pd.Series:
    h = df.loc[:d]
    return h.iloc[-1]


def symbol_stats(rm, d: dt.date, lookback: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(daily vol, ADV, spread bp) per symbol from the data before d"""
    end = d - dt.timedelta(days=1); r = rm.ret.loc[:end].tail(lookback); vol = r.std().fillna(0.02).clip(0.005, 0.15)
    adv = rm.volume.loc[:end].tail(lookback).mean().fillna(1e5); close = rm.close.loc[:end].ffill().iloc[-1]      # the last close per name, whatever its exchange did on the last day
    scale = rm.uni["price_scale"].reindex(close.index).fillna(1.0); dollar_adv = (adv * close * scale).fillna(1e6)
    spread = (25.0 / np.sqrt((dollar_adv.clip(lower=1e5)) / 1e6)).clip(1.5, 40.0).fillna(5.0)
    return vol, adv, spread


def run_day(d: dt.date, ctx: Context, state: BookState, *, fault_mix: dict | None = None, seed: int = 1, verbose: bool = False, checks_fn=None, until: str = "EOD") -> DayResult | None:
    """Run one day; until="SOD" stops after the start-of-day marks, exposures, limits and checks (the morning job)."""
    rng = np.random.default_rng(seed * 7919 + d.toordinal()); cfg = ctx.cfg; rm = ctx.rm; insts = ctx.insts; sett = ctx.sett
    if not cal.is_trading_day("XNYS", d):
        return None
    fault_day = fault_mix is not None
    # ---- the day's data --------------------------------------------------------------------------------------------------
    eq_today = ctx.prices[ctx.prices["date"] == d]
    prev_close = rm.close.loc[:d - dt.timedelta(days=1)].ffill().iloc[-1].dropna()
    for c, v in state.eod_marks.items():
        if c in prev_close.index:
            prev_close[c] = v
    vol, adv, spread_bp = symbol_stats(rm, d)
    # start-of-day marks are the marks the previous NAV was struck on (a day the fund did not run, e.g. a US holiday on
    # which London traded, is then booked with the next day); the first day falls back to the previous fixes
    fx_prev = state.eod_fx or _prev_row(ctx.fx, d).to_dict(); fx_now = _row_or_prev(ctx.fx, d).to_dict()
    rates_prev = state.eod_rates or _prev_row(ctx.rates, d).dropna().to_dict(); rates_now = _row_or_prev(ctx.rates, d).dropna().to_dict()
    ctx.fx_sod = fx_prev; ctx.rates_sod = rates_prev
    ctx.betas = rm.betas(d); ctx.adv = adv; ctx.factor_hist = rm.factor_history(d); ctx.cov = ctx.factor_hist.cov(); ctx.idio_vol = rm.idio_vol(d)
    spy = np.log(rm.etf["SPY"]).diff().reindex(ctx.factor_hist.index).fillna(0.0)
    ctx.fut_beta = {r: float(np.cov(ctx.factor_hist[f"FUT_{r}"], spy)[0, 1] / (spy.var() + 1e-12)) for r in ("ES", "NQ", "RTY")}
    # bond marks: yesterday's evaluated prices are the start-of-day marks, today's arrive at the close
    bp = ctx.bond_paths; bidx = bp.index
    prev_days = [x for x in bidx if x < d]; d_prev = prev_days[-1]; d_prev2 = prev_days[-2] if len(prev_days) > 1 else d_prev
    bond_sod = bp.loc[d_prev].to_dict(); bond_prev2 = bp.loc[d_prev2].to_dict(); bond_eod = bp.loc[d].to_dict() if d in bidx else dict(bond_sod)
    bond_sod = {c: state.eod_marks.get(c, v) for c, v in bond_sod.items()}
    bond_mark_stale = fault_day and (fault_mix or {}).get("bond_mark_stale", 0) > 0 and bool(state.bond_marks_prev)
    if bond_mark_stale:
        bond_sod_feed = dict(state.bond_marks_prev)          # the feed did not refresh: the desk sees the previous morning's marks again
    else:
        bond_sod_feed = dict(bond_sod)
    # ---- positions and the roll / hedge / FX calendar ----------------------------------------------------------------------------
    positions_sod = {k: v for k, v in state.positions.items() if abs(v) > 1e-9}; state.cash_bond_accrued_today = 0.0; state.cash_fx_today = 0.0
    held = sorted({iid for (_, iid) in positions_sod})
    cme_prev = cal.next_trading_day("CME", d, -1); d_prev = cal.next_trading_day("XNYS", d, -1); ctx.d_prev = d_prev
    fut_prev = {}; fut_now = {}; fut_vol = {}
    rolls_due = FU.roll_needed(positions_sod, insts, sett, d, cfg.roll_days_before)
    fut_needed = [iid for iid in held if insts[iid].asset_class == B.FUTURE] + [r[2] for r in rolls_due]
    # sizing day: the first CME day of the week (hedge, steepener, credit hedge); the sizing may open new contracts
    sizing_day = state.last_sizing is None or (d - state.last_sizing).days >= 7
    for root in ["ES", "NQ", "ZT", "ZN"] + list(cfg.macro_lots):
        fut_needed.append(sett.should_hold(root, d, cfg.roll_days_before))
    for c in sorted(set(fut_needed)):
        if c not in insts:
            insts[c] = B.futures_instrument(c, ctx.specs)
        p0, _ = sett.last_settle(c, cme_prev); p1, _ = sett.settle(c, d); p0 = state.eod_marks.get(c, p0)
        if not np.isfinite(p1):
            p1 = sett.continuous(insts[c].root, d)        # a month Yahoo does not list (the next month of an early roll): the front's price stands in; the roll row says the spread was unobserved
        if not np.isfinite(p0):
            p0 = p1
        fut_prev[c] = p0; fut_now[c] = p1; fut_vol[c] = float(rm.fut_ret[insts[c].root].loc[:d - dt.timedelta(days=1)].tail(20).std() or 0.01)
    fx_vol = {p: float(rm.fx_ret[p].loc[:d - dt.timedelta(days=1)].tail(20).std() or 0.004) for p in fx_prev}
    ctx.notice_dates = {}
    for c in fut_needed:
        rt, y, m = fi.parse_contract(c); fd = futures_dates(rt, y, m); fn = fd.get("first_notice") or fd.get("last_trade"); ctx.notice_dates[c] = min(x for x in (fd.get("first_notice"), fd.get("last_trade")) if x is not None)
    market = MK.DayMarket(d, insts, eq_today, prev_close, vol, adv, spread_bp, fut_prev, fut_now, fut_vol, fx_prev, fx_now, fx_vol, rates_prev, rates_now, bond_sod, bond_eod, rng)
    t_sod = market.t0; clock = Clock(t_sod)
    marks_sod = {}
    for iid in held:
        inst = insts[iid]
        if inst.asset_class == B.EQUITY and iid in prev_close.index:
            marks_sod[iid] = float(prev_close[iid])
        elif inst.asset_class == B.FUTURE:
            marks_sod[iid] = fut_prev[iid]
        elif inst.asset_class == B.BOND:
            marks_sod[iid] = bond_sod[iid]
        elif inst.asset_class == B.FX_FORWARD:
            marks_sod[iid] = 1.0
    nav0 = state.nav
    # ---- the monitor ------------------------------------------------------------------------------------------------------------------
    live_master = {k: _copy(v) for k, v in insts.items()}
    feed_bond_sod = {iid: bond_sod_feed[iid] for iid in held if insts[iid].asset_class == B.BOND}
    mon = Monitor(d, clock, ctx, positions_sod, live_master, {**marks_sod, **feed_bond_sod}, fx_prev, rates_prev, nav0, ctx.th, ctx.limits, ctx.mp)
    if state.bond_marks_prev:
        mon.check_marks_refreshed(feed_bond_sod, {iid: state.bond_marks_prev.get(iid, float("nan")) for iid in feed_bond_sod})
    ex_sod = X.compute(d, t_sod, positions_sod, insts, marks_sod, fx_prev, rates_prev, rm, nav0, ctx.betas, adv, fut_beta=ctx.fut_beta, factor_hist=ctx.factor_hist, idio_vol=ctx.idio_vol, cov=ctx.cov)
    n_us = max(int((ny_time(d, "16:00") - ny_time(d, "09:30")) // MK.STEP), 1)
    mon.sigma_step = ex_sod.sigma / np.sqrt(n_us) if ex_sod.sigma > 0 else 1e6
    for iid in held + fut_needed:
        inst = insts.get(iid)
        if inst is None:
            continue
        if inst.asset_class == B.EQUITY:
            mon.step_sigma[iid] = float(vol.get(iid, 0.02)) / np.sqrt(max(cal.session_minutes(inst.exchange, d) // 5, 1))
        elif inst.asset_class == B.FUTURE:
            mon.step_sigma[iid] = fut_vol.get(iid, 0.01) / np.sqrt(market.fut_ks)
    for p in fx_prev:
        mon.step_sigma[p] = fx_vol[p] / np.sqrt(market.n)
    limits_sod = L.evaluate(ctx.limits, ex_sod)
    if until == "SOD":
        checks = checks_fn(d, "SOD", ex_sod, limits_sod, mon, state, [], [s for s in state.settlements if s["value_date"] == d]) if checks_fn else []
        return DayResult(d, False, positions_sod, positions_sod, [], mon.alerts, [], [], {}, nav0, nav0, 0.0, ex_sod, ex_sod, {}, {}, limits_sod, [], [], [], [], {}, [], checks, mon, [], [])
    # ---- the day's events -----------------------------------------------------------------------------------------------------------
    events: list[dict] = []          # {"t", "kind", ...}; executed in time order

    def new_fill(strat, iid, qty, px, t, kind, cost=0.0):
        state.fill_seq += 1
        return {"id": f"F{state.fill_seq}", "strategy": strat, "instrument_id": iid, "qty": float(qty), "px": float(px), "time": float(t), "kind": kind, "cost": float(cost)}

    # equity deltas from execution-ops' targets
    tg = ctx.targets[ctx.targets["date"] == d]
    target = {(r.strategy, r.symbol): float(r.target) for r in tg.itertuples()}
    eq_strats = sorted(set(tg["strategy"]))
    for (strat, iid), q in list(positions_sod.items()):
        if insts[iid].asset_class == B.EQUITY and strat in eq_strats and (strat, iid) not in target:
            target[(strat, iid)] = 0.0
    slices = []
    for (strat, iid), q in target.items():
        delta = q - positions_sod.get((strat, iid), 0.0)
        if abs(delta) < 1 or iid not in insts:
            continue
        for (t, qq, px) in market.equity_slices(iid, delta):
            f = new_fill(strat, iid, qq, px, t, "equity"); slices.append(f); events.append({"t": t, "kind": "fill", "fill": f})
    # futures rolls at 14:00 New York
    t_roll = ny_time(d, "14:00"); roll_rows = []
    for (strat, old, new, lots) in rolls_due:
        events.append({"t": t_roll, "kind": "roll", "strategy": strat, "old": old, "new": new, "lots": lots}); mon.pending_rolls[old] = new
    # hedge and steepener sizing at 14:05 on sizing days
    t_size = ny_time(d, "14:05")
    if sizing_day:
        events.append({"t": t_size, "kind": "sizing"})
    # FX: maturing forwards settle at 10:00, the hedge is rebalanced at 10:05 on the first trading day of the week
    t_fx = ny_time(d, "10:00"); t_fxr = ny_time(d, "10:05")
    maturing = [iid for iid, q in state.forwards.items() if insts[iid].value_date <= d and abs(q) > 0]
    if maturing:
        events.append({"t": t_fx, "kind": "fx_settle", "ids": maturing})
    events.append({"t": t_fxr, "kind": "fx_rebalance", "live": False})       # every day on the open's exposure, and again at 15:30 on the live book (rebalance days move it)
    events.append({"t": ny_time(d, "15:30"), "kind": "fx_rebalance", "live": True})
    # settlement confirmations for instructions with value date today
    due = [s for s in state.settlements if s["value_date"] == d]
    for s in due:
        events.append({"t": float(rng.uniform(ny_time(d, "09:00"), ny_time(d, "11:30"))), "kind": "confirm", "key": s["key"]})
    mon.settlements = [dict(s, confirmed=False) for s in due]
    # RFQs
    rfq_specs = []
    bond_ids = [iid for iid in held if insts[iid].asset_class == B.BOND]
    for _ in range(cfg.rfq_trades_per_day if bond_ids else 0):
        iid = bond_ids[int(rng.integers(0, len(bond_ids)))]; inst = insts[iid]; side = 1 if rng.random() < 0.5 else -1
        par = float(np.round(rng.uniform(1e6, 5e6) if inst.rating_bucket == "IG" else rng.uniform(0.5e6, 2e6), -5))
        t = float(rng.uniform(ny_time(d, "09:30"), ny_time(d, "15:30"))); state.rfq_seq += 1
        rfq_specs.append({"id": f"RFQ{state.rfq_seq}", "time": t, "instrument_id": iid, "side": side, "par": par, "report_delay_s": float(np.exp(rng.normal(np.log(120.0), 0.6)))})
        events.append({"t": t, "kind": "rfq", "spec": rfq_specs[-1]})
    prints = CR.load_real_prints()
    prints = prints[prints["date"] == d] if prints is not None else CR.simulate_prints({iid: insts[iid] for iid in bond_ids}, bond_sod, d, ny_time(d, "08:00"), ny_time(d, "17:00"), rng)
    # prime-broker snapshots every 30 minutes
    for k in range(0, market.n + 1, 6):
        events.append({"t": market.grid[k], "kind": "pb_snapshot"})
    # faults
    us_open, us_close = ny_time(d, "09:30"), ny_time(d, "16:00")
    liquid = [iid for iid in held if insts[iid].asset_class == B.EQUITY and insts[iid].exchange == "XNYS" and iid in market.eq and market.eq[iid]["t_open"] is not None]
    big_eq = sorted(liquid, key=lambda s: -abs(sum(q for (st, i), q in positions_sod.items() if i == s) * prev_close.get(s, 0.0)))[:40]
    not_held = [s for s in ctx.uni["symbol"] if s in market.eq and s not in held and insts[s].exchange == "XNYS"]
    big_slices = sorted([f for f in slices if insts[f["instrument_id"]].exchange == "XNYS" and us_open + 1800 < f["time"] < us_close - 3600], key=lambda f: -abs(f["qty"] * f["px"]))[:40]
    day_info = {"t_open_us": us_open, "t_close_us": us_close, "t_sod": t_sod, "t_roll": t_roll, "t_settle_conf": ny_time(d, "09:00"), "liquid_equities": liquid, "big_equities": big_eq, "not_held_equities": not_held, "big_slices": big_slices,
                "held_futures": [iid for iid in held if insts[iid].asset_class == B.FUTURE], "sectors": {iid: insts[iid].sector for iid in big_eq}, "rolls_due": [r[1] for r in rolls_due], "settlements_due": [s["key"] for s in due], "rfqs": rfq_specs}
    faults = F.generate_schedule(day_info, rng, fault_mix) if fault_day else []
    faults = [f for f in faults if f.type != "bond_mark_stale" or bond_mark_stale]      # needs a previous morning to be stale against
    fmap = {}
    for f in faults:
        fmap.setdefault(f.type, []).append(f)
    if bond_mark_stale:
        for f in fmap.get("bond_mark_stale", []):
            f.armed = False; f.fired_at = t_sod
    skip_rolls = {f.scope for f in fmap.get("contract_not_rolled", [])}
    # ---- intraday ----------------------------------------------------------------------------------------------------------------------
    events.sort(key=lambda e: e["t"])
    fills: list[dict] = []; fx_trades: list[dict] = []; settled: list[dict] = []; rfqs: list[dict] = []; confirmed: set = set()
    stale_until: dict[str, float] = {}; stale_px: dict[str, float] = {}; spike_at: dict[str, tuple[float, float]] = {}; fx_stale_until: dict[str, float] = {}; fx_stale_val: dict[str, float] = {}; fx_invert_until: dict[str, float] = {}
    margin_params = ctx.mp; ei = 0; pending_snap = None
    truth = state.positions

    def truth_positions_by_iid() -> dict:
        out = {}
        for (s, i), q in truth.items():
            if abs(q) > 1e-9:
                out[i] = out.get(i, 0.0) + q
        return out

    def apply_fill(f: dict, to_monitor: bool = True, monitor_fill: dict | None = None):
        k = (f["strategy"], f["instrument_id"]); truth[k] = truth.get(k, 0.0) + f["qty"]; fills.append(f)
        if to_monitor:
            mon.on_fill(monitor_fill or f)

    opening_steps = {market.step_index(e["t_open"]) for e in market.eq.values() if e["t_open"] is not None}
    for k_step, t in enumerate(market.grid):
        clock.t = float(t)
        for f in faults:            # arm the ones due
            if f.armed and f.t <= t:
                f.armed = False; f.fired_at = float(t)
                if f.type == "stale_price":
                    stale_until[f.scope] = t + f.params["minutes"] * 60; stale_px[f.scope] = mon.last_px.get(f.scope, market.marks_at(t, [f.scope]).get(f.scope))
                elif f.type == "spiked_mark":
                    spike_at[f.scope] = (t, f.params["pct"])
                elif f.type == "fx_stale":
                    fx_stale_until[f.scope] = t + f.params["minutes"] * 60; fx_stale_val[f.scope] = mon.fx_last.get(f.scope, fx_prev[f.scope])
                elif f.type == "fx_inverted":
                    fx_invert_until[f.scope] = t + f.params["minutes"] * 60
                elif f.type == "wrong_multiplier":
                    live_master[f.scope].multiplier *= f.params["factor"]
                elif f.type == "sector_mislabel":
                    live_master[f.scope].sector = f.params["sector"]
                elif f.type == "margin_shortfall":
                    margin_params = MG.MarginParams(equity_long=f.params["equity_long"], equity_short=f.params["equity_short"])
                elif f.type == "rogue_fill":
                    px = market.equity_px(f.scope, t) or prev_close.get(f.scope, 100.0); inst = insts[f.scope]
                    side = 1.0 if sum(q for (st, i), q in truth.items() if i == f.scope) >= 0 else -1.0     # adds to the existing position
                    qty = side * np.floor(f.params["nav_frac"] * nav0 / (px * inst.price_scale * V.usd_per(inst.currency, fx_now)))
                    rf = new_fill("REV", f.scope, qty, px, t, "equity"); apply_fill(rf)
        # events due
        while ei < len(events) and events[ei]["t"] <= t:
            e = events[ei]; ei += 1; kind = e["kind"]
            if kind == "fill":
                f = e["fill"]; mf = f
                for fl in fmap.get("missing_fill", []):
                    if fl.params.get("fill_id") == f["id"]:
                        fl.fired_at = float(t); fl.armed = False; mf = None
                for fl in fmap.get("sign_flip", []):
                    if fl.params.get("fill_id") == f["id"]:
                        fl.fired_at = float(t); fl.armed = False; mf = dict(f, qty=-f["qty"])
                apply_fill(f, to_monitor=mf is not None, monitor_fill=mf)
            elif kind == "roll":
                if e["old"] in skip_rolls:
                    continue
                row = FU.execute_roll(e["strategy"], e["old"], e["new"], e["lots"], d, sett, ctx.specs, insts); roll_rows.append(row)
                if e["new"] not in live_master:
                    live_master[e["new"]] = _copy(insts[e["new"]])
                if e["new"] not in market.fut:
                    market.fut[e["new"]] = market.fut.get(e["old"])
                apply_fill(new_fill(e["strategy"], e["old"], -e["lots"], row["front_px"], t, "roll", row["crossing_usd"] / 2 + row["fees_usd"] / 2))
                apply_fill(new_fill(e["strategy"], e["new"], e["lots"], row["next_px"], t, "roll", row["crossing_usd"] / 2 + row["fees_usd"] / 2))
                mon.rolled.add(e["old"])
            elif kind == "sizing":
                for f in _sizing_fills(d, t, ctx, state, truth, market, ex_sod, fx_now, new_fill):
                    if f["instrument_id"] not in live_master:
                        live_master[f["instrument_id"]] = _copy(insts[f["instrument_id"]])
                    apply_fill(f)
                state.last_sizing = d
            elif kind == "fx_settle":
                fxr = market.fx_at(t)
                for iid in e["ids"]:
                    inst = insts[iid]; q = state.forwards[iid]; base, quote = B.CCY_OF_PAIR[inst.pair]
                    v0 = V.forward_value_usd(inst, q, d_prev, fx_prev, rates_prev); cash = q * (fxr[inst.pair] - inst.strike) * V.usd_per(quote, fxr)
                    state.cash += cash; state.cash_fx_today += cash; settled.append({"strategy": "FXHEDGE", "instrument_id": iid, "pnl": cash - v0, "cash": cash, "fix": fxr[inst.pair]})
                    truth[("FXHEDGE", iid)] = 0.0; state.forwards[iid] = 0.0
                    mon.positions[("FXHEDGE", iid)] = 0.0; mon.positions_sod.pop(("FXHEDGE", iid), None); mon.realised += cash - v0
            elif kind == "fx_rebalance":
                for tr in _fx_hedge_trades(d, t, ctx, state, truth, market, ex_sod, rng, live=e.get("live", False)):
                    fx_trades.append(tr); insts[tr["instrument_id"]] = tr["inst"]; live_master[tr["instrument_id"]] = _copy(tr["inst"])
                    apply_fill(new_fill("FXHEDGE", tr["instrument_id"], tr["qty"], tr["strike"], t, "fx_forward"))
                    state.forwards[tr["instrument_id"]] = state.forwards.get(tr["instrument_id"], 0.0) + tr["qty"]
                    for ins in FX.settlement_instructions(FX.Forward(tr["inst"], tr["qty"]), tr["inst"].value_date):
                        ins["key"] = f"{tr['instrument_id']}:{ins['ccy']}"; state.settlements.append(ins)
                state.last_fx_rebalance = d
            elif kind == "confirm":
                if any(fl.scope == e["key"] for fl in fmap.get("settlement_missing", [])):
                    for fl in fmap["settlement_missing"]:
                        if fl.scope == e["key"]:
                            fl.fired_at = max(float(t), ny_time(d, ctx.th.settlement_cutoff)); fl.armed = False     # a missing confirmation is only a fault once the cutoff has passed
                    continue
                confirmed.add(e["key"]); mon.on_settlement_confirmed(e["key"])
            elif kind == "rfq":
                sp = e["spec"]; inst = insts[sp["instrument_id"]]; mark = bond_sod[sp["instrument_id"]]
                tr = CR.rfq(inst, sp["side"], sp["par"], mark, t, cfg.fx_counterparties, rng); tr["id"] = sp["id"]; delay = sp["report_delay_s"]
                for fl in fmap.get("off_market_rfq", []):
                    if fl.scope == sp["id"]:
                        fl.fired_at = float(t); fl.armed = False; tr["price"] += sp["side"] * fl.params["points"]
                for fl in fmap.get("late_trace_report", []):
                    if fl.scope == sp["id"]:
                        fl.fired_at = float(t); fl.armed = False; delay = fl.params["delay_s"]
                pt = CR.post_trade(tr, inst, prints, delay); pt["date"] = d; rfqs.append(pt)
                apply_fill(new_fill("CREDIT", inst.instrument_id, sp["side"] * sp["par"], tr["price"], t, "rfq"))
                acc = -sp["side"] * sp["par"] * B.bond_accrued(inst, d) / 100.0; state.cash += acc; state.cash_bond_accrued_today += acc     # the accrued paid or received on top of the clean price
                mon.on_rfq(pt); events.append({"t": t + delay, "kind": "trace_report", "id": sp["id"]}); events.sort(key=lambda x: x["t"])
            elif kind == "trace_report":
                mon.on_rfq_report(e["id"], float(t))
            elif kind == "pb_snapshot":
                snap = truth_positions_by_iid()
                for fl in fmap.get("phantom_position", []):
                    if fl.fired_at is not None and fl.fired_at <= t <= fl.fired_at + 3600:
                        snap[fl.scope] = snap.get(fl.scope, 0.0) + fl.params["qty"]
                for fl in fmap.get("doubled_load", []):
                    if fl.fired_at is not None and fl.fired_at <= t <= fl.fired_at + 1800:
                        snap = {i: (2 * q if insts[i].asset_class == B.EQUITY else q) for i, q in snap.items()}
                pending_snap = snap
        # feeds with faults
        watch = sorted({i for (s, i), q in {**truth, **mon.positions}.items() if abs(q) > 1e-9} | set(fut_needed))
        truth_marks = market.marks_at(t, watch); feed = dict(truth_marks)
        for iid, until in stale_until.items():
            if t <= until and iid in feed and stale_px.get(iid) is not None:
                feed[iid] = stale_px[iid]
        for iid, (t_sp, pct) in spike_at.items():
            if t == t_sp and iid in feed:
                feed[iid] = feed[iid] * (1 + pct)
        fxr = market.fx_at(t); feed_fx = dict(fxr)
        for p, until in fx_stale_until.items():
            if t <= until:
                feed_fx[p] = fx_stale_val[p]
        for p, until in fx_invert_until.items():
            if t <= until:
                feed_fx[p] = 1.0 / fxr[p]
        mon.step(float(t), feed, feed_fx, market.rates_at(t), market, pb_snapshot=pending_snap, margin_params=margin_params, with_var=(k_step % 6 == 0), gap_step=(k_step in opening_steps)); pending_snap = None
    # ---- end of day -----------------------------------------------------------------------------------------------------------------------
    positions_eod = {k: v for k, v in truth.items() if abs(v) > 1e-9}
    held_eod = sorted({iid for (_, iid) in positions_eod} | set(held))
    marks_eod = {}
    for iid in held_eod:
        inst = insts[iid]
        if inst.asset_class == B.EQUITY:
            marks_eod[iid] = market.eq[iid]["close"] if iid in market.eq else marks_sod.get(iid)
        elif inst.asset_class == B.FUTURE:
            marks_eod[iid] = fut_now.get(iid, marks_sod.get(iid))
        elif inst.asset_class == B.BOND:
            marks_eod[iid] = bond_eod[iid]
        else:
            marks_eod[iid] = 1.0
    marks_eod = {k: v for k, v in marks_eod.items() if v is not None}
    # cash ledger: equity fills at the day's fix, futures settlement variation and costs, bond fills and coupons, financing
    cash_by_class = {"equity": 0.0, "future": 0.0, "bond": 0.0, "fx_forward": state.cash_fx_today, "financing": 0.0}
    for f in fills:
        inst = insts[f["instrument_id"]]
        if inst.asset_class == B.EQUITY:
            cash_by_class["equity"] -= f["qty"] * f["px"] * inst.price_scale * V.usd_per(inst.currency, fx_now)
        elif inst.asset_class == B.FUTURE:
            cash_by_class["future"] += f["qty"] * (marks_eod[f["instrument_id"]] - f["px"]) * inst.multiplier + f["cost"]
        elif inst.asset_class == B.BOND:
            cash_by_class["bond"] -= f["qty"] * f["px"] / 100.0
    for (strat, iid), q in positions_sod.items():
        inst = insts[iid]
        if inst.asset_class == B.FUTURE and iid in marks_eod:
            cash_by_class["future"] += q * (marks_eod[iid] - marks_sod[iid]) * inst.multiplier
        elif inst.asset_class == B.BOND:
            a0 = B.bond_accrued(inst, d_prev); a1 = B.bond_accrued(inst, d)
            if a1 < a0 - 1e-9:
                cash_by_class["bond"] += q * inst.coupon / 2.0 / 100.0
    state.cash += cash_by_class["equity"] + cash_by_class["future"] + cash_by_class["bond"]
    cash_by_class["bond"] += state.cash_bond_accrued_today          # already in the cash balance (booked at the RFQ); here for the breakdown
    ex_eod = X.compute(d, market.t1, positions_eod, insts, marks_eod, fx_now, rates_now, rm, nav0, ctx.betas, adv, fut_beta=ctx.fut_beta, factor_hist=ctx.factor_hist, idio_vol=ctx.idio_vol, cov=ctx.cov)
    fut_m = MG.futures_margin(positions_eod, insts, ctx.ranges, d, ctx.specs, cfg.roll_days_before); req = MG.requirement(ex_eod.positions, insts, fut_m, ctx.mp)
    coll_pre = MG.collateral(state.cash + float(ex_eod.positions.loc[ex_eod.positions["asset_class"].isin((B.EQUITY, B.BOND, B.FX_FORWARD)), "mv_usd"].sum()), ex_eod.positions)
    fin = MG.financing_cost(req, coll_pre, float(rates_now.get("SOFR", 4.0)), ctx.mp); state.cash += fin["total"]; cash_by_class["financing"] = fin["total"]
    nav1 = state.cash + float(ex_eod.positions.loc[ex_eod.positions["asset_class"].isin((B.EQUITY, B.BOND, B.FX_FORWARD)), "mv_usd"].sum())
    pos0_attr = {k: v for k, v in positions_sod.items() if not (insts[k[1]].asset_class == B.FX_FORWARD and k[1] in {s["instrument_id"] for s in settled})}
    spread_chg = {b: float(rm.cs.at[d, f"CS_{b}"]) if d in rm.cs.index and pd.notna(rm.cs.at[d, f"CS_{b}"]) else 0.0 for b in ("IG", "HY")}
    pnl_rows = PL.attribute(d, pos0_attr, fills, insts, marks_sod, marks_eod, fx_prev, fx_now, rates_prev, rates_now, rm, ctx.betas, spread_chg=spread_chg, financing=fin, settled_forwards=settled, d0=d_prev)
    gap = nav1 - nav0 - PL.total(pnl_rows)
    if abs(gap) > 1.0 and verbose:
        mv0 = r_mv0 = ex_sod.positions.groupby("asset_class")["mv_usd"].sum(); mv1 = ex_eod.positions.groupby("asset_class")["mv_usd"].sum(); att = PL.by(pnl_rows, 1)
        for cls in ("equity", "future", "bond", "fx_forward", "financing"):
            print(f"    identity {cls}: dmv {mv1.get(cls, 0.0) - mv0.get(cls, 0.0):+,.0f} cash {cash_by_class.get(cls, 0.0):+,.0f} sum {mv1.get(cls, 0.0) - mv0.get(cls, 0.0) + cash_by_class.get(cls, 0.0):+,.0f} vs attributed {att.get(cls, 0.0):+,.0f}")
    state.nav = nav1; state.positions = positions_eod; state.bond_marks_prev = bond_sod
    state.eod_marks = dict(marks_eod); state.eod_fx = dict(fx_now); state.eod_rates = dict(rates_now)
    state.settlements = [s for s in state.settlements if s["value_date"] > d]
    coll = MG.collateral(nav1, ex_eod.positions); util = req["total"] / max(coll["cash"], 1.0)
    margin_eod = {"requirement": req, "collateral": coll, "utilisation": util, "futures": fut_m}
    limits_eod = L.evaluate(ctx.limits, ex_eod, margin_util=util, intraday_pnl=nav1 - nav0)
    # settlement netting for today's value date
    inst_today = []
    for s in due:
        inst_today.append(dict(s, amount_usd=s["amount"] * V.usd_per(s["ccy"], fx_now), confirmed=s["key"] in confirmed))
    netting = FX.net_settlements(inst_today) if inst_today else {"gross_usd": 0.0, "bilateral_usd": 0.0, "multilateral_usd": 0.0, "instructions": 0, "by_ccy_net": {}}
    detections = F.score(faults, mon.alerts)
    checks = (checks_fn(d, "SOD", ex_sod, limits_sod, mon, state, roll_rows, inst_today) + checks_fn(d, "EOD", ex_eod, limits_eod, mon, state, roll_rows, inst_today, gap=gap, rfqs=rfqs)) if checks_fn else []
    if verbose:
        print(f"[{d}] {'fault' if fault_day else 'clean'}: nav {nav1 / 1e6:,.1f}m pnl {(nav1 - nav0) / 1e6:+.2f}m gap {gap:+.2f} fills {len(fills)} alerts {len(mon.alerts)} faults {sum(x.detected for x in detections)}/{len(detections)} rolls {len(roll_rows)} fx {len(fx_trades)} rfq {len(rfqs)} margin {100 * util:.0f}%", flush=True)
    return DayResult(d, fault_day, positions_sod, positions_eod, fills, mon.alerts, faults, detections, pnl_rows, nav0, nav1, gap, ex_sod, ex_eod, margin_eod, fin, limits_sod, limits_eod, roll_rows, fx_trades, inst_today, netting, rfqs, checks, mon, mon.pnl_path, mon.exposure_log)


def _copy(inst):
    import copy
    return copy.copy(inst)


def _sizing_fills(d, t, ctx, state, truth, market, ex_sod, fx_now, new_fill) -> list[dict]:
    """HEDGE: short index futures against the equity book's beta-dollars; RATES: the steepener in DV01; CREDIT: short
    ZN against the bonds' DV01.  Executed at the path plus a tick, fees per side."""
    cfg = ctx.cfg; rm = ctx.rm; sett = ctx.sett; specs = ctx.specs; out = []
    eq_beta = sum(v for k, v in ex_sod.beta_by_strategy.items() if k in ("MOM", "REV", "LOWVOL"))

    def trade(strat, root, target_lots):
        c = sett.should_hold(root, d, cfg.roll_days_before)
        if c not in ctx.insts:
            ctx.insts[c] = B.futures_instrument(c, specs)
        cur = sum(q for (s, i), q in truth.items() if s == strat and i == c)
        # anything the strategy still holds in another month of the same root is closed first (the roll should have done it)
        delta = float(np.round(target_lots - cur))
        if abs(delta) < 1:
            return
        px = market.futures_px(c, t)
        if px is None:
            p0, _ = sett.last_settle(c, d); px = p0
            if not np.isfinite(px):
                return
            market.fut[c] = np.full(market.fut_ks + 1, px)
        tick = float(specs.loc[root, "tick_size"]); fee = float(specs.loc[root, "fee_per_side_usd"])
        out.append(new_fill(strat, c, delta, px + np.sign(delta) * tick, t, "sizing", -fee * abs(delta)))

    es_px = market.futures_px(sett.should_hold("ES", d, cfg.roll_days_before), t) or 1.0; nq_px = market.futures_px(sett.should_hold("NQ", d, cfg.roll_days_before), t) or 1.0
    hedge_usd = -cfg.hedge_ratio * eq_beta
    trade("HEDGE", "ES", hedge_usd * cfg.hedge_split["ES"] / (es_px * 50.0 * ctx.fut_beta.get("ES", 1.0)))
    trade("HEDGE", "NQ", hedge_usd * cfg.hedge_split["NQ"] / (nq_px * 20.0 * ctx.fut_beta.get("NQ", 1.1)))
    trade("RATES", "ZT", cfg.steepener_dv01 / rm.treasury_dv01("ZT", d)); trade("RATES", "ZN", -cfg.steepener_dv01 / rm.treasury_dv01("ZN", d))
    if cfg.credit_hedge:
        bond_dv01 = sum(V.bond_dv01(ctx.insts[i], q, market.bond_px(i, t) or 100.0) for (s, i), q in truth.items() if s == "CREDIT" and ctx.insts[i].asset_class == B.BOND)
        trade("CREDIT", "ZN", -bond_dv01 / rm.treasury_dv01("ZN", d))
    for root, lots in cfg.macro_lots.items():
        trade("MACRO", root, lots)
    return out


def _fx_hedge_trades(d, t, ctx, state, truth, market, ex_sod, rng, live: bool = False) -> list[dict]:
    """sell the non-dollar equity exposure forward (EUR and GBP) to the 1-month date, best of four dealer quotes;
    on the open's exposure in the morning, on the live book in the afternoon"""
    cfg = ctx.cfg; fxr = market.fx_at(t); rates = market.rates_at(t); out = []
    for ccy, pair in (("EUR", "EURUSD"), ("GBP", "GBPUSD")):
        # what the equity book holds in that currency (the forwards themselves are excluded)
        if live:
            eq_usd = 0.0
            for (st, iid), q in truth.items():
                inst = ctx.insts.get(iid)
                if inst is not None and inst.asset_class == B.EQUITY and inst.currency == ccy and abs(q) > 0:
                    px = market.equity_px(iid, t)
                    if px is not None:
                        eq_usd += q * px * inst.price_scale * V.usd_per(ccy, fxr)
        else:
            eq_usd = float(ex_sod.positions.loc[(ex_sod.positions["asset_class"] == B.EQUITY) & (ex_sod.positions["currency"] == ccy), "notional_usd"].sum()) if ex_sod.positions is not None else 0.0
        target_base = -cfg.fx_hedge_ratio * eq_usd / fxr[pair]
        cur = sum(q for iid, q in state.forwards.items() if ctx.insts[iid].pair == pair and abs(q) > 0)
        delta = target_base - cur
        if abs(delta) * fxr[pair] < cfg.fx_rebalance_usd:
            continue
        spot = FX.spot_date(pair, d); vd = FX.forward_date(pair, spot, cfg.fx_tenor_months); days = (vd - d).days
        side = 1 if delta > 0 else -1
        quotes = FX.dealer_quotes(pair, fxr[pair], rates, days, side, cfg.fx_counterparties, rng); q = FX.best_quote(quotes, side)
        inst = B.forward_instrument(pair, vd, q.forward, q.counterparty, d)
        out.append({"date": d, "time": t, "instrument_id": inst.instrument_id, "inst": inst, "pair": pair, "qty": float(np.round(delta, -3)), "strike": q.forward, "spot": fxr[pair], "parity": V.forward_rate(pair, fxr[pair], rates, days), "value_date": vd, "counterparty": q.counterparty, "cover_bp": sorted(abs(x.points_bp - q.points_bp) for x in quotes)[1] if len(quotes) > 1 else 0.0, "days": days})
    return out
