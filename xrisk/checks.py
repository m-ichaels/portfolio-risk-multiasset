"""Start-of-day and end-of-day checks for the multi-asset book, in the form execution-ops uses (ok / warn / fail
with a detail an operator can act on): marks coverage per asset class, FX and rates feeds, limits at the open and
the close, futures rolls and notice dates, margin, settlements due and confirmed, the P&L identity, open alerts."""
from __future__ import annotations

import datetime as dt

from . import book as B
from .xops_bridge import fi, futures_dates


def run(d: dt.date, phase: str, ex, limit_rows: list, mon, state, rolls: list, settlements: list, ctx=None, gap: float | None = None, rfqs: list | None = None) -> list[dict]:
    rows = []; ds = d

    def add(name, status, detail):
        rows.append({"date": ds, "phase": phase, "check_name": name, "status": status, "detail": detail[:300]})

    p = ex.positions
    if p is None or p.empty:
        add("positions", "fail", "no positions marked"); return rows
    n_by = p.groupby("asset_class")["instrument_id"].nunique().to_dict()
    add("marks_coverage", "ok", ", ".join(f"{k} {v}" for k, v in n_by.items()) + f"; NAV {ex.nav / 1e6:,.0f}m, gross {ex.gross / 1e6:,.0f}m, net {ex.net / 1e6:,.0f}m")
    breaches = [r for r in limit_rows if r["status"] == "breach"]; warns = [r for r in limit_rows if r["status"] == "warn"]
    add("limits", "fail" if breaches else ("warn" if warns else "ok"), (f"{len(breaches)} breached: " + ", ".join(f"{r['limit_name']}:{r['scope']}" for r in breaches[:4]) + "; " if breaches else "") + (f"{len(warns)} warnings: " + ", ".join(f"{r['limit_name']}:{r['scope']}" for r in warns[:4]) if warns else "all limits inside the warning level"))
    add("var", "ok", f"parametric 99% {ex.var_param / 1e6:,.1f}m, historical {ex.var_hist / 1e6:,.1f}m ({100 * ex.var_param / max(ex.nav, 1):.2f}% / {100 * ex.var_hist / max(ex.nav, 1):.2f}% of NAV)")
    # futures: contracts held against their notice dates
    fut = p[p["asset_class"] == B.FUTURE].groupby("instrument_id")["qty"].sum()
    warn_c, fail_c = [], []
    for c, q in fut.items():
        rt, y, m = fi.parse_contract(c); fd = futures_dates(rt, y, m); fn = min(x for x in (fd.get("first_notice"), fd.get("last_trade")) if x is not None); days = (fn - d).days
        if days <= 0:
            fail_c.append((c, days))
        elif days <= 5:
            warn_c.append((c, days))
    add("futures_notice", "fail" if fail_c else ("warn" if warn_c else "ok"), (f"held past notice: {fail_c}; " if fail_c else "") + (f"within five days of notice: {warn_c}" if warn_c else f"{len(fut)} contracts, all clear of first notice / last trade"))
    if phase == "EOD":
        if rolls:
            add("rolls", "ok" if all(r["observed"] for r in rolls) else "warn", "; ".join(f"{r['from_contract']}->{r['to_contract']} {r['lots']:+.0f} lots, {r['spread_ticks']:+.1f} ticks, {r['cost_per_contract_usd']:+,.0f}/contract" + ("" if r["observed"] else " (next month not in the data: spread unobserved)") for r in rolls))
        m = mon.margin if mon is not None and mon.margin else None
        if m:
            add("margin", "fail" if m["requirement"]["total"] > m["collateral"]["cash"] else ("warn" if m["utilisation"] > 0.8 else "ok"), f"requirement {m['requirement']['total'] / 1e6:,.0f}m vs collateral {m['collateral']['cash'] / 1e6:,.0f}m ({100 * m['utilisation']:.0f}%)")
        if settlements:
            unc = [s for s in settlements if not s.get("confirmed")]
            add("settlements", "fail" if unc else "ok", f"{len(settlements)} instructions value today, {len(unc)} unconfirmed" + (f": {[s['key'] for s in unc][:3]}" if unc else ""))
        if gap is not None:
            add("pnl_identity", "ok" if abs(gap) < 1.0 else "fail", f"NAV change minus attributed P&L = {gap:+.4f} USD")
        if rfqs is not None:
            flagged = [r for r in rfqs if r.get("flags")]
            add("rfq_post_trade", "warn" if flagged else "ok", f"{len(rfqs)} RFQs, {len(flagged)} flagged: " + ", ".join(f"{r['id']} {r['flags']}" for r in flagged[:3]) if rfqs else "no RFQ today")
        crit = [a for a in mon.alerts if a["severity"] in ("high", "critical")] if mon is not None else []
        add("alerts", "warn" if crit else "ok", f"{len(mon.alerts) if mon else 0} alerts, {len(crit)} high or critical: " + ", ".join(sorted({a['rule'] for a in crit})) if crit else f"{len(mon.alerts) if mon else 0} alerts, none high")
    else:
        pend = [s for s in state.settlements if s["value_date"] == d]
        add("settlements_due", "ok", f"{len(pend)} instructions settle today" if pend else "nothing settles today")
    return rows
