"""python -m xrisk run | sod | day | eod | futures-check | fx-check | margin"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

from . import data


def _arg(name, default=None):
    if name in sys.argv:
        i = sys.argv.index(name); return sys.argv[i + 1] if i + 1 < len(sys.argv) else default
    return default


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "run":
        from .run import run_range
        start = dt.date.fromisoformat(_arg("--from", "2026-07-01")); end = dt.date.fromisoformat(_arg("--to", "2026-09-16"))
        out = run_range(start, end, fault_days=_arg("--faults", "alternate"), seed=int(_arg("--seed", "1")), results_dir=_arg("--results"))
        m = out["monitor"]; print(f"done: {m['detected']}/{m['injected']} faults detected, median ttd {m['median_ttd_s']}, {m.get('false_alerts_per_clean_day')} alerts per clean day; runtime {out['runtime_s']:.0f}s")
    elif cmd in ("sod", "day"):
        # the morning job and the trading day on one date, on the book the store holds from the previous close
        from . import run as R, engine as E, checks as CK, margin as MG
        d = dt.date.today() if _arg("--date", "today") == "today" else dt.date.fromisoformat(_arg("--date"))
        ctx = R.build_context(); bonds = data.load_bonds(); ctx.bond_paths = R.bond_mark_paths(ctx, bonds, d, d); ctx.ranges = MG.scan_ranges(data.load_futures(), ctx.specs, d)
        state = R.initial_state(ctx, d)
        r = E.run_day(d, ctx, state, fault_mix=None, seed=1, verbose=True, until="SOD" if cmd == "sod" else "EOD", checks_fn=lambda dd, ph, ex, lim, mon, st, rolls, settle, gap=None, rfqs=None: CK.run(dd, ph, ex, lim, mon, st, rolls, settle, ctx, gap=gap, rfqs=rfqs))
        if r is None:
            print(f"{d}: not a trading day"); return
        for c in r.checks:
            print(f"{c['phase']:3s} {c['status']:4s} {c['check_name']:22s} {c['detail']}")
        for a in r.alerts:
            print(f"    {a['severity']:8s} {a['rule']:22s} {a['scope']:24s} {a['detail']}")
        if cmd == "day":
            con = data.connect(); R.write_day(con, r, ctx.insts); con.close()
    elif cmd == "eod":
        d = dt.date.today() if _arg("--date", "today") == "today" else dt.date.fromisoformat(_arg("--date"))
        con = data.connect()
        for ph, st, name, detail in con.execute("select phase, status, check_name, detail from checks where date = ? order by phase desc, check_name", [d]).fetchall():
            print(f"{ph:3s} {st:4s} {name:22s} {detail}")
        for sev, rule, scope, detail in con.execute("select severity, rule, scope, detail from alerts where date = ? and severity in ('high', 'critical') order by time", [d]).fetchall():
            print(f"    {sev:8s} {rule:22s} {scope:24s} {detail}")
        con.close()
    elif cmd == "futures-check":
        from . import futures as FU
        sett = FU.Settlements(data.load_futures(), data.load_futures_specs())
        start = dt.date.fromisoformat(_arg("--from", "2025-09-01")); end = dt.date.fromisoformat(_arg("--to", "2026-09-18"))
        out = {"continuous_vs_front": sett.continuous_vs_front(start, end), "roll_calendar": FU.roll_calendar(sett, list(sett.specs.index), start, end, int(_arg("--roll-days", "2")))}
        for r in out["roll_calendar"]:
            r["front_px"], r["src_from"] = sett.settle(r["from_contract"], r["date"]); r["next_px"], r["src_to"] = sett.settle(r["to_contract"], r["date"])
            r["spread_ticks"] = (r["next_px"] - r["front_px"]) / float(sett.specs.loc[r["root"], "tick_size"]) if r["front_px"] == r["front_px"] and r["next_px"] == r["next_px"] else None
        data.to_json(out, os.path.join(data.RESULTS, "futures_check.json"))
        print(json.dumps(out["continuous_vs_front"], indent=1)); print(f"{len(out['roll_calendar'])} rolls, {sum(1 for r in out['roll_calendar'] if r['spread_ticks'] is not None)} with both settlements in the data")
    elif cmd == "fx-check":
        from . import fx as FX
        out = []
        for pair in ("EURUSD", "GBPUSD", "USDJPY"):
            for ds in ("2026-07-01", "2026-07-02", "2026-08-28", "2026-12-23", "2026-12-30", "2026-11-25"):
                d = dt.date.fromisoformat(ds); s = FX.spot_date(pair, d); f = FX.forward_date(pair, s, 1)
                out.append({"pair": pair, "trade": ds, "spot": s.isoformat(), "1M": f.isoformat()})
        data.to_json(out, os.path.join(data.RESULTS, "fx_check.json"))
        for r in out:
            print(r)
    elif cmd == "margin":
        from . import margin as MG
        d = dt.date.fromisoformat(_arg("--date", "2026-07-01")); r = MG.scan_ranges(data.load_futures(), data.load_futures_specs(), d)
        for k, v in r.items():
            print(f"{k:4s} scan {v['scan']:>10,.0f}  spread {v['spread']:>8,.0f}")
    else:
        print(__doc__)
