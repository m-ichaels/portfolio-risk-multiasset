#!/usr/bin/env python3
"""report.pdf from results/summary.md and results/figures/*.png (fpdf2).   python scripts/report.py [results] [report.pdf]"""
import os
import sys

from fpdf import FPDF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "report.pdf")

INTRO = """Question. What does a systematic multi-asset book look like intraday, how fast is a limit breach, a bad feed, a position break or a margin call caught, and what changes when the book holds futures, FX forwards and corporate bonds rather than US equities alone? This layer extends execution-ops (its equity book, calendars and fault-injection method) with positions-to-exposures, limits, intraday P&L attribution and margin on a virtual clock, futures rolls and first-notice handling with a SPAN-style margin, FX forwards with value dates, settlement instructions and CLS-style netting, and a credit RFQ post-trade against evaluated marks and TRACE-style prints; the monitor is fault-injected and scored like the order monitor.

Method. Python package (xrisk) on DuckDB. The book: execution-ops' three equity strategies on $2bn, an index-futures hedge, a Treasury-futures steepener, a commodity and yen macro book, a corporate-bond book hedged with Treasury futures, and an FX-forward hedge of the non-dollar equity exposure. A risk model estimated from the daily data (regional market, sector and style factors for equities; own return factors for futures and FX; par-yield and credit-spread factors with measured DV01s for Treasury futures and bonds) gives exposures, parametric and historical VaR and the attribution. Every daily mark is real; the intraday path between two marks is a Brownian bridge. Fault days alternate with clean days.

Caveats. Fills, the prime broker, the dealers and TRACE prints are simulated; prices, settlements, FX fixes, rates, bond marks (one date), calendars and their inconsistencies are real. Yahoo lists futures contracts only while they trade, so earlier front months come from its continuous series where the check shows it equals the front. The SPAN credit rates and the margin haircuts are this model's parameters."""

FIGS = [("book.png", "The book: gross by strategy and asset class, exposures over the run, VaR against the daily loss, net sector exposure."),
        ("monitor.png", "Fault injection: faults injected and caught by type, time to detect, incidents per day on fault and clean days, alerts on clean days by rule."),
        ("day.png", "One fault day: P&L since the open with the faults and the alerts, then faults by type with the line to their detection and alerts by rule."),
        ("pnl.png", "P&L attribution: daily by component, cumulative by strategy, the equity components, and the identity gap per day."),
        ("margin.png", "Margin: requirement against collateral, the SPAN-style futures margin against the sum of outrights, the calibrated scan ranges, financing."),
        ("futures.png", "Futures rolls: cost per contract and calendar spread per roll, and the continuous-versus-front check on the data."),
        ("fx.png", "FX: net currency exposure after the hedge, settlement amounts gross versus bilateral versus multilateral, instructions by value date."),
        ("credit.png", "Credit RFQs: markups in points and in yield by bucket, markup against cover, fills against the TRACE VWAP."),
        ("checks.png", "Start-of-day and end-of-day check outcomes over the run.")]


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9); self.set_text_color(120); self.cell(0, 6, "portfolio-risk-multiasset - risk monitor and multi-asset operations layer on execution-ops", align="R"); self.ln(8); self.set_text_color(0)

    def footer(self):
        self.set_y(-12); self.set_font("Helvetica", "", 8); self.set_text_color(120); self.cell(0, 6, f"{self.page_no()}", align="C")


def clean(s):
    return (s.replace("–", "-").replace("—", "-").replace("−", "-").replace("×", "x").replace("≥", ">=").replace("≤", "<=").replace("…", "...").replace("²", "^2").replace("±", "+/-").replace("**", "").replace("`", "")
             .replace("→", "->").replace("≈", "~").replace("é", "e").replace("ö", "o").replace("'", "'").replace("’", "'"))


def md_table(pdf, rows):
    cols = [c.strip() for c in rows[0].strip("|").split("|")]
    data = [[clean(c.strip()) for c in r.strip("|").split("|")] for r in rows[2:]]
    n = len(cols); w = (pdf.w - 20) / n; fs = 6.5 if n <= 7 else 5.2; cut = 42 if n <= 7 else 24
    pdf.set_font("Helvetica", "B", fs)
    for c in cols:
        pdf.cell(w, 5, clean(c)[:cut], border=1)
    pdf.ln(5); pdf.set_font("Helvetica", "", fs)
    for r in data[:80]:
        if pdf.get_y() > pdf.h - 20:
            pdf.add_page()
        for c in r:
            pdf.cell(w, 4.5, c[:cut], border=1)
        pdf.ln(4.5)
    pdf.ln(2)


def main():
    pdf = PDF(); pdf.set_auto_page_break(auto=True, margin=15); pdf.add_page()
    pdf.set_font("Helvetica", "B", 16); pdf.cell(0, 10, "Portfolio Risk Monitor and Multi-Asset Operations Layer", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    for para in INTRO.split("\n\n"):
        pdf.multi_cell(0, 4.5, clean(para)); pdf.ln(2)
    for fn, cap in FIGS:
        p = os.path.join(R, "figures", fn)
        if not os.path.exists(p):
            continue
        if pdf.get_y() > pdf.h - 90:
            pdf.add_page()
        pdf.image(p, w=pdf.w - 20); pdf.set_font("Helvetica", "I", 8); pdf.multi_cell(0, 4, clean(cap)); pdf.ln(3); pdf.set_font("Helvetica", "", 9)
    sm = os.path.join(R, "summary.md")
    if os.path.exists(sm):
        pdf.add_page(); lines = open(sm, encoding="utf-8").read().splitlines(); i = 0
        while i < len(lines):
            l = lines[i]
            if l.startswith("## "):
                pdf.set_font("Helvetica", "B", 11); pdf.ln(2); pdf.cell(0, 7, clean(l[3:]), new_x="LMARGIN", new_y="NEXT"); pdf.set_font("Helvetica", "", 9); i += 1
            elif l.startswith("### "):
                pdf.set_font("Helvetica", "B", 9); pdf.cell(0, 6, clean(l[4:]), new_x="LMARGIN", new_y="NEXT"); pdf.set_font("Helvetica", "", 9); i += 1
            elif l.startswith("|"):
                j = i
                while j < len(lines) and lines[j].startswith("|"):
                    j += 1
                if j - i >= 2:
                    md_table(pdf, lines[i:j])
                i = j
            elif l.startswith("# "):
                i += 1
            elif l.strip():
                pdf.set_x(pdf.l_margin); pdf.multi_cell(0, 4.5, clean(l.strip())); i += 1
            else:
                i += 1
    pdf.output(OUT); print("wrote", OUT)


if __name__ == "__main__":
    main()
