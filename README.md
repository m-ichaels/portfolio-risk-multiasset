# portfolio-risk-multiasset — risk monitor and multi-asset operations layer on execution-ops

**Question.** What does a systematic fund's book look like intraday, how fast is a limit breach, a bad feed, a position break or a margin call caught, and what changes when the book holds futures, FX forwards and corporate bonds rather than US equities alone? This layer extends [execution-ops](https://github.com/m-ichaels/execution-ops) (its equity book, calendars and fault-injection method) with positions-to-exposures, limits, intraday P&L attribution and margin on a virtual clock; futures rolls and first-notice handling with a SPAN-style margin; FX forwards with value dates, settlement instructions and CLS-style netting; and a credit RFQ post-trade against evaluated marks and TRACE-style prints. The risk monitor is fault-injected and scored exactly like the order monitor.

**Answer (§Results).**
- *Risk monitoring.* Over 27 fault days, 519 of 519 faults planted in the feeds, the fill stream, the prime broker's snapshots, the instrument master, the margin parameters, the roll schedule, the settlement confirmations and the bond marks were caught (100.0 %), median time to detect the same five-minute step (p90 25 min); on the 28 clean days in between the monitor raised 4.0 alerts a day, LIMIT_WARN 65, LIMIT_BREACH 47, LATE_TRACE_REPORT 1 (limit incidents the book itself causes, and the P&L rule on large days).
- *The book.* $2bn NAV: execution-ops' three equity strategies ($3.1bn gross), an index-futures hedge, a Treasury steepener, a commodity and yen book, $200m of corporate bonds hedged with ZN, and an FX-forward hedge; gross 2.11 x NAV, parametric one-day 99 % VaR 1.15 % of NAV (historical 1.31 %), exceeded on 5 of 55 days. P&L attributed by strategy, asset class and component reconciles to the NAV change on every day to 0.0000 USD.
- *Futures.* 9 rolls on the exchange's first-notice and last-trading rules; the roll P&L per contract (positive when the roll earns the calendar spread) was measured on 6J -616 (+96 ticks), CL +1,417 (-144 ticks), ES +3,334 (+269 ticks), GC -6,073 (+605 ticks), NQ +5,912 (+1185 ticks), ZN -283 (-16 ticks), ZT +256 (-35 ticks); 1 roll's next month was not listed on Yahoo, so its spread is unobserved. SPAN-style initial margin $15.0m on scan ranges calibrated to the data ($14.5m as the plain sum of outrights: the delivery charge inside the notice windows outweighs the one inter-commodity credit the book earns).
- *FX.* 75 forward trades hedging the non-dollar equity book, 70 settlement instructions on 15 value dates computed on four currency calendars; multilateral (CLS-style) netting settles $2,107m against $4,013m gross, a 47 % reduction; 8 instructions unconfirmed at the cutoff, all of them the injected fault.
- *Credit.* 220 RFQs to four dealers: mean markup 0.52 points (9.0 bp of yield) on investment grade and 0.78 points (22.9 bp) on high yield against the evaluated mark, cover 0.08 points; 104 fills flagged (off the mark or the TRACE VWAP, no prints, late report), the 54 planted ones among them.
- *Margin.* Requirement 38 % of the cash collateral on average, 40 % at worst on clean days; financing $+5.73m over the run.

Python package `xrisk` on DuckDB, 16 tests (marks per asset class, limits, the SPAN scan / spread / credit arithmetic, the attribution identity with fills, exposures and VaR scaling, the risk model's DV01s, front months and roll costs on the data, FX value dates on the four calendars and parity, netting, the RFQ post-trade, the fault scorer, one engine day with faults and a clean day after it), CI, systemd units and a runbook.

---

## Layout

| path | what |
|---|---|
| `xrisk/book.py`, `xrisk/valuation.py` | instruments (equity, future, FX forward, bond) and the stated book; marks to USD per asset class, covered-interest-parity forwards, 30/360 accrued, DV01 |
| `xrisk/riskmodel.py` | the risk model from the daily data: regional market betas, sector and style cross-sections, futures and FX factors, par-yield and credit-spread factors, Treasury-futures DV01 by regression, 250-day covariance |
| `xrisk/exposures.py`, `xrisk/limits.py` | positions to exposures (gross / net by class, strategy, sector, currency; beta-dollars; DV01 and CS01; factor vector; parametric and historical VaR; liquidity; top names) and the limit set with utilisation |
| `xrisk/pnl.py` | attribution by strategy, asset class and component, exact by construction |
| `xrisk/margin.py` | SPAN-style futures margin (sixteen scenarios, calendar-spread and delivery charges, inter-commodity credits, ranges calibrated on the data), prime-broker haircuts, forward variation margin, collateral, financing |
| `xrisk/futures.py` | front month and the roll rule on execution-ops' calendars, settlements from the listed contract or the continuous series with the check behind it, the roll and its cost, the roll calendar |
| `xrisk/fx.py` | USD, TARGET2, UK and Tokyo calendars; spot and forward value dates (modified following, end-end); dealer quotes; settlement instructions; gross, bilateral and multilateral netting |
| `xrisk/credit.py` | RFQs to four dealers, TRACE-style prints (real through the FINRA API when present), post-trade: markup in points and yield, cover, TRACE VWAP deviation, the 15-minute report |
| `xrisk/market.py` | the intraday market: Brownian bridges between real daily marks at five-minute steps, sessions per exchange, U-curve fills |
| `xrisk/monitor.py`, `xrisk/faults.py` | the risk monitor (seventeen rules, conditions and events) and the fault catalogue, schedule and scorer |
| `xrisk/engine.py`, `xrisk/run.py`, `xrisk/checks.py`, `xrisk/cli.py` | the day runner, the multi-day pipeline and the store, the SOD / EOD checks, `python -m xrisk run | sod | day | eod | futures-check | fx-check | margin` |
| `xrisk/xops_bridge.py`, `xrisk/_xops/` | execution-ops' calendar and first-notice modules: installed, the sibling checkout, or the vendored copy |
| `tools/download.py`, `tools/build_book.py` | the free data; the equity book from execution-ops' strategies |
| `deploy/` | systemd services and timers after execution-ops' own, `RUNBOOK.md` (every alert and check, what it means, what to do) |
| `tests/`, `.github/workflows/ci.yml`, `scripts/`, `notebooks/results.ipynb`, `report.pdf` | tests and CI; `run_all.sh`, `plots.py`, `summarize.py`, `report.py`; the notebook; the report |

Run: `pip install numpy pandas pyarrow duckdb matplotlib fpdf2 pytest`, then `scripts/run_all.sh` (the data is committed; the window takes about an hour) or `--quick` for four days. `--download` refreshes the data, `--build-book` regenerates the equity book from the sibling `../ProjectE`. Linux: copy `deploy/*.service` and `*.timer` to `/etc/systemd/system` next to execution-ops' and `systemctl enable --now xrisk-sod.timer xrisk-day.timer xrisk-eod.timer`.

---

## Data: what is real and what is simulated

| layer | source | real / simulated |
|---|---|---|
| equity book | execution-ops' momentum, reversal and low-vol strategies on its 190-name, 20-year panel, run from the sibling checkout (`tools/build_book.py`; London names sized in pounds and each strategy at its stated gross, two deliberate differences from execution-ops' own sizing) | real prices, stated strategies |
| futures settlements | Yahoo daily bars for every listed CME / CBOT / NYMEX / COMEX contract of ES, NQ, RTY, ZT, ZF, ZN, ZB, 6E, 6B, 6J, CL, GC (70 contracts) and the continuous `=F` series, two years | real; Yahoo drops expired contracts, so an expired front month is marked off the continuous series where the check shows it equals the front (§Futures) |
| FX | ECB euro reference rates since 1999 (EURUSD, GBPUSD, USDJPY derived) | real fixes |
| rates | SOFR (New York Fed), €STR (ECB), SONIA (Bank of England), JGB 1-year (MoF), the US Treasury par curve | real |
| bonds | iShares LQD and HYG holdings on 2026-09-17: evaluated price, modified duration, yield, coupon, maturity, sector for 4,502 CUSIPs; the book holds the 40 largest IG and 20 largest HY | real on that date; the daily mark path is modelled (§Method) |
| sectors | Yahoo's sector per equity | real |
| intraday paths, fills, the prime broker's snapshots, dealer quotes, TRACE prints, confirmations | generated (§Method) | simulated, stated |

FRED is unreachable from the build machine and FINRA's TRACE per-print data needs an API key (free registration); `tools/download.py trace` is written for it and skipped without the credentials.

---

## Method

**The book.** Equities from execution-ops (MOM $1bn capital at gross 2, REV $0.4bn, LOWVOL $0.6bn long-only). HEDGE: short ES and NQ (70 / 30) against 40 % of the equity book's beta-dollars, sized weekly with each contract's measured beta to SPY. RATES: a 2s10s steepener, long ZT / short ZN, $15k DV01 a leg. MACRO: long 150 CL, 60 GC, 100 6J. CREDIT: $150m par of IG and $50m of HY, pro rata to the ETFs' weights, hedged with short ZN on the bonds' DV01; four RFQs a day. FXHEDGE: sell the EUR and GBP equity value one month forward, checked twice a day (on the open's exposure at 10:05 New York, on the live book at 15:30, since rebalance days move it) and traded when the mismatch exceeds $20m, rolled at maturity. Everything rolls `roll_days_before` = 2 CME days ahead of first notice or last trade.

**Risk model.** Equities in two stages: a regional market factor $m_r$ (equal-weighted return of the region's names) with each stock's 250-day beta $\beta_i$, then a daily cross-section of the residuals on sector dummies and three style z-scores (size, momentum, volatility):

$$r_i = \beta_i\, m_{r(i)} + \sum_k x_{ik} f_k + \varepsilon_i .$$

Futures load on their own continuous return (ES, NQ, RTY, CL, GC), FX futures and forwards on spot, Treasury futures on the matched par-yield change with a DV01 per contract estimated by regressing the contract's daily dollar move on the yield change, bonds on the nearest par tenor and on a credit-spread factor per bucket backed out of LQD / HYG returns with the funds' holdings-weighted duration: $\Delta s = -10^4\, r_{\text{ETF}} / D - \Delta y$. With the factor exposure vector $x$ (dollars for return factors, dollars per bp for yield and spread factors) and the 250-day factor covariance $\Sigma$,

$$\sigma_p^2 = x^\top \Sigma x + \sum_i (w_i \sigma_i)^2, \qquad \text{VaR}_{99} = 2.326\,\sigma_p ,$$

and the historical VaR applies the last 250 days' actual instrument returns (bonds through their factors) to today's book.

**Marks and the intraday market.** Every daily mark is real: equity opens and closes, futures settlements, ECB fixes, par yields. Between two consecutive marks the path is a Brownian bridge in log price at five-minute steps with the instrument's trailing volatility; equities move inside their exchange session (before the open they sit at the previous close), futures until their 16:00 New York settlement, FX and rates across the 03:00–16:30 window. Fills execute on a U-shaped volume curve at the path plus half a spread; rolls at settlement plus a tick a leg and fees; forwards at the best of four dealer quotes around parity. Bond marks: iShares' evaluated price on the as-of date is the anchor, and the daily clean price is driven back and forward by $-D\,(\Delta y + \Delta s + \epsilon)/10^4$ with the real Treasury and ETF-implied spread moves and idiosyncratic yield noise of 1.5 bp (IG) and 4 bp (HY) a day. The desk sees yesterday's marks at the open and today's at the close.

**Exposures and limits.** Notional in USD per position (equities and bonds at market value, futures at price × multiplier, forwards at the base notional); gross and net by asset class, strategy, sector and currency; beta-dollars (equities to SPY, index futures with their measured beta); DV01 and CS01; days to liquidate at 20 % of ADV. Fourteen limits in one dataclass, each a dollar amount or a fraction of NAV; utilisation above 80 % is a warning, above 100 % a breach.

**P&L attribution.** Exact by construction: every component is defined and the last one is the remainder, so the sum equals the change in value the marks show. Equities: market $\text{mv}_0\,\beta_i\,m_r$, sector and style $\text{mv}_0\, x_{ik} f_k$ from the interval's cross-section, idio, FX translation $q\,p_1(u_1 - u_0)$, execution $q_f (p_1 - p_f) u_1$. Futures: settlement variation $q\,(s_1 - s_0)\,M$, execution (crossing and fees; the calendar spread paid on a roll is not a cash flow and is reported as the roll cost). Forwards: spot $q\,\Delta S\,u_{\text{quote}}$, forward points (the rest of the MTM change), execution (the dealer's price against parity), settled (the fixing). Bonds: carry (30/360 accrual and coupons), rates $-\text{DV01}\,\Delta y$, credit $-\text{DV01}\,\Delta s$, idio, execution. Financing: cash interest at SOFR − 25 bp on free cash and SOFR + 50 bp on a debit, stock borrow at 40 bp. The NAV is struck independently from the cash ledger and the marks, and the difference to the attributed total is the `pnl_identity` check.

**Margin.** Futures, SPAN-style: for each product the price scan range is the 99th percentile of the two-day dollar move per contract over the last 250 days, rounded to ticks, and the calendar-spread charge the same statistic on the front–next spread where two listed months overlap; the sixteen scenarios (price at $0, \pm\tfrac13, \pm\tfrac23, \pm 1$ of the range with volatility up and down, and $\pm 2$ ranges covered 35 %) are evaluated on the net position of each product,

$$\text{IM} = \sum_c \max_s(-\text{P\&L}_{c,s}) + \text{spread lots} \times \text{charge} + \text{delivery charge} - \text{inter-commodity credit},$$

with the delivery charge doubling the range inside the notice window and credits of 50 % (equity indices), 60 % (Treasuries) and 30 % (FX) on the smaller leg of an opposite-signed pair. The credit rates are this model's parameters, not CME's live ones. Equities: 15 % of long and 20 % of short market value plus 10 % on names above 4 % of the book. Bonds: 5 % (IG) and 15 % (HY) haircuts. Forwards: variation margin on the MTM, no initial margin (physically settled forwards are exempt under the uncleared-margin rules). Collateral is the fund's cash, $\text{NAV} - \text{physical MV} - \text{forward MTM}$; a margin call is a requirement above it.

**Futures operations.** The contract to hold is the exchange's front month (execution-ops' first-notice and last-trading rules for every root; 6B and 6J on the Euro FX rule; gold on its active months, skipping October) shifted two CME days earlier. A settlement comes from the listed contract's own history, else from Yahoo's continuous series when the contract is the front month. The roll closes the old month and opens the new one at the day's settlements; its P&L per contract is

$$c = -\operatorname{sign}(q)\,(P_{\text{next}} - P_{\text{front}})\,M - 2\,\text{tick}\,M - 2\,\text{fee},$$

negative when the roll pays the calendar spread (a long rolling into a dearer month, a short into a cheaper one) and positive when it earns it; the spread is observed only when both settlements are in the data, otherwise the new month is marked at the front's price on the roll day and the spread appears in the next day's settlement variation.

**FX operations.** Spot is T+2 with the currency-centre rule (T+1 must be a business day in the non-dollar centre, the value date in both), the one-month date modified following with the end-end rule, on the USD bank, TARGET2, UK and Tokyo calendars. Forwards are priced by covered interest parity on the overnight rates with each currency's day count,

$$F = S\,\frac{1 + r_q\,\tau_q}{1 + r_b\,\tau_b},$$

four dealers quote around it and the best wins. Each forward produces two settlement instructions (the base notional one way, the quote notional at the strike the other); on the value date the instructions are netted bilaterally (per counterparty and currency) and multilaterally (per currency, as CLS does), the reduction is $1 - \text{multilateral}/\text{gross}$, and every instruction expects a confirmation by 12:00 New York.

**Credit operations.** An RFQ names a bond of the book, a side and 0.5–5 million par; four dealers quote the evaluated mark plus a half spread that widens with size and differs by dealer; the best wins and the cover is the distance to the second best. Post-trade: the markup against the mark in points and in yield, $\Delta y = 10^4\,\Delta p / (D\,P)$; the deviation from the size-weighted TRACE prints of the same CUSIP within ±60 minutes (prints above 5 million IG and 1 million HY shown capped, as TRACE disseminates them); our own report's delay against the 15-minute rule.

**Monitor.** Every five minutes from the London open to after the US close: the price and FX feeds (stale for 15 / 30 minutes while trading; a one-step move beyond six sigma of the instrument's step volatility, eight for FX, excluding the first tick after an open), the OMS fill stream against the prime broker's half-hourly snapshot (a break per name; when more than five names differ at once it is one book-wide break, the snapshot's scaling against the OMS is estimated, and names that still differ after that scaling are raised on their own, so a missing fill is not hidden by a doubled load), the live instrument master against the open's, exposures and limits on the live master, P&L since the open on the same identity as the close (a one-step move beyond five step-sigmas outside an open), margin against collateral, the roll schedule after 14:30, notice dates, settlement confirmations after the cutoff, TRACE reports. Conditions are raised once when they appear and again only after they clear; events every thirty minutes at most per scope.

**Faults and scoring.** On fault days the generator plants nineteen faults from seventeen types: stale and spiked prices, a stale and an inverted FX rate, phantom positions, missing fills, a doubled position load, a fill booked the wrong way, a wrong multiplier and a wrong sector in the master, a rogue fill through the single-name limit, a roll that does not happen, a margin haircut jump, a settlement confirmation that never arrives, bond marks not refreshed, a late TRACE report, an off-market bond fill. A fault counts as detected when an alert of a matching rule for its scope arrives within an hour (four hours for the roll, the day for the marks); clean days measure the alerts the book itself causes.

---

## Results

Figures from `scripts/plots.py`; tables in `results/summary.md`; the run in `report.pdf`.

### The book

![book](results/figures/book.png)

| strategy | asset class | positions | gross $m | net $m |
|---|---|---|---|---|
| CREDIT | bond | 60 | 277 | +149 |
| CREDIT | future | 1 | 194 | -194 |
| FXHEDGE | fx_forward | 39 | 1,887 | -149 |
| HEDGE | future | 2 | 288 | -288 |
| LOWVOL | equity | 60 | 595 | +595 |
| MACRO | future | 3 | 50 | +50 |
| MOM | equity | 112 | 1,940 | +21 |
| RATES | future | 2 | 119 | +68 |
| REV | equity | 74 | 403 | +3 |

At the last close. Over the run: gross $4.21bn, net $+339m, beta-dollars $+397m after the hedge, DV01 $+536/bp; daily P&L mean $+0.04m, sd $16.19m, worst $-35.1m.

### Risk monitor under fault injection

![monitor](results/figures/monitor.png)

| fault | injected | caught | median time to detect | the monitor sees |
|---|---|---|---|---|
| stale_price | 54 | 54 | 10 min | STALE_PRICE (54) |
| spiked_mark | 54 | 54 | 0 s | PRICE_SPIKE (54) |
| phantom_position | 54 | 54 | 15 min | POSITION_BREAK (54) |
| missing_fill | 54 | 54 | 20 min | POSITION_BREAK (54) |
| fx_stale | 27 | 27 | 25 min | FX_STALE (27) |
| fx_inverted | 27 | 27 | 0 s | FX_SPIKE (27) |
| doubled_load | 27 | 27 | 15 min | POSITION_BREAK (27) |
| wrong_multiplier | 27 | 27 | 0 s | MASTER_CHANGE (27) |
| sector_mislabel | 27 | 27 | 0 s | MASTER_CHANGE (27) |
| rogue_fill | 27 | 27 | 0 s | LIMIT_BREACH (26), PNL_JUMP (1) |
| margin_shortfall | 27 | 27 | 0 s | MARGIN_CALL (27) |
| bond_mark_stale | 27 | 27 | 0 s | STALE_MARK (27) |
| late_trace_report | 27 | 27 | 20 min | LATE_TRACE_REPORT (27) |
| off_market_rfq | 27 | 27 | 0 s | RFQ_OFF_MARKET (27) |
| sign_flip | 24 | 24 | 25 min | POSITION_BREAK (24) |
| settlement_missing | 8 | 8 | 0 s | SETTLEMENT_UNCONFIRMED (8) |
| contract_not_rolled | 1 | 1 | 30 min | ROLL_OVERDUE (1) |

Alerts on clean days: 4.0 a day — LIMIT_WARN 65, LIMIT_BREACH 47, LATE_TRACE_REPORT 1 (limit incidents the book itself causes, and the P&L rule on large days).

### A fault day

![day](results/figures/day.png)

### P&L attribution

![pnl](results/figures/pnl.png)

| component | $m |
|---|---|
| bond:carry | +2.24 |
| bond:credit | +1.00 |
| bond:execution | -3.35 |
| bond:idio | +0.54 |
| bond:rates | -5.05 |
| equity:execution | +1.61 |
| equity:fx_translation | +1.40 |
| equity:idio | +75.91 |
| equity:market | +20.46 |
| equity:sector | +46.37 |
| equity:style | -157.58 |
| financing:borrow | -0.63 |
| financing:interest | +6.36 |
| future:execution | -0.11 |
| future:settlement_variation | +14.71 |
| fx_forward:execution | -0.82 |
| fx_forward:forward_points | -0.36 |
| fx_forward:settled | +1.12 |
| fx_forward:spot | -1.44 |

| strategy | $m |
|---|---|
| CREDIT | +0.95 |
| FUND | +5.73 |
| FXHEDGE | -1.50 |
| HEDGE | +0.91 |
| LOWVOL | +8.21 |
| MACRO | +8.19 |
| MOM | -86.46 |
| RATES | -0.07 |
| REV | +66.42 |

Total $+2.4m; the largest gap between the NAV change and the attributed total on any day is 0.0000 USD.

### Margin

![margin](results/figures/margin.png)

Requirement $487m on average against the cash collateral (38 % utilisation on clean days, 40 % at worst); the margin-shortfall fault (the prime broker doubling its haircuts to 60 %) takes it above 100 % on fault days and the monitor calls it. Futures: SPAN-style initial margin $15.0m a day on average = scan risk $14.5m + calendar-spread charges $0.00m (the roll days) + delivery charges $0.67m (contracts inside their notice window before the roll) - inter-commodity credit $0.20m (the steepener's ZT against ZN; the index hedge's ES and NQ are the same way round and earn none), against $14.5m as the plain sum of outright scan charges. Scan ranges per contract from the data: ES $10,325 (spread $400), NQ $22,480 (spread $790), RTY $5,440 (spread $200), ZT $734 (spread $102), ZF $719 (spread $23), ZN $1,062 (spread $47), ZB $2,031 (spread $62), 6E $2,156 (spread $62), 6B $1,212 (spread $50), 6J $1,950 (spread $56), CL $14,440 (spread $390), GC $36,280 (spread $310). Financing $+5.73m over the run (cash interest less stock borrow).

### Futures rolls

![futures](results/figures/futures.png)

| date | roll | lots | front | next | spread (ticks) | roll P&L $/contract | spread observed |
|---|---|---|---|---|---|---|---|
| 2026-07-20 00:00:00 | CLQ26->CLU26 | +150 | 83.2300 | 83.2300 | +0.0 | -23 | no |
| 2026-07-30 00:00:00 | GCQ26->GCZ26 | +60 | 4100.1001 | 4160.6001 | +605.0 | -6,073 | yes |
| 2026-08-19 00:00:00 | CLU26->CLV26 | +150 | 85.8300 | 84.3900 | -144.0 | +1,417 | yes |
| 2026-08-27 00:00:00 | ZTU26->ZTZ26 | +462 | 102.9766 | 102.8398 | -35.0 | +256 | yes |
| 2026-08-27 00:00:00 | ZNU26->ZNZ26 | -242 | 108.5781 | 108.3281 | -16.0 | -283 | yes |
| 2026-08-27 00:00:00 | ZNU26->ZNZ26 | -1743 | 108.5781 | 108.3281 | -16.0 | -283 | yes |
| 2026-09-11 00:00:00 | 6JU26->6JZ26 | +100 | 0.0065 | 0.0066 | +96.0 | -616 | yes |
| 2026-09-17 00:00:00 | ESU26->ESZ26 | -573 | 7640.0000 | 7707.2500 | +269.0 | +3,334 | yes |
| 2026-09-17 00:00:00 | NQU26->NQZ26 | -113 | 29446.7500 | 29743.0000 | +1185.0 | +5,912 | yes |

Yahoo's continuous series against the listed front month on the days both exist: ES 62/62; NQ 62/62; RTY 62/62; ZT 42/77; ZF 42/77; ZN 49/77; ZB 49/77; 6E 0/3; 6B 0/3; 6J 0/3; CL 19/19; GC 34/34. The substitution of the continuous series for an expired front month is used where that ratio is one (ES, NQ, RTY, CL); the Treasuries' listed months cover the window; the FX contracts' and gold's earlier front months are the continuous series on the evidence of the spread's smooth decay to the next month, stated as unverified.

### FX forwards and settlement

![fx](results/figures/fx.png)

75 forward trades, 70 settlement instructions on 15 value dates (T+2 and one month on the USD, TARGET2, UK and Tokyo calendars, modified following, end-end). Gross $4,013m; bilateral payment netting $3,160m; multilateral netting per currency $2,107m, a 47 % reduction. 8 instructions unconfirmed at the 12:00 cutoff, every one the injected fault.

| value date | ccy | instructions | gross | net |
|---|---|---|---|---|
| 2026-08-06 00:00:00 | EUR | 2 | 160,188,000 | 160,188,000 |
| 2026-08-06 00:00:00 | GBP | 2 | 203,005,000 | 203,005,000 |
| 2026-08-06 00:00:00 | USD | 4 | 450,682,114 | 85,731,796 |
| 2026-08-10 00:00:00 | GBP | 2 | 38,138,000 | 1,772,000 |
| 2026-08-10 00:00:00 | USD | 2 | 50,989,835 | 2,277,629 |
| 2026-08-13 00:00:00 | EUR | 1 | 20,835,000 | 20,835,000 |
| 2026-08-13 00:00:00 | GBP | 1 | 16,743,000 | 16,743,000 |
| 2026-08-13 00:00:00 | USD | 2 | 46,331,286 | 1,421,622 |
| 2026-08-14 00:00:00 | EUR | 1 | 17,828,000 | 17,828,000 |
| 2026-08-14 00:00:00 | USD | 1 | 20,403,406 | 20,403,406 |
| 2026-08-17 00:00:00 | EUR | 1 | 28,447,000 | 28,447,000 |
| 2026-08-17 00:00:00 | GBP | 2 | 39,461,000 | 5,777,000 |
| 2026-08-17 00:00:00 | USD | 3 | 85,451,326 | 40,206,252 |
| 2026-08-21 00:00:00 | GBP | 1 | 32,430,000 | 32,430,000 |
| 2026-08-21 00:00:00 | USD | 1 | 43,601,998 | 43,601,998 |
| 2026-08-24 00:00:00 | EUR | 1 | 24,399,000 | 24,399,000 |
| 2026-08-24 00:00:00 | GBP | 1 | 19,053,000 | 19,053,000 |
| 2026-08-24 00:00:00 | USD | 2 | 53,446,144 | 53,446,144 |
| 2026-08-28 00:00:00 | GBP | 1 | 16,642,000 | 16,642,000 |
| 2026-08-28 00:00:00 | USD | 1 | 22,198,157 | 22,198,157 |
| 2026-08-31 00:00:00 | EUR | 1 | 29,066,000 | 29,066,000 |
| 2026-08-31 00:00:00 | USD | 1 | 33,129,878 | 33,129,878 |
| 2026-09-04 00:00:00 | GBP | 1 | 30,804,000 | 30,804,000 |
| 2026-09-04 00:00:00 | USD | 1 | 41,390,875 | 41,390,875 |
| 2026-09-08 00:00:00 | EUR | 3 | 189,112,000 | 75,988,000 |
| 2026-09-08 00:00:00 | GBP | 3 | 168,469,000 | 60,093,000 |
| 2026-09-08 00:00:00 | USD | 6 | 445,271,891 | 6,933,793 |
| 2026-09-10 00:00:00 | EUR | 1 | 160,269,000 | 160,269,000 |
| 2026-09-10 00:00:00 | GBP | 1 | 203,310,000 | 203,310,000 |
| 2026-09-10 00:00:00 | USD | 2 | 458,583,491 | 88,219,839 |
| 2026-09-11 00:00:00 | EUR | 1 | 18,990,000 | 18,990,000 |
| 2026-09-11 00:00:00 | USD | 1 | 21,938,462 | 21,938,462 |
| 2026-09-14 00:00:00 | EUR | 2 | 46,255,000 | 7,845,000 |
| 2026-09-14 00:00:00 | GBP | 2 | 40,505,000 | 40,505,000 |
| 2026-09-14 00:00:00 | USD | 4 | 108,198,171 | 45,626,526 |
| 2026-09-17 00:00:00 | EUR | 2 | 49,121,000 | 49,121,000 |
| 2026-09-17 00:00:00 | GBP | 2 | 44,380,000 | 44,380,000 |
| 2026-09-17 00:00:00 | USD | 4 | 116,712,391 | 3,214,890 |

### Credit RFQ post-trade

![credit](results/figures/credit.png)

| bucket | trades | mean markup (pts) | median | mean markup (bp yield) | mean cover (pts) | mean dev. vs TRACE VWAP | with prints | flagged |
|---|---|---|---|---|---|---|---|---|
| HYG (HY) | 74 | 0.778 | 0.428 | 22.9 | 0.131 | 0.696 | 42 | 41 |
| LQD (IG) | 146 | 0.520 | 0.163 | 9.0 | 0.057 | 0.587 | 108 | 63 |

Flags: {'': 116, 'no_trace_prints': 49, 'off_market_vs_trace,off_market_vs_mark': 17, 'late_trace_report': 16, 'no_trace_prints,late_trace_report': 12, 'no_trace_prints,off_market_vs_mark': 9, 'off_market_vs_mark': 1}. Winning dealers: {'DLR1': 113, 'DLR2': 58, 'DLR3': 37, 'DLR4': 12} (the dealers differ in skill by construction, and the best quote wins). Late TRACE reports: 28, all the injected fault.

### Checks

![checks](results/figures/checks.png)

---

## Validation

- Marks: each asset class on hand-computed cases (pence through the scale and the fix, a futures notional, a forward against parity, 30/360 accrued, DV01).
- Limits: utilisation and status on constructed exposures.
- SPAN: an outright is charged the range per lot and the worst of the sixteen scenarios is the full range; a calendar spread nets the scan and pays the spread charge; an opposite-signed index pair earns the credit on the smaller leg; the delivery charge inside the notice window; the scan ranges from the data are whole ticks and ordered as they should be.
- Attribution: the identity on a mixed book with fills in each class to $10^{-6}$ relative; on the run, every day's gap is printed and checked (`pnl_identity`).
- Exposures and VaR: DV01 sign and size from the measured contract DV01, currency exposure from FX futures, the factor exposure's sign, VaR doubling with the book.
- Risk model: the four Treasury DV01s ordered and in range, a complete 250-day factor history, betas centred on one, a cross-section whose residuals average to zero.
- Futures: front months across the rule changes (first notice, gold's active months, the FX rule), the three settlement sources, the continuous-versus-front check exact for ES, NQ, RTY and CL, a roll's cost sign against its prices.
- FX: value dates across a US holiday, a UK bank holiday and the Tokyo new year, end-end and modified following, parity on both day counts; netting on a constructed set of instructions.
- Credit: the post-trade's flags each triggered on purpose.
- Fault scoring: detection inside and outside the window, scope matching, the clean-day rate.
- Engine: one fault day on the committed data with every planted fault caught and the identity exact, and a clean day after it with no fault-type alert.
- 16 tests and CI (the tests, both checks, a four-day pipeline with figures and summary, the morning job, the units).

## Traps stated

- *Fills, the prime broker, the dealers and TRACE prints are simulated;* prices, settlements, fixes, rates, the bond marks on one date, the calendars and Yahoo's own inconsistencies are real. The monitor's job is to catch what was planted, and the clean days say what it costs.
- *Expired futures are gone from Yahoo.* An expired front month is marked off the continuous series only because the check shows the series equals the listed front on every day both exist for ES, NQ, RTY and CL; for the Treasuries Yahoo's series switches at the volume roll about a week before first notice (the listed months cover the window, so it is not used), and for the FX contracts and gold the series tracks a different month, so their earlier front months are inferred from the smooth decay of the spread to the next month and stated as such. Rolls whose next month was not listed are reported with the spread unobserved.
- *The bond mark path is a model* anchored on one real evaluated price per CUSIP; its rates and credit components are real moves, its idiosyncratic part is noise. The RFQ markups are therefore a test of the post-trade logic, not a measurement of dealer behaviour.
- *Limits are set for the stated book.* Some are breached by construction on some days (a sector concentration of the momentum book, a currency mismatch on a rebalance day); those incidents are counted on clean days and not hidden.
- *SPAN credit rates and haircuts are this model's parameters.* The scan ranges are measured; the rest is stated in one dataclass.
- *The TRACE loader is untested against the live feed.*

## References

- CME Group, *SPAN Methodology* (scan ranges, intra- and inter-commodity spread charges, delivery charges); CME contract specifications for the twelve products.
- ISDA / BCBS-IOSCO, *Margin requirements for non-centrally cleared derivatives* (physically settled FX forwards exempt from initial margin).
- CLS Group, *Settlement service: multilateral netting*; ECB, *Euro foreign exchange reference rates*.
- FINRA Rule 6730 (TRACE reporting within 15 minutes) and the TRACE dissemination caps.
- Barra / Grinold & Kahn, *Active Portfolio Management* (the cross-sectional factor model and attribution).
- The Federal Reserve Bank of New York (SOFR), the ECB (€STR), the Bank of England (SONIA), the Japanese Ministry of Finance (JGB yields), the US Treasury (par yield curve); iShares (LQD and HYG holdings).
