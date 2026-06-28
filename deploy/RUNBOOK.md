# Runbook — the risk layer

This extends execution-ops' runbook (`deploy/RUNBOOK.md` there: the order monitor, the reconciliation, the
corporate-action checks). The risk layer runs on the same host, from the same start-of-day and end-of-day timers
(`deploy/xrisk-*.timer`): the risk start-of-day at 07:45 London after the order stack's, the intraday monitor from
08:00 London to 21:30 (the London open to after the US close, five-minute steps), the end-of-day at 22:30 after the
reconciliation. Logs are in `/var/log/xrisk/`, the store is `data/derived/xrisk.duckdb`; every alert, limit
evaluation, P&L row, margin row, roll, settlement instruction, RFQ and check is a row in it.

## Start of day (`xrisk sod`)

| check | fail or warn means | do |
|---|---|---|
| `marks_coverage` | positions without a mark, or a whole asset class missing | a feed did not arrive; do not size the hedges off yesterday's marks; fix the feed and rerun the SOD |
| `limits` | a limit breached at the open (the close's positions on this morning's marks and FX) | tell the portfolio manager before the day's orders add to it; the intraday monitor shows the same condition until it clears |
| `var` | informational: parametric and historical 99 % one-day VaR | if the two disagree by more than half, the last 250 days hold a regime the covariance does not: say so in the morning note |
| `futures_notice` | a contract within five days of first notice / last trade, or held past it | the roll is scheduled for `roll_days_before` CME days ahead; a `fail` means the contract is deliverable and must be closed before the session |
| `settlements_due` | FX instructions settle today | confirmations are expected by the cutoff (12:00 New York); see `SETTLEMENT_UNCONFIRMED` |
| `STALE_MARK` (alert at SOD) | the bond evaluated prices are yesterday's | the pricing file did not refresh; the CREDIT desk must not quote off it; reload before the first RFQ |

## During the day (monitor alerts, by severity)

Conditions (a stale price, a break, a breach, a call, an overdue roll, an unconfirmed settlement) are raised once
when they appear and again only after they clear; events (a spike, a jump, an off-market fill) every thirty
minutes at most per scope.

| alert | severity | meaning | action |
|---|---|---|---|
| `LIMIT_BREACH` | critical | a limit above 100 % (gross, net equity, beta-dollars, sector, single name, currency, DV01, CS01, VaR, futures per root, margin, liquidity, intraday loss) | the portfolio manager decides within 15 minutes: reduce, hedge, or a documented temporary increase; the alert stays open until the exposure is back inside |
| `LIMIT_WARN` | low | above 80 % of a limit | no action; it is the early warning for the breach |
| `MARGIN_CALL` | critical | the prime broker's and clearing requirement is above the cash collateral | a real call arrives from the prime broker: fund it from the T-bill line or reduce; a call the monitor raises before the broker's is the point of the rule |
| `POSITION_BREAK` | high | the OMS (start-of-day plus fills) and the prime broker's intraday snapshot disagree; `book` scope means many names at once (a position load), a name scope means one fill | one name: find the fill (missing, doubled, wrong side) with the broker and correct the OMS; book-wide: the prime broker reloaded positions; do not trade off either book until they agree |
| `MASTER_CHANGE` | high | an instrument's multiplier, sector or price scale changed since the open | reference data changed intraday: every exposure that uses it is wrong until it is reverted or confirmed; freeze the instrument in the OMS |
| `STALE_PRICE`, `FX_STALE` | medium | a price or an FX rate unchanged for 15 / 30 minutes while trading | the exposures and the intraday P&L on that name or currency are stale; check the feed; if the name is halted, mark it |
| `PRICE_SPIKE`, `FX_SPIKE` | high | a one-step move beyond six sigma (eight for FX) and the floor | a bad tick or an inverted rate corrupts every exposure: confirm against a second source before acting on any limit that fired with it |
| `PNL_JUMP` | high | the book's P&L moved more than five step-sigmas in five minutes outside an open | look at what moved: a spike or a fill through a limit shows up here first |
| `ROLL_OVERDUE`, `HELD_PAST_NOTICE` | high / critical | the scheduled roll did not execute by 14:30 New York; a contract is held on or after its notice date | roll now (calendar spread); past notice, close the position and call the clearing broker about delivery |
| `SETTLEMENT_UNCONFIRMED` | high | an FX instruction with today's value date not confirmed by the cutoff | chase the counterparty and the custodian; an unmatched instruction at the CLS cutoff fails and costs overdraft interest |
| `LATE_TRACE_REPORT` | medium | our TRACE report of an RFQ fill is more than 15 minutes after execution | report now; note the reason (FINRA Rule 6730) |
| `RFQ_OFF_MARKET` | high | a bond fill more than a point (IG) or two (HY) through the evaluated mark or the TRACE VWAP | confirm the price with the dealer before it books; a real print far from the mark is a mark problem, not a trade problem |
| `UNKNOWN_INSTRUMENT` | high | a fill on an instrument not in the master | set it up before the position is unmarked overnight |

Escalate to the portfolio manager when a `high` or `critical` alert is open for more than 15 minutes, to the prime
broker on any `POSITION_BREAK` that is not resolved by the next snapshot, and to the clearing broker on
`HELD_PAST_NOTICE`.

## End of day (`xrisk eod`)

| check | meaning | do |
|---|---|---|
| `pnl_identity` | the attributed P&L does not sum to the NAV change | never expected; the residual is by construction the remainder; a failure is a mark missing at one end of the day |
| `margin` | requirement against collateral at the close | above 80 %: fund it before the prime broker's call in the morning |
| `rolls` | the day's rolls with the spread paid; `warn` when the next month's settlement was not in the data | the roll cost per contract is booked as the spread plus a tick a leg plus fees |
| `settlements` | instructions value today, unconfirmed count | an unconfirmed instruction at the close is a fail: it did not settle |
| `rfq_post_trade` | flagged RFQs (off market, no prints, late report) | review each with the desk; the flags are reasons to look, not verdicts |
| `alerts` | high and critical alerts of the day | anything still open goes into the morning note |
