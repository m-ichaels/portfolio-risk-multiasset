# portfolio-risk-multiasset

A risk monitor and multi-asset operations layer on top of my [execution-ops](https://github.com/m-ichaels/execution-ops) stack, in a Python package (`xrisk`) on DuckDB. It adds exposures, limits, intraday P&L attribution and margin on a virtual clock, with futures rolls, FX forwards and corporate bonds alongside the US equity book. The monitor is fault-injected and scored the same way as execution-ops' order monitor.

The question: how fast is a limit breach, a bad feed, a position break or a margin call caught when the book holds futures, FX forwards and credit as well as equities?

## Results

- **Monitor.** Over 27 fault days the monitor caught 519 of 519 planted faults (100.0 %), median time to detect the same five-minute step (p90 25 min). On the 28 clean days it raised 4.0 alerts a day.
- **The book.** $2bn NAV across three equity strategies, an index-futures hedge, a Treasury steepener, commodity and yen futures, $200m of corporate bonds hedged with ZN and an FX-forward hedge. Gross is 2.11 x NAV and parametric one-day 99 % VaR is 1.15 % of NAV, exceeded on 5 of 55 days.
- **Attribution.** P&L by strategy, asset class and component reconciles to the NAV change on every day to 0.0000 USD.
- **Futures and margin.** 9 rolls on first-notice and last-trading rules. SPAN-style initial margin is $15.0m a day. Margin requirement is 38 % of cash collateral on average.
- **FX and credit.** Multilateral netting settles $2,107m against $4,013m gross, a 47 % reduction. 220 credit RFQs show a mean markup of 0.52 points on investment grade and 0.78 points on high yield against the evaluated mark.

![monitor](results/figures/monitor.png)
![pnl](results/figures/pnl.png)

All numbers are in `results/summary.md`; the detail is in [docs/method.md](docs/method.md).

## How it works

- **Book.** The equity book is execution-ops' three strategies. Futures, FX forwards and bonds are added and marked per asset class.
- **Intraday market.** Real daily marks joined by Brownian bridges at five-minute steps, with fills, prime-broker snapshots, dealer quotes and TRACE-style prints generated.
- **Risk.** Positions to exposures, DV01, factor exposures, parametric and historical VaR, and limits.
- **Monitor and faults.** Seventeen rules on a five-minute clock. A schedule plants faults in feeds, fills, snapshots, margin parameters, the roll schedule, settlement confirmations and bond marks, and scores detection.
- **Operations.** Futures rolls with a SPAN-style margin, FX value dates on four calendars with CLS-style netting, and a credit RFQ post-trade.

## Limitations

- Fills, the prime broker, the dealers and TRACE prints are simulated. Prices, settlements, fixes, rates and calendars are real. The monitor catches what was planted, so detection rates say little about real faults.
- The bond mark path is a model anchored on one real evaluated price per CUSIP, so the RFQ markups test the post-trade logic and measure nothing about dealers.
- SPAN credit rates and haircuts are this model's parameters. The TRACE loader is untested against the live feed.
- Expired futures are gone from Yahoo, so some earlier front months are inferred from the continuous series.

## Run

```
pip install numpy pandas pyarrow duckdb matplotlib pytest
scripts/run_all.sh --quick     # four days; without --quick the full window takes about an hour
python -m pytest -q            # 16 tests
```

CI runs the tests, the futures and FX checks, a short pipeline with figures and summary, the morning job and a systemd unit check. `README.md` and `docs/method.md` are written by `scripts/fill_readme.py`.
