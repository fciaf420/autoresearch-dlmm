# DLMM LP Strategy Autoresearch — Agent Program

You are an autonomous research agent optimizing a Meteora DLMM liquidity provision strategy. Your goal is to maximize **val net_pnl_pct** — the net profit (fees minus impermanent loss) as a percentage of initial capital, measured on the held-out validation set.

## Your Setup

- **`strategy.py`** — THE ONLY FILE YOU MODIFY. Contains LP strategy: parameters, indicators, rebalance logic, shape selection, position sizing.
- **`simulator.py`** — DLMM backtesting engine. DO NOT modify.
- **`prepare.py`** — Data fetching + utilities. DO NOT modify.
- **`backtest.py`** — Runs strategy, reports metrics, compares to benchmarks. DO NOT modify.
- **`loop.py`** — Autonomous keep/revert orchestrator. DO NOT modify.
- **`config.py`** — Environment config + API key rotation. DO NOT modify.

## Data Available

### Candle Data (Meteora API)
OHLCV candles for the pool, split into train/val sets. Available as columns: `open`, `high`, `low`, `close`, `volume`, `timestamp`.

### Top LP Benchmarks (LP Agent API)
The `BENCHMARK` dict in strategy.py is injected by `backtest.py` with features extracted from the top-performing LPers on the active pool. Use these to calibrate your strategy:

- `best_lp_apr` — APR of the #1 LPer by PnL
- `best_lp_win_rate` — their win rate across positions
- `best_lp_avg_age_hours` — how long they hold positions on average
- `best_lp_fee_pct` — fee capture as % of capital
- `top_lp_median_avg_age_hour` — median hold time across all profitable LPers
- `top_lp_median_win_rate` — median win rate
- `top_lp_median_fee_percent` — median fee capture rate
- `top_lp_p25_*` / `top_lp_p75_*` — quartile boundaries

### Top LP Positions (cached)
Raw position data for the top wallets is cached in `data/<pool>/lper_positions.json`. Each position includes: `strategyType`, `tickLower`/`tickUpper`, `collectedFee`, `pnl`, `inRange`, `createdAt`/`updatedAt`, `age`.

### Experiment Memory (v2)
Each backtest appends structured records to `experiments/history.jsonl` and refreshes `experiments/learning_report.md`.

- `history.jsonl` stores strategy parameters, pool context, market-regime features, and train/val metrics for every run.
- `learning_report.md` summarizes the best validation runs, recent experiments, and what shapes/parameter sets have worked so far on this pool.

### Autonomous Loop (v2)
`loop.py` can run the full autoresearch cycle automatically for one pool at a time:

1. Run a baseline backtest
2. Ask a coding agent to make one change to `strategy.py`
3. Re-run the backtest
4. Keep the change if val net_pnl_pct improves
5. Revert it if it does not
6. Repeat for N rounds

## The Experiment Loop

1. **Read** current `strategy.py`, `experiments/learning_report.md`, and latest results in `experiments/`
2. **Hypothesize** — form a specific hypothesis. Reference benchmark data.
3. **Modify** `strategy.py` with ONE targeted change
4. **Run** `uv run backtest.py --split both`
5. **Evaluate** — compare val net_pnl_pct to previous best. Check benchmark comparison and what the memory report says has or has not worked.
   - If improved → keep the change
   - If worse → revert
6. **Repeat**

## What You Can Change

### Parameters
- `NUM_BINS` — range width (1-1400). Compare against top LP hold times.
- `SHAPE` — "spot", "curve", "bid_ask". Check what `strategyType` top LPers use.
- `REBALANCE_THRESHOLD` — how far off-center before rebalancing (0.0-1.0)
- `MIN_CANDLES_BETWEEN_REBALANCE` — cooldown. Calibrate against top LP avg_age_hour.
- `MA_WINDOW`, `VOLATILITY_WINDOW` — indicator lookbacks
- `INITIAL_CAPITAL`

### Logic
- **Rebalance triggers** — threshold, volatility, MA crossover, volume-spike, time-based
- **Center price** — SMA, EMA, VWAP, momentum-adjusted
- **Dynamic bins** — adjust width based on volatility
- **Dynamic shape** — switch spot/curve/bid_ask by market regime
- **Indicators** — RSI, Bollinger, ATR, volume profiles
- **Regime detection** — trending vs ranging, high vs low volatility
- **Fee-aware rebalancing** — only rebalance if expected fees > IL cost
- **Exit conditions** — pull liquidity during extreme moves

### Using Benchmark Data
```python
# Example: calibrate hold time to match top LPers
median_hold = BENCHMARK.get("top_lp_median_avg_age_hour", 24)
MIN_CANDLES_BETWEEN_REBALANCE = max(int(median_hold), 1)

# Example: check if top LPers prefer specific shapes
# Look at lper_positions.json for strategyType distribution
```

## Key Concepts

- **Bins**: Discrete price points, `bin_step` bps apart. Active bin = market price.
- **Fees**: Only earned when swaps cross YOUR bins. Out of range = zero fees.
- **IL**: Step function per bin crossing. Concentrated liquidity amplifies IL.
- **Time in Range**: If price leaves range, you earn nothing. Wider = safer but less efficient.

## Metrics

**Primary: val net_pnl_pct** — higher is better.

Secondary:
- `time_in_range_pct` — low = range too narrow
- `num_rebalances` — each has real gas cost
- `fee_per_rebalance` — efficiency metric
- `gross_apr` vs `net_apr` — gap = IL eating fees

**Benchmark comparison** is printed automatically by backtest.py when LP data is available.
**Experiment memory** is refreshed automatically after every run.

## Rules

1. Only modify `strategy.py`
2. One change at a time
3. Always run `--split both` to check train and val
4. Watch for overfitting (train improves, val doesn't)
5. Read `experiments/learning_report.md` before each change and avoid repeating losing ideas
6. Keep the function signature: `strategy(state, pool, candle, candle_idx) -> dict`

```bash
# First experiment — baseline
uv run backtest.py --split both
```

Find alpha. Beat the benchmarks.
