# autoresearch-dlmm

Autonomous AI research for **Meteora DLMM** LP strategy optimization on Solana. Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

This repo is adapted from [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) for DLMM data prep, backtesting, strategy iteration, and now horizon-aware autonomous loops.

Give an AI agent a DLMM backtesting setup and let it experiment autonomously. It modifies the LP strategy, backtests against historical candle data, checks if net P&L improved, keeps or discards the change, and repeats. In `v2`, those experiments also build a lightweight memory so the agent can learn from prior runs instead of searching blind every time.

## How It Works

| File | Modified By | Purpose |
|------|-------------|---------|
| `prepare.py` | Nobody | Fetches candles from Meteora + top LP data from LP Agent API |
| `simulator.py` | Nobody | DLMM backtesting engine — bin-by-bin fees, IL, position management |
| `strategy.py` | **AI Agent** | LP strategy: bins, shape, rebalance logic, indicators |
| `backtest.py` | Nobody | Runs strategy, reports metrics, compares to top LP benchmarks |
| `loop.py` | Nobody | Runs the autonomous keep/revert loop for one selected pool |
| `memory.py` | Nobody | Stores structured experiment history and generates the v2 learning report |
| `config.py` | Nobody | Environment config, API key rotation, and horizon presets |
| `program.md` | **Human** | Instructions for the AI agent |

The agent iterates on `strategy.py` to maximize **val net_pnl_pct** — net profit (fees minus impermanent loss) as a percentage of initial capital on held-out validation data.

## Data Sources

### Meteora DLMM API (public, no auth)
- OHLCV candle data
- Volume history
- Pool metadata (bin step, fees, TVL)

### LP Agent API (key required)
- **Top LPers per pool** — ranked by PnL, with win rates, APR, hold times, fee capture
- **Wallet positions** — open and historical, with exact tick ranges, strategy types, P&L
- **Pool statistics** — on-chain metrics, position distributions

The top LP data gives the agent a benchmark to beat and features to learn from. Without LP Agent keys, the system still works — it just optimizes blind without benchmarks.

## Quick Start

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and install
git clone https://github.com/<your-username>/autoresearch-dlmm.git
cd autoresearch-dlmm
uv sync

# 3. Configure API keys
cp .env.example .env
# Edit .env and add your LP Agent API key(s)
# Get keys at: https://lpagent.mintlify.app/api-key-dashboard/dashboard

# 4. Fetch data (default horizon: swing)
uv run prepare.py

# 5. Run baseline backtest
uv run backtest.py

# 6. Run the autonomous loop
uv run loop.py --rounds 20 \
  --agent-cmd 'codex exec $(cat {prompt_file})'
```

## Horizon Modes

The repo is single-pool per run, but it is horizon-aware. Use `--horizon` to tell the system whether you are optimizing for short LPs or slow multi-day holds.

Supported modes:

- `scalp`: `5m` candles, 7-day lookback, benchmark LPs held roughly `0` to `6` hours
- `intraday`: `30m` candles, 14-day lookback, benchmark LPs held roughly `6` to `36` hours
- `swing`: `1h` candles, 30-day lookback, benchmark LPs held roughly `36` to `168` hours
- `7d_profile`: `1h` candles, 45-day lookback, benchmark LPs held roughly `72` to `240` hours

`7d_profile` is a multi-day hold preset, not a literal rolling 7-day evaluator. The strategy is tuned with slower defaults and longer-hold LP benchmarks, while the backtest still runs over the prepared history window. The old name `7d_hold` still works as a backward-compatible alias.

If you also pass `--days` or `--timeframe`, those explicit flags override the preset defaults.

## Common Commands

### Default run

```bash
uv run prepare.py
uv run backtest.py --split both

uv run loop.py --rounds 20 \
  --agent-cmd 'codex exec $(cat {prompt_file})'
```

### Multi-day meme pool

```bash
uv run prepare.py \
  --pool 81GpCm4d13y8TozYtThabuSCLQN2o3bbrvDogXFPn8sA \
  --horizon 7d_profile

uv run backtest.py \
  --pool 81GpCm4d13y8TozYtThabuSCLQN2o3bbrvDogXFPn8sA \
  --horizon 7d_profile \
  --split both

uv run loop.py \
  --pool 81GpCm4d13y8TozYtThabuSCLQN2o3bbrvDogXFPn8sA \
  --horizon 7d_profile \
  --rounds 25 \
  --sleep-seconds 300 \
  --agent-cmd 'codex exec $(cat {prompt_file})'
```

### Fast meme pool

```bash
uv run prepare.py --pool <POOL_ADDRESS> --horizon scalp
uv run backtest.py --pool <POOL_ADDRESS> --horizon scalp --split both
```

When you switch pools, run `prepare.py`, `backtest.py`, and `loop.py` with the same `--pool`. When you switch hold style, use the same `--horizon` across all three commands so cached candles, benchmark filtering, and strategy defaults stay aligned.

## Flag Reference

### `prepare.py`

- `--pool <POOL_ADDRESS>`: choose the pool to cache and benchmark
- `--horizon <MODE>`: one of `scalp`, `intraday`, `swing`, `7d_profile`
- `7d_hold` is accepted as an alias for `7d_profile`
- `--days <N>`: override the preset lookback window
- `--timeframe <5m|30m|1h|2h|4h|12h|24h>`: override the preset candle resolution
- `--top-wallets <N>`: number of top LP wallets to deep-dive for position history
- `--skip-lp`: skip LP Agent and fetch only Meteora market data

### `backtest.py`

- `--pool <POOL_ADDRESS>`: choose which cached pool to evaluate
- `--horizon <MODE>`: inject the horizon mode into strategy defaults and reporting
- `--split <train|val|both>`: choose which split(s) to run

### `loop.py`

- `--pool <POOL_ADDRESS>`: choose the pool to optimize
- `--horizon <MODE>`: choose the LP hold style
- `--rounds <N>`: number of autonomous edit/evaluate rounds
- `--sleep-seconds <N>`: wait time between rounds; use `300` for a five-minute cadence
- `--agent-cmd '<CMD>'`: shell command used to invoke your coding agent
- `--min-improvement <FLOAT>`: minimum validation improvement needed to keep a change
- `--prepare-if-missing`: auto-run `prepare.py` before the loop if the pool cache does not exist
- `--days <N>` / `--timeframe <...>`: only used with `--prepare-if-missing`, to override the horizon preset for bootstrap
- `--top-wallets <N>`: only used with `--prepare-if-missing`
- `--skip-lp`: only used with `--prepare-if-missing`

## API Key Rotation

LP Agent rate-limits to **5 requests per minute per key**. Add multiple keys for higher throughput:

```env
# .env
LPAGENT_API_KEYS=lpagent_key1,lpagent_key2,lpagent_key3
```

With 3 keys you get 15 RPM. The `KeyManager` in `config.py` automatically rotates across keys, tracks per-key call timestamps, and waits when all keys are rate-limited.

## Running the Agent

Point Claude Code, Codex, or any AI coding agent at the repo:

```
Have a look at program.md and let's kick off a new experiment! Let's do the setup first.
```

The agent will:
1. Read `program.md` for instructions
2. Run the baseline backtest
3. Examine benchmark data from top LPers
4. Read `experiments/learning_report.md` to see what has already worked or failed
5. Start iterating on `strategy.py` — one change at a time
6. Keep improvements, revert failures
7. Grow a memory of the pool through `experiments/history.jsonl`

## What Changed In v2

`v1` was the original autoresearch pattern adapted to DLMMs: modify `strategy.py`, run a backtest, compare metrics, keep or revert.

`v2` adds a learning layer on top of that loop:

- Every train/validation run is appended to `experiments/history.jsonl`
- Each record includes strategy parameters, pool config, benchmark context, and a market-regime summary
- `experiments/learning_report.md` is regenerated after each run with:
  - best validation run so far
  - recent experiments
  - which shapes have performed best on average
  - current LP benchmark context

This keeps the project lightweight while making the search process cumulative.

## Autonomous Loop

`loop.py` turns the repo into the full autoresearch workflow:

1. establish a baseline on the current `strategy.py`
2. hand the next-round prompt to a coding agent
3. run `backtest.py --split both`
4. keep the strategy change only if validation improves
5. revert regressions automatically
6. repeat for the requested number of rounds

It is single-pool by design in `v2`, which is usually what you want for meme pools because each pool has very different volatility, fee behavior, and rebalance patterns. The new `--horizon` flag lets the same repo behave very differently for `scalp` versus `7d_profile`.

Example:

```bash
uv run loop.py \
  --pool 9d9mb8kooFfaD3SctgZtkxQypkshx6ezhbKio89ixyy2 \
  --horizon scalp \
  --rounds 25 \
  --sleep-seconds 300 \
  --agent-cmd 'codex exec $(cat {prompt_file})'
```

Set `--sleep-seconds 300` if you want a five-minute cadence similar to the original training loop. Leave it at `0` if you want the backtest loop to iterate as fast as the agent can make changes.

## How v2 Was Built

This repo started from the `autoresearch-macos` idea of autonomous iteration, but the domain changed from language-model training to DLMM LP optimization.

The `v2` work added six pieces:

1. Live-data-safe Meteora integration.
   `prepare.py` now unwraps Meteora response envelopes, parses timestamps safely, and reads pool fee/bin config from `pool_config`.

2. Pool-correct benchmark loading.
   `backtest.py` injects the active pool's benchmark features into `strategy.py`, so `--pool` runs no longer read the default pool by mistake.

3. Structured experiment memory.
   `memory.py` captures the tunable strategy parameters, hashes the current strategy source, summarizes the train/val market regime, and appends normalized records to `experiments/history.jsonl`.

4. Agent-readable learning summaries.
   After every backtest, `backtest.py` regenerates `experiments/learning_report.md`, giving the agent a compact view of what has worked so far.

5. An autonomous loop runner.
   `loop.py` shells out to a coding agent, evaluates each proposed strategy change, and automatically keeps or reverts it based on validation performance.

6. Horizon-aware optimization.
   `config.py` now defines horizon presets, `prepare.py` filters LP benchmarks to matching hold times, `backtest.py` injects the selected horizon into strategy runtime context, and `strategy.py` adopts slower multi-day defaults for modes like `7d_profile` unless the agent has already tuned them away.

## What the Agent Experiments With

- **Liquidity shape**: spot vs curve vs bid_ask, dynamic switching based on regime
- **Bin count**: narrow concentrated vs wide passive, volatility-adaptive
- **Rebalance logic**: threshold, MA crossover, volume-spike, time-based triggers
- **Center price**: SMA, EMA, VWAP, momentum-adjusted
- **Indicators**: RSI, Bollinger Bands, ATR, volume profiles
- **Regime detection**: trending vs ranging market adaptation
- **Benchmark calibration**: matching hold times and shapes of top LPers
- **Exit conditions**: when to pull liquidity entirely

## Metrics

| Metric | Description |
|--------|-------------|
| **net_pnl_pct** | Primary. Net P&L as % of capital (fees - IL). Higher = better. |
| time_in_range_pct | % of candles where price was in range. Low = too narrow. |
| num_rebalances | Each costs gas IRL. Fewer is better at equal performance. |
| gross_apr / net_apr | Gap = how much IL eats your fees. |
| **Benchmark comparison** | Your strategy vs top LPers on the same pool. |

## Project Structure

```
config.py         — env loading, API key rotation
prepare.py        — data fetching (Meteora + LP Agent)
simulator.py      — DLMM backtesting engine
strategy.py       — LP strategy (agent modifies this)
backtest.py       — run + report + benchmark comparison
loop.py           — autonomous single-pool experiment runner
memory.py         — v2 experiment memory + learning report generation
program.md        — agent instructions
.env.example      — API key template
data/             — cached data (auto-created, gitignored)
experiments/      — run logs + history.jsonl + learning_report.md (auto-created, gitignored)
```

## Contributing

PRs welcome. The main areas for improvement:

- **Simulator fidelity** — model real bin liquidity distributions, dynamic fee spikes, slippage on rebalance
- **More data sources** — Birdeye price feeds, on-chain bin distributions via Helius
- **Multi-pool support** — optimize across multiple pools simultaneously
- **Live execution** — connect strategy output to LP Agent's Zap-In API for live deployment

## License

MIT
