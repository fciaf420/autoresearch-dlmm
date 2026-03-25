# autoresearch-dlmm

Autonomous AI research for **Meteora DLMM** LP strategy optimization on Solana. Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

This repo is adapted from [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) for DLMM data prep, backtesting, and strategy iteration.

Give an AI agent a DLMM backtesting setup and let it experiment autonomously. It modifies the LP strategy, backtests against historical candle data, checks if net P&L improved, keeps or discards the change, and repeats. In `v2`, those experiments also build a lightweight memory so the agent can learn from prior runs instead of searching blind every time.

## How It Works

| File | Modified By | Purpose |
|------|-------------|---------|
| `prepare.py` | Nobody | Fetches candles from Meteora + top LP data from LP Agent API |
| `simulator.py` | Nobody | DLMM backtesting engine — bin-by-bin fees, IL, position management |
| `strategy.py` | **AI Agent** | LP strategy: bins, shape, rebalance logic, indicators |
| `backtest.py` | Nobody | Runs strategy, reports metrics, compares to top LP benchmarks |
| `memory.py` | Nobody | Stores structured experiment history and generates the v2 learning report |
| `config.py` | Nobody | Environment config, API key rotation |
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

# 4. Fetch data (default: SOL/USDC, 30 days, 1h candles)
uv run prepare.py

# 5. Run baseline backtest
uv run backtest.py
```

### Custom Pools and Timeframes

```bash
# Different pool
uv run prepare.py --pool <POOL_ADDRESS>
uv run backtest.py --pool <POOL_ADDRESS>

# More history, different candles
uv run prepare.py --days 60 --timeframe 15m

# Analyze more top wallets for a custom pool
uv run prepare.py --pool <POOL_ADDRESS> --top-wallets 10

# Skip LP Agent data (Meteora only)
uv run prepare.py --skip-lp
```

When you switch pools, run both `prepare.py` and `backtest.py` with the same `--pool` so cached candles and LP benchmarks stay aligned.

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

## How v2 Was Built

This repo started from the `autoresearch-macos` idea of autonomous iteration, but the domain changed from language-model training to DLMM LP optimization.

The `v2` work added four pieces:

1. Live-data-safe Meteora integration.
   `prepare.py` now unwraps Meteora response envelopes, parses timestamps safely, and reads pool fee/bin config from `pool_config`.

2. Pool-correct benchmark loading.
   `backtest.py` injects the active pool's benchmark features into `strategy.py`, so `--pool` runs no longer read the default pool by mistake.

3. Structured experiment memory.
   `memory.py` captures the tunable strategy parameters, hashes the current strategy source, summarizes the train/val market regime, and appends normalized records to `experiments/history.jsonl`.

4. Agent-readable learning summaries.
   After every backtest, `backtest.py` regenerates `experiments/learning_report.md`, giving the agent a compact view of what has worked so far.

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
