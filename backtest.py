"""
backtest.py — Runs strategy against historical data and reports metrics.
Compares results against top LP benchmarks when available.
DO NOT MODIFY. The agent only modifies strategy.py.

Usage:
    uv run backtest.py                    # defaults
    uv run backtest.py --pool <address>   # specific pool
    uv run backtest.py --split val        # validation set only
    uv run backtest.py --split both       # train + val (default)
"""

import sys
import json
import time
import importlib
from pathlib import Path
from datetime import datetime

from tabulate import tabulate

import config
import memory
from prepare import load_data, split_data, compute_metrics, get_pool_runtime_config
from simulator import PoolConfig, run_backtest


def make_pool_config(pool_info: dict) -> PoolConfig:
    """Build PoolConfig from cached pool info."""
    runtime = get_pool_runtime_config(pool_info)
    return PoolConfig(
        bin_step_bps=runtime["bin_step"],
        base_fee_bps=runtime["base_fee_pct"],
        protocol_fee_pct=runtime["protocol_fee_pct"],
        name=runtime["name"],
    )


def run_one(candles, pool_config, strategy_fn, label, capital=1000.0):
    """Run backtest on a candle set and print results."""
    import strategy as strat_module
    if hasattr(strat_module, "reset"):
        strat_module.reset()

    t0 = time.time()
    raw_results = run_backtest(
        candles=candles, pool=pool_config,
        strategy_fn=strategy_fn, initial_capital=capital,
    )
    elapsed = time.time() - t0
    metrics = compute_metrics(raw_results)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    rows = [
        ["Net P&L (USD)", f"${metrics['net_pnl_usd']:+.4f}"],
        ["Net P&L (%)", f"{metrics['net_pnl_pct']:+.4f}%"],
        ["Total Fees", f"${metrics['total_fees_usd']:.4f}"],
        ["Total IL", f"${metrics['total_il_usd']:.4f}"],
        ["Gross APR", f"{metrics['gross_apr']:.1f}%"],
        ["Net APR", f"{metrics['net_apr']:.1f}%"],
        ["Rebalances", metrics['num_rebalances']],
        ["Time in Range", f"{metrics['time_in_range_pct']:.1f}%"],
        ["Fee/Rebalance", f"${metrics['fee_per_rebalance']:.4f}"],
        ["Final Portfolio", f"${metrics['final_portfolio_value_usd']:.2f}"],
        ["Backtest Time", f"{elapsed:.2f}s"],
    ]
    print(tabulate(rows, tablefmt="simple"))

    return metrics, raw_results


def print_benchmark_comparison(metrics: dict, lper_features: dict):
    """Compare backtest results against top LP benchmarks."""
    if not lper_features:
        return

    print(f"\n{'─'*60}")
    print(f"  BENCHMARK COMPARISON (vs Top LPers)")
    print(f"{'─'*60}")

    rows = []

    best_apr = lper_features.get("best_lp_apr")
    if best_apr is not None:
        diff = metrics["net_apr"] - best_apr
        status = "✓ BEATING" if diff > 0 else "✗ BEHIND"
        rows.append(["Best LP APR", f"{best_apr:.1f}%", f"{metrics['net_apr']:.1f}%",
                      f"{diff:+.1f}%", status])

    median_apr = lper_features.get("top_lp_median_apr")
    if median_apr is not None:
        diff = metrics["net_apr"] - median_apr
        status = "✓" if diff > 0 else "✗"
        rows.append(["Median LP APR", f"{median_apr:.1f}%", f"{metrics['net_apr']:.1f}%",
                      f"{diff:+.1f}%", status])

    median_wr = lper_features.get("top_lp_median_win_rate")
    if median_wr is not None:
        rows.append(["Median Win Rate", f"{median_wr:.1f}%", "—", "—", ""])

    best_wr = lper_features.get("best_lp_win_rate")
    if best_wr is not None:
        rows.append(["Best Win Rate", f"{best_wr:.1f}%", "—", "—", ""])

    median_age = lper_features.get("top_lp_median_avg_age_hour")
    if median_age is not None:
        rows.append(["Median Hold (hrs)", f"{median_age:.1f}", "—", "—", ""])

    best_positions = lper_features.get("best_lp_positions")
    if best_positions is not None:
        rows.append(["Best LP Positions", str(best_positions), "—", "—", ""])

    n_analyzed = lper_features.get("total_lpers_analyzed", 0)
    n_profitable = lper_features.get("profitable_lpers", 0)
    rows.append(["LPers Analyzed", str(n_analyzed), "—", "—", ""])
    rows.append(["Profitable LPers", str(n_profitable), "—", "—", ""])

    if rows:
        print(tabulate(rows, headers=["Metric", "Benchmark", "You", "Delta", ""],
                        tablefmt="simple"))


def save_results(
    metrics,
    raw_results,
    label,
    run_id,
    pool_address,
    pool_config,
    run_config,
    split_candles,
    strategy_module,
    lper_features=None,
):
    """Save experiment results."""
    config.EXPERIMENTS_DIR.mkdir(exist_ok=True)

    result_file = config.EXPERIMENTS_DIR / f"{run_id}_{label}.json"
    strategy_source = Path("strategy.py").read_text()
    strategy_params = memory.strategy_snapshot(strategy_module)

    log_entry = {
        "timestamp": run_id,
        "run_id": run_id,
        "pool_address": pool_address,
        "pool_name": pool_config.name,
        "label": label,
        "metrics": metrics,
        "strategy_params": strategy_params,
        "strategy_source": strategy_source,
        "benchmark": lper_features or {},
        "market_regime": memory.summarize_market_regime(split_candles),
        "rebalance_log": raw_results.get("rebalance_log", []),
    }

    with open(result_file, "w") as f:
        json.dump(log_entry, f, indent=2, default=str)

    history_record = memory.build_experiment_record(
        run_id=run_id,
        pool_address=pool_address,
        pool_config=pool_config,
        run_config=run_config,
        label=label,
        metrics=metrics,
        strategy_source=strategy_source,
        strategy_params=strategy_params,
        split_candles=split_candles,
        benchmark=lper_features,
        result_file=result_file,
    )
    history_file = memory.append_history(history_record)

    print(f"\nResults saved to {result_file}")
    print(f"Memory updated at {history_file}")
    return result_file


def main():
    pool_address = config.DEFAULT_POOL
    if "--pool" in sys.argv:
        pool_address = sys.argv[sys.argv.index("--pool") + 1]

    horizon_override = None
    if "--horizon" in sys.argv:
        horizon_override = sys.argv[sys.argv.index("--horizon") + 1]

    split_mode = "both"
    if "--split" in sys.argv:
        split_mode = sys.argv[sys.argv.index("--split") + 1]

    # Load all data
    print("Loading cached data...")
    pool_info, candles, volume, lper_features, top_lpers, lper_positions, run_config = \
        load_data(pool_address)
    if horizon_override:
        run_config = dict(run_config or {})
        run_config["horizon_mode"] = config.normalize_horizon_mode(horizon_override)
    else:
        run_config = dict(run_config or {})
        run_config["horizon_mode"] = config.normalize_horizon_mode(
            run_config.get("horizon_mode", config.HORIZON_MODE)
        )
    run_config["timeframe"] = run_config.get("timeframe", config.BACKTEST_TIMEFRAME)
    pool_config = make_pool_config(pool_info)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(
        f"Pool: {pool_config.name} | Bin Step: {pool_config.bin_step_bps} bps | "
        f"Base Fee: {pool_config.base_fee_bps:.3f}% | "
        f"Protocol Fee: {pool_config.protocol_fee_pct:.1f}%"
    )
    print(
        f"Horizon: {run_config['horizon_mode']} | "
        f"Prepared timeframe: {run_config.get('timeframe', 'unknown')}"
    )
    print(f"Candles: {len(candles)} total")
    if lper_features:
        print(f"Benchmark data: {lper_features.get('total_lpers_analyzed', 0)} LPers analyzed")

    train, val = split_data(candles)
    print(f"Train: {len(train)} | Val: {len(val)}")

    # Import strategy
    import strategy
    importlib.reload(strategy)
    if hasattr(strategy, "set_benchmark"):
        strategy.set_benchmark(lper_features)
    else:
        strategy.BENCHMARK = lper_features or {}
    if hasattr(strategy, "set_runtime_context"):
        strategy.set_runtime_context(run_config)
    strategy_fn = strategy.strategy

    print(f"\n--- Strategy Parameters ---")
    print(f"  SHAPE: {strategy.SHAPE}")
    print(f"  NUM_BINS: {strategy.NUM_BINS}")
    print(f"  REBALANCE_THRESHOLD: {strategy.REBALANCE_THRESHOLD}")
    print(f"  MIN_CANDLES_BETWEEN_REBALANCE: {strategy.MIN_CANDLES_BETWEEN_REBALANCE}")
    print(f"  MA_WINDOW: {strategy.MA_WINDOW}")
    print(f"  INITIAL_CAPITAL: ${strategy.INITIAL_CAPITAL}")
    if strategy.BENCHMARK:
        print(f"  BENCHMARK: loaded ({len(strategy.BENCHMARK)} features)")
    print(f"  MEMORY REPORT: {config.EXPERIMENTS_DIR / memory.LEARNING_REPORT}")

    # Run backtests
    if split_mode in ("train", "both"):
        train_metrics, train_raw = run_one(train, pool_config, strategy_fn, "TRAIN SET")
        save_results(
            train_metrics,
            train_raw,
            "train",
            run_id,
            pool_address,
            pool_config,
            run_config,
            train,
            strategy,
            lper_features,
        )

    if split_mode in ("val", "both"):
        val_metrics, val_raw = run_one(val, pool_config, strategy_fn, "VALIDATION SET")
        save_results(
            val_metrics,
            val_raw,
            "val",
            run_id,
            pool_address,
            pool_config,
            run_config,
            val,
            strategy,
            lper_features,
        )

        # Benchmark comparison
        print_benchmark_comparison(val_metrics, lper_features)

    report_path = memory.render_learning_report(pool_address)
    print(f"\nLearning report updated: {report_path}")

    if split_mode in ("val", "both"):
        print(f"\n{'='*60}")
        print(f"  >>> PRIMARY METRIC (val net_pnl_pct): {val_metrics['net_pnl_pct']:+.4f}% <<<")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
