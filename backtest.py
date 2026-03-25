"""
backtest.py — Runs strategy against historical data and reports metrics.
Compares results against top LP benchmarks when available.
DO NOT MODIFY. The agent only modifies strategy.py.

Usage:
    uv run backtest.py
    uv run backtest.py --pool <address>
    uv run backtest.py --split val
    uv run backtest.py --eval-mode rolling
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from datetime import datetime
from pathlib import Path

from tabulate import tabulate

import config
import memory
from prepare import (
    aggregate_window_metrics,
    compute_metrics,
    generate_rolling_windows,
    get_pool_runtime_config,
    load_data,
    split_data,
)
from simulator import PoolConfig, run_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DLMM backtests.")
    parser.add_argument("--pool", default=config.DEFAULT_POOL, help="Pool address to evaluate.")
    parser.add_argument(
        "--horizon",
        default=config.HORIZON_MODE,
        help=f"Horizon mode: {', '.join(sorted(config.HORIZON_PRESETS))}",
    )
    parser.add_argument(
        "--split",
        default="both",
        choices=["train", "val", "both"],
        help="Which data split(s) to evaluate.",
    )
    parser.add_argument(
        "--eval-mode",
        default=config.DEFAULT_EVAL_MODE,
        choices=["split", "rolling"],
        help="Evaluation mode: one continuous split path or rolling multi-start windows.",
    )
    parser.add_argument(
        "--window-hours",
        type=float,
        default=None,
        help="Rolling evaluation window length in hours.",
    )
    parser.add_argument(
        "--start-every-hours",
        type=float,
        default=None,
        help="Rolling evaluation step between entry starts in hours.",
    )
    parser.add_argument(
        "--objective",
        default=config.DEFAULT_EVAL_OBJECTIVE,
        choices=["balanced", "median", "mean", "worst"],
        help="Objective used to score rolling windows.",
    )
    return parser.parse_args()


def make_pool_config(pool_info: dict) -> PoolConfig:
    """Build PoolConfig from cached pool info."""
    runtime = get_pool_runtime_config(pool_info)
    return PoolConfig(
        bin_step_bps=runtime["bin_step"],
        base_fee_bps=runtime["base_fee_pct"],
        protocol_fee_pct=runtime["protocol_fee_pct"],
        name=runtime["name"],
    )


def primary_metric_name(metrics: dict) -> str:
    return metrics.get("primary_metric_name", "net_pnl_pct")


def primary_metric_value(metrics: dict) -> float:
    return float(metrics.get("primary_metric_value", metrics.get("net_pnl_pct", float("-inf"))))


def run_single_path(candles, pool_config, strategy_fn, capital=1000.0):
    """Run a single contiguous backtest path."""
    import strategy as strat_module

    if hasattr(strat_module, "reset"):
        strat_module.reset()

    raw_results = run_backtest(
        candles=candles,
        pool=pool_config,
        strategy_fn=strategy_fn,
        initial_capital=capital,
    )
    metrics = compute_metrics(raw_results)
    metrics.setdefault("eval_mode", "split")
    metrics.setdefault("primary_metric_name", "net_pnl_pct")
    metrics.setdefault("primary_metric_value", metrics["net_pnl_pct"])
    return metrics, raw_results


def run_rolling_windows(
    candles,
    pool_config,
    strategy_fn,
    timeframe: str,
    capital: float,
    window_hours: float,
    step_hours: float,
    objective: str,
):
    """Run many fixed-horizon windows across a candle history."""
    windows = generate_rolling_windows(candles, timeframe, window_hours, step_hours)
    if not windows:
        raise ValueError(
            f"Not enough candles for rolling evaluation. Need at least one {window_hours:g}h "
            f"window at timeframe {timeframe}, but only have {len(candles)} candles."
        )

    window_summaries = []
    for window_index, (start_idx, end_idx) in enumerate(windows, start=1):
        window_candles = candles.iloc[start_idx:end_idx].copy().reset_index(drop=True)
        metrics, raw_results = run_single_path(window_candles, pool_config, strategy_fn, capital=capital)
        window_summaries.append(
            {
                "window_index": window_index,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_timestamp": str(window_candles.iloc[0]["timestamp"]),
                "end_timestamp": str(window_candles.iloc[-1]["timestamp"]),
                "metrics": metrics,
                "num_rebalances": raw_results.get("num_rebalances", 0),
            }
        )

    aggregated = aggregate_window_metrics(
        [window["metrics"] for window in window_summaries],
        objective=objective,
        window_hours=window_hours,
        step_hours=step_hours,
    )
    raw_results = {
        "evaluation": {
            "mode": "rolling",
            "objective": objective,
            "window_hours": float(window_hours),
            "step_hours": float(step_hours),
            "window_count": len(window_summaries),
        },
        "rolling_windows": window_summaries,
        "rebalance_log": [],
    }
    return aggregated, raw_results


def render_rows(metrics: dict) -> list[list[object]]:
    """Render either split or rolling metrics for CLI output."""
    if metrics.get("eval_mode") == "rolling":
        return [
            ["Primary Score", f"{primary_metric_value(metrics):+.4f}%"],
            ["Objective", metrics.get("rolling_objective", "balanced")],
            ["Windows", metrics.get("window_count", 0)],
            ["Window Hours", metrics.get("window_hours", 0)],
            ["Start Every", f"{metrics.get('step_hours', 0)}h"],
            ["Median Net P&L (%)", f"{metrics.get('median_net_pnl_pct', 0.0):+.4f}%"],
            ["Average Net P&L (%)", f"{metrics.get('avg_net_pnl_pct', 0.0):+.4f}%"],
            ["P25 Net P&L (%)", f"{metrics.get('p25_net_pnl_pct', 0.0):+.4f}%"],
            ["Worst Net P&L (%)", f"{metrics.get('worst_net_pnl_pct', 0.0):+.4f}%"],
            ["Latest Window (%)", f"{metrics.get('latest_window_net_pnl_pct', 0.0):+.4f}%"],
            ["Win Rate", f"{metrics.get('win_rate_pct', 0.0):.1f}%"],
            ["Average APR", f"{metrics.get('net_apr', 0.0):.1f}%"],
            ["Average Time in Range", f"{metrics.get('avg_time_in_range_pct', 0.0):.1f}%"],
            ["Average Rebalances", metrics.get("avg_num_rebalances", 0.0)],
            ["Average Portfolio", f"${metrics.get('final_portfolio_value_usd', 0.0):.2f}"],
        ]

    return [
        ["Net P&L (USD)", f"${metrics['net_pnl_usd']:+.4f}"],
        ["Net P&L (%)", f"{metrics['net_pnl_pct']:+.4f}%"],
        ["Total Fees", f"${metrics['total_fees_usd']:.4f}"],
        ["Total IL", f"${metrics['total_il_usd']:.4f}"],
        ["Gross APR", f"{metrics['gross_apr']:.1f}%"],
        ["Net APR", f"{metrics['net_apr']:.1f}%"],
        ["Rebalances", metrics["num_rebalances"]],
        ["Time in Range", f"{metrics['time_in_range_pct']:.1f}%"],
        ["Fee/Rebalance", f"${metrics['fee_per_rebalance']:.4f}"],
        ["Final Portfolio", f"${metrics['final_portfolio_value_usd']:.2f}"],
    ]


def run_one(candles, pool_config, strategy_fn, label, eval_config, capital=1000.0):
    """Run either a classic split backtest or a rolling-window evaluation."""
    t0 = time.time()

    if eval_config["mode"] == "rolling":
        metrics, raw_results = run_rolling_windows(
            candles=candles,
            pool_config=pool_config,
            strategy_fn=strategy_fn,
            timeframe=eval_config["timeframe"],
            capital=capital,
            window_hours=eval_config["window_hours"],
            step_hours=eval_config["step_hours"],
            objective=eval_config["objective"],
        )
    else:
        metrics, raw_results = run_single_path(candles, pool_config, strategy_fn, capital=capital)

    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(tabulate(render_rows(metrics) + [["Backtest Time", f"{elapsed:.2f}s"]], tablefmt="simple"))
    return metrics, raw_results


def print_benchmark_comparison(metrics: dict, lper_features: dict):
    """Compare backtest results against top LP benchmarks."""
    if not lper_features:
        return

    print(f"\n{'─'*60}")
    print("  BENCHMARK COMPARISON (vs Top LPers)")
    print(f"{'─'*60}")

    rows = []
    your_apr = metrics.get("net_apr", 0.0)

    best_apr = lper_features.get("best_lp_apr")
    if best_apr is not None:
        diff = your_apr - best_apr
        status = "✓ BEATING" if diff > 0 else "✗ BEHIND"
        rows.append(["Best LP APR", f"{best_apr:.1f}%", f"{your_apr:.1f}%", f"{diff:+.1f}%", status])

    median_apr = lper_features.get("top_lp_median_apr")
    if median_apr is not None:
        diff = your_apr - median_apr
        status = "✓" if diff > 0 else "✗"
        rows.append(["Median LP APR", f"{median_apr:.1f}%", f"{your_apr:.1f}%", f"{diff:+.1f}%", status])

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

    rows.append(["LPers Analyzed", str(lper_features.get("total_lpers_analyzed", 0)), "—", "—", ""])
    rows.append(["Profitable LPers", str(lper_features.get("profitable_lpers", 0)), "—", "—", ""])

    if rows:
        print(tabulate(rows, headers=["Metric", "Benchmark", "You", "Delta", ""], tablefmt="simple"))


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
        "evaluation": raw_results.get("evaluation", {}),
        "rebalance_log": raw_results.get("rebalance_log", []),
    }
    if raw_results.get("rolling_windows"):
        log_entry["rolling_windows"] = raw_results["rolling_windows"]

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
    args = parse_args()

    print("Loading cached data...")
    pool_info, candles, volume, lper_features, top_lpers, lper_positions, run_config = load_data(args.pool)
    run_config = dict(run_config or {})

    horizon = config.resolve_horizon_settings(args.horizon)
    run_config["horizon_mode"] = config.normalize_horizon_mode(args.horizon)
    run_config["timeframe"] = run_config.get("timeframe", horizon["timeframe"])
    run_config["eval_mode"] = args.eval_mode
    run_config["eval_objective"] = args.objective
    run_config["eval_window_hours"] = float(
        args.window_hours if args.window_hours is not None else horizon["eval_window_hours"]
    )
    run_config["eval_step_hours"] = float(
        args.start_every_hours if args.start_every_hours is not None else horizon["eval_step_hours"]
    )

    pool_config = make_pool_config(pool_info)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(
        f"Pool: {pool_config.name} | Bin Step: {pool_config.bin_step_bps} bps | "
        f"Base Fee: {pool_config.base_fee_bps:.3f}% | Protocol Fee: {pool_config.protocol_fee_pct:.1f}%"
    )
    print(
        f"Horizon: {run_config['horizon_mode']} | Prepared timeframe: {run_config.get('timeframe', 'unknown')}"
    )
    print(
        f"Evaluation: {run_config['eval_mode']} | "
        f"Window: {run_config['eval_window_hours']}h | "
        f"Step: {run_config['eval_step_hours']}h | "
        f"Objective: {run_config['eval_objective']}"
    )
    print(f"Candles: {len(candles)} total")
    if lper_features:
        print(f"Benchmark data: {lper_features.get('total_lpers_analyzed', 0)} LPers analyzed")

    train, val = split_data(candles)
    print(f"Train: {len(train)} | Val: {len(val)}")

    import strategy

    importlib.reload(strategy)
    if hasattr(strategy, "set_benchmark"):
        strategy.set_benchmark(lper_features)
    else:
        strategy.BENCHMARK = lper_features or {}
    if hasattr(strategy, "set_runtime_context"):
        strategy.set_runtime_context(run_config)
    strategy_fn = strategy.strategy

    print("\n--- Strategy Parameters ---")
    print(f"  SHAPE: {strategy.SHAPE}")
    print(f"  NUM_BINS: {strategy.NUM_BINS}")
    print(f"  REBALANCE_THRESHOLD: {strategy.REBALANCE_THRESHOLD}")
    print(f"  MIN_CANDLES_BETWEEN_REBALANCE: {strategy.MIN_CANDLES_BETWEEN_REBALANCE}")
    print(f"  MA_WINDOW: {strategy.MA_WINDOW}")
    print(f"  INITIAL_CAPITAL: ${strategy.INITIAL_CAPITAL}")
    if strategy.BENCHMARK:
        print(f"  BENCHMARK: loaded ({len(strategy.BENCHMARK)} features)")
    print(f"  MEMORY REPORT: {config.EXPERIMENTS_DIR / memory.LEARNING_REPORT}")

    eval_config = {
        "mode": args.eval_mode,
        "timeframe": run_config["timeframe"],
        "window_hours": run_config["eval_window_hours"],
        "step_hours": run_config["eval_step_hours"],
        "objective": run_config["eval_objective"],
    }

    if args.split in ("train", "both"):
        train_metrics, train_raw = run_one(
            train,
            pool_config,
            strategy_fn,
            "TRAIN SET",
            eval_config,
            capital=strategy.INITIAL_CAPITAL,
        )
        save_results(
            train_metrics,
            train_raw,
            "train",
            run_id,
            args.pool,
            pool_config,
            run_config,
            train,
            strategy,
            lper_features,
        )

    if args.split in ("val", "both"):
        val_metrics, val_raw = run_one(
            val,
            pool_config,
            strategy_fn,
            "VALIDATION SET",
            eval_config,
            capital=strategy.INITIAL_CAPITAL,
        )
        save_results(
            val_metrics,
            val_raw,
            "val",
            run_id,
            args.pool,
            pool_config,
            run_config,
            val,
            strategy,
            lper_features,
        )
        print_benchmark_comparison(val_metrics, lper_features)

    report_path = memory.render_learning_report(args.pool)
    print(f"\nLearning report updated: {report_path}")

    if args.split in ("val", "both"):
        print(f"\n{'='*60}")
        print(
            f"  >>> PRIMARY METRIC ({primary_metric_name(val_metrics)}): "
            f"{primary_metric_value(val_metrics):+.4f}% <<<"
        )
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
