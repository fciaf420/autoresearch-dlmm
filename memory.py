"""
memory.py — Experiment memory and learning summaries for autoresearch v2.
Creates structured history plus an agent-readable report from past runs.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import config

HISTORY_FILE = "history.jsonl"
LEARNING_REPORT = "learning_report.md"


def strategy_snapshot(strategy_module) -> dict[str, Any]:
    """Capture the tunable strategy parameters that matter across experiments."""
    keys = [
        "NUM_BINS",
        "SHAPE",
        "REBALANCE_THRESHOLD",
        "MIN_CANDLES_BETWEEN_REBALANCE",
        "MA_WINDOW",
        "VOLATILITY_WINDOW",
        "INITIAL_CAPITAL",
    ]
    snapshot = {}
    for key in keys:
        if hasattr(strategy_module, key):
            value = getattr(strategy_module, key)
            if isinstance(value, (int, float, str, bool)) or value is None:
                snapshot[key] = value
    return snapshot


def strategy_signature(source: str) -> str:
    """Short content hash to group repeated runs of the same strategy code."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]


def summarize_market_regime(candles: pd.DataFrame) -> dict[str, Any]:
    """Create a compact regime summary the agent can reason about next time."""
    if candles.empty:
        return {}

    closes = pd.to_numeric(candles["close"], errors="coerce")
    opens = pd.to_numeric(candles["open"], errors="coerce")
    highs = pd.to_numeric(candles["high"], errors="coerce")
    lows = pd.to_numeric(candles["low"], errors="coerce")
    returns = closes.pct_change().dropna()
    intrabar_range_pct = (((highs - lows) / closes.replace(0, pd.NA)) * 100).dropna()

    avg_volume = None
    volume_cv = None
    if "volume" in candles.columns:
        volume = pd.to_numeric(candles["volume"], errors="coerce").dropna()
        if not volume.empty:
            avg_volume = float(volume.mean())
            if volume.mean() != 0:
                volume_cv = float(volume.std(ddof=0) / volume.mean())

    close_return_pct = 0.0
    if len(closes) > 1 and float(closes.iloc[0]) != 0:
        close_return_pct = float((closes.iloc[-1] / closes.iloc[0] - 1) * 100)

    realized_vol_pct = float(returns.std(ddof=0) * 100) if not returns.empty else 0.0
    avg_intrabar_range_pct = float(intrabar_range_pct.mean()) if not intrabar_range_pct.empty else 0.0

    if close_return_pct > 2:
        trend = "uptrend"
    elif close_return_pct < -2:
        trend = "downtrend"
    else:
        trend = "range"

    if realized_vol_pct > 1.5:
        volatility_regime = "high"
    elif realized_vol_pct > 0.6:
        volatility_regime = "medium"
    else:
        volatility_regime = "low"

    return {
        "candles": int(len(candles)),
        "close_return_pct": round(close_return_pct, 4),
        "realized_vol_pct": round(realized_vol_pct, 4),
        "avg_intrabar_range_pct": round(avg_intrabar_range_pct, 4),
        "avg_volume": round(avg_volume, 4) if avg_volume is not None else None,
        "volume_cv": round(volume_cv, 4) if volume_cv is not None else None,
        "trend": trend,
        "volatility_regime": volatility_regime,
    }


def append_history(record: dict[str, Any]) -> Path:
    """Append one experiment record to the v2 memory ledger."""
    config.EXPERIMENTS_DIR.mkdir(exist_ok=True)
    history_path = config.EXPERIMENTS_DIR / HISTORY_FILE
    with open(history_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return history_path


def load_history() -> list[dict[str, Any]]:
    """Load all historical experiment records."""
    history_path = config.EXPERIMENTS_DIR / HISTORY_FILE
    if not history_path.exists():
        return []

    records = []
    with open(history_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def build_experiment_record(
    run_id: str,
    pool_address: str,
    pool_config,
    run_config: dict[str, Any],
    label: str,
    metrics: dict[str, Any],
    strategy_source: str,
    strategy_params: dict[str, Any],
    split_candles: pd.DataFrame,
    benchmark: dict[str, Any] | None,
    result_file: Path,
) -> dict[str, Any]:
    """Create one normalized history record."""
    regime = summarize_market_regime(split_candles)
    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "pool_address": pool_address,
        "pool_name": getattr(pool_config, "name", "Unknown"),
        "pool_bin_step_bps": getattr(pool_config, "bin_step_bps", None),
        "pool_base_fee_pct": getattr(pool_config, "base_fee_bps", None),
        "pool_protocol_fee_pct": getattr(pool_config, "protocol_fee_pct", None),
        "run_config": run_config or {},
        "label": label.strip().lower(),
        "metrics": metrics,
        "strategy_params": strategy_params,
        "strategy_signature": strategy_signature(strategy_source),
        "market_regime": regime,
        "benchmark": benchmark or {},
        "result_file": str(result_file),
    }


def _best_validation_run(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    val_records = [r for r in records if r.get("label") in {"val", "validation set"}]
    if not val_records:
        return None
    return max(val_records, key=lambda r: r.get("metrics", {}).get("net_pnl_pct", float("-inf")))


def _shape_summary(val_records: list[dict[str, Any]]) -> list[tuple[str, int, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in val_records:
        params = record.get("strategy_params", {})
        shape = params.get("SHAPE", "unknown")
        grouped[shape].append(record.get("metrics", {}).get("net_pnl_pct", 0.0))

    summary = []
    for shape, vals in grouped.items():
        avg = sum(vals) / len(vals)
        summary.append((shape, len(vals), avg))
    return sorted(summary, key=lambda item: item[2], reverse=True)


def render_learning_report(pool_address: str) -> Path:
    """Render an agent-readable markdown summary of accumulated experiments."""
    config.EXPERIMENTS_DIR.mkdir(exist_ok=True)
    records = load_history()
    pool_records = [r for r in records if r.get("pool_address") == pool_address]

    report_path = config.EXPERIMENTS_DIR / LEARNING_REPORT
    lines = [
        "# Experiment Learning Report",
        "",
        f"Pool: `{pool_address}`",
        f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
    ]

    if not pool_records:
        lines.extend([
            "No experiment history yet for this pool.",
            "",
            "Run `uv run backtest.py --split both` to start building memory.",
        ])
        report_path.write_text("\n".join(lines) + "\n")
        return report_path

    val_records = [r for r in pool_records if r.get("label") in {"val", "validation set"}]
    train_records = [r for r in pool_records if r.get("label") in {"train", "train set"}]
    lines.extend([
        "## Memory Snapshot",
        "",
        f"- Horizon mode: {pool_records[-1].get('run_config', {}).get('horizon_mode', 'unknown')}",
        f"- Timeframe: {pool_records[-1].get('run_config', {}).get('timeframe', 'unknown')}",
        f"- Validation records: {len(val_records)}",
        f"- Train records: {len(train_records)}",
        f"- Unique strategy signatures: {len({r['strategy_signature'] for r in pool_records})}",
        "",
    ])

    best = _best_validation_run(pool_records)
    if best is not None:
        best_metrics = best.get("metrics", {})
        lines.extend([
            "## Best Validation Run",
            "",
            f"- net_pnl_pct: {best_metrics.get('net_pnl_pct', 0.0):+.4f}%",
            f"- net_apr: {best_metrics.get('net_apr', 0.0):+.2f}%",
            f"- time_in_range_pct: {best_metrics.get('time_in_range_pct', 0.0):.2f}%",
            f"- num_rebalances: {best_metrics.get('num_rebalances', 0)}",
            f"- strategy params: `{json.dumps(best.get('strategy_params', {}), sort_keys=True)}`",
            "",
        ])

    if val_records:
        lines.extend([
            "## What Has Worked So Far",
            "",
        ])
        for shape, count, avg in _shape_summary(val_records):
            lines.append(f"- Shape `{shape}`: {count} val runs, avg net_pnl_pct {avg:+.4f}%")
        lines.append("")

        recent = sorted(
            val_records,
            key=lambda r: r.get("recorded_at", ""),
            reverse=True,
        )[:5]
        lines.extend([
            "## Recent Validation Runs",
            "",
        ])
        for record in recent:
            params = record.get("strategy_params", {})
            regime = record.get("market_regime", {})
            metrics = record.get("metrics", {})
            lines.append(
                "- "
                f"{record.get('recorded_at', 'unknown')}: "
                f"net_pnl_pct {metrics.get('net_pnl_pct', 0.0):+.4f}%, "
                f"shape={params.get('SHAPE')}, bins={params.get('NUM_BINS')}, "
                f"rebalance={params.get('REBALANCE_THRESHOLD')}, "
                f"cooldown={params.get('MIN_CANDLES_BETWEEN_REBALANCE')}, "
                f"trend={regime.get('trend')}, vol={regime.get('volatility_regime')}"
            )
        lines.append("")

        latest = recent[0]
        benchmark = latest.get("benchmark", {})
        if benchmark:
            lines.extend([
                "## Current Benchmark Context",
                "",
                f"- LPers analyzed: {benchmark.get('total_lpers_analyzed', 0)}",
                f"- Profitable LPers: {benchmark.get('profitable_lpers', 0)}",
                f"- Best LP APR: {benchmark.get('best_lp_apr', 0):.2f}%",
                f"- Median LP APR: {benchmark.get('top_lp_median_apr', 0):.2f}%",
                "",
            ])

    report_path.write_text("\n".join(lines) + "\n")
    return report_path
