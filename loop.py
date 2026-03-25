"""
loop.py — Autonomous experiment orchestrator for autoresearch-dlmm.

This script closes the loop:
1. Run a baseline backtest on the current strategy
2. Ask an external coding agent to make one targeted change to strategy.py
3. Re-run the backtest
4. Keep improvements, revert regressions
5. Repeat

The agent command is intentionally generic so you can plug in Codex, Claude Code,
or a custom wrapper. The command string supports these placeholders:

    {prompt}          shell-quoted prompt text
    {prompt_file}     shell-quoted path to a markdown prompt file
    {pool}            shell-quoted pool address
    {round}           current round number
    {best_val}        current best validation net_pnl_pct
    {last_val}        validation net_pnl_pct from the last attempted run
    {strategy_file}   shell-quoted path to strategy.py
    {repo_dir}        shell-quoted repo root
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import config


ROOT = Path(__file__).parent
STRATEGY_FILE = ROOT / "strategy.py"
PROGRAM_FILE = ROOT / "program.md"
LOOP_HISTORY_FILE = config.EXPERIMENTS_DIR / "loop_history.jsonl"
LOOP_PROMPT_DIR = config.EXPERIMENTS_DIR / "loop_prompts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run autonomous DLMM strategy iterations.")
    parser.add_argument("--pool", default=config.DEFAULT_POOL, help="Pool address to optimize.")
    parser.add_argument(
        "--horizon",
        default=config.HORIZON_MODE,
        help=f"Horizon mode: {', '.join(sorted(config.HORIZON_PRESETS))}",
    )
    parser.add_argument("--rounds", type=int, default=10, help="Maximum experiment rounds.")
    parser.add_argument(
        "--agent-cmd",
        default=os.environ.get("AUTORESEARCH_AGENT_CMD", ""),
        help="Shell command used to invoke the coding agent.",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.0,
        help="Minimum val net_pnl_pct improvement required to keep a change.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Sleep between rounds. Set to 300 for a five-minute cadence.",
    )
    parser.add_argument(
        "--prepare-if-missing",
        action="store_true",
        help="Automatically run prepare.py if cached data for the pool is missing.",
    )
    parser.add_argument("--days", type=int, default=None, help="Prepare lookback days.")
    parser.add_argument(
        "--timeframe",
        default=None,
        help="Prepare timeframe, for example 1h or 15m.",
    )
    parser.add_argument(
        "--top-wallets",
        type=int,
        default=5,
        help="How many top LP wallets to deep-dive during prepare.",
    )
    parser.add_argument(
        "--skip-lp",
        action="store_true",
        help="Skip LP Agent data during auto-prepare.",
    )
    parser.add_argument(
        "--eval-mode",
        default=config.DEFAULT_EVAL_MODE,
        choices=["split", "rolling"],
        help="Evaluation mode used by backtest.py.",
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
        help="Primary objective for rolling evaluation.",
    )
    return parser.parse_args()


def ensure_cached_data(args: argparse.Namespace) -> None:
    pool_dir = config.DATA_DIR / args.pool[:12]
    if pool_dir.exists():
        return
    if not args.prepare_if_missing:
        raise FileNotFoundError(
            f"No cached data for {args.pool}. Run prepare.py first or pass --prepare-if-missing."
        )

    horizon = config.resolve_horizon_settings(args.horizon, timeframe=args.timeframe, days=args.days)
    cmd = [
        sys.executable,
        "prepare.py",
        "--pool",
        args.pool,
        "--horizon",
        horizon["mode"],
        "--days",
        str(horizon["days"]),
        "--timeframe",
        horizon["timeframe"],
        "--top-wallets",
        str(args.top_wallets),
    ]
    if args.skip_lp:
        cmd.append("--skip-lp")

    print("\n[setup] Cached data missing. Running prepare.py...")
    run_command(cmd, "prepare.py bootstrap")


def run_command(command, label: str, shell: bool = False) -> subprocess.CompletedProcess:
    print(f"\n[{label}]")
    if shell:
        print(command)
    else:
        print(" ".join(shlex.quote(str(part)) for part in command))

    completed = subprocess.run(
        command,
        cwd=ROOT,
        shell=shell,
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr.rstrip(), file=sys.stderr)
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}")
    return completed


def terminate_process_group(process: subprocess.Popen, label: str) -> None:
    if process.poll() is not None:
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
        print(f"[{label}] terminated lingering agent process after edit settled.")
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
        print(f"[{label}] killed lingering agent process after edit settled.")


def run_agent_round(
    command: str,
    label: str,
    watched_file: Path,
    settle_seconds: float = 12.0,
    timeout_seconds: float = 900.0,
) -> dict:
    print(f"\n[{label}]")
    print(command)

    pre_round_source = watched_file.read_text()
    last_source = pre_round_source
    edit_stable_since = None
    started_at = time.monotonic()

    process = subprocess.Popen(
        command,
        cwd=ROOT,
        shell=True,
        text=True,
        start_new_session=True,
    )

    while True:
        return_code = process.poll()
        current_source = watched_file.read_text()
        changed = current_source != pre_round_source

        if changed:
            if current_source != last_source:
                edit_stable_since = time.monotonic()
                last_source = current_source
            elif edit_stable_since is None:
                edit_stable_since = time.monotonic()
        else:
            last_source = current_source
            edit_stable_since = None

        if return_code is not None:
            if return_code != 0 and not changed:
                raise RuntimeError(f"{label} failed with exit code {return_code}")
            if return_code != 0:
                print(f"[{label}] agent exited with code {return_code} after editing strategy.py; continuing.")
            return {
                "changed": changed,
                "forced_exit": False,
                "exit_code": return_code,
            }

        if changed and edit_stable_since is not None:
            stable_for = time.monotonic() - edit_stable_since
            if stable_for >= settle_seconds:
                terminate_process_group(process, label)
                return {
                    "changed": True,
                    "forced_exit": True,
                    "exit_code": 0,
                }

        elapsed = time.monotonic() - started_at
        if elapsed >= timeout_seconds:
            if changed:
                print(f"[{label}] timed out after {timeout_seconds:.0f}s, but strategy.py changed; continuing.")
                terminate_process_group(process, label)
                return {
                    "changed": True,
                    "forced_exit": True,
                    "exit_code": 0,
                }

            terminate_process_group(process, label)
            raise RuntimeError(f"{label} timed out after {timeout_seconds:.0f}s without editing strategy.py")

        time.sleep(1)


def latest_val_result(before: set[str] | None = None) -> Path:
    candidates = sorted(config.EXPERIMENTS_DIR.glob("*_val.json"))
    if before is not None:
        new_candidates = [path for path in candidates if path.name not in before]
        if new_candidates:
            return new_candidates[-1]
    if not candidates:
        raise FileNotFoundError("No validation result files found in experiments/")
    return candidates[-1]


def metric_name(metrics: dict) -> str:
    return metrics.get("primary_metric_name", "net_pnl_pct")


def metric_value(metrics: dict) -> float:
    return float(metrics.get("primary_metric_value", metrics.get("net_pnl_pct", float("-inf"))))


def run_backtest(args: argparse.Namespace, horizon_mode: str) -> tuple[dict, Path]:
    config.EXPERIMENTS_DIR.mkdir(exist_ok=True)
    before = {path.name for path in config.EXPERIMENTS_DIR.glob("*_val.json")}
    horizon = config.resolve_horizon_settings(horizon_mode)
    window_hours = args.window_hours if args.window_hours is not None else horizon["eval_window_hours"]
    step_hours = args.start_every_hours if args.start_every_hours is not None else horizon["eval_step_hours"]
    command = [
        sys.executable,
        "backtest.py",
        "--split",
        "both",
        "--pool",
        args.pool,
        "--horizon",
        horizon_mode,
        "--eval-mode",
        args.eval_mode,
        "--window-hours",
        str(window_hours),
        "--start-every-hours",
        str(step_hours),
        "--objective",
        args.objective,
    ]
    run_command(command, "backtest.py")
    val_file = latest_val_result(before)
    with open(val_file) as f:
        payload = json.load(f)
    return payload["metrics"], val_file


def best_metric() -> float:
    history_path = config.EXPERIMENTS_DIR / "history.jsonl"
    if not history_path.exists():
        return float("-inf")

    best = float("-inf")
    with open(history_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("label") not in {"val", "validation set"}:
                continue
            metric = metric_value(record.get("metrics", {}))
            best = max(best, metric)
    return best


def build_prompt(
    args: argparse.Namespace,
    round_num: int,
    best_val: float,
    last_val: float,
) -> str:
    learning_report = config.EXPERIMENTS_DIR / "learning_report.md"
    program = PROGRAM_FILE.read_text()
    report_text = ""
    if learning_report.exists():
        report_text = learning_report.read_text()

    return f"""# Autoresearch DLMM Loop Prompt

Round: {round_num}
Pool: {args.pool}
Horizon mode: {args.horizon}
Evaluation mode: {args.eval_mode}
Current best validation score: {best_val:+.4f}%
Last attempted validation score: {last_val:+.4f}%

You are operating inside the repo at {ROOT}.
Modify only strategy.py.
Make exactly one targeted strategy change.
Use the benchmark context and experiment memory before making the change.
Do not edit prepare.py, backtest.py, simulator.py, config.py, memory.py, or docs.
Save your strategy.py changes and exit.

## program.md

{program}

## learning_report.md

{report_text or "No learning report yet. Run the baseline first and infer from current strategy."}
"""


def render_agent_command(template: str, prompt: str, prompt_file: Path, args, round_num: int, best_val: float, last_val: float) -> str:
    values = {
        "prompt": shlex.quote(prompt),
        "prompt_file": shlex.quote(str(prompt_file)),
        "pool": shlex.quote(args.pool),
        "round": round_num,
        "best_val": f"{best_val:.4f}" if math.isfinite(best_val) else "nan",
        "last_val": f"{last_val:.4f}" if math.isfinite(last_val) else "nan",
        "strategy_file": shlex.quote(str(STRATEGY_FILE)),
        "repo_dir": shlex.quote(str(ROOT)),
    }
    return template.format(**values)


def append_loop_record(record: dict) -> None:
    config.EXPERIMENTS_DIR.mkdir(exist_ok=True)
    with open(LOOP_HISTORY_FILE, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def main() -> None:
    args = parse_args()
    if not args.agent_cmd:
        raise SystemExit(
            "Missing --agent-cmd. Example: "
            "--agent-cmd 'codex exec {prompt_file}' or set AUTORESEARCH_AGENT_CMD."
        )

    config.EXPERIMENTS_DIR.mkdir(exist_ok=True)
    LOOP_PROMPT_DIR.mkdir(exist_ok=True)
    ensure_cached_data(args)

    print("=" * 60)
    print("  DLMM Autoresearch — Autonomous Loop")
    print("=" * 60)
    print(f"  Pool:       {args.pool}")
    print(f"  Horizon:    {config.normalize_horizon_mode(args.horizon)}")
    print(f"  Eval Mode:  {args.eval_mode}")
    if args.eval_mode == "rolling":
        horizon = config.resolve_horizon_settings(args.horizon)
        window_hours = args.window_hours if args.window_hours is not None else horizon["eval_window_hours"]
        step_hours = args.start_every_hours if args.start_every_hours is not None else horizon["eval_step_hours"]
        print(f"  Window:     {window_hours:g}h")
        print(f"  Step:       {step_hours:g}h")
        print(f"  Objective:  {args.objective}")
    print(f"  Rounds:     {args.rounds}")
    print(f"  Sleep:      {args.sleep_seconds:.1f}s")
    print(f"  Strategy:   {STRATEGY_FILE}")
    print()

    best_source = STRATEGY_FILE.read_text()
    baseline_metrics, baseline_file = run_backtest(args, config.normalize_horizon_mode(args.horizon))
    best_val = metric_value(baseline_metrics)
    last_val = best_val

    print(
        f"\n[baseline] Validation {metric_name(baseline_metrics)}: {best_val:+.4f}% "
        f"({baseline_file.name})"
    )

    for round_num in range(1, args.rounds + 1):
        append_loop_record(
            {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "round": round_num,
                "pool_address": args.pool,
                "status": "started",
                "best_val_net_pnl_pct": best_val,
                "last_val_net_pnl_pct": last_val,
            }
        )

        prompt = build_prompt(args, round_num, best_val, last_val)
        prompt_file = LOOP_PROMPT_DIR / f"round_{round_num:03d}.md"
        prompt_file.write_text(prompt)

        pre_round_source = STRATEGY_FILE.read_text()
        command = render_agent_command(
            args.agent_cmd,
            prompt,
            prompt_file,
            args,
            round_num,
            best_val,
            last_val,
        )

        try:
            agent_result = run_agent_round(command, f"agent round {round_num}", STRATEGY_FILE)
        except Exception as exc:
            append_loop_record(
                {
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "round": round_num,
                    "pool_address": args.pool,
                    "status": "agent_error",
                    "error": str(exc),
                }
            )
            raise

        post_round_source = STRATEGY_FILE.read_text()
        if post_round_source == pre_round_source:
            print(f"\n[round {round_num}] strategy.py unchanged. Skipping evaluation.")
            append_loop_record(
                {
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "round": round_num,
                    "pool_address": args.pool,
                    "status": "no_change",
                    "best_val_net_pnl_pct": best_val,
                    "agent_forced_exit": agent_result["forced_exit"],
                }
            )
            if args.sleep_seconds > 0 and round_num < args.rounds:
                time.sleep(args.sleep_seconds)
            continue

        metrics, val_file = run_backtest(args, config.normalize_horizon_mode(args.horizon))
        last_val = metric_value(metrics)
        improved = last_val > (best_val + args.min_improvement)

        if improved:
            best_val = last_val
            best_source = post_round_source
            status = "kept"
            print(
                f"\n[round {round_num}] kept change: "
                f"{metric_name(metrics)} improved to {last_val:+.4f}%"
            )
        else:
            STRATEGY_FILE.write_text(best_source)
            status = "reverted"
            print(
                f"\n[round {round_num}] reverted change: "
                f"trial {metric_name(metrics)} {last_val:+.4f}% did not beat best {best_val:+.4f}%"
            )

        append_loop_record(
            {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "round": round_num,
                "pool_address": args.pool,
                "status": status,
                "trial_val_net_pnl_pct": last_val,
                "best_val_net_pnl_pct": best_val,
                "agent_forced_exit": agent_result["forced_exit"],
                "result_file": str(val_file),
                "prompt_file": str(prompt_file),
            }
        )

        if args.sleep_seconds > 0 and round_num < args.rounds:
            time.sleep(args.sleep_seconds)

    print("\n" + "=" * 60)
    metric_label = "validation score"
    if args.eval_mode == "split":
        metric_label = "validation net_pnl_pct"
    print(f"  Best {metric_label}: {best_val:+.4f}%")
    print(f"  Loop history: {LOOP_HISTORY_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
