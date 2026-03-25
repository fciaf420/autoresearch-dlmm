"""
strategy.py — DLMM LP strategy definition.
████████████████████████████████████████████████████████████████████████████
██  THIS IS THE ONLY FILE THE AI AGENT MODIFIES.                        ██
██  Everything here is fair game: parameters, logic, signals, shapes.   ██
████████████████████████████████████████████████████████████████████████████

The strategy function is called once per candle. It receives the current
backtest state, pool config, the current candle row, and the candle index.
It must return a dict with at minimum {"action": "hold"} or:
    {"action": "rebalance", "center_price": ..., "num_bins": ..., "shape": ...}

Top LP features (from LP Agent API) are available in BENCHMARK below.
Use them to calibrate your strategy against what the best wallets do.
"""

import numpy as np

import config

# ─── Top LP Benchmarks (injected by backtest.py) ─────────────────────────────

BENCHMARK = {}
RUNTIME_CONTEXT = {
    "horizon_mode": config.HORIZON_MODE,
    "timeframe": config.BACKTEST_TIMEFRAME,
}

_BASELINE_DEFAULTS = {
    "NUM_BINS": 69,
    "SHAPE": "spot",
    "REBALANCE_THRESHOLD": 0.7,
    "MIN_CANDLES_BETWEEN_REBALANCE": 6,
    "MA_WINDOW": 20,
    "VOLATILITY_WINDOW": 12,
}


def set_benchmark(features: dict | None):
    """Inject pool-specific benchmark data for the active backtest run."""
    global BENCHMARK
    BENCHMARK = dict(features or {})


def set_runtime_context(context: dict | None):
    """Inject active horizon/timeframe and apply horizon defaults when untouched."""
    global RUNTIME_CONTEXT
    context = dict(context or {})
    horizon_mode = config.normalize_horizon_mode(context.get("horizon_mode"))
    settings = config.resolve_horizon_settings(horizon_mode)
    RUNTIME_CONTEXT = {
        "horizon_mode": horizon_mode,
        "timeframe": context.get("timeframe", settings["timeframe"]),
    }

    strategy_defaults = settings["strategy"]
    globals_ref = globals()
    for key, baseline in _BASELINE_DEFAULTS.items():
        if globals_ref.get(key) == baseline:
            globals_ref[key] = strategy_defaults[key]

# ─── Strategy Parameters (AGENT: tune these) ─────────────────────────────────

NUM_BINS = 1400                     # bins per position (max 1400)
SHAPE = "curve"                     # spot | curve | bid_ask
REBALANCE_THRESHOLD = 0.7           # rebalance when price is past this % of range
MIN_CANDLES_BETWEEN_REBALANCE = 6   # cooldown between rebalances
INITIAL_CAPITAL = config.INITIAL_CAPITAL

# ─── Indicator Parameters (AGENT: add/modify) ────────────────────────────────

MA_WINDOW = 20
VOLATILITY_WINDOW = 12

# ─── State (reset each backtest) ─────────────────────────────────────────────

_last_rebalance_idx = 0
_price_history = []


def _benchmark_curve_center(price: float) -> float:
    """Bias wide curve placements slightly forward for long-hold LP cohorts."""
    if SHAPE != "curve":
        return price

    median_hold_hours = BENCHMARK.get("top_lp_median_avg_age_hour")
    if median_hold_hours is None:
        median_hold_hours = BENCHMARK.get("benchmark_hold_min_hours", 24.0)

    hold_reference_hours = BENCHMARK.get("top_lp_p75_avg_age_hour", median_hold_hours)
    best_hold_hours = BENCHMARK.get("best_lp_avg_age_hours")
    if best_hold_hours is not None:
        # Let the strongest wallet modestly pull the center forward without
        # fully anchoring to its full hold duration.
        hold_reference_hours = max(
            float(hold_reference_hours),
            float(best_hold_hours) * 0.75,
        )
    bias_pct = min(max((float(hold_reference_hours) - 72.0) / 1600.0, 0.0), 0.06)
    return price * (1 + bias_pct)


def strategy(state, pool, candle, candle_idx) -> dict:
    """
    Core strategy function. Called once per candle.

    Args:
        state: BacktestState — running totals (fees, IL, rebalances, position)
        pool: PoolConfig — bin_step, fees, etc.
        candle: pandas Series — open, high, low, close, volume, timestamp
        candle_idx: int — index of current candle

    Returns:
        dict with:
            "action": "hold" | "rebalance" | "exit"
            "center_price": float (if rebalance)
            "num_bins": int (if rebalance)
            "shape": str (if rebalance)
            "capital_usd": float (if rebalance)

    Available benchmark data (from top LPers):
        BENCHMARK.get("best_lp_apr")
        BENCHMARK.get("best_lp_win_rate")
        BENCHMARK.get("best_lp_avg_age_hours")
        BENCHMARK.get("top_lp_median_avg_age_hour")
        BENCHMARK.get("top_lp_median_win_rate")
        BENCHMARK.get("top_lp_median_fee_percent")
        BENCHMARK.get("benchmark_horizon_mode")
    """
    global _last_rebalance_idx, _price_history

    price = candle["close"]
    _price_history.append(price)

    # ── Initial position (candle 0) ──
    if candle_idx == 0:
        _last_rebalance_idx = 0
        _price_history = [price]
        return {
            "action": "rebalance",
            "center_price": _benchmark_curve_center(price),
            "num_bins": NUM_BINS,
            "shape": SHAPE,
            "capital_usd": INITIAL_CAPITAL,
        }

    pos = state.position
    if pos is None:
        return {"action": "hold"}

    # ── Out of range → must rebalance ──
    if not pos.is_in_range(price):
        _last_rebalance_idx = candle_idx
        return {
            "action": "rebalance",
            "center_price": _benchmark_curve_center(price),
            "num_bins": NUM_BINS,
            "shape": SHAPE,
            "capital_usd": pos.capital_usd,
        }

    # ── Range utilization check ──
    active_idx = pos.get_active_bin_idx(price)
    if active_idx is not None:
        range_position = abs(active_idx - pos.num_bins / 2) / (pos.num_bins / 2)
        candles_since = candle_idx - _last_rebalance_idx

        if range_position > REBALANCE_THRESHOLD and \
           candles_since >= MIN_CANDLES_BETWEEN_REBALANCE:

            if len(_price_history) >= MA_WINDOW:
                ma = np.mean(_price_history[-MA_WINDOW:])
                new_center = 0.5 * price + 0.5 * ma
            else:
                new_center = price

            _last_rebalance_idx = candle_idx
            return {
                "action": "rebalance",
                "center_price": _benchmark_curve_center(new_center),
                "num_bins": NUM_BINS,
                "shape": SHAPE,
                "capital_usd": pos.capital_usd,
            }

    return {"action": "hold"}


def reset():
    """Reset strategy state between backtest runs."""
    global _last_rebalance_idx, _price_history
    _last_rebalance_idx = 0
    _price_history = []
