"""
simulator.py — DLMM backtesting engine.
Models bin-by-bin fee accrual, impermanent loss, and position management.
DO NOT MODIFY. The agent only modifies strategy.py.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PoolConfig:
    """Pool parameters derived from Meteora pool info."""
    bin_step_bps: int = 25
    base_fee_bps: float = 0.25
    protocol_fee_pct: float = 5.0
    name: str = "SOL/USDC"

    @property
    def bin_step_mult(self) -> float:
        return 1 + self.bin_step_bps / 10_000


@dataclass
class Position:
    """An LP position across a range of bins."""
    center_price: float
    num_bins: int = 69
    shape: str = "spot"
    capital_usd: float = 1000.0

    bin_prices: np.ndarray = field(default=None, repr=False)
    bin_liquidity: np.ndarray = field(default=None, repr=False)
    lower_price: float = 0.0
    upper_price: float = 0.0

    def initialize(self, pool: PoolConfig):
        """Compute bin prices and distribute liquidity based on shape."""
        half = self.num_bins // 2
        bin_indices = np.arange(-half, half + 1) if self.num_bins % 2 == 1 \
            else np.arange(-half, half)
        bin_indices = bin_indices[:self.num_bins]

        self.bin_prices = self.center_price * (pool.bin_step_mult ** bin_indices)
        self.lower_price = self.bin_prices[0]
        self.upper_price = self.bin_prices[-1]

        if self.shape == "spot":
            self.bin_liquidity = np.ones(self.num_bins) / self.num_bins
        elif self.shape == "curve":
            x = np.linspace(-3, 3, self.num_bins)
            weights = np.exp(-0.5 * x**2)
            self.bin_liquidity = weights / weights.sum()
        elif self.shape == "bid_ask":
            x = np.linspace(-3, 3, self.num_bins)
            weights = 1 - np.exp(-0.5 * x**2) + 0.1
            self.bin_liquidity = weights / weights.sum()
        else:
            raise ValueError(f"Unknown shape: {self.shape}")

        self.bin_liquidity = self.bin_liquidity * self.capital_usd

    def is_in_range(self, price: float) -> bool:
        return self.lower_price <= price <= self.upper_price

    def get_active_bin_idx(self, price: float) -> Optional[int]:
        if not self.is_in_range(price):
            return None
        return int(np.argmin(np.abs(self.bin_prices - price)))


@dataclass
class BacktestState:
    """Tracks running state during a backtest."""
    total_fees_usd: float = 0.0
    total_il_usd: float = 0.0
    num_rebalances: int = 0
    candles_in_range: int = 0
    candles_total: int = 0
    position: Optional[Position] = None
    fee_log: list = field(default_factory=list)
    il_log: list = field(default_factory=list)
    rebalance_log: list = field(default_factory=list)
    portfolio_value_log: list = field(default_factory=list)


def simulate_candle_fees(state, pool, price_open, price_close,
                          volume_usd, tvl_estimate_usd=500_000) -> float:
    """Simulate fee earnings for one price leg within a candle."""
    pos = state.position
    if pos is None or pos.bin_liquidity is None:
        return 0.0

    price_min = min(price_open, price_close)
    price_max = max(price_open, price_close)

    bins_crossed_mask = (pos.bin_prices >= price_min) & (pos.bin_prices <= price_max)

    open_idx = pos.get_active_bin_idx(price_open)
    close_idx = pos.get_active_bin_idx(price_close)
    if open_idx is not None:
        bins_crossed_mask[open_idx] = True
    if close_idx is not None:
        bins_crossed_mask[close_idx] = True

    bins_crossed = np.where(bins_crossed_mask)[0]
    if len(bins_crossed) == 0:
        return 0.0

    volume_per_bin = volume_usd / max(len(bins_crossed), 1)
    fee_rate_decimal = (pool.base_fee_bps / 100) / 100

    total_fees = 0.0
    for bin_idx in bins_crossed:
        our_liq = pos.bin_liquidity[bin_idx]
        est_bin_tvl = tvl_estimate_usd / pos.num_bins
        lp_share = our_liq / (est_bin_tvl + our_liq)
        bin_fee = volume_per_bin * fee_rate_decimal * lp_share
        bin_fee *= (1 - pool.protocol_fee_pct / 100)
        total_fees += bin_fee

    return total_fees


def simulate_candle_il(state, pool, price_open, price_close) -> float:
    """Simulate impermanent loss for one price leg."""
    pos = state.position
    if pos is None or pos.bin_liquidity is None:
        return 0.0

    price_min = min(price_open, price_close)
    price_max = max(price_open, price_close)

    bins_crossed = np.where(
        (pos.bin_prices >= price_min) & (pos.bin_prices <= price_max)
    )[0]

    if len(bins_crossed) == 0:
        return 0.0

    total_il = 0.0
    price_move_pct = abs(pool.bin_step_bps / 10_000)
    for bin_idx in bins_crossed:
        il_per_bin = pos.bin_liquidity[bin_idx] * (price_move_pct ** 2) / 4
        total_il += il_per_bin

    return total_il


def estimate_portfolio_value(state, current_price) -> float:
    if state.position is None:
        return 0.0
    value = state.position.capital_usd + state.total_fees_usd - abs(state.total_il_usd)
    return max(value, 0.0)


def run_backtest(candles, pool, strategy_fn, initial_capital=1000.0,
                  tvl_estimate_usd=500_000.0) -> dict:
    """
    Run a full backtest of a strategy against candle data.

    strategy_fn(state, pool, candle_row, candle_idx) -> dict:
        "action": "hold" | "rebalance" | "exit"
        "center_price", "num_bins", "shape", "capital_usd" (if rebalance)
    """
    state = BacktestState()
    first_price = candles.iloc[0]["close"]

    initial_decision = strategy_fn(state, pool, candles.iloc[0], 0)
    pos = Position(
        center_price=initial_decision.get("center_price", first_price),
        num_bins=initial_decision.get("num_bins", 69),
        shape=initial_decision.get("shape", "spot"),
        capital_usd=initial_decision.get("capital_usd", initial_capital),
    )
    pos.initialize(pool)
    state.position = pos

    has_volume = "volume" in candles.columns and candles["volume"].notna().any()

    for i in range(1, len(candles)):
        row = candles.iloc[i]
        state.candles_total += 1

        price_open = row["open"]
        price_close = row["close"]
        price_high = row["high"]
        price_low = row["low"]
        volume = row["volume"] if has_volume else tvl_estimate_usd * 0.01

        if state.position.is_in_range(price_close):
            state.candles_in_range += 1

        # Simulate O→H→L→C price path
        fees = 0.0
        il = 0.0
        for p1, p2, vol_share in [(price_open, price_high, 0.5),
                                    (price_high, price_low, 0.3),
                                    (price_low, price_close, 0.2)]:
            fees += simulate_candle_fees(state, pool, p1, p2,
                                          volume * vol_share, tvl_estimate_usd)
            il += simulate_candle_il(state, pool, p1, p2)

        state.total_fees_usd += fees
        state.total_il_usd += il
        state.fee_log.append(fees)
        state.il_log.append(il)
        state.portfolio_value_log.append(estimate_portfolio_value(state, price_close))

        decision = strategy_fn(state, pool, row, i)
        action = decision.get("action", "hold")

        if action == "rebalance":
            new_pos = Position(
                center_price=decision.get("center_price", price_close),
                num_bins=decision.get("num_bins", state.position.num_bins),
                shape=decision.get("shape", state.position.shape),
                capital_usd=decision.get("capital_usd", state.position.capital_usd),
            )
            new_pos.initialize(pool)
            state.position = new_pos
            state.num_rebalances += 1
            state.rebalance_log.append({
                "candle_idx": i, "price": price_close,
                "new_center": new_pos.center_price,
                "new_bins": new_pos.num_bins, "new_shape": new_pos.shape,
            })
        elif action == "exit":
            break

    if "timestamp" in candles.columns and len(candles) > 1:
        t0 = pd.Timestamp(candles.iloc[0]["timestamp"])
        t1 = pd.Timestamp(candles.iloc[min(i, len(candles)-1)]["timestamp"])
        hours = max((t1 - t0).total_seconds() / 3600, 1)
    else:
        hours = len(candles)

    return {
        "total_fees_usd": state.total_fees_usd,
        "total_il_usd": state.total_il_usd,
        "initial_capital_usd": initial_capital,
        "num_rebalances": state.num_rebalances,
        "time_in_range_pct": (state.candles_in_range / max(state.candles_total, 1)) * 100,
        "hours_elapsed": hours,
        "final_portfolio_value_usd": estimate_portfolio_value(state, candles.iloc[-1]["close"]),
        "fee_log": state.fee_log,
        "il_log": state.il_log,
        "rebalance_log": state.rebalance_log,
        "portfolio_value_log": state.portfolio_value_log,
    }
