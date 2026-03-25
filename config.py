"""
config.py — Configuration, environment loading, and API key rotation.
Supports multiple LP Agent API keys plus horizon-aware defaults.
"""

import os
import time
import threading
from pathlib import Path


def _load_env():
    """Load .env file if it exists (no dependency on python-dotenv)."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value:
                os.environ.setdefault(key, value)


_load_env()


# ─── Constants ────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
EXPERIMENTS_DIR = Path("experiments")

# Meteora DLMM API (public, no auth)
METEORA_API = "https://dlmm.datapi.meteora.ag"
METEORA_RATE_DELAY = 0.04  # 40ms between requests (safe under 30/s)

# LP Agent API
LPAGENT_API = "https://api.lpagent.io/open-api/v1"
LPAGENT_RPM_PER_KEY = 5  # 5 requests per minute per key

# Defaults (overridden by .env)
DEFAULT_POOL = os.environ.get("DEFAULT_POOL", "BVRbyLjjfSBcoyiYFuxbgKYnWuiFaF9CSXEa5vdSZ9Hh")
BACKTEST_DAYS = int(os.environ.get("BACKTEST_DAYS", "30"))
BACKTEST_TIMEFRAME = os.environ.get("BACKTEST_TIMEFRAME", "1h")
INITIAL_CAPITAL = float(os.environ.get("INITIAL_CAPITAL", "1000"))
HORIZON_MODE = os.environ.get("HORIZON_MODE", "swing")

HORIZON_PRESETS = {
    "scalp": {
        "timeframe": "5m",
        "days": 7,
        "hold_min_hours": 0.0,
        "hold_max_hours": 6.0,
        "strategy": {
            "NUM_BINS": 45,
            "SHAPE": "curve",
            "REBALANCE_THRESHOLD": 0.5,
            "MIN_CANDLES_BETWEEN_REBALANCE": 6,
            "MA_WINDOW": 12,
            "VOLATILITY_WINDOW": 12,
        },
    },
    "intraday": {
        "timeframe": "30m",
        "days": 14,
        "hold_min_hours": 6.0,
        "hold_max_hours": 36.0,
        "strategy": {
            "NUM_BINS": 57,
            "SHAPE": "spot",
            "REBALANCE_THRESHOLD": 0.6,
            "MIN_CANDLES_BETWEEN_REBALANCE": 4,
            "MA_WINDOW": 16,
            "VOLATILITY_WINDOW": 16,
        },
    },
    "swing": {
        "timeframe": "1h",
        "days": 30,
        "hold_min_hours": 36.0,
        "hold_max_hours": 168.0,
        "strategy": {
            "NUM_BINS": 81,
            "SHAPE": "spot",
            "REBALANCE_THRESHOLD": 0.75,
            "MIN_CANDLES_BETWEEN_REBALANCE": 12,
            "MA_WINDOW": 24,
            "VOLATILITY_WINDOW": 24,
        },
    },
    "7d_hold": {
        "timeframe": "1h",
        "days": 45,
        "hold_min_hours": 72.0,
        "hold_max_hours": 240.0,
        "strategy": {
            "NUM_BINS": 101,
            "SHAPE": "spot",
            "REBALANCE_THRESHOLD": 0.82,
            "MIN_CANDLES_BETWEEN_REBALANCE": 24,
            "MA_WINDOW": 48,
            "VOLATILITY_WINDOW": 36,
        },
    },
}


def normalize_horizon_mode(mode: str | None) -> str:
    """Return a supported horizon mode, falling back to the configured default."""
    mode = (mode or HORIZON_MODE or "swing").strip().lower()
    if mode not in HORIZON_PRESETS:
        raise ValueError(
            f"Unknown horizon mode: {mode}. "
            f"Supported modes: {', '.join(sorted(HORIZON_PRESETS))}"
        )
    return mode


def horizon_settings(mode: str | None = None) -> dict:
    """Return the preset settings for a horizon mode."""
    return HORIZON_PRESETS[normalize_horizon_mode(mode)].copy()


def resolve_horizon_settings(
    mode: str | None = None,
    timeframe: str | None = None,
    days: int | None = None,
) -> dict:
    """Resolve timeframe and lookback from the requested horizon plus overrides."""
    resolved_mode = normalize_horizon_mode(mode)
    preset = horizon_settings(resolved_mode)
    return {
        "mode": resolved_mode,
        "timeframe": timeframe or preset["timeframe"],
        "days": int(days if days is not None else preset["days"]),
        "hold_min_hours": preset["hold_min_hours"],
        "hold_max_hours": preset["hold_max_hours"],
        "strategy": dict(preset["strategy"]),
    }


def benchmark_filter_range(mode: str | None = None) -> tuple[float, float]:
    """Return the benchmark hold window in hours for a horizon mode."""
    settings = resolve_horizon_settings(mode)
    return settings["hold_min_hours"], settings["hold_max_hours"]


# ─── API Key Manager ─────────────────────────────────────────────────────────

class KeyManager:
    """
    Rotates across multiple LP Agent API keys to maximize throughput.

    Each key is rate-limited to 5 RPM. With N keys you get N*5 RPM.
    The manager tracks per-key timestamps and picks the least-recently-used
    key that isn't rate-limited.
    """

    def __init__(self):
        raw = os.environ.get("LPAGENT_API_KEYS", "")
        self.keys = [k.strip() for k in raw.split(",") if k.strip()]
        # Timestamps of last `LPAGENT_RPM_PER_KEY` calls per key
        self._call_log: dict[str, list[float]] = {k: [] for k in self.keys}
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return len(self.keys) > 0

    @property
    def total_rpm(self) -> int:
        return len(self.keys) * LPAGENT_RPM_PER_KEY

    def get_key(self) -> str:
        """
        Get the next available API key, waiting if all are rate-limited.
        Returns the key string with x-api-key header value.
        """
        if not self.keys:
            raise RuntimeError(
                "No LP Agent API keys configured.\n"
                "Add keys to .env: LPAGENT_API_KEYS=key1,key2,key3\n"
                "Get keys at: https://lpagent.mintlify.app/api-key-dashboard/dashboard"
            )

        while True:
            with self._lock:
                now = time.time()
                best_key = None
                best_wait = float("inf")

                for key in self.keys:
                    log = self._call_log[key]
                    # Prune calls older than 60s
                    log[:] = [t for t in log if now - t < 60]

                    if len(log) < LPAGENT_RPM_PER_KEY:
                        # This key has capacity
                        if not log or log[-1] < best_wait:
                            best_key = key
                            best_wait = log[-1] if log else 0
                    else:
                        # Key is at limit — how long until oldest call expires?
                        wait = 60 - (now - log[0])
                        if wait < best_wait and best_key is None:
                            best_wait = wait

                if best_key:
                    self._call_log[best_key].append(now)
                    return best_key

            # All keys rate-limited — wait for the shortest cooldown
            wait_time = min(
                60 - (time.time() - self._call_log[k][0])
                for k in self.keys
                if self._call_log[k]
            )
            wait_time = max(wait_time, 0.1)
            print(f"  [KeyManager] All keys rate-limited. Waiting {wait_time:.1f}s...")
            time.sleep(wait_time + 0.1)

    def status(self) -> str:
        """Human-readable status of all keys."""
        if not self.keys:
            return "No API keys configured"
        lines = [f"Keys: {len(self.keys)} | Max throughput: {self.total_rpm} RPM"]
        now = time.time()
        for i, key in enumerate(self.keys):
            masked = key[:12] + "..." + key[-4:]
            log = [t for t in self._call_log[key] if now - t < 60]
            lines.append(f"  [{i+1}] {masked} — {len(log)}/{LPAGENT_RPM_PER_KEY} calls in last 60s")
        return "\n".join(lines)


# Global key manager instance
keys = KeyManager()
