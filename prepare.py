"""
prepare.py — Data fetching, caching, and runtime utilities.
Pulls candle data from Meteora and top LP data from LP Agent API.
DO NOT MODIFY. The agent only modifies strategy.py.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import numpy as np
import pandas as pd

import config


# ─── Meteora API (public, no auth) ──────────────────────────────────────────

def meteora_get(endpoint: str, params: dict = None) -> dict:
    """GET from Meteora DLMM API with rate limiting."""
    url = f"{config.METEORA_API}{endpoint}"
    time.sleep(config.METEORA_RATE_DELAY)
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_pool_info(pool_address: str) -> dict:
    """Fetch pool metadata."""
    print(f"  Fetching pool info for {pool_address[:12]}...")
    return meteora_get(f"/pools/{pool_address}")


def fetch_ohlcv(pool_address: str, timeframe: str = "1h",
                start_time: int = None, end_time: int = None) -> list:
    """Fetch OHLCV candle data for a pool."""
    params = {"timeframe": timeframe}
    if start_time:
        params["start_time"] = start_time
    if end_time:
        params["end_time"] = end_time
    print(f"  Fetching OHLCV ({timeframe})...")
    data = meteora_get(f"/pools/{pool_address}/ohlcv", params=params)
    return data.get("data", data) if isinstance(data, dict) else data


def fetch_volume_history(pool_address: str, timeframe: str = "1h",
                         start_time: int = None, end_time: int = None) -> list:
    """Fetch volume history for a pool."""
    params = {"timeframe": timeframe}
    if start_time:
        params["start_time"] = start_time
    if end_time:
        params["end_time"] = end_time
    print(f"  Fetching volume history ({timeframe})...")
    data = meteora_get(f"/pools/{pool_address}/volume/history", params=params)
    return data.get("data", data) if isinstance(data, dict) else data


# ─── LP Agent API (key-authenticated) ───────────────────────────────────────

def lpagent_get(endpoint: str, params: dict = None) -> dict:
    """GET from LP Agent API with key rotation."""
    key = config.keys.get_key()
    url = f"{config.LPAGENT_API}{endpoint}"
    headers = {"x-api-key": key}
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_top_lpers(pool_address: str, order_by: str = "total_pnl",
                    sort_order: str = "desc", limit: int = 50,
                    pages: int = 3) -> list:
    """
    Fetch top LPers for a pool, paginated.
    Returns flat list of LP records.
    """
    all_lpers = []
    for page in range(1, pages + 1):
        print(f"  Fetching top LPers (page {page}/{pages}, sort={order_by})...")
        data = lpagent_get(
            f"/pools/{pool_address}/top-lpers",
            params={
                "order_by": order_by,
                "sort_order": sort_order,
                "page": page,
                "limit": limit,
            },
        )
        records = data.get("data", [])
        all_lpers.extend(records)

        pagination = data.get("pagination", {})
        if not pagination.get("hasNextPage", False):
            break

    print(f"  Got {len(all_lpers)} top LPers")
    return all_lpers


def fetch_lper_positions(wallet: str, position_type: str = "historical") -> list:
    """
    Fetch positions for a wallet.
    position_type: 'opening' or 'historical'
    """
    endpoint = f"/lp-positions/{position_type}"
    print(f"  Fetching {position_type} positions for {wallet[:8]}...")
    data = lpagent_get(endpoint, params={"owner": wallet})
    return data.get("data", [])


def fetch_lper_overview(wallet: str) -> dict:
    """Fetch overview metrics for a wallet's LP activity."""
    print(f"  Fetching overview for {wallet[:8]}...")
    data = lpagent_get("/lp-positions/overview", params={"owner": wallet})
    return data.get("data", {})


def fetch_lper_revenue(wallet: str) -> list:
    """Fetch revenue data for a wallet's positions."""
    print(f"  Fetching revenue for {wallet[:8]}...")
    data = lpagent_get("/lp-positions/revenue", params={"owner": wallet})
    return data.get("data", [])


def fetch_pool_details(pool_address: str) -> dict:
    """Fetch detailed pool info from LP Agent."""
    print(f"  Fetching LP Agent pool details...")
    data = lpagent_get(f"/pools/{pool_address}")
    return data.get("data", {})


def fetch_pool_positions(pool_address: str) -> list:
    """Fetch all positions in a pool from LP Agent."""
    print(f"  Fetching pool positions...")
    data = lpagent_get(f"/pools/{pool_address}/positions")
    return data.get("data", [])


def fetch_pool_onchain_stats(pool_address: str) -> dict:
    """Fetch on-chain statistics for a pool."""
    print(f"  Fetching on-chain stats...")
    data = lpagent_get(f"/pools/{pool_address}/onchain-stats")
    return data.get("data", {})


# ─── Data Processing ─────────────────────────────────────────────────────────

def _parse_timestamp_column(series: pd.Series) -> pd.Series:
    """Parse a timestamp-like column while preserving numeric Unix epochs."""
    non_null = series.dropna()
    if non_null.empty:
        return pd.to_datetime(series, utc=True, errors="coerce")

    sample = non_null.iloc[0]
    if isinstance(sample, (int, float)) or str(sample).isdigit():
        numeric = pd.to_numeric(series, errors="coerce")
        return pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def _normalize_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Prefer one timestamp source and avoid duplicate timestamp columns."""
    if "timestamp" in df.columns:
        df["timestamp"] = _parse_timestamp_column(df["timestamp"])
        return df

    for candidate in ("time", "date", "datetime", "timestamp_str", "date_str"):
        if candidate in df.columns:
            df["timestamp"] = _parse_timestamp_column(df[candidate])
            break
    return df


def get_pool_runtime_config(pool_info: dict) -> dict:
    """Extract runtime pool parameters with backward-compatible fallbacks."""
    pool_cfg = pool_info.get("pool_config") or {}
    return {
        "name": pool_info.get("name", "Unknown"),
        "bin_step": int(pool_cfg.get("bin_step", pool_info.get("bin_step", 25))),
        "base_fee_pct": float(pool_cfg.get("base_fee_pct", pool_info.get("base_fee_pct", 0.25))),
        "protocol_fee_pct": float(
            pool_cfg.get("protocol_fee_pct", pool_info.get("protocol_fee_pct", 5.0))
        ),
    }


def filter_lpers_for_horizon(lpers_df: pd.DataFrame, horizon_mode: str) -> pd.DataFrame:
    """Filter LP benchmarks to the target hold horizon when hold-time data exists."""
    if lpers_df.empty or "avg_age_hour" not in lpers_df.columns:
        return lpers_df

    min_hours, max_hours = config.benchmark_filter_range(horizon_mode)
    filtered = lpers_df[
        lpers_df["avg_age_hour"].between(min_hours, max_hours, inclusive="both")
    ].copy()
    return filtered if not filtered.empty else lpers_df

def process_ohlcv(raw_candles: list) -> pd.DataFrame:
    """Convert raw OHLCV to a clean DataFrame."""
    if not raw_candles:
        raise ValueError("No candle data returned from API")

    df = pd.DataFrame(raw_candles)

    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower in ("o", "open"):
            col_map[col] = "open"
        elif lower in ("h", "high"):
            col_map[col] = "high"
        elif lower in ("l", "low"):
            col_map[col] = "low"
        elif lower in ("c", "close"):
            col_map[col] = "close"
        elif lower in ("v", "volume"):
            col_map[col] = "volume"

    df = df.rename(columns=col_map)
    df = _normalize_timestamp(df)

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def process_volume(raw_volume: list) -> pd.DataFrame:
    """Convert raw volume history to a clean DataFrame."""
    if not raw_volume:
        return pd.DataFrame(columns=["timestamp", "volume"])

    df = pd.DataFrame(raw_volume)
    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if "vol" in lower:
            col_map[col] = "volume"

    df = df.rename(columns=col_map)
    df = _normalize_timestamp(df)
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def process_top_lpers(raw_lpers: list) -> pd.DataFrame:
    """Convert top LPer records to a DataFrame with extracted features."""
    if not raw_lpers:
        return pd.DataFrame()

    df = pd.DataFrame(raw_lpers)

    # Ensure numeric columns
    numeric_cols = [
        "total_inflow", "avg_inflow", "total_outflow", "total_fee",
        "total_pnl", "total_lp", "avg_age_hour", "win_lp", "win_rate",
        "fee_percent", "apr", "roi", "total_fee_native", "total_pnl_native",
        "win_rate_native", "fee_percent_native",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse timestamps
    for col in ["first_activity", "last_activity"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    return df


def extract_lper_features(lpers_df: pd.DataFrame, horizon_mode: str = None) -> dict:
    """
    Extract aggregate features from top LPers for the agent to use.
    These become the "benchmark" the strategy tries to beat.
    """
    if lpers_df.empty:
        return {}

    horizon_mode = config.normalize_horizon_mode(horizon_mode)
    filtered_lpers = filter_lpers_for_horizon(lpers_df, horizon_mode)

    # Filter to profitable LPers only
    profitable = (
        filtered_lpers[filtered_lpers["total_pnl"] > 0]
        if "total_pnl" in filtered_lpers.columns
        else filtered_lpers
    )

    features = {}

    if not profitable.empty:
        for col in ["avg_age_hour", "win_rate", "fee_percent", "apr", "roi",
                     "total_lp", "avg_inflow"]:
            if col in profitable.columns:
                features[f"top_lp_median_{col}"] = float(profitable[col].median())
                features[f"top_lp_p25_{col}"] = float(profitable[col].quantile(0.25))
                features[f"top_lp_p75_{col}"] = float(profitable[col].quantile(0.75))

        # Best LPer stats
        if "total_pnl" in profitable.columns:
            best = profitable.loc[profitable["total_pnl"].idxmax()]
            features["best_lp_pnl"] = float(best.get("total_pnl", 0))
            features["best_lp_win_rate"] = float(best.get("win_rate", 0))
            features["best_lp_avg_age_hours"] = float(best.get("avg_age_hour", 0))
            features["best_lp_fee_pct"] = float(best.get("fee_percent", 0))
            features["best_lp_apr"] = float(best.get("apr", 0))
            features["best_lp_positions"] = int(best.get("total_lp", 0))

    features["total_lpers_analyzed"] = len(filtered_lpers)
    features["total_lpers_raw"] = len(lpers_df)
    features["profitable_lpers"] = len(profitable) if not profitable.empty else 0
    features["benchmark_horizon_mode"] = horizon_mode
    features["benchmark_hold_min_hours"], features["benchmark_hold_max_hours"] = \
        config.benchmark_filter_range(horizon_mode)

    return features


# ─── Train/Val Split ─────────────────────────────────────────────────────────

def split_data(candles: pd.DataFrame, val_ratio: float = 0.2):
    """Temporal split into train and validation sets."""
    split_idx = int(len(candles) * (1 - val_ratio))
    train = candles.iloc[:split_idx].copy().reset_index(drop=True)
    val = candles.iloc[split_idx:].copy().reset_index(drop=True)
    return train, val


# ─── Evaluation Metric ───────────────────────────────────────────────────────

def compute_metrics(results: dict) -> dict:
    """
    Compute strategy performance metrics from backtest results.
    Primary metric: net_pnl_pct (net P&L as % of initial capital).
    """
    total_fees = results.get("total_fees_usd", 0.0)
    total_il = results.get("total_il_usd", 0.0)
    initial_capital = results.get("initial_capital_usd", 1000.0)
    num_rebalances = results.get("num_rebalances", 0)
    time_in_range_pct = results.get("time_in_range_pct", 0.0)
    hours = max(results.get("hours_elapsed", 1), 1)

    net_pnl = total_fees - abs(total_il)
    net_pnl_pct = (net_pnl / initial_capital) * 100
    gross_apr = (total_fees / initial_capital) * (365 * 24 / hours) * 100
    net_apr = (net_pnl / initial_capital) * (365 * 24 / hours) * 100
    fee_per_rebalance = total_fees / max(num_rebalances, 1)

    return {
        "net_pnl_usd": round(net_pnl, 4),
        "net_pnl_pct": round(net_pnl_pct, 4),
        "total_fees_usd": round(total_fees, 4),
        "total_il_usd": round(abs(total_il), 4),
        "gross_apr": round(gross_apr, 2),
        "net_apr": round(net_apr, 2),
        "num_rebalances": num_rebalances,
        "time_in_range_pct": round(time_in_range_pct, 2),
        "fee_per_rebalance": round(fee_per_rebalance, 4),
        "final_portfolio_value_usd": round(
            results.get("final_portfolio_value_usd", initial_capital), 4
        ),
    }


# ─── Persistence ─────────────────────────────────────────────────────────────

def save_data(pool_address: str, pool_info: dict, candles: pd.DataFrame,
              volume: pd.DataFrame, top_lpers: pd.DataFrame = None,
              lper_features: dict = None, lper_positions: dict = None,
              run_config: dict = None):
    """Cache all fetched data locally."""
    pool_dir = config.DATA_DIR / pool_address[:12]
    pool_dir.mkdir(parents=True, exist_ok=True)

    with open(pool_dir / "pool_info.json", "w") as f:
        json.dump(pool_info, f, indent=2, default=str)

    candles.to_parquet(pool_dir / "candles.parquet", index=False)
    volume.to_parquet(pool_dir / "volume.parquet", index=False)

    if top_lpers is not None and not top_lpers.empty:
        top_lpers.to_parquet(pool_dir / "top_lpers.parquet", index=False)

    if lper_features:
        with open(pool_dir / "lper_features.json", "w") as f:
            json.dump(lper_features, f, indent=2)

    if lper_positions:
        with open(pool_dir / "lper_positions.json", "w") as f:
            json.dump(lper_positions, f, indent=2, default=str)

    if run_config:
        with open(pool_dir / "run_config.json", "w") as f:
            json.dump(run_config, f, indent=2)

    print(f"\nData saved to {pool_dir}/")


def load_data(pool_address: str):
    """Load cached data."""
    pool_dir = config.DATA_DIR / pool_address[:12]
    if not pool_dir.exists():
        raise FileNotFoundError(
            f"No cached data for {pool_address}. Run: uv run prepare.py"
        )

    with open(pool_dir / "pool_info.json") as f:
        pool_info = json.load(f)

    candles = pd.read_parquet(pool_dir / "candles.parquet")
    volume = pd.read_parquet(pool_dir / "volume.parquet")

    lper_features = {}
    features_path = pool_dir / "lper_features.json"
    if features_path.exists():
        with open(features_path) as f:
            lper_features = json.load(f)

    top_lpers = None
    lpers_path = pool_dir / "top_lpers.parquet"
    if lpers_path.exists():
        top_lpers = pd.read_parquet(lpers_path)

    lper_positions = {}
    pos_path = pool_dir / "lper_positions.json"
    if pos_path.exists():
        with open(pos_path) as f:
            lper_positions = json.load(f)

    run_config = {}
    run_config_path = pool_dir / "run_config.json"
    if run_config_path.exists():
        with open(run_config_path) as f:
            run_config = json.load(f)

    return pool_info, candles, volume, lper_features, top_lpers, lper_positions, run_config


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    pool_address = config.DEFAULT_POOL
    if "--pool" in sys.argv:
        pool_address = sys.argv[sys.argv.index("--pool") + 1]

    horizon_mode = config.HORIZON_MODE
    if "--horizon" in sys.argv:
        horizon_mode = sys.argv[sys.argv.index("--horizon") + 1]

    days_override = None
    if "--days" in sys.argv:
        days_override = int(sys.argv[sys.argv.index("--days") + 1])

    timeframe_override = None
    if "--timeframe" in sys.argv:
        timeframe_override = sys.argv[sys.argv.index("--timeframe") + 1]

    horizon = config.resolve_horizon_settings(
        horizon_mode,
        timeframe=timeframe_override,
        days=days_override,
    )
    horizon_mode = horizon["mode"]
    days_back = horizon["days"]
    timeframe = horizon["timeframe"]

    top_n_wallets = 5  # How many top wallets to deep-dive
    if "--top-wallets" in sys.argv:
        top_n_wallets = int(sys.argv[sys.argv.index("--top-wallets") + 1])

    skip_lp = "--skip-lp" in sys.argv

    print("=" * 60)
    print("  DLMM Autoresearch — Data Preparation")
    print("=" * 60)
    print(f"  Pool:       {pool_address}")
    print(f"  Horizon:    {horizon_mode}")
    print(f"  History:    {days_back} days")
    print(f"  Timeframe:  {timeframe}")
    print(f"  LP Agent:   {'SKIP' if skip_lp else f'{len(config.keys.keys)} keys ({config.keys.total_rpm} RPM)'}")
    print()

    # ── 1. Meteora: Pool info ──
    print("[1/4] Pool info...")
    pool_info = fetch_pool_info(pool_address)
    pool_runtime = get_pool_runtime_config(pool_info)
    print(
        f"  → {pool_runtime['name']} | Bin Step: {pool_runtime['bin_step']} bps "
        f"| Base Fee: {pool_runtime['base_fee_pct']:.3f}% "
        f"| Protocol Fee: {pool_runtime['protocol_fee_pct']:.1f}%"
    )

    # ── 2. Meteora: Candles ──
    print("\n[2/4] Candle data...")
    now = int(datetime.now(timezone.utc).timestamp())
    start = now - (days_back * 86400)
    raw_candles = fetch_ohlcv(pool_address, timeframe=timeframe,
                               start_time=start, end_time=now)
    candles = process_ohlcv(raw_candles)
    print(f"  → {len(candles)} candles ({candles['timestamp'].min()} → {candles['timestamp'].max()})")

    # ── 3. Meteora: Volume ──
    print("\n[3/4] Volume history...")
    raw_volume = fetch_volume_history(pool_address, timeframe=timeframe,
                                      start_time=start, end_time=now)
    volume = process_volume(raw_volume)
    print(f"  → {len(volume)} rows")

    # ── 4. LP Agent: Top LPers + positions ──
    top_lpers_df = pd.DataFrame()
    lper_features = {}
    lper_positions = {}

    if not skip_lp and config.keys.available:
        print(f"\n[4/4] Top LP data (LP Agent API)...")
        print(f"  {config.keys.status()}")

        try:
            # Fetch top LPers ranked by PnL
            raw_lpers = fetch_top_lpers(pool_address, order_by="total_pnl",
                                         sort_order="desc", limit=50, pages=2)
            top_lpers_df = process_top_lpers(raw_lpers)
            lper_features = extract_lper_features(top_lpers_df, horizon_mode=horizon_mode)
            horizon_lpers_df = filter_lpers_for_horizon(top_lpers_df, horizon_mode)

            if not top_lpers_df.empty:
                print(f"\n  Top LPer benchmarks:")
                for key, val in sorted(lper_features.items()):
                    if isinstance(val, float):
                        print(f"    {key}: {val:.4f}")
                    else:
                        print(f"    {key}: {val}")

                # Deep-dive: fetch positions for top N wallets
                top_wallets = horizon_lpers_df.head(top_n_wallets)["owner"].tolist()
                print(f"\n  Fetching positions for top {len(top_wallets)} wallets...")

                for wallet in top_wallets:
                    try:
                        # Get historical positions for this wallet
                        positions = fetch_lper_positions(wallet, "historical")
                        if positions:
                            lper_positions[wallet] = positions
                            print(f"    {wallet[:8]}... → {len(positions)} historical positions")
                    except Exception as e:
                        print(f"    {wallet[:8]}... → Error: {e}")

                    try:
                        # Get open positions too
                        open_pos = fetch_lper_positions(wallet, "opening")
                        if open_pos:
                            lper_positions[f"{wallet}_open"] = open_pos
                            print(f"    {wallet[:8]}... → {len(open_pos)} open positions")
                    except Exception as e:
                        pass

        except Exception as e:
            print(f"  LP Agent error: {e}")
            print("  Continuing without LP data...")
    elif skip_lp:
        print("\n[4/4] Skipping LP Agent data (--skip-lp)")
    else:
        print("\n[4/4] Skipping LP Agent data (no API keys configured)")
        print("  Add keys to .env to enable. See .env.example")

    # ── Save everything ──
    run_config = {
        "pool_address": pool_address,
        "horizon_mode": horizon_mode,
        "days": days_back,
        "timeframe": timeframe,
        "top_wallets": top_n_wallets,
        "skip_lp": skip_lp,
    }
    save_data(pool_address, pool_info, candles, volume,
              top_lpers_df, lper_features, lper_positions, run_config=run_config)

    # ── Summary ──
    train, val = split_data(candles)
    print(f"\nTrain: {len(train)} candles ({train['timestamp'].min()} → {train['timestamp'].max()})")
    print(f"Val:   {len(val)} candles ({val['timestamp'].min()} → {val['timestamp'].max()})")

    if lper_features:
        best_apr = lper_features.get("best_lp_apr", "N/A")
        best_wr = lper_features.get("best_lp_win_rate", "N/A")
        print(f"\nBenchmark (best LPer): APR={best_apr}, Win Rate={best_wr}")

    print(f"\nDone. Run: uv run backtest.py")


if __name__ == "__main__":
    main()
