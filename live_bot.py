#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DEEP MARKET ANALOGUE BOT v1
===========================

Price Action only:
- Candlestick structure
- Wick/body/range
- Flat / compression
- Impulse
- Rejection
- Breakout / false breakout
- Swing structure HH/HL/LH/LL
- Retracement
- Exhaustion
- Historical analogue search
- Forward outcome statistics
- Continuous live monitoring
- Telegram text + chart

NO RSI
NO MACD
NO EMA
NO SMA
NO Bollinger
NO Stochastic

ATR is used only as a candle/range normalization unit.

IMPORTANT:
This program produces historical/statistical analysis.
It does NOT guarantee future movement.

Requirements:
    pip install aiohttp websockets numpy matplotlib

Run:
    python live_bot.py
"""

import os
import json
import time
import math
import asyncio
import hashlib
import traceback
from datetime import datetime, timezone
from collections import deque
from typing import List, Dict, Tuple, Optional

import aiohttp
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

SYMBOLS = [
    x.strip().upper()
    for x in os.getenv(
        "SYMBOLS",
        "BTCUSDT,ETHUSDT,SOLUSDT"
    ).split(",")
    if x.strip()
]

TIMEFRAMES = [
    x.strip()
    for x in os.getenv(
        "TIMEFRAMES",
        "5m,15m,1h"
    ).split(",")
    if x.strip()
]

# Historical context length.
CONTEXT_CANDLES = int(os.getenv("CONTEXT_CANDLES", "12"))

# How far into the future each historical analogue is measured.
FORWARD_CANDLES = int(os.getenv("FORWARD_CANDLES", "50"))

# Number of recent live candles kept.
CANDLE_BUFFER = int(os.getenv("CANDLE_BUFFER", "700"))

# Historical download amount.
DAYS_5M = int(os.getenv("DAYS_5M", "365"))
DAYS_15M = int(os.getenv("DAYS_15M", "730"))
DAYS_1H = int(os.getenv("DAYS_1H", "1825"))

# Historical analogue filtering.
MIN_ANALOGUES = int(os.getenv("MIN_ANALOGUES", "25"))
MIN_INDEPENDENT_ANALOGUES = int(
    os.getenv("MIN_INDEPENDENT_ANALOGUES", "15")
)

MIN_SIMILARITY = float(
    os.getenv("MIN_SIMILARITY", "0.72")
)

TOP_ANALOGUES = int(
    os.getenv("TOP_ANALOGUES", "8")
)

# To prevent counting the same market event hundreds of times.
ANALOGUE_SEPARATION = int(
    os.getenv(
        "ANALOGUE_SEPARATION",
        str(CONTEXT_CANDLES + FORWARD_CANDLES)
    )
)

# Signal gate.
SIGNAL_MIN_ANALOGUES = int(
    os.getenv("SIGNAL_MIN_ANALOGUES", "40")
)

SIGNAL_MIN_INDEPENDENT = int(
    os.getenv("SIGNAL_MIN_INDEPENDENT", "25")
)

SIGNAL_MIN_SIMILARITY = float(
    os.getenv("SIGNAL_MIN_SIMILARITY", "0.78")
)

SIGNAL_DIRECTION_MIN = float(
    os.getenv("SIGNAL_DIRECTION_MIN", "0.60")
)

SIGNAL_MOVE_MIN = float(
    os.getenv("SIGNAL_MOVE_MIN", "0.03")
)

# Current candle analysis.
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "50"))

# Scan interval.
RESCAN_SECONDS = int(
    os.getenv("RESCAN_SECONDS", "3600")
)

SIGNAL_COOLDOWN_SECONDS = int(
    os.getenv("SIGNAL_COOLDOWN_SECONDS", "1800")
)

# Historical download.
REQUEST_DELAY = float(
    os.getenv("REQUEST_DELAY", "0.12")
)

MAX_RETRIES = int(
    os.getenv("MAX_RETRIES", "6")
)

HTTP_TIMEOUT = int(
    os.getenv("HTTP_TIMEOUT", "30")
)

# Telegram
TELEGRAM_BOT_TOKEN = (
    os.getenv("TELEGRAM_BOT_TOKEN")
    or os.getenv("TELEGRAM_TOKEN")
    or ""
)

TELEGRAM_CHAT_ID = os.getenv(
    "TELEGRAM_CHAT_ID",
    ""
)

# Public Binance market-data endpoints.
BINANCE_REST = os.getenv(
    "BINANCE_REST",
    "https://data-api.binance.vision"
)

BINANCE_WS = os.getenv(
    "BINANCE_WS",
    "wss://stream.binance.com:9443"
)

RESULT_FILE = os.getenv(
    "RESULT_FILE",
    "historical_analogues_v1.json"
)

CHART_DIR = os.getenv(
    "CHART_DIR",
    "charts"
)

os.makedirs(CHART_DIR, exist_ok=True)


# ============================================================
# GLOBAL STATE
# ============================================================

HISTORY: Dict[Tuple[str, str], List[dict]] = {}
LIVE: Dict[Tuple[str, str], deque] = {}

LAST_SIGNAL: Dict[Tuple[str, str], float] = {}

LAST_STATE: Dict[Tuple[str, str], str] = {}

SCAN_LOCK = asyncio.Lock()


# ============================================================
# BASIC HELPERS
# ============================================================

def now_ms() -> int:
    return int(time.time() * 1000)


def utc_text(ms: int) -> str:
    return datetime.fromtimestamp(
        ms / 1000,
        tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M UTC")


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def pct(a, b):
    if not b:
        return 0.0
    return (a / b - 1.0) * 100.0


def fmt_pct(x):
    return f"{x:+.2f}%"


def fmt_num(x):
    if abs(x) >= 1000:
        return f"{x:,.2f}"
    if abs(x) >= 1:
        return f"{x:.4f}"
    return f"{x:.8f}"


# ============================================================
# KLINE CONVERSION
# ============================================================

def kline_from_binance(row) -> dict:
    return {
        "open_time": int(row[0]),
        "open": safe_float(row[1]),
        "high": safe_float(row[2]),
        "low": safe_float(row[3]),
        "close": safe_float(row[4]),
        "volume": safe_float(row[5]),
        "close_time": int(row[6]),
    }


def kline_from_ws(k) -> dict:
    return {
        "open_time": int(k["t"]),
        "open": safe_float(k["o"]),
        "high": safe_float(k["h"]),
        "low": safe_float(k["l"]),
        "close": safe_float(k["c"]),
        "volume": safe_float(k["v"]),
        "close_time": int(k["T"]),
    }


# ============================================================
# HTTP
# ============================================================

async def http_get_json(
    session: aiohttp.ClientSession,
    url: str,
    params: dict
):
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            async with session.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(
                    total=HTTP_TIMEOUT
                )
            ) as response:

                if response.status == 200:
                    return await response.json()

                body = await response.text()

                if response.status in (418, 429, 500, 502, 503, 504):
                    raise RuntimeError(
                        f"HTTP {response.status}: {body[:300]}"
                    )

                raise RuntimeError(
                    f"HTTP {response.status}: {body[:500]}"
                )

        except Exception as exc:
            last_error = exc

            wait = min(
                30,
                1.5 * (2 ** attempt)
            )

            print(
                f"[HTTP] retry {attempt + 1}/{MAX_RETRIES}: "
                f"{exc}; sleep={wait:.1f}s"
            )

            await asyncio.sleep(wait)

    raise last_error


# ============================================================
# HISTORICAL DATA
# ============================================================

def days_for_tf(tf: str) -> int:
    if tf == "5m":
        return DAYS_5M

    if tf == "15m":
        return DAYS_15M

    if tf == "1h":
        return DAYS_1H

    return 365


def interval_ms(tf: str) -> int:
    unit = tf[-1]
    value = int(tf[:-1])

    if unit == "m":
        return value * 60_000

    if unit == "h":
        return value * 3_600_000

    if unit == "d":
        return value * 86_400_000

    return 60_000


async def fetch_historical(
    session,
    symbol: str,
    timeframe: str
) -> List[dict]:

    days = days_for_tf(timeframe)

    end_time = now_ms()

    start_time = (
        end_time
        - days * 86_400_000
    )

    url = (
        f"{BINANCE_REST}/api/v3/klines"
    )

    output = []

    cursor = start_time

    print(
        f"[HISTORY] {symbol} {timeframe} "
        f"loading {days} days..."
    )

    while cursor < end_time:

        params = {
            "symbol": symbol,
            "interval": timeframe,
            "startTime": cursor,
            "endTime": end_time,
            "limit": 1000,
        }

        try:
            rows = await http_get_json(
                session,
                url,
                params
            )
        except Exception as exc:
            print(
                f"[HISTORY] {symbol} {timeframe} "
                f"FAILED: {exc}"
            )
            break

        if not rows:
            break

        for row in rows:
            try:
                output.append(
                    kline_from_binance(row)
                )
            except Exception:
                continue

        last_open = int(rows[-1][0])

        next_cursor = (
            last_open
            + interval_ms(timeframe)
        )

        if next_cursor <= cursor:
            break

        cursor = next_cursor

        print(
            f"[HISTORY] {symbol} {timeframe}: "
            f"{len(output):,} candles",
            end="\r",
            flush=True
        )

        await asyncio.sleep(REQUEST_DELAY)

        if len(rows) < 1000:
            break

    # Unique + sorted.
    unique = {}

    for c in output:
        unique[c["open_time"]] = c

    result = [
        unique[k]
        for k in sorted(unique)
    ]

    # Remove current incomplete candle.
    current = now_ms()

    result = [
        c for c in result
        if c["close_time"] < current
    ]

    print()

    print(
        f"[HISTORY] {symbol} {timeframe}: "
        f"{len(result):,} closed candles"
    )

    return result


# ============================================================
# PRICE ACTION FEATURES
# ============================================================

def candle_features(c: dict) -> dict:

    o = c["open"]
    h = c["high"]
    l = c["low"]
    cl = c["close"]

    rng = max(h - l, 1e-12)
    body = abs(cl - o)

    upper = max(
        0.0,
        h - max(o, cl)
    )

    lower = max(
        0.0,
        min(o, cl) - l
    )

    direction = (
        1 if cl > o
        else -1 if cl < o
        else 0
    )

    body_ratio = body / rng
    upper_ratio = upper / rng
    lower_ratio = lower / rng

    close_location = (
        (cl - l) / rng
    )

    return {
        "direction": direction,
        "range": rng,
        "body": body,
        "body_ratio": body_ratio,
        "upper_ratio": upper_ratio,
        "lower_ratio": lower_ratio,
        "close_location": close_location,
    }


def true_range(prev: Optional[dict], c: dict) -> float:

    if prev is None:
        return c["high"] - c["low"]

    return max(
        c["high"] - c["low"],
        abs(c["high"] - prev["close"]),
        abs(c["low"] - prev["close"])
    )


def atr_at(candles: List[dict], idx: int, period: int) -> float:

    start = max(
        1,
        idx - period + 1
    )

    values = []

    for i in range(start, idx + 1):
        values.append(
            true_range(
                candles[i - 1],
                candles[i]
            )
        )

    if not values:
        return max(
            candles[idx]["high"]
            - candles[idx]["low"],
            1e-12
        )

    return max(
        float(np.mean(values)),
        1e-12
    )


# ============================================================
# STRUCTURE ANALYSIS
# ============================================================

def local_swings(window: List[dict]):
    highs = []
    lows = []

    if len(window) < 5:
        return highs, lows

    for i in range(2, len(window) - 2):

        h = window[i]["high"]
        l = window[i]["low"]

        if (
            h >= window[i - 1]["high"]
            and h >= window[i - 2]["high"]
            and h >= window[i + 1]["high"]
            and h >= window[i + 2]["high"]
        ):
            highs.append(
                (i, h)
            )

        if (
            l <= window[i - 1]["low"]
            and l <= window[i - 2]["low"]
            and l <= window[i + 1]["low"]
            and l <= window[i + 2]["low"]
        ):
            lows.append(
                (i, l)
            )

    return highs, lows


def structure_state(window: List[dict]) -> str:

    highs, lows = local_swings(window)

    if len(highs) >= 2 and len(lows) >= 2:

        prev_h = highs[-2][1]
        last_h = highs[-1][1]

        prev_l = lows[-2][1]
        last_l = lows[-1][1]

        if last_h > prev_h and last_l > prev_l:
            return "HH_HL_UP"

        if last_h < prev_h and last_l < prev_l:
            return "LH_LL_DOWN"

        if last_h > prev_h and last_l < prev_l:
            return "EXPANSION"

        if last_h < prev_h and last_l > prev_l:
            return "COMPRESSION"

    return "MIXED"


# ============================================================
# MARKET STATE
# ============================================================

def analyse_market_state(
    candles: List[dict]
) -> dict:

    n = len(candles)

    if n < CONTEXT_CANDLES:
        return {
            "state": "INSUFFICIENT_DATA",
            "direction": 0,
            "features": {},
        }

    window = candles[-CONTEXT_CANDLES:]

    closes = np.array(
        [x["close"] for x in window],
        dtype=float
    )

    highs = np.array(
        [x["high"] for x in window],
        dtype=float
    )

    lows = np.array(
        [x["low"] for x in window],
        dtype=float
    )

    feats = [
        candle_features(x)
        for x in window
    ]

    ranges = np.array(
        [f["range"] for f in feats],
        dtype=float
    )

    bodies = np.array(
        [f["body_ratio"] for f in feats],
        dtype=float
    )

    directions = np.array(
        [f["direction"] for f in feats],
        dtype=float
    )

    net_return = (
        closes[-1] / closes[0] - 1
    )

    path = np.sum(
        np.abs(
            np.diff(closes) / closes[:-1]
        )
    )

    efficiency = (
        abs(net_return) / path
        if path > 0
        else 0.0
    )

    recent_range = (
        np.max(highs)
        / np.min(lows)
        - 1
    )

    # Split context into first/second half.
    half = max(2, CONTEXT_CANDLES // 2)

    first_range = np.mean(
        ranges[:half]
    )

    second_range = np.mean(
        ranges[-half:]
    )

    compression_ratio = (
        second_range / first_range
        if first_range > 0
        else 1.0
    )

    # Directional consistency.
    nonzero = directions[
        directions != 0
    ]

    if len(nonzero):
        directional_consistency = abs(
            np.sum(nonzero)
            / len(nonzero)
        )
    else:
        directional_consistency = 0.0

    # Longest same-direction run.
    longest_up = 0
    longest_down = 0

    run_up = 0
    run_down = 0

    for d in directions:
        if d > 0:
            run_up += 1
            run_down = 0
        elif d < 0:
            run_down += 1
            run_up = 0
        else:
            run_up = 0
            run_down = 0

        longest_up = max(
            longest_up,
            run_up
        )

        longest_down = max(
            longest_down,
            run_down
        )

    # Rejection.
    upper_rejection = sum(
        1 for f in feats
        if f["upper_ratio"] >= 0.45
    )

    lower_rejection = sum(
        1 for f in feats
        if f["lower_ratio"] >= 0.45
    )

    # Strong body candles.
    strong_bodies = sum(
        1 for f in feats
        if f["body_ratio"] >= 0.65
    )

    # Small candles.
    small_bodies = sum(
        1 for f in feats
        if f["body_ratio"] <= 0.25
    )

    # ATR normalized latest range.
    last_atr = atr_at(
        candles,
        n - 1,
        ATR_PERIOD
    )

    latest_range_norm = (
        ranges[-1] / last_atr
    )

    # Recent position in range.
    range_low = np.min(lows)
    range_high = np.max(highs)

    if range_high > range_low:
        position = (
            closes[-1] - range_low
        ) / (
            range_high - range_low
        )
    else:
        position = 0.5

    structure = structure_state(
        window
    )

    # Breakout tests.
    previous_high = np.max(
        highs[:-2]
    )

    previous_low = np.min(
        lows[:-2]
    )

    last_close = closes[-1]

    bullish_breakout = (
        last_close > previous_high
    )

    bearish_breakout = (
        last_close < previous_low
    )

    # False breakout approximation:
    prior_high = np.max(
        highs[:-1]
    )

    prior_low = np.min(
        lows[:-1]
    )

    false_up = (
        highs[-1] > prior_high
        and closes[-1] < prior_high
    )

    false_down = (
        lows[-1] < prior_low
        and closes[-1] > prior_low
    )

    # Impulse score.
    impulse_score = (
        abs(net_return) * 100
        * (
            0.5
            + 0.5 * directional_consistency
        )
        * (
            0.5
            + 0.5 * min(
                latest_range_norm / 2,
                1
            )
        )
    )

    # Compression.
    compression = (
        compression_ratio < 0.75
        and recent_range < 0.025
    )

    # Flat.
    flat = (
        recent_range < 0.018
        and directional_consistency < 0.45
    )

    # Exhaustion:
    exhaustion = (
        (
            longest_up >= 3
            and upper_rejection >= 2
            and net_return > 0.01
        )
        or
        (
            longest_down >= 3
            and lower_rejection >= 2
            and net_return < -0.01
        )
    )

    # State priority.
    if false_up:
        state = "FALSE_BREAKOUT_UP"

    elif false_down:
        state = "FALSE_BREAKOUT_DOWN"

    elif bullish_breakout:
        state = "BREAKOUT_UP"

    elif bearish_breakout:
        state = "BREAKOUT_DOWN"

    elif exhaustion:
        state = "EXHAUSTION"

    elif compression:
        state = "COMPRESSION"

    elif flat:
        state = "FLAT"

    elif abs(net_return) > 0.025:
        state = (
            "STRONG_UP_IMPULSE"
            if net_return > 0
            else "STRONG_DOWN_IMPULSE"
        )

    elif abs(net_return) > 0.01:
        state = (
            "UP_IMPULSE"
            if net_return > 0
            else "DOWN_IMPULSE"
        )

    else:
        state = "MIXED"

    direction = (
        1 if net_return > 0
        else -1 if net_return < 0
        else 0
    )

    # Feature vector used for analogue search.
    feature_vector = np.array([
        clamp((net_return + 0.10) / 0.20),

        clamp(
            directional_consistency
        ),

        clamp(
            efficiency
        ),

        clamp(
            recent_range / 0.10
        ),

        clamp(
            compression_ratio
        ),

        clamp(
            position
        ),

        clamp(
            latest_range_norm / 3
        ),

        clamp(
            longest_up / CONTEXT_CANDLES
        ),

        clamp(
            longest_down / CONTEXT_CANDLES
        ),

        clamp(
            upper_rejection / CONTEXT_CANDLES
        ),

        clamp(
            lower_rejection / CONTEXT_CANDLES
        ),

        clamp(
            strong_bodies / CONTEXT_CANDLES
        ),

        clamp(
            small_bodies / CONTEXT_CANDLES
        ),

        clamp(
            impulse_score / 10
        ),
    ], dtype=float)

    return {
        "state": state,
        "direction": direction,
        "structure": structure,
        "net_return": float(net_return),
        "recent_range": float(recent_range),
        "efficiency": float(efficiency),
        "compression_ratio": float(compression_ratio),
        "directional_consistency": float(
            directional_consistency
        ),
        "position": float(position),
        "latest_range_norm": float(
            latest_range_norm
        ),
        "longest_up": int(longest_up),
        "longest_down": int(longest_down),
        "upper_rejection": int(
            upper_rejection
        ),
        "lower_rejection": int(
            lower_rejection
        ),
        "strong_bodies": int(
            strong_bodies
        ),
        "small_bodies": int(
            small_bodies
        ),
        "impulse_score": float(
            impulse_score
        ),
        "bullish_breakout": bool(
            bullish_breakout
        ),
        "bearish_breakout": bool(
            bearish_breakout
        ),
        "false_up": bool(false_up),
        "false_down": bool(false_down),
        "exhaustion": bool(exhaustion),
        "feature_vector": feature_vector,
    }


# ============================================================
# CANDLE SIGNATURE
# ============================================================

def candle_code(c: dict) -> str:

    f = candle_features(c)

    d = (
        "U"
        if f["direction"] > 0
        else "D"
        if f["direction"] < 0
        else "F"
    )

    body = f["body_ratio"]

    if body < 0.15:
        b = "0"
    elif body < 0.35:
        b = "1"
    elif body < 0.65:
        b = "2"
    else:
        b = "3"

    upper = f["upper_ratio"]
    lower = f["lower_ratio"]

    if upper >= 0.45 and lower >= 0.45:
        w = "X"
    elif upper >= 0.45:
        w = "U"
    elif lower >= 0.45:
        w = "L"
    elif upper >= 0.20 and lower >= 0.20:
        w = "B"
    else:
        w = "N"

    return f"{d}{b}{w}"


def sequence_signature(
    window: List[dict]
) -> str:

    return "|".join(
        candle_code(c)
        for c in window
    )


# ============================================================
# CONTEXT SIMILARITY
# ============================================================

def vector_similarity(
    a: np.ndarray,
    b: np.ndarray
) -> float:

    if len(a) != len(b):
        return 0.0

    distance = np.mean(
        np.abs(a - b)
    )

    # 0 distance = 1 similarity.
    return clamp(
        1.0 - distance
    )


def candle_shape_similarity(
    a: List[dict],
    b: List[dict]
) -> float:

    if len(a) != len(b):
        return 0.0

    scores = []

    for ca, cb in zip(a, b):

        fa = candle_features(ca)
        fb = candle_features(cb)

        direction_score = (
            1.0
            if fa["direction"]
            == fb["direction"]
            else 0.0
        )

        body_score = 1.0 - abs(
            fa["body_ratio"]
            - fb["body_ratio"]
        )

        upper_score = 1.0 - abs(
            fa["upper_ratio"]
            - fb["upper_ratio"]
        )

        lower_score = 1.0 - abs(
            fa["lower_ratio"]
            - fb["lower_ratio"]
        )

        close_score = 1.0 - abs(
            fa["close_location"]
            - fb["close_location"]
        )

        score = (
            0.25 * direction_score
            + 0.20 * body_score
            + 0.20 * upper_score
            + 0.20 * lower_score
            + 0.15 * close_score
        )

        scores.append(
            clamp(score)
        )

    return float(
        np.mean(scores)
    )


def context_similarity(
    current_window: List[dict],
    historical_window: List[dict]
) -> float:

    current = analyse_market_state(
        current_window
    )

    historical = analyse_market_state(
        historical_window
    )

    if (
        "feature_vector" not in current
        or "feature_vector" not in historical
    ):
        return 0.0

    vector_score = vector_similarity(
        current["feature_vector"],
        historical["feature_vector"]
    )

    shape_score = candle_shape_similarity(
        current_window,
        historical_window
    )

    current_codes = [
        candle_code(c)
        for c in current_window
    ]

    historical_codes = [
        candle_code(c)
        for c in historical_window
    ]

    exact_count = sum(
        a == b
        for a, b in zip(
            current_codes,
            historical_codes
        )
    )

    code_score = (
        exact_count
        / len(current_codes)
        if current_codes
        else 0.0
    )

    # Main score.
    score = (
        0.45 * vector_score
        + 0.40 * shape_score
        + 0.15 * code_score
    )

    return float(
        clamp(score)
    )


# ============================================================
# FORWARD OUTCOME
# ============================================================

def forward_outcome(
    candles: List[dict],
    start_idx: int
) -> Optional[dict]:

    entry_idx = (
        start_idx
        + CONTEXT_CANDLES
        - 1
    )

    future_start = entry_idx + 1

    future_end = min(
        len(candles),
        future_start + FORWARD_CANDLES
    )

    if future_start >= len(candles):
        return None

    entry = candles[entry_idx]["close"]

    future = candles[
        future_start:future_end
    ]

    if len(future) < 3:
        return None

    max_high = max(
        c["high"]
        for c in future
    )

    min_low = min(
        c["low"]
        for c in future
    )

    max_up = (
        max_high / entry - 1
    )

    max_down = (
        min_low / entry - 1
    )

    final_return = (
        future[-1]["close"]
        / entry
        - 1
    )

    horizons = {}

    for h in (
        1,
        3,
        5,
        10,
        20,
        50
    ):

        if h <= len(future):

            p = future[h - 1]["close"]

            horizons[str(h)] = (
                p / entry - 1
            )

    thresholds_up = {}

    thresholds_down = {}

    for threshold in (
        0.01,
        0.03,
        0.05,
        0.10,
        0.15
    ):

        thresholds_up[
            str(int(threshold * 100))
        ] = any(
            c["high"]
            >= entry * (1 + threshold)
            for c in future
        )

        thresholds_down[
            str(int(threshold * 100))
        ] = any(
            c["low"]
            <= entry * (1 - threshold)
            for c in future
        )

    # Which directional move appeared first?
    first_direction = "FLAT"

    for c in future:

        up = (
            c["high"]
            / entry - 1
        )

        down = (
            c["low"]
            / entry - 1
        )

        if up >= 0.03 and down > -0.03:
            first_direction = "UP"
            break

        if down <= -0.03 and up < 0.03:
            first_direction = "DOWN"
            break

    if max_up >= 0.03 and max_down <= -0.03:
        scenario = "TWO_SIDED"

    elif max_up >= 0.03:
        scenario = "UP"

    elif max_down <= -0.03:
        scenario = "DOWN"

    else:
        scenario = "FLAT"

    return {
        "entry_price": entry,
        "max_up": float(max_up),
        "max_down": float(max_down),
        "final_return": float(final_return),
        "horizons": horizons,
        "thresholds_up": thresholds_up,
        "thresholds_down": thresholds_down,
        "first_direction": first_direction,
        "scenario": scenario,
    }


# ============================================================
# ANALOGUE SEARCH
# ============================================================

def independent_select(
    matches: List[dict]
) -> List[dict]:

    matches = sorted(
        matches,
        key=lambda x: (
            -x["similarity"],
            x["index"]
        )
    )

    selected = []

    for item in matches:

        idx = item["index"]

        too_close = any(
            abs(
                idx - old["index"]
            ) < ANALOGUE_SEPARATION
            for old in selected
        )

        if too_close:
            continue

        selected.append(item)

    return selected


def find_historical_analogues(
    candles: List[dict]
) -> dict:

    if len(candles) < (
        CONTEXT_CANDLES
        + FORWARD_CANDLES
        + 20
    ):
        return {
            "matches": [],
            "independent": [],
            "current_state": {},
        }

    current_window = candles[
        -CONTEXT_CANDLES:
    ]

    current_state = analyse_market_state(
        current_window
    )

    candidates = []

    max_start = (
        len(candles)
        - CONTEXT_CANDLES
        - FORWARD_CANDLES
    )

    # Search every historical context.
    for start in range(
        0,
        max_start
    ):

        # Avoid using current/latest data.
        window = candles[
            start:
            start + CONTEXT_CANDLES
        ]

        score = context_similarity(
            current_window,
            window
        )

        if score < MIN_SIMILARITY:
            continue

        outcome = forward_outcome(
            candles,
            start
        )

        if outcome is None:
            continue

        candidates.append({
            "index": start,
            "similarity": score,
            "open_time": candles[start][
                "open_time"
            ],
            "outcome": outcome,
            "signature": sequence_signature(
                window
            ),
            "state": analyse_market_state(
                window
            )["state"],
        })

    candidates.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    independent = independent_select(
        candidates
    )

    return {
        "matches": candidates,
        "independent": independent,
        "current_state": current_state,
    }


# ============================================================
# STATISTICS
# ============================================================

def safe_mean(values):
    if not values:
        return 0.0
    return float(np.mean(values))


def safe_median(values):
    if not values:
        return 0.0
    return float(np.median(values))


def build_statistics(
    analogues: List[dict]
) -> dict:

    if not analogues:
        return {
            "count": 0
        }

    outcomes = [
        x["outcome"]
        for x in analogues
    ]

    scenarios = {
        "UP": 0,
        "DOWN": 0,
        "FLAT": 0,
        "TWO_SIDED": 0,
    }

    first_direction = {
        "UP": 0,
        "DOWN": 0,
        "FLAT": 0,
    }

    max_up = []
    max_down = []
    final_returns = []

    for o in outcomes:

        scenarios[
            o["scenario"]
        ] = scenarios.get(
            o["scenario"],
            0
        ) + 1

        first_direction[
            o["first_direction"]
        ] = first_direction.get(
            o["first_direction"],
            0
        ) + 1

        max_up.append(
            o["max_up"]
        )

        max_down.append(
            o["max_down"]
        )

        final_returns.append(
            o["final_return"]
        )

    n = len(analogues)

    up_thresholds = {}
    down_thresholds = {}

    for threshold in (
        "1",
        "3",
        "5",
        "10",
        "15"
    ):

        up_thresholds[
            threshold
        ] = sum(
            bool(
                x["outcome"]
                ["thresholds_up"]
                .get(threshold, False)
            )
            for x in analogues
        )

        down_thresholds[
            threshold
        ] = sum(
            bool(
                x["outcome"]
                ["thresholds_down"]
                .get(threshold, False)
            )
            for x in analogues
        )

    horizon_stats = {}

    for h in (
        "1",
        "3",
        "5",
        "10",
        "20",
        "50"
    ):

        values = [
            x["outcome"]["horizons"][h]
            for x in analogues
            if h in x["outcome"]["horizons"]
        ]

        if values:

            horizon_stats[h] = {
                "mean": safe_mean(values),
                "median": safe_median(values),
                "positive": sum(
                    x > 0
                    for x in values
                ),
                "negative": sum(
                    x < 0
                    for x in values
                ),
            }

    return {
        "count": n,

        "scenario_counts": scenarios,

        "scenario_rates": {
            k: v / n
            for k, v in scenarios.items()
        },

        "first_direction_counts":
            first_direction,

        "first_direction_rates": {
            k: v / n
            for k, v
            in first_direction.items()
        },

        "max_up_mean":
            safe_mean(max_up),

        "max_up_median":
            safe_median(max_up),

        "max_down_mean":
            safe_mean(max_down),

        "max_down_median":
            safe_median(max_down),

        "final_return_mean":
            safe_mean(final_returns),

        "final_return_median":
            safe_median(final_returns),

        "up_thresholds":
            up_thresholds,

        "down_thresholds":
            down_thresholds,

        "horizons":
            horizon_stats,
    }


# ============================================================
# SCENARIO DECISION
# ============================================================

def determine_scenario(
    current_state: dict,
    stats: dict,
    independent_count: int,
    average_similarity: float
) -> dict:

    if stats.get("count", 0) < MIN_ANALOGUES:
        return {
            "status": "NO_SIGNAL",
            "scenario": "INSUFFICIENT_HISTORY",
            "reason": "Not enough historical analogues."
        }

    n = stats["count"]

    up_rate = (
        stats["first_direction_rates"]
        .get("UP", 0)
    )

    down_rate = (
        stats["first_direction_rates"]
        .get("DOWN", 0)
    )

    flat_rate = (
        stats["first_direction_rates"]
        .get("FLAT", 0)
    )

    up5 = (
        stats["up_thresholds"]
        .get("5", 0)
        / n
    )

    down5 = (
        stats["down_thresholds"]
        .get("5", 0)
        / n
    )

    reasons = []

    # --------------------------------------------------------
    # Signal requirements.
    # --------------------------------------------------------

    enough_count = (
        n >= SIGNAL_MIN_ANALOGUES
    )

    enough_independent = (
        independent_count
        >= SIGNAL_MIN_INDEPENDENT
    )

    enough_similarity = (
        average_similarity
        >= SIGNAL_MIN_SIMILARITY
    )

    if (
        enough_count
        and enough_independent
        and enough_similarity
    ):

        # Strong UP.
        if (
            up_rate >= SIGNAL_DIRECTION_MIN
            and up5 >= SIGNAL_MOVE_MIN
            and current_state["state"]
            in (
                "BREAKOUT_UP",
                "UP_IMPULSE",
                "STRONG_UP_IMPULSE",
                "COMPRESSION",
            )
        ):

            reasons.extend([
                "Historical UP direction dominates.",
                "5% upside occurrence is sufficient.",
                "Current structure supports upside scenario.",
            ])

            return {
                "status": "LONG_WATCH",
                "scenario": "UP",
                "reason": reasons,
            }

        # Strong DOWN.
        if (
            down_rate >= SIGNAL_DIRECTION_MIN
            and down5 >= SIGNAL_MOVE_MIN
            and current_state["state"]
            in (
                "BREAKOUT_DOWN",
                "DOWN_IMPULSE",
                "STRONG_DOWN_IMPULSE",
                "EXHAUSTION",
                "FALSE_BREAKOUT_UP",
                "COMPRESSION",
            )
        ):

            reasons.extend([
                "Historical DOWN direction dominates.",
                "5% downside occurrence is sufficient.",
                "Current structure supports downside scenario.",
            ])

            return {
                "status": "SHORT_WATCH",
                "scenario": "DOWN",
                "reason": reasons,
            }

    # --------------------------------------------------------
    # Watch states.
    # --------------------------------------------------------

    if current_state["state"] in (
        "FLAT",
        "COMPRESSION",
    ):

        if (
            up_rate > down_rate
        ):
            watch = "UP_BREAKOUT_WATCH"

        elif (
            down_rate > up_rate
        ):
            watch = "DOWN_BREAKOUT_WATCH"

        else:
            watch = "BREAKOUT_WATCH"

        return {
            "status": "WATCH",
            "scenario": watch,
            "reason": [
                "Market is compressed/flat.",
                "Direction is not sufficiently confirmed.",
                "Waiting for breakout confirmation.",
            ],
        }

    if current_state["state"] in (
        "FALSE_BREAKOUT_UP",
        "EXHAUSTION",
    ) and down_rate > up_rate:

        return {
            "status": "SHORT_WATCH",
            "scenario": "REVERSAL_DOWN",
            "reason": [
                "Current structure shows rejection/exhaustion.",
                "Historical analogues favour downside.",
                "Confirmation is still required.",
            ],
        }

    if current_state["state"] in (
        "FALSE_BREAKOUT_DOWN",
    ) and up_rate > down_rate:

        return {
            "status": "LONG_WATCH",
            "scenario": "REVERSAL_UP",
            "reason": [
                "Current structure shows downside rejection.",
                "Historical analogues favour upside.",
                "Confirmation is still required.",
            ],
        }

    return {
        "status": "NO_SIGNAL",
        "scenario": "MIXED",
        "reason": [
            "Historical outcomes are not sufficiently directional.",
            "Continue monitoring.",
        ],
    }


# ============================================================
# TEXT REPORT
# ============================================================

def make_report(
    symbol: str,
    timeframe: str,
    candles: List[dict],
    result: dict,
    full: bool = True
) -> str:

    current_state = result[
        "current_state"
    ]

    matches = result[
        "matches"
    ]

    independent = result[
        "independent"
    ]

    stats = result[
        "statistics"
    ]

    decision = result[
        "decision"
    ]

    current_price = candles[-1]["close"]

    average_similarity = result[
        "average_similarity"
    ]

    lines = []

    lines.append(
        f"🧠 DEEP MARKET ANALYSIS"
    )

    lines.append(
        f"{symbol} · {timeframe}"
    )

    lines.append("")

    lines.append(
        f"💰 PRICE: {fmt_num(current_price)}"
    )

    lines.append(
        f"🕐 {utc_text(candles[-1]['close_time'])}"
    )

    lines.append("")
    lines.append("━━━━━━━━━━━━━━━━━━")
    lines.append("📌 CURRENT MARKET STATE")
    lines.append("━━━━━━━━━━━━━━━━━━")

    lines.append(
        f"STATE: {current_state['state']}"
    )

    lines.append(
        f"STRUCTURE: {current_state['structure']}"
    )

    lines.append(
        f"Context return: "
        f"{fmt_pct(current_state['net_return'] * 100)}"
    )

    lines.append(
        f"Recent range: "
        f"{fmt_pct(current_state['recent_range'] * 100)}"
    )

    lines.append(
        f"Directional consistency: "
        f"{current_state['directional_consistency']:.2f}"
    )

    lines.append(
        f"Upper rejection: "
        f"{current_state['upper_rejection']}"
    )

    lines.append(
        f"Lower rejection: "
        f"{current_state['lower_rejection']}"
    )

    lines.append(
        f"Impulse score: "
        f"{current_state['impulse_score']:.2f}"
    )

    lines.append("")
    lines.append("━━━━━━━━━━━━━━━━━━")
    lines.append("🔎 HISTORICAL ANALOGUES")
    lines.append("━━━━━━━━━━━━━━━━━━")

    lines.append(
        f"Matches: {len(matches):,}"
    )

    lines.append(
        f"Independent: {len(independent):,}"
    )

    lines.append(
        f"Avg similarity: "
        f"{average_similarity * 100:.1f}%"
    )

    if stats.get("count", 0):

        n = stats["count"]

        lines.append("")
        lines.append(
            "📊 HISTORICAL DIRECTION"
        )

        for name in (
            "UP",
            "DOWN",
            "FLAT"
        ):

            count = (
                stats[
                    "first_direction_counts"
                ].get(name, 0)
            )

            rate = (
                count / n
            )

            icon = {
                "UP": "📈",
                "DOWN": "📉",
                "FLAT": "➡️",
            }[name]

            lines.append(
                f"{icon} {name}: "
                f"{count}/{n} "
                f"({rate * 100:.1f}%)"
            )

        lines.append("")
        lines.append(
            "📏 HISTORICAL MAX MOVE"
        )

        lines.append(
            f"UP median: "
            f"{fmt_pct(stats['max_up_median'] * 100)}"
        )

        lines.append(
            f"DOWN median: "
            f"{fmt_pct(stats['max_down_median'] * 100)}"
        )

        lines.append(
            f"Final median: "
            f"{fmt_pct(stats['final_return_median'] * 100)}"
        )

        lines.append("")
        lines.append(
            "🎯 THRESHOLD REACH"
        )

        for t in (
            "1",
            "3",
            "5",
            "10",
            "15"
        ):

            up = stats[
                "up_thresholds"
            ].get(t, 0)

            down = stats[
                "down_thresholds"
            ].get(t, 0)

            lines.append(
                f"+{t}%: {up}/{n}    "
                f"-{t}%: {down}/{n}"
            )

        lines.append("")
        lines.append(
            "⏱ FORWARD OUTCOMES"
        )

        for h in (
            "1",
            "3",
            "5",
            "10",
            "20",
            "50"
        ):

            item = stats[
                "horizons"
            ].get(h)

            if not item:
                continue

            lines.append(
                f"{h} candles: "
                f"median "
                f"{fmt_pct(item['median'] * 100)}"
            )

    lines.append("")
    lines.append("━━━━━━━━━━━━━━━━━━")
    lines.append("🔮 CURRENT SCENARIO")
    lines.append("━━━━━━━━━━━━━━━━━━")

    lines.append(
        f"STATUS: {decision['status']}"
    )

    lines.append(
        f"SCENARIO: {decision['scenario']}"
    )

    for reason in decision["reason"]:
        lines.append(
            f"• {reason}"
        )

    lines.append("")
    lines.append(
        "⚠️ Historical statistics are evidence, "
        "not a guarantee of future price movement."
    )

    return "\n".join(lines)


# ============================================================
# CHART
# ============================================================

def make_chart(
    symbol: str,
    timeframe: str,
    candles: List[dict],
    result: dict
) -> str:

    # Last 80 candles.
    display = candles[
        -min(100, len(candles)):
    ]

    closes = [
        c["close"]
        for c in display
    ]

    highs = [
        c["high"]
        for c in display
    ]

    lows = [
        c["low"]
        for c in display
    ]

    opens = [
        c["open"]
        for c in display
    ]

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    x = np.arange(
        len(display)
    )

    # Candles.
    for i, c in enumerate(display):

        o = c["open"]
        h = c["high"]
        l = c["low"]
        cl = c["close"]

        if cl >= o:
            face = "white"
        else:
            face = "black"

        ax.vlines(
            i,
            l,
            h,
            linewidth=1
        )

        body_low = min(o, cl)
        body_height = max(
            abs(cl - o),
            (h - l) * 0.002
        )

        rect = plt.Rectangle(
            (
                i - 0.32,
                body_low
            ),
            0.64,
            body_height,
            fill=True,
            facecolor=face,
            edgecolor="black",
            linewidth=0.8
        )

        ax.add_patch(rect)

    # Current context.
    context_start = max(
        0,
        len(display)
        - CONTEXT_CANDLES
    )

    ax.axvspan(
        context_start - 0.5,
        len(display) - 0.5,
        alpha=0.10
    )

    current_state = result[
        "current_state"
    ]

    ax.set_title(
        f"{symbol} {timeframe} | "
        f"{current_state['state']} | "
        f"Historical Analogue Analysis"
    )

    ax.set_ylabel("Price")
    ax.set_xlabel(
        "Recent candles"
    )

    ax.grid(
        alpha=0.18
    )

    path = os.path.join(
        CHART_DIR,
        (
            f"{symbol}_"
            f"{timeframe}_"
            f"{int(time.time())}.png"
        )
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=140
    )

    plt.close(fig)

    return path


# ============================================================
# TELEGRAM
# ============================================================

async def telegram_request(
    session,
    method: str,
    data=None,
    form=None
):

    if not TELEGRAM_BOT_TOKEN:
        return False

    url = (
        "https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/"
        f"{method}"
    )

    try:

        if form is not None:

            async with session.post(
                url,
                data=form,
                timeout=aiohttp.ClientTimeout(
                    total=60
                )
            ) as response:

                if response.status != 200:
                    print(
                        "[TELEGRAM]",
                        response.status,
                        await response.text()
                    )
                    return False

                return True

        async with session.post(
            url,
            json=data,
            timeout=aiohttp.ClientTimeout(
                total=60
            )
        ) as response:

            if response.status != 200:
                print(
                    "[TELEGRAM]",
                    response.status,
                    await response.text()
                )
                return False

            return True

    except Exception as exc:

        print(
            "[TELEGRAM ERROR]",
            exc
        )

        return False


async def telegram_text(
    session,
    text: str
):

    if not TELEGRAM_BOT_TOKEN:
        return

    if not TELEGRAM_CHAT_ID:
        return

    # Telegram text limit safety.
    chunks = []

    while len(text) > 3800:

        cut = text.rfind(
            "\n",
            0,
            3800
        )

        if cut < 500:
            cut = 3800

        chunks.append(
            text[:cut]
        )

        text = text[cut:]

    if text:
        chunks.append(text)

    for chunk in chunks:

        await telegram_request(
            session,
            "sendMessage",
            data={
                "chat_id":
                    TELEGRAM_CHAT_ID,
                "text":
                    chunk,
            }
        )

        await asyncio.sleep(
            0.3
        )


async def telegram_photo(
    session,
    path: str,
    caption: str
):

    if not (
        TELEGRAM_BOT_TOKEN
        and TELEGRAM_CHAT_ID
    ):
        return

    try:

        with open(
            path,
            "rb"
        ) as f:

            form = aiohttp.FormData()

            form.add_field(
                "chat_id",
                TELEGRAM_CHAT_ID
            )

            form.add_field(
                "caption",
                caption[:1000]
            )

            form.add_field(
                "photo",
                f,
                filename=os.path.basename(
                    path
                ),
                content_type="image/png"
            )

            await telegram_request(
                session,
                "sendPhoto",
                form=form
            )

    except Exception as exc:

        print(
            "[TELEGRAM PHOTO ERROR]",
            exc
        )


# ============================================================
# SAVE JSON
# ============================================================

def clean_json(obj):

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, dict):
        return {
            k: clean_json(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [
            clean_json(x)
            for x in obj
        ]

    return obj


def save_results(
    symbol: str,
    timeframe: str,
    result: dict
):

    data = {}

    if os.path.exists(
        RESULT_FILE
    ):

        try:

            with open(
                RESULT_FILE,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

        except Exception:
            data = {}

    if symbol not in data:
        data[symbol] = {}

    # Do not store every historical candle.
    compact = {
        "updated_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "current_state":
            result["current_state"],

        "statistics":
            result["statistics"],

        "decision":
            result["decision"],

        "match_count":
            len(result["matches"]),

        "independent_count":
            len(result["independent"]),

        "average_similarity":
            result["average_similarity"],

        "top_analogues": [
            {
                "open_time":
                    x["open_time"],

                "date":
                    utc_text(
                        x["open_time"]
                    ),

                "similarity":
                    x["similarity"],

                "state":
                    x["state"],

                "signature":
                    x["signature"],

                "outcome":
                    x["outcome"],
            }
            for x in result[
                "matches"
            ][:TOP_ANALOGUES]
        ],
    }

    data[symbol][timeframe] = clean_json(
        compact
    )

    tmp = (
        RESULT_FILE
        + ".tmp"
    )

    with open(
        tmp,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    os.replace(
        tmp,
        RESULT_FILE
    )


# ============================================================
# FULL ANALYSIS
# ============================================================

def analyse_candles(
    candles: List[dict]
) -> Optional[dict]:

    if len(candles) < (
        CONTEXT_CANDLES
        + FORWARD_CANDLES
        + 30
    ):
        return None

    result = find_historical_analogues(
        candles
    )

    matches = result[
        "matches"
    ]

    independent = result[
        "independent"
    ]

    # For statistics use independent historical events.
    stats = build_statistics(
        independent
    )

    if independent:

        avg_similarity = float(
            np.mean([
                x["similarity"]
                for x in independent
            ])
        )

    else:

        avg_similarity = 0.0

    decision = determine_scenario(
        result["current_state"],
        stats,
        len(independent),
        avg_similarity
    )

    return {
        "current_state":
            result["current_state"],

        "matches":
            matches,

        "independent":
            independent,

        "statistics":
            stats,

        "average_similarity":
            avg_similarity,

        "decision":
            decision,
    }


# ============================================================
# LIVE BUFFER
# ============================================================

async def refresh_live_buffer(
    session,
    symbol: str,
    timeframe: str
):

    url = (
        f"{BINANCE_REST}/api/v3/klines"
    )

    params = {
        "symbol": symbol,
        "interval": timeframe,
        "limit": CANDLE_BUFFER,
    }

    rows = await http_get_json(
        session,
        url,
        params
    )

    closed = []

    current = now_ms()

    for row in rows:

        c = kline_from_binance(
            row
        )

        if c["close_time"] < current:
            closed.append(c)

    LIVE[
        (symbol, timeframe)
    ] = deque(
        closed,
        maxlen=CANDLE_BUFFER
    )

    print(
        f"[LIVE BUFFER] {symbol} "
        f"{timeframe}: "
        f"{len(closed)}"
    )


def update_live_candle(
    symbol: str,
    timeframe: str,
    candle: dict
):

    key = (
        symbol,
        timeframe
    )

    if key not in LIVE:
        LIVE[key] = deque(
            maxlen=CANDLE_BUFFER
        )

    buf = LIVE[key]

    if buf and (
        buf[-1]["open_time"]
        == candle["open_time"]
    ):

        buf[-1] = candle

    elif not buf or (
        candle["open_time"]
        > buf[-1]["open_time"]
    ):

        buf.append(candle)

    else:

        # Out-of-order candle.
        data = {
            x["open_time"]: x
            for x in buf
        }

        data[
            candle["open_time"]
        ] = candle

        ordered = sorted(
            data.values(),
            key=lambda x:
                x["open_time"]
        )

        LIVE[key] = deque(
            ordered[-CANDLE_BUFFER:],
            maxlen=CANDLE_BUFFER
        )


# ============================================================
# SIGNAL CONFIRMATION
# ============================================================

def signal_confirmation(
    candles: List[dict],
    decision: dict
) -> bool:

    if len(candles) < 3:
        return False

    last = candles[-1]
    prev = candles[-2]

    lf = candle_features(
        last
    )

    if decision["status"] == "LONG_WATCH":

        # Strong bullish close.
        bullish = (
            last["close"]
            > last["open"]
        )

        close_high = (
            lf["close_location"]
            >= 0.70
        )

        breakout = (
            last["close"]
            > prev["high"]
        )

        return (
            bullish
            and close_high
            and breakout
        )

    if decision["status"] == "SHORT_WATCH":

        bearish = (
            last["close"]
            < last["open"]
        )

        close_low = (
            lf["close_location"]
            <= 0.30
        )

        breakout = (
            last["close"]
            < prev["low"]
        )

        return (
            bearish
            and close_low
            and breakout
        )

    return False


# ============================================================
# ANALYSE ONE MARKET
# ============================================================

async def analyse_symbol_tf(
    session,
    symbol: str,
    timeframe: str,
    send_update: bool = False
):

    key = (
        symbol,
        timeframe
    )

    candles = list(
        LIVE.get(
            key,
            []
        )
    )

    if len(candles) < (
        CONTEXT_CANDLES
        + FORWARD_CANDLES
        + 30
    ):
        print(
            f"[ANALYSE] {symbol} "
            f"{timeframe}: not enough candles"
        )
        return None

    result = analyse_candles(
        candles
    )

    if result is None:
        return None

    save_results(
        symbol,
        timeframe,
        result
    )

    report = make_report(
        symbol,
        timeframe,
        candles,
        result
    )

    print("")
    print("=" * 80)
    print(report)
    print("=" * 80)
    print("")

    state = result[
        "current_state"
    ]["state"]

    previous_state = LAST_STATE.get(
        key
    )

    decision = result[
        "decision"
    ]

    state_changed = (
        previous_state != state
    )

    LAST_STATE[key] = state

    # --------------------------------------------------------
    # Continuous important updates.
    # --------------------------------------------------------

    if send_update and (
        state_changed
        or decision["status"]
        in (
            "LONG_WATCH",
            "SHORT_WATCH",
        )
    ):

        await telegram_text(
            session,
            report
        )

    # --------------------------------------------------------
    # Final signal confirmation.
    # --------------------------------------------------------

    confirmed = signal_confirmation(
        candles,
        decision
    )

    cooldown_ok = (
        time.time()
        - LAST_SIGNAL.get(
            key,
            0
        )
        >= SIGNAL_COOLDOWN_SECONDS
    )

    if (
        confirmed
        and cooldown_ok
        and decision["status"]
        in (
            "LONG_WATCH",
            "SHORT_WATCH",
        )
        and len(result["independent"])
        >= SIGNAL_MIN_INDEPENDENT
    ):

        final_report = (
            report
            + "\n\n"
            + "🚨 CONFIRMATION RECEIVED"
            + "\n"
            + "Historical analogue evidence "
              "meets the configured gate."
            + "\n"
            + "This is a statistical setup, "
              "not a guarantee."
        )

        chart = make_chart(
            symbol,
            timeframe,
            candles,
            result
        )

        await telegram_text(
            session,
            final_report
        )

        await telegram_photo(
            session,
            chart,
            (
                f"{symbol} {timeframe} | "
                f"{decision['status']}"
            )
        )

        LAST_SIGNAL[key] = (
            time.time()
        )

        print(
            f"[SIGNAL] {symbol} "
            f"{timeframe}: "
            f"{decision['status']}"
        )

    return result


# ============================================================
# INITIAL HISTORY SCAN
# ============================================================

async def build_all_history(
    session
):

    for symbol in SYMBOLS:

        for timeframe in TIMEFRAMES:

            try:

                history = (
                    await fetch_historical(
                        session,
                        symbol,
                        timeframe
                    )
                )

                HISTORY[
                    (symbol, timeframe)
                ] = history

                # Live buffer initially from history.
                LIVE[
                    (symbol, timeframe)
                ] = deque(
                    history[
                        -CANDLE_BUFFER:
                    ],
                    maxlen=CANDLE_BUFFER
                )

            except Exception as exc:

                print(
                    f"[INIT ERROR] "
                    f"{symbol} {timeframe}: "
                    f"{exc}"
                )

                traceback.print_exc()


# ============================================================
# PERIODIC RESCAN
# ============================================================

async def periodic_rescan(
    session
):

    while True:

        try:

            await asyncio.sleep(
                RESCAN_SECONDS
            )

            async with SCAN_LOCK:

                print(
                    "[RESCAN] Historical "
                    "data refresh started."
                )

                for symbol in SYMBOLS:

                    for timeframe in TIMEFRAMES:

                        try:

                            history = (
                                await fetch_historical(
                                    session,
                                    symbol,
                                    timeframe
                                )
                            )

                            if history:

                                HISTORY[
                                    (
                                        symbol,
                                        timeframe
                                    )
                                ] = history

                                LIVE[
                                    (
                                        symbol,
                                        timeframe
                                    )
                                ] = deque(
                                    history[
                                        -CANDLE_BUFFER:
                                    ],
                                    maxlen=CANDLE_BUFFER
                                )

                        except Exception as exc:

                            print(
                                "[RESCAN ERROR]",
                                symbol,
                                timeframe,
                                exc
                            )

                print(
                    "[RESCAN] Complete."
                )

        except asyncio.CancelledError:
            raise

        except Exception as exc:

            print(
                "[RESCAN LOOP ERROR]",
                exc
            )

            await asyncio.sleep(
                30
            )


# ============================================================
# WEBSOCKET
# ============================================================

async def listen_symbol(
    session,
    symbol: str,
    timeframe: str
):

    stream = (
        f"{symbol.lower()}"
        f"@kline_{timeframe}"
    )

    url = (
        f"{BINANCE_WS}/ws/{stream}"
    )

    while True:

        try:

            print(
                f"[WS] connecting "
                f"{symbol} {timeframe}"
            )

            # websocket timeout deliberately generous.
            timeout = aiohttp.ClientTimeout(
                total=None
            )

            async with session.ws_connect(
                url,
                heartbeat=20,
                timeout=timeout
            ) as ws:

                print(
                    f"[WS] connected "
                    f"{symbol} {timeframe}"
                )

                # Refresh after reconnect so that
                # missed candles are restored.
                try:

                    await refresh_live_buffer(
                        session,
                        symbol,
                        timeframe
                    )

                except Exception as exc:

                    print(
                        "[WS] buffer refresh error:",
                        exc
                    )

                async for msg in ws:

                    if msg.type == aiohttp.WSMsgType.TEXT:

                        try:
                            payload = json.loads(
                                msg.data
                            )
                        except Exception:
                            continue

                        k = payload.get(
                            "k"
                        )

                        if not k:
                            continue

                        candle = (
                            kline_from_ws(k)
                        )

                        update_live_candle(
                            symbol,
                            timeframe,
                            candle
                        )

                        # Only analyze CLOSED candles.
                        if not bool(
                            k.get("x", False)
                        ):
                            continue

                        print(
                            f"[CLOSED] "
                            f"{symbol} "
                            f"{timeframe} "
                            f"{utc_text(candle['close_time'])}"
                        )

                        try:

                            await analyse_symbol_tf(
                                session,
                                symbol,
                                timeframe,
                                send_update=True
                            )

                        except Exception as exc:

                            print(
                                "[ANALYSE ERROR]",
                                symbol,
                                timeframe,
                                exc
                            )

                            traceback.print_exc()

                    elif msg.type in (
                        aiohttp.WSMsgType.ERROR,
                        aiohttp.WSMsgType.CLOSED,
                    ):

                        print(
                            f"[WS] closed/error "
                            f"{symbol} "
                            f"{timeframe}"
                        )

                        break

        except asyncio.CancelledError:
            raise

        except Exception as exc:

            print(
                f"[WS ERROR] "
                f"{symbol} "
                f"{timeframe}: "
                f"{exc}"
            )

        print(
            f"[WS] reconnecting "
            f"{symbol} {timeframe} "
            f"in 5 sec..."
        )

        await asyncio.sleep(5)


# ============================================================
# STARTUP ANALYSIS
# ============================================================

async def startup_analysis(
    session
):

    print(
        "\n[STARTUP] Running initial "
        "historical analogue analysis...\n"
    )

    for symbol in SYMBOLS:

        for timeframe in TIMEFRAMES:

            try:

                await analyse_symbol_tf(
                    session,
                    symbol,
                    timeframe,
                    send_update=False
                )

            except Exception as exc:

                print(
                    "[STARTUP ANALYSIS ERROR]",
                    symbol,
                    timeframe,
                    exc
                )

    print(
        "\n[STARTUP] Analysis complete.\n"
    )


# ============================================================
# MAIN
# ============================================================

async def main():

    print("")
    print("=" * 80)
    print("DEEP MARKET ANALOGUE BOT v1")
    print("=" * 80)
    print(
        "Symbols:",
        ", ".join(SYMBOLS)
    )
    print(
        "Timeframes:",
        ", ".join(TIMEFRAMES)
    )
    print(
        "Context candles:",
        CONTEXT_CANDLES
    )
    print(
        "Forward candles:",
        FORWARD_CANDLES
    )
    print(
        "Minimum similarity:",
        MIN_SIMILARITY
    )
    print(
        "Telegram:",
        "ENABLED"
        if TELEGRAM_BOT_TOKEN
        else "DISABLED"
    )
    print("=" * 80)
    print("")

    connector = aiohttp.TCPConnector(
        limit=50,
        ttl_dns_cache=300
    )

    async with aiohttp.ClientSession(
        connector=connector
    ) as session:

        # 1. Load history.
        await build_all_history(
            session
        )

        # 2. Initial deep analysis.
        await startup_analysis(
            session
        )

        # 3. Start periodic historical refresh.
        rescan_task = asyncio.create_task(
            periodic_rescan(
                session
            )
        )

        # 4. Start live websocket listeners.
        tasks = [
            asyncio.create_task(
                listen_symbol(
                    session,
                    symbol,
                    timeframe
                )
            )
            for symbol in SYMBOLS
            for timeframe in TIMEFRAMES
        ]

        print(
            "[BOT] LIVE MONITORING STARTED"
        )

        try:

            await asyncio.gather(
                *tasks
            )

        finally:

            rescan_task.cancel()

            for task in tasks:
                task.cancel()

            await asyncio.gather(
                rescan_task,
                *tasks,
                return_exceptions=True
            )


if __name__ == "__main__":

    try:
        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        print(
            "\nBOT STOPPED"
        )

    except Exception as exc:

        print(
            "\nFATAL ERROR:",
            exc
        )

        traceback.print_exc()