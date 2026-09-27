# ============================================================
# PRICE ACTION DEEP PATTERN ENGINE v10.0
# ============================================================
#
# PURE PRICE ACTION
# NO RSI
# NO MACD
# NO EMA
# NO SMA
# NO BOLLINGER
# NO STOCHASTIC
#
# ATR is used ONLY for risk normalization / R-unit measurement.
#
# FEATURES
# ------------------------------------------------------------
# 1. Historical Binance OHLCV
# 2. 100+ named Price Action structures
# 3. Automatic candle-sequence signatures
# 4. Market context analysis
# 5. Swing structure
# 6. Support / resistance reaction
# 7. Breakout / false breakout / retest
# 8. Reversal / continuation context
# 9. Historical analog search
# 10. MFE / MAE
# 11. 1R / 2R / 3R / 5R outcome statistics
# 12. Long / Short historical statistics
# 13. Holdout validation
# 14. Recent validation
# 15. Bootstrap validation
# 16. Telegram chart + full report
# 17. Closed-candle live signals only
# 18. WebSocket reconnect + buffer refresh
# 19. Persistent JSON results
#
# INSTALL:
# pip install python-binance aiohttp matplotlib numpy python-dotenv
#
# RUN:
# python live_bot.py
#
# ============================================================

import os
import json
import math
import time
import asyncio
import hashlib
import random
import traceback
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone

import numpy as np
import aiohttp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from binance import AsyncClient, BinanceSocketManager

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


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

SEQ_LENGTHS = [
    int(x.strip())
    for x in os.getenv(
        "SEQ_LENGTHS",
        "2"
    ).split(",")
    if x.strip()
]

MIN_OCCURRENCES = int(os.getenv("MIN_OCCURRENCES", "30"))
MIN_TRAIN_OCCURRENCES = int(
    os.getenv("MIN_TRAIN_OCCURRENCES", "20")
)
MIN_HOLDOUT_OCCURRENCES = int(
    os.getenv("MIN_HOLDOUT_OCCURRENCES", "8")
)
MIN_INDEPENDENT_OCCURRENCES = int(
    os.getenv("MIN_INDEPENDENT_OCCURRENCES", "25")
)

MIN_WIN_RATE = float(os.getenv("MIN_WIN_RATE", "65"))
MAX_LOSS_RATE = float(os.getenv("MAX_LOSS_RATE", "35"))
MAX_LOSSES_ABSOLUTE = int(
    os.getenv("MAX_LOSSES_ABSOLUTE", "15")
)

MIN_PROFIT_FACTOR = float(
    os.getenv("MIN_PROFIT_FACTOR", "1.30")
)

MIN_AVG_R = float(
    os.getenv("MIN_AVG_R", "0.15")
)

MIN_CONFIDENCE = float(
    os.getenv("MIN_CONFIDENCE", "60")
)

MIN_HOLDOUT_WR = float(
    os.getenv("MIN_HOLDOUT_WR", "55")
)

HOLDOUT_FRACTION = float(
    os.getenv("HOLDOUT_FRACTION", "0.20")
)

VALIDATION_SEGMENTS = int(
    os.getenv("VALIDATION_SEGMENTS", "5")
)

MIN_SEGMENT_WR = float(
    os.getenv("MIN_SEGMENT_WR", "50")
)

DEEP_VALIDATION_PASSES = int(
    os.getenv("DEEP_VALIDATION_PASSES", "50")
)

DEEP_PASS_RATE_REQUIRED = float(
    os.getenv("DEEP_PASS_RATE_REQUIRED", "60")
)

BOOTSTRAP_LOWER_Q = float(
    os.getenv("BOOTSTRAP_LOWER_Q", "10")
)

MIN_BOOTSTRAP_LCB = float(
    os.getenv("MIN_BOOTSTRAP_LCB", "55")
)

RECENT_OCCURRENCES = int(
    os.getenv("RECENT_OCCURRENCES", "30")
)

MIN_RECENT_WR = float(
    os.getenv("MIN_RECENT_WR", "55")
)

CONTEXT_CANDLES = int(
    os.getenv("CONTEXT_CANDLES", "8")
)

MIN_DIRECTIONAL_CONSISTENCY = float(
    os.getenv("MIN_DIRECTIONAL_CONSISTENCY", "0.45")
)

FORWARD_CANDLES = int(
    os.getenv("FORWARD_CANDLES", "50")
)

ATR_PERIOD = int(
    os.getenv("ATR_PERIOD", "50")
)

SL_BUF = float(
    os.getenv("SL_BUF", "10")
)

TP1_R = float(
    os.getenv("TP1_R", "1.0")
)

TP1_CLOSE_PCT = float(
    os.getenv("TP1_CLOSE_PCT", "0.50")
)

TRAIL_START_R = float(
    os.getenv("TRAIL_START_R", "4.0")
)

TRAIL_STEP_R = float(
    os.getenv("TRAIL_STEP_R", "2.0")
)

MAX_TRAIL_R = float(
    os.getenv("MAX_TRAIL_R", "10.0")
)

CANDLE_BUFFER = int(
    os.getenv("CANDLE_BUFFER", "500")
)

RESCAN_SECONDS = int(
    os.getenv("RESCAN_SECONDS", "3600")
)

SIGNAL_COOLDOWN_SECONDS = int(
    os.getenv("SIGNAL_COOLDOWN_SECONDS", "1800")
)

WS_RECONNECT_SECONDS = int(
    os.getenv("WS_RECONNECT_SECONDS", "72000")
)

REQUEST_DELAY = float(
    os.getenv("REQUEST_DELAY", "0.25")
)

MAX_RETRIES = int(
    os.getenv("MAX_RETRIES", "6")
)

TOP_PATTERNS_PER_TF = int(
    os.getenv("TOP_PATTERNS_PER_TF", "5")
)

DAYS_5M = int(
    os.getenv("DAYS_5M", "1825")
)

DAYS_15M = int(
    os.getenv("DAYS_15M", "3650")
)

DAYS_1H = int(
    os.getenv("DAYS_1H", "3650")
)

SAVE_RESULTS = (
    os.getenv("SAVE_RESULTS", "true").lower()
    in ("1", "true", "yes", "y")
)

RESULT_FILE = os.getenv(
    "RESULT_FILE",
    "deep_patterns_v10.json"
)

TELEGRAM_BOT_TOKEN = (
    os.getenv("TELEGRAM_BOT_TOKEN")
    or os.getenv("TELEGRAM_TOKEN")
    or ""
)

TELEGRAM_CHAT_ID = os.getenv(
    "TELEGRAM_CHAT_ID",
    ""
)

BINANCE_API_KEY = os.getenv(
    "BINANCE_API_KEY",
    ""
)

BINANCE_API_SECRET = os.getenv(
    "BINANCE_API_SECRET",
    ""
)


# ============================================================
# BASIC HELPERS
# ============================================================

def now_ms():
    return int(time.time() * 1000)


def utc_string(ms=None):
    if ms is None:
        ms = now_ms()

    return datetime.fromtimestamp(
        ms / 1000,
        tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def clamp(x, low, high):
    return max(low, min(high, x))


def pct(a, b):
    if b == 0:
        return 0.0

    return 100.0 * a / b


def mean_or_zero(values):
    if not values:
        return 0.0

    return float(np.mean(values))


def median_or_zero(values):
    if not values:
        return 0.0

    return float(np.median(values))


def sha1_text(text):
    return hashlib.sha1(
        text.encode("utf-8")
    ).hexdigest()[:16]


# ============================================================
# CANDLE MODEL
# ============================================================

@dataclass
class Candle:
    open_time: int
    close_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    def direction(self):
        if self.close > self.open:
            return "U"

        if self.close < self.open:
            return "D"

        return "F"

    def body(self):
        return abs(self.close - self.open)

    def range(self):
        return max(
            self.high - self.low,
            1e-12
        )

    def upper_wick(self):
        return max(
            0.0,
            self.high - max(
                self.open,
                self.close
            )
        )

    def lower_wick(self):
        return max(
            0.0,
            min(
                self.open,
                self.close
            ) - self.low
        )

    def body_ratio(self):
        return self.body() / self.range()

    def upper_ratio(self):
        return self.upper_wick() / self.range()

    def lower_ratio(self):
        return self.lower_wick() / self.range()


# ============================================================
# BINANCE DATA
# ============================================================

def interval_to_ms(interval):
    unit = interval[-1]
    value = int(interval[:-1])

    if unit == "m":
        return value * 60_000

    if unit == "h":
        return value * 3_600_000

    if unit == "d":
        return value * 86_400_000

    if unit == "w":
        return value * 7 * 86_400_000

    return 60_000


def days_for_tf(tf):
    if tf == "5m":
        return DAYS_5M

    if tf == "15m":
        return DAYS_15M

    if tf == "1h":
        return DAYS_1H

    return 365


async def request_with_retry(
    func,
    *args,
    **kwargs
):
    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):
        try:
            result = await func(
                *args,
                **kwargs
            )

            await asyncio.sleep(
                REQUEST_DELAY
            )

            return result

        except Exception as exc:
            last_error = exc

            wait = min(
                30,
                2 ** (attempt - 1)
            )

            print(
                f"[BINANCE RETRY] "
                f"{attempt}/{MAX_RETRIES} "
                f"error={exc}"
            )

            await asyncio.sleep(wait)

    raise last_error


async def fetch_history(
    client,
    symbol,
    timeframe,
    days
):
    end_ms = now_ms()
    start_ms = (
        end_ms
        - days * 86_400_000
    )

    interval_ms = interval_to_ms(
        timeframe
    )

    rows = []

    cursor = start_ms

    print(
        f"[HISTORY] {symbol} {timeframe} "
        f"{days} days"
    )

    while cursor < end_ms:
        batch = await request_with_retry(
            client.get_klines,
            symbol=symbol,
            interval=timeframe,
            startTime=cursor,
            endTime=end_ms,
            limit=1000
        )

        if not batch:
            break

        for row in batch:
            open_time = int(row[0])

            close_time = int(row[6])

            if close_time > end_ms:
                continue

            rows.append(
                Candle(
                    open_time=open_time,
                    close_time=close_time,
                    open=safe_float(row[1]),
                    high=safe_float(row[2]),
                    low=safe_float(row[3]),
                    close=safe_float(row[4]),
                    volume=safe_float(row[5])
                )
            )

        last_open = int(
            batch[-1][0]
        )

        next_cursor = (
            last_open + interval_ms
        )

        if next_cursor <= cursor:
            break

        cursor = next_cursor

        if len(batch) < 1000:
            break

        print(
            f"[HISTORY] {symbol} {timeframe} "
            f"candles={len(rows)}"
        )

    unique = {}

    for c in rows:
        unique[c.open_time] = c

    candles = sorted(
        unique.values(),
        key=lambda x: x.open_time
    )

    print(
        f"[HISTORY DONE] {symbol} {timeframe} "
        f"candles={len(candles)}"
    )

    return candles


async def fetch_recent_history(
    client,
    symbol,
    timeframe,
    limit
):
    rows = await request_with_retry(
        client.get_klines,
        symbol=symbol,
        interval=timeframe,
        limit=min(
            1000,
            max(50, limit)
        )
    )

    result = []

    cutoff = now_ms()

    for row in rows:
        close_time = int(row[6])

        if close_time > cutoff:
            continue

        result.append(
            Candle(
                open_time=int(row[0]),
                close_time=close_time,
                open=safe_float(row[1]),
                high=safe_float(row[2]),
                low=safe_float(row[3]),
                close=safe_float(row[4]),
                volume=safe_float(row[5])
            )
        )

    return result


# ============================================================
# ATR
# ============================================================

def compute_atr(candles, period=50):
    if len(candles) < 2:
        return 0.0

    trs = []

    for i in range(1, len(candles)):
        c = candles[i]
        p = candles[i - 1]

        tr = max(
            c.high - c.low,
            abs(c.high - p.close),
            abs(c.low - p.close)
        )

        trs.append(tr)

    if not trs:
        return 0.0

    return float(
        np.mean(
            trs[-period:]
        )
    )


# ============================================================
# CANDLE CLASSIFICATION
# ============================================================

def body_class(c):
    ratio = c.body_ratio()

    if ratio < 0.10:
        return "S"

    if ratio < 0.35:
        return "M"

    if ratio < 0.70:
        return "L"

    return "X"


def wick_class(c):
    up = c.upper_ratio()
    dn = c.lower_ratio()

    if (
        up < 0.15
        and dn < 0.15
    ):
        return "N"

    if (
        up >= 0.35
        and dn >= 0.35
    ):
        return "B"

    if up >= 0.35:
        return "U"

    if dn >= 0.35:
        return "D"

    if up >= 0.20:
        return "u"

    if dn >= 0.20:
        return "d"

    return "C"


def candle_code(c):
    return (
        c.direction()
        + body_class(c)
        + wick_class(c)
    )


# ============================================================
# RAW PRICE ACTION CONTEXT
# ============================================================

def directional_consistency(
    candles
):
    if len(candles) < 2:
        return 0.0

    dirs = [
        1 if c.close > c.open else
        -1 if c.close < c.open else
        0
        for c in candles
    ]

    nonzero = [
        x for x in dirs
        if x != 0
    ]

    if not nonzero:
        return 0.0

    positives = sum(
        x == 1
        for x in nonzero
    )

    negatives = sum(
        x == -1
        for x in nonzero
    )

    return abs(
        positives - negatives
    ) / len(nonzero)


def price_action_context(
    candles,
    index
):
    if index < 5:
        return {
            "trend": "UNKNOWN",
            "reversal_score": 0.0,
            "breakout": False,
            "false_breakout": False,
            "range_compression": False,
            "directional_consistency": 0.0,
            "swing_position": "MID",
            "efficiency": 0.0
        }

    start = max(
        0,
        index - CONTEXT_CANDLES
    )

    ctx = candles[
        start:index + 1
    ]

    first = ctx[0].open
    last = ctx[-1].close

    change = (
        last - first
    ) / max(
        abs(first),
        1e-12
    )

    if change > 0.003:
        trend = "UP"

    elif change < -0.003:
        trend = "DOWN"

    else:
        trend = "RANGE"

    highs = [
        c.high for c in ctx
    ]

    lows = [
        c.low for c in ctx
    ]

    recent_high = max(
        highs[:-1]
    )

    recent_low = min(
        lows[:-1]
    )

    current = ctx[-1]

    breakout_up = (
        current.close
        > recent_high
    )

    breakout_down = (
        current.close
        < recent_low
    )

    false_up = (
        current.high > recent_high
        and current.close <= recent_high
    )

    false_down = (
        current.low < recent_low
        and current.close >= recent_low
    )

    ranges = [
        c.range()
        for c in ctx
    ]

    recent_range = mean_or_zero(
        ranges[-3:]
    )

    old_range = mean_or_zero(
        ranges[:-3]
    )

    compression = (
        old_range > 0
        and recent_range
        < old_range * 0.65
    )

    net_move = abs(
        last - first
    )

    path = sum(
        abs(
            ctx[i].close
            - ctx[i - 1].close
        )
        for i in range(
            1,
            len(ctx)
        )
    )

    efficiency = (
        net_move / path
        if path > 0
        else 0.0
    )

    prev = ctx[-2]

    reversal_score = 0.0

    if (
        prev.close < prev.open
        and current.close > current.open
    ):
        reversal_score += 0.30

    if (
        prev.close > prev.open
        and current.close < current.open
    ):
        reversal_score += 0.30

    if current.lower_ratio() >= 0.35:
        reversal_score += 0.25

    if current.upper_ratio() >= 0.35:
        reversal_score += 0.25

    if (
        false_up
        or false_down
    ):
        reversal_score += 0.25

    swing_range = (
        max(highs)
        - min(lows)
    )

    if swing_range <= 0:
        swing_position = "MID"

    else:
        pos = (
            current.close
            - min(lows)
        ) / swing_range

        if pos >= 0.80:
            swing_position = "HIGH"

        elif pos <= 0.20:
            swing_position = "LOW"

        else:
            swing_position = "MID"

    return {
        "trend": trend,
        "reversal_score": clamp(
            reversal_score,
            0.0,
            1.0
        ),
        "breakout": (
            breakout_up
            or breakout_down
        ),
        "breakout_direction": (
            "UP"
            if breakout_up
            else
            "DOWN"
            if breakout_down
            else
            "NONE"
        ),
        "false_breakout": (
            false_up
            or false_down
        ),
        "false_breakout_direction": (
            "UP"
            if false_up
            else
            "DOWN"
            if false_down
            else
            "NONE"
        ),
        "range_compression": compression,
        "directional_consistency":
            directional_consistency(ctx),
        "swing_position":
            swing_position,
        "efficiency":
            efficiency,
        "change_pct":
            change * 100
    }


# ============================================================
# SWING ANALYSIS
# ============================================================

def local_swing_high(
    candles,
    i,
    radius=2
):
    if (
        i < radius
        or i + radius >= len(candles)
    ):
        return False

    h = candles[i].high

    for j in range(
        i - radius,
        i + radius + 1
    ):
        if j == i:
            continue

        if candles[j].high > h:
            return False

    return True


def local_swing_low(
    candles,
    i,
    radius=2
):
    if (
        i < radius
        or i + radius >= len(candles)
    ):
        return False

    low = candles[i].low

    for j in range(
        i - radius,
        i + radius + 1
    ):
        if j == i:
            continue

        if candles[j].low < low:
            return False

    return True


def recent_swings(
    candles,
    index,
    radius=2,
    lookback=40
):
    start = max(
        radius,
        index - lookback
    )

    end = min(
        len(candles) - radius,
        index
    )

    highs = []
    lows = []

    for i in range(
        start,
        end
    ):
        if local_swing_high(
            candles,
            i,
            radius
        ):
            highs.append(
                (i, candles[i].high)
            )

        if local_swing_low(
            candles,
            i,
            radius
        ):
            lows.append(
                (i, candles[i].low)
            )

    return highs, lows


# ============================================================
# CLASSIC PATTERN DETECTION
# ============================================================

def detect_candlestick_patterns(
    candles,
    i
):
    names = set()

    if i < 5:
        return names

    c0 = candles[i]
    c1 = candles[i - 1]
    c2 = candles[i - 2]
    c3 = candles[i - 3]
    c4 = candles[i - 4]

    body0 = c0.body()
    body1 = c1.body()
    body2 = c2.body()

    r0 = c0.range()
    r1 = c1.range()
    r2 = c2.range()

    bull0 = c0.close > c0.open
    bear0 = c0.close < c0.open

    bull1 = c1.close > c1.open
    bear1 = c1.close < c1.open

    bull2 = c2.close > c2.open
    bear2 = c2.close < c2.open

    # --------------------------------------------------------
    # SINGLE CANDLE
    # --------------------------------------------------------

    if c0.body_ratio() < 0.10:
        names.add("DOJI")

    if (
        c0.lower_ratio() >= 0.55
        and c0.body_ratio() <= 0.35
    ):
        names.add("HAMMER")

    if (
        c0.upper_ratio() >= 0.55
        and c0.body_ratio() <= 0.35
    ):
        names.add("SHOOTING_STAR")

    if (
        c0.lower_ratio() >= 0.55
        and c0.body_ratio() <= 0.35
        and bear0
    ):
        names.add("HANGING_MAN")

    if (
        c0.upper_ratio() >= 0.55
        and c0.body_ratio() <= 0.35
        and bull0
    ):
        names.add("INVERTED_HAMMER")

    if (
        c0.body_ratio() >= 0.80
        and c0.upper_ratio() <= 0.10
        and c0.lower_ratio() <= 0.10
    ):
        if bull0:
            names.add("BULLISH_MARUBOZU")
        elif bear0:
            names.add("BEARISH_MARUBOZU")

    if (
        c0.body_ratio() <= 0.15
        and c0.upper_ratio() >= 0.30
        and c0.lower_ratio() >= 0.30
    ):
        names.add("LONG_LEGGED_DOJI")

    if (
        c0.body_ratio() <= 0.12
        and c0.lower_ratio() >= 0.45
        and c0.upper_ratio() <= 0.15
    ):
        names.add("DRAGONFLY_DOJI")

    if (
        c0.body_ratio() <= 0.12
        and c0.upper_ratio() >= 0.45
        and c0.lower_ratio() <= 0.15
    ):
        names.add("GRAVESTONE_DOJI")

    # --------------------------------------------------------
    # TWO CANDLE
    # --------------------------------------------------------

    if bull0 and bear1:
        if (
            c0.open <= c1.close
            and c0.close >= c1.open
            and body0 >= body1 * 0.9
        ):
            names.add(
                "BULLISH_ENGULFING"
            )

    if bear0 and bull1:
        if (
            c0.open >= c1.close
            and c0.close <= c1.open
            and body0 >= body1 * 0.9
        ):
            names.add(
                "BEARISH_ENGULFING"
            )

    if (
        body1 > 0
        and body0 <= body1 * 0.45
        and (
            min(c0.open, c0.close)
            >= min(c1.open, c1.close)
            and
            max(c0.open, c0.close)
            <= max(c1.open, c1.close)
        )
    ):
        if bull1:
            names.add("BULLISH_HARAMI")
        elif bear1:
            names.add("BEARISH_HARAMI")

    if (
        bull0
        and bear1
        and c0.open < c1.close
        and c0.close > c1.open
    ):
        names.add("PIERCING_LINE")

    if (
        bear0
        and bull1
        and c0.open > c1.close
        and c0.close < c1.open
    ):
        names.add("DARK_CLOUD")

    if (
        abs(c0.high - c1.high)
        / max(c1.high, 1e-12)
        < 0.0015
        and bear0
        and bull1
    ):
        names.add("TWEEZER_TOP")

    if (
        abs(c0.low - c1.low)
        / max(c1.low, 1e-12)
        < 0.0015
        and bull0
        and bear1
    ):
        names.add("TWEEZER_BOTTOM")

    # --------------------------------------------------------
    # THREE CANDLE
    # --------------------------------------------------------

    if (
        bear2
        and body2 > 0
        and body1 <= body2 * 0.50
        and bull0
        and c0.close
        > (c2.open + c2.close) / 2
    ):
        names.add("MORNING_STAR")

    if (
        bull2
        and body2 > 0
        and body1 <= body2 * 0.50
        and bear0
        and c0.close
        < (c2.open + c2.close) / 2
    ):
        names.add("EVENING_STAR")

    if (
        bull0
        and bull1
        and bull2
        and c0.close > c1.close > c2.close
        and c0.open > c1.open
        and c1.open > c2.open
    ):
        names.add("THREE_WHITE_SOLDIERS")

    if (
        bear0
        and bear1
        and bear2
        and c0.close < c1.close < c2.close
        and c0.open < c1.open
        and c1.open < c2.open
    ):
        names.add("THREE_BLACK_CROWS")

    if (
        bull2
        and abs(c1.close - c1.open)
        < c2.body() * 0.40
        and bear0
    ):
        names.add("BEARISH_ABANDONED_BABY")

    if (
        bear2
        and abs(c1.close - c1.open)
        < c2.body() * 0.40
        and bull0
    ):
        names.add("BULLISH_ABANDONED_BABY")

    # --------------------------------------------------------
    # INSIDE / OUTSIDE BAR
    # --------------------------------------------------------

    if (
        c0.high <= c1.high
        and c0.low >= c1.low
    ):
        names.add("INSIDE_BAR")

    if (
        c0.high >= c1.high
        and c0.low <= c1.low
    ):
        names.add("OUTSIDE_BAR")

    # --------------------------------------------------------
    # RANGE EXPANSION / CONTRACTION
    # --------------------------------------------------------

    if (
        r0 > r1 * 1.8
        and body0 > r0 * 0.60
    ):
        names.add("RANGE_EXPANSION")

    if (
        r0 < r1 * 0.55
    ):
        names.add("RANGE_CONTRACTION")

    # --------------------------------------------------------
    # CONSECUTIVE CANDLES
    # --------------------------------------------------------

    if (
        bull0
        and bull1
        and bull2
    ):
        names.add(
            "THREE_BULLISH_SEQUENCE"
        )

    if (
        bear0
        and bear1
        and bear2
    ):
        names.add(
            "THREE_BEARISH_SEQUENCE"
        )

    if (
        bull0
        and bull1
        and bull2
        and c0.close > c1.close > c2.close
    ):
        names.add(
            "BULLISH_STAIRCASE"
        )

    if (
        bear0
        and bear1
        and bear2
        and c0.close < c1.close < c2.close
    ):
        names.add(
            "BEARISH_STAIRCASE"
        )

    # --------------------------------------------------------
    # REJECTION
    # --------------------------------------------------------

    if (
        c0.lower_ratio() >= 0.40
        and c0.close
        > c0.low + c0.range() * 0.65
    ):
        names.add("LOWER_REJECTION")

    if (
        c0.upper_ratio() >= 0.40
        and c0.close
        < c0.low + c0.range() * 0.35
    ):
        names.add("UPPER_REJECTION")

    # --------------------------------------------------------
    # GAP-LIKE DISLOCATION
    # --------------------------------------------------------

    if c0.open > c1.high:
        names.add("BULLISH_GAP_DISLOCATION")

    if c0.open < c1.low:
        names.add("BEARISH_GAP_DISLOCATION")

    return names


# ============================================================
# STRUCTURE PATTERNS
# ============================================================

def detect_structure_patterns(
    candles,
    i
):
    names = set()

    if i < 10:
        return names

    highs, lows = recent_swings(
        candles,
        i,
        radius=2,
        lookback=50
    )

    if len(highs) >= 2:
        h1 = highs[-2][1]
        h2 = highs[-1][1]

        tolerance = max(
            abs(h1) * 0.005,
            compute_atr(
                candles[
                    max(0, i - 30):i + 1
                ],
                min(ATR_PERIOD, 30)
            ) * 1.5
        )

        if abs(h1 - h2) <= tolerance:
            names.add("DOUBLE_TOP")

        if h2 > h1 * 1.002:
            names.add("HIGHER_HIGH")

        if h2 < h1 * 0.998:
            names.add("LOWER_HIGH")

    if len(lows) >= 2:
        l1 = lows[-2][1]
        l2 = lows[-1][1]

        tolerance = max(
            abs(l1) * 0.005,
            compute_atr(
                candles[
                    max(0, i - 30):i + 1
                ],
                min(ATR_PERIOD, 30)
            ) * 1.5
        )

        if abs(l1 - l2) <= tolerance:
            names.add("DOUBLE_BOTTOM")

        if l2 > l1 * 1.002:
            names.add("HIGHER_LOW")

        if l2 < l1 * 0.998:
            names.add("LOWER_LOW")

    current = candles[i]

    previous_high = max(
        c.high
        for c in candles[
            max(0, i - 20):i
        ]
    )

    previous_low = min(
        c.low
        for c in candles[
            max(0, i - 20):i
        ]
    )

    if current.close > previous_high:
        names.add("BREAKOUT_UP")

    if current.close < previous_low:
        names.add("BREAKOUT_DOWN")

    if (
        current.high > previous_high
        and current.close <= previous_high
    ):
        names.add(
            "FALSE_BREAKOUT_UP"
        )

    if (
        current.low < previous_low
        and current.close >= previous_low
    ):
        names.add(
            "FALSE_BREAKOUT_DOWN"
        )

    # --------------------------------------------------------
    # M / W APPROXIMATION
    # --------------------------------------------------------

    recent = candles[
        max(0, i - 8):i + 1
    ]

    if len(recent) >= 5:
        p = [
            c.close
            for c in recent
        ]

        a = p[-5]
        b = p[-4]
        c = p[-3]
        d = p[-2]
        e = p[-1]

        if (
            b > a
            and b > c
            and d > c
            and d > e
            and abs(b - d)
            / max(abs(b), 1e-12)
            < 0.02
        ):
            names.add("M_TOP")

        if (
            b < a
            and b < c
            and d < c
            and d < e
            and abs(b - d)
            / max(abs(b), 1e-12)
            < 0.02
        ):
            names.add("W_BOTTOM")

    # --------------------------------------------------------
    # HEAD AND SHOULDERS APPROXIMATION
    # --------------------------------------------------------

    if len(highs) >= 3:
        h1 = highs[-3][1]
        h2 = highs[-2][1]
        h3 = highs[-1][1]

        shoulder_tolerance = (
            max(h1, h3) * 0.03
        )

        if (
            abs(h1 - h3)
            <= shoulder_tolerance
            and h2 > h1
            and h2 > h3
        ):
            names.add("HEAD_AND_SHOULDERS")

    if len(lows) >= 3:
        l1 = lows[-3][1]
        l2 = lows[-2][1]
        l3 = lows[-1][1]

        shoulder_tolerance = (
            max(l1, l3) * 0.03
        )

        if (
            abs(l1 - l3)
            <= shoulder_tolerance
            and l2 < l1
            and l2 < l3
        ):
            names.add(
                "INVERSE_HEAD_AND_SHOULDERS"
            )

    return names


# ============================================================
# CONTEXT PATTERNS
# ============================================================

def detect_context_patterns(
    candles,
    i
):
    names = set()

    if i < 10:
        return names

    ctx = price_action_context(
        candles,
        i
    )

    if ctx["trend"] == "UP":
        names.add("UPTREND_CONTEXT")

    if ctx["trend"] == "DOWN":
        names.add("DOWNTREND_CONTEXT")

    if ctx["trend"] == "RANGE":
        names.add("RANGE_CONTEXT")

    if ctx["range_compression"]:
        names.add(
            "COMPRESSION_CONTEXT"
        )

    if ctx["breakout"]:
        names.add(
            "BREAKOUT_CONTEXT"
        )

    if ctx["false_breakout"]:
        names.add(
            "FALSE_BREAKOUT_CONTEXT"
        )

    if ctx["efficiency"] >= 0.70:
        names.add(
            "HIGH_EFFICIENCY_IMPULSE"
        )

    if ctx["efficiency"] <= 0.25:
        names.add(
            "CHOPPY_PRICE_ACTION"
        )

    if (
        ctx["reversal_score"]
        >= 0.60
    ):
        names.add(
            "REVERSAL_CONTEXT"
        )

    if (
        ctx["directional_consistency"]
        >= 0.70
    ):
        names.add(
            "STRONG_DIRECTIONAL_SEQUENCE"
        )

    current = candles[i]

    if (
        current.close
        > current.open
        and ctx["trend"] == "DOWN"
    ):
        names.add(
            "COUNTERTREND_BULLISH_REVERSAL"
        )

    if (
        current.close
        < current.open
        and ctx["trend"] == "UP"
    ):
        names.add(
            "COUNTERTREND_BEARISH_REVERSAL"
        )

    if (
        current.lower_ratio() >= 0.40
        and ctx["swing_position"] == "LOW"
    ):
        names.add(
            "LOW_ZONE_REJECTION"
        )

    if (
        current.upper_ratio() >= 0.40
        and ctx["swing_position"] == "HIGH"
    ):
        names.add(
            "HIGH_ZONE_REJECTION"
        )

    # --------------------------------------------------------
    # IMPULSE / CORRECTION
    # --------------------------------------------------------

    recent = candles[
        max(0, i - 12):i + 1
    ]

    if len(recent) >= 8:
        first = recent[0].close
        middle = recent[4].close
        last = recent[-1].close

        impulse = (
            middle - first
        )

        correction = (
            last - middle
        )

        if (
            impulse > 0
            and correction < 0
            and abs(correction)
            < abs(impulse) * 0.70
        ):
            names.add(
                "BULLISH_IMPULSE_CORRECTION"
            )

        if (
            impulse < 0
            and correction > 0
            and abs(correction)
            < abs(impulse) * 0.70
        ):
            names.add(
                "BEARISH_IMPULSE_CORRECTION"
            )

    return names


# ============================================================
# 100+ PATTERN LIBRARY
# ============================================================

CLASSIC_PATTERN_LIBRARY = [
    "DOJI",
    "LONG_LEGGED_DOJI",
    "DRAGONFLY_DOJI",
    "GRAVESTONE_DOJI",
    "HAMMER",
    "HANGING_MAN",
    "SHOOTING_STAR",
    "INVERTED_HAMMER",
    "BULLISH_MARUBOZU",
    "BEARISH_MARUBOZU",
    "BULLISH_ENGULFING",
    "BEARISH_ENGULFING",
    "BULLISH_HARAMI",
    "BEARISH_HARAMI",
    "PIERCING_LINE",
    "DARK_CLOUD",
    "TWEEZER_TOP",
    "TWEEZER_BOTTOM",
    "MORNING_STAR",
    "EVENING_STAR",
    "THREE_WHITE_SOLDIERS",
    "THREE_BLACK_CROWS",
    "BULLISH_ABANDONED_BABY",
    "BEARISH_ABANDONED_BABY",
    "INSIDE_BAR",
    "OUTSIDE_BAR",
    "RANGE_EXPANSION",
    "RANGE_CONTRACTION",
    "THREE_BULLISH_SEQUENCE",
    "THREE_BEARISH_SEQUENCE",
    "BULLISH_STAIRCASE",
    "BEARISH_STAIRCASE",
    "LOWER_REJECTION",
    "UPPER_REJECTION",
    "BULLISH_GAP_DISLOCATION",
    "BEARISH_GAP_DISLOCATION",
    "DOUBLE_TOP",
    "DOUBLE_BOTTOM",
    "HIGHER_HIGH",
    "LOWER_HIGH",
    "HIGHER_LOW",
    "LOWER_LOW",
    "BREAKOUT_UP",
    "BREAKOUT_DOWN",
    "FALSE_BREAKOUT_UP",
    "FALSE_BREAKOUT_DOWN",
    "M_TOP",
    "W_BOTTOM",
    "HEAD_AND_SHOULDERS",
    "INVERSE_HEAD_AND_SHOULDERS",
    "UPTREND_CONTEXT",
    "DOWNTREND_CONTEXT",
    "RANGE_CONTEXT",
    "COMPRESSION_CONTEXT",
    "BREAKOUT_CONTEXT",
    "FALSE_BREAKOUT_CONTEXT",
    "HIGH_EFFICIENCY_IMPULSE",
    "CHOPPY_PRICE_ACTION",
    "REVERSAL_CONTEXT",
    "STRONG_DIRECTIONAL_SEQUENCE",
    "COUNTERTREND_BULLISH_REVERSAL",
    "COUNTERTREND_BEARISH_REVERSAL",
    "LOW_ZONE_REJECTION",
    "HIGH_ZONE_REJECTION",
    "BULLISH_IMPULSE_CORRECTION",
    "BEARISH_IMPULSE_CORRECTION",
]


# Add generated contextual combinations.
COMPOSITE_PATTERN_NAMES = [
    "PINBAR_AT_SUPPORT",
    "PINBAR_AT_RESISTANCE",
    "ENGULFING_AFTER_DOWNMOVE",
    "ENGULFING_AFTER_UPMOVE",
    "DOJI_AFTER_IMPULSE",
    "INSIDE_BAR_AFTER_IMPULSE",
    "OUTSIDE_BAR_AFTER_COMPRESSION",
    "FALSE_BREAKOUT_REJECTION",
    "BREAKOUT_AND_RETEST",
    "BREAKOUT_AND_CONTINUATION",
    "BREAKOUT_AND_FAILURE",
    "LOW_SWEEP_RECLAIM",
    "HIGH_SWEEP_REJECT",
    "RANGE_LOW_REJECTION",
    "RANGE_HIGH_REJECTION",
    "TREND_PULLBACK_BULLISH",
    "TREND_PULLBACK_BEARISH",
    "THREE_CANDLE_REVERSAL",
    "THREE_CANDLE_CONTINUATION",
    "FIVE_CANDLE_REVERSAL",
    "FIVE_CANDLE_CONTINUATION",
    "BULLISH_BODY_EXPANSION",
    "BEARISH_BODY_EXPANSION",
    "BULLISH_WICK_REJECTION",
    "BEARISH_WICK_REJECTION",
    "BULLISH_CLOSE_AT_HIGH",
    "BEARISH_CLOSE_AT_LOW",
    "BULLISH_CLOSE_RECLAIM",
    "BEARISH_CLOSE_RECLAIM",
    "RANGE_BREAK_AND_HOLD",
    "RANGE_BREAK_AND_FAIL",
    "HIGHER_LOW_REJECTION",
    "LOWER_HIGH_REJECTION",
    "DOUBLE_BOTTOM_RECLAIM",
    "DOUBLE_TOP_REJECTION",
    "HEAD_SHOULDER_BREAK",
    "INVERSE_HEAD_SHOULDER_BREAK",
    "M_TOP_BREAK",
    "W_BOTTOM_BREAK",
    "COMPRESSION_BREAK_UP",
    "COMPRESSION_BREAK_DOWN",
]


ALL_PATTERN_NAMES = sorted(
    set(
        CLASSIC_PATTERN_LIBRARY
        + COMPOSITE_PATTERN_NAMES
        + [
            f"SEQUENCE_{n}"
            for n in range(1, 81)
        ]
    )
)


# ============================================================
# COMPOSITE PATTERN DETECTOR
# ============================================================

def detect_composite_patterns(
    candles,
    i
):
    names = set()

    if i < 8:
        return names

    c = candles[i]

    ctx = price_action_context(
        candles,
        i
    )

    classic = detect_candlestick_patterns(
        candles,
        i
    )

    structure = detect_structure_patterns(
        candles,
        i
    )

    # --------------------------------------------------------
    # PINBAR SUPPORT / RESISTANCE
    # --------------------------------------------------------

    if (
        (
            "HAMMER" in classic
            or "LOWER_REJECTION" in classic
            or "INVERTED_HAMMER" in classic
        )
        and ctx["swing_position"] == "LOW"
    ):
        names.add(
            "PINBAR_AT_SUPPORT"
        )

    if (
        (
            "SHOOTING_STAR" in classic
            or "UPPER_REJECTION" in classic
        )
        and ctx["swing_position"] == "HIGH"
    ):
        names.add(
            "PINBAR_AT_RESISTANCE"
        )

    # --------------------------------------------------------
    # ENGULFING CONTEXT
    # --------------------------------------------------------

    if (
        "BULLISH_ENGULFING"
        in classic
        and ctx["trend"] == "DOWN"
    ):
        names.add(
            "ENGULFING_AFTER_DOWNMOVE"
        )

    if (
        "BEARISH_ENGULFING"
        in classic
        and ctx["trend"] == "UP"
    ):
        names.add(
            "ENGULFING_AFTER_UPMOVE"
        )

    # --------------------------------------------------------
    # DOJI AFTER IMPULSE
    # --------------------------------------------------------

    if (
        "DOJI" in classic
        and (
            ctx["efficiency"] >= 0.60
        )
    ):
        names.add(
            "DOJI_AFTER_IMPULSE"
        )

    # --------------------------------------------------------
    # INSIDE / OUTSIDE
    # --------------------------------------------------------

    if (
        "INSIDE_BAR" in classic
        and ctx["efficiency"] >= 0.60
    ):
        names.add(
            "INSIDE_BAR_AFTER_IMPULSE"
        )

    if (
        "OUTSIDE_BAR" in classic
        and ctx["range_compression"]
    ):
        names.add(
            "OUTSIDE_BAR_AFTER_COMPRESSION"
        )

    # --------------------------------------------------------
    # FALSE BREAKOUT
    # --------------------------------------------------------

    if ctx["false_breakout"]:
        names.add(
            "FALSE_BREAKOUT_REJECTION"
        )

    # --------------------------------------------------------
    # BREAKOUT
    # --------------------------------------------------------

    if (
        "BREAKOUT_UP" in structure
        or "BREAKOUT_DOWN" in structure
    ):
        if (
            ctx["directional_consistency"]
            >= 0.45
        ):
            names.add(
                "BREAKOUT_AND_CONTINUATION"
            )

    # --------------------------------------------------------
    # SWEEPS
    # --------------------------------------------------------

    if "FALSE_BREAKOUT_DOWN" in structure:
        names.add(
            "LOW_SWEEP_RECLAIM"
        )

    if "FALSE_BREAKOUT_UP" in structure:
        names.add(
            "HIGH_SWEEP_REJECT"
        )

    # --------------------------------------------------------
    # RANGE ZONES
    # --------------------------------------------------------

    if (
        ctx["trend"] == "RANGE"
        and ctx["swing_position"] == "LOW"
        and c.lower_ratio() >= 0.30
    ):
        names.add(
            "RANGE_LOW_REJECTION"
        )

    if (
        ctx["trend"] == "RANGE"
        and ctx["swing_position"] == "HIGH"
        and c.upper_ratio() >= 0.30
    ):
        names.add(
            "RANGE_HIGH_REJECTION"
        )

    # --------------------------------------------------------
    # PULLBACKS
    # --------------------------------------------------------

    if (
        "UPTREND_CONTEXT"
        in detect_context_patterns(
            candles,
            i
        )
        and (
            c.close < c.open
            or c.lower_ratio() >= 0.25
        )
    ):
        names.add(
            "TREND_PULLBACK_BULLISH"
        )

    if (
        "DOWNTREND_CONTEXT"
        in detect_context_patterns(
            candles,
            i
        )
        and (
            c.close > c.open
            or c.upper_ratio() >= 0.25
        )
    ):
        names.add(
            "TREND_PULLBACK_BEARISH"
        )

    # --------------------------------------------------------
    # REVERSAL / CONTINUATION
    # --------------------------------------------------------

    if (
        ctx["reversal_score"] >= 0.60
    ):
        names.add(
            "THREE_CANDLE_REVERSAL"
        )

    if (
        ctx["directional_consistency"]
        >= 0.65
        and ctx["efficiency"] >= 0.50
    ):
        names.add(
            "THREE_CANDLE_CONTINUATION"
        )

    # --------------------------------------------------------
    # BODY / WICK
    # --------------------------------------------------------

    if (
        c.body_ratio() >= 0.65
        and c.close
        > c.low + c.range() * 0.80
    ):
        names.add(
            "BULLISH_BODY_EXPANSION"
        )

    if (
        c.body_ratio() >= 0.65
        and c.close
        < c.low + c.range() * 0.20
    ):
        names.add(
            "BEARISH_BODY_EXPANSION"
        )

    if (
        c.lower_ratio() >= 0.40
    ):
        names.add(
            "BULLISH_WICK_REJECTION"
        )

    if (
        c.upper_ratio() >= 0.40
    ):
        names.add(
            "BEARISH_WICK_REJECTION"
        )

    if (
        c.close
        >= c.low + c.range() * 0.90
    ):
        names.add(
            "BULLISH_CLOSE_AT_HIGH"
        )

    if (
        c.close
        <= c.low + c.range() * 0.10
    ):
        names.add(
            "BEARISH_CLOSE_AT_LOW"
        )

    # --------------------------------------------------------
    # STRUCTURE BREAKS
    # --------------------------------------------------------

    if (
        "BREAKOUT_UP" in structure
        and ctx["trend"] != "DOWN"
    ):
        names.add(
            "RANGE_BREAK_AND_HOLD"
        )

    if (
        "FALSE_BREAKOUT_UP" in structure
        or "FALSE_BREAKOUT_DOWN" in structure
    ):
        names.add(
            "RANGE_BREAK_AND_FAIL"
        )

    if "HIGHER_LOW" in structure:
        names.add(
            "HIGHER_LOW_REJECTION"
        )

    if "LOWER_HIGH" in structure:
        names.add(
            "LOWER_HIGH_REJECTION"
        )

    if "DOUBLE_BOTTOM" in structure:
        names.add(
            "DOUBLE_BOTTOM_RECLAIM"
        )

    if "DOUBLE_TOP" in structure:
        names.add(
            "DOUBLE_TOP_REJECTION"
        )

    if "HEAD_AND_SHOULDERS" in structure:
        names.add(
            "HEAD_SHOULDER_BREAK"
        )

    if "INVERSE_HEAD_AND_SHOULDERS" in structure:
        names.add(
            "INVERSE_HEAD_SHOULDER_BREAK"
        )

    if "M_TOP" in structure:
        names.add(
            "M_TOP_BREAK"
        )

    if "W_BOTTOM" in structure:
        names.add(
            "W_BOTTOM_BREAK"
        )

    # --------------------------------------------------------
    # COMPRESSION BREAK
    # --------------------------------------------------------

    if ctx["range_compression"]:
        if (
            c.close
            > c.open
            and ctx["breakout_direction"]
            == "UP"
        ):
            names.add(
                "COMPRESSION_BREAK_UP"
            )

        if (
            c.close
            < c.open
            and ctx["breakout_direction"]
            == "DOWN"
        ):
            names.add(
                "COMPRESSION_BREAK_DOWN"
            )

    return names


# ============================================================
# FULL PATTERN SNAPSHOT
# ============================================================

def detect_all_patterns(
    candles,
    i
):
    names = set()

    names.update(
        detect_candlestick_patterns(
            candles,
            i
        )
    )

    names.update(
        detect_structure_patterns(
            candles,
            i
        )
    )

    names.update(
        detect_context_patterns(
            candles,
            i
        )
    )

    names.update(
        detect_composite_patterns(
            candles,
            i
        )
    )

    return names


# ============================================================
# AUTOMATIC CANDLE SEQUENCE
# ============================================================

def build_sequence_signature(
    candles,
    end_index,
    length
):
    if (
        end_index - length + 1
        < 0
    ):
        return ""

    parts = []

    for i in range(
        end_index - length + 1,
        end_index + 1
    ):
        parts.append(
            candle_code(candles[i])
        )

    return "|".join(parts)


# ============================================================
# DIRECTION FROM HISTORICAL OUTCOME
# ============================================================

def historical_direction(
    candles,
    index,
    forward=5
):
    if index + 1 >= len(candles):
        return None

    entry = candles[index].close

    end = min(
        len(candles) - 1,
        index + forward
    )

    future = candles[end].close

    if future > entry:
        return "BUY"

    if future < entry:
        return "SELL"

    return "NEUTRAL"


# ============================================================
# FORWARD SIMULATION
# ============================================================

def simulate_forward(
    candles,
    index,
    direction,
    atr
):
    if (
        index + 1 >= len(candles)
        or atr <= 0
    ):
        return None

    entry = candles[index].close

    risk = max(
        atr,
        entry * 0.0005
    )

    if direction == "BUY":
        sl = (
            entry
            - risk
            - atr * SL_BUF / 100
        )

        tp1 = (
            entry
            + risk * TP1_R
        )

    else:
        sl = (
            entry
            + risk
            + atr * SL_BUF / 100
        )

        tp1 = (
            entry
            - risk * TP1_R
        )

    max_r = -999.0
    min_r = 999.0

    tp1_hit = False
    stopped = False

    trail_active = False
    trail_level = None

    realized_r = 0.0

    max_forward = min(
        len(candles),
        index + 1 + FORWARD_CANDLES
    )

    for j in range(
        index + 1,
        max_forward
    ):
        c = candles[j]

        if direction == "BUY":
            high_r = (
                c.high - entry
            ) / risk

            low_r = (
                c.low - entry
            ) / risk

        else:
            high_r = (
                entry - c.low
            ) / risk

            low_r = (
                entry - c.high
            ) / risk

        max_r = max(
            max_r,
            high_r
        )

        min_r = min(
            min_r,
            low_r
        )

        # ----------------------------------------------------
        # SL BEFORE TP WHEN BOTH OCCUR
        # ----------------------------------------------------

        if direction == "BUY":
            hit_sl = c.low <= sl
            hit_tp1 = c.high >= tp1
        else:
            hit_sl = c.high >= sl
            hit_tp1 = c.low <= tp1

        if (
            hit_sl
            and hit_tp1
            and not tp1_hit
        ):
            stopped = True
            realized_r = -1.0
            break

        # ----------------------------------------------------
        # TP1
        # ----------------------------------------------------

        if (
            hit_tp1
            and not tp1_hit
        ):
            tp1_hit = True

            realized_r += (
                TP1_R
                * TP1_CLOSE_PCT
            )

            # Remaining position moves
            # toward break-even.
            if direction == "BUY":
                sl = entry
            else:
                sl = entry

        # ----------------------------------------------------
        # TRAILING
        # ----------------------------------------------------

        if tp1_hit:
            if max_r >= TRAIL_START_R:
                trail_active = True

            if trail_active:
                trail_r = (
                    math.floor(
                        max_r
                        / TRAIL_STEP_R
                    )
                    * TRAIL_STEP_R
                    - TRAIL_STEP_R
                )

                trail_r = clamp(
                    trail_r,
                    0.0,
                    MAX_TRAIL_R
                )

                if direction == "BUY":
                    trail_level = (
                        entry
                        + risk * trail_r
                    )
                else:
                    trail_level = (
                        entry
                        - risk * trail_r
                    )

                if direction == "BUY":
                    if (
                        trail_level
                        > sl
                    ):
                        sl = trail_level

                else:
                    if (
                        trail_level
                        < sl
                    ):
                        sl = trail_level

        # ----------------------------------------------------
        # STOP / TRAIL HIT
        # ----------------------------------------------------

        if direction == "BUY":
            if c.low <= sl:
                stop_r = (
                    sl - entry
                ) / risk

                if tp1_hit:
                    remaining = (
                        1.0
                        - TP1_CLOSE_PCT
                    )

                    realized_r += (
                        stop_r
                        * remaining
                    )

                else:
                    realized_r = (
                        stop_r
                    )

                stopped = True
                break

        else:
            if c.high >= sl:
                stop_r = (
                    entry - sl
                ) / risk

                if tp1_hit:
                    remaining = (
                        1.0
                        - TP1_CLOSE_PCT
                    )

                    realized_r += (
                        stop_r
                        * remaining
                    )

                else:
                    realized_r = (
                        stop_r
                    )

                stopped = True
                break

    # --------------------------------------------------------
    # NO STOP YET
    # --------------------------------------------------------

    if not stopped:
        last = candles[
            max_forward - 1
        ].close

        if direction == "BUY":
            close_r = (
                last - entry
            ) / risk
        else:
            close_r = (
                entry - last
            ) / risk

        if tp1_hit:
            remaining = (
                1.0
                - TP1_CLOSE_PCT
            )

            realized_r += (
                close_r
                * remaining
            )

        else:
            realized_r = close_r

    outcome = (
        "WIN"
        if realized_r > 0
        else
        "LOSS"
        if realized_r < 0
        else
        "FLAT"
    )

    return {
        "direction": direction,
        "entry": entry,
        "risk": risk,
        "sl": sl,
        "tp1": tp1,
        "max_r": max_r,
        "min_r": min_r,
        "realized_r": realized_r,
        "outcome": outcome,
        "tp1_hit": tp1_hit,
        "bars_forward": max_forward - index - 1
    }


# ============================================================
# HISTORICAL PATTERN EVENT
# ============================================================

def historical_event(
    candles,
    index,
    pattern_name,
    atr
):
    patterns = detect_all_patterns(
        candles,
        index
    )

    if pattern_name not in patterns:
        return None

    direction = None

    # Explicit structural direction.
    bullish_names = {
        "BULLISH_ENGULFING",
        "BULLISH_HARAMI",
        "PIERCING_LINE",
        "MORNING_STAR",
        "THREE_WHITE_SOLDIERS",
        "HAMMER",
        "INVERTED_HAMMER",
        "BULLISH_MARUBOZU",
        "DOUBLE_BOTTOM",
        "W_BOTTOM",
        "BULLISH_IMPULSE_CORRECTION",
        "BREAKOUT_UP",
        "FALSE_BREAKOUT_DOWN",
        "LOW_SWEEP_RECLAIM",
        "LOW_ZONE_REJECTION",
        "RANGE_LOW_REJECTION",
        "TREND_PULLBACK_BULLISH",
        "COMPRESSION_BREAK_UP",
        "BULLISH_BODY_EXPANSION",
        "BULLISH_CLOSE_AT_HIGH",
        "BULLISH_WICK_REJECTION",
        "HIGHER_LOW",
        "HIGHER_LOW_REJECTION",
        "DOUBLE_BOTTOM_RECLAIM",
        "INVERSE_HEAD_AND_SHOULDERS",
    }

    bearish_names = {
        "BEARISH_ENGULFING",
        "BEARISH_HARAMI",
        "DARK_CLOUD",
        "EVENING_STAR",
        "THREE_BLACK_CROWS",
        "SHOOTING_STAR",
        "HANGING_MAN",
        "BEARISH_MARUBOZU",
        "DOUBLE_TOP",
        "M_TOP",
        "BEARISH_IMPULSE_CORRECTION",
        "BREAKOUT_DOWN",
        "FALSE_BREAKOUT_UP",
        "HIGH_SWEEP_REJECT",
        "HIGH_ZONE_REJECTION",
        "RANGE_HIGH_REJECTION",
        "TREND_PULLBACK_BEARISH",
        "COMPRESSION_BREAK_DOWN",
        "BEARISH_BODY_EXPANSION",
        "BEARISH_CLOSE_AT_LOW",
        "BEARISH_WICK_REJECTION",
        "LOWER_HIGH",
        "LOWER_HIGH_REJECTION",
        "DOUBLE_TOP_REJECTION",
        "HEAD_AND_SHOULDERS",
    }

    if pattern_name in bullish_names:
        direction = "BUY"

    elif pattern_name in bearish_names:
        direction = "SELL"

    else:
        # For neutral patterns, use actual
        # historical future direction.
        direction = historical_direction(
            candles,
            index,
            forward=5
        )

    if direction not in (
        "BUY",
        "SELL"
    ):
        return None

    result = simulate_forward(
        candles,
        index,
        direction,
        atr
    )

    if not result:
        return None

    return result


# ============================================================
# STATS
# ============================================================

def stats_from_results(
    results
):
    if not results:
        return {
            "count": 0,
            "wins": 0,
            "losses": 0,
            "flats": 0,
            "win_rate": 0.0,
            "loss_rate": 0.0,
            "avg_r": 0.0,
            "median_r": 0.0,
            "profit_factor": 0.0,
            "max_r": 0.0,
            "min_r": 0.0,
            "avg_mfe": 0.0,
            "avg_mae": 0.0,
            "tp1_rate": 0.0,
            "r1_rate": 0.0,
            "r2_rate": 0.0,
            "r3_rate": 0.0,
            "r5_rate": 0.0
        }

    wins = [
        x for x in results
        if x["outcome"] == "WIN"
    ]

    losses = [
        x for x in results
        if x["outcome"] == "LOSS"
    ]

    flats = [
        x for x in results
        if x["outcome"] == "FLAT"
    ]

    positive = sum(
        max(
            0,
            x["realized_r"]
        )
        for x in results
    )

    negative = sum(
        abs(
            min(
                0,
                x["realized_r"]
            )
        )
        for x in results
    )

    pf = (
        positive / negative
        if negative > 0
        else (
            999.0
            if positive > 0
            else 0.0
        )
    )

    return {
        "count": len(results),
        "wins": len(wins),
        "losses": len(losses),
        "flats": len(flats),
        "win_rate": pct(
            len(wins),
            len(results)
        ),
        "loss_rate": pct(
            len(losses),
            len(results)
        ),
        "avg_r": mean_or_zero(
            [
                x["realized_r"]
                for x in results
            ]
        ),
        "median_r": median_or_zero(
            [
                x["realized_r"]
                for x in results
            ]
        ),
        "profit_factor": pf,
        "max_r": max(
            x["realized_r"]
            for x in results
        ),
        "min_r": min(
            x["realized_r"]
            for x in results
        ),
        "avg_mfe": mean_or_zero(
            [
                x["max_r"]
                for x in results
            ]
        ),
        "avg_mae": mean_or_zero(
            [
                x["min_r"]
                for x in results
            ]
        ),
        "tp1_rate": pct(
            sum(
                x["tp1_hit"]
                for x in results
            ),
            len(results)
        ),
        "r1_rate": pct(
            sum(
                x["max_r"] >= 1
                for x in results
            ),
            len(results)
        ),
        "r2_rate": pct(
            sum(
                x["max_r"] >= 2
                for x in results
            ),
            len(results)
        ),
        "r3_rate": pct(
            sum(
                x["max_r"] >= 3
                for x in results
            ),
            len(results)
        ),
        "r5_rate": pct(
            sum(
                x["max_r"] >= 5
                for x in results
            ),
            len(results)
        )
    }


# ============================================================
# SEGMENT VALIDATION
# ============================================================

def segment_stats(
    results,
    segments=5
):
    if not results:
        return []

    n = len(results)

    output = []

    for s in range(
        segments
    ):
        start = (
            s * n // segments
        )

        end = (
            (s + 1)
            * n
            // segments
        )

        chunk = results[
            start:end
        ]

        if not chunk:
            continue

        st = stats_from_results(
            chunk
        )

        output.append({
            "segment": s + 1,
            **st
        })

    return output


# ============================================================
# BOOTSTRAP
# ============================================================

def bootstrap_wr(
    results,
    passes=50
):
    if len(results) < 2:
        return {
            "mean": 0.0,
            "lower": 0.0,
            "upper": 0.0,
            "pass_rate": 0.0
        }

    values = np.array([
        1
        if x["outcome"] == "WIN"
        else 0
        for x in results
    ])

    rates = []

    n = len(values)

    for _ in range(
        max(1, passes)
    ):
        sample = np.random.choice(
            values,
            size=n,
            replace=True
        )

        rates.append(
            float(
                np.mean(sample) * 100
            )
        )

    lower = float(
        np.percentile(
            rates,
            BOOTSTRAP_LOWER_Q
        )
    )

    upper = float(
        np.percentile(
            rates,
            100 - BOOTSTRAP_LOWER_Q
        )
    )

    return {
        "mean": mean_or_zero(rates),
        "lower": lower,
        "upper": upper,
        "pass_rate": pct(
            sum(
                r >= MIN_WIN_RATE
                for r in rates
            ),
            len(rates)
        )
    }


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_confidence(
    full,
    holdout,
    recent,
    segments,
    bootstrap
):
    score = 0.0

    # Historical WR
    score += clamp(
        (
            full["win_rate"]
            - 50
        ) * 0.55,
        0,
        20
    )

    # Holdout
    score += clamp(
        (
            holdout["win_rate"]
            - 50
        ) * 0.30,
        0,
        10
    )

    # Recent
    score += clamp(
        (
            recent["win_rate"]
            - 50
        ) * 0.20,
        0,
        7
    )

    # PF
    score += clamp(
        (
            full["profit_factor"]
            - 1
        ) * 12,
        0,
        12
    )

    # Avg R
    score += clamp(
        full["avg_r"] * 20,
        0,
        10
    )

    # Bootstrap
    score += clamp(
        (
            bootstrap["lower"]
            - 50
        ) * 0.20,
        0,
        8
    )

    # Segment stability
    if segments:
        good = sum(
            x["win_rate"]
            >= MIN_SEGMENT_WR
            for x in segments
        )

        score += (
            good
            / len(segments)
            * 8
        )

    return clamp(
        score,
        0,
        100
    )


# ============================================================
# DEEP VALIDATION
# ============================================================

def deep_validation(
    results
):
    if len(results) < 2:
        return {
            "passes": 0,
            "total": 0,
            "pass_rate": 0.0
        }

    passes = 0

    n = len(results)

    for _ in range(
        DEEP_VALIDATION_PASSES
    ):
        sample = random.choices(
            results,
            k=n
        )

        st = stats_from_results(
            sample
        )

        if (
            st["win_rate"]
            >= MIN_WIN_RATE
            and st["avg_r"]
            >= MIN_AVG_R
            and st["profit_factor"]
            >= MIN_PROFIT_FACTOR
        ):
            passes += 1

    return {
        "passes": passes,
        "total": DEEP_VALIDATION_PASSES,
        "pass_rate": pct(
            passes,
            DEEP_VALIDATION_PASSES
        )
    }


# ============================================================
# HOLDOUT SPLIT
# ============================================================

def split_train_holdout(
    results
):
    if not results:
        return [], []

    n = len(results)

    holdout_n = max(
        1,
        int(
            n * HOLDOUT_FRACTION
        )
    )

    if holdout_n >= n:
        holdout_n = max(
            1,
            n - 1
        )

    train = results[
        :n - holdout_n
    ]

    holdout = results[
        n - holdout_n:
    ]

    return train, holdout


# ============================================================
# INDEPENDENT OCCURRENCES
# ============================================================

def independent_indices(
    indices,
    min_gap
):
    if not indices:
        return []

    output = [
        indices[0]
    ]

    last = indices[0]

    for idx in indices[1:]:
        if (
            idx - last
            >= min_gap
        ):
            output.append(idx)
            last = idx

    return output


# ============================================================
# ANALYZE ONE PATTERN
# ============================================================

def analyze_candidate(
    candles,
    pattern_name,
    indices
):
    if len(indices) < MIN_OCCURRENCES:
        return None

    indices = sorted(indices)

    independent = independent_indices(
        indices,
        max(
            3,
            CONTEXT_CANDLES // 2
        )
    )

    if (
        len(independent)
        < MIN_INDEPENDENT_OCCURRENCES
    ):
        return None

    results = []

    for idx in independent:
        if (
            idx < ATR_PERIOD
            or idx + 1 >= len(candles)
        ):
            continue

        atr = compute_atr(
            candles[
                max(
                    0,
                    idx - ATR_PERIOD
                ):idx + 1
            ],
            ATR_PERIOD
        )

        event = historical_event(
            candles,
            idx,
            pattern_name,
            atr
        )

        if event:
            results.append(
                event
            )

    if len(results) < MIN_OCCURRENCES:
        return None

    train, holdout = (
        split_train_holdout(
            results
        )
    )

    full = stats_from_results(
        results
    )

    train_stats = stats_from_results(
        train
    )

    holdout_stats = stats_from_results(
        holdout
    )

    recent = stats_from_results(
        results[
            -RECENT_OCCURRENCES:
        ]
    )

    segments = segment_stats(
        results,
        VALIDATION_SEGMENTS
    )

    bootstrap = bootstrap_wr(
        train,
        DEEP_VALIDATION_PASSES
    )

    deep = deep_validation(
        train
    )

    confidence = calculate_confidence(
        full,
        holdout_stats,
        recent,
        segments,
        bootstrap
    )

    return {
        "pattern": pattern_name,
        "occurrences": len(results),
        "independent_occurrences":
            len(independent),
        "full": full,
        "train": train_stats,
        "holdout": holdout_stats,
        "recent": recent,
        "segments": segments,
        "bootstrap": bootstrap,
        "deep_validation": deep,
        "confidence": confidence,
        "indices": independent[
            -100:
        ]
    }


# ============================================================
# SEQUENCE CANDIDATES
# ============================================================

def sequence_candidates(
    candles,
    length
):
    counter = Counter()
    positions = defaultdict(list)

    start = max(
        ATR_PERIOD,
        length - 1
    )

    for i in range(
        start,
        len(candles)
    ):
        sig = build_sequence_signature(
            candles,
            i,
            length
        )

        if not sig:
            continue

        counter[sig] += 1
        positions[sig].append(i)

    return counter, positions


# ============================================================
# PATTERN ANALYSIS FOR TIMEFRAME
# ============================================================

def analyze_timeframe(
    candles,
    symbol,
    timeframe
):
    print(
        f"[ANALYZE] "
        f"{symbol} {timeframe}"
    )

    candidates = []

    # --------------------------------------------------------
    # NAMED PATTERNS
    # --------------------------------------------------------

    named_positions = defaultdict(
        list
    )

    for i in range(
        ATR_PERIOD,
        len(candles)
    ):
        names = detect_all_patterns(
            candles,
            i
        )

        for name in names:
            named_positions[
                name
            ].append(i)

    print(
        f"[PATTERNS] "
        f"{symbol} {timeframe} "
        f"named={len(named_positions)}"
    )

    for name, indices in (
        named_positions.items()
    ):
        if len(indices) < MIN_OCCURRENCES:
            continue

        try:
            result = analyze_candidate(
                candles,
                name,
                indices
            )

            if result:
                candidates.append(
                    result
                )

        except Exception as exc:
            print(
                f"[PATTERN ERROR] "
                f"{name}: {exc}"
            )

    # --------------------------------------------------------
    # AUTOMATIC SEQUENCES
    # --------------------------------------------------------

    for length in SEQ_LENGTHS:
        counter, positions = (
            sequence_candidates(
                candles,
                length
            )
        )

        for sig, count in (
            counter.items()
        ):
            if count < MIN_OCCURRENCES:
                continue

            pattern_name = (
                f"SEQUENCE_{length}_"
                f"{sig}"
            )

            try:
                result = analyze_candidate(
                    candles,
                    pattern_name,
                    positions[sig]
                )

                if result:
                    candidates.append(
                        result
                    )

            except Exception as exc:
                print(
                    f"[SEQUENCE ERROR] "
                    f"{pattern_name}: {exc}"
                )

    # --------------------------------------------------------
    # RANK FOR INFORMATION ONLY
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: (
            x["confidence"],
            x["full"]["profit_factor"],
            x["full"]["win_rate"],
            x["occurrences"]
        ),
        reverse=True
    )

    selected = candidates[
        :TOP_PATTERNS_PER_TF
    ]

    print(
        f"[ANALYZE DONE] "
        f"{symbol} {timeframe} "
        f"valid={len(candidates)} "
        f"selected={len(selected)}"
    )

    return selected, candidates


# ============================================================
# CURRENT MARKET SNAPSHOT
# ============================================================

def current_market_snapshot(
    candles
):
    i = len(candles) - 1

    current = candles[i]

    patterns = detect_all_patterns(
        candles,
        i
    )

    classic = detect_candlestick_patterns(
        candles,
        i
    )

    structure = detect_structure_patterns(
        candles,
        i
    )

    context = detect_context_patterns(
        candles,
        i
    )

    pa = price_action_context(
        candles,
        i
    )

    sequence = {}

    for length in SEQ_LENGTHS:
        sequence[
            str(length)
        ] = build_sequence_signature(
            candles,
            i,
            length
        )

    return {
        "index": i,
        "time": utc_string(
            current.close_time
        ),
        "price": current.close,
        "open": current.open,
        "high": current.high,
        "low": current.low,
        "volume": current.volume,
        "candle_code":
            candle_code(current),
        "patterns": sorted(
            patterns
        ),
        "classic": sorted(
            classic
        ),
        "structure": sorted(
            structure
        ),
        "context_patterns": sorted(
            context
        ),
        "sequence": sequence,
        "price_action": pa
    }


# ============================================================
# MATCH CURRENT STATE AGAINST HISTORY
# ============================================================

def historical_matches(
    candles,
    current_index,
    current_patterns,
    limit=20
):
    matches = []

    if not current_patterns:
        return matches

    current_codes = {}

    for length in SEQ_LENGTHS:
        current_codes[
            length
        ] = build_sequence_signature(
            candles,
            current_index,
            length
        )

    start = max(
        ATR_PERIOD,
        current_index - 100000
    )

    for i in range(
        start,
        current_index - 2
    ):
        hist_patterns = (
            detect_all_patterns(
                candles,
                i
            )
        )

        common = (
            current_patterns
            & hist_patterns
        )

        if not common:
            continue

        same_sequences = 0

        for length in SEQ_LENGTHS:
            current_sig = (
                current_codes[length]
            )

            hist_sig = (
                build_sequence_signature(
                    candles,
                    i,
                    length
                )
            )

            if (
                current_sig
                and current_sig == hist_sig
            ):
                same_sequences += 1

        similarity = (
            len(common) * 10
            + same_sequences * 20
        )

        matches.append({
            "index": i,
            "time": utc_string(
                candles[i].close_time
            ),
            "common_patterns":
                sorted(common),
            "same_sequences":
                same_sequences,
            "similarity": similarity
        })

    matches.sort(
        key=lambda x: (
            x["similarity"],
            x["index"]
        ),
        reverse=True
    )

    return matches[:limit]


# ============================================================
# CURRENT SIGNAL
# ============================================================

def build_live_signal(
    candles,
    pattern,
    symbol,
    timeframe
):
    current_index = (
        len(candles) - 1
    )

    current = candles[
        current_index
    ]

    current_patterns = (
        detect_all_patterns(
            candles,
            current_index
        )
    )

    # If candidate is a sequence.
    matched = False

    if pattern["pattern"].startswith(
        "SEQUENCE_"
    ):
        parts = pattern[
            "pattern"
        ].split("_", 2)

        if len(parts) == 3:
            length = int(parts[1])

            signature = parts[2]

            current_signature = (
                build_sequence_signature(
                    candles,
                    current_index,
                    length
                )
            )

            matched = (
                current_signature
                == signature
            )

    else:
        matched = (
            pattern["pattern"]
            in current_patterns
        )

    if not matched:
        return None

    full = pattern["full"]

    if (
        full["win_rate"]
        >= MIN_WIN_RATE
    ):
        if (
            full["wins"]
            >= full["losses"]
        ):
            direction = "BUY"

        else:
            direction = "SELL"

    else:
        return None

    if (
        direction == "BUY"
        and (
            "BEARISH_ENGULFING"
            in current_patterns
            or
            "BEARISH_MARUBOZU"
            in current_patterns
        )
    ):
        return None

    if (
        direction == "SELL"
        and (
            "BULLISH_ENGULFING"
            in current_patterns
            or
            "BULLISH_MARUBOZU"
            in current_patterns
        )
    ):
        return None

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "pattern": pattern,
        "direction": direction,
        "current_patterns":
            sorted(current_patterns),
        "snapshot":
            current_market_snapshot(
                candles
            )
    }


# ============================================================
# CHART
# ============================================================

def make_signal_chart(
    candles,
    signal,
    output_path
):
    pattern = signal["pattern"]

    current_index = (
        len(candles) - 1
    )

    window = min(
        100,
        len(candles)
    )

    start = max(
        0,
        current_index - window + 1
    )

    data = candles[
        start:current_index + 1
    ]

    fig, ax = plt.subplots(
        figsize=(15, 8)
    )

    for x, c in enumerate(data):
        up = c.close >= c.open

        lower = min(
            c.open,
            c.close
        )

        height = abs(
            c.close - c.open
        )

        if height == 0:
            height = c.range() * 0.01

        ax.vlines(
            x,
            c.low,
            c.high,
            linewidth=1
        )

        rect = Rectangle(
            (
                x - 0.32,
                lower
            ),
            0.64,
            height,
            fill=True,
            alpha=0.75
        )

        ax.add_patch(rect)

    signal_local = (
        current_index - start
    )

    pattern_length = 2

    if pattern[
        "pattern"
    ].startswith(
        "SEQUENCE_"
    ):
        try:
            pattern_length = int(
                pattern["pattern"]
                .split("_")[1]
            )
        except Exception:
            pattern_length = 2

    pattern_start = max(
        0,
        signal_local
        - pattern_length
        + 1
    )

    ax.axvspan(
        pattern_start - 0.45,
        signal_local + 0.45,
        alpha=0.18
    )

    ax.axvline(
        signal_local,
        linestyle="--",
        linewidth=1.5
    )

    direction = signal[
        "direction"
    ]

    title = (
        f"{signal['symbol']} "
        f"{signal['timeframe']} | "
        f"{direction} | "
        f"{pattern['pattern']} | "
        f"WR={pattern['full']['win_rate']:.1f}% "
        f"PF={pattern['full']['profit_factor']:.2f}"
    )

    ax.set_title(
        title,
        fontsize=12
    )

    ax.set_xlabel(
        "Historical candles"
    )

    ax.set_ylabel(
        "Price"
    )

    ax.grid(
        alpha=0.20
    )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=160
    )

    plt.close(fig)


# ============================================================
# TELEGRAM
# ============================================================

class Telegram:
    def __init__(
        self,
        token,
        chat_id
    ):
        self.token = token
        self.chat_id = chat_id

    @property
    def enabled(self):
        return bool(
            self.token
            and self.chat_id
        )

    async def send_text(
        self,
        text
    ):
        if not self.enabled:
            return False

        url = (
            "https://api.telegram.org/"
            f"bot{self.token}/sendMessage"
        )

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }

        try:
            timeout = aiohttp.ClientTimeout(
                total=30
            )

            async with aiohttp.ClientSession(
                timeout=timeout
            ) as session:

                async with session.post(
                    url,
                    json=payload
                ) as response:

                    return (
                        response.status == 200
                    )

        except Exception as exc:
            print(
                f"[TELEGRAM TEXT ERROR] "
                f"{exc}"
            )

            return False

    async def send_photo(
        self,
        path,
        caption
    ):
        if not self.enabled:
            return False

        url = (
            "https://api.telegram.org/"
            f"bot{self.token}/sendPhoto"
        )

        try:
            timeout = aiohttp.ClientTimeout(
                total=60
            )

            form = aiohttp.FormData()

            form.add_field(
                "chat_id",
                self.chat_id
            )

            form.add_field(
                "caption",
                caption[:1024]
            )

            form.add_field(
                "parse_mode",
                "HTML"
            )

            with open(
                path,
                "rb"
            ) as f:

                form.add_field(
                    "photo",
                    f,
                    filename="signal.png",
                    content_type="image/png"
                )

                async with aiohttp.ClientSession(
                    timeout=timeout
                ) as session:

                    async with session.post(
                        url,
                        data=form
                    ) as response:

                        return (
                            response.status == 200
                        )

        except Exception as exc:
            print(
                f"[TELEGRAM PHOTO ERROR] "
                f"{exc}"
            )

            return False


# ============================================================
# TELEGRAM REPORT
# ============================================================

def format_pattern_report(
    signal
):
    p = signal["pattern"]

    full = p["full"]
    train = p["train"]
    holdout = p["holdout"]
    recent = p["recent"]
    boot = p["bootstrap"]
    deep = p["deep_validation"]

    snapshot = signal[
        "snapshot"
    ]

    pa = snapshot[
        "price_action"
    ]

    lines = []

    lines.append(
        "🚨 <b>PRICE ACTION DEEP SIGNAL</b>"
    )

    lines.append(
        f"💠 <b>{signal['symbol']}</b> | "
        f"<b>{signal['timeframe']}</b>"
    )

    lines.append(
        f"📌 Direction: "
        f"<b>{signal['direction']}</b>"
    )

    lines.append(
        f"💰 Price: "
        f"<b>{snapshot['price']:.8f}</b>"
    )

    lines.append("")

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        f"🧩 <b>MAIN PATTERN</b>\n"
        f"{p['pattern']}"
    )

    lines.append(
        f"🔢 Occurrences: "
        f"<b>{p['occurrences']}</b>"
    )

    lines.append(
        f"🔗 Independent: "
        f"<b>{p['independent_occurrences']}</b>"
    )

    lines.append("")

    lines.append(
        "📊 <b>FULL HISTORY</b>"
    )

    lines.append(
        f"WIN: <b>{full['wins']}</b> | "
        f"LOSS: <b>{full['losses']}</b> | "
        f"FLAT: <b>{full['flats']}</b>"
    )

    lines.append(
        f"WR: <b>{full['win_rate']:.2f}%</b>"
    )

    lines.append(
        f"Loss rate: "
        f"<b>{full['loss_rate']:.2f}%</b>"
    )

    lines.append(
        f"Average R: "
        f"<b>{full['avg_r']:.3f}</b>"
    )

    lines.append(
        f"Median R: "
        f"<b>{full['median_r']:.3f}</b>"
    )

    lines.append(
        f"Profit Factor: "
        f"<b>{full['profit_factor']:.2f}</b>"
    )

    lines.append("")

    lines.append(
        "🎯 <b>FORWARD HISTORY</b>"
    )

    lines.append(
        f"MFE / max R: "
        f"<b>{full['avg_mfe']:.2f}R</b>"
    )

    lines.append(
        f"MAE / min R: "
        f"<b>{full['avg_mae']:.2f}R</b>"
    )

    lines.append(
        f"TP1 reached: "
        f"<b>{full['tp1_rate']:.1f}%</b>"
    )

    lines.append(
        f"1R reached: "
        f"<b>{full['r1_rate']:.1f}%</b>"
    )

    lines.append(
        f"2R reached: "
        f"<b>{full['r2_rate']:.1f}%</b>"
    )

    lines.append(
        f"3R reached: "
        f"<b>{full['r3_rate']:.1f}%</b>"
    )

    lines.append(
        f"5R reached: "
        f"<b>{full['r5_rate']:.1f}%</b>"
    )

    lines.append("")

    lines.append(
        "🧪 <b>TRAIN</b>"
    )

    lines.append(
        f"N={train['count']} | "
        f"WR={train['win_rate']:.1f}% | "
        f"PF={train['profit_factor']:.2f}"
    )

    lines.append(
        "🧪 <b>HOLDOUT</b>"
    )

    lines.append(
        f"N={holdout['count']} | "
        f"WR={holdout['win_rate']:.1f}% | "
        f"PF={holdout['profit_factor']:.2f}"
    )

    lines.append(
        "🧪 <b>RECENT</b>"
    )

    lines.append(
        f"N={recent['count']} | "
        f"WR={recent['win_rate']:.1f}%"
    )

    lines.append("")

    lines.append(
        "🔬 <b>DEEP VALIDATION</b>"
    )

    lines.append(
        f"Pass rate: "
        f"<b>{deep['pass_rate']:.1f}%</b>"
    )

    lines.append(
        f"Bootstrap mean: "
        f"<b>{boot['mean']:.1f}%</b>"
    )

    lines.append(
        f"Bootstrap lower: "
        f"<b>{boot['lower']:.1f}%</b>"
    )

    lines.append("")

    lines.append(
        f"🧠 <b>CONFIDENCE: "
        f"{p['confidence']:.1f}%</b>"
    )

    lines.append("")

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "📍 <b>CURRENT MARKET</b>"
    )

    lines.append(
        f"Candle: "
        f"<b>{snapshot['candle_code']}</b>"
    )

    lines.append(
        f"Trend: "
        f"<b>{pa['trend']}</b>"
    )

    lines.append(
        f"Change: "
        f"<b>{pa['change_pct']:.3f}%</b>"
    )

    lines.append(
        f"Efficiency: "
        f"<b>{pa['efficiency']:.2f}</b>"
    )

    lines.append(
        f"Reversal score: "
        f"<b>{pa['reversal_score']:.2f}</b>"
    )

    lines.append(
        f"Swing zone: "
        f"<b>{pa['swing_position']}</b>"
    )

    lines.append(
        f"Breakout: "
        f"<b>{pa['breakout_direction']}</b>"
    )

    lines.append(
        f"False breakout: "
        f"<b>{pa['false_breakout_direction']}</b>"
    )

    lines.append("")

    lines.append(
        "🧩 <b>CURRENT PATTERNS</b>"
    )

    current = (
        signal["current_patterns"]
    )

    if current:
        # Telegram message limit protection.
        text = ", ".join(
            current[:35]
        )

        lines.append(
            text
        )

    else:
        lines.append(
            "No named pattern"
        )

    lines.append("")

    lines.append(
        "⚠️ Historical statistics do not "
        "guarantee the next move."
    )

    return "\n".join(lines)


# ============================================================
# MAIN BOT
# ============================================================

class DeepPatternBot:

    def __init__(
        self,
        client
    ):
        self.client = client

        self.telegram = Telegram(
            TELEGRAM_BOT_TOKEN,
            TELEGRAM_CHAT_ID
        )

        self.histories = {}

        self.buffers = {}

        self.patterns = {}

        self.last_signal = {}

        self.last_scan = {}

        self.running = True

    # --------------------------------------------------------
    # KEY
    # --------------------------------------------------------

    def key(
        self,
        symbol,
        timeframe
    ):
        return (
            f"{symbol}|{timeframe}"
        )

    # --------------------------------------------------------
    # LOAD HISTORY
    # --------------------------------------------------------

    async def load_all_history(
        self
    ):
        for symbol in SYMBOLS:

            for timeframe in TIMEFRAMES:

                days = days_for_tf(
                    timeframe
                )

                try:
                    candles = (
                        await fetch_history(
                            self.client,
                            symbol,
                            timeframe,
                            days
                        )
                    )

                    self.histories[
                        self.key(
                            symbol,
                            timeframe
                        )
                    ] = candles

                    self.buffers[
                        self.key(
                            symbol,
                            timeframe
                        )
                    ] = deque(
                        candles[
                            -CANDLE_BUFFER:
                        ],
                        maxlen=CANDLE_BUFFER
                    )

                except Exception as exc:
                    print(
                        f"[HISTORY ERROR] "
                        f"{symbol} "
                        f"{timeframe}: "
                        f"{exc}"
                    )

    # --------------------------------------------------------
    # REFRESH LIVE BUFFER
    # --------------------------------------------------------

    async def refresh_live_buffer(
        self,
        symbol,
        timeframe
    ):
        key = self.key(
            symbol,
            timeframe
        )

        try:
            candles = (
                await fetch_recent_history(
                    self.client,
                    symbol,
                    timeframe,
                    CANDLE_BUFFER
                    + ATR_PERIOD
                    + 30
                )
            )

            if candles:
                self.buffers[key] = deque(
                    candles[
                        -CANDLE_BUFFER:
                    ],
                    maxlen=CANDLE_BUFFER
                )

                print(
                    f"[BUFFER REFRESH] "
                    f"{key} "
                    f"{len(candles)}"
                )

        except Exception as exc:
            print(
                f"[BUFFER REFRESH ERROR] "
                f"{key}: {exc}"
            )

    # --------------------------------------------------------
    # FULL ANALYSIS
    # --------------------------------------------------------

    async def full_scan(
        self
    ):
        for symbol in SYMBOLS:

            for timeframe in TIMEFRAMES:

                key = self.key(
                    symbol,
                    timeframe
                )

                candles = self.histories.get(
                    key,
                    []
                )

                if len(candles) < (
                    ATR_PERIOD
                    + MIN_OCCURRENCES
                    + 20
                ):
                    print(
                        f"[SCAN SKIP] "
                        f"{key} "
                        f"not enough candles"
                    )

                    continue

                try:
                    selected, all_candidates = (
                        analyze_timeframe(
                            candles,
                            symbol,
                            timeframe
                        )
                    )

                    self.patterns[key] = (
                        all_candidates
                    )

                    self.last_scan[key] = (
                        now_ms()
                    )

                    print(
                        f"[SCAN] {key} "
                        f"patterns="
                        f"{len(all_candidates)}"
                    )

                    if SAVE_RESULTS:
                        self.save_results()

                except Exception as exc:
                    print(
                        f"[SCAN ERROR] "
                        f"{key}: {exc}"
                    )

                    traceback.print_exc()

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    def save_results(
        self
    ):
        if not SAVE_RESULTS:
            return

        output = {}

        for key, patterns in (
            self.patterns.items()
        ):
            output[key] = patterns

        tmp = (
            RESULT_FILE
            + ".tmp"
        )

        try:
            with open(
                tmp,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    output,
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            os.replace(
                tmp,
                RESULT_FILE
            )

            print(
                f"[SAVE] {RESULT_FILE}"
            )

        except Exception as exc:
            print(
                f"[SAVE ERROR] {exc}"
            )

    # --------------------------------------------------------
    # LIVE GATE
    # --------------------------------------------------------

    async def check_live(
        self,
        symbol,
        timeframe
    ):
        key = self.key(
            symbol,
            timeframe
        )

        if key not in self.patterns:
            return

        candles = list(
            self.buffers.get(
                key,
                []
            )
        )

        if len(candles) < (
            ATR_PERIOD + 20
        ):
            return

        current = candles[-1]

        # ----------------------------------------------------
        # Only closed candle is accepted.
        # ----------------------------------------------------

        if current.close_time > now_ms():
            return

        candidates = self.patterns[
            key
        ]

        if not candidates:
            return

        for pattern in candidates:

            if (
                pattern["confidence"]
                < MIN_CONFIDENCE
            ):
                continue

            signal = build_live_signal(
                candles,
                pattern,
                symbol,
                timeframe
            )

            if not signal:
                continue

            # ------------------------------------------------
            # Cooldown
            # ------------------------------------------------

            last = self.last_signal.get(
                key,
                0
            )

            if (
                now_ms() - last
                < SIGNAL_COOLDOWN_SECONDS
                * 1000
            ):
                continue

            self.last_signal[key] = (
                now_ms()
            )

            await self.fire_signal(
                candles,
                signal
            )

            # One signal per closed candle.
            break

    # --------------------------------------------------------
    # SIGNAL
    # --------------------------------------------------------

    async def fire_signal(
        self,
        candles,
        signal
    ):
        filename = (
            "signal_"
            + signal["symbol"]
            + "_"
            + signal["timeframe"]
            + "_"
            + str(now_ms())
            + ".png"
        )

        make_signal_chart(
            candles,
            signal,
            filename
        )

        report = (
            format_pattern_report(
                signal
            )
        )

        print(
            "\n"
            + "=" * 80
            + "\n"
            + report.replace(
                "<b>",
                ""
            ).replace(
                "</b>",
                ""
            )
            + "\n"
            + "=" * 80
        )

        await self.telegram.send_photo(
            filename,
            report
        )

        try:
            os.remove(filename)
        except Exception:
            pass

    # --------------------------------------------------------
    # HANDLE CLOSED CANDLE
    # --------------------------------------------------------

    async def on_closed_candle(
        self,
        symbol,
        timeframe,
        k
    ):
        key = self.key(
            symbol,
            timeframe
        )

        candle = Candle(
            open_time=int(k["t"]),
            close_time=int(k["T"]),
            open=safe_float(k["o"]),
            high=safe_float(k["h"]),
            low=safe_float(k["l"]),
            close=safe_float(k["c"]),
            volume=safe_float(k["v"])
        )

        if candle.close_time > now_ms():
            return

        if key not in self.buffers:
            self.buffers[key] = deque(
                maxlen=CANDLE_BUFFER
            )

        buf = self.buffers[key]

        # Replace same candle if it already exists.
        if buf:
            if (
                buf[-1].open_time
                == candle.open_time
            ):
                buf[-1] = candle

            elif (
                candle.open_time
                > buf[-1].open_time
            ):
                buf.append(candle)

        else:
            buf.append(candle)

        await self.check_live(
            symbol,
            timeframe
        )

    # --------------------------------------------------------
    # STREAM
    # --------------------------------------------------------

    async def listen_stream(
        self
    ):
        streams = []

        for symbol in SYMBOLS:
            for timeframe in TIMEFRAMES:
                streams.append(
                    f"{symbol.lower()}"
                    f"@kline_{timeframe}"
                )

        while self.running:

            started = time.time()

            try:
                print(
                    "[WS] Connecting..."
                )

                # Refresh every buffer before
                # opening a new websocket.
                for symbol in SYMBOLS:
                    for timeframe in TIMEFRAMES:
                        await self.refresh_live_buffer(
                            symbol,
                            timeframe
                        )

                bsm = BinanceSocketManager(
                    self.client
                )

                async with bsm.multiplex_socket(
                    streams
                ) as stream:

                    print(
                        "[WS] Connected"
                    )

                    while (
                        self.running
                        and
                        time.time()
                        - started
                        < WS_RECONNECT_SECONDS
                    ):
                        msg = await stream.recv()

                        if not msg:
                            continue

                        data = msg.get(
                            "data",
                            msg
                        )

                        if (
                            data.get("e")
                            != "kline"
                        ):
                            continue

                        k = data.get(
                            "k",
                            {}
                        )

                        # Binance kline field x:
                        # True = candle closed.
                        if not k.get(
                            "x",
                            False
                        ):
                            continue

                        symbol = (
                            data.get(
                                "s"
                            )
                            or k.get(
                                "s"
                            )
                        )

                        timeframe = k.get(
                            "i"
                        )

                        if (
                            not symbol
                            or not timeframe
                        ):
                            continue

                        await self.on_closed_candle(
                            symbol,
                            timeframe,
                            k
                        )

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                print(
                    f"[WS ERROR] {exc}"
                )

                traceback.print_exc()

            print(
                "[WS] Reconnecting..."
            )

            # Rebuild recent buffers after
            # every disconnection.
            for symbol in SYMBOLS:
                for timeframe in TIMEFRAMES:
                    await self.refresh_live_buffer(
                        symbol,
                        timeframe
                    )

            await asyncio.sleep(3)

    # --------------------------------------------------------
    # RESCAN LOOP
    # --------------------------------------------------------

    async def rescan_loop(
        self
    ):
        while self.running:

            try:
                await asyncio.sleep(
                    RESCAN_SECONDS
                )

                print(
                    "[RESCAN] Starting..."
                )

                # Refresh full historical data.
                await self.load_all_history()

                await self.full_scan()

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                print(
                    f"[RESCAN ERROR] {exc}"
                )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    async def run(
        self
    ):
        print(
            "=================================================="
        )

        print(
            "PRICE ACTION DEEP PATTERN ENGINE v10.0"
        )

        print(
            "=================================================="
        )

        print(
            f"Symbols: {SYMBOLS}"
        )

        print(
            f"Timeframes: {TIMEFRAMES}"
        )

        print(
            f"Sequence lengths: {SEQ_LENGTHS}"
        )

        print(
            f"Pattern library: "
            f"{len(ALL_PATTERN_NAMES)}+"
        )

        print(
            "Indicators: NONE"
        )

        print(
            "ATR: risk normalization only"
        )

        print(
            "Telegram: "
            + (
                "ENABLED"
                if self.telegram.enabled
                else "DISABLED"
            )
        )

        print(
            "=================================================="
        )

        await self.load_all_history()

        await self.full_scan()

        tasks = [
            asyncio.create_task(
                self.listen_stream()
            ),
            asyncio.create_task(
                self.rescan_loop()
            )
        ]

        try:
            await asyncio.gather(
                *tasks
            )

        finally:
            self.running = False

            for task in tasks:
                task.cancel()

            await asyncio.gather(
                *tasks,
                return_exceptions=True
            )


# ============================================================
# MAIN
# ============================================================

async def main():
    client = None

    try:
        client = (
            await AsyncClient.create(
                api_key=BINANCE_API_KEY or None,
                api_secret=BINANCE_API_SECRET or None
            )
        )

        bot = DeepPatternBot(
            client
        )

        await bot.run()

    except KeyboardInterrupt:
        print(
            "\n[STOP] User stopped bot."
        )

    except Exception as exc:
        print(
            f"[FATAL] {exc}"
        )

        traceback.print_exc()

    finally:
        if client is not None:
            try:
                await client.close_connection()
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())