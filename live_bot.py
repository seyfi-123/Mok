#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import math
import signal
import hashlib
import logging
import threading
import traceback
import statistics
import re
import xml.etree.ElementTree as ET

from dataclasses import dataclass
from datetime import datetime, timezone
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

import requests
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import websocket


# ============================================================
# SOZLAMALAR
# ============================================================

APP_NAME = "Deep Historical Market Analogue Engine"
VERSION = "4.0.0"

BINANCE_REST_URL = os.getenv(
    "BINANCE_REST_URL",
    "https://api.binance.com"
)

BINANCE_WS_URL = os.getenv(
    "BINANCE_WS_URL",
    "wss://stream.binance.com:9443/stream"
)

TELEGRAM_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN",
    ""
)

TELEGRAM_CHAT_ID = os.getenv(
    "TELEGRAM_CHAT_ID",
    ""
)

PORT = int(os.getenv("PORT", "8080"))

SYMBOLS = [
    x.strip().upper()
    for x in os.getenv(
        "SYMBOLS",
        "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT"
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

MAX_CANDLES_MEMORY = int(
    os.getenv("MAX_CANDLES_MEMORY", "12000")
)

MIN_ANALOGUES = int(
    os.getenv("MIN_ANALOGUES", "100")
)

TARGET_ANALOGUES = int(
    os.getenv("TARGET_ANALOGUES", "500")
)

MIN_SIMILARITY = float(
    os.getenv("MIN_SIMILARITY", "0.72")
)

TOP_ANALOGUES = int(
    os.getenv("TOP_ANALOGUES", "500")
)

ANALOGUE_STEP = int(
    os.getenv("ANALOGUE_STEP", "3")
)

FORWARD_BARS = int(
    os.getenv("FORWARD_BARS", "12")
)

SIGNAL_COOLDOWN_MINUTES = int(
    os.getenv("SIGNAL_COOLDOWN_MINUTES", "60")
)

SCAN_SECONDS = int(
    os.getenv("SCAN_SECONDS", "300")
)

MARKET_SCAN_SECONDS = int(
    os.getenv("MARKET_SCAN_SECONDS", "600")
)

NEWS_SCAN_SECONDS = int(
    os.getenv("NEWS_SCAN_SECONDS", "300")
)

REQUEST_TIMEOUT = int(
    os.getenv("REQUEST_TIMEOUT", "20")
)

RSS_URL = os.getenv(
    "RSS_URL",
    "https://feeds.feedburner.com/CoinDesk"
)

MARKET_SCANNER_ENABLED = (
    os.getenv("MARKET_SCANNER_ENABLED", "true").lower()
    in ("1", "true", "yes", "on")
)

NEWS_ENABLED = (
    os.getenv("NEWS_ENABLED", "true").lower()
    in ("1", "true", "yes", "on")
)

CHART_ENABLED = (
    os.getenv("CHART_ENABLED", "true").lower()
    in ("1", "true", "yes", "on")
)

TELEGRAM_ENABLED = (
    os.getenv("TELEGRAM_ENABLED", "true").lower()
    in ("1", "true", "yes", "on")
    and bool(TELEGRAM_TOKEN)
    and bool(TELEGRAM_CHAT_ID)
)


# ============================================================
# LOG
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

log = logging.getLogger(APP_NAME)


# ============================================================
# GLOBAL HOLAT
# ============================================================

STOP = False

DATA_LOCK = threading.RLock()

MARKET_DATA = defaultdict(
    lambda: defaultdict(list)
)

LAST_SIGNAL = {}

NEWS_CACHE = []

LAST_NEWS_SCAN = 0

LAST_MARKET_SCAN = 0

SESSION_START = time.time()


# ============================================================
# MA'LUMOT MODELLARI
# ============================================================

@dataclass
class Candle:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int

    @property
    def body(self):
        return abs(self.close - self.open)

    @property
    def range(self):
        return max(
            self.high - self.low,
            1e-12
        )

    @property
    def upper_wick(self):
        return (
            self.high
            - max(self.open, self.close)
        )

    @property
    def lower_wick(self):
        return (
            min(self.open, self.close)
            - self.low
        )

    @property
    def direction(self):
        if self.close > self.open:
            return 1

        if self.close < self.open:
            return -1

        return 0

    @property
    def body_ratio(self):
        return self.body / self.range

    @property
    def upper_wick_ratio(self):
        return self.upper_wick / self.range

    @property
    def lower_wick_ratio(self):
        return self.lower_wick / self.range


@dataclass
class Analogue:
    index: int
    similarity: float
    direction: str
    max_up: float
    max_down: float
    close_move: float
    bars_to_extreme: int


@dataclass
class NewsItem:
    title: str
    link: str
    published: str
    source: str


# ============================================================
# YORDAMCHI FUNKSIYALAR
# ============================================================

def now_ms():
    return int(time.time() * 1000)


def utc_now():
    return datetime.now(timezone.utc)


def pct(a, b):
    if b == 0:
        return 0.0

    return (
        (a - b)
        / abs(b)
        * 100.0
    )


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def clamp(value, minimum, maximum):
    return max(
        minimum,
        min(maximum, value)
    )


def mean_or_zero(values):
    if not values:
        return 0.0

    return float(
        statistics.mean(values)
    )


def median_or_zero(values):
    if not values:
        return 0.0

    return float(
        statistics.median(values)
    )


def fmt_price(value):
    value = safe_float(value)

    if value >= 1000:
        return f"{value:,.2f}"

    if value >= 1:
        return f"{value:,.4f}"

    return f"{value:.8f}"


def timeframe_seconds(timeframe):
    if timeframe.endswith("m"):
        return int(
            timeframe[:-1]
        ) * 60

    if timeframe.endswith("h"):
        return int(
            timeframe[:-1]
        ) * 3600

    if timeframe.endswith("d"):
        return int(
            timeframe[:-1]
        ) * 86400

    return 300


def sleep_interruptible(seconds):
    end_time = time.time() + seconds

    while not STOP and time.time() < end_time:
        time.sleep(
            min(1.0, end_time - time.time())
        )


def normalize_symbol(symbol):
    return (
        symbol
        .replace("/", "")
        .replace("-", "")
        .upper()
    )


# ============================================================
# BINANCE
# ============================================================

HTTP_SESSION = requests.Session()

HTTP_SESSION.headers.update({
    "User-Agent":
        "DeepHistoricalMarketEngine/4.0"
})


def binance_get(
    path,
    params=None,
    retries=5
):
    url = (
        BINANCE_REST_URL.rstrip("/")
        + path
    )

    last_error = None

    for attempt in range(retries):
        try:
            response = HTTP_SESSION.get(
                url,
                params=params or {},
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 200:
                return response.json()

            if response.status_code in (
                418,
                429,
                500,
                502,
                503,
                504
            ):
                wait = min(
                    30,
                    2 ** attempt
                )

                log.warning(
                    "Binance javobi %s. %s soniya kutish.",
                    response.status_code,
                    wait
                )

                time.sleep(wait)
                continue

            response.raise_for_status()

        except Exception as exc:
            last_error = exc

            wait = min(
                30,
                2 ** attempt
            )

            log.warning(
                "Binance so'rovida xato: %s. %s soniya kutish.",
                exc,
                wait
            )

            time.sleep(wait)

    raise RuntimeError(
        f"Binance so'rovi muvaffaqiyatsiz: {last_error}"
    )


def history_days_for(timeframe):
    if timeframe == "5m":
        return 180

    if timeframe == "15m":
        return 365

    if timeframe == "1h":
        return 730

    if timeframe.endswith("m"):
        return 180

    if timeframe.endswith("h"):
        return 365

    return 365


def fetch_klines(
    symbol,
    interval,
    days
):
    symbol = normalize_symbol(symbol)

    seconds = timeframe_seconds(interval)

    total_ms = int(
        days
        * 86400
        * 1000
    )

    end_time = now_ms()

    start_time = end_time - total_ms

    all_rows = []

    cursor = start_time

    while cursor < end_time and not STOP:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_time,
            "limit": 1000
        }

        rows = binance_get(
            "/api/v3/klines",
            params
        )

        if not rows:
            break

        all_rows.extend(rows)

        last_open = int(
            rows[-1][0]
        )

        next_cursor = (
            last_open
            + seconds * 1000
        )

        if next_cursor <= cursor:
            break

        cursor = next_cursor

        if len(rows) < 1000:
            break

        time.sleep(0.08)

    candles = []

    seen = set()

    for row in all_rows:
        try:
            open_time = int(row[0])

            if open_time in seen:
                continue

            seen.add(open_time)

            candles.append(
                Candle(
                    open_time=open_time,
                    open=safe_float(row[1]),
                    high=safe_float(row[2]),
                    low=safe_float(row[3]),
                    close=safe_float(row[4]),
                    volume=safe_float(row[5]),
                    close_time=int(row[6])
                )
            )

        except Exception:
            continue

    candles.sort(
        key=lambda x: x.open_time
    )

    # Hozirgi yopilmagan shamni olib tashlash.
    current_ms = now_ms()

    candles = [
        c
        for c in candles
        if c.close_time <= current_ms
    ]

    if len(candles) > MAX_CANDLES_MEMORY:
        candles = candles[
            -MAX_CANDLES_MEMORY:
        ]

    return candles


# ============================================================
# SHAM PATTERNLARI
# ============================================================

def candle_color(candle):
    if candle.close > candle.open:
        return "BULLISH"

    if candle.close < candle.open:
        return "BEARISH"

    return "NEUTRAL"


def pin_bar(candle):
    body = candle.body
    upper = candle.upper_wick
    lower = candle.lower_wick

    if body <= 0:
        body = candle.range * 0.01

    bullish = (
        lower >= body * 2
        and upper <= body
    )

    bearish = (
        upper >= body * 2
        and lower <= body
    )

    if bullish:
        return "Bullish Pin Bar"

    if bearish:
        return "Bearish Pin Bar"

    return None


def engulfing(previous, current):
    if (
        previous.close < previous.open
        and current.close > current.open
        and current.open <= previous.close
        and current.close >= previous.open
    ):
        return "Bullish Engulfing"

    if (
        previous.close > previous.open
        and current.close < current.open
        and current.open >= previous.close
        and current.close <= previous.open
    ):
        return "Bearish Engulfing"

    return None


def inside_bar(previous, current):
    if (
        current.high <= previous.high
        and current.low >= previous.low
    ):
        return "Inside Bar"

    return None


def outside_bar(previous, current):
    if (
        current.high >= previous.high
        and current.low <= previous.low
    ):
        return "Outside Bar"

    return None


def detect_candlestick_patterns(candles):
    if len(candles) < 3:
        return []

    patterns = []

    current = candles[-1]
    previous = candles[-2]

    p = pin_bar(current)

    if p:
        patterns.append(p)

    p = engulfing(
        previous,
        current
    )

    if p:
        patterns.append(p)

    p = inside_bar(
        previous,
        current
    )

    if p:
        patterns.append(p)

    p = outside_bar(
        previous,
        current
    )

    if p:
        patterns.append(p)

    if current.body_ratio > 0.70:
        if current.direction > 0:
            patterns.append(
                "Strong Bullish Candle"
            )

        elif current.direction < 0:
            patterns.append(
                "Strong Bearish Candle"
            )

    return patterns


# ============================================================
# SWING VA STRUKTURA
# ============================================================

def local_swings(
    candles,
    strength=2
):
    highs = []
    lows = []

    if len(candles) < strength * 2 + 1:
        return highs, lows

    for i in range(
        strength,
        len(candles) - strength
    ):
        current = candles[i]

        left = candles[
            i - strength:i
        ]

        right = candles[
            i + 1:i + strength + 1
        ]

        if all(
            current.high >= c.high
            for c in left + right
        ):
            highs.append(i)

        if all(
            current.low <= c.low
            for c in left + right
        ):
            lows.append(i)

    return highs, lows


def structure_state(candles):
    if len(candles) < 20:
        return (
            "Маълумот етарли эмас",
            {}
        )

    data = candles[-100:]

    highs, lows = local_swings(
        data,
        strength=2
    )

    if len(highs) < 2 or len(lows) < 2:
        return (
            "Нейтрал структура",
            {}
        )

    last_high_1 = data[
        highs[-1]
    ].high

    last_high_2 = data[
        highs[-2]
    ].high

    last_low_1 = data[
        lows[-1]
    ].low

    last_low_2 = data[
        lows[-2]
    ].low

    higher_high = (
        last_high_1 > last_high_2
    )

    higher_low = (
        last_low_1 > last_low_2
    )

    lower_high = (
        last_high_1 < last_high_2
    )

    lower_low = (
        last_low_1 < last_low_2
    )

    if higher_high and higher_low:
        state = "Кўтарилиш структураси"

    elif lower_high and lower_low:
        state = "Пасайиш структураси"

    elif (
        higher_high
        or higher_low
        or lower_high
        or lower_low
    ):
        state = "Аралаш / нейтрал структура"

    else:
        state = "Нейтрал структура"

    details = {
        "higher_high": higher_high,
        "higher_low": higher_low,
        "lower_high": lower_high,
        "lower_low": lower_low,
        "last_high": last_high_1,
        "previous_high": last_high_2,
        "last_low": last_low_1,
        "previous_low": last_low_2
    }

    return state, details


# ============================================================
# PRICE ACTION KONTEXT
# ============================================================

def average_range(candles, window=20):
    if not candles:
        return 0.0

    data = candles[-window:]

    return mean_or_zero([
        c.range
        for c in data
    ])


def compression_score(candles):
    if len(candles) < 40:
        return 0.0

    short_range = average_range(
        candles,
        10
    )

    long_range = average_range(
        candles,
        40
    )

    if long_range <= 0:
        return 0.0

    ratio = (
        short_range
        / long_range
    )

    return clamp(
        1.0 - ratio,
        0.0,
        1.0
    )


def impulse_info(candles):
    if len(candles) < 10:
        return 0, 0.0

    data = candles[-10:]

    first = data[0].close
    last = data[-1].close

    move = pct(
        last,
        first
    )

    if move > 1.0:
        direction = 1

    elif move < -1.0:
        direction = -1

    else:
        direction = 0

    strength = clamp(
        abs(move) / 5.0,
        0.0,
        2.0
    )

    return direction, strength


def breakout_info(candles):
    if len(candles) < 30:
        return 0

    current = candles[-1]

    previous = candles[-21:-1]

    highest = max(
        c.high
        for c in previous
    )

    lowest = min(
        c.low
        for c in previous
    )

    if current.close > highest:
        return 1

    if current.close < lowest:
        return -1

    return 0


def rejection_info(candles):
    if not candles:
        return 0

    current = candles[-1]

    if (
        current.lower_wick_ratio > 0.55
        and current.direction >= 0
    ):
        return 1

    if (
        current.upper_wick_ratio > 0.55
        and current.direction <= 0
    ):
        return -1

    return 0


def exhaustion_info(candles):
    if len(candles) < 8:
        return 0

    data = candles[-8:]

    directions = [
        c.direction
        for c in data
    ]

    bullish = sum(
        1
        for x in directions
        if x > 0
    )

    bearish = sum(
        1
        for x in directions
        if x < 0
    )

    current = data[-1]

    if (
        bullish >= 6
        and current.upper_wick_ratio > 0.45
    ):
        return -1

    if (
        bearish >= 6
        and current.lower_wick_ratio > 0.45
    ):
        return 1

    return 0


def mw_pattern(candles):
    if len(candles) < 30:
        return "Йўқ"

    data = candles[-30:]

    highs, lows = local_swings(
        data,
        strength=2
    )

    if len(highs) >= 3:
        h1 = data[highs[-3]].high
        h2 = data[highs[-2]].high
        h3 = data[highs[-1]].high

        if (
            abs(h1 - h3)
            / max(h1, 1e-12)
            < 0.015
            and h2 < h1
        ):
            return "M шакл эҳтимоли"

    if len(lows) >= 3:
        l1 = data[lows[-3]].low
        l2 = data[lows[-2]].low
        l3 = data[lows[-1]].low

        if (
            abs(l1 - l3)
            / max(l1, 1e-12)
            < 0.015
            and l2 > l1
        ):
            return "W шакл эҳтимоли"

    return "Йўқ"


def build_context(candles):
    if not candles:
        return {}

    current = candles[-1]

    structure, structure_details = (
        structure_state(candles)
    )

    impulse_direction, impulse_strength = (
        impulse_info(candles)
    )

    breakout_direction = (
        breakout_info(candles)
    )

    rejection_direction = (
        rejection_info(candles)
    )

    exhaustion_direction = (
        exhaustion_info(candles)
    )

    patterns = detect_candlestick_patterns(
        candles
    )

    avg_range = average_range(
        candles,
        20
    )

    range_pct = 0.0

    if current.close:
        range_pct = (
            current.range
            / current.close
            * 100.0
        )

    return {
        "price": current.close,
        "direction": current.direction,
        "body_ratio": current.body_ratio,
        "upper_wick_ratio":
            current.upper_wick_ratio,
        "lower_wick_ratio":
            current.lower_wick_ratio,
        "compression":
            compression_score(candles),
        "impulse_direction":
            impulse_direction,
        "impulse_strength":
            impulse_strength,
        "breakout_direction":
            breakout_direction,
        "rejection_direction":
            rejection_direction,
        "exhaustion_direction":
            exhaustion_direction,
        "structure_state":
            structure,
        "structure_details":
            structure_details,
        "patterns":
            patterns,
        "mw":
            mw_pattern(candles),
        "range_pct":
            range_pct,
        "average_range":
            avg_range
    }


# ============================================================
# KONTEXT VEKTORI
# ============================================================

def context_vector(context):
    structure = (
        context.get(
            "structure_state",
            ""
        )
    )

    structure_up = (
        1.0
        if "Кўтарилиш" in structure
        else 0.0
    )

    structure_down = (
        1.0
        if "Пасайиш" in structure
        else 0.0
    )

    patterns = context.get(
        "patterns",
        []
    )

    pattern_text = " ".join(
        patterns
    )

    vector = [
        float(
            context.get(
                "direction",
                0
            )
        ),

        float(
            context.get(
                "body_ratio",
                0
            )
        ),

        float(
            context.get(
                "upper_wick_ratio",
                0
            )
        ),

        float(
            context.get(
                "lower_wick_ratio",
                0
            )
        ),

        float(
            context.get(
                "compression",
                0
            )
        ),

        float(
            context.get(
                "impulse_direction",
                0
            )
        ),

        float(
            context.get(
                "impulse_strength",
                0
            )
        ),

        float(
            context.get(
                "breakout_direction",
                0
            )
        ),

        float(
            context.get(
                "rejection_direction",
                0
            )
        ),

        float(
            context.get(
                "exhaustion_direction",
                0
            )
        ),

        structure_up,

        structure_down,

        1.0
        if "Pin Bar" in pattern_text
        else 0.0,

        1.0
        if "Engulfing" in pattern_text
        else 0.0,

        1.0
        if "Inside Bar" in pattern_text
        else 0.0,

        1.0
        if "Outside Bar" in pattern_text
        else 0.0,

        1.0
        if "Breakout" in pattern_text
        else 0.0,

        clamp(
            safe_float(
                context.get(
                    "range_pct",
                    0
                )
            ) / 5.0,
            0.0,
            2.0
        )
    ]

    return np.array(
        vector,
        dtype=float
    )


def cosine_similarity(a, b):
    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator <= 1e-12:
        return 0.0

    return float(
        np.dot(a, b)
        / denominator
    )


# ============================================================
# TARIXIY ANALOG ENGINE
# ============================================================

class HistoricalEngine:

    def __init__(self):
        self.indexes = {}
        self.lock = threading.RLock()

    def build_index(
        self,
        symbol,
        timeframe,
        candles
    ):
        key = (
            symbol,
            timeframe
        )

        vectors = []
        positions = []

        minimum_history = 45 + FORWARD_BARS

        if len(candles) <= minimum_history:
            with self.lock:
                self.indexes[key] = {
                    "vectors": np.empty(
                        (0, 18)
                    ),
                    "positions": []
                }

            return

        end_index = (
            len(candles)
            - FORWARD_BARS
        )

        for i in range(
            45,
            end_index,
            ANALOGUE_STEP
        ):
            sample = candles[
                :i + 1
            ]

            context = build_context(
                sample[-80:]
            )

            vector = context_vector(
                context
            )

            vectors.append(vector)
            positions.append(i)

        matrix = np.array(
            vectors,
            dtype=float
        )

        with self.lock:
            self.indexes[key] = {
                "vectors": matrix,
                "positions": positions
            }

        log.info(
            "Tarixiy indeks tayyor: %s %s | %s analog nuqta",
            symbol,
            timeframe,
            len(positions)
        )

    def search(
        self,
        symbol,
        timeframe,
        candles,
        target_context
    ):
        key = (
            symbol,
            timeframe
        )

        with self.lock:
            index = self.indexes.get(key)

        if not index:
            self.build_index(
                symbol,
                timeframe,
                candles
            )

            with self.lock:
                index = self.indexes.get(key)

        if not index:
            return []

        vectors = index["vectors"]
        positions = index["positions"]

        if (
            vectors is None
            or len(vectors) == 0
        ):
            return []

        target_vector = context_vector(
            target_context
        )

        norms = np.linalg.norm(
            vectors,
            axis=1
        )

        target_norm = np.linalg.norm(
            target_vector
        )

        if target_norm <= 1e-12:
            return []

        denominator = (
            norms
            * target_norm
        )

        similarities = np.zeros(
            len(vectors),
            dtype=float
        )

        valid = denominator > 1e-12

        similarities[valid] = (
            np.dot(
                vectors[valid],
                target_vector
            )
            / denominator[valid]
        )

        candidate_indexes = np.where(
            similarities >= MIN_SIMILARITY
        )[0]

        if len(candidate_indexes) == 0:
            candidate_indexes = np.argsort(
                similarities
            )[
                -TARGET_ANALOGUES:
            ]

        else:
            candidate_indexes = candidate_indexes[
                np.argsort(
                    similarities[
                        candidate_indexes
                    ]
                )[::-1]
            ]

        candidate_indexes = candidate_indexes[
            :TOP_ANALOGUES
        ]

        analogues = []

        for candidate in candidate_indexes:
            historical_index = positions[
                int(candidate)
            ]

            similarity = float(
                similarities[
                    int(candidate)
                ]
            )

            if (
                historical_index
                + FORWARD_BARS
                >= len(candles)
            ):
                continue

            entry_price = candles[
                historical_index
            ].close

            future = candles[
                historical_index
                + 1:
                historical_index
                + 1
                + FORWARD_BARS
            ]

            if not future:
                continue

            max_up = max(
                (
                    (
                        c.high
                        - entry_price
                    )
                    / entry_price
                    * 100.0
                )
                for c in future
            )

            max_down = min(
                (
                    (
                        c.low
                        - entry_price
                    )
                    / entry_price
                    * 100.0
                )
                for c in future
            )

            close_move = (
                (
                    future[-1].close
                    - entry_price
                )
                / entry_price
                * 100.0
            )

            if (
                max_up >= 0.30
                and max_up
                > abs(max_down)
            ):
                direction = "UP"

                bars_to_extreme = next(
                    (
                        i + 1
                        for i, c in enumerate(future)
                        if (
                            (
                                c.high
                                - entry_price
                            )
                            / entry_price
                            * 100.0
                        )
                        >= max_up
                    ),
                    FORWARD_BARS
                )

            elif (
                max_down <= -0.30
                and abs(max_down)
                > max_up
            ):
                direction = "DOWN"

                bars_to_extreme = next(
                    (
                        i + 1
                        for i, c in enumerate(future)
                        if (
                            (
                                c.low
                                - entry_price
                            )
                            / entry_price
                            * 100.0
                        )
                        <= max_down
                    ),
                    FORWARD_BARS
                )

            else:
                direction = "FLAT"

                bars_to_extreme = FORWARD_BARS

            analogues.append(
                Analogue(
                    index=historical_index,
                    similarity=similarity,
                    direction=direction,
                    max_up=float(max_up),
                    max_down=float(max_down),
                    close_move=float(close_move),
                    bars_to_extreme=int(
                        bars_to_extreme
                    )
                )
            )

        analogues.sort(
            key=lambda x: x.similarity,
            reverse=True
        )

        return analogues[
            :TOP_ANALOGUES
        ]


HISTORICAL = HistoricalEngine()


# ============================================================
# TARIXIY STATISTIKA
# ============================================================

def analogue_statistics(
    analogues
):
    total = len(analogues)

    if total == 0:
        return {
            "total": 0,
            "up": 0,
            "down": 0,
            "flat": 0,
            "up_pct": 0.0,
            "down_pct": 0.0,
            "flat_pct": 0.0,
            "median_move": 0.0,
            "median_up": 0.0,
            "median_down": 0.0,
            "targets": {},
            "profit_targets": {},
            "loss_targets": {},
            "avg_similarity": 0.0
        }

    up = sum(
        1
        for a in analogues
        if a.direction == "UP"
    )

    down = sum(
        1
        for a in analogues
        if a.direction == "DOWN"
    )

    flat = sum(
        1
        for a in analogues
        if a.direction == "FLAT"
    )

    up_moves = [
        a.max_up
        for a in analogues
    ]

    down_moves = [
        a.max_down
        for a in analogues
    ]

    close_moves = [
        a.close_move
        for a in analogues
    ]

    targets = {}

    for target in (
        -1,
        -2,
        -3,
        -4,
        -5,
        -10,
        -15,
        1,
        2,
        3,
        4,
        5,
        10,
        15
    ):
        if target < 0:
            hits = sum(
                1
                for a in analogues
                if a.max_down <= target
            )

        else:
            hits = sum(
                1
                for a in analogues
                if a.max_up >= target
            )

        targets[target] = {
            "count": hits,
            "pct": (
                hits
                / total
                * 100.0
            )
        }

    # ========================================================
    # QO'SHIMCHA 1: +1%, +2%, +3%, +4%
    # ========================================================

    profit_targets = {}

    for target in (
        1,
        2,
        3,
        4
    ):
        hits = sum(
            1
            for a in analogues
            if a.max_up >= target
        )

        profit_targets[target] = {
            "count": hits,
            "total": total,
            "pct": (
                hits
                / total
                * 100.0
            )
        }

    # ========================================================
    # QO'SHIMCHA 1: -1%, -2%, -3%, -4%
    # ========================================================

    loss_targets = {}

    for target in (
        1,
        2,
        3,
        4
    ):
        hits = sum(
            1
            for a in analogues
            if a.max_down <= -target
        )

        loss_targets[target] = {
            "count": hits,
            "total": total,
            "pct": (
                hits
                / total
                * 100.0
            )
        }

    return {
        "total": total,
        "up": up,
        "down": down,
        "flat": flat,
        "up_pct": up / total * 100.0,
        "down_pct": down / total * 100.0,
        "flat_pct": flat / total * 100.0,
        "median_move":
            median_or_zero(close_moves),
        "median_up":
            median_or_zero(up_moves),
        "median_down":
            median_or_zero(down_moves),
        "targets": targets,

        # QO'SHIMCHA STATISTIKALAR
        "profit_targets":
            profit_targets,

        "loss_targets":
            loss_targets,

        "avg_similarity":
            mean_or_zero([
                a.similarity
                for a in analogues
            ])
    }


# ============================================================
# SCENARIY
# ============================================================

def price_action_bias(context):
    score = 0.0

    score += (
        context.get(
            "impulse_direction",
            0
        )
        * min(
            2.0,
            context.get(
                "impulse_strength",
                0
            )
        )
    )

    score += (
        context.get(
            "breakout_direction",
            0
        )
        * 1.5
    )

    score += (
        context.get(
            "rejection_direction",
            0
        )
        * 1.0
    )

    score += (
        context.get(
            "exhaustion_direction",
            0
        )
        * 1.2
    )

    structure = context.get(
        "structure_state",
        ""
    )

    if "Кўтарилиш" in structure:
        score += 2.0

    elif "Пасайиш" in structure:
        score -= 2.0

    if score >= 2.0:
        return "UP"

    if score <= -2.0:
        return "DOWN"

    return "FLAT"


def historical_bias(stats):
    total = stats.get(
        "total",
        0
    )

    if total < 30:
        return "FLAT"

    up_pct = stats.get(
        "up_pct",
        0
    )

    down_pct = stats.get(
        "down_pct",
        0
    )

    if down_pct > up_pct + 8:
        return "DOWN"

    if up_pct > down_pct + 8:
        return "UP"

    return "FLAT"


def scenario_score(
    context,
    stats
):
    pa = price_action_bias(
        context
    )

    hist = historical_bias(
        stats
    )

    score = 0.0

    if pa == "UP":
        score += 1.0

    elif pa == "DOWN":
        score -= 1.0

    if hist == "UP":
        score += 1.0

    elif hist == "DOWN":
        score -= 1.0

    return score


# ============================================================
# TIMEFRAME ANALIZI
# ============================================================

def analyze_timeframe(
    symbol,
    timeframe,
    candles
):
    if len(candles) < 120:
        return {
            "ready": False,
            "timeframe": timeframe,
            "reason": "Маълумот етарли эмас"
        }

    context = build_context(
        candles[-120:]
    )

    analogues = HISTORICAL.search(
        symbol,
        timeframe,
        candles,
        context
    )

    stats = analogue_statistics(
        analogues
    )

    pa_bias = price_action_bias(
        context
    )

    hist_bias = historical_bias(
        stats
    )

    if (
        pa_bias == hist_bias
        and pa_bias != "FLAT"
    ):
        combined = pa_bias
    else:
        combined = "FLAT"

    return {
        "ready": True,
        "timeframe": timeframe,
        "context": context,
        "analogues": analogues,
        "stats": stats,
        "price_action_bias": pa_bias,
        "historical_bias": hist_bias,
        "combined_bias": combined
    }


def multi_timeframe_analysis(
    symbol
):
    results = {}

    with DATA_LOCK:
        data = {
            timeframe: list(
                MARKET_DATA[
                    symbol
                ][timeframe]
            )
            for timeframe in TIMEFRAMES
        }

    for timeframe in TIMEFRAMES:
        candles = data.get(
            timeframe,
            []
        )

        results[timeframe] = (
            analyze_timeframe(
                symbol,
                timeframe,
                candles
            )
        )

    return results


# ============================================================
# SIGNAL GATE
# ============================================================

def signal_gate(
    mtf
):
    ready = [
        item
        for item in mtf.values()
        if item.get("ready")
    ]

    if not ready:
        return {
            "status": "WATCH",
            "direction": "FLAT",
            "agreement": 0.0,
            "avg_hist": 0.0,
            "avg_similarity": 0.0,
            "avg_analogues": 0,
            "reason":
                "Таймфреймлар учун маълумот етарли эмас"
        }

    directions = [
        item.get(
            "combined_bias",
            "FLAT"
        )
        for item in ready
    ]

    non_flat = [
        d
        for d in directions
        if d in ("UP", "DOWN")
    ]

    if not non_flat:
        return {
            "status": "WATCH",
            "direction": "FLAT",
            "agreement": 0.0,
            "avg_hist": 0.0,
            "avg_similarity": mean_or_zero([
                item["stats"].get(
                    "avg_similarity",
                    0
                )
                for item in ready
            ]),
            "avg_analogues": mean_or_zero([
                item["stats"].get(
                    "total",
                    0
                )
                for item in ready
            ]),
            "reason":
                "Таймфреймлар умумий йўналишни тасдиқламади"
        }

    up_count = sum(
        1
        for d in non_flat
        if d == "UP"
    )

    down_count = sum(
        1
        for d in non_flat
        if d == "DOWN"
    )

    if up_count >= down_count:
        selected = "UP"
        count = up_count
    else:
        selected = "DOWN"
        count = down_count

    agreement = (
        count
        / len(ready)
    )

    hist_values = []

    for item in ready:
        stats = item["stats"]

        if selected == "UP":
            hist_values.append(
                stats.get(
                    "up_pct",
                    0
                )
            )

        else:
            hist_values.append(
                stats.get(
                    "down_pct",
                    0
                )
            )

    avg_hist = mean_or_zero(
        hist_values
    )

    avg_similarity = mean_or_zero([
        item["stats"].get(
            "avg_similarity",
            0
        )
        for item in ready
    ])

    avg_analogues = mean_or_zero([
        item["stats"].get(
            "total",
            0
        )
        for item in ready
    ])

    if (
        agreement >= 0.66
        and avg_hist >= 60.0
        and avg_similarity >= 0.76
        and avg_analogues >= MIN_ANALOGUES
    ):
        status = "SIGNAL"

    elif (
        avg_hist >= 55.0
        and avg_analogues >= 50
    ):
        status = "CONFIRMATION"

    else:
        status = "WATCH"

    reason = (
        f"{count}/{len(ready)} таймфрейм "
        f"бир томонга мос"
    )

    return {
        "status": status,
        "direction": selected,
        "agreement": agreement,
        "avg_hist": avg_hist,
        "avg_similarity": avg_similarity,
        "avg_analogues": avg_analogues,
        "reason": reason
    }


# ============================================================
# BOZOR SKANERI
# ============================================================

def quick_market_state(
    candles
):
    if len(candles) < 50:
        return "FLAT"

    context = build_context(
        candles[-100:]
    )

    return price_action_bias(
        context
    )


def market_scanner():
    results = {}

    for symbol in SYMBOLS:
        symbol_states = {}

        with DATA_LOCK:
            local_data = {
                tf: list(
                    MARKET_DATA[
                        symbol
                    ][tf]
                )
                for tf in TIMEFRAMES
            }

        for timeframe, candles in local_data.items():
            symbol_states[timeframe] = (
                quick_market_state(
                    candles
                )
            )

        results[symbol] = symbol_states

    return results


def market_breadth():
    scanner = market_scanner()

    states = []

    for symbol, timeframes in scanner.items():
        preferred = timeframes.get(
            "15m"
        )

        if preferred is None:
            values = list(
                timeframes.values()
            )

            preferred = (
                values[0]
                if values
                else "FLAT"
            )

        states.append(preferred)

    total = len(states)

    if total == 0:
        return {
            "up": 0.0,
            "down": 0.0,
            "flat": 0.0
        }

    up = sum(
        1
        for x in states
        if x == "UP"
    )

    down = sum(
        1
        for x in states
        if x == "DOWN"
    )

    flat = sum(
        1
        for x in states
        if x == "FLAT"
    )

    return {
        "up":
            up / total * 100.0,
        "down":
            down / total * 100.0,
        "flat":
            flat / total * 100.0
    }


# ============================================================
# RSS YANGILIKLAR
# ============================================================

def strip_xml(text):
    if not text:
        return ""

    return (
        text
        .replace("<![CDATA[", "")
        .replace("]]>", "")
        .strip()
    )


# ============================================================
# QO'SHIMCHA: NEWS TARJIMA CACHE
# ============================================================

TRANSLATION_CACHE = {}

TRANSLATION_LOCK = threading.RLock()


def clean_translation_text(text):
    if not text:
        return ""

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def translate_news_to_uzbek(text):
    """
    RSS orqali kelgan inglizcha yangilik sarlavhasini
    o'zbek tiliga tarjima qiladi.

    Tarjima servisi ishlamasa, original sarlavha
    qaytariladi. Bot ishlashdan to'xtamaydi.
    """

    text = clean_translation_text(
        text
    )

    if not text:
        return text

    cache_key = text.lower()

    with TRANSLATION_LOCK:
        cached = TRANSLATION_CACHE.get(
            cache_key
        )

    if cached:
        return cached

    try:
        response = HTTP_SESSION.get(
            "https://translate.googleapis.com/translate_a/single",
            params={
                "client": "gtx",
                "sl": "auto",
                "tl": "uz",
                "dt": "t",
                "q": text
            },
            timeout=REQUEST_TIMEOUT
        )

        if response.status_code != 200:
            log.warning(
                "News tarjima HTTP xato: %s",
                response.status_code
            )

            return text

        data = response.json()

        translated_parts = []

        if (
            isinstance(data, list)
            and len(data) > 0
            and isinstance(data[0], list)
        ):
            for part in data[0]:
                if (
                    isinstance(part, list)
                    and len(part) > 0
                    and part[0]
                ):
                    translated_parts.append(
                        str(part[0])
                    )

        translated = clean_translation_text(
            " ".join(
                translated_parts
            )
        )

        if not translated:
            return text

        with TRANSLATION_LOCK:
            TRANSLATION_CACHE[
                cache_key
            ] = translated

            if len(TRANSLATION_CACHE) > 1000:
                first_key = next(
                    iter(
                        TRANSLATION_CACHE
                    )
                )

                TRANSLATION_CACHE.pop(
                    first_key,
                    None
                )

        return translated

    except Exception as exc:
        log.warning(
            "News tarjimasida xato: %s",
            exc
        )

        return text


def parse_rss(url):
    response = requests.get(
        url,
        timeout=REQUEST_TIMEOUT,
        headers={
            "User-Agent":
                "DeepHistoricalMarketEngine/4.0"
        }
    )

    response.raise_for_status()

    root = ET.fromstring(
        response.content
    )

    items = []

    for item in root.iter():
        if (
            item.tag.lower().endswith("item")
            or item.tag.lower().endswith("entry")
        ):
            title = ""
            link = ""
            published = ""

            for child in list(item):
                tag = child.tag.lower()

                if tag.endswith("title"):
                    title = strip_xml(
                        child.text or ""
                    )

                elif tag.endswith("link"):
                    link = (
                        child.attrib.get(
                            "href",
                            ""
                        )
                        or child.text
                        or ""
                    )

                elif (
                    tag.endswith("pubdate")
                    or tag.endswith("published")
                    or tag.endswith("updated")
                ):
                    published = (
                        child.text
                        or ""
                    )

            if title:
                items.append(
                    NewsItem(
                        title=title,
                        link=link,
                        published=published,
                        source="RSS"
                    )
                )

    return items[:100]


def refresh_news():
    global NEWS_CACHE
    global LAST_NEWS_SCAN

    try:
        items = parse_rss(
            RSS_URL
        )

        unique = {}

        for item in items:
            key = item.title.strip().lower()

            if key:
                # =================================================
                # QO'SHIMCHA: INGLIZCHA NEWS -> O'ZBEKCHA
                # =================================================

                translated_title = (
                    translate_news_to_uzbek(
                        item.title
                    )
                )

                unique[key] = NewsItem(
                    title=translated_title,
                    link=item.link,
                    published=item.published,
                    source=item.source
                )

        NEWS_CACHE = list(
            unique.values()
        )[:100]

        LAST_NEWS_SCAN = time.time()

        log.info(
            "Yangiliklar yangilandi: %s ta",
            len(NEWS_CACHE)
        )

    except Exception as exc:
        log.warning(
            "RSS yangiliklarini olishda xato: %s",
            exc
        )


def relevant_news(symbol):
    base = (
        symbol
        .replace("USDT", "")
        .upper()
    )

    keywords = [
        base,
        "BITCOIN",
        "ETHEREUM",
        "CRYPTO",
        "BINANCE",
        "FED",
        "ETF",
        "SEC",
        "RATE",
        "INFLATION",
        "CPI",
        "FOMC",
        "REGULATION"
    ]

    result = []

    for item in NEWS_CACHE:
        title_upper = (
            item.title.upper()
        )

        if any(
            keyword in title_upper
            for keyword in keywords
        ):
            result.append(item)

    return result[:5]


# ============================================================
# GRAFIK
# ============================================================

def make_chart(
    symbol,
    timeframe,
    candles,
    analysis,
    path
):
    if not candles:
        return None

    data = candles[-80:]

    if not data:
        return None

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    x = np.arange(
        len(data)
    )

    for i, candle in enumerate(data):
        candle_color_value = (
            "green"
            if candle.close >= candle.open
            else "red"
        )

        ax.plot(
            [i, i],
            [
                candle.low,
                candle.high
            ],
            color=candle_color_value,
            linewidth=1
        )

        ax.plot(
            [i, i],
            [
                candle.open,
                candle.close
            ],
            color=candle_color_value,
            linewidth=5
        )

    context = analysis.get(
        "context",
        {}
    )

    structure = context.get(
        "structure_state",
        ""
    )

    price = context.get(
        "price",
        data[-1].close
    )

    ax.set_title(
        f"{symbol} | {timeframe} | "
        f"{structure} | "
        f"Нарх: {fmt_price(price)}"
    )

    ax.set_xlabel(
        "Ёпилган шамлар"
    )

    ax.set_ylabel(
        "Нарх"
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    try:
        fig.savefig(
            path,
            dpi=140
        )

        plt.close(fig)

        return path

    except Exception as exc:
        log.warning(
            "График сақлашда хато: %s",
            exc
        )

        plt.close(fig)

        return None


# ============================================================
# TELEGRAM
# ============================================================

def telegram_api(
    method,
    data=None,
    files=None
):
    if not TELEGRAM_ENABLED:
        return None

    url = (
        "https://api.telegram.org/bot"
        f"{TELEGRAM_TOKEN}/"
        f"{method}"
    )

    try:
        response = requests.post(
            url,
            data=data or {},
            files=files,
            timeout=REQUEST_TIMEOUT
        )

        return response

    except Exception as exc:
        log.warning(
            "Telegram API xatosi: %s",
            exc
        )

        return None


def telegram_send_text(
    text
):
    if not TELEGRAM_ENABLED:
        return False

    if not text:
        return False

    chunk_size = 3900

    chunks = [
        text[i:i + chunk_size]
        for i in range(
            0,
            len(text),
            chunk_size
        )
    ]

    success = True

    for chunk in chunks:
        response = telegram_api(
            "sendMessage",
            data={
                "chat_id":
                    TELEGRAM_CHAT_ID,
                "text":
                    chunk,
                "disable_web_page_preview":
                    "true"
            }
        )

        if (
            response is None
            or response.status_code != 200
        ):
            success = False

            if response is not None:
                log.warning(
                    "Telegram matn xatosi: %s",
                    response.text[:500]
                )

    return success


def telegram_send_photo(
    photo_path,
    caption
):
    if not TELEGRAM_ENABLED:
        return False

    if not photo_path:
        return False

    if not os.path.exists(photo_path):
        log.warning(
            "Grafik fayli topilmadi: %s",
            photo_path
        )

        return False

    try:
        with open(
            photo_path,
            "rb"
        ) as photo:

            response = telegram_api(
                "sendPhoto",
                data={
                    "chat_id":
                        TELEGRAM_CHAT_ID,
                    "caption":
                        caption[:1024]
                },
                files={
                    "photo": (
                        os.path.basename(
                            photo_path
                        ),
                        photo,
                        "image/png"
                    )
                }
            )

        if (
            response is not None
            and response.status_code == 200
        ):
            log.info(
                "Telegramga grafik yuborildi: %s",
                photo_path
            )

            return True

        if response is not None:
            log.warning(
                "Telegram grafik xatosi: %s",
                response.text[:500]
            )

    except Exception as exc:
        log.warning(
            "Telegram grafik yuborishda xato: %s",
            exc
        )

    return False


# ============================================================
# HISOBOT
# ============================================================

UZ_STATUS = {
    "SIGNAL": "SIGNAL",
    "CONFIRMATION": "TASDIQLASH",
    "WATCH": "KUZATUV"
}


UZ_DIRECTION = {
    "UP": "YUQORIGA",
    "DOWN": "PASTGA",
    "FLAT": "NEYTRAL"
}


def build_report(
    symbol,
    mtf,
    gate,
    breadth=None,
    news=None
):
    lines = []

    lines.append(
        f"📊 {APP_NAME}"
    )

    lines.append(
        f"Versiya: {VERSION}"
    )

    lines.append(
        f"🪙 Aktiv: {symbol}"
    )

    primary = None

    if "15m" in mtf:
        if mtf["15m"].get("ready"):
            primary = mtf["15m"]

    if primary is None:
        for item in mtf.values():
            if item.get("ready"):
                primary = item
                break

    if primary is not None:
        context = primary.get(
            "context",
            {}
        )

        stats = primary.get(
            "stats",
            {}
        )

        lines.append(
            f"💰 Нарх: "
            f"{fmt_price(context.get('price', 0))}"
        )

        lines.append(
            f"🏗 Структура: "
            f"{context.get('structure_state', 'Номаълум')}"
        )

        lines.append(
            "⚡ Импульс: "
            f"{context.get('impulse_direction', 0)} | "
            f"куч: "
            f"{context.get('impulse_strength', 0):.2f}"
        )

        lines.append(
            "🗜 Сиқилиш: "
            f"{context.get('compression', 0):.2f}"
        )

        lines.append(
            "🚀 Пробой: "
            f"{context.get('breakout_direction', 0)}"
        )

        lines.append(
            "↩️ Рад этиш: "
            f"{context.get('rejection_direction', 0)}"
        )

        lines.append(
            "⚠️ Чарчаш: "
            f"{context.get('exhaustion_direction', 0)}"
        )

        patterns = context.get(
            "patterns",
            []
        )

        lines.append(
            "🕯 Шам patternlari: "
            + (
                ", ".join(patterns)
                if patterns
                else "Йўқ"
            )
        )

        lines.append(
            f"M/W: "
            f"{context.get('mw', 'Йўқ')}"
        )

        lines.append("")

        lines.append(
            "📚 Тарихий аналоглар:"
        )

        lines.append(
            f"Сони: "
            f"{stats.get('total', 0)}"
        )

        lines.append(
            f"Ўртача ўхшашлик: "
            f"{stats.get('avg_similarity', 0):.3f}"
        )

        lines.append(
            f"UP: "
            f"{stats.get('up', 0)} "
            f"({stats.get('up_pct', 0):.2f}%)"
        )

        lines.append(
            f"DOWN: "
            f"{stats.get('down', 0)} "
            f"({stats.get('down_pct', 0):.2f}%)"
        )

        lines.append(
            f"FLAT: "
            f"{stats.get('flat', 0)} "
            f"({stats.get('flat_pct', 0):.2f}%)"
        )

        targets = stats.get(
            "targets",
            {}
        )

        lines.append(
            "🎯 PASTGA BORISH EHTIMOLI:"
        )

        for target in (
            -1,
            -3,
            -5,
            -10,
            -15
        ):
            item = targets.get(
                target,
                {}
            )

            lines.append(
                f"{target}%: "
                f"{item.get('pct', 0):.2f}%"
            )

        # ====================================================
        # QO'SHIMCHA: +1%, +2%, +3%, +4% STATISTIKA
        # ====================================================

        profit_targets = stats.get(
            "profit_targets",
            {}
        )

        loss_targets = stats.get(
            "loss_targets",
            {}
        )

        lines.append("")

        lines.append(
            "🎯 TARIXDA FOYDA YETISH STATISTIKASI:"
        )

        for target in (
            1,
            2,
            3,
            4
        ):
            item = profit_targets.get(
                target,
                {}
            )

            lines.append(
                f"+{target}% → "
                f"{item.get('count', 0)} marta / "
                f"{item.get('total', 0)} ta = "
                f"{item.get('pct', 0):.2f}%"
            )

        lines.append("")

        lines.append(
            "📉 TARIXDA PASTGA YETISH STATISTIKASI:"
        )

        for target in (
            1,
            2,
            3,
            4
        ):
            item = loss_targets.get(
                target,
                {}
            )

            lines.append(
                f"-{target}% → "
                f"{item.get('count', 0)} marta / "
                f"{item.get('total', 0)} ta = "
                f"{item.get('pct', 0):.2f}%"
            )

        lines.append(
            f"Median yopilish harakati: "
            f"{stats.get('median_move', 0):.2f}%"
        )

    lines.append("")

    lines.append(
        "🕐 MULTI-TIMEFRAME:"
    )

    for timeframe, item in mtf.items():
        if not item.get("ready"):
            lines.append(
                f"{timeframe}: "
                "маълумот етарли эмас"
            )

            continue

        stats = item["stats"]

        lines.append(
            f"{timeframe}: "
            f"PA={item['price_action_bias']} | "
            f"HIST={item['historical_bias']} | "
            f"FINAL={item['combined_bias']} | "
            f"Analog={stats.get('total', 0)} | "
            f"Sim={stats.get('avg_similarity', 0):.3f}"
        )

    lines.append("")

    lines.append(
        "🚦 SIGNAL GATE:"
    )

    status = gate.get(
        "status",
        "WATCH"
    )

    direction = gate.get(
        "direction",
        "FLAT"
    )

    lines.append(
        f"Holat: "
        f"{UZ_STATUS.get(status, status)}"
    )

    lines.append(
        f"Yo'nalish: "
        f"{UZ_DIRECTION.get(direction, direction)}"
    )

    lines.append(
        f"Kelishuv: "
        f"{gate.get('agreement', 0) * 100:.2f}%"
    )

    lines.append(
        f"Tarixiy dalil: "
        f"{gate.get('avg_hist', 0):.2f}%"
    )

    lines.append(
        f"Analoglar o'rtachasi: "
        f"{gate.get('avg_analogues', 0):.0f}"
    )

    lines.append(
        f"O'rtacha o'xshashlik: "
        f"{gate.get('avg_similarity', 0):.3f}"
    )

    lines.append(
        f"Sabab: "
        f"{gate.get('reason', '')}"
    )

    if breadth is not None:
        lines.append("")

        lines.append(
            "🌐 BOZOR KENGligi:"
        )

        lines.append(
            f"Yuqoriga: "
            f"{breadth.get('up', 0):.2f}%"
        )

        lines.append(
            f"Pastga: "
            f"{breadth.get('down', 0):.2f}%"
        )

        lines.append(
            f"Neytral: "
            f"{breadth.get('flat', 0):.2f}%"
        )

    if news:
        lines.append("")

        lines.append(
            "📰 YANGILIK / EVENT:"
        )

        for item in news[:5]:
            lines.append(
                f"• {item.title}"
            )

    lines.append("")

    lines.append(
        "⚠️ Eslatma: tarixiy analoglar "
        "kelajakdagi natijani kafolatlamaydi."
    )

    lines.append(
        "Bot avtomatik order ochmaydi."
    )

    return "\n".join(lines)


# ============================================================
# SIGNAL TAKRORLANISHINI NAZORAT QILISH
# ============================================================

def should_send_signal(
    symbol,
    gate
):
    key = (
        symbol,
        ",".join(TIMEFRAMES)
    )

    current_time = time.time()

    previous = LAST_SIGNAL.get(
        key
    )

    if previous is None:
        LAST_SIGNAL[key] = {
            "time": current_time,
            "direction":
                gate.get(
                    "direction",
                    "FLAT"
                ),
            "status":
                gate.get(
                    "status",
                    "WATCH"
                )
        }

        return True

    elapsed = (
        current_time
        - previous["time"]
    )

    cooldown = (
        SIGNAL_COOLDOWN_MINUTES
        * 60
    )

    same_direction = (
        previous.get("direction")
        == gate.get(
            "direction",
            "FLAT"
        )
    )

    current_status = gate.get(
        "status",
        "WATCH"
    )

    if (
        current_status == "SIGNAL"
        and (
            elapsed >= cooldown
            or not same_direction
        )
    ):
        LAST_SIGNAL[key] = {
            "time": current_time,
            "direction":
                gate.get(
                    "direction",
                    "FLAT"
                ),
            "status":
                current_status
        }

        return True

    if elapsed < cooldown:
        return False

    if (
        current_status == "WATCH"
        and same_direction
    ):
        return False

    LAST_SIGNAL[key] = {
        "time": current_time,
        "direction":
            gate.get(
                "direction",
                "FLAT"
            ),
        "status":
            current_status
    }

    return True


# ============================================================
# SYMBOL ANALIZI
# ============================================================

def analyze_symbol(
    symbol,
    send_telegram=True
):
    try:
        mtf = multi_timeframe_analysis(
            symbol
        )

        gate = signal_gate(
            mtf
        )

        breadth = None

        if MARKET_SCANNER_ENABLED:
            breadth = market_breadth()

        news = None

        if NEWS_ENABLED:
            news = relevant_news(
                symbol
            )

        report = build_report(
            symbol,
            mtf,
            gate,
            breadth,
            news
        )

        log.info(
            "\n%s",
            report
        )

        if (
            send_telegram
            and TELEGRAM_ENABLED
            and should_send_signal(
                symbol,
                gate
            )
        ):
            telegram_send_text(
                report
            )

            if CHART_ENABLED:
                primary = None

                if (
                    "15m" in mtf
                    and mtf["15m"].get(
                        "ready"
                    )
                ):
                    primary = mtf["15m"]

                else:
                    for item in mtf.values():
                        if item.get("ready"):
                            primary = item
                            break

                if primary is not None:
                    timeframe = primary.get(
                        "timeframe",
                        "15m"
                    )

                    with DATA_LOCK:
                        candles = list(
                            MARKET_DATA[
                                symbol
                            ][timeframe]
                        )

                    if candles:
                        filename = (
                            f"/tmp/"
                            f"{symbol}_"
                            f"{int(time.time())}.png"
                        )

                        chart_path = make_chart(
                            symbol,
                            timeframe,
                            candles,
                            primary,
                            filename
                        )

                        if chart_path:
                            status_text = (
                                UZ_STATUS.get(
                                    gate.get(
                                        "status",
                                        "WATCH"
                                    ),
                                    gate.get(
                                        "status",
                                        "WATCH"
                                    )
                                )
                            )

                            direction_text = (
                                UZ_DIRECTION.get(
                                    gate.get(
                                        "direction",
                                        "FLAT"
                                    ),
                                    gate.get(
                                        "direction",
                                        "FLAT"
                                    )
                                )
                            )

                            telegram_send_photo(
                                chart_path,
                                (
                                    f"{symbol} | "
                                    f"Holat: "
                                    f"{status_text} | "
                                    f"Yo'nalish: "
                                    f"{direction_text}"
                                )
                            )

                            try:
                                os.remove(
                                    chart_path
                                )
                            except Exception:
                                pass

        return {
            "symbol": symbol,
            "mtf": mtf,
            "gate": gate,
            "report": report
        }

    except Exception as exc:
        log.error(
            "Symbol analizida xato %s: %s",
            symbol,
            exc
        )

        traceback.print_exc()

        return None


# ============================================================
# TARIXNI YUKLASH
# ============================================================

def load_initial_history():
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            if STOP:
                return

            try:
                days = history_days_for(
                    timeframe
                )

                log.info(
                    "Tarix yuklanmoqda: "
                    "%s %s | %s kun",
                    symbol,
                    timeframe,
                    days
                )

                candles = fetch_klines(
                    symbol,
                    timeframe,
                    days
                )

                with DATA_LOCK:
                    MARKET_DATA[
                        symbol
                    ][timeframe] = candles

                log.info(
                    "Yuklandi: %s %s | %s ta sham",
                    symbol,
                    timeframe,
                    len(candles)
                )

                if len(candles) >= 120:
                    HISTORICAL.build_index(
                        symbol,
                        timeframe,
                        candles
                    )

            except Exception as exc:
                log.error(
                    "Tarix yuklashda xato: "
                    "%s %s | %s",
                    symbol,
                    timeframe,
                    exc
                )


# ============================================================
# WEBSOCKET
# ============================================================

def websocket_streams():
    streams = []

    for symbol in SYMBOLS:
        symbol_lower = symbol.lower()

        for timeframe in TIMEFRAMES:
            streams.append(
                f"{symbol_lower}@kline_{timeframe}"
            )

    return streams


def process_ws_message(
    message
):
    try:
        payload = json.loads(
            message
        )

        data = payload.get(
            "data",
            payload
        )

        kline = data.get(
            "k"
        )

        if not kline:
            return

        symbol = normalize_symbol(
            kline.get(
                "s",
                ""
            )
        )

        timeframe = kline.get(
            "i",
            ""
        )

        candle = Candle(
            open_time=int(
                kline["t"]
            ),
            open=safe_float(
                kline["o"]
            ),
            high=safe_float(
                kline["h"]
            ),
            low=safe_float(
                kline["l"]
            ),
            close=safe_float(
                kline["c"]
            ),
            volume=safe_float(
                kline["v"]
            ),
            close_time=int(
                kline["T"]
            )
        )

        is_closed = bool(
            kline.get(
                "x",
                False
            )
        )

        with DATA_LOCK:
            candles = MARKET_DATA[
                symbol
            ][timeframe]

            if candles:
                if (
                    candles[-1].open_time
                    == candle.open_time
                ):
                    candles[-1] = candle

                elif (
                    candle.open_time
                    > candles[-1].open_time
                ):
                    candles.append(
                        candle
                    )

                else:
                    replaced = False

                    for i in range(
                        len(candles) - 1,
                        max(
                            -1,
                            len(candles) - 20
                        ),
                        -1
                    ):
                        if (
                            candles[i].open_time
                            == candle.open_time
                        ):
                            candles[i] = candle
                            replaced = True
                            break

                    if not replaced:
                        candles.append(
                            candle
                        )

            else:
                candles.append(
                    candle
                )

            if len(candles) > MAX_CANDLES_MEMORY:
                del candles[
                    :-MAX_CANDLES_MEMORY
                ]

        if is_closed:
            threading.Thread(
                target=analyze_symbol,
                args=(symbol, True),
                daemon=True
            ).start()

    except Exception as exc:
        log.warning(
            "WebSocket xabarini qayta ishlashda xato: %s",
            exc
        )


def websocket_worker():
    streams = websocket_streams()

    if not streams:
        log.warning(
            "WebSocket uchun stream topilmadi."
        )
        return

    stream_path = "/".join(
        streams
    )

    url = (
        BINANCE_WS_URL.rstrip("/")
        + "?streams="
        + stream_path
    )

    reconnect_delay = 5

    while not STOP:
        try:
            log.info(
                "Binance WebSocket ulanmoqda..."
            )

            def on_open(
                ws
            ):
                nonlocal reconnect_delay

                reconnect_delay = 5

                log.info(
                    "Binance WebSocket ulandi."
                )

            def on_message(
                ws,
                message
            ):
                process_ws_message(
                    message
                )

            def on_error(
                ws,
                error
            ):
                log.warning(
                    "WebSocket xatosi: %s",
                    error
                )

            def on_close(
                ws,
                close_status_code,
                close_msg
            ):
                log.warning(
                    "WebSocket yopildi: %s %s",
                    close_status_code,
                    close_msg
                )

            ws_app = websocket.WebSocketApp(
                url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )

            ws_app.run_forever(
                ping_interval=120,
                ping_timeout=30
            )

        except Exception as exc:
            log.warning(
                "WebSocket ishida xato: %s",
                exc
            )

        if STOP:
            break

        log.info(
            "WebSocket %s soniyadan keyin qayta ulanadi.",
            reconnect_delay
        )

        sleep_interruptible(
            reconnect_delay
        )

        reconnect_delay = min(
            reconnect_delay * 2,
            60
        )


# ============================================================
# FON ANALIZ SIKLI
# ============================================================

def analysis_loop():
    while not STOP:
        started = time.time()

        for symbol in SYMBOLS:
            if STOP:
                break

            threading.Thread(
                target=analyze_symbol,
                args=(symbol, True),
                daemon=True
            ).start()

        elapsed = (
            time.time()
            - started
        )

        sleep_interruptible(
            max(
                1,
                SCAN_SECONDS - elapsed
            )
        )


def news_loop():
    while not STOP:
        try:
            refresh_news()

        except Exception as exc:
            log.warning(
                "Yangilik siklida xato: %s",
                exc
            )

        sleep_interruptible(
            NEWS_SCAN_SECONDS
        )


def market_loop():
    global LAST_MARKET_SCAN

    while not STOP:
        try:
            if MARKET_SCANNER_ENABLED:
                result = market_scanner()

                LAST_MARKET_SCAN = time.time()

                log.info(
                    "Bozor skaneri: %s",
                    result
                )

        except Exception as exc:
            log.warning(
                "Bozor skanerida xato: %s",
                exc
            )

        sleep_interruptible(
            MARKET_SCAN_SECONDS
        )


# ============================================================
# TELEGRAM BOSHLANG'ICH XABARI
# ============================================================

def startup_message():
    if not TELEGRAM_ENABLED:
        return

    text = (
        f"🚀 {APP_NAME}\n"
        f"Versiya: {VERSION}\n\n"
        f"Bot ishga tushdi.\n\n"
        f"🪙 Aktivlar: "
        f"{', '.join(SYMBOLS)}\n"
        f"🕐 Taymfreym: "
        f"{', '.join(TIMEFRAMES)}\n\n"
        f"📚 Tarixiy analoglar:\n"
        f"5m: 180 kun\n"
        f"15m: 365 kun\n"
        f"1h: 730 kun\n\n"
        f"🔎 Price Action\n"
        f"🕯 Candlestick Patterns\n"
        f"🏗 Market Structure\n"
        f"📚 Historical Analogues\n"
        f"📊 Statistik tahlil\n"
        f"🧭 Multi-Timeframe\n"
        f"🚦 Signal Gate\n"
        f"🌐 Market Scanner\n"
        f"📰 Yangiliklar/Event\n"
        f"📈 Grafik\n\n"
        f"Binance WebSocket faol.\n"
        f"Avtomatik order ochilmaydi."
    )

    telegram_send_text(
        text
    )


# ============================================================
# HEALTH SERVER
# ============================================================

from http.server import (
    BaseHTTPRequestHandler,
    HTTPServer
)


class HealthHandler(
    BaseHTTPRequestHandler
):

    def log_message(
        self,
        format_string,
        *args
    ):
        return

    def do_GET(self):
        uptime = (
            time.time()
            - SESSION_START
        )

        with DATA_LOCK:
            historical_indexes = len(
                HISTORICAL.indexes
            )

            loaded = {
                symbol: {
                    tf: len(
                        MARKET_DATA[
                            symbol
                        ][tf]
                    )
                    for tf in TIMEFRAMES
                }
                for symbol in SYMBOLS
            }

        body = json.dumps(
            {
                "status": "ok",
                "app": APP_NAME,
                "version": VERSION,
                "uptime_seconds": uptime,
                "symbols": SYMBOLS,
                "timeframes": TIMEFRAMES,
                "historical_indexes":
                    historical_indexes,
                "candles": loaded,
                "telegram_enabled":
                    TELEGRAM_ENABLED,
                "websocket": True
            },
            ensure_ascii=False
        ).encode(
            "utf-8"
        )

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8"
        )

        self.send_header(
            "Content-Length",
            str(len(body))
        )

        self.end_headers()

        self.wfile.write(
            body
        )


def health_server():
    try:
        server = HTTPServer(
            (
                "0.0.0.0",
                PORT
            ),
            HealthHandler
        )

        log.info(
            "Health server: 0.0.0.0:%s",
            PORT
        )

        while not STOP:
            server.handle_request()

        server.server_close()

    except Exception as exc:
        log.warning(
            "Health server xatosi: %s",
            exc
        )


# ============================================================
# O'CHIRISH
# ============================================================

def shutdown(
    signum=None,
    frame=None
):
    global STOP

    if STOP:
        return

    STOP = True

    log.info(
        "Bot to'xtatilmoqda..."
    )


# ============================================================
# MAIN
# ============================================================

def main():
    global STOP

    signal.signal(
        signal.SIGINT,
        shutdown
    )

    signal.signal(
        signal.SIGTERM,
        shutdown
    )

    log.info(
        "============================================================"
    )

    log.info(
        "%s v%s",
        APP_NAME,
        VERSION
    )

    log.info(
        "Symbol: %s",
        ", ".join(SYMBOLS)
    )

    log.info(
        "Timeframe: %s",
        ", ".join(TIMEFRAMES)
    )

    log.info(
        "Telegram: %s",
        "YOQILGAN"
        if TELEGRAM_ENABLED
        else "O'CHIRILGAN"
    )

    log.info(
        "Grafik: %s",
        "YOQILGAN"
        if CHART_ENABLED
        else "O'CHIRILGAN"
    )

    log.info(
        "Market scanner: %s",
        "YOQILGAN"
        if MARKET_SCANNER_ENABLED
        else "O'CHIRILGAN"
    )

    log.info(
        "Yangiliklar: %s",
        "YOQILGAN"
        if NEWS_ENABLED
        else "O'CHIRILGAN"
    )

    log.info(
        "============================================================"
    )

    try:
        refresh_news()
    except Exception:
        pass

    load_initial_history()

    startup_message()

    threads = []

    health_thread = threading.Thread(
        target=health_server,
        name="HealthServer",
        daemon=True
    )

    health_thread.start()

    threads.append(
        health_thread
    )

    websocket_thread = threading.Thread(
        target=websocket_worker,
        name="BinanceWebSocket",
        daemon=True
    )

    websocket_thread.start()

    threads.append(
        websocket_thread
    )

    analysis_thread = threading.Thread(
        target=analysis_loop,
        name="AnalysisLoop",
        daemon=True
    )

    analysis_thread.start()

    threads.append(
        analysis_thread
    )

    if NEWS_ENABLED:
        news_thread = threading.Thread(
            target=news_loop,
            name="NewsLoop",
            daemon=True
        )

        news_thread.start()

        threads.append(
            news_thread
        )

    if MARKET_SCANNER_ENABLED:
        market_thread = threading.Thread(
            target=market_loop,
            name="MarketLoop",
            daemon=True
        )

        market_thread.start()

        threads.append(
            market_thread
        )

    log.info(
        "Bot ishga tushdi va real vaqt rejimida ishlayapti."
    )

    while not STOP:
        time.sleep(1)

    log.info(
        "Barcha jarayonlar to'xtatildi."
    )


if __name__ == "__main__":
    main()