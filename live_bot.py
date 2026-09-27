#!/usr/bin/env bash
set -e

mkdir -p deep-market-bot
cd deep-market-bot

cat > requirements.txt <<'REQ'
requests==2.32.5
websocket-client==1.8.0
matplotlib==3.10.6
numpy==2.3.3
REQ

cat > Dockerfile <<'DOCKER'
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY bot.py .

CMD ["python", "-u", "bot.py"]
DOCKER

cat > bot.py <<'PY'
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
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple

import requests
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import websocket


# ============================================================
# DEEP MARKET ANALOGUE ENGINE
# ============================================================
#
# НЕ ИСПОЛЬЗУЕТ КЛАССИЧЕСКИЕ ИНДИКАТОРЫ КАК ОСНОВУ:
# RSI / MACD / Stochastic / Bollinger / EMA / SMA НЕ ИСПОЛЬЗУЮТСЯ
# ДЛЯ ФОРМИРОВАНИЯ СИГНАЛА.
#
# ОСНОВА:
# Price Action
# Candlesticks
# Structure
# Context
# Historical Analogues
# Multi-Timeframe
# Market Breadth
# News/Event Context
#
# SIGNAL != GUARANTEE
# Historical statistics are evidence, not certainty.
# ============================================================


APP_NAME = "Deep Historical Market Analogue Engine"
VERSION = "4.0.0"


# ============================================================
# CONFIG
# ============================================================

BINANCE_REST = os.getenv(
    "BINANCE_REST",
    "https://api.binance.com"
)

BINANCE_WS = os.getenv(
    "BINANCE_WS",
    "wss://stream.binance.com:9443/stream"
)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Example:
# BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT
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

# Historical data.
# More days = more analogues but more startup work.
HISTORY_DAYS_5M = int(os.getenv("HISTORY_DAYS_5M", "180"))
HISTORY_DAYS_15M = int(os.getenv("HISTORY_DAYS_15M", "365"))
HISTORY_DAYS_1H = int(os.getenv("HISTORY_DAYS_1H", "730"))

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

# Number of future candles used to determine historical result.
FORWARD_BARS = int(
    os.getenv("FORWARD_BARS", "12")
)

# Signal cooldown.
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

# RSS sources.
NEWS_FEEDS = [
    x.strip()
    for x in os.getenv(
        "NEWS_FEEDS",
        "https://feeds.feedburner.com/CoinDesk"
    ).split(",")
    if x.strip()
]

# Set true if you want all configured symbols scanned.
MARKET_SCANNER_ENABLED = (
    os.getenv("MARKET_SCANNER_ENABLED", "true").lower()
    == "true"
)

NEWS_ENABLED = (
    os.getenv("NEWS_ENABLED", "true").lower()
    == "true"
)

CHART_ENABLED = (
    os.getenv("CHART_ENABLED", "true").lower()
    == "true"
)

TELEGRAM_ENABLED = (
    os.getenv("TELEGRAM_ENABLED", "true").lower()
    == "true"
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

log = logging.getLogger(APP_NAME)


# ============================================================
# DATA MODEL
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
        return max(self.high - self.low, 1e-12)

    @property
    def upper_wick(self):
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self):
        return min(self.open, self.close) - self.low

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
# GLOBAL STATE
# ============================================================

STOP = False

DATA_LOCK = threading.RLock()

MARKET_DATA: Dict[str, Dict[str, List[Candle]]] = defaultdict(
    lambda: defaultdict(list)
)

LAST_SIGNAL: Dict[Tuple[str, str], Dict] = {}

NEWS_CACHE: List[NewsItem] = []

LAST_NEWS_SCAN = 0
LAST_MARKET_SCAN = 0

SESSION_START = time.time()


# ============================================================
# UTILS
# ============================================================

def now_ms():
    return int(time.time() * 1000)


def utc_now():
    return datetime.now(timezone.utc)


def pct(a, b):
    if b == 0:
        return 0.0
    return ((a / b) - 1.0) * 100.0


def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def mean_or_zero(values):
    values = list(values)
    return statistics.mean(values) if values else 0.0


def median_or_zero(values):
    values = list(values)
    return statistics.median(values) if values else 0.0


def fmt_price(price):
    if price >= 1000:
        return f"{price:,.2f}"
    if price >= 1:
        return f"{price:,.4f}"
    return f"{price:.8f}"


def timeframe_seconds(tf):
    unit = tf[-1]
    value = int(tf[:-1])

    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 3600
    if unit == "d":
        return value * 86400

    return 60


def sleep_interruptible(seconds):
    end = time.time() + seconds
    while time.time() < end and not STOP:
        time.sleep(min(1, end - time.time()))


# ============================================================
# BINANCE REST
# ============================================================

session = requests.Session()
session.headers.update(
    {
        "User-Agent": "DeepHistoricalMarketEngine/4.0"
    }
)


def binance_get(path, params=None, retries=5):
    url = BINANCE_REST + path

    last_error = None

    for attempt in range(retries):
        try:
            r = session.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            if r.status_code == 200:
                return r.json()

            if r.status_code in (418, 429, 500, 502, 503, 504):
                wait = min(30, 2 ** attempt)
                log.warning(
                    "Binance HTTP %s. Retry in %ss",
                    r.status_code,
                    wait,
                )
                time.sleep(wait)
                continue

            raise RuntimeError(
                f"Binance HTTP {r.status_code}: {r.text[:500]}"
            )

        except Exception as exc:
            last_error = exc
            wait = min(30, 2 ** attempt)
            log.warning(
                "Binance request error: %s. Retry in %ss",
                exc,
                wait,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Binance request failed: {last_error}"
    )


def fetch_klines(symbol, interval, days):
    """
    Binance klines endpoint.
    Max 1000 rows per request.
    """

    end_time = now_ms()
    start_time = end_time - days * 86400000

    all_rows = []

    current = start_time

    while current < end_time and not STOP:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current,
            "endTime": end_time,
            "limit": 1000,
        }

        rows = binance_get(
            "/api/v3/klines",
            params,
        )

        if not rows:
            break

        all_rows.extend(rows)

        last_open = int(rows[-1][0])

        next_current = last_open + 1

        if next_current <= current:
            break

        current = next_current

        if len(rows) < 1000:
            break

        # Small delay to be polite to REST.
        time.sleep(0.08)

    candles = []

    for r in all_rows:
        try:
            candles.append(
                Candle(
                    open_time=int(r[0]),
                    open=float(r[1]),
                    high=float(r[2]),
                    low=float(r[3]),
                    close=float(r[4]),
                    volume=float(r[5]),
                    close_time=int(r[6]),
                )
            )
        except Exception:
            continue

    # Remove duplicate timestamps.
    unique = {}

    for c in candles:
        unique[c.open_time] = c

    candles = sorted(
        unique.values(),
        key=lambda x: x.open_time,
    )

    # Do not use currently-open candle.
    if candles:
        current_time = now_ms()

        if candles[-1].close_time > current_time:
            candles = candles[:-1]

    return candles[-MAX_CANDLES_MEMORY:]


# ============================================================
# CANDLE PATTERNS
# ============================================================

def candle_color(c):
    if c.close > c.open:
        return "BULLISH"
    if c.close < c.open:
        return "BEARISH"
    return "DOJI"


def pin_bar(c):
    body = max(c.body, 1e-12)

    long_lower = (
        c.lower_wick >= body * 2.0
        and c.lower_wick_ratio >= 0.45
    )

    long_upper = (
        c.upper_wick >= body * 2.0
        and c.upper_wick_ratio >= 0.45
    )

    if long_lower and c.close >= c.open:
        return "Bullish Pin Bar"

    if long_upper and c.close <= c.open:
        return "Bearish Pin Bar"

    return None


def engulfing(prev, cur):
    if (
        prev.close < prev.open
        and cur.close > cur.open
        and cur.open <= prev.close
        and cur.close >= prev.open
    ):
        return "Bullish Engulfing"

    if (
        prev.close > prev.open
        and cur.close < cur.open
        and cur.open >= prev.close
        and cur.close <= prev.open
    ):
        return "Bearish Engulfing"

    return None


def inside_bar(prev, cur):
    if cur.high <= prev.high and cur.low >= prev.low:
        return "Inside Bar"
    return None


def outside_bar(prev, cur):
    if cur.high >= prev.high and cur.low <= prev.low:
        return "Outside Bar"
    return None


def detect_candlestick_patterns(candles):
    if len(candles) < 3:
        return []

    prev = candles[-2]
    cur = candles[-1]

    patterns = []

    p = pin_bar(cur)
    if p:
        patterns.append(p)

    p = engulfing(prev, cur)
    if p:
        patterns.append(p)

    p = inside_bar(prev, cur)
    if p:
        patterns.append(p)

    p = outside_bar(prev, cur)
    if p:
        patterns.append(p)

    if cur.body_ratio > 0.70:
        if cur.direction > 0:
            patterns.append("Strong Bullish Candle")
        elif cur.direction < 0:
            patterns.append("Strong Bearish Candle")

    return patterns


# ============================================================
# STRUCTURE
# ============================================================

def local_swings(candles, strength=2):
    highs = []
    lows = []

    if len(candles) < strength * 2 + 1:
        return highs, lows

    for i in range(
        strength,
        len(candles) - strength
    ):
        c = candles[i]

        left = candles[i-strength:i]
        right = candles[i+1:i+strength+1]

        if all(c.high >= x.high for x in left + right):
            highs.append((i, c.high))

        if all(c.low <= x.low for x in left + right):
            lows.append((i, c.low))

    return highs, lows


def structure_state(candles):
    if len(candles) < 15:
        return {
            "state": "Маълумот етарли эмас",
            "details": [],
        }

    highs, lows = local_swings(
        candles[-100:],
        strength=2,
    )

    if len(highs) < 2 or len(lows) < 2:
        return {
            "state": "Нейтрал структура",
            "details": [],
        }

    last_h = highs[-2:]
    last_l = lows[-2:]

    h1 = last_h[0][1]
    h2 = last_h[1][1]

    l1 = last_l[0][1]
    l2 = last_l[1][1]

    details = []

    if h2 > h1:
        details.append("HH")
    elif h2 < h1:
        details.append("LH")

    if l2 > l1:
        details.append("HL")
    elif l2 < l1:
        details.append("LL")

    if "HH" in details and "HL" in details:
        state = "Кўтарилиш структураси"

    elif "LH" in details and "LL" in details:
        state = "Пасайиш структураси"

    else:
        state = "Аралаш / нейтрал структура"

    return {
        "state": state,
        "details": details,
        "highs": last_h,
        "lows": last_l,
    }


# ============================================================
# CONTEXT ANALYSIS
# ============================================================

def average_range(candles, n=20):
    arr = candles[-n:]

    if not arr:
        return 0.0

    return mean_or_zero(
        [x.range for x in arr]
    )


def compression_score(candles):
    if len(candles) < 30:
        return 0.0

    old = average_range(candles[-30:-15])
    new = average_range(candles[-15:])

    if old <= 0:
        return 0.0

    ratio = new / old

    return clamp(
        1.0 - ratio
    )


def impulse_info(candles):
    if len(candles) < 10:
        return {
            "direction": 0,
            "strength": 0.0,
            "label": "Импульс аниқ эмас",
        }

    recent = candles[-6:]

    move = pct(
        recent[-1].close,
        recent[0].open,
    )

    ranges = [
        c.range
        for c in recent
    ]

    avg = mean_or_zero(ranges)

    last = recent[-1]

    strength = (
        abs(move) / max(
            (avg / max(last.close, 1e-12)) * 100,
            0.01,
        )
    )

    if move > 1.5:
        direction = 1
        label = "Кучли юқори импульс"

    elif move < -1.5:
        direction = -1
        label = "Кучли пастга импульс"

    elif move > 0.5:
        direction = 1
        label = "Юқори ҳаракат"

    elif move < -0.5:
        direction = -1
        label = "Пастга ҳаракат"

    else:
        direction = 0
        label = "Импульс аниқ эмас"

    return {
        "direction": direction,
        "strength": strength,
        "move": move,
        "label": label,
    }


def breakout_info(candles):
    if len(candles) < 25:
        return {
            "type": "Йўқ",
            "direction": 0,
        }

    recent = candles[-1]
    box = candles[-21:-1]

    high = max(c.high for c in box)
    low = min(c.low for c in box)

    if recent.close > high:
        return {
            "type": "Breakout юқорига",
            "direction": 1,
            "level": high,
        }

    if recent.close < low:
        return {
            "type": "Breakout пастга",
            "direction": -1,
            "level": low,
        }

    # False breakout detection.
    if recent.high > high and recent.close < high:
        return {
            "type": "False Breakout юқорида",
            "direction": -1,
            "level": high,
        }

    if recent.low < low and recent.close > low:
        return {
            "type": "False Breakout пастда",
            "direction": 1,
            "level": low,
        }

    return {
        "type": "Диапазон ичида",
        "direction": 0,
        "level": None,
    }


def rejection_info(candles):
    if not candles:
        return {
            "direction": 0,
            "label": "Йўқ",
        }

    c = candles[-1]

    if (
        c.lower_wick > c.body * 2
        and c.lower_wick_ratio > 0.45
    ):
        return {
            "direction": 1,
            "label": "Пастдан кучли rejection",
        }

    if (
        c.upper_wick > c.body * 2
        and c.upper_wick_ratio > 0.45
    ):
        return {
            "direction": -1,
            "label": "Юқоридан кучли rejection",
        }

    return {
        "direction": 0,
        "label": "Rejection аниқ эмас",
    }


def exhaustion_info(candles):
    if len(candles) < 8:
        return {
            "direction": 0,
            "label": "Маълумот етарли эмас",
        }

    recent = candles[-6:]

    bull = sum(
        1 for c in recent
        if c.direction > 0
    )

    bear = sum(
        1 for c in recent
        if c.direction < 0
    )

    last = recent[-1]

    if bear >= 4 and last.upper_wick_ratio > 0.35:
        return {
            "direction": 1,
            "label": "Пастга ҳаракатда exhaustion/rejection",
        }

    if bull >= 4 and last.lower_wick_ratio > 0.35:
        return {
            "direction": -1,
            "label": "Юқори ҳаракатда exhaustion/rejection",
        }

    return {
        "direction": 0,
        "label": "Exhaustion аниқ эмас",
    }


def mw_pattern(candles):
    if len(candles) < 30:
        return None

    segment = candles[-30:]

    highs, lows = local_swings(
        segment,
        strength=2,
    )

    if len(highs) >= 3:
        h = [x[1] for x in highs[-3:]]

        if (
            abs(h[0] - h[2])
            / max(h[0], 1e-12)
            < 0.015
            and h[1] < h[0]
        ):
            return "M / Double Top эҳтимоли"

    if len(lows) >= 3:
        l = [x[1] for x in lows[-3:]]

        if (
            abs(l[0] - l[2])
            / max(l[0], 1e-12)
            < 0.015
            and l[1] > l[0]
        ):
            return "W / Double Bottom эҳтимоли"

    return None


def build_context(candles):
    if len(candles) < 40:
        return None

    patterns = detect_candlestick_patterns(candles)

    structure = structure_state(candles)

    impulse = impulse_info(candles)

    breakout = breakout_info(candles)

    rejection = rejection_info(candles)

    exhaustion = exhaustion_info(candles)

    compression = compression_score(candles)

    mw = mw_pattern(candles)

    current = candles[-1]

    context = {
        "price": current.close,
        "direction": current.direction,
        "body_ratio": current.body_ratio,
        "upper_wick_ratio": current.upper_wick_ratio,
        "lower_wick_ratio": current.lower_wick_ratio,
        "compression": compression,
        "impulse_direction": impulse["direction"],
        "impulse_strength": impulse["strength"],
        "breakout_direction": breakout["direction"],
        "rejection_direction": rejection["direction"],
        "exhaustion_direction": exhaustion["direction"],
        "structure_state": structure["state"],
        "structure_details": structure["details"],
        "patterns": patterns,
        "mw": mw or "",
        "range_pct": (
            current.range
            / max(current.close, 1e-12)
            * 100
        ),
    }

    return context


# ============================================================
# CONTEXT VECTOR
# ============================================================

def context_vector(c):
    """
    Indicator-free structural vector.

    It describes:
    candle body/wicks,
    direction,
    compression,
    impulse,
    breakout,
    rejection,
    exhaustion,
    structure.
    """

    if not c:
        return np.zeros(18, dtype=float)

    structure_up = (
        1.0
        if c["structure_state"]
        == "Кўтарилиш структураси"
        else 0.0
    )

    structure_down = (
        1.0
        if c["structure_state"]
        == "Пасайиш структураси"
        else 0.0
    )

    return np.array(
        [
            c["direction"],
            c["body_ratio"],
            c["upper_wick_ratio"],
            c["lower_wick_ratio"],
            c["compression"],
            c["impulse_direction"],
            math.tanh(c["impulse_strength"] / 3),
            c["breakout_direction"],
            c["rejection_direction"],
            c["exhaustion_direction"],
            structure_up,
            structure_down,
            1.0 if "Pin Bar" in " ".join(c["patterns"]) else 0.0,
            1.0 if "Engulfing" in " ".join(c["patterns"]) else 0.0,
            1.0 if "Inside Bar" in " ".join(c["patterns"]) else 0.0,
            1.0 if "Outside Bar" in " ".join(c["patterns"]) else 0.0,
            1.0 if "Breakout" in " ".join(c["patterns"]) else 0.0,
            c["range_pct"] / 5.0,
        ],
        dtype=float,
    )


def cosine_similarity(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0

    value = float(
        np.dot(a, b)
        / (a_norm * b_norm)
    )

    return clamp(
        (value + 1.0) / 2.0
    )


# ============================================================
# HISTORICAL ANALOGUE ENGINE
# ============================================================

class HistoricalEngine:

    def __init__(self):
        self.cache = {}

    def build_index(
        self,
        symbol,
        timeframe,
        candles,
    ):
        """
        Build compact historical context index.

        We deliberately sample contexts to avoid
        calculating every single candle on huge histories.
        """

        key = (symbol, timeframe)

        vectors = []
        positions = []

        n = len(candles)

        start = 45
        end = n - FORWARD_BARS - 1

        for i in range(
            start,
            end,
            max(1, ANALOGUE_STEP),
        ):

            # Context ending at i.
            sample = candles[
                i - 80:i + 1
            ]

            context = build_context(sample)

            if not context:
                continue

            vec = context_vector(context)

            vectors.append(vec)
            positions.append(i)

        self.cache[key] = {
            "vectors": np.asarray(
                vectors,
                dtype=np.float32,
            ),
            "positions": np.asarray(
                positions,
                dtype=np.int32,
            ),
        }

        log.info(
            "Historical index %s %s: %s contexts",
            symbol,
            timeframe,
            len(positions),
        )

    def search(
        self,
        symbol,
        timeframe,
        candles,
        current_context,
    ):

        key = (symbol, timeframe)

        if key not in self.cache:
            self.build_index(
                symbol,
                timeframe,
                candles,
            )

        idx = self.cache[key]

        vectors = idx["vectors"]
        positions = idx["positions"]

        if len(vectors) == 0:
            return []

        current_vector = context_vector(
            current_context
        )

        # Vectorized cosine similarity.
        norms = np.linalg.norm(
            vectors,
            axis=1,
        )

        current_norm = np.linalg.norm(
            current_vector
        )

        if current_norm == 0:
            return []

        dots = vectors @ current_vector

        sims = dots / (
            np.maximum(norms, 1e-12)
            * current_norm
        )

        sims = (sims + 1.0) / 2.0

        candidate_indices = np.where(
            sims >= MIN_SIMILARITY
        )[0]

        if len(candidate_indices) == 0:
            # Take nearest cases even if threshold is not met.
            order = np.argsort(sims)[::-1]

            candidate_indices = order[
                :min(
                    TARGET_ANALOGUES,
                    len(order),
                )
            ]

        else:
            order = candidate_indices[
                np.argsort(
                    sims[candidate_indices]
                )[::-1]
            ]

            candidate_indices = order[
                :TOP_ANALOGUES
            ]

        analogues = []

        for j in candidate_indices:

            i = int(
                positions[j]
            )

            if i + FORWARD_BARS >= len(candles):
                continue

            base = candles[i].close

            future = candles[
                i + 1:
                i + FORWARD_BARS + 1
            ]

            if not future:
                continue

            max_up = max(
                pct(x.high, base)
                for x in future
            )

            max_down = min(
                pct(x.low, base)
                for x in future
            )

            close_move = pct(
                future[-1].close,
                base,
            )

            if (
                max_down <= -0.30
                and abs(max_down) > max_up
            ):
                direction = "DOWN"

            elif (
                max_up >= 0.30
                and max_up > abs(max_down)
            ):
                direction = "UP"

            else:
                direction = "FLAT"

            if direction == "DOWN":
                target = max_down
            elif direction == "UP":
                target = max_up
            else:
                target = abs(close_move)

            bars_to_extreme = 0

            if direction == "DOWN":
                extreme = min(
                    range(len(future)),
                    key=lambda k: future[k].low,
                )
                bars_to_extreme = extreme + 1

            elif direction == "UP":
                extreme = max(
                    range(len(future)),
                    key=lambda k: future[k].high,
                )
                bars_to_extreme = extreme + 1

            analogues.append(
                Analogue(
                    index=i,
                    similarity=float(sims[j]),
                    direction=direction,
                    max_up=max_up,
                    max_down=max_down,
                    close_move=close_move,
                    bars_to_extreme=bars_to_extreme,
                )
            )

        analogues.sort(
            key=lambda x: x.similarity,
            reverse=True,
        )

        return analogues[:TOP_ANALOGUES]


HISTORICAL = HistoricalEngine()


# ============================================================
# STATISTICS
# ============================================================

def analogue_statistics(analogues):

    total = len(analogues)

    if total == 0:
        return {
            "total": 0,
            "up": 0,
            "down": 0,
            "flat": 0,
            "up_pct": 0,
            "down_pct": 0,
            "flat_pct": 0,
            "median_move": 0,
            "median_up": 0,
            "median_down": 0,
            "targets": {},
            "avg_similarity": 0,
        }

    up = [
        x for x in analogues
        if x.direction == "UP"
    ]

    down = [
        x for x in analogues
        if x.direction == "DOWN"
    ]

    flat = [
        x for x in analogues
        if x.direction == "FLAT"
    ]

    all_moves = [
        x.close_move
        for x in analogues
    ]

    target_levels = [
        -1,
        -3,
        -5,
        -10,
        -15,
        1,
        3,
        5,
        10,
        15,
    ]

    targets = {}

    for level in target_levels:

        if level < 0:
            hits = sum(
                1
                for x in analogues
                if x.max_down <= level
            )
        else:
            hits = sum(
                1
                for x in analogues
                if x.max_up >= level
            )

        targets[level] = {
            "count": hits,
            "pct": hits / total * 100,
        }

    return {
        "total": total,
        "up": len(up),
        "down": len(down),
        "flat": len(flat),
        "up_pct": len(up) / total * 100,
        "down_pct": len(down) / total * 100,
        "flat_pct": len(flat) / total * 100,
        "median_move": median_or_zero(
            all_moves
        ),
        "median_up": median_or_zero(
            [x.max_up for x in up]
        ),
        "median_down": median_or_zero(
            [x.max_down for x in down]
        ),
        "targets": targets,
        "avg_similarity": mean_or_zero(
            [x.similarity for x in analogues]
        ),
    }


# ============================================================
# SCENARIO ENGINE
# ============================================================

def price_action_bias(context):

    score = 0.0

    score += (
        context["impulse_direction"]
        * min(
            2.0,
            context["impulse_strength"]
        )
    )

    score += (
        context["breakout_direction"]
        * 1.5
    )

    score += (
        context["rejection_direction"]
        * 1.0
    )

    # Exhaustion is reversal evidence.
    score += (
        context["exhaustion_direction"]
        * 1.2
    )

    if (
        context["structure_state"]
        == "Кўтарилиш структураси"
    ):
        score += 2.0

    elif (
        context["structure_state"]
        == "Пасайиш структураси"
    ):
        score -= 2.0

    if score >= 2.0:
        return "UP"

    if score <= -2.0:
        return "DOWN"

    return "FLAT"


def historical_bias(stats):

    if stats["total"] < 30:
        return "FLAT"

    if (
        stats["down_pct"]
        > stats["up_pct"] + 8
    ):
        return "DOWN"

    if (
        stats["up_pct"]
        > stats["down_pct"] + 8
    ):
        return "UP"

    return "FLAT"


def scenario_score(
    pa_bias,
    hist_bias,
    stats,
):

    score = 0

    if pa_bias == hist_bias:
        score += 2

    if (
        stats["total"]
        >= MIN_ANALOGUES
    ):
        score += 1

    if (
        stats["avg_similarity"]
        >= 0.80
    ):
        score += 1

    if (
        max(
            stats["up_pct"],
            stats["down_pct"],
        )
        >= 60
    ):
        score += 1

    return score


# ============================================================
# MULTI TIMEFRAME
# ============================================================

def analyze_timeframe(
    symbol,
    timeframe,
    candles,
):

    if len(candles) < 120:
        return {
            "ready": False,
            "timeframe": timeframe,
        }

    current_context = build_context(
        candles
    )

    if not current_context:
        return {
            "ready": False,
            "timeframe": timeframe,
        }

    analogues = HISTORICAL.search(
        symbol,
        timeframe,
        candles,
        current_context,
    )

    stats = analogue_statistics(
        analogues
    )

    pa_bias = price_action_bias(
        current_context
    )

    hist_bias = historical_bias(
        stats
    )

    combined = (
        pa_bias
        if pa_bias == hist_bias
        else "FLAT"
    )

    return {
        "ready": True,
        "timeframe": timeframe,
        "price": candles[-1].close,
        "context": current_context,
        "analogues": analogues,
        "stats": stats,
        "pa_bias": pa_bias,
        "hist_bias": hist_bias,
        "combined": combined,
    }


def multi_timeframe_analysis(
    symbol
):

    results = {}

    with DATA_LOCK:
        data = {
            tf: list(
                MARKET_DATA[
                    symbol
                ][tf]
            )
            for tf in TIMEFRAMES
        }

    for tf, candles in data.items():

        try:
            results[tf] = analyze_timeframe(
                symbol,
                tf,
                candles,
            )
        except Exception:
            log.exception(
                "Timeframe analysis failed %s %s",
                symbol,
                tf,
            )

    return results


# ============================================================
# SIGNAL GATE
# ============================================================

def signal_gate(
    symbol,
    mtf,
):

    ready = [
        x for x in mtf.values()
        if x.get("ready")
    ]

    if not ready:
        return {
            "status": "WAIT",
            "direction": "FLAT",
            "reason": "Маълумот етарли эмас",
        }

    directions = [
        x["combined"]
        for x in ready
        if x["combined"] in ("UP", "DOWN")
    ]

    if not directions:
        return {
            "status": "WATCH",
            "direction": "FLAT",
            "reason": "Тарихий ва жорий контекст бир томонга бирлашмади",
        }

    counts = Counter(directions)

    direction, count = counts.most_common(1)[0]

    agreement = count / len(ready)

    selected = [
        x for x in ready
        if x["combined"] == direction
    ]

    if not selected:
        return {
            "status": "WATCH",
            "direction": "FLAT",
            "reason": "Тасдиқ йўқ",
        }

    avg_hist = mean_or_zero(
        [
            x["stats"]["down_pct"]
            if direction == "DOWN"
            else x["stats"]["up_pct"]
            for x in selected
        ]
    )

    avg_similarity = mean_or_zero(
        [
            x["stats"]["avg_similarity"]
            for x in selected
        ]
    )

    avg_analogues = mean_or_zero(
        [
            x["stats"]["total"]
            for x in selected
        ]
    )

    # Strong confirmation.
    if (
        agreement >= 0.66
        and avg_hist >= 60
        and avg_similarity >= 0.76
        and avg_analogues >= MIN_ANALOGUES
    ):
        status = "SIGNAL"

    elif (
        avg_hist >= 55
        and avg_analogues >= 50
    ):
        status = "CONFIRMATION"

    else:
        status = "WATCH"

    return {
        "status": status,
        "direction": direction,
        "agreement": agreement,
        "avg_hist": avg_hist,
        "avg_similarity": avg_similarity,
        "avg_analogues": avg_analogues,
        "reason": (
            f"{count}/{len(ready)} timeframe бир томонга мос"
        ),
    }


# ============================================================
# MARKET SCANNER
# ============================================================

def quick_market_state(candles):

    if len(candles) < 20:
        return "UNKNOWN"

    context = build_context(
        candles
    )

    if not context:
        return "UNKNOWN"

    return price_action_bias(
        context
    )


def market_scanner():

    result = {}

    with DATA_LOCK:

        for symbol in SYMBOLS:

            result[symbol] = {}

            for tf in TIMEFRAMES:

                candles = MARKET_DATA[
                    symbol
                ][tf]

                if candles:
                    result[symbol][tf] = (
                        quick_market_state(
                            candles
                        )
                    )
                else:
                    result[symbol][tf] = "UNKNOWN"

    return result


def market_breadth(scanner):

    values = []

    for symbol_data in scanner.values():
        # Use 15m if available,
        # otherwise first timeframe.
        state = symbol_data.get(
            "15m"
        )

        if state is None:
            state = next(
                iter(
                    symbol_data.values()
                ),
                "UNKNOWN",
            )

        if state in (
            "UP",
            "DOWN",
            "FLAT",
        ):
            values.append(state)

    total = len(values)

    if total == 0:
        return {
            "up": 0,
            "down": 0,
            "flat": 0,
        }

    return {
        "up": sum(
            x == "UP"
            for x in values
        ) / total * 100,
        "down": sum(
            x == "DOWN"
            for x in values
        ) / total * 100,
        "flat": sum(
            x == "FLAT"
            for x in values
        ) / total * 100,
    }


# ============================================================
# NEWS / EVENT ENGINE
# ============================================================

def strip_xml(text):
    if not text:
        return ""

    return (
        text.replace(
            "<![CDATA[",
            "",
        )
        .replace(
            "]]>",
            "",
        )
        .strip()
    )


def parse_rss(url):

    try:

        r = session.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={
                "User-Agent":
                "Mozilla/5.0 DeepMarketBot"
            },
        )

        if r.status_code != 200:
            return []

        root = ET.fromstring(
            r.content
        )

        items = []

        for item in root.iter():

            if (
                item.tag.lower().endswith(
                    "item"
                )
            ):

                title = ""
                link = ""
                published = ""

                for child in list(item):

                    tag = child.tag.lower()

                    if tag.endswith("title"):
                        title = strip_xml(
                            child.text
                        )

                    elif tag.endswith("link"):
                        link = (
                            child.text
                            or ""
                        ).strip()

                    elif (
                        tag.endswith(
                            "pubdate"
                        )
                        or tag.endswith(
                            "published"
                        )
                        or tag.endswith(
                            "updated"
                        )
                    ):
                        published = (
                            child.text
                            or ""
                        ).strip()

                if title:
                    items.append(
                        NewsItem(
                            title=title,
                            link=link,
                            published=published,
                            source=url,
                        )
                    )

        return items[:30]

    except Exception as exc:
        log.warning(
            "RSS error %s: %s",
            url,
            exc,
        )
        return []


def refresh_news():

    global NEWS_CACHE
    global LAST_NEWS_SCAN

    all_items = []

    for feed in NEWS_FEEDS:
        all_items.extend(
            parse_rss(feed)
        )

    # Deduplicate.
    seen = set()
    final = []

    for item in all_items:

        key = (
            item.title.lower().strip()
        )

        if key in seen:
            continue

        seen.add(key)
        final.append(item)

    NEWS_CACHE = final[:100]

    LAST_NEWS_SCAN = time.time()

    log.info(
        "News cache updated: %s items",
        len(NEWS_CACHE),
    )


def relevant_news(symbol):

    if not NEWS_CACHE:
        return []

    base = symbol.replace(
        "USDT",
        "",
    ).upper()

    keywords = {
        base,
        "BITCOIN"
        if base == "BTC"
        else "",
        "ETHEREUM"
        if base == "ETH"
        else "",
        "CRYPTO",
        "BINANCE",
        "FED",
        "ETF",
        "SEC",
        "RATE",
        "INFLATION",
        "CPI",
        "FOMC",
        "REGULATION",
    }

    keywords = {
        x.lower()
        for x in keywords
        if x
    }

    result = []

    for item in NEWS_CACHE:

        text = item.title.lower()

        if any(
            k in text
            for k in keywords
        ):
            result.append(item)

    return result[:5]


# ============================================================
# CHART ENGINE
# ============================================================

def make_chart(
    symbol,
    timeframe,
    candles,
    analysis,
    path,
):

    if not candles:
        return None

    data = candles[-80:]

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    x = np.arange(
        len(data)
    )

    for i, c in enumerate(data):

        if c.close >= c.open:
            color = "green"
        else:
            color = "red"

        ax.plot(
            [i, i],
            [c.low, c.high],
            color=color,
            linewidth=1,
        )

        ax.plot(
            [i, i],
            [c.open, c.close],
            color=color,
            linewidth=5,
        )

    ctx = analysis.get(
        "context",
        {},
    )

    title = (
        f"{symbol} | {timeframe} | "
        f"{ctx.get('structure_state', '')}"
    )

    ax.set_title(title)

    ax.set_xlabel(
        "Ёпилган свечалар"
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
            dpi=140,
        )
        plt.close(fig)
        return path
    except Exception:
        plt.close(fig)
        return None


# ============================================================
# TELEGRAM
# ============================================================

def telegram_api(
    method,
    data=None,
    files=None,
):

    if not TELEGRAM_TOKEN:
        return None

    url = (
        "https://api.telegram.org/bot"
        + TELEGRAM_TOKEN
        + "/"
        + method
    )

    try:

        r = requests.post(
            url,
            data=data,
            files=files,
            timeout=30,
        )

        if r.status_code != 200:
            log.warning(
                "Telegram %s: %s",
                r.status_code,
                r.text[:500],
            )

        return r

    except Exception as exc:
        log.warning(
            "Telegram error: %s",
            exc,
        )

    return None


def telegram_send_text(text):

    if (
        not TELEGRAM_ENABLED
        or not TELEGRAM_TOKEN
        or not TELEGRAM_CHAT_ID
    ):
        return

    # Telegram message limit.
    chunks = [
        text[i:i+3900]
        for i in range(
            0,
            len(text),
            3900,
        )
    ]

    for chunk in chunks:

        telegram_api(
            "sendMessage",
            data={
                "chat_id":
                TELEGRAM_CHAT_ID,
                "text": chunk,
                "disable_web_page_preview":
                "true",
            },
        )


def telegram_send_photo(
    photo_path,
    caption,
):

    if (
        not TELEGRAM_ENABLED
        or not TELEGRAM_TOKEN
        or not TELEGRAM_CHAT_ID
    ):
        return

    try:

        with open(
            photo_path,
            "rb",
        ) as f:

            telegram_api(
                "sendPhoto",
                data={
                    "chat_id":
                    TELEGRAM_CHAT_ID,
                    "caption":
                    caption[:1000],
                },
                files={
                    "photo": f
                },
            )

    except Exception as exc:
        log.warning(
            "Telegram photo error: %s",
            exc,
        )


# ============================================================
# REPORT BUILDER
# ============================================================

def direction_icon(direction):

    if direction == "UP":
        return "🟢"

    if direction == "DOWN":
        return "🔴"

    return "⚪"


def tf_line(result):

    if not result.get("ready"):
        return (
            f"{result.get('timeframe')}: "
            f"⚪ маълумот етарли эмас"
        )

    return (
        f"{result['timeframe']}: "
        f"{direction_icon(result['combined'])} "
        f"{result['combined']} | "
        f"PA={result['pa_bias']} | "
        f"HIST={result['hist_bias']}"
    )


def build_report(
    symbol,
    mtf,
    gate,
    scanner=None,
    news=None,
):

    primary = None

    # Prefer 15m.
    if "15m" in mtf:
        primary = mtf["15m"]

    if primary is None:
        primary = next(
            iter(mtf.values()),
            None,
        )

    if not primary:
        return None

    ctx = primary.get(
        "context",
        {},
    )

    stats = primary.get(
        "stats",
        {},
    )

    patterns = ctx.get(
        "patterns",
        [],
    )

    if not patterns:
        patterns = [
            "Аниқ свеча pattern топилмади"
        ]

    status = gate.get(
        "status",
        "WATCH",
    )

    direction = gate.get(
        "direction",
        "FLAT",
    )

    if status == "SIGNAL":
        status_icon = "🚨"
    elif status == "CONFIRMATION":
        status_icon = "🟠"
    else:
        status_icon = "🟡"

    lines = []

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        f"🧠 {APP_NAME}"
    )

    lines.append(
        f"🪙 {symbol}"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        f"💰 Нарх: {fmt_price(primary['price'])}"
    )

    lines.append(
        f"📊 ҲОЛАТ: {ctx.get('structure_state', 'Нейтрал')}"
    )

    lines.append(
        f"⚡ Импульс: {ctx.get('impulse_direction')} "
        f"| {ctx.get('impulse_strength', 0):.2f}"
    )

    lines.append(
        f"📦 Compression: "
        f"{ctx.get('compression', 0) * 100:.1f}%"
    )

    lines.append(
        f"💥 Breakout: "
        f"{ctx.get('breakout_direction', 0)}"
    )

    lines.append(
        f"↩️ Rejection: "
        f"{ctx.get('rejection_direction', 0)}"
    )

    lines.append(
        f"⚠️ Exhaustion: "
        f"{ctx.get('exhaustion_direction', 0)}"
    )

    lines.append(
        f"🕯 Patterns: "
        f"{', '.join(patterns)}"
    )

    if ctx.get("mw"):
        lines.append(
            f"🔺 Pattern: {ctx['mw']}"
        )

    lines.append("")
    lines.append(
        "━━━━━━━━ ТАРИХИЙ АНАЛОГ ━━━━━━━━"
    )

    lines.append(
        f"🔎 Ўхшаш ҳолатлар: "
        f"{stats.get('total', 0)}"
    )

    lines.append(
        f"🎯 Ўртача ўхшашлик: "
        f"{stats.get('avg_similarity', 0) * 100:.1f}%"
    )

    lines.append(
        f"🔴 DOWN: "
        f"{stats.get('down', 0)} "
        f"({stats.get('down_pct', 0):.1f}%)"
    )

    lines.append(
        f"🟢 UP: "
        f"{stats.get('up', 0)} "
        f"({stats.get('up_pct', 0):.1f}%)"
    )

    lines.append(
        f"⚪ FLAT: "
        f"{stats.get('flat', 0)} "
        f"({stats.get('flat_pct', 0):.1f}%)"
    )

    lines.append("")

    targets = stats.get(
        "targets",
        {},
    )

    for level in [
        -1,
        -3,
        -5,
        -10,
        -15,
    ]:

        t = targets.get(
            level,
            {},
        )

        lines.append(
            f"📉 {level}%: "
            f"{t.get('count', 0)} / "
            f"{stats.get('total', 0)} = "
            f"{t.get('pct', 0):.1f}%"
        )

    lines.append(
        f"📈 Медиана close move: "
        f"{stats.get('median_move', 0):+.2f}%"
    )

    lines.append("")
    lines.append(
        "━━━━━━━━ TIMEFRAME ━━━━━━━━"
    )

    for tf in TIMEFRAMES:
        if tf in mtf:
            lines.append(
                tf_line(
                    mtf[tf]
                )
            )

    lines.append("")
    lines.append(
        "━━━━━━━━ СЦЕНАРИЙ ━━━━━━━━"
    )

    lines.append(
        f"{status_icon} {status}"
    )

    lines.append(
        f"Йўналиш: "
        f"{direction}"
    )

    lines.append(
        f"Timeframe agreement: "
        f"{gate.get('agreement', 0) * 100:.1f}%"
    )

    lines.append(
        f"Historical evidence: "
        f"{gate.get('avg_hist', 0):.1f}%"
    )

    lines.append(
        f"Historical analogues: "
        f"{gate.get('avg_analogues', 0):.0f}"
    )

    lines.append(
        f"Сабаб: {gate.get('reason', '')}"
    )

    if scanner:
        breadth = market_breadth(
            scanner
        )

        lines.append("")
        lines.append(
            "━━━━━━━━ MARKET BREADTH ━━━━━━━━"
        )

        lines.append(
            f"🟢 UP: {breadth['up']:.1f}%"
        )

        lines.append(
            f"🔴 DOWN: {breadth['down']:.1f}%"
        )

        lines.append(
            f"⚪ FLAT: {breadth['flat']:.1f}%"
        )

    if news:

        lines.append("")
        lines.append(
            "━━━━━━━━ 📰 NEWS / EVENT ━━━━━━━━"
        )

        for item in news[:3]:

            title = item.title.strip()

            if len(title) > 220:
                title = title[:217] + "..."

            lines.append(
                f"• {title}"
            )

            if item.published:
                lines.append(
                    f"  Вақт: {item.published}"
                )

    else:

        lines.append("")
        lines.append(
            "📰 News/Event: "
            "релевант хабар топилмади"
        )

    lines.append("")
    lines.append(
        "━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "ℹ️ Бу тарихий статистик далил."
    )

    lines.append(
        "❗ 100% кафолатланган прогноз эмас."
    )

    lines.append(
        "ℹ️ Бот автоматик order очмайди."
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━"
    )

    return "\n".join(lines)


# ============================================================
# SIGNAL DEDUPLICATION
# ============================================================

def should_send_signal(
    symbol,
    gate,
):

    key = (
        symbol,
        ",".join(TIMEFRAMES),
    )

    previous = LAST_SIGNAL.get(
        key
    )

    now = time.time()

    if previous:
        elapsed = (
            now - previous["time"]
        )

        same_direction = (
            previous["direction"]
            == gate["direction"]
        )

        if (
            elapsed
            < SIGNAL_COOLDOWN_MINUTES * 60
            and same_direction
            and gate["status"]
            != "SIGNAL"
        ):
            return False

    # WATCH isn't sent repeatedly.
    if gate["status"] == "WATCH":
        return False

    LAST_SIGNAL[key] = {
        "time": now,
        "direction":
            gate["direction"],
        "status":
            gate["status"],
    }

    return True


# ============================================================
# COMPLETE SYMBOL ANALYSIS
# ============================================================

def analyze_symbol(
    symbol,
    send_telegram=True,
):

    mtf = multi_timeframe_analysis(
        symbol
    )

    gate = signal_gate(
        symbol,
        mtf,
    )

    scanner = None

    if MARKET_SCANNER_ENABLED:
        scanner = market_scanner()

    news = []

    if NEWS_ENABLED:
        news = relevant_news(
            symbol
        )

    report = build_report(
        symbol,
        mtf,
        gate,
        scanner,
        news,
    )

    if not report:
        return

    log.info(
        "%s | %s | %s",
        symbol,
        gate["status"],
        gate["direction"],
    )

    # Telegram only when meaningful.
    if (
        send_telegram
        and should_send_signal(
            symbol,
            gate,
        )
    ):

        telegram_send_text(
            report
        )

        if CHART_ENABLED:

            primary = mtf.get(
                "15m"
            )

            if primary and primary.get(
                "ready"
            ):

                with DATA_LOCK:
                    candles = list(
                        MARKET_DATA[
                            symbol
                        ]["15m"]
                    )

                filename = (
                    "/tmp/"
                    + symbol
                    + "_"
                    + str(
                        int(
                            time.time()
                        )
                    )
                    + ".png"
                )

                chart = make_chart(
                    symbol,
                    "15m",
                    candles,
                    primary,
                    filename,
                )

                if chart:
                    telegram_send_photo(
                        chart,
                        (
                            f"{symbol} | "
                            f"{gate['status']} | "
                            f"{gate['direction']}"
                        ),
                    )


# ============================================================
# INITIAL HISTORY LOAD
# ============================================================

def history_days_for(tf):

    if tf == "5m":
        return HISTORY_DAYS_5M

    if tf == "15m":
        return HISTORY_DAYS_15M

    if tf == "1h":
        return HISTORY_DAYS_1H

    # Generic fallback.
    return 365


def load_initial_history():

    log.info(
        "Loading Binance historical data..."
    )

    for symbol in SYMBOLS:

        for tf in TIMEFRAMES:

            if STOP:
                return

            try:

                log.info(
                    "Loading %s %s",
                    symbol,
                    tf,
                )

                candles = fetch_klines(
                    symbol,
                    tf,
                    history_days_for(
                        tf
                    ),
                )

                with DATA_LOCK:
                    MARKET_DATA[
                        symbol
                    ][tf] = candles

                log.info(
                    "%s %s: %s candles",
                    symbol,
                    tf,
                    len(candles),
                )

                if len(candles) >= 120:

                    # Build historical index once.
                    HISTORICAL.build_index(
                        symbol,
                        tf,
                        candles,
                    )

            except Exception:
                log.exception(
                    "History load failed %s %s",
                    symbol,
                    tf,
                )

    log.info(
        "Historical loading complete."
    )


# ============================================================
# WEBSOCKET
# ============================================================

def websocket_streams():

    streams = []

    for symbol in SYMBOLS:

        s = symbol.lower()

        for tf in TIMEFRAMES:

            streams.append(
                f"{s}@kline_{tf}"
            )

    return streams


def process_ws_message(message):

    try:

        payload = json.loads(
            message
        )

        data = payload.get(
            "data",
            payload,
        )

        if data.get("e") != "kline":
            return

        k = data.get("k", {})

        symbol = (
            k.get("s", "")
            .upper()
        )

        interval = k.get(
            "i",
            ""
        )

        # x == True means closed candle.
        closed = bool(
            k.get("x", False)
        )

        if not symbol or not interval:
            return

        candle = Candle(
            open_time=int(
                k["t"]
            ),
            open=float(
                k["o"]
            ),
            high=float(
                k["h"]
            ),
            low=float(
                k["l"]
            ),
            close=float(
                k["c"]
            ),
            volume=float(
                k["v"]
            ),
            close_time=int(
                k["T"]
            ),
        )

        with DATA_LOCK:

            candles = MARKET_DATA[
                symbol
            ][interval]

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
                candles.append(
                    candle
                )

            # Keep memory bounded.
            if len(candles) > MAX_CANDLES_MEMORY:
                del candles[
                    :-MAX_CANDLES_MEMORY
                ]

        if closed:

            log.info(
                "CLOSED %s %s %s",
                symbol,
                interval,
                fmt_price(
                    candle.close
                ),
            )

            # Only analyze on candle close.
            threading.Thread(
                target=analyze_symbol,
                args=(symbol, True),
                daemon=True,
            ).start()

    except Exception:
        log.exception(
            "WS message processing error"
        )


def websocket_worker():

    reconnect_delay = 3

    while not STOP:

        ws = None

        try:

            streams = websocket_streams()

            if not streams:
                raise RuntimeError(
                    "No websocket streams"
                )

            url = (
                BINANCE_WS
                + "?streams="
                + "/".join(streams)
            )

            log.info(
                "Connecting Binance WebSocket..."
            )

            ws = websocket.WebSocketApp(
                url,
                on_open=lambda w:
                    log.info(
                        "Binance WebSocket connected"
                    ),
                on_message=lambda w, msg:
                    process_ws_message(
                        msg
                    ),
                on_error=lambda w, err:
                    log.warning(
                        "WebSocket error: %s",
                        err,
                    ),
                on_close=lambda w, code, msg:
                    log.warning(
                        "WebSocket closed: %s %s",
                        code,
                        msg,
                    ),
            )

            # Binance requires long-lived connection
            # management and reconnection.
            ws.run_forever(
                ping_interval=120,
                ping_timeout=30,
                reconnect=5,
            )

        except Exception:
            log.exception(
                "WebSocket worker crashed"
            )

        finally:

            try:
                if ws:
                    ws.close()
            except Exception:
                pass

        if STOP:
            break

        log.info(
            "Reconnecting Binance WebSocket in %ss",
            reconnect_delay,
        )

        sleep_interruptible(
            reconnect_delay
        )

        reconnect_delay = min(
            reconnect_delay * 2,
            60,
        )


# ============================================================
# BACKGROUND RESCAN
# ============================================================

def analysis_loop():

    while not STOP:

        try:

            for symbol in SYMBOLS:

                if STOP:
                    break

                analyze_symbol(
                    symbol,
                    send_telegram=True,
                )

                sleep_interruptible(
                    2
                )

        except Exception:
            log.exception(
                "Analysis loop error"
            )

        sleep_interruptible(
            SCAN_SECONDS
        )


def news_loop():

    while not STOP:

        try:

            if NEWS_ENABLED:
                refresh_news()

        except Exception:
            log.exception(
                "News loop error"
            )

        sleep_interruptible(
            NEWS_SCAN_SECONDS
        )


def market_loop():

    while not STOP:

        try:

            scanner = market_scanner()

            breadth = market_breadth(
                scanner
            )

            log.info(
                "MARKET | UP %.1f%% | DOWN %.1f%% | FLAT %.1f%%",
                breadth["up"],
                breadth["down"],
                breadth["flat"],
            )

        except Exception:
            log.exception(
                "Market scanner error"
            )

        sleep_interruptible(
            MARKET_SCAN_SECONDS
        )


# ============================================================
# TELEGRAM STARTUP
# ============================================================

def startup_message():

    if not TELEGRAM_ENABLED:
        return

    if not TELEGRAM_TOKEN:
        return

    text = f"""
🧠 {APP_NAME}

✅ Сервер ишга тушди.

Версия:
{VERSION}

🪙 Symbols:
{", ".join(SYMBOLS)}

⏱ Timeframes:
{", ".join(TIMEFRAMES)}

🔎 Historical Analogue:
ON

📊 Market Scanner:
{"ON" if MARKET_SCANNER_ENABLED else "OFF"}

📰 News/Event:
{"ON" if NEWS_ENABLED else "OFF"}

📈 Chart:
{"ON" if CHART_ENABLED else "OFF"}

🟢 Binance WebSocket:
ON

🔁 Auto Reconnect:
ON

🚫 Автоматик order:
OFF

Бот Price Action + тарихи аналоглар
асосида мониторинг бошлади.
"""

    telegram_send_text(
        text.strip()
    )


# ============================================================
# HEALTH CHECK HTTP SERVER
# ============================================================
#
# Railway web-service health / port detection.
# ============================================================

from http.server import (
    BaseHTTPRequestHandler,
    HTTPServer,
)


PORT = int(
    os.getenv(
        "PORT",
        "8080",
    )
)


class HealthHandler(
    BaseHTTPRequestHandler
):

    def do_GET(self):

        body = {
            "status": "ok",
            "app": APP_NAME,
            "version": VERSION,
            "uptime_seconds":
                int(
                    time.time()
                    - SESSION_START
                ),
            "symbols": SYMBOLS,
            "timeframes":
                TIMEFRAMES,
            "historical_indexes":
                len(
                    HISTORICAL.cache
                ),
        }

        payload = json.dumps(
            body,
            ensure_ascii=False,
        ).encode(
            "utf-8"
        )

        self.send_response(
            200
        )

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )

        self.send_header(
            "Content-Length",
            str(len(payload)),
        )

        self.end_headers()

        self.wfile.write(
            payload
        )

    def log_message(
        self,
        format,
        *args,
    ):
        return


def health_server():

    server = HTTPServer(
        ("0.0.0.0", PORT),
        HealthHandler,
    )

    log.info(
        "Health server listening on %s",
        PORT,
    )

    while not STOP:

        server.handle_request()


# ============================================================
# SHUTDOWN
# ============================================================

def shutdown(
    signum=None,
    frame=None,
):

    global STOP

    if STOP:
        return

    STOP = True

    log.info(
        "Shutdown requested."
    )


signal.signal(
    signal.SIGINT,
    shutdown,
)

signal.signal(
    signal.SIGTERM,
    shutdown,
)


# ============================================================
# MAIN
# ============================================================

def main():

    log.info(
        "======================================"
    )

    log.info(
        "%s",
        APP_NAME,
    )

    log.info(
        "Version %s",
        VERSION,
    )

    log.info(
        "Symbols: %s",
        SYMBOLS,
    )

    log.info(
        "Timeframes: %s",
        TIMEFRAMES,
    )

    log.info(
        "======================================"
    )

    # Initial news.
    if NEWS_ENABLED:
        try:
            refresh_news()
        except Exception:
            log.exception(
                "Initial news scan failed"
            )

    # Historical Binance data.
    load_initial_history()

    # Telegram startup.
    startup_message()

    # Health endpoint.
    threading.Thread(
        target=health_server,
        daemon=True,
    ).start()

    # Binance real-time stream.
    threading.Thread(
        target=websocket_worker,
        daemon=True,
    ).start()

    # Historical/periodic analysis.
    threading.Thread(
        target=analysis_loop,
        daemon=True,
    ).start()

    # News.
    threading.Thread(
        target=news_loop,
        daemon=True,
    ).start()

    # Market scanner.
    threading.Thread(
        target=market_loop,
        daemon=True,
    ).start()

    log.info(
        "BOT IS LIVE."
    )

    while not STOP:
        time.sleep(2)

    log.info(
        "BOT STOPPED."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        shutdown()
    except Exception:
        log.exception(
            "FATAL ERROR"
        )
        raise
PY

echo
echo "=========================================="
echo " DEEP MARKET BOT CREATED"
echo "=========================================="
echo
echo "Folder:"
pwd
echo
echo "Files:"
ls -lh
echo
echo "Next:"
echo "1) GitHub repositoryga deep-market-bot papkasini yuklang."
echo "2) Railway -> New Project -> Deploy from GitHub Repo."
echo "3) Railway Variables ga TELEGRAM_BOT_TOKEN va TELEGRAM_CHAT_ID kiriting."
echo "4) Deploy qiling."
echo