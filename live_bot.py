# ============================================================
# DEEP HISTORICAL MARKET BOT
# Payton-style deployment: GitHub -> Railway -> Binance -> Telegram
#
# Тил:
#   - Telegram ҳисоботлари: ЎЗБЕК КИРИЛЛ
#   - Логлар: ЎЗБЕК КИРИЛЛ
#   - Техник environment номлари: стандарт English
#
# Бу бот индикаторларга таянмайди.
# Асосий манбалар:
#   1. OHLC шамлар
#   2. Candle anatomy
#   3. Price Action
#   4. Market structure
#   5. Historical analogues
#   6. Forward historical outcomes
#   7. Multi-timeframe context
#
# ============================================================

import os
import sys
import json
import math
import time
import signal
import asyncio
import hashlib
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from collections import Counter, deque
from typing import Optional, List, Dict, Tuple

import aiohttp
import numpy as np


# ============================================================
# SOZLAMALAR
# ============================================================

BINANCE_REST = os.getenv(
    "BINANCE_REST",
    "https://api.binance.com"
)

BINANCE_WS = os.getenv(
    "BINANCE_WS",
    "wss://stream.binance.com:9443"
)

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN",
    os.getenv("TELEGRAM_TOKEN", "")
)

TELEGRAM_CHAT_ID = os.getenv(
    "TELEGRAM_CHAT_ID",
    ""
)

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


# История
DAYS_5M = int(os.getenv("DAYS_5M", "365"))
DAYS_15M = int(os.getenv("DAYS_15M", "730"))
DAYS_1H = int(os.getenv("DAYS_1H", "1825"))

HISTORY_LIMIT = int(
    os.getenv("HISTORY_LIMIT", "1000")
)

CONTEXT_CANDLES = int(
    os.getenv("CONTEXT_CANDLES", "30")
)

FORWARD_CANDLES = int(
    os.getenv("FORWARD_CANDLES", "50")
)

LIVE_BUFFER = int(
    os.getenv("LIVE_BUFFER", "2000")
)


# Аналоглар
MIN_ANALOGUES = int(
    os.getenv("MIN_ANALOGUES", "30")
)

MIN_INDEPENDENT_ANALOGUES = int(
    os.getenv("MIN_INDEPENDENT_ANALOGUES", "15")
)

MIN_SIMILARITY = float(
    os.getenv("MIN_SIMILARITY", "0.68")
)

TOP_ANALOGUES = int(
    os.getenv("TOP_ANALOGUES", "10")
)

ANALOGUE_SEPARATION = int(
    os.getenv(
        "ANALOGUE_SEPARATION",
        str(CONTEXT_CANDLES + FORWARD_CANDLES)
    )
)


# Сигнал фильтри
SIGNAL_MIN_ANALOGUES = int(
    os.getenv("SIGNAL_MIN_ANALOGUES", "60")
)

SIGNAL_MIN_INDEPENDENT = int(
    os.getenv("SIGNAL_MIN_INDEPENDENT", "30")
)

SIGNAL_MIN_SIMILARITY = float(
    os.getenv("SIGNAL_MIN_SIMILARITY", "0.76")
)

SIGNAL_DIRECTION_MIN = float(
    os.getenv("SIGNAL_DIRECTION_MIN", "0.60")
)

SIGNAL_MOVE_MIN = float(
    os.getenv("SIGNAL_MOVE_MIN", "0.03")
)


# Live режим
RESCAN_SECONDS = int(
    os.getenv("RESCAN_SECONDS", "3600")
)

SIGNAL_COOLDOWN_SECONDS = int(
    os.getenv("SIGNAL_COOLDOWN_SECONDS", "1800")
)

REQUEST_DELAY = float(
    os.getenv("REQUEST_DELAY", "0.12")
)

MAX_RETRIES = int(
    os.getenv("MAX_RETRIES", "6")
)


# Telegram
SEND_STATE_UPDATES = os.getenv(
    "SEND_STATE_UPDATES",
    "true"
).lower() == "true"

SEND_WATCH_UPDATES = os.getenv(
    "SEND_WATCH_UPDATES",
    "true"
).lower() == "true"


# Fayllar
RESULT_FILE = os.getenv(
    "RESULT_FILE",
    "deep_market_results.json"
)

DATA_DIR = os.getenv(
    "DATA_DIR",
    "data"
)


# ============================================================
# YORDAMCHI FUNKSIYALAR
# ============================================================

def now_utc() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def pct(value):
    return f"{value * 100:+.2f}%"


def log(message: str):
    print(
        f"[{now_utc()}] {message}",
        flush=True
    )


def timeframe_seconds(tf: str) -> int:
    unit = tf[-1]
    number = int(tf[:-1])

    if unit == "m":
        return number * 60

    if unit == "h":
        return number * 3600

    if unit == "d":
        return number * 86400

    raise ValueError(
        f"Noma'lum timeframe: {tf}"
    )


def history_days(tf: str) -> int:
    if tf == "5m":
        return DAYS_5M

    if tf == "15m":
        return DAYS_15M

    if tf == "1h":
        return DAYS_1H

    return 365


# ============================================================
# CANDLE
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

    quote_volume: float = 0.0
    trades: int = 0

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
        return self.high - max(
            self.open,
            self.close
        )

    @property
    def lower_wick(self):
        return min(
            self.open,
            self.close
        ) - self.low

    @property
    def direction(self):
        if self.close > self.open:
            return 1

        if self.close < self.open:
            return -1

        return 0


# ============================================================
# HISTORICAL ANALOGUE
# ============================================================

@dataclass
class Analogue:
    index: int
    similarity: float

    state: str
    structure: str

    final_return: float
    max_up: float
    max_down: float

    first_direction: str

    hit_1_down: bool
    hit_3_down: bool
    hit_5_down: bool
    hit_10_down: bool
    hit_15_down: bool

    hit_1_up: bool
    hit_3_up: bool
    hit_5_up: bool
    hit_10_up: bool
    hit_15_up: bool


# ============================================================
# BINANCE HTTP
# ============================================================

class BinanceClient:

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None

    async def start(self):

        if self.session is None:
            timeout = aiohttp.ClientTimeout(
                total=30
            )

            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "User-Agent":
                        "Deep-Historical-Market-Bot/1.0"
                }
            )

    async def close(self):

        if self.session:
            await self.session.close()

        self.session = None

    async def request(
        self,
        path: str,
        params: dict
    ):

        await self.start()

        url = (
            BINANCE_REST.rstrip("/")
            + path
        )

        last_error = None

        for attempt in range(MAX_RETRIES):

            try:

                async with self.session.get(
                    url,
                    params=params
                ) as response:

                    text = await response.text()

                    if response.status == 200:
                        return json.loads(text)

                    last_error = (
                        f"HTTP {response.status}: "
                        f"{text[:500]}"
                    )

            except Exception as exc:

                last_error = str(exc)

            await asyncio.sleep(
                min(
                    2 ** attempt,
                    10
                )
            )

        raise RuntimeError(
            f"Binance so'rovi muvaffaqiyatsiz: "
            f"{last_error}"
        )

    async def klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000
    ):

        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(
                max(limit, 1),
                1000
            )
        }

        if start_time is not None:
            params["startTime"] = start_time

        if end_time is not None:
            params["endTime"] = end_time

        return await self.request(
            "/api/v3/klines",
            params
        )


# ============================================================
# KLINE PARSER
# ============================================================

def parse_kline(row) -> Candle:

    return Candle(
        open_time=int(row[0]),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]),
        close_time=int(row[6]),
        quote_volume=float(row[7]),
        trades=int(row[8])
    )


# ============================================================
# TARIX YUKLASH
# ============================================================

async def load_history(
    client: BinanceClient,
    symbol: str,
    timeframe: str
) -> List[Candle]:

    days = history_days(timeframe)

    end_ms = int(
        time.time() * 1000
    )

    start_ms = end_ms - (
        days * 86400 * 1000
    )

    candles = []

    cursor = start_ms

    log(
        f"📥 {symbol} {timeframe}: "
        f"{days} kunlik tarix yuklanmoqda..."
    )

    while cursor < end_ms:

        rows = await client.klines(
            symbol=symbol,
            interval=timeframe,
            start_time=cursor,
            end_time=end_ms,
            limit=HISTORY_LIMIT
        )

        if not rows:
            break

        batch = [
            parse_kline(row)
            for row in rows
        ]

        candles.extend(batch)

        last_close = batch[-1].close_time

        next_cursor = (
            last_close + 1
        )

        if next_cursor <= cursor:
            break

        cursor = next_cursor

        await asyncio.sleep(
            REQUEST_DELAY
        )

        if len(rows) < HISTORY_LIMIT:
            break

    # Duplicate va tartib
    unique = {}

    for candle in candles:
        unique[candle.open_time] = candle

    candles = sorted(
        unique.values(),
        key=lambda x: x.open_time
    )

    # Faqat yopilgan shamlar
    current_ms = int(
        time.time() * 1000
    )

    candles = [
        c for c in candles
        if c.close_time <= current_ms
    ]

    log(
        f"✅ {symbol} {timeframe}: "
        f"{len(candles):,} ta yopilgan sham olindi."
    )

    return candles


# ============================================================
# PRICE ACTION
# ============================================================

def candle_body_ratio(c: Candle):
    return c.body / c.range


def upper_wick_ratio(c: Candle):
    return c.upper_wick / c.range


def lower_wick_ratio(c: Candle):
    return c.lower_wick / c.range


def candle_code(c: Candle) -> str:

    if c.range <= 0:
        return "DOJI"

    body = candle_body_ratio(c)
    upper = upper_wick_ratio(c)
    lower = lower_wick_ratio(c)

    if body <= 0.10:
        body_type = "DOJI"

    elif body <= 0.35:
        body_type = "SMALL"

    elif body <= 0.65:
        body_type = "MEDIUM"

    else:
        body_type = "LARGE"

    if upper >= 0.55:
        wick = "UPPER_REJECTION"

    elif lower >= 0.55:
        wick = "LOWER_REJECTION"

    elif upper >= 0.30 and lower >= 0.30:
        wick = "BOTH_WICKS"

    else:
        wick = "NORMAL"

    direction = (
        "BULL"
        if c.direction > 0
        else
        "BEAR"
        if c.direction < 0
        else
        "FLAT"
    )

    return (
        f"{direction}_"
        f"{body_type}_"
        f"{wick}"
    )


def is_pin_bar(c: Candle):

    body = max(
        c.body,
        1e-12
    )

    return (
        (
            c.lower_wick >= body * 2.5
            and
            c.upper_wick <= body * 1.2
        )
        or
        (
            c.upper_wick >= body * 2.5
            and
            c.lower_wick <= body * 1.2
        )
    )


def is_inside_bar(
    candles: List[Candle],
    i: int
):

    if i < 1:
        return False

    a = candles[i - 1]
    b = candles[i]

    return (
        b.high <= a.high
        and
        b.low >= a.low
    )


def is_outside_bar(
    candles: List[Candle],
    i: int
):

    if i < 1:
        return False

    a = candles[i - 1]
    b = candles[i]

    return (
        b.high >= a.high
        and
        b.low <= a.low
    )


def is_bull_engulfing(
    candles: List[Candle],
    i: int
):

    if i < 1:
        return False

    a = candles[i - 1]
    b = candles[i]

    return (
        a.direction < 0
        and
        b.direction > 0
        and
        b.open <= a.close
        and
        b.close >= a.open
    )


def is_bear_engulfing(
    candles: List[Candle],
    i: int
):

    if i < 1:
        return False

    a = candles[i - 1]
    b = candles[i]

    return (
        a.direction > 0
        and
        b.direction < 0
        and
        b.open >= a.close
        and
        b.close <= a.open
    )


# ============================================================
# ATR-SIMILAR NORMALIZATSIYA
# ============================================================

def local_range(
    candles: List[Candle],
    i: int,
    n: int = 20
):

    start = max(
        0,
        i - n + 1
    )

    values = [
        c.range
        for c in candles[start:i + 1]
    ]

    if not values:
        return 0.0

    return float(
        np.median(values)
    )


def normalized_move(
    candles: List[Candle],
    i: int,
    n: int = 20
):

    if i < 1:
        return 0.0

    base = max(
        local_range(
            candles,
            i,
            n
        ),
        1e-12
    )

    move = (
        candles[i].close
        -
        candles[i - 1].close
    )

    return move / base


# ============================================================
# IMPULSE / COMPRESSION
# ============================================================

def impulse_strength(
    candles: List[Candle],
    i: int,
    lookback: int = 10
):

    if i < lookback:
        return 0.0

    a = candles[
        i - lookback
    ].close

    b = candles[i].close

    move = (
        b - a
    ) / max(
        abs(a),
        1e-12
    )

    avg_range = np.mean([
        c.range
        for c in candles[
            i - lookback + 1:i + 1
        ]
    ])

    if avg_range <= 0:
        return 0.0

    return move * (
        abs(b - a)
        /
        avg_range
    )


def compression_score(
    candles: List[Candle],
    i: int,
    lookback: int = 12
):

    if i < lookback:
        return 0.0

    ranges = np.array([
        c.range
        for c in candles[
            i - lookback + 1:i + 1
        ]
    ])

    if len(ranges) < 3:
        return 0.0

    recent = np.mean(
        ranges[-4:]
    )

    old = np.mean(
        ranges[:4]
    )

    if old <= 0:
        return 0.0

    return clamp(
        1.0 - recent / old
    )


# ============================================================
# SWING STRUCTURE
# ============================================================

def detect_swings(
    candles: List[Candle],
    left: int = 2,
    right: int = 2
):

    highs = []
    lows = []

    for i in range(
        left,
        len(candles) - right
    ):

        high = candles[i].high
        low = candles[i].low

        left_highs = [
            candles[j].high
            for j in range(
                i - left,
                i
            )
        ]

        right_highs = [
            candles[j].high
            for j in range(
                i + 1,
                i + right + 1
            )
        ]

        left_lows = [
            candles[j].low
            for j in range(
                i - left,
                i
            )
        ]

        right_lows = [
            candles[j].low
            for j in range(
                i + 1,
                i + right + 1
            )
        ]

        if (
            high > max(left_highs)
            and
            high > max(right_highs)
        ):
            highs.append(i)

        if (
            low < min(left_lows)
            and
            low < min(right_lows)
        ):
            lows.append(i)

    return highs, lows


def structure_state(
    candles: List[Candle],
    i: int
):

    if i < 15:
        return "ARALASH"

    subset = candles[
        max(0, i - 80):i + 1
    ]

    highs, lows = detect_swings(
        subset
    )

    if len(highs) < 2 or len(lows) < 2:
        return "ARALASH"

    h1 = subset[
        highs[-2]
    ].high

    h2 = subset[
        highs[-1]
    ].high

    l1 = subset[
        lows[-2]
    ].low

    l2 = subset[
        lows[-1]
    ].low

    if h2 > h1 and l2 > l1:
        return "HH_HL_UP"

    if h2 < h1 and l2 < l1:
        return "LH_LL_DOWN"

    if h2 > h1 and l2 < l1:
        return "KENGAYUVCHI"

    return "ARALASH"


# ============================================================
# M / W
# ============================================================

def detect_m_w(
    candles: List[Candle],
    i: int
):

    if i < 12:
        return "YO'Q"

    data = candles[
        i - 12:i + 1
    ]

    highs = np.array([
        c.high
        for c in data
    ])

    lows = np.array([
        c.low
        for c in data
    ])

    high_peaks = []

    low_peaks = []

    for j in range(
        2,
        len(data) - 2
    ):

        if (
            highs[j]
            >
            highs[j - 1]
            and
            highs[j]
            >
            highs[j + 1]
        ):
            high_peaks.append(j)

        if (
            lows[j]
            <
            lows[j - 1]
            and
            lows[j]
            <
            lows[j + 1]
        ):
            low_peaks.append(j)

    if len(high_peaks) >= 2:

        a = highs[
            high_peaks[-2]
        ]

        b = highs[
            high_peaks[-1]
        ]

        tolerance = (
            max(a, b)
            * 0.003
        )

        if abs(a - b) <= tolerance:
            return "M"

    if len(low_peaks) >= 2:

        a = lows[
            low_peaks[-2]
        ]

        b = lows[
            low_peaks[-1]
        ]

        tolerance = (
            max(a, b)
            * 0.003
        )

        if abs(a - b) <= tolerance:
            return "W"

    return "YO'Q"


# ============================================================
# BREAKOUT / FALSE BREAKOUT / REJECTION
# ============================================================

def market_state(
    candles: List[Candle],
    i: int
):

    if i < 30:
        return "MA'LUMOT YETARLI EMAS"

    current = candles[i]

    previous = candles[
        i - 20:i
    ]

    high20 = max(
        c.high
        for c in previous
    )

    low20 = min(
        c.low
        for c in previous
    )

    close = current.close

    structure = structure_state(
        candles,
        i
    )

    impulse = impulse_strength(
        candles,
        i
    )

    compression = compression_score(
        candles,
        i
    )

    upper_rejection = (
        current.upper_wick
        /
        current.range
    )

    lower_rejection = (
        current.lower_wick
        /
        current.range
    )

    broke_high = (
        current.high > high20
    )

    broke_low = (
        current.low < low20
    )

    false_up = (
        broke_high
        and
        close < high20
    )

    false_down = (
        broke_low
        and
        close > low20
    )

    if false_up:
        return "YUQORIGA SOXTA TEShISH"

    if false_down:
        return "PASTGA SOXTA TEShISH"

    if broke_high and close > high20:
        return "YUQORIGA TEShISH"

    if broke_low and close < low20:
        return "PASTGA TEShISH"

    if (
        impulse > 3.0
        and
        upper_rejection > 0.45
    ):
        return "KUCHLI O'SISH + RAD ETILISH"

    if (
        impulse < -3.0
        and
        lower_rejection > 0.45
    ):
        return "KUCHLI TUSHISH + RAD ETILISH"

    if compression > 0.35:
        return "SIQILISH"

    if structure == "HH_HL_UP":
        if impulse > 1.0:
            return "KUCHLI O'SISH IMPULSI"

        return "O'SISH TUZILMASI"

    if structure == "LH_LL_DOWN":
        if impulse < -1.0:
            return "KUCHLI TUSHISH IMPULSI"

        return "TUSHISH TUZILMASI"

    if abs(impulse) > 2.5:
        if impulse > 0:
            return "KUCHLI O'SISH IMPULSI"

        return "KUCHLI TUSHISH IMPULSI"

    # Range / flat
    recent = [
        c.close
        for c in candles[
            i - 20:i + 1
        ]
    ]

    if recent:

        max_price = max(recent)
        min_price = min(recent)

        width = (
            max_price - min_price
        ) / max(
            current.close,
            1e-12
        )

        if width < 0.015:
            return "FLAT / RANGE"

    return "ARALASH"


# ============================================================
# KONTEXT VEKTORI
# ============================================================

def context_vector(
    candles: List[Candle],
    i: int
):

    start = max(
        0,
        i - CONTEXT_CANDLES + 1
    )

    seq = candles[
        start:i + 1
    ]

    if len(seq) < 5:
        return None

    closes = np.array([
        c.close
        for c in seq
    ])

    opens = np.array([
        c.open
        for c in seq
    ])

    highs = np.array([
        c.high
        for c in seq
    ])

    lows = np.array([
        c.low
        for c in seq
    ])

    ranges = np.maximum(
        highs - lows,
        1e-12
    )

    bodies = (
        np.abs(closes - opens)
        /
        ranges
    )

    upper = (
        highs
        -
        np.maximum(
            opens,
            closes
        )
    ) / ranges

    lower = (
        np.minimum(
            opens,
            closes
        )
        -
        lows
    ) / ranges

    directions = np.sign(
        closes - opens
    )

    returns = np.diff(
        closes
    ) / np.maximum(
        closes[:-1],
        1e-12
    )

    total_move = (
        closes[-1]
        /
        closes[0]
        - 1.0
    )

    positive_ratio = np.mean(
        directions > 0
    )

    negative_ratio = np.mean(
        directions < 0
    )

    efficiency = (
        abs(
            closes[-1]
            -
            closes[0]
        )
        /
        max(
            np.sum(
                np.abs(
                    np.diff(closes)
                )
            ),
            1e-12
        )
    )

    range_position = (
        closes[-1]
        -
        np.min(lows)
    ) / max(
        np.max(highs)
        -
        np.min(lows),
        1e-12
    )

    last = seq[-1]

    features = [
        total_move,
        efficiency,
        positive_ratio,
        negative_ratio,

        float(
            np.mean(bodies)
        ),

        float(
            np.mean(upper)
        ),

        float(
            np.mean(lower)
        ),

        float(
            bodies[-1]
        ),

        float(
            upper[-1]
        ),

        float(
            lower[-1]
        ),

        float(
            directions[-1]
        ),

        float(
            np.mean(ranges[-5:])
            /
            max(
                np.mean(ranges),
                1e-12
            )
        ),

        float(range_position),

        float(
            impulse_strength(
                candles,
                i
            )
        ),

        float(
            compression_score(
                candles,
                i
            )
        ),

        float(
            normalized_move(
                candles,
                i
            )
        ),

        float(
            int(
                is_pin_bar(last)
            )
        ),

        float(
            int(
                is_inside_bar(
                    candles,
                    i
                )
            )
        ),

        float(
            int(
                is_outside_bar(
                    candles,
                    i
                )
            )
        ),

        float(
            int(
                is_bull_engulfing(
                    candles,
                    i
                )
            )
        ),

        float(
            int(
                is_bear_engulfing(
                    candles,
                    i
                )
            )
        ),

        float(
            int(
                detect_m_w(
                    candles,
                    i
                ) == "M"
            )
        ),

        float(
            int(
                detect_m_w(
                    candles,
                    i
                ) == "W"
            )
        )
    ]

    return np.array(
        features,
        dtype=np.float64
    )


# ============================================================
# VEKTOR O'XSHASHLIGI
# ============================================================

def vector_similarity(
    a,
    b
):

    if a is None or b is None:
        return 0.0

    if len(a) != len(b):
        return 0.0

    a = np.asarray(a)
    b = np.asarray(b)

    scale = np.maximum(
        np.abs(a),
        np.abs(b)
    )

    scale = np.maximum(
        scale,
        1e-6
    )

    distance = np.mean(
        np.abs(a - b)
        /
        scale
    )

    similarity = math.exp(
        -2.5 * distance
    )

    return clamp(
        similarity
    )


# ============================================================
# FORWARD NATIJA
# ============================================================

def forward_outcome(
    candles: List[Candle],
    i: int
):

    current = candles[i]

    end = min(
        len(candles),
        i + 1 + FORWARD_CANDLES
    )

    future = candles[
        i + 1:end
    ]

    if not future:
        return None

    current_price = current.close

    future_closes = np.array([
        c.close
        for c in future
    ])

    returns = (
        future_closes
        /
        current_price
        - 1.0
    )

    max_up = float(
        np.max(returns)
    )

    max_down = float(
        np.min(returns)
    )

    final_return = float(
        returns[-1]
    )

    first_direction = "FLAT"

    for value in returns:

        if value >= 0.003:
            first_direction = "UP"
            break

        if value <= -0.003:
            first_direction = "DOWN"
            break

    def reached_down(level):
        return bool(
            np.min(returns)
            <= -level
        )

    def reached_up(level):
        return bool(
            np.max(returns)
            >= level
        )

    return {
        "final_return": final_return,

        "max_up": max_up,

        "max_down": max_down,

        "first_direction":
            first_direction,

        "hit_1_down":
            reached_down(0.01),

        "hit_3_down":
            reached_down(0.03),

        "hit_5_down":
            reached_down(0.05),

        "hit_10_down":
            reached_down(0.10),

        "hit_15_down":
            reached_down(0.15),

        "hit_1_up":
            reached_up(0.01),

        "hit_3_up":
            reached_up(0.03),

        "hit_5_up":
            reached_up(0.05),

        "hit_10_up":
            reached_up(0.10),

        "hit_15_up":
            reached_up(0.15)
    }


# ============================================================
# TARIXIY INDEKS
# ============================================================

class HistoricalIndex:

    def __init__(
        self,
        candles: List[Candle]
    ):

        self.candles = candles

        self.vectors = []

        self.states = []

        self.structures = []

        self.valid_indices = []

    def build(self):

        total = len(
            self.candles
        )

        start = (
            CONTEXT_CANDLES
            + 5
        )

        end = (
            total
            -
            FORWARD_CANDLES
            -
            1
        )

        log(
            f"🧠 Тарихий индекс қурилмоқда: "
            f"{max(0, end - start):,} та ҳолат..."
        )

        for i in range(
            start,
            max(start, end)
        ):

            vector = context_vector(
                self.candles,
                i
            )

            if vector is None:
                continue

            self.vectors.append(
                vector
            )

            self.states.append(
                market_state(
                    self.candles,
                    i
                )
            )

            self.structures.append(
                structure_state(
                    self.candles,
                    i
                )
            )

            self.valid_indices.append(
                i
            )

        log(
            f"✅ Тарихий индекс тайёр: "
            f"{len(self.valid_indices):,} та ҳолат."
        )


# ============================================================
# ANALOGUE ENGINE
# ============================================================

class AnalogueEngine:

    def __init__(
        self,
        candles: List[Candle],
        index: HistoricalIndex
    ):

        self.candles = candles
        self.index = index

    def find(
        self,
        current_index: int
    ):

        current_vector = context_vector(
            self.candles,
            current_index
        )

        if current_vector is None:
            return []

        current_state = market_state(
            self.candles,
            current_index
        )

        candidates = []

        for pos, historical_index in enumerate(
            self.index.valid_indices
        ):

            # Ҳозирги ҳолатга жуда яқин даврни
            # тарихий аналог сифатида ишлатмаймиз.
            if abs(
                historical_index
                -
                current_index
            ) < ANALOGUE_SEPARATION:

                continue

            historical_vector = (
                self.index.vectors[pos]
            )

            similarity = vector_similarity(
                current_vector,
                historical_vector
            )

            if similarity < MIN_SIMILARITY:
                continue

            outcome = forward_outcome(
                self.candles,
                historical_index
            )

            if outcome is None:
                continue

            candidates.append(
                Analogue(
                    index=historical_index,
                    similarity=similarity,

                    state=self.index.states[pos],

                    structure=
                        self.index.structures[pos],

                    final_return=
                        outcome[
                            "final_return"
                        ],

                    max_up=
                        outcome["max_up"],

                    max_down=
                        outcome["max_down"],

                    first_direction=
                        outcome[
                            "first_direction"
                        ],

                    hit_1_down=
                        outcome[
                            "hit_1_down"
                        ],

                    hit_3_down=
                        outcome[
                            "hit_3_down"
                        ],

                    hit_5_down=
                        outcome[
                            "hit_5_down"
                        ],

                    hit_10_down=
                        outcome[
                            "hit_10_down"
                        ],

                    hit_15_down=
                        outcome[
                            "hit_15_down"
                        ],

                    hit_1_up=
                        outcome[
                            "hit_1_up"
                        ],

                    hit_3_up=
                        outcome[
                            "hit_3_up"
                        ],

                    hit_5_up=
                        outcome[
                            "hit_5_up"
                        ],

                    hit_10_up=
                        outcome[
                            "hit_10_up"
                        ],

                    hit_15_up=
                        outcome[
                            "hit_15_up"
                        ]
                )
            )

        candidates.sort(
            key=lambda x:
                x.similarity,
            reverse=True
        )

        # Бир хил вақт зонасидаги
        # такрорий аналогларни қисқартамиз.
        selected = []

        for candidate in candidates:

            too_close = False

            for old in selected:

                if abs(
                    candidate.index
                    -
                    old.index
                ) < ANALOGUE_SEPARATION:

                    too_close = True
                    break

            if too_close:
                continue

            selected.append(
                candidate
            )

        return selected


# ============================================================
# STATISTIKA
# ============================================================

def build_statistics(
    analogues: List[Analogue]
):

    total = len(
        analogues
    )

    if total == 0:
        return {
            "total": 0
        }

    up = sum(
        1
        for a in analogues
        if a.first_direction == "UP"
    )

    down = sum(
        1
        for a in analogues
        if a.first_direction == "DOWN"
    )

    flat = total - up - down

    final_returns = [
        a.final_return
        for a in analogues
    ]

    max_ups = [
        a.max_up
        for a in analogues
    ]

    max_downs = [
        a.max_down
        for a in analogues
    ]

    def rate(n):
        return (
            n / total
            if total
            else 0.0
        )

    def threshold_down(name):
        return sum(
            1
            for a in analogues
            if getattr(a, name)
        )

    def threshold_up(name):
        return sum(
            1
            for a in analogues
            if getattr(a, name)
        )

    return {

        "total": total,

        "up": up,
        "down": down,
        "flat": flat,

        "up_rate": rate(up),
        "down_rate": rate(down),
        "flat_rate": rate(flat),

        "median_final":
            float(
                np.median(
                    final_returns
                )
            ),

        "median_max_up":
            float(
                np.median(
                    max_ups
                )
            ),

        "median_max_down":
            float(
                np.median(
                    max_downs
                )
            ),

        "mean_final":
            float(
                np.mean(
                    final_returns
                )
            ),

        "down_1":
            threshold_down(
                "hit_1_down"
            ),

        "down_3":
            threshold_down(
                "hit_3_down"
            ),

        "down_5":
            threshold_down(
                "hit_5_down"
            ),

        "down_10":
            threshold_down(
                "hit_10_down"
            ),

        "down_15":
            threshold_down(
                "hit_15_down"
            ),

        "up_1":
            threshold_up(
                "hit_1_up"
            ),

        "up_3":
            threshold_up(
                "hit_3_up"
            ),

        "up_5":
            threshold_up(
                "hit_5_up"
            ),

        "up_10":
            threshold_up(
                "hit_10_up"
            ),

        "up_15":
            threshold_up(
                "hit_15_up"
            )
    }


# ============================================================
# SCENARIY
# ============================================================

def determine_scenario(
    candles: List[Candle],
    i: int,
    statistics: dict
):

    state = market_state(
        candles,
        i
    )

    structure = structure_state(
        candles,
        i
    )

    m_w = detect_m_w(
        candles,
        i
    )

    down_rate = statistics.get(
        "down_rate",
        0
    )

    up_rate = statistics.get(
        "up_rate",
        0
    )

    median_final = statistics.get(
        "median_final",
        0
    )

    # Тарихий статистика йўналиши
    if (
        down_rate >= 0.60
        and
        abs(median_final)
        >= SIGNAL_MOVE_MIN
    ):
        historical_bias = "SHORT"

    elif (
        up_rate >= 0.60
        and
        abs(median_final)
        >= SIGNAL_MOVE_MIN
    ):
        historical_bias = "LONG"

    else:
        historical_bias = "NO_CLEAR_BIAS"

    # Жорий Price Action
    if "YUQORIGA" in state:
        current_bias = "LONG"

    elif "PASTGA" in state:
        current_bias = "SHORT"

    elif "O'SISH" in state:
        current_bias = "LONG"

    elif "TUSHISH" in state:
        current_bias = "SHORT"

    elif structure == "HH_HL_UP":
        current_bias = "LONG"

    elif structure == "LH_LL_DOWN":
        current_bias = "SHORT"

    else:
        current_bias = "NEUTRAL"

    return {
        "state": state,
        "structure": structure,
        "m_w": m_w,
        "historical_bias":
            historical_bias,
        "current_bias":
            current_bias
    }


# ============================================================
# SIGNAL GATE
# ============================================================

def signal_gate(
    analogues: List[Analogue],
    statistics: dict
):

    total = len(
        analogues
    )

    if total < SIGNAL_MIN_ANALOGUES:
        return {
            "status": "WATCH",
            "direction": None,
            "reason":
                f"Тарихий аналоглар {total} та. "
                f"Камида {SIGNAL_MIN_ANALOGUES} та керак."
        }

    independent = min(
        total,
        max(
            0,
            total // 2
        )
    )

    if independent < SIGNAL_MIN_INDEPENDENT:

        return {
            "status": "WATCH",
            "direction": None,
            "reason":
                "Мустақил тарихий ҳолатлар "
                "ҳали етарли эмас."
        }

    best_similarity = max(
        a.similarity
        for a in analogues
    )

    if (
        best_similarity
        <
        SIGNAL_MIN_SIMILARITY
    ):

        return {
            "status": "WATCH",
            "direction": None,
            "reason":
                "Энг яхши тарихий ўхшашлик "
                "сигнал чегарасидан паст."
        }

    up_rate = statistics.get(
        "up_rate",
        0
    )

    down_rate = statistics.get(
        "down_rate",
        0
    )

    median_final = statistics.get(
        "median_final",
        0
    )

    if (
        down_rate
        >= SIGNAL_DIRECTION_MIN
        and
        median_final
        <= -SIGNAL_MOVE_MIN
    ):

        return {
            "status": "SHORT_WATCH",
            "direction": "SHORT",
            "reason":
                "Тарихий аналогларда пастга "
                "йўналиш устун."
        }

    if (
        up_rate
        >= SIGNAL_DIRECTION_MIN
        and
        median_final
        >= SIGNAL_MOVE_MIN
    ):

        return {
            "status": "LONG_WATCH",
            "direction": "LONG",
            "reason":
                "Тарихий аналогларда юқорига "
                "йўналиш устун."
        }

    return {
        "status": "WATCH",
        "direction": None,
        "reason":
            "Тарихий маълумотда аниқ "
            "йўналиш устунлиги йўқ."
    }


# ============================================================
# TELEGRAM MATN
# ============================================================

def make_report(
    symbol: str,
    timeframe: str,
    candles: List[Candle],
    current_index: int,
    analogues: List[Analogue]
):

    current = candles[
        current_index
    ]

    state = market_state(
        candles,
        current_index
    )

    structure = structure_state(
        candles,
        current_index
    )

    m_w = detect_m_w(
        candles,
        current_index
    )

    statistics = build_statistics(
        analogues
    )

    scenario = determine_scenario(
        candles,
        current_index,
        statistics
    )

    gate = signal_gate(
        analogues,
        statistics
    )

    total = statistics.get(
        "total",
        0
    )

    def percent_count(
        count
    ):

        if total == 0:
            return "0.0%"

        return (
            f"{count / total * 100:.1f}%"
        )

    lines = []

    lines.append(
        "🔥 CHUQUR PRICE ACTION TAHLILI"
    )

    lines.append("")
    lines.append(
        f"💰 {symbol}"
    )

    lines.append(
        f"⏱ TF: {timeframe}"
    )

    lines.append(
        f"💵 Narx: {current.close:.8f}"
    )

    lines.append(
        f"🕯 Sham: {candle_code(current)}"
    )

    lines.append("")

    lines.append(
        "━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "📊 HOZIRGI BOZOR HOLATI"
    )

    lines.append(
        f"Holat: {state}"
    )

    lines.append(
        f"Tuzilma: {structure}"
    )

    lines.append(
        f"M/W: {m_w}"
    )

    lines.append(
        f"Pin Bar: "
        f"{'HA' if is_pin_bar(current) else 'YO‘Q'}"
    )

    lines.append(
        f"Inside Bar: "
        f"{'HA' if is_inside_bar(candles, current_index) else 'YO‘Q'}"
    )

    lines.append(
        f"Outside Bar: "
        f"{'HA' if is_outside_bar(candles, current_index) else 'YO‘Q'}"
    )

    lines.append("")

    lines.append(
        "━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🧠 TARIXIY ANALOGIYALAR"
    )

    lines.append(
        f"Topilgan holatlar: {total}"
    )

    if total:

        lines.append(
            f"🟢 Yuqoriga: "
            f"{statistics['up']} "
            f"({statistics['up_rate'] * 100:.1f}%)"
        )

        lines.append(
            f"🔴 Pastga: "
            f"{statistics['down']} "
            f"({statistics['down_rate'] * 100:.1f}%)"
        )

        lines.append(
            f"⚪ Tekis: "
            f"{statistics['flat']} "
            f"({statistics['flat_rate'] * 100:.1f}%)"
        )

        lines.append("")

        lines.append(
            "📉 TARIXIY HARAKAT"
        )

        lines.append(
            f"-1%: "
            f"{statistics['down_1']} "
            f"({percent_count(statistics['down_1'])})"
        )

        lines.append(
            f"-3%: "
            f"{statistics['down_3']} "
            f"({percent_count(statistics['down_3'])})"
        )

        lines.append(
            f"-5%: "
            f"{statistics['down_5']} "
            f"({percent_count(statistics['down_5'])})"
        )

        lines.append(
            f"-10%: "
            f"{statistics['down_10']} "
            f"({percent_count(statistics['down_10'])})"
        )

        lines.append(
            f"-15%: "
            f"{statistics['down_15']} "
            f"({percent_count(statistics['down_15'])})"
        )

        lines.append("")

        lines.append(
            "📈 YUQORIGA HARAKAT"
        )

        lines.append(
            f"+1%: "
            f"{statistics['up_1']} "
            f"({percent_count(statistics['up_1'])})"
        )

        lines.append(
            f"+3%: "
            f"{statistics['up_3']} "
            f"({percent_count(statistics['up_3'])})"
        )

        lines.append(
            f"+5%: "
            f"{statistics['up_5']} "
            f"({percent_count(statistics['up_5'])})"
        )

        lines.append(
            f"+10%: "
            f"{statistics['up_10']} "
            f"({percent_count(statistics['up_10'])})"
        )

        lines.append(
            f"+15%: "
            f"{statistics['up_15']} "
            f"({percent_count(statistics['up_15'])})"
        )

        lines.append("")

        lines.append(
            "📌 Медиана кейинги ҳаракат: "
            f"{pct(statistics['median_final'])}"
        )

        lines.append(
            "📌 Максимал юқори ҳаракат медианаси: "
            f"{pct(statistics['median_max_up'])}"
        )

        lines.append(
            "📌 Максимал паст ҳаракат медианаси: "
            f"{pct(statistics['median_max_down'])}"
        )

    lines.append("")

    lines.append(
        "━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🎯 СЦЕНАРИЙ"
    )

    lines.append(
        f"Ҳозирги Price Action: "
        f"{scenario['current_bias']}"
    )

    lines.append(
        f"Тарихий йўналиш: "
        f"{scenario['historical_bias']}"
    )

    lines.append(
        f"Тарихий M/W: "
        f"{scenario['m_w']}"
    )

    lines.append("")

    lines.append(
        "━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🚦 SIGNAL HOLATI"
    )

    if gate["status"] == "LONG_WATCH":

        lines.append(
            "🟢 LONG КУЗАТУВ"
        )

    elif gate["status"] == "SHORT_WATCH":

        lines.append(
            "🔴 SHORT КУЗАТУВ"
        )

    else:

        lines.append(
            "⚠️ ҲОЗИРЧА КИРИШ ЙЎҚ"
        )

    lines.append(
        f"Сабаб: {gate['reason']}"
    )

    lines.append("")

    lines.append(
        "❗ Бу натижа тарихий статистика."
    )

    lines.append(
        "❗ Ҳеч қандай ҳаракат кафолатланмайди."
    )

    lines.append(
        "❗ Киришдан олдин янги ёпилган шам "
        "тасдиғи керак."
    )

    if analogues:

        lines.append("")

        lines.append(
            "━━━━━━━━━━━━━━━━━━"
        )

        lines.append(
            "🔎 ЭНГ ЎХШАШ ТАРИХИЙ ҲОЛАТЛАР"
        )

        for n, a in enumerate(
            analogues[:TOP_ANALOGUES],
            1
        ):

            dt = datetime.fromtimestamp(
                candles[a.index].open_time / 1000,
                timezone.utc
            ).strftime(
                "%Y-%m-%d %H:%M"
            )

            lines.append(
                f"{n}. {dt} | "
                f"ўхшашлик "
                f"{a.similarity * 100:.1f}% | "
                f"{a.first_direction} | "
                f"кейин "
                f"{pct(a.final_return)}"
            )

    return "\n".join(lines)


# ============================================================
# TELEGRAM
# ============================================================

class Telegram:

    def __init__(self):
        self.session = None

    async def start(self):

        if self.session is None:

            self.session = (
                aiohttp.ClientSession()
            )

    async def close(self):

        if self.session:
            await self.session.close()

        self.session = None

    async def send(
        self,
        text: str
    ):

        if not TELEGRAM_BOT_TOKEN:
            log(
                "⚠️ TELEGRAM_BOT_TOKEN берилмаган."
            )
            return False

        if not TELEGRAM_CHAT_ID:
            log(
                "⚠️ TELEGRAM_CHAT_ID берилмаган."
            )
            return False

        await self.start()

        url = (
            "https://api.telegram.org/bot"
            f"{TELEGRAM_BOT_TOKEN}"
            "/sendMessage"
        )

        payload = {
            "chat_id":
                TELEGRAM_CHAT_ID,

            "text":
                text,

            "disable_web_page_preview":
                True
        }

        try:

            async with self.session.post(
                url,
                json=payload
            ) as response:

                if response.status == 200:
                    return True

                body = await response.text()

                log(
                    f"❌ Telegram xatosi: "
                    f"{response.status} "
                    f"{body[:300]}"
                )

        except Exception as exc:

            log(
                f"❌ Telegram ulanish xatosi: "
                f"{exc}"
            )

        return False


# ============================================================
# BOT
# ============================================================

class DeepMarketBot:

    def __init__(self):

        self.binance = BinanceClient()

        self.telegram = Telegram()

        self.histories: Dict[
            str,
            Dict[str, List[Candle]]
        ] = {}

        self.indices: Dict[
            str,
            Dict[str, HistoricalIndex]
        ] = {}

        self.engines: Dict[
            str,
            Dict[str, AnalogueEngine]
        ] = {}

        self.last_analysis = {}

        self.last_signal_time = {}

        self.last_state = {}

        self.running = True

    # --------------------------------------------------------

    async def build_history(self):

        for symbol in SYMBOLS:

            self.histories[symbol] = {}

            self.indices[symbol] = {}

            self.engines[symbol] = {}

            for timeframe in TIMEFRAMES:

                try:

                    candles = await load_history(
                        self.binance,
                        symbol,
                        timeframe
                    )

                    if len(candles) < (
                        CONTEXT_CANDLES
                        +
                        FORWARD_CANDLES
                        +
                        50
                    ):

                        log(
                            f"⚠️ {symbol} {timeframe}: "
                            "тарих етарли эмас."
                        )

                        continue

                    self.histories[
                        symbol
                    ][
                        timeframe
                    ] = candles

                    index = HistoricalIndex(
                        candles
                    )

                    index.build()

                    self.indices[
                        symbol
                    ][
                        timeframe
                    ] = index

                    self.engines[
                        symbol
                    ][
                        timeframe
                    ] = AnalogueEngine(
                        candles,
                        index
                    )

                except Exception as exc:

                    log(
                        f"❌ {symbol} {timeframe} "
                        f"тарихий маълумот хато: "
                        f"{exc}"
                    )

    # --------------------------------------------------------

    async def analyse(
        self,
        symbol: str,
        timeframe: str,
        send_telegram=True
    ):

        candles = self.histories.get(
            symbol,
            {}
        ).get(
            timeframe
        )

        engine = self.engines.get(
            symbol,
            {}
        ).get(
            timeframe
        )

        if not candles or not engine:
            return None

        current_index = (
            len(candles) - 1
        )

        if current_index < (
            CONTEXT_CANDLES + 5
        ):
            return None

        analogues = engine.find(
            current_index
        )

        report = make_report(
            symbol,
            timeframe,
            candles,
            current_index,
            analogues
        )

        state = market_state(
            candles,
            current_index
        )

        key = (
            f"{symbol}:{timeframe}"
        )

        previous_state = (
            self.last_state.get(key)
        )

        changed = (
            previous_state
            != state
        )

        self.last_state[key] = state

        if (
            send_telegram
            and
            changed
            and
            SEND_STATE_UPDATES
        ):

            await self.telegram.send(
                report
            )

        self.last_analysis[
            key
        ] = {
            "time":
                now_utc(),

            "state":
                state,

            "analogue_count":
                len(analogues)
        }

        return {
            "symbol":
                symbol,

            "timeframe":
                timeframe,

            "state":
                state,

            "analogues":
                len(analogues),

            "report":
                report
        }

    # --------------------------------------------------------

    async def analyse_all(
        self,
        send_telegram=True
    ):

        results = []

        for symbol in SYMBOLS:

            for timeframe in TIMEFRAMES:

                result = await self.analyse(
                    symbol,
                    timeframe,
                    send_telegram
                )

                if result:
                    results.append(
                        result
                    )

        return results

    # --------------------------------------------------------

    async def refresh_timeframe(
        self,
        symbol: str,
        timeframe: str
    ):

        candles = self.histories.get(
            symbol,
            {}
        ).get(
            timeframe
        )

        if not candles:
            return

        try:

            rows = await self.binance.klines(
                symbol=symbol,
                interval=timeframe,
                limit=3
            )

            if not rows:
                return

            for row in rows:

                candle = parse_kline(
                    row
                )

                # Фақат ёпилган шам
                now_ms = int(
                    time.time() * 1000
                )

                if candle.close_time > now_ms:
                    continue

                if (
                    candles
                    and
                    candle.open_time
                    ==
                    candles[-1].open_time
                ):

                    candles[-1] = candle

                elif (
                    not candles
                    or
                    candle.open_time
                    >
                    candles[-1].open_time
                ):

                    candles.append(
                        candle
                    )

            if len(candles) > LIVE_BUFFER:

                # Индекс билан аралашиб кетмаслиги учун
                # live тарихни ҳозирча тўлиқ сақлаймиз.
                #
                # Railway Volume / RAM режимида кейин
                # архивлаштиришни қўшиш мумкин.

                pass

        except Exception as exc:

            log(
                f"⚠️ {symbol} {timeframe} "
                f"янгилаш хато: {exc}"
            )

    # --------------------------------------------------------

    async def polling_loop(self):

        log(
            "🔄 Live кузатув бошланди."
        )

        while self.running:

            try:

                for symbol in SYMBOLS:

                    for timeframe in TIMEFRAMES:

                        await self.refresh_timeframe(
                            symbol,
                            timeframe
                        )

                await self.analyse_all(
                    send_telegram=True
                )

            except Exception as exc:

                log(
                    f"❌ Live цикл хато: {exc}"
                )

            await asyncio.sleep(
                RESCAN_SECONDS
            )

    # --------------------------------------------------------

    async def websocket_loop(self):

        """
        Binance WebSocket'дан реал вақт шамларини қабул қилиш.

        Муҳим:
        фақат kline 'x == true' бўлганда
        шам тўлиқ ёпилган деб ҳисоблаймиз.
        """

        streams = []

        for symbol in SYMBOLS:

            for timeframe in TIMEFRAMES:

                streams.append(
                    f"{symbol.lower()}@kline_{timeframe}"
                )

        if not streams:
            return

        stream_url = (
            BINANCE_WS.rstrip("/")
            +
            "/stream?streams="
            +
            "/".join(streams)
        )

        reconnect_delay = 5

        while self.running:

            try:

                log(
                    "🔌 Binance WebSocket "
                    "уланмоқда..."
                )

                timeout = aiohttp.ClientTimeout(
                    total=None
                )

                async with aiohttp.ClientSession(
                    timeout=timeout
                ) as session:

                    async with session.ws_connect(
                        stream_url,
                        heartbeat=30
                    ) as ws:

                        log(
                            "🟢 Binance WebSocket "
                            "уланди."
                        )

                        reconnect_delay = 5

                        async for message in ws:

                            if not self.running:
                                break

                            if message.type != aiohttp.WSMsgType.TEXT:
                                continue

                            try:
                                packet = json.loads(
                                    message.data
                                )
                            except Exception:
                                continue

                            data = packet.get(
                                "data",
                                {}
                            )

                            k = data.get(
                                "k",
                                {}
                            )

                            # Шам ҳали ёпилмаган бўлса,
                            # таҳлил қилмаймиз.
                            if not k.get(
                                "x",
                                False
                            ):
                                continue

                            symbol = k.get(
                                "s"
                            )

                            timeframe = k.get(
                                "i"
                            )

                            if (
                                not symbol
                                or
                                not timeframe
                            ):
                                continue

                            candles = (
                                self.histories
                                .get(
                                    symbol,
                                    {}
                                )
                                .get(
                                    timeframe
                                )
                            )

                            if candles is None:
                                continue

                            candle = Candle(
                                open_time=int(
                                    k["t"]
                                ),

                                close_time=int(
                                    k["T"]
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

                                quote_volume=float(
                                    k.get(
                                        "q",
                                        0
                                    )
                                ),

                                trades=int(
                                    k.get(
                                        "n",
                                        0
                                    )
                                )
                            )

                            if (
                                candles
                                and
                                candle.open_time
                                ==
                                candles[-1].open_time
                            ):

                                candles[-1] = candle

                            elif (
                                not candles
                                or
                                candle.open_time
                                >
                                candles[-1].open_time
                            ):

                                candles.append(
                                    candle
                                )

                            log(
                                f"🕯 {symbol} "
                                f"{timeframe} "
                                f"ёпилди: "
                                f"{candle.close}"
                            )

                            # Live index'ни янги candle билан
                            # қайта қуриш шарт эмас.
                            #
                            # Ҳозирги таҳлил:
                            # тарихий индекс +
                            # янги жорий ҳолат.
                            #
                            # Кейинги версияда incremental
                            # index қўшилади.

                            result = await self.analyse(
                                symbol,
                                timeframe,
                                send_telegram=True
                            )

                            if result:

                                log(
                                    f"📊 {symbol} "
                                    f"{timeframe}: "
                                    f"{result['state']} | "
                                    f"аналоглар: "
                                    f"{result['analogues']}"
                                )

            except Exception as exc:

                log(
                    f"🔴 WebSocket узилди: "
                    f"{exc}"
                )

                if self.running:

                    log(
                        f"🔄 {reconnect_delay} "
                        "секундан кейин қайта уланамиз..."
                    )

                    await asyncio.sleep(
                        reconnect_delay
                    )

                    reconnect_delay = min(
                        reconnect_delay * 2,
                        60
                    )

    # --------------------------------------------------------

    async def periodic_rescan(self):

        while self.running:

            await asyncio.sleep(
                RESCAN_SECONDS
            )

            try:

                log(
                    "🔍 Чуқур қайта скан бошланди..."
                )

                await self.analyse_all(
                    send_telegram=False
                )

                log(
                    "✅ Қайта скан тугади."
                )

            except Exception as exc:

                log(
                    f"❌ Қайта скан хато: "
                    f"{exc}"
                )

    # --------------------------------------------------------

    async def save_state(self):

        os.makedirs(
            DATA_DIR,
            exist_ok=True
        )

        data = {
            "time":
                now_utc(),

            "symbols":
                SYMBOLS,

            "timeframes":
                TIMEFRAMES,

            "last_analysis":
                self.last_analysis,

            "last_state":
                self.last_state
        }

        path = os.path.join(
            DATA_DIR,
            RESULT_FILE
        )

        try:

            with open(
                path,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    data,
                    f,
                    ensure_ascii=False,
                    indent=2
                )

        except Exception as exc:

            log(
                f"⚠️ Ҳолатни сақлаш хато: "
                f"{exc}"
            )

    # --------------------------------------------------------

    async def run(self):

        log(
            "================================================"
        )

        log(
            "🚀 DEEP HISTORICAL MARKET BOT"
        )

        log(
            "🚀 Бот ишга тушмоқда..."
        )

        log(
            f"Символлар: {', '.join(SYMBOLS)}"
        )

        log(
            f"Timeframe: {', '.join(TIMEFRAMES)}"
        )

        log(
            f"Контекст: {CONTEXT_CANDLES} шам"
        )

        log(
            f"Кейинги ҳаракат: "
            f"{FORWARD_CANDLES} шам"
        )

        log(
            "================================================"
        )

        await self.build_history()

        if not self.histories:

            raise RuntimeError(
                "Ҳеч қандай тарихий маълумот "
                "тайёрланмади."
            )

        # Бошланғич таҳлил
        try:

            results = await self.analyse_all(
                send_telegram=True
            )

            log(
                f"📊 Бошланғич таҳлил: "
                f"{len(results)} та."
            )

        except Exception as exc:

            log(
                f"⚠️ Бошланғич таҳлил хато: "
                f"{exc}"
            )

        await self.save_state()

        # WebSocket + periodic rescan
        tasks = [
            asyncio.create_task(
                self.websocket_loop()
            ),

            asyncio.create_task(
                self.periodic_rescan()
            )
        ]

        try:

            await asyncio.gather(
                *tasks
            )

        finally:

            for task in tasks:
                task.cancel()

            await self.telegram.close()

            await self.binance.close()

            await self.save_state()

    # --------------------------------------------------------

    def stop(self):

        if self.running:

            log(
                "🛑 Ботни тўхтатиш сигнали олинди."
            )

        self.running = False


# ============================================================
# MAIN
# ============================================================

BOT = None


def handle_signal(
    signum,
    frame
):

    global BOT

    if BOT:

        BOT.stop()


async def main():

    global BOT

    BOT = DeepMarketBot()

    signal.signal(
        signal.SIGINT,
        handle_signal
    )

    signal.signal(
        signal.SIGTERM,
        handle_signal
    )

    await BOT.run()


if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        print(
            "\n🛑 Бот тўхтатилди."
        )

    except Exception as exc:

        print(
            f"\n🔴 Критик хато: {exc}",
            file=sys.stderr
        )

        sys.exit(1)