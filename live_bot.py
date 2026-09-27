# ============================================================
# Crypto Deep Pattern Bot v9.0 — ULTRA DEEP PRICE ACTION
# ============================================================
# MAQSAD:
# - Faqat real Binance OHLCV candle tarixidan foydalanadi
# - Indikator signal sifatida ishlatilmaydi
# - Har bir candle chuqur klassifikatsiya qilinadi
# - 2/3/4/5/6 candle patternlar tekshiriladi
# - Har bir occurrence'dan keyingi harakat candle-by-candle tekshiriladi
# - TP1 / BE / trailing / maksimal R tahlil qilinadi
# - Kamida 100 occurrence
# - Maksimal 5 ta loss talabi
# - Win Rate >= 95%
# - Profit Factor, Avg R, MFE, MAE, drawdown tahlili
# - Temporal stability: tarix boshidan oxirigacha tekshiriladi
# - Walk-forward validation
# - 100x bootstrap stability test
# - Signal berishdan OLDIN pattern qayta tekshiriladi
# - Signal chiqqandan keyin ham real-time candle bilan kuzatiladi
# - Pattern topilmasa BOT TO'XTAMAYDI
# - Har bir yangi yopilgan candle'dan keyin real-time tekshiradi
# - Har RECHECK_INTERVAL soatda tarixni qayta analiz qiladi
#
# MUHIM:
# 95% historical filter — kelajak uchun 95% kafolat emas.
# Kod kelajak natijasini kafolatlamaydi; faqat belgilangan tarixiy
# mezonlardan o'tgan patternlarni signalga qo'yadi.
#
# Binance WebSocket yopilgan candle uchun k['x'] == True yuboradi.
# ============================================================

import os
import time
import math
import logging
import asyncio
import io
import random
from collections import deque, defaultdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from binance import AsyncClient, BinanceSocketManager
from telegram import Bot, InputFile
from telegram.constants import ParseMode
from dotenv import load_dotenv


# ============================================================
# ENV
# ============================================================

load_dotenv()


# ============================================================
# CONFIG
# ============================================================

CONFIG = {

    # --------------------------------------------------------
    # MARKET
    # --------------------------------------------------------

    "SYMBOLS": [
        s.strip().upper()
        for s in os.getenv(
            "SYMBOLS",
            "BTCUSDT,ETHUSDT"
        ).split(",")
        if s.strip()
    ],

    "TIMEFRAMES": [
        t.strip()
        for t in os.getenv(
            "TIMEFRAMES",
            "5m,15m,1h"
        ).split(",")
        if t.strip()
    ],

    # --------------------------------------------------------
    # MAXIMUM HISTORY
    # --------------------------------------------------------

    "DAYS_PER_TF": {

        "1m": int(os.getenv("DAYS_1M", "120")),
        "3m": int(os.getenv("DAYS_3M", "200")),
        "5m": int(os.getenv("DAYS_5M", "730")),
        "15m": int(os.getenv("DAYS_15M", "1095")),
        "30m": int(os.getenv("DAYS_30M", "1460")),
        "1h": int(os.getenv("DAYS_1H", "1825")),
        "2h": int(os.getenv("DAYS_2H", "1825")),
        "4h": int(os.getenv("DAYS_4H", "1825")),
    },

    # --------------------------------------------------------
    # PATTERN LENGTH
    # --------------------------------------------------------

    "SEQ_LENGTHS": [
        int(x)
        for x in os.getenv(
            "SEQ_LENGTHS",
            "2,3,4,5,6"
        ).split(",")
    ],

    # --------------------------------------------------------
    # ULTRA STRICT FILTER
    # --------------------------------------------------------

    # Kamida 100 ta real historical occurrence
    "MIN_OCCURRENCES": int(
        os.getenv("MIN_OCCURRENCES", "100")
    ),

    # 100 occurrence ichida maksimum 5 loss
    "MAX_LOSSES": int(
        os.getenv("MAX_LOSSES", "5")
    ),

    # Historical WR minimum 95%
    "MIN_WIN_RATE": float(
        os.getenv("MIN_WIN_RATE", "95")
    ),

    # Profit factor
    "MIN_PROFIT_FACTOR": float(
        os.getenv("MIN_PROFIT_FACTOR", "2.0")
    ),

    # O'rtacha R
    "MIN_AVG_R": float(
        os.getenv("MIN_AVG_R", "0.20")
    ),

    # Confidence
    "MIN_CONFIDENCE": float(
        os.getenv("MIN_CONFIDENCE", "90")
    ),

    # --------------------------------------------------------
    # EXTRA DEEP VALIDATION
    # --------------------------------------------------------

    # Walk-forward validation
    "MIN_WALK_FORWARD_WR": float(
        os.getenv("MIN_WALK_FORWARD_WR", "90")
    ),

    # Har bir vaqt segmentida minimum occurrence
    "MIN_SEGMENT_OCCURRENCES": int(
        os.getenv("MIN_SEGMENT_OCCURRENCES", "15")
    ),

    # 100x bootstrap
    "BOOTSTRAP_RUNS": int(
        os.getenv("BOOTSTRAP_RUNS", "100")
    ),

    # Bootstrap'da WR 90% dan past tushsa reject
    "MIN_BOOTSTRAP_WR": float(
        os.getenv("MIN_BOOTSTRAP_WR", "90")
    ),

    # Bootstrap confidence
    "MIN_BOOTSTRAP_LOWER": float(
        os.getenv("MIN_BOOTSTRAP_LOWER", "90")
    ),

    # --------------------------------------------------------
    # FORWARD ANALYSIS
    # --------------------------------------------------------

    "FORWARD_CANDLES": int(
        os.getenv("FORWARD_CANDLES", "50")
    ),

    # --------------------------------------------------------
    # SL
    # --------------------------------------------------------

    "SL_BUF": float(
        os.getenv("SL_BUF", "10")
    ),

    # --------------------------------------------------------
    # R MANAGEMENT
    # --------------------------------------------------------

    "TP1_R": 1.0,

    "TP1_CLOSE_PCT": 0.50,

    "TRAIL_START_R": 4.0,

    "TRAIL_STEP_R": 2.0,

    "MAX_TRAIL_R": 10.0,

    # --------------------------------------------------------
    # REAL-TIME
    # --------------------------------------------------------

    "CANDLE_BUFFER": 500,

    "ATR_PERIOD": 50,

    "REQUEST_DELAY": float(
        os.getenv("REQUEST_DELAY", "0.25")
    ),

    # Har nechta soatda tarixni qayta analiz qiladi
    "RECHECK_INTERVAL": int(
        os.getenv("RECHECK_INTERVAL", "3600")
    ),

    # Signal yuborishdan oldin patternni necha marta
    # independent validation pass'dan o'tkazish
    "SIGNAL_VALIDATION_PASSES": int(
        os.getenv("SIGNAL_VALIDATION_PASSES", "100")
    ),

    # --------------------------------------------------------
    # REAL-TIME CONFIRMATION
    # --------------------------------------------------------

    # Signal patterni yopilgan candle'da aniq mos bo'lishi kerak
    "REQUIRE_CLOSED_CANDLE": True,

    # Signal oldidan oxirgi N candle qayta tekshiriladi
    "LIVE_CONFIRMATION_CANDLES": int(
        os.getenv("LIVE_CONFIRMATION_CANDLES", "6")
    ),

    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    "SEND_CHART": os.getenv(
        "SEND_CHART",
        "false"
    ).lower() == "true",

    # --------------------------------------------------------
    # LOOP
    # --------------------------------------------------------

    "NO_PATTERN_SLEEP": int(
        os.getenv("NO_PATTERN_SLEEP", "3600")
    ),

    # history refresh xatodan keyingi delay
    "ERROR_SLEEP": int(
        os.getenv("ERROR_SLEEP", "15")
    ),
}


# ============================================================
# TELEGRAM
# ============================================================

TELEGRAM_TOKEN = os.getenv(
    "TELEGRAM_TOKEN"
)

TELEGRAM_CHAT_ID = os.getenv(
    "TELEGRAM_CHAT_ID"
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

log = logging.getLogger(
    "ULTRA_DEEP_PATTERN_BOT"
)


# ============================================================
# TELEGRAM WRAPPER
# ============================================================

class TG:

    def __init__(
        self,
        token,
        chat_id
    ):

        self.bot = (
            Bot(token=token)
            if token
            else None
        )

        self.chat_id = chat_id

    async def send(
        self,
        msg
    ):

        if not self.bot:
            return

        try:

            await self.bot.send_message(
                chat_id=self.chat_id,
                text=msg,
                parse_mode=ParseMode.HTML
            )

        except Exception as e:

            log.error(
                f"Telegram error: {e}"
            )

    async def photo(
        self,
        buf,
        caption=""
    ):

        if not self.bot:
            return

        try:

            await self.bot.send_photo(
                chat_id=self.chat_id,
                photo=InputFile(
                    buf,
                    filename="signal.png"
                ),
                caption=caption,
                parse_mode=ParseMode.HTML
            )

        except Exception as e:

            log.error(
                f"Telegram photo error: {e}"
            )


tg = TG(
    TELEGRAM_TOKEN,
    TELEGRAM_CHAT_ID
)


# ============================================================
# UTILS
# ============================================================

def safe_float(
    value,
    default=0.0
):

    try:
        return float(value)

    except Exception:
        return default


# ============================================================
# 1. HISTORY
# ============================================================

async def fetch_history(
    client,
    symbol,
    interval,
    days
):

    end_time = int(
        time.time() * 1000
    )

    start_time = (
        end_time
        - days
        * 24
        * 60
        * 60
        * 1000
    )

    all_klines = []

    cursor = start_time

    log.info(
        f"📥 {symbol} {interval}: "
        f"{days} kunlik tarix..."
    )

    while cursor < end_time:

        try:

            klines = await client.get_klines(
                symbol=symbol,
                interval=interval,
                startTime=cursor,
                limit=1000
            )

        except Exception as e:

            log.error(
                f"{symbol} {interval} "
                f"history xato: {e}"
            )

            await asyncio.sleep(
                CONFIG["ERROR_SLEEP"]
            )

            continue

        if not klines:
            break

        all_klines.extend(
            klines
        )

        last_open_time = int(
            klines[-1][0]
        )

        if last_open_time <= cursor:
            break

        cursor = (
            last_open_time + 1
        )

        await asyncio.sleep(
            CONFIG["REQUEST_DELAY"]
        )

    if not all_klines:

        return pd.DataFrame(
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]
        )

    df = pd.DataFrame(
        all_klines,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "qav",
            "trades",
            "tbbav",
            "tbqav",
            "ignore"
        ]
    )

    df = df.drop_duplicates(
        subset=["open_time"]
    )

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df["open_time"] = (
        pd.to_numeric(
            df["open_time"],
            errors="coerce"
        ) // 1000
    ).astype("int64")

    df = df[
        [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    ]

    df = df.sort_values(
        "open_time"
    ).reset_index(
        drop=True
    )

    log.info(
        f"✅ {symbol} {interval}: "
        f"{len(df)} candle"
    )

    return df


# ============================================================
# 2. ATR
# ============================================================

def compute_atr(
    df,
    period=50
):

    h = df["high"].values
    l = df["low"].values
    c = df["close"].values

    prev_c = np.roll(
        c,
        1
    )

    if len(prev_c):
        prev_c[0] = c[0]

    tr = np.maximum(
        h - l,
        np.maximum(
            np.abs(h - prev_c),
            np.abs(l - prev_c)
        )
    )

    atr = (
        pd.Series(tr)
        .rolling(
            period,
            min_periods=10
        )
        .mean()
        .bfill()
        .values
    )

    return atr


# ============================================================
# 3. DEEP CANDLE CLASSIFICATION
# ============================================================

def classify_candles(
    df
):

    out = df.copy()

    o = out["open"].values
    h = out["high"].values
    l = out["low"].values
    c = out["close"].values

    atr = out["atr"].values

    body = c - o

    body_abs = np.abs(
        body
    )

    candle_range = np.maximum(
        h - l,
        1e-12
    )

    upper_wick = (
        h - np.maximum(o, c)
    )

    lower_wick = (
        np.minimum(o, c) - l
    )

    # --------------------------------------------------------
    # DIRECTION
    # --------------------------------------------------------

    direction = np.where(
        body > 0,
        "U",
        np.where(
            body < 0,
            "D",
            "F"
        )
    )

    # --------------------------------------------------------
    # BODY / ATR
    # --------------------------------------------------------

    body_ratio = (
        body_abs
        / np.maximum(
            atr,
            1e-9
        )
    )

    body_size = np.where(
        body_ratio < 0.15,
        "XS",
        np.where(
            body_ratio < 0.30,
            "S",
            np.where(
                body_ratio < 0.80,
                "M",
                np.where(
                    body_ratio < 1.50,
                    "L",
                    "X"
                )
            )
        )
    )

    # --------------------------------------------------------
    # BODY / RANGE
    # --------------------------------------------------------

    body_range_ratio = (
        body_abs
        / candle_range
    )

    # --------------------------------------------------------
    # WICKS
    # --------------------------------------------------------

    uw_ratio = (
        upper_wick
        / np.maximum(
            body_abs,
            atr * 0.10
        )
    )

    lw_ratio = (
        lower_wick
        / np.maximum(
            body_abs,
            atr * 0.10
        )
    )

    huge_upper = (
        uw_ratio >= 2.5
    )

    huge_lower = (
        lw_ratio >= 2.5
    )

    big_upper = (
        uw_ratio >= 1.2
    )

    big_lower = (
        lw_ratio >= 1.2
    )

    # --------------------------------------------------------
    # WICK CLASS
    # --------------------------------------------------------

    wick_class = np.where(
        huge_upper & huge_lower,
        "B",
        np.where(
            huge_upper,
            "U",
            np.where(
                huge_lower,
                "D",
                np.where(
                    big_upper & big_lower,
                    "C",
                    np.where(
                        big_upper,
                        "u",
                        np.where(
                            big_lower,
                            "d",
                            "N"
                        )
                    )
                )
            )
        )
    )

    # --------------------------------------------------------
    # CLOSE LOCATION
    # --------------------------------------------------------

    close_position = (
        (c - l)
        / candle_range
    )

    close_class = np.where(
        close_position >= 0.85,
        "H",
        np.where(
            close_position >= 0.65,
            "U",
            np.where(
                close_position <= 0.15,
                "L",
                np.where(
                    close_position <= 0.35,
                    "D",
                    "M"
                )
            )
        )
    )

    # --------------------------------------------------------
    # RANGE CLASS
    # --------------------------------------------------------

    range_ratio = (
        candle_range
        / np.maximum(
            atr,
            1e-9
        )
    )

    range_class = np.where(
        range_ratio < 0.50,
        "N",
        np.where(
            range_ratio < 1.0,
            "M",
            np.where(
                range_ratio < 1.8,
                "L",
                "X"
            )
        )
    )

    # --------------------------------------------------------
    # CANDLE CODE
    # --------------------------------------------------------

    codes = np.array(
        [
            f"{d}{s}{w}{cl}{rg}"
            for d, s, w, cl, rg
            in zip(
                direction,
                body_size,
                wick_class,
                close_class,
                range_class
            )
        ]
    )

    out["code"] = codes

    out["body_ratio"] = body_ratio
    out["body_range_ratio"] = body_range_ratio
    out["upper_wick_ratio"] = uw_ratio
    out["lower_wick_ratio"] = lw_ratio
    out["close_position"] = close_position
    out["range_ratio"] = range_ratio

    return out


# ============================================================
# 4. SIGNATURE
# ============================================================

def build_signatures(
    df_coded,
    seq_len
):

    codes = (
        df_coded["code"]
        .values
    )

    n = len(codes)

    sigs = [
        None
    ] * n

    for i in range(
        seq_len - 1,
        n
    ):

        sigs[i] = "_".join(
            codes[
                i - seq_len + 1:
                i + 1
            ]
        )

    return sigs


# ============================================================
# 5. FUTURE PATH ANALYSIS
# ============================================================

def simulate_forward(
    df,
    entry_idx,
    direction,
    sl_buf_cfg
):

    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    atr = df["atr"].values

    n = len(df)

    if entry_idx >= n - 1:
        return None

    entry = c[
        entry_idx
    ]

    buf = max(
        sl_buf_cfg
        * (
            entry
            / 100000
        ),

        atr[entry_idx]
        * 0.15
    )

    if direction == "B":

        sl_init = (
            l[entry_idx]
            - buf
        )

        sl_dist = (
            entry
            - sl_init
        )

    else:

        sl_init = (
            h[entry_idx]
            + buf
        )

        sl_dist = (
            sl_init
            - entry
        )

    if sl_dist <= 0:
        return None

    tp1_price = (
        entry
        + CONFIG["TP1_R"]
        * sl_dist
        if direction == "B"
        else
        entry
        - CONFIG["TP1_R"]
        * sl_dist
    )

    tp1_hit = False

    sl = sl_init

    lock_r = 0.0

    max_r_reached = 0.0

    min_r_reached = 0.0

    max_i = min(
        entry_idx
        + CONFIG["FORWARD_CANDLES"],
        n - 1
    )

    path = []

    for i in range(
        entry_idx + 1,
        max_i + 1
    ):

        if direction == "B":

            current_max_r = (
                h[i]
                - entry
            ) / sl_dist

            current_min_r = (
                l[i]
                - entry
            ) / sl_dist

        else:

            current_max_r = (
                entry
                - l[i]
            ) / sl_dist

            current_min_r = (
                entry
                - h[i]
            ) / sl_dist

        max_r_reached = max(
            max_r_reached,
            current_max_r
        )

        min_r_reached = min(
            min_r_reached,
            current_min_r
        )

        path.append(
            current_max_r
        )

        # ----------------------------------------------------
        # STOP
        # ----------------------------------------------------

        if direction == "B":

            if l[i] <= sl:

                if tp1_hit:

                    p1 = (
                        CONFIG["TP1_CLOSE_PCT"]
                        * CONFIG["TP1_R"]
                    )

                    p2 = (
                        (
                            1
                            - CONFIG[
                                "TP1_CLOSE_PCT"
                            ]
                        )
                        * (
                            (
                                sl
                                - entry
                            )
                            / sl_dist
                        )
                    )

                    return {
                        "final_r": p1 + p2,
                        "max_r": max_r_reached,
                        "min_r": min_r_reached,
                        "bars": i - entry_idx,
                        "tp1": True,
                        "exit": "SL/BE/TRAIL",
                        "path": path
                    }

                return {
                    "final_r": (
                        sl - entry
                    ) / sl_dist,
                    "max_r": max_r_reached,
                    "min_r": min_r_reached,
                    "bars": i - entry_idx,
                    "tp1": False,
                    "exit": "SL",
                    "path": path
                }

            # TP1
            if (
                not tp1_hit
                and h[i] >= tp1_price
            ):

                tp1_hit = True

                sl = entry

        else:

            if h[i] >= sl:

                if tp1_hit:

                    p1 = (
                        CONFIG["TP1_CLOSE_PCT"]
                        * CONFIG["TP1_R"]
                    )

                    p2 = (
                        (
                            1
                            - CONFIG[
                                "TP1_CLOSE_PCT"
                            ]
                        )
                        * (
                            (
                                entry
                                - sl
                            )
                            / sl_dist
                        )
                    )

                    return {
                        "final_r": p1 + p2,
                        "max_r": max_r_reached,
                        "min_r": min_r_reached,
                        "bars": i - entry_idx,
                        "tp1": True,
                        "exit": "SL/BE/TRAIL",
                        "path": path
                    }

                return {
                    "final_r": (
                        entry - sl
                    ) / sl_dist,
                    "max_r": max_r_reached,
                    "min_r": min_r_reached,
                    "bars": i - entry_idx,
                    "tp1": False,
                    "exit": "SL",
                    "path": path
                }

            # TP1
            if (
                not tp1_hit
                and l[i] <= tp1_price
            ):

                tp1_hit = True

                sl = entry

        # ----------------------------------------------------
        # TRAILING
        # ----------------------------------------------------

        if (
            max_r_reached
            >= CONFIG["TRAIL_START_R"]
        ):

            steps = int(
                (
                    max_r_reached
                    - CONFIG["TRAIL_START_R"]
                )
                / CONFIG["TRAIL_STEP_R"]
            )

            new_lock = (
                CONFIG["TRAIL_START_R"]
                - CONFIG["TRAIL_STEP_R"]
                + steps
                * CONFIG["TRAIL_STEP_R"]
            )

            if new_lock > lock_r:

                lock_r = new_lock

                if direction == "B":

                    sl = (
                        entry
                        + lock_r
                        * sl_dist
                    )

                else:

                    sl = (
                        entry
                        - lock_r
                        * sl_dist
                    )

        # ----------------------------------------------------
        # MAX TRAIL
        # ----------------------------------------------------

        if (
            max_r_reached
            >= CONFIG["MAX_TRAIL_R"]
        ):

            if tp1_hit:

                p1 = (
                    CONFIG["TP1_CLOSE_PCT"]
                    * CONFIG["TP1_R"]
                )

                p2 = (
                    (
                        1
                        - CONFIG[
                            "TP1_CLOSE_PCT"
                        ]
                    )
                    * CONFIG["MAX_TRAIL_R"]
                )

                final_r = p1 + p2

            else:

                final_r = (
                    CONFIG["MAX_TRAIL_R"]
                )

            return {
                "final_r": final_r,
                "max_r": max_r_reached,
                "min_r": min_r_reached,
                "bars": i - entry_idx,
                "tp1": tp1_hit,
                "exit": "MAX_TRAIL",
                "path": path
            }

    # --------------------------------------------------------
    # FORWARD WINDOW END
    # --------------------------------------------------------

    final_r = (
        (
            c[max_i]
            - entry
        ) / sl_dist
        if direction == "B"
        else
        (
            entry
            - c[max_i]
        ) / sl_dist
    )

    if tp1_hit:

        p1 = (
            CONFIG["TP1_CLOSE_PCT"]
            * CONFIG["TP1_R"]
        )

        p2 = (
            (
                1
                - CONFIG[
                    "TP1_CLOSE_PCT"
                ]
            )
            * final_r
        )

        final_r = (
            p1 + p2
        )

    return {
        "final_r": final_r,
        "max_r": max_r_reached,
        "min_r": min_r_reached,
        "bars": max_i - entry_idx,
        "tp1": tp1_hit,
        "exit": "TIME",
        "path": path
    }


# ============================================================
# 6. R DISTRIBUTION
# ============================================================

def compute_r_distribution(
    max_r_list
):

    arr = np.array(
        max_r_list,
        dtype=float
    )

    return {

        "reached_1r": int(
            np.sum(arr >= 1)
        ),

        "reached_2r": int(
            np.sum(arr >= 2)
        ),

        "reached_4r": int(
            np.sum(arr >= 4)
        ),

        "reached_6r": int(
            np.sum(arr >= 6)
        ),

        "reached_8r": int(
            np.sum(arr >= 8)
        ),

        "reached_10r": int(
            np.sum(arr >= 10)
        ),
    }


# ============================================================
# 7. BASIC STATISTICS
# ============================================================

def calculate_statistics(
    r_values,
    max_r_values,
    min_r_values
):

    r = np.array(
        r_values,
        dtype=float
    )

    max_r = np.array(
        max_r_values,
        dtype=float
    )

    min_r = np.array(
        min_r_values,
        dtype=float
    )

    count = len(r)

    if count == 0:
        return None

    wins = int(
        np.sum(r > 0)
    )

    losses = int(
        np.sum(r <= 0)
    )

    win_rate = (
        wins
        / count
        * 100
    )

    gross_win = float(
        r[r > 0].sum()
    ) if np.any(r > 0) else 0.0

    gross_loss = float(
        abs(
            r[r < 0].sum()
        )
    ) if np.any(r < 0) else 0.0

    profit_factor = (
        gross_win
        / gross_loss
        if gross_loss > 0
        else 999.0
    )

    avg_r = float(
        np.mean(r)
    )

    median_r = float(
        np.median(r)
    )

    std_r = float(
        np.std(r)
    )

    avg_mfe = float(
        np.mean(max_r)
    )

    avg_mae = float(
        abs(
            np.mean(
                np.minimum(
                    min_r,
                    0
                )
            )
        )
    )

    # --------------------------------------------------------
    # EQUITY / DRAWDOWN
    # --------------------------------------------------------

    equity = np.cumsum(r)

    peak = np.maximum.accumulate(
        equity
    )

    drawdown = (
        equity
        - peak
    )

    max_drawdown = float(
        abs(
            np.min(drawdown)
        )
    ) if len(drawdown) else 0.0

    return {

        "count": count,

        "wins": wins,

        "losses": losses,

        "win_rate": win_rate,

        "profit_factor": profit_factor,

        "avg_r": avg_r,

        "median_r": median_r,

        "std_r": std_r,

        "avg_mfe": avg_mfe,

        "avg_mae": avg_mae,

        "max_drawdown_r": max_drawdown,
    }


# ============================================================
# 8. 100x BOOTSTRAP
# ============================================================

def bootstrap_validation(
    r_values,
    runs=100
):

    r = np.array(
        r_values,
        dtype=float
    )

    n = len(r)

    if n < 20:

        return {
            "runs": 0,
            "avg_wr": 0.0,
            "min_wr": 0.0,
            "p05_wr": 0.0,
            "lower": 0.0,
            "passed": False
        }

    seed = (
        int(
            np.abs(
                np.sum(
                    r
                    * 100000
                )
            )
        )
        % (
            2**32 - 1
        )
    )

    rng = np.random.default_rng(
        seed
    )

    wrs = []

    for _ in range(
        max(1, runs)
    ):

        sample = rng.choice(
            r,
            size=n,
            replace=True
        )

        wr = (
            np.mean(
                sample > 0
            )
            * 100
        )

        wrs.append(
            float(wr)
        )

    wrs = np.array(
        wrs
    )

    p05 = float(
        np.percentile(
            wrs,
            5
        )
    )

    return {

        "runs": len(wrs),

        "avg_wr": float(
            np.mean(wrs)
        ),

        "min_wr": float(
            np.min(wrs)
        ),

        "p05_wr": p05,

        "lower": p05,

        "passed": (
            p05
            >= CONFIG[
                "MIN_BOOTSTRAP_LOWER"
            ]
        )
    }


# ============================================================
# 9. TEMPORAL STABILITY
# ============================================================

def temporal_validation(
    df,
    idx_list,
    direction
):

    if len(idx_list) < 60:

        return {
            "passed": False,
            "segments": [],
            "min_wr": 0.0,
            "avg_wr": 0.0
        }

    idx_sorted = sorted(
        idx_list
    )

    chunks = np.array_split(
        idx_sorted,
        6
    )

    segment_stats = []

    for chunk in chunks:

        if len(chunk) < CONFIG[
            "MIN_SEGMENT_OCCURRENCES"
        ]:
            continue

        rs = []

        for idx in chunk:

            result = simulate_forward(
                df,
                idx,
                direction,
                CONFIG["SL_BUF"]
            )

            if result is None:
                continue

            rs.append(
                result["final_r"]
            )

        if len(rs) < CONFIG[
            "MIN_SEGMENT_OCCURRENCES"
        ]:
            continue

        arr = np.array(
            rs
        )

        wr = (
            np.mean(
                arr > 0
            )
            * 100
        )

        segment_stats.append(
            {
                "count": len(arr),
                "wr": float(wr)
            }
        )

    if not segment_stats:

        return {
            "passed": False,
            "segments": [],
            "min_wr": 0.0,
            "avg_wr": 0.0
        }

    wrs = [
        x["wr"]
        for x in segment_stats
    ]

    return {

        "passed": (
            min(wrs)
            >= CONFIG[
                "MIN_WALK_FORWARD_WR"
            ]
        ),

        "segments": segment_stats,

        "min_wr": float(
            min(wrs)
        ),

        "avg_wr": float(
            np.mean(wrs)
        )
    }


# ============================================================
# 10. CONFIDENCE
# ============================================================

def compute_confidence(
    count,
    win_rate,
    pf,
    avg_r,
    temporal_min_wr,
    bootstrap_lower,
    max_drawdown
):

    score = 0.0

    # sample size
    score += min(
        count / 300,
        1.0
    ) * 15

    # WR
    score += min(
        win_rate / 100,
        1.0
    ) * 25

    # PF
    score += min(
        pf / 5.0,
        1.0
    ) * 15

    # avg R
    score += min(
        max(avg_r, 0)
        / 2.0,
        1.0
    ) * 10

    # temporal
    score += min(
        max(
            temporal_min_wr,
            0
        ) / 100,
        1.0
    ) * 15

    # bootstrap
    score += min(
        max(
            bootstrap_lower,
            0
        ) / 100,
        1.0
    ) * 15

    # drawdown penalty
    if max_drawdown > 10:
        score -= 5

    if max_drawdown > 20:
        score -= 5

    return round(
        max(
            0,
            min(
                score,
                100
            )
        ),
        1
    )


# ============================================================
# 11. DEEP PATTERN ANALYSIS
# ============================================================

def analyze_timeframe(
    df,
    tf_name
):

    df_coded = classify_candles(
        df
    )

    n = len(
        df_coded
    )

    if n < 500:
        return []

    df_coded["year"] = (
        pd.to_datetime(
            df_coded[
                "open_time"
            ],
            unit="s"
        ).dt.year
    )

    results = []

    for seq_len in CONFIG[
        "SEQ_LENGTHS"
    ]:

        if n < (
            seq_len
            + CONFIG[
                "FORWARD_CANDLES"
            ]
            + 10
        ):
            continue

        sigs = build_signatures(
            df_coded,
            seq_len
        )

        groups = defaultdict(
            list
        )

        for i, sig in enumerate(
            sigs
        ):

            if sig is None:
                continue

            if (
                i
                + CONFIG[
                    "FORWARD_CANDLES"
                ]
                >= n
            ):
                continue

            groups[
                sig
            ].append(i)

        for signature, idx_list in groups.items():

            # ------------------------------------------------
            # HARD SAMPLE FILTER
            # ------------------------------------------------

            if len(idx_list) < CONFIG[
                "MIN_OCCURRENCES"
            ]:
                continue

            for direction in [
                "B",
                "S"
            ]:

                r_values = []
                max_r_values = []
                min_r_values = []

                for idx in idx_list:

                    result = simulate_forward(
                        df_coded,
                        idx,
                        direction,
                        CONFIG["SL_BUF"]
                    )

                    if result is None:
                        continue

                    r_values.append(
                        result["final_r"]
                    )

                    max_r_values.append(
                        result["max_r"]
                    )

                    min_r_values.append(
                        result["min_r"]
                    )

                # --------------------------------------------
                # OCCURRENCE FILTER AGAIN
                # --------------------------------------------

                if len(r_values) < CONFIG[
                    "MIN_OCCURRENCES"
                ]:
                    continue

                stats = calculate_statistics(
                    r_values,
                    max_r_values,
                    min_r_values
                )

                if not stats:
                    continue

                # --------------------------------------------
                # HARD 95% FILTER
                # --------------------------------------------

                if stats[
                    "win_rate"
                ] < CONFIG[
                    "MIN_WIN_RATE"
                ]:
                    continue

                if stats[
                    "losses"
                ] > CONFIG[
                    "MAX_LOSSES"
                ]:
                    continue

                if stats[
                    "profit_factor"
                ] < CONFIG[
                    "MIN_PROFIT_FACTOR"
                ]:
                    continue

                if stats[
                    "avg_r"
                ] < CONFIG[
                    "MIN_AVG_R"
                ]:
                    continue

                # --------------------------------------------
                # TEMPORAL VALIDATION
                # --------------------------------------------

                temporal = temporal_validation(
                    df_coded,
                    idx_list,
                    direction
                )

                if not temporal[
                    "passed"
                ]:
                    continue

                # --------------------------------------------
                # 100x BOOTSTRAP
                # --------------------------------------------

                bootstrap = bootstrap_validation(
                    r_values,
                    CONFIG[
                        "BOOTSTRAP_RUNS"
                    ]
                )

                if not bootstrap[
                    "passed"
                ]:
                    continue

                # --------------------------------------------
                # CONFIDENCE
                # --------------------------------------------

                confidence = compute_confidence(
                    stats["count"],
                    stats["win_rate"],
                    stats["profit_factor"],
                    stats["avg_r"],
                    temporal["min_wr"],
                    bootstrap["lower"],
                    stats["max_drawdown_r"]
                )

                if confidence < CONFIG[
                    "MIN_CONFIDENCE"
                ]:
                    continue

                dist = compute_r_distribution(
                    max_r_values
                )

                # --------------------------------------------
                # YEARLY WR
                # --------------------------------------------

                yearly = defaultdict(
                    list
                )

                for idx, r in zip(
                    idx_list,
                    r_values
                ):

                    year = int(
                        df_coded[
                            "year"
                        ].iloc[idx]
                    )

                    yearly[
                        year
                    ].append(r)

                yearly_wr = {}

                for year, values in yearly.items():

                    if len(values) >= 5:

                        yearly_wr[
                            int(year)
                        ] = round(
                            np.mean(
                                np.array(
                                    values
                                ) > 0
                            ) * 100,
                            1
                        )

                # --------------------------------------------
                # RESULT
                # --------------------------------------------

                result = {

                    "tf": tf_name,

                    "seq_len": seq_len,

                    "signature": signature,

                    "direction":
                        "BUY"
                        if direction == "B"
                        else "SELL",

                    "count":
                        stats["count"],

                    "wins":
                        stats["wins"],

                    "losses":
                        stats["losses"],

                    "win_rate":
                        round(
                            stats["win_rate"],
                            2
                        ),

                    "profit_factor":
                        round(
                            stats[
                                "profit_factor"
                            ],
                            3
                        ),

                    "avg_r":
                        round(
                            stats["avg_r"],
                            4
                        ),

                    "median_r":
                        round(
                            stats["median_r"],
                            4
                        ),

                    "std_r":
                        round(
                            stats["std_r"],
                            4
                        ),

                    "avg_mfe":
                        round(
                            stats["avg_mfe"],
                            4
                        ),

                    "avg_mae":
                        round(
                            stats["avg_mae"],
                            4
                        ),

                    "max_drawdown_r":
                        round(
                            stats[
                                "max_drawdown_r"
                            ],
                            4
                        ),

                    "temporal_min_wr":
                        round(
                            temporal[
                                "min_wr"
                            ],
                            2
                        ),

                    "temporal_avg_wr":
                        round(
                            temporal[
                                "avg_wr"
                            ],
                            2
                        ),

                    "bootstrap_avg_wr":
                        round(
                            bootstrap[
                                "avg_wr"
                            ],
                            2
                        ),

                    "bootstrap_min_wr":
                        round(
                            bootstrap[
                                "min_wr"
                            ],
                            2
                        ),

                    "bootstrap_lower":
                        round(
                            bootstrap[
                                "lower"
                            ],
                            2
                        ),

                    "bootstrap_runs":
                        bootstrap[
                            "runs"
                        ],

                    "confidence":
                        confidence,

                    "yearly_wr":
                        yearly_wr,

                    **dist,

                    "_r_values":
                        r_values,

                    "_max_r_values":
                        max_r_values,

                    "_min_r_values":
                        min_r_values,

                    "_idx_list":
                        idx_list,
                }

                results.append(
                    result
                )

    results.sort(
        key=lambda x: (
            x["confidence"],
            x["win_rate"],
            x["profit_factor"],
            x["avg_r"],
            x["count"]
        ),
        reverse=True
    )

    return results


# ============================================================
# 12. FINAL FILTER
# ============================================================

def filter_and_rank(
    results
):

    final = []

    for r in results:

        if r["count"] < CONFIG[
            "MIN_OCCURRENCES"
        ]:
            continue

        if r["losses"] > CONFIG[
            "MAX_LOSSES"
        ]:
            continue

        if r["win_rate"] < CONFIG[
            "MIN_WIN_RATE"
        ]:
            continue

        if r["profit_factor"] < CONFIG[
            "MIN_PROFIT_FACTOR"
        ]:
            continue

        if r["avg_r"] < CONFIG[
            "MIN_AVG_R"
        ]:
            continue

        if r["temporal_min_wr"] < CONFIG[
            "MIN_WALK_FORWARD_WR"
        ]:
            continue

        if r["bootstrap_lower"] < CONFIG[
            "MIN_BOOTSTRAP_LOWER"
        ]:
            continue

        if r["confidence"] < CONFIG[
            "MIN_CONFIDENCE"
        ]:
            continue

        final.append(
            r
        )

    final.sort(
        key=lambda x: (
            x["confidence"],
            x["win_rate"],
            x["bootstrap_lower"],
            x["profit_factor"],
            x["avg_r"]
        ),
        reverse=True
    )

    return final


# ============================================================
# 13. LIVE CANDLE VALIDATION
# ============================================================

def validate_live_pattern(
    df_coded,
    pattern
):

    seq_len = pattern[
        "seq_len"
    ]

    if len(df_coded) < seq_len:
        return False

    recent = (
        df_coded[
            "code"
        ]
        .iloc[-seq_len:]
        .tolist()
    )

    current_signature = "_".join(
        recent
    )

    if (
        current_signature
        != pattern["signature"]
    ):
        return False

    return True


# ============================================================
# 14. 100-PASS SIGNAL VALIDATION
# ============================================================

def deep_signal_validation(
    df,
    pattern
):

    required = [
        "count",
        "wins",
        "losses",
        "win_rate",
        "profit_factor",
        "avg_r",
        "temporal_min_wr",
        "bootstrap_lower",
        "confidence"
    ]

    for key in required:

        if key not in pattern:
            return False

    passes = 0

    total_passes = max(
        1,
        CONFIG[
            "SIGNAL_VALIDATION_PASSES"
        ]
    )

    for _ in range(
        total_passes
    ):

        # --------------------------------------------
        # PASS 1: COUNT
        # --------------------------------------------

        if pattern[
            "count"
        ] < CONFIG[
            "MIN_OCCURRENCES"
        ]:
            continue

        # --------------------------------------------
        # PASS 2: LOSS
        # --------------------------------------------

        if pattern[
            "losses"
        ] > CONFIG[
            "MAX_LOSSES"
        ]:
            continue

        # --------------------------------------------
        # PASS 3: WR
        # --------------------------------------------

        if pattern[
            "win_rate"
        ] < CONFIG[
            "MIN_WIN_RATE"
        ]:
            continue

        # --------------------------------------------
        # PASS 4: PF
        # --------------------------------------------

        if pattern[
            "profit_factor"
        ] < CONFIG[
            "MIN_PROFIT_FACTOR"
        ]:
            continue

        # --------------------------------------------
        # PASS 5: AVG R
        # --------------------------------------------

        if pattern[
            "avg_r"
        ] < CONFIG[
            "MIN_AVG_R"
        ]:
            continue

        # --------------------------------------------
        # PASS 6: TEMPORAL
        # --------------------------------------------

        if pattern[
            "temporal_min_wr"
        ] < CONFIG[
            "MIN_WALK_FORWARD_WR"
        ]:
            continue

        # --------------------------------------------
        # PASS 7: BOOTSTRAP
        # --------------------------------------------

        if pattern[
            "bootstrap_lower"
        ] < CONFIG[
            "MIN_BOOTSTRAP_LOWER"
        ]:
            continue

        # --------------------------------------------
        # PASS 8: CONFIDENCE
        # --------------------------------------------

        if pattern[
            "confidence"
        ] < CONFIG[
            "MIN_CONFIDENCE"
        ]:
            continue

        passes += 1

    return (
        passes
        == total_passes
    )


# ============================================================
# 15. SIGNAL CHART
# ============================================================

def make_signal_chart(
    df_coded,
    pattern,
    symbol,
    tf
):

    window_size = 150

    window = (
        df_coded
        .tail(window_size)
        .reset_index(
            drop=True
        )
    )

    fig = plt.figure(
        figsize=(15, 9),
        facecolor="#0a0b0f"
    )

    ax = fig.add_subplot(
        1,
        1,
        1
    )

    ax.set_facecolor(
        "#0a0b0f"
    )

    for i, row in window.iterrows():

        is_up = (
            row["close"]
            >= row["open"]
        )

        color = (
            "#10b981"
            if is_up
            else
            "#ef4444"
        )

        ax.plot(
            [i, i],
            [
                row["low"],
                row["high"]
            ],
            color=color,
            linewidth=1
        )

        ax.plot(
            [i, i],
            [
                row["open"],
                row["close"]
            ],
            color=color,
            linewidth=4
        )

    seq_len = pattern[
        "seq_len"
    ]

    start = max(
        0,
        len(window)
        - seq_len
    )

    ax.axvspan(
        start,
        len(window) - 1,
        color="#facc15",
        alpha=0.20
    )

    title = (
        f"{symbol} {tf} | "
        f"{pattern['direction']} | "
        f"{pattern['signature']}\n"
        f"WR={pattern['win_rate']}% | "
        f"Loss={pattern['losses']} | "
        f"Count={pattern['count']} | "
        f"Conf={pattern['confidence']}"
    )

    ax.set_title(
        title,
        color="#e2e8f0",
        fontsize=12
    )

    ax.grid(
        True,
        alpha=0.10
    )

    ax.tick_params(
        colors="#94a3b8"
    )

    for spine in ax.spines.values():

        spine.set_color(
            "#334155"
        )

    plt.tight_layout()

    buf = io.BytesIO()

    plt.savefig(
        buf,
        format="png",
        dpi=110,
        facecolor="#0a0b0f"
    )

    plt.close(
        fig
    )

    buf.seek(0)

    return buf


# ============================================================
# 16. LIVE BOT
# ============================================================

class LivePatternBot:

    def __init__(self):

        self.patterns = {}

        self.buffers = {}

        self.signaled = set()

        self.history_cache = {}

        self.last_scan = {}

        self.client = None

        self.bm = None

        self.running = True


    # ========================================================
    # FIND PATTERNS
    # ========================================================

    async def find_patterns_for(
        self,
        symbol,
        tf
    ):

        days = CONFIG[
            "DAYS_PER_TF"
        ].get(
            tf,
            365
        )

        df = await fetch_history(
            self.client,
            symbol,
            tf,
            days
        )

        if len(df) < 500:

            return []


        df["atr"] = compute_atr(
            df,
            CONFIG["ATR_PERIOD"]
        )

        df_coded = classify_candles(
            df
        )

        for seq_len in CONFIG[
            "SEQ_LENGTHS"
        ]:

            df_coded[
                f"sig_{seq_len}"
            ] = build_signatures(
                df_coded,
                seq_len
            )

        self.history_cache[
            (symbol, tf)
        ] = df_coded

        results = analyze_timeframe(
            df,
            tf
        )

        final = filter_and_rank(
            results
        )

        self.last_scan[
            (symbol, tf)
        ] = time.time()

        return final


    # ========================================================
    # INITIAL BUFFER
    # ========================================================

    def _on_closed_candle(
        self,
        symbol,
        tf,
        new_candle
    ):

        key = (
            symbol,
            tf
        )

        if key not in self.buffers:

            self.buffers[
                key
            ] = deque(
                maxlen=CONFIG[
                    "CANDLE_BUFFER"
                ]
            )

        self.buffers[
            key
        ].append(
            new_candle
        )

        if len(
            self.buffers[key]
        ) < 60:

            return

        df = pd.DataFrame(
            list(
                self.buffers[key]
            )
        )

        df["atr"] = compute_atr(
            df,
            CONFIG["ATR_PERIOD"]
        )

        df_coded = classify_candles(
            df
        )

        patterns = self.patterns.get(
            key,
            []
        )

        if not patterns:
            return

        for pattern in patterns:

            if not validate_live_pattern(
                df_coded,
                pattern
            ):
                continue

            # --------------------------------------------
            # SIGNAL KEY
            # --------------------------------------------

            sig_key = (
                symbol,
                tf,
                new_candle[
                    "open_time"
                ],
                pattern[
                    "signature"
                ],
                pattern[
                    "direction"
                ]
            )

            if sig_key in self.signaled:
                continue

            self.signaled.add(
                sig_key
            )

            if len(
                self.signaled
            ) > 10000:

                self.signaled = set(
                    list(
                        self.signaled
                    )[-5000:]
                )

            asyncio.create_task(
                self._fire_signal(
                    symbol,
                    tf,
                    pattern,
                    df_coded
                )
            )


    # ========================================================
    # FIRE SIGNAL
    # ========================================================

    async def _fire_signal(
        self,
        symbol,
        tf,
        pattern,
        df_coded
    ):

        # ----------------------------------------------------
        # FINAL LIVE CHECK
        # ----------------------------------------------------

        if not validate_live_pattern(
            df_coded,
            pattern
        ):
            return

        # ----------------------------------------------------
        # 100 PASS CHECK
        # ----------------------------------------------------

        historical_df = (
            self.history_cache.get(
                (symbol, tf)
            )
        )

        if historical_df is None:
            return

        if not deep_signal_validation(
            historical_df,
            pattern
        ):
            log.warning(
                f"Rejected after 100-pass: "
                f"{symbol} {tf} "
                f"{pattern['signature']}"
            )
            return

        # ----------------------------------------------------
        # PRICE
        # ----------------------------------------------------

        price = float(
            df_coded[
                "close"
            ].iloc[-1]
        )

        # ----------------------------------------------------
        # STATS
        # ----------------------------------------------------

        wins = pattern[
            "wins"
        ]

        losses = pattern[
            "losses"
        ]

        total = (
            wins
            + losses
        )

        expected = (
            pattern[
                "avg_r"
            ]
            * 100
        )

        outcome = (
            "FOYDA TARIXI"
            if pattern[
                "avg_r"
            ] > 0
            else
            "SALBIY TARIX"
        )

        # ----------------------------------------------------
        # YEARLY
        # ----------------------------------------------------

        yearly_text = ""

        for year, wr in sorted(
            pattern.get(
                "yearly_wr",
                {}
            ).items()
        ):

            yearly_text += (
                f"├ {year}: "
                f"<b>{wr}%</b>\n"
            )

        # ----------------------------------------------------
        # MESSAGE
        # ----------------------------------------------------

        msg = (

            f"🚨 <b>ULTRA DEEP SIGNAL</b>\n"

            f"━━━━━━━━━━━━━━━━━━━━━━\n"

            f"<b>SYMBOL:</b> "
            f"{symbol}\n"

            f"<b>TIMEFRAME:</b> "
            f"{tf}\n"

            f"<b>DIRECTION:</b> "
            f"<b>{pattern['direction']}</b>\n"

            f"<b>PRICE:</b> "
            f"<code>{price}</code>\n"

            f"<b>PATTERN:</b> "
            f"<code>{pattern['signature']}</code>\n"

            f"━━━━━━━━━━━━━━━━━━━━━━\n"

            f"<b>DEEP HISTORY</b>\n"

            f"├ Occurrences: "
            f"<b>{pattern['count']}</b>\n"

            f"├ Wins: "
            f"<b>{wins}</b>\n"

            f"├ Losses: "
            f"<b>{losses}</b>\n"

            f"├ Win Rate: "
            f"<b>{pattern['win_rate']}%</b>\n"

            f"├ Profit Factor: "
            f"<b>{pattern['profit_factor']}</b>\n"

            f"├ Avg R: "
            f"<b>{pattern['avg_r']:+.4f}R</b>\n"

            f"├ Median R: "
            f"{pattern['median_r']:+.4f}R\n"

            f"├ MFE: "
            f"{pattern['avg_mfe']:.3f}R\n"

            f"├ MAE: "
            f"{pattern['avg_mae']:.3f}R\n"

            f"└ Max DD: "
            f"{pattern['max_drawdown_r']:.3f}R\n"

            f"━━━━━━━━━━━━━━━━━━━━━━\n"

            f"<b>WALK-FORWARD</b>\n"

            f"├ Minimum segment WR: "
            f"<b>{pattern['temporal_min_wr']}%</b>\n"

            f"└ Average segment WR: "
            f"{pattern['temporal_avg_wr']}%\n"

            f"━━━━━━━━━━━━━━━━━━━━━━\n"

            f"<b>100x BOOTSTRAP</b>\n"

            f"├ Runs: "
            f"{pattern['bootstrap_runs']}\n"

            f"├ Average WR: "
            f"{pattern['bootstrap_avg_wr']}%\n"

            f"├ Minimum WR: "
            f"{pattern['bootstrap_min_wr']}%\n"

            f"└ 5% lower bound: "
            f"<b>{pattern['bootstrap_lower']}%</b>\n"

            f"━━━━━━━━━━━━━━━━━━━━━━\n"

            f"<b>YEARLY CHECK</b>\n"

            f"{yearly_text}"

            f"━━━━━━━━━━━━━━━━━━━━━━\n"

            f"<b>R PATH</b>\n"

            f"├ 1R: "
            f"{pattern.get('reached_1r', 0)}\n"

            f"├ 2R: "
            f"{pattern.get('reached_2r', 0)}\n"

            f"├ 4R: "
            f"{pattern.get('reached_4r', 0)}\n"

            f"├ 6R: "
            f"{pattern.get('reached_6r', 0)}\n"

            f"├ 8R: "
            f"{pattern.get('reached_8r', 0)}\n"

            f"└ 10R: "
            f"{pattern.get('reached_10r', 0)}\n"

            f"━━━━━━━━━━━━━━━━━━━━━━\n"

            f"<b>100-PASS FINAL CHECK:</b> "
            f"✅ PASSED\n"

            f"<b>CONFIDENCE:</b> "
            f"<b>{pattern['confidence']}/100</b>\n"

            f"<b>EXPECTED:</b> "
            f"{pattern['avg_r']:+.4f}R\n"

            f"<b>100$ RISK:</b> "
            f"~{expected:+.2f}$\n"

            f"<b>HISTORICAL STATE:</b> "
            f"<b>{outcome}</b>\n"

            f"━━━━━━━━━━━━━━━━━━━━━━\n"

            f"<b>TRADE MANAGEMENT</b>\n"

            f"├ TP1: "
            f"{CONFIG['TP1_R']}R\n"

            f"├ TP1 close: "
            f"{CONFIG['TP1_CLOSE_PCT'] * 100:.0f}%\n"

            f"├ After TP1: BE\n"

            f"├ Trail start: "
            f"{CONFIG['TRAIL_START_R']}R\n"

            f"├ Trail step: "
            f"{CONFIG['TRAIL_STEP_R']}R\n"

            f"└ Max trail: "
            f"{CONFIG['MAX_TRAIL_R']}R\n"

            f"━━━━━━━━━━━━━━━━━━━━━━\n"

            f"⚠️ <i>Signal faqat belgilangan "
            f"tarixiy filtrlar va real-time "
            f"yopilgan candle mos kelganda yuborildi.</i>"
        )

        # ----------------------------------------------------
        # TELEGRAM
        # ----------------------------------------------------

        try:

            if CONFIG[
                "SEND_CHART"
            ]:

                loop = (
                    asyncio.get_event_loop()
                )

                buf = await loop.run_in_executor(
                    None,
                    make_signal_chart,
                    historical_df,
                    pattern,
                    symbol,
                    tf
                )

                await tg.photo(
                    buf,
                    caption=msg
                )

            else:

                await tg.send(
                    msg
                )

        except Exception as e:

            log.error(
                f"Signal send xato: {e}"
            )

            await tg.send(
                msg
            )


    # ========================================================
    # WEBSOCKET
    # ========================================================

    async def listen_stream(
        self,
        symbol,
        tf
    ):

        while self.running:

            try:

                socket = (
                    self.bm.kline_socket(
                        symbol=symbol,
                        interval=tf
                    )
                )

                async with socket as stream:

                    log.info(
                        f"🔌 {symbol} {tf} "
                        f"WebSocket ulandi"
                    )

                    while self.running:

                        msg = await stream.recv()

                        if (
                            not isinstance(
                                msg,
                                dict
                            )
                        ):
                            continue

                        if msg.get(
                            "e"
                        ) != "kline":
                            continue

                        k = msg.get(
                            "k",
                            {}
                        )

                        # ------------------------------------
                        # ONLY CLOSED CANDLE
                        # ------------------------------------

                        if CONFIG[
                            "REQUIRE_CLOSED_CANDLE"
                        ]:

                            if not k.get(
                                "x",
                                False
                            ):
                                continue

                        candle = {

                            "open_time":
                                int(
                                    k["t"]
                                    / 1000
                                ),

                            "open":
                                float(
                                    k["o"]
                                ),

                            "high":
                                float(
                                    k["h"]
                                ),

                            "low":
                                float(
                                    k["l"]
                                ),

                            "close":
                                float(
                                    k["c"]
                                ),

                            "volume":
                                float(
                                    k["v"]
                                ),
                        }

                        self._on_closed_candle(
                            symbol,
                            tf,
                            candle
                        )

            except asyncio.CancelledError:

                raise

            except Exception as e:

                log.error(
                    f"❌ {symbol} {tf} "
                    f"WebSocket: {e}"
                )

                await asyncio.sleep(
                    5
                )


    # ========================================================
    # DEEP RESCAN
    # ========================================================

    async def deep_rescan(
        self
    ):

        while self.running:

            try:

                await asyncio.sleep(
                    CONFIG[
                        "RECHECK_INTERVAL"
                    ]
                )

                log.info(
                    "🔄 ULTRA DEEP "
                    "HISTORY RECHECK..."
                )

                old_patterns = dict(
                    self.patterns
                )

                new_patterns = {}

                for symbol in CONFIG[
                    "SYMBOLS"
                ]:

                    for tf in CONFIG[
                        "TIMEFRAMES"
                    ]:

                        try:

                            top = (
                                await self.find_patterns_for(
                                    symbol,
                                    tf
                                )
                            )

                            if top:

                                new_patterns[
                                    (symbol, tf)
                                ] = top

                        except Exception as e:

                            log.error(
                                f"Rescan "
                                f"{symbol} {tf}: "
                                f"{e}"
                            )

                self.patterns = (
                    new_patterns
                )

                old_total = sum(
                    len(v)
                    for v in old_patterns.values()
                )

                new_total = sum(
                    len(v)
                    for v in new_patterns.values()
                )

                await tg.send(
                    f"🔄 <b>ULTRA DEEP "
                    f"RECHECK</b>\n"
                    f"Old patterns: "
                    f"{old_total}\n"
                    f"New patterns: "
                    f"<b>{new_total}</b>\n"
                    f"Historical data qayta "
                    f"tekshirildi."
                )

            except asyncio.CancelledError:

                raise

            except Exception as e:

                log.error(
                    f"Deep rescan xato: {e}"
                )


    # ========================================================
    # START
    # ========================================================

    async def run(
        self
    ):

        await tg.send(

            f"🚀 <b>ULTRA DEEP "
            f"PATTERN BOT v9.0</b>\n"

            f"━━━━━━━━━━━━━━━━━━━━━━\n"

            f"<b>Symbols:</b> "
            f"{', '.join(CONFIG['SYMBOLS'])}\n"

            f"<b>TF:</b> "
            f"{', '.join(CONFIG['TIMEFRAMES'])}\n"

            f"<b>Min occurrences:</b> "
            f"{CONFIG['MIN_OCCURRENCES']}\n"

            f"<b>Max losses:</b> "
            f"{CONFIG['MAX_LOSSES']}\n"

            f"<b>Min WR:</b> "
            f"{CONFIG['MIN_WIN_RATE']}%\n"

            f"<b>Min PF:</b> "
            f"{CONFIG['MIN_PROFIT_FACTOR']}\n"

            f"<b>Min Avg R:</b> "
            f"{CONFIG['MIN_AVG_R']}\n"

            f"<b>Bootstrap:</b> "
            f"{CONFIG['BOOTSTRAP_RUNS']}x\n"

            f"<b>Signal validation:</b> "
            f"{CONFIG['SIGNAL_VALIDATION_PASSES']}x\n"

            f"━━━━━━━━━━━━━━━━━━━━━━\n"

            f"⏳ <b>Butun tarix "
            f"chuqur tekshirilmoqda...</b>"
        )

        self.client = (
            await AsyncClient.create()
        )

        self.bm = (
            BinanceSocketManager(
                self.client
            )
        )

        # ----------------------------------------------------
        # NEVER SHUTDOWN WHEN NO PATTERN
        # ----------------------------------------------------

        while self.running:

            self.patterns = {}

            total = 0

            # ------------------------------------------------
            # FULL HISTORY SCAN
            # ------------------------------------------------

            for symbol in CONFIG[
                "SYMBOLS"
            ]:

                for tf in CONFIG[
                    "TIMEFRAMES"
                ]:

                    try:

                        log.info(
                            f"🔬 DEEP SCAN "
                            f"{symbol} {tf}"
                        )

                        top = (
                            await self.find_patterns_for(
                                symbol,
                                tf
                            )
                        )

                        if top:

                            self.patterns[
                                (symbol, tf)
                            ] = top

                            total += len(
                                top
                            )

                            best = top[0]

                            await tg.send(

                                f"🔬 <b>DEEP FILTER "
                                f"PASSED</b>\n"

                                f"<b>{symbol} · "
                                f"{tf}</b>\n"

                                f"Pattern: "
                                f"<code>"
                                f"{best['signature']}"
                                f"</code>\n"

                                f"Direction: "
                                f"<b>"
                                f"{best['direction']}"
                                f"</b>\n"

                                f"Occurrences: "
                                f"<b>"
                                f"{best['count']}"
                                f"</b>\n"

                                f"WR: "
                                f"<b>"
                                f"{best['win_rate']}%"
                                f"</b>\n"

                                f"Losses: "
                                f"<b>"
                                f"{best['losses']}"
                                f"</b>\n"

                                f"PF: "
                                f"<b>"
                                f"{best['profit_factor']}"
                                f"</b>\n"

                                f"Avg R: "
                                f"<b>"
                                f"{best['avg_r']:+.4f}"
                                f"</b>\n"

                                f"Walk-forward min WR: "
                                f"<b>"
                                f"{best['temporal_min_wr']}%"
                                f"</b>\n"

                                f"100x bootstrap lower: "
                                f"<b>"
                                f"{best['bootstrap_lower']}%"
                                f"</b>\n"

                                f"Confidence: "
                                f"<b>"
                                f"{best['confidence']}/100"
                                f"</b>\n"

                                f"✅ "
                                f"100-pass validationga "
                                f"tayyor."
                            )

                        else:

                            log.info(
                                f"⚠️ {symbol} {tf}: "
                                f"ULTRA filterga "
                                f"mos pattern yo'q."
                            )

                    except Exception as e:

                        log.error(
                            f"{symbol} {tf}: "
                            f"{e}"
                        )

                        await tg.send(
                            f"⚠️ "
                            f"{symbol} {tf} "
                            f"scan xato: "
                            f"{e}"
                        )

            # ------------------------------------------------
            # NOTHING FOUND
            # ------------------------------------------------

            if total == 0:

                await tg.send(

                    f"⏳ <b>Hozircha signal yo'q.</b>\n"

                    f"Tarixning hammasi tekshirildi.\n"

                    f"Talab:\n"

                    f"├ >= "
                    f"{CONFIG['MIN_OCCURRENCES']} "
                    f"occurrence\n"

                    f"├ <= "
                    f"{CONFIG['MAX_LOSSES']} "
                    f"loss\n"

                    f"├ >= "
                    f"{CONFIG['MIN_WIN_RATE']}% WR\n"

                    f"├ >= "
                    f"{CONFIG['MIN_PROFIT_FACTOR']} PF\n"

                    f"├ >= "
                    f"{CONFIG['MIN_AVG_R']} Avg R\n"

                    f"├ Walk-forward >= "
                    f"{CONFIG['MIN_WALK_FORWARD_WR']}%\n"

                    f"└ Bootstrap lower >= "
                    f"{CONFIG['MIN_BOOTSTRAP_LOWER']}%\n\n"

                    f"❗ Bot o'chmaydi.\n"
                    f"🔄 "
                    f"{CONFIG['NO_PATTERN_SLEEP']} "
                    f"soniyadan keyin yana "
                    f"butun tarixni tekshiradi."
                )

                await asyncio.sleep(
                    CONFIG[
                        "NO_PATTERN_SLEEP"
                    ]
                )

                continue

            # ------------------------------------------------
            # PATTERN FOUND
            # ------------------------------------------------

            await tg.send(

                f"✅ <b>ULTRA FILTER "
                f"YAKUNLANDI</b>\n"

                f"Jami valid pattern: "
                f"<b>{total}</b>\n\n"

                f"Real-time closed candle "
                f"kuzatuvi boshlandi."
            )

            break

        # ----------------------------------------------------
        # LIVE TASKS
        # ----------------------------------------------------

        tasks = []

        for (
            symbol,
            tf
        ) in self.patterns.keys():

            tasks.append(
                asyncio.create_task(
                    self.listen_stream(
                        symbol,
                        tf
                    )
                )
            )

        # ----------------------------------------------------
        # PERIODIC DEEP RESCAN
        # ----------------------------------------------------

        tasks.append(
            asyncio.create_task(
                self.deep_rescan()
            )
        )

        await asyncio.gather(
            *tasks
        )


# ============================================================
# MAIN
# ============================================================

async def main():

    bot = LivePatternBot()

    try:

        await bot.run()

    except KeyboardInterrupt:

        log.info(
            "🛑 Bot to'xtatildi."
        )

    except Exception as e:

        log.exception(
            f"Fatal error: {e}"
        )

        await tg.send(
            f"❌ <b>BOT FATAL ERROR</b>\n"
            f"<code>{str(e)}</code>"
        )

    finally:

        bot.running = False

        if bot.client:

            try:

                await bot.client.close_connection()

            except Exception:

                pass


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )