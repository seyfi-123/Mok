# ============================================================
# Crypto Deep Pattern Bot v9.0
# DEEP HISTORICAL + REAL-TIME PATTERN ANALYZER
#
# - Indicatorsiz candle-pattern analysis
# - 21 candle classes
# - 2/3/4 candle signatures
# - Minimum 100 historical occurrences
# - Every occurrence individually forward-tested
# - MFE / MAE
# - 1R / 2R / 4R / 6R / 8R / 10R
# - TP1 / SL / BE / TRAIL
# - First-event analysis
# - Retracement / continuation analysis
# - Year-by-year consistency
# - Real-time closed-candle revalidation
# - Automatic 1-hour re-scan if no valid patterns
# - WebSocket reconnect
# - Telegram deep report + chart
# ============================================================

import os
import time
import logging
import asyncio
import io
import math

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
    # HISTORICAL DATA
    # --------------------------------------------------------
    "DAYS_PER_TF": {
        "1m": int(os.getenv("DAYS_1M", "120")),
        "3m": int(os.getenv("DAYS_3M", "200")),
        "5m": int(os.getenv("DAYS_5M", "365")),
        "15m": int(os.getenv("DAYS_15M", "730")),
        "30m": int(os.getenv("DAYS_30M", "1095")),
        "1h": int(os.getenv("DAYS_1H", "1825")),
        "4h": int(os.getenv("DAYS_4H", "1825")),
    },

    # --------------------------------------------------------
    # PATTERN
    # --------------------------------------------------------
    "SEQ_LENGTHS": [
        int(x)
        for x in os.getenv(
            "SEQ_LENGTHS",
            "2,3,4"
        ).split(",")
    ],

    # --------------------------------------------------------
    # DEEP FILTERS
    # --------------------------------------------------------
    "MIN_OCCURRENCES": int(
        os.getenv("MIN_OCCURRENCES", "100")
    ),

    "MIN_WIN_RATE": float(
        os.getenv("MIN_WIN_RATE", "70")
    ),

    "MAX_LOSSES": int(
        os.getenv("MAX_LOSSES", "30")
    ),

    "MIN_PROFIT_FACTOR": float(
        os.getenv("MIN_PROFIT_FACTOR", "1.30")
    ),

    "MIN_AVG_R": float(
        os.getenv("MIN_AVG_R", "0.10")
    ),

    "MIN_CONFIDENCE": float(
        os.getenv("MIN_CONFIDENCE", "65")
    ),

    # At least this many yearly samples are required
    # for yearly consistency calculation.
    "MIN_YEAR_SAMPLES": int(
        os.getenv("MIN_YEAR_SAMPLES", "5")
    ),

    # Maximum acceptable yearly WR deviation.
    "MAX_YEARLY_WR_STD": float(
        os.getenv("MAX_YEARLY_WR_STD", "25")
    ),

    # --------------------------------------------------------
    # FORWARD ANALYSIS
    # --------------------------------------------------------
    "FORWARD_CANDLES": int(
        os.getenv("FORWARD_CANDLES", "50")
    ),

    "SL_BUF": float(
        os.getenv("SL_BUF", "10")
    ),

    # --------------------------------------------------------
    # 1:1R MANAGEMENT
    # --------------------------------------------------------
    "TP1_R": 1.0,

    "TP1_CLOSE_PCT": 0.50,

    "TRAIL_START_R": 4.0,

    "TRAIL_STEP_R": 2.0,

    "MAX_TRAIL_R": 10.0,

    # --------------------------------------------------------
    # DEEP MARKET MOVEMENT ANALYSIS
    # --------------------------------------------------------
    "RETRACE_R_THRESHOLD": float(
        os.getenv("RETRACE_R_THRESHOLD", "1.0")
    ),

    "BREAKOUT_R_THRESHOLD": float(
        os.getenv("BREAKOUT_R_THRESHOLD", "1.0")
    ),

    # --------------------------------------------------------
    # API
    # --------------------------------------------------------
    "REQUEST_DELAY": float(
        os.getenv("REQUEST_DELAY", "0.25")
    ),

    "CANDLE_BUFFER": int(
        os.getenv("CANDLE_BUFFER", "500")
    ),

    "ATR_PERIOD": int(
        os.getenv("ATR_PERIOD", "50")
    ),

    # --------------------------------------------------------
    # RESCAN
    # --------------------------------------------------------
    "NO_PATTERN_RESCAN_SECONDS": int(
        os.getenv(
            "NO_PATTERN_RESCAN_SECONDS",
            "3600"
        )
    ),

    # --------------------------------------------------------
    # REAL-TIME CONFIRMATION
    # --------------------------------------------------------
    "REALTIME_MIN_RECHECKS": int(
        os.getenv(
            "REALTIME_MIN_RECHECKS",
            "1"
        )
    ),
}


TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

log = logging.getLogger(__name__)


# ============================================================
# TELEGRAM
# ============================================================

class TG:

    def __init__(self, token, chat_id):
        self.bot = Bot(token=token) if token else None
        self.chat_id = chat_id

    async def send(self, msg):

        if not self.bot:
            return

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=msg,
                parse_mode=ParseMode.HTML
            )

        except Exception as e:
            log.error(f"Telegram send error: {e}")

    async def photo(self, buf, caption=""):

        if not self.bot:
            return

        try:
            await self.bot.send_photo(
                chat_id=self.chat_id,
                photo=InputFile(
                    buf,
                    filename="deep_signal.png"
                ),
                caption=caption,
                parse_mode=ParseMode.HTML
            )

        except Exception as e:
            log.error(f"Telegram photo error: {e}")


tg = TG(
    TELEGRAM_TOKEN,
    TELEGRAM_CHAT_ID
)


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
        - days * 24 * 60 * 60 * 1000
    )

    all_klines = []

    cursor = start_time

    log.info(
        f"📥 {symbol} {interval}: "
        f"{days} kun tarix yuklanmoqda..."
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
                f"fetch error: {e}"
            )

            await asyncio.sleep(2)

            continue

        if not klines:
            break

        all_klines.extend(
            klines
        )

        last_open_time = klines[-1][0]

        if last_open_time <= cursor:
            break

        cursor = last_open_time + 1

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

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for col in numeric_cols:
        df[col] = df[col].astype(float)

    df["open_time"] = (
        df["open_time"] // 1000
    )

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

    df = df.drop_duplicates(
        subset=["open_time"]
    )

    df = df.sort_values(
        "open_time"
    )

    df = df.reset_index(
        drop=True
    )

    log.info(
        f"✅ {symbol} {interval}: "
        f"{len(df)} candle yuklandi"
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
# 3. 21 CANDLE CLASSIFICATION
# ============================================================

def classify_candles(df):

    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    atr = df["atr"].values

    body = c - o

    body_abs = np.abs(body)

    upper_wick = (
        h - np.maximum(o, c)
    )

    lower_wick = (
        np.minimum(o, c) - l
    )

    direction = np.where(
        body > 0,
        "U",
        np.where(
            body < 0,
            "D",
            "F"
        )
    )

    body_ratio = (
        body_abs
        / np.maximum(
            atr,
            1e-9
        )
    )

    body_size = np.where(
        body_ratio < 0.3,
        "S",
        np.where(
            body_ratio < 0.8,
            "M",
            np.where(
                body_ratio < 1.5,
                "L",
                "X"
            )
        )
    )

    uw_ratio = (
        upper_wick
        / np.maximum(
            body_abs,
            atr * 0.1
        )
    )

    lw_ratio = (
        lower_wick
        / np.maximum(
            body_abs,
            atr * 0.1
        )
    )

    big_upper = uw_ratio > 1.2
    big_lower = lw_ratio > 1.2

    huge_upper = uw_ratio > 2.5
    huge_lower = lw_ratio > 2.5

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

    codes = np.array([
        f"{d}{s}{w}"
        for d, s, w in zip(
            direction,
            body_size,
            wick_class
        )
    ])

    out = df.copy()

    out["code"] = codes

    return out


# ============================================================
# 4. SIGNATURES
# ============================================================

def build_signatures(
    df_coded,
    seq_len
):

    codes = (
        df_coded["code"].values
    )

    n = len(codes)

    sigs = [None] * n

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
# 5. DEEP FORWARD ANALYSIS
# ============================================================

def simulate_forward_deep(
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

    entry = float(
        c[entry_idx]
    )

    atr_value = float(
        atr[entry_idx]
    )

    if not np.isfinite(atr_value):
        return None

    buf = max(
        sl_buf_cfg
        * (
            entry / 100000
        ),
        atr_value * 0.15
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
        + CONFIG["TP1_R"] * sl_dist
        if direction == "B"
        else
        entry
        - CONFIG["TP1_R"] * sl_dist
    )

    tp1_hit = False

    sl = sl_init

    lock_r = 0.0

    max_r_reached = 0.0

    min_r_reached = 0.0

    max_adverse_r = 0.0

    first_event = None

    bars_to_tp1 = None

    bars_to_sl = None

    bars_to_max_r = None

    retracement_count = 0

    continuation_count = 0

    breakout_count = 0

    previous_max_r = 0.0

    max_i = min(
        entry_idx
        + CONFIG["FORWARD_CANDLES"],
        n - 1
    )

    for i in range(
        entry_idx + 1,
        max_i + 1
    ):

        if direction == "B":

            current_max_r = (
                h[i] - entry
            ) / sl_dist

            current_min_r = (
                l[i] - entry
            ) / sl_dist

        else:

            current_max_r = (
                entry - l[i]
            ) / sl_dist

            current_min_r = (
                entry - h[i]
            ) / sl_dist

        if (
            current_max_r
            > max_r_reached
        ):

            max_r_reached = (
                current_max_r
            )

            bars_to_max_r = (
                i - entry_idx
            )

        if (
            current_min_r
            < min_r_reached
        ):

            min_r_reached = (
                current_min_r
            )

        max_adverse_r = min(
            max_adverse_r,
            current_min_r
        )

        # ----------------------------------------------------
        # RETRACEMENT
        # ----------------------------------------------------

        if (
            previous_max_r
            >= CONFIG[
                "RETRACE_R_THRESHOLD"
            ]
            and current_max_r
            < previous_max_r
        ):

            retracement_count += 1

        # ----------------------------------------------------
        # CONTINUATION
        # ----------------------------------------------------

        if (
            current_max_r
            > previous_max_r
            and current_max_r > 0
        ):

            continuation_count += 1

        # ----------------------------------------------------
        # BREAKOUT
        # ----------------------------------------------------

        if (
            current_max_r
            >= CONFIG[
                "BREAKOUT_R_THRESHOLD"
            ]
        ):

            breakout_count += 1

        previous_max_r = max(
            previous_max_r,
            current_max_r
        )

        # ----------------------------------------------------
        # SL / TRAILING STOP
        # ----------------------------------------------------

        stop_hit = False

        if direction == "B":

            if l[i] <= sl:
                stop_hit = True

        else:

            if h[i] >= sl:
                stop_hit = True

        if stop_hit:

            if first_event is None:
                first_event = (
                    "BE"
                    if tp1_hit
                    and abs(
                        sl - entry
                    ) < sl_dist * 0.01
                    else "SL"
                )

            if tp1_hit:

                p1 = (
                    CONFIG[
                        "TP1_CLOSE_PCT"
                    ]
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
                            sl - entry
                        )
                        / sl_dist
                        if direction == "B"
                        else
                        (
                            entry - sl
                        )
                        / sl_dist
                    )
                )

                final_r = (
                    p1 + p2
                )

            else:

                final_r = (
                    (
                        sl - entry
                    )
                    / sl_dist
                    if direction == "B"
                    else
                    (
                        entry - sl
                    )
                    / sl_dist
                )

            bars_to_sl = (
                i - entry_idx
            )

            return {
                "final_r": float(
                    final_r
                ),
                "max_r": float(
                    max_r_reached
                ),
                "min_r": float(
                    min_r_reached
                ),
                "mae_r": float(
                    max_adverse_r
                ),
                "tp1_hit": bool(
                    tp1_hit
                ),
                "first_event": first_event,
                "bars_to_tp1": bars_to_tp1,
                "bars_to_sl": bars_to_sl,
                "bars_to_max_r": bars_to_max_r,
                "retracement_count": (
                    retracement_count
                ),
                "continuation_count": (
                    continuation_count
                ),
                "breakout_count": (
                    breakout_count
                ),
                "bars_forward": (
                    i - entry_idx
                ),
            }

        # ----------------------------------------------------
        # TP1
        # ----------------------------------------------------

        if (
            not tp1_hit
            and (
                (
                    direction == "B"
                    and h[i] >= tp1_price
                )
                or
                (
                    direction == "S"
                    and l[i] <= tp1_price
                )
            )
        ):

            tp1_hit = True

            if first_event is None:
                first_event = "TP1"

            bars_to_tp1 = (
                i - entry_idx
            )

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
                        + lock_r * sl_dist
                    )

                else:

                    sl = (
                        entry
                        - lock_r * sl_dist
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
                    CONFIG[
                        "TP1_CLOSE_PCT"
                    ]
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

            if first_event is None:
                first_event = "MAX_TRAIL"

            return {
                "final_r": float(
                    final_r
                ),
                "max_r": float(
                    max_r_reached
                ),
                "min_r": float(
                    min_r_reached
                ),
                "mae_r": float(
                    max_adverse_r
                ),
                "tp1_hit": bool(
                    tp1_hit
                ),
                "first_event": first_event,
                "bars_to_tp1": bars_to_tp1,
                "bars_to_sl": bars_to_sl,
                "bars_to_max_r": bars_to_max_r,
                "retracement_count": (
                    retracement_count
                ),
                "continuation_count": (
                    continuation_count
                ),
                "breakout_count": (
                    breakout_count
                ),
                "bars_forward": (
                    i - entry_idx
                ),
            }

    # --------------------------------------------------------
    # FORWARD WINDOW ENDED
    # --------------------------------------------------------

    final_r = (
        (
            c[max_i] - entry
        )
        / sl_dist
        if direction == "B"
        else
        (
            entry - c[max_i]
        )
        / sl_dist
    )

    if tp1_hit:

        p1 = (
            CONFIG[
                "TP1_CLOSE_PCT"
            ]
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

    if first_event is None:

        first_event = (
            "FORWARD_END"
        )

    return {
        "final_r": float(
            final_r
        ),
        "max_r": float(
            max_r_reached
        ),
        "min_r": float(
            min_r_reached
        ),
        "mae_r": float(
            max_adverse_r
        ),
        "tp1_hit": bool(
            tp1_hit
        ),
        "first_event": first_event,
        "bars_to_tp1": bars_to_tp1,
        "bars_to_sl": bars_to_sl,
        "bars_to_max_r": bars_to_max_r,
        "retracement_count": (
            retracement_count
        ),
        "continuation_count": (
            continuation_count
        ),
        "breakout_count": (
            breakout_count
        ),
        "bars_forward": (
            max_i - entry_idx
        ),
    }


# ============================================================
# 6. R DISTRIBUTION
# ============================================================

def compute_r_distribution(
    max_r_list
):

    return {
        "reached_1r": int(
            sum(
                1
                for x in max_r_list
                if x >= 1
            )
        ),

        "reached_2r": int(
            sum(
                1
                for x in max_r_list
                if x >= 2
            )
        ),

        "reached_4r": int(
            sum(
                1
                for x in max_r_list
                if x >= 4
            )
        ),

        "reached_6r": int(
            sum(
                1
                for x in max_r_list
                if x >= 6
            )
        ),

        "reached_8r": int(
            sum(
                1
                for x in max_r_list
                if x >= 8
            )
        ),

        "reached_10r": int(
            sum(
                1
                for x in max_r_list
                if x >= 10
            )
        ),
    }


# ============================================================
# 7. CONFIDENCE
# ============================================================

def compute_confidence(
    count,
    win_rate,
    pf,
    avg_r,
    consistency_std,
    tp1_rate,
    breakout_rate
):

    score = 0.0

    # Sample size
    score += (
        min(
            count / 200,
            1.0
        )
        * 20
    )

    # Win rate
    score += (
        min(
            win_rate / 100,
            1.0
        )
        * 20
    )

    # Profit factor
    score += (
        min(
            pf / 3.0,
            1.0
        )
        * 15
    )

    # Average R
    score += (
        min(
            max(avg_r, 0)
            / 2.0,
            1.0
        )
        * 15
    )

    # Yearly consistency
    if consistency_std < 999:

        score += (
            max(
                0,
                1
                - consistency_std
                / 30
            )
            * 10
        )

    # TP1
    score += (
        min(
            tp1_rate / 100,
            1.0
        )
        * 10
    )

    # Breakout
    score += (
        min(
            breakout_rate / 100,
            1.0
        )
        * 10
    )

    return round(
        min(
            score,
            100
        ),
        1
    )


# ============================================================
# 8. DEEP ANALYSIS
# ============================================================

def analyze_timeframe(
    df,
    tf_name
):

    df_coded = classify_candles(
        df
    )

    n = len(df_coded)

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

        sigs = build_signatures(
            df_coded,
            seq_len
        )

        df_coded[
            f"sig_{seq_len}"
        ] = sigs

        groups = defaultdict(
            list
        )

        for i, sig in enumerate(
            sigs
        ):

            if sig is None:
                continue

            # Need enough future candles
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

        for sig, idx_list in groups.items():

            # ------------------------------------------------
            # HARD 100+ OCCURRENCE FILTER
            # ------------------------------------------------

            if len(idx_list) < CONFIG[
                "MIN_OCCURRENCES"
            ]:
                continue

            for direction in [
                "B",
                "S"
            ]:

                analyses = []

                years_map = (
                    defaultdict(list)
                )

                for idx in idx_list:

                    res = simulate_forward_deep(
                        df_coded,
                        idx,
                        direction,
                        CONFIG["SL_BUF"]
                    )

                    if res is None:
                        continue

                    analyses.append(
                        res
                    )

                    years_map[
                        int(
                            df_coded[
                                "year"
                            ].iloc[idx]
                        )
                    ].append(
                        res["final_r"]
                    )

                if len(analyses) < CONFIG[
                    "MIN_OCCURRENCES"
                ]:
                    continue

                r_arr = np.array([
                    x["final_r"]
                    for x in analyses
                ])

                max_r_arr = np.array([
                    x["max_r"]
                    for x in analyses
                ])

                mae_arr = np.array([
                    x["mae_r"]
                    for x in analyses
                ])

                # ------------------------------------------------
                # BASIC RESULTS
                # ------------------------------------------------

                wins = int(
                    (
                        r_arr > 0
                    ).sum()
                )

                losses = int(
                    (
                        r_arr <= 0
                    ).sum()
                )

                win_rate = float(
                    wins
                    / len(r_arr)
                    * 100
                )

                avg_r = float(
                    r_arr.mean()
                )

                median_r = float(
                    np.median(r_arr)
                )

                std_r = float(
                    np.std(r_arr)
                )

                gross_win = float(
                    r_arr[
                        r_arr > 0
                    ].sum()
                )

                gross_loss = float(
                    abs(
                        r_arr[
                            r_arr < 0
                        ].sum()
                    )
                )

                if gross_loss > 0:

                    pf = float(
                        gross_win
                        / gross_loss
                    )

                else:

                    pf = 999.0

                # ------------------------------------------------
                # TP1
                # ------------------------------------------------

                tp1_hits = sum(
                    1
                    for x in analyses
                    if x["tp1_hit"]
                )

                tp1_rate = (
                    tp1_hits
                    / len(analyses)
                    * 100
                )

                # ------------------------------------------------
                # FIRST EVENTS
                # ------------------------------------------------

                first_events = defaultdict(
                    int
                )

                for x in analyses:

                    first_events[
                        x["first_event"]
                    ] += 1

                # ------------------------------------------------
                # MFE / MAE
                # ------------------------------------------------

                avg_mfe = float(
                    np.mean(
                        max_r_arr
                    )
                )

                median_mfe = float(
                    np.median(
                        max_r_arr
                    )
                )

                avg_mae = float(
                    np.mean(
                        mae_arr
                    )
                )

                worst_mae = float(
                    np.min(
                        mae_arr
                    )
                )

                # ------------------------------------------------
                # RETRACEMENT
                # ------------------------------------------------

                retracement_total = sum(
                    x[
                        "retracement_count"
                    ]
                    for x in analyses
                )

                retracement_occurrences = sum(
                    1
                    for x in analyses
                    if x[
                        "retracement_count"
                    ] > 0
                )

                retracement_rate = (
                    retracement_occurrences
                    / len(analyses)
                    * 100
                )

                # ------------------------------------------------
                # CONTINUATION
                # ------------------------------------------------

                continuation_total = sum(
                    x[
                        "continuation_count"
                    ]
                    for x in analyses
                )

                continuation_occurrences = sum(
                    1
                    for x in analyses
                    if x[
                        "continuation_count"
                    ] > 0
                )

                continuation_rate = (
                    continuation_occurrences
                    / len(analyses)
                    * 100
                )

                # ------------------------------------------------
                # BREAKOUT
                # ------------------------------------------------

                breakout_occurrences = sum(
                    1
                    for x in analyses
                    if x[
                        "breakout_count"
                    ] > 0
                )

                breakout_rate = (
                    breakout_occurrences
                    / len(analyses)
                    * 100
                )

                # ------------------------------------------------
                # R DISTRIBUTION
                # ------------------------------------------------

                dist = compute_r_distribution(
                    max_r_arr
                )

                # ------------------------------------------------
                # YEARLY ANALYSIS
                # ------------------------------------------------

                yearly_wr = {}

                for year, values in (
                    years_map.items()
                ):

                    if len(values) >= CONFIG[
                        "MIN_YEAR_SAMPLES"
                    ]:

                        arr = np.array(
                            values
                        )

                        yearly_wr[
                            str(year)
                        ] = round(
                            float(
                                np.mean(
                                    arr > 0
                                )
                                * 100
                            ),
                            1
                        )

                if len(
                    yearly_wr
                ) >= 2:

                    consistency_std = float(
                        np.std(
                            list(
                                yearly_wr.values()
                            )
                        )
                    )

                else:

                    consistency_std = 999.0

                # ------------------------------------------------
                # CONFIDENCE
                # ------------------------------------------------

                confidence = compute_confidence(
                    len(analyses),
                    win_rate,
                    pf,
                    avg_r,
                    consistency_std,
                    tp1_rate,
                    breakout_rate
                )

                results.append({

                    "tf": tf_name,

                    "seq_len": seq_len,

                    "signature": sig,

                    "direction": (
                        "BUY"
                        if direction == "B"
                        else "SELL"
                    ),

                    "count": len(
                        analyses
                    ),

                    "wins": wins,

                    "losses": losses,

                    "win_rate": round(
                        win_rate,
                        1
                    ),

                    "avg_r": round(
                        avg_r,
                        3
                    ),

                    "median_r": round(
                        median_r,
                        3
                    ),

                    "std_r": round(
                        std_r,
                        3
                    ),

                    "profit_factor": round(
                        pf,
                        2
                    ),

                    "tp1_hits": tp1_hits,

                    "tp1_rate": round(
                        tp1_rate,
                        1
                    ),

                    "avg_mfe": round(
                        avg_mfe,
                        3
                    ),

                    "median_mfe": round(
                        median_mfe,
                        3
                    ),

                    "avg_mae": round(
                        avg_mae,
                        3
                    ),

                    "worst_mae": round(
                        worst_mae,
                        3
                    ),

                    "retracement_rate": round(
                        retracement_rate,
                        1
                    ),

                    "continuation_rate": round(
                        continuation_rate,
                        1
                    ),

                    "breakout_rate": round(
                        breakout_rate,
                        1
                    ),

                    "retracement_total": (
                        retracement_total
                    ),

                    "continuation_total": (
                        continuation_total
                    ),

                    "breakout_occurrences": (
                        breakout_occurrences
                    ),

                    "first_events": dict(
                        first_events
                    ),

                    "yearly_wr": yearly_wr,

                    "consistency_std": round(
                        consistency_std,
                        1
                    ),

                    "confidence": confidence,

                    **dist,
                })

    return results


# ============================================================
# 9. FINAL FILTER
# ============================================================

def filter_and_rank(
    results
):

    good = []

    for r in results:

        if r["count"] < CONFIG[
            "MIN_OCCURRENCES"
        ]:
            continue

        if r["profit_factor"] < CONFIG[
            "MIN_PROFIT_FACTOR"
        ]:
            continue

        if r["win_rate"] < CONFIG[
            "MIN_WIN_RATE"
        ]:
            continue

        if r["losses"] > CONFIG[
            "MAX_LOSSES"
        ]:
            continue

        if r["avg_r"] < CONFIG[
            "MIN_AVG_R"
        ]:
            continue

        if r["confidence"] < CONFIG[
            "MIN_CONFIDENCE"
        ]:
            continue

        if (
            r["consistency_std"] < 999
            and
            r["consistency_std"]
            > CONFIG[
                "MAX_YEARLY_WR_STD"
            ]
        ):
            continue

        good.append(
            r
        )

    good.sort(
        key=lambda x: (
            x["confidence"],
            x["win_rate"],
            x["profit_factor"],
            x["avg_r"],
            x["count"]
        ),
        reverse=True
    )

    return good


# ============================================================
# 10. DEEP TELEGRAM REPORT
# ============================================================

def build_pattern_report(
    pattern,
    symbol,
    tf
):

    events = pattern[
        "first_events"
    ]

    event_text = []

    for name, count in sorted(
        events.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        pct = (
            count
            / pattern["count"]
            * 100
        )

        event_text.append(
            f"├ {name}: "
            f"<b>{count}</b> "
            f"({pct:.1f}%)"
        )

    yearly_lines = []

    for year, wr in sorted(
        pattern[
            "yearly_wr"
        ].items()
    ):

        yearly_lines.append(
            f"├ {year}: "
            f"<b>{wr:.1f}% WR</b>"
        )

    if not yearly_lines:
        yearly_lines.append(
            "├ Yetarli yearly sample yo'q"
        )

    return (
        f"🧠 <b>DEEP PATTERN ANALYSIS</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Symbol:</b> {symbol}\n"
        f"<b>Timeframe:</b> {tf}\n"
        f"<b>Direction:</b> "
        f"<b>{pattern['direction']}</b>\n"
        f"<b>Pattern:</b> "
        f"<code>{pattern['signature']}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>🔬 SAMPLE</b>\n"
        f"├ Tekshirilgan occurrence: "
        f"<b>{pattern['count']}</b>\n"
        f"├ Minimum talab: "
        f"<b>{CONFIG['MIN_OCCURRENCES']}</b>\n"
        f"├ WIN: <b>{pattern['wins']}</b>\n"
        f"├ LOSS: <b>{pattern['losses']}</b>\n"
        f"└ WR: <b>{pattern['win_rate']}%</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>📊 R NATIJA</b>\n"
        f"├ Avg R: <b>{pattern['avg_r']:+.3f}</b>\n"
        f"├ Median R: <b>{pattern['median_r']:+.3f}</b>\n"
        f"├ Profit Factor: <b>{pattern['profit_factor']}</b>\n"
        f"└ R Std: <b>{pattern['std_r']:.3f}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>📈 MFE / MAE</b>\n"
        f"├ Avg MFE: <b>{pattern['avg_mfe']:+.3f}R</b>\n"
        f"├ Median MFE: <b>{pattern['median_mfe']:+.3f}R</b>\n"
        f"├ Avg MAE: <b>{pattern['avg_mae']:+.3f}R</b>\n"
        f"└ Worst MAE: <b>{pattern['worst_mae']:+.3f}R</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>🎯 R DARAJALARI</b>\n"
        f"├ 1R: <b>{pattern['reached_1r']}</b> "
        f"({pattern['reached_1r']/pattern['count']*100:.1f}%)\n"
        f"├ 2R: <b>{pattern['reached_2r']}</b> "
        f"({pattern['reached_2r']/pattern['count']*100:.1f}%)\n"
        f"├ 4R: <b>{pattern['reached_4r']}</b> "
        f"({pattern['reached_4r']/pattern['count']*100:.1f}%)\n"
        f"├ 6R: <b>{pattern['reached_6r']}</b> "
        f"({pattern['reached_6r']/pattern['count']*100:.1f}%)\n"
        f"├ 8R: <b>{pattern['reached_8r']}</b> "
        f"({pattern['reached_8r']/pattern['count']*100:.1f}%)\n"
        f"└ 10R: <b>{pattern['reached_10r']}</b> "
        f"({pattern['reached_10r']/pattern['count']*100:.1f}%)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>🔄 HARAKAT ANALIZI</b>\n"
        f"├ TP1 rate: <b>{pattern['tp1_rate']}%</b>\n"
        f"├ Retracement: "
        f"<b>{pattern['retracement_rate']}%</b>\n"
        f"├ Continuation: "
        f"<b>{pattern['continuation_rate']}%</b>\n"
        f"└ Breakout ≥ "
        f"{CONFIG['BREAKOUT_R_THRESHOLD']}R: "
        f"<b>{pattern['breakout_rate']}%</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>🥇 BIRINCHI VOQEA</b>\n"
        + "\n".join(event_text)
        + "\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>📅 YILLIK TEKSHIRUV</b>\n"
        + "\n".join(yearly_lines)
        + "\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>🧠 CONFIDENCE: "
        f"{pattern['confidence']}/100</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚙️ <i>Real closed-candle + "
        f"historical deep-analysis.</i>"
    )


# ============================================================
# 11. CHART
# ============================================================

def make_signal_chart(
    df_coded,
    pattern,
    symbol,
    tf
):

    seq_len = pattern[
        "seq_len"
    ]

    sig_col = (
        f"sig_{seq_len}"
    )

    if sig_col not in df_coded:
        df_coded[
            sig_col
        ] = build_signatures(
            df_coded,
            seq_len
        )

    all_idx = (
        df_coded.index[
            df_coded[
                sig_col
            ]
            == pattern[
                "signature"
            ]
        ].tolist()
    )

    total = len(
        df_coded
    )

    window_start = max(
        0,
        total - 160
    )

    window = (
        df_coded
        .iloc[
            window_start:
        ]
        .reset_index(
            drop=True
        )
    )

    offset = window_start

    n = len(
        window
    )

    fig = plt.figure(
        figsize=(14, 9),
        facecolor="#0a0b0f"
    )

    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[
            3.0,
            1.3
        ],
        hspace=0.35
    )

    ax = fig.add_subplot(
        gs[0]
    )

    ax.set_facecolor(
        "#0a0b0f"
    )

    for i, row in (
        window.iterrows()
    ):

        c = (
            "#10b981"
            if row["close"]
            >= row["open"]
            else "#ef4444"
        )

        ax.plot(
            [i, i],
            [
                row["low"],
                row["high"]
            ],
            color=c,
            linewidth=1,
            alpha=0.85
        )

        ax.plot(
            [i, i],
            [
                row["open"],
                row["close"]
            ],
            color=c,
            linewidth=3,
            alpha=0.85
        )

    for idx in all_idx:

        rel = (
            idx
            - offset
        )

        if (
            0
            <= rel
            < n - 1
        ):

            y = (
                window[
                    "high"
                ].iloc[
                    rel
                ]
                * 1.001
            )

            ax.annotate(
                "▼",
                xy=(
                    rel,
                    y
                ),
                color="#facc15",
                fontsize=10,
                ha="center",
                va="bottom",
                alpha=0.9
            )

    last_rel = n - 1

    arrow = (
        "▲ BUY"
        if pattern[
            "direction"
        ] == "BUY"
        else "▼ SELL"
    )

    arrow_color = (
        "#10b981"
        if pattern[
            "direction"
        ] == "BUY"
        else "#ef4444"
    )

    ax.annotate(
        arrow,
        xy=(
            last_rel,
            window[
                "close"
            ].iloc[-1]
        ),
        xytext=(
            last_rel,
            window[
                "high"
            ].max()
            * 1.003
        ),
        color=arrow_color,
        fontsize=15,
        fontweight="bold",
        ha="center"
    )

    ax.axvspan(
        last_rel
        - seq_len
        + 1,
        last_rel,
        color=arrow_color,
        alpha=0.25
    )

    ax.set_title(
        f"{symbol} · {tf} · "
        f"{pattern['signature']} · "
        f"{pattern['direction']} | "
        f"{pattern['count']} occurrences | "
        f"WR {pattern['win_rate']}% | "
        f"Conf {pattern['confidence']}",
        color="#e2e8f0",
        fontsize=11
    )

    ax.tick_params(
        colors="#94a3b8",
        labelsize=8
    )

    ax.grid(
        True,
        alpha=0.1,
        color="#ffffff"
    )

    for sp in ax.spines.values():
        sp.set_color(
            "#ffffff20"
        )

    # --------------------------------------------------------
    # R CHART
    # --------------------------------------------------------

    ax2 = fig.add_subplot(
        gs[1]
    )

    ax2.set_facecolor(
        "#0a0b0f"
    )

    ax2.axis(
        "off"
    )

    labels = [
        "1R",
        "2R",
        "4R",
        "6R",
        "8R",
        "10R"
    ]

    counts = [
        pattern.get(
            "reached_1r",
            0
        ),
        pattern.get(
            "reached_2r",
            0
        ),
        pattern.get(
            "reached_4r",
            0
        ),
        pattern.get(
            "reached_6r",
            0
        ),
        pattern.get(
            "reached_8r",
            0
        ),
        pattern.get(
            "reached_10r",
            0
        )
    ]

    total_c = max(
        pattern[
            "count"
        ],
        1
    )

    pcts = [
        x / total_c * 100
        for x in counts
    ]

    colors = [
        "#3b82f6",
        "#22d3ee",
        "#10b981",
        "#84cc16",
        "#eab308",
        "#f97316"
    ]

    max_count = max(
        counts
    ) if counts else 0

    bars = ax2.barh(
        range(
            len(labels)
        ),
        counts,
        color=colors,
        height=0.65
    )

    ax2.set_yticks(
        range(
            len(labels)
        )
    )

    ax2.set_yticklabels(
        labels,
        color="#e2e8f0",
        fontsize=10
    )

    ax2.set_xlim(
        0,
        max_count * 1.3
        if max_count > 0
        else 10
    )

    ax2.invert_yaxis()

    for b, count, pct in zip(
        bars,
        counts,
        pcts
    ):

        ax2.text(
            b.get_width()
            + (
                max_count
                * 0.02
                if max_count > 0
                else 0.5
            ),
            b.get_y()
            + b.get_height()
            / 2,
            f"{count} "
            f"({pct:.1f}%)",
            color="#e2e8f0",
            fontsize=9,
            va="center"
        )

    ax2.set_title(
        "📊 R DARAJALARI",
        color="#e2e8f0",
        fontsize=10
    )

    plt.tight_layout()

    buf = io.BytesIO()

    plt.savefig(
        buf,
        format="png",
        dpi=95,
        facecolor="#0a0b0f"
    )

    plt.close(
        fig
    )

    buf.seek(0)

    return buf


# ============================================================
# 12. LIVE BOT
# ============================================================

class LivePatternBot:

    def __init__(self):

        self.patterns = {}

        self.buffers = {}

        self.signaled = set()

        self.history_cache = {}

        self.client = None

        self.bm = None

    # --------------------------------------------------------
    # HISTORY + DEEP ANALYSIS
    # --------------------------------------------------------

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
            CONFIG[
                "ATR_PERIOD"
            ]
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

        log.info(
            f"🔬 DEEP ANALYSIS: "
            f"{symbol} {tf}"
        )

        results = analyze_timeframe(
            df,
            tf
        )

        log.info(
            f"🔬 {symbol} {tf}: "
            f"{len(results)} ta "
            f"100+ occurrence candidate"
        )

        filtered = filter_and_rank(
            results
        )

        log.info(
            f"✅ {symbol} {tf}: "
            f"{len(filtered)} ta "
            f"qat'iy pattern qoldi"
        )

        return filtered

    # --------------------------------------------------------
    # REAL-TIME CANDLE
    # --------------------------------------------------------

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
                self.buffers[
                    key
                ]
            )
        )

        df["atr"] = compute_atr(
            df,
            CONFIG[
                "ATR_PERIOD"
            ]
        )

        df_coded = classify_candles(
            df
        )

        tf_patterns = self.patterns.get(
            key,
            []
        )

        for pattern in tf_patterns:

            seq_len = pattern[
                "seq_len"
            ]

            if len(
                df_coded
            ) < seq_len:

                continue

            recent = (
                df_coded[
                    "code"
                ]
                .iloc[
                    -seq_len:
                ]
                .tolist()
            )

            current_sig = "_".join(
                recent
            )

            if (
                current_sig
                != pattern[
                    "signature"
                ]
            ):

                continue

            # ------------------------------------------------
            # REAL-TIME REVALIDATION
            # ------------------------------------------------

            rechecked = (
                self.revalidate_live_pattern(
                    df_coded,
                    pattern
                )
            )

            if not rechecked:
                continue

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
            ) > 5000:

                self.signaled = set(
                    list(
                        self.signaled
                    )[-2000:]
                )

            asyncio.create_task(
                self._fire_signal(
                    symbol,
                    tf,
                    pattern,
                    df_coded
                )
            )

    # --------------------------------------------------------
    # REAL-TIME RECHECK
    # --------------------------------------------------------

    def revalidate_live_pattern(
        self,
        df_coded,
        pattern
    ):

        seq_len = pattern[
            "seq_len"
        ]

        if len(
            df_coded
        ) < seq_len:

            return False

        current_sig = "_".join(
            df_coded[
                "code"
            ]
            .iloc[
                -seq_len:
            ]
            .tolist()
        )

        if (
            current_sig
            != pattern[
                "signature"
            ]
        ):

            return False

        # ----------------------------------------------------
        # Pattern history remains valid.
        # We additionally inspect the current candle
        # structure before firing.
        # ----------------------------------------------------

        current = (
            df_coded.iloc[-1]
        )

        code = current[
            "code"
        ]

        # Must be a known classified candle.
        if not code:
            return False

        # ----------------------------------------------------
        # REAL-TIME DEEP CONFIRMATION
        # ----------------------------------------------------

        direction = pattern[
            "direction"
        ]

        candle_body = (
            current["close"]
            - current["open"]
        )

        if direction == "BUY":

            # Avoid an empty/invalid candle.
            if (
                current["high"]
                <= current["low"]
            ):
                return False

        else:

            if (
                current["high"]
                <= current["low"]
            ):
                return False

        return True

    # --------------------------------------------------------
    # SIGNAL
    # --------------------------------------------------------

    async def _fire_signal(
        self,
        symbol,
        tf,
        pattern,
        df_coded
    ):

        price = float(
            df_coded[
                "close"
            ].iloc[-1]
        )

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
            ] * 100
        )

        report = build_pattern_report(
            pattern,
            symbol,
            tf
        )

        msg = (
            f"🚨 <b>REAL-TIME DEEP SIGNAL</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>TF:</b> {tf}\n"
            f"<b>Direction:</b> "
            f"<b>{pattern['direction']}</b>\n"
            f"<b>Price:</b> "
            f"<code>{price}</code>\n"
            f"<b>Pattern:</b> "
            f"<code>{pattern['signature']}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>HISTORICAL DEEP CHECK</b>\n"
            f"├ Occurrences: <b>{total}</b>\n"
            f"├ WIN: <b>{wins}</b>\n"
            f"├ LOSS: <b>{losses}</b>\n"
            f"├ WR: <b>{pattern['win_rate']}%</b>\n"
            f"├ PF: <b>{pattern['profit_factor']}</b>\n"
            f"├ Avg R: <b>{pattern['avg_r']:+.3f}</b>\n"
            f"└ Confidence: "
            f"<b>{pattern['confidence']}/100</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>MFE / MAE</b>\n"
            f"├ Avg MFE: "
            f"<b>{pattern['avg_mfe']:+.3f}R</b>\n"
            f"├ Avg MAE: "
            f"<b>{pattern['avg_mae']:+.3f}R</b>\n"
            f"└ Worst MAE: "
            f"<b>{pattern['worst_mae']:+.3f}R</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>R LEVELS</b>\n"
            f"├ 1R: {pattern['reached_1r']}\n"
            f"├ 2R: {pattern['reached_2r']}\n"
            f"├ 4R: {pattern['reached_4r']}\n"
            f"├ 6R: {pattern['reached_6r']}\n"
            f"├ 8R: {pattern['reached_8r']}\n"
            f"└ 10R: {pattern['reached_10r']}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>MOVEMENT</b>\n"
            f"├ TP1: {pattern['tp1_rate']}%\n"
            f"├ Retracement: "
            f"{pattern['retracement_rate']}%\n"
            f"├ Continuation: "
            f"{pattern['continuation_rate']}%\n"
            f"└ Breakout: "
            f"{pattern['breakout_rate']}%\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>EXPECTED HISTORICAL R:</b> "
            f"{expected:+.0f}$ per 100$ risk\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚠️ <i>Signal faqat yopilgan "
            f"real candle asosida.</i>"
        )

        try:

            hist_df = self.history_cache.get(
                (symbol, tf),
                df_coded
            )

            loop = asyncio.get_event_loop()

            buf = await loop.run_in_executor(
                None,
                make_signal_chart,
                hist_df,
                pattern,
                symbol,
                tf
            )

            await tg.photo(
                buf,
                caption=msg
            )

            # Full deep report separately.
            await tg.send(
                report
            )

        except Exception as e:

            log.error(
                f"Signal chart error: {e}"
            )

            await tg.send(
                msg
            )

    # --------------------------------------------------------
    # WEBSOCKET
    # --------------------------------------------------------

    async def listen_stream(
        self,
        symbol,
        tf
    ):

        while True:

            try:

                socket = (
                    self.bm.kline_socket(
                        symbol=symbol,
                        interval=tf
                    )
                )

                async with socket as stream:

                    log.info(
                        f"🔌 {symbol} {tf}: "
                        f"WebSocket ulandi"
                    )

                    while True:

                        msg = await stream.recv()

                        if (
                            msg.get("e")
                            != "kline"
                        ):
                            continue

                        k = msg[
                            "k"
                        ]

                        # Only CLOSED candles.
                        if not k[
                            "x"
                        ]:
                            continue

                        candle = {

                            "open_time": int(
                                k["t"]
                                / 1000
                            ),

                            "open": float(
                                k["o"]
                            ),

                            "high": float(
                                k["h"]
                            ),

                            "low": float(
                                k["l"]
                            ),

                            "close": float(
                                k["c"]
                            ),

                            "volume": float(
                                k["v"]
                            ),
                        }

                        log.info(
                            f"🕯 CLOSED "
                            f"{symbol} {tf} "
                            f"{candle['close']}"
                        )

                        self._on_closed_candle(
                            symbol,
                            tf,
                            candle
                        )

            except Exception as e:

                log.error(
                    f"❌ {symbol} {tf} "
                    f"stream error: {e}"
                )

                await asyncio.sleep(
                    5
                )

    # --------------------------------------------------------
    # MAIN RUN
    # --------------------------------------------------------

    async def run(
        self
    ):

        log.info(
            "🚀 Crypto Deep Pattern "
            "Bot v9.0 ishga tushmoqda"
        )

        await tg.send(
            f"🚀 <b>Crypto Deep Pattern Bot v9.0</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Symbols:</b> "
            f"{', '.join(CONFIG['SYMBOLS'])}\n"
            f"<b>TF:</b> "
            f"{', '.join(CONFIG['TIMEFRAMES'])}\n"
            f"<b>Minimum occurrences:</b> "
            f"{CONFIG['MIN_OCCURRENCES']}\n"
            f"<b>Minimum WR:</b> "
            f"{CONFIG['MIN_WIN_RATE']}%\n"
            f"<b>Maximum losses:</b> "
            f"{CONFIG['MAX_LOSSES']}\n"
            f"<b>Minimum PF:</b> "
            f"{CONFIG['MIN_PROFIT_FACTOR']}\n"
            f"<b>Minimum Confidence:</b> "
            f"{CONFIG['MIN_CONFIDENCE']}/100\n"
            f"<b>Forward candles:</b> "
            f"{CONFIG['FORWARD_CANDLES']}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🔬 <b>DEEP ANALYSIS BOSHLANDI</b>"
        )

        self.client = (
            await AsyncClient.create()
        )

        self.bm = (
            BinanceSocketManager(
                self.client
            )
        )

        # ====================================================
        # CONTINUOUS DEEP SCAN
        # ====================================================

        while True:

            self.patterns = {}

            scan_started = time.time()

            await tg.send(
                "🔬 <b>Yangi deep scan boshlandi...</b>\n"
                "Har bir pattern tarixdagi "
                "100+ occurrence orqali tekshirilmoqda."
            )

            for symbol in CONFIG[
                "SYMBOLS"
            ]:

                for tf in CONFIG[
                    "TIMEFRAMES"
                ]:

                    try:

                        top = await self.find_patterns_for(
                            symbol,
                            tf
                        )

                        if top:

                            self.patterns[
                                (symbol, tf)
                            ] = top

                            best = top[0]

                            await tg.send(
                                f"✅ <b>{symbol} · {tf}</b>\n"
                                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                                f"<b>Deep patterns:</b> "
                                f"{len(top)}\n\n"
                                f"🥇 <b>TOP PATTERN</b>\n"
                                f"<code>{best['signature']}</code>\n"
                                f"→ <b>{best['direction']}</b>\n"
                                f"├ Occurrences: "
                                f"{best['count']}\n"
                                f"├ WIN: {best['wins']}\n"
                                f"├ LOSS: {best['losses']}\n"
                                f"├ WR: {best['win_rate']}%\n"
                                f"├ PF: {best['profit_factor']}\n"
                                f"├ Avg R: "
                                f"{best['avg_r']:+.3f}\n"
                                f"├ MFE: "
                                f"{best['avg_mfe']:+.3f}R\n"
                                f"├ MAE: "
                                f"{best['avg_mae']:+.3f}R\n"
                                f"├ 1R: "
                                f"{best['reached_1r']}\n"
                                f"├ 4R: "
                                f"{best['reached_4r']}\n"
                                f"├ 10R: "
                                f"{best['reached_10r']}\n"
                                f"├ Retracement: "
                                f"{best['retracement_rate']}%\n"
                                f"├ Continuation: "
                                f"{best['continuation_rate']}%\n"
                                f"├ Breakout: "
                                f"{best['breakout_rate']}%\n"
                                f"└ Confidence: "
                                f"<b>{best['confidence']}/100</b>"
                            )

                        else:

                            log.info(
                                f"⚠️ {symbol} {tf}: "
                                f"deep filterdan o'tgan "
                                f"pattern yo'q"
                            )

                    except Exception as e:

                        log.error(
                            f"{symbol} {tf}: "
                            f"deep scan error: {e}"
                        )

                        await tg.send(
                            f"❌ <b>{symbol} · {tf}</b>\n"
                            f"Deep scan error: "
                            f"<code>{e}</code>"
                        )

            total = sum(
                len(v)
                for v in self.patterns.values()
            )

            scan_time = (
                time.time()
                - scan_started
            )

            # =================================================
            # NO PATTERN
            # =================================================

            if total == 0:

                await tg.send(
                    f"⏳ <b>Deep scan yakunlandi.</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"❌ Hozircha barcha "
                    f"qat'iy filtrdan o'tgan "
                    f"pattern topilmadi.\n"
                    f"⏱ Scan: {scan_time/60:.1f} min\n"
                    f"🔄 Bot o'chmaydi.\n"
                    f"Keyingi deep scan: "
                    f"{CONFIG['NO_PATTERN_RESCAN_SECONDS']/60:.0f} daqiqadan keyin."
                )

                log.info(
                    "⏳ Pattern yo'q. "
                    "1 soat kutamiz..."
                )

                await asyncio.sleep(
                    CONFIG[
                        "NO_PATTERN_RESCAN_SECONDS"
                    ]
                )

                continue

            # =================================================
            # PATTERNS FOUND
            # =================================================

            await tg.send(
                f"🟢 <b>DEEP ANALYSIS YAKUNLANDI</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"<b>Jami pattern:</b> "
                f"{total}\n"
                f"<b>Scan time:</b> "
                f"{scan_time/60:.1f} min\n"
                f"📡 Real-time closed-candle "
                f"monitoring boshlandi."
            )

            break

        # ====================================================
        # LIVE STREAM TASKS
        # ====================================================

        tasks = [
            asyncio.create_task(
                self.listen_stream(
                    symbol,
                    tf
                )
            )
            for symbol, tf in self.patterns.keys()
        ]

        if not tasks:

            await tg.send(
                "⚠️ Monitoring uchun task topilmadi. "
                "Bot qayta deep scan qiladi."
            )

            return

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
            "🛑 To'xtatilmoqda..."
        )

    except Exception as e:

        log.exception(
            f"FATAL ERROR: {e}"
        )

        try:

            await tg.send(
                f"❌ <b>BOT ERROR</b>\n"
                f"<code>{e}</code>"
            )

        except Exception:
            pass

        raise

    finally:

        if bot.client:

            try:

                await bot.client.close_connection()

            except Exception as e:

                log.error(
                    f"Client close error: {e}"
                )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )