# ============================================================
# BTC Pattern Discovery Engine v1.0 — MUSTAQIL, ALOHIDA DASTUR
# Ishlayotgan botga (Railway) HECH QANDAY aloqasi yo'q.
# Vazifasi: BTCUSDT'ning butun tarixini (barcha TF'larda) yuklab,
# candle ketma-ketliklarini shakl bo'yicha guruhlab, statistik
# jihatdan barqaror va ko'p uchraydigan naqshlarni topadi.
# ============================================================
import os
import time
import json
import logging
import asyncio
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

from binance import AsyncClient
from telegram import Bot, InputFile
from telegram.constants import ParseMode
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# SOZLAMALAR
# ============================================================
CONFIG = {
    'SYMBOL': os.getenv('ANALYSIS_SYMBOL', 'BTCUSDT'),
    'TIMEFRAMES': os.getenv('ANALYSIS_TFS', '1m,3m,5m,15m,30m,1h').split(','),

    # Har TF uchun qancha orqaga (kun hisobida) tarix olinadi.
    # 1m uchun butun tarixni olish millionlab qator bo'ladi — juda uzoq va
    # og'ir bo'lgani uchun TF kichraygan sari lookback ham cheklab qo'yilgan.
    'DAYS_PER_TF': {
        '1m':  int(os.getenv('DAYS_1M', '180')),   # oxirgi ~6 oy
        '3m':  int(os.getenv('DAYS_3M', '365')),
        '5m':  int(os.getenv('DAYS_5M', '365')),
        '15m': int(os.getenv('DAYS_15M', '730')),
        '30m': int(os.getenv('DAYS_30M', '1095')),
        '1h':  int(os.getenv('DAYS_1H', '1825')),  # ~5 yil
    },

    # Naqsh (signature) necha ta ketma-ket candle'dan tuziladi
    'SEQ_LENGTHS': [int(x) for x in os.getenv('SEQ_LENGTHS', '2,3,4').split(',')],

    # Statistik ishonchlilik chegaralari
    'MIN_OCCURRENCES': int(os.getenv('MIN_OCCURRENCES', '100')),
    'FORWARD_CANDLES': int(os.getenv('FORWARD_CANDLES', '50')),
    'SL_BUF': float(os.getenv('SL_BUF', '10')),
    'BE_AT_R': 1.0,
    'TRAIL_STEP': 2.0,
    'MAX_TRAIL_R': 10.0,

    'TOP_N': int(os.getenv('TOP_N', '10')),

    'REQUEST_DELAY': float(os.getenv('REQUEST_DELAY', '0.25')),
}

TELEGRAM_TOKEN   = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
log = logging.getLogger(__name__)


class TG:
    def __init__(self, token, chat_id):
        self.bot = Bot(token=token) if token else None
        self.chat_id = chat_id

    async def send(self, msg):
        if not self.bot: return
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=msg, parse_mode=ParseMode.HTML)
        except Exception as e:
            log.error(f"TG: {e}")

    async def photo(self, buf, caption=""):
        if not self.bot: return
        try:
            await self.bot.send_photo(chat_id=self.chat_id, photo=InputFile(buf, filename='chart.png'),
                                       caption=caption, parse_mode=ParseMode.HTML)
        except Exception as e:
            log.error(f"TG photo: {e}")


tg = TG(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)


# ============================================================
# 1-QISM: TARIXIY MA'LUMOTNI YUKLASH (real Binance'dan)
# ============================================================
async def fetch_history(client, symbol, interval, days):
    end_time = int(time.time() * 1000)
    start_time = end_time - days * 24 * 60 * 60 * 1000

    all_klines = []
    cursor = start_time
    log.info(f"📥 {symbol} {interval}: {days} kunlik tarix yuklanmoqda...")

    while cursor < end_time:
        try:
            klines = await client.get_klines(
                symbol=symbol, interval=interval,
                startTime=cursor, limit=1000
            )
        except Exception as e:
            log.error(f"{symbol} {interval} fetch xato: {e} — 2s kutib qayta urinamiz")
            await asyncio.sleep(2)
            continue

        if not klines:
            break

        all_klines.extend(klines)
        last_open_time = klines[-1][0]
        if last_open_time <= cursor:
            break
        cursor = last_open_time + 1
        await asyncio.sleep(CONFIG['REQUEST_DELAY'])

        if len(all_klines) % 20000 == 0:
            log.info(f"   ...{symbol} {interval}: {len(all_klines)} ta candle yuklandi")

    df = pd.DataFrame(all_klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'trades', 'tbbav', 'tbqav', 'ignore'
    ])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df['open_time'] = df['open_time'] // 1000

    log.info(f"✅ {symbol} {interval}: jami {len(df)} ta candle yuklandi")
    return df[['open_time', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)


# ============================================================
# 2-QISM: CANDLE'NI SHAKL BO'YICHA DISKRETIZATSIYA QILISH
# ============================================================
def classify_candles(df):
    """
    Har candle uchun 3 harfli kod hisoblaydi:
      [Yo'nalish][Tana kattaligi][Fitil xarakteri]
      Yo'nalish:      U (ko'k), D (qizil), F (flat)
      Tana kattaligi: S (kichik), M (o'rta), L (katta) — lokal TR'ga nisbatan
      Fitil:          N (kichik), U (yuqori katta), D (past katta), B (ikkalasi ham)
    """
    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values

    tr = np.maximum(h - l, 1e-9)
    atr = pd.Series(tr).rolling(50, min_periods=10).mean().bfill().values

    body = c - o
    body_abs = np.abs(body)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l

    direction = np.where(body > 0, 'U', np.where(body < 0, 'D', 'F'))

    body_ratio = body_abs / atr
    body_size = np.where(body_ratio < 0.4, 'S', np.where(body_ratio < 1.2, 'M', 'L'))

    uw_ratio = upper_wick / np.maximum(body_abs, atr * 0.1)
    lw_ratio = lower_wick / np.maximum(body_abs, atr * 0.1)
    big_upper = uw_ratio > 1.0
    big_lower = lw_ratio > 1.0

    wick_class = np.where(big_upper & big_lower, 'B',
                  np.where(big_upper, 'U',
                  np.where(big_lower, 'D', 'N')))

    codes = np.array([f"{d}{s}{w}" for d, s, w in zip(direction, body_size, wick_class)])

    out = df.copy()
    out['code'] = codes
    out['atr'] = atr
    return out


def build_signatures(df_coded, seq_len):
    codes = df_coded['code'].values
    n = len(codes)
    sigs = [None] * n
    for i in range(seq_len - 1, n):
        sigs[i] = '_'.join(codes[i - seq_len + 1: i + 1])
    return sigs


# ============================================================
# 3-QISM: HAR NAQSH UCHUN OLDINGA QARAB NATIJA SIMULYATSIYASI
# ============================================================
def simulate_forward(df, entry_idx, direction, sl_buf_cfg):
    """
    entry_idx dagi candle yopilish narxidan kirib, live bot mantig'i bilan
    (SL, BE @1R, trailing har 2R, 10R cap) FORWARD_CANDLES ichida qanday
    natija (R multiple) berishini hisoblaydi.
    """
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    n = len(df)

    entry = c[entry_idx]
    buf = sl_buf_cfg * (entry / 100000)

    if direction == 'B':
        sl = l[entry_idx] - buf
        sl_dist = entry - sl
    else:
        sl = h[entry_idx] + buf
        sl_dist = sl - entry

    if sl_dist <= 0:
        return None

    be_set = False
    lock_r = 0.0
    max_i = min(entry_idx + CONFIG['FORWARD_CANDLES'], n - 1)

    for i in range(entry_idx + 1, max_i + 1):
        if direction == 'B':
            if l[i] <= sl:
                exit_r = (sl - entry) / sl_dist
                return exit_r
            max_r = (h[i] - entry) / sl_dist
        else:
            if h[i] >= sl:
                exit_r = (entry - sl) / sl_dist
                return exit_r
            max_r = (entry - l[i]) / sl_dist

        if max_r >= CONFIG['BE_AT_R'] and not be_set:
            sl = entry
            be_set = True

        if max_r >= CONFIG['MAX_TRAIL_R']:
            return CONFIG['MAX_TRAIL_R']

        if max_r >= CONFIG['BE_AT_R']:
            steps = int(max_r / CONFIG['TRAIL_STEP'])
            new_lock = (steps - 1) * CONFIG['TRAIL_STEP']
            if new_lock > lock_r:
                lock_r = new_lock
                if direction == 'B':
                    sl = entry + lock_r * sl_dist
                else:
                    sl = entry - lock_r * sl_dist

    if direction == 'B':
        return (c[max_i] - entry) / sl_dist
    else:
        return (entry - c[max_i]) / sl_dist


# ============================================================
# 4-QISM: NAQSHLARNI YIG'ISH, STATISTIKA, FILTRLASH
# ============================================================
def analyze_timeframe(df, tf_name):
    df_coded = classify_candles(df)
    n = len(df_coded)
    results = []

    df_coded['year'] = pd.to_datetime(df_coded['open_time'], unit='s').dt.year

    for seq_len in CONFIG['SEQ_LENGTHS']:
        sigs = build_signatures(df_coded, seq_len)
        df_coded[f'sig_{seq_len}'] = sigs

        groups = defaultdict(list)
        for i, s in enumerate(sigs):
            if s is None: continue
            if i + CONFIG['FORWARD_CANDLES'] >= n: continue
            groups[s].append(i)

        for sig, idx_list in groups.items():
            if len(idx_list) < CONFIG['MIN_OCCURRENCES']:
                continue

            for direction in ['B', 'S']:
                r_values = []
                years_map = defaultdict(list)
                for idx in idx_list:
                    r = simulate_forward(df_coded, idx, direction, CONFIG['SL_BUF'])
                    if r is None: continue
                    r_values.append(r)
                    yr = int(df_coded['year'].iloc[idx])
                    years_map[yr].append(r)

                if len(r_values) < CONFIG['MIN_OCCURRENCES']:
                    continue

                r_arr = np.array(r_values)
                win_rate = float((r_arr > 0).mean() * 100)
                avg_r = float(r_arr.mean())
                min_r = float(r_arr.min())
                max_r = float(r_arr.max())
                gross_win = r_arr[r_arr > 0].sum()
                gross_loss = abs(r_arr[r_arr < 0].sum())
                pf = float(gross_win / gross_loss) if gross_loss > 0 else float('inf')

                yearly_wr = []
                for yr, rs in years_map.items():
                    if len(rs) >= 10:
                        yearly_wr.append(np.mean(np.array(rs) > 0) * 100)
                consistency_std = float(np.std(yearly_wr)) if len(yearly_wr) >= 2 else 999.0
                is_consistent = len(yearly_wr) >= 2 and consistency_std < 15.0

                results.append({
                    'tf': tf_name,
                    'seq_len': seq_len,
                    'signature': sig,
                    'direction': 'BUY' if direction == 'B' else 'SELL',
                    'count': len(r_values),
                    'win_rate': round(win_rate, 1),
                    'avg_r': round(avg_r, 3),
                    'min_r': round(min_r, 2),
                    'max_r': round(max_r, 2),
                    'profit_factor': round(pf, 2) if pf != float('inf') else 999.0,
                    'consistent': is_consistent,
                    'consistency_std': round(consistency_std, 1),
                    'years_covered': len(yearly_wr),
                })

    return results, df_coded


def rank_results(results, top_n):
    consistent = [r for r in results if r['consistent'] and r['profit_factor'] >= 1.1]
    consistent.sort(key=lambda r: (r['avg_r'] * r['win_rate'] / 100), reverse=True)
    return consistent[:top_n]


# ============================================================
# 5-QISM: GRAFIK — TOP naqshning bitta misolini chizib ko'rsatish
# ============================================================
def make_pattern_chart(df_coded, signature, seq_col, symbol, tf):
    idx_list = df_coded.index[df_coded[seq_col] == signature].tolist()
    if not idx_list:
        return None
    example_idx = idx_list[len(idx_list) // 2]
    seq_len = len(signature.split('_'))
    start = max(0, example_idx - seq_len - 15)
    end = min(len(df_coded), example_idx + 30)
    window = df_coded.iloc[start:end].reset_index(drop=True)
    highlight_start = example_idx - seq_len + 1 - start
    highlight_end = example_idx - start

    fig, ax = plt.subplots(figsize=(11, 5), facecolor='#0a0b0f')
    ax.set_facecolor('#0a0b0f')
    for i, row in window.iterrows():
        c = '#10b981' if row['close'] >= row['open'] else '#ef4444'
        lw = 2.5 if highlight_start <= i <= highlight_end else 1
        alpha = 1.0 if highlight_start <= i <= highlight_end else 0.5
        ax.plot([i, i], [row['low'], row['high']], color=c, linewidth=1, alpha=alpha)
        ax.plot([i, i], [row['open'], row['close']], color=c, linewidth=4*lw/2.5, alpha=alpha)

    ax.axvspan(highlight_start - 0.4, highlight_end + 0.4, color='yellow', alpha=0.08)
    ax.set_title(f'{symbol} · {tf} · Naqsh: {signature}', color='#e2e8f0', fontsize=12)
    ax.tick_params(colors='#94a3b8', labelsize=8)
    ax.grid(True, alpha=0.1, color='#ffffff')
    for sp in ax.spines.values(): sp.set_color('#ffffff14')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=90, facecolor='#0a0b0f')
    plt.close(fig)
    buf.seek(0)
    return buf


# ============================================================
# 6-QISM: HISOBOT — Telegram'ga yuborish
# ============================================================
async def send_report(tf, top_patterns, total_candles):
    if not top_patterns:
        await tg.send(
            f"📊 <b>{CONFIG['SYMBOL']} · {tf}</b>\n"
            f"Jami {total_candles} ta candle tahlil qilindi.\n"
            f"⚠️ Statistik jihatdan barqaror (kamida {CONFIG['MIN_OCCURRENCES']} marta "
            f"takrorlangan va yillar bo'yicha barqaror) naqsh topilmadi."
        )
        return

    text = (f"📊 <b>{CONFIG['SYMBOL']} · {tf} — NAQSH TAHLILI</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Jami candle: {total_candles}\n"
            f"Topilgan barqaror naqshlar: {len(top_patterns)}\n\n")

    for i, r in enumerate(top_patterns, 1):
        text += (
            f"<b>{i}) {r['signature']}</b> ({r['seq_len']} sham) → <b>{r['direction']}</b>\n"
            f"   ├ Takrorlandi: <b>{r['count']}</b> marta\n"
            f"   ├ Win rate: <b>{r['win_rate']}%</b>\n"
            f"   ├ O'rtacha natija: <b>{r['avg_r']:+.2f}R</b>\n"
            f"   ├ Min/Max: {r['min_r']:+.2f}R / {r['max_r']:+.2f}R\n"
            f"   ├ Profit Factor: <b>{r['profit_factor']}</b>\n"
            f"   └ Barqarorlik: {r['years_covered']} yil, og'ish {r['consistency_std']}%\n\n"
        )

    for chunk_start in range(0, len(text), 3800):
        await tg.send(text[chunk_start:chunk_start+3800])
        await asyncio.sleep(0.5)


# ============================================================
# ASOSIY
# ============================================================
async def main():
    log.info("🚀 BTC Pattern Discovery Engine boshlandi")
    await tg.send(
        f"🚀 <b>BTC Pattern Discovery boshlandi</b>\n"
        f"Symbol: {CONFIG['SYMBOL']}\n"
        f"TF'lar: {', '.join(CONFIG['TIMEFRAMES'])}\n"
        f"Bu — <b>alohida, mustaqil tahlil</b>. Ishlayotgan botga tegishli emas.\n"
        f"⏳ Bu bir necha soat davom etishi mumkin, natija tayyor bo'lgach xabar keladi."
    )

    client = await AsyncClient.create()
    all_results = {}

    try:
        for tf in CONFIG['TIMEFRAMES']:
            tf = tf.strip()
            days = CONFIG['DAYS_PER_TF'].get(tf, 365)

            df = await fetch_history(client, CONFIG['SYMBOL'], tf, days)
            if len(df) < 500:
                log.warning(f"{tf}: yetarli ma'lumot yo'q, o'tkazib yuborildi")
                continue

            log.info(f"🔍 {tf}: naqsh tahlili boshlandi ({len(df)} candle)")
            results, df_coded = analyze_timeframe(df, tf)
            top = rank_results(results, CONFIG['TOP_N'])
            all_results[tf] = top

            log.info(f"✅ {tf}: {len(results)} ta naqsh topildi, {len(top)} tasi barqaror")
            await send_report(tf, top, len(df))

            if top:
                best = top[0]
                seq_col = f"sig_{best['seq_len']}"
                ch = make_pattern_chart(df_coded, best['signature'], seq_col, CONFIG['SYMBOL'], tf)
                if ch:
                    await tg.photo(ch, f"📈 {tf}: eng yaxshi naqsh — {best['signature']} ({best['direction']})")

            await asyncio.sleep(1)

        flat = []
        for tf, items in all_results.items():
            flat.extend(items)
        flat.sort(key=lambda r: (r['avg_r'] * r['win_rate'] / 100), reverse=True)

        if flat:
            top5 = flat[:5]
            text = "🏆 <b>UMUMIY TOP-5 (barcha TF'lar orasidan)</b>\n━━━━━━━━━━━━━━━━━━━━━━\n\n"
            for i, r in enumerate(top5, 1):
                text += (f"{i}) <b>{r['tf']} · {r['signature']}</b> → {r['direction']}\n"
                         f"   {r['count']} marta | WR {r['win_rate']}% | "
                         f"avg {r['avg_r']:+.2f}R | PF {r['profit_factor']}\n\n")
            await tg.send(text)
        else:
            await tg.send("⚠️ Hech qanday statistik jihatdan barqaror naqsh topilmadi.")

        with open('/tmp/pattern_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        log.info("💾 Natijalar /tmp/pattern_results.json ga saqlandi")

    finally:
        await client.close_connection()

    await tg.send("✅ <b>BTC Pattern Discovery yakunlandi.</b>")
    log.info("✅ Tugadi")


if __name__ == '__main__':
    asyncio.run(main())