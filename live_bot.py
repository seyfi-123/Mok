# ============================================================
# Crypto Live Pattern Bot v5.0
# - Tarixda naqsh necha marta ishlagan, grafikda belgilanadi
# - Telegramga oddiy tushunarli xabar (foyda/zarar)
# - Rasm har doim yuboriladi
# ============================================================
import os
import json
import time
import logging
import asyncio
import io
from collections import deque, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from binance import AsyncClient, BinanceSocketManager
from telegram import Bot, InputFile
from telegram.constants import ParseMode
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# SOZLAMALAR
# ============================================================
CONFIG = {
    'SYMBOLS': [s.strip().upper() for s in os.getenv('SYMBOLS', 'BTCUSDT').split(',') if s.strip()],
    'TIMEFRAMES': [t.strip() for t in os.getenv('TIMEFRAMES', '5m,15m,1h').split(',') if t.strip()],

    'DAYS_PER_TF': {
        '1m':  int(os.getenv('DAYS_1M', '180')),
        '3m':  int(os.getenv('DAYS_3M', '365')),
        '5m':  int(os.getenv('DAYS_5M', '365')),
        '15m': int(os.getenv('DAYS_15M', '730')),
        '30m': int(os.getenv('DAYS_30M', '1095')),
        '1h':  int(os.getenv('DAYS_1H', '1825')),
        '4h':  int(os.getenv('DAYS_4H', '1825')),
    },

    'SEQ_LENGTHS': [int(x) for x in os.getenv('SEQ_LENGTHS', '2,3,4').split(',')],

    'MIN_OCCURRENCES': int(os.getenv('MIN_OCCURRENCES', '100')),
    'MIN_WIN_RATE':   float(os.getenv('MIN_WIN_RATE', '50')),
    'MIN_PROFIT_FACTOR': float(os.getenv('MIN_PROFIT_FACTOR', '1.0')),

    'FORWARD_CANDLES': int(os.getenv('FORWARD_CANDLES', '50')),
    'SL_BUF': float(os.getenv('SL_BUF', '10')),

    # 1:1R → 1R da 50% yopiladi + BE, 4R dan trailing
    'TP1_R': 1.0,
    'TP1_CLOSE_PCT': 0.5,
    'TRAIL_START_R': 4.0,
    'TRAIL_STEP_R': 2.0,
    'MAX_TRAIL_R': 10.0,

    'REQUEST_DELAY': float(os.getenv('REQUEST_DELAY', '0.25')),
    'CANDLE_BUFFER': 250,
}

TELEGRAM_TOKEN   = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


# ============================================================
# TELEGRAM
# ============================================================
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
            await self.bot.send_photo(
                chat_id=self.chat_id,
                photo=InputFile(buf, filename='signal.png'),
                caption=caption,
                parse_mode=ParseMode.HTML
            )
        except Exception as e:
            log.error(f"TG photo: {e}")

tg = TG(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)


# ============================================================
# 1. TARIXNI YUKLASH
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
            log.error(f"{symbol} {interval} fetch xato: {e} — 2s kutamiz")
            await asyncio.sleep(2); continue
        if not klines:
            break
        all_klines.extend(klines)
        last_open_time = klines[-1][0]
        if last_open_time <= cursor:
            break
        cursor = last_open_time + 1
        await asyncio.sleep(CONFIG['REQUEST_DELAY'])

    df = pd.DataFrame(all_klines, columns=[
        'open_time','open','high','low','close','volume',
        'close_time','qav','trades','tbbav','tbqav','ignore'
    ])
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)
    df['open_time'] = df['open_time'] // 1000
    log.info(f"✅ {symbol} {interval}: {len(df)} ta candle yuklandi")
    return df[['open_time','open','high','low','close','volume']].reset_index(drop=True)


# ============================================================
# 2. SHAM KODI (O'ZGARTIRILMAGAN)
# ============================================================
def classify_candles(df):
    o, h, l, c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
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
# 3. SIMULYATSIYA (1:1R → TP1, BE, 4R dan trailing)
# ============================================================
def simulate_forward(df, entry_idx, direction, sl_buf_cfg):
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    n = len(df)

    entry = c[entry_idx]
    buf = sl_buf_cfg * (entry / 100000)

    if direction == 'B':
        sl_init = l[entry_idx] - buf
        sl_dist = entry - sl_init
    else:
        sl_init = h[entry_idx] + buf
        sl_dist = sl_init - entry

    if sl_dist <= 0:
        return None

    if direction == 'B':
        tp1_price = entry + CONFIG['TP1_R'] * sl_dist
    else:
        tp1_price = entry - CONFIG['TP1_R'] * sl_dist

    tp1_hit = False
    sl = sl_init
    lock_r = 0.0
    max_i = min(entry_idx + CONFIG['FORWARD_CANDLES'], n - 1)

    for i in range(entry_idx + 1, max_i + 1):
        if direction == 'B':
            if l[i] <= sl:
                if tp1_hit:
                    partial1 = CONFIG['TP1_CLOSE_PCT'] * CONFIG['TP1_R']
                    partial2 = (1 - CONFIG['TP1_CLOSE_PCT']) * ((sl - entry) / sl_dist)
                    return partial1 + partial2
                else:
                    return (sl - entry) / sl_dist
            if not tp1_hit and h[i] >= tp1_price:
                tp1_hit = True
                sl = entry
            max_r = (h[i] - entry) / sl_dist
        else:
            if h[i] >= sl:
                if tp1_hit:
                    partial1 = CONFIG['TP1_CLOSE_PCT'] * CONFIG['TP1_R']
                    partial2 = (1 - CONFIG['TP1_CLOSE_PCT']) * ((entry - sl) / sl_dist)
                    return partial1 + partial2
                else:
                    return (entry - sl) / sl_dist
            if not tp1_hit and l[i] <= tp1_price:
                tp1_hit = True
                sl = entry
            max_r = (entry - l[i]) / sl_dist

        if max_r >= CONFIG['TRAIL_START_R']:
            steps = int((max_r - CONFIG['TRAIL_START_R']) / CONFIG['TRAIL_STEP_R'])
            new_lock = CONFIG['TRAIL_START_R'] - CONFIG['TRAIL_STEP_R'] + steps * CONFIG['TRAIL_STEP_R']
            if new_lock > lock_r:
                lock_r = new_lock
                if direction == 'B':
                    sl = entry + lock_r * sl_dist
                else:
                    sl = entry - lock_r * sl_dist

        if max_r >= CONFIG['MAX_TRAIL_R']:
            if tp1_hit:
                partial1 = CONFIG['TP1_CLOSE_PCT'] * CONFIG['TP1_R']
                partial2 = (1 - CONFIG['TP1_CLOSE_PCT']) * CONFIG['MAX_TRAIL_R']
                return partial1 + partial2
            else:
                return CONFIG['MAX_TRAIL_R']

    if direction == 'B':
        final_r = (c[max_i] - entry) / sl_dist
    else:
        final_r = (entry - c[max_i]) / sl_dist

    if tp1_hit:
        partial1 = CONFIG['TP1_CLOSE_PCT'] * CONFIG['TP1_R']
        partial2 = (1 - CONFIG['TP1_CLOSE_PCT']) * final_r
        return partial1 + partial2
    return final_r


# ============================================================
# 4. TAHLIL
# ============================================================
def analyze_timeframe(df, tf_name):
    df_coded = classify_candles(df)
    n = len(df_coded)
    results = []

    for seq_len in CONFIG['SEQ_LENGTHS']:
        sigs = build_signatures(df_coded, seq_len)
        df_coded[f'sig_{seq_len}'] = sigs

        groups = defaultdict(list)
        for i, s in enumerate(sigs):
            if s is None or i + CONFIG['FORWARD_CANDLES'] >= n:
                continue
            groups[s].append(i)

        for sig, idx_list in groups.items():
            if len(idx_list) < CONFIG['MIN_OCCURRENCES']:
                continue

            for direction in ['B', 'S']:
                r_values = []
                for idx in idx_list:
                    r = simulate_forward(df_coded, idx, direction, CONFIG['SL_BUF'])
                    if r is None:
                        continue
                    r_values.append(r)

                if len(r_values) < CONFIG['MIN_OCCURRENCES']:
                    continue

                r_arr = np.array(r_values)
                wins = int((r_arr > 0).sum())
                losses = int((r_arr <= 0).sum())
                win_rate = float(wins / len(r_arr) * 100)
                avg_r = float(r_arr.mean())
                gross_win = r_arr[r_arr > 0].sum()
                gross_loss = abs(r_arr[r_arr < 0].sum())
                pf = float(gross_win / gross_loss) if gross_loss > 0 else 999.0

                results.append({
                    'tf': tf_name,
                    'seq_len': seq_len,
                    'signature': sig,
                    'direction': 'BUY' if direction == 'B' else 'SELL',
                    'count': len(r_values),
                    'wins': wins,
                    'losses': losses,
                    'win_rate': round(win_rate, 1),
                    'avg_r': round(avg_r, 3),
                    'profit_factor': round(pf, 2),
                })

    return results


def filter_and_rank(results):
    good = [r for r in results
            if r['profit_factor'] >= CONFIG['MIN_PROFIT_FACTOR']
            and r['win_rate'] >= CONFIG['MIN_WIN_RATE']]
    good.sort(key=lambda r: (r['avg_r'] * r['win_rate'] / 100), reverse=True)
    return good


# ============================================================
# 5. GRAFIK — tarixdagi barcha takrorlanishlar belgilanadi
# ============================================================
def make_signal_chart(df_coded, pattern, symbol, tf):
    """
    Grafikda:
      - Oxirgi 120 ta sham
      - Tarixda shu naqsh uchragan BARCHA joylar sariq strelka bilan belgilanadi
      - Oxirgi (hozirgi) signal katta yashil/qizil strelka bilan
    """
    seq_len = pattern['seq_len']
    sig_col = f'sig_{seq_len}'

    # Tarixda shu naqsh uchragan indekslar
    all_idx = df_coded.index[df_coded[sig_col] == pattern['signature']].tolist()

    # Oxirgi 120 shamni olamiz
    total = len(df_coded)
    window_start = max(0, total - 120)
    window = df_coded.iloc[window_start:].reset_index(drop=True)
    offset = window_start
    n = len(window)

    fig, ax = plt.subplots(figsize=(13, 6), facecolor='#0a0b0f')
    ax.set_facecolor('#0a0b0f')

    # Shamlar
    for i, row in window.iterrows():
        c = '#10b981' if row['close'] >= row['open'] else '#ef4444'
        ax.plot([i, i], [row['low'], row['high']], color=c, linewidth=1, alpha=0.85)
        ax.plot([i, i], [row['open'], row['close']], color=c, linewidth=3, alpha=0.85)

    # Tarixda uchragan joylar — kichik sariq strelka
    historical_marks = 0
    for idx in all_idx:
        rel = idx - offset
        if 0 <= rel < n - 1:
            # Naqsh tugagan joy: rel
            y = window['high'].iloc[rel] * 1.0005
            ax.annotate('▼', xy=(rel, y), color='#facc15', fontsize=11,
                        ha='center', va='bottom', alpha=0.9)
            historical_marks += 1
            ax.axvspan(rel - seq_len + 1, rel, color='#facc15', alpha=0.05)

    # Oxirgi (hozirgi) signal — katta strelka
    last_rel = n - 1
    arrow = '▲ BUY' if pattern['direction'] == 'BUY' else '▼ SELL'
    arrow_color = '#10b981' if pattern['direction'] == 'BUY' else '#ef4444'
    ax.annotate(arrow,
                xy=(last_rel, window['close'].iloc[-1]),
                xytext=(last_rel, window['high'].max() * 1.002),
                color=arrow_color, fontsize=14, fontweight='bold', ha='center')
    ax.axvspan(last_rel - seq_len + 1, last_rel, color=arrow_color, alpha=0.20)

    ax.set_title(
        f"{symbol} · {tf} · {pattern['signature']} · {pattern['direction']}   "
        f"| Tarixda {pattern['count']} marta uchragan | "
        f"Win: {pattern['wins']}  Loss: {pattern['losses']}",
        color='#e2e8f0', fontsize=11
    )
    ax.tick_params(colors='#94a3b8', labelsize=8)
    ax.grid(True, alpha=0.1, color='#ffffff')
    for sp in ax.spines.values():
        sp.set_color('#ffffff20')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=95, facecolor='#0a0b0f')
    plt.close(fig)
    buf.seek(0)
    return buf


# ============================================================
# 6. JONLI BOT
# ============================================================
class LivePatternBot:
    def __init__(self):
        self.patterns = {}
        self.buffers = {}
        self.signaled = set()
        self.history_cache = {}   # (symbol, tf) -> df_coded (grafik uchun)
        self.client = None
        self.bm = None

    async def find_patterns_for(self, symbol, tf):
        days = CONFIG['DAYS_PER_TF'].get(tf, 365)
        df = await fetch_history(self.client, symbol, tf, days)
        if len(df) < 500:
            return []
        df_coded = classify_candles(df)
        # Har bir seq_len uchun signature hisoblaymiz
        for seq_len in CONFIG['SEQ_LENGTHS']:
            df_coded[f'sig_{seq_len}'] = build_signatures(df_coded, seq_len)

        # Grafik uchun tarixni saqlaymiz
        self.history_cache[(symbol, tf)] = df_coded

        results = analyze_timeframe(df, tf)
        return filter_and_rank(results)

    def _on_closed_candle(self, symbol, tf, new_candle):
        key = (symbol, tf)
        if key not in self.buffers:
            self.buffers[key] = deque(maxlen=CONFIG['CANDLE_BUFFER'])
        self.buffers[key].append(new_candle)

        if len(self.buffers[key]) < 60:
            return

        df = pd.DataFrame(list(self.buffers[key]))
        df_coded = classify_candles(df)

        tf_patterns = self.patterns.get(key, [])
        for pattern in tf_patterns:
            seq_len = pattern['seq_len']
            if len(df_coded) < seq_len:
                continue
            recent = df_coded['code'].iloc[-seq_len:].tolist()
            current_sig = '_'.join(recent)
            if current_sig != pattern['signature']:
                continue

            sig_key = (symbol, tf, new_candle['open_time'], pattern['signature'], pattern['direction'])
            if sig_key in self.signaled:
                continue
            self.signaled.add(sig_key)
            if len(self.signaled) > 5000:
                self.signaled = set(list(self.signaled)[-2000:])

            asyncio.create_task(self._fire_signal(symbol, tf, pattern, df_coded))

    async def _fire_signal(self, symbol, tf, pattern, df_coded):
        price = float(df_coded['close'].iloc[-1])
        log.info(f"🚨 SIGNAL: {symbol} {tf} {pattern['signature']} {pattern['direction']} @ {price}")

        # FOYDA/ZARAR ni oddiy qilib yozamiz
        wins = pattern['wins']
        losses = pattern['losses']
        total = wins + losses
        if pattern['avg_r'] > 0:
            outcome = "✅ FOYDA"
            outcome_emoji = "🟢"
        else:
            outcome = "❌ ZARAR"
            outcome_emoji = "🔴"

        # 1R lik pul miqdorini hisoblash (masalan, 100$ risk)
        risk_example = 100
        expected_profit = pattern['avg_r'] * risk_example

        msg = (
            f"🚨 <b>YANGI SIGNAL — {symbol}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>📊 Juftlik:</b> {symbol}\n"
            f"<b>⏱ Timeframe:</b> {tf}\n"
            f"<b>🎯 Yo'nalish:</b> <b>{pattern['direction']}</b>\n"
            f"<b>💰 Narx:</b> <code>{price}</code>\n"
            f"<b>🔍 Naqsh:</b> <code>{pattern['signature']}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>📈 TARIXIY NATIJA:</b>\n"
            f"├ <b>Necha marta:</b> {total} marta\n"
            f"├ <b>✅ Foyda:</b> {wins} marta\n"
            f"├ <b>❌ Zarar:</b> {losses} marta\n"
            f"├ <b>🏆 Win Rate:</b> {pattern['win_rate']}%\n"
            f"└ <b>⚖️ Nisbat:</b> {wins}:{losses}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>📊 KUTILAYOTGAN NATIJA:</b>\n"
            f"├ <b>O'rtacha:</b> {pattern['avg_r']:+.2f}R\n"
            f"├ <b>100$ riskda:</b> ~{expected_profit:+.0f}$\n"
            f"└ <b>Holat:</b> {outcome_emoji} {outcome}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>⚙️ BOSHQARUV:</b>\n"
            f"├ 1R da 50% yopiladi + BE\n"
            f"├ 4R dan trailing (har 2R)\n"
            f"└ Maksimal 10R\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚠️ <i>Tarixiy tahlilga asoslangan signal.</i>"
        )

        try:
            # Tarixdagi barcha uchragan joylar bilan grafik
            hist_df = self.history_cache.get((symbol, tf))
            if hist_df is None:
                hist_df = df_coded

            loop = asyncio.get_event_loop()
            buf = await loop.run_in_executor(None, make_signal_chart, hist_df, pattern, symbol, tf)
            await tg.photo(buf, caption=msg)
        except Exception as e:
            log.error(f"Signal yuborishda xato: {e}")
            await tg.send(msg)

    async def listen_stream(self, symbol, tf):
        while True:
            try:
                socket = self.bm.kline_socket(symbol=symbol, interval=tf)
                async with socket as stream:
                    log.info(f"🔌 {symbol} {tf}: WebSocket ulandi")
                    while True:
                        msg = await stream.recv()
                        if msg.get('e') != 'kline' or not msg['k']['x']:
                            continue
                        k = msg['k']
                        candle = {
                            'open_time': int(k['t'] / 1000),
                            'open': float(k['o']),
                            'high': float(k['h']),
                            'low': float(k['l']),
                            'close': float(k['c']),
                            'volume': float(k['v']),
                        }
                        self._on_closed_candle(symbol, tf, candle)
            except Exception as e:
                log.error(f"❌ {symbol} {tf} stream xato: {e}. 5s dan keyin qayta")
                await asyncio.sleep(5)

    async def run(self):
        log.info("🚀 Live Pattern Bot v5.0 ishga tushmoqda")
        await tg.send(
            f"🚀 <b>Live Pattern Bot v5.0 ishga tushdi</b>\n"
            f"<b>Symbols:</b> {', '.join(CONFIG['SYMBOLS'])}\n"
            f"<b>TF:</b> {', '.join(CONFIG['TIMEFRAMES'])}\n"
            f"<b>Min takrorlanish:</b> {CONFIG['MIN_OCCURRENCES']}\n"
            f"<b>Min WR:</b> {CONFIG['MIN_WIN_RATE']}%\n"
            f"<b>Logika:</b> 1R da 50% + BE, 4R dan trailing\n"
            f"⏳ Tahlil boshlandi..."
        )

        self.client = await AsyncClient.create()
        self.bm = BinanceSocketManager(self.client)

        self.patterns = {}
        for symbol in CONFIG['SYMBOLS']:
            for tf in CONFIG['TIMEFRAMES']:
                try:
                    top = await self.find_patterns_for(symbol, tf)
                    if top:
                        self.patterns[(symbol, tf)] = top
                        best = top[0]
                        await tg.send(
                            f"✅ <b>{symbol} · {tf}</b>: {len(top)} ta naqsh topildi.\n\n"
                            f"🥇 Eng yaxshisi:\n"
                            f"<code>{best['signature']}</code> → {best['direction']}\n"
                            f"├ {best['count']} marta uchragan\n"
                            f"├ ✅ Foyda: {best['wins']}\n"
                            f"├ ❌ Zarar: {best['losses']}\n"
                            f"├ WR: {best['win_rate']}%\n"
                            f"└ avg: {best['avg_r']:+.2f}R"
                        )
                    else:
                        await tg.send(f"⚠️ <b>{symbol} · {tf}</b>: mos naqsh topilmadi.")
                except Exception as e:
                    log.error(f"{symbol} {tf} tahlil xatosi: {e}")
                    await tg.send(f"❌ <b>{symbol} · {tf}</b>: xato — {e}")

        total = sum(len(v) for v in self.patterns.values())
        if total == 0:
            await tg.send("❌ Hech qanday naqsh topilmadi. Bot to'xtatildi.")
            await self.client.close_connection()
            return

        await tg.send(
            f"✅ <b>Umumiy {total} ta naqsh kuzatilmoqda.</b>\n"
            f"Endi real vaqtda signal kutamiz."
        )

        tasks = [asyncio.create_task(self.listen_stream(s, tf)) for (s, tf) in self.patterns.keys()]
        await asyncio.gather(*tasks)


# ============================================================
# ISHGA TUSHIRISH
# ============================================================
async def main():
    bot = LivePatternBot()
    try:
        await bot.run()
    except KeyboardInterrupt:
        log.info("To'xtatilmoqda...")
    finally:
        if bot.client:
            await bot.client.close_connection()


if __name__ == '__main__':
    asyncio.run(main())