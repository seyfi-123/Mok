# ============================================================
# Crypto Live Pattern Bot v3.0 (Birlashtirilgan)
# Tahlil + Jonli kuzatuv + Telegram signal + Chart
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
    'MIN_WIN_RATE':   float(os.getenv('MIN_WIN_RATE', '0')),
    'MIN_PROFIT_FACTOR': float(os.getenv('MIN_PROFIT_FACTOR', '1.1')),
    'FORWARD_CANDLES': int(os.getenv('FORWARD_CANDLES', '50')),
    'SL_BUF': float(os.getenv('SL_BUF', '10')),
    'BE_AT_R': 1.0,
    'TRAIL_STEP': 2.0,
    'MAX_TRAIL_R': 10.0,
    'REQUEST_DELAY': float(os.getenv('REQUEST_DELAY', '0.25')),
    'CANDLE_BUFFER': 250,
    'PATTERNS_FILE': 'pattern_results.json',
}

TELEGRAM_TOKEN   = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
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
            await self.bot.send_photo(chat_id=self.chat_id, photo=InputFile(buf, filename='signal.png'), caption=caption, parse_mode=ParseMode.HTML)
        except Exception as e:
            log.error(f"TG photo: {e}")

tg = TG(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)

# ============================================================
# TAHLIL QISMI (O'ZGARTIRILMAGAN MANTIQ)
# ============================================================
async def fetch_history(client, symbol, interval, days):
    end_time = int(time.time() * 1000)
    start_time = end_time - days * 24 * 60 * 60 * 1000
    all_klines = []
    cursor = start_time
    log.info(f"📥 {symbol} {interval}: {days} kunlik tarix yuklanmoqda...")

    while cursor < end_time:
        try:
            klines = await client.get_klines(symbol=symbol, interval=interval, startTime=cursor, limit=1000)
        except Exception as e:
            log.error(f"{symbol} {interval} fetch xato: {e} — 2s kutamiz")
            await asyncio.sleep(2); continue
        if not klines: break
        all_klines.extend(klines)
        last_open_time = klines[-1][0]
        if last_open_time <= cursor: break
        cursor = last_open_time + 1
        await asyncio.sleep(CONFIG['REQUEST_DELAY'])

    df = pd.DataFrame(all_klines, columns=['open_time','open','high','low','close','volume','close_time','qav','trades','tbbav','tbqav','ignore'])
    for col in ['open','high','low','close','volume']: df[col] = df[col].astype(float)
    df['open_time'] = df['open_time'] // 1000
    log.info(f"✅ {symbol} {interval}: {len(df)} ta candle yuklandi")
    return df[['open_time','open','high','low','close','volume']].reset_index(drop=True)

def classify_candles(df):
    o, h, l, c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
    tr = np.maximum(h - l, 1e-9)
    atr = pd.Series(tr).rolling(50, min_periods=10).mean().bfill().values
    body = c - o; body_abs = np.abs(body)
    upper_wick = h - np.maximum(o, c); lower_wick = np.minimum(o, c) - l
    direction = np.where(body > 0, 'U', np.where(body < 0, 'D', 'F'))
    body_ratio = body_abs / atr
    body_size = np.where(body_ratio < 0.4, 'S', np.where(body_ratio < 1.2, 'M', 'L'))
    uw_ratio = upper_wick / np.maximum(body_abs, atr * 0.1)
    lw_ratio = lower_wick / np.maximum(body_abs, atr * 0.1)
    big_upper = uw_ratio > 1.0; big_lower = lw_ratio > 1.0
    wick_class = np.where(big_upper & big_lower, 'B', np.where(big_upper, 'U', np.where(big_lower, 'D', 'N')))
    codes = np.array([f"{d}{s}{w}" for d, s, w in zip(direction, body_size, wick_class)])
    out = df.copy(); out['code'] = codes; out['atr'] = atr
    return out

def build_signatures(df_coded, seq_len):
    codes = df_coded['code'].values; n = len(codes)
    sigs = [None] * n
    for i in range(seq_len - 1, n): sigs[i] = '_'.join(codes[i - seq_len + 1: i + 1])
    return sigs

def simulate_forward(df, entry_idx, direction, sl_buf_cfg):
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    n = len(df); entry = c[entry_idx]; buf = sl_buf_cfg * (entry / 100000)
    if direction == 'B': sl = l[entry_idx] - buf; sl_dist = entry - sl
    else: sl = h[entry_idx] + buf; sl_dist = sl - entry
    if sl_dist <= 0: return None
    be_set = False; lock_r = 0.0; max_i = min(entry_idx + CONFIG['FORWARD_CANDLES'], n - 1)
    for i in range(entry_idx + 1, max_i + 1):
        if direction == 'B':
            if l[i] <= sl: return (sl - entry) / sl_dist
            max_r = (h[i] - entry) / sl_dist
        else:
            if h[i] >= sl: return (entry - sl) / sl_dist
            max_r = (entry - l[i]) / sl_dist
        if max_r >= CONFIG['BE_AT_R'] and not be_set: sl = entry; be_set = True
        if max_r >= CONFIG['MAX_TRAIL_R']: return CONFIG['MAX_TRAIL_R']
        if max_r >= CONFIG['BE_AT_R']:
            steps = int(max_r / CONFIG['TRAIL_STEP']); new_lock = (steps - 1) * CONFIG['TRAIL_STEP']
            if new_lock > lock_r:
                lock_r = new_lock; sl = entry + lock_r * sl_dist if direction == 'B' else entry - lock_r * sl_dist
    return (c[max_i] - entry) / sl_dist if direction == 'B' else (entry - c[max_i]) / sl_dist

def analyze_timeframe(df, tf_name):
    df_coded = classify_candles(df); n = len(df_coded); results = []
    df_coded['year'] = pd.to_datetime(df_coded['open_time'], unit='s').dt.year
    for seq_len in CONFIG['SEQ_LENGTHS']:
        sigs = build_signatures(df_coded, seq_len); df_coded[f'sig_{seq_len}'] = sigs
        groups = defaultdict(list)
        for i, s in enumerate(sigs):
            if s is None or i + CONFIG['FORWARD_CANDLES'] >= n: continue
            groups[s].append(i)
        for sig, idx_list in groups.items():
            if len(idx_list) < CONFIG['MIN_OCCURRENCES']: continue
            for direction in ['B', 'S']:
                r_values = []; years_map = defaultdict(list)
                for idx in idx_list:
                    r = simulate_forward(df_coded, idx, direction, CONFIG['SL_BUF'])
                    if r is None: continue
                    r_values.append(r); years_map[int(df_coded['year'].iloc[idx])].append(r)
                if len(r_values) < CONFIG['MIN_OCCURRENCES']: continue
                r_arr = np.array(r_values)
                win_rate = float((r_arr > 0).mean() * 100); avg_r = float(r_arr.mean())
                gross_win = r_arr[r_arr > 0].sum(); gross_loss = abs(r_arr[r_arr < 0].sum())
                pf = float(gross_win / gross_loss) if gross_loss > 0 else 999.0
                yearly_wr = [np.mean(np.array(rs) > 0) * 100 for yr, rs in years_map.items() if len(rs) >= 10]
                consistency_std = float(np.std(yearly_wr)) if len(yearly_wr) >= 2 else 999.0
                is_consistent = len(yearly_wr) >= 2 and consistency_std < 15.0
                results.append({
                    'tf': tf_name, 'seq_len': seq_len, 'signature': sig,
                    'direction': 'BUY' if direction == 'B' else 'SELL',
                    'count': len(r_values), 'win_rate': round(win_rate, 1),
                    'avg_r': round(avg_r, 3), 'min_r': round(float(r_arr.min()), 2),
                    'max_r': round(float(r_arr.max()), 2), 'profit_factor': round(pf, 2),
                    'consistent': is_consistent, 'consistency_std': round(consistency_std, 1),
                    'years_covered': len(yearly_wr),
                })
    return results

def filter_and_rank(results):
    good = [r for r in results if r['consistent'] and r['profit_factor'] >= CONFIG['MIN_PROFIT_FACTOR'] and r['win_rate'] >= CONFIG['MIN_WIN_RATE']]
    good.sort(key=lambda r: (r['avg_r'] * r['win_rate'] / 100), reverse=True)
    return good

# ============================================================
# GRAFIK VA JONLI BOT
# ============================================================
def make_signal_chart(df_window, pattern, symbol, tf):
    seq_len = pattern['seq_len']; df_window = df_window.reset_index(drop=True); n = len(df_window)
    highlight_start = n - seq_len; highlight_end = n - 1
    fig, ax = plt.subplots(figsize=(12, 5.5), facecolor='#0a0b0f'); ax.set_facecolor('#0a0b0f')
    for i, row in df_window.iterrows():
        c = '#10b981' if row['close'] >= row['open'] else '#ef4444'
        in_pat = highlight_start <= i <= highlight_end
        alpha = 1.0 if in_pat else 0.55
        ax.plot([i, i], [row['low'], row['high']], color=c, linewidth=1, alpha=alpha)
        ax.plot([i, i], [row['open'], row['close']], color=c, linewidth=5 if in_pat else 3, alpha=alpha)
    ax.axvspan(highlight_start - 0.4, highlight_end + 0.4, color='yellow', alpha=0.10)
    arrow = '▲ BUY' if pattern['direction'] == 'BUY' else '▼ SELL'
    arrow_color = '#10b981' if pattern['direction'] == 'BUY' else '#ef4444'
    ax.annotate(arrow, xy=(highlight_end, df_window['close'].iloc[-1]), xytext=(highlight_end, df_window['high'].max() * 1.001), color=arrow_color, fontsize=13, fontweight='bold', ha='center')
    ax.set_title(f"{symbol} · {tf} · {pattern['signature']} · {pattern['direction']}   | WR {pattern['win_rate']}%  |  avg {pattern['avg_r']:+.2f}R  |  PF {pattern['profit_factor']}  |  {pattern['count']} marta", color='#e2e8f0', fontsize=11)
    ax.tick_params(colors='#94a3b8', labelsize=8); ax.grid(True, alpha=0.1, color='#ffffff')
    for sp in ax.spines.values(): sp.set_color('#ffffff20')
    plt.tight_layout(); buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=95, facecolor='#0a0b0f'); plt.close(fig); buf.seek(0)
    return buf

class LivePatternBot:
    def __init__(self):
        self.patterns = {}; self.buffers = {}; self.signaled = set()
        self.client = None; self.bm = None

    async def find_patterns_for(self, symbol, tf):
        days = CONFIG['DAYS_PER_TF'].get(tf, 365)
        df = await fetch_history(self.client, symbol, tf, days)
        if len(df) < 500: return []
        results = analyze_timeframe(df, tf)
        return filter_and_rank(results)

    def _on_closed_candle(self, symbol, tf, new_candle):
        key = (symbol, tf)
        if key not in self.buffers: self.buffers[key] = deque(maxlen=CONFIG['CANDLE_BUFFER'])
        self.buffers[key].append(new_candle)
        if len(self.buffers[key]) < 60: return
        df = pd.DataFrame(list(self.buffers[key])); df_coded = classify_candles(df)
        tf_patterns = self.patterns.get(key, [])
        for pattern in tf_patterns:
            seq_len = pattern['seq_len']
            if len(df_coded) < seq_len: continue
            recent = df_coded['code'].iloc[-seq_len:].tolist(); current_sig = '_'.join(recent)
            if current_sig != pattern['signature']: continue
            sig_key = (symbol, tf, new_candle['open_time'], pattern['signature'], pattern['direction'])
            if sig_key in self.signaled: continue
            self.signaled.add(sig_key)
            if len(self.signaled) > 5000: self.signaled = set(list(self.signaled)[-2000:])
            asyncio.create_task(self._fire_signal(symbol, tf, pattern, df_coded))

    async def _fire_signal(self, symbol, tf, pattern, df_coded):
        price = float(df_coded['close'].iloc[-1])
        log.info(f"🚨 SIGNAL: {symbol} {tf} {pattern['signature']} {pattern['direction']} @ {price}")
        msg = (f"🚨 <b>YANGI SIGNAL — {symbol}</b>\n━━━━━━━━━━━━━━━━━━━━━━\n"
               f"<b>TF:</b> {tf}\n<b>Naqsh:</b> <code>{pattern['signature']}</code>\n"
               f"<b>Yo'nalish:</b> <b>{pattern['direction']}</b>\n<b>Narx:</b> <code>{price}</code>\n"
               f"<b>Tarixiy WR:</b> {pattern['win_rate']}%\n<b>O'rtacha R:</b> {pattern['avg_r']:+.2f}R\n"
               f"<b>PF:</b> {pattern['profit_factor']} | <b>Uchragan:</b> {pattern['count']} marta\n"
               f"━━━━━━━━━━━━━━━━━━━━━━\n⚠️ <i>Tarixiy tahlilga asoslangan signal. Xavfni o'zingiz boshqaring.</i>")
        try:
            df_window = df_coded.tail(60).copy()
            loop = asyncio.get_event_loop()
            buf = await loop.run_in_executor(None, make_signal_chart, df_window, pattern, symbol, tf)
            await tg.photo(buf, caption=msg)
        except Exception as e:
            log.error(f"Signal yuborishda xato: {e}"); await tg.send(msg)

    async def listen_stream(self, symbol, tf):
        while True:
            try:
                socket = self.bm.kline_socket(symbol=symbol, interval=tf)
                async with socket as stream:
                    log.info(f"🔌 {symbol} {tf}: WebSocket ulandi")
                    while True:
                        msg = await stream.recv()
                        if msg.get('e') != 'kline' or not msg['k']['x']: continue
                        k = msg['k']
                        candle = {'open_time': int(k['t']/1000),'open': float(k['o']),'high': float(k['h']),'low': float(k['l']),'close': float(k['c']),'volume': float(k['v'])}
                        self._on_closed_candle(symbol, tf, candle)
            except Exception as e:
                log.error(f"❌ {symbol} {tf} stream xato: {e}. 5s dan keyin qayta ulanamiz"); await asyncio.sleep(5)

    async def run(self):
        log.info("🚀 Live Pattern Bot v3.0 ishga tushmoqda")
        await tg.send(f"🚀 <b>Live Pattern Bot v3.0 ishga tushdi</b>\n<b>Symbols:</b> {', '.join(CONFIG['SYMBOLS'])}\n<b>TF:</b> {', '.join(CONFIG['TIMEFRAMES'])}\n⏳ Tahlil boshlandi...")
        self.client = await AsyncClient.create()
        self.bm = BinanceSocketManager(self.client)
        
        self.patterns = {}
        for symbol in CONFIG['SYMBOLS']:
            for tf in CONFIG['TIMEFRAMES']:
                try:
                    top = await self.find_patterns_for(symbol, tf)
                    if top:
                        self.patterns[(symbol, tf)] = top
                        await tg.send(f"✅ <b>{symbol} · {tf}</b>: {len(top)} ta naqsh topildi.\nEng yaxshisi: <code>{top[0]['signature']}</code> ({top[0]['direction']}, WR {top[0]['win_rate']}%, avg {top[0]['avg_r']:+.2f}R)")
                    else:
                        await tg.send(f"⚠️ <b>{symbol} · {tf}</b>: mos naqsh topilmadi.")
                except Exception as e:
                    log.error(f"{symbol} {tf} tahlil xatosi: {e}")
        
        total = sum(len(v) for v in self.patterns.values())
        if total == 0:
            await tg.send("❌ Hech qanday naqsh topilmadi. Bot to'xtatildi.")
            await self.client.close_connection(); return

        await tg.send(f"✅ <b>Umumiy {total} ta naqsh kuzatilmoqda.</b>\nEndi real vaqtda signal kutamiz.")
        tasks = [asyncio.create_task(self.listen_stream(s, tf)) for (s, tf) in self.patterns.keys()]
        await asyncio.gather(*tasks)

async def main():
    bot = LivePatternBot()
    try: await bot.run()
    except KeyboardInterrupt: log.info("To'xtatilmoqda...")
    finally:
        if bot.client: await bot.client.close_connection()

if __name__ == '__main__':
    asyncio.run(main())