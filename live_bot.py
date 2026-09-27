# ============================================================
# Crypto Deep Pattern Bot v8.1 (indikatorsiz)
# - 21 xil sham kodi
# - Confidence Score
# - R taqsimoti + Chart
# - 1:1R logika (50% @ 1R + BE, trail 4R dan)
# ============================================================
import os
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
# CONFIG
# ============================================================
CONFIG = {
    'SYMBOLS': [s.strip().upper() for s in os.getenv('SYMBOLS', 'BTCUSDT,ETHUSDT').split(',') if s.strip()],
    'TIMEFRAMES': [t.strip() for t in os.getenv('TIMEFRAMES', '5m,15m,1h').split(',') if t.strip()],

    'DAYS_PER_TF': {
        '1m':  int(os.getenv('DAYS_1M', '120')),
        '3m':  int(os.getenv('DAYS_3M', '200')),
        '5m':  int(os.getenv('DAYS_5M', '365')),
        '15m': int(os.getenv('DAYS_15M', '730')),
        '30m': int(os.getenv('DAYS_30M', '1095')),
        '1h':  int(os.getenv('DAYS_1H', '1825')),
        '4h':  int(os.getenv('DAYS_4H', '1825')),
    },

    'SEQ_LENGTHS': [int(x) for x in os.getenv('SEQ_LENGTHS', '2,3,4').split(',')],

    # Filtirlar
    'MIN_OCCURRENCES': int(os.getenv('MIN_OCCURRENCES', '25')),
    'MIN_WIN_RATE':   float(os.getenv('MIN_WIN_RATE', '58')),
    'MAX_LOSSES':     int(os.getenv('MAX_LOSSES', '30')),
    'MIN_PROFIT_FACTOR': float(os.getenv('MIN_PROFIT_FACTOR', '1.15')),
    'MIN_AVG_R':      float(os.getenv('MIN_AVG_R', '0.10')),
    'MIN_CONFIDENCE': float(os.getenv('MIN_CONFIDENCE', '55')),

    # Simulyatsiya
    'FORWARD_CANDLES': int(os.getenv('FORWARD_CANDLES', '50')),
    'SL_BUF': float(os.getenv('SL_BUF', '10')),

    # 1:1R logika
    'TP1_R': 1.0,
    'TP1_CLOSE_PCT': 0.5,
    'TRAIL_START_R': 4.0,
    'TRAIL_STEP_R': 2.0,
    'MAX_TRAIL_R': 10.0,

    'REQUEST_DELAY': float(os.getenv('REQUEST_DELAY', '0.25')),
    'CANDLE_BUFFER': 300,
    'ATR_PERIOD': 50,
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
# 2. ATR (faqat sham klassifikatsiyasi va SL uchun kerak)
# ============================================================
def compute_atr(df, period=50):
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(period, min_periods=10).mean().bfill().values
    return atr


# ============================================================
# 3. CHUQUR SHAM KODI (21 variant)
# ============================================================
def classify_candles(df):
    """
    Format: [Yo'nalish][Tana][Fitil] = 3 harf
      Yo'nalish: U, D, F
      Tana: S, M, L, X
      Fitil: N, u, d, U, D, C, B
    """
    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    atr = df['atr'].values

    body = c - o
    body_abs = np.abs(body)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l

    direction = np.where(body > 0, 'U', np.where(body < 0, 'D', 'F'))

    body_ratio = body_abs / np.maximum(atr, 1e-9)
    body_size = np.where(body_ratio < 0.3, 'S',
                np.where(body_ratio < 0.8, 'M',
                np.where(body_ratio < 1.5, 'L', 'X')))

    uw_ratio = upper_wick / np.maximum(body_abs, atr * 0.1)
    lw_ratio = lower_wick / np.maximum(body_abs, atr * 0.1)

    big_upper = uw_ratio > 1.2
    big_lower = lw_ratio > 1.2
    huge_upper = uw_ratio > 2.5
    huge_lower = lw_ratio > 2.5

    wick_class = np.where(huge_upper & huge_lower, 'B',
                  np.where(huge_upper, 'U',
                  np.where(huge_lower, 'D',
                  np.where(big_upper & big_lower, 'C',
                  np.where(big_upper, 'u',
                  np.where(big_lower, 'd', 'N'))))))

    codes = np.array([f"{d}{s}{w}" for d, s, w in zip(direction, body_size, wick_class)])
    out = df.copy()
    out['code'] = codes
    return out


def build_signatures(df_coded, seq_len):
    codes = df_coded['code'].values
    n = len(codes)
    sigs = [None] * n
    for i in range(seq_len - 1, n):
        sigs[i] = '_'.join(codes[i - seq_len + 1: i + 1])
    return sigs


# ============================================================
# 4. SIMULYATSIYA (1:1R logika)
# ============================================================
def simulate_forward(df, entry_idx, direction, sl_buf_cfg):
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    atr = df['atr'].values
    n = len(df)

    entry = c[entry_idx]
    buf = max(sl_buf_cfg * (entry / 100000), atr[entry_idx] * 0.15)

    if direction == 'B':
        sl_init = l[entry_idx] - buf
        sl_dist = entry - sl_init
    else:
        sl_init = h[entry_idx] + buf
        sl_dist = sl_init - entry

    if sl_dist <= 0:
        return None

    tp1_price = entry + CONFIG['TP1_R'] * sl_dist if direction == 'B' else entry - CONFIG['TP1_R'] * sl_dist

    tp1_hit = False
    sl = sl_init
    lock_r = 0.0
    max_r_reached = 0.0
    max_i = min(entry_idx + CONFIG['FORWARD_CANDLES'], n - 1)

    for i in range(entry_idx + 1, max_i + 1):
        if direction == 'B':
            if l[i] <= sl:
                if tp1_hit:
                    p1 = CONFIG['TP1_CLOSE_PCT'] * CONFIG['TP1_R']
                    p2 = (1 - CONFIG['TP1_CLOSE_PCT']) * ((sl - entry) / sl_dist)
                    return (p1 + p2, max_r_reached)
                return ((sl - entry) / sl_dist, max_r_reached)
            if not tp1_hit and h[i] >= tp1_price:
                tp1_hit = True
                sl = entry
            max_r = (h[i] - entry) / sl_dist
        else:
            if h[i] >= sl:
                if tp1_hit:
                    p1 = CONFIG['TP1_CLOSE_PCT'] * CONFIG['TP1_R']
                    p2 = (1 - CONFIG['TP1_CLOSE_PCT']) * ((entry - sl) / sl_dist)
                    return (p1 + p2, max_r_reached)
                return ((entry - sl) / sl_dist, max_r_reached)
            if not tp1_hit and l[i] <= tp1_price:
                tp1_hit = True
                sl = entry
            max_r = (entry - l[i]) / sl_dist

        if max_r > max_r_reached:
            max_r_reached = max_r

        if max_r >= CONFIG['TRAIL_START_R']:
            steps = int((max_r - CONFIG['TRAIL_START_R']) / CONFIG['TRAIL_STEP_R'])
            new_lock = CONFIG['TRAIL_START_R'] - CONFIG['TRAIL_STEP_R'] + steps * CONFIG['TRAIL_STEP_R']
            if new_lock > lock_r:
                lock_r = new_lock
                sl = entry + lock_r * sl_dist if direction == 'B' else entry - lock_r * sl_dist

        if max_r >= CONFIG['MAX_TRAIL_R']:
            if tp1_hit:
                p1 = CONFIG['TP1_CLOSE_PCT'] * CONFIG['TP1_R']
                p2 = (1 - CONFIG['TP1_CLOSE_PCT']) * CONFIG['MAX_TRAIL_R']
                return (p1 + p2, max_r_reached)
            return (CONFIG['MAX_TRAIL_R'], max_r_reached)

    final_r = (c[max_i] - entry) / sl_dist if direction == 'B' else (entry - c[max_i]) / sl_dist
    if tp1_hit:
        p1 = CONFIG['TP1_CLOSE_PCT'] * CONFIG['TP1_R']
        p2 = (1 - CONFIG['TP1_CLOSE_PCT']) * final_r
        return (p1 + p2, max_r_reached)
    return (final_r, max_r_reached)


# ============================================================
# 5. TAHLIL + CONFIDENCE
# ============================================================
def compute_r_distribution(max_r_list):
    return {
        'reached_1r':  int(sum(1 for m in max_r_list if m >= 1.0)),
        'reached_2r':  int(sum(1 for m in max_r_list if m >= 2.0)),
        'reached_4r':  int(sum(1 for m in max_r_list if m >= 4.0)),
        'reached_6r':  int(sum(1 for m in max_r_list if m >= 6.0)),
        'reached_8r':  int(sum(1 for m in max_r_list if m >= 8.0)),
        'reached_10r': int(sum(1 for m in max_r_list if m >= 10.0)),
    }


def compute_confidence(count, win_rate, pf, avg_r, consistency_std):
    score = 0.0
    score += min(count / 200, 1.0) * 25
    score += min(win_rate / 100, 1.0) * 25
    score += min(pf / 3.0, 1.0) * 20
    score += min(max(avg_r, 0) / 2.0, 1.0) * 20
    if consistency_std < 999:
        score += max(0, (1 - consistency_std / 30)) * 10
    return round(min(score, 100), 1)


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
            if s is None or i + CONFIG['FORWARD_CANDLES'] >= n:
                continue
            groups[s].append(i)

        for sig, idx_list in groups.items():
            if len(idx_list) < CONFIG['MIN_OCCURRENCES']:
                continue

            for direction in ['B', 'S']:
                r_values = []
                max_r_values = []
                years_map = defaultdict(list)

                for idx in idx_list:
                    res = simulate_forward(df_coded, idx, direction, CONFIG['SL_BUF'])
                    if res is None:
                        continue
                    final_r, max_r = res
                    r_values.append(final_r)
                    max_r_values.append(max_r)
                    years_map[int(df_coded['year'].iloc[idx])].append(final_r)

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

                yearly_wr = [float(np.mean(np.array(rs) > 0) * 100)
                             for yr, rs in years_map.items() if len(rs) >= 5]
                consistency_std = float(np.std(yearly_wr)) if len(yearly_wr) >= 2 else 999.0

                dist = compute_r_distribution(max_r_values)
                confidence = compute_confidence(len(r_values), win_rate, pf, avg_r, consistency_std)

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
                    'consistency_std': round(consistency_std, 1),
                    'confidence': confidence,
                    **dist,
                })

    return results


def filter_and_rank(results):
    good = [r for r in results
            if r['profit_factor'] >= CONFIG['MIN_PROFIT_FACTOR']
            and r['win_rate'] >= CONFIG['MIN_WIN_RATE']
            and r['losses'] <= CONFIG['MAX_LOSSES']
            and r['avg_r'] >= CONFIG['MIN_AVG_R']
            and r['confidence'] >= CONFIG['MIN_CONFIDENCE']]
    good.sort(key=lambda r: r['confidence'], reverse=True)
    return good


# ============================================================
# 6. GRAFIK
# ============================================================
def make_signal_chart(df_coded, pattern, symbol, tf):
    seq_len = pattern['seq_len']
    sig_col = f'sig_{seq_len}'
    all_idx = df_coded.index[df_coded[sig_col] == pattern['signature']].tolist()

    total = len(df_coded)
    window_start = max(0, total - 120)
    window = df_coded.iloc[window_start:].reset_index(drop=True)
    offset = window_start
    n = len(window)

    fig = plt.figure(figsize=(14, 8), facecolor='#0a0b0f')
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.2], hspace=0.35)

    ax = fig.add_subplot(gs[0])
    ax.set_facecolor('#0a0b0f')

    for i, row in window.iterrows():
        c = '#10b981' if row['close'] >= row['open'] else '#ef4444'
        ax.plot([i, i], [row['low'], row['high']], color=c, linewidth=1, alpha=0.85)
        ax.plot([i, i], [row['open'], row['close']], color=c, linewidth=3, alpha=0.85)

    for idx in all_idx:
        rel = idx - offset
        if 0 <= rel < n - 1:
            y = window['high'].iloc[rel] * 1.001
            ax.annotate('▼', xy=(rel, y), color='#facc15', fontsize=10,
                        ha='center', va='bottom', alpha=0.9)
            ax.axvspan(rel - seq_len + 1, rel, color='#facc15', alpha=0.05)

    last_rel = n - 1
    arrow = '▲ BUY' if pattern['direction'] == 'BUY' else '▼ SELL'
    arrow_color = '#10b981' if pattern['direction'] == 'BUY' else '#ef4444'
    ax.annotate(arrow,
                xy=(last_rel, window['close'].iloc[-1]),
                xytext=(last_rel, window['high'].max() * 1.003),
                color=arrow_color, fontsize=15, fontweight='bold', ha='center')
    ax.axvspan(last_rel - seq_len + 1, last_rel, color=arrow_color, alpha=0.25)

    ax.set_title(
        f"{symbol} · {tf} · {pattern['signature']} · {pattern['direction']}   "
        f"| {pattern['count']} marta | WR {pattern['win_rate']}% | Conf {pattern['confidence']}",
        color='#e2e8f0', fontsize=11
    )
    ax.tick_params(colors='#94a3b8', labelsize=8)
    ax.grid(True, alpha=0.1, color='#ffffff')
    for sp in ax.spines.values():
        sp.set_color('#ffffff20')

    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#0a0b0f')
    ax2.axis('off')

    labels = ['1R', '2R', '4R', '6R', '8R', '10R']
    counts = [pattern.get('reached_1r', 0), pattern.get('reached_2r', 0),
              pattern.get('reached_4r', 0), pattern.get('reached_6r', 0),
              pattern.get('reached_8r', 0), pattern.get('reached_10r', 0)]
    total_c = pattern['count'] if pattern['count'] > 0 else 1
    pcts = [c / total_c * 100 for c in counts]
    colors = ['#3b82f6', '#22d3ee', '#10b981', '#84cc16', '#eab308', '#f97316']

    bars = ax2.barh(range(len(labels)), counts, color=colors, height=0.65)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, color='#e2e8f0', fontsize=10)
    ax2.set_xlim(0, max(counts) * 1.3 if max(counts) > 0 else 10)
    ax2.invert_yaxis()
    for b, c, p in zip(bars, counts, pcts):
        ax2.text(b.get_width() + max(counts) * 0.02 if max(counts) > 0 else 0.5,
                 b.get_y() + b.get_height() / 2,
                 f"{c} marta ({p:.1f}%)",
                 color='#e2e8f0', fontsize=9, va='center')
    ax2.set_title("📊 Necha marta qaysi R darajasiga yetgan",
                  color='#e2e8f0', fontsize=10)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=95, facecolor='#0a0b0f')
    plt.close(fig)
    buf.seek(0)
    return buf


# ============================================================
# 7. JONLI BOT
# ============================================================
class LivePatternBot:
    def __init__(self):
        self.patterns = {}
        self.buffers = {}
        self.signaled = set()
        self.history_cache = {}
        self.client = None
        self.bm = None

    async def find_patterns_for(self, symbol, tf):
        days = CONFIG['DAYS_PER_TF'].get(tf, 365)
        df = await fetch_history(self.client, symbol, tf, days)
        if len(df) < 500:
            return []
        df['atr'] = compute_atr(df, CONFIG['ATR_PERIOD'])
        df_coded = classify_candles(df)
        for seq_len in CONFIG['SEQ_LENGTHS']:
            df_coded[f'sig_{seq_len}'] = build_signatures(df_coded, seq_len)
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
        df['atr'] = compute_atr(df, CONFIG['ATR_PERIOD'])
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

            sig_key = (symbol, tf, new_candle['open_time'],
                       pattern['signature'], pattern['direction'])
            if sig_key in self.signaled:
                continue
            self.signaled.add(sig_key)
            if len(self.signaled) > 5000:
                self.signaled = set(list(self.signaled)[-2000:])

            asyncio.create_task(self._fire_signal(symbol, tf, pattern, df_coded))

    async def _fire_signal(self, symbol, tf, pattern, df_coded):
        price = float(df_coded['close'].iloc[-1])
        log.info(f"🚨 SIGNAL: {symbol} {tf} {pattern['signature']} {pattern['direction']} @ {price}")

        wins = pattern['wins']
        losses = pattern['losses']
        total = wins + losses
        outcome = "✅ FOYDA" if pattern['avg_r'] > 0 else "❌ ZARAR"
        expected = pattern['avg_r'] * 100

        msg = (
            f"🚨 <b>YANGI SIGNAL — {symbol}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>📊 Juftlik:</b> {symbol}\n"
            f"<b>⏱ Timeframe:</b> {tf}\n"
            f"<b>🎯 Yo'nalish:</b> <b>{pattern['direction']}</b>\n"
            f"<b>💰 Narx:</b> <code>{price}</code>\n"
            f"<b>🔍 Naqsh:</b> <code>{pattern['signature']}</code>\n"
            f"<b>🧠 Ishonch:</b> <b>{pattern['confidence']}/100</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>📈 TARIXIY NATIJA:</b>\n"
            f"├ Jami: <b>{total}</b> marta\n"
            f"├ ✅ Foyda: <b>{wins}</b> | ❌ Zarar: <b>{losses}</b>\n"
            f"├ 🏆 Win Rate: <b>{pattern['win_rate']}%</b>\n"
            f"├ ⚖️ Profit Factor: <b>{pattern['profit_factor']}</b>\n"
            f"└ 📉 Barqarorlik: {pattern['consistency_std']}%\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>🎯 R DARAJALARI:</b>\n"
            f"├ 1R → {pattern.get('reached_1r',0)} | 2R → {pattern.get('reached_2r',0)}\n"
            f"├ 4R → {pattern.get('reached_4r',0)} | 6R → {pattern.get('reached_6r',0)}\n"
            f"└ 8R → {pattern.get('reached_8r',0)} | 10R → {pattern.get('reached_10r',0)}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>💼 KUTILAYOTGAN:</b> {pattern['avg_r']:+.2f}R | 100$ riskda ~{expected:+.0f}$\n"
            f"<b>{outcome}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>⚙️ BOSHQARUV:</b>\n"
            f"├ 1R da 50% yopiladi + BE\n"
            f"├ 4R dan trailing (har 2R)\n"
            f"└ Maksimal 10R\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚠️ <i>Tarixiy tahlilga asoslangan.</i>"
        )

        try:
            hist_df = self.history_cache.get((symbol, tf), df_coded)
            loop = asyncio.get_event_loop()
            buf = await loop.run_in_executor(None, make_signal_chart, hist_df, pattern, symbol, tf)
            await tg.photo(buf, caption=msg)
        except Exception as e:
            log.error(f"Signal xato: {e}")
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
                log.error(f"❌ {symbol} {tf} stream xato: {e}. 5s dan keyin")
                await asyncio.sleep(5)

    async def run(self):
        log.info("🚀 Deep Pattern Bot v8.1 ishga tushmoqda")
        await tg.send(
            f"🚀 <b>Deep Pattern Bot v8.1</b>\n"
            f"<b>Symbols:</b> {', '.join(CONFIG['SYMBOLS'])}\n"
            f"<b>TF:</b> {', '.join(CONFIG['TIMEFRAMES'])}\n"
            f"<b>Min takrorlanish:</b> {CONFIG['MIN_OCCURRENCES']}\n"
            f"<b>Min WR:</b> {CONFIG['MIN_WIN_RATE']}%\n"
            f"<b>Min Confidence:</b> {CONFIG['MIN_CONFIDENCE']}/100\n"
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
                            f"✅ <b>{symbol} · {tf}</b>: {len(top)} ta naqsh.\n\n"
                            f"🥇 Eng yaxshisi:\n"
                            f"<code>{best['signature']}</code> → <b>{best['direction']}</b>\n"
                            f"├ {best['count']} marta | WR {best['win_rate']}%\n"
                            f"├ Foyda: {best['wins']} | Zarar: {best['losses']}\n"
                            f"├ PF: {best['profit_factor']} | avg: {best['avg_r']:+.2f}R\n"
                            f"├ 1R: {best.get('reached_1r',0)} | 4R: {best.get('reached_4r',0)} | 10R: {best.get('reached_10r',0)}\n"
                            f"└ 🧠 Ishonch: <b>{best['confidence']}/100</b>"
                        )
                    else:
                        await tg.send(f"⚠️ <b>{symbol} · {tf}</b>: mos naqsh yo'q")
                except Exception as e:
                    log.error(f"{symbol} {tf} xato: {e}")
                    await tg.send(f"❌ <b>{symbol} · {tf}</b>: {e}")

        total = sum(len(v) for v in self.patterns.values())
        if total == 0:
            await tg.send("❌ Naqsh topilmadi. Bot to'xtatildi.")
            await self.client.close_connection()
            return

        await tg.send(f"✅ <b>Jami {total} ta naqsh kuzatilmoqda.</b>\nSignal kutamiz...")

        tasks = [asyncio.create_task(self.listen_stream(s, tf)) for (s, tf) in self.patterns.keys()]
        await asyncio.gather(*tasks)


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