#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deep Historical Market Analogue Engine v5.1 (Elite filter)
pip install numpy requests websocket-client matplotlib
"""
import os, sys, json, math, time, re, signal, sqlite3, logging, threading
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import requests
import websocket
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from numpy.lib.stride_tricks import sliding_window_view as swv

np.seterr(all="ignore")
APP, VERSION = "Deep Historical Market Analogue Engine", "5.1.0"

# ======================= SOZLAMALAR =======================
def _s(n, d): return os.getenv(n, str(d))
def _f(n, d): return float(_s(n, d))
def _i(n, d): return int(_s(n, d))
def _b(n, d): return _s(n, d).lower() in ("1", "true", "yes", "on")
def _l(n, d): return [x.strip() for x in _s(n, d).split(",") if x.strip()]

SPOT = _s("BINANCE_REST_URL", "https://api.binance.com").rstrip("/")
FAPI = _s("BINANCE_FAPI_URL", "https://fapi.binance.com").rstrip("/")
WS_URL = _s("BINANCE_WS_URL", "wss://stream.binance.com:9443/stream")
TG_TOKEN, TG_CHAT = _s("TELEGRAM_BOT_TOKEN", ""), _s("TELEGRAM_CHAT_ID", "")
TG_ON = _b("TELEGRAM_ENABLED", True) and bool(TG_TOKEN and TG_CHAT)
PORT = _i("PORT", 8080)
SYMBOLS = [x.upper() for x in _l("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT")]
TFS = _l("TIMEFRAMES", "5m,15m,1h")
PRIMARY = _s("PRIMARY_TF", "15m")
if PRIMARY not in TFS:
    PRIMARY = TFS[len(TFS) // 2]
HIST_DAYS = {"5m": _i("DAYS_5M", 90), "15m": _i("DAYS_15M", 365), "1h": _i("DAYS_1H", 730)}
MAX_CANDLES = _i("MAX_CANDLES", 60000)
TIMEOUT = _i("REQUEST_TIMEOUT", 20)

# Label (triple-barrier): TP=+K*ATR, SL=-K*ATR, HORIZON sham ichida
HORIZON = _i("HORIZON_BARS", 16)
K = _f("BARRIER_ATR", 1.5)
COST_PCT = _f("COST_PCT", 0.08)          # komissiya+slippage, aylanma, %
# Kengar qatlam (statistika uchun)
TARGET_N = _i("TARGET_N", 150)
MIN_EFF = _i("MIN_EFF", 40)
MIN_DEC = _i("MIN_DECISIVE", 30)
MIN_SIM = _f("MIN_SIM", 0.50)
POOL = _i("POOL", 4000)
GAP = _i("CLUSTER_GAP", HORIZON)
REINDEX = _i("REINDEX_BARS", 48)
# ELITE qatlam (qattiq filtr): 1..10 ta eng kuchli analog
ELITE_SIM = _f("ELITE_SIM", 0.98)        # kosinus o'xshashlik, 0.98 - 0.99
ELITE_NORM = _f("ELITE_NORM", 0.85)      # vektor uzunligi nisbati (kuch mosligi)
ELITE_MAX = _i("ELITE_MAX", 10)
ELITE_MIN = _i("ELITE_MIN", 3)           # 1 ga tushirsa bo'ladi, lekin ishonchsiz
ELITE_AGREE = _f("ELITE_AGREE", 0.70)    # elite'ning shu qismi signal tomoniga mos bo'lishi shart
# Signal gate
MIN_WIN_LB = _f("MIN_WIN_LB", 0.53)
EDGE_OVER_BASE = _f("EDGE_OVER_BASE", 0.02)
MIN_EXP = _f("MIN_EXP_R", 0.05)
MIN_TF = _i("MIN_TF_AGREE", 2)
COOLDOWN = _i("COOLDOWN_MIN", 90)
FUND_EXT = _f("FUNDING_EXTREME_PCT", 0.05)
# Risk
EQUITY = _f("EQUITY_USDT", 1000)
RISK = _f("RISK_PCT", 0.5)
MAXLEV = _f("MAX_LEVERAGE", 3)
# Yangilik
NEWS_ON = _b("NEWS_ENABLED", True)
FEEDS = _l("RSS_URLS", "https://feeds.feedburner.com/CoinDesk,https://cointelegraph.com/rss")
NEWS_SEC = _i("NEWS_SCAN_SECONDS", 300)
RISK_WINDOW_MIN = _i("NEWS_RISK_WINDOW_MIN", 180)
BLACKOUTS = _l("EVENT_BLACKOUTS", "")     # masalan: 2026-10-29T18:00:00Z,2026-11-12T13:30:00Z
BLACKOUT_MIN = _i("BLACKOUT_MIN", 45)
CHART_ON = _b("CHART_ENABLED", True)
DB_PATH = _s("DB_PATH", "signals.db")
WARM = 100

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("engine")

STOP = False
LAST_WS = time.time()
WS_APP = None
START = time.time()
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "AnalogueEngine/5.1"
EX = ThreadPoolExecutor(max_workers=3)
SYM_LOCK = defaultdict(threading.Lock)

# ======================= YORDAMCHI =======================
def now_ms(): return int(time.time() * 1000)

def TF_SEC(tf): return int(tf[:-1]) * {"m": 60, "h": 3600, "d": 86400}[tf[-1]]

def fmt(v):
    v = float(v)
    return f"{v:,.2f}" if v >= 1000 else f"{v:,.4f}" if v >= 1 else f"{v:.8f}"

def wilson(k, n, z=1.96):
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - a) / d, (c + a) / d

def sleep_i(sec):
    end = time.time() + sec
    while not STOP and time.time() < end:
        time.sleep(min(1.0, max(0.0, end - time.time())))

def http_get(url, params=None, retries=4):
    err = None
    for a in range(retries):
        try:
            r = SESSION.get(url, params=params or {}, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (418, 429, 500, 502, 503, 504):
                time.sleep(min(30, 2 ** a))
                continue
            r.raise_for_status()
        except requests.HTTPError:
            raise
        except Exception as e:
            err = e
            time.sleep(min(30, 2 ** a))
    raise RuntimeError(f"so'rov muvaffaqiyatsiz: {url} {err}")

# ======================= MA'LUMOT OMBORI =======================
class Store:
    """Faqat YOPILGAN shamlar. Ustunlar: open_time, o, h, l, c, v"""
    def __init__(self):
        self.d, self.lock = {}, threading.RLock()

    def get(self, sym, tf):
        with self.lock:
            a = self.d.get((sym, tf))
            return None if a is None else a.copy()

    def merge(self, sym, tf, new):
        """Qo'shadi, dublikatni yangisi bilan almashtiradi. Oxirida gap bo'lsa True."""
        with self.lock:
            a = self.d.get((sym, tf))
            comb = new if a is None or len(a) == 0 else np.vstack([a, new])
            if len(comb) == 0:
                return False
            rev = comb[::-1]
            idx = np.unique(rev[:, 0], return_index=True)[1]
            comb = rev[idx]
            if len(comb) > MAX_CANDLES:
                comb = comb[-MAX_CANDLES:]
            self.d[(sym, tf)] = comb
            step = TF_SEC(tf) * 1000
            return len(comb) >= 2 and (comb[-1, 0] - comb[-2, 0]) != step

STORE = Store()

def fetch_klines(sym, tf, start_ms, end_ms=None):
    end_ms = end_ms or now_ms()
    step = TF_SEC(tf) * 1000
    out, cur = [], int(start_ms)
    while cur < end_ms and not STOP:
        rows = http_get(SPOT + "/api/v3/klines", {"symbol": sym, "interval": tf,
                        "startTime": cur, "endTime": int(end_ms), "limit": 1000})
        if not rows:
            break
        out.extend(rows)
        nxt = int(rows[-1][0]) + step
        if nxt <= cur or len(rows) < 1000:
            break
        cur = nxt
        time.sleep(0.05)
    if not out:
        return np.empty((0, 6))
    a = np.array([[r[0], r[1], r[2], r[3], r[4], r[5], r[6]] for r in out], dtype=float)
    a = a[a[:, 6] <= now_ms()]                       # faqat yopilgan
    a = a[np.unique(a[:, 0], return_index=True)[1]]
    return a[:, :6]

def backfill(sym, tf):
    try:
        arr = STORE.get(sym, tf)
        if arr is None:
            return
        new = fetch_klines(sym, tf, int(arr[-1, 0]) + TF_SEC(tf) * 1000)
        if len(new):
            STORE.merge(sym, tf, new)
            log.info("Backfill: %s %s +%d", sym, tf, len(new))
    except Exception as e:
        log.warning("Backfill xatosi %s %s: %s", sym, tf, e)

# ======================= FEATURELAR (ATR bilan normallashgan) =======================
def roll_mean(x, w):
    cs = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full(len(x), np.nan)
    out[w - 1:] = (cs[w:] - cs[:-w]) / w
    return out

def ema(x, span):
    a = 2.0 / (span + 1)
    out = np.empty(len(x))
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out

def compute(arr):
    """Qaytaradi: X (n,15), atr, regime. Warm-up qatorlarda NaN."""
    o, h, l, c, v = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5]
    n = len(c)
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = roll_mean(tr, 14)
    atr_l = roll_mean(tr, 100)
    a = np.where(atr > 0, atr, np.nan)

    def lag(x, k):
        r = np.full(n, np.nan)
        r[k:] = x[:-k]
        return r

    f = [np.clip((c - lag(c, k)) / a, -10, 10) for k in (1, 3, 6, 12, 24)]
    e20 = ema(c, 20)
    f.append(np.clip((c - e20) / a, -10, 10))
    slope = np.clip((e20 - lag(e20, 5)) / a, -10, 10)
    f.append(slope)
    hh, ll = np.full(n, np.nan), np.full(n, np.nan)
    if n >= 50:
        hh[49:] = swv(h, 50).max(1)
        ll[49:] = swv(l, 50).min(1)
    f.append(np.clip((c - ll) / np.where(hh - ll > 0, hh - ll, np.nan), 0, 1))
    f.append(np.clip(atr / atr_l, 0, 4))
    vm = roll_mean(v, 20)
    f.append(np.clip(v / np.where(vm > 0, vm, np.nan), 0, 5))
    rng = np.where(h - l > 0, h - l, np.nan)
    f.append(np.abs(c - o) / rng)
    f.append((h - np.maximum(o, c)) / rng)
    f.append((np.minimum(o, c) - l) / rng)
    f.append(np.clip((c - o) / a, -5, 5))
    f.append(np.clip(roll_mean(h - l, 5) / roll_mean(h - l, 30), 0, 3))
    X = np.column_stack(f)
    regime = np.where((slope > 0.5) & (c > e20), 1, np.where((slope < -0.5) & (c < e20), -1, 0))
    return X, atr, regime

# ======================= INDEKS (triple-barrier label) =======================
INDEX, INDEX_LOCK = {}, defaultdict(threading.Lock)

def build_index(arr, tf):
    X, atr, reg = compute(arr)
    c, h, l = arr[:, 4], arr[:, 2], arr[:, 3]
    n, H = len(c), HORIZON
    m = n - H
    if m <= WARM + 200:
        return None
    wh, wl = swv(h[1:], H)[:m], swv(l[1:], H)[:m]
    up, dn = c[:m] + K * atr[:m], c[:m] - K * atr[:m]
    uh, dh = wh >= up[:, None], wl <= dn[:, None]
    big = H + 1
    uf = np.where(uh.any(1), uh.argmax(1), big)
    df = np.where(dh.any(1), dh.argmax(1), big)
    out = np.where(uf < df, 1, np.where(df < H, -1, 0))       # bir shamda ikkalasi -> zarar (konservativ)
    r = np.where(out == 1, 1.0, np.where(out == -1, -1.0,
                 np.clip((c[H:H + m] - c[:m]) / (K * atr[:m]), -1, 1)))
    step = TF_SEC(tf) * 1000
    bad = np.r_[0, (np.diff(arr[:, 0]) != step).astype(int)]
    cs = np.cumsum(bad)
    idx = np.arange(m)
    ok = ~np.isnan(X[:m]).any(1) & (atr[:m] > 0) & ((cs[idx + H] - cs[np.maximum(idx - WARM, 0)]) == 0)
    sel = np.where(ok)[0]
    if len(sel) < 500:
        return None
    Xs = X[sel]
    mu, sd = Xs.mean(0), Xs.std(0)
    sd[sd < 1e-9] = 1.0
    return dict(Z=(Xs - mu) / sd, mu=mu, sd=sd, out=out[sel], r=r[sel], reg=reg[sel],
                pos=sel, built_t=float(arr[-1, 0]), size=len(sel))

def get_index(sym, tf, arr):
    key = (sym, tf)
    with INDEX_LOCK[key]:
        ix = INDEX.get(key)
        step = TF_SEC(tf) * 1000
        if ix is None or (arr[-1, 0] - ix["built_t"]) / step >= REINDEX:
            new = build_index(arr, tf)
            if new is not None:
                INDEX[key] = ix = new
                log.info("Indeks: %s %s | %d nuqta", sym, tf, new["size"])
        return ix

# ======================= ANALOG QIDIRISH + STATISTIKA =======================
def empty():
    return dict(n=0, up=0, dn=0, dec=0, signal=0, wr=0.0, lb=0.0, base=0.0, exp=0.0,
                avg_sim=0.0, reason="", side=0,
                e_n=0, e_win=0, e_sim=0.0, e_r=0.0)

def elite_set(ix, cand, z):
    """Eng kuchli 1..ELITE_MAX analog: kosinus >= ELITE_SIM va vektor uzunligi mos.
    Qo'shni shamlar embargo bilan bitta hodisaga aylantiriladi."""
    Zc = ix["Z"][cand]
    nz = np.linalg.norm(Zc, axis=1)
    nq = float(np.linalg.norm(z))
    if nq < 1e-9:
        return np.array([], dtype=int), np.array([])
    cos = (Zc @ z) / np.maximum(nz * nq, 1e-9)
    ratio = np.minimum(nz, nq) / np.maximum(np.maximum(nz, nq), 1e-9)
    ok = np.where((cos >= ELITE_SIM) & (ratio >= ELITE_NORM))[0]
    ok = ok[np.argsort(-cos[ok])]
    taken, ch = [], []
    for j in ok:
        p = int(ix["pos"][cand[j]])
        if any(abs(p - q) <= GAP for q in taken):
            continue
        taken.append(p)
        ch.append(int(j))
        if len(ch) >= ELITE_MAX:
            break
    ch = np.array(ch, dtype=int)
    return ch, (cos[ch] if len(ch) else np.array([]))

def evaluate(ix, x, regime, atr_pct):
    res = empty()
    z = (x - ix["mu"]) / ix["sd"]
    cand = np.where(ix["reg"] == regime)[0]          # faqat bir xil rejim
    if len(cand) < MIN_EFF * 3:
        res["reason"] = "rejim uchun tarix kam"
        return res

    # ---- ELITE qatlam ----
    ech, ecos = elite_set(ix, cand, z)
    erows = cand[ech] if len(ech) else np.array([], dtype=int)
    en = len(erows)
    eout = ix["out"][erows] if en else np.array([])
    er = ix["r"][erows] if en else np.array([])
    res["e_n"] = en
    res["e_sim"] = float(ecos.mean()) if en else 0.0

    # ---- Kengar qatlam ----
    d = np.sqrt(((ix["Z"][cand] - z) ** 2).sum(1))
    order = np.argsort(d)[:POOL]
    taken, chosen = [], []
    for j in order:                                    # declustering (embargo)
        p = int(ix["pos"][cand[j]])
        if any(abs(p - q) <= GAP for q in taken):
            continue
        taken.append(p)
        chosen.append(j)
        if len(chosen) >= TARGET_N:
            break
    if not chosen:
        res["reason"] = "analog topilmadi"
        return res
    chosen = np.array(chosen, dtype=int)
    sim = np.exp(-d[chosen] / math.sqrt(len(z)))
    keep = sim >= MIN_SIM
    rows, sim = cand[chosen[keep]], sim[keep]
    n = len(rows)
    res["n"], res["avg_sim"] = n, float(sim.mean()) if n else 0.0
    if n < MIN_EFF:
        res["reason"] = f"effektiv analog kam ({n}<{MIN_EFF})"
        return res
    out, r = ix["out"][rows], ix["r"][rows]
    up, dn = int((out == 1).sum()), int((out == -1).sum())
    dec = up + dn
    reg_out = ix["out"][cand]
    bdec = int((reg_out != 0).sum())
    base_up = float((reg_out == 1).sum() / bdec) if bdec else 0.5
    cost_r = COST_PCT / (K * atr_pct) if atr_pct > 0 else 9.0
    best = None
    for side in (1, -1):
        wins = up if side == 1 else dn
        lb, _ = wilson(wins, dec)
        base = base_up if side == 1 else 1 - base_up
        e = side * float(r.mean()) - cost_r
        cur = (e, side, wins / dec if dec else 0.0, lb, base)
        if best is None or e > best[0]:
            best = cur
    e, side, wr, lb, base = best
    ew = int((eout == side).sum()) if en else 0
    e_r = float(side * er.mean() - cost_r) if en else 0.0
    res.update(up=up, dn=dn, dec=dec, wr=wr, lb=lb, base=base, exp=e, side=side,
               e_win=ew, e_r=e_r)
    thr = max(MIN_WIN_LB, base + EDGE_OVER_BASE)
    if dec < MIN_DEC:
        res["reason"] = f"aniq natija kam ({dec}<{MIN_DEC})"
    elif lb < thr:
        res["reason"] = f"win-rate CI pastki {lb:.2f} < {thr:.2f}"
    elif e < MIN_EXP:
        res["reason"] = f"xarajatdan keyin kutilma {e:+.2f}R < {MIN_EXP}R"
    elif en < ELITE_MIN:
        res["reason"] = f"elite analog kam ({en}<{ELITE_MIN}, sim>={ELITE_SIM})"
    elif ew / en < ELITE_AGREE:
        res["reason"] = f"elite mos emas ({ew}/{en} < {ELITE_AGREE:.0%})"
    elif e_r <= 0:
        res["reason"] = f"elite kutilma {e_r:+.2f}R <= 0"
    else:
        res["signal"] = side
    return res

def analyze_tf(sym, tf):
    arr = STORE.get(sym, tf)
    step = TF_SEC(tf) * 1000
    if arr is None or len(arr) < WARM + HORIZON + 300:
        return None
    if now_ms() - arr[-1, 0] > 3 * step:                       # eskirgan ma'lumot
        log.warning("Ma'lumot eskirgan: %s %s", sym, tf)
        return None
    if np.any(np.diff(arr[-130:, 0]) != step):                 # gap
        return None
    ix = get_index(sym, tf, arr)
    if ix is None:
        return None
    X, atr, reg = compute(arr[-400:])
    x = X[-1]
    if np.isnan(x).any() or not atr[-1] > 0:
        return None
    price = float(arr[-1, 4])
    atr_pct = float(atr[-1] / price * 100)
    ev = evaluate(ix, x, int(reg[-1]), atr_pct)
    ev.update(price=price, atr=float(atr[-1]), atr_pct=atr_pct, regime=int(reg[-1]), t=int(arr[-1, 0]))
    return ev

# ======================= DERIVATIVLAR =======================
DERIV = {}

def derivatives(sym):
    c = DERIV.get(sym)
    if c and time.time() - c[0] < 60:
        return c[1]
    d = None
    try:
        p = http_get(FAPI + "/fapi/v1/premiumIndex", {"symbol": sym}, retries=2)
        oi = http_get(FAPI + "/futures/data/openInterestHist",
                      {"symbol": sym, "period": "5m", "limit": 13}, retries=2)
        ls = http_get(FAPI + "/futures/data/globalLongShortAccountRatio",
                      {"symbol": sym, "period": "5m", "limit": 1}, retries=2)
        d = dict(funding=float(p["lastFundingRate"]) * 100,
                 oi_chg=(float(oi[-1]["sumOpenInterestValue"]) / float(oi[0]["sumOpenInterestValue"]) - 1) * 100,
                 ls=float(ls[0]["longShortRatio"]))
    except Exception as e:
        log.warning("Derivativ ma'lumot yo'q %s: %s", sym, e)
    DERIV[sym] = (time.time(), d)
    return d

# ======================= YANGILIK / EVENT =======================
RX_MACRO = re.compile(r"\b(?:fomc|fed|federal reserve|cpi|pce|inflation|rate (?:cut|hike)s?|powell|tariffs?|wars?|"
                      r"strikes?|missiles?|sanctions?|hack(?:s|ed)?|exploits?|lawsuits?|sues?|etfs?|liquidations?)\b", re.I)
RX_BULL = re.compile(r"\b(?:approv(?:es|ed|al)|inflows?|records?|surg(?:es|ed|ing)|rall(?:y|ies|ied)|soars?|"
                     r"adopt(?:s|ion|ed)|boosts?|gains?)\b", re.I)
RX_BEAR = re.compile(r"\b(?:declin(?:es|ed|e)|bans?|hack(?:s|ed)?|exploits?|lawsuits?|outflows?|plunge[sd]?|"
                     r"crash(?:es|ed)?|drops?|falls?|selloff|strikes?|liquidations?)\b", re.I)
NAMES = {"btc": "bitcoin|btc", "eth": "ethereum|ether|eth", "sol": "solana|sol",
         "bnb": "bnb|binance", "xrp": "xrp|ripple"}
NEWS, NEWS_TS = [], 0.0

def parse_ts(s):
    try:
        dt = parsedate_to_datetime(s)
    except Exception:
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()

def fetch_news():
    global NEWS, NEWS_TS
    items = {}
    for url in FEEDS:
        try:
            r = SESSION.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            for it in ET.fromstring(r.content).iter():
                if it.tag.lower().rsplit("}", 1)[-1] not in ("item", "entry"):
                    continue
                title = pub = ""
                for ch in it:
                    t = ch.tag.lower().rsplit("}", 1)[-1]
                    if t == "title":
                        title = (ch.text or "").strip()
                    elif t in ("pubdate", "published", "updated"):
                        pub = (ch.text or "").strip()
                ts = parse_ts(pub)
                if title and ts:
                    items[title.lower()] = (ts, title)
        except Exception as e:
            log.warning("RSS xatosi %s: %s", url, e)
    if items:
        NEWS = sorted(items.values(), reverse=True)[:200]
    NEWS_TS = time.time()

def news_state(sym):
    base = sym.replace("USDT", "").lower()
    rx = re.compile(r"\b(?:" + NAMES.get(base, base) + r")\b", re.I)
    now = time.time()
    items, risk, sent = [], [], 0
    for ts, title in NEWS:
        age = (now - ts) / 60
        if age > 360:
            continue
        macro, mine = bool(RX_MACRO.search(title)), bool(rx.search(title))
        if not (macro or mine):
            continue
        s = len(RX_BULL.findall(title)) - len(RX_BEAR.findall(title))
        sent += max(-2, min(2, s))
        items.append((age, title))
        if macro and age <= RISK_WINDOW_MIN:
            risk.append(title)
    return dict(sent=sent, risk=risk, items=items[:2])

def in_blackout():
    now = time.time()
    for t in BLACKOUTS:
        try:
            ts = datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp()
        except Exception:
            continue
        if abs(now - ts) <= BLACKOUT_MIN * 60:
            return t
    return None

# ======================= BAZA (validatsiya) =======================
DB = sqlite3.connect(DB_PATH, check_same_thread=False)
DBL = threading.Lock()
with DBL:
    DB.execute("""CREATE TABLE IF NOT EXISTS signals(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, symbol TEXT, tf TEXT, side INTEGER,
        entry REAL, sl REAL, tp REAL, atr REAL, horizon INTEGER, win_lb REAL, exp_r REAL, n_eff INTEGER,
        status TEXT DEFAULT 'OPEN', r REAL, closed_ts INTEGER)""")
    DB.commit()

def save_signal(sym, v):
    p = v["primary"]
    with DBL:
        DB.execute("INSERT INTO signals(ts,symbol,tf,side,entry,sl,tp,atr,horizon,win_lb,exp_r,n_eff) "
                   "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                   (p["t"], sym, PRIMARY, v["side"], v["entry"], v["sl"], v["tp"], p["atr"],
                    HORIZON, p["lb"], p["exp"], p["n"]))
        DB.commit()

def recently_signaled(sym):
    since = (time.time() - COOLDOWN * 60) * 1000
    with DBL:
        return DB.execute("SELECT 1 FROM signals WHERE symbol=? AND ts>=?", (sym, since)).fetchone() is not None

def resolve_signals():
    with DBL:
        rows = DB.execute("SELECT id,ts,symbol,tf,side,entry,sl,tp,horizon FROM signals WHERE status='OPEN'").fetchall()
    for id_, ts, sym, tf, side, entry, sl, tp, H in rows:
        arr = STORE.get(sym, tf)
        if arr is None:
            continue
        sub = arr[arr[:, 0] > ts][:H]
        r = None
        for row in sub:
            hi, lo = row[2], row[3]
            hit_sl = lo <= sl if side == 1 else hi >= sl
            hit_tp = hi >= tp if side == 1 else lo <= tp
            if hit_sl:                      # konservativ: bir shamda ikkalasi bo'lsa SL
                r = -1.0
                break
            if hit_tp:
                r = 1.0
                break
        if r is None and len(sub) >= H:
            r = max(-1.0, min(1.0, (sub[-1, 4] - entry) * side / abs(tp - entry)))
        if r is None:
            continue
        cost = COST_PCT / (abs(tp - entry) / entry * 100)
        with DBL:
            DB.execute("UPDATE signals SET status='CLOSED', r=?, closed_ts=? WHERE id=?",
                       (r - cost, now_ms(), id_))
            DB.commit()
        log.info("Signal #%d yopildi: %+.2fR (%s)", id_, r - cost, sym)

def perf(days=30):
    since = (time.time() - days * 86400) * 1000
    with DBL:
        rs = [x[0] for x in DB.execute("SELECT r FROM signals WHERE status='CLOSED' AND ts>=? ORDER BY ts", (since,))]
        op = DB.execute("SELECT COUNT(*) FROM signals WHERE status='OPEN'").fetchone()[0]
    if not rs:
        return dict(n=0, open=op)
    a = np.array(rs)
    eq = np.cumsum(a)
    return dict(n=len(a), open=op, win=float((a > 0).mean()), avg=float(a.mean()),
                total=float(a.sum()), max_dd=float((np.maximum.accumulate(eq) - eq).max()))

# ======================= SIGNAL GATE =======================
REG = {1: "trend yuqori", -1: "trend past", 0: "range"}

def decide(sym, per_tf):
    v = dict(status="WATCH", side=0, reasons=[], per_tf=per_tf)
    p = per_tf.get(PRIMARY)
    if not p:
        v["reasons"].append("ma'lumot yetarli/yangi emas")
        return v
    v["primary"] = p
    side = p["signal"]
    if side == 0:
        v["reasons"].append(p.get("reason") or "asosiy TFda edge yo'q")
        return v
    opp = [tf for tf, e in per_tf.items() if e and e["signal"] == -side]
    if opp:
        v["reasons"].append("qarama-qarshi TF: " + ",".join(opp))
        return v
    agree = sum(1 for e in per_tf.values() if e and e["signal"] == side)
    if agree < MIN_TF:
        v["reasons"].append(f"TF kelishuvi {agree}/{MIN_TF}")
        return v
    bo = in_blackout()
    if bo:
        v["reasons"].append(f"katta event oynasi ({bo})")
        return v
    d = derivatives(sym)
    v["deriv"] = d
    if d:
        if side == 1 and d["funding"] > FUND_EXT:
            v["reasons"].append(f"funding juda baland {d['funding']:.3f}% (long gavjum)")
            return v
        if side == -1 and d["funding"] < -FUND_EXT:
            v["reasons"].append(f"funding juda past {d['funding']:.3f}% (short gavjum)")
            return v
    nw = news_state(sym) if NEWS_ON else dict(sent=0, risk=[], items=[])
    v["news"] = nw
    if nw["risk"]:
        v["reasons"].append("makro-xavf yangiligi: " + nw["risk"][0][:70])
        return v
    if side * nw["sent"] <= -3:
        v["reasons"].append("yangilik kayfiyati qarama-qarshi")
        return v
    if recently_signaled(sym):
        v["reasons"].append("cooldown")
        return v
    dist = K * p["atr"]
    entry = p["price"]
    qty = min(EQUITY * RISK / 100 / dist, EQUITY * MAXLEV / entry)
    v.update(status="SIGNAL", side=side, entry=entry, sl=entry - side * dist, tp=entry + side * dist,
             qty=qty, notional=qty * entry)
    return v

# ======================= XABAR / GRAFIK / TELEGRAM =======================
def arrow(e):
    return "?" if e is None else "↑" if e["signal"] == 1 else "↓" if e["signal"] == -1 else "–"

def build_msg(sym, v):
    p, side, d, nw = v["primary"], v["side"], v.get("deriv"), v.get("news") or {}
    base = sym.replace("USDT", "")
    L = [f"{'🟢 LONG' if side > 0 else '🔴 SHORT'} | {sym} | {PRIMARY}",
         f"Narx: {fmt(p['price'])}",
         f"Kirish ~{fmt(v['entry'])} | SL {fmt(v['sl'])} | TP {fmt(v['tp'])}",
         f"Gorizont: {HORIZON} sham (~{HORIZON * TF_SEC(PRIMARY) // 60} daq), R:R = 1:1",
         f"Win-rate: {p['wr'] * 100:.0f}% (95% CI pastki {p['lb'] * 100:.0f}%, baza {p['base'] * 100:.0f}%)",
         f"Kutilma: {p['exp']:+.2f}R (xarajatdan keyin) | N_eff={p['n']} | sim={p['avg_sim']:.2f}",
         f"⭐ Elite: {p['e_win']}/{p['e_n']} mos | sim={p['e_sim']:.3f} (>={ELITE_SIM}) | {p['e_r']:+.2f}R",
         "TF: " + "  ".join(f"{tf}{arrow(e)}" for tf, e in v["per_tf"].items()) + f" | Rejim: {REG[p['regime']]}",
         f"Hajm: {v['qty']:.4g} {base} (~{v['notional']:,.0f} USDT, risk {RISK}%)"]
    L.append(f"Funding {d['funding']:+.3f}% | OI 1h {d['oi_chg']:+.1f}% | L/S {d['ls']:.2f}" if d
             else "Derivativ ma'lumot: mavjud emas (ishonch pastroq)")
    for age, t in nw.get("items", []):
        L.append(f"📰 {int(age)} daq oldin: {t[:90]}")
    L.append("⚠️ Kafolat emas. Bot order ochmaydi.")
    return "\n".join(L)

def make_chart(sym, tf, v, path):
    arr = STORE.get(sym, tf)[-80:]
    x = mdates.date2num([datetime.fromtimestamp(t / 1000, timezone.utc) for t in arr[:, 0]])
    w = TF_SEC(tf) / 86400 * 0.7
    fig, (ax, av) = plt.subplots(2, 1, sharex=True, figsize=(11, 7), gridspec_kw={"height_ratios": [4, 1]})
    for xi, o, h, l, c, vol in zip(x, arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5]):
        col = "#16a34a" if c >= o else "#dc2626"
        ax.vlines(xi, l, h, color=col, lw=1)
        ax.bar(xi, max(abs(c - o), (h - l) * 0.01), bottom=min(o, c), width=w, color=col)
        av.bar(xi, vol, width=w, color=col)
    for lvl, col, name in ((v["entry"], "#2563eb", "Kirish"), (v["sl"], "#dc2626", "SL"), (v["tp"], "#16a34a", "TP")):
        ax.axhline(lvl, color=col, ls="--", lw=1)
        ax.text(x[0], lvl, f" {name} {fmt(lvl)}", color=col, va="bottom", fontsize=8)
    end = x[-1] + HORIZON * TF_SEC(tf) / 86400
    ax.axvspan(x[-1], end, color="#94a3b8", alpha=0.15)
    ax.set_xlim(x[0] - w, end)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M"))
    fig.autofmt_xdate()
    ax.set_title(f"{sym} {tf} | {'LONG' if v['side'] > 0 else 'SHORT'} | UTC")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path

def tg(method, data=None, files=None):
    if not TG_ON:
        return False
    try:
        r = SESSION.post(f"https://api.telegram.org/bot{TG_TOKEN}/{method}", data=data, files=files, timeout=TIMEOUT)
        if r.status_code != 200:
            log.warning("TG %s: %s", r.status_code, r.text[:200])
            return False
        return True
    except Exception as e:
        log.warning("TG xato: %s", e)
        return False

def tg_text(t):
    ok = True
    for i in range(0, len(t), 3900):
        ok = tg("sendMessage", {"chat_id": TG_CHAT, "text": t[i:i + 3900],
                                "disable_web_page_preview": "true"}) and ok
    return ok

def tg_photo(path, cap):
    with open(path, "rb") as f:
        return tg("sendPhoto", {"chat_id": TG_CHAT, "caption": cap[:1000]}, {"photo": f})

# ======================= ASOSIY TAHLIL =======================
def analyze_symbol(sym):
    lock = SYM_LOCK[sym]
    if not lock.acquire(blocking=False):
        return
    try:
        time.sleep(3)                              # yuqori TF shamlari ham yozilib olsin
        per_tf = {}
        for tf in TFS:
            try:
                per_tf[tf] = analyze_tf(sym, tf)
            except Exception:
                log.exception("analyze_tf %s %s", sym, tf)
                per_tf[tf] = None
        v = decide(sym, per_tf)
        short = " ".join(f"{tf}:{arrow(e)}n{e['n']}/lb{e['lb']:.2f}/e{e['e_n']}" if e else f"{tf}:?"
                         for tf, e in per_tf.items())
        log.info("%s %s | %s | %s", sym, v["status"], short, "; ".join(v["reasons"]))
        if v["status"] != "SIGNAL":
            return
        save_signal(sym, v)
        tg_text(build_msg(sym, v))
        if CHART_ON and TG_ON:
            path = f"/tmp/{sym}_{int(time.time())}.png"
            try:
                make_chart(sym, PRIMARY, v, path)
                tg_photo(path, f"{sym} | {'LONG' if v['side'] > 0 else 'SHORT'} | {PRIMARY}")
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
    except Exception:
        log.exception("analyze_symbol %s", sym)
    finally:
        lock.release()

# ======================= WEBSOCKET =======================
def on_message(ws, msg):
    global LAST_WS
    LAST_WS = time.time()
    try:
        d = json.loads(msg)
        k = d.get("data", d).get("k")
        if not k or not k.get("x"):                # faqat yopilgan sham
            return
        sym, tf = k["s"].upper(), k["i"]
        row = np.array([[k["t"], k["o"], k["h"], k["l"], k["c"], k["v"]]], dtype=float)
        if STORE.merge(sym, tf, row):
            EX.submit(backfill, sym, tf)
        if tf == PRIMARY:
            EX.submit(analyze_symbol, sym)
    except Exception:
        log.exception("ws xabar")

def ws_worker():
    global WS_APP
    streams = "/".join(f"{s.lower()}@kline_{tf}" for s in SYMBOLS for tf in TFS)
    url = f"{WS_URL.rstrip('/')}?streams={streams}"
    delay = 5
    while not STOP:
        try:
            WS_APP = websocket.WebSocketApp(
                url, on_message=on_message,
                on_open=lambda ws: log.info("WebSocket ulandi"),
                on_error=lambda ws, e: log.warning("WS xato: %s", e),
                on_close=lambda ws, c, m: log.warning("WS yopildi: %s %s", c, m))
            WS_APP.run_forever(ping_interval=60, ping_timeout=20)
        except Exception as e:
            log.warning("WS ishida xato: %s", e)
        if STOP:
            break
        for tf in TFS:                              # uzilish davridagi teshiklarni to'ldirish
            for s in SYMBOLS:
                backfill(s, tf)
        sleep_i(delay)
        delay = min(delay * 2, 60)

# ======================= HOUSEKEEPING / HEALTH =======================
def housekeeping():
    last_day = datetime.now(timezone.utc).date()
    while not STOP:
        try:
            resolve_signals()
            if time.time() - LAST_WS > 180 and WS_APP is not None:
                log.warning("WS jim (>180s), qayta ulanadi")
                WS_APP.close()
            if NEWS_ON and time.time() - NEWS_TS > NEWS_SEC:
                fetch_news()
            today = datetime.now(timezone.utc).date()
            if today != last_day:
                last_day = today
                p7, p30 = perf(7), perf(30)
                def line(n, p):
                    return (f"{n}: yopilgan {p['n']} | win {p['win'] * 100:.0f}% | o'rtacha {p['avg']:+.2f}R | "
                            f"jami {p['total']:+.1f}R | maxDD {p['max_dd']:.1f}R") if p["n"] else f"{n}: signal yo'q"
                tg_text(f"📈 Kunlik hisobot (UTC)\n{line('7 kun', p7)}\n{line('30 kun', p30)}\nOchiq: {p30['open']}")
        except Exception:
            log.exception("housekeeping")
        sleep_i(30)

class Health(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        body = json.dumps({"status": "ok", "version": VERSION, "uptime": int(time.time() - START),
                           "ws_age_sec": int(time.time() - LAST_WS),
                           "candles": {f"{s}_{tf}": (0 if STORE.get(s, tf) is None else len(STORE.get(s, tf)))
                                       for s in SYMBOLS for tf in TFS},
                           "indexes": len(INDEX), "perf_30d": perf(30)}, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

def load_history():
    for s in SYMBOLS:
        for tf in TFS:
            if STOP:
                return
            try:
                a = fetch_klines(s, tf, now_ms() - HIST_DAYS.get(tf, 365) * 86400000)
                STORE.merge(s, tf, a)
                gaps = int((np.diff(a[:, 0]) != TF_SEC(tf) * 1000).sum()) if len(a) > 1 else 0
                log.info("Yuklandi %s %s | %d sham | gap=%d", s, tf, len(a), gaps)
                arr = STORE.get(s, tf)
                if arr is not None and len(arr) > WARM + HORIZON + 300:
                    get_index(s, tf, arr)
            except Exception:
                log.exception("Tarix yuklash %s %s", s, tf)

def shutdown(*_):
    global STOP
    STOP = True
    log.info("To'xtatilmoqda...")

def main():
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    log.info("%s v%s | %s | %s | primary=%s | telegram=%s | elite sim>=%.2f max=%d min=%d",
             APP, VERSION, ",".join(SYMBOLS), ",".join(TFS), PRIMARY, TG_ON,
             ELITE_SIM, ELITE_MAX, ELITE_MIN)
    if NEWS_ON:
        fetch_news()
    load_history()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Health)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    threading.Thread(target=ws_worker, daemon=True).start()
    threading.Thread(target=housekeeping, daemon=True).start()
    tg_text(f"🚀 {APP} v{VERSION}\nAktivlar: {', '.join(SYMBOLS)}\nTF: {', '.join(TFS)} (asosiy {PRIMARY})\n"
            f"Elite filtr: sim>={ELITE_SIM}, 1..{ELITE_MAX} analog.\n"
            f"Signal faqat gate'dan o'tganda keladi. Bot order ochmaydi.")
    while not STOP:
        time.sleep(1)
    srv.shutdown()

if __name__ == "__main__":
    main()