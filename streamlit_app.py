"""
MAVRI — Day Trading & Momentum Screener.
Upgraded UI, optimized for 4000+ stocks, Intraday RVOL precision.
"""

import io
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timezone, timedelta

st.set_page_config(page_title="MAVRI Pro", layout="wide",
                   initial_sidebar_state="collapsed", page_icon="⚡")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;600;700&display=swap');
:root {
    --bg: #0A0E17;        /* Deep Space Black */
    --pan: #131824;       /* Panel Dark */
    --rz: #1E2536;        /* Raised Element */
    --ln: #2A3245;        /* Lines/Borders */
    --tx: #F8FAFC;        /* Primary Text */
    --mu: #94A3B8;        /* Muted Text */
    --dm: #64748B;        /* Dim/Secondary */
    --gd: #F59E0B;        /* Amber/Gold */
    --lg: #10B981;        /* Cyber Green */
    --sh: #EF4444;        /* Neon Red */
    --bl: #3B82F6;        /* Electric Blue */
}
html, body, [class*="css"] { font-family: 'Heebo', sans-serif; }
.stApp { background: var(--bg); color: var(--tx); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 4rem; max-width: 1400px; }
hr { border-color: var(--ln); }

/* ---- Top Bar ---- */
.bar { display: flex; align-items: center; justify-content: space-between; gap: 1rem;
       background: rgba(19, 24, 36, 0.7); backdrop-filter: blur(10px);
       border: 1px solid var(--ln); border-radius: 12px; padding: 1rem 1.5rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
.brand { display: flex; align-items: baseline; gap: 0.8rem; }
.brand h1 { font-size: 2rem; font-weight: 700; margin: 0; color: var(--tx); letter-spacing: 0.1em; 
            background: linear-gradient(90deg, #3B82F6, #10B981); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.brand em { font-style: normal; font-size: 0.9rem; color: var(--mu); letter-spacing: 0.05em; }
.status { display: flex; gap: 2rem; align-items: center; direction: rtl; }
.status i { font-style: normal; font-size: 0.75rem; color: var(--dm); display: block; text-transform: uppercase; }
.status b { font-size: 1.1rem; font-weight: 600; color: var(--tx); font-variant-numeric: tabular-nums; }
.led { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-left: 0.5rem; }
.led.on { background: var(--lg); box-shadow: 0 0 10px rgba(16, 185, 129, 0.8); }
.led.off { background: var(--sh); box-shadow: 0 0 10px rgba(239, 68, 68, 0.8); }

/* ---- Counters/KPI ---- */
.kpi { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
.kpi div { flex: 1; min-width: 120px; padding: 1rem; background: var(--pan); border: 1px solid var(--ln); 
           border-radius: 10px; direction: rtl; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
.kpi b { font-size: 1.5rem; font-weight: 600; display: block; line-height: 1.2; color: var(--tx); }
.kpi span { font-size: 0.75rem; color: var(--dm); font-weight: 500; }
.kpi div.hi b { color: var(--lg); }

/* ---- Cards ---- */
.tc { background: rgba(19, 24, 36, 0.6); backdrop-filter: blur(8px); border: 1px solid var(--ln); 
      border-radius: 12px; padding: 1.2rem; direction: rtl; height: 100%; transition: all 0.3s ease;
      box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
.tc:hover { transform: translateY(-4px); box-shadow: 0 12px 30px rgba(0,0,0,0.4); border-color: var(--bl); }
.tc.top { border-color: rgba(16, 185, 129, 0.5); box-shadow: 0 0 20px rgba(16, 185, 129, 0.1); }
.tc-h { display: flex; align-items: baseline; justify-content: space-between; padding-bottom: 0.5rem; border-bottom: 1px solid var(--ln); margin-bottom: 0.8rem; }
.tc-s { font-size: 1.8rem; font-weight: 700; color: var(--tx); direction: ltr; letter-spacing: 0.05em; }
.tc-d { text-align: left; direction: ltr; }
.tc-d b { font-size: 1.4rem; font-weight: 600; color: var(--gd); }
.tc-d span { display: block; font-size: 0.7rem; color: var(--mu); direction: rtl; }
.spark svg { width: 100%; height: 35px; display: block; opacity: 0.9; margin: 0.5rem 0; }

/* ---- Badges & Chips ---- */
.chips { display: flex; gap: 0.4rem; flex-wrap: wrap; margin: 0.5rem 0; }
.chip { font-size: 0.7rem; padding: 0.2rem 0.5rem; background: var(--rz); border: 1px solid var(--ln); border-radius: 6px; color: var(--mu); font-weight: 500; }
.chip.g { border-color: rgba(245, 158, 11, 0.4); color: var(--gd); background: rgba(245, 158, 11, 0.1); }
.badge { display: inline-block; font-size: 0.65rem; padding: 0.2rem 0.6rem; border-radius: 4px; background: rgba(16, 185, 129, 0.15); color: var(--lg); margin-right: 0.4rem; font-weight: 600; }

/* ---- Ladder ---- */
.lad { position: relative; height: 180px; margin: 1rem 0; background: rgba(0,0,0,0.1); border-radius: 8px; padding: 10px 0; }
.lad:before { content: ''; position: absolute; right: 80px; top: 10px; bottom: 10px; width: 2px; 
              background: linear-gradient(180deg, rgba(16, 185, 129, 0.5), rgba(239, 68, 68, 0.5)); border-radius: 2px; }
.lv { position: absolute; right: 0; left: 0; height: 0; display: flex; align-items: center; gap: 0.6rem; }
.lv .tg { width: 75px; text-align: left; font-size: 0.7rem; color: var(--dm); font-weight: 500; }
.lv .dt { width: 8px; height: 8px; border-radius: 50%; background: var(--dm); margin-right: -4px; flex: none; z-index: 2; border: 2px solid var(--pan); }
.lv .px { font-size: 0.95rem; font-weight: 600; color: var(--mu); font-variant-numeric: tabular-nums; direction: ltr; }
.lv.tp .dt { background: var(--lg); } .lv.tp .px { color: var(--lg); }
.lv.st .dt { background: var(--sh); } .lv.st .px, .lv.st .tg { color: var(--sh); }
.lv.en .dt { width: 12px; height: 12px; background: var(--bl); margin-right: -6px; box-shadow: 0 0 8px rgba(59, 130, 246, 0.6); border: none; }
.lv.en .px { font-size: 1.2rem; font-weight: 700; color: var(--tx); }
.lv.en .tg { color: var(--bl); font-weight: 700; }

/* ---- Bottom Details ---- */
.tl { display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--dm); border-top: 1px solid var(--ln); padding-top: 0.6rem; margin-top: 0.5rem; }
.tl b { color: var(--tx); font-weight: 600; }
.tw { font-size: 0.8rem; line-height: 1.6; color: var(--mu); margin-top: 0.8rem; padding: 0.8rem; background: var(--rz); border-radius: 8px; }

/* ---- Inputs & Expanders Overrides ---- */
div[data-testid="stExpander"] { border: 1px solid var(--ln); border-radius: 10px; background: var(--pan); overflow: hidden; }
div[data-testid="stExpander"] summary { background: rgba(255,255,255,0.02); }
h3, label { color: var(--mu) !important; }
.stButton button { border-radius: 8px; font-weight: 600; transition: all 0.2s; }
.stButton button:hover { transform: scale(1.02); }

/* ---- Grid & Mapping ---- */
.sg { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 0.5rem; direction: rtl; margin: 1rem 0; }
.sg div { background: var(--pan); padding: 0.8rem; border: 1px solid var(--ln); border-radius: 8px; transition: 0.2s; }
.sg div:hover { border-color: var(--mu); }
.sg b { display: block; font-size: 1.2rem; font-weight: 600; color: var(--tx); }
.sg span { font-size: 0.7rem; color: var(--dm); text-transform: uppercase; letter-spacing: 0.05em; }
.sg div.gd b { color: var(--gd); } .sg div.up b { color: var(--lg); } .sg div.dn b { color: var(--sh); }

.rmap { position: relative; height: 60px; margin: 1rem 0; direction: ltr; background: var(--pan); border-radius: 8px; padding: 0 10px; border: 1px solid var(--ln); }
.rmap .base { position: absolute; top: 28px; left: 5%; right: 5%; height: 6px; background: var(--rz); border-radius: 4px; }
.rmap .risk { position: absolute; top: 28px; height: 6px; background: linear-gradient(90deg, rgba(239, 68, 68, 0.2), rgba(239, 68, 68, 0.8)); border-radius: 4px 0 0 4px; }
.rmap .rew { position: absolute; top: 28px; height: 6px; background: linear-gradient(90deg, rgba(16, 185, 129, 0.8), rgba(16, 185, 129, 0.2)); border-radius: 0 4px 4px 0; }
.rmap .mk { position: absolute; top: 18px; width: 2px; height: 26px; background: var(--dm); z-index: 2; }
.rmap .mk.e { background: var(--bl); height: 32px; top: 15px; box-shadow: 0 0 5px var(--bl); }
.rmap .lb { position: absolute; top: 48px; font-size: 0.65rem; color: var(--dm); transform: translateX(-50%); white-space: nowrap; font-weight: 500; }
.rmap .lb.e { color: var(--bl); font-weight: 700; top: 2px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"}

# Updated presets for Day Trading/Momentum
PRESETS = {
    "מסחר יומי מומנטום": dict(dry=1.10, bb="כבוי", bbmode="בונוס", price=5.0, dv=15.0, sh=1.0, 
                             atr=0.5, atrp=4.0, rise=10, age=(1, 15), retr=(15, 60), rr=1.5, off=5),
    "התבנית שלי (סווינג)": dict(dry=1.30, bb="רצועה תחתונה", bbmode="בונוס", price=10.0, dv=15.0,
                             sh=1.0, atr=1.0, atrp=2.0, rise=12, age=(3, 20), retr=(20, 70), rr=1.5, off=12),
    "רחב": dict(dry=1.05, bb="אחת מהשתיים", bbmode="בונוס", price=5.0, dv=8.0, sh=0.5,
                atr=0.4, atrp=1.5, rise=6, age=(2, 35), retr=(12, 88), rr=1.0, off=20)
}

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_full_market():
    out = []
    for url, sym_col, etf_col, test_col in [
        ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", "Symbol", "ETF", "Test Issue"),
        ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", "ACT Symbol", "ETF", "Test Issue"),
    ]:
        try:
            txt = requests.get(url, headers=UA, timeout=25).text
            df = pd.read_csv(io.StringIO(txt), sep="|")
            df = df[df[test_col].astype(str).str.upper() != "Y"]
            if etf_col in df.columns:
                df = df[df[etf_col].astype(str).str.upper() != "Y"]
            syms = [str(x).strip().upper().replace(".", "-") for x in df[sym_col]]
            syms = [x for x in syms if x.isascii() and 1 <= len(x) <= 5 and x.replace("-", "").isalpha()]
            out += syms
        except Exception:
            pass
    return out

@st.cache_data(ttl=86400, show_spinner=False)
def build_universe(limit):
    got, log = [], []
    for name, url, cols in [
        ("S&P500", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", ("Symbol", "Ticker")),
        ("NDX", "https://en.wikipedia.org/wiki/Nasdaq-100", ("Ticker", "Symbol"))]:
        try:
            html = requests.get(url, headers=UA, timeout=25).text
            for tbl in pd.read_html(io.StringIO(html)):
                hit = False
                for col in cols:
                    if col in tbl.columns:
                        s = [str(x).strip().upper().replace(".", "-") for x in tbl[col]]
                        s = [x for x in s if x.isascii() and 1 <= len(x) <= 6 and x.replace("-", "").isalpha()]
                        if len(s) > 50: got += s; hit = True; break
                if hit: break
        except Exception: pass

    full = fetch_full_market()
    got += full
    seen, out = set(), []
    for t in got:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:limit], log

@st.cache_data(ttl=1800, max_entries=30, show_spinner=False)
def fetch(tickers, period="6mo"):
    out = {}
    try:
        raw = yf.download(list(tickers), period=period, interval="1d", group_by="ticker",
                          auto_adjust=False, threads=True, progress=False, timeout=45)
    except Exception:
        return out
    if raw is None or len(raw) == 0:
        return out
    for t in tickers:
        try:
            d = (raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw).dropna()
            if len(d) >= 60:  # Changed from 90 to 60 for 6mo compatibility
                out[t] = d
        except Exception:
            pass
    return out

@st.cache_data(ttl=900, max_entries=40, show_spinner=False)
def fetch_one(t, period, interval):
    for attempt in range(3):
        try:
            d = yf.download(t, period=period, interval=interval, auto_adjust=False, progress=False, timeout=20, threads=False)
            if d is not None and len(d):
                if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
                d = d.dropna()
                if len(d) >= 5: return d
        except Exception: pass
        time.sleep(0.5)
    return None

def prep(d):
    c, o = d["Close"].values.astype(float), d["Open"].values.astype(float)
    h, l = d["High"].values.astype(float), d["Low"].values.astype(float)
    v = d["Volume"].values.astype(float)
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    cs = pd.Series(c)
    mid, sd = cs.rolling(20).mean(), cs.rolling(20).std()
    return dict(c=c, o=o, h=h, l=l, v=v, atr=pd.Series(tr).rolling(14).mean().values,
                bbu=(mid + 2 * sd).values, bbl=(mid - 2 * sd).values, sma200=cs.rolling(200).mean().values)

def bb_hits(A, i, look, side):
    out = []
    for j in range(max(0, i - look + 1), i + 1):
        lo_b, up_b = A["bbl"][j], A["bbu"][j]
        if side in ("lower", "both") and np.isfinite(lo_b) and A["l"][j] <= lo_b and A["c"][j] > lo_b: out.append("תחתונה")
        if side in ("upper", "both") and np.isfinite(up_b) and A["h"][j] >= up_b and A["c"][j] < up_b: out.append("עליונה")
    return sorted(set(out))

def core(A, i, C):
    c, o, h, l, v = A["c"], A["o"], A["h"], A["l"], A["v"]
    if i < 50: return None, "היסטוריה קצרה"
    s = max(0, i - C["win"] + 1)
    cw, ow, lw, vw = c[s:i + 1], o[s:i + 1], l[s:i + 1], v[s:i + 1]
    n = len(cw)
    
    hi_idx = max(0, n - 1 - C["age_lo"])
    lo_idx = max(0, n - 1 - C["age_hi"])
    if hi_idx <= lo_idx: return None, "טווח ימים לא תקין"
    
    window = cw[lo_idx:hi_idx]
    if len(window) == 0: return None, "אין טווח לשיא"
    peak = lo_idx + int(np.argmax(window))
    age = n - 1 - peak
    st_ = max(0, peak - C["leg_max"])
    if peak - st_ < C["leg_min"]: return None, "אין מקום לזינוק"
    
    low_i = st_ + int(np.argmin(cw[st_:peak]))
    leg_bars = peak - low_i
    if leg_bars < C["leg_min"]: return None, "זינוק קצר מדי"
    lo_px, pk_px, px = float(cw[low_i]), float(cw[peak]), float(cw[-1])
    if lo_px <= 0 or pk_px <= lo_px: return None, "אין זינוק"
    
    rise = (pk_px / lo_px - 1) * 100
    if rise < C["rise_min"]: return None, "זינוק קטן מדי"
    retr = (pk_px - px) / (pk_px - lo_px) * 100
    if retr < C["retr_lo"]: return None, "כמעט לא תיקנה"
    if retr > C["retr_hi"]: return None, "החזירה את כל הזינוק"
    
    base = vw[max(0, low_i - 50):low_i]
    base_v = float(base.mean()) if len(base) >= 5 else float(vw[:max(low_i, 1)].mean())
    base_v = max(base_v, 1.0) # Prevent zero div
    
    leg_v, pull_v = vw[low_i:peak + 1], vw[peak:]
    if base_v <= 0 or len(leg_v) == 0 or len(pull_v) == 0 or pull_v.mean() <= 0: return None, "אין נתוני נפח"
    
    spike = float(leg_v.max() / base_v)
    dry = float(leg_v.mean() / pull_v.mean())
    if dry < C["dry_min"]: return None, "הנפח לא התייבש"
    
    tags = bb_hits(A, i, C["bb_look"], C["bb"]) if C["bb"] != "off" else []
    if C["bb_hard"] and C["bb"] != "off" and not tags: return None, "אין נר על בולינג׳ר"
    
    sup = float(lw[peak:].min())
    off_low = (px / sup - 1) * 100 if sup > 0 else 999
    if off_low > C["off_max"]: return None, "רחוקה מדי מהתמיכה"
    
    a = float(A["atr"][i])
    if not np.isfinite(a) or a <= 0: return None, "אין ATR"
    if a < C["atr_abs"]: return None, "ATR נמוך מדי"
    atr_pct = a / px * 100
    if atr_pct < C["atr_pct"]: return None, "תנודתיות נמוכה"
    
    # RVOL optimized for Day Trading (Current bar vs previous 50 days mean)
    base_50 = v[max(0, i - 50):i].mean() if i > 0 else 1.0
    rvol = float(v[i] / base_50) if base_50 > 0 else 0.0
    
    entry = max(px, float(h[i])) + .05 * a
    stop = max(min(sup - .35 * a, entry - 3 * a), entry - 4 * a)
    risk = entry - stop
    if risk <= 0: return None, "סטופ לא תקין"
    
    leg = pk_px - lo_px
    tp1, tp2, tp3 = pk_px, pk_px + .5 * leg, pk_px + leg
    rr = (tp1 - entry) / risk
    if rr < C["rr_min"]: return None, "אין מרווח ל-TP1"
    
    return dict(dry=round(dry, 2), spike=round(spike, 2), rise=round(rise, 1),
                leg_bars=int(leg_bars), retrace=round(retr, 1), price=round(px, 2),
                entry=round(entry, 2), stop=round(stop, 2), tp1=round(tp1, 2),
                tp2=round(tp2, 2), tp3=round(tp3, 2), rr=round(float(rr), 2),
                risk_share=round(float(risk), 2), to_entry=round((entry / px - 1) * 100, 2),
                rvol=round(rvol, 2), atr=round(a, 2), atr_pct=round(atr_pct, 1), age=int(age),
                support=round(sup, 2), bb=tags, spark=[float(x) for x in c[max(0, i - 59):i + 1]]), None

def core_breakdown(A, i, C):
    c, o, h, l, v = A["c"], A["o"], A["h"], A["l"], A["v"]
    if i < 50: return None, "היסטוריה קצרה"
    s = max(0, i - C["win"] + 1)
    cw, ow, lw, hw, vw = c[s:i + 1], o[s:i + 1], l[s:i + 1], h[s:i + 1], v[s:i + 1]
    n = len(cw)
    if n < C["sup_win"] + C["break_win"] + 5: return None, "היסטוריה קצרה לתבנית"

    pre_end = n - C["break_win"]
    pre_start = max(0, pre_end - C["sup_win"])
    if pre_end - pre_start < 10: return None, "אין מספיק נתוני תמיכה"
    
    support = float(np.min(lw[pre_start:pre_end]))
    if support <= 0: return None, "תמיכה לא תקינה"
    
    brk_low = float(np.min(lw[pre_end:]))
    brk_pct = (support - brk_low) / support * 100
    if brk_pct < C["break_min"]: return None, "לא נשברה תמיכה מספיק"
    if brk_pct > C["break_max"]: return None, "שבירה עמוקה מדי"
    
    px = float(cw[-1])
    reclaim_dist = (px / support - 1) * 100
    if reclaim_dist < C["reclaim_lo"]: return None, "עוד לא תפסה מחדש"
    if reclaim_dist > C["reclaim_hi"]: return None, "כבר התרחקה מדי מהתמיכה"

    base_v = float(np.mean(vw[max(0, pre_start - 30):pre_start])) if pre_start > 5 else float(np.mean(vw[:max(pre_start, 1)]))
    base_v = max(base_v, 1.0) # Prevent zero div
    brk_v = vw[pre_end:]
    if len(brk_v) == 0: return None, "אין נתוני נפח"
    brk_spike = float(np.max(brk_v) / base_v)

    # RVOL optimized for Day Trading
    base_50 = v[max(0, i - 50):i].mean() if i > 0 else 1.0
    rvol = float(v[i] / base_50) if base_50 > 0 else 0.0

    a = float(A["atr"][i])
    if not np.isfinite(a) or a <= 0: return None, "אין ATR"
    if a < C["atr_abs"]: return None, "ATR נמוך מדי"
    atr_pct = a / px * 100
    if atr_pct < C["atr_pct"]: return None, "תנודתיות נמוכה"

    entry = max(px, float(h[i])) + .05 * a
    stop = min(brk_low - .3 * a, entry - 2.2 * a)
    risk = entry - stop
    if risk <= 0: return None, "סטופ לא תקין"

    resistance = float(np.max(hw[pre_start:pre_end]))
    if resistance <= entry: resistance = entry + 2.6 * risk
    span = resistance - support
    t1 = max(support + span * .4, entry + 1.0 * risk)
    t2 = max(support + span * .7, entry + 1.8 * risk)
    t3 = max(resistance, entry + 2.6 * risk)
    rr = (t1 - entry) / risk
    if rr < C["rr_min"]: return None, "אין מרווח ליעד"

    return dict(kind="breakdown", support=round(support, 2), brk_low=round(brk_low, 2),
                brk_pct=round(brk_pct, 1), spike=round(brk_spike, 2), reclaim_pct=round(reclaim_dist, 1),
                resistance=round(resistance, 2), price=round(px, 2), entry=round(entry, 2), stop=round(stop, 2),
                tp1=round(t1, 2), tp2=round(t2, 2), tp3=round(t3, 2), rr=round(float(rr), 2),
                risk_share=round(float(risk), 2), to_entry=round((entry / px - 1) * 100, 2),
                rvol=round(rvol, 2), atr=round(a, 2), atr_pct=round(atr_pct, 1), age=0, 
                rise=round(brk_pct, 1), leg_bars=C["break_win"], dry=round(brk_spike, 2), 
                retrace=round(reclaim_dist, 1), spark=[float(x) for x in c[max(0, i - 59):i + 1]]), None

def why_he(r): return f"עלתה {r['rise']:.0f}% ב-{r['leg_bars']} ימים, תיקנה {r['retrace']:.0f}% בנפח קטן פי {r['dry']:.2f}, נתמכת ב-{r.get('support',0):.2f}"
def why_breakdown(r): return f"שברה תמיכה ב-{r['brk_pct']:.1f}% בנפח אגרסיבי, נתפסה מחדש {r['reclaim_pct']:+.1f}%."

def ltr(txt): return f'<bdi dir="ltr">{txt}</bdi>'

def rank_verdict(r):
    age = r.get("age", 0)
    if r.get("kind") == "breakdown":
        return dict(tier="?", color="#8B5CF6", sort=(2, 0, age), headline="שבירה ותפיסה מחדש", detail="תבנית אגרסיבית למומנטום.")
    dry = r["dry"]
    if 1.30 <= dry <= 1.60:
        return dict(tier="A", color="#10B981", sort=(0, round(abs(dry - 1.45)*20)/20, age), headline=f"יובש אידיאלי {dry:.2f}×", detail="טווח יובש אמין היסטורית.")
    if (1.10 <= dry < 1.30) or (1.60 < dry <= 2.20):
        return dict(tier="B", color="#F59E0B", sort=(1, round(abs(dry - 1.45)*20)/20, age), headline=f"יובש חזק {dry:.2f}×", detail="קרוב לטווח המוכח.")
    return dict(tier="C", color="#EF4444", sort=(2, round(abs(dry - 1.45)*20)/20, age), headline=f"יובש קיצוני {dry:.2f}×", detail="סיכון גבוה מהרגיל.")

def sparkline(vals):
    if not vals or len(vals) < 3: return ""
    lo, hi = min(vals), max(vals)
    rng = hi - lo if hi > lo else 1
    pts = " ".join(f"{i/(len(vals)-1)*100:.2f},{(1-(v-lo)/rng)*26+2:.2f}" for i, v in enumerate(vals))
    return f'<div class="spark"><svg viewBox="0 0 100 35" preserveAspectRatio="none"><polyline points="{pts}" fill="none" stroke="#3B82F6" stroke-width="2" vector-effect="non-scaling-stroke"/></svg></div>'

def ladder(r):
    lo, hi = r["stop"], r["tp3"]
    rng = hi - lo if hi > lo else 1
    lv = sorted([("TP3", r["tp3"], "tp"), ("TP2", r["tp2"], "tp"), ("TP1", r["tp1"], "tp"),
                 ("כניסה", r["entry"], "en"), ("מחיר", r["price"], "now"), ("סטופ", r["stop"], "st")], key=lambda x: -x[1])
    used, html = [], ""
    for tag, p, cls in lv:
        top = 10 + (1 - (p - lo) / rng) * 160
        for u in used:
            if abs(top - u) < 22: top = u + 22
        used.append(top)
        html += f'<div class="lv {cls}" style="top:{top:.0f}px"><div class="tg">{tag}</div><div class="dt"></div><div class="px">{p:.2f}</div></div>'
    return f'<div class="lad">{html}</div>'

def card(r, best=False):
    v = rank_verdict(r)
    ch = "".join(f'<span class="chip g">BB {b}</span>' for b in r.get("bb", []))
    ch += f'<span class="chip">ATR {r["atr_pct"]:.1f}%</span><span class="chip">RVOL {r["rvol"]:.2f}</span>'
    bd = '<span class="badge">🔥 מומנטום קרוב</span>' if r["to_entry"] <= 1.0 else ''
    return f"""
    <div class="tc{' top' if best else ''}">
        <div class="tc-h">
            <div style="display:flex;align-items:baseline;gap:.5rem"><div class="tc-s">{r['ticker']}</div></div>
            <div class="tc-d"><b>{v['tier']}</b><span>דירוג איכות</span></div>
        </div>
        <div style="color:{v['color']};font-weight:600;font-size:0.9rem;">{v['headline']}</div>
        {sparkline(r.get('spark', []))}
        <div class="chips">{bd}{ch}</div>
        {ladder(r)}
        <div class="tl"><span>סיכון <b>${r['risk_share']:.2f}</b> למניה</span><span>מרחק להפעלה <b>{ltr(f"{r['to_entry']:+.1f}%")}</b></span></div>
        <div class="tw">{r['why']}</div>
    </div>
    """

# ---- App Flow ----
il = timezone(timedelta(hours=3))
now = datetime.now(il)
ny = now.astimezone(timezone(timedelta(hours=-4)))
mkt_open = ny.weekday() < 5 and (9 * 60 + 30) <= (ny.hour * 60 + ny.minute) < 16 * 60

st.markdown(f"""
<div class="bar">
    <div class="brand"><h1>MAVRI <span>PRO</span></h1><em>סורק מומנטום ומסחר יומי</em></div>
    <div class="status">
        <div><i>שוק ניו-יורק</i><b><span class="led {'on' if mkt_open else 'off'}"></span>{'פתוח' if mkt_open else 'סגור'}</b></div>
        <div><i>שעה (IL)</i><b>{now:%H:%M}</b></div>
    </div>
</div>
""", unsafe_allow_html=True)

if "P" not in st.session_state: st.session_state["P"] = dict(PRESETS["מסחר יומי מומנטום"])
p_cols = st.columns([1, 1, 1, 3])
for col, name in zip(p_cols[:3], PRESETS):
    if col.button(name, use_container_width=True):
        st.session_state["P"] = dict(PRESETS[name])
        st.rerun()

P = st.session_state["P"]

with st.container():
    f1, f2, f3, f4, f5, f6 = st.columns([1, 1, 1, 1.2, 1, 1.5])
    dry_min = f1.number_input("יובש נפח מינ׳", 0.5, 3.0, float(P["dry"]), 0.05)
    bb_lbl = f2.selectbox("בולינג׳ר", ["כבוי", "רצועה תחתונה", "רצועה עליונה", "אחת מהשתיים"], index=["כבוי", "רצועה תחתונה", "רצועה עליונה", "אחת מהשתיים"].index(P["bb"]))
    min_px = f3.number_input("מחיר מינ׳ $", 1.0, 500.0, float(P["price"]), 1.0)
    min_dv = f4.number_input("מחזור $ (מיל׳)", 1.0, 500.0, float(P["dv"]), 1.0)
    atr_abs = f5.number_input("ATR מינ׳ %", 0.5, 20.0, float(P["atrp"]), 0.5)
    
    f6.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
    go_ = f6.button("🚀 סרוק את השוק עכשיו", type="primary", use_container_width=True)

ptype = st.radio("סוג תבנית", ["זינוק ותיקון לתמיכה", "שבירת תמיכה ותפיסה מחדש", "שתי התבניות"], horizontal=True)

with st.expander("⚙️ הגדרות סינון מתקדמות (נזילות, מניות ומסחר יומי)"):
    j1, j2, j3, j4 = st.columns(4)
    min_sh = j1.number_input("נפח מניות (מיל׳ ביום)", 0.0, 50.0, float(P.get("sh", 1.0)), 0.5, help="חובה למסחר יומי להבטחת Spread נמוך.")
    limit = j2.slider("מספר מניות לסריקה (Batches)", 200, 6000, 4000, 200, help="סריקה של 4000 מבוצעת במנות של 50 למניעת חסימה.")
    acct = j3.number_input("גודל תיק $ (לחישוב גודל)", 500, 1000000, 25000, 500)
    riskp = j4.slider("סיכון לעסקה %", 0.1, 5.0, 0.5, 0.1)

C = dict(leg_min=P.get("rise", 5), leg_max=30, rise_min=P.get("rise", 10), retr_lo=P["retr"][0],
         retr_hi=P["retr"][1], age_lo=P["age"][0], age_hi=P["age"][1], atr_pct=atr_abs,
         atr_abs=P.get("atr", 0.5), rr_min=P["rr"], dry_min=dry_min, bb_look=3, win=150,
         off_max=P.get("off", 10), bb_hard=(P.get("bbmode") == "חובה"),
         bb={"כבוי": "off", "רצועה תחתונה": "lower", "רצועה עליונה": "upper", "אחת מהשתיים": "both"}[bb_lbl])

CB = dict(win=150, sup_win=35, break_win=4, break_min=2.0, break_max=12.0, reclaim_lo=-1.0, reclaim_hi=8.0,
          atr_abs=0.5, atr_pct=atr_abs, rr_min=P["rr"])

if go_:
    t0 = time.time()
    uni, _ = build_universe(limit)
    st.info(f"מתחיל סריקה של {len(uni):,} מניות. המערכת מחלקת למנות של 50 מניות למניעת חסימת שרת...")
    
    prog, note = st.progress(0.0), st.empty()
    rows, drop, liq = [], {}, 0
    # FIX 1: Small batches (50) for Yahoo Finance stability
    batches = [tuple(uni[i:i + 50]) for i in range(0, len(uni), 50)]
    
    for i, b in enumerate(batches):
        note.caption(f"סורק מנה {i+1} מתוך {len(batches)}... התאמות עד כה: {len(rows)}")
        # FIX 2: period="6mo" limits data footprint significantly
        for t, d in fetch(b, period="6mo").items():
            try:
                p = float(d["Close"].iloc[-1])
                if p < min_px: drop["מחיר נמוך"] = drop.get("מחיר נמוך", 0) + 1; continue
                
                avg_sh = float(d["Volume"].tail(20).mean())
                if p * avg_sh < min_dv * 1e6: drop["מחזור נמוך $"] = drop.get("מחזור נמוך $", 0) + 1; continue
                
                # FIX 3: Volume filter check for day traders (Spread control)
                if min_sh > 0 and avg_sh < min_sh * 1e6: 
                    drop["נפח מניות דליל"] = drop.get("נפח מניות דליל", 0) + 1; continue
                
                liq += 1
                A = prep(d)
                found_here = []
                
                if ptype in ("זינוק ותיקון לתמיכה", "שתי התבניות"):
                    r, why = core(A, len(A["c"]) - 1, C)
                    if r is None: drop[why] = drop.get(why, 0) + 1
                    else: r["kind"] = "pullback"; r["why"] = why_he(r); found_here.append(r)
                        
                if ptype in ("שבירת תמיכה ותפיסה מחדש", "שתי התבניות"):
                    rb, whyb = core_breakdown(A, len(A["c"]) - 1, CB)
                    if rb is None: drop["שבירה: " + whyb] = drop.get("שבירה: " + whyb, 0) + 1
                    else: rb["kind"] = "breakdown"; rb["why"] = why_breakdown(rb); found_here.append(rb)
                        
                for r in found_here:
                    sh = int((acct * riskp / 100) / r["risk_share"]) if r["risk_share"] > 0 else 0
                    r.update(ticker=t, shares=sh, risk_total=round(sh * r["risk_share"], 2))
                    rows.append(r)
            except Exception:
                drop["שגיאה טכנית"] = drop.get("שגיאה טכנית", 0) + 1
        
        prog.progress((i + 1) / len(batches))
        # FIX 4: Safety sleep to respect YF Rate Limits
        time.sleep(0.3)
        
    prog.empty(); note.empty()
    st.session_state.update(rows=sorted(rows, key=lambda x: x["to_entry"]), stats=(len(uni), liq), took=time.time()-t0)

if "rows" in st.session_state:
    rows = st.session_state["rows"]
    u, l_ = st.session_state["stats"]
    
    st.markdown(f"""
    <div class="kpi">
        <div><b>{u:,}</b><span>מניות נסרקו</span></div>
        <div><b>{l_:,}</b><span>עברו סינון נזילות</span></div>
        <div class="hi"><b>{len(rows)}</b><span>תבניות אותרו</span></div>
        <div><b>{st.session_state.get('took',0):.1f}s</b><span>זמן סריקה</span></div>
    </div>
    """, unsafe_allow_html=True)

    if not rows:
        st.warning("לא נמצאו מניות העונות לקריטריונים המבוקשים. ה-ATR והנפח המינימלי למסחר יומי חותכים מניות דשדוש רבות. נסה להקל בפרמטרים.")
    else:
        st.subheader("🔥 הזדמנויות מסחר (ממוין לפי קרבה להפעלה)")
        cols = st.columns(3, gap="large")
        for idx, r in enumerate(rows[:15]):  # Show top 15 matches to maintain speed
            with cols[idx % 3]:
                st.markdown(card(r, best=(idx==0)), unsafe_allow_html=True)
                if st.button("📊 פתח גרף מורחב", key=f"c_{r['ticker']}", use_container_width=True):
                    st.toast(f"הוראות מסחר ל-{r['ticker']} — כניסה: {r['entry']:.2f}, סטופ: {r['stop']:.2f}", icon="📈")
