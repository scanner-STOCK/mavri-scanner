"""
MAVRI Pro — Day Trading & Momentum Screener.
Includes Historical Backtest, Drop Funnel, and strict Reversal Candle logic.
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

st.set_page_config(page_title="MAVRI Pro", layout="wide", initial_sidebar_state="collapsed", page_icon="⚡")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;600;700&display=swap');
:root { --bg:#0A0E17; --pan:#131824; --rz:#1E2536; --ln:#2A3245; --tx:#F8FAFC; --mu:#94A3B8; --dm:#64748B; --gd:#F59E0B; --lg:#10B981; --sh:#EF4444; --bl:#3B82F6; }
html,body,[class*="css"]{font-family:'Heebo',sans-serif;}
.stApp{background:var(--bg);color:var(--tx);}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1rem;padding-bottom:4rem;max-width:1400px;}
hr{border-color:var(--ln);}
.bar{display:flex;align-items:center;justify-content:space-between;gap:1rem;background:rgba(19,24,36,0.7);backdrop-filter:blur(10px);border:1px solid var(--ln);border-radius:12px;padding:1rem 1.5rem;flex-wrap:wrap;margin-bottom:1.5rem;}
.brand{display:flex;align-items:baseline;gap:0.8rem;}
.brand h1{font-size:2rem;font-weight:700;margin:0;color:var(--tx);letter-spacing:0.1em;background:linear-gradient(90deg,#3B82F6,#10B981);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.brand em{font-style:normal;font-size:0.9rem;color:var(--mu);letter-spacing:0.05em;}
.status{display:flex;gap:2rem;align-items:center;direction:rtl;}
.status i{font-style:normal;font-size:0.75rem;color:var(--dm);display:block;text-transform:uppercase;}
.status b{font-size:1.1rem;font-weight:600;color:var(--tx);font-variant-numeric:tabular-nums;}
.led{width:8px;height:8px;border-radius:50%;display:inline-block;margin-left:0.5rem;}
.led.on{background:var(--lg);box-shadow:0 0 10px rgba(16,185,129,0.8);}
.led.off{background:var(--sh);box-shadow:0 0 10px rgba(239,68,68,0.8);}
.kpi{display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:1.5rem;}
.kpi div{flex:1;min-width:120px;padding:1rem;background:var(--pan);border:1px solid var(--ln);border-radius:10px;direction:rtl;box-shadow:0 4px 6px -1px rgba(0,0,0,0.1);}
.kpi b{font-size:1.5rem;font-weight:600;display:block;line-height:1.2;color:var(--tx);}
.kpi span{font-size:0.75rem;color:var(--dm);font-weight:500;}
.kpi div.hi b{color:var(--lg);}
.tc{background:rgba(19,24,36,0.6);backdrop-filter:blur(8px);border:1px solid var(--ln);border-radius:12px;padding:1.2rem;direction:rtl;height:100%;transition:all 0.3s ease;box-shadow:0 4px 15px rgba(0,0,0,0.1);}
.tc:hover{transform:translateY(-4px);box-shadow:0 12px 30px rgba(0,0,0,0.4);border-color:var(--bl);}
.tc.top{border-color:rgba(16,185,129,0.5);box-shadow:0 0 20px rgba(16,185,129,0.1);}
.tc-h{display:flex;align-items:baseline;justify-content:space-between;padding-bottom:0.5rem;border-bottom:1px solid var(--ln);margin-bottom:0.8rem;}
.tc-s{font-size:1.8rem;font-weight:700;color:var(--tx);direction:ltr;letter-spacing:0.05em;}
.tc-d{text-align:left;direction:ltr;}
.tc-d b{font-size:1.4rem;font-weight:600;color:var(--gd);}
.tc-d span{display:block;font-size:0.7rem;color:var(--mu);direction:rtl;}
.spark svg{width:100%;height:35px;display:block;opacity:0.9;margin:0.5rem 0;}
.chips{display:flex;gap:0.4rem;flex-wrap:wrap;margin:0.5rem 0;}
.chip{font-size:0.7rem;padding:0.2rem 0.5rem;background:var(--rz);border:1px solid var(--ln);border-radius:6px;color:var(--mu);font-weight:500;}
.chip.g{border-color:rgba(245,158,11,0.4);color:var(--gd);background:rgba(245,158,11,0.1);}
.badge{display:inline-block;font-size:0.65rem;padding:0.2rem 0.6rem;border-radius:4px;background:rgba(16,185,129,0.15);color:var(--lg);margin-right:0.4rem;font-weight:600;}
.lad{position:relative;height:180px;margin:1rem 0;background:rgba(0,0,0,0.1);border-radius:8px;padding:10px 0;}
.lad:before{content:'';position:absolute;right:80px;top:10px;bottom:10px;width:2px;background:linear-gradient(180deg,rgba(16,185,129,0.5),rgba(239,68,68,0.5));border-radius:2px;}
.lv{position:absolute;right:0;left:0;height:0;display:flex;align-items:center;gap:0.6rem;}
.lv .tg{width:75px;text-align:left;font-size:0.7rem;color:var(--dm);font-weight:500;}
.lv .dt{width:8px;height:8px;border-radius:50%;background:var(--dm);margin-right:-4px;flex:none;z-index:2;border:2px solid var(--pan);}
.lv .px{font-size:0.95rem;font-weight:600;color:var(--mu);font-variant-numeric:tabular-nums;direction:ltr;}
.lv.tp .dt{background:var(--lg);} .lv.tp .px{color:var(--lg);}
.lv.st .dt{background:var(--sh);} .lv.st .px,.lv.st .tg{color:var(--sh);}
.lv.en .dt{width:12px;height:12px;background:var(--bl);margin-right:-6px;box-shadow:0 0 8px rgba(59,130,246,0.6);border:none;}
.lv.en .px{font-size:1.2rem;font-weight:700;color:var(--tx);}
.lv.en .tg{color:var(--bl);font-weight:700;}
.tl{display:flex;justify-content:space-between;font-size:0.75rem;color:var(--dm);border-top:1px solid var(--ln);padding-top:0.6rem;margin-top:0.5rem;}
.tl b{color:var(--tx);font-weight:600;}
.tw{font-size:0.8rem;line-height:1.6;color:var(--mu);margin-top:0.8rem;padding:0.8rem;background:var(--rz);border-radius:8px;}
div[data-testid="stExpander"]{border:1px solid var(--ln);border-radius:10px;background:var(--pan);overflow:hidden;}
div[data-testid="stExpander"] summary{background:rgba(255,255,255,0.02);}
h3,label{color:var(--mu) !important;}
.stButton button{border-radius:8px;font-weight:600;transition:all 0.2s;}
.stButton button:hover{transform:scale(1.02);}
.blocker{direction:rtl;font-size:0.85rem;color:var(--dm);margin:-0.5rem 0 1rem;background:rgba(239,68,68,0.1);padding:0.8rem;border-radius:8px;border:1px solid rgba(239,68,68,0.2);}
.blocker b{color:var(--sh);font-weight:600;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

UA = {"User-Agent": "Mozilla/5.0"}
PRESETS = {
    "מסחר יומי מומנטום": dict(dry=1.15, bb="כבוי", bbmode="בונוס", price=5.0, dv=15.0, sh=0.5, 
                             atr=0.5, atrp=2.5, rise=10, age=(1, 15), retr=(15, 60), rr=1.5, off=5),
    "סווינג התבנית שלי": dict(dry=1.30, bb="רצועה תחתונה", bbmode="בונוס", price=10.0, dv=15.0,
                             sh=1.0, atr=1.0, atrp=2.0, rise=12, age=(3, 20), retr=(20, 70), rr=1.5, off=12),
    "רחב מאוד": dict(dry=1.05, bb="אחת מהשתיים", bbmode="בונוס", price=5.0, dv=8.0, sh=0.5,
                atr=0.4, atrp=1.5, rise=6, age=(2, 35), retr=(12, 88), rr=1.0, off=20)
}

@st.cache_data(ttl=86400, show_spinner=False)
def build_universe(limit):
    out = []
    try:
        txt = requests.get("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", headers=UA).text
        df = pd.read_csv(io.StringIO(txt), sep="|")
        out += [x for x in df["Symbol"].dropna() if isinstance(x, str) and x.isalpha()]
        txt2 = requests.get("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", headers=UA).text
        df2 = pd.read_csv(io.StringIO(txt2), sep="|")
        out += [x for x in df2["ACT Symbol"].dropna() if isinstance(x, str) and x.isalpha()]
    except Exception: pass
    seen = set()
    return [x for x in out if not (x in seen or seen.add(x))][:limit]

@st.cache_data(ttl=1800, max_entries=30, show_spinner=False)
def fetch(tickers, period="6mo"):
    try:
        raw = yf.download(list(tickers), period=period, interval="1d", group_by="ticker", auto_adjust=False, threads=True, progress=False)
        return {t: raw[t].dropna() if isinstance(raw.columns, pd.MultiIndex) else raw.dropna() for t in tickers if t in raw}
    except: return {}

def prep(d):
    c, o, h, l, v = d["Close"].values.astype(float), d["Open"].values.astype(float), d["High"].values.astype(float), d["Low"].values.astype(float), d["Volume"].values.astype(float)
    tr = np.maximum.reduce([h - l, np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))])
    tr[0] = h[0] - l[0]
    mid = pd.Series(c).rolling(20).mean()
    sd = pd.Series(c).rolling(20).std()
    return dict(c=c, o=o, h=h, l=l, v=v, atr=pd.Series(tr).rolling(14).mean().values, bbu=(mid + 2 * sd).values, bbl=(mid - 2 * sd).values)

def bb_hits(A, i, look, side):
    out = []
    for j in range(max(0, i - look + 1), i + 1):
        if side in ("lower", "both") and A["l"][j] <= A["bbl"][j]: out.append("תחתונה")
        if side in ("upper", "both") and A["h"][j] >= A["bbu"][j]: out.append("עליונה")
    return list(set(out))

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
    
    low_i = st_ + int(np.argmin(cw[st_:peak])) if peak > st_ else st_
    leg_bars = peak - low_i
    if leg_bars < C["leg_min"]: return None, "זינוק קצר מדי"
    lo_px, pk_px, px = float(cw[low_i]), float(cw[peak]), float(cw[-1])
    
    rise = (pk_px / lo_px - 1) * 100 if lo_px > 0 else 0
    if rise < C["rise_min"]: return None, "זינוק קטן מדי"
    retr = (pk_px - px) / (pk_px - lo_px) * 100 if pk_px > lo_px else 0
    if retr < C["retr_lo"]: return None, "כמעט לא תיקנה"
    if retr > C["retr_hi"]: return None, "החזירה את כל הזינוק"
    
    base = vw[max(0, low_i - 50):low_i]
    base_v = max(float(base.mean()) if len(base) >= 5 else float(vw[:max(low_i, 1)].mean()), 1.0)
    leg_v, pull_v = vw[low_i:peak + 1], vw[peak:]
    if len(leg_v) == 0 or len(pull_v) == 0 or pull_v.mean() <= 0: return None, "אין נתוני נפח"
    
    spike = float(leg_v.max() / base_v)
    dry = float(leg_v.mean() / pull_v.mean())
    if dry < C["dry_min"]: return None, "הנפח לא התייבש"
    
    # REVERSAL CANDLE LOGIC (Crucial for Day Trading)
    is_green = float(cw[-1]) > float(ow[-1])
    total_size = float(h[i]) - float(l[i])
    lower_wick = min(float(cw[-1]), float(ow[-1])) - float(l[i])
    is_hammer = total_size > 0 and (lower_wick / total_size) > 0.4
    if not (is_green or is_hammer): return None, "אין נר היפוך/פטיש"

    tags = bb_hits(A, i, C["bb_look"], C["bb"]) if C["bb"] != "off" else []
    if C["bb_hard"] and C["bb"] != "off" and not tags: return None, "אין נר בולינג׳ר"
    
    sup = float(lw[peak:].min())
    off_low = (px / sup - 1) * 100 if sup > 0 else 999
    if off_low > C["off_max"]: return None, "רחוקה מדי מהתמיכה"
    
    a = float(A["atr"][i])
    if not np.isfinite(a) or a <= 0: return None, "אין ATR"
    if a < C["atr_abs"]: return None, "ATR נמוך מדי"
    atr_pct = a / px * 100
    if atr_pct < C["atr_pct"]: return None, "תנודתיות נמוכה"
    
    base_50 = v[max(0, i - 50):i].mean() if i > 0 else 1.0
    rvol = float(v[i] / base_50) if base_50 > 0 else 0.0
    
    entry = max(px, float(h[i])) + .05 * a
    stop = max(min(sup - .35 * a, entry - 3 * a), entry - 4 * a)
    risk = entry - stop
    if risk <= 0: return None, "סטופ לא תקין"
    
    tp1 = pk_px
    if (tp1 - entry) / risk < C["rr_min"]: return None, "אין מרווח ליעד"
    
    return dict(dry=round(dry,2), spike=round(spike,2), rise=round(rise,1), leg_bars=int(leg_bars), retrace=round(retr,1), price=round(px,2), entry=round(entry,2), stop=round(stop,2), tp1=round(tp1,2), tp2=round(tp1+.5*(pk_px-lo_px),2), tp3=round(tp1+(pk_px-lo_px),2), rr=round((tp1-entry)/risk,2), risk_share=round(risk,2), to_entry=round((entry/px-1)*100,2), rvol=round(rvol,2), atr_pct=round(atr_pct,1), age=int(age), support=round(sup,2), bb=tags, spark=[float(x) for x in c[max(0, i-59):i+1]]), None

def why_he(r): return f"זינקה בנפח פי {r['spike']:.1f}, תיקנה במחזור יבש ({r['dry']:.2f}x), וייצרה נר היפוך/פטיש על התמיכה."

def funnel_chart(drop):
    items = sorted([(k, v) for k, v in drop.items()], key=lambda x: x[1])
    if not items: return None
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    colors = ["#EF4444" if i == len(items) - 1 else "#8E4046" for i in range(len(items))]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors, text=[f"{v:,}" for v in values], textposition="outside", textfont=dict(color="#94A3B8")))
    fig.update_layout(height=max(250, 30 * len(items)), margin=dict(l=10, r=40, t=10, b=10), paper_bgcolor="#131824", plot_bgcolor="#131824", xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False))
    return fig

def backtest_walk(d, C):
    A = prep(d)
    out = []
    last = -999
    for i in range(60, len(A["c"]) - 10):
        if i - last < 5: continue
        r, _ = core(A, i, C)
        if r:
            entry, stop, tp1 = r["entry"], r["stop"], r["tp1"]
            filled, res = False, None
            for j in range(i + 1, min(i + 11, len(A["c"]))):
                if not filled:
                    if A["h"][j] >= entry: filled = True
                if filled:
                    if A["l"][j] <= stop: res = -1.0; break
                    if A["h"][j] >= tp1: res = (tp1 - entry) / (entry - stop); break
            if filled and res is not None:
                out.append(dict(R=res, dry=r["dry"], rr=r["rr"]))
                last = i
    return out

il = timezone(timedelta(hours=3))
now = datetime.now(il)
st.markdown(f"""
<div class="bar">
    <div class="brand"><h1>MAVRI <span>PRO</span></h1><em>סורק מומנטום חכם ⚡</em></div>
    <div class="status"><div><i>שעה (IL)</i><b>{now:%H:%M}</b></div></div>
</div>
""", unsafe_allow_html=True)

if "P" not in st.session_state: st.session_state["P"] = dict(PRESETS["מסחר יומי מומנטום"])
p_cols = st.columns([1, 1, 1, 3])
for col, name in zip(p_cols[:3], PRESETS):
    if col.button(name, use_container_width=True):
        st.session_state["P"] = dict(PRESETS[name]); st.rerun()

P = st.session_state["P"]

f1, f2, f3, f4, f5, f6 = st.columns([1, 1, 1, 1.2, 1, 1.5])
dry_min = f1.number_input("יובש נפח מינ׳", 0.5, 3.0, float(P["dry"]), 0.05)
bb_lbl = f2.selectbox("בולינג׳ר", ["כבוי", "רצועה תחתונה", "רצועה עליונה"], index=["כבוי", "רצועה תחתונה", "רצועה עליונה"].index(P["bb"]))
min_px = f3.number_input("מחיר מינ׳ $", 1.0, 500.0, float(P["price"]), 1.0)
min_dv = f4.number_input("מחזור $ (מיל׳)", 1.0, 500.0, float(P["dv"]), 1.0)
atr_abs = f5.number_input("ATR מינ׳ %", 0.5, 20.0, float(P["atrp"]), 0.5)
f6.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
go_ = f6.button("🚀 סרוק את השוק עכשיו", type="primary", use_container_width=True)

with st.expander("⚙️ הגדרות סינון מתקדמות (נזילות ומניות)"):
    j1, j2, j3, j4 = st.columns(4)
    min_sh = j1.number_input("נפח מניות (מיל׳)", 0.0, 50.0, float(P.get("sh", 0.5)), 0.5)
    limit = j2.slider("מספר מניות לסריקה", 200, 6000, 4000, 200)
    acct = j3.number_input("גודל תיק $", 500, 1000000, 25000, 500)
    riskp = j4.slider("סיכון לעסקה %", 0.1, 5.0, 0.5, 0.1)

C = dict(leg_min=P.get("rise", 5), leg_max=30, rise_min=P.get("rise", 10), retr_lo=P["retr"][0], retr_hi=P["retr"][1], age_lo=P["age"][0], age_hi=P["age"][1], atr_pct=atr_abs, atr_abs=P.get("atr", 0.5), rr_min=P["rr"], dry_min=dry_min, bb_look=3, win=150, off_max=P.get("off", 10), bb_hard=(P.get("bbmode") == "חובה"), bb={"כבוי": "off", "רצועה תחתונה": "lower", "רצועה עליונה": "upper"}[bb_lbl])

with st.expander("📊 בדיקה היסטורית (Backtest)"):
    st.write("מריץ לאחור את הסינון המדויק שבחרת למעלה על מנת לוודא תוחלת חיובית.")
    if st.button("הרץ בדיקה היסטורית", type="secondary"):
        uni_bt = build_universe(100)
        bt_trades = []
        bar = st.progress(0.0)
        bt_data = fetch(uni_bt, "2y")
        for idx, (t, d) in enumerate(bt_data.items()):
            bt_trades.extend(backtest_walk(d, C))
            bar.progress((idx + 1) / len(bt_data))
        bar.empty()
        if bt_trades:
            df_bt = pd.DataFrame(bt_trades)
            win_rate = (df_bt['R'] > 0).mean() * 100
            st.success(f"נמצאו {len(bt_trades)} עסקאות בעבר. אחוז הצלחה ליעד ראשון: {win_rate:.1f}% | תוחלת ממוצעת: {df_bt['R'].mean():+.2f}R")
        else:
            st.warning("לא נמצאו עסקאות היסטוריות עם הסינון הנוכחי - כנראה הוא מחמיר מדי.")

if go_:
    t0 = time.time()
    uni = build_universe(limit)
    prog, note = st.progress(0.0), st.empty()
    rows, drop, liq = [], {}, 0
    batches = [tuple(uni[i:i + 50]) for i in range(0, len(uni), 50)]
    
    for i, b in enumerate(batches):
        note.caption(f"סורק מנה {i+1} מתוך {len(batches)}... {len(rows)} התאמות")
        for t, d in fetch(b, "6mo").items():
            try:
                p = float(d["Close"].iloc[-1])
                if p < min_px: drop["מחיר נמוך מינ'"] = drop.get("מחיר נמוך מינ'", 0) + 1; continue
                avg_sh = float(d["Volume"].tail(20).mean())
                if p * avg_sh < min_dv * 1e6: drop["מחזור דולרי נמוך"] = drop.get("מחזור דולרי נמוך", 0) + 1; continue
                if min_sh > 0 and avg_sh < min_sh * 1e6: drop["מחזור מניות נמוך"] = drop.get("מחזור מניות נמוך", 0) + 1; continue
                liq += 1
                A = prep(d)
                r, why = core(A, len(A["c"]) - 1, C)
                if r is None: drop[why] = drop.get(why, 0) + 1
                else:
                    sh = int((acct * riskp / 100) / r["risk_share"]) if r["risk_share"] > 0 else 0
                    r.update(ticker=t, shares=sh, risk_total=round(sh * r["risk_share"], 2), why=why_he(r))
                    rows.append(r)
            except Exception:
                drop["שגיאה טכנית"] = drop.get("שגיאה טכנית", 0) + 1
        prog.progress((i + 1) / len(batches)); time.sleep(0.3)
    
    prog.empty(); note.empty()
    st.session_state.update(rows=sorted(rows, key=lambda x: x["to_entry"]), stats=(len(uni), liq), drop=drop, took=time.time()-t0)

if "rows" in st.session_state:
    rows = st.session_state["rows"]
    u, l_ = st.session_state["stats"]
    drop = st.session_state["drop"]
    
    st.markdown(f"""
    <div class="kpi">
        <div><b>{u:,}</b><span>נסרקו</span></div><div><b>{l_:,}</b><span>עברו נזילות</span></div>
        <div class="hi"><b>{len(rows)}</b><span>תבניות</span></div><div><b>{st.session_state.get('took',0):.1f}s</b><span>זמן</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    if drop:
        tb_k, tb_v = max(drop.items(), key=lambda x: x[1])
        st.markdown(f'<div class="blocker">המסנן שחותך הכי הרבה מניות הוא: <b>{tb_k}</b> ({tb_v:,} מניות). פתח את "מפל הסינון" למטה כדי לראות את השאר ולדייק את הסורק.</div>', unsafe_allow_html=True)
    
    with st.expander("🔻 מפל הסינון (איפה נפלו המניות)"):
        fc = funnel_chart(drop)
        if fc: st.plotly_chart(fc, use_container_width=True, config={"displayModeBar": False})

    if not rows:
        st.warning("לא נמצאו מניות בתבנית. נסה להוריד את יובש הנפח או את ה-ATR המינימלי.")
    else:
        st.subheader("🔥 הזדמנויות מסחר יומי / מומנטום")
        cols = st.columns(3, gap="large")
        for idx, r in enumerate(rows[:15]):
            with cols[idx % 3]:
                st.markdown(f"""
                <div class="tc{' top' if idx==0 else ''}">
                    <div class="tc-h"><div class="tc-s">{r['ticker']}</div><div class="tc-d"><b>{r['dry']}x</b><span>יובש נפח</span></div></div>
                    {sparkline(r.get('spark', []))}
                    <div class="chips"><span class="chip g">נר היפוך זוהה</span><span class="chip">ATR {r['atr_pct']:.1f}%</span></div>
                    <div class="tl"><span>כניסה: <b>${r['entry']:.2f}</b></span><span>סטופ: <b>${r['stop']:.2f}</b></span></div>
                    <div class="tw">{r['why']}</div>
                </div>
                """, unsafe_allow_html=True)
