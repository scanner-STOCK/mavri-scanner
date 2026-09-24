"""
MAVRI — סורק מסחר יומי v3.

זהה ל-v2 בלוגיקה. כל השינויים כאן הם בתצוגה:
  * ה-header של Streamlit נשאר גלוי — בתוכו כפתור התפריט והניווט בין העמודים.
    ב-v1/v2 הסתרתי אותו ב-CSS, וזה מה שגרם ל"חלקים נעלמים".
  * העמוד כולו RTL אמיתי, עם שדות מספר שנשארים LTR כדי שהמספרים לא יתהפכו.
  * גופנים גדולים יותר, פחות עמודות בשורה, ופריסה שנשברת יפה במסכים צרים.
  * הסרגל הצדדי נפתח כברירת מחדל, כדי שתמיד תראה לאיזה עמוד אתה עובר.

התקנה: pages/day_trading.py ליד app.py. מומלץ גם לשים את קובץ
.streamlit/config.toml ששלחתי איתו, כדי שהנושא הכהה יהיה אחיד.
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
IL = ZoneInfo("Asia/Jerusalem")
OPEN_M, CLOSE_M, ORB_M = 9 * 60 + 30, 16 * 60, 9 * 60 + 45

st.set_page_config(page_title="MAVRI — מסחר יומי", layout="wide",
                   initial_sidebar_state="auto", page_icon="◈")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Frank+Ruhl+Libre:wght@400;500;700&family=Heebo:wght@300;400;500;700&display=swap');
:root{--bg:#0B1216;--pan:#111C22;--ln:#243740;--tx:#E9EFF1;--mu:#A6B8C1;
      --dm:#6E858F;--gd:#D4A64B;--lg:#2FBF8F;--sh:#E0605F;}

/* גופן בסיס גדול יותר — הגרסה הקודמת הייתה קטנה מדי לקריאה */
html,body,[class*="css"]{font-family:'Heebo','Segoe UI',Arial,sans-serif;font-size:16px;}
.stApp{background:var(--bg);}

/* ה-header נשאר! בתוכו כפתור התפריט והניווט בין העמודים. */
header[data-testid="stHeader"]{background:transparent;}

.block-container{padding-top:1.1rem;padding-bottom:3rem;max-width:1180px;direction:rtl;}

/* RTL לכל התוויות, אבל שדות מספר נשארים LTR כדי שהספרות לא יתהפכו */
label,.stSelectbox label,.stRadio label{direction:rtl !important;text-align:right !important;
       color:var(--mu) !important;font-size:.85rem !important;font-weight:400 !important;}
input[type="number"],input[type="text"]{direction:ltr !important;text-align:left !important;}
div[data-testid="stCaptionContainer"]{direction:rtl;text-align:right;}
.stAlert{direction:rtl;text-align:right;}

.bar{display:flex;align-items:center;justify-content:space-between;gap:1rem;
     border-bottom:1px solid var(--ln);padding:.2rem 0 .7rem;flex-wrap:wrap;direction:rtl;}
.brand h1{font-family:'Frank Ruhl Libre',serif;font-size:1.6rem;margin:0;color:var(--tx);
          letter-spacing:.08em;}
.brand em{font-style:normal;font-size:.8rem;color:var(--dm);}
.status{display:flex;gap:1.5rem;}
.status i{font-style:normal;font-size:.72rem;color:var(--dm);display:block;}
.status b{font-family:'Frank Ruhl Libre',serif;font-size:1.05rem;color:var(--mu);
          font-variant-numeric:tabular-nums;}
.led{width:8px;height:8px;border-radius:50%;display:inline-block;margin-left:.35rem;}
.led.on{background:var(--lg);box-shadow:0 0 8px rgba(47,191,143,.8);}
.led.pre{background:var(--gd);box-shadow:0 0 8px rgba(212,166,75,.7);}
.led.off{background:var(--dm);}

.mode{font-size:.88rem;color:var(--mu);background:var(--pan);border:1px solid var(--ln);
      border-right:3px solid var(--gd);border-radius:5px;padding:.7rem .95rem;margin:.9rem 0;
      line-height:1.7;}

.kpi{display:flex;flex-wrap:wrap;border:1px solid var(--ln);border-radius:5px;
     overflow:hidden;margin:1rem 0;}
.kpi div{flex:1;min-width:120px;padding:.7rem .9rem;border-left:1px solid var(--ln);}
.kpi div:last-child{border-left:0;}
.kpi b{font-family:'Frank Ruhl Libre',serif;font-size:1.5rem;display:block;color:var(--tx);
       font-variant-numeric:tabular-nums;line-height:1.2;}
.kpi span{font-size:.75rem;color:var(--dm);}
.kpi div.hi b{color:var(--gd);}

.dcard{background:var(--pan);border:1px solid var(--ln);border-radius:6px;
       padding:1rem 1.05rem;height:100%;margin-bottom:.2rem;}
.dcard.up{border-right:3px solid var(--lg);} .dcard.dn{border-right:3px solid var(--sh);}
.dcard.miss{border-right:3px solid var(--dm);}
.dsym{font-family:'Frank Ruhl Libre',serif;font-size:1.6rem;font-weight:700;
      color:var(--tx);direction:ltr;}
.dgap{font-family:'Frank Ruhl Libre',serif;font-size:1.45rem;font-variant-numeric:tabular-nums;
      direction:ltr;}
.dgap.up{color:var(--lg);} .dgap.dn{color:var(--sh);}
.side{font-size:.82rem;color:var(--dm);margin:.25rem 0 .55rem;line-height:1.6;}
.chip{font-size:.74rem;padding:.16rem .5rem;border:1px solid var(--ln);border-radius:3px;
      color:var(--mu);margin:0 0 .3rem .3rem;display:inline-block;}
.chip.g{border-color:rgba(212,166,75,.5);color:var(--gd);}
.chip.r{border-color:rgba(224,96,95,.5);color:var(--sh);}
.plan{font-size:.84rem;line-height:1.95;color:var(--dm);border-top:1px solid var(--ln);
      margin-top:.6rem;padding-top:.55rem;}
.plan b{color:var(--tx);font-weight:500;font-variant-numeric:tabular-nums;}
.empty{border:1px dashed var(--ln);border-radius:6px;padding:2.4rem 1.4rem;text-align:center;
       color:var(--dm);line-height:2;font-size:.95rem;}
.note{font-size:.88rem;color:var(--mu);line-height:1.8;background:var(--pan);
      border:1px solid var(--ln);border-right:3px solid var(--sh);border-radius:5px;
      padding:.7rem .95rem;margin:.6rem 0 1rem;}
h3{font-family:'Frank Ruhl Libre',serif !important;color:var(--tx) !important;
   font-size:1.3rem !important;margin:1.4rem 0 .7rem !important;}
a.tv{font-size:.78rem;color:var(--dm);text-decoration:none;border-bottom:1px dotted var(--ln);}
a.tv:hover{color:var(--gd);}

@media(max-width:900px){
  [data-testid="stHorizontalBlock"]{flex-direction:column !important;gap:.4rem !important;}
  [data-testid="stHorizontalBlock"]>div{width:100% !important;flex:1 1 100% !important;}
  .block-container{padding-left:.6rem;padding-right:.6rem;}
  .kpi div{min-width:50%;} .status{gap:.9rem;}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------------- universe
UNIVERSE = """
AAPL MSFT NVDA GOOGL AMZN META TSLA AVGO AMD INTC QCOM MU MRVL SMCI ARM TSM
ON MPWR SWKS QRVO NXPI ALAB CRDO LRCX KLAC AMAT ADI TXN DELL HPQ STX WDC ASML
CRM ORCL ADBE NOW SNOW DDOG NET CRWD ZS OKTA PANW FTNT S PLTR MDB TEAM ESTC GTLB
PATH AI BBAI SOUN TTD APP U RBLX EA TTWO RDDT SNAP PINS BMBL DUOL
IONQ RGTI QBTS QUBT ARQQ
SHOP SQ PYPL COIN HOOD SOFI AFRM UPST LC ALLY COF SYF AXP V MA TOST NU
JPM BAC WFC C GS MS SCHW USB PNC RF KEY CFG HBAN FITB
UNH ELV CI CVS HUM CNC MOH LLY PFE MRK ABBV BMY AMGN GILD BIIB VRTX REGN
MRNA BNTX NVAX ILMN TMO SYK BSX MDT ABT ISRG EW HIMS OSCR
CRSP NTLA BEAM RXRX SRPT NBIX EXAS NTRA TEM PACB HALO VKTX APLS IOVA
ACAD FOLD INSM KRYS RARE PTCT AXSM SWTX RCKT VERA ANAB
XOM CVX COP EOG DVN FANG OXY HES SLB HAL BKR RIG VAL TDW
VLO MPC PSX KMI WMB OKE ET LNG AR RRC SWN EQT CHRD CRK GPOR
NEE DUK SO D AEP EXC XEL VST CEG NRG TLN
CAT DE HON GE MMM EMR ETN PH ROK DOV GEV PWR
BA LMT RTX NOC GD LHX AXON RKLB LUNR ASTS PL RDW VSAT
UPS FDX UNP CSX NSC ODFL XPO
DAL UAL AAL LUV JBLU
WMT COST TGT DG DLTR KR
HD LOW TSCO ORLY AZO BBY RH CHWY W ETSY EBAY WSM
NKE LULU DECK ONON SKX CROX VFC PVH RL TPR GPS ANF AEO URBN BIRK
SBUX MCD YUM CMG QSR DPZ WEN SHAK CAVA DRI TXRH WING CELH
PG KO PEP KDP MNST KHC GIS STZ TAP MO PM
DIS NFLX CMCSA WBD PARA LYV SPOT ROKU T VZ TMUS FUBO
LIN APD SHW ECL DD DOW LYB NUE STLD X CLF AA FCX SCCO NEM GOLD AEM HL CDE
RIO VALE MP ALB UEC CCJ LEU OKLO SMR NNE DNN UUUU
UBER LYFT DASH ABNB BKNG EXPE MAR HLT CZR MGM LVS WYNN PENN DKNG RCL CCL NCLH
F GM RIVN LCID NIO XPEV LI STLA BLNK CHPT
MARA RIOT CLSK HUT BITF WULF CIFR IREN CORZ MSTR BTDR GLXY BTBT
ENPH FSLR RUN NOVA ARRY SHLS CSIQ PLUG FCEL BE
LAZR OUST INDI AEVA MVIS
ANET CSCO JNPR FFIV CIEN LITE COHR VRT NVTS AOSL
IBM ACN ZM DOCU TWLO FSLY AKAM
GME AMC BYND TLRY ACB CGC SNDL OPEN CLOV NKLA
BABA JD PDD BIDU NTES TME BILI FUTU TIGR
SE MELI GRAB CPNG STNE PAGS ZK
SPY QQQ IWM
""".split()


def session_now():
    ny = datetime.now(NY)
    m = ny.hour * 60 + ny.minute
    if ny.weekday() >= 5:
        return ny, "סגור — סוף שבוע", "off", "closed"
    if 4 * 60 <= m < OPEN_M:
        return ny, "פרה-מרקט", "pre", "pre"
    if OPEN_M <= m < CLOSE_M:
        return ny, "פתוח", "on", "regular"
    if CLOSE_M <= m < 20 * 60:
        return ny, "אפטר-מרקט", "pre", "after"
    return ny, "סגור", "off", "closed"


# --------------------------------------------------------------- stage 1
def _daily_one(t):
    try:
        d = yf.Ticker(t).history(period="2mo", interval="1d",
                                 auto_adjust=False, actions=False)
    except Exception:
        return None
    if d is None or len(d) < 15:
        return None
    try:
        if getattr(d.index, "tz", None) is not None:
            d.index = d.index.tz_convert(NY)
        if d.index[-1].date() == datetime.now(NY).date():
            d = d.iloc[:-1]
        if len(d) < 15:
            return None
        c, h, l, v = (d["Close"].astype(float), d["High"].astype(float),
                      d["Low"].astype(float), d["Volume"].astype(float))
        pc = c.shift(1)
        tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        atr, px = float(tr.tail(14).mean()), float(c.iloc[-1])
        avg_vol = float(v.tail(20).mean())
        if px <= 0 or avg_vol <= 0 or not np.isfinite(atr) or atr <= 0:
            return None
        return dict(prev_close=px, atr=atr, atr_pct=atr / px * 100,
                    avg_vol=avg_vol, dollar_vol=px * avg_vol)
    except Exception:
        return None


# --------------------------------------------------------------- stage 2
def _cum_to(vols, mins, lo, hi):
    sel = (mins >= lo) & (mins <= hi)
    return float(vols[sel].sum()) if sel.any() else 0.0


def _intraday_one(t):
    try:
        d = yf.Ticker(t).history(period="5d", interval="5m", prepost=True,
                                 auto_adjust=False, actions=False)
    except Exception:
        return None
    if d is None or len(d) < 20:
        return None
    try:
        if getattr(d.index, "tz", None) is not None:
            d.index = d.index.tz_convert(NY)
        d = d.dropna()
        if len(d) < 20:
            return None
        mins = np.array([x.hour * 60 + x.minute for x in d.index])
        days = np.array([x.date() for x in d.index])
        vols = d["Volume"].astype(float).values
        uniq = sorted(set(days))
        if len(uniq) < 2:
            return None
        today, priors = uniq[-1], uniq[:-1]

        prev_reg = d[(days == priors[-1]) & (mins >= OPEN_M) & (mins < CLOSE_M)]
        if len(prev_reg) == 0:
            return None
        prev_close = float(prev_reg["Close"].iloc[-1])
        if prev_close <= 0:
            return None

        tmask = days == today
        td, tmins, tvols = d[tmask], mins[tmask], vols[tmask]
        now_m = int(tmins[-1])
        last = float(td["Close"].iloc[-1])

        pre = td[tmins < OPEN_M]
        pre_vol = float(pre["Volume"].sum()) if len(pre) else 0.0
        out = dict(last=last, prev_close=prev_close, now_min=now_m,
                   gap_pct=(last / prev_close - 1) * 100,
                   pre_vol=pre_vol, pre_dollar=pre_vol * last,
                   pre_high=float(pre["High"].max()) if len(pre) else np.nan,
                   pre_low=float(pre["Low"].min()) if len(pre) else np.nan)

        if now_m >= OPEN_M + 10:
            cut = min(now_m, CLOSE_M - 5)
            today_cum = _cum_to(tvols, tmins, OPEN_M, cut)
            base = [_cum_to(vols[days == p], mins[days == p], OPEN_M, cut) for p in priors]
        else:
            cut = min(now_m, OPEN_M - 5)
            today_cum = pre_vol
            base = [_cum_to(vols[days == p], mins[days == p], 4 * 60, cut) for p in priors]
        base = [b for b in base if b > 0]
        out["rvol"] = today_cum / float(np.median(base)) if base else np.nan
        out["day_vol"] = today_cum

        reg = td[(tmins >= OPEN_M) & (tmins < CLOSE_M)]
        if len(reg):
            rm = tmins[(tmins >= OPEN_M) & (tmins < CLOSE_M)]
            orb = reg[rm < ORB_M]
            if len(orb):
                out["orb_high"] = float(orb["High"].max())
                out["orb_low"] = float(orb["Low"].min())
            tp = (reg["High"] + reg["Low"] + reg["Close"]) / 3
            cv = float(reg["Volume"].sum())
            out["vwap"] = float((tp * reg["Volume"]).sum() / cv) if cv > 0 else np.nan
            out["day_high"] = float(reg["High"].max())
            out["day_low"] = float(reg["Low"].min())
        return out
    except Exception:
        return None


def pool_run(fn, items, workers, label, prog, note):
    out, done, total = {}, 0, max(len(items), 1)
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fn, t): t for t in items}
            for f in as_completed(futs):
                t = futs[f]
                try:
                    out[t] = f.result()
                except Exception:
                    out[t] = None
                done += 1
                if done % 10 == 0 or done == total:
                    prog.progress(done / total)
                    note.caption(f"{label} · {done:,}/{total:,}")
    except RuntimeError:
        for t in items:
            if t not in out:
                try:
                    out[t] = fn(t)
                except Exception:
                    out[t] = None
    return {k: v for k, v in out.items() if v is not None}


# --------------------------------------------------------------- top bar
ny, sess_txt, sess_cls, sess_kind = session_now()
il_now = datetime.now(IL)
st.markdown(f"""<div class="bar">
<div class="brand"><h1>MAVRI · מסחר יומי</h1>
<em>גאפ · ווליום יחסי · תנודתיות</em></div>
<div class="status">
  <div><i>שוק ניו־יורק</i><b><span class="led {sess_cls}"></span>{sess_txt}</b></div>
  <div><i>ניו־יורק</i><b>{ny:%H:%M}</b></div>
  <div><i>ישראל</i><b>{il_now:%H:%M}</b></div>
</div></div>""", unsafe_allow_html=True)

MODES = {"pre": "פרה-מרקט — גאפ וכסף לפני הפתיחה",
         "regular": "תוך-יומי — תנועה וווליום עכשיו",
         "after": "אפטר-מרקט — סיכום היום",
         "closed": "שוק סגור — הנתונים האחרונים שיש"}
auto_mode = sess_kind if sess_kind in MODES else "closed"
mode = st.selectbox("מצב סריקה", list(MODES), index=list(MODES).index(auto_mode),
                    format_func=lambda k: MODES[k])
is_pre = mode == "pre"
st.markdown(
    '<div class="mode">' + MODES[mode] + ' · ' +
    ("הפילטר המרכזי: מחזור דולרי בפרה-מרקט."
     if is_pre else
     "הפילטר המרכזי: ווליום יחסי (RVOL) — הווליום מתחילת היום מול החציון "
     "של אותה שעה בדיוק בימי המסחר הקודמים.") + '</div>',
    unsafe_allow_html=True)

# --------------------------------------------------------------- controls
st.markdown("### סינון")
c1, c2, c3 = st.columns(3)
min_gap = c1.number_input("שינוי מינימלי מול אתמול %", 0.0, 30.0, 2.0, 0.25)
direction = c2.selectbox("כיוון", ["שני הכיוונים", "למעלה בלבד", "למטה בלבד"])
if is_pre:
    min_pre_dollar = c3.number_input("מחזור פרה-מרקט מינ׳ (מ׳ $)", 0.0, 100.0, 0.3, 0.1)
    min_rvol = 0.0
else:
    min_rvol = c3.number_input("RVOL מינימלי", 0.0, 20.0, 1.5, 0.1,
                               help="1.0 = ווליום רגיל לשעה הזאת. 2.0 = פי שניים.")
    min_pre_dollar = 0.0

c4, c5, c6 = st.columns(3)
min_atr_pct = c4.number_input("ATR יומי מינ׳ %", 0.0, 20.0, 2.0, 0.25)
min_px = c5.number_input("מחיר מינ׳ $", 0.5, 200.0, 3.0, 0.5)
max_px = c6.number_input("מחיר מקס׳ $", 5.0, 2000.0, 400.0, 5.0)

c7, c8, c9 = st.columns(3)
min_dv = c7.number_input("מחזור יומי ממוצע מינ׳ (מ׳ $)", 1.0, 2000.0, 20.0, 5.0)
max_stage2 = c8.number_input("כמה מניות לבדוק לעומק", 50, 700, 300, 25)
extra = c9.text_input("מניות נוספות (מופרדות ברווח)", "")

st.markdown("### תיק וסיכון")
c10, c11 = st.columns(2)
acct = c10.number_input("גודל תיק $", 500, 5_000_000, 25_000, 500)
riskp = c11.slider("סיכון לעסקה %", 0.25, 3.0, 1.0, 0.25)

b1, b2 = st.columns([3, 1])
go = b1.button("סרוק מניות למסחר יומי", type="primary", use_container_width=True)
fresh = b2.button("↻ סריקה טרייה", use_container_width=True)
if fresh:
    for k in ("_d1", "_d2"):
        st.session_state.pop(k, None)
    go = True

# --------------------------------------------------------------- scan
if go:
    t0 = time.time()
    uni = list(dict.fromkeys(UNIVERSE + [x.strip().upper()
                                         for x in extra.split() if x.strip()]))
    prog, note = st.progress(0.0), st.empty()

    box = st.session_state.get("_d1")
    if box is None or time.time() - box["t"] > 1800:
        box = {"t": time.time(), "d": {}}
    daily = box["d"]
    missing = [t for t in uni if t not in daily]
    if missing:
        daily.update(pool_run(_daily_one, missing, 8,
                              "שלב 1/2 · נתונים יומיים", prog, note))
        box["d"] = daily
        st.session_state["_d1"] = box

    drops, survivors = {}, []
    for t in uni:
        s = daily.get(t)
        if not s:
            drops["לא התקבלו נתונים"] = drops.get("לא התקבלו נתונים", 0) + 1
            continue
        if not (min_px <= s["prev_close"] <= max_px):
            drops["מחוץ לטווח המחיר"] = drops.get("מחוץ לטווח המחיר", 0) + 1
            continue
        if s["dollar_vol"] < min_dv * 1e6:
            drops["מחזור יומי נמוך"] = drops.get("מחזור יומי נמוך", 0) + 1
            continue
        if s["atr_pct"] < min_atr_pct:
            drops["תנודתיות נמוכה"] = drops.get("תנודתיות נמוכה", 0) + 1
            continue
        survivors.append(t)
    survivors.sort(key=lambda t: -daily[t]["dollar_vol"])
    survivors = survivors[:int(max_stage2)]

    box2 = st.session_state.get("_d2")
    if box2 is None or time.time() - box2["t"] > 120:
        box2 = {"t": time.time(), "d": {}}
    intra = box2["d"]
    need = [t for t in survivors if t not in intra]
    if need:
        intra.update(pool_run(_intraday_one, need, 8,
                              "שלב 2/2 · נתוני 5 דקות", prog, note))
        box2["d"] = intra
        st.session_state["_d2"] = box2

    rows = []
    for t in survivors:
        q, s = intra.get(t), daily.get(t)
        if not q or not s:
            continue
        gap, px, atr = q["gap_pct"], q["last"], s["atr"]
        rv = q.get("rvol")
        rvol = float(rv) if rv is not None and np.isfinite(rv) else 0.0
        pre_m = q["pre_dollar"] / 1e6
        long_side = gap >= 0

        misses = []
        if abs(gap) < min_gap:
            misses.append(f"שינוי {gap:+.1f}% מול {min_gap:.1f}% נדרש")
        if direction == "למעלה בלבד" and gap < 0:
            misses.append("כיוון הפוך")
        if direction == "למטה בלבד" and gap > 0:
            misses.append("כיוון הפוך")
        if is_pre and pre_m < min_pre_dollar:
            misses.append(f"פרה-מרקט ${pre_m:.2f}מ׳ מול ${min_pre_dollar:.2f}מ׳")
        if not is_pre and min_rvol > 0 and rvol < min_rvol:
            misses.append(f"RVOL {rvol:.2f} מול {min_rvol:.2f} נדרש")

        if not is_pre and np.isfinite(q.get("orb_high", np.nan)):
            hi, lo, src = q["orb_high"], q["orb_low"], "טווח 15 הדקות הראשונות"
        elif np.isfinite(q.get("pre_high", np.nan)):
            hi, lo, src = q["pre_high"], q["pre_low"], "טווח הפרה-מרקט"
        else:
            hi, lo, src = px + 0.5 * atr, px - 0.5 * atr, "ATR"

        if long_side:
            trig = hi
            risk = max(trig - lo, 0.25 * atr)
            stop = trig - risk
            t1, t2 = trig + 1.5 * risk, trig + 2.5 * risk
        else:
            trig = lo
            risk = max(hi - trig, 0.25 * atr)
            stop = trig + risk
            t1, t2 = trig - 1.5 * risk, trig - 2.5 * risk

        shares = int((acct * riskp / 100) / risk) if risk > 0 else 0
        score = abs(gap) * (max(pre_m, 0.05) if is_pre else max(rvol, 0.1))

        rows.append(dict(
            ticker=t, gap=gap, last=px, prev_close=q["prev_close"], rvol=rvol,
            pre_dollar_m=pre_m, atr_pct=s["atr_pct"], level_src=src,
            trig=trig, stop=stop, t1=t1, t2=t2,
            risk_pct=risk / px * 100 if px > 0 else 99, shares=max(shares, 0),
            long_side=long_side, vwap=q.get("vwap"),
            passed=not misses, misses=misses, n_miss=len(misses), score=score))

    passed = sorted([r for r in rows if r["passed"]], key=lambda r: -r["score"])
    near = sorted([r for r in rows if not r["passed"]],
                  key=lambda r: (r["n_miss"], -r["score"]))[:12]
    prog.empty()
    note.empty()
    st.session_state["day"] = dict(
        passed=passed, near=near, drops=drops, uni=len(uni), daily=len(daily),
        surv=len(survivors), checked=len(intra), took=round(time.time() - t0, 1),
        at=datetime.now(IL).strftime("%H:%M"), sess=sess_txt)

# --------------------------------------------------------------- results
if "day" not in st.session_state:
    st.markdown('<div class="empty">לחץ על סריקה.<br>'
                'לפני הפתיחה המערכת מחפשת גאפ עם כסף אמיתי בפרה-מרקט. '
                'אחרי הפתיחה — תנועה עם ווליום חריג לשעה הזאת.</div>',
                unsafe_allow_html=True)
    st.stop()

D = st.session_state["day"]
passed, near = D["passed"], D["near"]
st.markdown(f"""<div class="kpi">
<div><b>{D['uni']:,}</b><span>ביקום</span></div>
<div><b>{D['surv']:,}</b><span>עברו נזילות</span></div>
<div><b>{D['checked']:,}</b><span>נבדקו לעומק</span></div>
<div class="hi"><b>{len(passed)}</b><span>עברו סינון</span></div>
<div><b>{len(near)}</b><span>קרובות</span></div>
<div><b>{D['took']:.0f}s</b><span>{D['sess']}</span></div>
<div><b>{D['at']}</b><span>נסרק בשעה</span></div>
</div>""", unsafe_allow_html=True)


def draw(rows_, miss_mode=False):
    for start in range(0, len(rows_), 2):     # שניים בשורה — כרטיס רחב וקריא
        cols = st.columns(2, gap="medium")
        for col, r in zip(cols, rows_[start:start + 2]):
            cls = "miss" if miss_mode else ("up" if r["long_side"] else "dn")
            side_txt = ("לונג — פריצה מעל " if r["long_side"] else "שורט — שבירה מתחת ל") \
                + f"{r['trig']:.2f} · {r['level_src']}"
            chips = (f'<span class="chip g">RVOL {r["rvol"]:.2f}</span>'
                     f'<span class="chip">ATR {r["atr_pct"]:.1f}%</span>'
                     f'<span class="chip">${r["pre_dollar_m"]:.1f}מ׳ פרה-מרקט</span>')
            if r.get("vwap") and np.isfinite(r["vwap"]):
                rel = "מעל" if r["last"] >= r["vwap"] else "מתחת ל"
                chips += f'<span class="chip">{rel}-VWAP {r["vwap"]:.2f}</span>'
            for mss in r["misses"]:
                chips += f'<span class="chip r">{mss}</span>'
            with col:
                st.markdown(f"""<div class="dcard {cls}">
<div style="display:flex;align-items:baseline;justify-content:space-between">
  <div class="dsym">{r['ticker']}</div>
  <div class="dgap {'up' if r['long_side'] else 'dn'}">{r['gap']:+.2f}%</div>
</div>
<div class="side">{side_txt}</div>
<div>{chips}</div>
<div class="plan">
  מחיר <b>{r['last']:.2f}</b> · אתמול <b>{r['prev_close']:.2f}</b><br>
  טריגר <b>{r['trig']:.2f}</b> · סטופ <b>{r['stop']:.2f}</b> (<b>{r['risk_pct']:.2f}%</b>)<br>
  יעד 1 <b>{r['t1']:.2f}</b> · יעד 2 <b>{r['t2']:.2f}</b> · כמות <b>{r['shares']:,}</b>
</div>
<div style="margin-top:.5rem">
  <a class="tv" href="https://www.tradingview.com/symbols/{r['ticker']}/"
     target="_blank">פתח ב-TradingView ↗</a>
</div></div>""", unsafe_allow_html=True)


if passed:
    st.markdown("### עברו את הסינון")
    draw(passed)
else:
    st.markdown('<div class="note">אף מנייה לא עברה את הסף המלא. למטה מוצגות '
                'הקרובות ביותר, ולכל אחת רשום בדיוק מה חסר — כך אפשר להחליט '
                'אם להזיז סף או שזה פשוט יום שקט.</div>', unsafe_allow_html=True)

if near:
    st.markdown("### הקרובות ביותר — ומה בדיוק חסר")
    draw(near, miss_mode=True)

if D["drops"]:
    with st.expander("איפה נפסלו מניות בשלב הנזילות"):
        st.dataframe(pd.DataFrame(sorted(D["drops"].items(), key=lambda x: -x[1]),
                                  columns=["סיבה", "מניות"]),
                     hide_index=True, use_container_width=True)

allr = passed or near
if allr:
    st.markdown("---")
    z1, z2 = st.columns([1, 2])
    z1.download_button("הורד CSV", pd.DataFrame(allr).to_csv(index=False),
                       f"mavri_day_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv",
                       use_container_width=True)
    z2.code(",".join(r["ticker"] for r in allr), language=None)
    st.caption("הרשימה מוכנה להדבקה כ-Watchlist ב-TradingView. נתוני Yahoo "
               "אינם כוללים חדשות, דוחות או Float — לבדוק ידנית לפני כניסה.")
