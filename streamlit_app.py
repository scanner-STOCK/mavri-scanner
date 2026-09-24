"""
MAVRI — סורק מסחר יומי (פרה-מרקט + תוך-יומי).

עמוד נפרד לחלוטין מהסורק הסווינגי. הלוגיקה שונה מיסודה:
סורק סווינג שואל "איזו מניה בנתה תבנית בשבועות האחרונים".
סורק יומי שואל "איזו מניה זזה עכשיו, בנפח חריג, עם מרווח תנועה".

שני שלבים:
  שלב 1 (זול)  — נר יומי לכל מניה ביקום: מחיר, מחזור ממוצע, ATR%.
                 מסנן מניות ישנוניות, יקרות מדי או לא נזילות.
  שלב 2 (יקר)  — נרות 5 דקות עם prepost=True רק לניצולים:
                 גאפ מול הסגירה של אתמול, ווליום פרה-מרקט,
                 שיא/שפל פרה-מרקט, VWAP אחרי הפתיחה.

התקנה: לשים את הקובץ בתיקייה בשם pages/ ליד app.py. Streamlit יזהה
אותו אוטומטית כעמוד נוסף. שום שינוי ב-app.py.
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

st.set_page_config(page_title="MAVRI — מסחר יומי", layout="wide",
                   initial_sidebar_state="collapsed", page_icon="◈")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Frank+Ruhl+Libre:wght@400;500;700&family=Heebo:wght@300;400;500;700&display=swap');
:root{--bg:#0B1216;--pan:#111C22;--ln:#22333C;--tx:#E9EFF1;--mu:#93A8B2;
      --dm:#5F7581;--gd:#D4A64B;--lg:#2FBF8F;--sh:#E0605F;}
html,body,[class*="css"]{font-family:'Heebo',sans-serif;}
.stApp{background:var(--bg);} #MainMenu,footer{visibility:hidden;}
.block-container{padding-top:.7rem;max-width:1320px;}
.bar{display:flex;align-items:center;justify-content:space-between;gap:1rem;
     border-bottom:1px solid var(--ln);padding:.1rem 0 .55rem;flex-wrap:wrap;}
.brand h1{font-family:'Frank Ruhl Libre',serif;font-size:1.45rem;margin:0;
          color:var(--tx);letter-spacing:.12em;}
.brand em{font-style:normal;font-size:.72rem;color:var(--dm);}
.status{display:flex;gap:1.3rem;direction:rtl;}
.status i{font-style:normal;font-size:.68rem;color:var(--dm);display:block;}
.status b{font-family:'Frank Ruhl Libre',serif;font-size:.95rem;color:var(--mu);
          font-variant-numeric:tabular-nums;}
.led{width:7px;height:7px;border-radius:50%;display:inline-block;margin-left:.35rem;}
.led.on{background:var(--lg);box-shadow:0 0 7px rgba(47,191,143,.8);}
.led.pre{background:var(--gd);box-shadow:0 0 7px rgba(212,166,75,.7);}
.led.off{background:var(--dm);}
.kpi{display:flex;border:1px solid var(--ln);border-radius:4px;overflow:hidden;margin:.9rem 0;}
.kpi div{flex:1;min-width:92px;padding:.55rem .8rem;border-left:1px solid var(--ln);direction:rtl;}
.kpi div:last-child{border-left:0;}
.kpi b{font-family:'Frank Ruhl Libre',serif;font-size:1.3rem;display:block;color:var(--tx);
       font-variant-numeric:tabular-nums;}
.kpi span{font-size:.66rem;color:var(--dm);}
.kpi div.hi b{color:var(--gd);}
.dcard{background:var(--pan);border:1px solid var(--ln);border-radius:5px;
       padding:.8rem .9rem;direction:rtl;height:100%;}
.dcard.up{border-right:3px solid var(--lg);} .dcard.dn{border-right:3px solid var(--sh);}
.dsym{font-family:'Frank Ruhl Libre',serif;font-size:1.35rem;font-weight:700;
      color:var(--tx);direction:ltr;}
.dgap{font-family:'Frank Ruhl Libre',serif;font-size:1.25rem;font-variant-numeric:tabular-nums;}
.dgap.up{color:var(--lg);} .dgap.dn{color:var(--sh);}
.chip{font-size:.62rem;padding:.11rem .4rem;border:1px solid var(--ln);border-radius:2px;
      color:var(--mu);margin-left:.25rem;}
.chip.g{border-color:rgba(212,166,75,.45);color:var(--gd);}
.plan{font-size:.74rem;line-height:1.75;color:var(--dm);border-top:1px solid var(--ln);
      margin-top:.5rem;padding-top:.4rem;}
.plan b{color:var(--mu);font-weight:500;font-variant-numeric:tabular-nums;}
.empty{border:1px dashed var(--ln);border-radius:5px;padding:2.2rem 1.2rem;text-align:center;
       color:var(--dm);direction:rtl;line-height:1.9;font-size:.87rem;}
h3,h5{font-family:'Frank Ruhl Libre',serif !important;color:var(--tx) !important;direction:rtl;}
label{color:var(--mu) !important;font-size:.75rem !important;}
a.tv{font-size:.68rem;color:var(--dm);text-decoration:none;border-bottom:1px dotted var(--ln);}
a.tv:hover{color:var(--gd);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------------- universe
# מניות שנסחרות בנפח גבוה ושזזות מספיק כדי להיסחר ביום. הרשימה קצרה בכוונה:
# בפרה-מרקט לכל מניה נדרשת קריאת רשת נפרדת, ולכן יקום של 6,000 מניות פשוט
# לא ריאלי מול Yahoo. הסריקה היומית לא צריכה את כל השוק — היא צריכה את
# המניות שבפועל אפשר להיכנס ולצאת מהן בשניות.
UNIVERSE = """
AAPL MSFT NVDA GOOGL AMZN META TSLA AVGO AMD INTC QCOM MU MRVL SMCI ARM TSM ASML
ON MPWR SWKS QRVO NXPI ALAB CRDO LRCX KLAC AMAT ADI TXN DELL HPQ STX WDC
CRM ORCL ADBE NOW SNOW DDOG NET CRWD ZS OKTA PANW FTNT S PLTR MDB TEAM ESTC GTLB
PATH AI BBAI SOUN TTD APP U RBLX EA TTWO RDDT SNAP PINS BMBL DUOL IONQ RGTI QBTS QUBT
SHOP SQ PYPL COIN HOOD SOFI AFRM UPST LC ALLY COF SYF AXP V MA TOST NU
JPM BAC WFC C GS MS SCHW USB PNC RF KEY CFG HBAN FITB
UNH ELV CI CVS HUM CNC MOH LLY PFE MRK ABBV BMY AMGN GILD BIIB VRTX REGN
MRNA BNTX NVAX ILMN TMO SYK BSX MDT ABT ISRG EW HIMS OSCR
CRSP NTLA BEAM RXRX SRPT NBIX EXAS NTRA TEM PACB HALO
XOM CVX COP EOG DVN FANG OXY HES SLB HAL BKR RIG VAL
VLO MPC PSX KMI WMB OKE ET LNG AR RRC SWN EQT CHRD
NEE DUK SO D AEP EXC XEL VST CEG NRG TLN
CAT DE HON GE MMM EMR ETN PH ROK DOV GEV PWR
BA LMT RTX NOC GD LHX AXON RKLB LUNR ASTS
UPS FDX UNP CSX NSC ODFL XPO
DAL UAL AAL LUV JBLU
WMT COST TGT DG DLTR KR
HD LOW TSCO ORLY AZO BBY RH CHWY W ETSY EBAY
NKE LULU DECK ONON SKX CROX VFC PVH RL TPR GPS ANF AEO URBN BIRK
SBUX MCD YUM CMG QSR DPZ WEN SHAK CAVA DRI TXRH WING CELH
PG KO PEP KDP MNST KHC GIS STZ TAP MO PM
DIS NFLX CMCSA WBD PARA LYV SPOT ROKU T VZ TMUS
LIN APD SHW ECL DD DOW LYB NUE STLD X CLF AA FCX SCCO NEM GOLD AEM
RIO VALE MP ALB UEC CCJ LEU OKLO SMR NNE
UBER LYFT DASH ABNB BKNG EXPE MAR HLT CZR MGM LVS WYNN PENN DKNG RCL CCL NCLH
F GM RIVN LCID NIO XPEV LI STLA BLNK CHPT
MARA RIOT CLSK HUT BITF WULF CIFR IREN CORZ MSTR BTDR GLXY
ENPH FSLR RUN NOVA ARRY SHLS CSIQ PLUG FCEL BE
LAZR OUST INDI AEVA MVIS
ANET CSCO JNPR FFIV CIEN LITE COHR VRT
IBM ACN ZM DOCU TWLO FSLY AKAM
GME AMC BYND TLRY ACB CGC SNDL OPEN CLOV
BABA JD PDD BIDU NTES TME BILI
SE MELI GRAB CPNG STNE PAGS
SPY QQQ IWM
""".split()


def session_now():
    ny = datetime.now(NY)
    m = ny.hour * 60 + ny.minute
    if ny.weekday() >= 5:
        return ny, "סגור — סוף שבוע", "off", "closed"
    if 4 * 60 <= m < 9 * 60 + 30:
        return ny, "פרה-מרקט", "pre", "pre"
    if 9 * 60 + 30 <= m < 16 * 60:
        return ny, "פתוח", "on", "regular"
    if 16 * 60 <= m < 20 * 60:
        return ny, "אפטר-מרקט", "pre", "after"
    return ny, "סגור", "off", "closed"


# --------------------------------------------------------------- stage 1
def _daily_one(t):
    """נר יומי: מחיר, מחזור דולרי ממוצע, ATR ו-ATR%. בלי פרה-מרקט."""
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
        today = datetime.now(NY).date()
        # אם הנר האחרון הוא של היום, הוא חלקי — לא סופרים אותו בממוצעים.
        if d.index[-1].date() == today:
            d = d.iloc[:-1]
        if len(d) < 15:
            return None
        c = d["Close"].astype(float)
        h = d["High"].astype(float)
        l = d["Low"].astype(float)
        v = d["Volume"].astype(float)
        pc = c.shift(1)
        tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        atr = float(tr.tail(14).mean())
        px = float(c.iloc[-1])
        avg_vol = float(v.tail(20).mean())
        if px <= 0 or avg_vol <= 0 or not np.isfinite(atr):
            return None
        return dict(prev_close=px, atr=atr, atr_pct=atr / px * 100,
                    avg_vol=avg_vol, dollar_vol=px * avg_vol)
    except Exception:
        return None


# --------------------------------------------------------------- stage 2
def _intraday_one(t):
    """נרות 5 דקות כולל פרה-מרקט: גאפ, ווליום מוקדם, שיא/שפל, VWAP."""
    try:
        d = yf.Ticker(t).history(period="2d", interval="5m", prepost=True,
                                 auto_adjust=False, actions=False)
    except Exception:
        return None
    if d is None or len(d) < 5:
        return None
    try:
        idx = d.index
        if getattr(idx, "tz", None) is not None:
            d.index = idx.tz_convert(NY)
        d = d.dropna()
        if len(d) < 5:
            return None
        mins = np.array([x.hour * 60 + x.minute for x in d.index])
        days = np.array([x.date() for x in d.index])
        today = days[-1]
        is_reg = (mins >= 570) & (mins < 960)

        prev = d[(days < today) & is_reg]
        if len(prev) == 0:
            return None
        prev_close = float(prev["Close"].iloc[-1])
        if prev_close <= 0:
            return None

        td = d[days == today]
        td_mins = mins[days == today]
        pre = td[td_mins < 570]
        reg = td[(td_mins >= 570) & (td_mins < 960)]

        last = float(td["Close"].iloc[-1])
        out = dict(last=last, prev_close=prev_close,
                   gap_pct=(last / prev_close - 1) * 100,
                   pre_vol=float(pre["Volume"].sum()) if len(pre) else 0.0,
                   pre_high=float(pre["High"].max()) if len(pre) else np.nan,
                   pre_low=float(pre["Low"].min()) if len(pre) else np.nan,
                   bars=int(len(td)))
        out["pre_dollar"] = out["pre_vol"] * last
        if len(reg):
            tp = (reg["High"] + reg["Low"] + reg["Close"]) / 3
            cv = reg["Volume"].cumsum()
            out.update(open_px=float(reg["Open"].iloc[0]),
                       day_high=float(reg["High"].max()),
                       day_low=float(reg["Low"].min()),
                       reg_vol=float(reg["Volume"].sum()),
                       vwap=float((tp * reg["Volume"]).cumsum().iloc[-1] /
                                  cv.iloc[-1]) if float(cv.iloc[-1]) > 0 else np.nan)
        return out
    except Exception:
        return None


def pool_run(fn, items, workers, label, prog, note):
    """מריץ fn על רשימה בפול חוטים אחד, עם התקדמות. נופל לריצה טורית
    אם התהליך לא מצליח לפתוח חוטים (קורה ב-Streamlit Cloud כשעמוס)."""
    out, done = {}, 0
    total = max(len(items), 1)
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
            if t in out:
                continue
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
<em>גאפ · ווליום פרה-מרקט · תנודתיות</em></div>
<div class="status">
  <div><i>שוק ניו־יורק</i><b><span class="led {sess_cls}"></span>{sess_txt}</b></div>
  <div><i>ניו־יורק</i><b>{ny:%H:%M}</b></div>
  <div><i>ישראל</i><b>{il_now:%H:%M}</b></div>
</div></div>""", unsafe_allow_html=True)

if sess_kind == "closed":
    st.info("השוק סגור. הסריקה תרוץ על הנתונים האחרונים שיש, אבל הזמן שבו "
            "היא שווה משהו הוא 8:00–9:30 בניו־יורק, כלומר 15:00–16:30 בישראל.")

# --------------------------------------------------------------- controls
c1, c2, c3, c4 = st.columns(4)
min_gap = c1.number_input("גאפ מינימלי %", 0.0, 30.0, 3.0, 0.5,
                          help="שינוי מול סגירת אתמול. בלי גאפ אין סיבה שהמניה "
                               "תזוז יותר מהרגיל היום.")
direction = c2.selectbox("כיוון", ["שני הכיוונים", "למעלה בלבד", "למטה בלבד"])
min_pre_dollar = c3.number_input("מחזור פרה-מרקט מינ׳ (מ׳ $)", 0.0, 100.0, 1.0, 0.5,
                                 help="כמה כסף כבר עבר לפני הפתיחה. זה המדד "
                                      "האמיתי לשאלה אם מישהו באמת מתעניין.")
min_atr_pct = c4.number_input("ATR יומי מינ׳ %", 0.0, 20.0, 2.5, 0.25,
                              help="מרחב תנועה. מניה שזזה 1% ביום לא תיתן "
                                   "מספיק מרחק בין סטופ ליעד.")

c5, c6, c7, c8 = st.columns(4)
min_px = c5.number_input("מחיר מינ׳ $", 0.5, 200.0, 3.0, 0.5)
max_px = c6.number_input("מחיר מקס׳ $", 5.0, 2000.0, 400.0, 5.0)
min_dv = c7.number_input("מחזור יומי ממוצע מינ׳ (מ׳ $)", 1.0, 2000.0, 30.0, 5.0,
                         help="נזילות. מתחת ל-30 מיליון דולר ביום המרווחים "
                              "מתרחבים ויציאה מהירה עולה כסף.")
max_stage2 = c8.number_input("כמה מניות לבדוק בפרה-מרקט", 50, 600, 250, 25,
                             help="כל מנייה כאן היא קריאת רשת נפרדת ל-Yahoo. "
                                  "250 לוקח כדקה. מעל 400 מתחילים לקבל חסימות.")

r1, r2, r3 = st.columns([1, 1, 2])
acct = r1.number_input("גודל תיק $", 500, 5_000_000, 25_000, 500)
riskp = r2.slider("סיכון לעסקה %", 0.25, 3.0, 1.0, 0.25)
r3.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
b1, b2 = r3.columns([2, 1])
go = b1.button("סרוק מניות למסחר יומי", type="primary", use_container_width=True)
fresh = b2.button("↻", use_container_width=True, help="מתעלם מהנתונים השמורים")

extra = st.text_input("מניות נוספות לבדיקה (מופרדות ברווח)", "",
                      help="סימולים שלא ברשימה הקבועה — למשל מניה שראית בחדשות הבוקר.")

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

    # שלב 1 — נתונים יומיים. נשמרים לחצי שעה: מה שמנייה עשתה אתמול
    # לא משתנה כשמזיזים סליידר.
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

    survivors = []
    for t in uni:
        s = daily.get(t)
        if not s:
            continue
        if s["prev_close"] < min_px or s["prev_close"] > max_px:
            continue
        if s["dollar_vol"] < min_dv * 1e6:
            continue
        if s["atr_pct"] < min_atr_pct:
            continue
        survivors.append(t)
    survivors.sort(key=lambda t: -daily[t]["dollar_vol"])
    survivors = survivors[:int(max_stage2)]

    # שלב 2 — פרה-מרקט. נשמר לשתי דקות בלבד: כאן כל דקה משנה.
    box2 = st.session_state.get("_d2")
    if box2 is None or time.time() - box2["t"] > 120:
        box2 = {"t": time.time(), "d": {}}
    intra = box2["d"]
    need = [t for t in survivors if t not in intra]
    if need:
        intra.update(pool_run(_intraday_one, need, 8,
                              "שלב 2/2 · פרה-מרקט", prog, note))
        box2["d"] = intra
        st.session_state["_d2"] = box2

    rows = []
    for t in survivors:
        q = intra.get(t)
        s = daily.get(t)
        if not q or not s:
            continue
        gap = q["gap_pct"]
        if direction == "למעלה בלבד" and gap < 0:
            continue
        if direction == "למטה בלבד" and gap > 0:
            continue
        if abs(gap) < min_gap:
            continue
        if q["pre_dollar"] < min_pre_dollar * 1e6:
            continue

        px = q["last"]
        atr = s["atr"]
        long_side = gap > 0
        # טריגר ברמה שהמחיר כבר הוכיח שהיא משמעותית לפני הפתיחה,
        # וסטופ בצד השני של אותו טווח. אם הטווח צר מדי, ATR ממלא את החסר.
        if long_side:
            trig = q["pre_high"] if np.isfinite(q.get("pre_high", np.nan)) else px
            stop = q["pre_low"] if np.isfinite(q.get("pre_low", np.nan)) else px - 0.5 * atr
            risk = max(trig - stop, 0.25 * atr)
            stop = trig - risk
            t1, t2 = trig + 1.5 * risk, trig + 2.5 * risk
        else:
            trig = q["pre_low"] if np.isfinite(q.get("pre_low", np.nan)) else px
            stop = q["pre_high"] if np.isfinite(q.get("pre_high", np.nan)) else px + 0.5 * atr
            risk = max(stop - trig, 0.25 * atr)
            stop = trig + risk
            t1, t2 = trig - 1.5 * risk, trig - 2.5 * risk

        risk_pct = risk / px * 100 if px > 0 else 99
        shares = int((acct * riskp / 100) / risk) if risk > 0 else 0
        pre_rvol = q["pre_vol"] / s["avg_vol"] * 100 if s["avg_vol"] > 0 else 0

        rows.append(dict(
            ticker=t, gap=gap, last=px, prev_close=q["prev_close"],
            pre_dollar_m=q["pre_dollar"] / 1e6, pre_rvol=pre_rvol,
            atr=atr, atr_pct=s["atr_pct"], dollar_vol_m=s["dollar_vol"] / 1e6,
            trig=trig, stop=stop, t1=t1, t2=t2, risk=risk, risk_pct=risk_pct,
            shares=max(shares, 0), long_side=long_side,
            vwap=q.get("vwap"), day_high=q.get("day_high"), day_low=q.get("day_low"),
            # דירוג: כסף שכבר עבר לפני הפתיחה, משוקלל בגודל הגאפ.
            score=(q["pre_dollar"] / 1e6) * min(abs(gap), 25)))

    rows.sort(key=lambda r: -r["score"])
    prog.empty()
    note.empty()
    st.session_state["day_rows"] = rows
    st.session_state["day_meta"] = dict(
        uni=len(uni), daily=len(daily), surv=len(survivors), checked=len(intra),
        took=round(time.time() - t0, 1),
        at=datetime.now(IL).strftime("%H:%M"), sess=sess_txt)

# --------------------------------------------------------------- results
if "day_rows" not in st.session_state:
    st.markdown('<div class="empty">לחץ על סריקה.<br>'
                'המערכת מחפשת מניות עם גאפ, ווליום פרה-מרקט אמיתי ומרווח תנועה — '
                'שלושת הדברים שקובעים אם אפשר לסחור במנייה היום.<br><br>'
                'הזמן הנכון להריץ: 15:00–16:30 שעון ישראל.</div>',
                unsafe_allow_html=True)
    st.stop()

rows = st.session_state["day_rows"]
m = st.session_state["day_meta"]
st.markdown(f"""<div class="kpi">
<div><b>{m['uni']:,}</b><span>ביקום</span></div>
<div><b>{m['daily']:,}</b><span>נתונים יומיים</span></div>
<div><b>{m['surv']:,}</b><span>עברו נזילות ותנודתיות</span></div>
<div><b>{m['checked']:,}</b><span>נבדקו בפרה-מרקט</span></div>
<div class="hi"><b>{len(rows)}</b><span>מועמדות</span></div>
<div><b>{m['took']:.0f}s</b><span>{m['sess']}</span></div>
<div><b>{m['at']}</b><span>נסרק בשעה</span></div>
</div>""", unsafe_allow_html=True)

if not rows:
    st.markdown('<div class="empty">אף מנייה לא עברה את הסינון.<br>'
                'בימים רגילים זה נורמלי לפני 8:00 בניו־יורק — הווליום בפרה-מרקט '
                'עוד לא הצטבר.<br>נסה להוריד את "גאפ מינימלי" ל-2% ואת '
                '"מחזור פרה-מרקט" ל-0.3.</div>', unsafe_allow_html=True)
    st.stop()

view = st.radio("תצוגה", ["כרטיסים", "טבלה"], horizontal=True, label_visibility="collapsed")

if view == "טבלה":
    df = pd.DataFrame([{
        "מניה": r["ticker"], "כיוון": "לונג" if r["long_side"] else "שורט",
        "גאפ %": round(r["gap"], 2), "מחיר": round(r["last"], 2),
        "$מ׳ פרה-מרקט": round(r["pre_dollar_m"], 1),
        "% מיום ממוצע": round(r["pre_rvol"], 1),
        "ATR %": round(r["atr_pct"], 1),
        "טריגר": round(r["trig"], 2), "סטופ": round(r["stop"], 2),
        "יעד 1": round(r["t1"], 2), "יעד 2": round(r["t2"], 2),
        "סיכון %": round(r["risk_pct"], 2), "כמות": r["shares"],
        "מחזור יומי מ׳$": round(r["dollar_vol_m"], 0)} for r in rows])
    st.dataframe(df, hide_index=True, use_container_width=True, height=520)
else:
    for start in range(0, len(rows), 3):
        cols = st.columns(3, gap="medium")
        for col, r in zip(cols, rows[start:start + 3]):
            side_cls = "up" if r["long_side"] else "dn"
            side_txt = "לונג — פריצת שיא פרה-מרקט" if r["long_side"] \
                else "שורט — שבירת שפל פרה-מרקט"
            vwap_txt = ""
            if r.get("vwap") and np.isfinite(r["vwap"]):
                rel = "מעל" if r["last"] >= r["vwap"] else "מתחת ל"
                vwap_txt = f'<span class="chip">{rel}-VWAP {r["vwap"]:.2f}</span>'
            with col:
                st.markdown(f"""<div class="dcard {side_cls}">
<div style="display:flex;align-items:baseline;justify-content:space-between">
  <div class="dsym">{r['ticker']}</div>
  <div class="dgap {side_cls}">{r['gap']:+.2f}%</div>
</div>
<div style="font-size:.72rem;color:var(--dm);margin:.15rem 0 .4rem">{side_txt}</div>
<div>
  <span class="chip g">${r['pre_dollar_m']:.1f}מ׳ פרה-מרקט</span>
  <span class="chip">{r['pre_rvol']:.1f}% מיום ממוצע</span>
  <span class="chip">ATR {r['atr_pct']:.1f}%</span>
  {vwap_txt}
</div>
<div class="plan">
  מחיר עכשיו <b>{r['last']:.2f}</b> · אתמול נסגרה ב-<b>{r['prev_close']:.2f}</b><br>
  טריגר <b>{r['trig']:.2f}</b> · סטופ <b>{r['stop']:.2f}</b>
  (<b>{r['risk_pct']:.2f}%</b>)<br>
  יעד 1 <b>{r['t1']:.2f}</b> (1.5R) · יעד 2 <b>{r['t2']:.2f}</b> (2.5R)<br>
  כמות לפי הסיכון שהגדרת: <b>{r['shares']:,}</b> מניות
</div>
<div style="margin-top:.45rem">
  <a class="tv" href="https://www.tradingview.com/symbols/{r['ticker']}/"
     target="_blank">פתח ב-TradingView ↗</a>
</div>
</div>""", unsafe_allow_html=True)

st.markdown("---")
z1, z2 = st.columns([1, 2])
z1.download_button("הורד CSV",
                   pd.DataFrame(rows).to_csv(index=False),
                   f"mavri_day_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv",
                   use_container_width=True)
z2.code(",".join(r["ticker"] for r in rows), language=None)
st.caption("הרשימה למעלה מוכנה להדבקה כ-Watchlist ב-TradingView. "
           "הטריגר והסטופ מחושבים מטווח הפרה-מרקט — הם משתנים כל עוד השוק "
           "לא נפתח, ולכן כדאי לרענן סמוך ל-9:30 בניו־יורק. "
           "נתוני הפרה-מרקט מגיעים מ-Yahoo והם חלקיים: אין כאן חדשות, "
           "דוחות או Float, ואת אלה צריך לבדוק ידנית לפני כניסה.")
