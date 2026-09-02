"""
MAVRI — pullback scanner.
Impulse leg on volume -> pullback on drying volume -> entry at support.
Filters were set from a walk-forward backtest; volume dry-up >= 1.3 was the
only parameter that held up across periods.
"""

import io
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(page_title="MAVRI", layout="wide",
                   initial_sidebar_state="collapsed", page_icon="◈")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Frank+Ruhl+Libre:wght@400;500;700&family=Heebo:wght@300;400;500;700&display=swap');
:root{--paper:#F7F8FA;--card:#FFF;--ink:#16202B;--soft:#5A6875;--faint:#93A0AC;
      --rule:#E2E7ED;--long:#0E7C5A;--short:#B4293B;}
html,body,[class*="css"]{font-family:'Heebo',sans-serif;}
.stApp{background:var(--paper);}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1rem;padding-bottom:3rem;max-width:1180px;}
.mast{display:flex;align-items:baseline;justify-content:space-between;
      border-bottom:2px solid var(--ink);padding-bottom:.5rem;}
.mast h1{font-family:'Frank Ruhl Libre',serif;font-weight:700;font-size:2rem;margin:0;
         letter-spacing:-.01em;color:var(--ink);}
.mast .sub{font-size:.85rem;color:var(--soft);}
.mast .stamp{font-size:.78rem;color:var(--faint);font-variant-numeric:tabular-nums;}
.counts{display:flex;gap:2.2rem;flex-wrap:wrap;padding:.7rem 0 1rem;
        border-bottom:1px solid var(--rule);margin-bottom:1.2rem;}
.counts div{direction:rtl;}
.counts b{font-family:'Frank Ruhl Libre',serif;font-size:1.45rem;font-weight:500;display:block;
          line-height:1.1;font-variant-numeric:tabular-nums;color:var(--ink);}
.counts span{font-size:.75rem;color:var(--faint);}
.tc{background:var(--card);border:1px solid var(--rule);border-radius:3px;
    padding:1rem 1.1rem .9rem;direction:rtl;}
.tc.top{border-right:3px solid var(--long);}
.tc-head{display:flex;align-items:baseline;justify-content:space-between;
         border-bottom:1px solid var(--rule);padding-bottom:.5rem;margin-bottom:.8rem;}
.tc-sym{font-family:'Frank Ruhl Libre',serif;font-size:1.6rem;font-weight:700;
        color:var(--ink);direction:ltr;}
.tc-dry{text-align:left;direction:ltr;}
.tc-dry b{font-family:'Frank Ruhl Libre',serif;font-size:1.25rem;font-weight:500;
          color:var(--long);font-variant-numeric:tabular-nums;}
.tc-dry span{display:block;font-size:.67rem;color:var(--faint);direction:rtl;}
.lad{position:relative;height:186px;margin:.1rem 0 .8rem;}
.lad:before{content:'';position:absolute;right:78px;top:6px;bottom:6px;width:1px;background:var(--rule);}
.lv{position:absolute;right:0;left:0;height:0;display:flex;align-items:center;gap:.5rem;}
.lv .tag{width:74px;text-align:left;font-size:.71rem;color:var(--soft);}
.lv .dot{width:7px;height:7px;border-radius:50%;background:var(--faint);margin-right:-3.5px;flex:none;}
.lv .px{font-family:'Frank Ruhl Libre',serif;font-size:.96rem;color:var(--ink);
        font-variant-numeric:tabular-nums;direction:ltr;}
.lv.tp .dot{background:var(--long);} .lv.tp .px{color:var(--long);}
.lv.st .dot{background:var(--short);} .lv.st .px,.lv.st .tag{color:var(--short);}
.lv.en .dot{width:11px;height:11px;background:var(--ink);margin-right:-5.5px;}
.lv.en .px{font-size:1.18rem;font-weight:700;} .lv.en .tag{color:var(--ink);font-weight:500;}
.lv.now .dot{background:transparent;border:1px solid var(--faint);}
.lv.now .px,.lv.now .tag{color:var(--faint);font-size:.8rem;}
.tc-line{display:flex;justify-content:space-between;font-size:.77rem;color:var(--soft);
         border-top:1px solid var(--rule);padding-top:.55rem;}
.tc-line b{color:var(--ink);font-weight:500;font-variant-numeric:tabular-nums;}
.tc-why{font-size:.78rem;line-height:1.55;color:var(--soft);margin-top:.5rem;}
.stButton>button{background:var(--ink);color:#fff;border:0;border-radius:2px;
                 font-family:'Heebo';font-weight:500;padding:.6rem 1rem;width:100%;}
.stButton>button:hover{background:#2B3A4A;color:#fff;}
.empty{border:1px dashed var(--rule);padding:2rem 1.3rem;text-align:center;
       color:var(--soft);direction:rtl;line-height:1.7;font-size:.9rem;}
h3{font-family:'Frank Ruhl Libre',serif !important;font-weight:500 !important;
   color:var(--ink);direction:rtl;}
@media(max-width:640px){.block-container{padding-left:.7rem;padding-right:.7rem;}
  .mast h1{font-size:1.5rem;} .counts{gap:1.2rem;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"}

BACKUP = """AAPL MSFT NVDA GOOGL GOOG AMZN META TSLA AVGO AMD INTC QCOM MU LRCX KLAC AMAT ADI TXN
ON MPWR SWKS QRVO MRVL NXPI ASML TSM ARM SMCI DELL HPQ HPE NTAP STX WDC ALAB CRDO
CRM ORCL ADBE NOW SNOW DDOG NET CRWD ZS OKTA PANW FTNT S PLTR MDB TEAM WDAY ESTC GTLB
CFLT PATH AI BBAI SOUN TTD APP U RBLX EA TTWO RDDT SNAP PINS BMBL MTCH DUOL
SHOP SQ PYPL COIN HOOD SOFI AFRM UPST LC ALLY COF DFS SYF AXP V MA TOST NU
JPM BAC WFC C GS MS SCHW BK STT TFC USB PNC RF KEY CFG HBAN FITB ZION CMA
BX KKR APO ARES TPG OWL BLK STEP BDC WKC ACIW ABG LAD FTDR AWI BCC ETN ACHC SHW CSW TXRH
UNH ELV CI CVS HUM CNC MOH LLY PFE MRK ABBV BMY AMGN GILD BIIB VRTX REGN ALNY
MRNA BNTX NVAX ILMN TMO DHR SYK BSX MDT ABT ISRG EW ZBH BAX BDX HIMS OSCR MRVI
CRSP NTLA BEAM EDIT VERV SANA RXRX IONS SRPT NBIX EXAS NTRA TEM TWST PACB HALO HAE VEEV
XOM CVX COP EOG DVN FANG OXY HES MRO APA SLB HAL BKR NOV FTI RIG VAL
VLO MPC PSX DINO PBF DK CVI KMI WMB OKE ET EPD MPLX TRGP LNG AR RRC SWN EQT CHRD
NEE DUK SO D AEP EXC XEL ED PEG WEC ES AEE CMS DTE PPL FE VST CEG NRG TLN
CAT DE HON GE MMM EMR ETN PH ROK DOV ITW PNR XYL IEX FLS GEV PWR
BA LMT RTX NOC GD LHX TDG HWM HEI TXT SPR AXON RKLB LUNR ASTS PL RDW
UPS FDX UNP CSX NSC ODFL JBHT XPO CHRW EXPD LSTR SAIA
DAL UAL AAL LUV ALK JBLU SAVE ALGT
WMT COST TGT DG DLTR BJ KR ACI SFM CASY
HD LOW TSCO ORLY AZO AAP GPC BBY WSM RH CHWY W ETSY EBAY
NKE LULU DECK ONON SKX CROX VFC PVH RL TPR CPRI GPS ANF AEO URBN BIRK
SBUX MCD YUM CMG QSR DPZ WEN JACK SHAK CAVA DRI EAT TXRH BLMN WING SG CELH
PG KO PEP KDP MNST KHC GIS K CPB CAG SJM HSY MDLZ STZ TAP MO PM
DIS NFLX CMCSA WBD PARA FOX LYV SPOT ROKU FUBO T VZ TMUS
AMT CCI SBAC EQIX DLR PLD SPG O VICI WELL VTR ARE BXP KIM REG FRT IRM
LIN APD SHW ECL DD DOW LYB PPG NUE STLD X CLF AA FCX SCCO NEM GOLD AEM
RIO BHP VALE MP ALB SQM UEC CCJ LEU OKLO SMR NNE
UBER LYFT DASH ABNB BKNG EXPE MAR HLT H WH CZR MGM LVS WYNN PENN DKNG RCL CCL NCLH
F GM RIVN LCID NIO XPEV LI STLA HMC TM BLNK CHPT
MARA RIOT CLSK HUT BITF WULF CIFR IREN CORZ MSTR GLXY BTDR
ENPH SEDG FSLR RUN NOVA ARRY SHLS CSIQ JKS DQ PLUG FCEL BE BLDP
LAZR OUST INDI AEVA MVIS IONQ RGTI QBTS QUBT
ANET CSCO JNPR FFIV CIEN LITE INFN VIAV COHR VRT FIVN
IBM ACN CTSH INFY WIT EPAM DXC ZM DOCU TWLO FSLY AKAM
GME AMC KOSS BYND TLRY ACB CGC SNDL OPEN WISH CLOV
NVO AZN SNY GSK NVS TAK BABA JD PDD BIDU NTES TME BILI IQ TCOM YMM
SE MELI GRAB CPNG STNE PAGS VIST GGAL BMA PAM YPF TS TX
""".split()


@st.cache_data(ttl=86400, show_spinner=False)
def build_universe(limit):
    got = []
    for url, cols in [
        ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", ("Symbol", "Ticker")),
        ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", ("Symbol", "Ticker")),
        ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", ("Symbol", "Ticker")),
        ("https://en.wikipedia.org/wiki/Nasdaq-100", ("Ticker", "Symbol"))]:
        try:
            html = requests.get(url, headers=UA, timeout=25).text
            for tbl in pd.read_html(io.StringIO(html)):
                hit = False
                for col in cols:
                    if col in tbl.columns:
                        s = [str(x).strip().upper().replace(".", "-") for x in tbl[col]]
                        s = [x for x in s if x.isascii() and 1 <= len(x) <= 6
                             and x.replace("-", "").isalpha()]
                        if len(s) > 50:
                            got += s; hit = True
                        break
                if hit:
                    break
        except Exception:
            pass
    got += [t.upper() for t in BACKUP]
    seen, out = set(), []
    for t in got:
        if t not in seen:
            seen.add(t); out.append(t)
    return out[:limit]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch(tickers, period="1y"):
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
            if len(d) >= 90:
                out[t] = d
        except Exception:
            pass
    return out


def prep(d):
    c, o = d["Close"].values.astype(float), d["Open"].values.astype(float)
    h, l = d["High"].values.astype(float), d["Low"].values.astype(float)
    v = d["Volume"].values.astype(float)
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return dict(c=c, o=o, h=h, l=l, v=v, atr=pd.Series(tr).rolling(14).mean().values)


def core(A, i, C):
    c, o, h, l, v = A["c"], A["o"], A["h"], A["l"], A["v"]
    if i < 80:
        return None, "היסטוריה קצרה"
    s = max(0, i - 109)
    cw, ow, lw, vw = c[s:i + 1], o[s:i + 1], l[s:i + 1], v[s:i + 1]
    n = len(cw)
    peak = int(np.argmax(cw[:-1]))
    age = n - 1 - peak
    if age < C["age_lo"]:
        return None, "עוד בשיא"
    if age > C["age_hi"]:
        return None, "השיא ישן מדי"
    st_ = max(0, peak - C["leg_max"])
    if peak - st_ < C["leg_min"]:
        return None, "אין מקום לזינוק"
    low_i = st_ + int(np.argmin(cw[st_:peak]))
    leg_bars = peak - low_i
    if leg_bars < C["leg_min"]:
        return None, "זינוק קצר מדי"
    lo_px, pk_px, px = float(cw[low_i]), float(cw[peak]), float(cw[-1])
    if lo_px <= 0 or pk_px <= lo_px:
        return None, "אין זינוק"
    rise = (pk_px / lo_px - 1) * 100
    if rise < C["rise_min"]:
        return None, "זינוק קטן מדי"
    retr = (pk_px - px) / (pk_px - lo_px) * 100
    if retr < C["retr_lo"]:
        return None, "כמעט לא תיקנה"
    if retr > C["retr_hi"]:
        return None, "החזירה את כל הזינוק"
    base = vw[max(0, low_i - 50):low_i]
    base_v = float(base.mean()) if len(base) >= 5 else float(vw[:max(low_i, 1)].mean())
    leg_v, pull_v = vw[low_i:peak + 1], vw[peak:]
    if base_v <= 0 or len(leg_v) == 0 or len(pull_v) == 0 or pull_v.mean() <= 0:
        return None, "אין נתוני נפח"
    spike = float(leg_v.max() / base_v)
    dry = float(leg_v.mean() / pull_v.mean())
    if dry < C["dry_min"]:
        return None, "הנפח לא התייבש"
    sup = float(lw[peak:].min())
    a = float(A["atr"][i])
    if not np.isfinite(a) or a <= 0:
        return None, "אין ATR"
    atr_pct = a / px * 100
    if atr_pct < C["atr_min"]:
        return None, "תנודתיות נמוכה"
    rvol = float(v[i - 2:i + 1].mean() / v[max(0, i - 49):i + 1].mean())
    entry = max(px, float(h[i])) + .05 * a
    stop = max(min(sup - .35 * a, entry - 3 * a), entry - 4 * a)
    risk = entry - stop
    if risk <= 0:
        return None, "סטופ לא תקין"
    leg = pk_px - lo_px
    tp1, tp2, tp3 = pk_px, pk_px + .5 * leg, pk_px + leg
    rr = (tp1 - entry) / risk
    if rr < C["rr_min"]:
        return None, "אין מרווח ל-TP1"
    return dict(dry=round(dry, 2), spike=round(spike, 2), rise=round(rise, 1),
                leg_bars=int(leg_bars), retrace=round(retr, 1), price=round(px, 2),
                entry=round(entry, 2), stop=round(stop, 2), tp1=round(tp1, 2),
                tp2=round(tp2, 2), tp3=round(tp3, 2), rr=round(float(rr), 2),
                rr3=round(float((tp3 - entry) / risk), 2), risk_share=round(float(risk), 2),
                to_entry=round((entry / px - 1) * 100, 2), rvol=round(rvol, 2),
                atr_pct=round(atr_pct, 1), age=int(age), support=round(sup, 2),
                peak_px=round(pk_px, 2), base=round(lo_px, 2),
                low_i=s + low_i, peak_i=s + peak), None


def why_he(r):
    return (f"עלתה {r['rise']:.0f}% ב-{r['leg_bars']} ימים בנפח פי {r['spike']:.1f}, "
            f"תיקנה {r['retrace']:.0f}% בנפח קטן פי {r['dry']:.2f}, "
            f"ונתמכת ב-{r['support']:.2f}")


def ladder(r):
    lo, hi = r["stop"], r["tp3"]
    rng = hi - lo if hi > lo else 1
    lv = sorted([("TP3", r["tp3"], "tp"), ("TP2", r["tp2"], "tp"), ("TP1", r["tp1"], "tp"),
                 ("כניסה", r["entry"], "en"), ("מחיר", r["price"], "now"),
                 ("סטופ", r["stop"], "st")], key=lambda x: -x[1])
    used, html = [], ""
    for tag, p, cls in lv:
        top = 8 + (1 - (p - lo) / rng) * 170
        for u in used:
            if abs(top - u) < 19:
                top = u + 19
        used.append(top)
        html += (f'<div class="lv {cls}" style="top:{top:.0f}px"><div class="tag">{tag}</div>'
                 f'<div class="dot"></div><div class="px">{p:.2f}</div></div>')
    return f'<div class="lad">{html}</div>'


def card(r, best=False):
    return f"""<div class="tc{' top' if best else ''}">
<div class="tc-head"><div class="tc-sym">{r['ticker']}</div>
<div class="tc-dry"><b>{r['dry']:.2f}×</b><span>יובש נפח</span></div></div>
{ladder(r)}
<div class="tc-line"><span>{int(r['shares'])} מניות · סיכון <b>${r['risk_total']:.0f}</b></span>
<span>יחס <b>1:{r['rr']:.1f}</b></span></div>
<div class="tc-line" style="border:0;padding-top:.3rem">
<span>{r['age']} ימים מהשיא · ATR {r['atr_pct']}%</span>
<span>הפעלה ב-<b>{r['to_entry']:+.1f}%</b></span></div>
<div class="tc-why">{r['why']}</div></div>"""


# ---------------------------------------------------------------- filters

with st.sidebar:
    st.markdown("### כיול")
    st.caption("נקבע לפי בדיקה היסטורית. יובש נפח 1.3 הוא הפרמטר היחיד שהוכיח יציבות.")
    limit = st.slider("מספר מניות", 200, 2500, 1500, 100)
    dry_min = st.slider("יובש נפח מינימלי", 0.5, 2.5, 1.30, 0.05)
    agev = st.slider("ימים מהשיא", 1, 60, (3, 35))
    retr = st.slider("עומק תיקון %", 5, 95, (20, 70))
    rise_min = st.slider("גודל זינוק מינימלי %", 3, 100, 12)
    leg_min = st.slider("ימי זינוק מינימלי", 3, 20, 5)
    leg_max = st.slider("ימי זינוק מקסימלי", 8, 60, 30)
    atr_min = st.slider("ATR מינימלי %", 0.0, 10.0, 2.0, 0.25)
    rr_min = st.slider("יחס סיכון-סיכוי", 0.5, 5.0, 1.5, 0.1)
    min_px = st.number_input("מחיר מינימלי $", 1.0, 200.0, 3.0, 1.0)
    min_dv = st.number_input("מחזור מינימלי (מיליון $)", 1.0, 500.0, 10.0, 1.0)
    st.markdown("### חשבון")
    acct = st.number_input("גודל תיק $", 500, 5_000_000, 25_000, 500)
    riskp = st.slider("סיכון לעסקה %", 0.25, 5.0, 0.5, 0.25)

C = dict(leg_min=leg_min, leg_max=leg_max, rise_min=rise_min, retr_lo=retr[0],
         retr_hi=retr[1], age_lo=agev[0], age_hi=agev[1], atr_min=atr_min,
         rr_min=rr_min, dry_min=dry_min)

st.markdown(f"""<div class="mast">
<div><h1>MAVRI</h1><div class="sub">זינוק בנפח · תיקון בנפח יבש · כניסה בתמיכה</div></div>
<div class="stamp">{datetime.now():%d.%m.%Y  %H:%M}</div></div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------- detail view

if st.session_state.get("open"):
    r = next((x for x in st.session_state.get("rows", [])
              if x["ticker"] == st.session_state["open"]), None)
    if r is None:
        st.session_state["open"] = None
        st.rerun()

    if st.button("→  חזרה לרשימה"):
        st.session_state["open"] = None
        st.rerun()

    st.markdown(f"### {r['ticker']}")
    st.markdown(f'<div class="tc-why" style="font-size:.92rem">{r["why"]}</div>',
                unsafe_allow_html=True)

    d1 = fetch((r["ticker"],)).get(r["ticker"])
    if d1 is not None:
        v = d1.tail(120)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[.76, .24], vertical_spacing=.02)
        fig.add_trace(go.Candlestick(x=v.index, open=v.Open, high=v.High, low=v.Low,
                                     close=v.Close, name="",
                                     increasing_line_color="#0E7C5A",
                                     decreasing_line_color="#B4293B",
                                     increasing_fillcolor="#0E7C5A",
                                     decreasing_fillcolor="#B4293B"), 1, 1)
        try:
            fig.add_vrect(x0=d1.index[r["low_i"]], x1=d1.index[r["peak_i"]],
                          fillcolor="#0E7C5A", opacity=.07, line_width=0)
            fig.add_vrect(x0=d1.index[r["peak_i"]], x1=v.index[-1],
                          fillcolor="#B4293B", opacity=.07, line_width=0)
        except Exception:
            pass
        for y_, lab, col, dash in [(r["entry"], "כניסה", "#16202B", "solid"),
                                   (r["stop"], "סטופ", "#B4293B", "dash"),
                                   (r["tp1"], "TP1", "#0E7C5A", "dot"),
                                   (r["tp2"], "TP2", "#0E7C5A", "dot"),
                                   (r["tp3"], "TP3", "#0E7C5A", "dot")]:
            fig.add_hline(y=y_, line_dash=dash, line_color=col, line_width=1,
                          annotation_text=f"{lab} {y_:.2f}", annotation_position="right",
                          annotation_font_size=11, row=1, col=1)
        vc = ["#0E7C5A" if x >= y2 else "#B4293B" for x, y2 in zip(v.Close, v.Open)]
        fig.add_trace(go.Bar(x=v.index, y=v.Volume, marker_color=vc, marker_line_width=0,
                             opacity=.55, showlegend=False), 2, 1)
        fig.update_layout(height=560, template="plotly_white", showlegend=False,
                          xaxis_rangeslider_visible=False, font_family="Heebo",
                          paper_bgcolor="#FFF", plot_bgcolor="#FFF",
                          margin=dict(l=8, r=88, t=14, b=8))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="#EEF1F4", zeroline=False)
        st.plotly_chart(fig, use_container_width=True)

    a, b, c_ = st.columns(3)
    a.metric("יובש נפח", f"{r['dry']}×")
    b.metric("יחס סיכון-סיכוי", f"1:{r['rr']}")
    c_.metric("מרחק להפעלה", f"{r['to_entry']:+.1f}%")

    st.code(f"{r['ticker']}\n"
            f"BUY STOP {int(r['shares'])} @ {r['entry']:.2f}\n"
            f"STOP {r['stop']:.2f}   (סיכון ${r['risk_total']:.0f})\n"
            f"TP1 {r['tp1']:.2f} · TP2 {r['tp2']:.2f} · TP3 {r['tp3']:.2f}\n"
            f"תמיכה {r['support']:.2f} · שיא {r['peak_px']:.2f} · "
            f"תיקון {r['retrace']}% · {r['age']} ימים מהשיא", language=None)
    st.stop()

# ---------------------------------------------------------------- scan

if st.button("סרוק את השוק"):
    uni = build_universe(limit)
    prog, note = st.progress(0.0), st.empty()
    rows, drop, liq = [], {}, 0
    batches = [tuple(uni[i:i + 120]) for i in range(0, len(uni), 120)]
    for i, b in enumerate(batches):
        note.caption(f"{i+1} מתוך {len(batches)} · נמצאו {len(rows)}")
        for t, d in fetch(b).items():
            try:
                p = float(d["Close"].iloc[-1])
                if p < min_px:
                    drop["מחיר נמוך"] = drop.get("מחיר נמוך", 0) + 1; continue
                if p * float(d["Volume"].tail(20).mean()) < min_dv * 1e6:
                    drop["מחזור נמוך"] = drop.get("מחזור נמוך", 0) + 1; continue
                liq += 1
                A = prep(d)
                r, why = core(A, len(A["c"]) - 1, C)
                if r is None:
                    drop[why] = drop.get(why, 0) + 1; continue
                sh = int((acct * riskp / 100) / r["risk_share"])
                r.update(ticker=t, shares=sh, why=why_he(r),
                         risk_total=round(sh * r["risk_share"], 2))
                rows.append(r)
            except Exception:
                drop["שגיאה"] = drop.get("שגיאה", 0) + 1
        prog.progress((i + 1) / len(batches))
    prog.empty(); note.empty()
    rows.sort(key=lambda x: -x["dry"])
    st.session_state.update(rows=rows, stats=(len(uni), liq), drop=drop, open=None)

# ---------------------------------------------------------------- list

if "rows" not in st.session_state:
    st.markdown('<div class="empty">לחץ על סריקה כדי לעבור על השוק.<br>'
                'החיפוש הוא אחר מניות שזינקו בנפח, תיקנו בנפח נמוך, '
                'ונמצאות עכשיו על התמיכה.</div>', unsafe_allow_html=True)
else:
    rows = st.session_state["rows"]
    u, l_ = st.session_state["stats"]
    st.markdown(f"""<div class="counts">
<div><b>{u:,}</b><span>מניות נסרקו</span></div>
<div><b>{l_:,}</b><span>עברו נזילות</span></div>
<div><b>{len(rows)}</b><span>בתבנית</span></div>
<div><b>{riskp}%</b><span>סיכון לעסקה</span></div></div>""", unsafe_allow_html=True)

    if not rows:
        st.markdown('<div class="empty">אין מניות בתבנית היום.<br>'
                    'התבנית נדירה — אל תרפה את הפילטרים רק כדי לקבל תוצאות.</div>',
                    unsafe_allow_html=True)
        with st.expander("מה נפסל"):
            st.dataframe(pd.DataFrame(sorted(st.session_state["drop"].items(),
                                             key=lambda x: -x[1]), columns=["סיבה", "כמות"]),
                         hide_index=True, use_container_width=True)
    else:
        for start in range(0, len(rows), 3):
            cols = st.columns(3, gap="medium")
            for col, r in zip(cols, rows[start:start + 3]):
                with col:
                    st.markdown(card(r, best=(r is rows[0])), unsafe_allow_html=True)
                    if st.button("פתח", key=f"o{r['ticker']}"):
                        st.session_state["open"] = r["ticker"]
                        st.rerun()

        df = pd.DataFrame(rows)
        c1, c2 = st.columns(2)
        c1.download_button("הורד CSV", df.to_csv(index=False),
                           f"mavri_{datetime.now():%Y%m%d}.csv", "text/csv",
                           use_container_width=True)
        c2.code(",".join(df["ticker"].tolist()), language=None)