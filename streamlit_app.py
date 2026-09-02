"""
MAVRI — impulse / pullback screener.
Leg up on volume -> pullback on drying volume -> reversal at support.
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

st.set_page_config(page_title="MAVRI", layout="wide",
                   initial_sidebar_state="collapsed", page_icon="◈")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Frank+Ruhl+Libre:wght@400;500;700&family=Heebo:wght@200;300;400;500;700&display=swap');
:root{--bg:#0B1216;--pan:#111C22;--rz:#17242B;--ln:#22333C;
      --tx:#E9EFF1;--mu:#93A8B2;--dm:#5F7581;--gd:#D4A64B;--lg:#2FBF8F;--sh:#E0605F;}
html,body,[class*="css"]{font-family:'Heebo',sans-serif;}
.stApp{background:var(--bg);}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:.7rem;padding-bottom:4rem;max-width:1320px;}
hr{border-color:var(--ln);}

/* ---- top bar ---- */
.bar{display:flex;align-items:center;justify-content:space-between;gap:1rem;
     border-bottom:1px solid var(--ln);padding:.1rem 0 .55rem;flex-wrap:wrap;}
.brand{display:flex;align-items:baseline;gap:.7rem;}
.brand h1{font-family:'Frank Ruhl Libre',serif;font-size:1.5rem;font-weight:700;margin:0;
          color:var(--tx);letter-spacing:.14em;}
.brand em{font-style:normal;font-size:.72rem;color:var(--dm);letter-spacing:.02em;}
.status{display:flex;gap:1.4rem;align-items:center;direction:rtl;}
.status i{font-style:normal;font-size:.7rem;color:var(--dm);display:block;letter-spacing:.02em;}
.status b{font-family:'Frank Ruhl Libre',serif;font-size:.98rem;font-weight:500;color:var(--mu);
          font-variant-numeric:tabular-nums;}
.led{width:7px;height:7px;border-radius:50%;display:inline-block;margin-left:.35rem;}
.led.on{background:var(--lg);box-shadow:0 0 7px rgba(47,191,143,.8);}
.led.off{background:var(--dm);}

/* ---- counters ---- */
.kpi{display:flex;gap:0;flex-wrap:wrap;border:1px solid var(--ln);border-radius:4px;
     overflow:hidden;margin:.9rem 0 1rem;}
.kpi div{flex:1;min-width:110px;padding:.6rem .85rem;border-left:1px solid var(--ln);direction:rtl;}
.kpi div:last-child{border-left:0;}
.kpi b{font-family:'Frank Ruhl Libre',serif;font-size:1.35rem;font-weight:500;display:block;
       line-height:1.15;color:var(--tx);font-variant-numeric:tabular-nums;}
.kpi span{font-size:.68rem;color:var(--dm);letter-spacing:.02em;}
.kpi div.hi b{color:var(--gd);}

/* ---- card ---- */
.tc{background:var(--pan);border:1px solid var(--ln);border-radius:5px;
    padding:.85rem .95rem .75rem;direction:rtl;height:100%;}
.tc.top{border-color:rgba(212,166,75,.4);}
.tc-h{display:flex;align-items:baseline;justify-content:space-between;padding-bottom:.35rem;}
.tc-s{font-family:'Frank Ruhl Libre',serif;font-size:1.4rem;font-weight:700;color:var(--tx);
      direction:ltr;letter-spacing:.04em;}
.tc-d{text-align:left;direction:ltr;}
.tc-d b{font-family:'Frank Ruhl Libre',serif;font-size:1.25rem;font-weight:500;color:var(--gd);
        font-variant-numeric:tabular-nums;}
.tc-d span{display:block;font-size:.62rem;color:var(--dm);direction:rtl;}
.spark svg{width:100%;height:30px;display:block;opacity:.8;}
.chips{display:flex;gap:.28rem;flex-wrap:wrap;margin:.3rem 0 .1rem;}
.chip{font-size:.62rem;padding:.11rem .4rem;border:1px solid var(--ln);border-radius:2px;color:var(--mu);}
.chip.g{border-color:rgba(212,166,75,.45);color:var(--gd);}
.lad{position:relative;height:168px;margin:.4rem 0 .6rem;}
.lad:before{content:'';position:absolute;right:70px;top:5px;bottom:5px;width:1px;
            background:linear-gradient(180deg,rgba(47,191,143,.45),rgba(224,96,95,.45));}
.lv{position:absolute;right:0;left:0;height:0;display:flex;align-items:center;gap:.45rem;}
.lv .tg{width:66px;text-align:left;font-size:.66rem;color:var(--dm);}
.lv .dt{width:6px;height:6px;border-radius:50%;background:var(--dm);margin-right:-3px;flex:none;}
.lv .px{font-family:'Frank Ruhl Libre',serif;font-size:.9rem;color:var(--mu);
        font-variant-numeric:tabular-nums;direction:ltr;}
.lv.tp .dt{background:var(--lg);} .lv.tp .px{color:var(--lg);}
.lv.st .dt{background:var(--sh);} .lv.st .px,.lv.st .tg{color:var(--sh);}
.lv.en .dt{width:10px;height:10px;background:var(--gd);margin-right:-5px;
           box-shadow:0 0 0 3px rgba(212,166,75,.18);}
.lv.en .px{font-size:1.14rem;font-weight:700;color:var(--tx);}
.lv.en .tg{color:var(--gd);}
.lv.now .dt{background:transparent;border:1px solid var(--dm);}
.lv.now .px,.lv.now .tg{color:var(--dm);font-size:.75rem;}
.tl{display:flex;justify-content:space-between;font-size:.71rem;color:var(--dm);
    border-top:1px solid var(--ln);padding-top:.42rem;}
.tl b{color:var(--mu);font-weight:500;font-variant-numeric:tabular-nums;}
.tw{font-size:.72rem;line-height:1.6;color:var(--dm);margin-top:.4rem;}

.empty{border:1px dashed var(--ln);border-radius:5px;padding:2.4rem 1.4rem;text-align:center;
       color:var(--dm);direction:rtl;line-height:1.9;font-size:.87rem;}
h3{font-family:'Frank Ruhl Libre',serif !important;font-weight:500 !important;
   color:var(--tx) !important;direction:rtl;}
label{color:var(--mu) !important;font-size:.75rem !important;}
div[data-testid="stExpander"]{border:1px solid var(--ln);border-radius:4px;background:var(--pan);}
.fn{direction:rtl;margin:.2rem 0 .4rem;}
.fn .row{display:flex;align-items:center;gap:.6rem;margin:.22rem 0;}
.fn .nm{width:150px;font-size:.72rem;color:var(--mu);text-align:right;}
.fn .tr{flex:1;height:16px;background:var(--rz);border-radius:2px;overflow:hidden;}
.fn .fl{height:100%;background:linear-gradient(90deg,#2C4A57,#3E6B7C);}
.fn .fl.cut{background:linear-gradient(90deg,#5A2A2E,#8E4046);}
.fn .vl{width:64px;font-size:.72rem;color:var(--dm);font-variant-numeric:tabular-nums;
        text-align:left;direction:ltr;}

.dh{display:flex;align-items:flex-end;justify-content:space-between;gap:1rem;
    border-bottom:1px solid var(--ln);padding-bottom:.7rem;margin-bottom:.9rem;flex-wrap:wrap;}
.dh .sym{font-family:'Frank Ruhl Libre',serif;font-size:2.3rem;font-weight:700;
         color:var(--tx);letter-spacing:.04em;line-height:1;}
.dh .pr{font-family:'Frank Ruhl Libre',serif;font-size:1.5rem;color:var(--mu);
        font-variant-numeric:tabular-nums;}
.dh .ch{font-size:.9rem;font-variant-numeric:tabular-nums;}
.dh .up{color:var(--lg);} .dh .dn{color:var(--sh);}

.sg{display:grid;grid-template-columns:repeat(auto-fit,minmax(112px,1fr));gap:1px;
    background:var(--ln);border:1px solid var(--ln);border-radius:4px;overflow:hidden;
    direction:rtl;margin:.2rem 0 1rem;}
.sg div{background:var(--pan);padding:.55rem .7rem;}
.sg b{display:block;font-family:'Frank Ruhl Libre',serif;font-size:1.08rem;font-weight:500;
      color:var(--tx);font-variant-numeric:tabular-nums;line-height:1.25;}
.sg span{font-size:.65rem;color:var(--dm);}
.sg div.gd b{color:var(--gd);} .sg div.up b{color:var(--lg);} .sg div.dn b{color:var(--sh);}

.rmap{position:relative;height:52px;margin:.5rem 0 1rem;direction:ltr;}
.rmap .base{position:absolute;top:23px;left:0;right:0;height:5px;background:var(--rz);border-radius:3px;}
.rmap .risk{position:absolute;top:23px;height:5px;background:rgba(224,96,95,.55);}
.rmap .rew{position:absolute;top:23px;height:5px;background:rgba(47,191,143,.5);}
.rmap .mk{position:absolute;top:12px;width:2px;height:27px;background:var(--dm);}
.rmap .mk.e{background:var(--gd);height:33px;top:9px;}
.rmap .lb{position:absolute;top:42px;font-size:.62rem;color:var(--dm);transform:translateX(-50%);
          white-space:nowrap;}
.rmap .lb.e{color:var(--gd);}

.near{border-color:rgba(47,191,143,.5) !important;}
.badge{display:inline-block;font-size:.6rem;padding:.1rem .4rem;border-radius:2px;
       background:rgba(47,191,143,.16);color:var(--lg);margin-right:.3rem;}
a.tv{font-size:.68rem;color:var(--dm);text-decoration:none;border-bottom:1px dotted var(--ln);}
a.tv:hover{color:var(--gd);}
@media(max-width:820px){
  [data-testid="stHorizontalBlock"]{flex-direction:column !important;gap:.5rem !important;}
  [data-testid="stHorizontalBlock"]>div{width:100% !important;flex:1 1 100% !important;}
  .block-container{padding-left:.55rem;padding-right:.55rem;}
  .status{gap:.85rem;} .kpi div{min-width:80px;padding:.5rem .55rem;}
  .kpi b{font-size:1.12rem;} .dh .sym{font-size:1.7rem;}
  .fn .nm{width:98px;font-size:.64rem;} .lad{height:158px;}
  .rmap .lb{font-size:.55rem;}}
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

PRESETS = {
    "התבנית שלי": dict(dry=1.30, bb="רצועה תחתונה", bbmode="בונוס", price=10.0, dv=15.0,
                       sh=0.3, atr=1.0, atrp=2.0, rise=12, age=(3, 45), retr=(20, 70), rr=1.5),
    "רחב": dict(dry=1.10, bb="אחת מהשתיים", bbmode="בונוס", price=5.0, dv=8.0, sh=0.2,
                atr=0.4, atrp=1.5, rise=8, age=(2, 60), retr=(15, 85), rr=1.0),
    "מחמיר": dict(dry=1.45, bb="רצועה תחתונה", bbmode="חובה", price=15.0, dv=40.0, sh=1.0,
                  atr=1.5, atrp=3.0, rise=20, age=(3, 25), retr=(30, 65), rr=2.0),
}


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
                            got += s
                            hit = True
                        break
                if hit:
                    break
        except Exception:
            pass
    got += [t.upper() for t in BACKUP]
    seen, out = set(), []
    for t in got:
        if t not in seen:
            seen.add(t)
            out.append(t)
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


@st.cache_data(ttl=900, show_spinner=False)
def fetch_one(t, period, interval):
    """Single ticker, any interval. Returns DataFrame or None."""
    for attempt in (1, 2):
        try:
            d = yf.download(t, period=period, interval=interval, auto_adjust=False,
                            progress=False, timeout=45, threads=False)
            if d is None or len(d) == 0:
                continue
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            d = d.dropna()
            if len(d) >= 5:
                return d
        except Exception:
            pass
    return None


def prep(d):
    c, o = d["Close"].values.astype(float), d["Open"].values.astype(float)
    h, l = d["High"].values.astype(float), d["Low"].values.astype(float)
    v = d["Volume"].values.astype(float)
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    cs = pd.Series(c)
    mid, sd = cs.rolling(20).mean(), cs.rolling(20).std()
    return dict(c=c, o=o, h=h, l=l, v=v,
                atr=pd.Series(tr).rolling(14).mean().values,
                bbu=(mid + 2 * sd).values, bbl=(mid - 2 * sd).values)


def bb_hits(A, i, look, side):
    out = []
    for j in range(max(0, i - look + 1), i + 1):
        lo_b, up_b = A["bbl"][j], A["bbu"][j]
        if side in ("lower", "both") and np.isfinite(lo_b) and A["l"][j] <= lo_b and A["c"][j] > lo_b:
            out.append("תחתונה")
        if side in ("upper", "both") and np.isfinite(up_b) and A["h"][j] >= up_b and A["c"][j] < up_b:
            out.append("עליונה")
    return sorted(set(out))


def core(A, i, C):
    c, o, h, l, v = A["c"], A["o"], A["h"], A["l"], A["v"]
    if i < 80:
        return None, "היסטוריה קצרה"
    s = max(0, i - C["win"] + 1)
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
    tags = []
    if C["bb"] != "off":
        tags = bb_hits(A, i, C["bb_look"], C["bb"])
        if C["bb_hard"] and not tags:
            return None, "אין נר על בולינג׳ר"
    sup = float(lw[peak:].min())
    a = float(A["atr"][i])
    if not np.isfinite(a) or a <= 0:
        return None, "אין ATR"
    if a < C["atr_abs"]:
        return None, "ATR נמוך מדי"
    atr_pct = a / px * 100
    if atr_pct < C["atr_pct"]:
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
                atr=round(a, 2), atr_pct=round(atr_pct, 1), age=int(age),
                support=round(sup, 2), peak_px=round(pk_px, 2), base=round(lo_px, 2),
                bb=tags, spark=[float(x) for x in c[max(0, i - 59):i + 1]]), None


def why_he(r):
    return (f"עלתה {r['rise']:.0f}% ב-{r['leg_bars']} ימים בנפח פי {r['spike']:.1f}, "
            f"תיקנה {r['retrace']:.0f}% בנפח קטן פי {r['dry']:.2f}, "
            f"ונתמכת ב-{r['support']:.2f}")


def sparkline(vals):
    if not vals or len(vals) < 3:
        return ""
    lo, hi = min(vals), max(vals)
    rng = hi - lo if hi > lo else 1
    n = len(vals)
    pts = " ".join(f"{i/(n-1)*100:.2f},{(1-(v-lo)/rng)*26+2:.2f}" for i, v in enumerate(vals))
    return (f'<div class="spark"><svg viewBox="0 0 100 30" preserveAspectRatio="none">'
            f'<polyline points="{pts}" fill="none" stroke="#93A8B2" stroke-width="1" '
            f'vector-effect="non-scaling-stroke"/></svg></div>')


def ladder(r):
    lo, hi = r["stop"], r["tp3"]
    rng = hi - lo if hi > lo else 1
    lv = sorted([("TP3", r["tp3"], "tp"), ("TP2", r["tp2"], "tp"), ("TP1", r["tp1"], "tp"),
                 ("כניסה", r["entry"], "en"), ("מחיר", r["price"], "now"),
                 ("סטופ", r["stop"], "st")], key=lambda x: -x[1])
    used, html = [], ""
    for tag, p, cls in lv:
        top = 5 + (1 - (p - lo) / rng) * 156
        for u in used:
            if abs(top - u) < 18:
                top = u + 18
        used.append(top)
        html += (f'<div class="lv {cls}" style="top:{top:.0f}px"><div class="tg">{tag}</div>'
                 f'<div class="dt"></div><div class="px">{p:.2f}</div></div>')
    return f'<div class="lad">{html}</div>'


def card(r, best=False):
    ch = "".join(f'<span class="chip g">בולינג׳ר {b}</span>' for b in r.get("bb", []))
    ch += f'<span class="chip">ATR {r["atr"]:.2f}$</span><span class="chip">RVOL {r["rvol"]:.2f}</span>'
    near = r["to_entry"] <= 1.0
    if near:
        ch = '<span class="badge">קרוב להפעלה</span>' + ch
    return f"""<div class="tc{' top' if best else ''}{' near' if near else ''}">
<div class="tc-h"><div class="tc-s">{r['ticker']}</div>
<div class="tc-d"><b>{r['dry']:.2f}×</b><span>יובש נפח</span></div></div>
{sparkline(r.get('spark', []))}<div class="chips">{ch}</div>{ladder(r)}
<div class="tl"><span>{int(r['shares'])} מניות · סיכון <b>${r['risk_total']:.0f}</b></span>
<span>יחס <b>1:{r['rr']:.1f}</b></span></div>
<div class="tl" style="border:0;padding-top:.22rem"><span>{r['age']} ימים מהשיא</span>
<span>הפעלה ב-<b>{r['to_entry']:+.1f}%</b></span></div>
<div class="tw">{r['why']}</div></div>"""


def funnel_html(total, drop):
    order = ["מחזור נמוך", "נפח מניות נמוך", "מחיר נמוך", "השיא ישן מדי", "עוד בשיא",
             "זינוק קטן מדי", "זינוק קצר מדי", "אין מקום לזינוק", "כמעט לא תיקנה",
             "החזירה את כל הזינוק", "הנפח לא התייבש", "אין נר בולינג׳ר",
             "ATR נמוך מדי", "תנודתיות נמוכה", "אין מרווח ל-TP1", "היסטוריה קצרה",
             "אין נתוני נפח", "סטופ לא תקין", "אין ATR", "שגיאה"]
    items = [(k, drop[k]) for k in order if drop.get(k)]
    items += [(k, v) for k, v in drop.items() if k not in order and v]
    mx = max([v for _, v in items], default=1)
    rows = ""
    for k, v in items:
        rows += (f'<div class="row"><div class="nm">{k}</div><div class="tr">'
                 f'<div class="fl cut" style="width:{v/mx*100:.1f}%"></div></div>'
                 f'<div class="vl">{v:,}</div></div>')
    return f'<div class="fn">{rows}</div>'


def stat_grid(items):
    cells = "".join(f'<div class="{c}"><b>{v}</b><span>{lab}</span></div>'
                    for lab, v, c in items)
    return f'<div class="sg">{cells}</div>'


def rmap(r):
    """Trade laid out in R multiples from stop to TP3."""
    risk = r["entry"] - r["stop"]
    if risk <= 0:
        return ""
    lo_r, hi_r = -1.0, (r["tp3"] - r["entry"]) / risk
    span = hi_r - lo_r
    def pos(x):
        return (x - lo_r) / span * 100
    e, t1 = pos(0), pos((r["tp1"] - r["entry"]) / risk)
    t2, t3 = pos((r["tp2"] - r["entry"]) / risk), pos(hi_r)
    h = (f'<div class="rmap"><div class="base"></div>'
         f'<div class="risk" style="left:0;width:{e:.1f}%"></div>'
         f'<div class="rew" style="left:{e:.1f}%;width:{t3-e:.1f}%"></div>'
         f'<div class="mk" style="left:0"></div><div class="lb" style="left:2%">סטופ −1R</div>'
         f'<div class="mk e" style="left:{e:.1f}%"></div>'
         f'<div class="lb e" style="left:{e:.1f}%">כניסה</div>')
    for x, lab in [(t1, f"TP1 +{(r['tp1']-r['entry'])/risk:.1f}R"),
                   (t2, f"TP2 +{(r['tp2']-r['entry'])/risk:.1f}R"),
                   (t3, f"TP3 +{hi_r:.1f}R")]:
        h += (f'<div class="mk" style="left:{x:.1f}%"></div>'
              f'<div class="lb" style="left:{min(x,96):.1f}%">{lab}</div>')
    return h + "</div>"


# ---------------------------------------------------------------- top bar

il = timezone(timedelta(hours=3))
now = datetime.now(il)
ny = now.astimezone(timezone(timedelta(hours=-4)))
mkt_open = ny.weekday() < 5 and (9 * 60 + 30) <= (ny.hour * 60 + ny.minute) < 16 * 60
st.markdown(f"""<div class="bar">
<div class="brand"><h1>MAVRI</h1><em>סורק תבניות זינוק ותיקון</em></div>
<div class="status">
  <div><i>שוק ניו־יורק</i><b><span class="led {'on' if mkt_open else 'off'}"></span>
     {'פתוח' if mkt_open else 'סגור'}</b></div>
  <div><i>שעה בישראל</i><b>{now:%H:%M}</b></div>
  <div><i>תאריך</i><b>{now:%d.%m.%y}</b></div>
</div></div>""", unsafe_allow_html=True)

if "P" not in st.session_state:
    st.session_state["P"] = dict(PRESETS["התבנית שלי"])

p1, p2, p3, p4 = st.columns([1, 1, 1, 3])
for col, name in zip((p1, p2, p3), PRESETS):
    if col.button(name, use_container_width=True, key=f"ps{name}"):
        st.session_state["P"] = dict(PRESETS[name])
        st.rerun()
P = st.session_state["P"]

f1, f2, f3, f4, f5, f6 = st.columns([1.05, 1.15, 1, 1.15, 1, 1.35])
dry_min = f1.number_input("יובש נפח מינ׳", 0.5, 3.0, float(P["dry"]), 0.05)
bb_lbl = f2.selectbox("נר בולינג׳ר", ["כבוי", "רצועה תחתונה", "רצועה עליונה", "אחת מהשתיים"],
                      index=["כבוי", "רצועה תחתונה", "רצועה עליונה",
                             "אחת מהשתיים"].index(P["bb"]))
min_px = f3.number_input("מחיר מינ׳ $", 1.0, 500.0, float(P["price"]), 1.0)
min_dv = f4.number_input("מחזור מינ׳ (מ׳ $)", 1.0, 500.0, float(P["dv"]), 1.0,
                         help="מחזור בדולרים, לא במניות. זה המדד הנכון לנזילות.")
atr_abs = f5.number_input("ATR מינ׳ $", 0.0, 20.0, float(P["atr"]), 0.1)
f6.markdown("<div style='height:1.55rem'></div>", unsafe_allow_html=True)
s1, s2 = f6.columns([2, 1])
go_ = s1.button("סרוק את השוק", type="primary", use_container_width=True)
fresh = s2.button("↻", use_container_width=True,
                  help="סריקה טרייה — מתעלם מהנתונים השמורים ומוריד הכל מחדש. איטי יותר.")
if fresh:
    fetch.clear()
    go_ = True

with st.expander("סינון מתקדם"):
    g1, g2, g3, g4 = st.columns(4)
    agev = g1.slider("ימים מהשיא", 1, 90, tuple(P["age"]))
    retr = g2.slider("עומק תיקון %", 5, 95, tuple(P["retr"]))
    rise_min = g3.slider("גודל זינוק מינ׳ %", 3, 100, int(P["rise"]))
    atr_pct = g4.slider("ATR מינ׳ %", 0.0, 10.0, float(P["atrp"]), 0.25)
    h1, h2, h3, h4 = st.columns(4)
    leg_min = h1.slider("ימי זינוק מינ׳", 3, 20, 5)
    leg_max = h2.slider("ימי זינוק מקס׳", 8, 60, 30)
    rr_min = h3.slider("יחס סיכון־סיכוי", 0.5, 5.0, float(P["rr"]), 0.1)
    min_sh = h4.number_input("נפח מינ׳ (מ׳ מניות)", 0.0, 50.0, float(P["sh"]), 0.1,
                             help="0 מכבה. סינון לפי מספר מניות פוסל מניות יקרות ונזילות.")
    j1, j2, j3, j4 = st.columns(4)
    bbmode = j1.selectbox("בולינג׳ר כ־", ["בונוס", "חובה"],
                          index=["בונוס", "חובה"].index(P.get("bbmode", "בונוס")),
                          help="בונוס = מסומן בתווית אך לא פוסל. חובה = מסנן.")
    bb_look = j2.slider("בולינג׳ר: ימים אחורה", 1, 10, 3)
    win = j3.slider("חלון חיפוש השיא (ימים)", 60, 200, 110, 10,
                    help="ככל שהחלון גדול יותר, כך 'השיא ישן מדי' פוסל יותר.")
    limit = j4.slider("מספר מניות לסריקה", 200, 2500, 1500, 100)
    k1, k2 = st.columns(2)
    acct = k1.number_input("גודל תיק $", 500, 5_000_000, 25_000, 500)
    riskp = k2.slider("סיכון לעסקה %", 0.25, 5.0, 0.5, 0.25)

C = dict(leg_min=leg_min, leg_max=leg_max, rise_min=rise_min, retr_lo=retr[0],
         retr_hi=retr[1], age_lo=agev[0], age_hi=agev[1], atr_pct=atr_pct,
         atr_abs=atr_abs, rr_min=rr_min, dry_min=dry_min, bb_look=bb_look, win=win,
         bb_hard=(bbmode == "חובה"),
         bb={"כבוי": "off", "רצועה תחתונה": "lower",
             "רצועה עליונה": "upper", "אחת מהשתיים": "both"}[bb_lbl])

# ---------------------------------------------------------------- detail

if st.session_state.get("open"):
    r = next((x for x in st.session_state.get("rows", [])
              if x["ticker"] == st.session_state["open"]), None)
    if r is None:
        st.session_state["open"] = None
        st.rerun()

    nav1, nav2 = st.columns([1, 4])
    if nav1.button("→  חזרה לרשימה", use_container_width=True):
        st.session_state["open"] = None
        st.rerun()
    others = [x["ticker"] for x in st.session_state["rows"]]
    jump = nav2.selectbox("מעבר מהיר", others, index=others.index(r["ticker"]),
                          label_visibility="collapsed")
    if jump != r["ticker"]:
        st.session_state["open"] = jump
        st.rerun()

    INTERVALS = {
        "יומי": ("1d", {"3 חודשים": "3mo", "6 חודשים": "6mo", "שנה": "1y", "שנתיים": "2y"}),
        "שבועי": ("1wk", {"שנה": "1y", "שנתיים": "2y", "5 שנים": "5y", "10 שנים": "10y"}),
        "שעתי": ("60m", {"5 ימים": "5d", "חודש": "1mo", "3 חודשים": "3mo", "6 חודשים": "6mo"}),
        "30 דקות": ("30m", {"5 ימים": "5d", "10 ימים": "10d", "חודש": "1mo"}),
        "15 דקות": ("15m", {"3 ימים": "3d", "5 ימים": "5d", "10 ימים": "10d"}),
        "5 דקות": ("5m", {"יום": "1d", "3 ימים": "3d", "5 ימים": "5d"}),
    }

    cc1, cc2, cc3, cc4 = st.columns([1, 1, 2.2, 1])
    tf = cc1.selectbox("נרות", list(INTERVALS), index=0)
    iv, ranges = INTERVALS[tf]
    per = cc2.selectbox("טווח", list(ranges), index=min(1, len(ranges) - 1))
    ind = cc3.multiselect("שכבות", ["בולינג׳ר", "SMA20", "SMA50", "SMA200", "VWAP",
                                    "רמות עסקה", "אזורי תבנית"],
                          default=["בולינג׳ר", "SMA20", "SMA50", "רמות עסקה", "אזורי תבנית"])
    show_rsi = cc4.checkbox("RSI", value=True)

    v = fetch_one(r["ticker"], ranges[per], iv)
    if v is None:
        st.error(f"לא הצלחתי להוריד נתונים ל-{r['ticker']} בטווח הזה. "
                 "אינטרוול תוך-יומי מוגבל בהיסטוריה — נסה טווח קצר יותר, או נרות יומיים.")
        if st.button("נסה שוב"):
            fetch_one.clear()
            st.rerun()
    else:
        rows_n = 3 if show_rsi else 2
        heights = [.60, .20, .20] if show_rsi else [.78, .22]
        fig = make_subplots(rows=rows_n, cols=1, shared_xaxes=True,
                            row_heights=heights, vertical_spacing=.018)
        if "בולינג׳ר" in ind and len(v) > 20:
            mid, sd = v["Close"].rolling(20).mean(), v["Close"].rolling(20).std()
            fig.add_trace(go.Scatter(x=v.index, y=mid + 2 * sd, line=dict(color="#33505E", width=1),
                                     name="BB עליון", hoverinfo="skip"), 1, 1)
            fig.add_trace(go.Scatter(x=v.index, y=mid - 2 * sd, line=dict(color="#33505E", width=1),
                                     fill="tonexty", fillcolor="rgba(51,80,94,.13)",
                                     name="BB תחתון", hoverinfo="skip"), 1, 1)
        fig.add_trace(go.Candlestick(x=v.index, open=v.Open, high=v.High, low=v.Low,
                                     close=v.Close, name=r["ticker"],
                                     increasing_line_color="#2FBF8F",
                                     decreasing_line_color="#E0605F",
                                     increasing_fillcolor="#2FBF8F",
                                     decreasing_fillcolor="#E0605F"), 1, 1)
        for nm, per_, col in [("SMA20", 20, "#4E8FB0"), ("SMA50", 50, "#D4A64B"),
                              ("SMA200", 200, "#8A6FB0")]:
            if nm in ind and len(v) > per_:
                fig.add_trace(go.Scatter(x=v.index, y=v["Close"].rolling(per_).mean(),
                                         line=dict(color=col, width=1), name=nm), 1, 1)
        if "VWAP" in ind:
            tp = (v["High"] + v["Low"] + v["Close"]) / 3
            vw = (tp * v["Volume"]).cumsum() / v["Volume"].cumsum().replace(0, np.nan)
            fig.add_trace(go.Scatter(x=v.index, y=vw, line=dict(color="#C77DBA", width=1,
                                                                dash="dot"), name="VWAP"), 1, 1)
        if "אזורי תבנית" in ind and iv == "1d":
            try:
                pk = v.index[-1 - r["age"]]
                lg = v.index[-1 - r["age"] - r["leg_bars"]]
                fig.add_vrect(x0=lg, x1=pk, fillcolor="#2FBF8F", opacity=.07,
                              line_width=0, row=1, col=1)
                fig.add_vrect(x0=pk, x1=v.index[-1], fillcolor="#E0605F", opacity=.07,
                              line_width=0, row=1, col=1)
            except Exception:
                pass
        if "רמות עסקה" in ind:
            for y_, lab, col, dash in [(r["entry"], "כניסה", "#D4A64B", "solid"),
                                       (r["stop"], "סטופ", "#E0605F", "dash"),
                                       (r["support"], "תמיכה", "#6C8896", "dot"),
                                       (r["tp1"], "TP1", "#2FBF8F", "dot"),
                                       (r["tp2"], "TP2", "#2FBF8F", "dot"),
                                       (r["tp3"], "TP3", "#2FBF8F", "dot")]:
                fig.add_hline(y=y_, line_dash=dash, line_color=col, line_width=1,
                              annotation_text=f"{lab} {y_:.2f}", annotation_position="right",
                              annotation_font_size=10, annotation_font_color=col, row=1, col=1)
        vc = ["#2FBF8F" if x >= y2 else "#E0605F" for x, y2 in zip(v.Close, v.Open)]
        fig.add_trace(go.Bar(x=v.index, y=v.Volume, marker_color=vc, marker_line_width=0,
                             opacity=.5, name="נפח", showlegend=False), 2, 1)
        if len(v) > 20:
            fig.add_trace(go.Scatter(x=v.index, y=v["Volume"].rolling(20).mean(),
                                     line=dict(color="#93A8B2", width=1), showlegend=False,
                                     hoverinfo="skip"), 2, 1)
        if show_rsi and len(v) > 15:
            dl = v["Close"].diff()
            up = dl.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
            dn = (-dl.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
            fig.add_trace(go.Scatter(x=v.index, y=100 - 100 / (1 + up / dn.replace(0, np.nan)),
                                     line=dict(color="#D4A64B", width=1.2), name="RSI",
                                     showlegend=False), 3, 1)
            for lvl, cl in [(70, "#E0605F"), (30, "#2FBF8F"), (50, "#2A3D47")]:
                fig.add_hline(y=lvl, line_color=cl, line_width=.8, line_dash="dot", row=3, col=1)
            fig.update_yaxes(range=[0, 100], row=3, col=1)
        if iv != "1d":
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]),
                                          dict(bounds=[16, 9.5], pattern="hour")])
        fig.update_layout(height=680 if show_rsi else 560, xaxis_rangeslider_visible=False,
                          font_family="Heebo", font_color="#93A8B2", hovermode="x unified",
                          dragmode="pan", paper_bgcolor="#0B1216", plot_bgcolor="#0B1216",
                          legend=dict(orientation="h", y=1.04, x=0, bgcolor="rgba(0,0,0,0)",
                                      font=dict(size=10)),
                          margin=dict(l=6, r=88, t=26, b=6))
        fig.update_xaxes(showgrid=False, linecolor="#22333C")
        fig.update_yaxes(gridcolor="#17242B", zeroline=False, linecolor="#22333C")
        st.plotly_chart(fig, use_container_width=True, config={
            "scrollZoom": True, "displaylogo": False,
            "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
            "displayModeBar": True, "responsive": True,
            "doubleClick": "reset"})
        st.caption("גלגלת העכבר מקרבת ומרחיקה · גרירה מזיזה · לחיצה כפולה מאפסת · "
                   "בנייד צביטה שתי אצבעות")

    st.markdown("### מפת הסיכון")
    st.markdown(rmap(r), unsafe_allow_html=True)

    q1, q2 = st.columns([1, 1])
    with q1:
        st.markdown("##### תוכנית מסחר")
        st.code(f"{r['ticker']}\n"
                f"BUY STOP {int(r['shares'])} @ {r['entry']:.2f}\n"
                f"STOP {r['stop']:.2f}\n"
                f"TP1 {r['tp1']:.2f}  ·  TP2 {r['tp2']:.2f}  ·  TP3 {r['tp3']:.2f}",
                language=None)
    with q2:
        st.markdown("##### כסף")
        st.markdown(stat_grid([
            ("כמות מניות", f"{int(r['shares']):,}", ""),
            ("שווי פוזיציה", f"${r['shares']*r['entry']:,.0f}", ""),
            ("סיכון", f"${r['risk_total']:,.0f}", "dn"),
            ("רווח ב-TP1", f"${r['shares']*(r['tp1']-r['entry']):,.0f}", "up"),
            ("רווח ב-TP3", f"${r['shares']*(r['tp3']-r['entry']):,.0f}", "up"),
            ("סיכון למניה", f"${r['risk_share']:.2f}", ""),
        ]), unsafe_allow_html=True)

    st.caption("הכניסה היא הוראת BUY STOP מעל שיא היום. אם המחיר לא מגיע לשם — אין עסקה.")
    st.stop()

# ---------------------------------------------------------------- scan

if go_:
    t0 = time.time()
    uni = build_universe(limit)
    prog, note = st.progress(0.0), st.empty()
    rows, drop, liq, last_bar = [], {}, 0, None
    batches = [tuple(uni[i:i + 120]) for i in range(0, len(uni), 120)]
    for i, b in enumerate(batches):
        note.caption(f"{i+1} / {len(batches)}  ·  {len(rows)} התאמות")
        for t, d in fetch(b).items():
            try:
                p = float(d["Close"].iloc[-1])
                if p < min_px:
                    drop["מחיר נמוך"] = drop.get("מחיר נמוך", 0) + 1
                    continue
                avg_sh = float(d["Volume"].tail(20).mean())
                if p * avg_sh < min_dv * 1e6:
                    drop["מחזור נמוך"] = drop.get("מחזור נמוך", 0) + 1
                    continue
                if min_sh > 0 and avg_sh < min_sh * 1e6:
                    drop["נפח מניות נמוך"] = drop.get("נפח מניות נמוך", 0) + 1
                    continue
                liq += 1
                if last_bar is None:
                    last_bar = d.index[-1]
                A = prep(d)
                r, why = core(A, len(A["c"]) - 1, C)
                if r is None:
                    drop[why] = drop.get(why, 0) + 1
                    continue
                sh = int((acct * riskp / 100) / r["risk_share"])
                r.update(ticker=t, shares=sh, why=why_he(r),
                         risk_total=round(sh * r["risk_share"], 2))
                rows.append(r)
            except Exception:
                drop["שגיאה"] = drop.get("שגיאה", 0) + 1
        prog.progress((i + 1) / len(batches))
    prog.empty()
    note.empty()
    rows.sort(key=lambda x: -x["dry"])
    st.session_state.update(
        rows=rows, stats=(len(uni), liq), drop=drop, open=None,
        scanned_at=datetime.now(il).strftime("%H:%M"),
        took=round(time.time() - t0, 1), cached=not fresh,
        data_date=last_bar.strftime("%d.%m.%y") if last_bar is not None else "—")

# ---------------------------------------------------------------- results

if "rows" not in st.session_state:
    st.markdown('<div class="empty">בחר פריסט או הגדר סינון, ולחץ על סריקה.<br>'
                'המערכת מחפשת מניות שזינקו בנפח, תיקנו בנפח נמוך, '
                'ונמצאות עכשיו על התמיכה.</div>', unsafe_allow_html=True)
else:
    rows = st.session_state["rows"]
    u, l_ = st.session_state["stats"]
    top_dry = max((r["dry"] for r in rows), default=0)
    took = st.session_state.get("took", 0)
    src = "מטמון" if st.session_state.get("cached") else "הורדה טרייה"
    st.markdown(f"""<div class="kpi">
<div><b>{u:,}</b><span>נסרקו</span></div>
<div><b>{l_:,}</b><span>עברו נזילות</span></div>
<div class="hi"><b>{len(rows)}</b><span>בתבנית</span></div>
<div><b>{top_dry:.2f}×</b><span>יובש מרבי</span></div>
<div><b>{st.session_state.get('data_date','—')}</b><span>נתונים עד</span></div>
<div><b>{took:.0f}s</b><span>{src}</span></div>
<div><b>{st.session_state.get('scanned_at','—')}</b><span>נסרק בשעה</span></div>
</div>""", unsafe_allow_html=True)
    if st.session_state.get("cached") and took < 20:
        st.caption("הסריקה הייתה מהירה כי הנתונים כבר היו שמורים מסריקה קודמת "
                   "(נשמרים לשעה). ללחיצה על ↻ תרד הורדה טרייה מהשוק.")

    if not rows:
        st.markdown('<div class="empty">אין מניות בתבנית בסינון הזה.<br>'
                    'התבנית נדירה — נסה את הפריסט הרחב לפני שאתה מרפה ידנית.</div>',
                    unsafe_allow_html=True)
        with st.expander("מפל הסינון — איפה המניות נפלו", expanded=True):
            st.markdown(funnel_html(u, st.session_state["drop"]), unsafe_allow_html=True)
            st.caption("השורה הארוכה ביותר היא המסנן שחוסם הכי הרבה. הרפה אותו ראשון.")
    else:
        v1, v2, v3 = st.columns([1.4, 1.6, 1])
        view = v1.radio("תצוגה", ["כרטיסים", "טבלה"], horizontal=True,
                        label_visibility="collapsed")
        srt = v2.selectbox("מיון", ["יובש נפח", "יחס סיכון־סיכוי", "קרוב להפעלה",
                                    "טרי (ימים מהשיא)", "גודל הזינוק"],
                           label_visibility="collapsed")
        only_near = v3.checkbox("רק קרובות להפעלה")
        keyf = {"יובש נפח": lambda x: -x["dry"],
                "יחס סיכון־סיכוי": lambda x: -x["rr"],
                "קרוב להפעלה": lambda x: x["to_entry"],
                "טרי (ימים מהשיא)": lambda x: x["age"],
                "גודל הזינוק": lambda x: -x["rise"]}[srt]
        rows = sorted(rows, key=keyf)
        if only_near:
            rows = [x for x in rows if x["to_entry"] <= 1.0]
        if not rows:
            st.info("אין מניות בטווח של 1% מרמת ההפעלה כרגע.")
            st.stop()
        if view == "טבלה":
            t = pd.DataFrame([{
                "מניה": r["ticker"], "יובש נפח": r["dry"], "מחיר": r["price"],
                "כניסה": r["entry"], "% לכניסה": r["to_entry"], "סטופ": r["stop"],
                "TP1": r["tp1"], "TP2": r["tp2"], "TP3": r["tp3"], "R:R": r["rr"],
                "כמות": r["shares"], "סיכון $": r["risk_total"], "ATR $": r["atr"],
                "RVOL": r["rvol"], "תיקון %": r["retrace"], "ימים": r["age"],
                "בולינג׳ר": ", ".join(r.get("bb", [])) or "—"} for r in rows])
            st.dataframe(t, hide_index=True, use_container_width=True, height=460,
                         column_config={
                             "יובש נפח": st.column_config.ProgressColumn(
                                 "יובש נפח", format="%.2f×", min_value=1.0, max_value=2.2),
                             "R:R": st.column_config.ProgressColumn(
                                 "R:R", format="1:%.1f", min_value=0.0, max_value=4.0),
                             "מחיר": st.column_config.NumberColumn(format="$%.2f"),
                             "כניסה": st.column_config.NumberColumn(format="$%.2f"),
                             "סטופ": st.column_config.NumberColumn(format="$%.2f"),
                             "TP1": st.column_config.NumberColumn(format="$%.2f"),
                             "TP2": st.column_config.NumberColumn(format="$%.2f"),
                             "TP3": st.column_config.NumberColumn(format="$%.2f"),
                             "% לכניסה": st.column_config.NumberColumn(format="%+.2f%%")})
            pick = st.selectbox("פתח ניתוח", [r["ticker"] for r in rows])
            if st.button("פתח", type="primary"):
                st.session_state["open"] = pick
                st.rerun()
        else:
            for start in range(0, len(rows), 3):
                cols = st.columns(3, gap="medium")
                for col, r in zip(cols, rows[start:start + 3]):
                    with col:
                        st.markdown(card(r, best=(r is rows[0])), unsafe_allow_html=True)
                        if st.button("פתח ניתוח", key=f"o{r['ticker']}",
                                     use_container_width=True):
                            st.session_state["open"] = r["ticker"]
                            st.rerun()

        with st.expander("מפל הסינון — איפה המניות נפלו"):
            st.markdown(funnel_html(u, st.session_state["drop"]), unsafe_allow_html=True)

        df = pd.DataFrame([{k: v for k, v in r.items() if k != "spark"} for r in rows])
        z1, z2 = st.columns(2)
        z1.download_button("הורד CSV", df.to_csv(index=False),
                           f"mavri_{datetime.now():%Y%m%d}.csv", "text/csv",
                           use_container_width=True)
        z2.code(",".join(df["ticker"].tolist()), language=None)
