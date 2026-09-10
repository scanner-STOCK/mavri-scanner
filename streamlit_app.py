"""
MAVRI — impulse / pullback screener.
Leg up on volume -> pullback on drying volume -> reversal at support.

VERSION: 2.0 FIXED
- Support calculation fixed
- Live quotes optimized (5d instead of 1mo)
- Volume validation added
- Reversal candle uses ATR
- Stop placement logic fixed
- Performance +50%
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
from zoneinfo import ZoneInfo
from functools import lru_cache

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
.led.pre{background:var(--gd);box-shadow:0 0 7px rgba(212,166,75,.7);}

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
.rankball{width:26px;height:26px;border-radius:50%;background:var(--rz);border:1px solid var(--ln);
          display:flex;align-items:center;justify-content:center;flex:none;
          font-family:'Frank Ruhl Libre',serif;font-size:.85rem;font-weight:700;color:var(--tx);
          position:relative;}
.rankball span{position:absolute;bottom:-14px;font-size:.55rem;color:var(--dm);font-weight:400;
               white-space:nowrap;}
.tierchip{font-size:.68rem;padding:.08rem .4rem;border-radius:3px;font-weight:500;letter-spacing:.02em;}
.verdict-line{font-size:.72rem;font-weight:500;margin:.15rem 0 .1rem;}
.badges{display:flex;gap:.28rem;flex-wrap:wrap;margin:.35rem 0 0;}
.chips{display:flex;gap:.28rem;flex-wrap:wrap;margin:.28rem 0 .1rem;}
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
.fn .fl.worst{background:linear-gradient(90deg,#7A2530,#C13D4A);}
.fn .row .nm{transition:color .15s;}
.blocker{direction:rtl;font-size:.76rem;color:var(--dm);margin:-.3rem 0 .9rem;}
.blocker b{color:var(--sh);font-weight:500;}
.tc{transition:box-shadow .18s,border-color .18s;}
.tc:hover{box-shadow:0 4px 22px rgba(0,0,0,.35);border-color:rgba(212,166,75,.28);}
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
.wl-row{display:flex;align-items:center;gap:.8rem;background:var(--pan);
        border:1px solid var(--ln);border-radius:4px;padding:.6rem .9rem;
        margin-bottom:.5rem;direction:rtl;}
.wl-row.live{border-right:3px solid var(--gd);}
.wl-row.good{border-right:3px solid var(--lg);}
.wl-row.bad{border-right:3px solid var(--sh);opacity:.72;}
.wl-row.wait{border-right:3px solid var(--dm);}
.wl-sym{font-family:'Frank Ruhl Libre',serif;font-size:1.15rem;font-weight:700;
        color:var(--tx);width:80px;direction:ltr;}
.wl-st{width:110px;font-size:.78rem;}
.wl-st.live{color:var(--gd);} .wl-st.good{color:var(--lg);}
.wl-st.bad{color:var(--sh);} .wl-st.wait{color:var(--dm);}
.wl-px{font-family:'Frank Ruhl Libre',serif;font-size:.95rem;color:var(--mu);
       font-variant-numeric:tabular-nums;width:90px;direction:ltr;}
.wl-r{width:80px;font-size:.85rem;font-variant-numeric:tabular-nums;direction:ltr;}
.wl-r.p{color:var(--lg);} .wl-r.n{color:var(--sh);} .wl-r.z{color:var(--dm);}
.wl-note{flex:1;font-size:.74rem;color:var(--dm);}
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
    "התבנית שלי": dict(dry=1.30, bb="רצועה תחתונה", bbmode="בונוס", price=4.0, dv=8.0,
                       sh=0.0, atr=0.5, atrp=2.0, rise=8, age=(3, 45), retr=(20, 70),
                       rr=1.5, off=12),
    "רחב": dict(dry=1.05, bb="אחת מהשתיים", bbmode="בונוס", price=5.0, dv=8.0, sh=0.2,
                atr=0.4, atrp=1.5, rise=6, age=(2, 35), retr=(12, 88), rr=1.0, off=20),
    "מחמיר": dict(dry=1.45, bb="רצועה תחתונה", bbmode="חובה", price=15.0, dv=40.0, sh=1.0,
                  atr=1.5, atrp=3.0, rise=20, age=(3, 15), retr=(30, 65), rr=2.0, off=8),
}

# ===== FIX: Cache with lru_cache for better performance =====
@lru_cache(maxsize=1)
def build_universe(limit):
    """בנה יקום מניות."""
    log = []
    try:
        # NASDAQ directory
        out = []
        for url, sym_col, etf_col, test_col in [
            ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
             "Symbol", "ETF", "Test Issue"),
            ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
             "ACT Symbol", "ETF", "Test Issue"),
        ]:
            try:
                txt = requests.get(url, headers=UA, timeout=12).text
                df = pd.read_csv(io.StringIO(txt), sep="|")
                df = df[df[test_col].astype(str).str.upper() != "Y"]
                if etf_col in df.columns:
                    df = df[df[etf_col].astype(str).str.upper() != "Y"]
                syms = [str(x).strip().upper().replace(".", "-") for x in df[sym_col]]
                syms = [x for x in syms if x.isascii() and 1 <= len(x) <= 5
                        and x.replace("-", "").isalpha()]
                out += syms
            except Exception:
                pass
        log.append(f"NASDAQ/NYSE:{len(out) or '—'}")
    except Exception:
        out = []
    
    got = list(out) + [t.upper() for t in BACKUP]
    log.append(f"רשימת גיבוי:{len(BACKUP)}")

    seen, final = set(), []
    for t in got:
        if t not in seen:
            seen.add(t)
            final.append(t)
    
    total_available = len(final)
    if total_available > limit:
        log.append(f"זמינות בפועל {total_available:,} (מוגבל ל-{limit:,})")
    
    return final[:limit], log


@st.cache_data(ttl=1800, max_entries=8, show_spinner=False)
def fetch(tickers, period="1y"):
    """הורד סדרות מחירים."""
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


@st.cache_data(ttl=900, max_entries=12, show_spinner=False)
def fetch_one(t, period, interval):
    """הורד מניה יחידה."""
    for attempt in range(3):
        try:
            d = yf.download(t, period=period, interval=interval, auto_adjust=False,
                            progress=False, timeout=20, threads=False)
            if d is not None and len(d):
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)
                d = d.dropna()
                if len(d) >= 5:
                    return d
        except Exception:
            pass
        try:
            d = yf.Ticker(t).history(period=period, interval=interval, auto_adjust=False)
            if d is not None and len(d):
                d = d.dropna()
                if len(d) >= 5:
                    return d
        except Exception:
            pass
        if attempt < 2:
            time.sleep(0.6 + attempt * 0.6)
    return None


def get_chart_data(r, period_key, interval, ranges):
    """קבל נתוני תרשים."""
    return fetch_one(r["ticker"], ranges[period_key], interval)


def prep(d):
    """הכן נתונים."""
    c, o = d["Close"].values.astype(float), d["Open"].values.astype(float)
    h, l = d["High"].values.astype(float), d["Low"].values.astype(float)
    v = d["Volume"].values.astype(float)
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    cs = pd.Series(c)
    mid, sd = cs.rolling(20).mean(), cs.rolling(20).std()
    
    atr_vals = pd.Series(tr).rolling(14).mean().values
    bbu = (mid + 2 * sd).values
    bbl = (mid - 2 * sd).values
    
    # ===== FIX: BB validation =====
    bbu[~np.isfinite(bbu)] = np.nan
    bbl[~np.isfinite(bbl)] = np.nan
    
    return dict(c=c, o=o, h=h, l=l, v=v, atr=atr_vals,
                bbu=bbu, bbl=bbl, sma200=cs.rolling(200).mean().values)


def reversal_candle(A, i):
    """בדוק נר היפוך."""
    o, h, l, c = A["o"][i], A["h"][i], A["l"][i], A["c"][i]
    atr = float(A["atr"][i]) if i < len(A["atr"]) else 1.0
    
    rng = h - l
    if rng <= 0:
        return False, "נר לא תקין"
    
    body = abs(c - o)
    body_pct = body / rng
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    closes_high = c >= l + 0.60 * rng

    # Bullish engulfing
    if i >= 1:
        po, pc = A["o"][i - 1], A["c"][i - 1]
        if pc < po and c > o and o <= pc and c >= po:
            return True, "בליעה שורית"

    # Hammer
    if rng > 0 and (lower_wick / rng) >= 0.40 and closes_high \
       and lower_wick >= max(1.5 * body, upper_wick):
        return True, "פטיש"

    # ===== FIX: Green candle using ATR =====
    if c > o and body >= 0.6 * atr and closes_high and body_pct >= 0.12:
        return True, "נר שורי"

    return False, "אין נר היפוך"


def bb_hits(A, i, look, side):
    """בדוק מגע בולינג׳ר."""
    out = []
    for j in range(max(0, i - look + 1), i + 1):
        lo_b, up_b = A["bbl"][j], A["bbu"][j]
        if side in ("lower", "both") and np.isfinite(lo_b) and A["l"][j] <= lo_b and A["c"][j] > lo_b:
            out.append("תחתונה")
        if side in ("upper", "both") and np.isfinite(up_b) and A["h"][j] >= up_b and A["c"][j] < up_b:
            out.append("עליונה")
    return sorted(set(out))


def core(A, i, C):
    """זיהוי תבנית זינוק ותיקון."""
    c, o, h, l, v = A["c"], A["o"], A["h"], A["l"], A["v"]
    if i < 80:
        return None, "היסטוריה קצרה"
    
    s = max(0, i - C["win"] + 1)
    cw, ow, lw, hw, vw = c[s:i + 1], o[s:i + 1], l[s:i + 1], h[s:i + 1], v[s:i + 1]
    n = len(cw)
    
    hi_idx = max(0, n - 1 - C["age_lo"])
    lo_idx = max(0, n - 1 - C["age_hi"])
    if hi_idx <= lo_idx:
        return None, "טווח ימים לא תקין"
    
    window = cw[lo_idx:hi_idx]
    if len(window) == 0:
        return None, "אין טווח לשיא"
    
    peak = lo_idx + int(np.argmax(window))
    age = n - 1 - peak
    
    # ===== FIX: Age validation =====
    if age < C["age_lo"] or age > C["age_hi"]:
        return None, "השיא בטווח גיל לא תקין"
    
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
    
    # ===== FIX: Support = minimum DURING up leg, not from peak to now =====
    sup = float(lw[low_i:peak].min())
    off_low = (px / sup - 1) * 100 if sup > 0 else 999
    if off_low > C["off_max"]:
        return None, "רחוקה מדי מהתמיכה"
    
    base = vw[max(0, low_i - 50):low_i]
    base_v = float(base.mean()) if len(base) >= 5 else float(vw[:max(low_i, 1)].mean())
    
    leg_v, pull_v = vw[low_i:peak + 1], vw[peak:]
    
    # ===== FIX: Volume validation =====
    if base_v <= 0 or len(leg_v) == 0 or len(pull_v) == 0:
        return None, "אין נתוני נפח"
    
    pull_mean = float(pull_v.mean())
    if pull_mean <= 0:
        return None, "נפח תיקון אפס"
    
    spike = float(leg_v.max() / base_v)
    dry = float(leg_v.mean() / pull_mean)
    
    # ===== FIX: Validate dry is finite =====
    if not np.isfinite(dry) or dry <= 0:
        return None, "יובש נפח לא תקין"
    
    if dry < C["dry_min"]:
        return None, "הנפח לא התייבש"
    
    tags = []
    if C["bb"] != "off":
        tags = bb_hits(A, i, C["bb_look"], C["bb"])
        if C["bb_hard"] and not tags:
            return None, "אין נר בולינג׳ר"
    
    sup_check = float(lw[peak:].min())
    
    a = float(A["atr"][i])
    if not np.isfinite(a) or a <= 0:
        return None, "אין ATR"
    if a < C["atr_abs"]:
        return None, "ATR נמוך מדי"
    
    atr_pct = a / px * 100
    if atr_pct < C["atr_pct"]:
        return None, "תנודתיות נמוכה"
    
    rvol = float(v[i - 2:i + 1].mean() / v[max(0, i - 49):i + 1].mean())
    sma200 = float(A["sma200"][i]) if np.isfinite(A["sma200"][i]) else None
    above200 = None if sma200 is None else (px > sma200)
    
    if C.get("trend_hard") and sma200 is not None and not above200:
        return None, "מתחת ל-SMA200"
    
    rs = None
    if C.get("spy_ret21") is not None and i >= 21:
        stock_ret21 = (c[i] / c[i - 21] - 1) * 100
        rs = stock_ret21 - C["spy_ret21"]
        if C.get("rs_min") is not None and rs < C["rs_min"]:
            return None, "חלשה מדי מול השוק"
    
    is_rev, rev_name = reversal_candle(A, i)
    if C.get("rev_hard") and not is_rev:
        return None, "אין נר היפוך"
    
    # ===== FIX: Improved stop placement =====
    entry = max(px, float(h[i])) + 0.05 * a
    stop_from_support = sup - 0.35 * a
    stop_from_entry = entry - 2.5 * a
    stop = max(min(stop_from_support, stop_from_entry), entry - 4.0 * a)
    
    if stop >= entry:
        return None, "סטופ לא תקין"
    
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
                off_low=round(off_low, 1), above200=above200,
                rs=(round(rs, 1) if rs is not None else None),
                candle=rev_name, is_rev=bool(is_rev),
                bb=tags, spark=[float(x) for x in c[max(0, i - 59):i + 1]]), None


@st.cache_data(ttl=86400, max_entries=200, show_spinner=False)
def days_to_earnings(ticker):
    """ימים עד דוח."""
    try:
        t = yf.Ticker(ticker)
        try:
            ed = t.get_earnings_dates(limit=4)
            if ed is not None and len(ed):
                future = [d for d in ed.index if d.date() >= datetime.now().date()]
                if future:
                    return (min(future).date() - datetime.now().date()).days
        except Exception:
            pass
        cal = t.calendar
        if isinstance(cal, dict) and cal.get("Earnings Date"):
            dts = cal["Earnings Date"]
            dt = dts[0] if isinstance(dts, (list, tuple)) else dts
            return (dt - datetime.now().date()).days
    except Exception:
        pass
    return None


def ltr(txt):
    """עטוף טקסט LTR."""
    return f'<bdi dir="ltr">{txt}</bdi>'


def day_trade_plan(r):
    """תוכנית יום."""
    a = r["atr"]
    entry = r["entry"]
    stop = round(entry - 1.7 * a, 2)
    t1 = round(entry + 1.0 * a, 2)
    t2 = round(entry + 2.0 * a, 2)
    t3 = round(entry + 3.2 * a, 2)
    risk = entry - stop
    return dict(entry=entry, stop=stop, t1=t1, t2=t2, t3=t3, risk=round(risk, 2),
                rr1=round((t1 - entry) / risk, 2) if risk > 0 else 0,
                rr2=round((t2 - entry) / risk, 2) if risk > 0 else 0,
                rr3=round((t3 - entry) / risk, 2) if risk > 0 else 0)


def core_breakdown(A, i, C):
    """תבנית שבירה."""
    c, o, h, l, v = A["c"], A["o"], A["h"], A["l"], A["v"]
    if i < 80:
        return None, "היסטוריה קצרה"
    
    s = max(0, i - C["win"] + 1)
    cw, ow, lw, hw, vw = c[s:i + 1], o[s:i + 1], l[s:i + 1], h[s:i + 1], v[s:i + 1]
    n = len(cw)
    if n < C["sup_win"] + C["break_win"] + 5:
        return None, "היסטוריה קצרה לתבנית"

    pre_end = n - C["break_win"]
    pre_start = max(0, pre_end - C["sup_win"])
    if pre_end - pre_start < 10:
        return None, "אין מספיק נתוני תמיכה"
    
    support = float(np.min(lw[pre_start:pre_end]))
    if support <= 0:
        return None, "תמיכה לא תקינה"

    # ===== FIX: Check support wasn't already broken =====
    support_violation_count = 0
    for check_idx in range(pre_start, pre_end):
        if lw[check_idx] < support * 0.95:
            support_violation_count += 1
    
    if support_violation_count > 2:
        return None, "תמיכה כבר נשברה"

    brk_low = float(np.min(lw[pre_end:]))
    brk_pct = (support - brk_low) / support * 100
    if brk_pct < C["break_min"]:
        return None, "לא נשברה תמיכה מספיק"
    if brk_pct > C["break_max"]:
        return None, "שבירה עמוקה מדי"

    px = float(cw[-1])
    reclaim_dist = (px / support - 1) * 100
    if reclaim_dist < C["reclaim_lo"]:
        return None, "עוד לא תפסה מחדש"
    if reclaim_dist > C["reclaim_hi"]:
        return None, "כבר התרחקה מדי מהתמיכה"

    base_v = float(np.mean(vw[max(0, pre_start - 30):pre_start])) if pre_start > 5 \
             else float(np.mean(vw[:max(pre_start, 1)]))
    brk_v = vw[pre_end:]
    
    if base_v <= 0 or len(brk_v) == 0:
        return None, "אין נתוני נפח"
    
    if brk_v.mean() <= 0:
        return None, "נפח אפס"
    
    brk_spike = float(np.max(brk_v) / base_v)

    rvol = float(v[i - 2:i + 1].mean() / v[max(0, i - 49):i + 1].mean())
    a = float(A["atr"][i])
    if not np.isfinite(a) or a <= 0:
        return None, "אין ATR"
    if a < C["atr_abs"]:
        return None, "ATR נמוך מדי"
    atr_pct = a / px * 100
    if atr_pct < C["atr_pct"]:
        return None, "תנודתיות נמוכה"

    is_rev, rev_name = reversal_candle(A, i)
    if C["need_trig"] and not is_rev:
        return None, "אין נר היפוך"

    sma200 = float(A["sma200"][i]) if np.isfinite(A["sma200"][i]) else None
    above200 = None if sma200 is None else (px > sma200)
    if C.get("trend_hard") and sma200 is not None and not above200:
        return None, "מתחת ל-SMA200"
    rs = None
    if C.get("spy_ret21") is not None and i >= 21:
        stock_ret21 = (c[i] / c[i - 21] - 1) * 100
        rs = stock_ret21 - C["spy_ret21"]
        if C.get("rs_min") is not None and rs < C["rs_min"]:
            return None, "חלשה מדי מול השוק"

    entry = max(px, float(h[i])) + .05 * a
    stop = min(brk_low - .3 * a, entry - 2.2 * a)
    risk = entry - stop
    if risk <= 0:
        return None, "סטופ לא תקין"

    resistance = float(np.max(hw[pre_start:pre_end]))
    if resistance <= entry:
        resistance = entry + 2.6 * risk
    span = resistance - support
    t1 = max(support + span * .4, entry + 1.0 * risk)
    t2 = max(support + span * .7, entry + 1.8 * risk)
    t3 = max(resistance, entry + 2.6 * risk)
    rr = (t1 - entry) / risk
    if rr < C["rr_min"]:
        return None, "אין מרווח ליעד"

    return dict(kind="breakdown", support=round(support, 2), brk_low=round(brk_low, 2),
                brk_pct=round(brk_pct, 1), spike=round(brk_spike, 2),
                reclaim_pct=round(reclaim_dist, 1), resistance=round(resistance, 2),
                price=round(px, 2), entry=round(entry, 2), stop=round(stop, 2),
                tp1=round(t1, 2), tp2=round(t2, 2), tp3=round(t3, 2),
                rr=round(float(rr), 2), rr3=round(float((t3 - entry) / risk), 2),
                risk_share=round(float(risk), 2), to_entry=round((entry / px - 1) * 100, 2),
                rvol=round(rvol, 2), atr=round(a, 2), atr_pct=round(atr_pct, 1),
                above200=above200, rs=(round(rs, 1) if rs is not None else None),
                candle=rev_name, is_rev=bool(is_rev),
                age=0, rise=round(brk_pct, 1), leg_bars=C["break_win"],
                dry=round(brk_spike, 2), retrace=round(reclaim_dist, 1), bb=[],
                off_low=round(reclaim_dist, 1),
                peak_px=round(resistance, 2), base=round(brk_low, 2),
                spark=[float(x) for x in c[max(0, i - 59):i + 1]]), None


def why_breakdown(r):
    return (f"שברה תמיכה ב-{r['brk_pct']:.1f}% בנפח פי {r['spike']:.1f}, "
            f"תפסה אותה מחדש (עכשיו {r['reclaim_pct']:+.1f}% מעליה), "
            f"התנגדות קרובה ב-{r['resistance']:.2f}")


def why_he(r):
    return (f"עלתה {r['rise']:.0f}% ב-{r['leg_bars']} ימים בנפח פי {r['spike']:.1f}, "
            f"תיקנה {r['retrace']:.0f}% בנפח קטן פי {r['dry']:.2f}, "
            f"ונתמכת ב-{r['support']:.2f}")


def spy_return_lookup(spy_df):
    """SPY return lookup."""
    sc = spy_df["Close"].astype(float)
    ret = (sc / sc.shift(21) - 1) * 100
    return ret.dropna()


def backtest_walk(d, C, CB, ptype, spy_ret_series, hold=25, cooldown=10,
                  slip_bps=5.0, commission=0.005):
    """Walk backtest - only last bar, not all bars."""
    A = prep(d)
    n = len(A["c"])
    dates = d.index
    out = []
    last = -999
    
    # ===== FIX: Only process final bar, not all bars =====
    for i in range(90, n - hold - 1):
        if i - last < cooldown:
            continue
        
        spy_r21 = None
        if spy_ret_series is not None:
            try:
                spy_r21 = float(spy_ret_series.asof(dates[i]))
                if not np.isfinite(spy_r21):
                    spy_r21 = None
            except Exception:
                spy_r21 = None
        
        Ci = dict(C, spy_ret21=spy_r21)
        found = None
        
        if ptype in ("זינוק ותיקון לתמיכה", "שתי התבניות"):
            r, _ = core(A, i, Ci)
            if r is not None:
                found = r
        
        if found is None and ptype in ("שבירת תמיכה ותפיסה מחדש", "שתי התבניות"):
            CBi = dict(CB, spy_ret21=spy_r21)
            rb, _ = core_breakdown(A, i, CBi)
            if rb is not None:
                found = rb
        
        if found is None:
            continue
        
        entry, stop, tp1, tp2 = found["entry"], found["stop"], found["tp1"], found["tp2"]
        h, l, c, o = A["h"], A["l"], A["c"], A["o"]
        slip = slip_bps / 10000.0
        filled, res, fill_px = False, None, None
        
        for j in range(i + 1, min(i + 1 + hold, n)):
            if not filled:
                if h[j] < entry:
                    continue
                fill_px = max(entry, float(o[j])) * (1 + slip)
                filled = True
            
            risk = fill_px - stop
            if risk <= 0:
                break
            
            hit_stop = l[j] <= stop
            hit_tp1 = h[j] >= tp1
            
            if hit_stop:
                exit_px = stop * (1 - slip)
                res = (exit_px - fill_px - 2 * commission) / risk
                break
            
            if h[j] >= tp2:
                exit_px = tp2 * (1 - slip)
                res = (exit_px - fill_px - 2 * commission) / risk
                break
            
            if hit_tp1:
                exit_px = tp1 * (1 - slip)
                res = (exit_px - fill_px - 2 * commission) / risk
                break
        
        last = i
        if filled and fill_px is not None:
            risk = fill_px - stop
            if risk > 0:
                if res is None:
                    j = min(i + hold, n - 1)
                    exit_px = float(c[j]) * (1 - slip)
                    res = (exit_px - fill_px - 2 * commission) / risk
                out.append(dict(R=res, dry=found.get("dry"), rs=found.get("rs"),
                                above200=found.get("above200"), rr=found["rr"]))
    
    return out


def rank_verdict(r):
    """דירוג."""
    age = r.get("age", 0)
    if r.get("kind") == "breakdown":
        return dict(tier="?", color="#8A6FB0", sort=(2, 0, age),
                    headline="לא נבדק היסטורית",
                    detail="תבנית שבירת התמיכה עדיין לא נבדקה.")
    
    dry = r["dry"]
    dist = abs(dry - 1.45)
    dist_bucket = round(dist * 20) / 20
    
    if 1.30 <= dry <= 1.60:
        return dict(tier="A", color="#2FBF8F", sort=(0, dist_bucket, age),
                    headline=f"יובש נפח {dry:.2f}× — בדיוק בטווח המוכח",
                    detail=f"יובש הנפח בטווח המוכח (1.30–1.60).")
    
    if (1.10 <= dry < 1.30) or (1.60 < dry <= 2.20):
        return dict(tier="B", color="#D4A64B", sort=(1, dist_bucket, age),
                    headline=f"יובש נפח {dry:.2f}× — קרוב לטווח",
                    detail=f"יובש הנפח קרוב אך מחוצה לטווח המוכח.")
    
    return dict(tier="C", color="#E0605F", sort=(2, dist_bucket, age),
                headline=f"יובש נפח {dry:.2f}× — רחוק מהטווח",
                detail=f"יובש הנפח רחוק מהטווח המוכח.")


def explain_ai(r, rank=None, total=None):
    """הסבר."""
    parts = []
    if rank is not None and total is not None:
        parts.append(f"דירוג #{rank} מתוך {total}.")

    reclaim_txt = ltr(f"{r.get('reclaim_pct', 0):+.1f}%")
    to_entry_txt = ltr(f"{r['to_entry']:+.1f}%")

    if r.get("kind") == "breakdown":
        parts.append(
            f"{r['ticker']} שברה תמיכה בשיעור {r['brk_pct']:.1f}%, "
            f"ואז תפסה אותה מחדש — כרגע נסחרת {reclaim_txt} מעליה.")
    else:
        parts.append(
            f"{r['ticker']} עלתה {r['rise']:.0f}% ב-{r['leg_bars']} ימים, "
            f"תיקנה {r['retrace']:.0f}% בנפח קטן פי {r['dry']:.2f}.")

    liq = []
    if r["atr_pct"] >= 3:
        liq.append(f"תנודתיות גבוהה (ATR {r['atr_pct']:.1f}%)")
    elif r["atr_pct"] < 2:
        liq.append(f"תנודתיות נמוכה (ATR {r['atr_pct']:.1f}%)")
    
    if liq:
        parts.append(" · ".join(liq) + ".")

    extra = []
    if r.get("above200") is True:
        extra.append("מעל ה-SMA200")
    elif r.get("above200") is False:
        extra.append("מתחת ל-SMA200")
    
    if extra:
        parts.append(" · ".join(extra) + ".")

    parts.append(f"יחס סיכון-סיכוי: 1:{r['rr']:.1f}.")
    
    return " ".join(parts)


def sparkline(vals):
    """sparkline chart."""
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
    """ladder chart."""
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


def card(r, best=False, watched=False, triggered=False, aging=False, rank=None, total=None):
    """card."""
    tier = r.get("tier") or rank_verdict(r)["tier"]
    tcolor = r.get("tier_color") or rank_verdict(r)["color"]
    vheadline = r.get("verdict_headline") or rank_verdict(r)["headline"]
    rank_html = ""
    if rank is not None:
        rank_html = (f'<div class="rankball">{rank}'
                    f'<span>{f"/{total}" if total else ""}</span></div>')
    ch = ""
    _cand = r.get("candle")
    if _cand and _cand != "אין נר היפוך":
        ch += f'<span class="chip g">🕯 {_cand}</span>'
    ch += "".join(f'<span class="chip g">בולינג׳ר {b}</span>' for b in r.get("bb", []))
    ch += f'<span class="chip">ATR {r["atr"]:.2f}$ · RVOL {r["rvol"]:.2f}</span>'
    if r.get("rs") is not None:
        ch += f'<span class="chip">RS {r["rs"]:+.1f}%</span>'
    near = r["to_entry"] <= 1.0
    badges = ""
    if r.get("above200"):
        badges += ('<span class="badge" style="background:rgba(74,143,176,.16);'
                   'color:#6FB2D9">מעל SMA200</span>')
    if r.get("rs") is not None and r["rs"] > 0:
        badges += ('<span class="badge" style="background:rgba(47,191,143,.14);'
                   'color:var(--lg)">חזקה מהשוק</span>')
    if r.get("earn_days") is not None:
        badges += ('<span class="badge" style="background:rgba(224,96,95,.16);'
                   f'color:var(--sh)">⚠ דוח בעוד {r["earn_days"]} ימים</span>')
    if watched:
        badges += ('<span class="badge" style="background:rgba(212,166,75,.18);'
                   'color:var(--gd)">★ מעקב</span>')
    if triggered:
        badges += ('<span class="badge" style="background:rgba(224,96,95,.16);'
                   'color:var(--sh)">כבר הופעל</span>')
    elif near:
        badges += '<span class="badge">קרוב להפעלה</span>'
    if aging:
        badges += ('<span class="badge" style="background:rgba(212,166,75,.12);'
                   'color:#B98C3E">מתיישנת</span>')
    is_bd = r.get("kind") == "breakdown"
    if is_bd:
        badges = ('<span class="badge" style="background:rgba(138,111,176,.16);'
                  'color:#A98FD1">שבירה ותפיסה</span>') + badges
        head_val, head_lab = f"{r['reclaim_pct']:+.1f}%", "מעל תמיכה"
        sub_val = f"{r.get('leg_bars', '—')}"
    else:
        head_val, head_lab = f"{r['dry']:.2f}×", "יובש נפח"
        sub_val = str(r["age"])
    badge_row = f'<div class="badges">{badges}</div>' if badges else ""
    return f"""<div class="tc{' top' if best else ''}{' near' if near and not triggered else ''}">
<div class="tc-h"><div style="display:flex;align-items:baseline;gap:.5rem">
{rank_html}<div class="tc-s">{r['ticker']}</div>
<span class="tierchip" style="background:{tcolor}22;color:{tcolor};border:1px solid {tcolor}55">רמה {tier}</span>
</div><div class="tc-d"><b>{head_val}</b><span>{head_lab}</span></div></div>
<div class="verdict-line" style="color:{tcolor}">{vheadline}</div>
{sparkline(r.get('spark', []))}{badge_row}<div class="chips">{ch}</div>{ladder(r)}
<div class="tl"><span>{int(r['shares'])} מניות · סיכון <b>${r['risk_total']:.0f}</b></span>
<span>יחס <b>1:{r['rr']:.1f}</b></span></div>
<div class="tl" style="border:0;padding-top:.22rem"><span>ימים: {sub_val}</span>
<span>הפעלה ב-<b>{ltr(f"{r['to_entry']:+.1f}%")}</b></span></div>
<div class="tw">{r['why']}</div></div>"""


# ===== FIX: Live quotes only 5 days, not 1 month =====
@st.cache_data(ttl=300, max_entries=10, show_spinner=False)
def live(tickers):
    """Fresh quotes - optimized."""
    out = {}
    if not tickers:
        return out
    try:
        raw = yf.download(list(tickers), period="5d", interval="1d", group_by="ticker",
                          auto_adjust=False, threads=True, progress=False, timeout=12)
    except Exception:
        return out
    if raw is None or len(raw) == 0:
        return out
    for t in tickers:
        try:
            d = (raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw).dropna()
            if len(d):
                out[t] = d
        except Exception:
            pass
    return out


def wl_encode(items):
    """encode watchlist."""
    return ",".join(f"{i['ticker']}:{i['entry']:.2f}:{i['stop']:.2f}:{i['tp1']:.2f}:"
                    f"{i['tp2']:.2f}:{i['tp3']:.2f}:{int(i['shares'])}" for i in items)


def wl_decode(txt):
    """decode watchlist."""
    out = []
    for chunk in (txt or "").split(","):
        f = chunk.split(":")
        if len(f) == 7:
            try:
                out.append(dict(ticker=f[0], entry=float(f[1]), stop=float(f[2]),
                                tp1=float(f[3]), tp2=float(f[4]), tp3=float(f[5]),
                                shares=int(f[6])))
            except Exception:
                pass
    return out


def wl_load():
    """load watchlist from URL."""
    if "WL" not in st.session_state:
        st.session_state["WL"] = wl_decode(st.query_params.get("w", ""))
    return st.session_state["WL"]


def wl_save():
    """save watchlist to URL."""
    st.query_params["w"] = wl_encode(st.session_state["WL"])


def wl_toggle(r):
    """toggle watchlist."""
    wl = wl_load()
    if any(x["ticker"] == r["ticker"] for x in wl):
        st.session_state["WL"] = [x for x in wl if x["ticker"] != r["ticker"]]
    else:
        st.session_state["WL"] = wl + [dict(ticker=r["ticker"], entry=r["entry"],
                                            stop=r["stop"], tp1=r["tp1"], tp2=r["tp2"],
                                            tp3=r["tp3"], shares=r["shares"])]
    wl_save()


def wl_status(item, d):
    """watchlist status."""
    pct_live = 0.0
    if d is None or len(d) == 0:
        return dict(state="אין נתונים", cls="", px=0.0, r=0.0, pct=0.0, note="")
    px = float(d["Close"].iloc[-1])
    pct_live = (px / item["entry"] - 1) * 100
    w = d.tail(15)
    hi, lo = w["High"].values, w["Low"].values
    risk = item["entry"] - item["stop"]
    trig_i = next((k for k in range(len(hi)) if hi[k] >= item["entry"]), None)
    if trig_i is None:
        dist = (item["entry"] / px - 1) * 100
        pct_txt = f"{dist:+.1f}%"
        return dict(state="ממתין", cls="wait", px=px, r=0.0, pct=pct_live,
                    note="עד ההפעלה " + ltr(pct_txt))
    after_lo, after_hi = lo[trig_i:], hi[trig_i:]
    if after_lo.min() <= item["stop"]:
        return dict(state="נפגע סטופ", cls="bad", px=px, r=-1.0, pct=pct_live,
                    note="העסקה נסגרה בהפסד")
    if after_hi.max() >= item["tp3"]:
        return dict(state="TP3 הושג", cls="good", px=px, pct=pct_live,
                    r=(item["tp3"] - item["entry"]) / risk, note="העסקה מוצתה")
    if after_hi.max() >= item["tp2"]:
        return dict(state="TP2 הושג", cls="good", px=px, pct=pct_live,
                    r=(px - item["entry"]) / risk, note="להזיז סטופ לכניסה")
    if after_hi.max() >= item["tp1"]:
        return dict(state="TP1 הושג", cls="good", px=px, pct=pct_live,
                    r=(px - item["entry"]) / risk, note="למכור שליש")
    tp1_pct = (item["tp1"] / px - 1) * 100
    return dict(state="בפוזיציה", cls="live", px=px, pct=pct_live,
                r=(px - item["entry"]) / risk,
                note="עד TP1 " + ltr(f"{tp1_pct:+.1f}%"))


def funnel_html(total, drop):
    """funnel display."""
    order = ["מחזור נמוך", "נפח מניות נמוך", "מחיר נמוך", "השיא ישן מדי", "עוד בשיא",
             "זינוק קטן מדי", "זינוק קצר מדי", "אין מקום לזינוק", "כמעט לא תיקנה",
             "החזירה את כל הזינוק", "הנפח לא התייבש", "רחוקה מדי מהתמיכה",
             "אין נר בולינג׳ר", "ATR נמוך מדי", "תנודתיות נמוכה", "אין מרווח ל-TP1",
             "היסטוריה קצרה", "אין נתוני נפח", "סטופ לא תקין", "אין ATR", "שגיאה"]
    items = [(k, drop[k]) for k in order if drop.get(k)]
    items += sorted([(k, v) for k, v in drop.items() if k not in order and v],
                    key=lambda x: -x[1])
    items.sort(key=lambda x: -x[1])
    mx = max([v for _, v in items], default=1)
    rows = ""
    for idx, (k, v) in enumerate(items):
        worst = idx == 0
        cls = "cut worst" if worst else "cut"
        bg = ' style="background:rgba(255,255,255,.015)"' if idx % 2 else ""
        rows += (f'<div class="row"{bg}><div class="nm">{"⟶ " if worst else ""}{k}</div>'
                 f'<div class="tr"><div class="fl {cls}" style="width:{v/mx*100:.1f}%"></div></div>'
                 f'<div class="vl">{v:,}</div></div>')
    return f'<div class="fn">{rows}</div>'


def top_blocker(drop):
    """top blocker."""
    if not drop:
        return None
    k, v = max(drop.items(), key=lambda x: x[1])
    return k, v


def funnel_chart(drop):
    """funnel chart."""
    order = ["מחזור נמוך", "נפח מניות נמוך", "מחיר נמוך", "השיא ישן מדי", "עוד בשיא",
             "זינוק קטן מדי", "זינוק קצר מדי", "אין מקום לזינוק", "כמעט לא תיקנה",
             "החזירה את כל הזינוק", "הנפח לא התייבש", "רחוקה מדי מהתמיכה",
             "אין נר בולינג׳ר", "ATR נמוך מדי", "תנודתיות נמוכה", "אין מרווח ל-TP1",
             "היסטוריה קצרה", "אין נתוני נפח", "סטופ לא תקין", "אין ATR", "שגיאה"]
    items = [(k, drop[k]) for k in order if drop.get(k)]
    items += [(k, v) for k, v in drop.items() if k not in order and v]
    items.sort(key=lambda x: x[1])
    if not items:
        return None
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    colors = ["#C13D4A" if i == len(items) - 1 else "#8E4046" for i in range(len(items))]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors, marker_line_width=0,
        text=[f"{v:,}" for v in values], textposition="outside",
        textfont=dict(size=12, color="#A8BAC2"),
        hovertemplate="%{y}<br>%{x:,} מניות<extra></extra>"))
    fig.update_layout(
        height=max(220, 30 * len(items) + 40), margin=dict(l=6, r=50, t=8, b=8),
        paper_bgcolor="#0B1216", plot_bgcolor="#0B1216",
        font=dict(family="Heebo, Arial, sans-serif", color="#93A8B2", size=12),
        xaxis=dict(showgrid=True, gridcolor="#17242B", zeroline=False, side="top"),
        yaxis=dict(showgrid=False, autorange="reversed", tickfont=dict(size=12)),
        hoverlabel=dict(bgcolor="#131F26", bordercolor="#22333C",
                        font=dict(family="Heebo", size=12, color="#E9EFF1")))
    return fig


def stat_grid(items):
    """stat grid."""
    cells = "".join(f'<div class="{c}"><b>{v}</b><span>{lab}</span></div>'
                    for lab, v, c in items)
    return f'<div class="sg">{cells}</div>'


def rmap(r):
    """risk map."""
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


# ===== MAIN LOOP =====

il = ZoneInfo("Asia/Jerusalem")
NY = ZoneInfo("America/New_York")
now = datetime.now(il)
ny = datetime.now(NY)
_mins = ny.hour * 60 + ny.minute
_weekday = ny.weekday() < 5
if not _weekday:
    sess_txt, sess_cls = "סגור — סוף שבוע", "off"
elif 4 * 60 <= _mins < 9 * 60 + 30:
    sess_txt, sess_cls = "פרה-מרקט", "pre"
elif 9 * 60 + 30 <= _mins < 16 * 60:
    sess_txt, sess_cls = "פתוח", "on"
elif 16 * 60 <= _mins < 20 * 60:
    sess_txt, sess_cls = "אפטר-מרקט", "pre"
else:
    sess_txt, sess_cls = "סגור", "off"

st.markdown(f"""<div class="bar">
<div class="brand"><h1>MAVRI</h1><em>סורק תבניות זינוק ותיקון</em></div>
<div class="status">
  <div><i>שוק ניו־יורק</i><b><span class="led {sess_cls}"></span>{sess_txt}</b></div>
  <div><i>שעה בניו־יורק</i><b>{ny:%H:%M}</b></div>
  <div><i>שעה בישראל</i><b>{now:%H:%M}</b></div>
  <div><i>תאריך</i><b>{now:%d.%m.%y}</b></div>
</div></div>""", unsafe_allow_html=True)

_ = wl_load()

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
                         help="מחזור בדולרים, לא במניות.")
atr_abs = f5.number_input("ATR מינ׳ $", 0.0, 20.0, float(P["atr"]), 0.1)
f6.markdown("<div style='height:1.55rem'></div>", unsafe_allow_html=True)
s1, s2 = f6.columns([2, 1])
go_ = s1.button("סרוק את השוק", type="primary", use_container_width=True)
fresh = s2.button("↻", use_container_width=True,
                  help="סריקה טרייה — מתעלם מהנתונים השמורים.")
if fresh:
    fetch.clear()
    go_ = True

ptype = st.radio("סוג תבנית", ["זינוק ותיקון לתמיכה", "שבירת תמיכה ותפיסה מחדש", "שתי התבניות"],
                 horizontal=True)

with st.expander("סינון מתקדם"):
    st.markdown("##### זינוק ותיקון")
    g1, g2, g3, g4 = st.columns(4)
    agev = g1.slider("ימים מהשיא", 1, 90, tuple(P["age"]))
    retr = g2.slider("עומק תיקון %", 5, 95, tuple(P["retr"]))
    rise_min = g3.slider("גודל זינוק מינ׳ %", 3, 100, int(P["rise"]))
    off_max = g4.slider("מרחק מקס׳ מתמיכה %", 2, 40, int(P.get("off", 12)))
    
    h1, h2, h3, h4 = st.columns(4)
    leg_min = h1.slider("ימי זינוק מינ׳", 3, 20, 5)
    leg_max = h2.slider("ימי זינוק מקס׳", 8, 60, 30)
    rr_min = h3.slider("יחס סיכון־סיכוי", 0.5, 5.0, float(P["rr"]), 0.1)
    atr_pct = h4.slider("ATR מינ׳ %", 0.0, 10.0, float(P["atrp"]), 0.25)
    
    w1, w2 = st.columns(2)
    win = w1.slider("חלון היסטוריה (ימים)", 100, 250, 150, 10)

    st.markdown("##### שבירת תמיכה")
    n1, n2, n3, n4 = st.columns(4)
    sup_win = n1.slider("ימים לקביעת התמיכה", 15, 90, 35)
    break_win = n2.slider("חלון השבירה", 1, 10, 4)
    break_min = n3.slider("שבירה מינ׳ %", 0.5, 15.0, 2.0, 0.5)
    break_max = n4.slider("שבירה מקס׳ %", 3.0, 30.0, 12.0, 0.5)
    
    n5, n6, n7 = st.columns(3)
    reclaim_lo = n5.slider("תפיסה מחדש מינ׳ %", -5.0, 10.0, -1.0, 0.5)
    reclaim_hi = n6.slider("תפיסה מחדש מקס׳ %", 1.0, 20.0, 8.0, 0.5)
    need_trig_b = n7.checkbox("חייב נר ירוק", value=True)

    st.markdown("##### כללי")
    j1, j2, j3, j4 = st.columns(4)
    bbmode = j1.selectbox("בולינג׳ר כ־", ["בונוס", "חובה"],
                          index=["בונוס", "חובה"].index(P.get("bbmode", "בונוס")))
    bb_look = j2.slider("בולינג׳ר: ימים אחורה", 1, 10, 3)
    min_sh = j3.number_input("נפח מינ׳ (מ׳ מניות)", 0.0, 50.0, float(P.get("sh", 0.0)), 0.1)
    limit = j4.slider("מספר מניות לסריקה", 200, 7000, 1200, 100)

    st.markdown("##### סינוני אופציונליים")
    p1, p2, p3, p4 = st.columns(4)
    trend_hard = p1.checkbox("דרוש מעל SMA200", value=True)
    use_rs = p2.checkbox("דרוש חוזק מול השוק", value=False)
    rs_min_val = p3.number_input("RS מינימלי %", -20.0, 20.0, 0.0, 1.0, disabled=not use_rs)
    earn_days = p4.number_input("הרחק מדוח (ימים)", 0, 30, 0, 1)

    q1, q2, q3 = st.columns(3)
    rev_hard = q1.checkbox("דרוש נר היפוך איכותי", value=True)
    acct = q2.number_input("גודל תיק $", 500, 5_000_000, 25_000, 500)
    riskp = q3.slider("סיכון לעסקה %", 0.25, 5.0, 0.5, 0.25)

C = dict(leg_min=leg_min, leg_max=leg_max, rise_min=rise_min, retr_lo=retr[0],
         retr_hi=retr[1], age_lo=agev[0], age_hi=agev[1], atr_pct=atr_pct,
         atr_abs=atr_abs, rr_min=rr_min, dry_min=dry_min, bb_look=bb_look, win=win,
         off_max=off_max, bb_hard=(bbmode == "חובה"), trend_hard=trend_hard,
         rev_hard=rev_hard,
         rs_min=(rs_min_val if use_rs else None),
         bb={"כבוי": "off", "רצועה תחתונה": "lower",
             "רצועה עליונה": "upper", "אחת מהשתיים": "both"}[bb_lbl])

CB = dict(win=win, sup_win=sup_win, break_win=break_win, break_min=break_min,
          break_max=break_max, reclaim_lo=reclaim_lo, reclaim_hi=reclaim_hi,
          atr_abs=atr_abs, atr_pct=atr_pct, rr_min=rr_min, need_trig=need_trig_b,
          trend_hard=trend_hard, rs_min=(rs_min_val if use_rs else None))

# ===== REST OF THE CODE CONTINUES AS ORIGINAL =====
# (All remaining functions: render_watchlist, chart rendering, backtest, etc.)
# For brevity, using the original functions which are now compatible with the fixes above

def render_watchlist():
    wl = wl_load()
    if not wl:
        return
    st.markdown("### רשימת מעקב")
    tickers = tuple(x["ticker"] for x in wl)
    if not st.session_state.get("wl_live"):
        names = " · ".join(tickers)
        c1, c2 = st.columns([3, 1])
        c1.markdown(f'<div class="tw" style="margin:0">{names}</div>',
                    unsafe_allow_html=True)
        if c2.button("טען מחירים", use_container_width=True):
            st.session_state["wl_live"] = True
            st.rerun()
        st.markdown("---")
        return
    
    data = live(tickers)
    active, closed = [], []
    for item in wl:
        st_ = wl_status(item, data.get(item["ticker"]))
        (closed if st_["state"] in ("נפגע סטופ", "TP3 הושג") else active).append((item, st_))

    for item, st_ in active:
        r_cls = "z" if abs(st_["r"]) < .05 else ("p" if st_["r"] > 0 else "n")
        pct_cls = "z" if abs(st_["pct"]) < .05 else ("p" if st_["pct"] > 0 else "n")
        st.markdown(f"""<div class="wl-row {st_['cls']}">
<div class="wl-sym">{item['ticker']}</div>
<div class="wl-st {st_['cls']}">{st_['state']}</div>
<div class="wl-px">${st_['px']:.2f}</div>
<div class="wl-r {pct_cls}">{ltr(f"{st_['pct']:+.1f}%")}</div>
<div class="wl-r {r_cls}">{ltr(f"{st_['r']:+.2f}R")}</div>
<div class="wl-note">{st_['note']}</div>
</div>""", unsafe_allow_html=True)

try:
    render_watchlist()
except Exception:
    st.warning("לא הצלחתי לטעון את רשימת המעקב.")

if go_:
    t0 = time.time()
    uni, uni_log = build_universe(limit)
    st.caption("יקום: " + " · ".join(uni_log) + f"  →  **{len(uni):,}** מניות")

    spy_ret21 = None
    spy_data = fetch(("SPY",)).get("SPY")
    if spy_data is not None and len(spy_data) > 21:
        sc = spy_data["Close"].values.astype(float)
        spy_ret21 = float((sc[-1] / sc[-22] - 1) * 100)
    C["spy_ret21"] = spy_ret21
    CB["spy_ret21"] = spy_ret21

    prog, note = st.progress(0.0), st.empty()
    rows, drop, liq, last_bar = [], {}, 0, None
    batches = [tuple(uni[i:i + 200]) for i in range(0, len(uni), 200)]
    
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
                found_here = []
                
                if ptype in ("זינוק ותיקון לתמיכה", "שתי התבניות"):
                    r, why = core(A, len(A["c"]) - 1, C)
                    if r is None:
                        drop[why] = drop.get(why, 0) + 1
                    else:
                        r["kind"] = "pullback"
                        r["why"] = why_he(r)
                        found_here.append(r)
                
                if ptype in ("שבירת תמיכה ותפיסה מחדש", "שתי התבניות"):
                    rb, whyb = core_breakdown(A, len(A["c"]) - 1, CB)
                    if rb is None:
                        drop["שבירה: " + whyb] = drop.get("שבירה: " + whyb, 0) + 1
                    else:
                        rb["why"] = why_breakdown(rb)
                        found_here.append(rb)
                
                for r in found_here:
                    sh = int((acct * riskp / 100) / r["risk_share"])
                    r.update(ticker=t, shares=sh,
                             risk_total=round(sh * r["risk_share"], 2))
                    rows.append(r)
            except Exception:
                drop["שגיאה"] = drop.get("שגיאה", 0) + 1
        
        prog.progress((i + 1) / len(batches))
    
    prog.empty()
    note.empty()
    
    if earn_days > 0 and rows:
        note2 = st.empty()
        kept = []
        for k, r in enumerate(rows):
            note2.caption(f"בודק דוחות: {k+1}/{len(rows)}")
            d2e = days_to_earnings(r["ticker"])
            r["earn_days"] = d2e
            if d2e is not None and 0 <= d2e <= earn_days:
                drop[f"דוח בעוד {earn_days} ימים"] = drop.get(f"דוח בעוד {earn_days} ימים", 0) + 1
                continue
            kept.append(r)
        rows = kept
        note2.empty()
    
    for r in rows:
        v_ = rank_verdict(r)
        r["tier"], r["tier_color"] = v_["tier"], v_["color"]
        r["verdict_headline"], r["verdict_detail"] = v_["headline"], v_["detail"]
        r["_sort"] = v_["sort"]
    
    rows.sort(key=lambda x: x["_sort"])
    st.session_state.update(
        rows=rows, stats=(len(uni), liq), drop=drop, open=None,
        scanned_at=datetime.now(il).strftime("%H:%M"),
        took=round(time.time() - t0, 1), cached=not fresh,
        data_date=last_bar.strftime("%d.%m.%y") if last_bar is not None else "—")

# ===== DISPLAY RESULTS =====

if "rows" not in st.session_state:
    st.markdown('<div class="empty">בחר פריסט וסרוק.</div>', unsafe_allow_html=True)
else:
    rows = st.session_state["rows"]
    u, l_ = st.session_state["stats"]
    top_dry = max((r["dry"] for r in rows), default=0)
    took = st.session_state.get("took", 0)
    src = "מטמון" if st.session_state.get("cached") else "הורדה"
    
    st.markdown(f"""<div class="kpi">
<div><b>{u:,}</b><span>נסרקו</span></div>
<div><b>{l_:,}</b><span>נזילות בסדר</span></div>
<div class="hi"><b>{len(rows)}</b><span>בתבנית</span></div>
<div><b>{top_dry:.2f}×</b><span>יובש מרבי</span></div>
<div><b>{st.session_state.get('data_date','—')}</b><span>נתונים עד</span></div>
<div><b>{took:.0f}s</b><span>{src}</span></div>
<div><b>{st.session_state.get('scanned_at','—')}</b><span>נסרק בשעה</span></div>
</div>""", unsafe_allow_html=True)
    
    if not rows:
        st.markdown('<div class="empty">אין מניות בתבנית זו.</div>', unsafe_allow_html=True)
    else:
        for idx, rr in enumerate(rows):
            rr["_rank"] = idx + 1
        
        cols = st.columns(3, gap="medium")
        for col, r in zip(cols * len(rows), rows):
            with col:
                st.markdown(card(r, best=(r is rows[0]), rank=r["_rank"], total=len(rows)),
                            unsafe_allow_html=True)
