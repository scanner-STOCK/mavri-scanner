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
import os
import time
import shutil
import tempfile
import importlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

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

# Hard ceiling on yfinance's own internal threads per download call.
# threads=True lets yfinance size its pool from os.cpu_count(), which on a
# shared cloud host reports the HOST's cores — dozens — not the handful this
# container is actually allowed. Multiplied by our own worker pool, a single
# backtest tried to open ~360 threads and the platform refused with
# "RuntimeError: can't start new thread". An explicit integer keeps the total
# at (our workers x 6), which stays well inside any container limit.
YF_THREADS = 10

# yf.download is NOT thread-safe. Its results live in one module-level dict,
# yfinance.shared._DFS, which every call RESETS on entry (`shared._DFS = {}`)
# and then polls until it holds as many entries as that call requested. Two
# calls in flight at once wipe each other's results mid-download — and a call
# whose entries were wiped can poll forever. This is a known yfinance issue
# ("yfinance.download is not thread-safe", GitHub #2557).
#
# That single fact explains the whole history of this scanner's data problems:
# 12% missing, then 52%, then 100%, and the page that "never finished loading".
# Each was the same race with different timing.
#
# The fix is to never run two downloads at once: every call goes through one
# process-wide lock, and parallelism comes from yfinance's OWN internal threads
# (YF_THREADS), which is how the library is designed to be used. The lock is
# stored on the yfinance module object so it survives Streamlit reruns and is
# shared by every browser session in the process — sessions on Streamlit Cloud
# run as threads of one process and share the same yfinance globals.
if not hasattr(yf, "_mavri_lock"):
    yf._mavri_lock = threading.Lock()


# ---- diagnostics ---------------------------------------------------------
# Every download failure used to be swallowed by a bare `except: return {}`,
# so a broken environment and a quiet market looked identical: "no data". The
# wrapper now records the real exception and yfinance's own per-ticker error
# messages, so the UI can show what actually went wrong. Stored on the yfinance
# module, like the lock, so it is process-wide.
def _yf_diag():
    d = getattr(yf, "_mavri_diag", None)
    if d is None:
        d = {"exc": None, "exc_when": None, "ticker_errs": {}, "fallback": 0,
             "calls": 0}
        yf._mavri_diag = d
    return d


def _yf_diag_reset():
    yf._mavri_diag = None


def _yf_note(e):
    d = _yf_diag()
    d["exc"] = f"{type(e).__name__}: {str(e)[:300]}"
    d["exc_when"] = time.strftime("%H:%M:%S")


def _yf_collect_ticker_errors():
    """yfinance does not raise for failed tickers — it files a message per
    ticker in yfinance.shared._ERRORS and returns empty columns. Harvest those
    right after each call (the next call resets the dict)."""
    try:
        sh = importlib.import_module("yfinance.shared")
        errs = dict(getattr(sh, "_ERRORS", {}) or {})
    except Exception:
        return
    if not errs:
        return
    te = _yf_diag()["ticker_errs"]
    for msg in list(errs.values())[:300]:
        m = str(msg)[:160]
        te[m] = te.get(m, 0) + 1
    if len(te) > 20:
        keep = sorted(te.items(), key=lambda x: -x[1])[:20]
        te.clear()
        te.update(keep)


# ---- private tz cache ------------------------------------------------------
# yfinance keeps a small SQLite cache of ticker timezones. If a process dies
# mid-write (as it did when the old code ran out of threads) that file can be
# left locked or corrupt, and from then on EVERY lookup fails instantly —
# before any network request. A private per-process directory sidesteps a
# damaged shared file, and the refresh button can replace it on demand.
def _yf_cache_dir(fresh=False):
    cur = getattr(yf, "_mavri_cache_dir", None)
    if cur and not fresh:
        return cur
    base = os.path.join(tempfile.gettempdir(), "mavri-yf")
    if cur and fresh:
        shutil.rmtree(cur, ignore_errors=True)
    path = os.path.join(base, f"{os.getpid()}-{int(time.time())}")
    try:
        os.makedirs(path, exist_ok=True)
        yf.set_tz_cache_location(path)
        yf._mavri_cache_dir = path
    except Exception:
        pass
    return getattr(yf, "_mavri_cache_dir", None)


if not getattr(yf, "_mavri_cache_dir", None):
    _yf_cache_dir()


def yf_download(*args, **kwargs):
    """The only way this app calls yf.download. Serialised; see above.

    Thread-ceiling fallback: if the process can no longer start threads
    (RuntimeError: can't start new thread — e.g. threads left stuck forever by
    the old race), a synchronous download (threads=False) starts none at all
    and still works. Slower, but data keeps flowing until the next reboot."""
    with yf._mavri_lock:
        d = _yf_diag()
        d["calls"] += 1
        try:
            df = yf.download(*args, **kwargs)
        except RuntimeError as e:
            _yf_note(e)
            if "thread" not in str(e).lower():
                raise
            d["fallback"] += 1
            kwargs["threads"] = False
            try:
                df = yf.download(*args, **kwargs)
            except Exception as e2:
                _yf_note(e2)
                raise
        except Exception as e:
            _yf_note(e)
            raise
        _yf_collect_ticker_errors()
        return df


def yf_hint(d, n_threads):
    """Turn the recorded evidence into one plain-language diagnosis."""
    top = sorted((d.get("ticker_errs") or {}).items(), key=lambda x: -x[1])[:3]
    blob = " ".join([d.get("exc") or ""] + [m for m, _ in top]).lower()
    if "rate" in blob or "too many" in blob or "429" in blob:
        return ("Yahoo חוסם זמנית בגלל כמות בקשות (Rate limit). זה לא באג בקוד — "
                "המתן 30-60 דקות ונסה שוב, עם פחות מניות לסריקה.")
    if "database" in blob or "sqlite" in blob or "operationalerror" in blob:
        return "מטמון פנימי של yfinance נפגם. לחץ על ↻ למעלה — הוא נמחק ונבנה מחדש."
    if "thread" in blob or d.get("fallback") or n_threads > 150:
        return ("התהליך הגיע לתקרת החוטים — שאריות מגרסה קודמת. Manage app → "
                "Reboot app מנקה את זה. עד אז ההורדה רצה במצב איטי בלי חוטים.")
    if not d.get("exc") and not top:
        return ("לא נרשמה שום שגיאה — Yahoo החזיר תשובות ריקות בלי הסבר. "
                "הרץ את בדיקת החיבור למטה.")
    return "השגיאה המדויקת כתובה למעלה — צלם ושלח."


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
    "מסחר יומי": dict(dry=1.10, bb="אחת מהשתיים", bbmode="בונוס", price=2.0, dv=8.0,
                      sh=0.0, atr=0.25, atrp=4.0, rise=12, age=(2, 20), retr=(25, 75),
                      rr=1.2, off=15, maxpx=80.0, rvol=1.5, legdv=30.0),
    "התבנית שלי": dict(dry=1.30, bb="רצועה תחתונה", bbmode="בונוס", price=4.0, dv=8.0,
                       sh=0.0, atr=0.5, atrp=2.0, rise=8, age=(3, 45), retr=(20, 70),
                       rr=1.5, off=12),
    "רחב": dict(dry=1.05, bb="אחת מהשתיים", bbmode="בונוס", price=5.0, dv=8.0, sh=0.2,
                atr=0.4, atrp=1.5, rise=6, age=(2, 35), retr=(12, 88), rr=1.0, off=20),
    "מחמיר": dict(dry=1.45, bb="רצועה תחתונה", bbmode="חובה", price=15.0, dv=40.0, sh=1.0,
                  atr=1.5, atrp=3.0, rise=20, age=(3, 15), retr=(30, 65), rr=2.0, off=8),
}


_JUNK_RE = (r"(?<!common )\bunits?\b|\bwarrants?\b|\brights?\b|"
            r"\bpreferred\b|\bnotes? due\b|\bdebentures?\b|"
            r"\bsubordinated\b|% ")
# Not stock: SPAC units/warrants/rights, preferreds, notes. Two deliberate
# exceptions, both verified against the live NASDAQ file:
#   "common units"  — MLPs such as ARLP and DMLP trade exactly like shares.
#   "depositary"    — ADRs (BABA, NIO, XPEV...) are "American Depositary
#                     Shares"; the pattern never mentions the word at all.

DIR_SOURCES = [
    ("NASDAQ", "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
     "Symbol", "ETF", "Test Issue"),
    ("NYSE", "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
     "ACT Symbol", "ETF", "Test Issue"),
]


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_dir_file(url, sym_col, etf_col, test_col):
    """One directory file, parsed to common-stock symbols.

    RAISES on failure instead of returning a partial list — deliberately.
    st.cache_data does not cache a call that raises, so a timeout is retried on
    the next scan. The old code swallowed the error and returned whatever it
    had, and that half-market list was then cached for 24 hours: one slow
    response from nasdaqtrader.com and the scanner silently ran on ~2,500
    names instead of ~6,000 until the next day."""
    last = None
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=25)
            r.raise_for_status()
            txt = r.text
            if sym_col not in txt[:300]:
                raise ValueError("unexpected payload")
            df = pd.read_csv(io.StringIO(txt), sep="|")
            df = df[df[test_col].astype(str).str.upper() != "Y"]
            if etf_col in df.columns:
                df = df[df[etf_col].astype(str).str.upper() != "Y"]
            if "Security Name" in df.columns:
                nm = df["Security Name"].astype(str).str.lower()
                df = df[~nm.str.contains(_JUNK_RE, regex=True, na=False)]
            syms = [str(x).strip().upper().replace(".", "-") for x in df[sym_col]]
            syms = [x for x in syms if x.isascii() and 1 <= len(x) <= 5
                    and x.replace("-", "").isalpha()]
            if len(syms) < 800:
                raise ValueError(f"only {len(syms)} symbols — truncated download?")
            return syms
        except Exception as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{url}: {last}")


def fetch_full_market():
    """Both exchange files. Returns (symbols, per-source counts). A source that
    failed reports 0 and is retried on the next scan, because its failure was
    never cached."""
    out, counts = [], {}
    for name, url, sc, ec, tc in DIR_SOURCES:
        try:
            got = _fetch_dir_file(url, sc, ec, tc)
            out += got
            counts[name] = len(got)
        except Exception:
            counts[name] = 0
    return out, counts


def build_universe(limit):
    """Universe = curated liquid list FIRST, then NASDAQ Trader's official
    directory (~7,000 common stocks) for breadth.

    BUG FIX (critical): the previous version built the list as
    `full + BACKUP` and then cut it with `out[:limit]`. The NASDAQ directory
    arrives in ALPHABETICAL order, so `limit=1200` silently scanned roughly
    A→C only — NVDA, TSLA, PLTR, MSTR and most of the liquid market were never
    downloaded at all. The sidebar help text promised "ordered by liquidity",
    but nothing ever ordered it. That is why scans came back with few or zero
    matches no matter how the filters were relaxed.

    Now BACKUP (the hand-picked liquid names) is placed first, so any limit —
    even 200 — always covers the tradeable universe before reaching into the
    alphabetical long tail."""
    log = []
    full, counts = fetch_full_market()
    for k_, v_ in counts.items():
        log.append(f"{k_}:{v_:,}" if v_ else f"{k_}: ⚠ לא נטען — ינסה שוב בסריקה הבאה")

    # Liquid, hand-curated names first — these must never be cut by `limit`.
    got = [t.upper() for t in BACKUP] + list(full)
    log.append(f"נזילות תחילה:{len(set(BACKUP))}")

    seen, out = set(), []
    for t in got:
        if t not in seen:
            seen.add(t)
            out.append(t)
    total_available = len(out)
    if total_available > limit:
        log.append(f"זמינות בפועל {total_available:,} (מוגבל ל-{limit:,})")
    return out[:limit], log


# Parallelism without yf.download. Its thread-safety bug lives entirely in
# the global results dict (shared._DFS); Ticker.history downloads ONE ticker
# into its own DataFrame and never touches that dict. It is the exact call
# yf.download makes internally for each ticker — minus the broken part. So we
# run our own pool of Ticker.history calls: full concurrency, no race, no lock.
PER_TICKER_WORKERS = 24


def _diag_lock():
    lk = getattr(yf, "_mavri_diag_lock", None)
    if lk is None:
        lk = threading.Lock()
        yf._mavri_diag_lock = lk
    return lk


def _note_ticker_error(e):
    msg = f"{type(e).__name__}: {str(e)[:140]}"
    with _diag_lock():
        d = _yf_diag()
        d["exc"] = msg
        d["exc_when"] = time.strftime("%H:%M:%S")
        te = d["ticker_errs"]
        te[msg] = te.get(msg, 0) + 1
        if len(te) > 40:
            keep = sorted(te.items(), key=lambda x: -x[1])[:20]
            te.clear()
            te.update(keep)


def _history_one(t, period):
    """One ticker, daily bars, tz-naive index (same shape yf.download gave)."""
    with _diag_lock():
        _yf_diag()["calls"] += 1
    tk = yf.Ticker(t)
    try:
        try:
            d = tk.history(period=period, interval="1d", auto_adjust=False,
                           actions=False, raise_errors=True)
        except TypeError:       # older yfinance without raise_errors
            d = tk.history(period=period, interval="1d", auto_adjust=False,
                           actions=False)
    except Exception as e:
        _note_ticker_error(e)
        return None
    if d is None or len(d) == 0:
        return None
    try:
        if getattr(d.index, "tz", None) is not None:
            d.index = d.index.tz_localize(None)
        return d[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception as e:
        _note_ticker_error(e)
        return None


def _pool_map(fn, items, workers=PER_TICKER_WORKERS, on_progress=None):
    """Map fn over items on ONE continuous thread pool; if the process cannot
    start threads, finish the remainder inline instead of failing.

    One pool over the whole list, not a pool per batch: with batches, every
    batch waited for its slowest ticker before the next could start, leaving
    most workers idle at each boundary. on_progress(done_count) is called from
    the calling thread, so it may safely touch Streamlit elements."""
    out, done = {}, set()
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fn, it): it for it in items}
            for fut in as_completed(futs):
                it = futs[fut]
                try:
                    out[it] = fut.result()
                except Exception:
                    out[it] = None
                done.add(it)
                if on_progress and (len(done) % 25 == 0 or len(done) == len(futs)):
                    on_progress(len(done))
    except RuntimeError:
        with _diag_lock():
            _yf_diag()["fallback"] += 1
        for it in items:
            if it not in done:
                try:
                    out[it] = fn(it)
                except Exception:
                    out[it] = None
    return out


def _download_batch(tickers, period="1y", on_progress=None, workers=PER_TICKER_WORKERS):
    """Daily history for a list of tickers, 24 at a time."""
    need = 90 if period not in ("5d", "10d") else 3
    res = _pool_map(lambda t: _history_one(t, period), list(tickers),
                    workers=workers, on_progress=on_progress)
    return {t: d for t, d in res.items() if d is not None and len(d) >= need}


@st.cache_data(ttl=1800, max_entries=120, show_spinner=False)
def fetch(tickers, period="1y"):
    """Cached single-batch fetch. max_entries raised from 8 to 120: with a
    7,000-name universe the old ceiling evicted almost every batch before the
    scan finished, so nothing was ever actually cached and every rescan paid
    full download cost again."""
    return _download_batch(tuple(tickers), period)


def _price_cache(period):
    """Session-level price store for the parallel path.

    st.cache_data cannot be called from worker threads, so the concurrent
    downloader keeps its own store here. Without it the ↻ button would be
    meaningless — every scan would be a full fresh download whether you asked
    for one or not. Entries live for 30 minutes, the same TTL the cached
    `fetch` used."""
    key = f"_px_{period}"
    box = st.session_state.get(key)
    if box is None or (time.time() - box["t"]) > 1800:
        box = {"t": time.time(), "d": {}}
        st.session_state[key] = box
    return box["d"]


def clear_price_cache():
    for k in [k for k in list(st.session_state.keys())
              if k.startswith("_px_") or k == "_snap"]:
        del st.session_state[k]


def render_yf_diagnostics(key):
    """Shown whenever most of the universe came back empty. States the real
    error, a plain diagnosis, and offers a 3-ticker live connection test."""
    d = _yf_diag()
    n_thr = threading.active_count()
    ver = getattr(yf, "__version__", "?")
    lines = [f"yfinance {ver} · חוטים פעילים בתהליך: {n_thr} · "
             f"קריאות הורדה בריצה הזאת: {d.get('calls', 0)}"]
    if d.get("fallback"):
        lines.append(f"הורדות שרצו במצב חירום בלי חוטים: {d['fallback']}")
    if d.get("exc"):
        lines.append(f"שגיאה אחרונה ({d.get('exc_when')}): `{d['exc']}`")
    for m, c in sorted((d.get("ticker_errs") or {}).items(),
                       key=lambda x: -x[1])[:3]:
        lines.append(f"{c:,}× `{m}`")
    st.error("**אבחון הורדת נתונים**  \n" + "  \n".join(lines) +
             f"  \n\n**מה זה אומר:** {yf_hint(d, n_thr)}")
    if st.button("בדוק חיבור ל-Yahoo עכשיו (3 מניות)", key=f"yfdiag_{key}"):
        t0 = time.time()
        try:
            raw = yf_download(["AAPL", "MSFT", "NVDA"], period="5d", interval="1d",
                              group_by="ticker", auto_adjust=False, threads=False,
                              progress=False, timeout=15)
            ok = []
            for t in ("AAPL", "MSFT", "NVDA"):
                try:
                    dd = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                    if len(dd.dropna()):
                        ok.append(t)
                except Exception:
                    pass
            res = (f"התקבלו {len(ok)}/3 ({', '.join(ok) or '—'}) "
                   f"תוך {time.time() - t0:.1f} שניות.")
            if not ok:
                res += " " + yf_hint(_yf_diag(), threading.active_count())
        except Exception as e:
            res = f"נכשל: {type(e).__name__}: {str(e)[:200]}"
        st.session_state[f"yfdiag_res_{key}"] = res
    if st.session_state.get(f"yfdiag_res_{key}"):
        st.info("בדיקת חיבור: " + st.session_state[f"yfdiag_res_{key}"])


def run_batches(fn, batches, workers, on_each=None):
    """Run fn over batches on a small thread pool, and never crash on the
    platform's thread limit.

    If the host refuses to start a thread (RuntimeError: can't start new
    thread) — which happens on shared cloud containers when other sessions are
    busy — the remaining batches run one by one on the current thread instead.
    Slower, but the scan finishes instead of dying with a red traceback.
    Yields each batch's result dict."""
    results = []
    pending = list(batches)
    if workers <= 1:
        # With the download lock in place, extra workers would only queue on it.
        # Running inline needs no thread at all — which matters exactly when
        # the process is at its thread ceiling.
        for b in pending:
            try:
                r = fn(b) or {}
            except Exception:
                r = {}
            results.append(r)
            if on_each:
                on_each(r)
        return results
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {}
            for b in pending:
                futs[ex.submit(fn, b)] = b
            for fut in as_completed(futs):
                try:
                    r = fut.result() or {}
                except Exception:
                    r = {}
                results.append(r)
                pending.remove(futs[fut])
                if on_each:
                    on_each(r)
    except RuntimeError:
        for b in list(pending):
            try:
                r = fn(b) or {}
            except Exception:
                r = {}
            results.append(r)
            pending.remove(b)
            if on_each:
                on_each(r)
    return results


def prescreen_universe(tickers, min_price, min_dollar_vol_m, batch=200,
                       workers=1, on_progress=None):
    """Cheap first pass: 10 daily bars per ticker instead of a full year.

    WHY THIS EXISTS — this is what makes a 4,000-name scan possible at all.
    Downloading 252 daily bars for 4,000 tickers means parsing roughly a
    million rows, and unpacking that many MultiIndex frames is exactly what
    pegged the CPU and got the app throttled for 24 hours. But the first two
    filters the scan applies — minimum price and minimum dollar volume — only
    ever look at the last few bars. There is no reason to pay for a year of
    history on a name that is about to be rejected for trading 200k a day.

    So: pull 10 bars for everyone (about 25x less data), throw out everything
    illiquid, and let the expensive full-history download run only on the
    survivors. On a typical 4,000-name universe that is ~700 survivors, so the
    heavy stage shrinks by roughly 5x on top of the 25x saved here.

    Returns (survivor_tickers, price_map, drop_counts)."""
    out, drops = [], {}
    prices = {}
    tickers = list(tickers)

    # Liquidity snapshots are reused across scans within the session. Changing
    # a pattern slider does not change what a stock traded yesterday, yet every
    # rescan used to re-download 10 bars for the entire universe before doing
    # anything. That was the single slowest part of tuning filters: one minute
    # of network per slider nudge. Cached for 30 minutes, keyed per ticker.
    snap = st.session_state.get("_snap")
    if snap is None or (time.time() - snap["t"]) > 1800:
        snap = {"t": time.time(), "d": {}}
        st.session_state["_snap"] = snap
    store = snap["d"]

    cached = {t: store[t] for t in tickers if t in store}
    missing = [t for t in tickers if t not in store]

    for t, (px, dv) in cached.items():
        if px < min_price:
            drops["מחיר נמוך"] = drops.get("מחיר נמוך", 0) + 1
            continue
        if dv < min_dollar_vol_m * 1e6:
            drops["מחזור נמוך"] = drops.get("מחזור נמוך", 0) + 1
            continue
        out.append(t)
        prices[t] = px

    if not missing:
        if on_progress:
            on_progress(1.0, len(out))
        miss0 = len(tickers) - sum(drops.values()) - len(out)
        if miss0 > 0:
            drops["לא התקבלו נתונים"] = miss0
        return out, prices, drops

    def _run(names, nworkers, frac0, frac1):
        total = max(len(names), 1)

        def _p(k):
            if on_progress:
                on_progress(frac0 + (frac1 - frac0) * k / total, len(out) + k)

        return _download_light(names, on_progress=_p, workers=nworkers)

    # One continuous pool across every missing ticker.
    got = _run(missing, PER_TICKER_WORKERS, 0.0, 0.85)

    # Recovery pass at gentler concurrency for whatever came back empty —
    # mostly transient refusals, which a quieter second request gets through.
    retry = [t for t in missing if t not in got]
    if retry and len(retry) < 0.8 * len(missing):
        time.sleep(1.0)
        got.update(_run(retry, 8, 0.85, 1.0))

    store.update(got)
    for t, (px, dv) in got.items():
        if px < min_price:
            drops["מחיר נמוך"] = drops.get("מחיר נמוך", 0) + 1
            continue
        if dv < min_dollar_vol_m * 1e6:
            drops["מחזור נמוך"] = drops.get("מחזור נמוך", 0) + 1
            continue
        out.append(t)
        prices[t] = px
    missing = len(tickers) - sum(drops.values()) - len(out)
    if missing > 0:
        drops["לא התקבלו נתונים"] = missing
    return out, prices, drops


def _download_light(tickers, on_progress=None, workers=PER_TICKER_WORKERS):
    """10 bars per ticker reduced to (last_close, avg_dollar_volume). Scalars
    only — nothing heavy is kept, which is the point of the prescreen."""
    def one(t):
        d = _history_one(t, "10d")
        if d is None or len(d) < 3:
            return None
        px = float(d["Close"].iloc[-1])
        vol = float(d["Volume"].tail(10).mean())
        return (px, px * vol) if px > 0 and vol > 0 else None
    res = _pool_map(one, list(tickers), workers=workers, on_progress=on_progress)
    return {t: v for t, v in res.items() if v is not None}


def extended_hours_quote(ticker):
    """Latest pre-market / after-hours price against the regular-session close.

    yfinance exposes extended-hours bars through prepost=True on intraday
    intervals. This is only ever called on the handful of names that already
    matched the pattern — never on the scanned universe — because it is one
    network call per ticker.

    Returns dict(price, change_pct, session) or None."""
    try:
        d = yf_download(ticker, period="2d", interval="5m", prepost=True,
                        auto_adjust=False, progress=False, timeout=15,
                        threads=False)
        if d is None or len(d) == 0:
            return None
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d.dropna()
        if len(d) < 2:
            return None
        idx = d.index.tz_convert(NY) if d.index.tz is not None else d.index
        last_ts = idx[-1]
        mins = last_ts.hour * 60 + last_ts.minute
        if mins < 9 * 60 + 30:
            sess = "פרה-מרקט"
        elif mins >= 16 * 60:
            sess = "אפטר-מרקט"
        else:
            sess = "מסחר רגיל"
        # Regular-session close = last bar at or before 16:00 on the last day.
        reg = d[[(t.hour * 60 + t.minute) < 16 * 60 for t in idx]]
        if len(reg) == 0:
            return None
        ref = float(reg["Close"].iloc[-1])
        last = float(d["Close"].iloc[-1])
        if ref <= 0:
            return None
        return dict(price=round(last, 2),
                    change_pct=round((last / ref - 1) * 100, 2),
                    session=sess)
    except Exception:
        return None


def fetch_universe_parallel(tickers, period="1y", batch=150, workers=1,
                            on_progress=None):
    """Download the universe with a few batches in flight at once.

    TUNED FOR STREAMLIT COMMUNITY CLOUD: the free tier gives roughly one CPU,
    and it throttles apps that peg it. Ten worker threads each unpacking a
    MultiIndex DataFrame was enough to trip that throttle within a single scan.
    Three workers on larger batches move the same data with a fraction of the
    thread-scheduling and parsing overhead — slower by a few seconds, but the
    app keeps its CPU instead of getting capped for 24 hours."""
    tickers = list(tickers)
    store = _price_cache(period)

    data = {t: store[t] for t in tickers if t in store}
    missing = [t for t in tickers if t not in store]
    if not missing:
        if on_progress:
            on_progress(1.0, len(data))
        return data, len(tickers) - len(data)

    total = max(len(missing), 1)

    def _p(k):
        if on_progress:
            on_progress(k / total, len(data) + k)

    got = _download_batch(missing, period, on_progress=_p)
    data.update(got)
    store.update(got)
    no_data = len(tickers) - len(data)
    return data, no_data


@st.cache_data(ttl=900, max_entries=12, show_spinner=False)
def fetch_one(t, period, interval):
    """Single ticker, any interval. Tries two different Yahoo endpoints with
    backoff before giving up — the two methods sometimes succeed independently
    of each other when Yahoo is rate-limiting one path."""
    for attempt in range(3):
        try:
            d = yf_download(t, period=period, interval=interval, auto_adjust=False,
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
    """Fetch chart data for one ticker. fetch_one is cached and retries across
    two Yahoo endpoints, so this is the reliable path; we no longer keep price
    DataFrames in session_state because that grew memory without bound."""
    return fetch_one(r["ticker"], ranges[period_key], interval)


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
                bbu=(mid + 2 * sd).values, bbl=(mid - 2 * sd).values,
                sma200=cs.rolling(200).mean().values)


def reversal_candle(A, i):
    """Classify the signal bar. A small green candle is not a reversal — we
    require either a decisive bullish body closing in the upper part of its
    range, a true hammer (long lower wick, small body), or a bullish engulfing
    of the prior down bar. Returns (is_reversal, name)."""
    o, h, l, c = A["o"][i], A["h"][i], A["l"][i], A["c"][i]
    rng = h - l
    if rng <= 0:
        return False, "נר לא תקין"
    body = abs(c - o)
    body_pct = body / rng
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    closes_high = c >= l + 0.60 * rng

    if i >= 1:
        po, pc = A["o"][i - 1], A["c"][i - 1]
        if pc < po and c > o and o <= pc and c >= po:
            return True, "בליעה שורית"

    if rng > 0 and (lower_wick / rng) >= 0.40 and closes_high \
       and lower_wick >= max(1.5 * body, upper_wick):
        return True, "פטיש"

    if c > o and body_pct >= 0.15 and closes_high:
        return True, "נר שורי"

    return False, "אין נר היפוך"


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
    # Search for the peak ONLY inside the allowed recency window, instead of
    # finding the global high of the whole lookback and then rejecting it for
    # being old. This directly finds fresh local highs even when an older,
    # bigger high exists earlier in the window — which is exactly the recent
    # pullback we're looking for, not ancient history.
    hi_idx = max(0, n - 1 - C["age_lo"])
    lo_idx = max(0, n - 1 - C["age_hi"])
    if hi_idx <= lo_idx:
        return None, "טווח ימים לא תקין"
    window = cw[lo_idx:hi_idx]
    if len(window) == 0:
        return None, "אין טווח לשיא"
    peak = lo_idx + int(np.argmax(window))
    age = n - 1 - peak
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

    # --- "כמה כסף נכנס בזינוק" -------------------------------------------
    # Volume ratios alone say nothing about whether real money moved: a stock
    # that trades 80k shares a day can post a 5x volume spike and still be a
    # $400k day that no day trader can work with. This measures the average
    # DOLLAR volume across the up-leg itself, so the filter can demand actual
    # capital inflow during the move rather than a ratio against a tiny base.
    leg_px = cw[low_i:peak + 1]
    leg_dollars = float(np.mean(leg_v * leg_px)) if len(leg_px) == len(leg_v) else 0.0
    if C.get("leg_dv_min") and leg_dollars < C["leg_dv_min"] * 1e6:
        return None, "מעט כסף נכנס בזינוק"
    dry = float(leg_v.mean() / pull_v.mean())
    if dry < C["dry_min"]:
        return None, "הנפח לא התייבש"
    tags = []
    if C["bb"] != "off":
        tags = bb_hits(A, i, C["bb_look"], C["bb"])
        if C["bb_hard"] and not tags:
            return None, "אין נר על בולינג׳ר"
    sup = float(lw[peak:].min())
    off_low = (px / sup - 1) * 100 if sup > 0 else 999
    if off_low > C["off_max"]:
        return None, "רחוקה מדי מהתמיכה"
    a = float(A["atr"][i])
    if not np.isfinite(a) or a <= 0:
        return None, "אין ATR"
    if a < C["atr_abs"]:
        return None, "ATR נמוך מדי"
    atr_pct = a / px * 100
    if atr_pct < C["atr_pct"]:
        return None, "תנודתיות נמוכה"
    rvol = float(v[i - 2:i + 1].mean() / v[max(0, i - 49):i + 1].mean())
    # A name trading at half its normal volume is not waking up, whatever the
    # chart shape looks like. This is the filter that keeps sleepy large caps
    # out of a day-trading list.
    if C.get("rvol_min") and rvol < C["rvol_min"]:
        return None, "נפח יחסי נמוך (RVOL)"
    if C.get("max_px") and px > C["max_px"]:
        return None, "מחיר גבוה מדי"
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
                leg_dollars=round(leg_dollars, 0),
                off_low=round(off_low, 1), above200=above200,
                rs=(round(rs, 1) if rs is not None else None),
                candle=rev_name, is_rev=bool(is_rev),
                bb=tags, spark=[float(x) for x in c[max(0, i - 59):i + 1]]), None


@st.cache_data(ttl=86400, max_entries=200, show_spinner=False)
def days_to_earnings(ticker):
    """Trading-unaware calendar days until the next known earnings date, or None
    if unavailable. Only ever called on the small matched subset — never on the
    full scanned universe — because this is one extra network call per ticker."""
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
    """Wrap a numeric/LTR fragment so it doesn't get reordered inside RTL text."""
    return f'<bdi dir="ltr">{txt}</bdi>'


def fast_rsi(prices, period=14):
    """Fast RSI calculation - vectorized, handles edge cases"""
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-period-1:])
    gains = deltas[deltas > 0].mean() if np.any(deltas > 0) else 0.0
    losses = -deltas[deltas < 0].mean() if np.any(deltas < 0) else 0.0
    if losses == 0:
        return 100.0 if gains > 0 else 50.0
    rs = gains / losses
    return float(100 - (100 / (1 + rs)))


def split_volume(high, low, close, volume):
    """
    Estimate buy vs sell volume using close position within daily bar.
    BUG FIX: Original core_breakdown checked only total volume.
    This measures buying pressure by close positioning:
    - Close near top of bar (>70% of range) = buying dominated
    - Close near bottom (<30%) = selling dominated
    Result: more actionable volume profile for entry confirmation.
    """
    rng = high - low
    rng = np.where(rng == 0, 1e-10, rng)
    close_pos = (close - low) / rng
    
    buy_vol = volume * np.where(close_pos >= 0.7, 1.5 + close_pos * 0.5, 1.0)
    sell_vol = volume * np.where(close_pos <= 0.3, 1.5 - close_pos, 1.0)
    
    return buy_vol, sell_vol


def day_trade_plan(r):
    """Tight ATR-based plan for a max-4-day hold, separate from the swing TP1/2/3."""
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
    """Support broken, then reclaimed: potential failed-breakdown reversal.
    
    UPGRADES (v2):
    - RSI Oversold confirmation (<40)
    - Buy Volume Profile (not just total volume)
    - Better threshold tuning for 4000-stock scans
    """
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
    resistance = float(np.max(hw[pre_start:pre_end]))
    if support <= 0:
        return None, "תמיכה לא תקינה"

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

    # === RSI Oversold Check (NEW) ===
    rsi_val = fast_rsi(c)
    if rsi_val > 40:
        return None, f"RSI לא oversold"

    # === Buy Volume Profile (NEW) ===
    buy_vol, _ = split_volume(hw, lw, cw, vw)
    base_v = float(np.mean(vw[max(0, pre_start - 30):pre_start])) if pre_start > 5 \
             else float(np.mean(vw[:max(pre_start, 1)]))
    
    reclaim_buy = buy_vol[max(0, n - C["break_win"]):]
    if len(reclaim_buy) == 0 or reclaim_buy.mean() <= 0:
        return None, "אין נתוני נפח קונים"
    
    buy_pressure = float(reclaim_buy.mean() / base_v)
    if buy_pressure < 0.75:
        return None, "אין לחץ קנייה בתפיסה"
    
    brk_v = vw[pre_end:]
    if base_v <= 0 or len(brk_v) == 0:
        return None, "אין נתוני נפח"
    brk_spike = float(np.max(brk_v) / base_v)

    rvol = float(v[i - 2:i + 1].mean() / v[max(0, i - 49):i + 1].mean())
    if C.get("rvol_min") and rvol < C["rvol_min"]:
        return None, "נפח יחסי נמוך (RVOL)"
    if C.get("max_px") and px > C["max_px"]:
        return None, "מחיר גבוה מדי"
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
                rsi=round(rsi_val, 1), buy_vol_ratio=round(buy_pressure, 2),
                above200=above200, rs=(round(rs, 1) if rs is not None else None),
                candle=rev_name, is_rev=bool(is_rev),
                age=0, rise=round(brk_pct, 1), leg_bars=C["break_win"],
                dry=round(brk_spike, 2), retrace=round(reclaim_dist, 1), bb=[],
                off_low=round(reclaim_dist, 1),
                peak_px=round(resistance, 2), base=round(brk_low, 2),
                spark=[float(x) for x in c[max(0, i - 59):i + 1]]), None


def core_parabolic(A, i, C):
    """Parabolic run, deep correction, money still parked: the second-leg setup.

    This is a different animal from the ordinary pullback and cannot be
    expressed with the same knobs. A stock that runs 300% over three months and
    then corrects for six weeks fails core() on every axis: the leg is far
    longer than leg_max allows, the peak is far older than age_hi allows, and
    the retracement is measured against a base so distant that the percentages
    stop meaning the same thing.

    What defines the setup instead:
      1. A move large enough to be a repricing, not a swing (default 60%+).
      2. A correction deep enough to shake out the late buyers.
      3. Price still far above where the move started — the capital that came
         in did NOT all leave. This is the condition that separates a base for
         a second leg from a failed pump round-tripping to zero.
      4. Volume drying through the correction: distribution finished.
      5. A reversal bar at the current support.

    Targets run back toward the old high rather than out to new ground, because
    the realistic first objective for a second leg is a retest of the peak."""
    c, o, h, l, v = A["c"], A["o"], A["h"], A["l"], A["v"]
    # 90 bars, not 120: a stock that IPO'd nine months ago and then ran 300%
    # is a prime candidate for this pattern, and the stricter floor excluded
    # exactly that profile.
    if i < 90:
        return None, "היסטוריה קצרה"
    s_ = max(0, i - C["pwin"] + 1)
    cw, ow, hw, lw, vw = c[s_:i + 1], o[s_:i + 1], h[s_:i + 1], l[s_:i + 1], v[s_:i + 1]
    n = len(cw)
    if n < 90:
        return None, "היסטוריה קצרה"

    # --- locate the blow-off high inside the allowed recency band -----------
    hi_idx = max(0, n - 1 - C["page_lo"])
    lo_idx = max(0, n - 1 - C["page_hi"])
    if hi_idx <= lo_idx:
        return None, "טווח ימים לא תקין"
    peak = lo_idx + int(np.argmax(hw[lo_idx:hi_idx]))
    age = n - 1 - peak
    peak_px = float(hw[peak])

    # --- the base the run started from --------------------------------------
    run_start = max(0, peak - C["prun_max"])
    if peak - run_start < 10:
        return None, "אין מקום לריצה"
    base_i = run_start + int(np.argmin(lw[run_start:peak]))
    base_px = float(lw[base_i])
    if base_px <= 0:
        return None, "בסיס לא תקין"
    run_bars = peak - base_i
    if run_bars < C["prun_min"]:
        return None, "הריצה קצרה מדי"

    rise = (peak_px / base_px - 1) * 100
    if rise < C["prise_min"]:
        return None, "הפריצה קטנה מדי"

    px = float(cw[-1])
    span = peak_px - base_px
    if span <= 0:
        return None, "אין פריצה"
    retr = (peak_px - px) / span * 100
    if retr < C["pretr_lo"]:
        return None, "עוד לא תיקנה מספיק"
    if retr > C["pretr_hi"]:
        return None, "מחקה את כל הפריצה"

    # --- did the money actually stay? ---------------------------------------
    hold = (px / base_px - 1) * 100
    if hold < C["phold_min"]:
        return None, "הכסף יצא — חזרה לבסיס"

    # --- volume: heavy on the run, dry on the correction --------------------
    run_v = vw[base_i:peak + 1]
    pull_v = vw[peak:]
    if len(run_v) == 0 or len(pull_v) == 0 or pull_v.mean() <= 0:
        return None, "אין נתוני נפח"
    dry = float(run_v.mean() / pull_v.mean())
    if dry < C["pdry_min"]:
        return None, "הנפח לא התייבש"
    pre = vw[max(0, base_i - 40):base_i]
    base_v = float(pre.mean()) if len(pre) >= 5 else float(run_v.mean())
    spike = float(run_v.max() / base_v) if base_v > 0 else 1.0
    run_dollars = float(np.mean(run_v * cw[base_i:peak + 1]))
    if C.get("pdv_min") and run_dollars < C["pdv_min"] * 1e6:
        return None, "מעט כסף נכנס בפריצה"

    # Independent gates. A parabolic runner is usually a sub-$5 micro-cap, and
    # during its correction the relative volume often falls BELOW normal — the
    # day-trading floors (price >= $2, RVOL >= 1.5) would reject exactly the
    # names this pattern exists to find. So it carries its own thresholds and
    # ignores the day-trading ones entirely.
    rvol = float(v[i - 2:i + 1].mean() / v[max(0, i - 49):i + 1].mean())
    if C.get("prvol_min") and rvol < C["prvol_min"]:
        return None, "נפח יחסי נמוך (RVOL)"
    if C.get("pmin_px") and px < C["pmin_px"]:
        return None, "מחיר נמוך"
    if C.get("pmax_px") and px > C["pmax_px"]:
        return None, "מחיר גבוה מדי"

    a = float(A["atr"][i])
    if not np.isfinite(a) or a <= 0:
        return None, "אין ATR"
    if a < C["atr_abs"]:
        return None, "ATR נמוך מדי"
    atr_pct = a / px * 100
    if atr_pct < C["atr_pct"]:
        return None, "תנודתיות נמוכה"

    sup = float(lw[peak:].min())
    off_low = (px / sup - 1) * 100 if sup > 0 else 999
    if off_low > C["poff_max"]:
        return None, "רחוקה מדי מהתמיכה"

    is_rev, rev_name = reversal_candle(A, i)
    if C.get("need_trig") and not is_rev:
        return None, "אין נר היפוך"

    sma200 = float(A["sma200"][i]) if np.isfinite(A["sma200"][i]) else None
    above200 = None if sma200 is None else (px > sma200)
    rs = None
    if C.get("spy_ret21") is not None and i >= 21:
        rs = (c[i] / c[i - 21] - 1) * 100 - C["spy_ret21"]

    entry = max(px, float(h[i])) + .05 * a
    # Structural stop, NOT an ATR multiple. On a name that just ran 300% the
    # ATR is enormous, and "entry - 3*ATR" put the stop ~75% below entry —
    # inflating risk until the R:R test rejected every real candidate. The
    # swing low since the peak is the level that actually invalidates the
    # setup; ATR only pads it slightly. A hard risk ceiling keeps the position
    # sizeable even when the support sits far away.
    stop = sup - .25 * a
    max_risk = entry * (C.get("pmax_risk", 25) / 100.0)
    if entry - stop > max_risk:
        stop = entry - max_risk
    risk = entry - stop
    if risk <= 0:
        return None, "סטופ לא תקין"
    # A second leg aims back at the old high, not at blue sky.
    gap = peak_px - entry
    tp1 = entry + gap * .35
    tp2 = entry + gap * .70
    tp3 = peak_px
    rr = (tp1 - entry) / risk
    if rr < C["rr_min"]:
        return None, "אין מרווח ל-TP1"

    return dict(kind="parabolic", dry=round(dry, 2), spike=round(spike, 2),
                rise=round(rise, 1), leg_bars=int(run_bars),
                retrace=round(retr, 1), price=round(px, 2),
                entry=round(entry, 2), stop=round(stop, 2), tp1=round(tp1, 2),
                tp2=round(tp2, 2), tp3=round(tp3, 2), rr=round(float(rr), 2),
                rr3=round(float((tp3 - entry) / risk), 2),
                risk_share=round(float(risk), 2),
                to_entry=round((entry / px - 1) * 100, 2), rvol=round(rvol, 2),
                atr=round(a, 2), atr_pct=round(atr_pct, 1), age=int(age),
                support=round(sup, 2), peak_px=round(peak_px, 2),
                base=round(base_px, 2), hold_pct=round(hold, 1),
                leg_dollars=round(run_dollars, 0),
                off_low=round(off_low, 1), above200=above200,
                rs=(round(rs, 1) if rs is not None else None),
                candle=rev_name, is_rev=bool(is_rev), bb=[],
                spark=[float(x) for x in c[max(0, i - 119):i + 1]]), None


def why_parabolic(r):
    return (f"זינקה {r['rise']:.0f}% ב-{r['leg_bars']} ימים, תיקנה "
            f"{r['retrace']:.0f}% מהמהלך בנפח קטן פי {r['dry']:.2f}, "
            f"ועדיין {r['hold_pct']:.0f}% מעל הבסיס — הכסף לא יצא. "
            f"יעד אחרון: חזרה לשיא ב-{r['peak_px']:.2f}")


def why_breakdown(r):
    return (f"שברה תמיכה ב-{r['brk_pct']:.1f}% בנפח פי {r['spike']:.1f}, "
            f"תפסה אותה מחדש (עכשיו {r['reclaim_pct']:+.1f}% מעליה), "
            f"התנגדות קרובה ב-{r['resistance']:.2f}")


def why_he(r):
    return (f"עלתה {r['rise']:.0f}% ב-{r['leg_bars']} ימים בנפח פי {r['spike']:.1f}, "
            f"תיקנה {r['retrace']:.0f}% בנפח קטן פי {r['dry']:.2f}, "
            f"ונתמכת ב-{r['support']:.2f}")


def spy_return_lookup(spy_df):
    """21-day rolling return of SPY, indexed by date, for point-in-time lookup
    during the walk-forward backtest (as-of, tolerant of small calendar gaps)."""
    sc = spy_df["Close"].astype(float)
    ret = (sc / sc.shift(21) - 1) * 100
    return ret.dropna()


def backtest_walk(d, C, CB, ptype, spy_ret_series, hold=25, cooldown=10,
                  slip_bps=5.0, commission=0.005, CPX=None):
    """Walk a single ticker's full history bar by bar, calling the SAME core()/
    core_breakdown() the live scanner uses — so a backtest 'pass' means exactly
    what a live match means, nothing reimplemented separately."""
    A = prep(d)
    n = len(A["c"])
    dates = d.index
    out = []
    last = -999
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
        if ptype in ("זינוק ותיקון לתמיכה", "כל התבניות"):
            r, _ = core(A, i, Ci)
            if r is not None:
                found = r
        if found is None and ptype in ("פריצה פרבולית ותיקון", "כל התבניות") and CPX:
            rp, _ = core_parabolic(A, i, dict(CPX, spy_ret21=spy_r21))
            if rp is not None:
                found = rp
        if found is None and ptype in ("שבירת תמיכה ותפיסה מחדש", "כל התבניות"):
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
                # A gap above the buy-stop fills at the open, not at the trigger.
                fill_px = max(entry, float(o[j])) * (1 + slip)
                filled = True
            risk = fill_px - stop
            if risk <= 0:
                break
            hit_stop = l[j] <= stop
            hit_tp1 = h[j] >= tp1
            if hit_stop:
                # Same-bar stop and target: daily bars can't tell which came
                # first, so assume the stop. Conservative on purpose.
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
    """
    Ranking grounded in the walk-forward backtest, not an invented composite score.
    The ONLY range with repeated, large-sample positive expectancy was dry-up 1.3-1.6
    (+0.25R to +0.39R across multiple runs, 100+ trades). Below 1.3 was negative in
    two separate runs. Above 1.6 was negative in the one large-sample run we have
    (-0.09R, n=28). This ranks by closeness to the PROVEN range, not by raw magnitude —
    a high dry-up number is not automatically better.

    Within a quality tier, freshness breaks near-ties: a pullback whose peak was
    20 days ago and one whose peak was 41 days ago can have almost identical dry-up
    (1.43 vs 1.45) — a trivial, meaningless difference — while looking completely
    different on a chart. Distance-from-center is bucketed to 0.05 so noise-level
    differences don't decide the ranking; age (ascending, fresher first) does instead.
    """
    age = r.get("age", 0)
    if r.get("kind") == "parabolic":
        hold = r.get("hold_pct", 0)
        rise = r.get("rise", 0)
        # Ranked by how much of the original move is still held, because that
        # is the whole thesis: capital that stayed in is what funds a second
        # leg. Not backtested — flagged honestly as experimental.
        return dict(tier="P", color="#4E8FB0", sort=(1, -hold / 100, age),
                    headline=f"פריצה {rise:.0f}% · מחזיקה {hold:.0f}% מעל הבסיס",
                    detail=f"המנייה עשתה מהלך של {rise:.0f}% ותיקנה "
                           f"{r.get('retrace', 0):.0f}% ממנו, אבל היא עדיין "
                           f"{hold:.0f}% מעל הנקודה שממנה יצאה — כלומר רוב הכסף "
                           f"שנכנס במהלך עוד שם. זה ההבדל בין בסיס לרגל שני לבין "
                           f"פאמפ שחוזר לאפס. חשוב: התבנית הזאת לא עברה בדיקה "
                           f"היסטורית במערכת, בניגוד ליובש הנפח. היא הגיונית "
                           f"טכנית — אין לה עדיין הוכחה סטטיסטית.")
    if r.get("kind") == "breakdown":
        return dict(tier="?", color="#8A6FB0", sort=(2, 0, age),
                    headline="לא נבדק היסטורית",
                    detail="תבנית שבירת התמיכה לא עברה עדיין בדיקה היסטורית (backtest) "
                           "כמו תבנית הזינוק-ותיקון. המספרים כאן חושבו נכון, אבל אין לנו "
                           "עדיין הוכחה סטטיסטית שהתבנית הזאת רווחית. התייחס אליה בזהירות "
                           "יתרה עד שתיבדק.")
    dry = r["dry"]
    dist = abs(dry - 1.45)
    dist_bucket = round(dist * 20) / 20  # nearest 0.05 — kills 1.43-vs-1.45-style noise
    if 1.30 <= dry <= 1.60:
        return dict(tier="A", color="#2FBF8F", sort=(0, dist_bucket, age),
                    headline=f"יובש נפח {dry:.2f}× — בדיוק בטווח המוכח",
                    detail=f"יובש הנפח ({dry:.2f}×) נמצא בטווח 1.30–1.60, האזור היחיד שהראה "
                           f"תוחלת חיובית עקבית בבדיקה ההיסטורית (כ-0.25 עד 0.39R לעסקה, "
                           f"על יותר מ-100 עסקאות בשתי ריצות נפרדות). בין מניות ברמה הזאת "
                           f"המערכת מעדיפה את התיקון הטרי ביותר — {age} ימים מהשיא כאן.")
    if (1.10 <= dry < 1.30) or (1.60 < dry <= 2.20):
        return dict(tier="B", color="#D4A64B", sort=(1, dist_bucket, age),
                    headline=f"יובש נפח {dry:.2f}× — מחוץ לטווח המוכח",
                    detail=f"יובש הנפח ({dry:.2f}×) קרוב לטווח המוכח (1.30–1.60) אך מחוצה לו. "
                           f"אין לנו נתון היסטורי חד-משמעי על הטווח הזה — בריצה אחת הוא "
                           f"הראה תוצאה שלילית, במדגם קטן מדי כדי לקבוע. אפשרי, אך פחות בטוח.")
    return dict(tier="C", color="#E0605F", sort=(2, dist_bucket, age),
                headline=f"יובש נפח {dry:.2f}× — רחוק מהטווח שנבדק",
                detail=f"יובש הנפח ({dry:.2f}×) רחוק מהטווח שהוכח (1.30–1.60). "
                       f"{'מתחת לטווח' if dry < 1.10 else 'מעל לטווח'} — במקום שבו בדקנו "
                       f"היסטורית, התוצאה הייתה שלילית. מספר גבוה או נמוך יותר אינו "
                       f"אוטומטית טוב יותר; זה בדיוק הטעות שגילינו והפכה אותנו לזהירים.")


def explain_ai(r, rank=None, total=None):
    """A full written explanation combining every factor we actually computed —
    no invented score, just the numbers put into plain sentences."""
    parts = []
    if rank is not None and total is not None:
        parts.append(f"דירוג #{rank} מתוך {total} מניות שנמצאו בסריקה הזאת.")

    reclaim_txt = ltr(f"{r.get('reclaim_pct', 0):+.1f}%")
    to_entry_txt = ltr(f"{r['to_entry']:+.1f}%")

    if r.get("kind") == "parabolic":
        parts.append(
            f"{r['ticker']} עשתה מהלך של {r['rise']:.0f}% מ-{r['base']:.2f} "
            f"עד {r['peak_px']:.2f} על פני {r['leg_bars']} ימי מסחר, ואז תיקנה "
            f"{r['retrace']:.0f}% מהמהלך בנפח שהתייבש פי {r['dry']:.2f}. "
            f"הנתון המכריע כאן: היא נסחרת עכשיו {r['hold_pct']:.0f}% מעל הבסיס "
            f"שממנו יצאה. כלומר רוב הכסף שנכנס במהלך לא יצא — מי שקנה מוקדם "
            f"עדיין ברווח ואין לו סיבה לברוח, ומי שקנה בשיא כבר נשטף. "
            f"זו ההגדרה של בסיס לרגל שני, להבדיל מפאמפ שמתרוקן. "
            f"היעד האחרון הוא חזרה לשיא ב-{r['peak_px']:.2f}, לא שטח חדש. "
            f"התבנית הזאת לא נבדקה היסטורית במערכת — התייחס בזהירות.")
    elif r.get("kind") == "breakdown":
        parts.append(
            f"{r['ticker']} שברה תמיכה שנקבעה על פני כחודש בשיעור של {r['brk_pct']:.1f}%, "
            f"בנפח שהגיע לפי {r['spike']:.1f} מהרגיל, ואז תפסה אותה מחדש — כרגע נסחרת "
            f"{reclaim_txt} מעליה. זו תבנית 'שייקאאוט' קלאסית: מי שקנה מוקדם מדי נבהל "
            f"ומכר בשבירה, ועכשיו המחיר מתאושש. חשוב לדעת: בניגוד לתבנית הזינוק-ותיקון, "
            f"את התבנית הזאת עדיין לא בדקנו היסטורית — היא הגיונית מבחינה טכנית, אבל "
            f"אין לנו אישוש סטטיסטי שהיא רווחית.")
    else:
        tier_txt = {
            "A": ("המצב הכי אמין שיש למערכת: יובש הנפח בדיוק בטווח שהוכח רווחי "
                  "(1.30–1.60) על יותר מ-100 עסקאות היסטוריות בשתי ריצות נפרדות."),
            "B": ("מצב בינוני: יובש הנפח קרוב לטווח המוכח אך לא בתוכו. אין נתון "
                  "היסטורי חד-משמעי על הטווח המדויק הזה."),
            "C": ("מצב לא מבוסס: יובש הנפח רחוק מהטווח שנבדק. איפה שכן נבדק "
                  "(מעל 1.6 או מתחת ל-1.1), התוצאה ההיסטורית הייתה שלילית."),
        }.get(r.get("tier"), "")
        parts.append(
            f"{r['ticker']} עלתה {r['rise']:.0f}% תוך {r['leg_bars']} ימי מסחר בנפח "
            f"שהגיע לשיא של פי {r['spike']:.1f} מהרגיל, ואז תיקנה {r['retrace']:.0f}% "
            f"מהעלייה על נפח קטן פי {r['dry']:.2f}. {tier_txt}")

    liq = []
    if r["atr_pct"] >= 3:
        liq.append(f"תנודתיות גבוהה יחסית (ATR {r['atr_pct']:.1f}%) — מתאימה לקצב מסחר יומי")
    elif r["atr_pct"] < 2:
        liq.append(f"תנודתיות נמוכה יחסית (ATR {r['atr_pct']:.1f}%) — התנועה עשויה להיות איטית")
    if r["rvol"] < 0.85:
        liq.append(f"נפח הימים האחרונים עדיין נמוך מהרגיל (RVOL {r['rvol']:.2f}) — "
                   f"ההוראה עוד לא הופעלה, אין עדיין אישור קונים")
    elif r["rvol"] >= 1.3:
        liq.append(f"נפח הימים האחרונים כבר גבוה מהרגיל (RVOL {r['rvol']:.2f}) — "
                   f"ייתכן שהתנועה כבר החלה")
    if r.get("candle") and r["candle"] != "אין נר היפוך":
        liq.append(f"נר האיתות הוא {r['candle']} — אישור היפוך ולא סתם נר ירוק")
    if r.get("bb"):
        liq.append(f"נגעה ברצועת בולינג׳ר {' ו'.join(r['bb'])}, אישור טכני נוסף")
    if liq:
        parts.append(" · ".join(liq) + ".")

    extra = []
    if r.get("above200") is True:
        extra.append("נסחרת מעל ה-SMA200, כלומר בתוך מגמת עלייה ארוכת טווח")
    elif r.get("above200") is False:
        extra.append("נסחרת מתחת ל-SMA200 — נגד המגמה הארוכה, סימן אזהרה קל")
    if r.get("rs") is not None:
        rs_txt = "חזקה מהמדד" if r["rs"] > 0 else "חלשה מהמדד"
        rs_abs_txt = ltr(f"{abs(r['rs']):.1f}%")
        extra.append(f"{rs_txt} ב-{rs_abs_txt} על פני החודש האחרון (RS)")
    if extra:
        parts.append(" · ".join(extra) + ". שים לב: שני הגורמים האלה — מגמה וחוזק "
                     "יחסי — לא נבדקו עדיין בבדיקה היסטורית במערכת הזאת, בניגוד "
                     "ליובש הנפח. הם היגיון מקובל במסחר, לא הוכחה.")
    if r.get("earn_days") is not None:
        parts.append(f"⚠ שים לב: דוח רבעוני צפוי בעוד {r['earn_days']} ימים — "
                     f"תוך כדי חלון ההחזקה המתוכנן. שקול לצאת לפני הדוח או להקטין גודל פוזיציה.")

    waiting = r["to_entry"] > 0.3
    wait_txt = "עדיין ממתינים לפריצה" if waiting else "קרוב מאוד להפעלה כרגע"
    parts.append(
        f"יחס סיכון-סיכוי ליעד הראשון: 1:{r['rr']:.1f}. נקודת הכניסה נמצאת "
        f"{to_entry_txt} מהמחיר הנוכחי — {wait_txt}.")

    if r.get("kind") == "parabolic":
        bottom = ("שורה תחתונה: מבנה קלאסי לרגל שני, אך ניסיוני — "
                  "הכסף עוד בפנים, וזה התנאי המרכזי. גודל פוזיציה מוקטן.")
    elif r.get("kind") == "breakdown":
        bottom = "שורה תחתונה: תבנית טכנית הגיונית אך ניסיונית — הזדמנות משנית, לא ראשית."
    elif r.get("tier") == "A":
        bottom = "שורה תחתונה: אחת ההזדמנויות המבוססות ביותר שהמערכת מצאה בסריקה הזאת."
    elif r.get("tier") == "B":
        bottom = ("שורה תחתונה: הזדמנות סבירה — שקול גודל פוזיציה קטן יותר "
                 "בהעדר אישוש היסטורי מלא לטווח הזה.")
    else:
        bottom = "שורה תחתונה: המערכת ממליצה בזהירות רבה — הנתונים רחוקים מכל מה שנבדק בעבר."
    parts.append(bottom)
    return " ".join(parts)


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


def card(r, best=False, watched=False, triggered=False, aging=False, rank=None, total=None):
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
    _rv = r.get("rvol", 0)
    _rvcls = "g" if _rv >= 1.5 else ""
    ch += f'<span class="chip {_rvcls}">RVOL {_rv:.2f}</span>'
    ch += f'<span class="chip">ATR {r["atr"]:.2f}$ ({r.get("atr_pct", 0):.1f}%)</span>'
    if r.get("leg_dollars"):
        ch += f'<span class="chip g">${r["leg_dollars"]/1e6:.0f}מ׳ בזינוק</span>'
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
    if r.get("undersized"):
        badges += ('<span class="badge" style="background:rgba(224,96,95,.16);'
                   'color:var(--sh)">סטופ רחב לתיק</span>')
    if r.get("eh_change") is not None:
        _ehc = r["eh_change"]
        _col = "var(--lg)" if _ehc > 0 else ("var(--sh)" if _ehc < 0 else "var(--dm)")
        _bg = ("rgba(47,191,143,.14)" if _ehc > 0
               else ("rgba(224,96,95,.14)" if _ehc < 0 else "rgba(147,168,178,.12)"))
        badges += (f'<span class="badge" style="background:{_bg};color:{_col}">'
                   f'{r.get("eh_session", "מורחב")} {_ehc:+.1f}%</span>')
    if r.get("eh_triggered"):
        badges += ('<span class="badge" style="background:rgba(212,166,75,.2);'
                   'color:var(--gd)">פרץ במסחר מורחב</span>')
    if r.get("kind") == "parabolic":
        badges = ('<span class="badge" style="background:rgba(78,143,176,.18);'
                  'color:#6FB2D9">פרבולי · רגל שני</span>') + badges
        head_val, head_lab = f"+{r['rise']:.0f}%", "גודל הפריצה"
        sub_lab, sub_val = "מעל הבסיס", f"{r.get('hold_pct', 0):.0f}%"
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
<div class="tl" style="border:0;padding-top:.22rem"><span>{sub_lab}: {sub_val}</span>
<span>הפעלה ב-<b>{ltr(f"{r['to_entry']:+.1f}%")}</b></span></div>
<div class="tw">{r['why']}</div></div>"""
    is_bd = r.get("kind") == "breakdown"
    if is_bd:
        badges = ('<span class="badge" style="background:rgba(138,111,176,.16);'
                  'color:#A98FD1">שבירה ותפיסה</span>') + badges
        head_val, head_lab = f"{r['reclaim_pct']:+.1f}%", "מעל תמיכה"
        sub_lab = "מאז השבירה" if r.get("break_win") is None else "ימי חלון"
        sub_val = f"{r.get('leg_bars', '—')}"
    else:
        head_val, head_lab = f"{r['dry']:.2f}×", "יובש נפח"
        sub_lab, sub_val = "ימים מהשיא", str(r["age"])
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
<div class="tl" style="border:0;padding-top:.22rem"><span>{sub_lab}: {sub_val}</span>
<span>הפעלה ב-<b>{ltr(f"{r['to_entry']:+.1f}%")}</b></span></div>
<div class="tw">{r['why']}</div></div>"""


@st.cache_data(ttl=300, max_entries=10, show_spinner=False)
def live(tickers):
    """Fresh-ish quotes for the watchlist. 60-second cache."""
    out = {}
    if not tickers:
        return out
    try:
        raw = yf_download(list(tickers), period="1mo", interval="1d", group_by="ticker",
                          auto_adjust=False, threads=YF_THREADS, progress=False, timeout=12)
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
    return ",".join(f"{i['ticker']}:{i['entry']:.2f}:{i['stop']:.2f}:{i['tp1']:.2f}:"
                    f"{i['tp2']:.2f}:{i['tp3']:.2f}:{int(i['shares'])}" for i in items)


def wl_decode(txt):
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
    if "WL" not in st.session_state:
        st.session_state["WL"] = wl_decode(st.query_params.get("w", ""))
    return st.session_state["WL"]


def wl_save():
    st.query_params["w"] = wl_encode(st.session_state["WL"])


def wl_toggle(r):
    wl = wl_load()
    if any(x["ticker"] == r["ticker"] for x in wl):
        st.session_state["WL"] = [x for x in wl if x["ticker"] != r["ticker"]]
    else:
        st.session_state["WL"] = wl + [dict(ticker=r["ticker"], entry=r["entry"],
                                            stop=r["stop"], tp1=r["tp1"], tp2=r["tp2"],
                                            tp3=r["tp3"], shares=r["shares"])]
    wl_save()


def wl_status(item, d):
    """Where the trade stands, using daily bars since it was saved."""
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
                    r=(px - item["entry"]) / risk, note="למכור שליש, סטופ לכניסה")
    tp1_pct = (item["tp1"] / px - 1) * 100
    return dict(state="בפוזיציה", cls="live", px=px, pct=pct_live,
                r=(px - item["entry"]) / risk,
                note="עד TP1 " + ltr(f"{tp1_pct:+.1f}%"))


def funnel_html(total, drop):
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
    if not drop:
        return None
    k, v = max(drop.items(), key=lambda x: x[1])
    return k, v


def funnel_chart(drop):
    order = ["מחזור נמוך", "נפח מניות נמוך", "מחיר נמוך", "השיא ישן מדי", "עוד בשיא",
             "זינוק קטן מדי", "זינוק קצר מדי", "אין מקום לזינוק", "כמעט לא תיקנה",
             "החזירה את כל הזינוק", "הנפח לא התייבש", "רחוקה מדי מהתמיכה",
             "אין נר בולינג׳ר", "ATR נמוך מדי", "תנודתיות נמוכה", "אין מרווח ל-TP1",
             "היסטוריה קצרה", "אין נתוני נפח", "סטופ לא תקין", "אין ATR", "שגיאה"]
    items = [(k, drop[k]) for k in order if drop.get(k)]
    items += [(k, v) for k, v in drop.items() if k not in order and v]
    items.sort(key=lambda x: x[1])  # ascending so the worst ends up on top visually
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
_ = wl_load()  # ensure state hydrated from URL before anything renders

if "P" not in st.session_state:
    st.session_state["P"] = dict(PRESETS["מסחר יומי"])

p1, p2, p3, p4, p5 = st.columns([1, 1, 1, 1, 2])
for col, name in zip((p1, p2, p3, p4), PRESETS):
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
    clear_price_cache()
    _yf_cache_dir(fresh=True)
    go_ = True

ptype = st.radio("סוג תבנית",
                 ["זינוק ותיקון לתמיכה", "פריצה פרבולית ותיקון",
                  "שבירת תמיכה ותפיסה מחדש", "כל התבניות"],
                 index=3, horizontal=True,
                 help="'פריצה פרבולית ותיקון' מחפש מניות שעשו מהלך עצום, "
                      "תיקנו עמוק, ועדיין רחוק מעל הנקודה שממנה יצאו — "
                      "כלומר הכסף שנכנס לא יצא. זה הרגל השני.")

with st.expander("סינון מתקדם"):
    st.markdown("##### זינוק ותיקון")
    g1, g2, g3, g4 = st.columns(4)
    agev = g1.slider("ימים מהשיא", 1, 90, tuple(P["age"]))
    retr = g2.slider("עומק תיקון %", 5, 95, tuple(P["retr"]))
    rise_min = g3.slider("גודל זינוק מינ׳ %", 3, 100, int(P["rise"]))
    off_max = g4.slider("מרחק מקס׳ מתמיכה %", 2, 40, int(P.get("off", 12)),
                        help="כמה אחוז מעל התמיכה המחיר יכול להיות ועדיין להיחשב 'עליה'.")
    h1, h2, h3, h4 = st.columns(4)
    leg_min = h1.slider("ימי זינוק מינ׳", 3, 20, 5)
    leg_max = h2.slider("ימי זינוק מקס׳", 8, 60, 30)
    rr_min = h3.slider("יחס סיכון־סיכוי", 0.5, 5.0, float(P["rr"]), 0.1)
    atr_pct = h4.slider("ATR מינ׳ %", 0.0, 10.0, float(P["atrp"]), 0.25)
    w1, w2 = st.columns(2)
    win = w1.slider("חלון היסטוריה (ימים)", 100, 250, 150, 10,
                    help="רקע היסטורי לחישוב נפח בסיס. השיא עצמו נחפש רק בטווח "
                         "'ימים מהשיא' שהגדרת למעלה — לא כאן.")

    st.markdown("##### פריצה פרבולית ותיקון")
    y1, y2, y3, y4 = st.columns(4)
    prise_min = y1.slider("גודל הפריצה מינ׳ %", 20, 500, 60, 10,
                          help="כמה המנייה עלתה מהבסיס לשיא. 60% = מהלך רציני. "
                               "200% = פרבולי אמיתי כמו שתיארת.")
    pretr = y2.slider("עומק התיקון %", 5, 95, (25, 80),
                      help="כמה מהמהלך הוחזר. 30-70 = תיקון בריא שמנער "
                           "את מי שקנה מאוחר, בלי למחוק את המהלך.")
    phold_min = y3.slider("מעל הבסיס מינ׳ %", 5, 300, 25, 5,
                          help="התנאי הכי חשוב בתבנית הזאת: כמה אחוז המחיר "
                               "עדיין מעל הנקודה שממנה יצא המהלך. אם ירד "
                               "מתחת לזה — הכסף יצא וזו לא הזדמנות לרגל שני.")
    page = y4.slider("ימים מהשיא", 1, 120, (1, 45),
                     help="כמה זמן עבר מאז השיא. תיקון של מהלך גדול "
                          "לוקח שבועות, לא ימים.")
    y5, y6, y7, y8 = st.columns(4)
    prun_min = y5.slider("ימי ריצה מינ׳", 2, 60, 3,
                         help="DLXY עשתה 325% בחמישה ימים. סף של 10 ימים "
                              "היה פוסל אותה. 3 = תופס גם ספייקים מהירים.")
    prun_max = y6.slider("ימי ריצה מקס׳", 20, 200, 120, 10,
                         help="חלון החיפוש אחורה מהשיא אל הבסיס.")
    pdry_min = y7.number_input("יובש נפח מינ׳ (פרבולי)", 0.5, 5.0, 1.05, 0.05)
    poff_max = y8.slider("מרחק מקס׳ מתמיכה %", 3, 90, 40,
                         help="בפרבולי הטווחים ענקיים — 20% מעל התמיכה זה "
                              "כלום במנייה שזזה 300%.")

    st.markdown("###### שערים עצמאיים לתבנית הזאת (לא מושפעים מהמסחר היומי)")
    z1, z2, z3, z4 = st.columns(4)
    pmin_px = z1.number_input("מחיר מינ׳ $ (פרבולי)", 0.1, 100.0, 0.5, 0.1,
                              help="רוב הפרבוליים הם פני-סטוקס. DLXY נסחרה "
                                   "ב-1.85$ אחרי המהלך. אל תעלה את זה מעל 2.")
    pmax_px = z2.number_input("מחיר מקס׳ $ (פרבולי)", 0.0, 2000.0, 50.0, 5.0,
                              help="0 = ללא הגבלה.")
    prvol_min = z3.number_input("RVOL מינ׳ (פרבולי)", 0.0, 10.0, 0.0, 0.1,
                                help="0 = כבוי, וזו ברירת המחדל בכוונה. "
                                     "בזמן התיקון הנפח של פרבולי יורד מתחת "
                                     "לרגיל — סף RVOL כאן פוסל בדיוק את "
                                     "ההתכנסות שאתה מחפש.")
    pmax_risk = z4.slider("סיכון מקס׳ לעסקה % (פרבולי)", 5, 60, 25, 5,
                          help="מרחק הסטופ ממחיר הכניסה, באחוזים. בפרבולי "
                               "התמיכה יכולה להיות רחוקה מאוד — התקרה הזאת "
                               "מונעת סטופ מטורף שהורס את יחס הסיכון-סיכוי.")
    z5, z6 = st.columns([1, 3])
    pdv_min = z5.number_input("כסף בפריצה מינ׳ (מ׳ $)", 0.0, 500.0, 5.0, 1.0,
                              help="ממוצע המחזור הדולרי בימי הפריצה. "
                                   "מוציא ספייקים על מניות רפאים.")

    st.markdown("##### שבירת תמיכה ותפיסה מחדש")
    n1, n2, n3, n4 = st.columns(4)
    sup_win = n1.slider("ימים לקביעת התמיכה", 15, 90, 35)
    break_win = n2.slider("חלון השבירה (ימים אחרונים)", 1, 10, 4)
    break_min = n3.slider("שבירה מינ׳ %", 0.5, 15.0, 2.0, 0.5)
    break_max = n4.slider("שבירה מקס׳ %", 3.0, 30.0, 12.0, 0.5)
    n5, n6, n7 = st.columns(3)
    reclaim_lo = n5.slider("תפיסה מחדש מינ׳ %", -5.0, 10.0, -1.0, 0.5,
                           help="שלילי = עוד מתחת לתמיכה אך קרוב. חיובי = כבר מעליה.")
    reclaim_hi = n6.slider("תפיסה מחדש מקס׳ %", 1.0, 20.0, 8.0, 0.5)
    need_trig_b = n7.checkbox("חייב נר ירוק בתפיסה", value=True)

    st.markdown("##### כללי")
    j1, j2, j3, j4 = st.columns(4)
    bbmode = j1.selectbox("בולינג׳ר כ־", ["בונוס", "חובה"],
                          index=["בונוס", "חובה"].index(P.get("bbmode", "בונוס")),
                          help="בונוס = מסומן בתווית אך לא פוסל. חובה = מסנן.")
    bb_look = j2.slider("בולינג׳ר: ימים אחורה", 1, 10, 3)
    min_sh = j3.number_input("נפח מינ׳ (מ׳ מניות)", 0.0, 50.0, float(P.get("sh", 0.0)), 0.1,
                             help="0 מכבה. סינון לפי מספר מניות פוסל מניות יקרות ונזילות.")
    limit = j4.slider("מספר מניות לסריקה", 100, 7038, 3000, 100,
                      help="הצוואר הוא הרשת, לא המעבד: ניתוח 7000 מניות לוקח "
                           "8 שניות, אבל ההורדה שלהן לוקחת דקות. "
                           "3000 = כדקה. 7038 = כל הבורסה, 2-3 דקות. "
                           "אם האפליקציה נחנקת — הורד ל-1500.")

    st.markdown("##### מגמה, חוזק יחסי ודוחות — עדיין לא נבדקו בבדיקה היסטורית")
    p1, p2, p3, p4 = st.columns(4)
    trend_hard = p1.checkbox("דרוש מעל SMA200", value=True,
                             help="לסחור רק תיקונים בתוך מגמת עלייה ארוכת טווח. "
                                  "היגיון סטנדרטי במסחר, אך לא נבדק עדיין במערכת הזאת.")
    use_rs = p2.checkbox("דרוש חוזק מול השוק", value=False,
                         help="RS מול SPY חיובי — המנייה עלתה יותר מהמדד בחודש האחרון.")
    rs_min_val = p3.number_input("RS מינימלי %", -20.0, 20.0, 0.0, 1.0, disabled=not use_rs)
    earn_days = p4.number_input("הרחק מדוח (ימים)", 0, 30, 0, 1,
                                help="0 = כבוי (מהיר). כל ערך אחר מוסיף קריאת רשת "
                                     "לכל מנייה שנמצאה, ומאט את סוף הסריקה בכמה שניות.")

    st.markdown("##### תנועה ונפח — מה הופך מנייה לסחירה ביום")
    d1, d2, d3, d4 = st.columns(4)
    max_px = d1.number_input("מחיר מקס׳ $", 0.0, 2000.0, float(P.get("maxpx", 0.0)), 5.0,
                             help="0 = ללא הגבלה. מניה ב-400$ דורשת הון גדול לכל "
                                  "פוזיציה ומגיבה לאט. למסחר יומי נסה 20–80.")
    rvol_min = d2.number_input("RVOL מינ׳", 0.0, 10.0, float(P.get("rvol", 0.0)), 0.1,
                               help="נפח 3 הימים האחרונים חלקי הממוצע של 50 יום. "
                                    "1.0 = נפח רגיל. מתחת ל-1 המנייה ישנה. "
                                    "למסחר יומי דרוש 1.5 ומעלה.")
    leg_dv_min = d3.number_input("כסף בזינוק מינ׳ (מ׳ $)", 0.0, 500.0,
                                 float(P.get("legdv", 0.0)), 1.0,
                                 help="ממוצע המחזור הדולרי בימי העלייה עצמם. "
                                      "זה 'כמה כסף נכנס'. יחס נפח גבוה על מנייה "
                                      "דלילה עדיין אומר מעט מאוד כסף אמיתי.")
    d4.markdown("<div style='height:1.55rem'></div>", unsafe_allow_html=True)
    d4.caption("שלושת אלה יחד מוציאים מניות גדולות ומנומנמות.")

    q1, q2, q3 = st.columns(3)
    rev_hard = q1.checkbox("דרוש נר היפוך איכותי", value=True,
                           help="נר שורי עם גוף משמעותי וסגירה בחלק העליון, פטיש, "
                                "או בליעה שורית. נר ירוק קטן לא נחשב.")
    acct = q2.number_input("גודל תיק $", 500, 5_000_000, 25_000, 500)
    riskp = q3.slider("סיכון לעסקה %", 0.25, 5.0, 0.5, 0.25)

    use_prepost = st.checkbox(
        "בדוק פרה-מרקט ואפטר-מרקט למניות שנמצאו", value=True,
        help="הנרות היומיים נגמרים בסגירה ב-16:00. מניה שקפצה 6% בפרה-מרקט "
             "היא כבר לא אותה עסקה שהכרטיס מתאר. הבדיקה רצה רק על ההתאמות "
             "(לא על כל היקום) ומוסיפה כשנייה לכל מנייה שנמצאה.")

C = dict(leg_min=leg_min, leg_max=leg_max, rise_min=rise_min, retr_lo=retr[0],
         retr_hi=retr[1], age_lo=agev[0], age_hi=agev[1], atr_pct=atr_pct,
         atr_abs=atr_abs, rr_min=rr_min, dry_min=dry_min, bb_look=bb_look, win=win,
         off_max=off_max, bb_hard=(bbmode == "חובה"), trend_hard=trend_hard,
         rev_hard=rev_hard,
         rs_min=(rs_min_val if use_rs else None),
         rvol_min=(rvol_min or None), max_px=(max_px or None),
         leg_dv_min=(leg_dv_min or None),
         bb={"כבוי": "off", "רצועה תחתונה": "lower",
             "רצועה עליונה": "upper", "אחת מהשתיים": "both"}[bb_lbl])

CP = dict(pwin=max(win, 250), prise_min=prise_min, pretr_lo=pretr[0],
          pretr_hi=pretr[1], phold_min=phold_min, page_lo=page[0], page_hi=page[1],
          prun_min=prun_min, prun_max=prun_max, pdry_min=pdry_min,
          poff_max=poff_max, atr_abs=atr_abs, atr_pct=atr_pct, rr_min=rr_min,
          need_trig=rev_hard,
          prvol_min=(prvol_min or None), pmin_px=(pmin_px or None),
          pmax_px=(pmax_px or None), pdv_min=(pdv_min or None),
          pmax_risk=pmax_risk)

CB = dict(win=win, sup_win=sup_win, break_win=break_win, break_min=break_min,
          break_max=break_max, reclaim_lo=reclaim_lo, reclaim_hi=reclaim_hi,
          atr_abs=atr_abs, atr_pct=atr_pct, rr_min=rr_min, need_trig=need_trig_b,
          trend_hard=trend_hard, rs_min=(rs_min_val if use_rs else None),
          rvol_min=(rvol_min or None), max_px=(max_px or None))

with st.expander("בדיקה היסטורית — לפני שסומכים על סינון חדש",
                 expanded=bool(st.session_state.get("bt_diag"))):
    st.markdown("בודק כל איתות היסטורי שהמסננים שלמעלה היו מזהים, ועוקב קדימה: "
               "האם ההוראה התמלאה, והאם המחיר הגיע ליעד לפני הסטופ. משתמש **באותן** "
               "פונקציות שהסורק החי משתמש בהן — לא חישוב נפרד.")
    bc1, bc2, bc3 = st.columns(3)
    bt_n = bc1.slider("כמה מניות לבדוק", 20, 500, 100, 10,
                      help="יותר מניות = בדיקה אמינה יותר, אבל איטית יותר.")
    bt_period = bc2.selectbox("תקופה", ["2y", "5y"], index=0)
    bt_hold = bc3.slider("ימי החזקה מקסימלי", 5, 60, 25, 5)
    bcc1, bcc2 = st.columns(2)
    bt_slip = bcc1.number_input("Slippage (נקודות בסיס)", 0.0, 100.0, 5.0, 1.0,
                                help="עלות החלקה בכניסה וביציאה. 5 נ\"ב = 0.05%.")
    bt_comm = bcc2.number_input("עמלה למניה $", 0.0, 1.0, 0.005, 0.001, format="%.3f")

    if st.button("הרץ בדיקה היסטורית", type="primary"):
        _yf_diag_reset()
        # The backtest used to take the first N names of the universe — and the
        # universe is deliberately liquidity-first, so those are AAPL, MSFT,
        # NVDA, GOOGL... With the day-trading preset (price <= $80, ATR >= 4%)
        # almost none of them could pass the filters on ANY day in two years.
        # Result: zero trades, which answered "is this pattern profitable?"
        # on stocks the scanner would never have looked at.
        #
        # Now the sample is drawn from names that clear the same price band and
        # dollar-volume floor the live scan uses, so the backtest tests the
        # population the pattern actually trades. The liquidity snapshot is
        # shared with the scanner, so after a scan this step is nearly free.
        note_pre = st.empty()
        prog_pre = st.progress(0.0)
        note_pre.caption("שלב 1/3 · בוחר מניות שמתאימות לסינון הנוכחי…")
        _par = ptype in ("פריצה פרבולית ותיקון", "כל התבניות")
        _only_par = ptype == "פריצה פרבולית ותיקון"
        lo_px = (pmin_px if _only_par else
                 (min(min_px, pmin_px) if _par else min_px))
        lo_dv = (pdv_min if _only_par and pdv_min else
                 (min(min_dv, pdv_min) if (_par and pdv_min) else min_dv))
        _inf = float("inf")
        if _only_par:
            hi_px = pmax_px or _inf
        elif _par:
            hi_px = max(max_px or _inf, pmax_px or _inf)
        else:
            hi_px = max_px or _inf
        # 4x the requested sample is enough: roughly a quarter to a third of
        # names clear the price/volume band. The old 8x (4,000 names for a
        # 500-stock test) spent minutes on candidates that were never used.
        # After a scan these snapshots are already cached and this is instant.
        pool, _ = build_universe(max(bt_n * 4, 1200))

        def _tick_bt_pre(frac, kept):
            prog_pre.progress(min(frac, 1.0))
            note_pre.caption(f"שלב 1/3 · בוחר מניות · {kept:,} מתאימות עד כה")

        cand, cand_px, bt_pre_drops = prescreen_universe(
            pool, min_price=lo_px, min_dollar_vol_m=lo_dv, on_progress=_tick_bt_pre)
        prog_pre.empty()
        in_band = [t for t in cand if cand_px.get(t, 0) <= hi_px]
        uni_bt = in_band[:bt_n]
        note_pre.empty()
        spy_hist = fetch(("SPY",), period=bt_period).get("SPY")
        spy_series = spy_return_lookup(spy_hist) if spy_hist is not None else None
        prog_b, note_b, live_b = st.progress(0.0), st.empty(), st.empty()
        trades, got_data, errored = [], 0, 0
        # Download the whole backtest universe in parallel first, then walk it.
        # Previously each 60-name chunk was downloaded and processed before the
        # next one even started, so a 500-name / 5y backtest could take many
        # minutes of pure waiting on the network.
        note_b.caption(f"שלב 2/3 · מוריד {len(uni_bt):,} היסטוריות…")
        bt_market, bt_nodata = fetch_universe_parallel(
            uni_bt, period=bt_period, batch=100, workers=1,
            on_progress=lambda f, g: prog_b.progress(min(f * 0.5, 0.5)))
        got_data = len(bt_market)
        chunks = [list(bt_market.items())[i:i + 60]
                  for i in range(0, len(bt_market), 60)]
        for ci, ch in enumerate(chunks):
            note_b.caption(f"שלב 3/3 · מנתח מנה {ci+1}/{len(chunks)} · "
                           f"{len(trades)} עסקאות עד כה")
            for t, dd in ch:
                try:
                    if float(dd["Close"].iloc[-1]) < min_px:
                        continue
                    if float(dd["Close"].iloc[-1] * dd["Volume"].tail(20).mean()) < min_dv * 1e6:
                        continue
                    for tr in backtest_walk(dd, C, CB, ptype, spy_series, hold=bt_hold,
                                            slip_bps=bt_slip, commission=bt_comm,
                                            CPX=CP):
                        tr["ticker"] = t
                        trades.append(tr)
                except Exception:
                    errored += 1
            prog_b.progress(0.5 + 0.5 * ((ci + 1) / max(len(chunks), 1)))
            if trades:
                bb = pd.DataFrame(trades)
                live_b.caption(f"ביניים: {len(bb)} עסקאות · "
                              f"{(bb['R']>0).mean()*100:.1f}% הצלחה · {bb['R'].mean():+.2f}R")
        prog_b.empty(); note_b.empty(); live_b.empty()
        st.session_state["bt"] = trades
        st.session_state["bt_diag"] = dict(
            requested=len(uni_bt), got_data=got_data, errored=errored,
            pool=len(pool), in_band=len(in_band), ptype=ptype,
            pre_drops=dict(bt_pre_drops), hi_px=hi_px, lo_px=lo_px,
            spy_ok=spy_hist is not None, when=datetime.now(il).strftime("%H:%M:%S"))

    if "bt_diag" in st.session_state:
        dg = st.session_state["bt_diag"]
        _pool_txt = (f"מתוך {dg['pool']:,} מניות, {dg['in_band']:,} מתאימות לטווח "
                     f"המחיר והמחזור של הסינון · " if dg.get("pool") else "")
        st.caption(f"ריצה אחרונה ({dg['when']}) · תבנית: {dg.get('ptype', '—')} · "
                  f"{_pool_txt}נבדקו {dg['requested']} · "
                  f"קיבלנו נתונים ל-{dg['got_data']} · SPY {'הצליח' if dg['spy_ok'] else 'נכשל'} "
                  f"· {dg['errored']} שגיאות בעיבוד.")
        if not st.session_state.get("bt"):
            # Say WHERE it stopped. The old message blamed the network whenever
            # nothing came back — including when SPY had just downloaded fine
            # and the real cause was that zero stocks survived the prescreen.
            pdr = dg.get("pre_drops") or {}
            if dg.get("requested", 0) == 0:
                nod = pdr.get("לא התקבלו נתונים", 0)
                pool_n = max(dg.get("pool", 0), 1)
                if nod >= 0.8 * pool_n:
                    st.caption(f"הסינון המקדים לא קיבל נתונים עבור {nod:,} מתוך "
                               f"{pool_n:,} מניות.")
                    render_yf_diagnostics("bt")
                else:
                    parts = " · ".join(f"{k}: {v:,}" for k, v in
                                       sorted(pdr.items(), key=lambda x: -x[1]))
                    _hi = dg.get("hi_px")
                    _hi_txt = (f"עד ${_hi:,.0f}" if _hi and _hi != float("inf")
                               else "ללא תקרה")
                    st.warning(f"אף מנייה לא עברה את טווח המחיר והמחזור של הסינון "
                               f"(מחיר מ-${dg.get('lo_px', 0):,.2f} {_hi_txt}). "
                               f"פירוט: {parts or '—'}. הרפה את מחיר או מחזור "
                               f"מינימלי בשורה העליונה.")
            elif dg["got_data"] == 0:
                st.error("המניות נבחרו אבל ההורדה ההיסטורית לא החזירה נתונים. "
                         "כנראה תקלה זמנית מול Yahoo Finance — נסה שוב בעוד רגע.")
            else:
                st.warning("התקבלו נתונים אבל אף עסקה לא נמצאה. המניות שנבדקו כבר "
                          "מתאימות לטווח המחיר של הסינון, כך שהסיבה היא התבנית עצמה: "
                          "השילוב של כל התנאים לא קרה אפילו פעם אחת. נסה קודם "
                          "500 מניות ותקופה של 5y. אם גם אז אפס — התבנית בהגדרות "
                          "האלה נדירה מדי כדי למדוד אותה.")

    if "bt" in st.session_state and st.session_state["bt"]:
        b = pd.DataFrame(st.session_state["bt"])
        wr = (b["R"] > 0).mean() * 100
        avg = b["R"].mean()
        gp = b.loc[b["R"] > 0, "R"].sum()
        gl = -b.loc[b["R"] < 0, "R"].sum()
        pf = gp / gl if gl > 0 else float("inf")
        k = st.columns(5)
        k[0].metric("עסקאות", len(b))
        k[1].metric("אחוז הצלחה", f"{wr:.1f}%")
        k[2].metric("תוחלת", f"{avg:+.2f}R")
        k[3].metric("Profit factor", f"{pf:.2f}")
        k[4].metric("סה\"כ", f"{b['R'].sum():+.0f}R")

        if len(b) < 60:
            st.warning(f"{len(b)} עסקאות בלבד — מדגם קטן מדי כדי לסמוך עליו. "
                      "הגדל את מספר המניות או את התקופה.")

        def bucket(col, bins, labels, title):
            d2 = b.dropna(subset=[col]).copy()
            if not len(d2):
                st.caption(f"{title}: אין מספיק נתונים.")
                return
            d2["_b"] = pd.cut(d2[col], bins, labels=labels)
            g = d2.groupby("_b", observed=True).agg(
                עסקאות=("R", "size"),
                אחוז_הצלחה=("R", lambda x: round((x > 0).mean() * 100, 1)),
                תוחלת_R=("R", lambda x: round(x.mean(), 2))).reset_index()
            g.columns = [title, "עסקאות", "אחוז הצלחה", "תוחלת R"]
            st.dataframe(g, hide_index=True, use_container_width=True)

        st.markdown("##### יובש נפח — בדיקת אימות חוזרת")
        bucket("dry", [0, 1.1, 1.3, 1.6, 2.2, 99],
              ["<1.1", "1.1-1.3", "1.3-1.6", "1.6-2.2", "2.2+"], "יובש נפח")

        st.markdown("##### חוזק מול השוק (RS) — גורם חדש, לא נבדק לפני כן")
        bucket("rs", [-100, 0, 100], ["שלילי (חלשה מהשוק)", "חיובי (חזקה מהשוק)"], "RS")

        st.markdown("##### מגמת יסוד (SMA200) — גורם חדש, לא נבדק לפני כן")
        b["trend_txt"] = b["above200"].map({True: "מעל SMA200", False: "מתחת ל-SMA200"})
        d3 = b.dropna(subset=["trend_txt"])
        if len(d3):
            g3 = d3.groupby("trend_txt").agg(
                עסקאות=("R", "size"),
                אחוז_הצלחה=("R", lambda x: round((x > 0).mean() * 100, 1)),
                תוחלת_R=("R", lambda x: round(x.mean(), 2))).reset_index()
            g3.columns = ["מגמה", "עסקאות", "אחוז הצלחה", "תוחלת R"]
            st.dataframe(g3, hide_index=True, use_container_width=True)
        else:
            st.caption("אין מספיק נתונים.")

        st.caption("אם קטגוריה מסוימת מראה תוחלת גבוהה יותר באופן עקבי — שווה להפוך "
                  "אותה לסינון קבוע. אם ההבדלים קטנים או לא עקביים, זה כנראה רעש. "
                  "דוחות רבעוניים לא ניתנים לבדיקה כאן — אין לנו נתוני דוחות עבר אמינים בחינם.")
        st.download_button("הורד את כל העסקאות (CSV)", b.to_csv(index=False),
                          f"backtest_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv")

# ---------------------------------------------------------------- detail

def render_watchlist():
    wl = wl_load()
    if not wl:
        return
    st.markdown("### רשימת מעקב")
    tickers = tuple(x["ticker"] for x in wl)
    # Do NOT hit the network on every page render — that added seconds to each
    # rerun and made the whole app feel frozen. Quotes load on demand.
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
<div class="wl-r {pct_cls}">{ltr(f"{st_['pct']:+.1f}%")}<span style="color:var(--dm);font-size:.62rem"> מהכניסה</span></div>
<div class="wl-r {r_cls}">{ltr(f"{st_['r']:+.2f}R")}</div>
<div class="wl-note">{st_['note']}</div>
</div>""", unsafe_allow_html=True)
        b1, b2 = st.columns([1, 6])
        if b1.button("פתח", key=f"wlopen_{item['ticker']}"):
            st.session_state["open"] = item["ticker"]
            st.rerun()

    if closed:
        with st.expander(f"נסגרו ({len(closed)})"):
            for item, st_ in closed:
                r_cls = "p" if st_["r"] > 0 else "n"
                st.markdown(f"""<div class="wl-row {st_['cls']}">
<div class="wl-sym">{item['ticker']}</div>
<div class="wl-st {st_['cls']}">{st_['state']}</div>
<div class="wl-px">${st_['px']:.2f}</div>
<div class="wl-r {r_cls}">{ltr(f"{st_['r']:+.2f}R")}</div>
<div class="wl-note">{st_['note']}</div></div>""", unsafe_allow_html=True)
            if st.button("נקה עסקאות שנסגרו"):
                keep = {i["ticker"] for i, s_ in active}
                st.session_state["WL"] = [x for x in wl if x["ticker"] in keep]
                wl_save()
                st.rerun()
    st.markdown("---")


try:
    render_watchlist()
except Exception:
    st.warning("לא הצלחתי לטעון את רשימת המעקב כרגע (כנראה תקלת רשת זמנית). "
               "שאר המערכת עובדת כרגיל.")

if st.session_state.get("open"):
    r = next((x for x in st.session_state.get("rows", [])
              if x["ticker"] == st.session_state["open"]), None)
    if r is None:
        st.session_state["open"] = None
        st.rerun()

    nav1, nav2, nav3 = st.columns([1, 3.4, 1])
    if nav1.button("→  חזרה לרשימה", use_container_width=True):
        st.session_state["open"] = None
        st.rerun()
    others = [x["ticker"] for x in st.session_state["rows"]]
    jump = nav2.selectbox("מעבר מהיר", others, index=others.index(r["ticker"]),
                          label_visibility="collapsed")
    if jump != r["ticker"]:
        st.session_state["open"] = jump
        st.rerun()
    in_wl = r["ticker"] in {x["ticker"] for x in wl_load()}
    if nav3.button("★ במעקב" if in_wl else "☆ הוסף למעקב", use_container_width=True):
        wl_toggle(r)
        st.rerun()

    v_ = rank_verdict(r)
    rank_disp = r.get("_rank")
    total_disp = len(st.session_state.get("rows", []))
    rank_tag = f'דירוג #{rank_disp} מתוך {total_disp} · ' if rank_disp else ""
    st.markdown(f"""<div style="direction:rtl;display:flex;align-items:center;gap:.9rem;
    margin:.8rem 0 1rem;padding:.85rem 1rem;background:var(--pan);border:1px solid var(--ln);
    border-right:4px solid {v_['color']};border-radius:5px">
    <div style="font-family:'Frank Ruhl Libre',serif;font-size:1.9rem;font-weight:700;
    color:{v_['color']}">{r['ticker']}</div>
    <div style="width:1px;align-self:stretch;background:var(--ln)"></div>
    <div style="flex:1">
      <div style="font-size:.95rem;color:{v_['color']};font-weight:500">{rank_tag}רמה {v_['tier']} — {v_['headline']}</div>
      <div style="font-size:.8rem;color:var(--dm);margin-top:.15rem">{r['why']}</div>
    </div></div>""", unsafe_allow_html=True)
    with st.expander("הסבר AI מלא על ההמלצה", expanded=True):
        full_txt = explain_ai(r, rank=rank_disp, total=total_disp)
        st.markdown(f'<div class="tw" style="font-size:.86rem;line-height:1.9">{full_txt}</div>',
                   unsafe_allow_html=True)

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

    v = get_chart_data(r, per, iv, ranges)
    if v is None:
        st.error(f"לא הצלחתי להוריד נתונים ל-{r['ticker']} בטווח הזה. "
                 "אינטרוול תוך-יומי מוגבל בהיסטוריה — נסה טווח קצר יותר, או נרות יומיים.")
        if st.button("נסה שוב"):
            fetch_one.clear()
            st.rerun()
    else:
        # ---- categorical x-axis: real trading bars only, no timezone-dependent
        # rangebreaks, no dead space. Every pixel of width is real data. ----
        n = len(v)
        xs = list(range(n))
        if iv == "1d":
            dtxt = [d.strftime("%d.%m.%y") for d in v.index]
        elif iv == "1wk":
            dtxt = [d.strftime("שבוע %d.%m.%y") for d in v.index]
        else:
            dtxt = [d.strftime("%d.%m  %H:%M") for d in v.index]

        n_ticks = min(8, n)
        tick_idx = sorted(set(int(k) for k in np.linspace(0, n - 1, n_ticks)))
        tick_txt = [dtxt[k] for k in tick_idx]

        rows_n = 3 if show_rsi else 2
        heights = [.62, .19, .19] if show_rsi else [.78, .22]
        fig = make_subplots(rows=rows_n, cols=1, shared_xaxes=True,
                            row_heights=heights, vertical_spacing=.02)

        if "בולינג׳ר" in ind and n > 20:
            mid, sd = v["Close"].rolling(20).mean(), v["Close"].rolling(20).std()
            fig.add_trace(go.Scatter(x=xs, y=mid + 2 * sd, line=dict(color="#33505E", width=1.1),
                                     name="BB עליון", hoverinfo="skip"), 1, 1)
            fig.add_trace(go.Scatter(x=xs, y=mid - 2 * sd, line=dict(color="#33505E", width=1.1),
                                     fill="tonexty", fillcolor="rgba(51,80,94,.14)",
                                     name="BB תחתון", hoverinfo="skip"), 1, 1)

        fig.add_trace(go.Candlestick(
            x=xs, open=v.Open, high=v.High, low=v.Low, close=v.Close,
            name=r["ticker"], text=dtxt, hovertext=dtxt,
            increasing_line_color="#2FBF8F", decreasing_line_color="#E0605F",
            increasing_fillcolor="#2FBF8F", decreasing_fillcolor="#E0605F",
            increasing_line_width=1.4, decreasing_line_width=1.4,
            hovertemplate="%{text}<br>O %{open:.2f}  H %{high:.2f}<br>"
                          "L %{low:.2f}  C %{close:.2f}<extra></extra>"), 1, 1)

        for nm, per_, col in [("SMA20", 20, "#4E8FB0"), ("SMA50", 50, "#D4A64B"),
                              ("SMA200", 200, "#8A6FB0")]:
            if nm in ind and n > per_:
                fig.add_trace(go.Scatter(x=xs, y=v["Close"].rolling(per_).mean(),
                                         line=dict(color=col, width=1.6), name=nm), 1, 1)
        if "VWAP" in ind:
            tp = (v["High"] + v["Low"] + v["Close"]) / 3
            vw = (tp * v["Volume"]).cumsum() / v["Volume"].cumsum().replace(0, np.nan)
            fig.add_trace(go.Scatter(x=xs, y=vw, line=dict(color="#C77DBA", width=1.4,
                                                            dash="dot"), name="VWAP"), 1, 1)
        if "אזורי תבנית" in ind and iv == "1d" and r["age"] + r["leg_bars"] < n:
            lg_x, pk_x = n - 1 - r["age"] - r["leg_bars"], n - 1 - r["age"]
            fig.add_vrect(x0=lg_x, x1=pk_x, fillcolor="#2FBF8F", opacity=.08,
                          line_width=0, row=1, col=1)
            fig.add_vrect(x0=pk_x, x1=n - 1, fillcolor="#E0605F", opacity=.08,
                          line_width=0, row=1, col=1)
        if "רמות עסקה" in ind:
            lvl_list = [(r["entry"], "כניסה", "#D4A64B", "solid"),
                       (r["stop"], "סטופ", "#E0605F", "dash"),
                       (r.get("support", r["stop"]), "תמיכה", "#6C8896", "dot"),
                       (r["tp1"], "TP1", "#2FBF8F", "dot"),
                       (r["tp2"], "TP2", "#2FBF8F", "dot"),
                       (r["tp3"], "TP3", "#2FBF8F", "dot")]
            if r.get("kind") == "breakdown":
                lvl_list.append((r.get("resistance", r["tp3"]), "התנגדות", "#8A6FB0", "dot"))
            for y_, lab, col, dash in lvl_list:
                fig.add_hline(y=y_, line_dash=dash, line_color=col, line_width=1.2,
                              annotation_text=f"{lab}  {y_:.2f}", annotation_position="right",
                              annotation_font_size=12, annotation_font_color=col, row=1, col=1)

        vc = ["#2FBF8F" if x >= y2 else "#E0605F" for x, y2 in zip(v.Close, v.Open)]
        fig.add_trace(go.Bar(x=xs, y=v.Volume, marker_color=vc, marker_line_width=0,
                             opacity=.55, name="נפח", hovertext=dtxt,
                             hovertemplate="%{hovertext}<br>נפח %{y:,.0f}<extra></extra>",
                             showlegend=False), 2, 1)
        if n > 20:
            fig.add_trace(go.Scatter(x=xs, y=v["Volume"].rolling(20).mean(),
                                     line=dict(color="#93A8B2", width=1.2), showlegend=False,
                                     hoverinfo="skip"), 2, 1)

        if show_rsi and n > 15:
            dl = v["Close"].diff()
            up = dl.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
            dn = (-dl.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
            fig.add_trace(go.Scatter(x=xs, y=100 - 100 / (1 + up / dn.replace(0, np.nan)),
                                     line=dict(color="#D4A64B", width=1.5), name="RSI",
                                     hovertext=dtxt, hovertemplate="%{hovertext}<br>RSI %{y:.1f}"
                                                                    "<extra></extra>",
                                     showlegend=False), 3, 1)
            for lvl, cl in [(70, "#E0605F"), (30, "#2FBF8F"), (50, "#2A3D47")]:
                fig.add_hline(y=lvl, line_color=cl, line_width=.9, line_dash="dot", row=3, col=1)
            fig.update_yaxes(range=[0, 100], row=3, col=1)

        fig.update_layout(
            height=700 if show_rsi else 580, xaxis_rangeslider_visible=False,
            font=dict(family="Heebo, Arial, sans-serif", color="#A8BAC2", size=13),
            hovermode="x unified", dragmode="pan",
            paper_bgcolor="#0B1216", plot_bgcolor="#0B1216",
            legend=dict(orientation="h", y=1.045, x=0, bgcolor="rgba(0,0,0,0)",
                        font=dict(size=11)),
            margin=dict(l=6, r=118, t=30, b=6),
            hoverlabel=dict(bgcolor="#131F26", bordercolor="#22333C",
                            font=dict(family="Heebo", size=12, color="#E9EFF1")))
        fig.update_xaxes(showgrid=False, linecolor="#22333C", zeroline=False,
                         tickmode="array", tickvals=tick_idx, ticktext=tick_txt,
                         tickfont=dict(size=11))
        fig.update_xaxes(tickmode="array", tickvals=tick_idx, ticktext=tick_txt,
                         row=rows_n, col=1)
        fig.update_yaxes(gridcolor="#17242B", zeroline=False, linecolor="#22333C",
                         tickfont=dict(size=11), side="right")
        fig.update_xaxes(range=[-1, n], row=1, col=1)
        fig.add_annotation(text="MAVRI", x=0.01, y=0.06, xref="paper", yref="paper",
                          showarrow=False, font=dict(size=22, color="rgba(147,168,178,.07)",
                                                     family="Frank Ruhl Libre"))

        st.plotly_chart(fig, use_container_width=True, config={
            "scrollZoom": True, "displaylogo": False,
            "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
            "displayModeBar": True, "responsive": True, "doubleClick": "reset",
            "toImageButtonOptions": {"scale": 2}})
        st.caption("גלגלת העכבר מקרבת ומרחיקה · גרירה מזיזה · לחיצה כפולה מאפסת · "
                   "בנייד צביטה בשתי אצבעות. הנתונים כוללים ימי מסחר בלבד — "
                   "ללא רווחים מלאכותיים בין ימים.")

    st.markdown("### תוכנית מסחר — עד 4 ימים")
    dtp = day_trade_plan(r)
    dt_shares = int((acct * riskp / 100) / dtp["risk"]) if dtp["risk"] > 0 else 0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("כניסה", f"${dtp['entry']:.2f}")
    m2.metric("סטופ", f"${dtp['stop']:.2f}", f"-${dtp['risk']:.2f}/מניה", delta_color="off")
    m3.metric("יעד ראשון", f"${dtp['t1']:.2f}", f"1:{dtp['rr1']:.1f}")
    m4.metric("כמות מניות", f"{dt_shares:,}", f"סיכון ${round(dt_shares*dtp['risk']):,}",
              delta_color="off")
    st.markdown(f"""<div class="tw" style="font-size:.83rem;line-height:1.9">
<b style="color:var(--tx)">יום 1</b> — ממתינים להפעלה: פריצה מעל {ltr(f"${dtp['entry']:.2f}")} בנפח מוגבר. לא הופעל? לא נוגעים.<br>
<b style="color:var(--tx)">יום 1–2</b> — לאחר הפעלה, סטופ קבוע ב-{ltr(f"${dtp['stop']:.2f}")}. הגעה ל-{ltr(f"${dtp['t1']:.2f}")} (T1) → מוכרים שליש, מעלים סטופ לכניסה.<br>
<b style="color:var(--tx)">יום 2–3</b> — הגעה ל-{ltr(f"${dtp['t2']:.2f}")} (T2) → מוכרים שליש נוסף, סטופ עוקב מתחת לשפל היום הקודם.<br>
<b style="color:var(--tx)">יום 3–4</b> — יתרה ל-{ltr(f"${dtp['t3']:.2f}")} (T3) או יציאה ידנית.
<span style="color:var(--sh)">לא הופעל תוך 4 ימי מסחר → מבטלים, התבנית התיישנה.</span>
</div>""", unsafe_allow_html=True)
    st.code(f"{r['ticker']}\n"
            f"BUY STOP {dt_shares} @ {dtp['entry']:.2f}\n"
            f"STOP {dtp['stop']:.2f}\n"
            f"T1 {dtp['t1']:.2f}  ·  T2 {dtp['t2']:.2f}  ·  T3 {dtp['t3']:.2f}\n"
            f"בטל אם לא הופעל תוך 4 ימי מסחר", language=None)

    with st.expander("תוכנית סווינג מלאה — למי שרוצה להחזיק מעבר ל-4 ימים"):
        st.markdown(rmap(r), unsafe_allow_html=True)
        q1, q2 = st.columns(2)
        with q1:
            st.code(f"{r['ticker']}\n"
                    f"BUY STOP {int(r['shares'])} @ {r['entry']:.2f}\n"
                    f"STOP {r['stop']:.2f}\n"
                    f"TP1 {r['tp1']:.2f}  ·  TP2 {r['tp2']:.2f}  ·  TP3 {r['tp3']:.2f}",
                    language=None)
        with q2:
            st.markdown(stat_grid([
                ("כמות מניות", f"{int(r['shares']):,}", ""),
                ("שווי פוזיציה", f"${r['shares']*r['entry']:,.0f}", ""),
                ("סיכון", f"${r['risk_total']:,.0f}", "dn"),
                ("רווח ב-TP1", f"${r['shares']*(r['tp1']-r['entry']):,.0f}", "up"),
                ("רווח ב-TP3", f"${r['shares']*(r['tp3']-r['entry']):,.0f}", "up"),
            ]), unsafe_allow_html=True)
        st.caption("שתי התוכניות משתמשות באותה נקודת כניסה. ההבדל הוא מרחק הסטופ "
                   "והיעדים — התוכנית היומית מהודקת ל-ATR לטווח של ימים.")

    with st.expander("נתוני התבנית המלאים"):
        if r.get("kind") == "breakdown":
            st.markdown(stat_grid([
                ("שבירה", f"{r['brk_pct']:.1f}%", ""), ("נפח שבירה", f"{r['spike']:.2f}×", ""),
                ("תפיסה מחדש", f"{r['reclaim_pct']:+.1f}%", ""),
                ("תמיכה", f"${r['support']:.2f}", ""), ("התנגדות", f"${r['resistance']:.2f}", ""),
                ("ATR", f"${r['atr']:.2f}", ""), ("ATR %", f"{r['atr_pct']:.1f}%", ""),
                ("RVOL", f"{r['rvol']:.2f}", ""),
                ("מעל SMA200", "כן" if r.get("above200") else ("לא" if r.get("above200") is False else "—"), ""),
                ("RS מול SPY", f"{r['rs']:+.1f}%" if r.get("rs") is not None else "—", ""),
                ("דוח בעוד", f"{r['earn_days']} ימים" if r.get("earn_days") is not None else "לא נבדק", ""),
            ]), unsafe_allow_html=True)
        else:
            st.markdown(stat_grid([
                ("יובש נפח", f"{r['dry']:.2f}×", "gd"), ("נפח בזינוק", f"{r['spike']:.2f}×", ""),
                ("זינוק", f"+{r['rise']:.0f}%", "up"), ("ימי זינוק", f"{r['leg_bars']}", ""),
                ("תיקון", f"{r['retrace']:.0f}%", ""), ("ימים מהשיא", f"{r['age']}", ""),
                ("ATR", f"${r['atr']:.2f}", ""), ("ATR %", f"{r['atr_pct']:.1f}%", ""),
                ("RVOL", f"{r['rvol']:.2f}", ""), ("R:R ל-TP1", f"1:{r['rr']:.1f}", "gd"),
                ("R:R ל-TP3", f"1:{r['rr3']:.1f}", ""),
                ("בולינג׳ר", ", ".join(r.get("bb", [])) or "—", ""),
                ("נר איתות", r.get("candle", "—"), "gd" if r.get("is_rev") else ""),
                ("מעל SMA200", "כן" if r.get("above200") else ("לא" if r.get("above200") is False else "—"), ""),
                ("RS מול SPY", f"{r['rs']:+.1f}%" if r.get("rs") is not None else "—", ""),
                ("דוח בעוד", f"{r['earn_days']} ימים" if r.get("earn_days") is not None else "לא נבדק", ""),
            ]), unsafe_allow_html=True)
    st.stop()

# ---------------------------------------------------------------- scan

if go_:
    t0 = time.time()
    _yf_diag_reset()
    uni, uni_log = build_universe(limit)
    st.caption("יקום: " + " · ".join(uni_log) + f"  →  **{len(uni):,}** מניות ייחודיות")

    spy_ret21 = None
    spy_data = fetch(("SPY",)).get("SPY")
    if spy_data is not None and len(spy_data) > 21:
        sc = spy_data["Close"].values.astype(float)
        spy_ret21 = float((sc[-1] / sc[-22] - 1) * 100)
    C["spy_ret21"] = spy_ret21
    CB["spy_ret21"] = spy_ret21

    prog, note = st.progress(0.0), st.empty()
    rows, drop, liq, last_bar = [], {}, 0, None

    # --- stage 1: cheap prescreen (10 bars) to kill illiquid names ----------
    def _tick_pre(frac, kept):
        prog.progress(min(frac * 0.35, 0.35))
        note.caption(f"סינון מקדים · {kept:,} מניות עברו נזילות")

    # The prescreen is a hard gate: anything it rejects never reaches ANY
    # pattern. So its floors must be the loosest of every enabled pattern, not
    # the day-trading ones. A $1.85 parabolic runner would otherwise be thrown
    # out here and core_parabolic would never see it.
    _par_on = ptype in ("פריצה פרבולית ותיקון", "כל התבניות")
    pre_min_px = min(min_px, pmin_px) if _par_on else min_px
    pre_min_dv = min(min_dv, pdv_min) if (_par_on and pdv_min) else min_dv
    # Measured: CPU is NOT the constraint — parsing 7,000 tickers costs about
    # 8 seconds total. Network round trips are. With batches of 200 and three
    # workers a full universe needed ~20 sequential rounds of waiting on Yahoo.
    # Fewer, larger batches cut that to about five, which is where the minutes
    # actually go. 10-bar payloads are small, so a 500-name batch is still a
    # modest request.
    survivors, _pre_px, pre_drops = prescreen_universe(
        uni, min_price=pre_min_px, min_dollar_vol_m=pre_min_dv,
        batch=200, workers=1, on_progress=_tick_pre)
    drop.update(pre_drops)

    # --- stage 2: full history for survivors only ---------------------------
    def _tick(frac, got):
        prog.progress(0.35 + min(frac * 0.45, 0.45))
        note.caption(f"מוריד היסטוריה · {got:,} מתוך {len(survivors):,}")

    market, no_data = fetch_universe_parallel(survivors, period="1y", batch=150,
                                              workers=1, on_progress=_tick)
    if no_data:
        drop["לא התקבלו נתונים"] = drop.get("לא התקבלו נתונים", 0) + no_data

    # --- stage 3: pattern match (pure CPU, no network) ----------------------
    items = list(market.items())
    for i, (t, d) in enumerate(items):
        if i % 150 == 0:
            note.caption(f"מנתח {i:,} / {len(items):,}  ·  {len(rows)} התאמות")
            prog.progress(0.80 + 0.20 * (i / max(len(items), 1)))
        try:
            p = float(d["Close"].iloc[-1])
            if p < pre_min_px:
                drop["מחיר נמוך"] = drop.get("מחיר נמוך", 0) + 1
                continue
            avg_sh = float(d["Volume"].tail(20).mean())
            if p * avg_sh < pre_min_dv * 1e6:
                drop["מחזור נמוך"] = drop.get("מחזור נמוך", 0) + 1
                continue
            # Each pattern still applies its OWN price floor inside its core;
            # this outer gate only removes what no pattern could ever want.
            if min_sh > 0 and avg_sh < min_sh * 1e6:
                drop["נפח מניות נמוך"] = drop.get("נפח מניות נמוך", 0) + 1
                continue
            liq += 1
            if last_bar is None:
                last_bar = d.index[-1]
            A = prep(d)
            found_here = []
            if ptype in ("זינוק ותיקון לתמיכה", "כל התבניות"):
                r, why = core(A, len(A["c"]) - 1, C)
                if r is None:
                    drop[why] = drop.get(why, 0) + 1
                else:
                    r["kind"] = "pullback"
                    r["why"] = why_he(r)
                    found_here.append(r)
            if ptype in ("פריצה פרבולית ותיקון", "כל התבניות"):
                CP["spy_ret21"] = C.get("spy_ret21")
                rp, whyp = core_parabolic(A, len(A["c"]) - 1, CP)
                if rp is None:
                    drop["פרבולי: " + whyp] = drop.get("פרבולי: " + whyp, 0) + 1
                else:
                    rp["why"] = why_parabolic(rp)
                    found_here.append(rp)
            if ptype in ("שבירת תמיכה ותפיסה מחדש", "כל התבניות"):
                rb, whyb = core_breakdown(A, len(A["c"]) - 1, CB)
                if rb is None:
                    drop["שבירה: " + whyb] = drop.get("שבירה: " + whyb, 0) + 1
                else:
                    rb["why"] = why_breakdown(rb)
                    found_here.append(rb)
            for r in found_here:
                # BUG FIX: integer division could yield 0 shares on wide-stop
                # names, and the card then displayed a meaningless
                # "0 מניות · סיכון $0" trade. Floor at 1 and expose the real
                # dollar risk so position sizing stays honest.
                raw_sh = (acct * riskp / 100) / r["risk_share"]
                sh = max(int(raw_sh), 1)
                r.update(ticker=t, shares=sh,
                         undersized=bool(raw_sh < 1),
                         risk_total=round(sh * r["risk_share"], 2))
                rows.append(r)
        except Exception:
            drop["שגיאה"] = drop.get("שגיאה", 0) + 1
    prog.empty()
    note.empty()
    if earn_days > 0 and rows:
        note2 = st.empty()
        kept = []
        for k, r in enumerate(rows):
            note2.caption(f"בודק דוחות רבעוניים: {k+1}/{len(rows)}")
            d2e = days_to_earnings(r["ticker"])
            r["earn_days"] = d2e
            if d2e is not None and 0 <= d2e <= earn_days:
                drop[f"דוח בעוד {earn_days} ימים או פחות"] = \
                    drop.get(f"דוח בעוד {earn_days} ימים או פחות", 0) + 1
                continue
            kept.append(r)
        rows = kept
        note2.empty()
    # ---- near-miss pass -------------------------------------------------
    # When the scan comes back empty the funnel says WHICH filter blocked the
    # most names, but not how close any single stock actually was. Re-running
    # the same cores against the already-downloaded data with each numeric
    # threshold widened costs nothing on the network and answers the real
    # question: "was anything one hair away, or is today simply not the day?"
    near_rows = []
    if not rows and market:
        C_soft = dict(C)
        C_soft.update(rise_min=max(C["rise_min"] * 0.5, 4),
                      retr_lo=max(C["retr_lo"] - 12, 5),
                      retr_hi=min(C["retr_hi"] + 12, 92),
                      dry_min=max(C["dry_min"] - 0.25, 1.0),
                      atr_pct=max(C["atr_pct"] - 1.5, 0.5),
                      rvol_min=(max(C["rvol_min"] - 0.4, 0.8)
                                if C.get("rvol_min") else None),
                      leg_dv_min=(C["leg_dv_min"] * 0.5
                                  if C.get("leg_dv_min") else None),
                      off_max=C["off_max"] + 6,
                      rev_hard=False, trend_hard=False, bb_hard=False)
        # prep() is 97% of the analysis cost (measured: 0.54 ms/ticker vs
        # 0.005 ms for core itself). Re-running it across the entire market
        # doubled the heaviest step just to build a diagnostic table nobody
        # reads past the first few rows. 600 names is plenty to answer "was
        # anything close?" and keeps this pass under a second.
        for t, d in list(market.items())[:600]:
            try:
                A = prep(d)
                rs_, _ = core(A, len(A["c"]) - 1, C_soft)
                if rs_ is None:
                    continue
                # Record exactly which of the user's real thresholds it misses.
                misses = []
                if rs_["rise"] < C["rise_min"]:
                    misses.append(f"זינוק {rs_['rise']:.0f}% מול {C['rise_min']}% נדרש")
                if rs_["retrace"] < C["retr_lo"]:
                    misses.append(f"תיקון {rs_['retrace']:.0f}% מול {C['retr_lo']}% נדרש")
                if rs_["retrace"] > C["retr_hi"]:
                    misses.append(f"תיקון {rs_['retrace']:.0f}% מעל {C['retr_hi']}% מותר")
                if rs_["dry"] < C["dry_min"]:
                    misses.append(f"יובש {rs_['dry']:.2f}× מול {C['dry_min']:.2f}× נדרש")
                if rs_["atr_pct"] < C["atr_pct"]:
                    misses.append(f"ATR {rs_['atr_pct']:.1f}% מול {C['atr_pct']}% נדרש")
                if C.get("rvol_min") and rs_["rvol"] < C["rvol_min"]:
                    misses.append(f"RVOL {rs_['rvol']:.2f} מול {C['rvol_min']:.2f} נדרש")
                if C.get("leg_dv_min") and rs_.get("leg_dollars", 0) < C["leg_dv_min"] * 1e6:
                    misses.append(f"${rs_['leg_dollars']/1e6:.0f}מ׳ בזינוק מול "
                                  f"${C['leg_dv_min']:.0f}מ׳ נדרש")
                if not rs_.get("is_rev"):
                    misses.append("אין נר היפוך")
                if rs_.get("above200") is False:
                    misses.append("מתחת ל-SMA200")
                if not misses or len(misses) > 2:
                    continue  # only genuinely close calls are useful
                rs_["ticker"] = t
                rs_["misses"] = misses
                rs_["n_miss"] = len(misses)
                near_rows.append(rs_)
            except Exception:
                pass
        near_rows.sort(key=lambda x: (x["n_miss"], -x["rise"]))
        near_rows = near_rows[:25]

    for r in rows:
        v_ = rank_verdict(r)
        r["tier"], r["tier_color"] = v_["tier"], v_["color"]
        r["verdict_headline"], r["verdict_detail"] = v_["headline"], v_["detail"]
        r["_sort"] = v_["sort"]
    rows.sort(key=lambda x: x["_sort"])

    # Extended-hours check on matches only. The daily bars the pattern is built
    # from stop at the 16:00 close, so a name that already gapped 6% in the
    # pre-market is a different trade than the one the card describes. One
    # network call per match — affordable because matches are few.
    if use_prepost and rows:
        note3 = st.empty()
        for k, r in enumerate(rows[:40]):
            note3.caption(f"בודק פרה/אפטר מרקט: {k+1}/{min(len(rows), 40)}")
            eh = extended_hours_quote(r["ticker"])
            if eh:
                r["eh_price"] = eh["price"]
                r["eh_change"] = eh["change_pct"]
                r["eh_session"] = eh["session"]
                # A gap straight through the trigger changes the plan.
                if eh["price"] >= r["entry"]:
                    r["eh_triggered"] = True
        note3.empty()
    st.session_state.update(
        rows=rows, near=near_rows, stats=(len(uni), liq),
        got_data=len(market), drop=drop, open=None,
        scanned_at=datetime.now(il).strftime("%H:%M"),
        took=round(time.time() - t0, 1), cached=not fresh,
        data_date=last_bar.strftime("%d.%m.%y") if last_bar is not None else "—")

# ---------------------------------------------------------------- results

if "rows" not in st.session_state:
    if wl_load():
        st.info("הרשימה שלך למעלה נשמרת בכתובת הדף — סמן אותה במועדפים כדי לחזור אליה "
                "מכל מכשיר, כולל הטלפון.")
    else:
        st.markdown('<div class="empty">בחר פריסט או הגדר סינון, ולחץ על סריקה.<br>'
                    'המערכת מחפשת מניות שזינקו בנפח, תיקנו בנפח נמוך, '
                    'ונמצאות עכשיו על התמיכה.<br><br>'
                    'מניה שתסמן ב-☆ תישאר ברשימת המעקב למעלה, בזמן אמת, '
                    'עד שהיא תפגע בסטופ או תגיע ל-TP3.</div>', unsafe_allow_html=True)
else:
    rows = st.session_state["rows"]
    u, l_ = st.session_state["stats"]
    top_dry = max((r["dry"] for r in rows), default=0)
    took = st.session_state.get("took", 0)
    src = "מטמון" if st.session_state.get("cached") else "הורדה טרייה"
    gd = st.session_state.get("got_data", u)
    st.markdown(f"""<div class="kpi">
<div><b>{u:,}</b><span>ביקום</span></div>
<div><b>{gd:,}</b><span>נתונים התקבלו</span></div>
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

    tb = top_blocker(st.session_state["drop"])
    if tb:
        tb_k, tb_v = tb
        tb_pct = tb_v / max(u, 1) * 100
        st.markdown(f'<div class="blocker">המסנן שחוסם הכי הרבה מניות: '
                    f'<b>{tb_k}</b> ({tb_v:,} מניות, {tb_pct:.0f}% מהיקום) — '
                    f'פרטים מלאים בפאנל "מפל הסינון" למטה.</div>',
                    unsafe_allow_html=True)

    _nod = st.session_state["drop"].get("לא התקבלו נתונים", 0)
    if _nod >= 0.8 * max(u, 1):
        # Most of the universe returned nothing: this is a data failure, not an
        # empty market. Say so plainly before any advice about loosening filters.
        render_yf_diagnostics("scan")
    if not rows:
        st.markdown('<div class="empty">אין מניות בתבנית בסינון הזה.<br>'
                    'התבנית נדירה — נסה את הפריסט הרחב לפני שאתה מרפה ידנית.</div>',
                    unsafe_allow_html=True)

        # Actionable triage instead of a bare "try relaxing something".
        # Each blocker maps to the exact control that removes it, so the user
        # is never left guessing which of a dozen sliders is the one biting.
        FIXES = {
            "לא התקבלו נתונים": "Yahoo לא החזיר נתונים לחלק מהמניות — לחץ ↻ לסריקה טרייה, או הקטן את 'מספר מניות לסריקה'.",
            "מחזור נמוך": "הורד את 'מחזור מינ׳ (מ׳ $)' — הוא פוסל מניות לפי נזילות לפני כל בדיקה טכנית.",
            "מחיר נמוך": "הורד את 'מחיר מינ׳ $'.",
            "נפח מניות נמוך": "אפס את 'נפח מינ׳ (מ׳ מניות)' — הוא פוסל מניות יקרות ונזילות לגמרי.",
            "הנפח לא התייבש": "הורד את 'יובש נפח מינ׳' (נסה 1.05–1.15).",
            "מתחת ל-SMA200": "בטל את 'דרוש מעל SMA200' בסינון המתקדם.",
            "אין נר היפוך": "בטל את 'דרוש נר היפוך איכותי' — הוא לבדו פוסל את רוב המניות בכל יום נתון.",
            "אין נר על בולינג׳ר": "שנה 'בולינג׳ר כ־' ל'בונוס' במקום 'חובה'.",
            "רחוקה מדי מהתמיכה": "הגדל את 'מרחק מקס׳ מתמיכה %'.",
            "זינוק קטן מדי": "הורד את 'גודל זינוק מינ׳ %'.",
            "כמעט לא תיקנה": "הורד את הגבול התחתון של 'עומק תיקון %'.",
            "החזירה את כל הזינוק": "הגדל את הגבול העליון של 'עומק תיקון %'.",
            "תנודתיות נמוכה": "הורד את 'ATR מינ׳ %'.",
            "ATR נמוך מדי": "הורד את 'ATR מינ׳ $'.",
            "אין מרווח ל-TP1": "הורד את 'יחס סיכון־סיכוי'.",
            "חלשה מדי מול השוק": "בטל את 'דרוש חוזק מול השוק'.",
            "היסטוריה קצרה": "נורמלי — מניות שנסחרות פחות מ-80 ימים לא ניתנות לניתוח.",
        }
        near = st.session_state.get("near") or []
        if near:
            st.markdown("#### כמעט התאימו — ובכמה בדיוק פספסו")
            st.caption("המניות האלה עברו את כל שאר התנאים. לכל אחת רשום מה חסר "
                       "ומה הערך שלה בפועל, כדי שתדע אם שווה להזיז סליידר או שזה "
                       "באמת לא היום. חושב על הנתונים שכבר ירדו — בלי הורדה נוספת.")
            near_tbl = pd.DataFrame([{
                "מניה": x["ticker"],
                "מחיר": x["price"],
                "זינוק %": x["rise"],
                "תיקון %": x["retrace"],
                "יובש": x["dry"],
                "RVOL": x["rvol"],
                "ATR %": x["atr_pct"],
                "$מ׳ בזינוק": round(x.get("leg_dollars", 0) / 1e6, 1),
                "מה חסר": " · ".join(x["misses"]),
            } for x in near])
            st.dataframe(near_tbl, hide_index=True, use_container_width=True,
                         height=min(420, 40 + 35 * len(near)),
                         column_config={
                             "מחיר": st.column_config.NumberColumn(format="$%.2f"),
                             "זינוק %": st.column_config.NumberColumn(format="%.0f%%"),
                             "תיקון %": st.column_config.NumberColumn(format="%.0f%%"),
                             "יובש": st.column_config.NumberColumn(format="%.2f×"),
                             "RVOL": st.column_config.NumberColumn(format="%.2f"),
                             "ATR %": st.column_config.NumberColumn(format="%.1f%%")})
            one_miss = [x for x in near if x["n_miss"] == 1]
            if one_miss:
                from collections import Counter
                top_single = Counter(x["misses"][0].split(" ")[0]
                                     for x in one_miss).most_common(1)[0]
                st.markdown(
                    f'<div class="blocker">{len(one_miss)} מניות פספסו תנאי '
                    f'<b>אחד בלבד</b>. הנפוץ ביותר ביניהן: <b>{top_single[0]}</b> '
                    f'({top_single[1]} מניות). הזזת הסליידר הזה היא השינוי '
                    f'היחיד שיפתח אותן.</div>', unsafe_allow_html=True)
            st.markdown("---")

        dsorted = sorted(st.session_state["drop"].items(), key=lambda x: -x[1])[:3]
        if dsorted:
            st.markdown("#### מה לעשות עכשיו")
            for k, v in dsorted:
                base = k.replace("שבירה: ", "")
                tip = FIXES.get(base, "הרפה את המסנן הזה בסינון המתקדם.")
                st.markdown(
                    f'<div class="blocker" style="margin:.25rem 0">'
                    f'<b>{k}</b> — {v:,} מניות. {tip}</div>',
                    unsafe_allow_html=True)
            st.caption("הסיבות למעלה הן הסיבה הראשונה שכל מנייה נפסלה. "
                       "הרפה את הראשונה, סרוק שוב, וראה מה נהיה החוסם הבא.")

        with st.expander("מפל הסינון — איפה המניות נפלו", expanded=True):
            fc = funnel_chart(st.session_state["drop"])
            if fc:
                st.plotly_chart(fc, use_container_width=True, config={"displayModeBar": False})
            st.caption("השורה הארוכה ביותר היא המסנן שחוסם הכי הרבה. הרפה אותו ראשון.")
    else:
        v1, v2, v3 = st.columns([1.4, 1.6, 1])
        view = v1.radio("תצוגה", ["כרטיסים", "טבלה"], horizontal=True,
                        label_visibility="collapsed")
        srt = v2.selectbox("מיון", ["המלצה (מבוסס בדיקה היסטורית)", "יובש נפח",
                                    "יחס סיכון־סיכוי", "קרוב להפעלה", "הכי קרוב לתמיכה",
                                    "טרי (ימים מהשיא)", "גודל הזינוק"],
                           label_visibility="collapsed")
        only_near = v3.checkbox("רק קרובות להפעלה")
        keyf = {"המלצה (מבוסס בדיקה היסטורית)": lambda x: x.get("_sort", (9, 0)),
                "יובש נפח": lambda x: -x["dry"],
                "יחס סיכון־סיכוי": lambda x: -x["rr"],
                "קרוב להפעלה": lambda x: x["to_entry"],
                "הכי קרוב לתמיכה": lambda x: x.get("off_low", 999),
                "טרי (ימים מהשיא)": lambda x: x["age"],
                "גודל הזינוק": lambda x: -x["rise"]}[srt]
        rows = sorted(rows, key=keyf)
        if srt == "המלצה (מבוסס בדיקה היסטורית)":
            has_pullback = any(x.get("kind") != "breakdown" for x in rows)
            has_breakdown = any(x.get("kind") == "breakdown" for x in rows)
            if has_pullback and has_breakdown:
                st.caption("ממוין קודם לפי איכות התבנית (יובש נפח מוכח לעומת שבירת "
                           "תמיכה שטרם נבדקה), ובתוך כל רמה — לפי טריות התיקון. "
                           "מספר גבוה יותר של יובש נפח אינו אוטומטית טוב יותר.")
            elif has_pullback:
                st.caption("ממוין לפי קרבה לטווח היובש שהוכח רווחי (1.30–1.60), ובתוך "
                           "הטווח — לפי טריות התיקון. מניה עם יובש 3.5× לא בהכרח טובה "
                           "יותר ממניה עם 1.45×, וגם לא אם התיקון שלה ישן בהרבה יותר.")
            else:
                st.caption("תבנית שבירת התמיכה עדיין לא נבדקה היסטורית — כל התוצאות "
                           "כאן מסומנות ברמה '?' ומוין רק לפי טריות. התייחס בזהירות.")
        if only_near:
            rows = [x for x in rows if x["to_entry"] <= 1.0]
        if not rows:
            st.info("אין מניות בטווח של 1% מרמת ההפעלה כרגע.")
            st.stop()
        for idx, rr in enumerate(rows):
            rr["_rank"] = idx + 1
        total_n = len(rows)
        if view == "טבלה":
            t = pd.DataFrame([{
                "#": r["_rank"],
                "מניה": r["ticker"],
                "סוג": {"breakdown": "שבירה", "parabolic": "פרבולי"}.get(
                    r.get("kind"), "תיקון"),
                "יובש/שבירה": r["dry"], "מחיר": r["price"],
                "כניסה": r["entry"], "% לכניסה": r["to_entry"], "סטופ": r["stop"],
                "TP1": r["tp1"], "TP2": r["tp2"], "TP3": r["tp3"], "R:R": r["rr"],
                "כמות": r["shares"], "סיכון $": r["risk_total"], "ATR $": r["atr"],
                "RVOL": r["rvol"], "תיקון %": r["retrace"], "ימים": r["age"],
                "בולינג׳ר": ", ".join(r.get("bb", [])) or "—"} for r in rows])
            st.dataframe(t, hide_index=True, use_container_width=True, height=460,
                         column_config={
                             "יובש/שבירה": st.column_config.ProgressColumn(
                                 "יובש/שבירה", format="%.2f", min_value=1.0, max_value=2.5),
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
            tb1, tb2 = st.columns([3, 1])
            if tb1.button("פתח", type="primary", use_container_width=True):
                st.session_state["open"] = pick
                st.rerun()
            wl_tickers = {x["ticker"] for x in wl_load()}
            picked_row = next(r for r in rows if r["ticker"] == pick)
            star = "★ הסר ממעקב" if pick in wl_tickers else "☆ הוסף למעקב"
            if tb2.button(star, use_container_width=True):
                wl_toggle(picked_row)
                st.rerun()
        else:
            wl_tickers = {x["ticker"] for x in wl_load()}
            # CPU FIX: this used to call live() on every matched ticker on every
            # single rerun — a fresh multi-ticker download just to decide whether
            # to print a "כבר הופעל" badge. Clicking any button paid for it again.
            # The scan's own closing price answers the same question for free;
            # the live quote is now opt-in.
            if st.session_state.get("cards_live"):
                live_now = live(tuple(r["ticker"] for r in rows))
            else:
                live_now = {}
                st.caption("מחירי הכרטיסים הם מסגירת הסריקה. "
                           "לבדיקה מול מחיר עכשווי לחץ על הכפתור למטה.")
                if st.button("טען מחירים עדכניים לכרטיסים"):
                    st.session_state["cards_live"] = True
                    st.rerun()
            age_ceiling = C["age_hi"]
            for start in range(0, len(rows), 3):
                cols = st.columns(3, gap="medium")
                for col, r in zip(cols, rows[start:start + 3]):
                    with col:
                        lp = live_now.get(r["ticker"])
                        cur_px = float(lp["Close"].iloc[-1]) if lp is not None and len(lp) else r["price"]
                        triggered = cur_px >= r["entry"]
                        aging = (age_ceiling - r["age"]) <= 3
                        st.markdown(card(r, best=(r is rows[0]),
                                         watched=(r["ticker"] in wl_tickers),
                                         triggered=triggered, aging=aging,
                                         rank=r["_rank"], total=total_n),
                                    unsafe_allow_html=True)
                        bo, bw = st.columns([3, 1])
                        if bo.button("פתח ניתוח", key=f"o{r['ticker']}",
                                     use_container_width=True):
                            st.session_state["open"] = r["ticker"]
                            st.rerun()
                        star = "★" if r["ticker"] in wl_tickers else "☆"
                        if bw.button(star, key=f"w{r['ticker']}", use_container_width=True,
                                     help="הוסף/הסר מרשימת המעקב"):
                            wl_toggle(r)
                            st.rerun()

        with st.expander("מפל הסינון — איפה המניות נפלו"):
            fc = funnel_chart(st.session_state["drop"])
            if fc:
                st.plotly_chart(fc, use_container_width=True, config={"displayModeBar": False})

        df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("spark", "_daily")}
                           for r in rows])
        z1, z2 = st.columns(2)
        z1.download_button("הורד CSV", df.to_csv(index=False),
                           f"mavri_{datetime.now():%Y%m%d}.csv", "text/csv",
                           use_container_width=True)
        z2.code(",".join(df["ticker"].tolist()), language=None)
