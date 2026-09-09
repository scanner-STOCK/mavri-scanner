"""
MAVRI Pro
Daily momentum and pre-breakout stock screener.

Important:
- Daily data identifies setups, not real-time intraday execution.
- yfinance is suitable for research/prototyping, not guaranteed live execution.
- Backtests are estimates and do not guarantee future performance.
"""

from __future__ import annotations

import io
import math
import time
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime
from html import escape
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf


# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="MAVRI Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")
UA = {"User-Agent": "Mozilla/5.0 MAVRI-Pro/2.0"}
BATCH_SIZE = 75
MAX_RESULTS = 100
DEFAULT_SCAN_SIZE = 4000

CSS = """
<style>
@import url('[fonts.googleapis.com](https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;600;700&display=swap)');

:root {
    --bg:#090D15;
    --panel:#121826;
    --panel2:#1A2232;
    --line:#293348;
    --text:#F8FAFC;
    --muted:#94A3B8;
    --dim:#64748B;
    --gold:#F59E0B;
    --green:#10B981;
    --red:#EF4444;
    --blue:#3B82F6;
}

html, body, [class*="css"] {
    font-family:'Heebo', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 90% 0%, rgba(59,130,246,.10), transparent 28%),
        radial-gradient(circle at 5% 20%, rgba(16,185,129,.07), transparent 24%),
        var(--bg);
    color:var(--text);
}

#MainMenu, footer, header {
    visibility:hidden;
}

.block-container {
    padding-top:1rem;
    padding-bottom:4rem;
    max-width:1450px;
}

div[data-testid="stExpander"] {
    border:1px solid var(--line);
    border-radius:12px;
    background:rgba(18,24,38,.88);
    overflow:hidden;
}

div[data-testid="stExpander"] summary {
    background:rgba(255,255,255,.02);
}

label, h2, h3 {
    color:var(--muted) !important;
}

.stButton button {
    border-radius:9px;
    font-weight:600;
    transition:.18s ease;
}

.stButton button:hover {
    transform:translateY(-1px);
}

.topbar {
    display:flex;
    justify-content:space-between;
    align-items:center;
    gap:1rem;
    padding:1rem 1.4rem;
    margin-bottom:1.2rem;
    border:1px solid var(--line);
    border-radius:14px;
    background:rgba(18,24,38,.80);
    backdrop-filter:blur(12px);
}

.brand {
    display:flex;
    align-items:baseline;
    gap:.75rem;
}

.brand-name {
    margin:0;
    font-size:2rem;
    font-weight:700;
    letter-spacing:.10em;
    background:linear-gradient(90deg,var(--blue),var(--green));
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.brand-subtitle {
    color:var(--muted);
    font-size:.88rem;
}

.clock-label {
    color:var(--dim);
    font-size:.68rem;
    text-align:right;
}

.clock-value {
    color:var(--text);
    font-size:1.1rem;
    font-weight:600;
    direction:ltr;
}

.kpis {
    display:grid;
    grid-template-columns:repeat(5,minmax(120px,1fr));
    gap:.6rem;
    margin:1rem 0 1.2rem;
}

.kpi {
    direction:rtl;
    border:1px solid var(--line);
    border-radius:11px;
    padding:1rem;
    background:rgba(18,24,38,.90);
}

.kpi strong {
    display:block;
    color:var(--text);
    font-size:1.45rem;
    line-height:1.2;
}

.kpi span {
    color:var(--dim);
    font-size:.73rem;
}

.kpi.good strong {
    color:var(--green);
}

.notice {
    direction:rtl;
    padding:.85rem 1rem;
    margin:.5rem 0 1rem;
    border-radius:9px;
    border:1px solid rgba(245,158,11,.25);
    background:rgba(245,158,11,.08);
    color:var(--muted);
    font-size:.84rem;
}

.notice strong {
    color:var(--gold);
}

.card {
    direction:rtl;
    height:100%;
    padding:1rem;
    border:1px solid var(--line);
    border-radius:13px;
    background:rgba(18,24,38,.86);
    box-shadow:0 7px 24px rgba(0,0,0,.16);
    transition:.2s ease;
}

.card:hover {
    transform:translateY(-3px);
    border-color:rgba(59,130,246,.65);
    box-shadow:0 13px 32px rgba(0,0,0,.28);
}

.card.top {
    border-color:rgba(16,185,129,.55);
}

.card-head {
    display:flex;
    justify-content:space-between;
    align-items:flex-start;
    border-bottom:1px solid var(--line);
    padding-bottom:.55rem;
    margin-bottom:.65rem;
}

.ticker {
    direction:ltr;
    font-size:1.55rem;
    font-weight:700;
    letter-spacing:.05em;
}

.score {
    direction:ltr;
    color:var(--green);
    font-size:1.2rem;
    font-weight:700;
    text-align:left;
}

.score-label {
    color:var(--dim);
    font-size:.65rem;
    text-align:left;
}

.spark {
    margin:.45rem 0 .6rem;
}

.spark svg {
    width:100%;
    height:42px;
    display:block;
}

.chips {
    display:flex;
    flex-wrap:wrap;
    gap:.35rem;
    margin:.55rem 0;
}

.chip {
    display:inline-block;
    padding:.22rem .5rem;
    border:1px solid var(--line);
    border-radius:6px;
    color:var(--muted);
    background:var(--panel2);
    font-size:.67rem;
}

.chip.green {
    color:var(--green);
    border-color:rgba(16,185,129,.35);
    background:rgba(16,185,129,.08);
}

.chip.gold {
    color:var(--gold);
    border-color:rgba(245,158,11,.35);
    background:rgba(245,158,11,.08);
}

.levels {
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:.4rem;
    margin-top:.65rem;
}

.level {
    padding:.48rem;
    border-radius:7px;
    background:rgba(255,255,255,.025);
    border:1px solid var(--line);
}

.level span {
    display:block;
    color:var(--dim);
    font-size:.63rem;
}

.level b {
    direction:ltr;
    display:block;
    color:var(--text);
    font-size:.82rem;
}

.explain {
    color:var(--muted);
    line-height:1.55;
    font-size:.76rem;
    margin-top:.65rem;
    padding:.65rem;
    background:var(--panel2);
    border-radius:8px;
}

@media(max-width:900px) {
    .kpis {
        grid-template-columns:repeat(2,1fr);
    }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ============================================================
# MODELS
# ============================================================

@dataclass(frozen=True)
class ScanConfig:
    min_price: float = 5.0
    max_price: float = 500.0
    min_dollar_volume_m: float = 15.0
    min_share_volume_m: float = 0.5

    setup_window: int = 150
    peak_age_min: int = 2
    peak_age_max: int = 20

    leg_bars_min: int = 3
    leg_bars_max: int = 30
    rise_min_pct: float = 10.0

    retrace_min_pct: float = 15.0
    retrace_max_pct: float = 65.0

    dry_ratio_min: float = 1.15
    spike_ratio_min: float = 1.5

    atr_min_abs: float = 0.30
    atr_min_pct: float = 2.0
    support_distance_max_pct: float = 7.0

    bb_side: str = "off"
    bb_required: bool = False
    bb_lookback: int = 3

    reversal_required: bool = True
    min_body_pct: float = 0.15
    min_lower_wick_pct: float = 0.40

    rr_min: float = 1.5
    breakout_buffer_atr: float = 0.05
    stop_buffer_atr: float = 0.25
    max_stop_atr: float = 4.0

    earnings_filter: bool = False


PRESETS: dict[str, ScanConfig] = {
    "מסחר יומי מומנטום": ScanConfig(
        min_price=5,
        min_dollar_volume_m=20,
        min_share_volume_m=0.75,
        peak_age_min=1,
        peak_age_max=15,
        leg_bars_min=3,
        rise_min_pct=10,
        retrace_min_pct=12,
        retrace_max_pct=60,
        dry_ratio_min=1.15,
        spike_ratio_min=1.5,
        atr_min_abs=0.30,
        atr_min_pct=2.5,
        support_distance_max_pct=6,
        rr_min=1.5,
    ),
    "לפני פריצה": ScanConfig(
        min_price=7,
        min_dollar_volume_m=15,
        min_share_volume_m=0.5,
        peak_age_min=3,
        peak_age_max=25,
        leg_bars_min=4,
        rise_min_pct=12,
        retrace_min_pct=18,
        retrace_max_pct=55,
        dry_ratio_min=1.25,
        spike_ratio_min=1.7,
        atr_min_abs=0.30,
        atr_min_pct=1.8,
        support_distance_max_pct=10,
        rr_min=1.8,
    ),
    "רחב": ScanConfig(
        min_price=3,
        min_dollar_volume_m=8,
        min_share_volume_m=0.25,
        peak_age_min=1,
        peak_age_max=35,
        leg_bars_min=2,
        rise_min_pct=6,
        retrace_min_pct=10,
        retrace_max_pct=80,
        dry_ratio_min=1.05,
        spike_ratio_min=1.1,
        atr_min_abs=0.15,
        atr_min_pct=1.2,
        support_distance_max_pct=16,
        rr_min=1.0,
        reversal_required=False,
    ),
}


# ============================================================
# DATA
# ============================================================

def chunked(values: list[str], size: int) -> Iterable[tuple[str, ...]]:
    for start in range(0, len(values), size):
        yield tuple(values[start:start + size])


def normalize_symbol(value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    symbol = value.strip().upper()

    if not symbol or symbol in {"FILE CREATION TIME", "SYMBOL", "ACT SYMBOL"}:
        return None

    # Yahoo commonly uses "-" instead of "." for class shares.
    symbol = symbol.replace(".", "-")

    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ-")
    if not all(ch in allowed for ch in symbol):
        return None

    return symbol


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def build_universe(limit: int = DEFAULT_SCAN_SIZE) -> list[str]:
    urls = [
        (
            "[nasdaqtrader.com](https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt)",
            "Symbol",
        ),
        (
            "[nasdaqtrader.com](https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt)",
            "ACT Symbol",
        ),
    ]

    symbols: list[str] = []

    for url, symbol_column in urls:
        response = requests.get(
            url,
            headers=UA,
            timeout=20,
        )
        response.raise_for_status()

        frame = pd.read_csv(io.StringIO(response.text), sep="|")

        if symbol_column not in frame.columns:
            raise ValueError(f"Missing symbol column: {symbol_column}")

        for raw_symbol in frame[symbol_column].dropna():
            symbol = normalize_symbol(raw_symbol)
            if symbol:
                symbols.append(symbol)

    # Keep original exchange-file order while removing duplicates.
    unique_symbols = list(dict.fromkeys(symbols))

    if not unique_symbols:
        raise RuntimeError("Universe download returned no symbols.")

    return unique_symbols[:limit]


def normalize_download(
    raw: pd.DataFrame,
    tickers: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}

    if raw is None or raw.empty:
        return output

    if isinstance(raw.columns, pd.MultiIndex):
        level_zero = set(raw.columns.get_level_values(0))

        for ticker in tickers:
            if ticker not in level_zero:
                continue

            frame = raw[ticker].copy()
            frame = frame.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

            if not frame.empty:
                output[ticker] = frame

        return output

    # Single-ticker response.
    if len(tickers) == 1:
        frame = raw.copy()
        frame = frame.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

        if not frame.empty:
            output[tickers[0]] = frame

    return output


@st.cache_data(ttl=30 * 60, max_entries=160, show_spinner=False)
def fetch_batch(
    tickers: tuple[str, ...],
    period: str = "1y",
) -> dict[str, pd.DataFrame]:
    if not tickers:
        return {}

    try:
        raw = yf.download(
            tickers=list(tickers),
            period=period,
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            actions=False,
            threads=True,
            progress=False,
            timeout=25,
        )
        return normalize_download(raw, tickers)

    except Exception:
        # Retry individually so one bad ticker does not destroy the batch.
        recovered: dict[str, pd.DataFrame] = {}

        for ticker in tickers:
            try:
                raw = yf.download(
                    tickers=ticker,
                    period=period,
                    interval="1d",
                    auto_adjust=False,
                    actions=False,
                    threads=False,
                    progress=False,
                    timeout=15,
                )
                recovered.update(normalize_download(raw, (ticker,)))
            except Exception:
                continue

        return recovered


# ============================================================
# INDICATORS
# ============================================================

def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["Open", "High", "Low", "Close", "Volume"]

    if not set(required).issubset(frame.columns):
        raise ValueError("OHLCV columns are missing.")

    out = frame[required].copy()

    for column in required:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=required)
    out = out[out["Volume"] >= 0]

    if len(out) < 60:
        raise ValueError("Not enough history.")

    previous_close = out["Close"].shift(1)

    true_range = pd.concat(
        [
            out["High"] - out["Low"],
            (out["High"] - previous_close).abs(),
            (out["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder ATR.
    out["ATR14"] = true_range.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    out["SMA20"] = out["Close"].rolling(20).mean()
    out["STD20"] = out["Close"].rolling(20).std(ddof=0)
    out["BBU"] = out["SMA20"] + 2 * out["STD20"]
    out["BBL"] = out["SMA20"] - 2 * out["STD20"]

    out["SMA50"] = out["Close"].rolling(50).mean()
    out["SMA200"] = out["Close"].rolling(200).mean()

    out["AVG_VOL20"] = out["Volume"].rolling(20).mean()
    out["AVG_VOL50"] = out["Volume"].rolling(50).mean()
    out["DOLLAR_VOL20"] = (out["Close"] * out["Volume"]).rolling(20).mean()

    return out


def detect_reversal(
    row: pd.Series,
    previous_row: pd.Series,
    cfg: ScanConfig,
) -> tuple[bool, str]:
    open_price = float(row["Open"])
    high = float(row["High"])
    low = float(row["Low"])
    close = float(row["Close"])

    candle_range = high - low

    if candle_range <= 0:
        return False, "נר לא תקין"

    body = abs(close - open_price)
    lower_wick = min(open_price, close) - low
    upper_wick = high - max(open_price, close)

    body_pct = body / candle_range
    lower_wick_pct = lower_wick / candle_range

    previous_open = float(previous_row["Open"])
    previous_close = float(previous_row["Close"])

    bullish_body = (
        close > open_price
        and body_pct >= cfg.min_body_pct
        and close >= low + 0.60 * candle_range
    )

    hammer = (
        lower_wick_pct >= cfg.min_lower_wick_pct
        and close >= low + 0.60 * candle_range
        and lower_wick >= max(1.5 * body, upper_wick)
    )

    bullish_engulfing = (
        previous_close < previous_open
        and close > open_price
        and open_price <= previous_close
        and close >= previous_open
    )

    if bullish_engulfing:
        return True, "בליעה שורית"

    if hammer:
        return True, "פטיש"

    if bullish_body:
        return True, "נר שורי"

    return False, "אין נר היפוך"


def bollinger_tags(
    window: pd.DataFrame,
    lookback: int,
    side: str,
) -> list[str]:
    if side == "off":
        return []

    recent = window.tail(lookback)
    tags: list[str] = []

    if side in {"lower", "both"}:
        if (recent["Low"] <= recent["BBL"]).fillna(False).any():
            tags.append("תחתונה")

    if side in {"upper", "both"}:
        if (recent["High"] >= recent["BBU"]).fillna(False).any():
            tags.append("עליונה")

    return tags


def score_setup(result: dict[str, Any]) -> float:
    score = 0.0

    score += min(result["dry"] / 2.0, 1.0) * 18
    score += min(result["spike"] / 4.0, 1.0) * 15
    score += min(result["rr"] / 3.0, 1.0) * 20
    score += min(result["dollar_volume_m"] / 100.0, 1.0) * 10
    score += min(result["atr_pct"] / 6.0, 1.0) * 10
    score += min(result["rvol"] / 2.0, 1.0) * 8
    score += max(0.0, 1.0 - result["support_distance_pct"] / 15.0) * 10
    score += max(0.0, 1.0 - result["to_entry_pct"] / 8.0) * 9

    return round(float(np.clip(score, 0, 100)), 1)


def evaluate_setup(
    prepared: pd.DataFrame,
    index: int,
    cfg: ScanConfig,
) -> tuple[dict[str, Any] | None, str]:
    if index < 59:
        return None, "היסטוריה קצרה"

    start = max(0, index - cfg.setup_window + 1)
    window = prepared.iloc[start:index + 1].copy()

    if len(window) < 60:
        return None, "היסטוריה קצרה"

    current = window.iloc[-1]
    previous = window.iloc[-2]

    close = float(current["Close"])
    atr = float(current["ATR14"])

    if not np.isfinite(close) or close <= 0:
        return None, "מחיר לא תקין"

    if not np.isfinite(atr) or atr <= 0:
        return None, "אין ATR"

    if close < cfg.min_price:
        return None, "מחיר נמוך"

    if close > cfg.max_price:
        return None, "מחיר גבוה"

    avg_volume20 = float(current["AVG_VOL20"])
    dollar_volume20 = float(current["DOLLAR_VOL20"])

    if not np.isfinite(avg_volume20):
        return None, "אין נפח ממוצע"

    if not np.isfinite(dollar_volume20):
        return None, "אין מחזור דולרי"

    if avg_volume20 < cfg.min_share_volume_m * 1_000_000:
        return None, "מחזור מניות נמוך"

    if dollar_volume20 < cfg.min_dollar_volume_m * 1_000_000:
        return None, "מחזור דולרי נמוך"

    atr_pct = atr / close * 100

    if atr < cfg.atr_min_abs:
        return None, "ATR כספי נמוך"

    if atr_pct < cfg.atr_min_pct:
        return None, "תנודתיות נמוכה"

    n = len(window)
    peak_right = n - 1 - cfg.peak_age_min
    peak_left = n - 1 - cfg.peak_age_max

    peak_left = max(0, peak_left)
    peak_right = min(n - 1, peak_right)

    if peak_right < peak_left:
        return None, "טווח שיא לא תקין"

    peak_search = window.iloc[peak_left:peak_right + 1]

    if peak_search.empty:
        return None, "אין טווח שיא"

    # Use the actual daily high, not closing price.
    peak_label = peak_search["High"].idxmax()
    peak_pos = window.index.get_loc(peak_label)
    peak_price = float(window.iloc[peak_pos]["High"])
    peak_age = n - 1 - peak_pos

    leg_start = max(0, peak_pos - cfg.leg_bars_max)
    leg_search = window.iloc[leg_start:peak_pos + 1]

    if leg_search.empty:
        return None, "אין רגל עלייה"

    low_label = leg_search["Low"].idxmin()
    low_pos = window.index.get_loc(low_label)
    low_price = float(window.iloc[low_pos]["Low"])
    leg_bars = peak_pos - low_pos

    if leg_bars < cfg.leg_bars_min:
        return None, "זינוק קצר מדי"

    if low_price <= 0 or peak_price <= low_price:
        return None, "רגל מחיר לא תקינה"

    rise_pct = (peak_price / low_price - 1) * 100

    if rise_pct < cfg.rise_min_pct:
        return None, "זינוק קטן מדי"

    retrace_pct = (peak_price - close) / (peak_price - low_price) * 100

    if retrace_pct < cfg.retrace_min_pct:
        return None, "כמעט לא תיקנה"

    if retrace_pct > cfg.retrace_max_pct:
        return None, "תיקון עמוק מדי"

    pre_leg_volume = window["Volume"].iloc[max(0, low_pos - 50):low_pos]
    leg_volume = window["Volume"].iloc[low_pos:peak_pos + 1]

    # Exclude the peak bar from pullback average where possible.
    pullback_volume = window["Volume"].iloc[peak_pos + 1:]

    if len(pre_leg_volume) < 5:
        pre_leg_volume = window["Volume"].iloc[:max(low_pos, 1)]

    if pre_leg_volume.empty or leg_volume.empty or pullback_volume.empty:
        return None, "אין מספיק נתוני נפח"

    base_volume = max(float(pre_leg_volume.mean()), 1.0)
    pullback_average = float(pullback_volume.mean())

    if pullback_average <= 0:
        return None, "אין נפח בתיקון"

    spike_ratio = float(leg_volume.max() / base_volume)
    dry_ratio = float(leg_volume.mean() / pullback_average)

    if spike_ratio < cfg.spike_ratio_min:
        return None, "אין קפיצת נפח"

    if dry_ratio < cfg.dry_ratio_min:
        return None, "הנפח לא התייבש"

    is_reversal, candle_name = detect_reversal(current, previous, cfg)

    if cfg.reversal_required and not is_reversal:
        return None, candle_name

    bb_tags = bollinger_tags(window, cfg.bb_lookback, cfg.bb_side)

    if cfg.bb_required and cfg.bb_side != "off" and not bb_tags:
        return None, "אין מגע בולינגר"

    post_peak = window.iloc[peak_pos:]

    if post_peak.empty:
        return None, "אין תיקון לאחר השיא"

    # Recent support gives a more relevant stop than the lowest point
    # of the entire pullback in many setups.
    support_window = post_peak.tail(min(10, len(post_peak)))
    support = float(support_window["Low"].min())

    if support <= 0:
        return None, "תמיכה לא תקינה"

    support_distance_pct = (close / support - 1) * 100

    if support_distance_pct > cfg.support_distance_max_pct:
        return None, "רחוקה מהתמיכה"

    current_high = float(current["High"])
    entry = current_high + cfg.breakout_buffer_atr * atr

    stop_candidate = support - cfg.stop_buffer_atr * atr
    max_risk_stop = entry - cfg.max_stop_atr * atr
    stop = max(stop_candidate, max_risk_stop)

    risk_per_share = entry - stop

    if risk_per_share <= 0:
        return None, "סטופ לא תקין"

    target_1 = peak_price
    target_2 = peak_price + 0.5 * (peak_price - low_price)
    target_3 = peak_price + 1.0 * (peak_price - low_price)

    rr = (target_1 - entry) / risk_per_share

    if rr < cfg.rr_min:
        return None, "יחס סיכוי-סיכון נמוך"

    avg_volume50 = float(current["AVG_VOL50"])
    rvol = (
        float(current["Volume"]) / avg_volume50
        if np.isfinite(avg_volume50) and avg_volume50 > 0
        else 0.0
    )

    to_entry_pct = (entry / close - 1) * 100

    sma50 = float(current["SMA50"]) if np.isfinite(current["SMA50"]) else np.nan
    sma200 = float(current["SMA200"]) if np.isfinite(current["SMA200"]) else np.nan

    trend_ok = bool(
        np.isfinite(sma50)
        and close >= sma50
        and (not np.isfinite(sma200) or sma50 >= sma200)
    )

    result: dict[str, Any] = {
        "price": round(close, 2),
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "tp1": round(target_1, 2),
        "tp2": round(target_2, 2),
        "tp3": round(target_3, 2),
        "risk_share": round(risk_per_share, 2),
        "rr": round(rr, 2),
        "rise_pct": round(rise_pct, 1),
        "retrace_pct": round(retrace_pct, 1),
        "dry": round(dry_ratio, 2),
        "spike": round(spike_ratio, 2),
        "rvol": round(rvol, 2),
        "atr_pct": round(atr_pct, 1),
        "peak_age": int(peak_age),
        "leg_bars": int(leg_bars),
        "support": round(support, 2),
        "support_distance_pct": round(support_distance_pct, 2),
        "to_entry_pct": round(to_entry_pct, 2),
        "dollar_volume_m": round(dollar_volume20 / 1_000_000, 1),
        "avg_volume_m": round(avg_volume20 / 1_000_000, 2),
        "candle": candle_name,
        "bb": bb_tags,
        "trend_ok": trend_ok,
        "spark": [
            round(float(value), 4)
            for value in window["Close"].tail(60).to_numpy()
        ],
    }

    result["score"] = score_setup(result)

    return result, "עבר"


# ============================================================
# BACKTEST
# ============================================================

def resolve_trade(
    prepared: pd.DataFrame,
    signal_index: int,
    entry: float,
    stop: float,
    target: float,
    max_hold_days: int,
    slippage_bps: float,
    commission_per_share: float,
) -> dict[str, Any] | None:
    end = min(signal_index + max_hold_days, len(prepared) - 1)

    if signal_index + 1 > end:
        return None

    filled = False
    fill_price = math.nan
    fill_index = -1

    for j in range(signal_index + 1, end + 1):
        row = prepared.iloc[j]
        day_open = float(row["Open"])
        day_high = float(row["High"])
        day_low = float(row["Low"])
        day_close = float(row["Close"])

        if not filled:
            if day_high < entry:
                continue

            # Gap above the stop-entry gets a worse fill at the open.
            raw_fill = max(entry, day_open)
            fill_price = raw_fill * (1 + slippage_bps / 10_000)
            fill_index = j
            filled = True

        actual_stop = stop * (1 - slippage_bps / 10_000)
        actual_target = target * (1 - slippage_bps / 10_000)
        risk = fill_price - actual_stop

        if risk <= 0:
            return None

        hit_stop = day_low <= stop
        hit_target = day_high >= target

        if hit_stop and hit_target:
            # Daily bars do not reveal which happened first.
            # Conservative assumption: stop occurred first.
            exit_price = actual_stop
            outcome = "stop_same_bar"
            r_multiple = (exit_price - fill_price) / risk
            r_multiple -= (2 * commission_per_share) / risk

            return {
                "R": r_multiple,
                "outcome": outcome,
                "hold_days": j - fill_index + 1,
            }

        if hit_stop:
            exit_price = actual_stop
            r_multiple = (exit_price - fill_price) / risk
            r_multiple -= (2 * commission_per_share) / risk

            return {
                "R": r_multiple,
                "outcome": "stop",
                "hold_days": j - fill_index + 1,
            }

        if hit_target:
            exit_price = actual_target
            r_multiple = (exit_price - fill_price) / risk
            r_multiple -= (2 * commission_per_share) / risk

            return {
                "R": r_multiple,
                "outcome": "target",
                "hold_days": j - fill_index + 1,
            }

        if j == end:
            exit_price = day_close * (1 - slippage_bps / 10_000)
            r_multiple = (exit_price - fill_price) / risk
            r_multiple -= (2 * commission_per_share) / risk

            return {
                "R": r_multiple,
                "outcome": "time_exit",
                "hold_days": j - fill_index + 1,
            }

    return None


def backtest_symbol(
    ticker: str,
    frame: pd.DataFrame,
    cfg: ScanConfig,
    max_hold_days: int,
    cooldown_days: int,
    slippage_bps: float,
    commission_per_share: float,
) -> list[dict[str, Any]]:
    prepared = prepare_frame(frame)
    trades: list[dict[str, Any]] = []
    last_signal = -10_000

    # Leave enough future bars for trade resolution.
    final_signal_index = len(prepared) - max_hold_days - 1

    for i in range(200, max(200, final_signal_index + 1)):
        if i - last_signal < cooldown_days:
            continue

        result, _ = evaluate_setup(prepared, i, cfg)

        if result is None:
            continue

        resolved = resolve_trade(
            prepared=prepared,
            signal_index=i,
            entry=float(result["entry"]),
            stop=float(result["stop"]),
            target=float(result["tp1"]),
            max_hold_days=max_hold_days,
            slippage_bps=slippage_bps,
            commission_per_share=commission_per_share,
        )

        if resolved is None:
            continue

        trades.append(
            {
                "ticker": ticker,
                "date": prepared.index[i],
                "R": float(resolved["R"]),
                "outcome": resolved["outcome"],
                "hold_days": int(resolved["hold_days"]),
                "score": float(result["score"]),
                "dry": float(result["dry"]),
                "rr_planned": float(result["rr"]),
            }
        )
        last_signal = i

    return trades


def max_drawdown_r(returns: pd.Series) -> float:
    equity = returns.cumsum()
    drawdown = equity - equity.cummax()
    return float(drawdown.min()) if not drawdown.empty else 0.0


def summarize_backtest(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {}

    wins = trades["R"] > 0
    losses = trades["R"] < 0

    gross_profit = float(trades.loc[wins, "R"].sum())
    gross_loss = abs(float(trades.loc[losses, "R"].sum()))

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else float("inf")
    )

    return {
        "trades": float(len(trades)),
        "win_rate": float(wins.mean() * 100),
        "expectancy": float(trades["R"].mean()),
        "median_r": float(trades["R"].median()),
        "profit_factor": profit_factor,
        "max_drawdown_r": max_drawdown_r(trades["R"]),
        "avg_hold": float(trades["hold_days"].mean()),
    }


# ============================================================
# PRESENTATION
# ============================================================

def sparkline(values: list[float]) -> str:
    if len(values) < 2:
        return '<div class="spark"></div>'

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]

    if len(array) < 2:
        return '<div class="spark"></div>'

    width = 320.0
    height = 42.0
    padding = 2.0

    low = float(array.min())
    high = float(array.max())
    span = high - low if high > low else 1.0

    xs = np.linspace(padding, width - padding, len(array))
    ys = height - padding - ((array - low) / span) * (height - 2 * padding)

    points = " ".join(
        f"{x:.1f},{y:.1f}"
        for x, y in zip(xs, ys)
    )

    color = "#10B981" if array[-1] >= array[0] else "#EF4444"

    return (
        '<div class="spark">'
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" '
        'preserveAspectRatio="none" role="img" aria-label="Price trend">'
        f'<polyline fill="none" stroke="{color}" stroke-width="2.2" '
        f'stroke-linejoin="round" stroke-linecap="round" points="{points}" />'
        "</svg>"
        "</div>"
    )


def setup_explanation(result: dict[str, Any]) -> str:
    return (
        f"זינוק של {result['rise_pct']:.1f}% עם קפיצת נפח פי "
        f"{result['spike']:.1f}, תיקון של {result['retrace_pct']:.1f}% "
        f"בנפח מתייבש, ונר {result['candle']} סמוך לתמיכה."
    )


def card_html(result: dict[str, Any], is_top: bool) -> str:
    ticker = escape(str(result["ticker"]))
    candle = escape(str(result["candle"]))
    explanation = escape(setup_explanation(result))

    trend_chip = (
        '<span class="chip green">מגמה תומכת</span>'
        if result["trend_ok"]
        else '<span class="chip">ללא אישור מגמה</span>'
    )

    bb_chip = ""
    if result["bb"]:
        bb_text = escape(", ".join(result["bb"]))
        bb_chip = f'<span class="chip">BB {bb_text}</span>'

    top_class = " top" if is_top else ""

    return f"""
    <div class="card{top_class}">
        <div class="card-head">
            <div class="ticker">{ticker}</div>
            <div>
                <div class="score">{result['score']:.1f}</div>
                <div class="score-label">ציון איכות</div>
            </div>
        </div>

        {sparkline(result.get("spark", []))}

        <div class="chips">
            <span class="chip gold">{candle}</span>
            <span class="chip">ATR {result['atr_pct']:.1f}%</span>
            <span class="chip">RVOL {result['rvol']:.2f}</span>
            <span class="chip">R:R {result['rr']:.2f}</span>
            {trend_chip}
            {bb_chip}
        </div>

        <div class="levels">
            <div class="level">
                <span>כניסה</span>
                <b>${result['entry']:.2f}</b>
            </div>
            <div class="level">
                <span>סטופ</span>
                <b>${result['stop']:.2f}</b>
            </div>
            <div class="level">
                <span>יעד ראשון</span>
                <b>${result['tp1']:.2f}</b>
            </div>
        </div>

        <div class="levels">
            <div class="level">
                <span>מניות</span>
                <b>{result['shares']:,}</b>
            </div>
            <div class="level">
                <span>סיכון מתוכנן</span>
                <b>${result['risk_total']:,.0f}</b>
            </div>
            <div class="level">
                <span>מרחק כניסה</span>
                <b>{result['to_entry_pct']:.2f}%</b>
            </div>
        </div>

        <div class="explain">{explanation}</div>
    </div>
    """


def funnel_chart(drop_counts: Counter[str]) -> go.Figure | None:
    if not drop_counts:
        return None

    items = sorted(
        drop_counts.items(),
        key=lambda item: item[1],
    )

    labels = [label for label, _ in items]
    values = [value for _, value in items]

    colors = [
        "#EF4444" if value == max(values) else "#8E4046"
        for value in values
    ]

    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{value:,}" for value in values],
            textposition="outside",
            textfont={"color": "#94A3B8"},
            hovertemplate="%{y}: %{x:,}<extra></extra>",
        )
    )

    figure.update_layout(
        height=max(300, 32 * len(items)),
        margin={"l": 10, "r": 70, "t": 15, "b": 20},
        paper_bgcolor="#121826",
        plot_bgcolor="#121826",
        font={"color": "#94A3B8"},
        xaxis={"showgrid": False, "zeroline": False},
        yaxis={"showgrid": False},
    )

    return figure


def backtest_equity_chart(trades: pd.DataFrame) -> go.Figure:
    ordered = trades.sort_values("date").copy()
    ordered["equity_r"] = ordered["R"].cumsum()

    figure = go.Figure(
        go.Scatter(
            x=ordered["date"],
            y=ordered["equity_r"],
            mode="lines",
            line={"color": "#10B981", "width": 2},
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}R<extra></extra>",
        )
    )

    figure.update_layout(
        height=320,
        margin={"l": 10, "r": 10, "t": 20, "b": 20},
        paper_bgcolor="#121826",
        plot_bgcolor="#121826",
        font={"color": "#94A3B8"},
        xaxis={"showgrid": False},
        yaxis={"gridcolor": "#293348", "title": "R מצטבר"},
    )

    return figure


def results_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    columns = [
        "ticker",
        "score",
        "price",
        "entry",
        "stop",
        "tp1",
        "rr",
        "to_entry_pct",
        "atr_pct",
        "rvol",
        "dry",
        "spike",
        "dollar_volume_m",
        "shares",
        "risk_total",
    ]

    frame = pd.DataFrame(rows)[columns].copy()

    frame.columns = [
        "Ticker",
        "Score",
        "Price",
        "Entry",
        "Stop",
        "Target 1",
        "R:R",
        "To entry %",
        "ATR %",
        "RVOL",
        "Dry ratio",
        "Volume spike",
        "Dollar volume M",
        "Shares",
        "Risk $",
    ]

    return frame


# ============================================================
# UI
# ============================================================

now = datetime.now(ISRAEL_TZ)

st.markdown(
    f"""
    <div class="topbar">
        <div class="brand">
            <div class="brand-name">MAVRI PRO</div>
            <div class="brand-subtitle">סורק מומנטום ותבניות לפני פריצה</div>
        </div>
        <div>
            <div class="clock-label">זמן ישראל</div>
            <div class="clock-value">{now:%d/%m/%Y %H:%M}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "preset_name" not in st.session_state:
    st.session_state["preset_name"] = "מסחר יומי מומנטום"

preset_columns = st.columns([1, 1, 1, 3])

for column, preset_name in zip(preset_columns[:3], PRESETS):
    if column.button(
        preset_name,
        use_container_width=True,
        type=(
            "primary"
            if st.session_state["preset_name"] == preset_name
            else "secondary"
        ),
    ):
        st.session_state["preset_name"] = preset_name
        st.rerun()

base_cfg = PRESETS[st.session_state["preset_name"]]

control_1, control_2, control_3, control_4, control_5, control_6 = st.columns(
    [1, 1, 1, 1.2, 1, 1.5]
)

dry_min = control_1.number_input(
    "יחס יובש מינימלי",
    min_value=0.5,
    max_value=5.0,
    value=float(base_cfg.dry_ratio_min),
    step=0.05,
)

spike_min = control_2.number_input(
    "קפיצת נפח מינימלית",
    min_value=0.5,
    max_value=10.0,
    value=float(base_cfg.spike_ratio_min),
    step=0.1,
)

min_price = control_3.number_input(
    "מחיר מינימום $",
    min_value=1.0,
    max_value=500.0,
    value=float(base_cfg.min_price),
    step=1.0,
)

min_dollar_volume = control_4.number_input(
    "מחזור דולרי ממוצע, מיליון",
    min_value=1.0,
    max_value=1000.0,
    value=float(base_cfg.min_dollar_volume_m),
    step=1.0,
)

min_atr_pct = control_5.number_input(
    "ATR מינימלי %",
    min_value=0.1,
    max_value=20.0,
    value=float(base_cfg.atr_min_pct),
    step=0.1,
)

control_6.markdown(
    "<div style='height:1.75rem'></div>",
    unsafe_allow_html=True,
)

scan_clicked = control_6.button(
    "🚀 סרוק את השוק",
    type="primary",
    use_container_width=True,
)

with st.expander("⚙️ הגדרות מתקדמות"):
    row_1 = st.columns(4)

    min_share_volume = row_1[0].number_input(
        "נפח מניות ממוצע, מיליון",
        min_value=0.0,
        max_value=100.0,
        value=float(base_cfg.min_share_volume_m),
        step=0.25,
    )

    scan_limit = row_1[1].slider(
        "מספר סימולים לסריקה",
        min_value=500,
        max_value=6000,
        value=DEFAULT_SCAN_SIZE,
        step=250,
    )

    account_size = row_1[2].number_input(
        "גודל תיק $",
        min_value=500.0,
        max_value=10_000_000.0,
        value=25_000.0,
        step=500.0,
    )

    risk_percent = row_1[3].slider(
        "סיכון לעסקה %",
        min_value=0.1,
        max_value=3.0,
        value=0.5,
        step=0.1,
    )

    row_2 = st.columns(4)

    rise_min = row_2[0].number_input(
        "זינוק מינימלי %",
        min_value=1.0,
        max_value=200.0,
        value=float(base_cfg.rise_min_pct),
        step=1.0,
    )

    leg_bars_min = row_2[1].number_input(
        "מינימום נרות ברגל העלייה",
        min_value=1,
        max_value=30,
        value=int(base_cfg.leg_bars_min),
        step=1,
    )

    retrace_range = row_2[2].slider(
        "טווח תיקון %",
        min_value=0,
        max_value=100,
        value=(
            int(base_cfg.retrace_min_pct),
            int(base_cfg.retrace_max_pct),
        ),
        step=1,
    )

    rr_min = row_2[3].number_input(
        "יחס סיכוי-סיכון מינימלי",
        min_value=0.5,
        max_value=10.0,
        value=float(base_cfg.rr_min),
        step=0.1,
    )

    row_3 = st.columns(4)

    reversal_required = row_3[0].checkbox(
        "נר היפוך חובה",
        value=base_cfg.reversal_required,
    )

    bb_label = row_3[1].selectbox(
        "פילטר בולינגר",
        options=["כבוי", "תחתונה", "עליונה", "אחת מהשתיים"],
        index=0,
    )

    bb_required = row_3[2].checkbox(
        "מגע בולינגר חובה",
        value=base_cfg.bb_required,
    )

    support_distance = row_3[3].number_input(
        "מרחק מקסימלי מתמיכה %",
        min_value=1.0,
        max_value=50.0,
        value=float(base_cfg.support_distance_max_pct),
        step=1.0,
    )

bb_mapping = {
    "כבוי": "off",
    "תחתונה": "lower",
    "עליונה": "upper",
    "אחת מהשתיים": "both",
}

cfg = replace(
    base_cfg,
    min_price=float(min_price),
    min_dollar_volume_m=float(min_dollar_volume),
    min_share_volume_m=float(min_share_volume),
    dry_ratio_min=float(dry_min),
    spike_ratio_min=float(spike_min),
    atr_min_pct=float(min_atr_pct),
    rise_min_pct=float(rise_min),
    leg_bars_min=int(leg_bars_min),
    retrace_min_pct=float(retrace_range[0]),
    retrace_max_pct=float(retrace_range[1]),
    rr_min=float(rr_min),
    reversal_required=bool(reversal_required),
    bb_side=bb_mapping[bb_label],
    bb_required=bool(bb_required),
    support_distance_max_pct=float(support_distance),
)

st.markdown(
    """
    <div class="notice">
        <strong>חשוב:</strong>
        הסריקה משתמשת בנרות יומיים ולכן מאתרת תבניות וטריגרים אפשריים.
        למסחר יומי בזמן אמת יש לאשר מחיר, נפח, Spread וחדשות באמצעות מקור נתונים תוך-יומי אמין.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# BACKTEST UI
# ============================================================

with st.expander("📊 בדיקה היסטורית שמרנית"):
    st.write(
        "הבדיקה כוללת נזילות היסטורית, יציאת זמן, Slippage, "
        "עמלה והנחה שמרנית כאשר יעד וסטופ נפגעים באותו נר."
    )

    bt_1, bt_2, bt_3, bt_4 = st.columns(4)

    backtest_symbols = bt_1.slider(
        "כמות סימולים לבדיקה",
        min_value=50,
        max_value=1000,
        value=250,
        step=50,
    )

    max_hold_days = bt_2.slider(
        "מקסימום ימי החזקה",
        min_value=1,
        max_value=30,
        value=10,
        step=1,
    )

    slippage_bps = bt_3.number_input(
        "Slippage, נקודות בסיס",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=1.0,
    )

    commission_per_share = bt_4.number_input(
        "עמלה למניה $",
        min_value=0.0,
        max_value=1.0,
        value=0.005,
        step=0.001,
        format="%.3f",
    )

    run_backtest = st.button(
        "הרץ Backtest",
        type="secondary",
        use_container_width=True,
    )

    if run_backtest:
        try:
            universe = build_universe(backtest_symbols)
            batches = list(chunked(universe, BATCH_SIZE))
            progress = st.progress(0.0)
            status = st.empty()
            trades: list[dict[str, Any]] = []

            for batch_index, batch in enumerate(batches):
                status.caption(
                    f"בדיקה היסטורית: מנה {batch_index + 1} "
                    f"מתוך {len(batches)}"
                )

                data = fetch_batch(batch, period="5y")

                for ticker, frame in data.items():
                    try:
                        trades.extend(
                            backtest_symbol(
                                ticker=ticker,
                                frame=frame,
                                cfg=cfg,
                                max_hold_days=int(max_hold_days),
                                cooldown_days=5,
                                slippage_bps=float(slippage_bps),
                                commission_per_share=float(
                                    commission_per_share
                                ),
                            )
                        )
                    except Exception:
                        continue

                progress.progress((batch_index + 1) / len(batches))

            progress.empty()
            status.empty()

            if not trades:
                st.warning(
                    "לא נמצאו עסקאות. ייתכן שהסינון מחמיר מדי "
                    "או שלא התקבל מספיק מידע היסטורי."
                )
            else:
                bt_frame = pd.DataFrame(trades)
                bt_frame = bt_frame.sort_values("date")
                summary = summarize_backtest(bt_frame)

                kpi_columns = st.columns(6)
                kpi_columns[0].metric(
                    "עסקאות",
                    f"{int(summary['trades']):,}",
                )
                kpi_columns[1].metric(
                    "אחוז הצלחה",
                    f"{summary['win_rate']:.1f}%",
                )
                kpi_columns[2].metric(
                    "תוחלת",
                    f"{summary['expectancy']:+.2f}R",
                )
                kpi_columns[3].metric(
                    "Profit Factor",
                    (
                        f"{summary['profit_factor']:.2f}"
                        if np.isfinite(summary["profit_factor"])
                        else "∞"
                    ),
                )
                kpi_columns[4].metric(
                    "Max Drawdown",
                    f"{summary['max_drawdown_r']:.1f}R",
                )
                kpi_columns[5].metric(
                    "זמן החזקה ממוצע",
                    f"{summary['avg_hold']:.1f} ימים",
                )

                st.plotly_chart(
                    backtest_equity_chart(bt_frame),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

                outcome_counts = (
                    bt_frame["outcome"]
                    .value_counts()
                    .rename_axis("Outcome")
                    .reset_index(name="Trades")
                )

                st.dataframe(
                    outcome_counts,
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as exc:
            st.error(f"שגיאה בבדיקה ההיסטורית: {exc}")


# ============================================================
# MARKET SCAN
# ============================================================

if scan_clicked:
    started = time.perf_counter()

    try:
        universe = build_universe(int(scan_limit))
    except Exception as exc:
        st.error(f"לא ניתן לבנות את רשימת הסימולים: {exc}")
        st.stop()

    batches = list(chunked(universe, BATCH_SIZE))
    progress = st.progress(0.0)
    status = st.empty()

    rows: list[dict[str, Any]] = []
    drops: Counter[str] = Counter()
    downloaded_count = 0
    technical_errors = 0

    for batch_index, batch in enumerate(batches):
        status.caption(
            f"סורק מנה {batch_index + 1} מתוך {len(batches)} | "
            f"נבדקו {downloaded_count:,} | נמצאו {len(rows):,}"
        )

        batch_data = fetch_batch(batch, period="1y")
        missing = len(batch) - len(batch_data)

        if missing > 0:
            drops["לא התקבלו נתונים"] += missing

        for ticker, frame in batch_data.items():
            downloaded_count += 1

            try:
                prepared = prepare_frame(frame)
                result, reason = evaluate_setup(
                    prepared,
                    len(prepared) - 1,
                    cfg,
                )

                if result is None:
                    drops[reason] += 1
                    continue

                risk_budget = account_size * risk_percent / 100
                risk_per_share = float(result["risk_share"])

                risk_based_shares = (
                    math.floor(risk_budget / risk_per_share)
                    if risk_per_share > 0
                    else 0
                )

                # Prevent position value from exceeding the account.
                cash_based_shares = math.floor(
                    account_size / float(result["entry"])
                )

                shares = max(
                    0,
                    min(risk_based_shares, cash_based_shares),
                )

                result.update(
                    ticker=ticker,
                    shares=shares,
                    risk_total=round(
                        shares * risk_per_share,
                        2,
                    ),
                )

                rows.append(result)

            except Exception:
                technical_errors += 1
                drops["שגיאת נתונים"] += 1

        progress.progress((batch_index + 1) / len(batches))

    progress.empty()
    status.empty()

    rows.sort(
        key=lambda row: (
            -row["score"],
            row["to_entry_pct"],
            -row["dollar_volume_m"],
        )
    )

    elapsed = time.perf_counter() - started

    st.session_state["scan_result"] = {
        "rows": rows[:MAX_RESULTS],
        "universe_count": len(universe),
        "downloaded_count": downloaded_count,
        "drops": dict(drops),
        "elapsed": elapsed,
        "technical_errors": technical_errors,
        "scanned_at": datetime.now(ISRAEL_TZ).isoformat(),
    }


# ============================================================
# RESULTS
# ============================================================

scan_result = st.session_state.get("scan_result")

if scan_result:
    rows = scan_result["rows"]
    drops = Counter(scan_result["drops"])

    top_score = rows[0]["score"] if rows else 0.0

    st.markdown(
        f"""
        <div class="kpis">
            <div class="kpi">
                <strong>{scan_result['universe_count']:,}</strong>
                <span>סימולים ביקום</span>
            </div>
            <div class="kpi">
                <strong>{scan_result['downloaded_count']:,}</strong>
                <span>התקבלו נתונים</span>
            </div>
            <div class="kpi good">
                <strong>{len(rows):,}</strong>
                <span>תבניות שנמצאו</span>
            </div>
            <div class="kpi">
                <strong>{top_score:.1f}</strong>
                <span>הציון הגבוה ביותר</span>
            </div>
            <div class="kpi">
                <strong>{scan_result['elapsed']:.1f}s</strong>
                <span>זמן סריקה</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if drops:
        top_reason, top_count = drops.most_common(1)[0]

        st.markdown(
            f"""
            <div class="notice">
                המסנן שהוציא הכי הרבה סימולים:
                <strong>{escape(top_reason)}</strong>
                ({top_count:,}).
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("🔻 מפל הסינון"):
        figure = funnel_chart(drops)

        if figure is not None:
            st.plotly_chart(
                figure,
                use_container_width=True,
                config={"displayModeBar": False},
            )

    if not rows:
        st.warning(
            "לא נמצאו תבניות. במקום לבטל את כל המסננים, "
            "הפחת בהדרגה את דרישת קפיצת הנפח, יובש הנפח "
            "או יחס הסיכוי-סיכון."
        )

    else:
        st.subheader("🔥 התבניות המדורגות")

        card_columns = st.columns(3, gap="large")

        for index, result in enumerate(rows[:15]):
            with card_columns[index % 3]:
                st.markdown(
                    card_html(result, is_top=index == 0),
                    unsafe_allow_html=True,
                )

        st.subheader("טבלת תוצאות מלאה")

        table = results_table(rows)

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score",
                    min_value=0,
                    max_value=100,
                    format="%.1f",
                ),
                "Price": st.column_config.NumberColumn(
                    "Price",
                    format="$%.2f",
                ),
                "Entry": st.column_config.NumberColumn(
                    "Entry",
                    format="$%.2f",
                ),
                "Stop": st.column_config.NumberColumn(
                    "Stop",
                    format="$%.2f",
                ),
                "Target 1": st.column_config.NumberColumn(
                    "Target 1",
                    format="$%.2f",
                ),
                "Risk $": st.column_config.NumberColumn(
                    "Risk $",
                    format="$%.2f",
                ),
            },
        )

        csv_bytes = table.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="הורד תוצאות CSV",
            data=csv_bytes,
            file_name="mavri_scan.csv",
            mime="text/csv",
            use_container_width=True,
        )
