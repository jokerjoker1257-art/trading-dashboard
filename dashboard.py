"""
Pro Trading Terminal — Gold + BTC
Manual trading decision support. Signals suggestion hain, order nahi.

Chalane ke liye:  streamlit run dashboard.py
Optional .env / environment:
    FINNHUB_API_KEY=...        (asli economic calendar ke liye)
    LOCAL_TZ=Asia/Karachi      (aapka local timezone)
"""
from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta, date, time as dtime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import ta
import yfinance as yf
from plotly.subplots import make_subplots

UTC = timezone.utc
NY = ZoneInfo("America/New_York")
LOCAL = ZoneInfo(os.getenv("LOCAL_TZ", "Asia/Karachi"))
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")

SYMBOLS = {"Gold (XAUUSD)": "GC=F", "Bitcoin (BTC)": "BTC-USD"}

SIGNAL_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signals.csv")
LOG_COLS = ["logged_at", "symbol", "signal", "score", "price", "atr"]
SCORE_AFTER_H = 12          # kitne ghante baad signal ko score karna hai

# FOMC 2026 — federalreserve.gov se verify karein, ye badal sakti hain
FOMC_DATES = ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
              "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"]

st.set_page_config(page_title="Pro Trading Terminal", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp { background: linear-gradient(180deg,#0f172a 0%,#1e293b 100%); color:#e2e8f0; }
.main-title { font-size:1.9rem; font-weight:700; color:#38bdf8; margin-bottom:4px; }
.signal-box { padding:14px 20px; border-radius:10px; font-size:1.25rem; font-weight:700;
              text-align:center; margin:10px 0 6px 0; letter-spacing:.5px; }
.buy  { background:linear-gradient(90deg,#065f46,#10b981); color:#fff;
        box-shadow:0 4px 15px rgba(16,185,129,.25); }
.sell { background:linear-gradient(90deg,#7f1d1d,#ef4444); color:#fff;
        box-shadow:0 4px 15px rgba(239,68,68,.25); }
.neutral { background:linear-gradient(90deg,#334155,#64748b); color:#e2e8f0; }
div[data-testid="stMetricValue"] { color:#f8fafc !important; font-weight:600; }
div[data-testid="stMetricLabel"] { color:#94a3b8 !important; }
.muted { color:#94a3b8; font-size:.82rem; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════ DATA ═══════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def get_data(symbol: str, period: str = "12d", interval: str = "1h"):
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval)
    except (requests.RequestException, ValueError, KeyError) as e:
        st.session_state.setdefault("errors", []).append(f"{symbol}: {e}")
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.copy()
    df["EMA9"] = ta.trend.ema_indicator(df["Close"], window=9)
    df["EMA20"] = ta.trend.ema_indicator(df["Close"], window=20)
    df["EMA50"] = ta.trend.ema_indicator(df["Close"], window=50)
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
    df["MACD"] = ta.trend.macd_diff(df["Close"])
    return df


@st.cache_data(ttl=600, show_spinner=False)
def get_crypto_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=6)
        r.raise_for_status()
        d = r.json()["data"][0]
        return int(d["value"]), d["value_classification"]
    except (requests.RequestException, KeyError, ValueError, IndexError):
        return None, "N/A"


@st.cache_data(ttl=900, show_spinner=False)
def get_news(symbol: str, limit: int = 6) -> list[dict]:
    out = []
    try:
        for n in (yf.Ticker(symbol).news or [])[:limit]:
            c = n.get("content", n)
            title = c.get("title") or n.get("title", "")
            if not title:
                continue
            prov = c.get("provider", {})
            out.append(dict(
                title=title[:120],
                source=(prov.get("displayName") if isinstance(prov, dict) else None)
                       or n.get("publisher", "Yahoo Finance"),
                url=(c.get("canonicalUrl") or {}).get("url", "") or n.get("link", "")))
    except (requests.RequestException, KeyError, TypeError, AttributeError):
        pass
    return out


# ═══════════════════ ECONOMIC CALENDAR ═══════════════════
def _first_friday(y: int, m: int) -> date:
    d = date(y, m, 1)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def _ny_to_utc(d: date, hh: int, mm: int) -> datetime:
    """New York local -> UTC. DST khud handle hoti hai."""
    return datetime.combine(d, dtime(hh, mm), tzinfo=NY).astimezone(UTC)


def _builtin_calendar(days: int) -> list[dict]:
    out, today = [], datetime.now(UTC).date()
    for off in range(days + 1):
        d = today + timedelta(days=off)
        if d == _first_friday(d.year, d.month):
            out += [dict(when=_ny_to_utc(d, 8, 30), name="Non-Farm Payrolls (NFP)",
                         cur="USD", impact="High"),
                    dict(when=_ny_to_utc(d, 8, 30), name="Unemployment Rate",
                         cur="USD", impact="High")]
        if 10 <= d.day <= 15 and d.weekday() < 5:
            out.append(dict(when=_ny_to_utc(d, 8, 30), name="CPI / Core CPI (approx)",
                            cur="USD", impact="High"))
        if d.isoformat() in FOMC_DATES:
            out += [dict(when=_ny_to_utc(d, 14, 0), name="FOMC Rate Decision",
                         cur="USD", impact="High"),
                    dict(when=_ny_to_utc(d, 14, 30), name="FOMC Press Conference",
                         cur="USD", impact="High")]
        if d.day <= 3 and d.weekday() == 0:
            out.append(dict(when=_ny_to_utc(d, 10, 0), name="ISM Manufacturing PMI",
                            cur="USD", impact="Medium"))
    return out


@st.cache_data(ttl=1800, show_spinner=False)
def get_calendar(days: int = 2) -> tuple[list[dict], str]:
    today = datetime.now(UTC).date()
    if FINNHUB_KEY:
        try:
            r = requests.get("https://finnhub.io/api/v1/calendar/economic",
                             params={"from": str(today),
                                     "to": str(today + timedelta(days=days)),
                                     "token": FINNHUB_KEY}, timeout=8)
            r.raise_for_status()
            evs = []
            for e in (r.json().get("economicCalendar") or []):
                imp = str(e.get("impact", "")).capitalize()
                if imp not in ("High", "Medium"):
                    continue
                try:
                    w = datetime.strptime(e["time"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
                except (KeyError, ValueError):
                    continue
                evs.append(dict(when=w, name=e.get("event", "?"),
                                cur=e.get("country", ""), impact=imp))
            if evs:
                return sorted(evs, key=lambda x: x["when"]), "FinnHub (live)"
        except (requests.RequestException, ValueError):
            pass
    return sorted(_builtin_calendar(days), key=lambda x: x["when"]), "Built-in schedule"


def news_window_active(events: list[dict], mins: int = 30) -> bool:
    now = datetime.now(UTC)
    return any(e["impact"] == "High"
               and abs((e["when"] - now).total_seconds()) <= mins * 60
               for e in events)


# ═══════════════════════ SIGNAL ═══════════════════════
def get_signal(df):
    if df is None or len(df) < 50:
        return "WAIT", "neutral", 0.0, "Insufficient data"
    last = df.iloc[-1]
    close, e9, e20, e50 = last["Close"], last["EMA9"], last["EMA20"], last["EMA50"]
    rsi, macd = last["RSI"], last["MACD"]
    score, why = 0.0, []

    if close > e50 and e20 > e50:
        score += 2; why.append("Uptrend structure")
    elif close < e50 and e20 < e50:
        score -= 2; why.append("Downtrend structure")

    if e9 > e20:
        score += 1.5; why.append("EMA9 > EMA20")
    else:
        score -= 1.5; why.append("EMA9 < EMA20")

    if 52 < rsi < 68:
        score += 1; why.append("RSI bullish")
    elif 32 < rsi < 48:
        score -= 1; why.append("RSI bearish")
    elif rsi >= 70:
        score -= 0.5; why.append("RSI overbought")
    elif rsi <= 30:
        score += 0.5; why.append("RSI oversold")

    score += 1 if macd > 0 else -1

    if score >= 3.5:   return "STRONG BUY", "buy", score, " | ".join(why[:3])
    if score >= 1.8:   return "BUY", "buy", score, " | ".join(why[:3])
    if score <= -3.5:  return "STRONG SELL", "sell", score, " | ".join(why[:3])
    if score <= -1.8:  return "SELL", "sell", score, " | ".join(why[:3])
    return "WAIT", "neutral", score, "No clear edge"


def get_session() -> str:
    h = datetime.now(UTC).hour
    if 0 <= h < 7:   return "Asian"
    if 7 <= h < 12:  return "London"
    if 12 <= h < 16: return "London–NY Overlap"
    if 16 <= h < 21: return "New York"
    return "Off Session"


def make_chart(df, title):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.03,
                        row_heights=[.73, .27], subplot_titles=(title, "RSI (14)"))
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                 low=df["Low"], close=df["Close"], name="Price",
                                 increasing_line_color="#10b981",
                                 decreasing_line_color="#ef4444"), row=1, col=1)
    for col, colr, w in (("EMA9", "#facc15", 1.3), ("EMA20", "#22d3ee", 1.5),
                         ("EMA50", "#fb923c", 1.7)):
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col,
                                 line=dict(color=colr, width=w)), row=1, col=1)
    hi, lo = df["High"].tail(40).max(), df["Low"].tail(40).min()
    fig.add_hline(y=hi, line_dash="dot", line_color="#ef4444", line_width=1.4,
                  annotation_text="40-bar high", annotation_position="top right", row=1, col=1)
    fig.add_hline(y=lo, line_dash="dot", line_color="#10b981", line_width=1.4,
                  annotation_text="40-bar low", annotation_position="bottom right", row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                             line=dict(color="#c084fc", width=1.4)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,68,68,.55)", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(16,185,129,.55)", row=2, col=1)
    fig.update_layout(height=510, template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,.6)",
                      xaxis_rangeslider_visible=False, hovermode="x unified",
                      legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"),
                      margin=dict(l=10, r=10, t=40, b=10))
    fig.update_xaxes(gridcolor="rgba(148,163,184,.12)", row=1, col=1)
    fig.update_yaxes(gridcolor="rgba(148,163,184,.12)", row=1, col=1)
    fig.update_yaxes(range=[15, 85], gridcolor="rgba(148,163,184,.12)", row=2, col=1)
    return fig


# ═══════════════ SIGNAL LOG + SELF-SCORING ═══════════════
# Dashboard ka sabse ahem hissa: apna record khud rakhta hai.
# Signal tabhi log hota hai jab wo BADALTA hai (har 60s nahi).

def read_log() -> list[dict]:
    if not os.path.exists(SIGNAL_LOG):
        return []
    try:
        with open(SIGNAL_LOG, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except OSError:
        return []


def log_signal(symbol: str, signal: str, score: float, price: float, atr: float) -> None:
    rows = read_log()
    prev = [r for r in rows if r["symbol"] == symbol]
    if prev and prev[-1]["signal"] == signal:
        return                                  # wahi signal — dobara log nahi
    new = dict(logged_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
               symbol=symbol, signal=signal, score=f"{score:.1f}",
               price=f"{price:.2f}", atr=f"{atr:.2f}")
    try:
        exists = os.path.exists(SIGNAL_LOG)
        with open(SIGNAL_LOG, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=LOG_COLS)
            if not exists:
                w.writeheader()
            w.writerow(new)
    except OSError:
        pass


def signal_age(symbol: str) -> str:
    rows = [r for r in read_log() if r["symbol"] == symbol]
    if not rows:
        return "naya"
    try:
        t = datetime.strptime(rows[-1]["logged_at"], "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
    except ValueError:
        return "?"
    h = (datetime.now(UTC) - t).total_seconds() / 3600
    return f"{h:.0f}h" if h >= 1 else f"{h*60:.0f}m"


def score_log(price_frames: dict[str, "pd.DataFrame"]) -> "pd.DataFrame":
    """Har purane signal ko dekho: SCORE_AFTER_H ghante baad price kya hui."""
    rows, now = read_log(), datetime.now(UTC)
    out = []
    for r in rows:
        try:
            t = datetime.strptime(r["logged_at"], "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
            entry = float(r["price"])
        except (ValueError, KeyError):
            continue
        if (now - t).total_seconds() / 3600 < SCORE_AFTER_H:
            continue                             # abhi pakka nahi
        df = price_frames.get(r["symbol"])
        if df is None or df.empty:
            continue
        target = t + timedelta(hours=SCORE_AFTER_H)
        idx = df.index.tz_convert(UTC) if df.index.tz is not None else df.index.tz_localize(UTC)
        later = df[idx >= target]
        if later.empty:
            continue
        exit_px = float(later.iloc[0]["Close"])
        ret = (exit_px / entry - 1) * 100
        direction = 1 if "BUY" in r["signal"] else -1 if "SELL" in r["signal"] else 0
        if direction == 0:
            continue
        out.append(dict(symbol=r["symbol"], signal=r["signal"], ret=direction * ret))
    return pd.DataFrame(out)


def calc_lot(symbol: str, equity: float, risk_pct: float, sl_dist: float) -> float:
    """XAUUSD: 1 lot = 100 oz. BTC: 1 lot = 1 BTC. Broker se verify karein."""
    if sl_dist <= 0:
        return 0.0
    contract = 100.0 if "GC" in symbol or "XAU" in symbol else 1.0
    return (equity * risk_pct / 100.0) / (sl_dist * contract)


# ═══════════════════════ RENDER ═══════════════════════
def render():
    now = datetime.now(UTC)
    events, cal_src = get_calendar(2)
    blocked = news_window_active(events)

    st.markdown('<p class="main-title">Pro Trading Terminal — Gold + BTC</p>',
                unsafe_allow_html=True)
    st.caption(f"{now.strftime('%Y-%m-%d %H:%M:%S')} UTC · "
               f"{now.astimezone(LOCAL).strftime('%H:%M')} local ({LOCAL.key})")

    if blocked:
        st.error("**NEWS WINDOW ACTIVE** — high-impact release ±30 min. "
                 "Naye entries se bachein; khuli trades break-even pe le aayein.")

    fg, fg_txt = get_crypto_fear_greed()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Crypto Fear & Greed", fg if fg else "—", fg_txt)
    c2.metric("Session (UTC)", get_session())
    c3.metric("Mode", "Manual")
    c4.metric("News", "BLOCKED" if blocked else "Clear")
    st.divider()

    frames = {}
    cols = st.columns(2)
    for col, (name, sym) in zip(cols, SYMBOLS.items()):
        with col:
            st.subheader(name)
            df = get_data(sym)
            if df is None or len(df) <= 50:
                st.error("Data not available right now")
                continue
            last = df.iloc[-1]
            price = float(last["Close"])
            atr = 0.0 if pd.isna(last["ATR"]) else float(last["ATR"])
            rsi = 0.0 if pd.isna(last["RSI"]) else float(last["RSI"])
            sig, kind, score, why = get_signal(df)
            if blocked:
                sig, kind = "WAIT (news)", "neutral"

            st.metric("Price", f"${price:,.2f}")
            st.markdown(f'<div class="signal-box {kind}">{sig} | Score: {score:.1f}</div>',
                        unsafe_allow_html=True)
            st.caption(f"Reason: {why} · signal {signal_age(sym)} purana")
            if not blocked:
                log_signal(sym, sig, score, price, atr)

            m = st.columns(4)
            m[0].metric("RSI", f"{rsi:.1f}")
            m[1].metric("ATR", f"{atr:.2f}")
            m[2].metric("40-bar low", f"{df['Low'].tail(40).min():.2f}")
            m[3].metric("40-bar high", f"{df['High'].tail(40).max():.2f}")
            sl_dist = atr * 1.4
            lot = calc_lot(sym, st.session_state.get("equity", 5000.0),
                           st.session_state.get("risk_pct", 0.5), sl_dist)
            st.caption(f"SL distance ≈ {sl_dist:.2f} → lot ≈ **{lot:.2f}** "
                       f"({st.session_state.get('risk_pct', 0.5)}% risk) · "
                       f"40-bar high/low asli S/R nahi hai")
            frames[sym] = df
            st.plotly_chart(make_chart(df.tail(100), name), use_container_width=True)

    st.divider()

    # ── Economic calendar ──
    st.subheader("High Impact Economic Events")
    st.caption(f"Source: {cal_src} · UTC / local ({LOCAL.key})")
    if not events:
        st.info("Agle 2 din koi high/medium impact event nahi.")
    for e in events:
        mins = (e["when"] - now).total_seconds() / 60
        if abs(mins) <= 30:
            badge, note = "🔴", "**ABHI — trade mat lein**"
        elif 0 < mins <= 120:
            badge, note = "🟠", f"{int(mins)} min baaki"
        elif mins < 0:
            badge, note = "⚪", "guzar chuka"
        else:
            badge = "🟡" if e["impact"] == "High" else "⚪"
            note = e["when"].strftime("%a %d %b")
        st.markdown(
            f"{badge} **{e['name']}** · {e['cur']} · `{e['impact']}`  \n"
            f"<span class='muted'>{e['when'].strftime('%H:%M')} UTC — "
            f"{e['when'].astimezone(LOCAL).strftime('%H:%M')} local — {note}</span>",
            unsafe_allow_html=True)

    st.divider()

    # ── Signal track record ──
    st.subheader("Signal Track Record")
    st.caption(f"Har signal {SCORE_AFTER_H}h baad khud score hota hai · signals.csv")
    scored = score_log(frames)
    if scored.empty:
        n_logged = len(read_log())
        st.info(f"{n_logged} signals log hue hain. Scoring {SCORE_AFTER_H}h baad "
                f"shuru hoti hai — chalte rehne dein.")
    else:
        n = len(scored)
        avg = scored.ret.mean()
        hit = 100 * (scored.ret > 0).mean()
        sd = scored.ret.std()
        tstat = avg / (sd / (n ** 0.5)) if n > 1 and sd and sd > 0 else 0.0
        k = st.columns(4)
        k[0].metric("Scored signals", n)
        k[1].metric("Hit rate", f"{hit:.1f}%")
        k[2].metric(f"Avg {SCORE_AFTER_H}h move", f"{avg:+.3f}%")
        k[3].metric("t-statistic", f"{tstat:+.2f}")
        if n < 30:
            st.warning(f"Abhi faisla nahi — {30 - n} aur signals chahiye.")
        elif tstat > 2:
            st.success("Coin flip se behtar lag raha hai. 50+ tak jaari rakhein.")
        elif tstat < -2:
            st.error("Signal ULTA chal raha hai — istemal band karein.")
        else:
            st.info("Coin flip se farq nahi. Abhi tak koi edge nahi.")
        st.dataframe(scored.groupby("signal").ret.agg(["count", "mean"])
                     .round(3).rename(columns={"count": "n", "mean": "avg %"}),
                     use_container_width=True)

    st.divider()

    # ── News ──
    st.subheader("Latest News")
    for tab, (name, sym) in zip(st.tabs(list(SYMBOLS.keys())), SYMBOLS.items()):
        with tab:
            items = get_news(sym)
            if not items:
                st.caption("Is waqt koi news nahi mili.")
            for it in items:
                t = f"[{it['title']}]({it['url']})" if it["url"] else f"**{it['title']}**"
                st.markdown(f"• {t}  \n<span class='muted'>{it['source']}</span>",
                            unsafe_allow_html=True)


with st.sidebar:
    st.title("Settings")
    auto = st.checkbox("Auto refresh (60s)", value=True)
    if st.button("Refresh now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.session_state["equity"] = st.number_input("Account equity ($)", 100.0,
                                                 1_000_000.0, 5000.0, 100.0)
    st.session_state["risk_pct"] = st.number_input("Risk per trade (%)", 0.1,
                                                   2.0, 0.5, 0.1)
    st.markdown("---")
    st.markdown("**How to use**")
    st.markdown("1. News window check karein\n2. Signal + score dekhein\n"
                "3. Reason parhein\n4. Apna faisla karke manual trade lein")
    st.markdown("---")
    st.markdown("**Signal log** (cloud pe restart se mit jata hai)")
    _rows = read_log()
    st.caption(f"{len(_rows)} signals logged")
    if _rows:
        import io as _io
        _buf = _io.StringIO()
        _w = csv.DictWriter(_buf, fieldnames=LOG_COLS)
        _w.writeheader(); _w.writerows(_rows)
        st.download_button("Download signals.csv", _buf.getvalue(),
                           "signals.csv", "text/csv", use_container_width=True)
    _up = st.file_uploader("Restore signals.csv", type="csv",
                           label_visibility="collapsed")
    if _up is not None:
        try:
            _new = list(csv.DictReader(_up.getvalue().decode("utf-8").splitlines()))
            _seen = {(r["logged_at"], r["symbol"]) for r in _rows}
            _add = [r for r in _new if (r.get("logged_at"), r.get("symbol")) not in _seen]
            if _add:
                _ex = os.path.exists(SIGNAL_LOG)
                with open(SIGNAL_LOG, "a", newline="", encoding="utf-8") as _fh:
                    _ww = csv.DictWriter(_fh, fieldnames=LOG_COLS)
                    if not _ex:
                        _ww.writeheader()
                    _ww.writerows([{k: r.get(k, "") for k in LOG_COLS} for r in _add])
                st.success(f"{len(_add)} signals restore ho gaye")
            else:
                st.info("Koi naya signal nahi mila")
        except (ValueError, KeyError, UnicodeDecodeError) as _e:
            st.error(f"File parh nahi saka: {_e}")
    st.markdown("---")
    st.caption("Signals suggestion hain, order nahi. "
               "Risk per trade 0.5% se zyada nahi.")

if auto and hasattr(st, "fragment"):
    render = st.fragment(run_every=60)(render)   # UI block nahi hoti
render()
