import yfinance as yf
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Expande Tu Futuro Web")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ─────────────────────────────────────────────
# MAPEO DE INTERVALOS
# 1d  → vela diaria   → descarga 2 años
# 1wk → vela semanal  → descarga max
# 1mo → vela mensual  → descarga max
# 3mo → vela trimest. → descarga max
# ─────────────────────────────────────────────
INTERVAL_MAP = {
    "1d":  ("1d",  "2y"),
    "1wk": ("1wk", "max"),
    "1mo": ("1mo", "max"),
    "3mo": ("3mo", "max"),
}


def clean_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ─────────────────────────────────────────────
# INDICADORES (siempre sobre datos diarios 2y)
# ─────────────────────────────────────────────

def calcular_indicadores(df):
    df["SMA20"]  = df["Close"].rolling(20).mean()
    df["SMA50"]  = df["Close"].rolling(50).mean()
    df["SMA100"] = df["Close"].rolling(100).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    # Supertrend
    df["TR"] = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = df["TR"].rolling(7).mean()
    hl2 = (df["High"] + df["Low"]) / 2
    df["UB"] = hl2 + 3.0 * df["ATR"]
    df["LB"] = hl2 - 3.0 * df["ATR"]
    df["ST"]  = np.nan
    df["Dir"] = 0
    for i in range(1, len(df)):
        ps = df.iloc[i-1]["ST"]; pd_ = df.iloc[i-1]["Dir"]
        cl = float(df.iloc[i]["LB"]); cu = float(df.iloc[i]["UB"]); cc = float(df.iloc[i]["Close"])
        if np.isnan(ps):
            df.iloc[i, df.columns.get_loc("ST")]  = cl
            df.iloc[i, df.columns.get_loc("Dir")] = 1
            continue
        st = max(cl, ps) if pd_ == 1 else min(cu, ps)
        df.iloc[i, df.columns.get_loc("ST")]  = st
        df.iloc[i, df.columns.get_loc("Dir")] = 1 if cc > st else -1
    return df


# ─────────────────────────────────────────────
# ALERTAS
# ─────────────────────────────────────────────

def detectar_alertas(df_daily, ticker=""):
    alertas = []
    n = len(df_daily) - 1
    if n < 2:
        return alertas

    precio_now  = float(df_daily["Close"].iloc[n])
    precio_prev = float(df_daily["Close"].iloc[n - 1])
    prefix = f"[{ticker}] " if ticker else ""

    smas = {
        "SMA20":  (df_daily["SMA20"].iloc[n],  df_daily["SMA20"].iloc[n-1]),
        "SMA50":  (df_daily["SMA50"].iloc[n],  df_daily["SMA50"].iloc[n-1]),
        "SMA100": (df_daily["SMA100"].iloc[n], df_daily["SMA100"].iloc[n-1]),
        "SMA200": (df_daily["SMA200"].iloc[n], df_daily["SMA200"].iloc[n-1]),
    }

    for nombre, (sma_now, sma_prev) in smas.items():
        if not (pd.notna(sma_now) and pd.notna(sma_prev)):
            continue
        if precio_prev < sma_prev and precio_now >= sma_now:
            alertas.append({"nivel": "bullish",
                "msg": f"📈 {prefix}Precio cruza {nombre} al alza — ${precio_now:,.2f}"})
        elif precio_prev > sma_prev and precio_now <= sma_now:
            alertas.append({"nivel": "bearish",
                "msg": f"📉 {prefix}Precio cruza {nombre} a la baja — ${precio_now:,.2f}"})
        elif abs(precio_now - sma_now) / sma_now * 100 <= 0.4:
            alertas.append({"nivel": "info",
                "msg": f"⚠️ {prefix}Precio tocando {nombre} — ${precio_now:,.2f}"})

    # Cruces entre medias importantes
    s100_n, s100_p = smas["SMA100"]
    s200_n, s200_p = smas["SMA200"]
    if pd.notna(s100_n) and pd.notna(s200_n):
        if s100_p < s200_p and s100_n >= s200_n:
            alertas.append({"nivel": "bullish", "msg": f"🟢 {prefix}Golden Cross — SMA100 sobre SMA200"})
        elif s100_p > s200_p and s100_n <= s200_n:
            alertas.append({"nivel": "bearish", "msg": f"🔴 {prefix}Death Cross — SMA100 bajo SMA200"})

    s20_n, s20_p = smas["SMA20"]
    s50_n, s50_p = smas["SMA50"]
    if pd.notna(s20_n) and pd.notna(s50_n):
        if s20_p < s50_p and s20_n >= s50_n:
            alertas.append({"nivel": "bullish", "msg": f"🟡 {prefix}SMA20 cruza sobre SMA50"})
        elif s20_p > s50_p and s20_n <= s50_n:
            alertas.append({"nivel": "bearish", "msg": f"🟠 {prefix}SMA20 cruza bajo SMA50"})

    return alertas


def get_daily_df(ticker):
    df = yf.download(ticker.upper(), period="2y", interval="1d", progress=False)
    if df.empty:
        return None
    df = clean_df(df)
    df = calcular_indicadores(df)
    return df


def ts_ms(idx):
    """Convierte un DatetimeIndex a lista de milliseconds UTC para Chart.js."""
    return [int(t.timestamp() * 1000) for t in idx]


def safe(v):
    return float(v) if pd.notna(v) else None


# ─────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/chart/{ticker}")
async def get_chart(ticker: str, interval: str = "1d"):
    try:
        yf_interval, yf_period = INTERVAL_MAP.get(interval, ("1d", "2y"))

        # ── Datos de velas (OHLC) según el intervalo elegido ──
        df_candles = yf.download(ticker.upper(), period=yf_period,
                                 interval=yf_interval, progress=False)
        if df_candles.empty:
            return {"error": f"Símbolo no encontrado: {ticker}"}
        df_candles = clean_df(df_candles)

        # ── Datos diarios para SMAs e indicadores (siempre 2y) ──
        if interval == "1d":
            df_daily = df_candles.copy()
            df_daily = calcular_indicadores(df_daily)
        else:
            df_daily = get_daily_df(ticker)
            if df_daily is None:
                df_daily = df_candles.copy()
                df_daily = calcular_indicadores(df_daily)

        # ── Construir velas OHLC para Chart.js Financial ──
        timestamps = ts_ms(df_candles.index)
        candles = [
            {"x": timestamps[i],
             "o": safe(df_candles["Open"].iloc[i]),
             "h": safe(df_candles["High"].iloc[i]),
             "l": safe(df_candles["Low"].iloc[i]),
             "c": safe(df_candles["Close"].iloc[i])}
            for i in range(len(df_candles))
        ]

        # ── SMAs: interpolar al mismo eje temporal que las velas ──
        # Para intervalos no-diarios, usamos los valores diarios y los
        # resampleamos al mismo timeframe para que la línea tenga el mismo eje x.
        def sma_series(col):
            """Devuelve [{x, y}] para una columna SMA del df_daily."""
            series = []
            for i in range(len(df_daily)):
                v = df_daily[col].iloc[i]
                if pd.notna(v):
                    series.append({"x": int(df_daily.index[i].timestamp() * 1000),
                                   "y": float(v)})
            return series

        # RSI señales sobre precio
        rsi_vals   = df_daily["RSI"].values
        close_vals = df_daily["Close"].values
        daily_ts   = ts_ms(df_daily.index)

        rsi_os = [{"x": daily_ts[i], "y": float(close_vals[i])}
                  for i in range(len(df_daily))
                  if pd.notna(rsi_vals[i]) and rsi_vals[i] < 30]

        rsi_ob = [{"x": daily_ts[i], "y": float(close_vals[i])}
                  for i in range(len(df_daily))
                  if pd.notna(rsi_vals[i]) and rsi_vals[i] > 70]

        # Supertrend (solo en modo diario, el resto quedaría confuso)
        st_buy  = [{"x": daily_ts[i], "y": float(df_daily["ST"].iloc[i])}
                   for i in range(len(df_daily))
                   if pd.notna(df_daily["ST"].iloc[i]) and df_daily["Dir"].iloc[i] == 1]
        st_sell = [{"x": daily_ts[i], "y": float(df_daily["ST"].iloc[i])}
                   for i in range(len(df_daily))
                   if pd.notna(df_daily["ST"].iloc[i]) and df_daily["Dir"].iloc[i] == -1]

        chart_data = {
            "candles":  candles,
            "sma20":    sma_series("SMA20"),
            "sma50":    sma_series("SMA50"),
            "sma100":   sma_series("SMA100"),
            "sma200":   sma_series("SMA200"),
            "rsi_os":   rsi_os,
            "rsi_ob":   rsi_ob,
            "st_buy":   st_buy,
            "st_sell":  st_sell,
        }

        alertas = detectar_alertas(df_daily)

        last  = float(df_candles["Close"].iloc[-1])
        first = float(df_candles["Close"].iloc[0])
        rsi_c = float(df_daily["RSI"].dropna().iloc[-1]) if not df_daily["RSI"].dropna().empty else 50

        return {
            "chart":       chart_data,
            "last_price":  last,
            "change":      last - first,
            "change_pct":  (last - first) / first * 100,
            "rsi_current": rsi_c,
            "alertas":     alertas,
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/api/watch")
async def watch_favorites(tickers: str = ""):
    all_alertas = []
    for t in tickers.split(","):
        t = t.strip()
        if not t:
            continue
        try:
            df = get_daily_df(t)
            if df is not None:
                all_alertas.extend(detectar_alertas(df, ticker=t))
        except Exception:
            pass
    return {"alertas": all_alertas}


@app.get("/api/sparkline/{ticker}")
async def sparkline(ticker: str):
    try:
        df = yf.download(ticker.upper(), period="1mo", interval="1d", progress=False)
        if df.empty:
            return {"closes": [], "pct": 0}
        df = clean_df(df)
        closes = df["Close"].dropna().tolist()
        pct = (closes[-1] - closes[0]) / closes[0] * 100 if len(closes) > 1 else 0
        return {"closes": [float(c) for c in closes], "pct": round(pct, 2)}
    except Exception:
        return {"closes": [], "pct": 0}


if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
