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
# INDICADORES (sobre datos diarios 2y)
# ─────────────────────────────────────────────

def calcular_smas(df):
    df["SMA20"]  = df["Close"].rolling(20).mean()
    df["SMA50"]  = df["Close"].rolling(50).mean()
    df["SMA100"] = df["Close"].rolling(100).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    return df


def calcular_rsi(df):
    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))
    return df


def calcular_supertrend(df, atr_period=7, factor=3.0):
    df["TR"] = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = df["TR"].rolling(atr_period).mean()
    hl2 = (df["High"] + df["Low"]) / 2
    df["UpperBand"] = hl2 + factor * df["ATR"]
    df["LowerBand"] = hl2 - factor * df["ATR"]
    df["Supertrend"] = np.nan
    df["Direction"]  = 0
    for i in range(1, len(df)):
        ps  = df.iloc[i-1]["Supertrend"]
        pd_ = df.iloc[i-1]["Direction"]
        cl  = float(df.iloc[i]["LowerBand"])
        cu  = float(df.iloc[i]["UpperBand"])
        cc  = float(df.iloc[i]["Close"])
        if np.isnan(ps):
            df.iloc[i, df.columns.get_loc("Supertrend")] = cl
            df.iloc[i, df.columns.get_loc("Direction")]  = 1
            continue
        st = max(cl, ps) if pd_ == 1 else min(cu, ps)
        d  = 1 if cc > st else -1
        df.iloc[i, df.columns.get_loc("Supertrend")] = st
        df.iloc[i, df.columns.get_loc("Direction")]  = d
    return df


# ─────────────────────────────────────────────
# ALERTAS: precio vs SMAs (cruce y contacto)
# ─────────────────────────────────────────────

def detectar_alertas_smas(df_daily, ticker=""):
    """Detecta cruces precio/SMA y cruces entre SMAs en los últimos 2 días."""
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
                "msg": f"⚠️ {prefix}Precio tocando {nombre} — ${precio_now:,.2f} / MA ${sma_now:,.2f}"})

    # Cruces entre medias
    s100_n, s100_p = smas["SMA100"]
    s200_n, s200_p = smas["SMA200"]
    if pd.notna(s100_n) and pd.notna(s200_n):
        if s100_p < s200_p and s100_n >= s200_n:
            alertas.append({"nivel": "bullish",
                "msg": f"🟢 {prefix}Golden Cross — SMA100 cruza sobre SMA200"})
        elif s100_p > s200_p and s100_n <= s200_n:
            alertas.append({"nivel": "bearish",
                "msg": f"🔴 {prefix}Death Cross — SMA100 cruza bajo SMA200"})

    s20_n, s20_p = smas["SMA20"]
    s50_n, s50_p = smas["SMA50"]
    if pd.notna(s20_n) and pd.notna(s50_n):
        if s20_p < s50_p and s20_n >= s50_n:
            alertas.append({"nivel": "bullish",
                "msg": f"🟡 {prefix}Cruce alcista SMA20 sobre SMA50"})
        elif s20_p > s50_p and s20_n <= s50_n:
            alertas.append({"nivel": "bearish",
                "msg": f"🟠 {prefix}Cruce bajista SMA20 bajo SMA50"})

    return alertas


# ─────────────────────────────────────────────
# DESCARGA DIARIA BASE (con caché simple)
# ─────────────────────────────────────────────

def get_daily(ticker):
    df = yf.download(ticker.upper(), period="2y", interval="1d", progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = calcular_smas(df)
    df = calcular_rsi(df)
    df = calcular_supertrend(df)
    return df


# ─────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/chart/{ticker}")
async def get_chart(ticker: str, period: str = "1mo"):
    try:
        # ── 1. Siempre descargamos 2 años diarios para SMAs correctas ──
        df_daily = get_daily(ticker)
        if df_daily is None:
            return {"error": f"Símbolo no encontrado: {ticker}"}

        # SMAs actuales (último valor conocido) — para usar como horizontales en intraday
        last_smas = {
            "SMA20":  float(df_daily["SMA20"].dropna().iloc[-1])  if not df_daily["SMA20"].dropna().empty  else None,
            "SMA50":  float(df_daily["SMA50"].dropna().iloc[-1])  if not df_daily["SMA50"].dropna().empty  else None,
            "SMA100": float(df_daily["SMA100"].dropna().iloc[-1]) if not df_daily["SMA100"].dropna().empty else None,
            "SMA200": float(df_daily["SMA200"].dropna().iloc[-1]) if not df_daily["SMA200"].dropna().empty else None,
        }

        # ── 2. Elegir fuente de datos según periodo ──
        intraday_mode = period in ("1d", "5d")

        if intraday_mode:
            interval = "5m" if period == "1d" else "15m"
            df_display = yf.download(ticker.upper(), period=period, interval=interval, progress=False)
            if df_display.empty:
                return {"error": f"Sin datos intraday para {ticker}"}
            if isinstance(df_display.columns, pd.MultiIndex):
                df_display.columns = df_display.columns.get_level_values(0)
            # RSI sobre intraday (limitado pero útil)
            delta = df_display["Close"].diff()
            gain  = delta.where(delta > 0, 0).rolling(14).mean()
            loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df_display["RSI"] = 100 - (100 / (1 + gain / loss))
        else:
            # Filtrar del daily según periodo
            from datetime import timedelta
            dias = {"1mo": 30, "6mo": 180, "1y": 365, "2y": 730}.get(period, 30)
            cutoff = df_daily.index[-1] - timedelta(days=dias)
            df_display = df_daily[df_daily.index >= cutoff].copy()

        # ── 3. Construir datasets ──
        def safe(v):
            return float(v) if pd.notna(v) else None

        labels     = df_display.index.strftime('%Y-%m-%d %H:%M').tolist()
        close_vals = df_display["Close"].values
        n_pts      = len(close_vals)
        rsi_vals   = df_display["RSI"].values if "RSI" in df_display.columns else [50]*n_pts

        # Supertrend solo en modo diario
        if not intraday_mode and "Supertrend" in df_display.columns:
            st_buy  = [safe(v) if df_display["Direction"].iloc[i] == 1  else None for i, v in enumerate(df_display["Supertrend"])]
            st_sell = [safe(v) if df_display["Direction"].iloc[i] == -1 else None for i, v in enumerate(df_display["Supertrend"])]
        else:
            st_buy  = [None] * n_pts
            st_sell = [None] * n_pts

        rsi_os = [safe(close_vals[i]) if pd.notna(rsi_vals[i]) and rsi_vals[i] < 30 else None for i in range(n_pts)]
        rsi_ob = [safe(close_vals[i]) if pd.notna(rsi_vals[i]) and rsi_vals[i] > 70 else None for i in range(n_pts)]

        # En intraday las SMAs son líneas horizontales al último valor diario conocido
        if intraday_mode:
            sma20_data  = [last_smas["SMA20"]]  * n_pts if last_smas["SMA20"]  else [None] * n_pts
            sma50_data  = [last_smas["SMA50"]]  * n_pts if last_smas["SMA50"]  else [None] * n_pts
            sma100_data = [last_smas["SMA100"]] * n_pts if last_smas["SMA100"] else [None] * n_pts
            sma200_data = [last_smas["SMA200"]] * n_pts if last_smas["SMA200"] else [None] * n_pts
        else:
            sma20_data  = [safe(v) for v in df_display["SMA20"]]
            sma50_data  = [safe(v) for v in df_display["SMA50"]]
            sma100_data = [safe(v) for v in df_display["SMA100"]]
            sma200_data = [safe(v) for v in df_display["SMA200"]]

        chart_data = {
            "labels": labels,
            "datasets": [
                {"label": "Precio", "data": [safe(v) for v in close_vals],
                 "borderColor": "#ffffff", "backgroundColor": "rgba(255,255,255,0.07)",
                 "borderWidth": 2, "fill": "origin", "tension": 0.3,
                 "pointRadius": 0, "yAxisID": "y"},

                {"label": "SMA 20", "data": sma20_data,
                 "borderColor": "#00ffff", "backgroundColor": "transparent",
                 "borderWidth": 1, "borderDash": [4,3] if intraday_mode else [],
                 "fill": False, "tension": 0.1, "pointRadius": 0, "yAxisID": "y"},

                {"label": "SMA 50", "data": sma50_data,
                 "borderColor": "#ffff00", "backgroundColor": "transparent",
                 "borderWidth": 1.5, "borderDash": [4,3] if intraday_mode else [],
                 "fill": False, "tension": 0.1, "pointRadius": 0, "yAxisID": "y"},

                {"label": "SMA 100", "data": sma100_data,
                 "borderColor": "#fff176", "backgroundColor": "transparent",
                 "borderWidth": 2, "borderDash": [4,3] if intraday_mode else [],
                 "fill": False, "tension": 0.1, "pointRadius": 0, "yAxisID": "y"},

                {"label": "SMA 200", "data": sma200_data,
                 "borderColor": "#ce93d8", "backgroundColor": "transparent",
                 "borderWidth": 3, "borderDash": [4,3] if intraday_mode else [],
                 "fill": False, "tension": 0.1, "pointRadius": 0, "yAxisID": "y"},

                {"label": "ST ▲", "data": st_buy,
                 "borderColor": "transparent", "backgroundColor": "#00ff00",
                 "borderWidth": 0, "fill": False, "showLine": False,
                 "pointRadius": 2, "pointHoverRadius": 5, "yAxisID": "y"},

                {"label": "ST ▼", "data": st_sell,
                 "borderColor": "transparent", "backgroundColor": "#ff3333",
                 "borderWidth": 0, "fill": False, "showLine": False,
                 "pointRadius": 2, "pointHoverRadius": 5, "yAxisID": "y"},

                {"label": "RSI <30", "data": rsi_os,
                 "borderColor": "transparent", "backgroundColor": "#ff007f",
                 "borderWidth": 0, "fill": False, "showLine": False,
                 "pointRadius": 6, "pointHoverRadius": 9, "yAxisID": "y"},

                {"label": "RSI >70", "data": rsi_ob,
                 "borderColor": "transparent", "backgroundColor": "#ff007f",
                 "borderWidth": 0, "fill": False, "showLine": False,
                 "pointRadius": 6, "pointHoverRadius": 9, "yAxisID": "y"},
            ]
        }

        # Alertas sobre datos diarios completos
        alertas = detectar_alertas_smas(df_daily)

        last  = float(df_display["Close"].iloc[-1])
        first = float(df_display["Close"].iloc[0])
        rsi_c = float(df_display["RSI"].dropna().iloc[-1]) if "RSI" in df_display.columns and not df_display["RSI"].dropna().empty else 50

        return {
            "chart":       chart_data,
            "last_price":  last,
            "change":      last - first,
            "change_pct":  (last - first) / first * 100,
            "rsi_current": rsi_c,
            "alertas":     alertas,
            "intraday":    intraday_mode,
            "last_smas":   last_smas,
        }

    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────
# ENDPOINT DE VIGILANCIA DE FAVORITOS
# ─────────────────────────────────────────────

@app.get("/api/watch")
async def watch_favorites(tickers: str = ""):
    """Comprueba alertas SMA para una lista de tickers separados por coma."""
    all_alertas = []
    if not tickers:
        return {"alertas": []}
    for t in tickers.split(","):
        t = t.strip()
        if not t:
            continue
        try:
            df = get_daily(t)
            if df is not None:
                alertas = detectar_alertas_smas(df, ticker=t)
                all_alertas.extend(alertas)
        except Exception:
            pass
    return {"alertas": all_alertas}


@app.get("/api/sparkline/{ticker}")
async def sparkline(ticker: str):
    try:
        df = yf.download(ticker.upper(), period="5d", interval="1h", progress=False)
        if df.empty:
            return {"closes": [], "pct": 0}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        closes = df["Close"].dropna().tolist()
        pct = (closes[-1] - closes[0]) / closes[0] * 100 if len(closes) > 1 else 0
        return {"closes": [float(c) for c in closes], "pct": round(pct, 2)}
    except Exception:
        return {"closes": [], "pct": 0}


if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
