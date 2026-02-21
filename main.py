import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Expande Tu Futuro Web")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ─────────────────────────────────────────────
# INDICADORES
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

    atr_period, factor = 7, 3.0
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
# ALERTAS AUTOMÁTICAS
# ─────────────────────────────────────────────

def detectar_alertas(df, gemas: list):
    alertas = []
    n = len(df) - 1
    if n < 2:
        return alertas

    precio_actual = float(df["Close"].iloc[n])
    precio_prev   = float(df["Close"].iloc[n - 1])

    smas = {
        "SMA20":  (df["SMA20"].iloc[n],  df["SMA20"].iloc[n-1]),
        "SMA50":  (df["SMA50"].iloc[n],  df["SMA50"].iloc[n-1]),
        "SMA100": (df["SMA100"].iloc[n], df["SMA100"].iloc[n-1]),
        "SMA200": (df["SMA200"].iloc[n], df["SMA200"].iloc[n-1]),
    }

    # ── Cruce precio / media móvil ──────────────────────────
    for nombre, (sma_now, sma_prev) in smas.items():
        if not (pd.notna(sma_now) and pd.notna(sma_prev)):
            continue
        # precio cruza hacia arriba
        if precio_prev < sma_prev and precio_actual >= sma_now:
            alertas.append({
                "nivel": "bullish",
                "msg":   f"📈 Precio cruza {nombre} hacia arriba — ${precio_actual:,.2f}"
            })
        # precio cruza hacia abajo
        elif precio_prev > sma_prev and precio_actual <= sma_now:
            alertas.append({
                "nivel": "bearish",
                "msg":   f"📉 Precio cruza {nombre} hacia abajo — ${precio_actual:,.2f}"
            })
        # precio muy cerca (≤0.5%)
        elif abs(precio_actual - sma_now) / sma_now * 100 <= 0.5:
            alertas.append({
                "nivel": "info",
                "msg":   f"⚠️ Precio tocando {nombre} — ${precio_actual:,.2f} / MA ${sma_now:,.2f}"
            })

    # ── Cruces entre medias (100/200 y 20/50) ─────────────
    s100_n, s100_p = smas["SMA100"]
    s200_n, s200_p = smas["SMA200"]
    if pd.notna(s100_n) and pd.notna(s200_n):
        if s100_p < s200_p and s100_n >= s200_n:
            alertas.append({"nivel": "bullish", "msg": "🟢 Golden Cross — SMA100 cruza sobre SMA200"})
        elif s100_p > s200_p and s100_n <= s200_n:
            alertas.append({"nivel": "bearish", "msg": "🔴 Death Cross — SMA100 cruza bajo SMA200"})

    s20_n, s20_p = smas["SMA20"]
    s50_n, s50_p = smas["SMA50"]
    if pd.notna(s20_n) and pd.notna(s50_n):
        if s20_p < s50_p and s20_n >= s50_n:
            alertas.append({"nivel": "bullish", "msg": "🟡 Cruce alcista — SMA20 sobre SMA50"})
        elif s20_p > s50_p and s20_n <= s50_n:
            alertas.append({"nivel": "bearish", "msg": "🟠 Cruce bajista — SMA20 bajo SMA50"})

    # ── Niveles de precio personalizados (gemas) ──────────
    for g in gemas:
        pct = abs(precio_actual - g) / g * 100
        if pct <= 1.0:
            alertas.append({
                "nivel": "info",
                "msg":   f"💎 Precio cerca de nivel ${g:,.2f} — {pct:.2f}% de distancia"
            })

    return alertas


# ─────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/chart/{ticker}")
async def get_chart(ticker: str, period: str = "1mo", niveles: str = ""):
    try:
        # ── Siempre descargar 2 años diarios para que SMA100/200 estén disponibles ──
        df_full = yf.download(ticker.upper(), period="2y", interval="1d", progress=False)
        if df_full.empty:
            return {"error": f"Símbolo no encontrado: {ticker}"}

        if isinstance(df_full.columns, pd.MultiIndex):
            df_full.columns = df_full.columns.get_level_values(0)

        # Calcular indicadores sobre el histórico completo
        df_full = calcular_indicadores(df_full)

        # ── Filtrar al periodo de visualización ──────────────────────────────────
        periodo_dias = {
            "1d": 1, "5d": 5, "1mo": 30, "6mo": 180, "1y": 365, "2y": 730
        }
        dias = periodo_dias.get(period, 30)
        cutoff = df_full.index[-1] - timedelta(days=dias)
        df = df_full[df_full.index >= cutoff].copy()

        if df.empty:
            df = df_full.tail(30).copy()

        def safe(v):
            return float(v) if pd.notna(v) else None

        labels     = df.index.strftime('%Y-%m-%d').tolist()
        close_vals = df["Close"].values
        rsi_vals   = df["RSI"].values

        st_buy  = [safe(v) if df["Direction"].iloc[i] == 1  else None for i, v in enumerate(df["Supertrend"])]
        st_sell = [safe(v) if df["Direction"].iloc[i] == -1 else None for i, v in enumerate(df["Supertrend"])]

        # RSI señales en precio — color fucsia
        rsi_os = [safe(close_vals[i]) if pd.notna(rsi_vals[i]) and rsi_vals[i] < 30 else None for i in range(len(rsi_vals))]
        rsi_ob = [safe(close_vals[i]) if pd.notna(rsi_vals[i]) and rsi_vals[i] > 70 else None for i in range(len(rsi_vals))]

        chart_data = {
            "labels": labels,
            "datasets": [
                # Precio — blanco con sombra
                {"label": "Precio", "data": [safe(v) for v in close_vals],
                 "borderColor": "#ffffff", "backgroundColor": "rgba(255,255,255,0.07)",
                 "borderWidth": 2, "fill": "origin", "tension": 0.3,
                 "pointRadius": 0, "yAxisID": "y"},

                # SMAs escalonadas en grosor
                {"label": "SMA 20", "data": [safe(v) for v in df["SMA20"]],
                 "borderColor": "#00ffff", "backgroundColor": "transparent",
                 "borderWidth": 1, "fill": False, "tension": 0.1,
                 "pointRadius": 0, "yAxisID": "y"},

                {"label": "SMA 50", "data": [safe(v) for v in df["SMA50"]],
                 "borderColor": "#ffff00", "backgroundColor": "transparent",
                 "borderWidth": 1.5, "fill": False, "tension": 0.1,
                 "pointRadius": 0, "yAxisID": "y"},

                {"label": "SMA 100", "data": [safe(v) for v in df["SMA100"]],
                 "borderColor": "#fff176", "backgroundColor": "transparent",
                 "borderWidth": 2, "fill": False, "tension": 0.1,
                 "pointRadius": 0, "yAxisID": "y"},

                {"label": "SMA 200", "data": [safe(v) for v in df["SMA200"]],
                 "borderColor": "#ce93d8", "backgroundColor": "transparent",
                 "borderWidth": 3, "fill": False, "tension": 0.1,
                 "pointRadius": 0, "yAxisID": "y"},

                # Supertrend — puntos muy pequeños
                {"label": "ST ▲", "data": st_buy,
                 "borderColor": "transparent", "backgroundColor": "#00ff00",
                 "borderWidth": 0, "fill": False, "showLine": False,
                 "pointRadius": 2, "pointHoverRadius": 5, "yAxisID": "y"},

                {"label": "ST ▼", "data": st_sell,
                 "borderColor": "transparent", "backgroundColor": "#ff3333",
                 "borderWidth": 0, "fill": False, "showLine": False,
                 "pointRadius": 2, "pointHoverRadius": 5, "yAxisID": "y"},

                # RSI señales — fucsia
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

        # Gemas (niveles manuales)
        gemas = []
        if niveles:
            for n in niveles.split(","):
                try:
                    gemas.append(float(n.strip()))
                except Exception:
                    pass

        # Alertas sobre el dataset completo (último día disponible)
        alertas = detectar_alertas(df_full, gemas)

        last  = float(df["Close"].iloc[-1])
        first = float(df["Close"].iloc[0])
        rsi_c = float(df["RSI"].dropna().iloc[-1]) if not df["RSI"].dropna().empty else 50

        return {
            "chart":       chart_data,
            "last_price":  last,
            "change":      last - first,
            "change_pct":  (last - first) / first * 100,
            "rsi_current": rsi_c,
            "alertas":     alertas,
            "gemas":       gemas,
        }

    except Exception as e:
        return {"error": str(e)}


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
