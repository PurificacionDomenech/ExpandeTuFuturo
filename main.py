import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.utils
import json
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional

app = FastAPI(title="Expande Tu Futuro Web")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Lógica de Indicadores ---

def calcular_indicadores(data):
    # Medias Móviles
    data["SMA20"] = data["Close"].rolling(window=20).mean()
    data["SMA50"] = data["Close"].rolling(window=50).mean()
    data["SMA100"] = data["Close"].rolling(window=100).mean()
    data["SMA200"] = data["Close"].rolling(window=200).mean()

    # RSI
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # Supertrend (7, 3.0)
    atr_period = 7
    factor = 3.0

    data["H-L"] = data["High"] - data["Low"]
    data["H-PC"] = abs(data["High"] - data["Close"].shift(1))
    data["L-PC"] = abs(data["Low"] - data["Close"].shift(1))
    data["TR"] = data[["H-L", "H-PC", "L-PC"]].max(axis=1)
    data["ATR"] = data["TR"].rolling(window=atr_period).mean()

    hl2 = (data["High"] + data["Low"]) / 2
    data["UpperBand"] = hl2 + factor * data["ATR"]
    data["LowerBand"] = hl2 - factor * data["ATR"]

    data["Supertrend"] = np.nan
    data["Direction"] = 0

    for i in range(1, len(data)):
        prev_st = data.iloc[i-1]["Supertrend"]
        prev_dir = data.iloc[i-1]["Direction"]

        curr_lower = float(data.iloc[i]["LowerBand"])
        curr_upper = float(data.iloc[i]["UpperBand"])
        curr_close = float(data.iloc[i]["Close"])

        if np.isnan(prev_st):
            data.iloc[i, data.columns.get_loc("Supertrend")] = curr_lower
            data.iloc[i, data.columns.get_loc("Direction")] = 1
            continue

        if prev_dir == 1:
            curr_st = max(curr_lower, prev_st)
            curr_dir = 1 if curr_close > curr_st else -1
        else:
            curr_st = min(curr_upper, prev_st)
            curr_dir = 1 if curr_close > curr_st else -1

        data.iloc[i, data.columns.get_loc("Supertrend")] = curr_st
        data.iloc[i, data.columns.get_loc("Direction")] = curr_dir

    return data

# --- Rutas ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/chart/{ticker}")
async def get_chart(ticker: str, period: str = "1mo"):
    interval = "1d"
    if period == "1d": interval = "5m"
    elif period == "5d": interval = "30m"

    try:
        df = yf.download(ticker, period=period, interval=interval)
        if df.empty:
            return {"error": "No se encontraron datos"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = calcular_indicadores(df)

        labels = df.index.strftime('%Y-%m-%d %H:%M').tolist()

        # --- Supertrend: puntos en vez de línea ---
        st_buy_y = [float(v) if pd.notna(v) and df["Direction"].iloc[i] == 1 else None
                    for i, v in enumerate(df["Supertrend"].values)]
        st_sell_y = [float(v) if pd.notna(v) and df["Direction"].iloc[i] == -1 else None
                     for i, v in enumerate(df["Supertrend"].values)]

        # --- RSI: señales sobre precio en zonas extremas ---
        rsi_vals = df["RSI"].values
        close_vals = df["Close"].values

        # Puntos verdes = sobreventa (RSI < 30), puntos rojos = sobrecompra (RSI > 70)
        rsi_oversold_y  = [float(close_vals[i]) if rsi_vals[i] < 30 else None for i in range(len(rsi_vals))]
        rsi_overbought_y = [float(close_vals[i]) if rsi_vals[i] > 70 else None for i in range(len(rsi_vals))]

        chart_data = {
            "labels": labels,
            "datasets": [
                {
                    "label": "Precio",
                    "data": [float(v) for v in close_vals],
                    "borderColor": "#ff007f",
                    "backgroundColor": "rgba(255, 0, 127, 0.05)",
                    "borderWidth": 2,
                    "fill": True,
                    "tension": 0.1,
                    "pointRadius": 0,
                    "yAxisID": "y"
                },
                {
                    "label": "SMA 20",
                    "data": [float(v) if pd.notna(v) else None for v in df["SMA20"].values],
                    "borderColor": "#00ffff",
                    "backgroundColor": "transparent",
                    "borderWidth": 2,
                    "fill": False,
                    "tension": 0.1,
                    "pointRadius": 0,
                    "yAxisID": "y"
                },
                {
                    "label": "SMA 50",
                    "data": [float(v) if pd.notna(v) else None for v in df["SMA50"].values],
                    "borderColor": "#ffff00",
                    "backgroundColor": "transparent",
                    "borderWidth": 2,
                    "fill": False,
                    "tension": 0.1,
                    "pointRadius": 0,
                    "yAxisID": "y"
                },
                {
                    "label": "SMA 100",
                    "data": [float(v) if pd.notna(v) else None for v in df["SMA100"].values],
                    "borderColor": "#fff176",
                    "backgroundColor": "transparent",
                    "borderWidth": 1.8,
                    "fill": False,
                    "tension": 0.1,
                    "pointRadius": 0,
                    "yAxisID": "y"
                },
                {
                    "label": "SMA 200",
                    "data": [float(v) if pd.notna(v) else None for v in df["SMA200"].values],
                    "borderColor": "#ce93d8",
                    "backgroundColor": "transparent",
                    "borderWidth": 2,
                    "fill": False,
                    "tension": 0.1,
                    "pointRadius": 0,
                    "yAxisID": "y"
                },
                {
                    "label": "Supertrend Buy",
                    "data": st_buy_y,
                    "borderColor": "transparent",
                    "backgroundColor": "#00ff00",
                    "borderWidth": 0,
                    "fill": False,
                    "tension": 0,
                    "pointRadius": 4,
                    "pointHoverRadius": 6,
                    "showLine": False,
                    "yAxisID": "y"
                },
                {
                    "label": "Supertrend Sell",
                    "data": st_sell_y,
                    "borderColor": "transparent",
                    "backgroundColor": "#ff0000",
                    "borderWidth": 0,
                    "fill": False,
                    "tension": 0,
                    "pointRadius": 4,
                    "pointHoverRadius": 6,
                    "showLine": False,
                    "yAxisID": "y"
                },
                {
                    "label": "RSI Sobreventa",
                    "data": rsi_oversold_y,
                    "borderColor": "transparent",
                    "backgroundColor": "#00e676",
                    "borderWidth": 0,
                    "fill": False,
                    "tension": 0,
                    "pointRadius": 7,
                    "pointHoverRadius": 9,
                    "pointStyle": "circle",
                    "showLine": False,
                    "yAxisID": "y"
                },
                {
                    "label": "RSI Sobrecompra",
                    "data": rsi_overbought_y,
                    "borderColor": "transparent",
                    "backgroundColor": "#ff1744",
                    "borderWidth": 0,
                    "fill": False,
                    "tension": 0,
                    "pointRadius": 7,
                    "pointHoverRadius": 9,
                    "pointStyle": "circle",
                    "showLine": False,
                    "yAxisID": "y"
                }
            ]
        }

        last_close = float(df["Close"].iloc[-1])
        first_close = float(df["Close"].iloc[0])

        # RSI actual para el indicador lateral
        rsi_current = float(df["RSI"].dropna().iloc[-1])

        return {
            "chart": chart_data,
            "last_price": last_close,
            "change": last_close - first_close,
            "change_pct": ((last_close - first_close) / first_close) * 100,
            "rsi_current": rsi_current
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
