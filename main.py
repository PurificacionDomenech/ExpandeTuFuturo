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

# Configuración de archivos estáticos y plantillas
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Lógica de Indicadores ---

def calcular_indicadores(data):
    # Medias Móviles
    data["SMA20"] = data["Close"].rolling(window=20).mean()
    data["SMA50"] = data["Close"].rolling(window=50).mean()
    
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
        
        if np.isnan(prev_st):
            data.iloc[i, data.columns.get_loc("Supertrend")] = data.iloc[i]["LowerBand"]
            data.iloc[i, data.columns.get_loc("Direction")] = 1
            continue
            
        if prev_dir == 1:
            curr_st = max(data.iloc[i]["LowerBand"], prev_st)
            curr_dir = 1 if data.iloc[i]["Close"] > curr_st else -1
        else:
            curr_st = min(data.iloc[i]["UpperBand"], prev_st)
            curr_dir = 1 if data.iloc[i]["Close"] > curr_st else -1
            
        data.iloc[i, data.columns.get_loc("Supertrend")] = curr_st
        data.iloc[i, data.columns.get_loc("Direction")] = curr_dir
        
    return data

# --- Rutas ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/chart/{ticker}")
async def get_chart(ticker: str, period: str = "1mo"):
    # period: 1d, 5d, 1mo, 6mo, 1y, 5y, max
    # interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    
    interval = "1d"
    if period == "1d": interval = "5m"
    elif period == "5d": interval = "30m"
    
    try:
        df = yf.download(ticker, period=period, interval=interval)
        if df.empty:
            return {"error": "No se encontraron datos"}
        
        # Limpiar columnas si son multi-index
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        
        df = calcular_indicadores(df)
        
        # Crear Gráfica de Área (Fucsia/Rosa)
        fig = go.Figure()
        
        # Área principal
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"],
            fill='tozeroy',
            mode='lines',
            line=dict(color='#ff007f', width=2),
            fillcolor='rgba(255, 0, 127, 0.2)',
            name="Precio"
        ))
        
        # SMA 20
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA20"],
            mode='lines',
            line=dict(color='rgba(0, 255, 255, 0.6)', width=1.5),
            name="SMA 20"
        ))
        
        # SMA 50
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA50"],
            mode='lines',
            line=dict(color='rgba(255, 255, 0, 0.6)', width=1.5),
            name="SMA 50"
        ))
        
        # Supertrend
        fig.add_trace(go.Scatter(
            x=df.index,
            y=np.where(df["Direction"] == 1, df["Supertrend"], np.nan),
            mode='lines',
            line=dict(color='#00ff00', width=2),
            name="Supertrend Buy"
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=np.where(df["Direction"] == -1, df["Supertrend"], np.nan),
            mode='lines',
            line=dict(color='#ff0000', width=2),
            name="Supertrend Sell"
        ))
        
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # RSI Data para gráfica separada
        rsi_data = {
            "x": df.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            "y": df["RSI"].fillna(50).tolist()
        }
        
        chart_json = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
        
        return {
            "chart": chart_json,
            "rsi": rsi_data,
            "last_price": float(df["Close"].iloc[-1]),
            "change": float(df["Close"].iloc[-1] - df["Close"].iloc[0]),
            "change_pct": float((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100)
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
