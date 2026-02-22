import yfinance as yf
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

INTERVAL_MAP = {
    "1d":  ("1d",  "max"),
    "1wk": ("1wk", "max"),
    "1mo": ("1mo", "max"),
}


def clean_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def calcular_indicadores(df):
    df["SMA20"]  = df["Close"].rolling(20).mean()
    df["SMA50"]  = df["Close"].rolling(50).mean()
    df["SMA100"] = df["Close"].rolling(100).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

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
        ps  = df.iloc[i-1]["ST"]
        pd_ = df.iloc[i-1]["Dir"]
        cl  = float(df.iloc[i]["LB"])
        cu  = float(df.iloc[i]["UB"])
        cc  = float(df.iloc[i]["Close"])
        if np.isnan(ps):
            df.iloc[i, df.columns.get_loc("ST")]  = cl
            df.iloc[i, df.columns.get_loc("Dir")] = 1
            continue
        st = max(cl, ps) if pd_ == 1 else min(cu, ps)
        df.iloc[i, df.columns.get_loc("ST")]  = st
        df.iloc[i, df.columns.get_loc("Dir")] = 1 if cc > st else -1

    return df


def detectar_alertas(df, ticker=""):
    alertas = []
    n = len(df) - 1
    if n < 2:
        return alertas

    precio_now  = float(df["Close"].iloc[n])
    precio_prev = float(df["Close"].iloc[n - 1])
    prefix = f"[{ticker}] " if ticker else ""

    smas = {
        "SMA20":  (df["SMA20"].iloc[n],  df["SMA20"].iloc[n-1]),
        "SMA50":  (df["SMA50"].iloc[n],  df["SMA50"].iloc[n-1]),
        "SMA100": (df["SMA100"].iloc[n], df["SMA100"].iloc[n-1]),
        "SMA200": (df["SMA200"].iloc[n], df["SMA200"].iloc[n-1]),
    }

    for nombre, (sma_now, sma_prev) in smas.items():
        if not (pd.notna(sma_now) and pd.notna(sma_prev)):
            continue
        if precio_prev < sma_prev and precio_now >= sma_now:
            alertas.append({"nivel": "bullish",
                "msg": f"Precio cruza {nombre} al alza [{prefix}] ${precio_now:,.2f}"})
        elif precio_prev > sma_prev and precio_now <= sma_now:
            alertas.append({"nivel": "bearish",
                "msg": f"Precio cruza {nombre} a la baja [{prefix}] ${precio_now:,.2f}"})
        elif abs(precio_now - sma_now) / sma_now * 100 <= 0.4:
            alertas.append({"nivel": "info",
                "msg": f"Precio tocando {nombre} [{prefix}] ${precio_now:,.2f}"})

    s100_n, s100_p = smas["SMA100"]
    s200_n, s200_p = smas["SMA200"]
    if pd.notna(s100_n) and pd.notna(s200_n):
        if s100_p < s200_p and s100_n >= s200_n:
            alertas.append({"nivel": "bullish",
                "msg": f"Golden Cross SMA100/SMA200 [{prefix}]"})
        elif s100_p > s200_p and s100_n <= s200_n:
            alertas.append({"nivel": "bearish",
                "msg": f"Death Cross SMA100/SMA200 [{prefix}]"})

    s20_n, s20_p = smas["SMA20"]
    s50_n, s50_p = smas["SMA50"]
    if pd.notna(s20_n) and pd.notna(s50_n):
        if s20_p < s50_p and s20_n >= s50_n:
            alertas.append({"nivel": "bullish",
                "msg": f"SMA20 cruza sobre SMA50 [{prefix}]"})
        elif s20_p > s50_p and s20_n <= s50_n:
            alertas.append({"nivel": "bearish",
                "msg": f"SMA20 cruza bajo SMA50 [{prefix}]"})

    return alertas


def get_daily_df(ticker):
    df = yf.download(ticker.upper(), period="max", interval="1d", progress=False)
    if df.empty:
        return None
    df = clean_df(df)
    df = calcular_indicadores(df)
    return df


def ts_ms(idx):
    return [int(t.timestamp() * 1000) for t in idx]


def safe(v):
    return float(v) if pd.notna(v) else None


@app.get("/")
async def index():
    return FileResponse("templates/index.html")


@app.get("/api/chart/{ticker}")
async def get_chart(ticker: str, interval: str = "1d"):
    try:
        yf_interval, yf_period = INTERVAL_MAP.get(interval, ("1d", "max"))

        df_candles = yf.download(
            ticker.upper(), period=yf_period,
            interval=yf_interval, progress=False
        )
        if df_candles.empty:
            return {"error": f"Simbolo no encontrado: {ticker}"}
        df_candles = clean_df(df_candles)

        df_daily = df_candles.copy()
        df_daily = calcular_indicadores(df_daily)

        timestamps = ts_ms(df_candles.index)
        candles = [
            {
                "x": timestamps[i],
                "o": safe(df_candles["Open"].iloc[i]),
                "h": safe(df_candles["High"].iloc[i]),
                "l": safe(df_candles["Low"].iloc[i]),
                "c": safe(df_candles["Close"].iloc[i]),
            }
            for i in range(len(df_candles))
        ]

        def sma_series(col):
            out = []
            for i in range(len(df_daily)):
                v = df_daily[col].iloc[i]
                if pd.notna(v):
                    out.append({
                        "x": int(df_daily.index[i].timestamp() * 1000),
                        "y": float(v),
                    })
            return out

        rsi_vals   = df_daily["RSI"].values
        close_vals = df_daily["Close"].values
        daily_ts   = ts_ms(df_daily.index)

        rsi_os = [
            {"x": daily_ts[i], "y
