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
    prefix = "[" + ticker + "] " if ticker else ""

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
                "msg": prefix + "Precio cruza " + nombre + " al alza $" + str(round(precio_now, 2))})
        elif precio_prev > sma_prev and precio_now <= sma_now:
            alertas.append({"nivel": "bearish",
                "msg": prefix + "Precio cruza " + nombre + " a la baja $" + str(round(precio_now, 2))})
        elif sma_now > 0 and abs(precio_now - sma_now) / sma_now * 100 <= 0.4:
            alertas.append({"nivel": "info",
                "msg": prefix + "Precio tocando " + nombre + " $" + str(round(precio_now, 2))})

    s100_n, s100_p = smas["SMA100"]
    s200_n, s200_p = smas["SMA200"]
    if pd.notna(s100_n) and pd.notna(s200_n) and pd.notna(s100_p) and pd.notna(s200_p):
        if s100_p < s200_p and s100_n >= s200_n:
            alertas.append({"nivel": "bullish", "msg": prefix + "Golden Cross SMA100/200"})
        elif s100_p > s200_p and s100_n <= s200_n:
            alertas.append({"nivel": "bearish", "msg": prefix + "Death Cross SMA100/200"})

    s20_n, s20_p = smas["SMA20"]
    s50_n, s50_p = smas["SMA50"]
    if pd.notna(s20_n) and pd.notna(s50_n) and pd.notna(s20_p) and pd.notna(s50_p):
        if s20_p < s50_p and s20_n >= s50_n:
            alertas.append({"nivel": "bullish", "msg": prefix + "SMA20 cruza sobre SMA50"})
        elif s20_p > s50_p and s20_n <= s50_n:
            alertas.append({"nivel": "bearish", "msg": prefix + "SMA20 cruza bajo SMA50"})

    return alertas


def safe(v):
    return float(v) if pd.notna(v) else None


def ts_ms(idx):
    return [int(t.timestamp() * 1000) for t in idx]


@app.get("/")
async def index():
    return FileResponse("templates/index.html")


@app.get("/api/chart/{ticker}")
async def get_chart(ticker: str, interval: str = "1d"):
    try:
        yf_interval, yf_period = INTERVAL_MAP.get(interval, ("1d", "max"))
        df = yf.download(ticker.upper(), period=yf_period, interval=yf_interval, progress=False)
        if df.empty:
            return {"error": "Simbolo no encontrado: " + ticker}
        df = clean_df(df)
        df = calcular_indicadores(df)

        timestamps = ts_ms(df.index)
        candles = []
        for i in range(len(df)):
            candles.append({
                "x": timestamps[i],
                "o": safe(df["Open"].iloc[i]),
                "h": safe(df["High"].iloc[i]),
                "l": safe(df["Low"].iloc[i]),
                "c": safe(df["Close"].iloc[i]),
            })

        def sma_series(col):
            out = []
            for i in range(len(df)):
                v = df[col].iloc[i]
                if pd.notna(v):
                    out.append({"x": timestamps[i], "y": float(v)})
            return out

        rsi_vals  = df["RSI"].values
        close_vals = df["Close"].values

        rsi_os = []
        rsi_ob = []
        for i in range(len(df)):
            if pd.notna(rsi_vals[i]):
                if rsi_vals[i] < 30:
                    rsi_os.append({"x": timestamps[i], "y": float(close_vals[i])})
                elif rsi_vals[i] > 70:
                    rsi_ob.append({"x": timestamps[i], "y": float(close_vals[i])})

        st_buy  = []
        st_sell = []
        for i in range(len(df)):
            st_v = df["ST"].iloc[i]
            dr_v = df["Dir"].iloc[i]
            if pd.notna(st_v):
                if dr_v == 1:
                    st_buy.append({"x": timestamps[i], "y": float(st_v)})
                elif dr_v == -1:
                    st_sell.append({"x": timestamps[i], "y": float(st_v)})

        last  = float(df["Close"].iloc[-1])
        first = float(df["Close"].iloc[0])
        rsi_series = df["RSI"].dropna()
        rsi_c = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50

        alertas = detectar_alertas(df)

        return {
            "chart": {
                "candles":  candles,
                "sma20":    sma_series("SMA20"),
                "sma50":    sma_series("SMA50"),
                "sma100":   sma_series("SMA100"),
                "sma200":   sma_series("SMA200"),
                "rsi_os":   rsi_os,
                "rsi_ob":   rsi_ob,
                "st_buy":   st_buy,
                "st_sell":  st_sell,
            },
            "last_price":  last,
            "change":      last - first,
            "change_pct":  (last - first) / first * 100,
            "rsi_current": rsi_c,
            "alertas":     alertas,
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/api/row/{ticker}")
async def get_row(ticker: str):
    try:
        df = yf.download(ticker.upper(), period="1y", interval="1d", progress=False)
        if df.empty:
            return {"error": "not found"}
        df = clean_df(df)
        df = calcular_indicadores(df)

        last  = float(df["Close"].iloc[-1])
        first = float(df["Close"].iloc[0])

        def last_val(col):
            s = df[col].dropna()
            return float(s.iloc[-1]) if not s.empty else None

        rsi_s = df["RSI"].dropna()
        rsi   = float(rsi_s.iloc[-1]) if not rsi_s.empty else None

        dir_s  = df["Dir"].dropna()
        st_dir = int(dir_s.iloc[-1]) if not dir_s.empty else 0

        return {
            "ticker":     ticker.upper(),
            "price":      last,
            "change_pct": round((last - first) / first * 100, 2),
            "rsi":        round(rsi, 1) if rsi is not None else None,
            "sma20":      last_val("SMA20"),
            "sma50":      last_val("SMA50"),
            "sma100":     last_val("SMA100"),
            "sma200":     last_val("SMA200"),
            "st_dir":     st_dir,
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
            df = yf.download(t.upper(), period="1y", interval="1d", progress=False)
            if not df.empty:
                df = clean_df(df)
                df = calcular_indicadores(df)
                all_alertas.extend(detectar_alertas(df, ticker=t.upper()))
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
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
