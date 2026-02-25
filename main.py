import yfinance as yf
import pandas as pd
import numpy as np
import asyncio
import os
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from notifications import load_prefs, save_prefs, dispatch_notifications, format_alerts_text, get_db

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"GLOBAL ERROR: {exc}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"ok": False, "error": "Internal Server Error", "details": str(exc)}
    )

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    if request.url.path.startswith("/api/"):
        return JSONResponse(status_code=404, content={"ok": False, "error": "Endpoint not found"})
    return FileResponse("templates/splash.html")

INTERVAL_MAP = {
    "1d":  ("1d",  "max"),
    "1wk": ("1wk", "max"),
    "1mo": ("1mo", "max"),
}

def clean_df(df):
    try:
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return pd.DataFrame()

def calcular_indicadores(df):
    if df is None or df.empty or len(df) < 20:
        return df
    try:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if 'Close' not in df.columns:
            return df
        df["SMA20"]  = df["Close"].rolling(20).mean()
        df["SMA50"]  = df["Close"].rolling(50).mean()
        df["SMA100"] = df["Close"].rolling(100).mean()
        df["SMA200"] = df["Close"].rolling(200).mean()
        delta = df["Close"].diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["RSI"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
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
    except Exception as e:
        print(f"Error in indicators: {e}")
    return df

def detectar_alertas(df, ticker=""):
    alertas = []
    if len(df) < 2:
        return alertas
    try:
        n = len(df) - 1
        p_now  = float(df["Close"].iloc[n])
        p_prev = float(df["Close"].iloc[n - 1])
        prefix = f"[{ticker.upper()}] " if ticker else ""
        smas = {"SMA20": "SMA20", "SMA50": "SMA50", "SMA100": "SMA100", "SMA200": "SMA200"}
        for k, v in smas.items():
            if v in df.columns:
                now = df[v].iloc[n]
                prev = df[v].iloc[n-1]
                if pd.notna(now) and pd.notna(prev):
                    if p_prev < prev and p_now >= now:
                        alertas.append({"nivel": "bullish", "msg": f"{prefix}Precio cruza {k} al alza $" + str(round(p_now, 2))})
                    elif p_prev > prev and p_now <= now:
                        alertas.append({"nivel": "bearish", "msg": f"{prefix}Precio cruza {k} a la baja $" + str(round(p_now, 2))})
    except:
        pass
    return alertas

@app.get("/")
async def splash():
    return FileResponse("templates/splash.html")

@app.get("/app")
async def index():
    return FileResponse("templates/index.html")

@app.get("/api/health")
async def health():
    return {"status": "ok"}

@app.get("/api/chart/{ticker}")
async def get_chart(ticker: str, interval: str = "1d"):
    try:
        yf_i, yf_p = INTERVAL_MAP.get(interval, ("1d", "max"))
        df = yf.download(ticker.upper(), period=yf_p, interval=yf_i, progress=False)
        df = clean_df(df)
        if df.empty:
            return {"error": "No data available for " + ticker}
        df = calcular_indicadores(df)
        ts = [int(t.timestamp() * 1000) for t in df.index]
        candles = []
        for i in range(len(df)):
            candles.append({
                "x": ts[i],
                "o": float(df["Open"].iloc[i]),
                "h": float(df["High"].iloc[i]),
                "l": float(df["Low"].iloc[i]),
                "c": float(df["Close"].iloc[i]),
            })
        
        # Preparar indicadores para el frontend
        indicadores = {}
        for col in ["SMA20", "SMA50", "SMA100", "SMA200", "RSI", "ST"]:
            if col in df.columns:
                # Convertir a lista de floats manejando NaNs
                indicadores[col] = [float(x) if pd.notna(x) else None for x in df[col].tolist()]
        
        last_price = float(df["Close"].iloc[-1])
        alertas = detectar_alertas(df, ticker)
        return {
            "chart": {"candles": candles},
            "indicadores": indicadores,
            "last_price": last_price,
            "alertas": alertas
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/sparkline/{ticker}")
async def get_sparkline(ticker: str):
    try:
        df = yf.download(ticker.upper(), period="7d", interval="1h", progress=False)
        df = clean_df(df)
        if df.empty: return {"ticker": ticker.upper(), "points": []}
        points = [float(c) for c in df["Close"].tolist() if pd.notna(c)]
        return {"ticker": ticker.upper(), "points": points}
    except Exception as e:
        return {"ticker": ticker.upper(), "points": [], "error": str(e)}

@app.get("/api/row/{ticker}")
async def get_row(ticker: str):
    try:
        ticker_clean = ticker.upper().strip()
        df = yf.download(ticker_clean, period="1mo", interval="1d", progress=False)
        df = clean_df(df)
        if df.empty:
            return {"ticker": ticker_clean, "error": "No data"}
        df = calcular_indicadores(df)
        try:
            last = float(df["Close"].iloc[-1])
            first = float(df["Close"].iloc[0])
        except:
            return {"ticker": ticker_clean, "price": 0, "change_pct": 0, "error": "Insufficient data"}
        return {
            "ticker": ticker_clean,
            "price": round(last, 2),
            "change_pct": round((last - first) / first * 100, 2) if first != 0 else 0
        }
    except Exception as e:
        return {"ticker": ticker.upper(), "error": str(e)}

@app.get("/api/watch")
async def watch_favorites(tickers: str = ""):
    all_alertas = []
    if not tickers: return {"alertas": []}
    for t in tickers.split(","):
        t = t.strip()
        if not t: continue
        try:
            df = yf.download(t.upper(), period="5d", interval="1d", progress=False)
            df = clean_df(df)
            if not df.empty:
                df = calcular_indicadores(df)
                all_alertas.extend(detectar_alertas(df, ticker=t.upper()))
        except: pass
    return {"alertas": all_alertas}

@app.post("/api/notifications/redeem")
async def redeem_code(request: Request):
    try:
        uid = request.query_params.get("user_id")
        try:
            body = await request.json()
        except:
            body = {}
        if not uid:
            uid = body.get("user_id", "default")
        code = body.get("code", "").upper().strip()
        if not code:
            return {"ok": False, "error": "Código vacío"}
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT is_used FROM access_codes WHERE code = %s", (code,))
                row = cur.fetchone()
                if not row:
                    if code == "VIP333":
                        cur.execute("INSERT INTO access_codes (code, is_used) VALUES ('VIP333', FALSE)")
                        row = (False,)
                    else:
                        return {"ok": False, "error": "Código inválido"}
                if row[0] and code != "VIP333":
                    return {"ok": False, "error": "Código ya utilizado"}
                if code != "VIP333":
                    cur.execute("UPDATE access_codes SET is_used = TRUE WHERE code = %s", (code,))
                cur.execute("INSERT INTO notification_prefs (user_id, is_vip, updated_at) VALUES (%s, TRUE, NOW()) ON CONFLICT (user_id) DO UPDATE SET is_vip = TRUE, updated_at = NOW()", (uid,))
            conn.commit()
        return {"ok": True, "msg": "¡VIP Activado!"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/api/notifications/prefs")
async def get_prefs(user_id: str = "default"):
    return load_prefs(user_id)

@app.post("/api/notifications/prefs")
async def set_prefs(request: Request, user_id: str = "default"):
    try:
        body = await request.json()
        p = load_prefs(user_id)
        p.update(body)
        save_prefs(p, user_id)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/api/notifications/status")
async def status():
    return {
        "telegram": bool(os.environ.get("TELEGRAM_BOT_TOKEN")),
        "email": bool(os.environ.get("SMTP_USER"))
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
