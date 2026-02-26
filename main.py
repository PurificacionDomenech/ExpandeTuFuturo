import os
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import yfinance as yf
import pandas as pd
import numpy as np
import asyncio
from notifications import load_prefs, save_prefs, dispatch_notifications, get_db
from indicators import clean_df, calcular_indicadores, detectar_alertas
import cron_scanner

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cron_scanner.main())

INTERVAL_MAP = {"1d": ("1d", "max"), "1wk": ("1wk", "max"), "1mo": ("1mo", "max")}
def ts_ms(idx): return [int(t.timestamp() * 1000) for t in idx]
def safe(v): return float(v) if pd.notna(v) else None

@app.get("/")
async def splash(): return FileResponse("templates/splash.html")

@app.get("/app")
async def index(): return FileResponse("templates/index.html")

@app.get("/api/chart/{ticker}")
async def get_chart(ticker: str, interval: str = "1d"):
    try:
        yf_interval, yf_period = INTERVAL_MAP.get(interval, ("1d", "1y"))
        df = yf.download(ticker.upper(), period=yf_period, interval=yf_interval, progress=False)
        if df.empty:
            df = yf.download(ticker.upper(), period="1y", interval=yf_interval, progress=False)
            if df.empty: return {"error": "Simbolo no encontrado"}
        df = clean_df(df)
        df = calcular_indicadores(df)
        timestamps = ts_ms(df.index)
        candles = [{"x": timestamps[i], "o": safe(df["Open"].iloc[i]), "h": safe(df["High"].iloc[i]), "l": safe(df["Low"].iloc[i]), "c": safe(df["Close"].iloc[i])} for i in range(len(df))]
        
        def sma_series(col): return [{"x": timestamps[i], "y": float(df[col].iloc[i])} for i in range(len(df)) if pd.notna(df[col].iloc[i])]
        
        rsi_vals, close_vals = df["RSI"].values, df["Close"].values
        rsi_os = [{"x": timestamps[i], "y": float(close_vals[i])} for i in range(len(df)) if pd.notna(rsi_vals[i]) and rsi_vals[i] < 30]
        rsi_ob = [{"x": timestamps[i], "y": float(close_vals[i])} for i in range(len(df)) if pd.notna(rsi_vals[i]) and rsi_vals[i] > 70]
        
        st_buy = [{"x": timestamps[i], "y": float(df["ST"].iloc[i])} for i in range(len(df)) if pd.notna(df["ST"].iloc[i]) and df["Dir"].iloc[i] == 1]
        st_sell = [{"x": timestamps[i], "y": float(df["ST"].iloc[i])} for i in range(len(df)) if pd.notna(df["ST"].iloc[i]) and df["Dir"].iloc[i] == -1]
        
        last, first = float(df["Close"].iloc[-1]), float(df["Close"].iloc[0])
        rsi_c = float(df["RSI"].dropna().iloc[-1]) if not df["RSI"].dropna().empty else 50
        
        return {
            "chart": {"candles": candles, "sma20": sma_series("SMA20"), "sma50": sma_series("SMA50"), "sma100": sma_series("SMA100"), "sma200": sma_series("SMA200"), "rsi_os": rsi_os, "rsi_ob": rsi_ob, "st_buy": st_buy, "st_sell": st_sell},
            "last_price": last, "change": last - first, "change_pct": (last - first) / first * 100, "rsi_current": rsi_c, "alertas": detectar_alertas(df)
        }
    except Exception as e: return {"error": str(e)}

@app.get("/api/vip/etf-details")
async def get_vip_etf_details(user_id: str = "default"):
    prefs = load_prefs(user_id)
    if not prefs.get("is_vip"): return {"ok": False, "error": "Acceso restringido"}
    watchlist_str = prefs.get("watchlist", "")
    if not watchlist_str: return {"ok": True, "details": []}
    favs = [t.strip().upper() for t in watchlist_str.split(",") if t.strip()]
    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM etf_details WHERE ticker = ANY(%s)", (favs,))
                return {"ok": True, "details": [dict(r) for r in cur.fetchall()]}
    except Exception as e: return {"ok": False, "error": str(e)}

@app.get("/api/row/{ticker}")
async def get_row(ticker: str):
    try:
        df = yf.download(ticker.upper(), period="1y", interval="1d", progress=False)
        if df.empty: return {"error": "not found"}
        df = calcular_indicadores(clean_df(df))
        last, first = float(df["Close"].iloc[-1]), float(df["Close"].iloc[0])
        def lv(col): return float(df[col].dropna().iloc[-1]) if not df[col].dropna().empty else None
        return {
            "ticker": ticker.upper(), "price": last, "change_pct": round((last - first) / first * 100, 2),
            "rsi": round(df["RSI"].dropna().iloc[-1], 1) if not df["RSI"].dropna().empty else None,
            "sma20": lv("SMA20"), "sma50": lv("SMA50"), "sma100": lv("SMA100"), "sma200": lv("SMA200"),
            "st_dir": int(df["Dir"].dropna().iloc[-1]) if not df["Dir"].dropna().empty else 0
        }
    except Exception as e: return {"error": str(e)}

@app.get("/api/watch")
async def watch_favorites(tickers: str = ""):
    alertas = []
    for t in [x.strip().upper() for x in tickers.split(",") if x.strip()]:
        try:
            df = yf.download(t, period="1y", interval="1d", progress=False)
            if not df.empty: alertas.extend(detectar_alertas(calcular_indicadores(clean_df(df)), ticker=t))
        except: pass
    return {"alertas": alertas}

@app.get("/api/sparkline/{ticker}")
async def sparkline(ticker: str):
    try:
        df = yf.download(ticker.upper(), period="1mo", interval="1d", progress=False)
        if df.empty: return {"closes": [], "pct": 0}
        closes = clean_df(df)["Close"].dropna().tolist()
        return {"closes": [float(c) for c in closes], "pct": round((closes[-1]-closes[0])/closes[0]*100, 2) if len(closes)>1 else 0}
    except: return {"closes": [], "pct": 0}

@app.get("/api/vip/details-view")
async def vip_details_view():
    return FileResponse("templates/vip_details.html")

@app.post("/api/notifications/redeem")
async def redeem_code(request: Request, user_id: str = "default"):
    body = await request.json()
    code = body.get("code", "").upper().strip()
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT is_used FROM access_codes WHERE code = %s", (code,))
                row = cur.fetchone()
                if not row: return {"ok": False, "error": "Invalido"}
                if row[0]: return {"ok": False, "error": "Usado"}
                if code not in ["VIP333"] + [f"VIP{i:03d}" for i in range(1, 11)]:
                    cur.execute("UPDATE access_codes SET is_used = TRUE WHERE code = %s", (code,))
                cur.execute("INSERT INTO notification_prefs (user_id, is_vip, updated_at) VALUES (%s, TRUE, NOW()) ON CONFLICT (user_id) DO UPDATE SET is_vip = TRUE, updated_at = NOW()", (user_id,))
            conn.commit()
        return {"ok": True, "msg": "VIP activado"}
    except Exception as e: return {"ok": False, "error": str(e)}

@app.get("/api/notifications/prefs")
async def get_notification_prefs(user_id: str = "default"):
    prefs = load_prefs(user_id)
    if not prefs.get("is_vip"): prefs.update({"telegram_enabled": False, "email_enabled": False})
    return prefs

@app.post("/api/notifications/prefs")
async def set_notification_prefs(request: Request, user_id: str = "default"):
    body = await request.json()
    prefs = load_prefs(user_id)
    if not prefs.get("is_vip"): body.update({"telegram_enabled": False, "email_enabled": False})
    prefs.update(body)
    save_prefs(prefs, user_id)
    return {"ok": True}

@app.post("/api/notifications/send")
async def send_alerts_now(request: Request, user_id: str = "default"):
    body = await request.json()
    alertas = []
    for t in [x.strip().upper() for x in body.get("tickers", "").split(",") if x.strip()]:
        try:
            df = yf.download(t, period="1y", interval="1d", progress=False)
            if not df.empty: alertas.extend(detectar_alertas(calcular_indicadores(clean_df(df)), ticker=t))
        except: pass
    if alertas:
        await dispatch_notifications(load_prefs(user_id), alertas)
        return {"ok": True, "count": len(alertas)}
    return {"ok": True, "count": 0}

@app.get("/api/notifications/status")
async def notification_status():
    return {
        "telegram_configured": bool(os.environ.get("TELEGRAM_BOT_TOKEN")),
        "email_configured": bool(os.environ.get("SMTP_USER") and os.environ.get("SMTP_PASS")),
    }

@app.post("/api/telegram/webhook")
async def telegram_webhook(request: Request):
    import httpx
    try:
        data = await request.json()
        msg = data.get("message") or data.get("edited_message")
        if not msg: return {"ok": True}
        chat_id, token = msg["chat"]["id"], os.environ.get("TELEGRAM_BOT_TOKEN", "")
        reply = f"✅ Tu Chat ID es: <code>{chat_id}</code>"
        async with httpx.AsyncClient() as client:
            await client.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": reply, "parse_mode": "HTML"})
        return {"ok": True}
    except: return {"ok": False}

