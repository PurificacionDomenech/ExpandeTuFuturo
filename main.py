import yfinance as yf
import pandas as pd
import numpy as np
import asyncio
import os
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from notifications import load_prefs, save_prefs, dispatch_notifications, format_alerts_text, get_db
from indicators import clean_df, calcular_indicadores, detectar_alertas
import cron_scanner

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    # Iniciar el escáner automático en segundo plano
    asyncio.create_task(cron_scanner.main())
    print("Background scanner task started")

INTERVAL_MAP = {
    "1d":  ("1d",  "max"),
    "1wk": ("1wk", "max"),
    "1mo": ("1mo", "max"),
}


def ts_ms(idx):
    return [int(t.timestamp() * 1000) for t in idx]


@app.get("/")
async def splash():
    return FileResponse("templates/splash.html")


@app.get("/app")
async def index():
    return FileResponse("templates/index.html")


@app.get("/api/chart/{ticker}")
async def get_chart(ticker: str, interval: str = "1d"):
    try:
        yf_interval, yf_period = INTERVAL_MAP.get(interval, ("1d", "1y"))
        # Forzar un periodo razonable si 'max' falla
        df = yf.download(ticker.upper(), period=yf_period, interval=yf_interval, progress=False)
        if df.empty:
            # Reintentar con 1y por si acaso
            df = yf.download(ticker.upper(), period="1y", interval=yf_interval, progress=False)
            if df.empty:
                return {"error": "Simbolo no encontrado o sin datos: " + ticker}
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


@app.post("/api/admin/set-vip")
async def set_vip_status(request: Request):
    # Endpoint simple para que el admin dé acceso VIP
    body = await request.json()
    user_id = body.get("user_id")
    is_vip = body.get("is_vip", True)
    if not user_id:
        return {"ok": False, "error": "Falta user_id"}
    
    prefs = load_prefs(user_id)
    prefs["is_vip"] = is_vip
    save_prefs(prefs, user_id)
    return {"ok": True, "user_id": user_id, "is_vip": is_vip}


@app.post("/api/admin/generate-code")
async def generate_code():
    import secrets
    import string
    # Solo el admin debería poder llamar a esto, por ahora es libre para que tú lo uses
    code = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8))
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO access_codes (code) VALUES (%s)", (code,))
            conn.commit()
        return {"ok": True, "code": code}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/notifications/redeem")
async def redeem_code(request: Request, user_id: str = "default"):
    body = await request.json()
    code = body.get("code", "").upper().strip()
    if not code:
        return {"ok": False, "error": "Código vacío"}
    
    # DEBUG: Imprimir para ver qué llega
    print(f"DEBUG: Intentando canjear código '{code}' para usuario '{user_id}'")
    
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT is_used FROM access_codes WHERE code = %s", (code,))
                row = cur.fetchone()
                if not row:
                    print(f"DEBUG: Código '{code}' no encontrado en BD")
                    return {"ok": False, "error": "Código inválido"}
                if row[0]:
                    print(f"DEBUG: Código '{code}' ya está marcado como usado")
                    return {"ok": False, "error": "Código ya utilizado"}
                
                # Marcar código como usado (excepto para VIP333 y los nuevos VIP001-VIP010)
                special_codes = ["VIP333"] + [f"VIP{i:03d}" for i in range(1, 11)]
                if code not in special_codes:
                    cur.execute("UPDATE access_codes SET is_used = TRUE WHERE code = %s", (code,))
                
                # ACTUALIZACIÓN DIRECTA DE VIP
                cur.execute("SELECT user_id FROM notification_prefs WHERE user_id = %s", (user_id,))
                if not cur.fetchone():
                    cur.execute("INSERT INTO notification_prefs (user_id, is_vip, updated_at) VALUES (%s, TRUE, NOW())", (user_id,))
                else:
                    cur.execute("UPDATE notification_prefs SET is_vip = TRUE, updated_at = NOW() WHERE user_id = %s", (user_id,))
                
            conn.commit()
        print(f"DEBUG: Canje exitoso para {user_id}")
        return {"ok": True, "msg": "¡Felicidades! Ahora tienes acceso VIP"}
    except Exception as e:
        print(f"DEBUG: Error en redeem: {e}")
        return {"ok": False, "error": str(e)}


@app.get("/api/notifications/prefs")
async def get_notification_prefs(user_id: str = "default"):
    try:
        prefs = load_prefs(user_id)
        # Si NO es VIP, capamos las opciones para que el front lo sepa
        if not prefs.get("is_vip"):
            prefs["telegram_enabled"] = False
            prefs["email_enabled"] = False
        return prefs
    except Exception as e:
        print(f"ERROR in get_notification_prefs: {e}")
        return {"is_vip": False, "error": str(e)}


@app.post("/api/notifications/prefs")
async def set_notification_prefs(request: Request, user_id: str = "default"):
    body = await request.json()
    prefs = load_prefs(user_id)
    
    # Impedir que un no-vip active notificaciones vía API
    if not prefs.get("is_vip"):
        body["telegram_enabled"] = False
        body["email_enabled"] = False
        
    prefs.update(body)
    save_prefs(prefs, user_id)
    return {"ok": True}


@app.post("/api/notifications/test")
async def test_notification(request: Request, user_id: str = "default"):
    body = await request.json()
    channel = body.get("channel", "all")
    prefs = load_prefs(user_id)

    test_alertas = [
        {"nivel": "bullish", "msg": "[AAPL] Precio cruza SMA50 al alza $185.20"},
        {"nivel": "bearish", "msg": "[BTC-USD] Death Cross SMA100/200 detectado"},
        {"nivel": "info",    "msg": "[ETH-USD] Precio tocando SMA200 $182.50"},
    ]

    test_prefs = dict(prefs)
    if channel != "all":
        for ch in ["telegram", "whatsapp", "email"]:
            if ch != channel:
                test_prefs[f"{ch}_enabled"] = False

    results = await dispatch_notifications(test_prefs, test_alertas)
    return {"ok": True, "results": results}


@app.post("/api/notifications/send")
async def send_alerts_now(request: Request, user_id: str = "default"):
    body = await request.json()
    tickers_str = body.get("tickers", "")
    prefs = load_prefs(user_id)

    all_alertas = []
    for t in tickers_str.split(","):
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

    if all_alertas:
        results = await dispatch_notifications(prefs, all_alertas)
        return {"ok": True, "alertas_count": len(all_alertas), "results": results}
    return {"ok": True, "alertas_count": 0, "msg": "Sin alertas activas para enviar"}


@app.get("/api/notifications/status")
async def notification_status():
    try:
        has_telegram = bool(os.environ.get("TELEGRAM_BOT_TOKEN"))
        has_twilio   = bool(os.environ.get("TWILIO_ACCOUNT_SID") and os.environ.get("TWILIO_AUTH_TOKEN"))
        has_smtp     = bool(os.environ.get("SMTP_USER") and os.environ.get("SMTP_PASS"))
        return {
            "telegram_configured":  has_telegram,
            "whatsapp_configured":  has_twilio,
            "email_configured":     has_smtp,
        }
    except Exception as e:
        print(f"ERROR in notification_status: {e}")
        return {"error": str(e)}


@app.post("/api/telegram/webhook")
async def telegram_webhook(request: Request):
    import httpx
    try:
        data = await request.json()
        message = data.get("message") or data.get("edited_message")
        if not message:
            return {"ok": True}

        chat_id = message["chat"]["id"]
        text = message.get("text", "")
        first_name = message["from"].get("first_name", "")
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if not token:
            return {"ok": False}

        if text.startswith("/start") or text.startswith("/chatid"):
            reply = (
                f"¡Hola {first_name}! 👋\n\n"
                f"✅ Tu <b>Chat ID</b> es:\n"
                f"<code>{chat_id}</code>\n\n"
                f"Copia ese número y pégalo en el panel de Alertas del <b>ETF Market Scanner</b> "
                f"para recibir notificaciones de mercado aquí."
            )
        else:
            reply = (
                f"Tu <b>Chat ID</b> es: <code>{chat_id}</code>\n\n"
                f"Usa /start para ver las instrucciones completas."
            )

        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": reply, "parse_mode": "HTML"}
            )
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/telegram/setup-webhook")
async def setup_telegram_webhook(request: Request):
    import httpx
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        return {"ok": False, "error": "TELEGRAM_BOT_TOKEN no configurado"}

    domain = os.environ.get("REPLIT_DEV_DOMAIN", "")
    if not domain:
        return {"ok": False, "error": "Dominio no detectado"}

    webhook_url = f"https://{domain}/api/telegram/webhook"
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"https://api.telegram.org/bot{token}/setWebhook",
            json={"url": webhook_url, "allowed_updates": ["message"]}
        )
        result = r.json()
    return {"ok": result.get("ok"), "webhook_url": webhook_url, "telegram_response": result}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
