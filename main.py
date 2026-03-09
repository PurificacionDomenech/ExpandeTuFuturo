import yfinance as yf
import pandas as pd
import numpy as np
import asyncio
import os
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from notifications import load_prefs, save_prefs, dispatch_notifications, format_alerts_text, get_db

app = FastAPI()

class CatchAllMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            if request.url.path == "/health":
                return JSONResponse(status_code=200, content={"status": "ok"})
            print(f"UNHANDLED ERROR on {request.url.path}: {e}")
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "detail": str(e)}
            )

app.add_middleware(CatchAllMiddleware)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    import cron_scanner
    asyncio.create_task(cron_scanner.main())
    print("Background scanner task started")
    asyncio.create_task(auto_setup_telegram_webhook())

async def auto_setup_telegram_webhook():
    import httpx
    await asyncio.sleep(5)
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        print("TELEGRAM: No bot token, skipping webhook setup")
        return

    deployment_url = os.environ.get("REPLIT_DEPLOYMENT_URL", "")
    dev_domain = os.environ.get("REPLIT_DEV_DOMAIN", "")

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            # Si estamos en producción, siempre actualizamos el webhook
            if deployment_url:
                domain = deployment_url if deployment_url.startswith("http") else f"https://{deployment_url}"
                webhook_url = f"{domain}/api/telegram/webhook"
                r = await client.post(
                    f"https://api.telegram.org/bot{token}/setWebhook",
                    json={"url": webhook_url, "allowed_updates": ["message"]}
                )
                print(f"TELEGRAM: Webhook set to {webhook_url} -> {r.json()}")
                return

            # En desarrollo: solo cambiamos el webhook si no hay ya uno de producción activo
            if dev_domain:
                info_r = await client.get(f"https://api.telegram.org/bot{token}/getWebhookInfo")
                info = info_r.json().get("result", {})
                current_url = info.get("url", "")
                # Si el webhook actual apunta a una URL de producción, no lo sobreescribimos
                if current_url and ".replit.app" in current_url:
                    print(f"TELEGRAM: Webhook de producción activo ({current_url}), no se sobreescribe desde dev")
                    return
                domain = dev_domain if dev_domain.startswith("http") else f"https://{dev_domain}"
                webhook_url = f"{domain}/api/telegram/webhook"
                r = await client.post(
                    f"https://api.telegram.org/bot{token}/setWebhook",
                    json={"url": webhook_url, "allowed_updates": ["message"]}
                )
                print(f"TELEGRAM: Webhook (dev) set to {webhook_url} -> {r.json()}")
                return

        print("TELEGRAM: No domain detected, skipping webhook setup")
    except Exception as e:
        print(f"TELEGRAM: Webhook setup error: {e}")

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
    
    ticker_clean = str(ticker).upper().strip()
    prefix = f"[{ticker_clean}] " if ticker_clean else ""

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

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
async def splash():
    from starlette.responses import HTMLResponse
    with open("templates/splash.html", "r") as f:
        content = f.read()
    return HTMLResponse(content, headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"})

@app.get("/app")
async def index():
    from starlette.responses import HTMLResponse
    import time
    with open("templates/index.html", "r") as f:
        content = f.read()
    content = content.replace("</head>", f'<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate"><meta http-equiv="Pragma" content="no-cache"><meta http-equiv="Expires" content="0"><meta name="v" content="{int(time.time())}"></head>')
    return HTMLResponse(content, headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"})

@app.get("/api/chart/{ticker}")
async def get_chart(ticker: str, interval: str = "1d"):
    try:
        yf_interval, yf_period = INTERVAL_MAP.get(interval, ("1d", "1y"))
        df = yf.download(ticker.upper(), period=yf_period, interval=yf_interval, progress=False)
        if df.empty:
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
    code = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8))
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO access_codes (code) VALUES (%s)", (code,))
            conn.commit()
        return {"ok": True, "code": code}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _translate_text(text, max_len=500):
    if not text:
        return text
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source='en', target='es').translate(text[:max_len]) or text
    except Exception:
        return text

def _parse_news(news_list, max_items=8, include_thumb=False, translate=False):
    items = []
    for n in news_list[:max_items]:
        c = n.get("content", {})
        title = c.get("title", "")
        if not title:
            continue
        summary = c.get("summary", "")
        pub = c.get("pubDate", "")
        provider = c.get("provider", {}).get("displayName", "")
        click = c.get("clickThroughUrl", {})
        canon = c.get("canonicalUrl", {})
        url = ""
        if click and click.get("url"):
            url = click["url"]
        elif canon and canon.get("url"):
            url = canon["url"]
        if translate:
            title = _translate_text(title)
            if summary:
                summary = _translate_text(summary, 300)
        item = {"title": title, "summary": summary[:200], "url": url, "date": pub, "source": provider}
        if include_thumb:
            thumb = ""
            tn = c.get("thumbnail")
            if tn and tn.get("resolutions"):
                for r in tn["resolutions"]:
                    if r.get("tag") == "original":
                        thumb = r.get("url", "")
                        break
                if not thumb:
                    thumb = tn["resolutions"][0].get("url", "")
            item["thumb"] = thumb
        items.append(item)
    return items

def _fetch_ticker_news(ticker, max_items=8, include_thumb=False, translate=False):
    t = yf.Ticker(ticker)
    news = t.news or []
    return _parse_news(news, max_items, include_thumb, translate)

def _fetch_epicentro():
    indices = [
        {"ticker": "^VIX", "name": "VIX", "icon": "🌡️"},
        {"ticker": "^GSPC", "name": "S&P 500", "icon": "📊"},
        {"ticker": "^DJI", "name": "Dow Jones", "icon": "📊"},
        {"ticker": "^IXIC", "name": "Nasdaq", "icon": "📊"},
        {"ticker": "^RUT", "name": "Russell 2000", "icon": "📊"},
    ]
    results = []
    for idx in indices:
        try:
            tk = yf.Ticker(idx["ticker"])
            h = tk.history(period="2d")
            if h.empty:
                continue
            price = float(h["Close"].iloc[-1])
            change_pct = 0.0
            if len(h) >= 2:
                prev = float(h["Close"].iloc[-2])
                if prev > 0:
                    change_pct = ((price - prev) / prev) * 100
            results.append({
                "name": idx["name"],
                "icon": idx["icon"],
                "price": round(price, 2),
                "change_pct": round(change_pct, 2),
            })
        except Exception:
            pass
    return results

@app.get("/api/radar/epicentro")
async def get_epicentro():
    try:
        data = await asyncio.to_thread(_fetch_epicentro)
        return {"indicators": data}
    except Exception as e:
        return {"indicators": [], "error": str(e)}

@app.get("/api/radar/batch/{tickers}")
async def get_radar_batch(tickers: str):
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()][:10]
    async def fetch_one(tk):
        try:
            return tk, await asyncio.to_thread(_fetch_ticker_news, tk, 5, False, True)
        except Exception:
            return tk, []
    tasks = [fetch_one(tk) for tk in ticker_list]
    results_list = await asyncio.gather(*tasks)
    return {"results": dict(results_list)}

@app.get("/api/radar/{ticker}")
async def get_radar_news(ticker: str):
    try:
        items = await asyncio.to_thread(_fetch_ticker_news, ticker, 8, True, True)
        return {"ticker": ticker.upper(), "news": items}
    except Exception as e:
        return {"ticker": ticker.upper(), "news": [], "error": str(e)}

@app.post("/api/notifications/redeem")
async def redeem_code(request: Request, user_id: str = "default"):
    try:
        body = await request.json()
    except Exception:
        return {"ok": False, "error": "Invalid request"}
    code = body.get("code", "").upper().strip()
    if not code:
        return {"ok": False, "error": "Código vacío"}
    
    permanent_codes = ["VIP001", "VIP002", "VIP003", "VIP004", "VIP005", "VIP333", "VIP777"]
    trial_codes = [f"VIP{i:03d}" for i in range(6, 45)]
    all_special = permanent_codes + trial_codes
    is_special = code in all_special
    is_trial = code in trial_codes
    
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT user_id, vip_code FROM notification_prefs WHERE user_id = %s", (user_id,))
                existing = cur.fetchone()
                if existing and existing[1] in permanent_codes:
                    return {"ok": True, "msg": "Ya tienes acceso VIP permanente"}
                
                if not is_special:
                    cur.execute("SELECT is_used FROM access_codes WHERE code = %s", (code,))
                    row = cur.fetchone()
                    if not row:
                        return {"ok": False, "error": "Código inválido"}
                    if row[0]:
                        return {"ok": False, "error": "Código ya utilizado"}
                    cur.execute("UPDATE access_codes SET is_used = TRUE WHERE code = %s", (code,))
                
                expires_sql = "NOW() + INTERVAL '3 months'" if is_trial else "NULL"
                
                if not existing:
                    cur.execute(f"INSERT INTO notification_prefs (user_id, is_vip, vip_code, vip_expires_at, updated_at) VALUES (%s, TRUE, %s, {expires_sql}, NOW())", (user_id, code))
                else:
                    cur.execute(f"UPDATE notification_prefs SET is_vip = TRUE, vip_code = %s, vip_expires_at = {expires_sql}, updated_at = NOW() WHERE user_id = %s", (code, user_id))
                
            conn.commit()
        
        if is_trial:
            return {"ok": True, "msg": "¡Acceso VIP activado! Tienes 3 meses de prueba gratuita"}
        return {"ok": True, "msg": "¡Felicidades! Ahora tienes acceso VIP permanente"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/api/notifications/prefs")
async def get_notification_prefs(user_id: str = "default"):
    try:
        prefs = load_prefs(user_id)
        if not prefs.get("is_vip"):
            prefs["telegram_enabled"] = False
            prefs["email_enabled"] = False
        return prefs
    except Exception as e:
        return {"is_vip": False, "error": str(e)}

@app.post("/api/notifications/prefs")
async def set_notification_prefs(request: Request, user_id: str = "default"):
    try:
        body = await request.json()
    except Exception:
        return {"ok": False, "error": "Invalid request"}

    prefs = load_prefs(user_id)

    if prefs.get("is_vip"):
        prefs["is_vip"] = True

    # Si el body solo trae el watchlist (sincronización silenciosa desde favoritos),
    # solo actualizamos el watchlist y no tocamos telegram/email
    watchlist_only = set(body.keys()) <= {"watchlist", "is_vip"}
    if watchlist_only:
        if "watchlist" in body:
            prefs["watchlist"] = body["watchlist"]
        save_prefs(prefs, user_id)
        return {"ok": True}

    # Guardado completo desde el modal de configuración
    if not prefs.get("is_vip"):
        body["telegram_enabled"] = False
        body["email_enabled"] = False
    else:
        # Para usuarios VIP: telegram/email solo se desactivan si el usuario
        # lo hace explícitamente — nunca desde un guardado accidental
        if "telegram_enabled" not in body:
            body["telegram_enabled"] = prefs.get("telegram_enabled", False)
        if "email_enabled" not in body:
            body["email_enabled"] = prefs.get("email_enabled", False)

    prefs.update(body)
    save_prefs(prefs, user_id)
    return {"ok": True}

@app.post("/api/notifications/test")
async def test_notification(request: Request, user_id: str = "default"):
    try:
        body = await request.json()
    except Exception:
        return {"ok": False, "error": "Invalid request"}
    channel = body.get("channel", "all")
    prefs = load_prefs(user_id)

    test_alertas = [
        {"nivel": "bullish", "msg": "[AAPL] Precio cruza SMA50 al alza $215.40 [🟢 Favorable]"},
        {"nivel": "info",    "msg": "[ETH-USD] Precio tocando SMA200 $2150.80 [🟡 Interesante]"},
        {"nivel": "info",    "msg": "[BTC-USD] Precio tocando SMA100 $84200.50 [🟠 Considerar]"},
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
    try:
        body = await request.json()
    except Exception:
        return {"ok": False, "error": "Invalid request"}
    tickers_str = body.get("tickers", "")
    prefs = load_prefs(user_id)

    import cron_scanner
    filtered_alertas = []
    for t in tickers_str.split(","):
        t = t.strip()
        if not t:
            continue
        try:
            df = yf.download(t.upper(), period="1y", interval="1d", progress=False)
            if df.empty:
                continue
            df = clean_df(df)
            df = calcular_indicadores(df)
            n = len(df) - 1
            if n < 2:
                continue

            price  = cron_scanner.safe_float(df["Close"].iloc[n])
            sma20  = cron_scanner.safe_float(df["SMA20"].iloc[n])
            sma50  = cron_scanner.safe_float(df["SMA50"].iloc[n])
            sma100 = cron_scanner.safe_float(df["SMA100"].iloc[n])
            sma200 = cron_scanner.safe_float(df["SMA200"].iloc[n])
            rsi_s  = df["RSI"].dropna() if "RSI" in df.columns else None
            rsi    = cron_scanner.safe_float(rsi_s.iloc[-1]) if rsi_s is not None and not rsi_s.empty else None

            if price is None:
                continue

            pts = cron_scanner.calcular_estado_pts(price, sma20, sma50, sma100, sma200, rsi)
            estado = cron_scanner.get_estado_label(pts)

            if estado == "No ahora":
                continue

            estado_icon = {"Favorable": "🟢", "Interesante": "🟡", "Considerar": "🟠"}.get(estado, "")
            for alert in detectar_alertas(df, ticker=t.upper()):
                if alert.get("nivel") == "bearish":
                    continue
                original_msg = alert["msg"]
                if cron_scanner._already_sent(user_id, original_msg):
                    continue
                filtered_alertas.append({**alert, "msg": f"{original_msg} [{estado_icon} {estado}]"})
                cron_scanner._mark_sent(user_id, original_msg)
        except Exception:
            pass

    if filtered_alertas:
        results = await dispatch_notifications(prefs, filtered_alertas)
        return {"ok": True, "alertas_count": len(filtered_alertas), "results": results}
    return {"ok": True, "alertas_count": 0, "msg": "Sin alertas nuevas para enviar"}

@app.get("/api/notifications/status")
async def notification_status():
    try:
        has_telegram = bool(os.environ.get("TELEGRAM_BOT_TOKEN"))
        has_twilio   = bool(os.environ.get("TWILIO_ACCOUNT_SID") and os.environ.get("TWILIO_AUTH_TOKEN"))
        has_smtp     = bool(os.environ.get("SMTP_USER") and os.environ.get("SMTP_PASS"))
        total_users = 0
        try:
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM notification_prefs")
                    total_users = cur.fetchone()[0]
        except Exception:
            pass
        return {
            "telegram_configured":  has_telegram,
            "whatsapp_configured":  has_twilio,
            "email_configured":     has_smtp,
            "total_users":          total_users,
        }
    except Exception as e:
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
            resp = await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": reply, "parse_mode": "HTML"}
            )
            result = resp.json()
            print(f"TELEGRAM BOT reply to {chat_id}: ok={result.get('ok')}, error={result.get('description','')}")
        return {"ok": True}
    except Exception as e:
        print(f"TELEGRAM BOT ERROR: {e}")
        return {"ok": False, "error": str(e)}

@app.get("/api/telegram/setup-webhook")
async def setup_telegram_webhook(request: Request):
    import httpx
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        return {"ok": False, "error": "TELEGRAM_BOT_TOKEN no configurado"}

    domain = os.environ.get("REPLIT_DEPLOYMENT_URL", "") or os.environ.get("REPLIT_DEV_DOMAIN", "")
    if not domain:
        return {"ok": False, "error": "Dominio no detectado"}
    if not domain.startswith("http"):
        domain = f"https://{domain}"

    webhook_url = f"{domain}/api/telegram/webhook"
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
