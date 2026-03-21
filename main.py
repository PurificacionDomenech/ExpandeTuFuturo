import yfinance as yf
import pandas as pd
import numpy as np
import asyncio
import os
import sys
import traceback
import logging
from functools import partial
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from notifications import load_prefs, save_prefs, dispatch_notifications, format_alerts_text, get_db

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("etf")


async def _run_blocking(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))

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

def _get_public_url():
    for var in ("REPLIT_DEPLOYMENT_URL", "REPLIT_DOMAINS", "REPLIT_DEV_DOMAIN"):
        val = os.environ.get(var, "").split(",")[0].strip()
        if val:
            return val if val.startswith("http") else f"https://{val}"
    return ""


def _is_production():
    domains = os.environ.get("REPLIT_DOMAINS", "")
    if ".replit.app" in domains:
        return True
    if os.environ.get("REPLIT_DEPLOYMENT_URL"):
        return True
    return False


@app.on_event("startup")
async def startup_event():
    public_url = _get_public_url()
    is_prod = _is_production()
    logger.info(f"=== ETF STARTUP === prod={is_prod} url={public_url}")
    logger.info(f"  REPLIT_DEPLOYMENT_URL={os.environ.get('REPLIT_DEPLOYMENT_URL','')}")
    logger.info(f"  REPLIT_DOMAINS={os.environ.get('REPLIT_DOMAINS','')}")
    logger.info(f"  REPLIT_DEV_DOMAIN={os.environ.get('REPLIT_DEV_DOMAIN','')}")

    import cron_scanner
    asyncio.create_task(cron_scanner.main())
    logger.info("Background scanner task started")
    asyncio.create_task(auto_setup_telegram_webhook())
    asyncio.create_task(keep_alive_ping())


async def keep_alive_ping():
    import httpx
    await asyncio.sleep(30)
    public_url = _get_public_url()
    if not public_url:
        logger.warning("KEEP-ALIVE: No public URL found, skipping")
        return
    health_url = f"{public_url}/health"
    logger.info(f"KEEP-ALIVE: Pinging {health_url} every 5 minutes")
    while True:
        try:
            async with httpx.AsyncClient(timeout=15, verify=False) as client:
                r = await client.get(health_url)
                logger.info(f"KEEP-ALIVE: ping {r.status_code}")
        except Exception as e:
            logger.warning(f"KEEP-ALIVE: ping failed ({e})")
        await asyncio.sleep(300)


async def auto_setup_telegram_webhook():
    import httpx
    await asyncio.sleep(5)
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        logger.warning("TELEGRAM: No bot token, skipping webhook setup")
        return

    is_prod = _is_production()
    public_url = _get_public_url()

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            if is_prod and public_url:
                webhook_url = f"{public_url}/api/telegram/webhook"
                r = await client.post(
                    f"https://api.telegram.org/bot{token}/setWebhook",
                    json={"url": webhook_url, "allowed_updates": ["message"]}
                )
                logger.info(f"TELEGRAM: Webhook (PROD) set to {webhook_url} -> {r.json()}")
                return

            if not is_prod and public_url:
                info_r = await client.get(f"https://api.telegram.org/bot{token}/getWebhookInfo")
                info = info_r.json().get("result", {})
                current_url = info.get("url", "")
                if current_url and (".replit.app" in current_url or current_url != f"{public_url}/api/telegram/webhook"):
                    if ".replit.app" in current_url:
                        logger.info(f"TELEGRAM: Webhook de producción activo ({current_url}), no se sobreescribe desde dev")
                        return
                webhook_url = f"{public_url}/api/telegram/webhook"
                r = await client.post(
                    f"https://api.telegram.org/bot{token}/setWebhook",
                    json={"url": webhook_url, "allowed_updates": ["message"]}
                )
                logger.info(f"TELEGRAM: Webhook (dev) set to {webhook_url} -> {r.json()}")
                return

        logger.warning("TELEGRAM: No domain detected, skipping webhook setup")
    except Exception as e:
        logger.error(f"TELEGRAM: Webhook setup error: {e}")

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

    event_dt = df.index[n]
    try:
        if event_dt.hour == 0 and event_dt.minute == 0:
            event_str = event_dt.strftime("%d/%m/%Y")
        else:
            event_str = event_dt.strftime("%d/%m %H:%M")
    except Exception:
        event_str = str(event_dt)[:10]
    
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
                "msg": prefix + "Precio cruza " + nombre + " al alza $" + str(round(precio_now, 2)) + f" ({event_str})"})
        elif precio_prev > sma_prev and precio_now <= sma_now:
            alertas.append({"nivel": "bearish",
                "msg": prefix + "Precio cruza " + nombre + " a la baja $" + str(round(precio_now, 2)) + f" ({event_str})"})
        elif sma_now > 0 and abs(precio_now - sma_now) / sma_now * 100 <= 0.4:
            alertas.append({"nivel": "info",
                "msg": prefix + "Precio tocando " + nombre + " $" + str(round(precio_now, 2)) + f" ({event_str})"})

    s100_n, s100_p = smas["SMA100"]
    s200_n, s200_p = smas["SMA200"]
    if pd.notna(s100_n) and pd.notna(s200_n) and pd.notna(s100_p) and pd.notna(s200_p):
        if s100_p < s200_p and s100_n >= s200_n:
            alertas.append({"nivel": "bullish", "msg": prefix + f"Golden Cross SMA100/200 ({event_str})"})
        elif s100_p > s200_p and s100_n <= s200_n:
            alertas.append({"nivel": "bearish", "msg": prefix + f"Death Cross SMA100/200 ({event_str})"})

    s20_n, s20_p = smas["SMA20"]
    s50_n, s50_p = smas["SMA50"]
    if pd.notna(s20_n) and pd.notna(s50_n) and pd.notna(s20_p) and pd.notna(s50_p):
        if s20_p < s50_p and s20_n >= s50_n:
            alertas.append({"nivel": "bullish", "msg": prefix + f"SMA20 cruza sobre SMA50 ({event_str})"})
        elif s20_p > s50_p and s20_n <= s50_n:
            alertas.append({"nivel": "bearish", "msg": prefix + f"SMA20 cruza bajo SMA50 ({event_str})"})

    return alertas


def evaluar_confluencias(df, ticker=""):
    n = len(df) - 1
    if n < 2:
        return None

    precio = float(df["Close"].iloc[n])
    rsi_val = float(df["RSI"].iloc[n]) if "RSI" in df.columns and pd.notna(df["RSI"].iloc[n]) else None
    st_dir = int(df["Dir"].iloc[n]) if "Dir" in df.columns and pd.notna(df["Dir"].iloc[n]) else 0

    sma100 = float(df["SMA100"].iloc[n]) if pd.notna(df["SMA100"].iloc[n]) else None
    sma200 = float(df["SMA200"].iloc[n]) if pd.notna(df["SMA200"].iloc[n]) else None

    event_dt = df.index[n]
    try:
        if event_dt.hour == 0 and event_dt.minute == 0:
            event_str = event_dt.strftime("%d/%m/%Y")
        else:
            event_str = event_dt.strftime("%d/%m %H:%M")
    except Exception:
        event_str = str(event_dt)[:10]

    confluencias = []
    puntos = 0

    rsi_ok = rsi_val is not None and rsi_val < 30
    confluencias.append({
        "id": "rsi_sobreventa",
        "ok": rsi_ok,
        "texto": f"RSI sobrevendido ({rsi_val:.1f}) — señal de rebote" if rsi_ok and rsi_val else "RSI no en sobreventa",
        "tipo": "momentum"
    })
    if rsi_ok:
        puntos += 1

    st_ok = st_dir == 1
    confluencias.append({
        "id": "supertrend",
        "ok": st_ok,
        "texto": "SuperTrend alcista — tendencia confirmada" if st_ok else "SuperTrend bajista",
        "tipo": "tendencia"
    })
    if st_ok:
        puntos += 1

    cerca_sma200 = sma200 is not None and sma200 != 0 and abs(precio - sma200) / sma200 * 100 <= 0.8
    confluencias.append({
        "id": "zona_sma200",
        "ok": cerca_sma200,
        "texto": f"Precio en zona SMA200 (${sma200:.2f}) — soporte clave" if cerca_sma200 and sma200 else "Lejos de SMA200",
        "tipo": "soporte"
    })
    if cerca_sma200:
        puntos += 1

    cerca_sma100 = not cerca_sma200 and sma100 is not None and sma100 != 0 and abs(precio - sma100) / sma100 * 100 <= 0.8
    confluencias.append({
        "id": "zona_sma100",
        "ok": cerca_sma100,
        "texto": f"Precio en zona SMA100 (${sma100:.2f}) — soporte medio" if cerca_sma100 and sma100 else "Lejos de SMA100",
        "tipo": "soporte"
    })
    if cerca_sma100:
        puntos += 1

    cruce_dorado = sma100 is not None and sma200 is not None and sma100 > sma200
    confluencias.append({
        "id": "golden_cross",
        "ok": cruce_dorado,
        "texto": "SMA100 > SMA200 — Cruce Dorado activo" if cruce_dorado else "Sin Cruce Dorado",
        "tipo": "estructura"
    })
    if cruce_dorado:
        puntos += 1

    bajo_sma200 = sma200 is not None and precio < sma200
    confluencias.append({
        "id": "zona_valor",
        "ok": bajo_sma200,
        "texto": "Precio por debajo de SMA200 — zona de valor" if bajo_sma200 else "Precio sobre SMA200",
        "tipo": "valor"
    })
    if bajo_sma200:
        puntos += 1

    if puntos < 3:
        return None

    if puntos >= 5:
        estado = "FAVORABLE"
    elif puntos >= 3:
        estado = "INTERESANTE"
    else:
        estado = "CONSIDERAR"

    return {
        "ticker": str(ticker).upper().strip(),
        "precio": precio,
        "rsi": rsi_val,
        "st_dir": st_dir,
        "puntos": puntos,
        "estado": estado,
        "nivel": "bullish" if puntos >= 5 else "info",
        "confluencias": confluencias,
        "event_str": event_str,
    }


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

@app.get("/welcome")
async def welcome():
    from starlette.responses import HTMLResponse
    with open("templates/welcome.html", "r") as f:
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
        df = await _run_blocking(yf.download, ticker.upper(), period=yf_period, interval=yf_interval, progress=False)
        if df.empty:
            df = await _run_blocking(yf.download, ticker.upper(), period="1y", interval=yf_interval, progress=False)
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
        df = await _run_blocking(yf.download, ticker.upper(), period="1y", interval="1d", progress=False)
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
            df = await _run_blocking(yf.download, t.upper(), period="1y", interval="1d", progress=False)
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
        df = await _run_blocking(yf.download, ticker.upper(), period="1mo", interval="1d", progress=False)
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
    from notifications import dispatch_confluencias
    try:
        body = await request.json()
    except Exception:
        return {"ok": False, "error": "Invalid request"}
    channel = body.get("channel", "all")
    prefs = load_prefs(user_id)

    test_resultado = {
        "ticker": "AAPL",
        "precio": 215.40,
        "rsi": 28.5,
        "st_dir": 1,
        "puntos": 5,
        "estado": "FAVORABLE",
        "nivel": "bullish",
        "event_str": "20/03/2026",
        "confluencias": [
            {"id": "rsi_sobreventa", "ok": True, "texto": "RSI sobrevendido (28.5) — señal de rebote", "tipo": "momentum"},
            {"id": "supertrend", "ok": True, "texto": "SuperTrend alcista — tendencia confirmada", "tipo": "tendencia"},
            {"id": "zona_sma200", "ok": True, "texto": "Precio en zona SMA200 ($214.80) — soporte clave", "tipo": "soporte"},
            {"id": "zona_sma100", "ok": False, "texto": "Lejos de SMA100", "tipo": "soporte"},
            {"id": "golden_cross", "ok": True, "texto": "SMA100 > SMA200 — Cruce Dorado activo", "tipo": "estructura"},
            {"id": "zona_valor", "ok": True, "texto": "Precio por debajo de SMA200 — zona de valor", "tipo": "valor"},
        ],
    }

    test_prefs = dict(prefs)
    if channel != "all":
        for ch in ["telegram", "whatsapp", "email"]:
            if ch != channel:
                test_prefs[f"{ch}_enabled"] = False

    results = await dispatch_confluencias(test_prefs, test_resultado)
    return {"ok": True, "results": results}

@app.post("/api/notifications/send")
async def send_alerts_now(request: Request, user_id: str = "default"):
    from notifications import dispatch_confluencias
    try:
        body = await request.json()
    except Exception:
        return {"ok": False, "error": "Invalid request"}
    tickers_str = body.get("tickers", "")
    prefs = load_prefs(user_id)

    import cron_scanner
    sent_count = 0
    all_results = {}
    legacy_alertas = []

    for t in tickers_str.split(","):
        t = t.strip()
        if not t:
            continue
        try:
            df = await _run_blocking(yf.download, t.upper(), period="1y", interval="1d", progress=False)
            if df.empty:
                continue
            df = clean_df(df)
            df = calcular_indicadores(df)

            resultado = evaluar_confluencias(df, ticker=t.upper())
            if resultado:
                dedup_key = f"{t.upper()}_{resultado['estado']}_{round(resultado['precio'], -1)}"
                if cron_scanner._already_sent(user_id, dedup_key):
                    continue
                r = await dispatch_confluencias(prefs, resultado)
                any_ok = any(v.get("ok") for v in r.values()) if r else False
                if any_ok:
                    cron_scanner._mark_sent(user_id, dedup_key)
                all_results[t.upper()] = r
                sent_count += 1
            else:
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
                    legacy_alertas.append({**alert, "msg": f"{original_msg} [{estado_icon} {estado}]"})
                    cron_scanner._mark_sent(user_id, original_msg)
        except Exception:
            pass

    if legacy_alertas:
        r = await dispatch_notifications(prefs, legacy_alertas)
        all_results["_legacy"] = r
        sent_count += len(legacy_alertas)

    if sent_count > 0:
        return {"ok": True, "alertas_count": sent_count, "results": all_results}
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
            logger.info(f"TELEGRAM BOT reply to {chat_id}: ok={result.get('ok')}, error={result.get('description','')}")
        return {"ok": True}
    except Exception as e:
        logger.error(f"TELEGRAM BOT ERROR: {e}")
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
