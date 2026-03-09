import asyncio
import yfinance as yf
import gc
import os
import time
import hashlib
from datetime import datetime, timedelta
from notifications import dispatch_notifications, get_db

# ── Deduplicación en memoria: {alert_hash -> datetime_sent} ──────────────────
_sent_alerts: dict = {}
_DEDUP_HOURS = 4  # No reenviar la misma alerta en las próximas 4 horas


def _alert_hash(user_id: str, msg: str) -> str:
    raw = f"{user_id}|{msg}"
    return hashlib.md5(raw.encode()).hexdigest()


def _already_sent(user_id: str, msg: str) -> bool:
    h = _alert_hash(user_id, msg)
    sent_at = _sent_alerts.get(h)
    if sent_at and datetime.now() - sent_at < timedelta(hours=_DEDUP_HOURS):
        return True
    return False


def _mark_sent(user_id: str, msg: str):
    h = _alert_hash(user_id, msg)
    _sent_alerts[h] = datetime.now()


def _clean_old_sent():
    cutoff = datetime.now() - timedelta(hours=_DEDUP_HOURS + 1)
    to_del = [k for k, v in _sent_alerts.items() if v < cutoff]
    for k in to_del:
        del _sent_alerts[k]


# ── Cálculo de estado (misma lógica que el frontend) ─────────────────────────
def calcular_estado_pts(price, sma20, sma50, sma100, sma200, rsi):
    pts = 0
    if rsi is not None:
        if rsi < 30:
            pts += 3
        elif rsi < 40:
            pts += 2
        elif rsi < 50:
            pts += 1

    if sma200 is not None and abs(price - sma200) / sma200 * 100 <= 0.8:
        pts += 4
    elif sma100 is not None and abs(price - sma100) / sma100 * 100 <= 0.8:
        pts += 3
    elif sma50 is not None and abs(price - sma50) / sma50 * 100 <= 0.8:
        pts += 2
    elif sma20 is not None and abs(price - sma20) / sma20 * 100 <= 0.8:
        pts += 1

    if sma200 is not None and price < sma200:
        pts += 1

    return pts


def get_estado_label(pts: int) -> str:
    if pts >= 5:
        return "Favorable"
    elif pts >= 3:
        return "Interesante"
    elif pts >= 2:
        return "Considerar"
    else:
        return "No ahora"


def safe_float(val):
    try:
        f = float(val)
        import math
        return None if math.isnan(f) else f
    except Exception:
        return None


async def scan_and_notify():
    from main import calcular_indicadores, detectar_alertas, clean_df
    import pandas as pd

    now_str = time.ctime()
    print(f"[{now_str}] Iniciando escaneo automático de vigilancia...")
    _clean_old_sent()

    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT user_id, telegram_enabled, telegram_chat_id, "
                    "email_enabled, email_address, watchlist "
                    "FROM notification_prefs WHERE is_vip = TRUE AND (vip_expires_at IS NULL OR vip_expires_at > NOW())"
                )
                users = cur.fetchall()

        for user in users:
            uid, tg_en, tg_id, em_en, em_adr, wl_str = user
            print(f"  Procesando {uid} (TG:{tg_en}, EM:{em_en}, WL:{wl_str})")
            if not wl_str:
                continue
            if not tg_en and not em_en:
                continue

            tickers = [t.strip() for t in wl_str.split(",") if t.strip()]
            filtered_alerts = []

            for t in tickers:
                try:
                    df_full = yf.download(t.upper(), period="1y", interval="1d", progress=False)
                    if df_full.empty:
                        continue

                    df_full = clean_df(df_full)
                    df_full = calcular_indicadores(df_full)

                    n = len(df_full) - 1
                    if n < 2:
                        continue

                    price   = safe_float(df_full["Close"].iloc[n])
                    sma20   = safe_float(df_full["SMA20"].iloc[n])
                    sma50   = safe_float(df_full["SMA50"].iloc[n])
                    sma100  = safe_float(df_full["SMA100"].iloc[n])
                    sma200  = safe_float(df_full["SMA200"].iloc[n])
                    rsi_val = safe_float(df_full["RSI"].iloc[n]) if "RSI" in df_full.columns else None

                    if price is None:
                        continue

                    pts = calcular_estado_pts(price, sma20, sma50, sma100, sma200, rsi_val)
                    estado = get_estado_label(pts)

                    # ── Solo enviar si el estado es Favorable, Interesante o Considerar ──
                    if estado == "No ahora":
                        print(f"    [{t}] Estado '{estado}' (pts={pts}) — omitido")
                        continue

                    raw_alerts = detectar_alertas(df_full, ticker=t.upper())

                    estado_icon = {"Favorable": "🟢", "Interesante": "🟡", "Considerar": "🟠"}.get(estado, "")

                    for alert in raw_alerts:
                        # Solo señales alcistas e informativas — NO señales de caída
                        if alert.get("nivel") == "bearish":
                            continue

                        original_msg = alert["msg"]
                        enriched_msg = f"{original_msg} [{estado_icon} {estado}]"

                        # Deduplicación: no reenviar si ya se mandó en las últimas 4h
                        if _already_sent(uid, original_msg):
                            print(f"    [{t}] DUPLICADO omitido: {original_msg[:60]}")
                            continue

                        filtered_alerts.append({**alert, "msg": enriched_msg})
                        _mark_sent(uid, original_msg)

                    del df_full
                except Exception as e:
                    print(f"    Error escaneando {t}: {e}")

                await asyncio.sleep(0.5)

            gc.collect()

            print(f"  → {len(filtered_alerts)} alertas nuevas para {uid}")
            if filtered_alerts:
                prefs = {
                    "telegram_enabled": tg_en,
                    "telegram_chat_id": tg_id,
                    "email_enabled": em_en,
                    "email_address": em_adr
                }
                await dispatch_notifications(prefs, filtered_alerts)
                print(f"  ✓ Enviadas {len(filtered_alerts)} alertas a {uid}")
            else:
                print(f"  – Sin alertas nuevas para {uid} en este ciclo")

    except Exception as e:
        print(f"Error en scan_and_notify: {e}")
        import traceback
        traceback.print_exc()


def _radar_already_sent_today(user_id):
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM radar_daily_log WHERE sent_date = CURRENT_DATE AND user_id = %s", (user_id,))
                return cur.fetchone() is not None
    except Exception:
        return False

def _mark_radar_sent(user_id):
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO radar_daily_log (user_id) VALUES (%s) ON CONFLICT DO NOTHING", (user_id,))
            conn.commit()
    except Exception:
        pass

async def send_daily_radar():
    from notifications import send_telegram
    print(f"[{time.ctime()}] Enviando resumen diario de Radar Financiero...")

    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT user_id, telegram_chat_id, watchlist "
                    "FROM notification_prefs WHERE is_vip = TRUE AND telegram_enabled = TRUE "
                    "AND telegram_chat_id IS NOT NULL AND telegram_chat_id != '' "
                    "AND watchlist IS NOT NULL AND watchlist != '' "
                    "AND (vip_expires_at IS NULL OR vip_expires_at > NOW())"
                )
                users = cur.fetchall()

        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if not token:
            print("  TELEGRAM_BOT_TOKEN no configurado, omitiendo radar diario")
            return

        for uid, chat_id, wl_str in users:
            if _radar_already_sent_today(uid):
                print(f"  – Radar ya enviado hoy a {uid}, omitiendo")
                continue

            tickers = [t.strip().upper() for t in wl_str.split(",") if t.strip()][:10]
            if not tickers:
                continue

            msg_parts = ["📡 <b>Radar Financiero Diario</b>\n"]

            for ticker in tickers:
                try:
                    t = yf.Ticker(ticker)
                    news_raw = t.news or []
                    if not news_raw:
                        continue

                    msg_parts.append(f"\n<b>🔸 {ticker}</b>")
                    count = 0
                    for n in news_raw[:3]:
                        c = n.get("content", {})
                        title = c.get("title", "")
                        if not title:
                            continue
                        try:
                            from deep_translator import GoogleTranslator
                            title_es = GoogleTranslator(source='en', target='es').translate(title[:400]) or title
                        except Exception:
                            title_es = title
                        click = c.get("clickThroughUrl", {})
                        canon = c.get("canonicalUrl", {})
                        url = ""
                        if click and click.get("url"):
                            url = click["url"]
                        elif canon and canon.get("url"):
                            url = canon["url"]
                        if url:
                            msg_parts.append(f"  • <a href=\"{url}\">{title_es}</a>")
                        else:
                            msg_parts.append(f"  • {title_es}")
                        count += 1
                    if count == 0:
                        msg_parts.pop()
                except Exception as e:
                    print(f"  Error obteniendo noticias de {ticker}: {e}")

                await asyncio.sleep(0.3)

            if len(msg_parts) > 1:
                msg_parts.append("\n<i>ETF · Expande Tu Futuro</i>")
                full_msg = "\n".join(msg_parts)
                if len(full_msg) > 4000:
                    full_msg = full_msg[:3950] + "\n\n<i>… (mensaje recortado)</i>"
                try:
                    r = await send_telegram(token, str(chat_id), full_msg)
                    if r.get("ok"):
                        _mark_radar_sent(uid)
                        print(f"  ✓ Radar diario enviado a {uid}")
                    else:
                        print(f"  ✗ Telegram rechazó radar para {uid}: {r.get('description', 'unknown')}")
                except Exception as e:
                    print(f"  ✗ Error enviando radar a {uid}: {e}")

    except Exception as e:
        print(f"Error en send_daily_radar: {e}")
        import traceback
        traceback.print_exc()


async def main():
    print("Scanner background loop active")
    await asyncio.sleep(60)
    while True:
        try:
            await scan_and_notify()
        except Exception as e:
            print(f"Loop error: {e}")

        now = datetime.now()
        if now.hour >= 8:
            try:
                await send_daily_radar()
            except Exception as e:
                print(f"Radar daily error: {e}")

        await asyncio.sleep(1800)


if __name__ == "__main__":
    asyncio.run(main())
