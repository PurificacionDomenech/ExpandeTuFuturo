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

def _esc_html(text):
    import html
    return html.escape(str(text)) if text else ""

def _get_epicentro_data():
    """Obtiene los indicadores del Epicentro. Retorna lista de dicts."""
    epi_indices = [
        ("^VIX",  "VIX",         "🌡️", True),
        ("^GSPC", "S&P 500",     "📊", False),
        ("^DJI",  "Dow Jones",   "📊", False),
        ("^IXIC", "Nasdaq",      "📊", False),
        ("^RUT",  "Russell 2000","📊", False),
    ]
    results = []
    for sym, name, icon, is_vix in epi_indices:
        try:
            tk = yf.Ticker(sym)
            h = tk.history(period="2d")
            if h.empty:
                continue
            price = float(h["Close"].iloc[-1])
            chg = 0.0
            if len(h) >= 2:
                prev = float(h["Close"].iloc[-2])
                if prev > 0:
                    chg = ((price - prev) / prev) * 100
            results.append({"sym": sym, "name": name, "icon": icon,
                             "is_vix": is_vix, "price": price, "chg": chg})
        except Exception:
            pass
    return results


def _format_epicentro_telegram(epi_data: list) -> str:
    import html as _html
    lines = ["📊 <b>Indicadores del Epicentro</b>\n"]
    for d in epi_data:
        arrow = ("📉" if d["chg"] >= 0 else "📈") if d["is_vix"] else ("📈" if d["chg"] >= 0 else "📉")
        sign = "↑" if d["chg"] >= 0 else "↓"
        safe_name = _html.escape(d['name'])
        lines.append(f"{arrow} <b>{safe_name}</b>  {d['price']:,.2f}  {sign}{abs(d['chg']):.2f}%")
    lines.append("\n<i>ETF · Expande Tu Futuro</i>")
    return "\n".join(lines)


def _format_epicentro_html(epi_data: list) -> str:
    rows = ""
    for d in epi_data:
        color = "#ef4444" if (d["chg"] >= 0 and d["is_vix"]) or (d["chg"] < 0 and not d["is_vix"]) else "#22c55e"
        arrow = "↓" if d["chg"] < 0 else "↑"
        rows += (f'<tr><td style="padding:6px 10px;color:#e8c96d;font-weight:700">{d["name"]}</td>'
                 f'<td style="padding:6px 10px;font-family:monospace">{d["price"]:,.2f}</td>'
                 f'<td style="padding:6px 10px;color:{color};font-weight:700">{arrow}{abs(d["chg"]):.2f}%</td></tr>')
    return (f'<table style="width:100%;border-collapse:collapse;margin-bottom:20px">'
            f'<tr><th style="text-align:left;padding:6px 10px;color:rgba(232,228,217,.5);font-size:11px">ÍNDICE</th>'
            f'<th style="text-align:left;padding:6px 10px;color:rgba(232,228,217,.5);font-size:11px">PRECIO</th>'
            f'<th style="text-align:left;padding:6px 10px;color:rgba(232,228,217,.5);font-size:11px">CAMBIO</th></tr>'
            f'{rows}</table>')


async def _get_ticker_news(ticker: str) -> list:
    """Retorna lista de {title_es, url} para un ticker (max 3 noticias)."""
    try:
        t = yf.Ticker(ticker)
        news_raw = t.news or []
        items = []
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
            url = (click.get("url") if click else None) or (canon.get("url") if canon else None) or ""
            items.append({"title_es": title_es, "url": url})
        return items
    except Exception:
        return []


async def send_daily_radar():
    from notifications import send_telegram, send_email
    import re
    print(f"[{time.ctime()}] Enviando resumen diario de Radar Financiero...")

    # ── Credenciales ──────────────────────────────────────────────────────────
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "")
    smtp_from = os.environ.get("SMTP_FROM", smtp_user)

    try:
        # ── Obtener usuarios VIP activos (telegram O email) ───────────────────
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT user_id, telegram_enabled, telegram_chat_id, "
                    "email_enabled, email_address, watchlist "
                    "FROM notification_prefs WHERE is_vip = TRUE "
                    "AND (vip_expires_at IS NULL OR vip_expires_at > NOW()) "
                    "AND watchlist IS NOT NULL AND watchlist != '' "
                    "AND (telegram_enabled = TRUE OR email_enabled = TRUE)"
                )
                users = cur.fetchall()

        if not users:
            print("  Sin usuarios VIP con notificaciones activas")
            return

        # ── Epicentro (una sola vez) ──────────────────────────────────────────
        epi_data = []
        try:
            epi_data = _get_epicentro_data()
        except Exception as e:
            print(f"  Error obteniendo epicentro: {e}")

        # ── Caché de noticias por ticker ──────────────────────────────────────
        news_cache: dict = {}

        for uid, tg_on, chat_id, em_on, em_addr, wl_str in users:
            if _radar_already_sent_today(uid):
                print(f"  – Radar ya enviado hoy a {uid}, omitiendo")
                continue

            tickers = [t.strip().upper() for t in wl_str.split(",") if t.strip()][:10]
            if not tickers:
                continue

            # Obtener noticias de todos los tickers del usuario
            for ticker in tickers:
                if ticker not in news_cache:
                    news_cache[ticker] = await _get_ticker_news(ticker)
                    await asyncio.sleep(0.3)

            sent_ok = False

            # ── TELEGRAM: un mensaje por ticker ──────────────────────────────
            if tg_on and chat_id and token:
                try:
                    # 1) Mensaje del Epicentro
                    if epi_data:
                        epi_msg = f"📡 <b>Radar Financiero Diario</b>\n\n{_format_epicentro_telegram(epi_data)}"
                        r_epi = await send_telegram(token, str(chat_id), epi_msg)
                        if not r_epi.get("ok"):
                            print(f"    ✗ Epicentro TG rechazado para {uid}: {r_epi.get('description','?')}")
                        await asyncio.sleep(0.5)

                    # 2) Un mensaje por ticker
                    for ticker in tickers:
                        items = news_cache.get(ticker, [])
                        if not items:
                            continue
                        lines = [f"🔸 <b>{ticker}</b>"]
                        for item in items:
                            safe_title = _esc_html(item["title_es"])
                            url = item["url"]
                            if url:
                                safe_url = url.replace("&", "&amp;").replace('"', "%22")
                                lines.append(f'  • <a href="{safe_url}">{safe_title}</a>')
                            else:
                                lines.append(f"  • {safe_title}")
                        lines.append("\n<i>ETF · Expande Tu Futuro</i>")
                        ticker_msg = "\n".join(lines)
                        r = await send_telegram(token, str(chat_id), ticker_msg)
                        if not r.get("ok"):
                            print(f"    ✗ TG ticker {ticker} para {uid}: {r.get('description','?')}")
                        await asyncio.sleep(0.5)

                    sent_ok = True
                    print(f"  ✓ Radar Telegram enviado a {uid} ({len(tickers)} tickers)")
                except Exception as e:
                    print(f"  ✗ Error Telegram radar para {uid}: {e}")

            # ── EMAIL: un único correo HTML con todo ─────────────────────────
            if em_on and em_addr and smtp_user and smtp_pass:
                try:
                    from datetime import datetime
                    try:
                        import pytz
                        now_str = datetime.now(pytz.timezone('Europe/Madrid')).strftime("%d/%m/%Y %H:%M")
                    except Exception:
                        now_str = datetime.now().strftime("%d/%m/%Y %H:%M")

                    html_parts = [f"""
<div style="font-family:'DM Sans',Arial,sans-serif;background:#060810;color:#e8e4d9;
            padding:30px;border-radius:12px;max-width:600px;margin:0 auto">
  <div style="border-bottom:1px solid rgba(201,168,76,.3);padding-bottom:15px;margin-bottom:20px">
    <h2 style="color:#e8c96d;margin:0;font-size:18px">📡 Radar Financiero Diario</h2>
    <p style="color:rgba(232,228,217,.5);font-size:12px;margin:4px 0 0">
      ETF · Expande Tu Futuro · {now_str}</p>
  </div>"""]

                    if epi_data:
                        html_parts.append('<h3 style="color:#e8c96d;font-size:14px;margin:0 0 10px">📊 Indicadores del Epicentro</h3>')
                        html_parts.append(_format_epicentro_html(epi_data))

                    for ticker in tickers:
                        items = news_cache.get(ticker, [])
                        if not items:
                            continue
                        html_parts.append(f'<h3 style="color:#e8c96d;font-size:14px;margin:16px 0 8px">🔸 {ticker}</h3>')
                        for item in items:
                            url = item["url"]
                            title = item["title_es"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                            if url:
                                safe_url = url.replace("&", "&amp;").replace('"', "%22")
                                html_parts.append(
                                    f'<div style="padding:8px 12px;margin:4px 0;border-radius:6px;'
                                    f'background:rgba(255,255,255,.03);border-left:3px solid rgba(201,168,76,.4);font-size:13px">'
                                    f'• <a href="{safe_url}" style="color:#e8c96d;text-decoration:none">{title}</a></div>')
                            else:
                                html_parts.append(
                                    f'<div style="padding:8px 12px;margin:4px 0;border-radius:6px;'
                                    f'background:rgba(255,255,255,.03);border-left:3px solid rgba(201,168,76,.4);font-size:13px">'
                                    f'• {title}</div>')

                    html_parts.append("""
  <div style="margin-top:25px;padding-top:15px;border-top:1px solid rgba(201,168,76,.15);
              font-size:10px;color:rgba(232,228,217,.3)">
    Enviado por ETF · Expande Tu Futuro
  </div>
</div>""")

                    full_html = "\n".join(html_parts)
                    r = await send_email(smtp_host, smtp_port, smtp_user, smtp_pass,
                                         smtp_from, em_addr,
                                         "📡 Radar Financiero Diario · ETF Expande Tu Futuro",
                                         full_html)
                    if r.get("ok"):
                        sent_ok = True
                        print(f"  ✓ Radar Email enviado a {uid} ({em_addr})")
                    else:
                        print(f"  ✗ Error Email radar para {uid}: {r.get('error','?')}")
                except Exception as e:
                    print(f"  ✗ Error Email radar para {uid}: {e}")

            if sent_ok:
                _mark_radar_sent(uid)

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
