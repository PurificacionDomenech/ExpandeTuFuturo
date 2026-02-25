import os
import asyncio
import httpx
import smtplib
import psycopg2
import psycopg2.extras
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def get_db():
    try:
        return psycopg2.connect(os.environ["DATABASE_URL"], connect_timeout=5)
    except Exception as e:
        print(f"Database Connection Error: {e}")
        raise e

def load_prefs(user_id: str = "default") -> dict:
    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM notification_prefs WHERE user_id = %s",
                    (user_id,)
                )
                row = cur.fetchone()
                if row:
                    d = dict(row)
                    d["is_vip"] = bool(d.get("is_vip", False))
                    return d
    except Exception as e:
        print(f"Error loading prefs for {user_id}: {e}")
    return {"is_vip": False}

def save_prefs(prefs: dict, user_id: str = "default"):
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO notification_prefs
                        (user_id, telegram_enabled, telegram_chat_id,
                         email_enabled, email_address, watchlist, is_vip, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (user_id) DO UPDATE SET
                        telegram_enabled = EXCLUDED.telegram_enabled,
                        telegram_chat_id = EXCLUDED.telegram_chat_id,
                        email_enabled    = EXCLUDED.email_enabled,
                        email_address    = EXCLUDED.email_address,
                        watchlist        = EXCLUDED.watchlist,
                        is_vip           = COALESCE(notification_prefs.is_vip, EXCLUDED.is_vip),
                        updated_at       = NOW()
                """, (
                    user_id,
                    prefs.get("telegram_enabled", False),
                    prefs.get("telegram_chat_id", ""),
                    prefs.get("email_enabled", False),
                    prefs.get("email_address", ""),
                    prefs.get("watchlist", ""),
                    prefs.get("is_vip", False),
                ))
            conn.commit()
    except Exception as e:
        print(f"DB save error for {user_id}: {e}")
        raise RuntimeError(f"DB save error: {e}")

def list_all_users_with_notifications() -> list:
    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM notification_prefs
                    WHERE telegram_enabled = TRUE
                       OR email_enabled    = TRUE
                """)
                return [dict(r) for r in cur.fetchall()]
    except Exception:
        return []

async def send_telegram(bot_token: str, chat_id: str, message: str) -> dict:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(url, json=payload)
        return r.json()

async def send_email(smtp_host: str, smtp_port: int, smtp_user: str,
                     smtp_pass: str, from_addr: str, to_addr: str,
                     subject: str, body_html: str) -> dict:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr

    import re
    plain = body_html.replace("<br>", "\n").replace("</div>", "\n")
    plain = re.sub(r"<[^>]+>", "", plain)

    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(body_html, "html"))

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            None, _send_smtp, smtp_host, smtp_port, smtp_user, smtp_pass,
            from_addr, to_addr, msg
        )
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _send_smtp(host, port, user, password, from_addr, to_addr, msg):
    port = int(port)
    if port == 465:
        with smtplib.SMTP_SSL(host, port, timeout=10) as server:
            server.login(user, password)
            server.sendmail(from_addr, to_addr, msg.as_string())
    else:
        with smtplib.SMTP(host, port, timeout=10) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(from_addr, to_addr, msg.as_string())

def format_alerts_text(alertas: list) -> str:
    if not alertas:
        return "Sin alertas activas."
    icon_map = {"bullish": "🟢", "bearish": "🔴", "info": "🟡"}
    from datetime import datetime
    try:
        import pytz
        tz = pytz.timezone('Europe/Madrid')
        now = datetime.now(tz)
    except:
        now = datetime.now()
    now_str = now.strftime("%H:%M")
    lines = [f"⚡ ETF Market Scanner · {now_str}\n"]
    for a in alertas:
        icon = icon_map.get(a.get("nivel", "info"), "🟡")
        lines.append(f"{icon} {a['msg']}")
    return "\n".join(lines)

async def dispatch_notifications(prefs: dict, alertas: list) -> dict:
    results = {}
    text = format_alerts_text(alertas)
    if prefs.get("telegram_enabled") and prefs.get("telegram_chat_id"):
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if token:
            try:
                r = await send_telegram(token, str(prefs["telegram_chat_id"]), text)
                results["telegram"] = {"ok": r.get("ok", False), "error": r.get("description")}
            except Exception as e:
                results["telegram"] = {"ok": False, "error": str(e)}
    return results
