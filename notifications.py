import os
import asyncio
import httpx
import smtplib
import psycopg2
import psycopg2.extras
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def get_db():
    # Usar un pool o asegurar cierre, pero aquí forzamos sslmode si es necesario o manejamos el error
    return psycopg2.connect(os.environ["DATABASE_URL"])

def load_prefs(user_id: str = "default") -> dict:
    conn = None
    try:
        conn = get_db()
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
        print(f"DEBUG: load_prefs error for {user_id}: {e}")
    finally:
        if conn: conn.close()
    return {"is_vip": False}

def save_prefs(prefs: dict, user_id: str = "default"):
    conn = None
    try:
        conn = get_db()
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
        if conn: conn.rollback()
        print(f"DEBUG: save_prefs error for {user_id}: {e}")
        raise RuntimeError(f"DB save error: {e}")
    finally:
        if conn: conn.close()


def list_all_users_with_notifications() -> list:
    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM notification_prefs
                    WHERE telegram_enabled = TRUE
                       OR whatsapp_enabled = TRUE
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


async def send_whatsapp(account_sid: str, auth_token: str, from_number: str,
                        to_number: str, message: str) -> dict:
    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    data = {
        "From": f"whatsapp:{from_number}",
        "To": f"whatsapp:{to_number}",
        "Body": message,
    }
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(url, data=data, auth=(account_sid, auth_token))
        return r.json()


async def send_email(smtp_host: str, smtp_port: int, smtp_user: str,
                     smtp_pass: str, from_addr: str, to_addr: str,
                     subject: str, body_html: str) -> dict:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr

    plain = body_html.replace("<br>", "\n").replace("</div>", "\n")
    import re
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
    # Usar Madrid/Europe para la hora local
    try:
        import pytz
        tz = pytz.timezone('Europe/Madrid')
        now = datetime.now(tz)
    except Exception:
        now = datetime.now()
    now_str = now.strftime("%H:%M")
    lines = [f"🔱 ETF · Expande Tu Futuro · {now_str}\n"]
    for a in alertas:
        icon = icon_map.get(a.get("nivel", "info"), "🟡")
        lines.append(f"{icon} {a['msg']}")
    return "\n".join(lines)


def format_alerts_html(alertas: list) -> str:
    if not alertas:
        return "<p>Sin alertas activas.</p>"
    color_map = {"bullish": "#2dd47e", "bearish": "#f05858", "info": "#e8c96d"}
    icon_map  = {"bullish": "🟢",      "bearish": "🔴",       "info": "🟡"}
    from datetime import datetime
    try:
        import pytz
        tz = pytz.timezone('Europe/Madrid')
        now = datetime.now(tz)
    except Exception:
        now = datetime.now()
    now_str = now.strftime("%H:%M")
    parts = [f"""
    <div style="font-family:'DM Sans',Arial,sans-serif;background:#060810;
                color:#e8e4d9;padding:30px;border-radius:12px;max-width:600px">
      <div style="border-bottom:1px solid rgba(201,168,76,.3);padding-bottom:15px;margin-bottom:20px">
        <h2 style="color:#e8c96d;margin:0;font-size:18px">🔱 ETF · Expande Tu Futuro</h2>
        <p style="color:rgba(232,228,217,.5);font-size:12px;margin:4px 0 0">
          Alertas de mercado · {now_str}</p>
      </div>
    """]
    for a in alertas:
        c    = color_map.get(a.get("nivel", "info"), "#e8c96d")
        icon = icon_map.get(a.get("nivel", "info"), "🟡")
        parts.append(
            f'<div style="padding:10px 14px;margin:6px 0;border-radius:8px;'
            f'background:rgba(255,255,255,.03);border-left:3px solid {c};font-size:13px">'
            f'{icon} {a["msg"]}</div>'
        )
    parts.append("""
      <div style="margin-top:25px;padding-top:15px;border-top:1px solid rgba(201,168,76,.15);
                  font-size:10px;color:rgba(232,228,217,.3)">
        Enviado por ETF · Expande Tu Futuro
      </div>
    </div>""")
    return "\n".join(parts)


async def dispatch_notifications(prefs: dict, alertas: list) -> dict:
    results = {}
    text = format_alerts_text(alertas)
    html = format_alerts_html(alertas)

    if prefs.get("telegram_enabled") and prefs.get("telegram_chat_id"):
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if token:
            try:
                r = await send_telegram(token, str(prefs["telegram_chat_id"]), text)
                results["telegram"] = {"ok": r.get("ok", False),
                                       "error": r.get("description")}
            except Exception as e:
                results["telegram"] = {"ok": False, "error": str(e)}
        else:
            results["telegram"] = {"ok": False, "error": "TELEGRAM_BOT_TOKEN no configurado"}

    if prefs.get("email_enabled") and prefs.get("email_address"):
        smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        smtp_user = os.environ.get("SMTP_USER", "")
        smtp_pass = os.environ.get("SMTP_PASS", "")
        smtp_from = os.environ.get("SMTP_FROM", smtp_user)
        if smtp_user and smtp_pass:
            try:
                r = await send_email(smtp_host, smtp_port, smtp_user, smtp_pass,
                                     smtp_from, prefs["email_address"],
                                     "⚡ ETF · Alertas de Mercado", html)
                results["email"] = r
            except Exception as e:
                results["email"] = {"ok": False, "error": str(e)}
        else:
            results["email"] = {"ok": False,
                                 "error": "Credenciales SMTP no configuradas en servidor"}

    return results
