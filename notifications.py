import os
import json
import asyncio
import httpx
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

PREFS_FILE = Path("data/notification_prefs.json")
PREFS_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_prefs():
    if PREFS_FILE.exists():
        return json.loads(PREFS_FILE.read_text())
    return {}


def save_prefs(prefs: dict):
    PREFS_FILE.write_text(json.dumps(prefs, indent=2))


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


async def send_whatsapp(account_sid: str, auth_token: str, from_number: str, to_number: str, message: str) -> dict:
    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    data = {
        "From": f"whatsapp:{from_number}",
        "To": f"whatsapp:{to_number}",
        "Body": message,
    }
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(url, data=data, auth=(account_sid, auth_token))
        return r.json()


async def send_email(smtp_host: str, smtp_port: int, smtp_user: str, smtp_pass: str, from_addr: str, to_addr: str, subject: str, body: str) -> dict:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr

    html_body = f"""
    <div style="font-family:'DM Sans',Arial,sans-serif;background:#060810;color:#e8e4d9;padding:30px;border-radius:12px;">
      <div style="border-bottom:1px solid rgba(201,168,76,0.3);padding-bottom:15px;margin-bottom:20px;">
        <h2 style="color:#e8c96d;margin:0;font-size:18px;">⚡ ETF · Expande Tu Futuro</h2>
        <p style="color:rgba(232,228,217,0.5);font-size:12px;margin:5px 0 0;">LC Market Scanner · Alertas</p>
      </div>
      <div style="font-size:14px;line-height:1.8;">
        {body}
      </div>
      <div style="margin-top:25px;padding-top:15px;border-top:1px solid rgba(201,168,76,0.15);font-size:10px;color:rgba(232,228,217,0.3);">
        Enviado automáticamente por ETF Market Scanner
      </div>
    </div>
    """
    msg.attach(MIMEText(body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, _send_smtp, smtp_host, smtp_port, smtp_user, smtp_pass, from_addr, to_addr, msg)
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
    lines = ["⚡ ETF · Alertas de Mercado\n"]
    for a in alertas:
        icon = icon_map.get(a.get("nivel", "info"), "🟡")
        lines.append(f"{icon} {a['msg']}")
    return "\n".join(lines)


def format_alerts_html(alertas: list) -> str:
    if not alertas:
        return "<p>Sin alertas activas.</p>"
    color_map = {"bullish": "#2dd47e", "bearish": "#f05858", "info": "#e8c96d"}
    icon_map = {"bullish": "🟢", "bearish": "🔴", "info": "🟡"}
    parts = []
    for a in alertas:
        c = color_map.get(a.get("nivel", "info"), "#e8c96d")
        icon = icon_map.get(a.get("nivel", "info"), "🟡")
        parts.append(f'<div style="padding:8px 12px;margin:4px 0;border-radius:6px;background:rgba(255,255,255,0.03);border-left:3px solid {c};font-size:13px;">{icon} {a["msg"]}</div>')
    return "\n".join(parts)


async def dispatch_notifications(prefs: dict, alertas: list):
    results = {}
    text = format_alerts_text(alertas)
    html = format_alerts_html(alertas)

    if prefs.get("telegram_enabled") and prefs.get("telegram_chat_id"):
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if token:
            try:
                tg_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                r = await send_telegram(token, prefs["telegram_chat_id"], tg_text)
                results["telegram"] = {"ok": r.get("ok", False)}
            except Exception as e:
                results["telegram"] = {"ok": False, "error": str(e)}
        else:
            results["telegram"] = {"ok": False, "error": "TELEGRAM_BOT_TOKEN no configurado"}

    if prefs.get("whatsapp_enabled") and prefs.get("whatsapp_number"):
        sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
        token = os.environ.get("TWILIO_AUTH_TOKEN", "")
        from_num = os.environ.get("TWILIO_WHATSAPP_FROM", "")
        if sid and token and from_num:
            try:
                r = await send_whatsapp(sid, token, from_num, prefs["whatsapp_number"], text)
                results["whatsapp"] = {"ok": "sid" in r}
            except Exception as e:
                results["whatsapp"] = {"ok": False, "error": str(e)}
        else:
            results["whatsapp"] = {"ok": False, "error": "Credenciales Twilio no configuradas"}

    if prefs.get("email_enabled") and prefs.get("email_address"):
        smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        smtp_user = os.environ.get("SMTP_USER", "")
        smtp_pass = os.environ.get("SMTP_PASS", "")
        smtp_from = os.environ.get("SMTP_FROM", smtp_user)
        if smtp_user and smtp_pass:
            try:
                r = await send_email(smtp_host, smtp_port, smtp_user, smtp_pass, smtp_from, prefs["email_address"], "⚡ ETF Alertas de Mercado", html)
                results["email"] = r
            except Exception as e:
                results["email"] = {"ok": False, "error": str(e)}
        else:
            results["email"] = {"ok": False, "error": "Credenciales SMTP no configuradas"}

    return results
