import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
from main import calcular_indicadores, detectar_alertas, clean_df
from notifications import dispatch_notifications, get_db

async def scan_and_notify():
    print(f"[{time.ctime()}] Iniciando escaneo automático de vigilancia...")
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                # Obtener todos los usuarios VIP que tienen algo en su watchlist
                cur.execute("SELECT user_id, telegram_enabled, telegram_chat_id, email_enabled, email_address, watchlist FROM notification_prefs WHERE is_vip = TRUE")
                users = cur.fetchall()
        
        for user in users:
            uid, tg_en, tg_id, em_en, em_adr, wl_str = user
            if not wl_str: continue
            
            tickers = [t.strip() for t in wl_str.split(",") if t.strip()]
            all_alerts = []
            
            for t in tickers:
                try:
                    # Usar un periodo más corto para evitar timeouts y asegurar datos frescos
                    df = yf.download(t.upper(), period="5d", interval="1h", progress=False)
                    if not df.empty:
                        df = clean_df(df)
                        # Para calcular indicadores de largo plazo (SMA200) necesitamos más datos
                        df_full = yf.download(t.upper(), period="1y", interval="1d", progress=False)
                        if not df_full.empty:
                            df_full = clean_df(df_full)
                            df_full = calcular_indicadores(df_full)
                            alerts = detectar_alertas(df_full, ticker=t.upper())
                            all_alerts.extend(alerts)
                except Exception as e:
                    print(f"Error escaneando {t}: {e}")
            
            if all_alerts:
                prefs = {
                    "telegram_enabled": tg_en,
                    "telegram_chat_id": tg_id,
                    "email_enabled": em_en,
                    "email_address": em_adr
                }
                await dispatch_notifications(prefs, all_alerts)
                print(f"Enviadas {len(all_alerts)} alertas al usuario {uid}")
                
    except Exception as e:
        print(f"Error en scan_and_notify: {e}")

async def main():
    while True:
        await scan_and_notify()
        # Esperar 1 hora entre escaneos (3600 segundos)
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
