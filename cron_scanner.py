import asyncio
import yfinance as yf
import gc
import os
import time
from notifications import dispatch_notifications, get_db

async def scan_and_notify():
    from main import calcular_indicadores, detectar_alertas, clean_df
    print(f"[{time.ctime()}] Iniciando escaneo automático de vigilancia...")
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT user_id, telegram_enabled, telegram_chat_id, email_enabled, email_address, watchlist FROM notification_prefs WHERE is_vip = TRUE")
                users = cur.fetchall()
        
        for user in users:
            uid, tg_en, tg_id, em_en, em_adr, wl_str = user
            if not wl_str: continue
            if not tg_en and not em_en: continue
            
            tickers = [t.strip() for t in wl_str.split(",") if t.strip()]
            all_alerts = []
            
            for t in tickers:
                try:
                    df_full = yf.download(t.upper(), period="5d", interval="1d", progress=False)
                    if not df_full.empty:
                        df_full = clean_df(df_full)
                        df_full = calcular_indicadores(df_full)
                        alerts = detectar_alertas(df_full, ticker=t.upper())
                        all_alerts.extend(alerts)
                    del df_full
                except Exception as e:
                    print(f"Error escaneando {t}: {e}")
                await asyncio.sleep(0.5)
            
            gc.collect()
            
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
    print("Scanner background loop active")
    await asyncio.sleep(60)
    while True:
        try:
            await scan_and_notify()
        except Exception as e:
            print(f"Loop error: {e}")
        await asyncio.sleep(1800)

if __name__ == "__main__":
    asyncio.run(main())
