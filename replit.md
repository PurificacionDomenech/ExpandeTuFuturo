# Expande Tu Futuro — LC Market Scanner

Aplicación web de escaneo de mercados financieros construida con FastAPI y Python.

## Descripción

Esta app proporciona análisis técnico para acciones, ETFs y criptoactivos, incluyendo:
- Gráficos de velas con indicadores SMA 20/50/100/200
- RSI (Índice de Fuerza Relativa) con señales de zonas extremas
- Indicador SuperTrend
- Alertas automáticas por cruces de SMAs y toques de precio
- Panel de vigilancia (watchlist)
- Página de inicio (splash) con autenticación vía Supabase
- Sistema de notificaciones multicanal (Telegram, Email)
- Sistema VIP/Free con códigos de acceso reutilizables

## Arquitectura

- **Backend**: FastAPI (Python) servido con Uvicorn (dev) / Gunicorn (prod)
- **Frontend**: Plantillas HTML/CSS/JS vanilla (FileResponse)
- **Datos**: yfinance para datos de mercado en tiempo real
- **Auth**: Supabase (del lado del cliente, en splash.html)
- **Base de datos**: PostgreSQL (notification_prefs, access_codes)
- **Puerto**: 5000
- **Scanner**: cron_scanner.py ejecuta escaneos automáticos cada 30 min

## Archivos principales

- `main.py` — App FastAPI con rutas API, indicadores técnicos y lógica de gráficos
- `cron_scanner.py` — Scanner automático de watchlists (cada 30 min para VIP)
- `notifications.py` — Sistema de notificaciones (Telegram, Email) con branding "ETF · Expande Tu Futuro"
- `templates/splash.html` — Página de inicio / login
- `templates/index.html` — Dashboard principal con toda la UI
- `static/` — Recursos estáticos (logo, etc.)

## Sistema VIP

- Códigos VIP permanentes: VIP001-VIP005, VIP333, VIP777
- Códigos VIP prueba (3 meses): VIP006-VIP044
- Usuarios Free: máximo 3 activos en vigilancia, sin alertas
- Usuarios VIP/PRO: activos ilimitados, alertas Telegram/Email
- Precios PRO: 17€/mes, 57€ semestral (9,50€/mes, 44% dto.), 87€ anual (7,25€/mes, 57% dto.)
- Modal Planes PRO con campo unico para codigo VIP y tabla comparativa
- Los controles de notificaciones (Telegram/Email) están activos para todos los usuarios
- Las llamadas API usan URLs absolutas (window.location.origin) para compatibilidad con despliegue

## Endpoints de la API

- `GET /health` — Health check (para despliegue)
- `GET /` — Página de inicio (splash)
- `GET /app` — Dashboard principal
- `GET /api/chart/{ticker}?interval=1d` — Datos del gráfico (velas, SMAs, RSI, SuperTrend, alertas)
- `GET /api/row/{ticker}` — Datos resumidos por activo para la tabla
- `GET /api/watch?tickers=X,Y,Z` — Alertas de la watchlist
- `GET /api/sparkline/{ticker}` — Datos de sparkline (cierres de 1 mes)
- `GET /api/notifications/prefs?user_id=X` — Preferencias de notificaciones
- `POST /api/notifications/prefs?user_id=X` — Guardar preferencias
- `POST /api/notifications/redeem?user_id=X` — Canjear código VIP
- `POST /api/notifications/send?user_id=X` — Enviar alertas manuales
- `GET /api/notifications/status` — Estado del sistema de alertas

## Workflow

- **Start application**: `uvicorn main:app --host 0.0.0.0 --port 5000`

## Despliegue

Configurado para despliegue autoscale con gunicorn:
- `gunicorn --bind=0.0.0.0:5000 --timeout=120 --workers=2 --worker-class=uvicorn.workers.UvicornWorker main:app`
- Health check en `/health`

## Secretos configurados

- TELEGRAM_BOT_TOKEN
- SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS
