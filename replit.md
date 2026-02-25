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

## Arquitectura

- **Backend**: FastAPI (Python) servido con Uvicorn
- **Frontend**: Plantillas HTML/CSS/JS vanilla (Jinja2 vía FileResponse)
- **Datos**: yfinance para datos de mercado en tiempo real
- **Auth**: Supabase (del lado del cliente, en splash.html)
- **Puerto**: 5000

## Archivos principales

- `main.py` — App FastAPI con todas las rutas API y lógica de datos de gráficos
- `app.py` — Interfaz de escritorio legacy con tkinter (no se usa en la versión web)
- `templates/splash.html` — Página de inicio / login
- `templates/index.html` — Dashboard principal
- `static/` — Recursos estáticos (logo, etc.)
- `graficos/chart_tv.py` — Módulo de gráficos legacy (para tkinter)
- `indicadores/etf.py` — Helpers de indicadores ETF

## Endpoints de la API

- `GET /` — Página de inicio (splash)
- `GET /app` — Dashboard principal
- `GET /api/chart/{ticker}?interval=1d` — Datos del gráfico (velas, SMAs, RSI, SuperTrend, alertas)
- `GET /api/row/{ticker}` — Datos resumidos por activo para la tabla
- `GET /api/watch?tickers=X,Y,Z` — Alertas de la watchlist
- `GET /api/sparkline/{ticker}` — Datos de sparkline (cierres de 1 mes)

## Workflow

- **Start application**: `uvicorn main:app --host 0.0.0.0 --port 5000`

## Despliegue

Configurado para despliegue autoscale usando gunicorn con workers de uvicorn.
