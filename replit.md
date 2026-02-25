# Expande Tu Futuro — LC Market Scanner

A financial market scanner web application built with FastAPI and Python.

## Overview

This app provides technical analysis for stocks, ETFs, and crypto assets including:
- Candlestick charts with SMA 20/50/100/200 indicators
- RSI (Relative Strength Index) with extreme signal markers
- SuperTrend indicator
- Automatic alerts for SMA crossovers and price touches
- Watchlist / watchdog panel
- Splash/landing page with Supabase authentication

## Architecture

- **Backend**: FastAPI (Python) served via Uvicorn
- **Frontend**: Vanilla HTML/CSS/JS templates (Jinja2 via FileResponse)
- **Data**: yfinance for real-time market data
- **Auth**: Supabase (client-side, in splash.html)
- **Port**: 5000

## Key Files

- `main.py` — FastAPI app with all API routes and chart data logic
- `app.py` — Legacy tkinter desktop UI (not used in web version)
- `templates/splash.html` — Landing/login page
- `templates/index.html` — Main dashboard
- `static/` — Static assets (logo, etc.)
- `graficos/chart_tv.py` — Legacy chart module (for tkinter)
- `indicadores/etf.py` — ETF indicator helpers

## API Endpoints

- `GET /` — Splash/landing page
- `GET /app` — Main dashboard
- `GET /api/chart/{ticker}?interval=1d` — Chart data (candles, SMAs, RSI, SuperTrend, alerts)
- `GET /api/row/{ticker}` — Summary row data for asset table
- `GET /api/watch?tickers=X,Y,Z` — Watchlist alerts
- `GET /api/sparkline/{ticker}` — Sparkline data (1 month closes)

## Workflow

- **Start application**: `uvicorn main:app --host 0.0.0.0 --port 5000`

## Deployment

Configured for autoscale deployment using gunicorn with uvicorn workers.
