# Expande Tu Futuro Web 🚀

Aplicación web de trading minimalista y profesional diseñada para el análisis técnico rápido.

## Características
- **Gráficas de Área Modernas**: Visualización fucsia/rosa vibrante con degradados.
- **Indicadores Técnicos**: 
  - Supertrend (7, 3.0)
  - Medias Móviles (SMA 20, SMA 50)
  - RSI (14 periodos)
- **Buscador Inteligente**: Acceso a miles de activos (Cripto, ETFs, Acciones) vía yfinance.
- **Responsive**: Diseño optimizado para escritorio y móviles.

## Tecnologías
- **Backend**: Python + FastAPI
- **Frontend**: HTML5 + Tailwind CSS + Plotly.js
- **Datos**: yfinance API

## Despliegue en Railway
1. Conecta tu repositorio de GitHub a Railway.
2. Railway detectará automáticamente el `requirements.txt` y el `Procfile`.
3. La aplicación se ejecutará en el puerto asignado por Railway (variable `PORT`).
4. ¡Listo! Tu app estará en línea.

## Cambios Recientes
- Compatibilidad con yfinance v1.2.0+
- Corrección del cálculo de indicadores técnicos
- Optimización de la visualización de gráficas
- Configuración mejorada para Railway

## Ejecución Local
```bash
pip install -r requirements.txt
python main.py
```
Accede a `http://localhost:8000`
