import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.utils
import json

# Descargar datos
df = yf.download('BTC-USD', period='1mo', interval='1d')
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

print("Columnas disponibles:", df.columns.tolist())
print("Primeras filas:")
print(df.head())

# Crear figura con velas
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="Precio"
))

# Convertir a JSON
chart_json = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))

# Verificar estructura
print("\nEstructura de datos en JSON:")
print("Número de trazos:", len(chart_json.get('data', [])))
if chart_json.get('data'):
    print("Tipo de primer trazo:", chart_json['data'][0].get('type'))
    print("Claves del primer trazo:", list(chart_json['data'][0].keys())[:10])
