import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from indicadores.etf import calcular_medias, calcular_supertrend

# ============================================================
# FUNCIÓN PRINCIPAL DEL GRÁFICO
# ============================================================

def mostrar_grafico(ticker):

    # DESCARGA DE DATOS
    data = yf.download(ticker, period="3y", interval="1d")
    data = data.copy()
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

    # INDICADORES
    data = calcular_medias(data)
    data = calcular_supertrend(data)

    # GRÁFICO
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a",
        decreasing_fillcolor="#ef5350",
        name="Precio"
    ))

    # MEDIAS
    fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], mode="lines",
                             name="MA20", line=dict(color="#4dd0e1", width=1.8)))

    fig.add_trace(go.Scatter(x=data.index, y=data["MA50"], mode="lines",
                             name="MA50", line=dict(color="#ffb74d", width=1.8)))

    fig.add_trace(go.Scatter(x=data.index, y=data["MA100"], mode="lines",
                             name="MA100", line=dict(color="#fff176", width=1.8)))

    fig.add_trace(go.Scatter(x=data.index, y=data["MA200"], mode="lines",
                             name="MA200", line=dict(color="#ce93d8", width=2)))

    # SUPERTREND
    fig.add_trace(go.Scatter(
        x=data.index,
        y=np.where(data["Direction"] == 1, data["Supertrend"], np.nan),
        mode="lines",
        name="Supertrend UP",
        line=dict(color="#00e676", width=2, dash="dot")
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=np.where(data["Direction"] == -1, data["Supertrend"], np.nan),
        mode="lines",
        name="Supertrend DOWN",
        line=dict(color="#ff1744", width=2, dash="dot")
    ))

    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=False),
            showgrid=False
        ),
        yaxis=dict(showgrid=False),
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font=dict(color="#e0e0e0"),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.5,
                y=1.12,
                buttons=list([
                    dict(label="1M", method="relayout", args=[{"xaxis.range": [data.index[-30], data.index[-1]]}]),
                    dict(label="3M", method="relayout", args=[{"xaxis.range": [data.index[-90], data.index[-1]]}]),
                    dict(label="6M", method="relayout", args=[{"xaxis.range": [data.index[-180], data.index[-1]]}]),
                    dict(label="1Y", method="relayout", args=[{"xaxis.range": [data.index[-365], data.index[-1]]}]),
                    dict(label="ALL", method="relayout", args=[{"xaxis.autorange": True}])
                ])
            )
        ],
        title=f"Gráfico estilo TradingView — {ticker}"
    )

    fig.show()