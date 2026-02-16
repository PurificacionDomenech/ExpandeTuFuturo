import numpy as np
import pandas as pd

# ============================================================
# MEDIAS MÓVILES
# ============================================================

def calcular_medias(data):
    data["MA20"] = data["Close"].rolling(window=20).mean()
    data["MA50"] = data["Close"].rolling(window=50).mean()
    data["MA100"] = data["Close"].rolling(window=100).mean()
    data["MA200"] = data["Close"].rolling(window=200).mean()
    return data


# ============================================================
# SUPERTREND
# ============================================================

def calcular_supertrend(data, atr_period=7, factor=3.0):

    data["H-L"] = data["High"] - data["Low"]
    data["H-PC"] = abs(data["High"] - data["Close"].shift(1))
    data["L-PC"] = abs(data["Low"] - data["Close"].shift(1))
    data["TR"] = data[["H-L", "H-PC", "L-PC"]].max(axis=1)

    data["ATR"] = data["TR"].rolling(window=atr_period).mean()

    hl2 = (data["High"] + data["Low"]) / 2
    data["UpperBand"] = hl2 + factor * data["ATR"]
    data["LowerBand"] = hl2 - factor * data["ATR"]

    data["Supertrend"] = np.nan
    data["Direction"] = 0

    for i in range(1, len(data)):
        prev_supertrend = data.loc[data.index[i-1], "Supertrend"]
        prev_direction = data.loc[data.index[i-1], "Direction"]

        if np.isnan(prev_supertrend):
            data.loc[data.index[i], "Supertrend"] = data.loc[data.index[i], "LowerBand"]
            data.loc[data.index[i], "Direction"] = 1
            continue

        if prev_direction == 1:
            curr_supertrend = max(data.loc[data.index[i], "LowerBand"], prev_supertrend)
            curr_direction = 1 if data.loc[data.index[i], "Close"] > curr_supertrend else -1
        else:
            curr_supertrend = min(data.loc[data.index[i], "UpperBand"], prev_supertrend)
            curr_direction = 1 if data.loc[data.index[i], "Close"] > curr_supertrend else -1

        data.loc[data.index[i], "Supertrend"] = curr_supertrend
        data.loc[data.index[i], "Direction"] = curr_direction

    return data