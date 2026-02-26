import pandas as pd
import numpy as np

def clean_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def calcular_indicadores(df):
    df["SMA20"]  = df["Close"].rolling(20).mean()
    df["SMA50"]  = df["Close"].rolling(50).mean()
    df["SMA100"] = df["Close"].rolling(100).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    df["TR"] = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = df["TR"].rolling(7).mean()
    hl2 = (df["High"] + df["Low"]) / 2
    df["UB"] = hl2 + 3.0 * df["ATR"]
    df["LB"] = hl2 - 3.0 * df["ATR"]
    df["ST"]  = np.nan
    df["Dir"] = 0

    for i in range(1, len(df)):
        ps  = df.iloc[i-1]["ST"]
        pd_ = df.iloc[i-1]["Dir"]
        cl  = float(df.iloc[i]["LB"])
        cu  = float(df.iloc[i]["UB"])
        cc  = float(df.iloc[i]["Close"])
        if np.isnan(ps):
            df.iloc[i, df.columns.get_loc("ST")]  = cl
            df.iloc[i, df.columns.get_loc("Dir")] = 1
            continue
        st = max(cl, ps) if pd_ == 1 else min(cu, ps)
        df.iloc[i, df.columns.get_loc("ST")]  = st
        df.iloc[i, df.columns.get_loc("Dir")] = 1 if cc > st else -1

    return df

def detectar_alertas(df, ticker=""):
    alertas = []
    n = len(df) - 1
    if n < 2:
        return alertas

    precio_now  = float(df["Close"].iloc[n])
    precio_prev = float(df["Close"].iloc[n - 1])
    
    ticker_clean = str(ticker).upper().strip()
    prefix = f"[{ticker_clean}] " if ticker_clean else ""

    smas = {
        "SMA20":  (df["SMA20"].iloc[n],  df["SMA20"].iloc[n-1]),
        "SMA50":  (df["SMA50"].iloc[n],  df["SMA50"].iloc[n-1]),
        "SMA100": (df["SMA100"].iloc[n], df["SMA100"].iloc[n-1]),
        "SMA200": (df["SMA200"].iloc[n], df["SMA200"].iloc[n-1]),
    }

    for nombre, (sma_now, sma_prev) in smas.items():
        if not (pd.notna(sma_now) and pd.notna(sma_prev)):
            continue
        if precio_prev < sma_prev and precio_now >= sma_now:
            alertas.append({"nivel": "bullish",
                "msg": prefix + "Precio cruza " + nombre + " al alza $" + str(round(precio_now, 2))})
        elif precio_prev > sma_prev and precio_now <= sma_now:
            alertas.append({"nivel": "bearish",
                "msg": prefix + "Precio cruza " + nombre + " a la baja $" + str(round(precio_now, 2))})
        elif sma_now > 0 and abs(precio_now - sma_now) / sma_now * 100 <= 0.4:
            alertas.append({"nivel": "info",
                "msg": prefix + "Precio tocando " + nombre + " $" + str(round(precio_now, 2))})

    s100_n, s100_p = smas["SMA100"]
    s200_n, s200_p = smas["SMA200"]
    if pd.notna(s100_n) and pd.notna(s200_n) and pd.notna(s100_p) and pd.notna(s200_p):
        if s100_p < s200_p and s100_n >= s200_n:
            alertas.append({"nivel": "bullish", "msg": prefix + "Golden Cross SMA100/200"})
        elif s100_p > s200_p and s100_n <= s200_n:
            alertas.append({"nivel": "bearish", "msg": prefix + "Death Cross SMA100/200"})

    s20_n, s20_p = smas["SMA20"]
    s50_n, s50_p = smas["SMA50"]
    if pd.notna(s20_n) and pd.notna(s50_n) and pd.notna(s20_p) and pd.notna(s50_p):
        if s20_p < s50_p and s20_n >= s50_n:
            alertas.append({"nivel": "bullish", "msg": prefix + "SMA20 cruza sobre SMA50"})
        elif s20_p > s50_p and s20_n <= s50_n:
            alertas.append({"nivel": "bearish", "msg": prefix + "SMA20 cruza bajo SMA50"})

    return alertas
