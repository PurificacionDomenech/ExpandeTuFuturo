En main.py, cambia el INTERVAL_MAP así:

ANTES:
INTERVAL_MAP = {
    "1d":  ("1d",  "2y"),
    "1wk": ("1wk", "max"),
    "1mo": ("1mo", "max"),
    "3mo": ("3mo", "max"),
}

DESPUÉS:
INTERVAL_MAP = {
    "1d":  ("1d",  "max"),   # <-- cambia "2y" a "max"
    "1wk": ("1wk", "max"),
    "1mo": ("1mo", "max"),
    "3mo": ("3mo", "max"),
}

Eso es todo. El HTML ya no envía el parámetro de período,
siempre llama a /api/chart/{ticker}?interval=1d
y el backend descargará el máximo histórico disponible con velas diarias.
