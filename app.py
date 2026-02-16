import tkinter as tk
from tkinter import ttk
from graficos.chart_tv import mostrar_grafico

# ============================================================
# PANEL ELEGANTE TIPO TRADINGVIEW
# ============================================================

def seleccionar_ticker():
    root = tk.Tk()
    root.title("Selector de Índices — Estilo TradingView")
    root.geometry("420x350")
    root.configure(bg="#0d1117")

    titulo = tk.Label(
        root,
        text="Selecciona un índice",
        font=("Segoe UI", 18, "bold"),
        fg="#e0e0e0",
        bg="#0d1117"
    )
    titulo.pack(pady=20)

    opciones = {
        "Nasdaq 100 (^NDX)": "^NDX",
        "S&P 500 (^GSPC)": "^GSPC",
        "Dow Jones (^DJI)": "^DJI",
        "Russell 2000 (^RUT)": "^RUT",
        "Apple (AAPL)": "AAPL",
        "IBEX 35 (^IBEX)": "^IBEX"
    }

    seleccion = tk.StringVar()
    seleccion.set("S&P 500 (^GSPC)")

    combo = ttk.Combobox(
        root,
        textvariable=seleccion,
        values=list(opciones.keys()),
        font=("Segoe UI", 12),
        state="readonly",
        width=30
    )
    combo.pack(pady=20)

    def confirmar():
        ticker = opciones[seleccion.get()]
        root.destroy()
        mostrar_grafico(ticker)

    boton = tk.Button(
        root,
        text="Cargar gráfico",
        font=("Segoe UI", 14, "bold"),
        bg="#238636",
        fg="white",
        activebackground="#2ea043",
        activeforeground="white",
        relief="flat",
        padx=20,
        pady=10,
        command=confirmar
    )
    boton.pack(pady=30)

    root.mainloop()


if __name__ == "__main__":
    seleccionar_ticker()