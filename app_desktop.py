"""
Native Windows Desktop App for Global Multi-CFD Momentum Signal
Nikkei 225, S&P 500, Nasdaq 100, Russell 2000, 10Y Treasury (100% CFD)
"""

import threading
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import requests

ASSETS = {
    "S&P 500 (SPX)": "^GSPC",
    "Nasdaq 100 (NDX)": "^NDX",
    "Russell 2000 (RUT)": "^RUT",
    "Nikkei 225 (NI225)": "^N225",
    "10Y Treasury (TNX)": "^TNX"
}

def fetch_data_and_signal(symbol: str, timeframe="5m"):
    range_str = "5d" if timeframe == "5m" else ("1mo" if timeframe == "1h" else "3mo")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={timeframe}&range={range_str}"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=10)
    data = resp.json()["chart"]["result"][0]
    
    timestamps = data["timestamp"]
    quote = data["indicators"]["quote"][0]
    
    df = pd.DataFrame({
        "Open": quote["open"],
        "High": quote["high"],
        "Low": quote["low"],
        "Close": quote["close"],
        "Volume": quote.get("volume", [0]*len(timestamps)),
    }, index=pd.to_datetime(timestamps, unit="s", utc=True))
    
    df = df.dropna(subset=["Close"])
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    volume = df["Volume"].fillna(0).values
    
    total_vol = volume.sum()
    if total_vol > 0:
        typical_price = (high + low + close) / 3.0
        vwap = (typical_price * volume).sum() / (total_vol + 1e-9)
    else:
        vwap = float(np.mean(close[-20:]))

    tr = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
    atr_14 = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.std(close))
    vwap_z = (close[-1] - vwap) / (atr_14 + 1e-9)
    
    ret_3 = (close[-1] / close[-4] - 1) * 100 if len(close) > 4 else 0.0
    ret_6 = (close[-1] / close[-7] - 1) * 100 if len(close) > 7 else 0.0
    
    raw_score = 0.35 * ret_3 + 0.35 * ret_6 + 0.30 * (vwap_z * 0.4)
    prob_up = float(1.0 / (1.0 + np.exp(- (0.19 + 0.35 * raw_score))))
    conviction = int(np.clip(raw_score * 35, -100, 100))
    
    if prob_up > 0.54 and vwap_z > 0.15:
        action = "BUY / LONG"
        color = "#00e676"
        bg_card = "#0b2e1b"
        desc = "Bullish Momentum: CFD price confirmed above Session VWAP."
    elif prob_up < 0.46 and vwap_z < -0.15:
        action = "SELL / SHORT"
        color = "#ff5252"
        bg_card = "#380d12"
        desc = "Bearish Momentum: CFD price confirmed below Session VWAP."
    else:
        action = "HOLD CASH"
        color = "#ffd740"
        bg_card = "#2e240b"
        desc = "Neutral / Chop: CFD price fluctuating near VWAP."
        
    dec = 3 if "^TNX" in symbol else 2
    ind_price = float(np.round(close[-1] * (1.0 + (prob_up - 0.50) * 0.005), dec))
    ind_delta_pct = float(np.round(((ind_price / (close[-1] + 1e-9)) - 1.0) * 100, 2))

    return {
        "action": action,
        "color": color,
        "bg_card": bg_card,
        "desc": desc,
        "price": close[-1],
        "indicative_price": ind_price,
        "indicative_delta_pct": ind_delta_pct,
        "vwap": vwap,
        "vwap_z": vwap_z,
        "prob_up": prob_up,
        "conviction": conviction,
        "ret_3": ret_3,
        "time": df.index[-1].strftime("%Y-%m-%d %H:%M UTC")
    }

class MultiIndexApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Global Multi-CFD Momentum Radar")
        self.geometry("500x600")
        self.configure(bg="#121214")
        self.resizable(False, False)
        
        self.build_ui()
        self.refresh_signal()

    def build_ui(self):
        lbl_title = tk.Label(self, text="Global Multi-CFD Momentum Radar", font=("Helvetica", 15, "bold"), fg="#ffffff", bg="#121214")
        lbl_title.pack(pady=(16, 2))
        lbl_sub = tk.Label(self, text="Nikkei 225 • S&P 500 • Nasdaq 100 • Russell 2000 • 10Y Treasury", font=("Helvetica", 8), fg="#8e8e93", bg="#121214")
        lbl_sub.pack(pady=(0, 12))

        # Controls Row
        frame_top = tk.Frame(self, bg="#121214")
        frame_top.pack(fill="x", padx=24, pady=4)
        
        tk.Label(frame_top, text="Asset:", font=("Helvetica", 10), fg="#cccccc", bg="#121214").pack(side="left")
        self.asset_var = tk.StringVar(value="S&P 500 (SPX)")
        cb_asset = ttk.Combobox(frame_top, textvariable=self.asset_var, values=list(ASSETS.keys()), state="readonly", width=18)
        cb_asset.pack(side="left", padx=6)
        cb_asset.bind("<<ComboboxSelected>>", lambda e: self.refresh_signal())

        self.tf_var = tk.StringVar(value="5m")
        cb_tf = ttk.Combobox(frame_top, textvariable=self.tf_var, values=["5m", "1h", "1d"], state="readonly", width=5)
        cb_tf.pack(side="left", padx=4)
        cb_tf.bind("<<ComboboxSelected>>", lambda e: self.refresh_signal())
        
        self.btn_refresh = tk.Button(frame_top, text="⚡ Refresh", font=("Helvetica", 9, "bold"), bg="#2c2c2e", fg="#ffffff", activebackground="#3a3a3c", relief="flat", command=self.refresh_signal, cursor="hand2")
        self.btn_refresh.pack(side="right")

        # Signal Card
        self.card = tk.Frame(self, bg="#1c1c1e", bd=2, relief="solid")
        self.card.pack(fill="both", padx=24, pady=16, ipady=12)

        self.lbl_sig_header = tk.Label(self.card, text="CURRENT ACTION", font=("Helvetica", 10, "bold"), fg="#8e8e93", bg="#1c1c1e")
        self.lbl_sig_header.pack(pady=(8, 0))

        self.lbl_action = tk.Label(self.card, text="CALCULATING...", font=("Helvetica", 28, "bold"), fg="#ffffff", bg="#1c1c1e")
        self.lbl_action.pack(pady=4)

        self.lbl_desc = tk.Label(self.card, text="", font=("Helvetica", 10), fg="#cccccc", bg="#1c1c1e", wraplength=420)
        self.lbl_desc.pack(pady=(0, 8))

        # Metrics Grid
        frame_metrics = tk.Frame(self, bg="#121214")
        frame_metrics.pack(fill="x", padx=24, pady=8)

        self.val_price = self.add_metric(frame_metrics, "Current Price", "$0.00", 0, 0)
        self.val_prob = self.add_metric(frame_metrics, "Win Probability", "0.0%", 0, 1)
        self.val_conviction = self.add_metric(frame_metrics, "Conviction", "+0", 1, 0)
        self.val_vwap = self.add_metric(frame_metrics, "Session VWAP", "$0.00", 1, 1)

        self.lbl_time = tk.Label(self, text="", font=("Helvetica", 9), fg="#636366", bg="#121214")
        self.lbl_time.pack(side="bottom", pady=12)

    def add_metric(self, parent, title, initial_val, r, c):
        f = tk.Frame(parent, bg="#1c1c1e", padx=12, pady=8)
        f.grid(row=r, column=c, padx=6, pady=6, sticky="nsew")
        parent.grid_columnconfigure(c, weight=1)
        
        tk.Label(f, text=title, font=("Helvetica", 9), fg="#8e8e93", bg="#1c1c1e").pack(anchor="w")
        lbl_v = tk.Label(f, text=initial_val, font=("Helvetica", 14, "bold"), fg="#ffffff", bg="#1c1c1e")
        lbl_v.pack(anchor="w", pady=(2, 0))
        return lbl_v

    def refresh_signal(self):
        self.btn_refresh.config(text="Loading...", state="disabled")
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            symbol = ASSETS[self.asset_var.get()]
            res = fetch_data_and_signal(symbol, self.tf_var.get())
            self.after(0, self._update_ui, res)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"Failed: {e}"))
        finally:
            self.after(0, lambda: self.btn_refresh.config(text="⚡ Refresh", state="normal"))

    def _update_ui(self, res):
        self.lbl_action.config(text=res["action"], fg=res["color"])
        self.card.config(bg=res["bg_card"], highlightbackground=res["color"], highlightthickness=2)
        self.lbl_sig_header.config(bg=res["bg_card"])
        self.lbl_desc.config(text=res["desc"], bg=res["bg_card"])
        
        self.val_price.config(text=f"${res['price']:,.2f}")
        self.val_prob.config(text=f"{res['prob_up']*100:.1f}%")
        self.val_conviction.config(text=f"{res['conviction']:+d} / 100")
        self.val_vwap.config(text=f"${res['vwap']:,.2f} ({res['vwap_z']:+.1f}σ)")
        self.lbl_time.config(text=f"Last updated: {res['time']}")

if __name__ == "__main__":
    app = MultiIndexApp()
    app.mainloop()
