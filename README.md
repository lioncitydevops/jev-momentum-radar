# 🌐 Global Multi-Index Momentum Radar (TypeSafe Jev System One)

Real-time short-term momentum signals and cross-market intelligence for **S&P 500 (SPY)**, **Nasdaq 100 (QQQ)**, **Russell 2000 (IWM)**, and **Nikkei 225 (^N225)**.

Powered by **TypeSafe Jev System One** models for micro-tactical execution judgments and probabilistic continuation forecasts.

---

## 🚀 Deploying to Vercel (Step-by-Step)

This repository is optimized for instant deployment on **Vercel**:

1. **Push to your GitHub repository** (`https://github.com/lioncitydevops/<your-repo-name>`).
2. Log in to [Vercel](https://vercel.com) and click **"Add New Project"** -> **"Import Git Repository"**.
3. Select this repository.
4. In the **Environment Variables** section on Vercel:
   - Key: `TYPESAFE_API_KEY`
   - Value: `<your-typesafe-api-key>`
5. Click **"Deploy"**!
6. Vercel will build and serve your web dashboard live at `https://<your-project>.vercel.app`.

---

## ⚡ Architecture & Features

- **Frontend (`public/index.html`)**: Sleek dark-mode interface with glassmorphism, responsive grid, dynamic continuation probability meters, cross-market relative strength matrix, and interactive deep-dive candlestick charts with Session VWAP.
- **Serverless API (`api/radar.py`)**: Stateless Vercel serverless function that fetches real-time market data from Yahoo Finance, computes technical metrics (VWAP z-score, 14-RSI, multi-scale momentum), and queries TypeSafe Jev System One API.
- **Streamlit Local App (`dashboard.py`)**: Desktop/local multi-asset dashboard executable via `Launch_Signal_App.bat`.
- **TradingView Bridge (`tradingview_jev_bridge.py` & `.pine`)**: Webhook bridge connecting TradingView Pine Script alerts directly into TypeSafe Jev.

---

## 💻 Local Development

### 1. Setup Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API Key
Create a `.env` file in the root directory:
```env
TYPESAFE_API_KEY=your_typesafe_api_key_here
```

### 3. Run Streamlit Dashboard
```bash
streamlit run dashboard.py
```
Or double-click `Launch_Signal_App.bat`.
