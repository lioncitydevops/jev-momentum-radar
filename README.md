# 🌐 Global Multi-CFD Momentum Radar (TypeSafe Jev System One)

Real-time short-term CFD momentum signals and cross-market intelligence for **S&P 500 (SPX)**, **Nasdaq 100 (NDX)**, **Russell 2000 (RUT)**, **Nikkei 225 (NI225)**, and **10Y Treasury (TNX)**.

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

- **Frontend (`public/index.html`)**: Sleek dark-mode interface with glassmorphism, responsive grid, dynamic continuation probability meters, cross-market CFD relative strength matrix, and interactive deep-dive TradingView CFD charts with Session VWAP.
- **Serverless API (`api/index.py`)**: Stateless Vercel serverless function that fetches real-time market data from TradingView CFD Scanners & Yahoo Finance CFD tickers, computes technical metrics (VWAP z-score, 14-RSI, multi-scale momentum), and queries TypeSafe Jev System One API.
- **Streamlit Local App (`dashboard.py`)**: Desktop/local multi-asset CFD dashboard executable via `Launch_Signal_App.bat`.
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

### 2. Configure API Keys
Create or update `.env` in the root directory:
```env
TYPESAFE_API_KEY=your_typesafe_api_key_here

# Optional: OANDA v20 Real-Time Zero-Delay CFD Feed (24/5 streaming)
OANDA_API_KEY=your_oanda_personal_access_token_here
OANDA_ENVIRONMENT=practice  # or 'live'
```

> **Note on Price Lag**: Yahoo Finance free quotes are delayed by 15-20 minutes and do not update outside regular US cash market hours. Connecting your free **OANDA Practice Account API Token** enables continuous 24/5 zero-lag institutional CFD pricing for S&P 500 (`SPX500_USD`), Nasdaq 100 (`NAS100_USD`), Russell 2000 (`US2000_USD`), Nikkei 225 (`JP225_USD`), and US Treasuries (`USB10Y_USD`).

### 3. Run Streamlit Dashboard
```bash
streamlit run dashboard.py
```
Or double-click `Launch_Signal_App.bat`.
You can also toggle between OANDA Real-Time and Yahoo Finance directly within the dashboard sidebar.
