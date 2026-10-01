"""
OANDA v20 CFD Real-Time Price & Candlestick Feed
Provides continuous 24/5 streaming and historical candlestick data for global CFDs:
- S&P 500 (SPX500_USD)
- Nasdaq 100 (NAS100_USD)
- Russell 2000 (US2000_USD)
- Nikkei 225 (JP225_USD)
- 10-Year US Treasury Yield/Note (USB10Y_USD)
- Dow Jones 30 (US30_USD)
- Gold (XAU_USD)
- Crude Oil (WTICO_USD)
"""

import os
import requests
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Instrument Mappings from standard radar names to OANDA instrument identifiers
OANDA_INSTRUMENTS = {
    "S&P 500 (SPX)": "SPX500_USD",
    "Nasdaq 100 (NDX)": "NAS100_USD",
    "Russell 2000 (RUT)": "US2000_USD",
    "Nikkei 225 (NI225)": "JP225_USD",
    "10Y Treasury (TNX)": "USB10Y_USD",
    "WTI Crude (WTI)": "WTICO_USD",
    "Brent Crude (BRENT)": "BCO_USD",
    "Dow Jones 30 (US30)": "US30_USD",
    "Gold (XAU)": "XAU_USD"
}

# Reverse and alias lookup
SYMBOL_TO_OANDA = {
    "^GSPC": "SPX500_USD",
    "SPY": "SPX500_USD",
    "SPX": "SPX500_USD",
    "SPX500_USD": "SPX500_USD",
    "^NDX": "NAS100_USD",
    "QQQ": "NAS100_USD",
    "NDX": "NAS100_USD",
    "NAS100_USD": "NAS100_USD",
    "^RUT": "US2000_USD",
    "IWM": "US2000_USD",
    "RUT": "US2000_USD",
    "US2000_USD": "US2000_USD",
    "^N225": "JP225_USD",
    "NI225": "JP225_USD",
    "JP225_USD": "JP225_USD",
    "^TNX": "USB10Y_USD",
    "TNX": "USB10Y_USD",
    "USB10Y_USD": "USB10Y_USD",
    "CL=F": "WTICO_USD",
    "WTI": "WTICO_USD",
    "WTICO_USD": "WTICO_USD",
    "BZ=F": "BCO_USD",
    "BRENT": "BCO_USD",
    "BCO_USD": "BCO_USD"
}

# Granularity mapping
TIMEFRAME_TO_GRANULARITY = {
    "1m": "M1",
    "5m": "M5",
    "15m": "M15",
    "1h": "H1",
    "1d": "D"
}

def get_oanda_base_url(environment: str = "practice") -> str:
    """Returns the REST API base URL for practice or live accounts."""
    if environment.lower() == "live":
        return "https://api-fxtrade.oanda.com/v3"
    return "https://api-fxpractice.oanda.com/v3"

def get_configured_credentials() -> Tuple[str, str]:
    """Retrieves API key and environment from environment variables."""
    api_key = os.getenv("OANDA_API_KEY", "").strip()
    environment = os.getenv("OANDA_ENVIRONMENT", "practice").strip().lower()
    return api_key, environment

def test_oanda_connection(api_key: Optional[str] = None, environment: Optional[str] = None) -> Tuple[bool, str]:
    """Tests connection to Oanda with the provided or environment API key."""
    env_key, env_cfg = get_configured_credentials()
    if not api_key:
        api_key = env_key
    if not environment:
        environment = env_cfg
        
    if not api_key:
        return False, "OANDA_API_KEY is empty. Please set it in .env or provide your token."
        
    base_url = get_oanda_base_url(environment)
    url = f"{base_url}/instruments/SPX500_USD/candles?count=1&granularity=M5"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=6)
        if resp.status_code == 200:
            return True, f"Connected to OANDA ({environment.upper()}) successfully! Real-time CFD feed active."
        elif resp.status_code == 401:
            return False, f"401 Unauthorized on {environment.upper()}: Check token permissions."
        else:
            return False, f"OANDA API error {resp.status_code}: {resp.text}"
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def fetch_oanda_candles(
    instrument: str,
    timeframe: str = "5m",
    count: int = 100,
    api_key: Optional[str] = None,
    environment: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetches real-time completed and current midpoint candles from OANDA.
    
    Args:
        instrument: OANDA instrument e.g. 'SPX500_USD' or asset name/symbol
        timeframe: '1m', '5m', '1h', or '1d'
        count: number of bars to fetch (e.g. 50-200)
        api_key: optional API key override
        environment: 'practice' or 'live'
        
    Returns:
        pd.DataFrame with index DatetimeIndex (UTC) and Open, High, Low, Close, Volume columns.
    """
    env_key, env_cfg = get_configured_credentials()
    if not api_key:
        api_key = env_key
    if not environment:
        environment = env_cfg

    if not api_key:
        raise ValueError("Missing OANDA_API_KEY. Please provide or set in .env.")

    # Resolve instrument if alias or display name is given
    oanda_inst = OANDA_INSTRUMENTS.get(instrument, SYMBOL_TO_OANDA.get(instrument, instrument))
    granularity = TIMEFRAME_TO_GRANULARITY.get(timeframe, "M5")
    
    base_url = get_oanda_base_url(environment)
    url = f"{base_url}/instruments/{oanda_inst}/candles"
    params = {
        "price": "M",
        "granularity": granularity,
        "count": count
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    resp = requests.get(url, headers=headers, params=params, timeout=8)
    resp.raise_for_status()
    data = resp.json()
    
    candles = data.get("candles", [])
    if not candles:
        raise ValueError(f"No candle data returned from OANDA for {oanda_inst}")
        
    records = []
    for c in candles:
        t = pd.to_datetime(c["time"])
        mid = c["mid"]
        records.append({
            "Timestamp": t,
            "Open": float(mid["o"]),
            "High": float(mid["h"]),
            "Low": float(mid["l"]),
            "Close": float(mid["c"]),
            "Volume": float(c.get("volume", 0)),
            "Complete": bool(c.get("complete", True))
        })
        
    df = pd.DataFrame(records)
    df.set_index("Timestamp", inplace=True)
    return df

def fetch_oanda_pricing(
    instruments: list,
    account_id: Optional[str] = None,
    api_key: Optional[str] = None,
    environment: str = "practice"
) -> Dict[str, Any]:
    """
    Fetches real-time bid/ask pricing snapshot for multiple instruments.
    """
    if not api_key:
        api_key, environment = get_configured_credentials()
    if not account_id:
        account_id = os.getenv("OANDA_ACCOUNT_ID", "")
        
    if not api_key or not account_id:
        # Fall back to candle endpoints for latest close if account_id is not given
        results = {}
        for inst in instruments:
            try:
                df = fetch_oanda_candles(inst, timeframe="1m", count=1, api_key=api_key, environment=environment)
                results[inst] = {
                    "price": df["Close"].iloc[-1],
                    "time": df.index[-1].isoformat()
                }
            except Exception:
                pass
        return results

    base_url = get_oanda_base_url(environment)
    inst_str = ",".join(instruments)
    url = f"{base_url}/accounts/{account_id}/pricing?instruments={inst_str}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    resp = requests.get(url, headers=headers, timeout=6)
    resp.raise_for_status()
    return resp.json()

if __name__ == "__main__":
    key, env = get_configured_credentials()
    print(f"Testing OANDA module: env={env}, key_set={bool(key)}")
    ok, msg = test_oanda_connection(key, env)
    print(f"Connection result: {msg}")
