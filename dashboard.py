import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from fpdf import FPDF
import yfinance as yf
import tempfile

# ======================
# INICIALIZAÇÃO DO SESSION STATE
# ======================
if not hasattr(st, 'session_state'):
    st.session_state = {}
if 'config' not in st.session_state:
    st.session_state.config = {
        'rsi_window': 14,
        'bb_window': 20,
        'ma_windows': [7, 30, 200]
    }

# ======================
# CONFIGURAÇÕES INICIAIS
# ======================
st.set_page_config(layout="wide", page_title="BTC Super Dashboard Pro+")
st.title("🚀 BTC Super Dashboard Pro+ - Edição Premium")

# ======================
# FUNÇÕES DE CÁLCULO (ATUALIZADAS PARA USAR CONFIG)
# ======================

def calculate_ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def calculate_rsi(series, window=None):
    window = window or st.session_state.config['rsi_window']
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd = ema_fast - ema_slow
    signal_line = calculate_ema(macd, signal)
    return macd, signal_line

def calculate_bollinger_bands(series, window=None, num_std=2):
    window = window or st.session_state.config['bb_window']
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, lower

# ======================
# FUNÇÕES AUXILIARES
# ======================

def get_market_sentiment():
    try:
        response = requests.get("https://api.alternative.me/fng/", timeout=5)
        data = response.json()
        return {
            "value": int(data["data"][0]["value"]),
            "sentiment": data["data"][0]["value_classification"]
        }
    except:
        return {"value": 50, "sentiment": "Neutral"}

def get_traditional_assets():
    assets = {
        "S&P 500": "^GSPC",
        "Ouro": "GC=F",
        "ETH-USD": "ETH-USD"
    }
    dfs = []
    for name, ticker in assets.items():
        data = yf.Ticker(ticker).history(period="90d", interval="1d")
        data = data.reset_index()[['Date', 'Close']].rename(columns={'Close': 'value', 'Date': 'date'})
        data['asset'] = name
        dfs.append(data)
    return pd.concat(dfs)

# ======================
# CARREGAMENTO DE DADOS (ATUALIZADO)
# ======================

@st.cache_data(ttl=3600)
def load_data():
    data = {}
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=90"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        market_data = response.json()
        
        data['prices'] = pd.DataFrame(market_data["prices"], columns=["timestamp", "price"])
        data['prices']["date"] = pd.to_datetime(data['prices']["timestamp"], unit="ms")
        
        price_series = data['prices']['price']
        
        # Aplicando configurações do usuário
        data['prices']['RSI'] = calculate_rsi(price_series)
        data['prices']['BB_Upper'], data['prices']['BB_Lower'] = calculate_bollinger_bands(price_series)
        
        # Adicionando médias móveis selecionadas
        for window in st.session_state.config['ma_windows']:
            data['prices'][f'MA{window}'] = price_series.rolling(window).mean()
        
        # Restante do carregamento de dados...
        hr_response = requests.get("https://api.blockchain.info/charts/hash-rate?format=json&timespan=3months", timeout=10)
        hr_response.raise_for_status()
        data['hashrate'] = pd.DataFrame(hr_response.json()["values"])
        data['hashrate']["date"] = pd.to_datetime(data['hashrate']["x"], unit="s")
        
        diff_response = requests.get("https://api.blockchain.info/charts/difficulty?timespan=2years&format=json", timeout=10)
        diff_response.raise_for_status()
        data['difficulty'] = pd.DataFrame(diff_response.json()["values"])
        data['difficulty']["date"] = pd.to_datetime(data['difficulty']["x"], unit="s")
        
        data['exchanges'] = {
            "binance": {"inflow": 1500, "outflow": 1200, "reserves": 500000},
            "coinbase": {"inflow": 800, "outflow": 750, "reserves": 350000},
            "kraken": {"inflow": 600, "outflow": 550, "reserves": 200000}
        }
        
        data['whale_alert'] = pd.DataFrame({
            "date": [datetime.now() - timedelta(hours=h) for h in [1, 3, 5, 8, 12]],
            "amount": [250, 180, 120, 300, 150],
            "exchange": ["Binance", "Coinbase", "Kraken", "Binance", "FTX"]
        })
        
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        dates = pd.date_range(end=datetime.today(), periods=90)
        data = {
            'prices': pd.DataFrame({
                'date': dates,
                'price': np.linspace(45000, 50000, 90),
                'RSI': np.linspace(30, 70, 90),
                'BB_Upper': np.linspace(46000, 51000, 90),
                'BB_Lower': np.linspace(44000, 49000, 90)
            })
        }
        for window in st.session_state.config['ma_windows']:
            data['prices'][f'MA{window}'] = np.linspace(44800, 50200, 90)
    return data

# ======================
# BACKTESTING (ATUALIZADO)
# ======================

def backtest_strategy(data):
    df = data['prices'].copy()
    ma_window = st.session_state.config['ma_windows'][1] if len(st.session_state.config['ma_windows']) > 1 else 30
    
    df['signal'] = np.where(
        (df['RSI'] < 30) & (df['price'] < df[f'MA{ma_window}']), 1,
        np.where(
            (df['RSI'] > 70) & (df['price'] > df[f'MA{ma_window}']), -1, 0
        )
    )
    
    df['daily_return'] = df['price'].pct_change()
    df['strategy_return'] = df['signal'].shift(1) * df['daily_return']
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
    
    return df

# ======================
# INTERFACE DO USUÁRIO
# ======================

def update_config():
    st.session_state.config = {
        'rsi_window': st.session_state.rsi_slider,
        'bb_window': st.session_state.bb_slider,
        'ma_windows': st.session_state.ma_multiselect
    }
    st.rerun()

# Sidebar - Controles do Usuário
with st.sidebar:
    st.header("⚙️ Painel de Controle")
    with st.form("config_form"):
        st.subheader("🔧 Parâmetros Técnicos")
        
        st.slider(
            "Período do RSI",
            7, 21, st.session_state.config['rsi_window'],
            key="rsi_slider"
        )
        
        st.slider(
            "Janela das Bandas de Bollinger",
            10, 50, st.session_state.config['bb_window'],
            key="bb_slider"
        )
        
        st.multiselect(
            "Médias Móveis para Exibir",
            [7, 20, 30, 50, 100, 200],
            default=st.session_state.config['ma_windows'],
            key="ma_multiselect"
        )
        
        if st.form_submit_button("💾 Salvar Configurações", on_click=update_config):
            st.success("Configurações atualizadas!")

# Carregar dados
data = load_data()
bt_data = backtest_strategy(data)
sentiment = get_market_sentiment()
traditional_assets = get_traditional_assets()

# Restante da interface (mantida igual ao seu código original)
# ... (seções de métricas, abas, gráficos, etc.)

# [CONTINUA COM O RESTO DO SEU CÓDIGO ORIGINAL]
