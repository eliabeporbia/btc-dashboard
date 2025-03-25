import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from fpdf import FPDF
import tempfile
from sklearn.linear_model import LinearRegression

# ======================
# CONFIGURAÇÕES INICIAIS
# ======================
st.set_page_config(layout="wide", page_title="BTC Super Dashboard Pro+")
st.title("🚀 BTC Super Dashboard Pro+ - Análise de Confluência")

# ======================
# FUNÇÕES DE CÁLCULO
# ======================

def calculate_ema(series, window):
    """Calcula a Média Móvel Exponencial (EMA)"""
    return series.ewm(span=window, adjust=False).mean()

def calculate_rsi(series, window=14):
    """Calcula o Índice de Força Relativa (RSI)"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calcula o MACD com linha de sinal"""
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd = ema_fast - ema_slow
    signal_line = calculate_ema(macd, signal)
    return macd, signal_line

def calculate_bollinger_bands(series, window=20, num_std=2):
    """Calcula as Bandas de Bollinger"""
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, lower

# ======================
# ANÁLISE DE SENTIMENTO (FEAR & GREED INDEX)
# ======================

def get_market_sentiment():
    """Coleta dados de sentimentos do mercado"""
    try:
        response = requests.get("https://api.alternative.me/fng/", timeout=5)
        data = response.json()
        return {
            "value": int(data["data"][0]["value"]),
            "sentiment": data["data"][0]["value_classification"]
        }
    except:
        return {"value": 50, "sentiment": "Neutral"}

# ======================
# CARREGAMENTO DE DADOS
# ======================

@st.cache_data(ttl=3600)
def load_data():
    data = {}
    try:
        # Preço do Bitcoin (últimos 90 dias)
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=90"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        market_data = response.json()
        
        data['prices'] = pd.DataFrame(market_data["prices"], columns=["timestamp", "price"])
        data['prices']["date"] = pd.to_datetime(data['prices']["timestamp"], unit="ms")
        
        # Calculando todos os indicadores técnicos
        price_series = data['prices']['price']
        data['prices']['MA7'] = price_series.rolling(7).mean()
        data['prices']['MA30'] = price_series.rolling(30).mean()
        data['prices']['MA200'] = price_series.rolling(200).mean()
        data['prices']['RSI'] = calculate_rsi(price_series)
        data['prices']['MACD'], data['prices']['MACD_Signal'] = calculate_macd(price_series)
        data['prices']['BB_Upper'], data['prices']['BB_Lower'] = calculate_bollinger_bands(price_series)
        
        # Hashrate (taxa de hash)
        hr_response = requests.get("https://api.blockchain.info/charts/hash-rate?format=json&timespan=3months", timeout=10)
        hr_response.raise_for_status()
        data['hashrate'] = pd.DataFrame(hr_response.json()["values"])
        data['hashrate']["date"] = pd.to_datetime(data['hashrate']["x"], unit="s")
        
        # Dificuldade de mineração
        diff_response = requests.get("https://api.blockchain.info/charts/difficulty?timespan=2years&format=json", timeout=10)
        diff_response.raise_for_status()
        data['difficulty'] = pd.DataFrame(diff_response.json()["values"])
        data['difficulty']["date"] = pd.to_datetime(data['difficulty']["x"], unit="s")
        
        # Dados simulados de exchanges
        data['exchanges'] = {
            "binance": {"inflow": 1500, "outflow": 1200, "reserves": 500000},
            "coinbase": {"inflow": 800, "outflow": 750, "reserves": 350000},
            "kraken": {"inflow": 600, "outflow": 550, "reserves": 200000}
        }
        
        # Atividade de "baleias" (grandes investidores)
        data['whale_alert'] = pd.DataFrame({
            "date": [datetime.now() - timedelta(hours=h) for h in [1, 3, 5, 8, 12]],
            "amount": [250, 180, 120, 300, 150],
            "exchange": ["Binance", "Coinbase", "Kraken", "Binance", "FTX"]
        })
        
    except requests.exceptions.RequestException as e:
        st.error(f"Erro na requisição à API: {str(e)}")
    except Exception as e:
        st.error(f"Erro ao processar dados: {str(e)}")
    return data

# ======================
# GERADOR DE SINAIS (COM CONFLUÊNCIA)
# ======================

def generate_signals(data):
    signals = []
    buy_signals = 0
    sell_signals = 0
    
    if not data['prices'].empty:
        last_price = data['prices']['price'].iloc[-1]
        last_rsi = data['prices']['RSI'].iloc[-1]
        sentiment = get_market_sentiment()
        
        # 1. Sinais de Médias Móveis
        ma_signals = [
            ("Preço vs MA7", data['prices']['MA7'].iloc[-1]),
            ("Preço vs MA30", data['prices']['MA30'].iloc[-1]),
            ("Preço vs MA200", data['prices']['MA200'].iloc[-1]),
            ("MA7 vs MA30", data['prices']['MA7'].iloc[-1], data['prices']['MA30'].iloc[-1])
        ]
        
        for name, *values in ma_signals:
            if len(values) == 1:
                signal = "COMPRA" if last_price > values[0] else "VENDA"
                change = (last_price/values[0] - 1)
            else:
                signal = "COMPRA" if values[0] > values[1] else "VENDA"
                change = (values[0]/values[1] - 1)
            signals.append((name, signal, f"{change:.2%}"))
        
        # 2. RSI
        rsi_signal = "COMPRA" if last_rsi < 30 else "VENDA" if last_rsi > 70 else "NEUTRO"
        signals.append(("RSI (14)", rsi_signal, f"{last_rsi:.2f}"))
        
        # 3. MACD
        macd = data['prices']['MACD'].iloc[-1]
        macd_signal = "COMPRA" if macd > 0 else "VENDA"
        signals.append(("MACD", macd_signal, f"{macd:.2f}"))
        
        # 4. Bandas de Bollinger
        bb_upper = data['prices']['BB_Upper'].iloc[-1]
        bb_lower = data['prices']['BB_Lower'].iloc[-1]
        bb_signal = "COMPRA" if last_price < bb_lower else "VENDA" if last_price > bb_upper else "NEUTRO"
        signals.append(("Bollinger Bands", bb_signal, f"Atual: ${last_price:,.0f}"))
        
        # 5. Sentimento do Mercado (Fear & Greed)
        sentiment_signal = "COMPRA" if sentiment['value'] < 25 else "VENDA" if sentiment['value'] > 75 else "NEUTRO"
        signals.append(("📢 Sentimento", sentiment_signal, f"{sentiment['value']} ({sentiment['sentiment']})"))
        
        # 6. CONFLUÊNCIA: RSI + Sentimento
        if (last_rsi < 35) and (sentiment['value'] < 30):
            signals.append(("🔥 RSI + Medo Extremo", "COMPRA FORTE", f"RSI: {last_rsi:.1f} | Sentimento: {sentiment['value']}"))
        
        # 7. CONFLUÊNCIA: Bollinger + Sentimento
        if (last_price < bb_lower) and (sentiment['value'] < 30):
            signals.append(("🔥 Bollinger + Medo", "COMPRA", "Preço na Banda Inferior + Medo"))
    
    # [...] (restante da função mantido igual)

    # Contagem de sinais
    buy_signals = sum(1 for s in signals if s[1] == "COMPRA" or "COMPRA FORTE" in s[1])
    sell_signals = sum(1 for s in signals if s[1] == "VENDA")
    
    # Análise consolidada
    if buy_signals >= sell_signals + 3:
        final_verdict = "✅ FORTE COMPRA"
    elif buy_signals > sell_signals:
        final_verdict = "📈 COMPRA"
    elif sell_signals >= buy_signals + 3:
        final_verdict = "❌ FORTE VENDA"
    elif sell_signals > buy_signals:
        final_verdict = "📉 VENDA"
    else:
        final_verdict = "➖ NEUTRO"
    
    return signals, final_verdict, buy_signals, sell_signals

# ======================
# INTERFACE DO USUÁRIO
# ======================

# Carregar dados
data = load_data()
signals, final_verdict, buy_signals, sell_signals = generate_signals(data)

# Sidebar - Controles do Usuário
st.sidebar.header("⚙️ Painel de Controle")
st.sidebar.subheader("🔧 Parâmetros Técnicos")
rsi_window = st.sidebar.slider("Período do RSI", 7, 21, 14)
bb_window = st.sidebar.slider("Janela das Bandas de Bollinger", 10, 50, 20)

# Seção principal
st.header("📊 Análise de Confluência BTC")

# Métricas
col1, col2, col3 = st.columns(3)
col1.metric("Preço Atual", f"${data['prices']['price'].iloc[-1]:,.2f}")
col2.metric("Sentimento", f"{get_market_sentiment()['value']}/100", get_market_sentiment()['sentiment'])
col3.metric("Análise Final", final_verdict)

# Tabela de Sinais
st.subheader(f"📈 Sinais de Mercado (COMPRA: {buy_signals} | VENDA: {sell_signals})")
df_signals = pd.DataFrame(signals, columns=["Indicador", "Sinal", "Valor"])

def color_signal(val):
    if "FORTE" in val:
        return 'background-color: #4CAF50; font-weight: bold;'
    elif "COMPRA" in val:
        return 'background-color: #4CAF50'
    elif "VENDA" in val:
        return 'background-color: #F44336'
    else:
        return 'background-color: #FFC107'

st.dataframe(
    df_signals.style.applymap(color_signal, subset=["Sinal"]),
    hide_index=True,
    use_container_width=True
)

# Gráficos (mantidos conforme original)
# [...]
