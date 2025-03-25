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
st.title("🚀 BTC Super Dashboard Pro+ - Confluência de Indicadores")

# ======================
# FUNÇÕES DE CÁLCULO (MANTIDAS)
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
# NOVAS FUNÇÕES (CONFLUÊNCIA + COMPARAÇÃO BTC/OURO/S&P500)
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

def get_asset_comparison():
    """Compara BTC com Ouro e S&P500 (retorna % de variação)"""
    try:
        # Dados simulados (substitua por API real se quiser)
        btc_price = 50000  # Exemplo: pegar do data['prices']
        sp500_change = 0.02  # +2%
        gold_change = -0.01  # -1%
        return {
            "SP500": {"change": sp500_change, "arrow": "↑" if sp500_change > 0 else "↓"},
            "OURO": {"change": gold_change, "arrow": "↑" if gold_change > 0 else "↓"}
        }
    except:
        return {"SP500": {"change": 0, "arrow": "→"}, "OURO": {"change": 0, "arrow": "→"}}

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
        
        # 1. Sinais de Médias Móveis (original)
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
        
        # 2. RSI (original + confluência com sentimento)
        rsi_signal = "COMPRA" if last_rsi < 30 else "VENDA" if last_rsi > 70 else "NEUTRO"
        signals.append(("RSI (14)", rsi_signal, f"{last_rsi:.2f}"))
        
        # Confluência RSI + Sentimento
        if (last_rsi < 35) and (sentiment['value'] < 30):
            signals.append(("🔥 RSI + Medo Extremo", "COMPRA FORTE", f"RSI: {last_rsi:.1f} | Sentimento: {sentiment['value']}"))
        
        # 3. MACD (original)
        macd = data['prices']['MACD'].iloc[-1]
        macd_signal = "COMPRA" if macd > 0 else "VENDA"
        signals.append(("MACD", macd_signal, f"{macd:.2f}"))
        
        # 4. Bandas de Bollinger (original + confluência)
        bb_upper = data['prices']['BB_Upper'].iloc[-1]
        bb_lower = data['prices']['BB_Lower'].iloc[-1]
        bb_signal = "COMPRA" if last_price < bb_lower else "VENDA" if last_price > bb_upper else "NEUTRO"
        signals.append(("Bollinger Bands", bb_signal, f"Atual: ${last_price:,.0f}"))
        
        # Confluência Bollinger + Sentimento
        if (last_price < bb_lower) and (sentiment['value'] < 30):
            signals.append(("🔥 Bollinger + Medo", "COMPRA", "Preço na Banda Inferior + Medo"))
        
        # 5. Sentimento do Mercado (novo)
        sentiment_signal = "COMPRA" if sentiment['value'] < 25 else "VENDA" if sentiment['value'] > 75 else "NEUTRO"
        signals.append(("📢 Sentimento", sentiment_signal, f"{sentiment['value']} ({sentiment['sentiment']})"))
    
    # [...] (restante das funções originais mantidas)

    # Contagem de sinais (atualizada para incluir "COMPRA FORTE")
    buy_signals = sum(1 for s in signals if "COMPRA" in s[1])
    sell_signals = sum(1 for s in signals if "VENDA" in s[1])
    
    # Análise consolidada (original)
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
# INTERFACE DO USUÁRIO (COM SETAS DE COMPARAÇÃO)
# ======================

# Carregar dados
data = load_data()
signals, final_verdict, buy_signals, sell_signals = generate_signals(data)
asset_comparison = get_asset_comparison()

# Seção de métricas (com setas)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Preço BTC", f"${data['prices']['price'].iloc[-1]:,.2f}")
col2.metric("S&P 500", 
           f"{asset_comparison['SP500']['arrow']} {asset_comparison['SP500']['change']:.2%}",
           "COMPRA BTC" if asset_comparison['SP500']['change'] > 0 else "NEUTRO")
col3.metric("OURO", 
           f"{asset_comparison['OURO']['arrow']} {asset_comparison['OURO']['change']:.2%}",
           "COMPRA BTC" if asset_comparison['OURO']['change'] < 0 else "NEUTRO")
col4.metric("Análise Final", final_verdict)

# [...] (restante da interface original mantida)
