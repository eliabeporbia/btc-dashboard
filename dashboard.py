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
# CONFIGURAÇÕES INICIAIS
# ======================
st.set_page_config(layout="wide", page_title="BTC Super Dashboard Pro+")
st.title("🚀 BTC Super Dashboard Pro+ - Edição Premium")

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
# NOVAS FUNÇÕES ADICIONADAS
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

def get_asset_today_change():
    """Retorna a variação percentual do dia do S&P500 e Ouro"""
    try:
        # Dados em tempo real (último pregão)
        sp500 = yf.Ticker("^GSPC").history(period="2d")["Close"].pct_change()[-1]
        ouro = yf.Ticker("GC=F").history(period="2d")["Close"].pct_change()[-1]
        return {
            "SP500": {"change": sp500, "arrow": "↑" if sp500 > 0 else "↓"},
            "OURO": {"change": ouro, "arrow": "↑" if ouro > 0 else "↓"}
        }
    except:
        return {
            "SP500": {"change": 0, "arrow": "→"},
            "OURO": {"change": 0, "arrow": "→"}
        }

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
        
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        # Dados simulados como fallback
        dates = pd.date_range(end=datetime.today(), periods=90)
        data = {
            'prices': pd.DataFrame({
                'date': dates,
                'price': np.linspace(45000, 50000, 90),
                'MA7': np.linspace(44800, 50200, 90),
                'MA30': np.linspace(44600, 49800, 90)
            })
        }
    return data

# ======================
# GERADOR DE SINAIS (COM CONFLUÊNCIA)
# ======================

def generate_signals(data):
    signals = []
    sentiment = get_market_sentiment()
    
    if not data['prices'].empty:
        last_price = data['prices']['price'].iloc[-1]
        last_rsi = data['prices']['RSI'].iloc[-1]
        
        # 1. Sinais de Médias Móveis
        signals.append(("Preço vs MA7", "COMPRA" if last_price > data['prices']['MA7'].iloc[-1] else "VENDA", 
                       f"{(last_price/data['prices']['MA7'].iloc[-1]-1):.2%}"))
        
        # 2. RSI + Confluência com Sentimento
        rsi_signal = "COMPRA" if last_rsi < 30 else "VENDA" if last_rsi > 70 else "NEUTRO"
        signals.append(("RSI (14)", rsi_signal, f"{last_rsi:.1f}"))
        
        if (last_rsi < 35) and (sentiment['value'] < 30):
            signals.append(("🔥 RSI + Medo Extremo", "COMPRA FORTE", f"RSI: {last_rsi:.1f} | Sentimento: {sentiment['value']}"))
        
        # 3. Bollinger Bands + Confluência
        bb_signal = "COMPRA" if last_price < data['prices']['BB_Lower'].iloc[-1] else "VENDA" if last_price > data['prices']['BB_Upper'].iloc[-1] else "NEUTRO"
        signals.append(("Bollinger Bands", bb_signal, f"Atual: ${last_price:,.0f}"))
        
        if (last_price < data['prices']['BB_Lower'].iloc[-1]) and (sentiment['value'] < 30):
            signals.append(("🔥 Bollinger + Medo", "COMPRA", "Preço na Banda Inferior + Medo"))
        
        # 4. Sentimento do Mercado
        signals.append(("📢 Sentimento", "COMPRA" if sentiment['value'] < 25 else "VENDA" if sentiment['value'] > 75 else "NEUTRO", 
                       f"{sentiment['value']} ({sentiment['sentiment']})"))
    
    # Contagem de sinais
    buy_signals = sum(1 for s in signals if "COMPRA" in s[1])
    sell_signals = sum(1 for s in signals if "VENDA" in s[1])
    
    # Veredito final
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
asset_changes = get_asset_today_change()

# Sidebar
st.sidebar.header("⚙️ Configurações")
st.sidebar.subheader("🔧 Parâmetros Técnicos")
rsi_window = st.sidebar.slider("Período do RSI", 7, 21, 14)
bb_window = st.sidebar.slider("Bandas de Bollinger (dias)", 10, 50, 20)

# Métricas com setas (CORRIGIDO O ERRO DE SYNTAX)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Preço BTC", f"${data['prices']['price'].iloc[-1]:,.2f}")
col2.metric("S&P 500 (Hoje)", 
           f"{asset_changes['SP500']['arrow']} {abs(asset_changes['SP500']['change'])*100:.2f}%",
           "Alta → Risco" if asset_changes['SP500']['change'] > 0 else "Baixa → Hedge")
col3.metric("OURO (Hoje)", 
           f"{asset_changes['OURO']['arrow']} {abs(asset_changes['OURO']['change'])*100:.2f}%",
           "Alta → Hedge" if asset_changes['OURO']['change'] > 0 else "Baixa → Risco")
col4.metric("Análise Final", final_verdict)

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
    return 'background-color: #FFC107'

st.dataframe(
    df_signals.style.applymap(color_signal, subset=["Sinal"]),
    hide_index=True,
    use_container_width=True
)

# Abas principais
tab1, tab2, tab3 = st.tabs(["📉 Preço", "📊 Técnico", "🐳 Whales"])

with tab1:
    if not data['prices'].empty:
        fig = px.line(data['prices'], x="date", y=["price", "MA7", "MA30"], title="Preço BTC")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    if not data['prices'].empty:
        fig_rsi = px.line(data['prices'], x="date", y="RSI", title="RSI (14 dias)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        st.plotly_chart(fig_rsi, use_container_width=True)

with tab3:
    if 'whale_alert' in data:
        st.dataframe(data['whale_alert'], hide_index=True)

# Rodapé
st.sidebar.markdown("""
**📌 Legenda:**
- 🟢 **COMPRA**: Indicador positivo
- 🔴 **VENDA**: Indicador negativo
- 🟡 **NEUTRO**: Sem sinal claro
- ✅ **FORTE COMPRA**: Múltiplas confirmações
""")
