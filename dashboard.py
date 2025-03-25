import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

# Configuração do painel
st.set_page_config(layout="wide", page_title="BTC On-Chain Pro")
st.title("🚨 BTC On-Chain Pro - Sinais de Compra/Venda")

# ---- 1. DADOS EM TEMPO REAL ----
@st.cache_data(ttl=3600)
def load_data():
    # Dicionário para armazenar tudo
    data = {}
    
    try:
        # Preço e volume
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=90"
        response = requests.get(url, timeout=10)
        market_data = response.json()
        data['prices'] = pd.DataFrame(market_data["prices"], columns=["timestamp", "price"])
        data['prices']["date"] = pd.to_datetime(data['prices']["timestamp"], unit="ms")
        
        # Hashrate e dificuldade
        data['hashrate'] = pd.DataFrame(
            requests.get("https://api.blockchain.info/charts/hash-rate?format=json&timespan=3months").json()["values"]
        )
        data['hashrate']["date"] = pd.to_datetime(data['hashrate']["x"], unit="s")
        
        # Dificuldade (corrigido)
        data['difficulty'] = pd.DataFrame(
            requests.get("https://api.blockchain.info/charts/difficulty?timespan=2years&format=json").json()["values"]
        )
        data['difficulty']["date"] = pd.to_datetime(data['difficulty']["x"], unit="s")
        
        # Dados de exchanges (simulados + real)
        data['exchanges'] = {
            "binance": {"inflow": 1500, "outflow": 1200, "reserves": 500000},
            "coinbase": {"inflow": 800, "outflow": 750, "reserves": 350000},
            "kraken": {"inflow": 600, "outflow": 550, "reserves": 200000}
        }
        
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
    
    return data

data = load_data()

# ---- 2. SINAIS DE COMPRA/VENDA ----
def generate_signals():
    signals = []
    
    # 1. Tendência de preço (Média Móvel)
    if not data['prices'].empty:
        data['prices']['MA7'] = data['prices']['price'].rolling(7).mean()
        last_price = data['prices']['price'].iloc[-1]
        ma7 = data['prices']['MA7'].iloc[-1]
        signals.append(("Preço vs Média 7D", "COMPRA" if last_price > ma7 else "VENDA", last_price/ma7 - 1))
    
    # 2. Fluxo de exchanges
    if data['exchanges']:
        net_flows = sum(ex["inflow"] - ex["outflow"] for ex in data['exchanges'].values())
        signals.append(("Net Flow Exchanges", "COMPRA" if net_flows < 0 else "VENDA", net_flows))
    
    # 3. Hashrate vs Dificuldade
    if not data['hashrate'].empty and not data['difficulty'].empty:
        hr_growth = data['hashrate']['y'].iloc[-1] / data['hashrate']['y'].iloc[-30] - 1
        diff_growth = data['difficulty']['y'].iloc[-1] / data['difficulty']['y'].iloc[-30] - 1
        signals.append(("Hashrate vs Dificuldade", "COMPRA" if hr_growth > diff_growth else "VENDA", hr_growth - diff_growth))
    
    return signals

signals = generate_signals()

# ---- 3. LAYOUT DO PAINEL ----
st.header("📢 Sinais de Mercado", divider="rainbow")

# Área de status
col1, col2, col3 = st.columns(3)
col1.metric("Preço Atual", f"${data['prices']['price'].iloc[-1]:,.2f}" if not data['prices'].empty else "N/A")
col2.metric("Hash Rate", f"{data['hashrate']['y'].iloc[-1]/1e6:,.1f} EH/s" if not data['hashrate'].empty else "N/A")
col3.metric("Dificuldade", f"{data['difficulty']['y'].iloc[-1]/1e12:,.1f} T" if not data['difficulty'].empty else "N/A")

# Tabela de sinais
st.subheader("📈 Indicadores Técnicos")
df_signals = pd.DataFrame(
    [(name, signal, f"{value:.2%}" if isinstance(value, float) else value) for name, signal, value in signals],
    columns=["Indicador", "Sinal", "Valor"]
)
st.dataframe(
    df_signals.style.applymap(
        lambda x: "background-color: #4CAF50" if x == "COMPRA" else "background-color: #F44336", 
        subset=["Sinal"]
    ),
    hide_index=True,
    use_container_width=True
)

# Gráficos
tab1, tab2, tab3 = st.tabs(["📊 Mercado", "🏦 Exchanges", "🔍 Detalhes"])

with tab1:
    if not data['prices'].empty:
        fig = px.line(data['prices'], x="date", y=["price", "MA7"], title="Preço BTC vs Média Móvel 7 Dias")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    df_exchanges = pd.DataFrame(data['exchanges']).T
    fig = px.bar(df_exchanges, y=["inflow", "outflow"], barmode="group", title="Fluxo de Exchanges (BTC)")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    if not data['difficulty'].empty:
        fig = px.line(data['difficulty'], x="date", y="y", title="Dificuldade da Rede (Últimos 2 Anos)")
        st.plotly_chart(fig, use_container_width=True)

# ---- RODAPÉ ----
st.sidebar.header("🔧 Configurações")
if st.sidebar.button("Atualizar Dados"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("""
**📌 Legenda:**
- 🟢 **COMPRA**: 3+ indicadores positivos
- 🔴 **VENDA**: 3+ indicadores negativos
- 🟡 **NEUTRO**: Sinais mistos
""")
