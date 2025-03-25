import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# Configuração do painel
st.set_page_config(layout="wide", page_title="BTC On-Chain Dashboard")
st.title("📊 Painel BTC On-Chain (Dados Reais)")

# ---- 1. PREÇO DO BTC (COINGECKO) ----
@st.cache_data(ttl=3600)
def get_btc_price():
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=90"
        response = requests.get(url, timeout=10)
        data = response.json()
        prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        prices["date"] = pd.to_datetime(prices["timestamp"], unit="ms")
        return prices
    except Exception as e:
        st.error(f"Erro ao obter preço: {str(e)}")
        return pd.DataFrame(columns=["timestamp", "price", "date"])

# ---- 2. HASH RATE (BLOCKCHAIN.COM) ----
@st.cache_data(ttl=3600)
def get_hash_rate():
    try:
        url = "https://api.blockchain.info/charts/hash-rate?format=json&timespan=3months"
        response = requests.get(url, timeout=10)
        data = response.json()
        df = pd.DataFrame(data["values"])
        df["date"] = pd.to_datetime(df["x"], unit="s")
        return df
    except Exception as e:
        st.error(f"Erro ao obter hash rate: {str(e)}")
        return pd.DataFrame(columns=["x", "y", "date"])

# ---- 3. FLUXO DE EXCHANGES (SIMULADO) ----
@st.cache_data(ttl=3600)
def get_exchange_flows():
    # Dados simulados (como na versão anterior)
    return {
        "binance": {"inflow": 1500, "outflow": 1200},
        "coinbase": {"inflow": 800, "outflow": 750},
        "kraken": {"inflow": 600, "outflow": 550}
    }

# ---- 4. DIFICULDADE DA REDE (CORREÇÃO) ----
@st.cache_data(ttl=3600)
def get_difficulty_data():
    try:
        # Nova fonte confiável para dados históricos
        url = "https://api.blockchain.info/charts/difficulty?timespan=2years&format=json"
        response = requests.get(url, timeout=15)
        data = response.json()
        df = pd.DataFrame(data["values"])
        df["date"] = pd.to_datetime(df["x"], unit="s")
        return df
    except Exception as e:
        st.error(f"Erro ao obter dificuldade: {str(e)}")
        return pd.DataFrame(columns=["x", "y", "date"])

# ---- CARREGAMENTO DE DADOS ----
with st.spinner("Atualizando dados..."):
    df_price = get_btc_price()
    df_hashrate = get_hash_rate()
    exchanges = get_exchange_flows()
    df_difficulty = get_difficulty_data()

# ---- LAYOUT DO PAINEL (IGUAL À VERSÃO ANTERIOR) ----
tab1, tab2, tab3 = st.tabs(["📈 Mercado", "🏦 Exchanges", "⚙️ Rede Bitcoin"])

with tab1:
    if not df_price.empty:
        st.subheader("Preço do BTC (Últimos 90 dias)")
        fig_price = px.line(df_price, x="date", y="price")
        st.plotly_chart(fig_price, use_container_width=True)
    
    if not df_hashrate.empty:
        st.subheader("Hash Rate (TH/s)")
        fig_hash = px.line(df_hashrate, x="date", y="y")
        st.plotly_chart(fig_hash, use_container_width=True)

with tab2:
    st.subheader("Fluxo de Exchanges (Simulado)")
    df_exchanges = pd.DataFrame(exchanges).T
    st.dataframe(df_exchanges.style.format("{:,.0f} BTC"))
    
    fig_flows = px.bar(
        df_exchanges,
        x=df_exchanges.index,
        y=["inflow", "outflow"],
        barmode="group",
        title="Inflow vs Outflow"
    )
    st.plotly_chart(fig_flows, use_container_width=True)

with tab3:
    st.subheader("Dificuldade da Rede (Corrigido)")
    if not df_difficulty.empty:
        fig_diff = px.line(
            df_difficulty,
            x="date",
            y="y",
            title="Dificuldade de Mineração (Últimos 2 Anos)"
        )
        st.plotly_chart(fig_diff, use_container_width=True)
    else:
        st.info("Dados de dificuldade carregados parcialmente")

# ---- RODAPÉ (MESMO DA VERSÃO ANTERIOR) ----
st.sidebar.header("Configurações")
if st.sidebar.button("Atualizar Dados"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption(f"Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
