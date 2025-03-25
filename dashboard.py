import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

# Configurações
ACCESS_TOKEN = "ory_at_JiREYSDxh-JYMCRaTPbOXDBWOMjAbqxAUKnIOQ8aaIc.7jI9xzEhmVH4dkzyvOly1rtC5aut-qNhSW0RQlUH-A8"
BITQUERY_HEADERS = {"Authorization": f"Bearer {ACCESS_TOKEN}"}

# ---- 1. PREÇO DO BTC (COINGECKO) ----
@st.cache_data(ttl=3600)
def get_btc_price():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=90"
    data = requests.get(url).json()
    prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    prices["date"] = pd.to_datetime(prices["timestamp"], unit="ms")
    return prices

# ---- 2. HASH RATE (BLOCKCHAIN.COM) ----
@st.cache_data(ttl=3600)
def get_hash_rate():
    url = "https://api.blockchain.info/charts/hash-rate?format=json&timespan=3months"
    data = requests.get(url).json()
    df = pd.DataFrame(data["values"])
    df["date"] = pd.to_datetime(df["x"], unit="s")
    return df

# ---- 3. INFLOW/OUTFLOW (BITQUERY - REAL) ----
@st.cache_data(ttl=3600)
def get_exchange_flows():
    query = """
    {
      bitcoin(network: bitcoin) {
        inputs(exchange: {is: "binance"}, date: {since: "2024-01-01"}) {
          value
        }
        outputs(exchange: {is: "binance"}, date: {since: "2024-01-01"}) {
          value
        }
      }
    }
    """
    response = requests.post(
        "https://graphql.bitquery.io",
        json={"query": query},
        headers=BITQUERY_HEADERS
    )
    data = response.json()
    # Processa os dados (exemplo simplificado)
    inflow = sum(float(tx["value"]) for tx in data["data"]["bitcoin"]["inputs"])
    outflow = sum(float(tx["value"]) for tx in data["data"]["bitcoin"]["outputs"])
    return {"inflow": inflow / 1e8, "outflow": outflow / 1e8}  # Converte de satoshis para BTC

# ---- 4. MVRV/SOPR (SIMULADO) ----
@st.cache_data
def mock_onchain_metrics():
    dates = pd.date_range(end=datetime.today(), periods=90)
    mvrv = [1.0 + 0.03*i for i in range(90)]
    sopr = [0.98 + 0.01*i for i in range(90)]
    return pd.DataFrame({"date": dates, "MVRV": mvrv, "SOPR": sopr})

# ---- LAYOUT DO PAINEL ----
st.title("📊 Painel BTC On-Chain (Dados Reais + Bitquery)")
df_price = get_btc_price()
df_hashrate = get_hash_rate()
flows = get_exchange_flows()
df_metrics = mock_onchain_metrics()

# Métricas rápidas
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Preço BTC", f"${df_price['price'].iloc[-1]:,.2f}")
with col2:
    st.metric("Hash Rate", f"{df_hashrate['y'].iloc[-1]:,.0f} TH/s")
with col3:
    st.metric("Net Flow (Binance)", f"{(flows['inflow'] - flows['outflow']):.2f} BTC")

# Abas
tab1, tab2, tab3 = st.tabs(["📈 Mercado", "🏦 Exchanges", "📊 On-Chain"])

with tab1:
    st.subheader("Preço do BTC (CoinGecko)")
    fig_price = px.line(df_price, x="date", y="price")
    st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("Hash Rate (Blockchain.com)")
    fig_hash = px.line(df_hashrate, x="date", y="y", labels={"y": "TH/s"})
    st.plotly_chart(fig_hash, use_container_width=True)

with tab2:
    st.subheader("Inflow/Outflow - Binance (Bitquery)")
    fig_flows = px.bar(
        x=["Inflow", "Outflow"],
        y=[flows["inflow"], flows["outflow"]],
        color=["Inflow", "Outflow"],
        color_discrete_map={"Inflow": "green", "Outflow": "red"}
    )
    st.plotly_chart(fig_flows, use_container_width=True)

with tab3:
    st.subheader("MVRV Ratio (Simulado)")
    fig_mvrv = px.line(df_metrics, x="date", y="MVRV")
    fig_mvrv.add_hline(y=3.7, line_color="red", annotation_text="Topo Histórico")
    st.plotly_chart(fig_mvrv, use_container_width=True)

    st.subheader("SOPR (Simulado)")
    fig_sopr = px.line(df_metrics, x="date", y="SOPR")
    fig_sopr.add_hline(y=1.0, line_color="gray", annotation_text="Break-even")
    st.plotly_chart(fig_sopr, use_container_width=True)

# Notas
st.divider()
st.caption("""
    **Notas**:  
    - Dados de inflow/outflow são reais (via Bitquery).  
    - MVRV/SOPR são simulados (para dados reais, use Glassnode).  
    - Token válido por 100 anos (não compartilhe publicamente!).  
""")
