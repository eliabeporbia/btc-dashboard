import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# Configurações
ACCESS_TOKEN = "ory_at_JiREYSDxh-JYMCRaTPbOXDBWOMjAbqxAUKnIOQ8aaIc.7jI9xzEhmVH4dkzyvOly1rtC5aut-qNhSW0RQlUH-A8"
BITQUERY_HEADERS = {"Authorization": f"Bearer {ACCESS_TOKEN}"}

# ---- 1. PREÇO DO BTC (COINGECKO) ----
@st.cache_data(ttl=3600)
def get_btc_price():
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=90"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
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
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data["values"])
        df["date"] = pd.to_datetime(df["x"], unit="s")
        return df
    except Exception as e:
        st.error(f"Erro ao obter hash rate: {str(e)}")
        return pd.DataFrame(columns=["x", "y", "date"])

# ---- 3. INFLOW/OUTFLOW (BITQUERY) ----
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
    try:
        response = requests.post(
            "https://graphql.bitquery.io",
            json={"query": query},
            headers=BITQUERY_HEADERS,
            timeout=15
        )
        response.raise_for_status()
        data = response.json()
        
        # Debug opcional
        if st.session_state.get("debug_mode", False):
            st.json(data)
        
        # Processamento seguro
        bitcoin_data = data.get("data", {}).get("bitcoin", {})
        inflows = bitcoin_data.get("inputs", [])
        outflows = bitcoin_data.get("outputs", [])
        
        if not isinstance(inflows, list) or not isinstance(outflows, list):
            st.warning("Estrutura de dados inesperada da API Bitquery")
            return {"inflow": 0, "outflow": 0}
        
        inflow = sum(float(tx.get("value", 0)) for tx in inflows) / 1e8  # Converter satoshis para BTC
        outflow = sum(float(tx.get("value", 0)) for tx in outflows) / 1e8
        
        return {"inflow": inflow, "outflow": outflow}
        
    except requests.exceptions.RequestException as e:
        st.error(f"Erro na requisição: {str(e)}")
    except ValueError as e:
        st.error(f"Erro ao processar JSON: {str(e)}")
    except Exception as e:
        st.error(f"Erro inesperado: {str(e)}")
    
    return {"inflow": 0, "outflow": 0}  # Fallback

# ---- 4. MVRV/SOPR (SIMULADO) ----
@st.cache_data
def mock_onchain_metrics():
    try:
        dates = pd.date_range(end=datetime.today(), periods=90)
        mvrv = [1.0 + 0.03*i for i in range(90)]
        sopr = [0.98 + 0.01*i for i in range(90)]
        return pd.DataFrame({"date": dates, "MVRV": mvrv, "SOPR": sopr})
    except Exception as e:
        st.error(f"Erro ao gerar dados simulados: {str(e)}")
        return pd.DataFrame(columns=["date", "MVRV", "SOPR"])

# ---- CONFIGURAÇÃO DO PAINEL ----
st.set_page_config(layout="wide", page_title="BTC On-Chain Dashboard")
st.title("📊 Painel BTC On-Chain")

# Modo debug
if st.sidebar.checkbox("Modo Debug"):
    st.session_state.debug_mode = True
    st.warning("Modo debug ativado - mostrando dados brutos")

# ---- CARREGAMENTO DE DADOS ----
with st.spinner("Carregando dados..."):
    df_price = get_btc_price()
    df_hashrate = get_hash_rate()
    flows = get_exchange_flows()
    df_metrics = mock_onchain_metrics()

# ---- LAYOUT PRINCIPAL ----
tab1, tab2, tab3 = st.tabs(["📈 Mercado", "🏦 Exchanges", "📊 On-Chain"])

with tab1:
    if not df_price.empty:
        st.subheader("Preço do BTC (CoinGecko)")
        fig_price = px.line(df_price, x="date", y="price", title="Variação de Preço")
        st.plotly_chart(fig_price, use_container_width=True)
    
    if not df_hashrate.empty:
        st.subheader("Hash Rate (Blockchain.com)")
        fig_hash = px.line(df_hashrate, x="date", y="y", labels={"y": "TH/s"}, title="Poder de Mineração")
        st.plotly_chart(fig_hash, use_container_width=True)

with tab2:
    st.subheader("Fluxo de Exchanges (Binance)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Inflow", f"{flows['inflow']:.2f} BTC")
    with col2:
        st.metric("Outflow", f"{flows['outflow']:.2f} BTC")
    
    fig_flows = px.bar(
        x=["Inflow", "Outflow"],
        y=[flows["inflow"], flows["outflow"]],
        color=["Inflow", "Outflow"],
        color_discrete_map={"Inflow": "green", "Outflow": "red"},
        title="Movimentação na Binance"
    )
    st.plotly_chart(fig_flows, use_container_width=True)

with tab3:
    if not df_metrics.empty:
        st.subheader("MVRV Ratio (Simulado)")
        fig_mvrv = px.line(df_metrics, x="date", y="MVRV")
        fig_mvrv.add_hline(y=3.7, line_color="red", annotation_text="Topo Histórico")
        fig_mvrv.add_hline(y=1.0, line_color="green", annotation_text="Fundo Histórico")
        st.plotly_chart(fig_mvrv, use_container_width=True)
        
        st.subheader("SOPR (Simulado)")
        fig_sopr = px.line(df_metrics, x="date", y="SOPR")
        fig_sopr.add_hline(y=1.0, line_color="gray", annotation_text="Break-even")
        st.plotly_chart(fig_sopr, use_container_width=True)

# ---- RODAPÉ ----
st.divider()
st.caption("""
    **Notas**:  
    - Dados de exchange: Bitquery API (Binance)  
    - Preço: CoinGecko | Hash Rate: Blockchain.com  
    - MVRV/SOPR: Dados simulados (para versão real, use Glassnode)  
    - Atualização automática a cada 1 hora  
""")

# Botão para forçar atualização
if st.button("Atualizar Dados"):
    st.cache_data.clear()
    st.rerun()
