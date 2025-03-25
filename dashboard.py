import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

# Configuração do painel
st.set_page_config(layout="wide", page_title="BTC On-Chain Simplificado")
st.title("💰 Painel BTC Simplificado (Dados Reais)")

# ---- DADOS EM TEMPO REAL ----
@st.cache_data(ttl=600)  # Atualiza a cada 10 minutos
def get_data():
    try:
        # Preço do BTC
        price_url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
        price = requests.get(price_url).json()["bitcoin"]["usd"]
        
        # Dificuldade e Hashrate (simplificado)
        diff = float(requests.get("https://blockchain.info/q/getdifficulty").text)
        hashrate = diff / (600 * 1e12)  # Fórmula simplificada
        
        return {
            "price": price,
            "difficulty": diff,
            "hashrate": hashrate,
            "status": "✅ Saudável" if hashrate > 500000 else "⚠️ Fraco"
        }
    except:
        return {"price": 0, "difficulty": 0, "hashrate": 0, "status": "Erro"}

data = get_data()

# ---- LAYOUT SIMPLES ----
st.header("📌 Status Atual")
col1, col2, col3 = st.columns(3)
col1.metric("Preço BTC", f"${data['price']:,.2f}")
col2.metric("Dificuldade", f"{data['difficulty']/1e12:,.1f}T")
col3.metric("Hash Rate", f"{data['hashrate']:,.0f} TH/s", data['status'])

st.header("📊 Gráfico de Preço (Últimos 30 Dias")
try:
    # Gráfico de preço histórico
    hist_url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=30"
    hist_data = requests.get(hist_url).json()
    df = pd.DataFrame(hist_data["prices"], columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    fig = px.line(df, x="date", y="price", title="Preço do BTC")
    st.plotly_chart(fig, use_container_width=True)
except:
    st.warning("Gráfico indisponível no momento")

# ---- DICAS RÁPIDAS ----
st.header("💡 Interpretação Rápida")
if data['hashrate'] > 500000 and data['price'] > 30000:
    st.success("**Mercado Forte**: Hash rate alto e preço estável - bom momento para HODL")
elif data['hashrate'] < 300000:
    st.warning("**Cautela**: Hash rate baixo - rede menos segura")
else:
    st.info("**Mercado Neutro**: Aguarde mais sinais")

# ---- ATUALIZAÇÃO ----
if st.button("🔄 Atualizar Agora"):
    st.cache_data.clear()
    st.rerun()

st.caption("Dados atualizados a cada 10 minutos | Fontes: CoinGecko, Blockchain.com")
