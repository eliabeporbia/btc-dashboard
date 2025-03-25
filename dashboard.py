import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from transformers import pipeline
from fpdf import FPDF

# Configuração do painel
st.set_page_config(layout="wide", page_title="BTC Super Dashboard")
st.title("🚀 BTC Super Dashboard - Análise Profissional")

# ---- 1. FUNÇÕES PRINCIPAIS ----
@st.cache_data(ttl=3600)
def load_data():
    data = {}
    try:
        # Preço e volume
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=90"
        market_data = requests.get(url, timeout=10).json()
        data['prices'] = pd.DataFrame(market_data["prices"], columns=["timestamp", "price"])
        data['prices']["date"] = pd.to_datetime(data['prices']["timestamp"], unit="ms")
        
        # Hashrate e dificuldade
        data['hashrate'] = pd.DataFrame(
            requests.get("https://api.blockchain.info/charts/hash-rate?format=json&timespan=3months").json()["values"]
        )
        data['hashrate']["date"] = pd.to_datetime(data['hashrate']["x"], unit="s")
        
        # Dificuldade
        data['difficulty'] = pd.DataFrame(
            requests.get("https://api.blockchain.info/charts/difficulty?timespan=2years&format=json").json()["values"]
        )
        data['difficulty']["date"] = pd.to_datetime(data['difficulty']["x"], unit="s")
        
        # Dados de exchanges (simulados)
        data['exchanges'] = {
            "binance": {"inflow": 1500, "outflow": 1200, "reserves": 500000},
            "coinbase": {"inflow": 800, "outflow": 750, "reserves": 350000},
            "kraken": {"inflow": 600, "outflow": 550, "reserves": 200000}
        }
        
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
    return data

def analyze_sentiment():
    try:
        tweets = [
            "Bitcoin está subindo forte hoje!",
            "Tenho medo da queda do BTC",
            "Ótimo momento para comprar Bitcoin"
        ]
        results = pipeline("sentiment-analysis")(tweets)
        positive = sum(1 for r in results if r["label"] == "POSITIVE") / len(results)
        return {"score": positive, "status": "✅ Positivo" if positive > 0.6 else "⚠️ Neutro" if positive > 0.4 else "❌ Negativo"}
    except:
        return {"score": 0.5, "status": "⚠️ Análise falhou"}

def generate_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Relatório BTC - {datetime.now().strftime('%d/%m/%Y')}", ln=1)
    pdf.cell(200, 10, txt=f"Preço Atual: ${data['prices']['price'].iloc[-1]:,.2f}", ln=1)
    pdf.output("report.pdf")
    return open("report.pdf", "rb")

# ---- 2. CARREGAMENTO DE DADOS ----
data = load_data()
sentiment = analyze_sentiment()

# ---- 3. LAYOUT DO PAINEL ----
tab1, tab2, tab3, tab4 = st.tabs(["📈 Mercado", "📊 Análise", "🐳 Whales", "📑 Relatório"])

with tab1:
    # Gráfico de preço
    if not data['prices'].empty:
        fig = px.line(data['prices'], x="date", y="price", title="Preço do BTC (90 dias)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Candlesticks (dados simulados)
    ohlc_data = pd.DataFrame({
        "date": pd.date_range(end=datetime.today(), periods=30),
        "open": np.random.normal(50000, 1000, 30),
        "high": np.random.normal(51000, 800, 30),
        "low": np.random.normal(49000, 800, 30),
        "close": np.random.normal(50500, 1000, 30)
    })
    fig_candle = go.Figure(go.Candlestick(
        x=ohlc_data['date'],
        open=ohlc_data['open'],
        high=ohlc_data['high'],
        low=ohlc_data['low'],
        close=ohlc_data['close']
    ))
    st.plotly_chart(fig_candle, use_container_width=True)

with tab2:
    # Análise de sentimento
    st.subheader("📢 Sentimento do Mercado")
    st.metric("Score", f"{sentiment['score']:.0%}", sentiment['status'])
    
    # Heatmap de liquidez
    st.subheader("🗺️ Liquidez por Exchange")
    liquidity_data = pd.DataFrame({
        "Exchange": ["Binance", "Coinbase", "Kraken"],
        "Volume (BTC)": [1500, 800, 600]
    })
    fig_heatmap = px.bar(liquidity_data, x="Exchange", y="Volume (BTC)", color="Exchange")
    st.plotly_chart(fig_heatmap, use_container_width=True)

with tab3:
    # Atividade de Whales
    st.subheader("🐳 Movimentações Recentes")
    whales_data = pd.DataFrame({
        "Horário": [datetime.now() - timedelta(hours=h) for h in [1, 3, 5]],
        "Quantidade (BTC)": [250, 180, 120],
        "Exchange": ["Binance", "Coinbase", "Kraken"]
    })
    st.dataframe(whales_data.style.format({
        "Quantidade (BTC)": "{:,.0f}",
        "Horário": lambda x: x.strftime("%d/%m %H:%M")
    }), use_container_width=True)

with tab4:
    # Relatório PDF
    st.subheader("📑 Gerar Relatório")
    if st.button("🖨️ Criar PDF"):
        with st.spinner("Gerando relatório..."):
            pdf_file = generate_pdf()
            st.download_button(
                "⬇️ Baixar Relatório",
                data=pdf_file,
                file_name="relatorio_btc.pdf",
                mime="application/pdf"
            )

# ---- BOTÃO DE ATUALIZAÇÃO ----
if st.sidebar.button("🔄 Atualizar Dados"):
    st.cache_data.clear()
    st.rerun()
