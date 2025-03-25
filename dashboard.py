import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np

# Configuração do painel
st.set_page_config(layout="wide", page_title="Analisador BTC Profissional")
st.title("📈 Analisador BTC - Indicadores em Tempo Real")

# ---- CONSTANTES ----
EXCHANGES = ["binance", "coinbase", "kraken", "bybit", "okx"]
COLORS = {"binance": "#F0B90B", "coinbase": "#0052FF", "kraken": "#582C87", "bybit": "#FFD100", "okx": "#00296B"}

# ---- 1. PREÇO E MERCADO ----
@st.cache_data(ttl=3600)
def get_market_data():
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=90"
        data = requests.get(url, timeout=10).json()
        prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        prices["date"] = pd.to_datetime(prices["timestamp"], unit="ms")
        
        # Cálculo de tendência (média móvel 7 dias)
        prices["MA7"] = prices["price"].rolling(7).mean()
        prices["trend"] = np.where(prices["price"] > prices["MA7"], "alta", "baixa")
        
        return prices
    except Exception as e:
        st.error(f"Erro mercado: {str(e)}")
        return pd.DataFrame()

# ---- 2. DADOS DE EXCHANGES ----
@st.cache_data(ttl=3600)
def get_exchanges_data():
    results = {}
    for exchange in EXCHANGES:
        try:
            url = f"https://api.coingecko.com/api/v3/exchanges/{exchange}/tickers?coin_id=bitcoin"
            data = requests.get(url, timeout=10).json()
            
            # Métricas-chave
            volume_btc = data["tickers"][0]["converted_volume"]["btc"]
            spread = abs(data["tickers"][0]["bid_ask_spread_percentage"])
            
            results[exchange] = {
                "volume": volume_btc,
                "spread": spread,
                "inflow": volume_btc * 0.3,  # Estimativa
                "outflow": volume_btc * 0.25
            }
        except:
            results[exchange] = {"volume": 0, "spread": 0, "inflow": 0, "outflow": 0}
    
    return results

# ---- 3. INDICADORES ON-CHAIN ----
@st.cache_data(ttl=3600)
def get_onchain_indicators():
    try:
        # Dados de dificuldade
        url = "https://blockchain.info/q/getdifficulty"
        difficulty = float(requests.get(url, timeout=10).text)
        
        # Hashrate (estimado)
        hashrate = difficulty / (600 * 1e12)  # TH/s
        
        return {
            "hashrate": hashrate,
            "difficulty": difficulty,
            "status": "alta" if hashrate > 500000 else "baixa"
        }
    except Exception as e:
        st.error(f"Erro on-chain: {str(e)}")
        return {"hashrate": 0, "difficulty": 0, "status": "neutro"}

# ---- CARREGAMENTO DE DADOS ----
with st.spinner("Analisando mercado..."):
    df_price = get_market_data()
    exchanges_data = get_exchanges_data()
    onchain = get_onchain_indicators()

# ---- SISTEMA DE ALERTAS ----
def generate_signal():
    # Fatores de ponderação
    price_trend = 0.4
    volume_trend = 0.3
    onchain_trend = 0.3
    
    # 1. Tendência de preço
    last_trend = df_price["trend"].iloc[-1]
    price_score = 1 if last_trend == "alta" else -1
    
    # 2. Volume nas exchanges
    total_volume = sum(ex["volume"] for ex in exchanges_data.values())
    volume_score = 1 if total_volume > 50000 else -1  # 50k BTC como limiar
    
    # 3. Saúde da rede
    onchain_score = 1 if onchain["status"] == "alta" else -1
    
    # Cálculo final
    total_score = (price_score * price_trend + 
                 volume_score * volume_trend + 
                 onchain_score * onchain_trend)
    
    if total_score > 0.5:
        return "📈 FORTE ALTA", "green"
    elif total_score > 0:
        return "📈 Tendência de Alta", "lightgreen"
    elif total_score < -0.5:
        return "📉 FORTE BAIXA", "red"
    else:
        return "📉 Tendência de Baixa", "orange"

signal, color = generate_signal()

# ---- LAYOUT PRINCIPAL ----
st.markdown(f"""
    <div style='background-color:{color}; padding:10px; border-radius:5px; text-align:center'>
        <h2 style='color:white; margin:0;'>{signal}</h2>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---- ABAS ----
tab1, tab2, tab3 = st.tabs(["📊 Mercado", "🏦 Exchanges", "⚙️ Rede Bitcoin"])

with tab1:
    if not df_price.empty:
        fig = px.line(df_price, x="date", y=["price", "MA7"], 
                     title="Preço BTC vs Média Móvel 7 Dias",
                     color_discrete_map={"price": "blue", "MA7": "orange"})
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar últimas tendências
        st.subheader("Últimas Tendências")
        cols = st.columns(5)
        for i in range(1, 6):
            day = df_price.iloc[-i]
            cols[i-1].metric(
                f"{day['date'].strftime('%d/%m')}",
                f"${day['price']:,.0f}",
                "Alta" if day["trend"] == "alta" else "Baixa"
            )

with tab2:
    st.subheader("Volume 24h por Exchange")
    
    # Gráfico de volume
    volume_data = pd.DataFrame.from_dict(exchanges_data, orient="index")
    fig_volume = px.bar(volume_data, x=volume_data.index, y="volume",
                       color=volume_data.index, color_discrete_map=COLORS)
    st.plotly_chart(fig_volume, use_container_width=True)
    
    # Tabela detalhada
    st.subheader("Métricas Detalhadas")
    df_exchanges = pd.DataFrame(exchanges_data).T
    st.dataframe(df_exchanges.style.format({
        "volume": "{:,.0f} BTC",
        "spread": "{:.2f}%",
        "inflow": "{:,.0f} BTC",
        "outflow": "{:,.0f} BTC"
    }), use_container_width=True)

with tab3:
    st.subheader("Saúde da Rede Bitcoin")
    
    cols = st.columns(3)
    cols[0].metric("Hash Rate", f"{onchain['hashrate']:,.0f} TH/s", 
                  "Forte" if onchain["status"] == "alta" else "Fraco")
    cols[1].metric("Dificuldade", f"{onchain['difficulty']/1e12:,.2f} T")
    cols[2].metric("Status", onchain["status"].upper())
    
    # Gráfico de dificuldade histórica
    st.subheader("Dificuldade da Rede")
    try:
        url = "https://blockchain.info/charts/difficulty?format=json"
        data = requests.get(url, timeout=10).json()
        df_diff = pd.DataFrame(data["values"])
        df_diff["date"] = pd.to_datetime(df_diff["x"], unit="s")
        fig_diff = px.line(df_diff, x="date", y="y", 
                          title="Dificuldade de Mineração")
        st.plotly_chart(fig_diff, use_container_width=True)
    except:
        st.warning("Dados históricos não disponíveis")

# ---- RODAPÉ ----
st.sidebar.header("Configurações")
if st.sidebar.button("Atualizar Dados Agora"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("""
**Legenda dos Sinais**:
- 📈 FORTE ALTA: Todos os indicadores positivos
- 📈 Tendência de Alta: Maioria positiva
- 📉 Tendência de Baixa: Maioria negativa
- 📉 FORTE BAIXA: Todos negativos
""")
