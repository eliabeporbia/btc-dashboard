import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from fpdf import FPDF

# Configuração do painel
st.set_page_config(layout="wide", page_title="BTC Super Dashboard Pro")
st.title("🚀 BTC Super Dashboard Pro - Análise Avançada")

# ---- 1. FUNÇÕES PRINCIPAIS ----
@st.cache_data(ttl=3600)
def load_data():
    data = {}
    try:
        # Preço e volume
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=90"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        market_data = response.json()
        
        data['prices'] = pd.DataFrame(market_data["prices"], columns=["timestamp", "price"])
        data['prices']["date"] = pd.to_datetime(data['prices']["timestamp"], unit="ms")
        data['prices']['MA7'] = data['prices']['price'].rolling(7).mean()
        data['prices']['MA30'] = data['prices']['price'].rolling(30).mean()
        
        # Hashrate
        hr_response = requests.get("https://api.blockchain.info/charts/hash-rate?format=json&timespan=3months", timeout=10)
        hr_response.raise_for_status()
        data['hashrate'] = pd.DataFrame(hr_response.json()["values"])
        data['hashrate']["date"] = pd.to_datetime(data['hashrate']["x"], unit="s")
        
        # Dificuldade
        diff_response = requests.get("https://api.blockchain.info/charts/difficulty?timespan=2years&format=json", timeout=10)
        diff_response.raise_for_status()
        data['difficulty'] = pd.DataFrame(diff_response.json()["values"])
        data['difficulty']["date"] = pd.to_datetime(data['difficulty']["x"], unit="s")
        
        # Dados de exchanges (simulados + real)
        data['exchanges'] = {
            "binance": {"inflow": 1500, "outflow": 1200, "reserves": 500000},
            "coinbase": {"inflow": 800, "outflow": 750, "reserves": 350000},
            "kraken": {"inflow": 600, "outflow": 550, "reserves": 200000}
        }
        
    except requests.exceptions.RequestException as e:
        st.error(f"Erro na requisição à API: {str(e)}")
    except Exception as e:
        st.error(f"Erro ao processar dados: {str(e)}")
    return data

def generate_signals(data):
    signals = []
    
    # 1. Tendência de preço (Médias Móveis)
    if not data['prices'].empty:
        last_price = data['prices']['price'].iloc[-1]
        ma7 = data['prices']['MA7'].iloc[-1]
        ma30 = data['prices']['MA30'].iloc[-1]
        
        signals.append(("Preço vs MA7", "COMPRA" if last_price > ma7 else "VENDA", f"{(last_price/ma7 - 1):.2%}"))
        signals.append(("Preço vs MA30", "COMPRA" if last_price > ma30 else "VENDA", f"{(last_price/ma30 - 1):.2%}"))
        signals.append(("MA7 vs MA30", "COMPRA" if ma7 > ma30 else "VENDA", f"{(ma7/ma30 - 1):.2%}"))
    
    # 2. Fluxo de exchanges
    if data['exchanges']:
        net_flows = sum(ex["inflow"] - ex["outflow"] for ex in data['exchanges'].values())
        signals.append(("Net Flow Exchanges", "COMPRA" if net_flows < 0 else "VENDA", f"{net_flows:,} BTC"))
    
    # 3. Hashrate vs Dificuldade
    if not data['hashrate'].empty and not data['difficulty'].empty:
        hr_growth = data['hashrate']['y'].iloc[-1] / data['hashrate']['y'].iloc[-30] - 1
        diff_growth = data['difficulty']['y'].iloc[-1] / data['difficulty']['y'].iloc[-30] - 1
        signals.append(("Hashrate vs Dificuldade", "COMPRA" if hr_growth > diff_growth else "VENDA", f"{(hr_growth - diff_growth):.2%}"))
    
    return signals

def generate_pdf(data, signals):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Cabeçalho
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt=f"Relatório BTC Pro - {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=1)
    pdf.set_font("Arial", size=12)
    
    # Dados principais
    pdf.cell(200, 10, txt=f"Preço Atual: ${data['prices']['price'].iloc[-1]:,.2f}", ln=1)
    pdf.cell(200, 10, txt=f"Hashrate: {data['hashrate']['y'].iloc[-1]/1e6:,.1f} EH/s", ln=1)
    pdf.cell(200, 10, txt=f"Dificuldade: {data['difficulty']['y'].iloc[-1]/1e12:,.1f} T", ln=1)
    
    # Sinais
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Sinais de Mercado:", ln=1)
    pdf.set_font("Arial", size=12)
    
    for name, signal, value in signals:
        pdf.cell(200, 10, txt=f"{name}: {signal} ({value})", ln=1)
    
    pdf.output("report.pdf")
    return open("report.pdf", "rb")

# ---- 2. CARREGAMENTO DE DADOS ----
data = load_data()
signals = generate_signals(data) if data else []

# ---- 3. LAYOUT DO PAINEL ----
st.header("📢 Status do Mercado", divider="rainbow")

# Área de status
col1, col2, col3 = st.columns(3)
col1.metric("Preço Atual", f"${data['prices']['price'].iloc[-1]:,.2f}" if not data['prices'].empty else "N/A")
col2.metric("Hash Rate", f"{data['hashrate']['y'].iloc[-1]/1e6:,.1f} EH/s" if not data['hashrate'].empty else "N/A")
col3.metric("Dificuldade", f"{data['difficulty']['y'].iloc[-1]/1e12:,.1f} T" if not data['difficulty'].empty else "N/A")

# Tabela de sinais
st.subheader("📈 Sinais de Compra/Venda")
if signals:
    df_signals = pd.DataFrame(signals, columns=["Indicador", "Sinal", "Valor"])
    st.dataframe(
        df_signals.style.applymap(
            lambda x: "background-color: #4CAF50" if x == "COMPRA" else "background-color: #F44336", 
            subset=["Sinal"]
        ),
        hide_index=True,
        use_container_width=True
    )
else:
    st.warning("Não foi possível gerar sinais. Verifique os dados.")

# Gráficos e abas
tab1, tab2, tab3, tab4 = st.tabs(["📊 Mercado", "🏦 Exchanges", "⛏️ Mineração", "📑 Relatório"])

with tab1:
    if not data['prices'].empty:
        fig = px.line(data['prices'], x="date", y=["price", "MA7", "MA30"], 
                     title="Preço BTC e Médias Móveis")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    if data['exchanges']:
        df_exchanges = pd.DataFrame(data['exchanges']).T
        fig = px.bar(df_exchanges, y=["inflow", "outflow"], barmode="group", 
                     title="Fluxo de Exchanges (BTC)")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    if not data['hashrate'].empty and not data['difficulty'].empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['hashrate']['date'], y=data['hashrate']['y']/1e6, 
                       name="Hashrate (EH/s)"))
        fig.add_trace(go.Scatter(x=data['difficulty']['date'], y=data['difficulty']['y']/1e12, 
                       name="Dificuldade (T)", yaxis="y2"))
        fig.update_layout(
            title="Hashrate vs Dificuldade",
            yaxis=dict(title="Hashrate (EH/s)"),
            yaxis2=dict(title="Dificuldade (T)", overlaying="y", side="right")
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("📑 Gerar Relatório Completo")
    if st.button("🖨️ Criar PDF"):
        with st.spinner("Gerando relatório..."):
            pdf_file = generate_pdf(data, signals)
            st.download_button(
                "⬇️ Baixar Relatório",
                data=pdf_file,
                file_name="relatorio_btc_pro.pdf",
                mime="application/pdf"
            )

# ---- BOTÃO DE ATUALIZAÇÃO ----
st.sidebar.header("Configurações")
if st.sidebar.button("🔄 Atualizar Dados"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("""
**📌 Legenda:**
- 🟢 **COMPRA**: Indicador positivo
- 🔴 **VENDA**: Indicador negativo
- 📈 **MA7/MA30**: Médias móveis de 7 e 30 dias
""")
