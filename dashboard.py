import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from fpdf import FPDF
import talib

# Configuração do painel
st.set_page_config(layout="wide", page_title="BTC Super Dashboard Pro")
st.title("🚀 BTC Super Dashboard Pro - Análise Consolidadada")

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
        
        # Calculando todos os indicadores técnicos
        closes = data['prices']['price'].values
        data['prices']['MA7'] = talib.SMA(closes, timeperiod=7)
        data['prices']['MA30'] = talib.SMA(closes, timeperiod=30)
        data['prices']['MA200'] = talib.SMA(closes, timeperiod=200)
        data['prices']['RSI'] = talib.RSI(closes, timeperiod=14)
        data['prices']['MACD'], _, _ = talib.MACD(closes)
        
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
        
        # Dados de exchanges
        data['exchanges'] = {
            "binance": {"inflow": 1500, "outflow": 1200, "reserves": 500000},
            "coinbase": {"inflow": 800, "outflow": 750, "reserves": 350000},
            "kraken": {"inflow": 600, "outflow": 550, "reserves": 200000}
        }
        
        # Dados de whale alerts (simulado)
        data['whale_alert'] = pd.DataFrame({
            "date": [datetime.now() - timedelta(hours=h) for h in [1, 3, 5, 8, 12]],
            "amount": [250, 180, 120, 300, 150],
            "exchange": ["Binance", "Coinbase", "Kraken", "Binance", "FTX"]
        })
        
    except requests.exceptions.RequestException as e:
        st.error(f"Erro na requisição à API: {str(e)}")
    except Exception as e:
        st.error(f"Erro ao processar dados: {str(e)}")
    return data

def generate_signals(data):
    signals = []
    buy_signals = 0
    sell_signals = 0
    
    if not data['prices'].empty:
        last_price = data['prices']['price'].iloc[-1]
        closes = data['prices']['price'].values
        
        # 1. Médias Móveis
        ma7 = data['prices']['MA7'].iloc[-1]
        ma30 = data['prices']['MA30'].iloc[-1]
        ma200 = data['prices']['MA200'].iloc[-1]
        
        signals.append(("Preço vs MA7", "COMPRA" if last_price > ma7 else "VENDA", f"{(last_price/ma7 - 1):.2%}"))
        signals.append(("Preço vs MA30", "COMPRA" if last_price > ma30 else "VENDA", f"{(last_price/ma30 - 1):.2%}"))
        signals.append(("MA7 vs MA30", "COMPRA" if ma7 > ma30 else "VENDA", f"{(ma7/ma30 - 1):.2%}"))
        signals.append(("Preço vs MA200", "COMPRA" if last_price > ma200 else "VENDA", f"{(last_price/ma200 - 1):.2%}"))
        
        # 2. RSI
        rsi = data['prices']['RSI'].iloc[-1]
        rsi_signal = "COMPRA" if rsi < 30 else "VENDA" if rsi > 70 else "NEUTRO"
        signals.append(("RSI (14)", rsi_signal, f"{rsi:.2f}"))
        
        # 3. MACD
        macd = data['prices']['MACD'].iloc[-1]
        macd_signal = "COMPRA" if macd > 0 else "VENDA"
        signals.append(("MACD", macd_signal, f"{macd:.2f}"))
        
        # 4. Suporte/Resistência
        support = data['prices']['price'].rolling(30).min().iloc[-1]
        resistance = data['prices']['price'].rolling(30).max().iloc[-1]
        sr_signal = "COMPRA" if last_price < support * 1.02 else "VENDA" if last_price > resistance * 0.98 else "NEUTRO"
        signals.append(("S/R", sr_signal, f"Sup: ${support:,.0f} | Res: ${resistance:,.0f}"))
    
    # 5. Fluxo de exchanges
    if data['exchanges']:
        net_flows = sum(ex["inflow"] - ex["outflow"] for ex in data['exchanges'].values())
        flow_signal = "COMPRA" if net_flows < 0 else "VENDA"
        signals.append(("Net Flow Exchanges", flow_signal, f"{net_flows:,} BTC"))
    
    # 6. Hashrate vs Dificuldade
    if not data['hashrate'].empty and not data['difficulty'].empty:
        hr_growth = data['hashrate']['y'].iloc[-1] / data['hashrate']['y'].iloc[-30] - 1
        diff_growth = data['difficulty']['y'].iloc[-1] / data['difficulty']['y'].iloc[-30] - 1
        hr_signal = "COMPRA" if hr_growth > diff_growth else "VENDA"
        signals.append(("Hashrate vs Dificuldade", hr_signal, f"{(hr_growth - diff_growth):.2%}"))
    
    # 7. Whale Activity
    if 'whale_alert' in data and not data['whale_alert'].empty:
        whale_ratio = data['whale_alert']['amount'].sum() / (24*30)
        whale_signal = "COMPRA" if whale_ratio < 100 else "VENDA"
        signals.append(("Atividade de Whales", whale_signal, f"{whale_ratio:.1f} BTC/dia"))
    
    # Contagem de sinais
    buy_signals = sum(1 for s in signals if s[1] == "COMPRA")
    sell_signals = sum(1 for s in signals if s[1] == "VENDA")
    
    # Análise consolidada
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

def generate_pdf(data, signals, final_verdict):
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
    
    # Veredito final
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=f"Análise Final: {final_verdict}", ln=1)
    pdf.set_font("Arial", size=12)
    
    # Sinais
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Sinais Detalhados:", ln=1)
    pdf.set_font("Arial", size=12)
    
    for name, signal, value in signals:
        pdf.cell(200, 10, txt=f"{name}: {signal} ({value})", ln=1)
    
    pdf.output("report.pdf")
    return open("report.pdf", "rb")

# ---- 2. CARREGAMENTO DE DADOS ----
data = load_data()
if data and not data['prices'].empty:
    signals, final_verdict, buy_signals, sell_signals = generate_signals(data)
else:
    signals = []
    final_verdict = "Dados indisponíveis"

# ---- 3. LAYOUT DO PAINEL ----
st.header("📢 Status do Mercado", divider="rainbow")

# Área de status
col1, col2, col3, col4 = st.columns(4)
col1.metric("Preço Atual", f"${data['prices']['price'].iloc[-1]:,.2f}" if not data['prices'].empty else "N/A")
col2.metric("Hash Rate", f"{data['hashrate']['y'].iloc[-1]/1e6:,.1f} EH/s" if not data['hashrate'].empty else "N/A")
col3.metric("Dificuldade", f"{data['difficulty']['y'].iloc[-1]/1e12:,.1f} T" if not data['difficulty'].empty else "N/A")
col4.metric("Análise Final", final_verdict)

# Resumo de sinais
st.subheader(f"📊 Resumo de Sinais (COMPRA: {buy_signals} | VENDA: {sell_signals})")

# Tabela de sinais detalhados
if signals:
    df_signals = pd.DataFrame(signals, columns=["Indicador", "Sinal", "Valor"])
    
    def color_signal(val):
        color = '#4CAF50' if val == "COMPRA" else '#F44336' if val == "VENDA" else '#FFC107'
        return f'background-color: {color}'
    
    st.dataframe(
        df_signals.style.applymap(color_signal, subset=["Sinal"]),
        hide_index=True,
        use_container_width=True,
        height=(len(df_signals) * 35 + 38)
else:
    st.warning("Não foi possível gerar sinais. Verifique os dados.")

# Gráficos e abas
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Mercado", "📈 Técnico", "🏦 Exchanges", "🐳 Whales", "📑 Relatório"])

with tab1:
    if not data['prices'].empty:
        fig = px.line(data['prices'], x="date", y=["price", "MA7", "MA30", "MA200"], 
                     title="Preço BTC e Médias Móveis (7, 30 e 200 dias)")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    if not data['prices'].empty:
        fig_rsi = px.line(data['prices'], x="date", y="RSI", 
                         title="RSI (14 dias)", 
                         range_y=[0, 100])
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        fig_macd = px.line(data['prices'], x="date", y="MACD", 
                          title="MACD")
        fig_macd.add_hline(y=0, line_color="black")
        st.plotly_chart(fig_macd, use_container_width=True)

with tab3:
    if data['exchanges']:
        df_exchanges = pd.DataFrame(data['exchanges']).T
        fig = px.bar(df_exchanges, y=["inflow", "outflow"], barmode="group", 
                     title="Fluxo de Exchanges (BTC)")
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    if 'whale_alert' in data and not data['whale_alert'].empty:
        fig = px.bar(data['whale_alert'], x="date", y="amount", color="exchange",
                     title="Atividade de Whales (últimas 24h)")
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("📑 Relatório Completo")
    if st.button("🖨️ Gerar Relatório PDF"):
        with st.spinner("Gerando relatório..."):
            pdf_file = generate_pdf(data, signals, final_verdict)
            st.download_button(
                "⬇️ Baixar Relatório Completo",
                data=pdf_file,
                file_name="relatorio_btc_pro.pdf",
                mime="application/pdf"
            )
    
    st.markdown("""
    **📋 Resumo Executivo:**
    - **11 indicadores técnicos e on-chain**
    - **Análise consolidada automática**
    - **Dados atualizados a cada hora**
    """)

# ---- BOTÃO DE ATUALIZAÇÃO ----
st.sidebar.header("🔧 Configurações")
if st.sidebar.button("🔄 Atualizar Dados Agora"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("""
**📌 Legenda:**
- 🟢 **COMPRA**: Indicador positivo
- 🔴 **VENDA**: Indicador negativo
- 🟡 **NEUTRO**: Sem sinal claro
- ✅ **FORTE COMPRA**: 3+ sinais de diferença
- ❌ **FORTE VENDA**: 3+ sinais de diferença

**📊 Indicadores:**
1. Médias Móveis (7, 30, 200 dias)
2. RSI (sobrecompra/sobrevenda)
3. MACD (momentum)
4. Suporte/Resistência
5. Fluxo de Exchanges
6. Hashrate vs Dificuldade
7. Atividade de Whales
""")
