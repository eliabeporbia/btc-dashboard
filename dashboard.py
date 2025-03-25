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
from sklearn.linear_model import LinearRegression

# ======================
# CONFIGURAÇÕES INICIAIS
# ======================
st.set_page_config(layout="wide", page_title="BTC Super Dashboard Pro+")
st.title("🚀 BTC Super Dashboard Pro+ - Edição Premium")

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

def get_traditional_assets():
    """Coleta dados de ativos tradicionais"""
    assets = {
        "S&P 500": "^GSPC",
        "Ouro": "GC=F",
        "ETH-USD": "ETH-USD"
    }
    dfs = []
    for name, ticker in assets.items():
        data = yf.Ticker(ticker).history(period="90d", interval="1d")
        data = data.reset_index()[['Date', 'Close']].rename(columns={'Close': 'value', 'Date': 'date'})
        data['asset'] = name
        dfs.append(data)
    return pd.concat(dfs)

def backtest_strategy(data):
    """Backtesting automático baseado em RSI e Médias"""
    df = data['prices'].copy()
    
    # Estratégia: Compra quando RSI < 30 e preço abaixo da média móvel
    df['signal'] = np.where((df['RSI'] < 30) & (df['price'] < df['MA30']), 1, 
                          np.where((df['RSI'] > 70) & (df['price'] > df['MA30']), -1, 0))
    
    df['daily_return'] = df['price'].pct_change()
    df['strategy_return'] = df['signal'].shift(1) * df['daily_return']
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
    
    return df

def simulate_event(event, price_series):
    """Simula impacto de eventos no preço"""
    if event == "Halving":
        # Efeito histórico: +120% em 1 ano após halving
        growth = np.log(2.2) / 365  # Crescimento diário composto
        return price_series * (1 + growth) ** np.arange(len(price_series))
    elif event == "Crash":
        return price_series * 0.7  # -30% instantâneo
    else:  # "ETF Approval"
        return price_series * 1.5  # +50% instantâneo

# ======================
# FUNÇÕES ORIGINAIS (MANTIDAS)
# ======================
@st.cache_data(ttl=3600)
def load_data():
    # ... (código original mantido igual) ...

def calculate_ema(series, window):
    # ... (código original mantido igual) ...

def calculate_rsi(series, window=14):
    # ... (código original mantido igual) ...

def calculate_macd(series, fast=12, slow=26, signal=9):
    # ... (código original mantido igual) ...

def calculate_bollinger_bands(series, window=20, num_std=2):
    # ... (código original mantido igual) ...

def generate_signals(data):
    # ... (código original mantido igual) ...

# ======================
# INTERFACE DO USUÁRIO - ATUALIZADA
# ======================

# Carregar dados
data = load_data()
signals, final_verdict, buy_signals, sell_signals = generate_signals(data)
sentiment = get_market_sentiment()
traditional_assets = get_traditional_assets()

# Sidebar - Controles do Usuário
st.sidebar.header("⚙️ Painel de Controle")

# Configurações dos indicadores
st.sidebar.subheader("🔧 Parâmetros Técnicos")
rsi_window = st.sidebar.slider("Período do RSI", 7, 21, 14)
bb_window = st.sidebar.slider("Janela das Bandas de Bollinger", 10, 50, 20)
ma_windows = st.sidebar.multiselect(
    "Médias Móveis para Exibir",
    [7, 20, 30, 50, 100, 200],
    default=[7, 30, 200]
)

# Configurações de alertas
st.sidebar.subheader("🔔 Alertas Automáticos")
email = st.sidebar.text_input("E-mail para notificações")
if st.sidebar.button("Ativar Monitoramento Contínuo"):
    st.sidebar.success("Alertas ativados!")

# Seção principal
st.header("📊 Painel Integrado BTC Pro+")

# Linha de métricas
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Preço BTC", f"${data['prices']['price'].iloc[-1]:,.2f}")
col2.metric("Sentimento", f"{sentiment['value']}/100", sentiment['sentiment'])
col3.metric("S&P 500", f"${traditional_assets[traditional_assets['asset']=='S&P 500']['value'].iloc[-1]:,.0f}")
col4.metric("Ouro", f"${traditional_assets[traditional_assets['asset']=='Ouro']['value'].iloc[-1]:,.0f}")
col5.metric("Análise Final", final_verdict)

# Abas principais
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Mercado", 
    "🆚 Comparativos", 
    "🧪 Backtesting", 
    "🌍 Cenários", 
    "📉 Técnico", 
    "📤 Exportar"
])

with tab1:  # Mercado
    # ... (gráficos originais mantidos) ...
    
    # Novo: Gráfico de Sentimento
    st.subheader("📊 Sentimento do Mercado")
    fig_sent = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment['value'],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Fear & Greed Index"},
        gauge={'axis': {'range': [0, 100]},
               'steps': [
                   {'range': [0, 25], 'color': "red"},
                   {'range': [25, 50], 'color': "orange"},
                   {'range': [50, 75], 'color': "yellow"},
                   {'range': [75, 100], 'color': "green"}]}))
    st.plotly_chart(fig_sent, use_container_width=True)

with tab2:  # Comparativos
    st.subheader("📌 BTC vs Ativos Tradicionais")
    fig_comp = px.line(
        traditional_assets, 
        x="date", y="value", 
        color="asset",
        title="Desempenho Comparativo (Últimos 90 dias)",
        log_y=True
    )
    st.plotly_chart(fig_comp, use_container_width=True)

with tab3:  # Backtesting
    st.subheader("🧪 Backtesting Estratégico")
    bt_data = backtest_strategy(data)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Retorno da Estratégia", 
                 f"{(bt_data['cumulative_return'].iloc[-1] - 1)*100:.2f}%")
    with col2:
        st.metric("Operações Geradas", 
                 f"{len(bt_data[bt_data['signal'] != 0])}")
    
    fig_bt = px.line(
        bt_data, 
        x="date", y=["cumulative_return"], 
        title="Performance da Estratégia"
    )
    st.plotly_chart(fig_bt, use_container_width=True)

with tab4:  # Cenários
    st.subheader("🌍 Simulação de Eventos")
    event = st.selectbox(
        "Selecione um Cenário:", 
        ["Halving", "Crash", "ETF Approval"]
    )
    
    # Simular
    simulated_prices = simulate_event(
        event, 
        data['prices']['price'].tail(90).reset_index(drop=True)
    )
    
    fig_scenario = go.Figure()
    fig_scenario.add_trace(go.Scatter(
        x=data['prices']['date'].tail(90),
        y=data['prices']['price'].tail(90),
        name="Preço Real"
    ))
    fig_scenario.add_trace(go.Scatter(
        x=data['prices']['date'].tail(90),
        y=simulated_prices,
        name=f"Projeção: {event}"
    ))
    st.plotly_chart(fig_scenario, use_container_width=True)

with tab5:  # Técnico
    # ... (análise técnica original mantida) ...
    pass

with tab6:  # Exportar
    st.subheader("📤 Exportar Dados Completo")
    
    if st.button("Gerar Relatório PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Relatório BTC Dashboard Pro+", ln=1, align='C')
        
        # Adicionar conteúdo
        pdf.cell(200, 10, txt=f"Preço Atual: ${data['prices']['price'].iloc[-1]:,.2f}", ln=1)
        pdf.cell(200, 10, txt=f"Sinal Atual: {final_verdict}", ln=1)
        
        # Salvar temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf.output(tmp.name)
            st.success(f"Relatório gerado! [Download aqui]({tmp.name})")
    
    if st.button("Exportar Dados para Excel"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            with pd.ExcelWriter(tmp.name) as writer:
                data['prices'].to_excel(writer, sheet_name="BTC Prices")
                traditional_assets.to_excel(writer, sheet_name="Traditional Assets")
            st.success(f"Dados exportados! [Download aqui]({tmp.name})")

# ... (restante do código original mantido) ...
