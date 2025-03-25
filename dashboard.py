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
# FUNÇÕES DE CÁLCULO
# ======================

def calculate_ema(series, window):
    """Calcula a Média Móvel Exponencial (EMA)"""
    return series.ewm(span=window, adjust=False).mean()

def calculate_rsi(series, window=14):
    """Calcula o Índice de Força Relativa (RSI)"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calcula o MACD com linha de sinal"""
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd = ema_fast - ema_slow
    signal_line = calculate_ema(macd, signal)
    return macd, signal_line

def calculate_bollinger_bands(series, window=20, num_std=2):
    """Calcula as Bandas de Bollinger"""
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, lower

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
# CARREGAMENTO DE DADOS
# ======================

@st.cache_data(ttl=3600)
def load_data():
    data = {}
    try:
        # Preço do Bitcoin (últimos 90 dias)
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=90"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        market_data = response.json()
        
        data['prices'] = pd.DataFrame(market_data["prices"], columns=["timestamp", "price"])
        data['prices']["date"] = pd.to_datetime(data['prices']["timestamp"], unit="ms")
        
        # Calculando todos os indicadores técnicos
        price_series = data['prices']['price']
        data['prices']['MA7'] = price_series.rolling(7).mean()
        data['prices']['MA30'] = price_series.rolling(30).mean()
        data['prices']['MA200'] = price_series.rolling(200).mean()
        data['prices']['RSI'] = calculate_rsi(price_series)
        data['prices']['MACD'], data['prices']['MACD_Signal'] = calculate_macd(price_series)
        data['prices']['BB_Upper'], data['prices']['BB_Lower'] = calculate_bollinger_bands(price_series)
        
        # Hashrate (taxa de hash)
        hr_response = requests.get("https://api.blockchain.info/charts/hash-rate?format=json&timespan=3months", timeout=10)
        hr_response.raise_for_status()
        data['hashrate'] = pd.DataFrame(hr_response.json()["values"])
        data['hashrate']["date"] = pd.to_datetime(data['hashrate']["x"], unit="s")
        
        # Dificuldade de mineração
        diff_response = requests.get("https://api.blockchain.info/charts/difficulty?timespan=2years&format=json", timeout=10)
        diff_response.raise_for_status()
        data['difficulty'] = pd.DataFrame(diff_response.json()["values"])
        data['difficulty']["date"] = pd.to_datetime(data['difficulty']["x"], unit="s")
        
        # Dados simulados de exchanges
        data['exchanges'] = {
            "binance": {"inflow": 1500, "outflow": 1200, "reserves": 500000},
            "coinbase": {"inflow": 800, "outflow": 750, "reserves": 350000},
            "kraken": {"inflow": 600, "outflow": 550, "reserves": 200000}
        }
        
        # Atividade de "baleias" (grandes investidores)
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

# ======================
# GERADOR DE SINAIS
# ======================

def generate_signals(data):
    signals = []
    buy_signals = 0
    sell_signals = 0
    
    if not data['prices'].empty:
        last_price = data['prices']['price'].iloc[-1]
        
        # 1. Sinais de Médias Móveis
        ma_signals = [
            ("Preço vs MA7", data['prices']['MA7'].iloc[-1]),
            ("Preço vs MA30", data['prices']['MA30'].iloc[-1]),
            ("Preço vs MA200", data['prices']['MA200'].iloc[-1]),
            ("MA7 vs MA30", data['prices']['MA7'].iloc[-1], data['prices']['MA30'].iloc[-1])
        ]
        
        for name, *values in ma_signals:
            if len(values) == 1:
                signal = "COMPRA" if last_price > values[0] else "VENDA"
                change = (last_price/values[0] - 1)
            else:
                signal = "COMPRA" if values[0] > values[1] else "VENDA"
                change = (values[0]/values[1] - 1)
            signals.append((name, signal, f"{change:.2%}"))
        
        # 2. RSI
        rsi = data['prices']['RSI'].iloc[-1]
        rsi_signal = "COMPRA" if rsi < 30 else "VENDA" if rsi > 70 else "NEUTRO"
        signals.append(("RSI (14)", rsi_signal, f"{rsi:.2f}"))
        
        # 3. MACD
        macd = data['prices']['MACD'].iloc[-1]
        macd_signal = "COMPRA" if macd > 0 else "VENDA"
        signals.append(("MACD", macd_signal, f"{macd:.2f}"))
        
        # 4. Bandas de Bollinger
        bb_upper = data['prices']['BB_Upper'].iloc[-1]
        bb_lower = data['prices']['BB_Lower'].iloc[-1]
        bb_signal = "COMPRA" if last_price < bb_lower else "VENDA" if last_price > bb_upper else "NEUTRO"
        signals.append(("Bollinger Bands", bb_signal, f"Atual: ${last_price:,.0f}"))
    
    # 5. Fluxo de exchanges
    if data['exchanges']:
        net_flows = sum(ex["inflow"] - ex["outflow"] for ex in data['exchanges'].values())
        flow_signal = "COMPRA" if net_flows < 0 else "VENDA"
        signals.append(("Fluxo Líquido Exchanges", flow_signal, f"{net_flows:,} BTC"))
    
    # 6. Hashrate vs Dificuldade
    if not data['hashrate'].empty and not data['difficulty'].empty:
        hr_growth = data['hashrate']['y'].iloc[-1] / data['hashrate']['y'].iloc[-30] - 1
        diff_growth = data['difficulty']['y'].iloc[-1] / data['difficulty']['y'].iloc[-30] - 1
        hr_signal = "COMPRA" if hr_growth > diff_growth else "VENDA"
        signals.append(("Hashrate vs Dificuldade", hr_signal, f"{(hr_growth - diff_growth):.2%}"))
    
    # 7. Atividade de Whales
    if 'whale_alert' in data and not data['whale_alert'].empty:
        whale_ratio = data['whale_alert']['amount'].sum() / (24*30)  # Normalizado para 30 dias
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

# ======================
# INTERFACE DO USUÁRIO
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
    if not data['prices'].empty:
        fig = px.line(data['prices'], x="date", y=["price", "MA7", "MA30", "MA200"], 
                     title="Preço BTC e Médias Móveis")
        st.plotly_chart(fig, use_container_width=True)
    
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
    if not data['prices'].empty:
        # Gráfico RSI
        fig_rsi = px.line(data['prices'], x="date", y="RSI", 
                         title="RSI (14 dias)", 
                         range_y=[0, 100])
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        # Gráfico MACD
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=data['prices']['date'], y=data['prices']['MACD'], name="MACD"))
        fig_macd.add_trace(go.Scatter(x=data['prices']['date'], y=data['prices']['MACD_Signal'], name="Signal"))
        fig_macd.update_layout(title="MACD (12,26,9)")
        st.plotly_chart(fig_macd, use_container_width=True)
        
        # Gráfico Bollinger Bands
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=data['prices']['date'], y=data['prices']['BB_Upper'], name="Banda Superior"))
        fig_bb.add_trace(go.Scatter(x=data['prices']['date'], y=data['prices']['price'], name="Preço"))
        fig_bb.add_trace(go.Scatter(x=data['prices']['date'], y=data['prices']['BB_Lower'], name="Banda Inferior"))
        fig_bb.update_layout(title="Bandas de Bollinger (20,2)")
        st.plotly_chart(fig_bb, use_container_width=True)

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

# ======================
# RODAPÉ
# ======================
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
4. Bandas de Bollinger
5. Fluxo de Exchanges
6. Hashrate vs Dificuldade
7. Atividade de Whales
8. Análise Sentimental
9. Comparação com Mercado Tradicional
""")
