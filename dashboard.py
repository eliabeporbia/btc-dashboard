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
from sklearn.model_selection import ParameterGrid
from itertools import product

# ======================
# CONFIGURAÇÕES INICIAIS
# ======================
st.set_page_config(layout="wide", page_title="BTC Super Dashboard Pro+")
st.title("🚀 BTC Super Dashboard Pro+ - Edição Premium")

# ======================
# NOVAS FUNÇÕES DE BACKTESTING AVANÇADO
# ======================

def calculate_daily_returns(df):
    """Calcula retornos diários e cumulativos"""
    df['daily_return'] = df['price'].pct_change()
    df['cumulative_return'] = (1 + df['daily_return']).cumprod()
    return df

def calculate_strategy_returns(df, signal_col='signal'):
    """Calcula retornos da estratégia baseada em coluna de sinais"""
    df['strategy_return'] = df[signal_col].shift(1) * df['daily_return']
    df['strategy_cumulative'] = (1 + df['strategy_return']).cumprod()
    return df

def backtest_rsi_strategy(df, rsi_window=14, overbought=70, oversold=30):
    """Estratégia avançada de RSI com zonas personalizadas"""
    df = df.copy()
    df['RSI'] = calculate_rsi(df['price'], rsi_window)
    
    # Sinais mais sofisticados com confirmação
    df['signal'] = 0
    df.loc[(df['RSI'] < oversold) & (df['price'] > df['MA30']), 'signal'] = 1
    df.loc[(df['RSI'] > overbought) & (df['price'] < df['MA30']), 'signal'] = -1
    
    return calculate_strategy_returns(df)

def backtest_macd_strategy(df, fast=12, slow=26, signal=9):
    """Estratégia MACD cruzamento de linha zero e linha de sinal"""
    df = df.copy()
    df['MACD'], df['MACD_Signal'] = calculate_macd(df['price'], fast, slow, signal)
    
    # Cruzamento de linha zero
    df['signal'] = 0
    df.loc[df['MACD'] > 0, 'signal'] = 1
    df.loc[df['MACD'] < 0, 'signal'] = -1
    
    # Cruzamento de linha de sinal (sobrescreve se for mais forte)
    df.loc[(df['MACD'] > df['MACD_Signal']) & (df['MACD'] > 0), 'signal'] = 1.5  # Compra forte
    df.loc[(df['MACD'] < df['MACD_Signal']) & (df['MACD'] < 0), 'signal'] = -1.5 # Venda forte
    
    return calculate_strategy_returns(df)

def backtest_bollinger_strategy(df, window=20, num_std=2):
    """Estratégia Bandas de Bollinger com saída progressiva"""
    df = df.copy()
    df['BB_Upper'], df['BB_Lower'] = calculate_bollinger_bands(df['price'], window, num_std)
    df['MA'] = df['price'].rolling(window).mean()
    
    df['signal'] = 0
    # Entrada quando toca banda inferior
    df.loc[df['price'] < df['BB_Lower'], 'signal'] = 1
    # Saída progressiva - 50% na média, 50% na banda superior
    df.loc[(df['price'] > df['MA']) & (df['signal'].shift(1) == 1), 'signal'] = 0.5
    df.loc[df['price'] > df['BB_Upper'], 'signal'] = -1  # Venda se tocar banda superior
    
    return calculate_strategy_returns(df)

def backtest_ema_cross_strategy(df, short_window=9, long_window=21):
    """Estratégia de cruzamento de EMAs"""
    df = df.copy()
    df['EMA_Short'] = calculate_ema(df['price'], short_window)
    df['EMA_Long'] = calculate_ema(df['price'], long_window)
    
    df['signal'] = 0
    df.loc[df['EMA_Short'] > df['EMA_Long'], 'signal'] = 1  # Compra quando EMA curta cruza acima
    df.loc[df['EMA_Short'] < df['EMA_Long'], 'signal'] = -1 # Venda quando EMA curta cruza abaixo
    
    return calculate_strategy_returns(df)

def calculate_metrics(df):
    """Calcula métricas avançadas de performance"""
    metrics = {}
    returns = df['strategy_return'].dropna()
    buy_hold_returns = df['daily_return'].dropna()
    
    # Retornos
    metrics['Retorno Estratégia'] = df['strategy_cumulative'].iloc[-1] - 1
    metrics['Retorno Buy & Hold'] = df['cumulative_return'].iloc[-1] - 1
    
    # Volatilidade
    metrics['Vol Estratégia'] = returns.std() * np.sqrt(365)
    metrics['Vol Buy & Hold'] = buy_hold_returns.std() * np.sqrt(365)
    
    # Razão Sharpe (assumindo risco zero)
    metrics['Sharpe Estratégia'] = returns.mean() / returns.std() * np.sqrt(365)
    metrics['Sharpe Buy & Hold'] = buy_hold_returns.mean() / buy_hold_returns.std() * np.sqrt(365)
    
    # Drawdown
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.expanding(min_periods=1).max()
    drawdown = (cum_returns - peak) / peak
    metrics['Max Drawdown'] = drawdown.min()
    
    # Win Rate
    metrics['Win Rate'] = len(returns[returns > 0]) / len(returns)
    
    # Taxa de Acerto
    trades = df[df['signal'] != 0]
    if len(trades) > 0:
        metrics['Taxa Acerto'] = len(trades[trades['strategy_return'] > 0]) / len(trades)
    else:
        metrics['Taxa Acerto'] = 0
    
    return metrics

def optimize_strategy_parameters(data, strategy_name, param_space):
    """Otimiza os parâmetros de uma estratégia específica"""
    best_sharpe = -np.inf
    best_params = None
    best_results = None
    
    # Gerar todas combinações de parâmetros
    param_combinations = list(ParameterGrid(param_space))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, params in enumerate(param_combinations):
        try:
            # Executar backtest com os parâmetros atuais
            if strategy_name == 'RSI':
                df = backtest_rsi_strategy(data['prices'], **params)
            elif strategy_name == 'MACD':
                df = backtest_macd_strategy(data['prices'], **params)
            elif strategy_name == 'Bollinger':
                df = backtest_bollinger_strategy(data['prices'], **params)
            elif strategy_name == 'EMA Cross':
                df = backtest_ema_cross_strategy(data['prices'], **params)
            
            # Calcular métricas
            returns = df['strategy_return'].dropna()
            if len(returns) > 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(365)
                
                # Atualizar melhor combinação se necessário
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params
                    best_results = df
        except:
            continue
        
        # Atualizar barra de progresso
        progress = (i + 1) / len(param_combinations)
        progress_bar.progress(progress)
        status_text.text(f"Testando combinação {i+1}/{len(param_combinations)} | Melhor Sharpe: {best_sharpe:.2f}")
    
    progress_bar.empty()
    status_text.empty()
    
    return best_params, best_sharpe, best_results

# ======================
# FUNÇÕES ORIGINAIS (MANTIDAS)
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

def backtest_strategy(data, rsi_window=14):
    """Backtesting automático baseado em RSI e Médias"""
    df = data['prices'].copy()
    
    # Calcular RSI com período personalizado
    rsi_col = f'RSI_{rsi_window}'
    if rsi_col not in df:
        df[rsi_col] = calculate_rsi(df['price'], rsi_window)
    
    # Estratégia: Compra quando RSI < 30 e preço abaixo da média móvel
    df['signal'] = np.where(
        (df[rsi_col] < 30) & (df['price'] < df['MA30']), 1, 
        np.where((df[rsi_col] > 70) & (df['price'] > df['MA30']), -1, 0))
    
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
        
        # Calculando todos os indicadores técnicos básicos
        price_series = data['prices']['price']
        data['prices']['MA7'] = price_series.rolling(7).mean()
        data['prices']['MA30'] = price_series.rolling(30).mean()
        data['prices']['MA200'] = price_series.rolling(200).mean()
        data['prices']['RSI_14'] = calculate_rsi(price_series, 14)  # RSI padrão
        data['prices']['MACD'], data['prices']['MACD_Signal'] = calculate_macd(price_series)
        data['prices']['BB_Upper_20'], data['prices']['BB_Lower_20'] = calculate_bollinger_bands(price_series, 20)
        
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

def generate_signals(data, rsi_window=14, bb_window=20):
    signals = []
    buy_signals = 0
    sell_signals = 0
    
    if not data['prices'].empty:
        last_price = data['prices']['price'].iloc[-1]
        
        # 1. Sinais de Médias Móveis (usando as médias selecionadas)
        ma_signals = []
        for window in st.session_state.user_settings['ma_windows']:
            col_name = f'MA{window}'
            if col_name not in data['prices']:
                data['prices'][col_name] = data['prices']['price'].rolling(window).mean()
            ma_signals.append((f"Preço vs MA{window}", data['prices'][col_name].iloc[-1]))
        
        # Adicionar comparação entre médias
        if len(st.session_state.user_settings['ma_windows']) > 1:
            ma1 = st.session_state.user_settings['ma_windows'][0]
            ma2 = st.session_state.user_settings['ma_windows'][1]
            ma_signals.append((f"MA{ma1} vs MA{ma2}", 
                             data['prices'][f'MA{ma1}'].iloc[-1], 
                             data['prices'][f'MA{ma2}'].iloc[-1]))
        
        for name, *values in ma_signals:
            if len(values) == 1:
                signal = "COMPRA" if last_price > values[0] else "VENDA"
                change = (last_price/values[0] - 1)
            else:
                signal = "COMPRA" if values[0] > values[1] else "VENDA"
                change = (values[0]/values[1] - 1)
            signals.append((name, signal, f"{change:.2%}"))
        
        # 2. RSI com período personalizado
        rsi_col = f'RSI_{rsi_window}'
        if rsi_col not in data['prices']:
            data['prices'][rsi_col] = calculate_rsi(data['prices']['price'], rsi_window)
        rsi = data['prices'][rsi_col].iloc[-1]
        rsi_signal = "COMPRA" if rsi < 30 else "VENDA" if rsi > 70 else "NEUTRO"
        signals.append((f"RSI ({rsi_window})", rsi_signal, f"{rsi:.2f}"))
        
        # 3. MACD
        macd = data['prices']['MACD'].iloc[-1]
        macd_signal = "COMPRA" if macd > 0 else "VENDA"
        signals.append(("MACD", macd_signal, f"{macd:.2f}"))
        
        # 4. Bandas de Bollinger com janela personalizada
        bb_upper_col = f'BB_Upper_{bb_window}'
        bb_lower_col = f'BB_Lower_{bb_window}'
        if bb_upper_col not in data['prices']:
            data['prices'][bb_upper_col], data['prices'][bb_lower_col] = calculate_bollinger_bands(
                data['prices']['price'], window=bb_window)
        
        bb_upper = data['prices'][bb_upper_col].iloc[-1]
        bb_lower = data['prices'][bb_lower_col].iloc[-1]
        bb_signal = "COMPRA" if last_price < bb_lower else "VENDA" if last_price > bb_upper else "NEUTRO"
        signals.append((f"Bollinger Bands ({bb_window})", bb_signal, f"Atual: ${last_price:,.0f}"))
    
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

# Configurações padrão
DEFAULT_SETTINGS = {
    'rsi_window': 14,
    'bb_window': 20,
    'ma_windows': [7, 30, 200],
    'email': ''
}

# Inicializar session_state para configurações
if 'user_settings' not in st.session_state:
    st.session_state.user_settings = DEFAULT_SETTINGS.copy()

# Sidebar - Controles do Usuário
st.sidebar.header("⚙️ Painel de Controle")

# Configurações dos indicadores
st.sidebar.subheader("🔧 Parâmetros Técnicos")

# Usar valores do session_state como padrão
rsi_window = st.sidebar.slider(
    "Período do RSI", 
    7, 21, 
    st.session_state.user_settings['rsi_window']
)

bb_window = st.sidebar.slider(
    "Janela das Bandas de Bollinger", 
    10, 50, 
    st.session_state.user_settings['bb_window']
)

ma_windows = st.sidebar.multiselect(
    "Médias Móveis para Exibir",
    [7, 20, 30, 50, 100, 200],
    st.session_state.user_settings['ma_windows']
)

# Configurações de alertas
st.sidebar.subheader("🔔 Alertas Automáticos")
email = st.sidebar.text_input(
    "E-mail para notificações", 
    st.session_state.user_settings['email']
)

# Botões de controle
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("💾 Salvar Configurações"):
        st.session_state.user_settings = {
            'rsi_window': rsi_window,
            'bb_window': bb_window,
            'ma_windows': ma_windows,
            'email': email
        }
        st.sidebar.success("Configurações salvas com sucesso!")
        
with col2:
    if st.button("🔄 Resetar"):
        st.session_state.user_settings = DEFAULT_SETTINGS.copy()
        st.sidebar.success("Configurações resetadas para padrão!")
        if hasattr(st, 'rerun'):
            st.rerun()
        else:
            st.experimental_rerun()

if st.sidebar.button("Ativar Monitoramento Contínuo"):
    st.sidebar.success("Alertas ativados!")

# Gerar sinais com configurações atuais
signals, final_verdict, buy_signals, sell_signals = generate_signals(
    data, 
    rsi_window=st.session_state.user_settings['rsi_window'],
    bb_window=st.session_state.user_settings['bb_window']
)

sentiment = get_market_sentiment()
traditional_assets = get_traditional_assets()

# Seção principal
st.header("📊 Painel Integrado BTC Pro+")

# Linha de métricas
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Preço BTC", f"${data['prices']['price'].iloc[-1]:,.2f}")
col2.metric("Sentimento", f"{sentiment['value']}/100", sentiment['sentiment'])

# S&P 500 com %
sp500_data = traditional_assets[traditional_assets['asset']=='S&P 500']
sp500_value = sp500_data['value'].iloc[-1]
sp500_prev = sp500_data['value'].iloc[-2] if len(sp500_data) > 1 else sp500_value
sp500_change = (sp500_value/sp500_prev - 1)*100
col3.metric(
    "S&P 500", 
    f"${sp500_value:,.0f}",
    f"{sp500_change:+.2f}%",
    delta_color="normal"
)

# Ouro com %
ouro_data = traditional_assets[traditional_assets['asset']=='Ouro']
ouro_value = ouro_data['value'].iloc[-1]
ouro_prev = ouro_data['value'].iloc[-2] if len(ouro_data) > 1 else ouro_value
ouro_change = (ouro_value/ouro_prev - 1)*100
col4.metric(
    "Ouro", 
    f"${ouro_value:,.0f}",
    f"{ouro_change:+.2f}%",
    delta_color="normal"
)

# Análise Final
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
    col1, col2 = st.columns([3, 2])
    
    with col1:
        if not data['prices'].empty:
            # Mostrar apenas as médias móveis selecionadas
            ma_cols = ['price'] + [f'MA{window}' for window in st.session_state.user_settings['ma_windows']]
            fig = px.line(data['prices'], x="date", y=ma_cols, 
                         title="Preço BTC e Médias Móveis")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Análise Técnica")
        
        # Container para os indicadores
        indicators_container = st.container()
        
        with indicators_container:
            # 1. Médias Móveis
            for signal in signals:
                if "MA" in signal[0] or "Preço vs" in signal[0]:
                    color = "🟢" if signal[1] == "COMPRA" else "🔴" if signal[1] == "VENDA" else "🟡"
                    st.markdown(f"{color} **{signal[0]}**: {signal[1]} ({signal[2]})")
            
            # 2. RSI
            rsi_signal = next(s for s in signals if "RSI" in s[0])
            rsi_color = "🟢" if rsi_signal[1] == "COMPRA" else "🔴" if rsi_signal[1] == "VENDA" else "🟡"
            st.markdown(f"{rsi_color} **{rsi_signal[0]}**: {rsi_signal[1]} ({rsi_signal[2]})")
            
            # 3. MACD
            macd_signal = next(s for s in signals if "MACD" in s[0])
            macd_color = "🟢" if macd_signal[1] == "COMPRA" else "🔴"
            st.markdown(f"{macd_color} **{macd_signal[0]}**: {macd_signal[1]} ({macd_signal[2]})")
            
            # 4. Bollinger Bands
            bb_signal = next(s for s in signals if "Bollinger" in s[0])
            bb_color = "🟢" if bb_signal[1] == "COMPRA" else "🔴" if bb_signal[1] == "VENDA" else "🟡"
            st.markdown(f"{bb_color} **{bb_signal[0]}**: {bb_signal[1]} ({bb_signal[2]})")
            
            # 5. Fluxo de Exchanges
            flow_signal = next(s for s in signals if "Fluxo" in s[0])
            flow_color = "🟢" if flow_signal[1] == "COMPRA" else "🔴"
            st.markdown(f"{flow_color} **{flow_signal[0]}**: {flow_signal[1]} ({flow_signal[2]})")
            
            # 6. Hashrate vs Dificuldade
            hr_signal = next(s for s in signals if "Hashrate" in s[0])
            hr_color = "🟢" if hr_signal[1] == "COMPRA" else "🔴"
            st.markdown(f"{hr_color} **{hr_signal[0]}**: {hr_signal[1]} ({hr_signal[2]})")
            
            # 7. Atividade de Whales
            whale_signal = next(s for s in signals if "Whales" in s[0])
            whale_color = "🟢" if whale_signal[1] == "COMPRA" else "🔴"
            st.markdown(f"{whale_color} **{whale_signal[0]}**: {whale_signal[1]} ({whale_signal[2]})")
        
        # Análise Final em destaque
        st.divider()
        st.subheader("📌 Análise Consolidada")
        
        if final_verdict == "✅ FORTE COMPRA":
            st.success(f"## {final_verdict} ({buy_signals}/{len(signals)} indicadores)")
        elif final_verdict == "❌ FORTE VENDA":
            st.error(f"## {final_verdict} ({sell_signals}/{len(signals)} indicadores)")
        elif "COMPRA" in final_verdict:
            st.info(f"## {final_verdict} ({buy_signals}/{len(signals)} indicadores)")
        elif "VENDA" in final_verdict:
            st.warning(f"## {final_verdict} ({sell_signals}/{len(signals)} indicadores)")
        else:
            st.write(f"## {final_verdict}")
        
        st.caption(f"*Baseado na análise de {len(signals)} indicadores técnicos*")
    
    # Gráfico de Sentimento abaixo
    st.subheader("📈 Sentimento do Mercado")
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

with tab3:  # Backtesting (COMPLETAMENTE REFEITO)
    st.subheader("🧪 Backtesting Avançado")
    
    # Seletor de estratégia
    strategy = st.selectbox(
        "Escolha sua Estratégia:",
        ["RSI", "MACD", "Bollinger", "EMA Cross"],
        key="backtest_strategy"
    )
    
    # Parâmetros dinâmicos
    params_col1, params_col2 = st.columns(2)
    with params_col1:
        if strategy == "RSI":
            rsi_window = st.slider("Período RSI", 7, 21, 14)
            overbought = st.slider("Zona de Sobrevenda", 70, 90, 70)
            oversold = st.slider("Zona de Sobrecompra", 10, 30, 30)
            df = backtest_rsi_strategy(data['prices'], rsi_window, overbought, oversold)
            
        elif strategy == "MACD":
            fast = st.slider("EMA Rápida", 5, 20, 12)
            slow = st.slider("EMA Lenta", 20, 50, 26)
            signal = st.slider("Linha de Sinal", 5, 20, 9)
            df = backtest_macd_strategy(data['prices'], fast, slow, signal)
            
        elif strategy == "Bollinger":
            window = st.slider("Janela", 10, 50, 20)
            num_std = st.slider("Nº de Desvios", 1.0, 3.0, 2.0, 0.1)
            df = backtest_bollinger_strategy(data['prices'], window, num_std)
            
        else:  # EMA Cross
            short_window = st.slider("EMA Curta", 5, 20, 9)
            long_window = st.slider("EMA Longa", 20, 50, 21)
            df = backtest_ema_cross_strategy(data['prices'], short_window, long_window)
    
    with params_col2:
        # Mostrar descrição da estratégia
        st.markdown("**📝 Descrição da Estratégia**")
        if strategy == "RSI":
            st.markdown("""
            - **Compra**: Quando RSI < Zona de Sobrecompra e preço > MA30
            - **Venda**: Quando RSI > Zona de Sobrevenda e preço < MA30
            """)
        elif strategy == "MACD":
            st.markdown("""
            - **Compra Forte**: MACD > 0 e cruzando linha de sinal para cima
            - **Venda Forte**: MACD < 0 e cruzando linha de sinal para baixo
            """)
        elif strategy == "Bollinger":
            st.markdown("""
            - **Compra**: Preço toca banda inferior
            - **Venda Parcial**: Preço cruza a média móvel
            - **Venda Total**: Preço toca banda superior
            """)
        else:  # EMA Cross
            st.markdown("""
            - **Compra**: EMA curta cruza EMA longa para cima
            - **Venda**: EMA curta cruza EMA longa para baixo
            """)
    
    # Calcular métricas
    metrics = calculate_metrics(df)
    
    # Mostrar resultados
    st.subheader("📊 Resultados do Backtesting")
    
    # Gráfico comparativo
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['strategy_cumulative'],
        name="Estratégia",
        line=dict(color='green', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['cumulative_return'],
        name="Buy & Hold",
        line=dict(color='blue', width=2)
    ))
    fig.update_layout(
        title="Desempenho Comparativo",
        yaxis_title="Retorno Acumulado",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Métricas de performance
    st.subheader("📈 Métricas de Performance")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Retorno Estratégia", f"{metrics['Retorno Estratégia']:.2%}",
                 delta=f"{(metrics['Retorno Estratégia'] - metrics['Retorno Buy & Hold']):.2%} vs B&H")
    with col2:
        st.metric("Retorno Buy & Hold", f"{metrics['Retorno Buy & Hold']:.2%}")
    with col3:
        st.metric("Sharpe Ratio", f"{metrics['Sharpe Estratégia']:.2f}",
                 delta=f"{(metrics['Sharpe Estratégia'] - metrics['Sharpe Buy & Hold']):.2f} vs B&H")
    
    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Volatilidade", f"{metrics['Vol Estratégia']:.2%}",
                 delta=f"{(metrics['Vol Estratégia'] - metrics['Vol Buy & Hold']):.2%} vs B&H")
    with col5:
        st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
    with col6:
        st.metric("Taxa de Acerto", f"{metrics['Taxa Acerto']:.2%}")
    
    # Otimização de parâmetros
    st.subheader("⚙️ Otimização Automática de Parâmetros")
    if st.checkbox("🔍 Executar Otimização (Pode demorar)"):
        with st.spinner("Otimizando parâmetros..."):
            if strategy == "RSI":
                param_space = {
                    'rsi_window': range(10, 21),
                    'overbought': range(70, 81, 5),
                    'oversold': range(20, 31, 5)
                }
            elif strategy == "MACD":
                param_space = {
                    'fast': range(10, 21),
                    'slow': range(20, 31),
                    'signal': range(5, 16)
                }
            elif strategy == "Bollinger":
                param_space = {
                    'window': range(15, 26),
                    'num_std': [1.5, 2.0, 2.5]
                }
            else:  # EMA Cross
                param_space = {
                    'short_window': range(5, 16),
                    'long_window': range(15, 26)
                }
            
            best_params, best_sharpe, best_df = optimize_strategy_parameters(
                data, strategy, param_space)
            
            st.success(f"🎯 Melhores parâmetros encontrados (Sharpe: {best_sharpe:.2f}):")
            st.write(best_params)
            
            if st.button("Aplicar Parâmetros Otimizados"):
                if strategy == "RSI":
                    st.session_state.user_settings['rsi_window'] = best_params['rsi_window']
                elif strategy == "Bollinger":
                    st.session_state.user_settings['bb_window'] = best_params['window']
                st.rerun()

with tab4:  # Cenários
    st.subheader("🌍 Simulação de Eventos")
    event = st.selectbox(
        "Selecione um Cenário:", 
        ["Halving", "Crash", "ETF Approval"]
    )
    
    # Simular (linha corrigida)
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
        # Gráfico RSI com período personalizado
        rsi_window = st.session_state.user_settings['rsi_window']
        rsi_col = f'RSI_{rsi_window}'
        if rsi_col not in data['prices']:
            data['prices'][rsi_col] = calculate_rsi(data['prices']['price'], rsi_window)
        
        fig_rsi = px.line(data['prices'], x="date", y=rsi_col, 
                         title=f"RSI ({rsi_window} dias)", 
                         range_y=[0, 100])
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        # Gráfico Bollinger Bands com janela personalizada
        bb_window = st.session_state.user_settings['bb_window']
        bb_upper_col = f'BB_Upper_{bb_window}'
        bb_lower_col = f'BB_Lower_{bb_window}'
        if bb_upper_col not in data['prices']:
            data['prices'][bb_upper_col], data['prices'][bb_lower_col] = calculate_bollinger_bands(
                data['prices']['price'], window=bb_window)
        
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(
            x=data['prices']['date'], 
            y=data['prices'][bb_upper_col], 
            name="Banda Superior"))
        fig_bb.add_trace(go.Scatter(
            x=data['prices']['date'], 
            y=data['prices']['price'], 
            name="Preço"))
        fig_bb.add_trace(go.Scatter(
            x=data['prices']['date'], 
            y=data['prices'][bb_lower_col], 
            name="Banda Inferior"))
        fig_bb.update_layout(title=f"Bandas de Bollinger ({bb_window},2)")
        st.plotly_chart(fig_bb, use_container_width=True)
        
        # Gráfico MACD (mantido padrão)
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(
            x=data['prices']['date'], 
            y=data['prices']['MACD'], 
            name="MACD"))
        fig_macd.add_trace(go.Scatter(
            x=data['prices']['date'], 
            y=data['prices']['MACD_Signal'], 
            name="Signal"))
        fig_macd.update_layout(title="MACD (12,26,9)")
        st.plotly_chart(fig_macd, use_container_width=True)

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
        pdf.cell(200, 10, txt=f"Configurações:", ln=1)
        pdf.cell(200, 10, txt=f"- Período RSI: {st.session_state.user_settings['rsi_window']}", ln=1)
        pdf.cell(200, 10, txt=f"- BB Window: {st.session_state.user_settings['bb_window']}", ln=1)
        pdf.cell(200, 10, txt=f"- Médias Móveis: {', '.join(map(str, st.session_state.user_settings['ma_windows']))}", ln=1)
        
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
