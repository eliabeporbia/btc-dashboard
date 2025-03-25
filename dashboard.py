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
# FUNÇÕES DE CÁLCULO (ATUALIZADAS)
# ======================

def calculate_ema(series, window):
    """Calcula a Média Móvel Exponencial (EMA)"""
    if series.empty:
        return pd.Series()
    return series.ewm(span=window, adjust=False).mean()

def calculate_rsi(series, window=14):
    """Calcula o Índice de Força Relativa (RSI) com tratamento para dados vazios"""
    if len(series) < window + 1:
        return pd.Series(np.nan, index=series.index)
    
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calcula o MACD com linha de sinal"""
    if len(series) < slow + signal:
        return pd.Series(np.nan, index=series.index), pd.Series(np.nan, index=series.index)
    
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd = ema_fast - ema_slow
    signal_line = calculate_ema(macd, signal)
    return macd, signal_line

def calculate_bollinger_bands(series, window=20, num_std=2):
    """Calcula as Bandas de Bollinger com tratamento para dados insuficientes"""
    if len(series) < window:
        return pd.Series(np.nan, index=series.index), pd.Series(np.nan, index=series.index)
    
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, lower

def get_market_sentiment():
    """Coleta dados de sentimentos do mercado com tratamento de erro robusto"""
    try:
        response = requests.get("https://api.alternative.me/fng/", timeout=10)
        response.raise_for_status()
        data = response.json()
        return {
            "value": int(data["data"][0]["value"]),
            "sentiment": data["data"][0]["value_classification"]
        }
    except Exception as e:
        st.warning(f"Não foi possível obter o sentimento do mercado: {str(e)}")
        return {"value": 50, "sentiment": "Neutral"}

def get_traditional_assets():
    """Coleta dados de ativos tradicionais com tratamento de erro"""
    assets = {
        "S&P 500": "^GSPC",
        "Ouro": "GC=F",
        "ETH-USD": "ETH-USD"
    }
    dfs = []
    
    for name, ticker in assets.items():
        try:
            data = yf.Ticker(ticker).history(period="90d", interval="1d")
            if not data.empty:
                data = data.reset_index()[['Date', 'Close']].rename(columns={'Close': 'value', 'Date': 'date'})
                data['asset'] = name
                dfs.append(data)
        except Exception as e:
            st.warning(f"Não foi possível obter dados para {name}: {str(e)}")
    
    return pd.concat(dfs) if dfs else pd.DataFrame()

# ======================
# FUNÇÕES DE BACKTESTING (REVISTAS E APRIMORADAS)
# ======================

def calculate_daily_returns(df):
    """Calcula retornos diários e cumulativos com verificação de dados"""
    if df.empty or 'price' not in df.columns:
        return pd.DataFrame()
    
    df = df.copy()
    df['daily_return'] = df['price'].pct_change()
    df['cumulative_return'] = (1 + df['daily_return']).cumprod()
    return df

def calculate_strategy_returns(df, signal_col='signal'):
    """Calcula retornos da estratégia com verificações de segurança"""
    if df.empty or 'daily_return' not in df.columns or signal_col not in df.columns:
        return pd.DataFrame()
    
    df = df.copy()
    df['strategy_return'] = df[signal_col].shift(1) * df['daily_return']
    df['strategy_cumulative'] = (1 + df['strategy_return']).cumprod()
    return df

def backtest_rsi_strategy(df, rsi_window=14, overbought=70, oversold=30):
    """Estratégia RSI aprimorada com verificações robustas"""
    if df.empty or 'price' not in df.columns:
        return pd.DataFrame()
    
    df = df.copy()
    
    # Garantir colunas necessárias
    if 'MA30' not in df.columns:
        df['MA30'] = df['price'].rolling(30).mean()
    
    df['RSI'] = calculate_rsi(df['price'], rsi_window)
    
    # Sinais com confirmação
    df['signal'] = 0
    df.loc[(df['RSI'] < oversold) & (df['price'] > df['MA30']), 'signal'] = 1
    df.loc[(df['RSI'] > overbought) & (df['price'] < df['MA30']), 'signal'] = -1
    
    df = calculate_daily_returns(df)
    return calculate_strategy_returns(df)

def backtest_macd_strategy(df, fast=12, slow=26, signal=9):
    """Estratégia MACD com tratamento robusto"""
    if df.empty or 'price' not in df.columns:
        return pd.DataFrame()
    
    df = df.copy()
    df['MACD'], df['MACD_Signal'] = calculate_macd(df['price'], fast, slow, signal)
    
    df['signal'] = 0
    df.loc[df['MACD'] > 0, 'signal'] = 1
    df.loc[df['MACD'] < 0, 'signal'] = -1
    
    # Sinal forte no cruzamento
    df.loc[(df['MACD'] > df['MACD_Signal']) & (df['MACD'] > 0), 'signal'] = 1.5
    df.loc[(df['MACD'] < df['MACD_Signal']) & (df['MACD'] < 0), 'signal'] = -1.5
    
    df = calculate_daily_returns(df)
    return calculate_strategy_returns(df)

def backtest_bollinger_strategy(df, window=20, num_std=2):
    """Estratégia Bandas de Bollinger robusta"""
    if df.empty or 'price' not in df.columns:
        return pd.DataFrame()
    
    df = df.copy()
    df['BB_Upper'], df['BB_Lower'] = calculate_bollinger_bands(df['price'], window, num_std)
    df['MA'] = df['price'].rolling(window).mean()
    
    df['signal'] = 0
    df.loc[df['price'] < df['BB_Lower'], 'signal'] = 1
    df.loc[(df['price'] > df['MA']) & (df['signal'].shift(1) == 1), 'signal'] = 0.5
    df.loc[df['price'] > df['BB_Upper'], 'signal'] = -1
    
    df = calculate_daily_returns(df)
    return calculate_strategy_returns(df)

def backtest_ema_cross_strategy(df, short_window=9, long_window=21):
    """Estratégia EMA Cross com verificações"""
    if df.empty or 'price' not in df.columns:
        return pd.DataFrame()
    
    df = df.copy()
    df['EMA_Short'] = calculate_ema(df['price'], short_window)
    df['EMA_Long'] = calculate_ema(df['price'], long_window)
    
    df['signal'] = 0
    df.loc[df['EMA_Short'] > df['EMA_Long'], 'signal'] = 1
    df.loc[df['EMA_Short'] < df['EMA_Long'], 'signal'] = -1
    
    df = calculate_daily_returns(df)
    return calculate_strategy_returns(df)

def calculate_metrics(df):
    """Calcula métricas de performance com tratamento robusto"""
    metrics = {}
    
    if df.empty or 'strategy_return' not in df.columns or 'daily_return' not in df.columns:
        return metrics
    
    returns = df['strategy_return'].dropna()
    buy_hold_returns = df['daily_return'].dropna()
    
    if len(returns) == 0 or len(buy_hold_returns) == 0:
        return metrics
    
    # Retornos
    metrics['Retorno Estratégia'] = df['strategy_cumulative'].iloc[-1] - 1 if 'strategy_cumulative' in df.columns else 0
    metrics['Retorno Buy & Hold'] = df['cumulative_return'].iloc[-1] - 1 if 'cumulative_return' in df.columns else 0
    
    # Volatilidade
    metrics['Vol Estratégia'] = returns.std() * np.sqrt(365) if len(returns) > 1 else 0
    metrics['Vol Buy & Hold'] = buy_hold_returns.std() * np.sqrt(365) if len(buy_hold_returns) > 1 else 0
    
    # Razão Sharpe
    metrics['Sharpe Estratégia'] = (returns.mean() / returns.std() * np.sqrt(365)) if returns.std() != 0 else 0
    metrics['Sharpe Buy & Hold'] = (buy_hold_returns.mean() / buy_hold_returns.std() * np.sqrt(365)) if buy_hold_returns.std() != 0 else 0
    
    # Drawdown
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.expanding(min_periods=1).max()
    drawdown = (cum_returns - peak) / peak
    metrics['Max Drawdown'] = drawdown.min() if len(drawdown) > 0 else 0
    
    # Win Rate
    metrics['Win Rate'] = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
    
    # Taxa de Acerto
    trades = df[df['signal'] != 0] if 'signal' in df.columns else pd.DataFrame()
    metrics['Taxa Acerto'] = len(trades[trades['strategy_return'] > 0]) / len(trades) if len(trades) > 0 else 0
    
    return metrics

def optimize_strategy_parameters(data, strategy_name, param_space):
    """Otimização robusta de parâmetros"""
    best_sharpe = -np.inf
    best_params = None
    best_results = None
    
    if 'prices' not in data or data['prices'].empty:
        return best_params, best_sharpe, best_results
    
    param_combinations = list(ParameterGrid(param_space))
    if not param_combinations:
        return best_params, best_sharpe, best_results
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, params in enumerate(param_combinations):
        try:
            if strategy_name == 'RSI':
                df = backtest_rsi_strategy(data['prices'], **params)
            elif strategy_name == 'MACD':
                df = backtest_macd_strategy(data['prices'], **params)
            elif strategy_name == 'Bollinger':
                df = backtest_bollinger_strategy(data['prices'], **params)
            elif strategy_name == 'EMA Cross':
                df = backtest_ema_cross_strategy(data['prices'], **params)
            else:
                continue
                
            if df.empty or 'strategy_return' not in df.columns:
                continue
                
            returns = df['strategy_return'].dropna()
            if len(returns) > 1:
                sharpe = returns.mean() / returns.std() * np.sqrt(365) if returns.std() != 0 else 0
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params
                    best_results = df
                    
        except Exception as e:
            continue
        
        progress = (i + 1) / len(param_combinations)
        progress_bar.progress(progress)
        status_text.text(f"Testando combinação {i+1}/{len(param_combinations)} | Melhor Sharpe: {max(best_sharpe, 0):.2f}")
    
    progress_bar.empty()
    status_text.empty()
    
    return best_params, best_sharpe, best_results

# ======================
# CARREGAMENTO DE DADOS (REVISADO)
# ======================

@st.cache_data(ttl=3600, show_spinner="Carregando dados do mercado...")
def load_data():
    data = {}
    try:
        # Preço do Bitcoin (últimos 90 dias)
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=90"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        market_data = response.json()
        
        data['prices'] = pd.DataFrame(market_data["prices"], columns=["timestamp", "price"])
        data['prices']["date"] = pd.to_datetime(data['prices']["timestamp"], unit="ms")
        
        # Calculando indicadores técnicos com tratamento de erro
        price_series = data['prices']['price']
        
        if not price_series.empty:
            data['prices']['MA7'] = price_series.rolling(7).mean()
            data['prices']['MA30'] = price_series.rolling(30).mean()
            data['prices']['MA200'] = price_series.rolling(200).mean()
            data['prices']['RSI_14'] = calculate_rsi(price_series, 14)
            
            macd, signal = calculate_macd(price_series)
            data['prices']['MACD'] = macd
            data['prices']['MACD_Signal'] = signal
            
            upper, lower = calculate_bollinger_bands(price_series)
            data['prices']['BB_Upper_20'] = upper
            data['prices']['BB_Lower_20'] = lower
        
        # Hashrate (taxa de hash)
        try:
            hr_response = requests.get("https://api.blockchain.info/charts/hash-rate?format=json&timespan=3months", timeout=10)
            hr_response.raise_for_status()
            data['hashrate'] = pd.DataFrame(hr_response.json()["values"])
            data['hashrate']["date"] = pd.to_datetime(data['hashrate']["x"], unit="s")
        except Exception:
            data['hashrate'] = pd.DataFrame()
        
        # Dificuldade de mineração
        try:
            diff_response = requests.get("https://api.blockchain.info/charts/difficulty?timespan=2years&format=json", timeout=10)
            diff_response.raise_for_status()
            data['difficulty'] = pd.DataFrame(diff_response.json()["values"])
            data['difficulty']["date"] = pd.to_datetime(data['difficulty']["x"], unit="s")
        except Exception:
            data['difficulty'] = pd.DataFrame()
        
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
        data['prices'] = pd.DataFrame()
    except Exception as e:
        st.error(f"Erro ao processar dados: {str(e)}")
        data['prices'] = pd.DataFrame()
    
    return data

# ======================
# GERADOR DE SINAIS (REVISADO)
# ======================

def generate_signals(data, rsi_window=14, bb_window=20):
    """Geração robusta de sinais com tratamento de erro"""
    signals = []
    buy_signals = 0
    sell_signals = 0
    
    if 'prices' not in data or data['prices'].empty:
        return signals, "➖ DADOS INDISPONÍVEIS", buy_signals, sell_signals
    
    try:
        last_price = data['prices']['price'].iloc[-1]
        
        # 1. Sinais de Médias Móveis
        ma_signals = []
        for window in st.session_state.user_settings['ma_windows']:
            col_name = f'MA{window}'
            if col_name not in data['prices'].columns:
                data['prices'][col_name] = data['prices']['price'].rolling(window).mean()
            ma_signals.append((f"Preço vs MA{window}", data['prices'][col_name].iloc[-1]))
        
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
        if rsi_col not in data['prices'].columns:
            data['prices'][rsi_col] = calculate_rsi(data['prices']['price'], rsi_window)
        
        if not data['prices'][rsi_col].isna().all():
            rsi = data['prices'][rsi_col].iloc[-1]
            rsi_signal = "COMPRA" if rsi < 30 else "VENDA" if rsi > 70 else "NEUTRO"
            signals.append((f"RSI ({rsi_window})", rsi_signal, f"{rsi:.2f}"))
        
        # 3. MACD
        if 'MACD' in data['prices'].columns and not data['prices']['MACD'].isna().all():
            macd = data['prices']['MACD'].iloc[-1]
            macd_signal = "COMPRA" if macd > 0 else "VENDA"
            signals.append(("MACD", macd_signal, f"{macd:.2f}"))
        
        # 4. Bandas de Bollinger
        bb_upper_col = f'BB_Upper_{bb_window}'
        bb_lower_col = f'BB_Lower_{bb_window}'
        
        if bb_upper_col not in data['prices'].columns:
            upper, lower = calculate_bollinger_bands(data['prices']['price'], window=bb_window)
            data['prices'][bb_upper_col] = upper
            data['prices'][bb_lower_col] = lower
        
        if not data['prices'][bb_upper_col].isna().all():
            bb_upper = data['prices'][bb_upper_col].iloc[-1]
            bb_lower = data['prices'][bb_lower_col].iloc[-1]
            bb_signal = "COMPRA" if last_price < bb_lower else "VENDA" if last_price > bb_upper else "NEUTRO"
            signals.append((f"Bollinger Bands ({bb_window})", bb_signal, f"Atual: ${last_price:,.0f}"))
    
    except Exception as e:
        st.error(f"Erro ao gerar sinais: {str(e)}")
        return signals, "➖ ERRO NA ANÁLISE", buy_signals, sell_signals
    
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
# INTERFACE DO USUÁRIO (REVISADA)
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
        st.rerun()

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

if 'prices' in data and not data['prices'].empty:
    col1.metric("Preço BTC", f"${data['prices']['price'].iloc[-1]:,.2f}")
else:
    col1.metric("Preço BTC", "N/A")

col2.metric("Sentimento", f"{sentiment['value']}/100", sentiment['sentiment'])

# S&P 500 com %
if not traditional_assets.empty:
    sp500_data = traditional_assets[traditional_assets['asset']=='S&P 500']
    if not sp500_data.empty:
        sp500_value = sp500_data['value'].iloc[-1]
        sp500_prev = sp500_data['value'].iloc[-2] if len(sp500_data) > 1 else sp500_value
        sp500_change = (sp500_value/sp500_prev - 1)*100
        col3.metric("S&P 500", f"${sp500_value:,.0f}", f"{sp500_change:+.2f}%")
    else:
        col3.metric("S&P 500", "N/A")
else:
    col3.metric("S&P 500", "N/A")

# Ouro com %
if not traditional_assets.empty:
    ouro_data = traditional_assets[traditional_assets['asset']=='Ouro']
    if not ouro_data.empty:
        ouro_value = ouro_data['value'].iloc[-1]
        ouro_prev = ouro_data['value'].iloc[-2] if len(ouro_data) > 1 else ouro_value
        ouro_change = (ouro_value/ouro_prev - 1)*100
        col4.metric("Ouro", f"${ouro_value:,.0f}", f"{ouro_change:+.2f}%")
    else:
        col4.metric("Ouro", "N/A")
else:
    col4.metric("Ouro", "N/A")

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
        if 'prices' in data and not data['prices'].empty:
            ma_cols = ['price'] + [f'MA{window}' for window in st.session_state.user_settings['ma_windows'] 
                                 if f'MA{window}' in data['prices'].columns]
            fig = px.line(data['prices'], x="date", y=ma_cols, 
                         title="Preço BTC e Médias Móveis")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Dados de preços não disponíveis")
    
    with col2:
        st.subheader("📊 Análise Técnica")
        
        if not signals:
            st.warning("Nenhum sinal disponível")
        else:
            indicators_container = st.container()
            with indicators_container:
                for signal in signals:
                    if "MA" in signal[0] or "Preço vs" in signal[0]:
                        color = "🟢" if signal[1] == "COMPRA" else "🔴" if signal[1] == "VENDA" else "🟡"
                        st.markdown(f"{color} **{signal[0]}**: {signal[1]} ({signal[2]})")
                
                rsi_signal = next((s for s in signals if "RSI" in s[0]), None)
                if rsi_signal:
                    rsi_color = "🟢" if rsi_signal[1] == "COMPRA" else "🔴" if rsi_signal[1] == "VENDA" else "🟡"
                    st.markdown(f"{rsi_color} **{rsi_signal[0]}**: {rsi_signal[1]} ({rsi_signal[2]})")
                
                macd_signal = next((s for s in signals if "MACD" in s[0]), None)
                if macd_signal:
                    macd_color = "🟢" if macd_signal[1] == "COMPRA" else "🔴"
                    st.markdown(f"{macd_color} **{macd_signal[0]}**: {macd_signal[1]} ({macd_signal[2]})")
                
                bb_signal = next((s for s in signals if "Bollinger" in s[0]), None)
                if bb_signal:
                    bb_color = "🟢" if bb_signal[1] == "COMPRA" else "🔴" if bb_signal[1] == "VENDA" else "🟡"
                    st.markdown(f"{bb_color} **{bb_signal[0]}**: {bb_signal[1]} ({bb_signal[2]})")
        
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
    if not traditional_assets.empty:
        fig_comp = px.line(
            traditional_assets, 
            x="date", y="value", 
            color="asset",
            title="Desempenho Comparativo (Últimos 90 dias)",
            log_y=True
        )
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.warning("Dados comparativos não disponíveis")

with tab3:  # Backtesting (REVISADO E APRIMORADO)
    st.subheader("🧪 Backtesting Avançado")
    
    if 'prices' not in data or data['prices'].empty:
        st.error("Dados de preços não disponíveis para backtesting")
        st.stop()
    
    # Seletor de estratégia
    strategy = st.selectbox(
        "Escolha sua Estratégia:",
        ["RSI", "MACD", "Bollinger", "EMA Cross"],
        key="backtest_strategy"
    )
    
    # Garantir colunas necessárias
    if 'MA30' not in data['prices'].columns:
        data['prices']['MA30'] = data['prices']['price'].rolling(30).mean()
    
    # Parâmetros dinâmicos
    params_col1, params_col2 = st.columns(2)
    df = pd.DataFrame()
    
    with params_col1:
        try:
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
                
        except Exception as e:
            st.error(f"Erro ao configurar estratégia: {str(e)}")
            st.stop()
    
    with params_col2:
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
    
    if df.empty:
        st.error("Não foi possível executar o backtesting. Dados insuficientes.")
        st.stop()
    
    # Calcular métricas
    metrics = calculate_metrics(df)
    
    if not metrics:
        st.error("Não foi possível calcular métricas de performance.")
        st.stop()
    
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
            
            if best_params:
                st.success(f"🎯 Melhores parâmetros encontrados (Sharpe: {best_sharpe:.2f}):")
                st.json(best_params)
                
                if st.button("Aplicar Parâmetros Otimizados"):
                    if strategy == "RSI":
                        st.session_state.user_settings['rsi_window'] = best_params['rsi_window']
                    elif strategy == "Bollinger":
                        st.session_state.user_settings['bb_window'] = best_params['window']
                    st.rerun()
            else:
                st.warning("Não foi possível encontrar parâmetros otimizados")

with tab4:  # Cenários
    st.subheader("🌍 Simulação de Eventos")
    event = st.selectbox(
        "Selecione um Cenário:", 
        ["Halving", "Crash", "ETF Approval"]
    )
    
    if 'prices' not in data or data['prices'].empty:
        st.warning("Dados de preços não disponíveis para simulação")
    else:
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
    if 'prices' not in data or data['prices'].empty:
        st.warning("Dados técnicos não disponíveis")
    else:
        # Gráfico RSI com período personalizado
        rsi_window = st.session_state.user_settings['rsi_window']
        rsi_col = f'RSI_{rsi_window}'
        if rsi_col not in data['prices'].columns:
            data['prices'][rsi_col] = calculate_rsi(data['prices']['price'], rsi_window)
        
        if not data['prices'][rsi_col].isna().all():
            fig_rsi = px.line(data['prices'], x="date", y=rsi_col, 
                             title=f"RSI ({rsi_window} dias)", 
                             range_y=[0, 100])
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            st.plotly_chart(fig_rsi, use_container_width=True)
        else:
            st.warning("Não foi possível calcular o RSI")
        
        # Gráfico Bollinger Bands
        bb_window = st.session_state.user_settings['bb_window']
        bb_upper_col = f'BB_Upper_{bb_window}'
        bb_lower_col = f'BB_Lower_{bb_window}'
        
        if bb_upper_col not in data['prices'].columns:
            upper, lower = calculate_bollinger_bands(data['prices']['price'], window=bb_window)
            data['prices'][bb_upper_col] = upper
            data['prices'][bb_lower_col] = lower
        
        if not data['prices'][bb_upper_col].isna().all():
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
        else:
            st.warning("Não foi possível calcular as Bandas de Bollinger")
        
        # Gráfico MACD
        if 'MACD' in data['prices'].columns and not data['prices']['MACD'].isna().all():
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
        else:
            st.warning("Não foi possível calcular o MACD")

with tab6:  # Exportar
    st.subheader("📤 Exportar Dados Completo")
    
    if st.button("Gerar Relatório PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Relatório BTC Dashboard Pro+", ln=1, align='C')
        
        # Adicionar conteúdo
        if 'prices' in data and not data['prices'].empty:
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
                if 'prices' in data and not data['prices'].empty:
                    data['prices'].to_excel(writer, sheet_name="BTC Prices")
                if not traditional_assets.empty:
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
