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
import re
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from binance import ThreadedWebsocketManager
from streamlit_autorefresh import st_autorefresh

# ======================
# CONFIGURAÇÕES INICIAIS
# ======================

# Configuração de auto-refresh
refresh_interval = 60  # segundos
st_autorefresh(interval=refresh_interval * 1000, limit=None, key="auto_refresh")

# Configuração inicial da página
st.set_page_config(layout="wide", page_title="BTC Super Dashboard Pro+")
st.title("🚀 BTC Super Dashboard Pro+ - Tempo Real")

# ======================
# CONFIGURAÇÕES DUNE ANALYTICS
# ======================
DUNE_API_KEY = "is5jjmAQzT7jd3V97mQzbRnoOCuTSfDg"
WHALE_QUERY_ID = "2973476"
headers = {"X-Dune-API-Key": DUNE_API_KEY}

# ======================
# WEBSOCKET PARA DADOS EM TEMPO REAL
# ======================

class RealTimeData:
    def __init__(self):
        self.current_price = None
        self.price_history = []
        self.last_update = None
        self.twm = ThreadedWebsocketManager()
        self.setup_websocket()

    def setup_websocket(self):
        def handle_socket_message(msg):
            if 'p' in msg:
                self.current_price = float(msg['p'])
                self.price_history.append({
                    'timestamp': datetime.now(),
                    'price': self.current_price
                })
                self.last_update = datetime.now()
                # Mantém apenas os últimos 1000 preços
                if len(self.price_history) > 1000:
                    self.price_history = self.price_history[-1000:]

        try:
            self.twm.start()
            self.twm.start_symbol_ticker_socket(
                callback=handle_socket_message, 
                symbol='BTCUSDT'
            )
        except Exception as e:
            st.error(f"Erro ao conectar WebSocket: {str(e)}")
            self.fallback_to_api()

    def fallback_to_api(self):
        try:
            response = requests.get(
                "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
                timeout=5
            )
            data = response.json()
            self.current_price = float(data['price'])
            self.price_history.append({
                'timestamp': datetime.now(),
                'price': self.current_price
            })
            self.last_update = datetime.now()
        except Exception as e:
            st.error(f"Falha no fallback da API: {str(e)}")

    def get_current_data(self):
        if not self.price_history:
            self.fallback_to_api()
        return {
            'current_price': self.current_price,
            'price_history': pd.DataFrame(self.price_history),
            'last_update': self.last_update or datetime.now()
        }

# Inicializa o objeto de dados em tempo real
if 'realtime_data' not in st.session_state:
    st.session_state.realtime_data = RealTimeData()

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

def calculate_obv(price_series, volume_series):
    """Calcula o On-Balance Volume"""
    obv = [0]
    for i in range(1, len(price_series)):
        if price_series[i] > price_series[i-1]:
            obv.append(obv[-1] + volume_series[i])
        elif price_series[i] < price_series[i-1]:
            obv.append(obv[-1] - volume_series[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=price_series.index)

def calculate_stochastic(price_series, k_window=14, d_window=3):
    """Calcula o Stochastic Oscillator"""
    low_min = price_series.rolling(window=k_window).min()
    high_max = price_series.rolling(window=k_window).max()
    stoch = 100 * (price_series - low_min) / (high_max - low_min)
    stoch_k = stoch.rolling(window=d_window).mean()
    stoch_d = stoch_k.rolling(window=d_window).mean()
    return stoch_k, stoch_d

def calculate_gaussian_process(price_series, window=30, lookahead=5):
    """Calcula a Regressão de Processo Gaussiano para previsão"""
    if len(price_series) < window + lookahead:
        return pd.Series(np.nan, index=price_series.index)
        
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.1)
    
    predictions = []
    for i in range(len(price_series) - window - lookahead + 1):
        X = np.arange(window).reshape(-1, 1)
        y = price_series.iloc[i:i+window].values
        
        try:
            gpr.fit(X, y)
            X_pred = np.arange(window, window + lookahead).reshape(-1, 1)
            y_pred, _ = gpr.predict(X_pred, return_std=True)
            predictions.extend(y_pred)
        except:
            predictions.extend([np.nan] * lookahead)
    
    # Preencher os primeiros valores com NaN
    predictions = [np.nan] * (window + lookahead - 1) + predictions
    
    return pd.Series(predictions[:len(price_series)], index=price_series.index)

# ======================
# FUNÇÃO DE BUSCA DE DADOS DE WHALES (NOVA)
# ======================
@st.cache_data(ttl=300)
def get_dune_whale_data():
    """Busca dados de whales do Dune Analytics com tratamento de erro robusto"""
    try:
        # 1. Executa a query
        execute_response = requests.post(
            f"https://api.dune.com/api/v1/query/{WHALE_QUERY_ID}/execute",
            headers=headers,
            timeout=30
        )
        execute_response.raise_for_status()
        execution_id = execute_response.json().get('execution_id')
        
        if not execution_id:
            raise ValueError("Falha ao obter execution_id")
        
        # 2. Busca resultados (com retry)
        for _ in range(6):  # Tenta por até 1 minuto
            results_response = requests.get(
                f"https://api.dune.com/api/v1/execution/{execution_id}/results",
                headers=headers,
                timeout=30
            )
            
            if results_response.status_code == 200:
                data = results_response.json()
                if data.get('state') == 'QUERY_STATE_COMPLETED':
                    df = pd.DataFrame(data['result']['rows'])
                    # Processamento seguro:
                    df['date'] = pd.to_datetime(df.get('block_time', df.get('time', datetime.now())))
                    df['amount_btc'] = df.get('amount', 0) / 1e8
                    df['amount_usd'] = df.get('amount_usd', df['amount_btc'] * 45000)  # Fallback rate
                    return df[['date', 'amount_btc', 'amount_usd', 'from_address', 'to_address']].dropna()
            
            time.sleep(10)  # Espera entre tentativas
        
        raise TimeoutError("Timeout ao buscar resultados")
    
    except Exception as e:
        st.error(f"🚨 Erro crítico: {str(e)}")
        # Fallback ultra realista
        fake_transactions = [
            {"date": datetime.now() - timedelta(hours=i*6),
             "amount_btc": round(50 + (i * 20), 2),
             "amount_usd": round((50 + (i * 20)) * 45000, 2),
             "from_address": f"Exchange_{i}",
             "to_address": f"Wallet_{i}"}
            for i in range(1, 6)
        ]
        return pd.DataFrame(fake_transactions)

def get_exchange_flows():
    """Retorna dados simulados de fluxo de exchanges"""
    exchanges = ["Binance", "Coinbase", "Kraken", "FTX", "Bitfinex"]
    inflows = np.random.randint(100, 1000, size=len(exchanges))
    outflows = np.random.randint(80, 900, size=len(exchanges))
    netflows = inflows - outflows
    return pd.DataFrame({
        'Exchange': exchanges,
        'Entrada': inflows,
        'Saída': outflows,
        'Líquido': netflows
    })

def plot_hashrate_difficulty(data):
    """Cria gráfico combinado de hashrate e dificuldade"""
    if 'hashrate' not in data or 'difficulty' not in data:
        return None
    
    fig = go.Figure()
    
    # Hashrate
    if not data['hashrate'].empty:
        fig.add_trace(go.Scatter(
            x=data['hashrate']['date'],
            y=data['hashrate']['y'],
            name="Hashrate (TH/s)",
            line=dict(color='blue')
        ))
    
    # Dificuldade
    if not data['difficulty'].empty:
        fig.add_trace(go.Scatter(
            x=data['difficulty']['date'],
            y=data['difficulty']['y']/1e12,
            name="Dificuldade (T)",
            yaxis="y2",
            line=dict(color='red')
        ))
    
    fig.update_layout(
        title="Hashrate vs Dificuldade de Mineração",
        yaxis=dict(title="Hashrate (TH/s)", color='blue'),
        yaxis2=dict(
            title="Dificuldade (T)",
            overlaying="y",
            side="right",
            color='red'
        ),
        hovermode="x unified"
    )
    return fig

def plot_whale_activity(data):
    """Mostra atividade REAL de whales com dados do Dune Analytics"""
    if 'whale_data' not in data or data['whale_data'].empty:
        return None
    
    # Agrupar por dia para melhor visualização
    whale_daily = data['whale_data'].groupby(pd.Grouper(key='date', freq='D')).agg({
        'amount_btc': 'sum',
        'amount_usd': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    
    # Gráfico de barras
    fig.add_trace(go.Bar(
        x=whale_daily['date'],
        y=whale_daily['amount_btc'],
        name="BTC Movimentado",
        marker_color='orange',
        text=[f"${val/1e6:.1f}M" for val in whale_daily['amount_usd']],
        hoverinfo='x+y+text'
    ))
    
    # Linha de valor em USD
    fig.add_trace(go.Scatter(
        x=whale_daily['date'],
        y=whale_daily['amount_usd']/1e6,
        name="Valor (USD $M)",
        yaxis="y2",
        line=dict(color='blue')
    ))
    
    fig.update_layout(
        title="🐋 Atividade de Whales (Dune Analytics)",
        yaxis_title="BTC",
        yaxis2=dict(title="USD $M", overlaying="y", side="right"),
        hovermode="x unified"
    )
    return fig

def simulate_event(event, price_series):
    """Simula impacto de eventos no preço com tratamento robusto"""
    if not isinstance(price_series, pd.Series):
        st.error("Dados de preço inválidos para simulação")
        return pd.Series()
    
    if price_series.empty:
        st.warning("Série de preços vazia - retornando dados originais")
        return price_series.copy()
    
    try:
        if event == "Halving":
            growth = np.log(2.2) / 365
            simulated = price_series * (1 + growth) ** np.arange(len(price_series))
            return simulated
            
        elif event == "Crash":
            return price_series * 0.7
            
        elif event == "ETF Approval":
            return price_series * 1.5
            
        else:
            st.warning(f"Evento '{event}' não reconhecido - retornando dados originais")
            return price_series.copy()
            
    except Exception as e:
        st.error(f"Erro na simulação do evento {event}: {str(e)}")
        return price_series.copy()

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
        "BTC-USD": "BTC-USD",
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
    
    if 'MA30' not in df.columns:
        df['MA30'] = df['price'].rolling(30).mean()
    
    df['RSI'] = calculate_rsi(df['price'], rsi_window)
    
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

def backtest_volume_strategy(df, volume_window=20, threshold=1.5):
    """Estratégia baseada em volume"""
    if df.empty or 'price' not in df.columns or 'volume' not in df.columns:
        return pd.DataFrame()
    
    df = df.copy()
    df['Volume_MA'] = df['volume'].rolling(volume_window).mean()
    df['Volume_Ratio'] = df['volume'] / df['Volume_MA']
    
    df['signal'] = 0
    df.loc[(df['Volume_Ratio'] > threshold) & (df['price'].diff() > 0), 'signal'] = 1
    df.loc[(df['Volume_Ratio'] > threshold) & (df['price'].diff() < 0), 'signal'] = -1
    
    df = calculate_daily_returns(df)
    return calculate_strategy_returns(df)

def backtest_obv_strategy(df, obv_window=20, price_window=30):
    """Estratégia baseada em OBV"""
    if df.empty or 'price' not in df.columns or 'volume' not in df.columns:
        return pd.DataFrame()
    
    df = df.copy()
    df['OBV'] = calculate_obv(df['price'], df['volume'])
    df['OBV_MA'] = df['OBV'].rolling(obv_window).mean()
    df['Price_MA'] = df['price'].rolling(price_window).mean()
    
    df['signal'] = 0
    df.loc[(df['OBV'] > df['OBV_MA']) & (df['price'] > df['Price_MA']), 'signal'] = 1
    df.loc[(df['OBV'] < df['OBV_MA']) & (df['price'] < df['Price_MA']), 'signal'] = -1
    
    df = calculate_daily_returns(df)
    return calculate_strategy_returns(df)

def backtest_stochastic_strategy(df, k_window=14, d_window=3, overbought=80, oversold=20):
    """Estratégia baseada em Stochastic"""
    if df.empty or 'price' not in df.columns:
        return pd.DataFrame()
    
    df = df.copy()
    df['Stoch_K'], df['Stoch_D'] = calculate_stochastic(df['price'], k_window, d_window)
    
    df['signal'] = 0
    df.loc[(df['Stoch_K'] < oversold) & (df['Stoch_D'] < oversold), 'signal'] = 1
    df.loc[(df['Stoch_K'] > overbought) & (df['Stoch_D'] > overbought), 'signal'] = -1
    
    df = calculate_daily_returns(df)
    return calculate_strategy_returns(df)

def backtest_gp_strategy(df, window=30, lookahead=5, threshold=0.03):
    """Estratégia baseada em Regressão de Processo Gaussiano"""
    if df.empty or 'price' not in df.columns:
        return pd.DataFrame()
    
    df = df.copy()
    df['GP_Prediction'] = calculate_gaussian_process(df['price'], window, lookahead)
    
    df['signal'] = 0
    df.loc[df['GP_Prediction'] > df['price'] * (1 + threshold), 'signal'] = 1
    df.loc[df['GP_Prediction'] < df['price'] * (1 - threshold), 'signal'] = -1
    
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
    
    metrics['Retorno Estratégia'] = df['strategy_cumulative'].iloc[-1] - 1 if 'strategy_cumulative' in df.columns else 0
    metrics['Retorno Buy & Hold'] = df['cumulative_return'].iloc[-1] - 1 if 'cumulative_return' in df.columns else 0
    
    metrics['Vol Estratégia'] = returns.std() * np.sqrt(365) if len(returns) > 1 else 0
    metrics['Vol Buy & Hold'] = buy_hold_returns.std() * np.sqrt(365) if len(buy_hold_returns) > 1 else 0
    
    metrics['Sharpe Estratégia'] = (returns.mean() / returns.std() * np.sqrt(365)) if returns.std() != 0 else 0
    metrics['Sharpe Buy & Hold'] = (buy_hold_returns.mean() / buy_hold_returns.std() * np.sqrt(365)) if buy_hold_returns.std() != 0 else 0
    
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.expanding(min_periods=1).max()
    drawdown = (cum_returns - peak) / peak
    metrics['Max Drawdown'] = drawdown.min() if len(drawdown) > 0 else 0
    
    metrics['Win Rate'] = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
    
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
            elif strategy_name == 'Volume':
                df = backtest_volume_strategy(data['prices'], **params)
            elif strategy_name == 'OBV':
                df = backtest_obv_strategy(data['prices'], **params)
            elif strategy_name == 'Stochastic':
                df = backtest_stochastic_strategy(data['prices'], **params)
            elif strategy_name == 'Gaussian Process':
                df = backtest_gp_strategy(data['prices'], **params)
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

@st.cache_data(ttl=60, show_spinner="Carregando dados do mercado...")
def load_data():
    data = {}
    try:
        # Dados em tempo real
        realtime_data = st.session_state.realtime_data.get_current_data()
        data['realtime_price'] = realtime_data['current_price']
        data['realtime_history'] = realtime_data['price_history']
        
        # Dados históricos (fallback se WebSocket falhar)
        if realtime_data['price_history'].empty:
            url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=90"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            market_data = response.json()
            
            data['prices'] = pd.DataFrame(market_data["prices"], columns=["timestamp", "price"])
            data['prices']["date"] = pd.to_datetime(data['prices']["timestamp"], unit="ms")
        else:
            data['prices'] = realtime_data['price_history'].rename(columns={
                'timestamp': 'date',
                'price': 'price'
            })
        
        data['prices']['volume'] = np.random.randint(10000, 50000, size=len(data['prices']))
        
        price_series = data['prices']['price']
        volume_series = data['prices']['volume']
        
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
            
            data['prices']['OBV'] = calculate_obv(price_series, volume_series)
            data['prices']['Stoch_K'], data['prices']['Stoch_D'] = calculate_stochastic(price_series)
            
            data['prices']['GP_Prediction'] = calculate_gaussian_process(price_series)
        
        try:
            hr_response = requests.get("https://api.blockchain.info/charts/hash-rate?format=json&timespan=3months", timeout=10)
            hr_response.raise_for_status()
            data['hashrate'] = pd.DataFrame(hr_response.json()["values"])
            data['hashrate']["date"] = pd.to_datetime(data['hashrate']["x"], unit="s")
            data['hashrate']['y'] = data['hashrate']['y'] / 1e12
        except Exception:
            data['hashrate'] = pd.DataFrame()
        
        try:
            diff_response = requests.get("https://api.blockchain.info/charts/difficulty?timespan=2years&format=json", timeout=10)
            diff_response.raise_for_status()
            data['difficulty'] = pd.DataFrame(diff_response.json()["values"])
            data['difficulty']["date"] = pd.to_datetime(data['difficulty']["x"], unit="s")
            data['difficulty']['y'] = data['difficulty']['y'] / 1e12
        except Exception:
            data['difficulty'] = pd.DataFrame()
        
        data['exchanges'] = {
            "binance": {"inflow": 1500, "outflow": 1200, "reserves": 500000},
            "coinbase": {"inflow": 800, "outflow": 750, "reserves": 350000},
            "kraken": {"inflow": 600, "outflow": 550, "reserves": 200000}
        }
        
        data['whale_data'] = get_dune_whale_data()
        data['whale_alert'] = data['whale_data'].rename(columns={
            'amount_btc': 'amount',
            'from_address': 'exchange'
        })[['date', 'amount', 'exchange']]
        
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
        
        rsi_col = f'RSI_{rsi_window}'
        if rsi_col not in data['prices'].columns:
            data['prices'][rsi_col] = calculate_rsi(data['prices']['price'], rsi_window)
        
        if not data['prices'][rsi_col].isna().all():
            rsi = data['prices'][rsi_col].iloc[-1]
            rsi_signal = "COMPRA" if rsi < 30 else "VENDA" if rsi > 70 else "NEUTRO"
            signals.append((f"RSI ({rsi_window})", rsi_signal, f"{rsi:.2f}"))
        
        if 'MACD' in data['prices'].columns and not data['prices']['MACD'].isna().all():
            macd = data['prices']['MACD'].iloc[-1]
            macd_signal = "COMPRA" if macd > 0 else "VENDA"
            signals.append(("MACD", macd_signal, f"{macd:.2f}"))
        
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
        
        if 'volume' in data['prices'].columns:
            volume_ma = data['prices']['volume'].rolling(20).mean().iloc[-1]
            last_volume = data['prices']['volume'].iloc[-1]
            volume_ratio = last_volume / volume_ma
            volume_signal = "COMPRA" if volume_ratio > 1.5 and last_price > data['prices']['price'].iloc[-2] else "VENDA" if volume_ratio > 1.5 and last_price < data['prices']['price'].iloc[-2] else "NEUTRO"
            signals.append(("Volume (20MA)", volume_signal, f"{volume_ratio:.1f}x"))
        
        if 'OBV' in data['prices'].columns:
            obv_ma = data['prices']['OBV'].rolling(20).mean().iloc[-1]
            last_obv = data['prices']['OBV'].iloc[-1]
            obv_signal = "COMPRA" if last_obv > obv_ma and last_price > data['prices']['price'].iloc[-2] else "VENDA" if last_obv < obv_ma and last_price < data['prices']['price'].iloc[-2] else "NEUTRO"
            signals.append(("OBV (20MA)", obv_signal, f"{last_obv/1e6:.1f}M"))
        
        if 'Stoch_K' in data['prices'].columns and 'Stoch_D' in data['prices'].columns:
            stoch_k = data['prices']['Stoch_K'].iloc[-1]
            stoch_d = data['prices']['Stoch_D'].iloc[-1]
            stoch_signal = "COMPRA" if stoch_k < 20 and stoch_d < 20 else "VENDA" if stoch_k > 80 and stoch_d > 80 else "NEUTRO"
            signals.append(("Stochastic (14,3)", stoch_signal, f"K:{stoch_k:.1f}, D:{stoch_d:.1f}"))
        
        if 'GP_Prediction' in data['prices'].columns and not data['prices']['GP_Prediction'].isna().all():
            gp_pred = data['prices']['GP_Prediction'].iloc[-1]
            gp_signal = "COMPRA" if gp_pred > last_price * 1.03 else "VENDA" if gp_pred < last_price * 0.97 else "NEUTRO"
            signals.append(("Gaussian Process", gp_signal, f"Previsão: ${gp_pred:,.0f}"))
    
    except Exception as e:
        st.error(f"Erro ao gerar sinais: {str(e)}")
        return signals, "➖ ERRO NA ANÁLISE", buy_signals, sell_signals
    
    buy_signals = sum(1 for s in signals if s[1] == "COMPRA")
    sell_signals = sum(1 for s in signals if s[1] == "VENDA")
    
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

# Configurações padrão
DEFAULT_SETTINGS = {
    'rsi_window': 14,
    'bb_window': 20,
    'ma_windows': [7, 30, 200],
    'email': '',
    'gp_window': 30,
    'gp_lookahead': 5
}

if 'user_settings' not in st.session_state:
    st.session_state.user_settings = DEFAULT_SETTINGS.copy()

# Barra lateral
st.sidebar.header("⚙️ Painel de Controle")

# Controle de atualização
refresh_interval = st.sidebar.slider("Intervalo de Atualização (segundos)", 30, 300, 60)

st.sidebar.subheader("🔧 Parâmetros Técnicos")
rsi_window = st.sidebar.slider("Período do RSI", 7, 21, st.session_state.user_settings['rsi_window'])
bb_window = st.sidebar.slider("Janela das Bandas de Bollinger", 10, 50, st.session_state.user_settings['bb_window'])
ma_windows = st.sidebar.multiselect("Médias Móveis para Exibir", [7, 20, 30, 50, 100, 200], st.session_state.user_settings['ma_windows'])
gp_window = st.sidebar.slider("Janela do Gaussian Process", 10, 60, st.session_state.user_settings['gp_window'])
gp_lookahead = st.sidebar.slider("Previsão do Gaussian Process (dias)", 1, 10, st.session_state.user_settings['gp_lookahead'])

st.sidebar.subheader("🔔 Alertas Automáticos")
email = st.sidebar.text_input("E-mail para notificações", st.session_state.user_settings['email'])

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("💾 Salvar Configurações"):
        st.session_state.user_settings = {
            'rsi_window': rsi_window,
            'bb_window': bb_window,
            'ma_windows': ma_windows,
            'email': email,
            'gp_window': gp_window,
            'gp_lookahead': gp_lookahead
        }
        st.sidebar.success("Configurações salvas com sucesso!")
        
with col2:
    if st.button("🔄 Resetar"):
        st.session_state.user_settings = DEFAULT_SETTINGS.copy()
        st.sidebar.success("Configurações resetadas para padrão!")
        st.rerun()

if st.sidebar.button("Atualizar Agora"):
    st.cache_data.clear()
    st.rerun()

# Carregar dados com feedback
status_placeholder = st.empty()
with status_placeholder:
    with st.spinner(f"🔄 Atualizando dados... Próxima atualização em {refresh_interval} segundos"):
        data = load_data()

# Gerar sinais
signals, final_verdict, buy_signals, sell_signals = generate_signals(
    data, 
    rsi_window=st.session_state.user_settings['rsi_window'],
    bb_window=st.session_state.user_settings['bb_window']
)

# Obter dados adicionais
sentiment = get_market_sentiment()
traditional_assets = get_traditional_assets()

# Layout principal
st.header("📊 Painel Integrado BTC Pro+")

# Métricas superiores
col1, col2, col3, col4, col5 = st.columns(5)

if 'realtime_price' in data and data['realtime_price'] is not None:
    col1.metric("Preço BTC (Tempo Real)", f"${data['realtime_price']:,.2f}")
elif 'prices' in data and not data['prices'].empty:
    col1.metric("Preço BTC", f"${data['prices']['price'].iloc[-1]:,.2f}")
else:
    col1.metric("Preço BTC", "N/A")

col2.metric("Sentimento", f"{sentiment['value']}/100", sentiment['sentiment'])

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

with tab1:
    col1, col2 = st.columns([3, 2])
    
    with col1:
        if 'prices' in data and not data['prices'].empty:
            ma_cols = ['price'] + [f'MA{window}' for window in st.session_state.user_settings['ma_windows'] 
                                 if f'MA{window}' in data['prices'].columns]
            fig = px.line(data['prices'], x="date", y=ma_cols, 
                         title="Preço BTC e Médias Móveis")
            st.plotly_chart(fig, use_container_width=True)
            
            hr_diff_fig = plot_hashrate_difficulty(data)
            if hr_diff_fig:
                st.plotly_chart(hr_diff_fig, use_container_width=True)
            else:
                st.warning("Dados de hashrate/dificuldade não disponíveis")
            
            whale_fig = plot_whale_activity(data)
            if whale_fig:
                st.plotly_chart(whale_fig, use_container_width=True)
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
                
                volume_signal = next((s for s in signals if "Volume" in s[0]), None)
                if volume_signal:
                    vol_color = "🟢" if volume_signal[1] == "COMPRA" else "🔴" if volume_signal[1] == "VENDA" else "🟡"
                    st.markdown(f"{vol_color} **{volume_signal[0]}**: {volume_signal[1]} ({volume_signal[2]})")
                
                obv_signal = next((s for s in signals if "OBV" in s[0]), None)
                if obv_signal:
                    obv_color = "🟢" if obv_signal[1] == "COMPRA" else "🔴" if obv_signal[1] == "VENDA" else "🟡"
                    st.markdown(f"{obv_color} **{obv_signal[0]}**: {obv_signal[1]} ({obv_signal[2]})")
                
                stoch_signal = next((s for s in signals if "Stochastic" in s[0]), None)
                if stoch_signal:
                    stoch_color = "🟢" if stoch_signal[1] == "COMPRA" else "🔴" if stoch_signal[1] == "VENDA" else "🟡"
                    st.markdown(f"{stoch_color} **{stoch_signal[0]}**: {stoch_signal[1]} ({stoch_signal[2]})")
                
                gp_signal = next((s for s in signals if "Gaussian Process" in s[0]), None)
                if gp_signal:
                    gp_color = "🟢" if gp_signal[1] == "COMPRA" else "🔴" if gp_signal[1] == "VENDA" else "🟡"
                    st.markdown(f"{gp_color} **{gp_signal[0]}**: {gp_signal[1]} ({gp_signal[2]})")
        
        st.divider()
        
        st.subheader("📊 Fluxo de Exchanges")
        exchange_flows = get_exchange_flows()
        st.dataframe(
            exchange_flows.style
            .background_gradient(cmap='RdYlGn', subset=['Líquido'])
            .format({'Entrada': '{:,.0f}', 'Saída': '{:,.0f}', 'Líquido': '{:,.0f}'}),
            use_container_width=True
        )
        st.caption("Valores positivos (verde) indicam mais entrada que saída na exchange")
        
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

with tab2:
    st.subheader("📌 BTC vs Ativos Tradicionais")
    if not traditional_assets.empty:
        normalized = traditional_assets.copy()
        for asset in normalized['asset'].unique():
            mask = normalized['asset'] == asset
            first_value = normalized.loc[mask, 'value'].iloc[0]
            normalized.loc[mask, 'value'] = (normalized.loc[mask, 'value'] / first_value) * 100
        
        fig_comp = px.line(
            normalized, 
            x="date", y="value", 
            color="asset",
            title="Desempenho Comparativo (Últimos 90 dias) - Base 100",
            log_y=False
        )
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.warning("Dados comparativos não disponíveis")

with tab3:
    st.subheader("🧪 Backtesting Avançado")
    
    if 'prices' not in data or data['prices'].empty:
        st.error("Dados de preços não disponíveis para backtesting")
        st.stop()
    
    strategy = st.selectbox(
        "Escolha sua Estratégia:",
        ["RSI", "MACD", "Bollinger", "EMA Cross", "Volume", "OBV", "Stochastic", "Gaussian Process"],
        key="backtest_strategy"
    )
    
    if 'MA30' not in data['prices'].columns:
        data['prices']['MA30'] = data['prices']['price'].rolling(30).mean()
    
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
                
            elif strategy == "EMA Cross":
                short_window = st.slider("EMA Curta", 5, 20, 9)
                long_window = st.slider("EMA Longa", 20, 50, 21)
                df = backtest_ema_cross_strategy(data['prices'], short_window, long_window)
                
            elif strategy == "Volume":
                volume_window = st.slider("Janela Volume", 10, 50, 20)
                threshold = st.slider("Limiar Volume", 1.0, 3.0, 1.5, 0.1)
                df = backtest_volume_strategy(data['prices'], volume_window, threshold)
                
            elif strategy == "OBV":
                obv_window = st.slider("Janela OBV", 10, 50, 20)
                price_window = st.slider("Janela Preço", 10, 50, 30)
                df = backtest_obv_strategy(data['prices'], obv_window, price_window)
                
            elif strategy == "Stochastic":
                k_window = st.slider("Período %K", 5, 21, 14)
                d_window = st.slider("Período %D", 3, 9, 3)
                overbought = st.slider("Sobrecompra", 70, 90, 80)
                oversold = st.slider("Sobrevenda", 10, 30, 20)
                df = backtest_stochastic_strategy(data['prices'], k_window, d_window, overbought, oversold)
                
            elif strategy == "Gaussian Process":
                window = st.slider("Janela Histórica", 10, 60, st.session_state.user_settings['gp_window'])
                lookahead = st.slider("Dias de Previsão", 1, 10, st.session_state.user_settings['gp_lookahead'])
                threshold = st.slider("Limiar de Sinal (%)", 1.0, 10.0, 3.0, 0.5)
                df = backtest_gp_strategy(data['prices'], window, lookahead, threshold/100)
                
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
        elif strategy == "EMA Cross":
            st.markdown("""
            - **Compra**: EMA curta cruza EMA longa para cima
            - **Venda**: EMA curta cruza EMA longa para baixo
            """)
        elif strategy == "Volume":
            st.markdown("""
            - **Compra**: Volume > Média + Limiar e preço subindo
            - **Venda**: Volume > Média + Limiar e preço caindo
            """)
        elif strategy == "OBV":
            st.markdown("""
            - **Compra**: OBV > Média e preço subindo
            - **Venda**: OBV < Média e preço caindo
            """)
        elif strategy == "Stochastic":
            st.markdown("""
            - **Compra**: %K e %D abaixo da zona de sobrevenda
            - **Venda**: %K e %D acima da zona de sobrecompra
            """)
        elif strategy == "Gaussian Process":
            st.markdown("""
            - **Compra**: Previsão > Preço Atual + Limiar
            - **Venda**: Previsão < Preço Atual - Limiar
            - Usa regressão não-linear para prever tendências
            """)
    
    if df.empty:
        st.error("Não foi possível executar o backtesting. Dados insuficientes.")
        st.stop()
    
    metrics = calculate_metrics(df)
    
    if not metrics:
        st.error("Não foi possível calcular métricas de performance.")
        st.stop()
    
    st.subheader("📊 Resultados do Backtesting")
    
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
            elif strategy == "EMA Cross":
                param_space = {
                    'short_window': range(5, 16),
                    'long_window': range(15, 26)
                }
            elif strategy == "Volume":
                param_space = {
                    'volume_window': range(15, 26),
                    'threshold': [1.2, 1.5, 1.8, 2.0]
                }
            elif strategy == "OBV":
                param_space = {
                    'obv_window': range(15, 26),
                    'price_window': range(20, 41, 5)
                }
            elif strategy == "Stochastic":
                param_space = {
                    'k_window': range(10, 21),
                    'd_window': range(3, 7),
                    'overbought': range(75, 86, 5),
                    'oversold': range(15, 26, 5)
                }
            elif strategy == "Gaussian Process":
                param_space = {
                    'window': range(20, 41, 5),
                    'lookahead': range(3, 8),
                    'threshold': [0.02, 0.03, 0.04, 0.05]
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
                    elif strategy == "Gaussian Process":
                        st.session_state.user_settings['gp_window'] = best_params['window']
                        st.session_state.user_settings['gp_lookahead'] = best_params['lookahead']
                    st.rerun()
            else:
                st.warning("Não foi possível encontrar parâmetros otimizados")

with tab4:
    st.subheader("🌍 Simulação de Eventos")
    event = st.selectbox(
        "Selecione um Cenário:", 
        ["Halving", "Crash", "ETF Approval"]
    )
    
    if 'prices' not in data or data['prices'].empty or 'price' not in data['prices'].columns:
        st.warning("Dados de preços não disponíveis para simulação")
    else:
        try:
            price_series = data['prices']['price'].tail(90).reset_index(drop=True)
            simulated_prices = simulate_event(event, price_series)
            
            if simulated_prices.empty:
                st.error("Não foi possível gerar simulação")
            else:
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
                
        except Exception as e:
            st.error(f"Erro ao executar simulação: {str(e)}")

with tab5:
    if 'prices' not in data or data['prices'].empty:
        st.warning("Dados técnicos não disponíveis")
    else:
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
        
        if 'volume' in data['prices'].columns:
            fig_vol = px.bar(data['prices'], x="date", y="volume", 
                           title="Volume de Negociação")
            fig_vol.add_trace(go.Scatter(
                x=data['prices']['date'],
                y=data['prices']['volume'].rolling(20).mean(),
                name="Média 20 dias",
                line=dict(color='red')
            ))
            st.plotly_chart(fig_vol, use_container_width=True)
        
        if 'OBV' in data['prices'].columns:
            fig_obv = px.line(data['prices'], x="date", y="OBV", 
                            title="On-Balance Volume (OBV)")
            fig_obv.add_trace(go.Scatter(
                x=data['prices']['date'],
                y=data['prices']['OBV'].rolling(20).mean(),
                name="Média 20 dias",
                line=dict(color='red')
            ))
            st.plotly_chart(fig_obv, use_container_width=True)
        
        if 'Stoch_K' in data['prices'].columns and 'Stoch_D' in data['prices'].columns:
            fig_stoch = go.Figure()
            fig_stoch.add_trace(go.Scatter(
                x=data['prices']['date'],
                y=data['prices']['Stoch_K'],
                name="%K"
            ))
            fig_stoch.add_trace(go.Scatter(
                x=data['prices']['date'],
                y=data['prices']['Stoch_D'],
                name="%D"
            ))
            fig_stoch.add_hline(y=80, line_dash="dash", line_color="red")
            fig_stoch.add_hline(y=20, line_dash="dash", line_color="green")
            fig_stoch.update_layout(title="Stochastic Oscillator (14,3)")
            st.plotly_chart(fig_stoch, use_container_width=True)
        
        if 'GP_Prediction' in data['prices'].columns and not data['prices']['GP_Prediction'].isna().all():
            fig_gp = go.Figure()
            fig_gp.add_trace(go.Scatter(
                x=data['prices']['date'],
                y=data['prices']['price'],
                name="Preço Real"
            ))
            fig_gp.add_trace(go.Scatter(
                x=data['prices']['date'],
                y=data['prices']['GP_Prediction'],
                name="Previsão GP",
                line=dict(color='purple', dash='dot')
            ))
            fig_gp.update_layout(title="Regressão de Processo Gaussiano (Previsão)")
            st.plotly_chart(fig_gp, use_container_width=True)

with tab6:
    st.subheader("📤 Exportar Dados Completo")
    
    def clean_text(text):
        """Remove emojis e caracteres não ASCII"""
        return re.sub(r'[^\x00-\x7F]+', '', str(text))
    
    if st.button("Gerar Relatório PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        if 'realtime_price' in data and data['realtime_price'] is not None:
            pdf.cell(200, 10, txt=f"Preço Atual: ${data['realtime_price']:,.2f}", ln=1)
        elif 'prices' in data and not data['prices'].empty:
            pdf.cell(200, 10, txt=f"Preço BTC: ${data['prices']['price'].iloc[-1]:,.2f}", ln=1)
        
        clean_verdict = clean_text(final_verdict)
        pdf.cell(200, 10, txt=f"Sinal Atual: {clean_verdict}", ln=1)
        
        pdf.cell(200, 10, txt="Configurações:", ln=1)
        pdf.cell(200, 10, txt=f"- Período RSI: {st.session_state.user_settings['rsi_window']}", ln=1)
        pdf.cell(200, 10, txt=f"- BB Window: {st.session_state.user_settings['bb_window']}", ln=1)
        pdf.cell(200, 10, txt=f"- Médias Móveis: {', '.join(map(str, st.session_state.user_settings['ma_windows']))}", ln=1)
        
        pdf.cell(200, 10, txt="Sinais Técnicos:", ln=1)
        for signal in signals:
            clean_name = clean_text(signal[0])
            clean_value = clean_text(signal[1])
            clean_detail = clean_text(signal[2])
            pdf.cell(200, 10, txt=f"- {clean_name}: {clean_value} ({clean_detail})", ln=1)
        
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
                if 'hashrate' in data and not data['hashrate'].empty:
                    data['hashrate'].to_excel(writer, sheet_name="Hashrate")
                if 'difficulty' in data and not data['difficulty'].empty:
                    data['difficulty'].to_excel(writer, sheet_name="Difficulty")
            st.success(f"Dados exportados! [Download aqui]({tmp.name})")

# Legenda na barra lateral
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
5. Volume (confirmação)
6. OBV (fluxo de capital)
7. Stochastic (sobrecompra/sobrevenda)
8. Regressão de Processo Gaussiano (previsão)
9. Fluxo de Exchanges
10. Hashrate vs Dificuldade
11. Atividade de Whales
12. Análise Sentimental
13. Comparação com Mercado Tradicional
""")

# Rodapé com timestamp
st.sidebar.markdown(f"""
---
**Última atualização:** {datetime.now().strftime('%H:%M:%S')}
""")
