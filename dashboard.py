import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from fpdf import FPDF
import yfinance as yf
import tempfile
import re
import io
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.model_selection import ParameterGrid
from transformers import pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
import traceback

# ======================
# CONFIGURAÇÕES INICIAIS
# ======================
st.set_page_config(layout="wide", page_title="BTC AI Dashboard Pro+")
st.title("🚀 BTC AI Dashboard Pro+ - Edição Premium")

# ======================
# CONSTANTES E CONFIGURAÇÕES
# ======================
INDICATOR_WEIGHTS = {
    'order_blocks': 2.0,
    'gaussian_process': 1.0,
    'rsi': 1.5,
    'macd': 1.3,
    'bollinger': 1.2,
    'volume': 1.1,
    'obv': 1.1,
    'stochastic': 1.1,
    'ma_cross': 1.0,
    'lstm': 1.8,
    'sentiment': 1.4
}

DEFAULT_SETTINGS = {
    'rsi_window': 14,
    'bb_window': 20,
    'ma_windows': [7, 30, 200],
    'email': '',
    'gp_window': 30,
    'gp_lookahead': 5,
    'ob_swing_length': 11,
    'ob_show_bull': 3,
    'ob_show_bear': 3,
    'ob_use_body': True,
    'min_confidence': 0.7,
    'n_clusters': 5,
    'lstm_window': 60,
    'lstm_epochs': 50,
    'lstm_units': 50,
    'rl_episodes': 10000
}

# ======================
# FUNÇÕES AUXILIARES
# ======================

def display_asset_metric(column, asset_name, assets_df):
    """Exibe métrica para um ativo tradicional."""
    if assets_df is None or asset_name not in assets_df['asset'].values:
        column.metric(asset_name, "N/A")
        return
    
    asset_data = assets_df[assets_df['asset'] == asset_name]
    if not asset_data.empty:
        last_value = asset_data.iloc[-1]['value']
        prev_value = asset_data.iloc[-2]['value'] if len(asset_data) > 1 else last_value
        change_pct = ((last_value / prev_value) - 1) * 100 if prev_value != 0 else 0
        column.metric(asset_name, f"${last_value:,.2f}", f"{change_pct:.2f}%")
    else:
        column.metric(asset_name, "N/A")

# ======================
# FUNÇÕES DE IA
# ======================

class BitcoinTradingEnv(gym.Env):
    """Ambiente customizado para trading de Bitcoin com Gymnasium."""
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, df, initial_balance=10000, render_mode=None):
        super(BitcoinTradingEnv, self).__init__()
        
        required_cols = ['price', 'volume', f'RSI_{DEFAULT_SETTINGS["rsi_window"]}',
                        'MACD', 'MACD_Signal', f'BB_Upper_{DEFAULT_SETTINGS["bb_window"]}',
                        f'BB_Lower_{DEFAULT_SETTINGS["bb_window"]}']
        
        # Verificação robusta dos dados de entrada
        if df is None or not isinstance(df, pd.DataFrame) or len(df) < 2:
            raise ValueError("DataFrame de entrada inválido ou insuficiente")
            
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame não possui colunas necessárias: {missing_cols}")

        self.df = df.dropna(subset=required_cols).reset_index(drop=True)
        if len(self.df) < 2:
            raise ValueError("DataFrame tem menos de 2 linhas válidas após limpeza")

        self.initial_balance = initial_balance
        self.current_step = 0
        self.render_mode = render_mode

        # Ações: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)

        # Espaço de observação
        self.observation_shape = (10,)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.btc_held = 0
        self.current_step = 0
        self.total_profit = 0
        self.last_portfolio_value = self.initial_balance

        observation = self._next_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _get_info(self):
        return {
            "step": self.current_step,
            "balance": self.balance,
            "btc_held": self.btc_held,
            "total_profit": self.total_profit,
            "portfolio_value": self.last_portfolio_value
        }

    def _next_observation(self):
        if self.current_step >= len(self.df):
            return np.zeros(self.observation_shape, dtype=np.float32)

        current_data = self.df.iloc[self.current_step]
        price = current_data['price']
        volume = current_data['volume']
        rsi = current_data[f'RSI_{DEFAULT_SETTINGS["rsi_window"]}']
        macd = current_data['MACD']
        macd_signal = current_data['MACD_Signal']
        bb_upper = current_data[f'BB_Upper_{DEFAULT_SETTINGS["bb_window"]}']
        bb_lower = current_data[f'BB_Lower_{DEFAULT_SETTINGS["bb_window"]}']

        obs = np.array([
            np.log1p(price / (self.df['price'].mean() if self.df['price'].mean() else 1),
            np.log1p(volume / (self.df['volume'].mean() if self.df['volume'].mean() else 1),
            rsi / 100.0,
            macd / (price if price else 1),
            macd_signal / (price if price else 1),
            self.balance / self.initial_balance,
            (self.btc_held * price) / self.initial_balance,
            self.current_step / (len(self.df) - 1 if len(self.df) > 1 else 1),
            bb_upper / (price if price else 1),
            bb_lower / (price if price else 1),
        ], dtype=np.float32)

        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            terminated = True
            obs = self._next_observation()
            return obs, 0, terminated, False, self._get_info()

        current_price = self.df.iloc[self.current_step]['price']
        if pd.isna(current_price) or current_price <= 0:
            current_price = self.df.iloc[self.current_step-1]['price'] if self.current_step > 0 else self.initial_balance

        if action == 1 and self.balance > 10:
            amount_to_buy = self.balance / current_price
            self.btc_held += amount_to_buy
            self.balance = 0
        elif action == 2 and self.btc_held > 0.0001:
            self.balance += self.btc_held * current_price
            self.btc_held = 0

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        next_price = self.df.iloc[self.current_step]['price'] if not terminated else current_price
        if pd.isna(next_price) or next_price <= 0:
            next_price = current_price

        current_portfolio_value = self.balance + (self.btc_held * next_price)
        reward = (current_portfolio_value - self.last_portfolio_value) / self.initial_balance
        self.last_portfolio_value = current_portfolio_value
        self.total_profit = current_portfolio_value - self.initial_balance

        observation = self._next_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _render_frame(self):
        current_price = self.df.iloc[min(self.current_step, len(self.df)-1]['price']
        portfolio_value = self.balance + self.btc_held * current_price
        print(f"Step: {self.current_step}, Price: {current_price:.2f}, Balance: {self.balance:.2f}, BTC: {self.btc_held:.6f}, Portfolio: {portfolio_value:.2f}, Profit: {self.total_profit:.2f}")

    def close(self):
        pass

# ... (continua com o resto das funções, mantendo a mesma estrutura mas com verificações robustas)

# ======================
# FUNÇÃO PRINCIPAL
# ======================

def main():
    # Inicializar estado da sessão
    if 'user_settings' not in st.session_state:
        st.session_state.user_settings = DEFAULT_SETTINGS.copy()
    if 'last_backtest_strategy' not in st.session_state:
        st.session_state.last_backtest_strategy = None
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = pd.DataFrame()
    if 'backtest_metrics' not in st.session_state:
        st.session_state.backtest_metrics = {}

    # Sidebar
    st.sidebar.header("⚙️ Painel de Controle AI")
    
    with st.sidebar.expander("🧠 Configurações de IA", expanded=False):
        st.session_state.user_settings['lstm_window'] = st.slider("Janela LSTM", 30, 120, st.session_state.user_settings.get('lstm_window', 60), 10, key='cfg_lstm_w')
        st.session_state.user_settings['lstm_epochs'] = st.slider("Épocas LSTM", 10, 100, st.session_state.user_settings.get('lstm_epochs', 50), 10, key='cfg_lstm_e')
        st.session_state.user_settings['lstm_units'] = st.slider("Unidades LSTM", 30, 100, st.session_state.user_settings.get('lstm_units', 50), 10, key='cfg_lstm_u')
        st.session_state.user_settings['rl_episodes'] = st.slider("Timesteps RL", 5000, 50000, st.session_state.user_settings.get('rl_episodes', 10000), 1000, key='cfg_rl_ts')

    # ... (restante do código do sidebar)

    # Carregar dados
    @st.cache_data(ttl=1800, show_spinner="Carregando dados de mercado...")
    def load_cached_data():
        try:
            # Buscar dados do Yahoo Finance com verificação robusta
            ticker = "BTC-USD"
            hist = yf.download(ticker, period="2y", interval="1d", progress=False)
            
            if hist is None or not isinstance(hist, pd.DataFrame) or hist.empty:
                st.error("Falha ao obter dados do Yahoo Finance")
                return {'prices': pd.DataFrame(), 'prices_full': pd.DataFrame()}
                
            # Processamento dos dados (igual ao original)
            hist = hist.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'price', 'Volume': 'volume', 'Adj Close': 'adj_close'
            })
            hist = hist[['open', 'high', 'low', 'price', 'volume']]
            hist = hist[hist.index.tz_localize(None) < datetime.now().replace(tzinfo=None)]
            hist['date'] = pd.to_datetime(hist.index)
            
            data = {'prices_full': hist.copy(), 'prices': hist.tail(180).copy()}
            
            # Calcular indicadores (igual ao original)
            # ...
            
            return data
            
        except Exception as e:
            st.error(f"Erro crítico ao carregar dados: {str(e)}")
            return {'prices': pd.DataFrame(), 'prices_full': pd.DataFrame()}

    data = load_cached_data()
    
    # Verificação robusta dos dados carregados
    if not data or 'prices' not in data or not isinstance(data['prices'], pd.DataFrame) or data['prices'].empty:
        st.error("Falha crítica ao carregar dados. O dashboard não pode continuar.")
        st.stop()

    # ... (restante do código principal)

if __name__ == "__main__":
    main()
