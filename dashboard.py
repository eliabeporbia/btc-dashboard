# -*- coding: utf-8 -*-
import streamlit as st
# st.cache_resource.clear() # Comente/remova para produção

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
import os
import joblib
import warnings

# Ignorar warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='stable_baselines3')
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')

# --- Importações ML/IA ---
SKLEARN_AVAILABLE = False
try: from sklearn.preprocessing import MinMaxScaler, StandardScaler; from sklearn.cluster import KMeans; from sklearn.gaussian_process import GaussianProcessRegressor; from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel; from sklearn.model_selection import ParameterGrid; SKLEARN_AVAILABLE = True
except ImportError: st.warning("Scikit-learn não encontrado.")
TRANSFORMERS_AVAILABLE = False; ACCELERATE_AVAILABLE = False
try:
    from transformers import pipeline; TRANSFORMERS_AVAILABLE = True
    try: import accelerate; ACCELERATE_AVAILABLE = True
    except ImportError: st.warning("Accelerate não encontrado.")
except ImportError: st.warning("Transformers não encontrado.")
except ValueError as e:
    if "Keras is Keras 3" in str(e): st.error("Erro Keras 3/Transformers TF. Use 'tf-keras'.")
    else: st.warning(f"Erro import Transformers: {e}.")
    TRANSFORMERS_AVAILABLE = False
TF_AVAILABLE = False
try:
    try: from tf_keras.models import Sequential, load_model; from tf_keras.layers import LSTM, Dense, Dropout; from tf_keras.optimizers import Adam
    except ImportError: from tensorflow.keras.models import Sequential, load_model; from tensorflow.keras.layers import LSTM, Dense, Dropout; from tensorflow.keras.optimizers import Adam
    import tensorflow as tf; TF_AVAILABLE = True
except ImportError: st.warning("TensorFlow/Keras ou tf-keras não encontrado. LSTM desativado.")
except Exception as e: st.warning(f"Erro import TF/Keras: {e}. LSTM desativado.")
SB3_AVAILABLE = False
try: from stable_baselines3 import PPO; from stable_baselines3.common.env_checker import check_env; from stable_baselines3.common.vec_env import DummyVecEnv; from stable_baselines3.common.callbacks import BaseCallback; SB3_AVAILABLE = True
except ImportError: st.warning("Stable-Baselines3 não encontrado. RL desativado.")
GYM_AVAILABLE = False
try: import gymnasium as gym; from gymnasium import spaces; GYM_AVAILABLE = True
except ImportError: st.warning("Gymnasium não encontrado. RL desativado.")
TORCH_AVAILABLE = False
try: import torch; TORCH_AVAILABLE = True
except ImportError: st.warning("PyTorch não encontrado.")

# ======================
# CONSTANTES E CONFIGS
# ======================
st.set_page_config(layout="wide", page_title="BTC AI Dashboard Pro+")
st.title("🚀 BTC AI Dashboard Pro+ v2.4 - CoinAnk Style Report") # Versão incrementada

# Arquivos e Diretórios
BASE_DIR = os.path.dirname(os.path.abspath(__file__)); MODEL_DIR = os.path.join(BASE_DIR, "saved_models"); os.makedirs(MODEL_DIR, exist_ok=True)
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_btc_model.keras"); LSTM_SCALER_PATH = os.path.join(MODEL_DIR, "lstm_btc_scaler.joblib")
RL_MODEL_PATH = os.path.join(MODEL_DIR, "rl_ppo_btc_model.zip"); RL_SCALER_PATH = os.path.join(MODEL_DIR, "rl_observation_scaler.joblib"); RL_ENV_CONFIG_PATH = os.path.join(MODEL_DIR, "rl_env_config.joblib")

# Pesos e Configs Padrão
INDICATOR_WEIGHTS = { 'order_blocks': 2.0, 'gaussian_process': 1.5, 'rsi': 1.5, 'macd': 1.3, 'bollinger': 1.2, 'volume': 1.0, 'obv': 1.0, 'stochastic': 1.1, 'ma_cross': 1.0, 'lstm_pred': 1.8, 'rl_action': 2.0, 'sentiment': 1.2, 'divergence': 1.2, 'kdj': 1.1 } # Adicionado KDJ
DEFAULT_SETTINGS = { 'rsi_window': 14, 'bb_window': 20, 'ma_windows': [20, 50, 100], 'gp_window': 30, 'gp_lookahead': 1, 'ob_swing_length': 11, 'ob_show_bull': 3, 'ob_show_bear': 3, 'ob_use_body': True, 'min_confidence': 0.7, 'n_clusters': 5, 'lstm_window': 60, 'lstm_epochs': 30, 'lstm_units': 50, 'rl_total_timesteps': 20000, 'rl_transaction_cost': 0.001, 'email': '' }
RL_OBSERVATION_COLS_BASE = ['price', 'volume', 'RSI_14', 'MACD', 'MACD_Signal', 'BB_Upper_20', 'BB_Lower_20', 'Stoch_K_14_3']
RL_OBSERVATION_COLS_NORM = [f'{col}_norm' for col in RL_OBSERVATION_COLS_BASE]

# ======================
# FUNÇÕES AUXILIARES E CLASSES
# ======================

# --- Callback SB3 ---
class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps: int, progress_bar, status_text, verbose=0): super().__init__(verbose); self.total_timesteps = total_timesteps; self.progress_bar = progress_bar; self.status_text = status_text; self.current_step = 0
    def _on_step(self) -> bool:
        self.current_step = self.model.num_timesteps; progress = self.current_step / self.total_timesteps; reward_str = ""
        if self.model.ep_info_buffer and len(self.model.ep_info_buffer) > 0:
            try: reward_str = f"| R Média: {np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer]):.3f}"
            except Exception: pass
        self.status_text.text(f"Treinando RL: {self.current_step}/{self.total_timesteps} {reward_str}"); self.progress_bar.progress(progress)
        return True

# --- Ambiente RL ---
if GYM_AVAILABLE and SKLEARN_AVAILABLE:
    class BitcoinTradingEnv(gym.Env):
        metadata = {'render_modes': ['human']}
        def __init__(self, df, feature_cols_norm, scaler, transaction_cost=0.001, initial_balance=10000, render_mode=None):
            super().__init__(); self.feature_cols_base = [c.replace('_norm','') for c in feature_cols_norm]; required_cols = ['price'] + self.feature_cols_base
            if df.empty or not all(col in df.columns for col in required_cols): missing = [c for c in required_cols if c not in df.columns]; raise ValueError(f"DataFrame RL vazio ou colunas ausentes: {missing}")
            self.df = df.copy().reset_index(drop=True); self.scaler = scaler; self.feature_cols_norm = feature_cols_norm; self.transaction_cost = transaction_cost; self.initial_balance = initial_balance; self.render_mode = render_mode; self.current_step = 0
            self.action_space = spaces.Discrete(3); self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(len(self.feature_cols_norm),), dtype=np.float32)
        def _get_observation(self):
            idx = min(self.current_step, len(self.df) - 1); features = self.df.loc[idx, self.feature_cols_base].values.reshape(1, -1); scaled_features = self.scaler.transform(features).astype(np.float32)
            scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=5.0, neginf=-5.0); scaled_features = np.clip(scaled_features, -5.0, 5.0); return scaled_features.flatten()
        def _get_info(self): idx = min(self.current_step, len(self.df) - 1); current_price = self.df.loc[idx, 'price']; portfolio_value = self.balance + (self.btc_held * current_price); return {'total_profit': portfolio_value - self.initial_balance, 'portfolio_value': portfolio_value, 'balance': self.balance, 'btc_held': self.btc_held, 'current_step': self.current_step}
        def reset(self, seed=None, options=None): super().reset(seed=seed); self.balance = self.initial_balance; self.btc_held = 0; self.current_step = 0; self.last_portfolio_value = self.initial_balance; observation = self._get_observation(); info = self._get_info(); return observation, info
        def step(self, action):
            idx = min(self.current_step, len(self.df) - 1); current_price = self.df.loc[idx, 'price']; cost_penalty = 0
            if action == 1 and self.balance > 10: cost_penalty = self.balance * self.transaction_cost; self.balance -= cost_penalty;
            if self.balance > 0: self.btc_held += self.balance / current_price; self.balance = 0
            elif action == 2 and self.btc_held > 1e-6: sell_value = self.btc_held * current_price; cost_penalty = sell_value * self.transaction_cost; self.balance += sell_value - cost_penalty; self.btc_held = 0
            self.current_step += 1; terminated = self.current_step >= len(self.df); truncated = False
            if not terminated:
                next_idx = min(self.current_step, len(self.df) - 1); next_price = self.df.loc[next_idx, 'price']; current_portfolio_value = self.balance + (self.btc_held * next_price)
                reward = (current_portfolio_value - self.last_portfolio_value) - cost_penalty; self.last_portfolio_value = current_portfolio_value; observation = self._get_observation()
            else: reward = 0; observation = self._get_observation() if len(self.df)>1 else np.zeros(self.observation_space.shape)
            info = self._get_info(); return observation, reward, terminated, truncated, info
        def render(self): print(f"Step:{self.current_step}|Bal:${self.balance:,.2f}|BTC:{self.btc_held:.6f}|Port:${self.last_portfolio_value:,.2f}")
        def close(self): pass

# --- Funções Sentimento ---
@st.cache_resource(show_spinner="Carregando modelo de sentimento...")
def load_sentiment_model():
    if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE: st.warning("Transformers/PyTorch ausente. Sentimento desativado."); return None
    try: return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", framework="pt", device=-1)
    except Exception as e: st.error(f"Erro load Sentimento (pt): {e}"); return None

def analyze_news_sentiment(news_list, _model): # Mantida
    if _model is None: return news_list;
    if not news_list: return []; results = []
    for news in news_list:
        news['sentiment'] = 'NEUTRAL'; news['sentiment_score'] = 0.5
        try: text = news.get('title', '');
        if text: result = _model(text[:512])[0]; news['sentiment'] = result['label']; news['sentiment_score'] = result['score'] if result['label'] == 'POSITIVE' else (1 - result['score'])
        results.append(news)
        except Exception as e: st.warning(f"Erro análise sentimento: {e}"); results.append(news)
    return results

# --- Funções LSTM ---
@st.cache_resource
def create_lstm_architecture(input_shape, units=50): # Mantida
    if not TF_AVAILABLE: return None
    try: from tf_keras.models import Sequential; from tf_keras.layers import LSTM, Dense, Dropout; from tf_keras.optimizers import Adam
    except ImportError: from tensorflow.keras.models import Sequential; from tensorflow.keras.layers import LSTM, Dense, Dropout; from tensorflow.keras.optimizers import Adam
    model = Sequential([LSTM(units, return_sequences=True, input_shape=input_shape), Dropout(0.2), LSTM(units, return_sequences=False), Dropout(0.2), Dense(25, activation='relu'), Dense(1)])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error'); return model

@st.cache_resource(show_spinner="Verificando modelo LSTM treinado...")
def load_lstm_model_and_scaler(): # Mantida
    model, scaler = None, None
    if TF_AVAILABLE and os.path.exists(LSTM_MODEL_PATH) and os.path.exists(LSTM_SCALER_PATH):
        try:
             try: from tf_keras.models import load_model as load_model_tfk
             except ImportError: from tensorflow.keras.models import load_model as load_model_tfk
             model = load_model_tfk(LSTM_MODEL_PATH, compile=False)
             if not model.optimizer: model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
             scaler = joblib.load(LSTM_SCALER_PATH)
        except Exception as e: st.error(f"Erro ao carregar LSTM: {e}"); model, scaler = None, None
    return model, scaler

def train_and_save_lstm(data_prices, window, epochs, units): # Mantida
    if not TF_AVAILABLE or not SKLEARN_AVAILABLE: st.error("TF/Keras ou SKlearn ausente."); return False
    price_data = data_prices['price'].dropna().values.reshape(-1, 1)
    if len(price_data) < window + 1: st.error(f"Dados insuficientes LSTM ({len(price_data)}<{window+1})."); return False
    scaler = MinMaxScaler(feature_range=(0, 1)); scaled_data = scaler.fit_transform(price_data)
    X, y = [], [];
    for i in range(window, len(scaled_data)): X.append(scaled_data[i-window:i, 0]); y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    if len(X) == 0: st.error("Amostras LSTM não geradas."); return False
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    model = create_lstm_architecture(input_shape=(X.shape[1], 1), units=units)
    if model is None: return False
    try:
        status_lstm = st.status(f"Treinando LSTM ({epochs} épocas)...", expanded=True)
        history = model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
        final_loss = history.history['loss'][-1]
        status_lstm.update(label=f"Treino LSTM concluído! Loss: {final_loss:.4f}", state="complete", expanded=False)
        model.save(LSTM_MODEL_PATH); joblib.dump(scaler, LSTM_SCALER_PATH)
        st.success(f"Modelo LSTM salvo em '{LSTM_MODEL_PATH}'.")
        st.cache_resource.clear()
        return True
    except Exception as e: st.error(f"Erro treino/save LSTM: {e}"); return False

def predict_with_lstm(model, scaler, data_prices, window): # Mantida
    if model is None or scaler is None: return None
    try:
        last_window_data = data_prices['price'].dropna().values[-window:]
        if len(last_window_data) < window: return None
        last_window_scaled = scaler.transform(last_window_data.reshape(-1, 1))
        X_pred = np.reshape(last_window_scaled, (1, window, 1))
        pred_scaled = model.predict(X_pred, verbose=0)
        return scaler.inverse_transform(pred_scaled)[0][0]
    except Exception: return None

# --- Funções Indicadores Técnicos ---
def calculate_ema(series, window):
    if not isinstance(series, pd.Series) or series.empty: return pd.Series(dtype=np.float64)
    return series.dropna().ewm(span=window, adjust=False).mean().reindex(series.index)
def calculate_rsi(series, window=14):
    if not isinstance(series, pd.Series) or len(series.dropna()) < window + 1: return pd.Series(np.nan, index=series.index, dtype=np.float64)
    series_clean = series.dropna(); delta = series_clean.diff()
    gain = delta.where(delta > 0, 0.0); loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean(); avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss); rsi = 100.0 - (100.0 / (1.0 + rs)); rsi = np.where(np.isinf(rs), 100.0, rsi)
    return rsi.reindex(series.index)
def calculate_macd(series, fast=12, slow=26, signal=9):
    if not isinstance(series, pd.Series): return pd.Series(dtype=np.float64), pd.Series(dtype=np.float64)
    series_clean = series.dropna();
    if len(series_clean) < slow: return pd.Series(np.nan, index=series.index), pd.Series(np.nan, index=series.index)
    ema_fast = calculate_ema(series_clean, fast); ema_slow = calculate_ema(series_clean, slow); macd = ema_fast - ema_slow
    signal_line = calculate_ema(macd, signal) if len(macd.dropna()) >= signal else pd.Series(np.nan, index=macd.index)
    return macd.reindex(series.index), signal_line.reindex(series.index)
def calculate_bollinger_bands(series, window=20, num_std=2):
    if not isinstance(series, pd.Series): return pd.Series(dtype=np.float64), pd.Series(dtype=np.float64)
    series_clean = series.dropna();
    if len(series_clean) < window: return pd.Series(np.nan, index=series.index), pd.Series(np.nan, index=series.index)
    sma = series_clean.rolling(window).mean(); std = series_clean.rolling(window).std(ddof=0)
    upper = sma + (std * num_std); lower = sma - (std * num_std)
    return upper.reindex(series.index), lower.reindex(series.index)
def calculate_obv(price_series, volume_series):
    if not isinstance(price_series, pd.Series) or not isinstance(volume_series, pd.Series) or price_series.empty or volume_series.empty or len(price_series) != len(volume_series): return pd.Series(dtype=np.float64)
    df_temp = pd.DataFrame({'price': price_series, 'volume': volume_series}).dropna();
    if len(df_temp) < 2: return pd.Series(np.nan, index=price_series.index)
    price = df_temp['price']; volume = df_temp['volume']; price_diff = price.diff(); volume_signed = np.where(price_diff > 0, volume, np.where(price_diff < 0, -volume, 0))
    obv = volume_signed.cumsum(); obv.iloc[0] = 0; obv_series = pd.Series(obv, index=df_temp.index); return obv_series.reindex(price_series.index)

# --- MODIFICADO: Calcula K, D e J ---
def calculate_kdj(price_series, high_series, low_series, k_window=14, d_window=3, j_smooth=3):
    """Calcula o KDJ (%K, %D, %J)"""
    if not all(isinstance(s, pd.Series) for s in [price_series, high_series, low_series]) or price_series.empty or high_series.empty or low_series.empty:
        return pd.Series(dtype=np.float64), pd.Series(dtype=np.float64), pd.Series(dtype=np.float64)
    df_temp = pd.DataFrame({'close': price_series, 'high': high_series, 'low': low_series}).dropna();
    if len(df_temp) < k_window:
        return pd.Series(np.nan, index=price_series.index), pd.Series(np.nan, index=price_series.index), pd.Series(np.nan, index=price_series.index)

    close = df_temp['close']; high = df_temp['high']; low = df_temp['low']
    low_min = low.rolling(k_window).min(); high_max = high.rolling(k_window).max(); delta_hl = high_max - low_min
    # RSV (Raw Stochastic Value)
    rsv = 100 * (close - low_min) / delta_hl.replace(0, np.nan); rsv.fillna(50, inplace=True); rsv = rsv.clip(0, 100)
    # %K é a média móvel de RSV
    stoch_k = rsv.ewm(span=d_window, adjust=False).mean() # Usar EMA para suavização é comum
    # %D é a média móvel de %K
    stoch_d = stoch_k.ewm(span=d_window, adjust=False).mean()
    # %J
    stoch_j = 3 * stoch_k - 2 * stoch_d
    stoch_j = stoch_j.clip(0, 100) # J é frequentemente clipado entre 0 e 100

    return stoch_k.reindex(price_series.index), stoch_d.reindex(price_series.index), stoch_j.reindex(price_series.index)

def calculate_gaussian_process(price_series, window=30, lookahead=1):
    if not isinstance(price_series, pd.Series) or not SKLEARN_AVAILABLE: return pd.Series(dtype=np.float64)
    series_clean = price_series.dropna();
    if len(series_clean) < window + 1: return pd.Series(np.nan, index=price_series.index)
    kernel = ConstantKernel(1.0,(1e-3,1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-1,1e2)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5,1e1))
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=5); predictions = np.full(len(series_clean), np.nan); scaler = StandardScaler()
    for i in range(window, len(series_clean)):
        X_train = np.arange(i - window, i).reshape(-1, 1); y_train = series_clean.iloc[i - window:i].values.reshape(-1, 1); y_train_scaled = scaler.fit_transform(y_train).flatten()
        try: gpr.fit(X_train, y_train_scaled); X_pred = np.array([[i]]); y_pred_scaled, _ = gpr.predict(X_pred, return_std=True); predictions[i] = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()[0]
        except Exception: pass
    return pd.Series(predictions, index=series_clean.index).reindex(price_series.index)
def identify_order_blocks(df, swing_length=11, show_bull=3, show_bear=3, use_body=True):
    if df.empty or not all(col in df.columns for col in ['date','open','high','low','close']): return df.copy(), []
    df_copy = df.copy(); df_copy['date'] = pd.to_datetime(df_copy['date']); df_copy = df_copy.sort_values('date');
    if swing_length % 2 == 0: swing_length += 1; half_swing = swing_length // 2
    else: half_swing = swing_length // 2
    if use_body: df_copy['is_pivot_high'] = df_copy['close'].rolling(swing_length, center=True).apply(lambda x: x[half_swing] == x.max(), raw=True).fillna(0).astype(bool); df_copy['is_pivot_low'] = df_copy['close'].rolling(swing_length, center=True).apply(lambda x: x[half_swing] == x.min(), raw=True).fillna(0).astype(bool)
    else: df_copy['is_pivot_high'] = df_copy['high'].rolling(swing_length, center=True).apply(lambda x: x[half_swing] == x.max(), raw=True).fillna(0).astype(bool); df_copy['is_pivot_low'] = df_copy['low'].rolling(swing_length, center=True).apply(lambda x: x[half_swing] == x.min(), raw=True).fillna(0).astype(bool)
    blocks = []; pivot_high_indices = df_copy[df_copy['is_pivot_high']].index; pivot_low_indices = df_copy[df_copy['is_pivot_low']].index; last_bull_ob_end = pd.Timestamp.min.tz_localize(df_copy['date'].dt.tz); processed_indices_bull = set(); bullish_obs_found = 0
    for idx_pivot_low in reversed(pivot_low_indices):
        if bullish_obs_found >= show_bull or idx_pivot_low == df_copy.index[0]: continue
        lookback_range = df_copy.loc[:idx_pivot_low].iloc[-swing_length-1:-1]; down_candles = lookback_range[lookback_range['close'] < lookback_range['open']];
        if not down_candles.empty: ob_candle_idx = down_candles.index[-1]; current_ob_start = df_copy.loc[ob_candle_idx, 'date'];
        if current_ob_start <= last_bull_ob_end or ob_candle_idx in processed_indices_bull: continue
        ob_candle = df_copy.loc[ob_candle_idx]; block_high = ob_candle['high']; block_low = ob_candle['low']; trigger_price = block_low; subsequent_high_after_pivot = df_copy.loc[idx_pivot_low:]['high'].max()
        if subsequent_high_after_pivot > block_high: blocks.append({'type': 'bullish_ob','start_date': ob_candle['date'],'end_date': ob_candle['date'],'high': block_high,'low': block_low,'trigger_price': trigger_price,'pivot_date': df_copy.loc[idx_pivot_low, 'date'],'broken': False,'weight': INDICATOR_WEIGHTS['order_blocks']}); last_bull_ob_end = ob_candle['date']; processed_indices_bull.add(ob_candle_idx); bullish_obs_found += 1
    last_bear_ob_end = pd.Timestamp.min.tz_localize(df_copy['date'].dt.tz); processed_indices_bear = set(); bearish_obs_found = 0
    for idx_pivot_high in reversed(pivot_high_indices):
        if bearish_obs_found >= show_bear or idx_pivot_high == df_copy.index[0]: continue
        lookback_range = df_copy.loc[:idx_pivot_high].iloc[-swing_length-1:-1]; up_candles = lookback_range[lookback_range['close'] > lookback_range['open']];
        if not up_candles.empty: ob_candle_idx = up_candles.index[-1]; current_ob_start = df_copy.loc[ob_candle_idx, 'date'];
        if current_ob_start <= last_bear_ob_end or ob_candle_idx in processed_indices_bear: continue
        ob_candle = df_copy.loc[ob_candle_idx]; block_high = ob_candle['high']; block_low = ob_candle['low']; trigger_price = block_high; subsequent_low_after_pivot = df_copy.loc[idx_pivot_high:]['low'].min()
        if subsequent_low_after_pivot < block_low: blocks.append({'type': 'bearish_ob','start_date': ob_candle['date'],'end_date': ob_candle['date'],'high': block_high,'low': block_low,'trigger_price': trigger_price,'pivot_date': df_copy.loc[idx_pivot_high, 'date'],'broken': False,'weight': INDICATOR_WEIGHTS['order_blocks']}); last_bear_ob_end = ob_candle['date']; processed_indices_bear.add(ob_candle_idx); bearish_obs_found += 1
    last_date = df_copy['date'].iloc[-1]
    for block in blocks:
        if block['end_date'] < last_date:
            if block['type'] == 'bullish_ob': subsequent_data = df_copy[df_copy['date'] > block['pivot_date']];
            if not subsequent_data.empty and (subsequent_data['close'] < block['low']).any(): block['broken'] = True; block['breaker_type'] = 'bullish_breaker'
            elif block['type'] == 'bearish_ob': subsequent_data = df_copy[df_copy['date'] > block['pivot_date']];
            if not subsequent_data.empty and (subsequent_data['close'] > block['high']).any(): block['broken'] = True; block['breaker_type'] = 'bearish_breaker'
    blocks.sort(key=lambda x: x['start_date']); return df_copy, blocks
def plot_order_blocks(fig, blocks, current_price): # ...
    if not isinstance(fig, go.Figure) or not blocks: return fig
    colors = {"bull_ob": "rgba(0,0,255,0.2)", "bear_ob": "rgba(255,165,0,0.2)", "bull_breaker": "rgba(255,0,0,0.1)", "bear_breaker": "rgba(0,255,0,0.1)", "line_bull": "blue", "line_bear": "orange", "line_bull_br": "red", "line_bear_br": "green"}
    max_blocks_plot = 10; plotted = 0
    for block in reversed(blocks):
        if plotted >= max_blocks_plot: break
        is_breaker = block.get('broken',False); b_type=block.get('type'); br_type=block.get('breaker_type'); fill_color, line_color, line_width = None,None,0
        if not is_breaker:
            if b_type=='bullish_ob': fill_color, line_color = colors['bull_ob'], colors['line_bull']
            elif b_type=='bearish_ob': fill_color, line_color = colors['bear_ob'], colors['line_bear']
            else: continue
        else:
            line_width=1
            if br_type=='bullish_breaker': fill_color, line_color = colors['bull_breaker'], colors['line_bull_br']
            elif br_type=='bearish_breaker': fill_color, line_color = colors['bear_breaker'], colors['line_bear_br']
            else: continue
        end_visual = block['end_date'] + pd.Timedelta(hours=12) if block['start_date'] == block['end_date'] else block['end_date']
        try: fig.add_shape(type="rect", x0=block['start_date'], y0=block['low'], x1=end_visual, y1=block['high'], line=dict(color=line_color, width=line_width), fillcolor=fill_color, layer="below"); plotted += 1
        except Exception as e: print(f"Warning plot block: {e}")
    return fig
def detect_support_resistance_clusters(prices, n_clusters=5): # ...
    if not SKLEARN_AVAILABLE or not isinstance(prices,(np.ndarray, pd.Series)): return []
    prices_clean = prices[~np.isnan(prices)];
    if len(prices_clean) < n_clusters: return []
    X = np.array(prices_clean).reshape(-1, 1); scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
    try: kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42).fit(X_scaled); clusters = sorted([c[0] for c in scaler.inverse_transform(kmeans.cluster_centers_)]); return clusters
    except Exception as e: st.warning(f"Erro K-Means S/R: {e}"); return []
def detect_divergences(price_series, indicator_series, window=14): # ...
    if not isinstance(price_series, pd.Series) or not isinstance(indicator_series, pd.Series) or price_series.empty or indicator_series.empty: return pd.DataFrame({'divergence': 0}, index=price_series.index)
    df = pd.DataFrame({'price': price_series, 'indicator': indicator_series}).dropna();
    if len(df) < window: return pd.DataFrame({'divergence': 0}, index=price_series.index).fillna(0)
    price_roll_max=df['price'].rolling(window, center=True).max(); price_roll_min=df['price'].rolling(window, center=True).min(); ind_roll_max=df['indicator'].rolling(window, center=True).max(); ind_roll_min=df['indicator'].rolling(window, center=True).min()
    bearish_div = (df['price']==price_roll_max) & (df['indicator'] < ind_roll_max.shift(1)) & (df['price'] > df['price'].shift(1)); bullish_div = (df['price']==price_roll_min) & (df['indicator'] > ind_roll_min.shift(1)) & (df['price'] < df['price'].shift(1))
    df['divergence'] = 0; df.loc[bearish_div, 'divergence'] = -1; df.loc[bullish_div, 'divergence'] = 1
    return df[['divergence']].reindex(price_series.index).fillna(0)

# --- Nova Função: Calcular Pivot Points ---
def calculate_pivot_points(high, low, close):
    """Calcula Pivot Points e níveis de Suporte/Resistência."""
    if pd.isna(high) or pd.isna(low) or pd.isna(close):
        return {} # Retorna vazio se dados ausentes
    PP = (high + low + close) / 3
    S1 = (PP * 2) - high
    R1 = (PP * 2) - low
    S2 = PP - (high - low)
    R2 = PP + (high - low)
    S3 = low - 2 * (high - PP)
    R3 = high + 2 * (PP - low)
    return {'PP': PP, 'S1': S1, 'S2': S2, 'S3': S3, 'R1': R1, 'R2': R2, 'R3': R3}

# --- Funções Dados/Plotagem ---
def get_exchange_flows_simulated(): exchanges = ["Binance","Coinbase","Kraken","Outros"]; inflows=np.random.uniform(500, 5000, size=len(exchanges)); outflows=np.random.uniform(400, 4800, size=len(exchanges)); netflows=inflows-outflows; return pd.DataFrame({'Exchange':exchanges,'Entrada':inflows,'Saída':outflows,'Líquido':netflows})
def plot_hashrate_difficulty(data): # ... (mantida) ...
    if 'hashrate' not in data or 'difficulty' not in data or data['hashrate'].empty or data['difficulty'].empty: return None
    fig = go.Figure(); fig.add_trace(go.Scatter(x=data['hashrate']['date'], y=data['hashrate']['y'], name="Hashrate (TH/s)", line=dict(color='blue'))); fig.add_trace(go.Scatter(x=data['difficulty']['date'], y=data['difficulty']['y'], name="Dificuldade (T)", yaxis="y2", line=dict(color='red'))); fig.update_layout(title_text="Hashrate vs Dificuldade", xaxis_title="Data", yaxis=dict(title_text="Hashrate (TH/s)", titlefont=dict(color="blue"), tickfont=dict(color="blue")), yaxis2=dict(title_text="Dificuldade (T)", titlefont=dict(color="red"), tickfont=dict(color="red"), overlaying="y", side="right"), hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)); return fig
def plot_whale_activity_simulated(data): # ... (mantida) ...
    if 'whale_alert_simulated' not in data or data['whale_alert_simulated'].empty: return None
    df_whale = data['whale_alert_simulated']; fig = go.Figure(go.Bar(x=df_whale['date'],y=df_whale['amount'],name="BTC Mov.", marker_color='orange', text=df_whale['exchange'])); fig.update_layout(title_text="Atividade Whale (Simulado)", xaxis_title="Data", yaxis_title="BTC", hovermode="x unified"); return fig
def simulate_event(event, price_series): # ... (mantida) ...
    if not isinstance(price_series, pd.Series) or price_series.empty: return None
    simulated = price_series.copy(); n_days = len(simulated)
    try:
        if event=="Halving": daily_growth_factor=(2.2)**(1/365); factors=daily_growth_factor**np.arange(n_days); return simulated*factors
        elif event=="Crash": return simulated*0.7
        elif event=="ETF Approval": return simulated*1.5
        else: return price_series.copy()
    except Exception as e: st.error(f"Erro simulação {event}: {e}"); return price_series.copy()
def get_market_sentiment(): # ... (mantida) ...
    try: response = requests.get("https://api.alternative.me/fng/", timeout=10); response.raise_for_status(); data = response.json(); value = int(data.get("data", [{}])[0].get("value", 50)); sentiment = data.get("data", [{}])[0].get("value_classification", "Neutral"); return {"value": value, "sentiment": sentiment}
    except Exception as e: st.warning(f"Falha F&G: {e}"); return {"value": 50, "sentiment": "Neutral"}
def filter_news_by_confidence(news_data, min_confidence=0.7): # ... (mantida) ...
    if not isinstance(news_data, list): return []
    return [news for news in news_data if news.get('confidence', news.get('sentiment_score', 0)) >= min_confidence]
def get_traditional_assets(): # ... (mantida com correção) ...
    assets = {"BTC-USD": "BTC-USD", "S&P 500": "^GSPC", "Ouro": "GC=F", "ETH-USD": "ETH-USD"}; dfs = []; end_date = datetime.now(); start_date = end_date - timedelta(days=95)
    for name, ticker in assets.items():
        try:
            data_yf = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
            if not data_yf.empty and 'Close' in data_yf.columns:
                data_proc = data_yf['Close'].resample('1D').ffill().to_frame()
                data_proc = data_proc.reset_index().rename(columns={'Close': 'value', 'Date': 'date'})
                data_proc['date'] = pd.to_datetime(data_proc['date']).dt.normalize()
                data_proc['asset'] = name
                dfs.append(data_proc.tail(90))
        except Exception as e:
            st.warning(f"Falha ao buscar {name} ({ticker}): {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# --- Funções Backtesting ---
def calculate_daily_returns(df): # ... (mantida) ...
    if df.empty or 'price' not in df.columns: return df
    df_copy = df.copy(); df_copy['daily_return'] = df_copy['price'].pct_change(); df_copy['cumulative_return'] = (1 + df_copy['daily_return']).cumprod(); return df_copy
def calculate_strategy_returns(df, signal_col='signal'): # ... (mantida) ...
    if df.empty or 'daily_return' not in df.columns or signal_col not in df.columns: return df
    df_copy = df.copy(); df_copy['strategy_return'] = df_copy[signal_col].shift(1) * df_copy['daily_return']; df_copy['strategy_cumulative'] = (1 + df_copy['strategy_return'].fillna(0)).cumprod(); return df_copy
# Colar funções backtest_* aqui
def backtest_rsi_strategy(df_input, rsi_window=14, overbought=70, oversold=30):
    if df_input.empty or 'price' not in df_input.columns: return pd.DataFrame()
    df = df_input.copy(); df[f'RSI_{rsi_window}'] = calculate_rsi(df['price'], rsi_window)
    df['signal'] = 0.0; buy_condition = (df[f'RSI_{rsi_window}'] < oversold); sell_condition = (df[f'RSI_{rsi_window}'] > overbought)
    df.loc[buy_condition, 'signal'] = 1.0 * INDICATOR_WEIGHTS['rsi']; df.loc[sell_condition, 'signal'] = -1.0 * INDICATOR_WEIGHTS['rsi']
    df['signal'] = df['signal'].fillna(0.0); df = calculate_daily_returns(df); df = calculate_strategy_returns(df); return df
def backtest_macd_strategy(df_input, fast=12, slow=26, signal_macd_window=9):
    if df_input.empty or 'price' not in df_input.columns: return pd.DataFrame()
    df = df_input.copy(); macd_col, signal_col = f'MACD_{fast}_{slow}', f'MACD_Signal_{signal_macd_window}'; df[macd_col], df[signal_col] = calculate_macd(df['price'], fast, slow, signal_macd_window)
    df['signal'] = 0.0; buy_cross = (df[macd_col] > df[signal_col]) & (df[macd_col].shift(1) <= df[signal_col].shift(1)); sell_cross = (df[macd_col] < df[signal_col]) & (df[macd_col].shift(1) >= df[signal_col].shift(1))
    df.loc[buy_cross, 'signal'] = 1.0 * INDICATOR_WEIGHTS['macd']; df.loc[sell_cross, 'signal'] = -1.0 * INDICATOR_WEIGHTS['macd']
    df['signal'] = df['signal'].fillna(0.0); df = calculate_daily_returns(df); df = calculate_strategy_returns(df); return df
def backtest_bollinger_strategy(df_input, window=20, num_std=2):
    if df_input.empty or 'price' not in df_input.columns: return pd.DataFrame()
    df = df_input.copy(); bb_upper_col, bb_lower_col = f'BB_Upper_{window}', f'BB_Lower_{window}'; df[bb_upper_col], df[bb_lower_col] = calculate_bollinger_bands(df['price'], window, num_std)
    df['signal'] = 0.0; buy_condition = df['price'] <= df[bb_lower_col]; sell_condition = df['price'] >= df[bb_upper_col]
    df.loc[buy_condition, 'signal'] = 1.0 * INDICATOR_WEIGHTS['bollinger']; df.loc[sell_condition, 'signal'] = -1.0 * INDICATOR_WEIGHTS['bollinger']
    df['signal'] = df['signal'].fillna(0.0); df = calculate_daily_returns(df); df = calculate_strategy_returns(df); return df
def backtest_ema_cross_strategy(df_input, short_window=9, long_window=21):
    if df_input.empty or 'price' not in df_input.columns or short_window >= long_window: return pd.DataFrame()
    df = df_input.copy(); ema_short_col, ema_long_col = f'EMA_{short_window}', f'EMA_{long_window}'; df[ema_short_col] = calculate_ema(df['price'], short_window); df[ema_long_col] = calculate_ema(df['price'], long_window)
    df['signal'] = 0.0; buy_condition = (df[ema_short_col] > df[ema_long_col]) & (df[ema_short_col].shift(1) <= df[ema_long_col].shift(1)); sell_condition = (df[ema_short_col] < df[ema_long_col]) & (df[ema_short_col].shift(1) >= df[ema_long_col].shift(1))
    df.loc[buy_condition, 'signal'] = 1.0 * INDICATOR_WEIGHTS['ma_cross']; df.loc[sell_condition, 'signal'] = -1.0 * INDICATOR_WEIGHTS['ma_cross']
    df['signal'] = df['signal'].fillna(0.0); df = calculate_daily_returns(df); df = calculate_strategy_returns(df); return df
def backtest_volume_strategy(df_input, volume_window=20, threshold=1.5):
    if df_input.empty or 'price' not in df_input.columns or 'volume' not in df_input.columns: return pd.DataFrame()
    df = df_input.copy(); vol_ma_col = f'Volume_MA_{volume_window}'; df[vol_ma_col] = df['volume'].rolling(volume_window).mean(); df['Volume_Ratio'] = df['volume'] / df[vol_ma_col]; df['Price_Change'] = df['price'].diff()
    df['signal'] = 0.0; buy_condition = (df['Volume_Ratio'] > threshold) & (df['Price_Change'] > 0); sell_condition = (df['Volume_Ratio'] > threshold) & (df['Price_Change'] < 0)
    df.loc[buy_condition, 'signal'] = 1.0 * INDICATOR_WEIGHTS['volume']; df.loc[sell_condition, 'signal'] = -1.0 * INDICATOR_WEIGHTS['volume']
    df['signal'] = df['signal'].fillna(0.0); df = calculate_daily_returns(df); df = calculate_strategy_returns(df); return df
def backtest_obv_strategy(df_input, obv_window=20, price_window=30):
    if df_input.empty or 'price' not in df_input.columns or 'volume' not in df_input.columns: return pd.DataFrame()
    df = df_input.copy(); df['OBV'] = calculate_obv(df['price'], df['volume']); obv_ma_col = f'OBV_MA_{obv_window}'; price_ma_col = f'Price_MA_{price_window}'; df[obv_ma_col] = df['OBV'].rolling(obv_window).mean(); df[price_ma_col] = df['price'].rolling(price_window).mean()
    df['signal'] = 0.0; buy_condition = (df['OBV'] > df[obv_ma_col]) & (df['price'] > df[price_ma_col]); sell_condition = (df['OBV'] < df[obv_ma_col]) & (df['price'] < df[price_ma_col])
    df.loc[buy_condition, 'signal'] = 1.0 * INDICATOR_WEIGHTS['obv']; df.loc[sell_condition, 'signal'] = -1.0 * INDICATOR_WEIGHTS['obv']
    df['signal'] = df['signal'].fillna(0.0); df = calculate_daily_returns(df); df = calculate_strategy_returns(df); return df
def backtest_stochastic_strategy(df_input, k_window=14, d_window=3, overbought=80, oversold=20):
    if df_input.empty or not all(c in df_input.columns for c in ['price', 'high', 'low']): return pd.DataFrame()
    # Usa calculate_kdj para obter K e D
    df = df_input.copy(); kdj_k, kdj_d, _ = calculate_kdj(df['price'], df['high'], df['low'], k_window, d_window)
    df[f'Stoch_K_{k_window}_{d_window}'] = kdj_k; df[f'Stoch_D_{k_window}_{d_window}'] = kdj_d # Salva no df se precisar
    df['signal'] = 0.0; buy_condition = (kdj_k < oversold) & (kdj_d < oversold); sell_condition = (kdj_k > overbought) & (kdj_d > overbought)
    df.loc[buy_condition, 'signal'] = 1.0 * INDICATOR_WEIGHTS['stochastic']; df.loc[sell_condition, 'signal'] = -1.0 * INDICATOR_WEIGHTS['stochastic']
    df['signal'] = df['signal'].fillna(0.0); df = calculate_daily_returns(df); df = calculate_strategy_returns(df); return df
def backtest_gp_strategy(df_input, window=30, threshold=0.02):
    if df_input.empty or 'price' not in df_input.columns or not SKLEARN_AVAILABLE: return pd.DataFrame()
    df = df_input.copy(); gp_pred_col = f'GP_Prediction_{window}'; df[gp_pred_col] = calculate_gaussian_process(df['price'], window=window, lookahead=1)
    df['signal'] = 0.0; buy_condition = df[gp_pred_col] > df['price'] * (1 + threshold); sell_condition = df[gp_pred_col] < df['price'] * (1 - threshold)
    df.loc[buy_condition, 'signal'] = 1.0 * INDICATOR_WEIGHTS['gaussian_process']; df.loc[sell_condition, 'signal'] = -1.0 * INDICATOR_WEIGHTS['gaussian_process']
    df['signal'] = df['signal'].fillna(0.0); df = calculate_daily_returns(df); df = calculate_strategy_returns(df); return df
def backtest_order_block_strategy(df_input, swing_length=11, use_body=True):
    if df_input.empty or not all(c in df_input.columns for c in ['date', 'open', 'high', 'low', 'close']): return pd.DataFrame()
    df = df_input.copy(); df, blocks = identify_order_blocks(df, swing_length=swing_length, show_bull=999, show_bear=999, use_body=use_body)
    df['signal'] = 0.0; last_signal = 0.0
    for i in range(1, len(df)):
        current_price = df['price'].iloc[i]; current_date = df['date'].iloc[i]; active_signal = 0.0; relevant_blocks = [b for b in blocks if b['end_date'] < current_date]
        for block in reversed(relevant_blocks):
            is_br, b_type, br_type = block.get('broken',False), block.get('type'), block.get('breaker_type'); in_b, in_brz = block['low']<=current_price<=block['high'], block['low']*0.99<=current_price<=block['high']*1.01
            if not is_br:
                if b_type=='bullish_ob' and in_b: active_signal = 1.0 * block['weight']; break
                elif b_type=='bearish_ob' and in_b: active_signal = -1.0 * block['weight']; break
            else:
                if br_type=='bullish_breaker' and in_brz: active_signal = -1.0 * block['weight']; break
                elif br_type=='bearish_breaker' and in_brz: active_signal = 1.0 * block['weight']; break
        if active_signal != 0.0: last_signal = active_signal
        df.loc[df.index[i], 'signal'] = last_signal
    df['signal'] = df['signal'].fillna(0.0); df = calculate_daily_returns(df); df = calculate_strategy_returns(df); return df
def calculate_metrics(df): # ... (mantida) ...
    metrics = {}; required_cols = ['strategy_return', 'daily_return', 'strategy_cumulative', 'cumulative_return', 'signal']
    if df.empty or not all(col in df.columns for col in required_cols): return metrics
    returns = df['strategy_return'].dropna(); buy_hold_returns = df['daily_return'].dropna();
    if len(returns) < 2 or len(buy_hold_returns) < 2: return metrics
    metrics['Retorno Estratégia'] = df['strategy_cumulative'].iloc[-1] - 1; metrics['Retorno Buy & Hold'] = df['cumulative_return'].iloc[-1] - 1; metrics['Vol Estratégia'] = returns.std() * np.sqrt(365); metrics['Vol Buy & Hold'] = buy_hold_returns.std() * np.sqrt(365)
    strat_std = metrics['Vol Estratégia'] / np.sqrt(365); bh_std = metrics['Vol Buy & Hold'] / np.sqrt(365); metrics['Sharpe Estratégia'] = (returns.mean() * 365) / metrics['Vol Estratégia'] if metrics['Vol Estratégia'] > 1e-9 else 0.0; metrics['Sharpe Buy & Hold'] = (buy_hold_returns.mean() * 365) / metrics['Vol Buy & Hold'] if metrics['Vol Buy & Hold'] > 1e-9 else 0.0
    cum_returns_strat = df['strategy_cumulative']; peak_strat = cum_returns_strat.expanding(min_periods=1).max(); drawdown_strat = (cum_returns_strat - peak_strat) / peak_strat; metrics['Max Drawdown Estratégia'] = drawdown_strat.min() if not drawdown_strat.empty else 0.0
    cum_returns_bh = df['cumulative_return']; peak_bh = cum_returns_bh.expanding(min_periods=1).max(); drawdown_bh = (cum_returns_bh - peak_bh) / peak_bh; metrics['Max Drawdown Buy & Hold'] = drawdown_bh.min() if not drawdown_bh.empty else 0.0
    metrics['Win Rate Diário Estratégia'] = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0.0; trades = df[df['signal'].diff().fillna(0) != 0]; winning_trades = trades[trades['strategy_return'] > 0] # Aprox.
    metrics['Taxa Acerto (Trades Aprox.)'] = len(winning_trades) / len(trades) if not trades.empty else 0.0; return metrics
def optimize_strategy_parameters(data, strategy_name, param_space): # ... (mantida) ...
    if 'prices' not in data or data['prices'].empty: st.error("Dados ausentes otim."); return None, -np.inf, None;
    if not SKLEARN_AVAILABLE: st.error("SKlearn necessário otim."); return None, -np.inf, None
    param_combinations = list(ParameterGrid(param_space));
    if not param_combinations: st.warning("Sem combinações params."); return None, -np.inf, None
    best_sharpe = -np.inf; best_params = None; best_results_df = None; total_combinations = len(param_combinations); progress_bar = st.progress(0); status_text = st.empty(); st.warning(f"Otimizando {total_combinations} combinações...")
    required_cols_map={'RSI': ['price'], 'MACD': ['price'], 'Bollinger': ['price'], 'EMA Cross': ['price'], 'Volume': ['price', 'volume'], 'OBV': ['price', 'volume'], 'Stochastic': ['price', 'high', 'low'], 'Gaussian Process': ['price'], 'Order Blocks': ['date', 'open', 'high', 'low', 'close']}
    strat_req_cols = required_cols_map.get(strategy_name, ['price'])
    if not all(col in data['prices'].columns for col in strat_req_cols): st.error(f"Dados insuficientes ({strat_req_cols}) otimizar {strategy_name}."); return None, -np.inf, None
    for i, params in enumerate(param_combinations):
        df_result = None; current_sharpe = -np.inf
        try:
            if strategy_name=='RSI': df_result=backtest_rsi_strategy(data['prices'], **params)
            elif strategy_name=='MACD': m_params=params.copy(); m_params['signal_macd_window']=m_params.pop('signal', 9); df_result=backtest_macd_strategy(data['prices'], **m_params)
            elif strategy_name=='Bollinger': df_result=backtest_bollinger_strategy(data['prices'], **params)
            elif strategy_name=='EMA Cross': df_result=backtest_ema_cross_strategy(data['prices'], **params)
            elif strategy_name=='Volume': df_result=backtest_volume_strategy(data['prices'], **params)
            elif strategy_name=='OBV': df_result=backtest_obv_strategy(data['prices'], **params)
            elif strategy_name=='Stochastic': df_result=backtest_stochastic_strategy(data['prices'], **params)
            elif strategy_name=='Gaussian Process': gp_params=params.copy(); gp_params['lookahead']=1; df_result=backtest_gp_strategy(data['prices'], **gp_params)
            elif strategy_name=='Order Blocks': df_result=backtest_order_block_strategy(data['prices'], **params)
            else: continue
            if df_result is not None and not df_result.empty and 'strategy_return' in df_result.columns:
                returns = df_result['strategy_return'].dropna();
                if len(returns) > 1: current_std = returns.std(); current_sharpe = (returns.mean()/current_std)*np.sqrt(365) if current_std > 1e-9 else 0.0
                if current_sharpe > best_sharpe: best_sharpe=current_sharpe; best_params=params; best_results_df=df_result
        except Exception: continue
        progress=(i+1)/total_combinations; progress_bar.progress(progress); status_text.text(f"Testando {i+1}/{total_combinations} | Melhor Sharpe: {best_sharpe:.2f}")
    progress_bar.empty(); status_text.empty();
    if best_params: st.success(f"Otimização Concluída! Melhor Sharpe: {best_sharpe:.2f}")
    else: st.warning("Nenhuma combinação válida.")
    return best_params, best_sharpe, best_results_df

# --- Carregamento de Dados (com correção erro tuple) ---
@st.cache_data(ttl=3600, show_spinner="Carregando e processando dados de mercado...")
def load_and_process_data():
    data = {'prices': pd.DataFrame()} # Inicia com DF vazio
    try:
        ticker = "BTC-USD"; btc_data_raw = yf.download(ticker, period="1y", interval="1d", progress=False)
        if not isinstance(btc_data_raw, pd.DataFrame) or btc_data_raw.empty: raise ValueError(f"yfinance não retornou DataFrame para {ticker}.")
        btc_data = btc_data_raw.copy()
        # --- CORREÇÃO ERRO TUPLE / COLUNAS ---
        column_map = {'open': ['Open', 'open'],'high': ['High', 'high'],'low': ['Low', 'low'],'close': ['Close', 'close'],'volume': ['Volume', 'volume']}
        renamed_cols = {}; found_cols = {}
        for standard_name, possible_names in column_map.items():
            for possible_name in btc_data.columns: # Itera sobre colunas existentes
                col_str = possible_name[0] if isinstance(possible_name, tuple) else possible_name
                if isinstance(col_str, str) and col_str.lower() == standard_name: renamed_cols[possible_name] = standard_name; found_cols[standard_name] = True; break
        btc_data.rename(columns=renamed_cols, inplace=True)
        if not found_cols.get('close'): adj_close_cols = [c for c in btc_data.columns if isinstance(c, str) and c.lower().replace(' ','') == 'adjclose'];
        if adj_close_cols: btc_data.rename(columns={adj_close_cols[0]: 'close'}, inplace=True); found_cols['close'] = True
        required_ohlcv = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_ohlcv if col not in btc_data.columns]
        if missing_cols: raise ValueError(f"Colunas OHLCV ausentes: {missing_cols}. Disponíveis: {list(btc_data.columns)}")
        # --- FIM CORREÇÃO COLUNAS ---
        btc_data.reset_index(inplace=True); date_col = 'Date' if 'Date' in btc_data.columns else 'index' if 'index' in btc_data.columns else None
        if not date_col: raise ValueError("Coluna Date/index não encontrada.")
        btc_data.rename(columns={date_col: 'date'}, inplace=True)
        btc_data['date'] = pd.to_datetime(btc_data['date']).dt.normalize(); btc_data['price'] = btc_data['close']
        cols_order = ['date', 'open', 'high', 'low', 'close', 'price', 'volume']
        cols_order = [c for c in cols_order if c in btc_data.columns]
        # Garante que as colunas base para RL existam antes de incluir
        rl_base_existing = [c for c in RL_OBSERVATION_COLS_BASE if c in btc_data.columns]
        all_needed_cols = list(set(cols_order + rl_base_existing)); all_needed_cols = [c for c in all_needed_cols if c in btc_data.columns]
        data['prices'] = btc_data[all_needed_cols].sort_values('date').reset_index(drop=True)

        # --- Calcular Indicadores ---
        if not data['prices'].empty and SKLEARN_AVAILABLE:
            df = data['prices']
            # MAs
            all_ma_windows = list(set(DEFAULT_SETTINGS['ma_windows'] + [5, 10, 20, 120])) # Inclui MAs da CoinAnk
            for window in all_ma_windows: df[f'MA{window}'] = df['price'].rolling(window).mean()
            # RSIs
            rsi_windows = list(set([DEFAULT_SETTINGS["rsi_window"], 6, 12, 14]))
            for window in rsi_windows: df[f'RSI_{window}'] = calculate_rsi(df['price'], window)
            df['RSI_14'] = df[f'RSI_{DEFAULT_SETTINGS["rsi_window"]}'] # Alias
            # MACD
            df[f'MACD_12_26'], df[f'MACD_Signal_9'] = calculate_macd(df['price'], 12, 26, 9); df['MACD'] = df['MACD_12_26']; df['MACD_Signal'] = df['MACD_Signal_9']
            # Bollinger
            bb_w = DEFAULT_SETTINGS["bb_window"]; upper, lower = calculate_bollinger_bands(df['price'], bb_w); df[f'BB_Upper_{bb_w}'] = upper; df[f'BB_Lower_{bb_w}'] = lower; df['BB_Upper_20'] = upper; df['BB_Lower_20'] = lower
            # Bollinger %B
            df['BB_PercentB'] = (df['price'] - df[f'BB_Lower_{bb_w}']) / (df[f'BB_Upper_{bb_w}'] - df[f'BB_Lower_{bb_w}']).replace(0, np.nan)
            # OBV
            df['OBV'] = calculate_obv(df['price'], df['volume']); df['OBV_MA_20'] = df['OBV'].rolling(20).mean()
            # KDJ (Stochastic com J)
            k, d, j = calculate_kdj(df['price'], df['high'], df['low'], 14, 3, 3); df['Stoch_K_14_3'] = k; df['Stoch_D_14_3'] = d; df['Stoch_J_14_3'] = j
            # Gaussian Process
            gp_w = DEFAULT_SETTINGS["gp_window"]; df[f'GP_Prediction_{gp_w}'] = calculate_gaussian_process(df['price'], gp_w, 1)
            # Divergências
            df['RSI_Divergence'] = detect_divergences(df['price'], df[f'RSI_{DEFAULT_SETTINGS["rsi_window"]}'])
            # S/R Clusters
            lookback_sr = 90
            if len(df['price'].dropna()) >= lookback_sr: data['support_resistance'] = detect_support_resistance_clusters(df['price'].dropna().tail(lookback_sr).values, DEFAULT_SETTINGS['n_clusters'])
            else: data['support_resistance'] = []
            # Volume MA
            df['Volume_MA_20'] = df['volume'].rolling(20).mean()
            # Pivot Points (para o dia atual, baseado no dia anterior)
            if len(df) > 1:
                 pivot_data = calculate_pivot_points(df['high'].iloc[-2], df['low'].iloc[-2], df['close'].iloc[-2])
                 data['pivot_points'] = pivot_data
            else: data['pivot_points'] = {}

            data['prices'] = df

        # --- Dados On-Chain / Simulados ---
        try: hr_response=requests.get("https://api.blockchain.info/charts/hash-rate?format=json&timespan=1year", timeout=10); hr_response.raise_for_status(); hr_data=pd.DataFrame(hr_response.json()["values"]); hr_data["date"]=pd.to_datetime(hr_data["x"], unit="s").dt.normalize(); hr_data['y']=hr_data['y']/1e12; data['hashrate']=hr_data[['date', 'y']].dropna()
        except Exception: data['hashrate']=pd.DataFrame({'date': [], 'y': []})
        try: diff_response=requests.get("https://api.blockchain.info/charts/difficulty?timespan=1year&format=json", timeout=10); diff_response.raise_for_status(); diff_data=pd.DataFrame(diff_response.json()["values"]); diff_data["date"]=pd.to_datetime(diff_data["x"], unit="s").dt.normalize(); diff_data['y']=diff_data['y']/1e12; data['difficulty']=diff_data[['date', 'y']].dropna()
        except Exception: data['difficulty']=pd.DataFrame({'date': [], 'y': []})
        data['exchanges_simulated'] = get_exchange_flows_simulated()
        news_end_date = datetime.now(tz='UTC').normalize()
        data['whale_alert_simulated'] = pd.DataFrame({"date": pd.date_range(end=news_end_date - timedelta(days=1), periods=5, freq='12H'), "amount": np.random.randint(500, 5000, 5), "exchange": ["Binance", "Coinbase", "Kraken", "Unknown", "Binance"]})
        data['news'] = [{"title": f"Notícia Simulada {i}", "date": news_end_date - timedelta(days=i), "confidence": np.random.uniform(0.6, 0.95), "source": "Fonte Simulada"} for i in range(5)]

    except Exception as e:
        st.error(f"Erro fatal ao carregar/processar dados: {e}")
        # Retorna estrutura mínima para evitar quebrar o resto do app
        data = {'prices': pd.DataFrame(), 'hashrate': pd.DataFrame(), 'difficulty': pd.DataFrame(),
                'exchanges_simulated': pd.DataFrame(), 'whale_alert_simulated': pd.DataFrame(),
                'news': [], 'support_resistance': [], 'pivot_points': {}}
    return data

# --- Geração de Sinais V2 ---
def generate_signals_v2(data, settings, lstm_prediction=None, rl_action=None):
    signals = []; buy_score, sell_score, neutral_score = 0.0, 0.0, 0.0
    df = data.get('prices', pd.DataFrame());
    if df.empty: return signals, "➖ DADOS INDISP.", 0, 0
    if len(df) < 2: return signals, "➖ DADOS INSUF.", 0, 0
    last_row = df.iloc[-1]; prev_row = df.iloc[-2]
    last_price = last_row.get('price', np.nan)
    if pd.isna(last_price): return signals, "➖ PREÇO INDISP.", 0, 0

    def add_signal(name, condition_buy, condition_sell, value_display, weight):
        nonlocal buy_score, sell_score, neutral_score; signal_text = "NEUTRO"; score = 0.0
        if condition_buy: signal_text = "COMPRA"; score = 1.0 * weight; buy_score += score
        elif condition_sell: signal_text = "VENDA"; score = -1.0 * weight; sell_score += abs(score)
        else: neutral_score += weight
        signals.append({'name': name, 'signal': signal_text, 'value': str(value_display), 'score': score, 'weight': weight})

    # --- Sinais Técnicos ---
    rsi_val = last_row.get(f'RSI_{settings["rsi_window"]}', np.nan);
    if not pd.isna(rsi_val): add_signal(f"RSI ({settings['rsi_window']})", rsi_val < 30, rsi_val > 70, f"{rsi_val:.1f}", INDICATOR_WEIGHTS['rsi'])
    macd_val, macd_sig = last_row.get('MACD', np.nan), last_row.get('MACD_Signal', np.nan);
    if not pd.isna(macd_val) and not pd.isna(macd_sig): add_signal("MACD", macd_val > macd_sig, macd_val < macd_sig, f"{macd_val:.2f}/{macd_sig:.2f}", INDICATOR_WEIGHTS['macd'])
    bb_u, bb_l = last_row.get(f'BB_Upper_{settings["bb_window"]}', np.nan), last_row.get(f'BB_Lower_{settings["bb_window"]}', np.nan);
    if not pd.isna(bb_u) and not pd.isna(bb_l): add_signal(f"Bollinger ({settings['bb_window']})", last_price < bb_l, last_price > bb_u, f"${bb_l:,.0f}-${bb_u:,.0f}", INDICATOR_WEIGHTS['bollinger'])
    vol, vol_ma = last_row.get('volume', np.nan), last_row.get('Volume_MA_20', np.nan); price_chg = last_price - prev_row.get('price', last_price);
    if not pd.isna(vol) and not pd.isna(vol_ma) and vol_ma > 0: vol_r = vol/vol_ma; add_signal("Volume", vol_r > 1.5 and price_chg > 0, vol_r > 1.5 and price_chg < 0, f"{vol_r:.1f}x", INDICATOR_WEIGHTS['volume'])
    obv, obv_ma = last_row.get('OBV', np.nan), last_row.get('OBV_MA_20', np.nan);
    if not pd.isna(obv) and not pd.isna(obv_ma): add_signal("OBV", obv > obv_ma and price_chg > 0, obv < obv_ma and price_chg < 0, f"{obv/1e6:.1f}M", INDICATOR_WEIGHTS['obv'])
    k, d, j = last_row.get('Stoch_K_14_3', np.nan), last_row.get('Stoch_D_14_3', np.nan), last_row.get('Stoch_J_14_3', np.nan); # KDJ
    if not pd.isna(k) and not pd.isna(d): add_signal("KDJ", k < 20 and d < 20, k > 80 and d > 80, f"K:{k:.1f},D:{d:.1f},J:{j:.1f}", INDICATOR_WEIGHTS['kdj']) # Sinal baseado em K/D, mas mostra J
    gp_pred = last_row.get(f'GP_Prediction_{settings["gp_window"]}', np.nan);
    if not pd.isna(gp_pred): gp_thresh=0.02; add_signal("Gauss Process", gp_pred > last_price*(1+gp_thresh), gp_pred < last_price*(1-gp_thresh), f"P:${gp_pred:,.0f}", INDICATOR_WEIGHTS['gaussian_process'])
    _, current_blocks = identify_order_blocks(data['prices'], **settings); ob_buy, ob_sell = False, False; ob_disp = "N/A"
    for block in reversed(current_blocks):
        is_br, b_type, br_type = block.get('broken',False), block.get('type'), block.get('breaker_type'); in_b, in_brz = block['low']<=last_price<=block['high'], block['low']*0.99<=last_price<=block['high']*1.01
        if not is_br:
            if b_type=='bullish_ob' and in_b: ob_buy=True; ob_disp=f"BullOB:{block['low']:,.0f}-{block['high']:,.0f}"; break
            if b_type=='bearish_ob' and in_b: ob_sell=True; ob_disp=f"BearOB:{block['low']:,.0f}-{block['high']:,.0f}"; break
        else:
            if br_type=='bullish_breaker' and in_brz: ob_sell=True; ob_disp=f"BullBrk:{block['low']:,.0f}-{block['high']:,.0f}"; break
            if br_type=='bearish_breaker' and in_brz: ob_buy=True; ob_disp=f"BearBrk:{block['low']:,.0f}-{block['high']:,.0f}"; break
    add_signal("Order Block", ob_buy, ob_sell, ob_disp, INDICATOR_WEIGHTS['order_blocks'])
    rsi_div = last_row.get('RSI_Divergence', 0); div_disp = "Nenhuma"; div_buy = (rsi_div == 1); div_sell = (rsi_div == -1)
    if div_buy: div_disp = "Alta"
    elif div_sell: div_disp = "Baixa"
    add_signal("Divergência", div_buy, div_sell, div_disp, INDICATOR_WEIGHTS['divergence'])

    # --- Sinais IA ---
    if lstm_prediction is not None: thresh=0.01; add_signal("LSTM Pred", lstm_prediction>last_price*(1+thresh), lstm_prediction<last_price*(1-thresh), f"P:${lstm_prediction:,.0f}", INDICATOR_WEIGHTS['lstm_pred'])
    if rl_action is not None: act_disp = {0:'Manter', 1:'Compra', 2:'Venda'}.get(rl_action,'N/A'); add_signal("RL Ação", rl_action==1, rl_action==2, act_disp, INDICATOR_WEIGHTS['rl_action'])

    # --- Veredito Final ---
    total_w = buy_score + sell_score + neutral_score; total_w = 1 if total_w==0 else total_w
    if buy_score > sell_score*1.8: final_verdict="✅ FORTE COMPRA"
    elif buy_score > sell_score*1.1: final_verdict="📈 COMPRA"
    elif sell_score > buy_score*1.8: final_verdict="❌ FORTE VENDA"
    elif sell_score > buy_score*1.1: final_verdict="📉 VENDA"
    else: final_verdict="➖ NEUTRO"
    buy_c = sum(1 for s in signals if s['signal']=='COMPRA'); sell_c = sum(1 for s in signals if s['signal']=='VENDA')
    return signals, final_verdict, buy_c, sell_c

# --- Funções PDF e Excel ---
def clean_text(text): # Mantida
    if text is None: return ""
    try: text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', str(text)); return str(text).encode('latin-1', 'ignore').decode('latin-1')
    except: return re.sub(r'[^\x20-\x7E]+', '', str(text))

# --- Nova Função: Relatório estilo CoinAnk ---
def generate_coinank_style_report(data):
    report_lines = []
    prices_df = data.get('prices')
    pivots = data.get('pivot_points', {})

    if prices_df is None or prices_df.empty or len(prices_df) < 2:
        return "Erro: Dados de preço insuficientes para gerar relatório."

    last = prices_df.iloc[-1]
    prev = prices_df.iloc[-2]

    # --- Cabeçalho ---
    current_price = last.get('price', np.nan)
    prev_price = prev.get('price', np.nan)
    change_24h = ((current_price / prev_price) - 1) * 100 if pd.notna(current_price) and pd.notna(prev_price) and prev_price != 0 else 0.0

    report_lines.append(f"**BTCUSDT 1d Market Analysis Report** (Gerado em {datetime.now().strftime('%Y-%m-%d %H:%M')})")
    report_lines.append(f"*Baseado nos dados de {prices_df['date'].iloc[-1].strftime('%Y-%m-%d')}*")
    report_lines.append("-" * 30)
    report_lines.append(f"Preço Atual: {current_price:,.2f} USDT" if pd.notna(current_price) else "Preço Atual: N/A")
    report_lines.append(f"Variação 24h: {change_24h:.3f}%")
    report_lines.append("-" * 30)

    # --- Suporte e Resistência (Pivot Points) ---
    report_lines.append("**Níveis Chave (Pivot Points Diários):**")
    if pivots:
        report_lines.append(f"  Suporte: S1={pivots.get('S1', 0):.2f}, S2={pivots.get('S2', 0):.2f}, S3={pivots.get('S3', 0):.2f} USDT")
        report_lines.append(f"  Resistência: R1={pivots.get('R1', 0):.2f}, R2={pivots.get('R2', 0):.2f}, R3={pivots.get('R3', 0):.2f} USDT")
        report_lines.append(f"  Pivot Point (PP): {pivots.get('PP', 0):.2f} USDT")
    else:
        report_lines.append("  N/A (Dados do dia anterior insuficientes)")
    report_lines.append("-" * 30)

    # --- Análise de Indicadores ---
    report_lines.append("**Visão Geral dos Indicadores Técnicos:**")
    ma_texts = []; macd_text = "MACD: N/A"; boll_text = "BOLL: N/A"; rsi_text = "RSI: N/A"; kdj_text = "KDJ: N/A"
    trend_score = 0 # +1 para bullish, -1 para bearish

    # MAs
    ma5, ma10, ma20, ma120 = last.get('MA5'), last.get('MA10'), last.get('MA20'), last.get('MA120')
    if all(pd.notna(v) for v in [ma5, ma10, ma20]):
        ma_short_term = "Altista" if ma5 > ma10 > ma20 else "Baixista" if ma5 < ma10 < ma20 else "Lateral"
        trend_score += 1 if ma_short_term == "Altista" else -1 if ma_short_term == "Baixista" else 0
        ma_texts.append(f"MA5({ma5:.0f}) {'acima' if ma5>ma10 else 'abaixo'} MA10({ma10:.0f}) e {'acima' if ma10>ma20 else 'abaixo'} MA20({ma20:.0f}) -> Alinhamento {ma_short_term}.")
    if pd.notna(ma120) and pd.notna(current_price):
        ma_long_term = "Altista" if current_price > ma120 else "Baixista"
        trend_score += 1 if ma_long_term == "Altista" else -1
        ma_texts.append(f"Preço {'acima' if ma_long_term=='Altista' else 'abaixo'} MA120({ma120:.0f}) -> Tendência longa {ma_long_term}.")
    if ma_texts: ma_text = " ".join(ma_texts)
    else: ma_text = "Sistema MA: N/A"

    # MACD
    macd_val, dea_val = last.get('MACD'), last.get('MACD_Signal')
    if pd.notna(macd_val) and pd.notna(dea_val):
        macd_hist = macd_val - dea_val
        macd_momentum = "Altista" if macd_val > dea_val else "Baixista"
        trend_score += 1 if macd_momentum == "Altista" else -1
        macd_text = f"MACD: DIF({macd_val:.2f}) {'acima' if macd_momentum=='Altista' else 'abaixo'} DEA({dea_val:.2f}) | Histograma({macd_hist:.2f}) -> Momentum {macd_momentum}."

    # Bollinger
    bb_u, bb_l, bb_p = last.get('BB_Upper_20'), last.get('BB_Lower_20'), last.get('BB_PercentB')
    if all(pd.notna(v) for v in [bb_u, bb_l, bb_p, current_price]):
        boll_pos = "Superior" if bb_p > 0.8 else "Inferior" if bb_p < 0.2 else "Médio"
        boll_sugg = "potencial sobrecompra" if boll_pos == "Superior" else "potencial sobrevenda" if boll_pos == "Inferior" else "sem sinal claro"
        boll_text = f"BOLL: Preço ({current_price:.0f}) na banda {boll_pos} (%B={bb_p:.2f}), sugerindo {boll_sugg}."
        # Não adiciona ao trend_score diretamente, mais um indicador de volatilidade/extremo

    # RSI
    rsi6, rsi12, rsi14 = last.get('RSI_6'), last.get('RSI_12'), last.get('RSI_14')
    if all(pd.notna(v) for v in [rsi6, rsi12, rsi14]):
        rsi_avg = (rsi6 + rsi12 + rsi14) / 3
        rsi_cond = "Sobrevenda (<30)" if rsi_avg < 30 else "Sobrecompra (>70)" if rsi_avg > 70 else "Neutro (30-70)"
        rsi_momentum = "Baixista" if rsi_avg < 50 else "Altista"
        trend_score += 1 if rsi_momentum == "Altista" else -1
        rsi_text = f"RSI: RSI6({rsi6:.2f}), RSI12({rsi12:.2f}), RSI14({rsi14:.2f}). Média({rsi_avg:.2f}) indica {rsi_cond} e momentum {rsi_momentum}."

    # KDJ
    k, d, j = last.get('Stoch_K_14_3'), last.get('Stoch_D_14_3'), last.get('Stoch_J_14_3')
    if all(pd.notna(v) for v in [k, d, j]):
        kdj_cond = "Sobrevenda (<20)" if k < 20 and d < 20 else "Sobrecompra (>80)" if k > 80 and d > 80 else "Neutro (20-80)"
        kdj_momentum = "Baixista" if k < d else "Altista" # Cruzamento K vs D
        trend_score += 1 if kdj_momentum == "Altista" else -1
        kdj_text = f"KDJ: K({k:.2f}), D({d:.2f}), J({j:.2f}). Condição {kdj_cond}, momentum {kdj_momentum}."

    report_lines.append(f"- {ma_text}")
    report_lines.append(f"- {macd_text}")
    report_lines.append(f"- {boll_text}")
    report_lines.append(f"- {rsi_text}")
    report_lines.append(f"- {kdj_text}")

    # Funding Rate (Placeholder)
    report_lines.append("- Funding Rate: N/A (Fonte de dados não integrada).")

    # Volume Analysis (Simples)
    vol_ma20 = last.get('Volume_MA_20')
    volume = last.get('volume')
    if pd.notna(vol_ma20) and pd.notna(volume):
         vol_status = "Acima" if volume > vol_ma20 else "Abaixo"
         report_lines.append(f"- Volume: Volume atual ({volume:,.0f}) está {vol_status.lower()} da média de 20 dias ({vol_ma20:,.0f}).")
    else: report_lines.append("- Volume: N/A.")

    # Fund Flow (Placeholder)
    report_lines.append("- Fluxo de Fundos: N/A (Fonte de dados não integrada/simulada).")
    report_lines.append("-" * 30)

    # --- Resultado da Análise ---
    report_lines.append("**Resultado da Análise:**")
    # Define tendência geral baseado no score
    overall_trend = "Altista" if trend_score > 1 else "Baixista" if trend_score < -1 else "Neutra/Indefinida"
    report_lines.append(f"  **Direção Geral:** {overall_trend}")

    # Sugestões (Genéricas e com Aviso)
    if pivots:
        r1 = pivots.get('R1', last_price * 1.05) # Usa preço + 5% se pivot ausente
        s1 = pivots.get('S1', last_price * 0.95)
        stop_loss_pct = 0.03 # 3%
        target_pct = 0.05 # 5%

        if overall_trend == "Baixista":
             entry_zone = pivots.get('PP', last_price) # Sugere entrada perto do Pivot Point
             report_lines.append(f"  Sugestão Entrada (Short): Considerar perto de resistências ({pivots.get('PP'):.0f} ou R1={r1:.0f}).")
             report_lines.append(f"  Stop Loss Sugerido: ~{stop_loss_pct:.1%} acima da entrada (ex: {entry_zone * (1 + stop_loss_pct):.0f}).")
             report_lines.append(f"  Preço Alvo Sugerido: Próximo a S1 ({s1:.0f}).")
        elif overall_trend == "Altista":
             entry_zone = pivots.get('PP', last_price)
             report_lines.append(f"  Sugestão Entrada (Long): Considerar perto de suportes ({pivots.get('PP'):.0f} ou S1={s1:.0f}).")
             report_lines.append(f"  Stop Loss Sugerido: ~{stop_loss_pct:.1%} abaixo da entrada (ex: {entry_zone * (1 - stop_loss_pct):.0f}).")
             report_lines.append(f"  Preço Alvo Sugerido: Próximo a R1 ({r1:.0f}).")
        else:
             report_lines.append("  Sugestão: Aguardar confirmação de tendência.")
    else:
        report_lines.append("  Sugestão: N/A (Pivot Points não calculados).")

    report_lines.append("-" * 30)
    report_lines.append("*Nota: Esta análise é gerada automaticamente para referência e não constitui conselho de investimento.*")

    return "\n".join(report_lines)


# --- Funções PDF e Excel ---
def clean_text(text): # Mantida
    if text is None: return ""
    try: text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', str(text)); return str(text).encode('latin-1', 'ignore').decode('latin-1')
    except: return re.sub(r'[^\x20-\x7E]+', '', str(text))

def generate_pdf_report(data, signals, final_verdict, settings, coinank_report_str): # Adiciona coinank_report_str
    pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", 'B', 16); pdf.cell(0, 10, txt="Relatório BTC AI Dashboard Pro+ v2.1", ln=1, align='C'); pdf.ln(5)
    pdf.set_font("Arial", size=10); pdf.cell(0, 5, txt=f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", ln=1, align='C'); pdf.ln(10)

    # --- Seção Relatório CoinAnk Style ---
    pdf.set_font("Arial", 'B', 12); pdf.cell(0, 7, txt="Análise Técnica Detalhada (Estilo CoinAnk)", ln=1)
    pdf.set_font("Arial", size=9)
    # Adiciona o relatório CoinAnk formatado
    for line in coinank_report_str.split('\n'):
        # Remove negrito markdown para PDF
        cleaned_line = clean_text(line.replace("**", ""))
        if cleaned_line.startswith("-"): pdf.cell(0, 5, txt=cleaned_line, ln=1) # Mantém ifen
        elif ":" in cleaned_line and not cleaned_line.startswith(" "): # Linhas de título
            pdf.set_font("Arial", 'B', 9); pdf.cell(0, 5, txt=cleaned_line, ln=1); pdf.set_font("Arial", size=9)
        else: pdf.multi_cell(0, 5, txt=cleaned_line) # Usa multi_cell para quebrar linha

    pdf.ln(5); pdf.cell(0, 5, txt="-"*70, ln=1, align='C'); pdf.ln(5) # Separador

    # --- Seção Sinais Consolidados e Individuais ---
    pdf.set_font("Arial", 'B', 12); pdf.cell(0, 7, txt=f"Sinal Consolidado (Ponderado): {clean_text(final_verdict)}", ln=1); pdf.ln(5)
    pdf.set_font("Arial", 'B', 11); pdf.cell(0, 7, txt="Sinais Individuais (Dashboard):", ln=1); pdf.set_font("Arial", size=9)
    if signals:
        for signal in signals: c_name=clean_text(signal.get('name','N/A')); c_val=clean_text(signal.get('signal','N/A')); c_det=clean_text(str(signal.get('value',''))); w=signal.get('weight',0); s=signal.get('score',0); pdf.cell(0, 5, txt=f"- {c_name}: {c_val} ({c_det}) | W:{w:.1f}, S:{s:.2f}", ln=1)
    else: pdf.cell(0, 5, txt="- Nenhum sinal individual gerado.", ln=1)

    # Disclaimer final
    pdf.ln(5); pdf.set_font("Arial", 'I', 8); pdf.ln(10); pdf.multi_cell(0, 4, txt=clean_text("Disclaimer: Relatório gerado automaticamente apenas para fins informativos. Não constitui aconselhamento financeiro."))

    try: with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp: pdf_output_path = tmp.name; pdf.output(pdf_output_path); return pdf_output_path
    except Exception as e: st.error(f"Erro salvar PDF: {e}"); return None


# ======================
# |||| LOOP PRINCIPAL ||||
# ======================
def main():
    # --- Inicialização ---
    if 'user_settings' not in st.session_state: st.session_state.user_settings = DEFAULT_SETTINGS.copy()
    settings = st.session_state.user_settings
    sentiment_model = load_sentiment_model()
    lstm_model, lstm_scaler = load_lstm_model_and_scaler()
    rl_model, rl_scaler, rl_env_config = None, None, None
    if SB3_AVAILABLE and SKLEARN_AVAILABLE and os.path.exists(RL_MODEL_PATH):
        try: rl_model = PPO.load(RL_MODEL_PATH, device='auto')
        except Exception as e: st.error(f"Erro load RL model: {e}"); rl_model=None
        if rl_model:
            if os.path.exists(RL_SCALER_PATH): rl_scaler = joblib.load(RL_SCALER_PATH)
            else: rl_model = None; st.warning(f"Scaler RL não encontrado.")
            if os.path.exists(RL_ENV_CONFIG_PATH): rl_env_config = joblib.load(RL_ENV_CONFIG_PATH)
            else: rl_model = None; st.warning(f"Config RL não encontrada.")

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Painel de Controle AI v2.4") # Versão incrementada
        with st.expander("🧠 Config IA", expanded=False):
            settings['lstm_window'] = st.slider("Janela LSTM", 30, 120, settings['lstm_window'], 10, key='sl_lstm_w')
            settings['lstm_epochs'] = st.slider("Épocas LSTM", 10, 100, settings['lstm_epochs'], 10, key='sl_lstm_e')
            settings['rl_total_timesteps'] = st.slider("Timesteps RL", 10000, 100000, settings['rl_total_timesteps'], 5000, key='sl_rl_ts')
            settings['rl_transaction_cost'] = st.slider("Custo Tx RL (%)", 0.0, 0.5, settings['rl_transaction_cost']*100, 0.05, key='sl_rl_cost') / 100.0
            st.caption(f"Modelos: '{os.path.basename(LSTM_MODEL_PATH)}', '{os.path.basename(RL_MODEL_PATH)}'")
        with st.expander("🔧 Técnicos", expanded=True):
            settings['rsi_window'] = st.slider("RSI", 7, 30, settings['rsi_window'], 1, key='sl_rsi')
            settings['bb_window'] = st.slider("BB", 10, 50, settings['bb_window'], 1, key='sl_bb')
            settings['ma_windows'] = st.multiselect("MAs", [7, 14, 20, 30, 50, 100, 200], settings['ma_windows'], key='sl_ma')
            settings['gp_window'] = st.slider("GP Win", 10, 60, settings['gp_window'], 5, key='sl_gp')
        with st.expander("📊 OB & S/R", expanded=False):
            settings['ob_swing_length'] = st.slider("OB Swing", 5, 21, settings['ob_swing_length'], 2, key='sl_ob_sw')
            settings['ob_use_body'] = st.checkbox("OB Corpo", settings['ob_use_body'], key='sl_ob_bd')
            settings['n_clusters'] = st.slider("Clusters S/R", 3, 10, settings['n_clusters'], 1, key='sl_cls')
        with st.expander("📰 Notícias", expanded=False):
            settings['min_confidence'] = st.slider("Confiança Mín.", 0.5, 1.0, settings['min_confidence'], 0.05, key='sl_news')
        st.divider()
        if st.button("🔄 Atualizar Dados", type="primary", use_container_width=True, key='bt_upd'): st.cache_data.clear(); st.success("Cache limpo..."); st.rerun()
        with st.expander("ℹ️ Legenda", expanded=False): st.markdown("""**📌 Legenda Sinais:** ... (Resto da legenda) ...""")

    # --- Carregamento Principal ---
    master_data = load_and_process_data()
    if 'prices' not in master_data or master_data['prices'].empty: st.error("Erro dados. Dashboard parado."); st.stop()

    # --- Previsão/Ação IA ---
    current_lstm_prediction = None
    if lstm_model and lstm_scaler: current_lstm_prediction = predict_with_lstm(lstm_model, lstm_scaler, master_data['prices'], settings['lstm_window'])
    current_rl_action = None
    if rl_model and rl_scaler and rl_env_config:
         try:
             rl_feature_cols_norm = rl_env_config['feature_cols']; rl_feature_cols_base = [c.replace('_norm','') for c in rl_feature_cols_norm]
             df_rl_current = master_data['prices'].copy()
             if not all(c in df_rl_current.columns for c in rl_feature_cols_base): raise ValueError(f"Colunas RL base ausentes: {rl_feature_cols_base}")
             df_rl_current.dropna(subset=rl_feature_cols_base, inplace=True); df_rl_current = df_rl_current.reset_index(drop=True)
             if not df_rl_current.empty:
                 env_sim = BitcoinTradingEnv(df_rl_current, rl_feature_cols_norm, rl_scaler, settings['rl_transaction_cost'])
                 obs, _ = env_sim.reset()
                 current_rl_action, _ = rl_model.predict(obs, deterministic=True)
             else: st.warning("Dados insuficientes simular RL.")
         except Exception as e: st.warning(f"Erro simulação ação RL: {e}")

    # --- Geração Sinais ---
    signals, final_verdict, buy_count, sell_count = generate_signals_v2(master_data, settings, current_lstm_prediction, current_rl_action)

    # --- Geração Relatório CoinAnk Style ---
    coinank_report = generate_coinank_style_report(master_data)

    # --- Busca Dados Adicionais ---
    sentiment = get_market_sentiment(); traditional_assets = get_traditional_assets()
    if sentiment_model and 'news' in master_data: analyzed_news = analyze_news_sentiment(master_data['news'], sentiment_model)
    else: analyzed_news = master_data.get('news', [])
    filtered_news = filter_news_by_confidence(analyzed_news, settings['min_confidence'])

    # --- Layout Principal ---
    st.header("📊 Painel Integrado BTC AI Pro+ v2.4")
    # Métricas
    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
    last_price_val = master_data['prices']['price'].iloc[-1] if not master_data['prices'].empty else None
    prev_price_val = master_data['prices']['price'].iloc[-2] if len(master_data['prices']) > 1 else last_price_val
    price_change = ((last_price_val / prev_price_val) - 1) * 100 if prev_price_val and last_price_val and prev_price_val != 0 else 0.0
    mcol1.metric("Preço BTC", f"${last_price_val:,.2f}" if last_price_val else "N/A", f"{price_change:.2f}%" if price_change else " ", delta_color="normal")
    mcol2.metric("Sentimento (F&G)", f"{sentiment['value']}/100", sentiment['sentiment'])
    def get_asset_metric(asset_name):
        asset_data = traditional_assets[traditional_assets['asset'] == asset_name] if not traditional_assets.empty else pd.DataFrame()
        if not asset_data.empty and len(asset_data) > 1: val = asset_data['value'].iloc[-1]; prev = asset_data['value'].iloc[-2]; change = ((val / prev) - 1) * 100 if prev and prev!=0 else 0.0; return f"${val:,.0f}", f"{change:+.2f}%"
        elif not asset_data.empty: val = asset_data['value'].iloc[-1]; return f"${val:,.0f}", " "
        else: return "N/A", " "
    sp_val, sp_delta = get_asset_metric("S&P 500"); mcol3.metric("S&P 500", sp_val, sp_delta)
    gold_val, gold_delta = get_asset_metric("Ouro"); mcol4.metric("Ouro", gold_val, gold_delta)
    mcol5.metric("Análise Final AI", final_verdict)

    # --- Tabs ---
    tab_titles = ["📈 Mercado", "📝 Relatório AI", "🆚 Comparativos", "🧪 Backtesting", "🌍 Cenários", "🤖 IA Training", "📉 Técnico", "📤 Exportar"]
    tabs = st.tabs(tab_titles)

    # Tab 1: Mercado (Gráficos e Sinais Visuais)
    with tabs[0]:
        col1, col2 = st.columns([3, 1])
        with col1: # Gráficos
            st.subheader("Preço BTC, MAs e Níveis Chave")
            if not master_data['prices'].empty:
                fig_price = go.Figure()
                if st.checkbox("Candlestick", value=False, key='cb_ohlc'): fig_price.add_trace(go.Candlestick(x=master_data['prices']['date'], open=master_data['prices']['open'], high=master_data['prices']['high'], low=master_data['prices']['low'], close=master_data['prices']['close'], name='BTC OHLC'))
                else: fig_price.add_trace(go.Scatter(x=master_data['prices']['date'], y=master_data['prices']['price'], mode='lines', name='Preço BTC', line=dict(color='orange', width=2)))
                for window in settings['ma_windows']:
                     if f'MA{window}' in master_data['prices'].columns: fig_price.add_trace(go.Scatter(x=master_data['prices']['date'], y=master_data['prices'][f'MA{window}'], mode='lines', name=f'MA {window}', opacity=0.7))
                sr_levels = data.get('support_resistance', []) # Usa K-Means S/R para visualização
                if sr_levels:
                     for level in sr_levels: fig_price.add_hline(y=level, line_dash="dot", line_color="gray", opacity=0.6, annotation_text=f" S/R Cluster: {level:,.0f}", annotation_position="bottom right")
                _, current_blocks = identify_order_blocks(master_data['prices'], **settings); fig_price = plot_order_blocks(fig_price, current_blocks, last_price_val)
                if 'RSI_Divergence' in master_data['prices'].columns:
                     div_df = master_data['prices'][master_data['prices']['RSI_Divergence'] != 0].copy(); div_df['plot_y'] = np.where(div_df['RSI_Divergence'] == 1, div_df['low'] * 0.98, div_df['high'] * 1.02)
                     fig_price.add_trace(go.Scatter(x=div_df[div_df['RSI_Divergence'] == 1]['date'], y=div_df[div_df['RSI_Divergence'] == 1]['plot_y'], mode='markers', name='Div. Alta', marker=dict(symbol='triangle-up', color='green', size=10)))
                     fig_price.add_trace(go.Scatter(x=div_df[div_df['RSI_Divergence'] == -1]['date'], y=div_df[div_df['RSI_Divergence'] == -1]['plot_y'], mode='markers', name='Div. Baixa', marker=dict(symbol='triangle-down', color='red', size=10)))
                fig_price.update_layout(title="Preço BTC com Indicadores e Níveis", xaxis_rangeslider_visible=False, height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_price, use_container_width=True)
            st.divider(); gcol1, gcol2 = st.columns(2)
            with gcol1: st.subheader("⛏️ On-Chain"); hr_diff_fig = plot_hashrate_difficulty(master_data); st.plotly_chart(hr_diff_fig, use_container_width=True) if hr_diff_fig else st.caption("N/A")
            with gcol2: st.subheader("🐳 Atividade Whale (Simulado)"); whale_fig = plot_whale_activity_simulated(master_data); st.plotly_chart(whale_fig, use_container_width=True) if whale_fig else st.caption("N/A")
        with col2: # Sinais Visuais
            st.subheader("🚦 Sinais Atuais (Dashboard)");
            if signals:
                 for sig in signals: color = {"COMPRA": "🟢", "VENDA": "🔴", "NEUTRO": "🟡"}.get(sig['signal'], "⚪"); name_s = sig['name'][:20] + '..' if len(sig['name']) > 20 else sig['name']; val_s = str(sig['value'])[:18] + '..' if len(str(sig['value'])) > 18 else str(sig['value']); st.markdown(f"<small>{color} **{name_s}:** {sig['signal']} ({val_s})</small>", unsafe_allow_html=True)
            st.divider(); st.subheader("Consolidado (Dashboard)")
            if final_verdict == "✅ FORTE COMPRA": st.success(f"### {final_verdict}")
            elif final_verdict == "❌ FORTE VENDA": st.error(f"### {final_verdict}")
            elif "COMPRA" in final_verdict: st.info(f"### {final_verdict}")
            elif "VENDA" in final_verdict: st.warning(f"### {final_verdict}")
            else: st.write(f"### {final_verdict}")
            st.caption(f"{len(signals)} Sinais | Compra: {buy_count}, Venda: {sell_count}")
            st.divider(); st.subheader("📰 Notícias");
            if filtered_news:
                 for news in filtered_news[:5]: s_label = news.get('sentiment', 'N'); s_score = news.get('sentiment_score', 0.5)*100; s_col = "green" if s_label=="POSITIVE" else "red" if s_label=="NEGATIVE" else "gray"; st.markdown(f"<small><font color='{s_col}'>[{s_label[0]} {s_score:.0f}%]</font> {news['title'][:60]}...</small>", unsafe_allow_html=True)
            else: st.caption("Nenhuma notícia.")
            st.divider(); st.subheader("📊 Fluxo Exchanges (Simulado)")
            ex_df = master_data.get('exchanges_simulated');
            if ex_df is not None and not ex_df.empty: st.dataframe(ex_df.style.background_gradient(cmap='RdYlGn', subset=['Líquido'], vmin=-500, vmax=500).format("{:,.0f}"), use_container_width=True, height=180)
        st.divider(); st.subheader("😨 Sentimento do Mercado (F&G)")
        if sentiment: fig_sent = go.Figure(go.Indicator(mode="gauge+number+delta",value=sentiment['value'],title={'text': f"F&G: {sentiment['sentiment']}"}, gauge={'axis': {'range': [0, 100]},'bar': {'color': "darkblue"},'steps': [{'range': [0, 25], 'color': "#d62728"},{'range': [25, 45], 'color': "#ff7f0e"},{'range': [45, 55], 'color': "#f0de69"},{'range': [55, 75], 'color': "#aec7e8"},{'range': [75, 100], 'color': "#2ca02c"}],'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': sentiment['value']}})); fig_sent.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20)); st.plotly_chart(fig_sent, use_container_width=True)

    # Tab 2: Relatório AI (Novo)
    with tabs[1]:
        st.subheader("📝 Relatório de Análise Técnica (Estilo CoinAnk)")
        st.markdown(f"```text\n{coinank_report}\n```") # Exibe como bloco de texto pré-formatado

    # Tab 3: Comparativos
    with tabs[2]:
        st.subheader("📌 BTC vs Ativos Tradicionais (Últimos 90 dias)")
        if not traditional_assets.empty: pivot_df = traditional_assets.pivot(index='date', columns='asset', values='value'); normalized_pivot = (pivot_df / pivot_df.iloc[0] * 100).ffill(); normalized_plot = normalized_pivot.reset_index().melt(id_vars='date', var_name='asset', value_name='normalized_value'); fig_comp = px.line(normalized_plot, x="date", y="normalized_value", color="asset", title="Desempenho Normalizado (Início = 100)", labels={'normalized_value': 'Performance Normalizada', 'date': 'Data', 'asset': 'Ativo'}); fig_comp.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)); st.plotly_chart(fig_comp, use_container_width=True)
        else: st.warning("Dados comparativos não disponíveis.")
        st.subheader("🔄 Correlação entre Ativos (Últimos 90 dias)")
        if not traditional_assets.empty:
             pivot_df_corr = traditional_assets.pivot(index='date', columns='asset', values='value'); returns_df = pivot_df_corr.pct_change().dropna()
             if not returns_df.empty and len(returns_df)>1: corr_matrix = returns_df.corr(); fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Matriz de Correlação (Retornos Diários)", color_continuous_scale='RdBu_r', zmin=-1, zmax=1); st.plotly_chart(fig_corr, use_container_width=True)
             else: st.warning("Não foi possível calcular correlação.")

    # Tab 4: Backtesting
    with tabs[3]:
        st.subheader("🧪 Backtesting de Estratégias")
        if 'prices' not in master_data or master_data['prices'].empty: st.error("Dados ausentes."); st.stop()
        strategy_list = ["RSI", "MACD", "Bollinger", "EMA Cross", "Volume", "OBV", "Stochastic", "Gaussian Process", "Order Blocks"]
        strategy = st.selectbox("Escolha a Estratégia:", strategy_list, key="backtest_strategy")
        params_container = st.container(); params = {}
        with params_container: # Parâmetros específicos
            st.markdown(f"**Parâmetros para {strategy}:**"); bcol1, bcol2 = st.columns(2)
            # Sliders para cada estratégia (mantidos)
            with bcol1:
                if strategy == "RSI": params['rsi_window'] = st.slider("RSI Win", 7, 30, settings['rsi_window'], key="bt_rsi_win"); params['oversold'] = st.slider("Oversold", 10, 40, 30, key="bt_rsi_os")
                elif strategy == "MACD": params['fast'] = st.slider("Fast EMA", 5, 20, 12, key="bt_macd_f"); params['signal_macd_window'] = st.slider("Signal Line", 5, 20, 9, key="bt_macd_s")
                elif strategy == "Bollinger": params['window'] = st.slider("BB Win", 10, 50, settings['bb_window'], key="bt_bb_win")
                elif strategy == "EMA Cross": params['short_window'] = st.slider("Short EMA", 5, 50, 9, key="bt_ema_s")
                elif strategy == "Volume": params['volume_window'] = st.slider("Vol Win", 10, 50, 20, key="bt_vol_win")
                elif strategy == "OBV": params['obv_window'] = st.slider("OBV MA Win", 10, 50, 20, key="bt_obv_win")
                elif strategy == "Stochastic": params['k_window'] = st.slider("%K Win", 5, 30, 14, key="bt_stoch_k"); params['oversold'] = st.slider("Stoch Oversold", 10, 40, 20, key="bt_stoch_os")
                elif strategy == "Gaussian Process": params['window'] = st.slider("GP Win", 10, 60, settings['gp_window'], key="bt_gp_win")
                elif strategy == "Order Blocks": params['swing_length'] = st.slider("OB Swing", 5, 21, settings['ob_swing_length'], step=2, key="bt_ob_swing")
            with bcol2:
                if strategy == "RSI": params['overbought'] = st.slider("Overbought", 60, 90, 70, key="bt_rsi_ob")
                elif strategy == "MACD": params['slow'] = st.slider("Slow EMA", 20, 60, 26, key="bt_macd_sl")
                elif strategy == "Bollinger": params['num_std'] = st.slider("Std Devs", 1.0, 3.0, 2.0, 0.1, key="bt_bb_std")
                elif strategy == "EMA Cross": params['long_window'] = st.slider("Long EMA", 10, 100, 21, key="bt_ema_l")
                elif strategy == "Volume": params['threshold'] = st.slider("Vol Threshold", 1.0, 3.0, 1.5, 0.1, key="bt_vol_thr")
                elif strategy == "OBV": params['price_window'] = st.slider("Price MA Win", 10, 50, 30, key="bt_obv_price")
                elif strategy == "Stochastic": params['d_window'] = st.slider("%D Smooth", 3, 9, 3, key="bt_stoch_d"); params['overbought'] = st.slider("Stoch Overbought", 60, 90, 80, key="bt_stoch_ob")
                elif strategy == "Gaussian Process": params['threshold'] = st.slider("GP Threshold (%)", 1.0, 5.0, 2.0, 0.5, key="bt_gp_thr") / 100.0
                elif strategy == "Order Blocks": params['use_body'] = st.checkbox("OB Use Body", settings['ob_use_body'], key="bt_ob_body")

        if st.button(f"▶️ Executar Backtest {strategy}", type="primary", key=f'bt_run_{strategy}'):
            st.session_state.backtest_results = None; df_backtest = pd.DataFrame()
            with st.spinner(f"Executando backtest {strategy}..."):
                try: # Chama função de backtest correta
                    if strategy == 'RSI': df_backtest = backtest_rsi_strategy(master_data['prices'], **params)
                    elif strategy == 'MACD': df_backtest = backtest_macd_strategy(master_data['prices'], **params)
                    elif strategy == 'Bollinger': df_backtest = backtest_bollinger_strategy(master_data['prices'], **params)
                    elif strategy == 'EMA Cross': df_backtest = backtest_ema_cross_strategy(master_data['prices'], **params)
                    elif strategy == 'Volume': df_backtest = backtest_volume_strategy(master_data['prices'], **params)
                    elif strategy == 'OBV': df_backtest = backtest_obv_strategy(master_data['prices'], **params)
                    elif strategy == 'Stochastic': df_backtest = backtest_stochastic_strategy(master_data['prices'], **params)
                    elif strategy == 'Gaussian Process': df_backtest = backtest_gp_strategy(master_data['prices'], **params)
                    elif strategy == 'Order Blocks': df_backtest = backtest_order_block_strategy(master_data['prices'], **params)
                    if df_backtest is not None and not df_backtest.empty: st.session_state.backtest_results = df_backtest; st.success("Backtest concluído!")
                    else: st.error("Falha no backtest.")
                except Exception as e: st.error(f"Erro no backtest: {e}")
        if 'backtest_results' in st.session_state and st.session_state.backtest_results is not None:
            df_results = st.session_state.backtest_results; metrics = calculate_metrics(df_results)
            if metrics:
                 st.subheader("📊 Resultados Backtesting"); fig_bt = go.Figure(); fig_bt.add_trace(go.Scatter(x=df_results['date'], y=df_results['strategy_cumulative'], name="Estratégia", line=dict(color='green', width=2))); fig_bt.add_trace(go.Scatter(x=df_results['date'], y=df_results['cumulative_return'], name="Buy & Hold", line=dict(color='blue', width=2))); fig_bt.update_layout(title=f"Desempenho: {strategy}", yaxis_title="Retorno Acumulado", hovermode="x unified"); st.plotly_chart(fig_bt, use_container_width=True)
                 st.subheader("📈 Métricas"); m_col1, m_col2, m_col3 = st.columns(3); m_col1.metric("Retorno Estratégia", f"{metrics.get('Retorno Estratégia', 0):.2%}"); m_col2.metric("Retorno B&H", f"{metrics.get('Retorno Buy & Hold', 0):.2%}"); m_col3.metric("Sharpe Estratégia", f"{metrics.get('Sharpe Estratégia', 0):.2f}"); m_col4, m_col5, m_col6 = st.columns(3); m_col4.metric("Volatilidade", f"{metrics.get('Vol Estratégia', 0):.2%}"); m_col5.metric("Max Drawdown", f"{metrics.get('Max Drawdown Estratégia', 0):.2%}"); m_col6.metric("Taxa Acerto (Aprox.)", f"{metrics.get('Taxa Acerto (Trades Aprox.)', 0):.2%}")
            else: st.warning("Não foi possível calcular métricas.")
            st.divider(); st.subheader("⚙️ Otimização Automática (Experimental)")
            if st.checkbox("Executar Otimização (LENTO!)", key='cb_opt'):
                param_space = {} # Definir espaços aqui
                if strategy == "RSI": param_space = {'rsi_window': range(10, 21, 2), 'overbought': range(65, 81, 5), 'oversold': range(20, 36, 5)}
                elif strategy == "MACD": param_space = {'fast': range(9, 16, 3), 'slow': range(20, 31, 3), 'signal': range(7, 13, 2)}
                elif strategy == "Bollinger": param_space = {'window': range(15, 31, 5), 'num_std': [1.8, 2.0, 2.2]}
                elif strategy == "EMA Cross": param_space = {'short_window': range(7, 14, 3), 'long_window': range(18, 31, 4)}
                elif strategy == "Stochastic": param_space = {'k_window': range(10, 21, 3), 'd_window': [3, 5], 'overbought': [75, 80], 'oversold': [20, 25]}
                elif strategy == "Order Blocks": param_space = {'swing_length': range(7, 17, 4), 'use_body': [True, False]}
                else: st.warning(f"Otimização não definida para {strategy}.")
                if param_space:
                    if strategy == 'MACD' and 'signal' in param_space: param_space['signal_macd_window'] = param_space.pop('signal')
                    best_params, best_sharpe, best_df = optimize_strategy_parameters(master_data, strategy, param_space)
                    if best_params:
                        st.success(f"🎯 Melhores parâmetros (Sharpe: {best_sharpe:.2f}):"); st.json(best_params)
                        if st.button("Aplicar Parâmetros Otimizados", key='bt_apply_opt'):
                             st.session_state.backtest_params = best_params; st.session_state.backtest_results = best_df
                             st.info("Parâmetros aplicados."); st.rerun()

    # Tab 5: Cenários
    with tabs[4]:
        st.subheader("🌍 Simulação de Cenários de Mercado")
        event = st.selectbox("Selecione Evento:", ["Halving", "Crash", "ETF Approval"], key="scenario_event")
        if 'prices' not in master_data or master_data['prices'].empty: st.warning("Dados ausentes.")
        else:
            df_scenario = master_data['prices'].tail(90).copy()
            if not df_scenario.empty:
                 simulated_prices = simulate_event(event, df_scenario['price'])
                 if simulated_prices is not None and not simulated_prices.empty:
                     fig_scenario = go.Figure(); fig_scenario.add_trace(go.Scatter(x=df_scenario['date'], y=df_scenario['price'], name="Real Recente", line=dict(color='blue'))); fig_scenario.add_trace(go.Scatter(x=df_scenario['date'], y=simulated_prices, name=f"Projeção Pós-{event}", line=dict(color='red', dash='dash'))); fig_scenario.update_layout(title=f"Simulação: {event}", yaxis_title="Preço BTC", hovermode="x unified"); st.plotly_chart(fig_scenario, use_container_width=True)
                 else: st.error("Falha na simulação.")

    # Tab 6: IA Training
    with tabs[5]:
        st.header("🤖 Treinamento e Gerenciamento de Modelos IA")
        st.info("Use esta aba para treinar os modelos LSTM e RL. Os modelos treinados serão salvos localmente e carregados automaticamente nas próximas execuções.")
        with st.container(border=True):
            st.subheader("🧠 Treinamento LSTM")
            if TF_AVAILABLE:
                lstm_loaded = (lstm_model is not None and lstm_scaler is not None); status_lstm = "✅ Carregado" if lstm_loaded else "❌ Não Carregado"
                st.write(f"Status Modelo LSTM: {status_lstm}")
                if st.button("Iniciar Treinamento LSTM", key="train_lstm_button", disabled=not TF_AVAILABLE):
                    if 'prices' in master_data and not master_data['prices'].empty: success = train_and_save_lstm(master_data['prices'], settings['lstm_window'], settings['lstm_epochs'], DEFAULT_SETTINGS['lstm_units']);
                    if success: st.rerun()
                    else: st.error("Dados de preço necessários para treino LSTM.")
            else: st.warning("TensorFlow/Keras indisponível.")
        with st.container(border=True):
            st.subheader("🤖 Treinamento Agente RL (PPO)")
            if SB3_AVAILABLE and GYM_AVAILABLE and SKLEARN_AVAILABLE:
                rl_loaded = (rl_model is not None and rl_scaler is not None and rl_env_config is not None); status_rl = "✅ Carregado" if rl_loaded else "❌ Não Carregado"
                st.write(f"Status Modelo RL: {status_rl}")
                if st.button("Iniciar Treinamento RL (LENTO!)", key="train_rl_button"):
                    if 'prices' in master_data and not master_data['prices'].empty:
                         with st.spinner("Preparando dados e ambiente RL..."):
                             try:
                                 df_rl_train = master_data['prices'].copy(); feature_cols_base = RL_OBSERVATION_COLS_BASE
                                 if not all(c in df_rl_train.columns for c in feature_cols_base): raise ValueError(f"Colunas RL ausentes: {feature_cols_base}")
                                 df_rl_train.dropna(subset=feature_cols_base, inplace=True); df_rl_train = df_rl_train.reset_index(drop=True)
                                 if len(df_rl_train) < 100: raise ValueError("Dados insuficientes pós limpeza.")
                                 scaler_rl = StandardScaler(); scaler_rl.fit(df_rl_train[feature_cols_base])
                                 joblib.dump(scaler_rl, RL_SCALER_PATH)
                                 env_config = {'feature_cols': RL_OBSERVATION_COLS_NORM}; joblib.dump(env_config, RL_ENV_CONFIG_PATH)
                                 env_train = BitcoinTradingEnv(df_rl_train, RL_OBSERVATION_COLS_NORM, scaler_rl, settings['rl_transaction_cost'])
                                 vec_env_train = DummyVecEnv([lambda: env_train])
                             except Exception as e: st.error(f"Erro preparação RL: {e}"); st.stop()
                         try: # Treinamento
                             st.info(f"Treinando RL PPO ({settings['rl_total_timesteps']} timesteps)...")
                             progress_bar_rl = st.progress(0); status_text_rl = st.empty(); callback = ProgressBarCallback(settings['rl_total_timesteps'], progress_bar_rl, status_text_rl)
                             model_rl = PPO('MlpPolicy', vec_env_train, verbose=0, device='auto', learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2)
                             model_rl.learn(total_timesteps=settings['rl_total_timesteps'], callback=callback)
                             model_rl.save(RL_MODEL_PATH); st.success(f"Modelo RL salvo em '{RL_MODEL_PATH}'.")
                             st.cache_resource.clear(); st.rerun()
                         except Exception as e: st.error(f"Erro treino/save RL: {e}")
                         finally: progress_bar_rl.empty(); status_text_rl.empty()
                    else: st.error("Dados de preço necessários para treino RL.")
            else: st.warning("SB3/Gym/SKlearn indisponível.")
        with st.container(border=True):
            st.subheader("📰 Modelo Análise Sentimento")
            if TRANSFORMERS_AVAILABLE: status_sent = "✅ Carregado" if sentiment_model else "❌ Falha"; st.write(f"Status: {status_sent}")
            else: st.warning("Transformers indisponível.")

    # Tab 7: Técnico
    with tabs[6]:
        st.header("📉 Indicadores Técnicos Detalhados")
        if 'prices' not in master_data or master_data['prices'].empty: st.warning("Dados ausentes.")
        else:
            df_tech = master_data['prices']; num_cols = 2; cols = st.columns(num_cols); plot_idx = 0
            def add_tech_plot(figure, title): nonlocal plot_idx;
            if figure: cols[plot_idx % num_cols].plotly_chart(figure, use_container_width=True); plot_idx += 1
            rsi_col = f'RSI_{settings["rsi_window"]}';
            if rsi_col in df_tech.columns: fig = px.line(df_tech, x="date", y=rsi_col, title=f"RSI ({settings['rsi_window']})", range_y=[0, 100]); fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5); fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5); fig.update_layout(height=300, margin=dict(t=30, b=20)); add_tech_plot(fig, f"RSI ({settings['rsi_window']})")
            if 'MACD' in df_tech.columns and 'MACD_Signal' in df_tech.columns: fig = go.Figure(); fig.add_trace(go.Scatter(x=df_tech['date'], y=df_tech['MACD'], name="MACD", line_color='blue')); fig.add_trace(go.Scatter(x=df_tech['date'], y=df_tech['MACD_Signal'], name="Signal", line_color='red')); macd_hist = df_tech['MACD'] - df_tech['MACD_Signal']; colors = ['green' if v >= 0 else 'red' for v in macd_hist]; fig.add_trace(go.Bar(x=df_tech['date'], y=macd_hist, name='Hist', marker_color=colors)); fig.update_layout(title="MACD (12, 26, 9)", height=300, margin=dict(t=30, b=20)); add_tech_plot(fig, "MACD")
            bb_col_u, bb_col_l = f'BB_Upper_{settings["bb_window"]}', f'BB_Lower_{settings["bb_window"]}';
            if bb_col_u in df_tech.columns and bb_col_l in df_tech.columns: fig = go.Figure(); fig.add_trace(go.Scatter(x=df_tech['date'], y=df_tech['price'], name="Preço", line=dict(color='orange'))); fig.add_trace(go.Scatter(x=df_tech['date'], y=df_tech[bb_col_u], name="Sup", line=dict(color='lightblue', width=1))); fig.add_trace(go.Scatter(x=df_tech['date'], y=df_tech[bb_col_l], name="Inf", line=dict(color='lightblue', width=1), fill='tonexty', fillcolor='rgba(173,216,230,0.1)')); fig.update_layout(title=f"Bollinger ({settings['bb_window']})", height=300, margin=dict(t=30, b=20)); add_tech_plot(fig, f"Bollinger ({settings['bb_window']})")
            stoch_k_col, stoch_d_col = 'Stoch_K_14_3', 'Stoch_D_14_3'; # Usa KDJ para plotar K e D
            if stoch_k_col in df_tech.columns and stoch_d_col in df_tech.columns: fig = go.Figure(); fig.add_trace(go.Scatter(x=df_tech['date'], y=df_tech[stoch_k_col], name="%K", line_color='blue')); fig.add_trace(go.Scatter(x=df_tech['date'], y=df_tech[stoch_d_col], name="%D", line_color='red')); fig.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.5); fig.add_hline(y=20, line_dash="dash", line_color="green", opacity=0.5); fig.update_layout(title="Stochastic (KDJ)", height=300, margin=dict(t=30, b=20), range_y=[0,100]); add_tech_plot(fig, "Stochastic (KDJ)")
            if 'volume' in df_tech.columns: fig = px.bar(df_tech, x="date", y="volume", title="Volume"); vol_ma_col = 'Volume_MA_20';
            if vol_ma_col in df_tech.columns: fig.add_trace(go.Scatter(x=df_tech['date'], y=df_tech[vol_ma_col], name='Vol MA20', line=dict(color='red'))); fig.update_layout(height=300, margin=dict(t=30, b=20)); add_tech_plot(fig, "Volume")
            if 'OBV' in df_tech.columns: fig = px.line(df_tech, x="date", y="OBV", title="OBV"); obv_ma_col = 'OBV_MA_20';
            if obv_ma_col in df_tech.columns: fig.add_trace(go.Scatter(x=df_tech['date'], y=df_tech[obv_ma_col], name='OBV MA20', line=dict(color='red'))); fig.update_layout(height=300, margin=dict(t=30, b=20)); add_tech_plot(fig, "OBV")
            gp_pred_col = f'GP_Prediction_{settings["gp_window"]}';
            if gp_pred_col in df_tech.columns: fig = go.Figure(); fig.add_trace(go.Scatter(x=df_tech['date'], y=df_tech['price'], name="Preço", line=dict(color='blue', width=1))); fig.add_trace(go.Scatter(x=df_tech['date'], y=df_tech[gp_pred_col], name="Pred GP", line=dict(color='purple', dash='dot'))); fig.update_layout(title=f"Predição GP ({settings['gp_window']}d)", height=300, margin=dict(t=30, b=20)); add_tech_plot(fig, "Gaussian Process")

    # Tab 8: Exportar
    with tabs[7]: # Ajustado índice da tab
        st.header("📤 Exportar Relatório e Dados")
        st.subheader("📄 Relatório PDF"); st.caption("Gera um resumo da análise atual em PDF.")
        if st.button("Gerar Relatório PDF", key='bt_pdf'):
            with st.spinner("Gerando PDF..."):
                # Passa o relatório CoinAnk para a função PDF
                pdf_path = generate_pdf_report(master_data, signals, final_verdict, settings, coinank_report)
                if pdf_path and os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as pdf_file: st.download_button(label="Baixar Relatório PDF", data=pdf_file, file_name=f"btc_ai_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", mime="application/octet-stream", key='dl_pdf')
                    try: os.remove(pdf_path)
                    except Exception as e: st.warning(f"Não foi possível remover PDF temp: {e}")
                else: st.error("Falha ao gerar PDF.")
        st.divider(); st.subheader("💾 Dados em Excel (.xlsx)"); st.caption("Exporta os DataFrames para Excel.")
        if st.button("Exportar Dados para Excel", key='bt_excel'):
            with st.spinner("Preparando Excel..."):
                 try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp: excel_path = tmp.name
                    with pd.ExcelWriter(excel_path) as writer:
                        if 'prices' in master_data and not master_data['prices'].empty: master_data['prices'].to_excel(writer, sheet_name="BTC_Data", index=False)
                        if not traditional_assets.empty: traditional_assets.to_excel(writer, sheet_name="Trad_Assets", index=False)
                        if 'hashrate' in master_data and not master_data['hashrate'].empty: master_data['hashrate'].to_excel(writer, sheet_name="Hashrate", index=False)
                        if 'difficulty' in master_data and not master_data['difficulty'].empty: master_data['difficulty'].to_excel(writer, sheet_name="Difficulty", index=False)
                        if 'support_resistance' in master_data and master_data['support_resistance']: pd.DataFrame({'SR_Level': master_data['support_resistance']}).to_excel(writer, sheet_name="Support_Resistance", index=False)
                        if analyzed_news: pd.DataFrame(analyzed_news).to_excel(writer, sheet_name="News_Sentiment", index=False)
                        if 'backtest_results' in st.session_state and st.session_state.backtest_results is not None: st.session_state.backtest_results.to_excel(writer, sheet_name="Last_Backtest", index=False)
                    with open(excel_path, "rb") as excel_file: st.download_button(label="Baixar Arquivo Excel", data=excel_file, file_name=f"btc_ai_data_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key='dl_excel')
                    try: os.remove(excel_path)
                    except Exception as e: st.warning(f"Não foi possível remover Excel temp: {e}")
                 except Exception as e: st.error(f"Erro ao gerar Excel: {e}")

if __name__ == "__main__":
    main()
