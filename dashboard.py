# -*- coding: utf-8 -*-
# ↑↑↑ Adicionado para garantir codificação correta ↑↑↑

import streamlit as st
# st.cache_resource.clear() # Comente ou remova para produção para usar cache

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

# Ignorar alguns warnings comuns de bibliotecas
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='stable_baselines3')
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')


# --- Importações de ML/IA com Verificação ---
SKLEARN_AVAILABLE = False
try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from sklearn.model_selection import ParameterGrid
    SKLEARN_AVAILABLE = True
except ImportError:
    st.warning("Scikit-learn não encontrado ('pip install scikit-learn'). Funcionalidades limitadas.")

TRANSFORMERS_AVAILABLE = False
try:
    # Adicionar import explícito de TF se for usar backend TF do Transformers
    # import tensorflow
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    st.warning("Transformers não encontrado ('pip install transformers accelerate'). Análise de sentimento desativada.")
# Captura erro de Keras 3 aqui também
except ValueError as e:
    if "Your currently installed version of Keras is Keras 3" in str(e):
         st.error("Erro de Keras 3 detectado. Instale 'tf-keras' ('pip install tf-keras') e reinicie. Análise de sentimento TF desativada.")
         TRANSFORMERS_AVAILABLE = False # Desativa se der erro
    else:
         st.warning(f"Erro ao importar Transformers: {e}. Análise de sentimento desativada.")
         TRANSFORMERS_AVAILABLE = False


TF_AVAILABLE = False
try:
    # Import Keras specifics from TensorFlow
    # Tenta importar de tf_keras primeiro, depois de tensorflow.keras
    try:
        from tf_keras.models import Sequential, load_model
        from tf_keras.layers import LSTM, Dense, Dropout
        from tf_keras.optimizers import Adam
    except ImportError:
        from tensorflow.keras.models import Sequential, load_model
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    st.warning("TensorFlow/Keras ou tf-keras não encontrado ('pip install tensorflow tf-keras'). Funcionalidades LSTM desativadas.")
except Exception as e: # Captura outros erros de TF
    st.warning(f"Erro ao importar TensorFlow/Keras: {e}. Funcionalidades LSTM desativadas.")


SB3_AVAILABLE = False
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    st.warning("Stable-Baselines3 não encontrado ('pip install stable-baselines3[extra]'). Funcionalidades de RL desativadas.")

GYM_AVAILABLE = False
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    st.warning("Gymnasium não encontrado (geralmente instalado com SB3[extra]). Funcionalidades de RL desativadas.")

TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    st.warning("PyTorch não encontrado ('pip install torch'). Pode afetar Transformers ou RL.")


# ======================
# CONSTANTES E CONFIGURAÇÕES GLOBAIS
# ======================
st.set_page_config(layout="wide", page_title="BTC AI Dashboard Pro+")
st.title("🚀 BTC AI Dashboard Pro+ v2.1 - Mais Harmonia")

# Arquivos de persistência
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Diretório do script
MODEL_DIR = os.path.join(BASE_DIR, "saved_models") # Subdiretório para modelos
os.makedirs(MODEL_DIR, exist_ok=True) # Cria o diretório se não existir

LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_btc_model.keras")
LSTM_SCALER_PATH = os.path.join(MODEL_DIR, "lstm_btc_scaler.joblib")
RL_MODEL_PATH = os.path.join(MODEL_DIR, "rl_ppo_btc_model.zip")
RL_SCALER_PATH = os.path.join(MODEL_DIR, "rl_observation_scaler.joblib")
RL_ENV_CONFIG_PATH = os.path.join(MODEL_DIR, "rl_env_config.joblib")

# Pesos e Configurações Padrão (mantidos como antes)
INDICATOR_WEIGHTS = {
    'order_blocks': 2.0, 'gaussian_process': 1.5, 'rsi': 1.5, 'macd': 1.3,
    'bollinger': 1.2, 'volume': 1.0, 'obv': 1.0, 'stochastic': 1.1,
    'ma_cross': 1.0, 'lstm_pred': 1.8, 'rl_action': 2.0,
    'sentiment': 1.2, 'divergence': 1.2
}
DEFAULT_SETTINGS = {
    'rsi_window': 14, 'bb_window': 20, 'ma_windows': [20, 50, 100],
    'gp_window': 30, 'gp_lookahead': 1, 'ob_swing_length': 11, 'ob_show_bull': 3,
    'ob_show_bear': 3, 'ob_use_body': True, 'min_confidence': 0.7, 'n_clusters': 5,
    'lstm_window': 60, 'lstm_epochs': 30, 'lstm_units': 50, 'rl_total_timesteps': 20000,
    'rl_transaction_cost': 0.001, 'email': ''
}
RL_OBSERVATION_COLS_BASE = ['price', 'volume', 'RSI_14', 'MACD', 'MACD_Signal', 'BB_Upper_20', 'BB_Lower_20', 'Stoch_K_14_3'] # Nomes base usados para scaler
RL_OBSERVATION_COLS_NORM = [f'{col}_norm' for col in RL_OBSERVATION_COLS_BASE] # Nomes das colunas normalizadas

# ======================
# FUNÇÕES AUXILIARES E CLASSES
# ======================

# --- Callback para Feedback SB3 (mantida) ---
class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps: int, progress_bar, status_text, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps; self.progress_bar = progress_bar
        self.status_text = status_text; self.current_step = 0
    def _on_step(self) -> bool:
        self.current_step = self.model.num_timesteps; progress = self.current_step / self.total_timesteps
        reward_str = ""
        if self.model.ep_info_buffer:
            try: reward_str = f"| R Média: {np.mean(self.model.ep_info_buffer['r']):.3f}"
            except: pass # Ignora erro se buffer vazio
        self.status_text.text(f"Treinando RL: {self.current_step}/{self.total_timesteps} {reward_str}")
        self.progress_bar.progress(progress); return True

# --- Ambiente RL Refatorado (mantida) ---
if GYM_AVAILABLE and SKLEARN_AVAILABLE:
    class BitcoinTradingEnv(gym.Env):
        metadata = {'render_modes': ['human']}
        def __init__(self, df, feature_cols_norm, scaler, transaction_cost=0.001, initial_balance=10000, render_mode=None):
            super().__init__()
            feature_cols_base = [c.replace('_norm','') for c in feature_cols_norm]
            required_cols = ['price'] + feature_cols_base # Precisa de 'price' além das features
            if df.empty or not all(col in df.columns for col in required_cols): raise ValueError("DataFrame RL vazio ou colunas ausentes.")
            self.df = df.copy().reset_index(drop=True)
            self.scaler = scaler; self.feature_cols_base = feature_cols_base
            self.feature_cols_norm = feature_cols_norm; self.transaction_cost = transaction_cost
            self.initial_balance = initial_balance; self.render_mode = render_mode; self.current_step = 0
            self.action_space = spaces.Discrete(3)
            self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(len(self.feature_cols_norm),), dtype=np.float32) # Limites razoáveis pós StandardScaler

        def _get_observation(self):
            # Seleciona features BASE, escala e retorna
            features = self.df.loc[self.current_step, self.feature_cols_base].values.reshape(1, -1)
            scaled_features = self.scaler.transform(features).astype(np.float32)
            scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=5.0, neginf=-5.0) # Limita
            return scaled_features.flatten()

        def _get_info(self): # Mantida
             current_price = self.df.loc[min(self.current_step, len(self.df)-1), 'price'] # Evita erro no fim
             portfolio_value = self.balance + (self.btc_held * current_price)
             return {'total_profit': portfolio_value - self.initial_balance, 'portfolio_value': portfolio_value, 'balance': self.balance, 'btc_held': self.btc_held, 'current_step': self.current_step}

        def reset(self, seed=None, options=None): # Mantida
            super().reset(seed=seed); self.balance = self.initial_balance; self.btc_held = 0
            self.current_step = 0; self.last_portfolio_value = self.initial_balance
            observation = self._get_observation(); info = self._get_info()
            return observation, info

        def step(self, action): # Mantida (com custo)
            current_price = self.df.loc[self.current_step, 'price']; cost_penalty = 0
            if action == 1 and self.balance > 10: # Buy
                cost_penalty = self.balance * self.transaction_cost; self.balance -= cost_penalty
                if self.balance > 0: self.btc_held += self.balance / current_price; self.balance = 0
            elif action == 2 and self.btc_held > 1e-6: # Sell
                sell_value = self.btc_held * current_price; cost_penalty = sell_value * self.transaction_cost
                self.balance += sell_value - cost_penalty; self.btc_held = 0
            self.current_step += 1; terminated = self.current_step >= len(self.df) - 1; truncated = False
            if not terminated:
                next_price = self.df.loc[self.current_step, 'price']; current_portfolio_value = self.balance + (self.btc_held * next_price)
                reward = (current_portfolio_value - self.last_portfolio_value) - cost_penalty; self.last_portfolio_value = current_portfolio_value
                observation = self._get_observation()
            else: reward = 0; observation = self._get_observation() # Última obs
            info = self._get_info()
            # if self.render_mode == "human": self.render() # Descomente para debug
            return observation, reward, terminated, truncated, info

        def render(self): print(f"Step:{self.current_step}|Bal:${self.balance:,.2f}|BTC:{self.btc_held:.6f}|Port:${self.last_portfolio_value:,.2f}")
        def close(self): pass

# --- Funções de Análise de Sentimento (mantida) ---
@st.cache_resource(show_spinner="Carregando modelo de sentimento...")
def load_sentiment_model():
    if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE: return None
    try: return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)
    except Exception as e: st.error(f"Erro ao carregar modelo Sentimento: {e}"); return None

# analyze_news_sentiment (mantida)
def analyze_news_sentiment(news_list, _model):
    if _model is None: return news_list
    if not news_list: return []
    results = []
    for news in news_list:
        news['sentiment'] = 'NEUTRAL'; news['sentiment_score'] = 0.5
        try:
            text = news.get('title', '')
            if text:
                result = _model(text[:512])[0] # Limita tamanho
                news['sentiment'] = result['label']
                news['sentiment_score'] = result['score'] if result['label'] == 'POSITIVE' else (1 - result['score'])
            results.append(news)
        except Exception: results.append(news) # Adiciona mesmo com erro
    return results

# --- Funções LSTM (mantida com tf_keras) ---
@st.cache_resource
def create_lstm_architecture(input_shape, units=50):
    if not TF_AVAILABLE: return None
    # Usa tf_keras ou tensorflow.keras dependendo do import
    model = Sequential([LSTM(units, return_sequences=True, input_shape=input_shape), Dropout(0.2), LSTM(units, return_sequences=False), Dropout(0.2), Dense(25, activation='relu'), Dense(1)])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error'); return model

@st.cache_resource(show_spinner="Verificando modelo LSTM treinado...")
def load_lstm_model_and_scaler():
    model, scaler = None, None
    if TF_AVAILABLE and os.path.exists(LSTM_MODEL_PATH) and os.path.exists(LSTM_SCALER_PATH):
        try:
            model = load_model(LSTM_MODEL_PATH) # load_model do Keras/TF
            scaler = joblib.load(LSTM_SCALER_PATH)
        except Exception as e: st.error(f"Erro ao carregar LSTM: {e}"); model, scaler = None, None
    return model, scaler

def train_and_save_lstm(data_prices, window, epochs, units):
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
        # Usar tf.device para garantir CPU se necessário, mas Keras deve gerenciar
        history = model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
        final_loss = history.history['loss'][-1]
        status_lstm.update(label=f"Treino LSTM concluído! Loss: {final_loss:.4f}", state="complete", expanded=False)
        model.save(LSTM_MODEL_PATH); joblib.dump(scaler, LSTM_SCALER_PATH)
        st.success(f"Modelo LSTM salvo em '{LSTM_MODEL_PATH}'.")
        st.cache_resource.clear() # Limpa cache para forçar recarga
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
    except Exception: return None # Silencioso na previsão

# --- Funções Indicadores Técnicos (mantidas como antes) ---
# calculate_ema, calculate_rsi, calculate_macd, calculate_bollinger_bands,
# calculate_obv, calculate_stochastic, calculate_gaussian_process,
# identify_order_blocks, plot_order_blocks, detect_support_resistance_clusters,
# detect_divergences
# (Colar as versões robustecidas aqui se não estiverem já no seu código)
# ...

# --- Funções Dados/Plotagem (mantidas) ---
# get_exchange_flows_simulated, plot_hashrate_difficulty, plot_whale_activity_simulated,
# simulate_event, get_market_sentiment, get_traditional_assets, filter_news_by_confidence
# ...

# --- Funções Backtesting (mantidas) ---
# calculate_daily_returns, calculate_strategy_returns, backtest_*, calculate_metrics, optimize_strategy_parameters
# ...

# --- Carregamento de Dados Refatorado (com correção do erro 'tuple') ---
@st.cache_data(ttl=3600, show_spinner="Carregando e processando dados de mercado...")
def load_and_process_data():
    data = {'prices': pd.DataFrame()}
    try:
        ticker = "BTC-USD"
        btc_data_raw = yf.download(ticker, period="1y", interval="1d", progress=False)

        # *** CORREÇÃO ERRO TUPLE: Verifica se é DataFrame e trata colunas ***
        if not isinstance(btc_data_raw, pd.DataFrame) or btc_data_raw.empty:
            raise ValueError(f"yfinance não retornou um DataFrame válido para {ticker}.")

        btc_data = btc_data_raw.copy() # Trabalha com cópia

        # Trata nomes de coluna (string ou tuple/MultiIndex)
        new_cols = []
        for col in btc_data.columns:
            if isinstance(col, tuple): # Achatamento simples de MultiIndex
                new_cols.append('_'.join(map(str, col)).lower().strip().replace(' ', '_'))
            elif isinstance(col, str):
                new_cols.append(col.lower().strip().replace(' ', '_')) # Minúsculas e substitui espaço
            else:
                new_cols.append(str(col).lower().strip().replace(' ', '_')) # Fallback
        btc_data.columns = new_cols
        # *** FIM DA CORREÇÃO ***

        required_ohlcv = ['open', 'high', 'low', 'close', 'volume']
        if not all(c in btc_data.columns for c in required_ohlcv):
             raise ValueError(f"Colunas OHLCV ausentes após renomear: {required_ohlcv}")

        btc_data.reset_index(inplace=True)
        btc_data['date'] = pd.to_datetime(btc_data['date']).dt.normalize()
        btc_data['price'] = btc_data['close'] # Usa 'close' como 'price' principal
        cols_order = ['date', 'open', 'high', 'low', 'close', 'price', 'volume']
        # Garante que todas as colunas existem antes de reordenar
        cols_order = [c for c in cols_order if c in btc_data.columns]
        data['prices'] = btc_data[cols_order].sort_values('date').reset_index(drop=True)

        # --- Calcular Indicadores (mantido) ---
        if not data['prices'].empty and SKLEARN_AVAILABLE: # Precisa do sklearn para scaler do GP
            df = data['prices']
            for window in DEFAULT_SETTINGS['ma_windows']: df[f'MA{window}'] = df['price'].rolling(window).mean()
            rsi_w = DEFAULT_SETTINGS["rsi_window"]; df[f'RSI_{rsi_w}'] = calculate_rsi(df['price'], rsi_w); df['RSI_14'] = df[f'RSI_{rsi_w}'] # Alias
            df[f'MACD_12_26'], df[f'MACD_Signal_9'] = calculate_macd(df['price'], 12, 26, 9); df['MACD'] = df['MACD_12_26']; df['MACD_Signal'] = df['MACD_Signal_9']
            bb_w = DEFAULT_SETTINGS["bb_window"]; upper, lower = calculate_bollinger_bands(df['price'], bb_w); df[f'BB_Upper_{bb_w}'] = upper; df[f'BB_Lower_{bb_w}'] = lower; df['BB_Upper_20'] = upper; df['BB_Lower_20'] = lower # Alias
            df['OBV'] = calculate_obv(df['price'], df['volume']); df['OBV_MA_20'] = df['OBV'].rolling(20).mean()
            k, d = calculate_stochastic(df['price'], df['high'], df['low'], 14, 3); df['Stoch_K_14_3'] = k; df['Stoch_D_14_3'] = d
            gp_w = DEFAULT_SETTINGS["gp_window"]; df[f'GP_Prediction_{gp_w}'] = calculate_gaussian_process(df['price'], gp_w, 1)
            df['RSI_Divergence'] = detect_divergences(df['price'], df[f'RSI_{rsi_w}'])
            lookback_sr = 90
            if len(df['price'].dropna()) >= lookback_sr: data['support_resistance'] = detect_support_resistance_clusters(df['price'].dropna().tail(lookback_sr).values, DEFAULT_SETTINGS['n_clusters'])
            else: data['support_resistance'] = []
            df['Volume_MA_20'] = df['volume'].rolling(20).mean()
            data['prices'] = df

        # --- Dados On-Chain / Simulados (mantido) ---
        # ... (código para hashrate, difficulty, exchanges_simulated, whale_alert_simulated, news) ...
        try: # Hashrate
            hr_response=requests.get("https://api.blockchain.info/charts/hash-rate?format=json×pan=1year", timeout=10); hr_response.raise_for_status(); hr_data=pd.DataFrame(hr_response.json()["values"]); hr_data["date"]=pd.to_datetime(hr_data["x"], unit="s").dt.normalize(); hr_data['y']=hr_data['y']/1e12; data['hashrate']=hr_data[['date', 'y']].dropna()
        except Exception: data['hashrate']=pd.DataFrame({'date': [], 'y': []})
        try: # Difficulty
            diff_response=requests.get("https://api.blockchain.info/charts/difficulty?timespan=1year&format=json", timeout=10); diff_response.raise_for_status(); diff_data=pd.DataFrame(diff_response.json()["values"]); diff_data["date"]=pd.to_datetime(diff_data["x"], unit="s").dt.normalize(); diff_data['y']=diff_data['y']/1e12; data['difficulty']=diff_data[['date', 'y']].dropna()
        except Exception: data['difficulty']=pd.DataFrame({'date': [], 'y': []})
        data['exchanges_simulated'] = get_exchange_flows_simulated()
        news_end_date = datetime.now(tz='UTC').normalize()
        data['whale_alert_simulated'] = pd.DataFrame({"date": pd.date_range(end=news_end_date - timedelta(days=1), periods=5, freq='12H'), "amount": np.random.randint(500, 5000, 5), "exchange": ["Binance", "Coinbase", "Kraken", "Unknown", "Binance"]})
        data['news'] = [{"title": f"Notícia Simulada {i}", "date": news_end_date - timedelta(days=i), "confidence": np.random.uniform(0.6, 0.95), "source": "Fonte Simulada"} for i in range(5)]

    except Exception as e:
        st.error(f"Erro fatal ao carregar/processar dados: {e}")
        # Retorna estrutura vazia ou parcial para evitar parada total
        data = {'prices': pd.DataFrame(), 'hashrate': pd.DataFrame(), 'difficulty': pd.DataFrame(),
                'exchanges_simulated': pd.DataFrame(), 'whale_alert_simulated': pd.DataFrame(),
                'news': [], 'support_resistance': []}
    return data

# --- Geração de Sinais V2 (mantida) ---
def generate_signals_v2(data, settings, lstm_prediction=None, rl_action=None):
    signals = []; buy_score, sell_score, neutral_score = 0.0, 0.0, 0.0
    df = data.get('prices', pd.DataFrame());
    if df.empty: return signals, "➖ DADOS INDISP.", 0, 0
    last_row = df.iloc[-1]; prev_row = df.iloc[-2] if len(df) > 1 else last_row
    last_price = last_row.get('price', np.nan)
    if pd.isna(last_price): return signals, "➖ PREÇO INDISP.", 0, 0

    def add_signal(name, condition_buy, condition_sell, value_display, weight):
        nonlocal buy_score, sell_score, neutral_score; signal_text = "NEUTRO"; score = 0.0
        if condition_buy: signal_text = "COMPRA"; score = 1.0 * weight; buy_score += score
        elif condition_sell: signal_text = "VENDA"; score = -1.0 * weight; sell_score += abs(score)
        else: neutral_score += weight
        signals.append({'name': name, 'signal': signal_text, 'value': value_display, 'score': score, 'weight': weight})

    # --- Sinais Técnicos ---
    # RSI
    rsi_val = last_row.get(f'RSI_{settings["rsi_window"]}', np.nan)
    if not pd.isna(rsi_val): add_signal(f"RSI ({settings['rsi_window']})", rsi_val < 30, rsi_val > 70, f"{rsi_val:.1f}", INDICATOR_WEIGHTS['rsi'])
    # MACD
    macd_val, macd_sig = last_row.get('MACD', np.nan), last_row.get('MACD_Signal', np.nan)
    if not pd.isna(macd_val) and not pd.isna(macd_sig): add_signal("MACD (Linha vs Sinal)", macd_val > macd_sig, macd_val < macd_sig, f"{macd_val:.2f}/{macd_sig:.2f}", INDICATOR_WEIGHTS['macd'])
    # Bollinger
    bb_u, bb_l = last_row.get(f'BB_Upper_{settings["bb_window"]}', np.nan), last_row.get(f'BB_Lower_{settings["bb_window"]}', np.nan)
    if not pd.isna(bb_u) and not pd.isna(bb_l): add_signal(f"Bollinger ({settings['bb_window']})", last_price < bb_l, last_price > bb_u, f"${bb_l:,.0f}-${bb_u:,.0f}", INDICATOR_WEIGHTS['bollinger'])
    # Volume
    vol, vol_ma = last_row.get('volume', np.nan), last_row.get('Volume_MA_20', np.nan)
    price_chg = last_price - prev_row.get('price', last_price)
    if not pd.isna(vol) and not pd.isna(vol_ma) and vol_ma > 0: vol_r = vol/vol_ma; add_signal("Volume (vs MA20)", vol_r > 1.5 and price_chg > 0, vol_r > 1.5 and price_chg < 0, f"{vol_r:.1f}x", INDICATOR_WEIGHTS['volume'])
    # OBV
    obv, obv_ma = last_row.get('OBV', np.nan), last_row.get('OBV_MA_20', np.nan)
    if not pd.isna(obv) and not pd.isna(obv_ma): add_signal("OBV (vs MA20)", obv > obv_ma and price_chg > 0, obv < obv_ma and price_chg < 0, f"{obv/1e6:.1f}M", INDICATOR_WEIGHTS['obv'])
    # Stochastic
    k, d = last_row.get('Stoch_K_14_3', np.nan), last_row.get('Stoch_D_14_3', np.nan)
    if not pd.isna(k) and not pd.isna(d): add_signal("Stochastic (14,3)", k < 20 and d < 20, k > 80 and d > 80, f"K:{k:.1f},D:{d:.1f}", INDICATOR_WEIGHTS['stochastic'])
    # Gaussian Process
    gp_pred = last_row.get(f'GP_Prediction_{settings["gp_window"]}', np.nan)
    if not pd.isna(gp_pred): gp_thresh=0.02; add_signal("Gaussian Process", gp_pred > last_price*(1+gp_thresh), gp_pred < last_price*(1-gp_thresh), f"P:${gp_pred:,.0f}", INDICATOR_WEIGHTS['gaussian_process'])
    # Order Blocks
    _, current_blocks = identify_order_blocks(data['prices'], **settings) # Usa dados completos para ID
    ob_buy, ob_sell = False, False; ob_disp = "N/A"
    for block in reversed(current_blocks):
        is_br, b_type, br_type = block.get('broken',False), block.get('type'), block.get('breaker_type')
        in_b, in_brz = block['low']<=last_price<=block['high'], block['low']*0.99<=last_price<=block['high']*1.01
        if not is_br:
            if b_type=='bullish_ob' and in_b: ob_buy=True; ob_disp=f"BullOB:{block['low']:,.0f}-{block['high']:,.0f}"; break
            if b_type=='bearish_ob' and in_b: ob_sell=True; ob_disp=f"BearOB:{block['low']:,.0f}-{block['high']:,.0f}"; break
        else:
            if br_type=='bullish_breaker' and in_brz: ob_sell=True; ob_disp=f"BullBrk:{block['low']:,.0f}-{block['high']:,.0f}"; break
            if br_type=='bearish_breaker' and in_brz: ob_buy=True; ob_disp=f"BearBrk:{block['low']:,.0f}-{block['high']:,.0f}"; break
    add_signal("Order/Breaker Block", ob_buy, ob_sell, ob_disp, INDICATOR_WEIGHTS['order_blocks'])
    # Divergência
    rsi_div = last_row.get('RSI_Divergence', 0); div_disp="Nenhuma"; div_buy=(rsi_div==1); div_sell=(rsi_div==-1)
    if div_buy: div_disp="Alta"; elif div_sell: div_disp="Baixa"
    add_signal("Divergência RSI", div_buy, div_sell, div_disp, INDICATOR_WEIGHTS['divergence'])

    # --- Sinais IA ---
    if lstm_prediction is not None: thresh=0.01; add_signal("LSTM Previsão", lstm_prediction>last_price*(1+thresh), lstm_prediction<last_price*(1-thresh), f"P:${lstm_prediction:,.0f}", INDICATOR_WEIGHTS['lstm_pred'])
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


# --- Funções PDF e Excel (mantidas) ---
# clean_text, generate_pdf_report
# ...

# ======================
# |||| LOOP PRINCIPAL ||||
# ======================
def main():
    # --- Inicialização ---
    if 'user_settings' not in st.session_state: st.session_state.user_settings = DEFAULT_SETTINGS.copy()
    settings = st.session_state.user_settings

    # Carrega modelos (IA e Scalers)
    sentiment_model = load_sentiment_model()
    lstm_model, lstm_scaler = load_lstm_model_and_scaler()
    rl_model, rl_scaler, rl_env_config = None, None, None
    if SB3_AVAILABLE and SKLEARN_AVAILABLE and os.path.exists(RL_MODEL_PATH):
        try:
            rl_model = PPO.load(RL_MODEL_PATH, device='auto') # Carrega na GPU se possível
            if os.path.exists(RL_SCALER_PATH): rl_scaler = joblib.load(RL_SCALER_PATH)
            else: rl_model = None
            if os.path.exists(RL_ENV_CONFIG_PATH): rl_env_config = joblib.load(RL_ENV_CONFIG_PATH)
            else: rl_model = None
        except Exception as e: st.error(f"Erro load RL: {e}"); rl_model=None

    # --- Sidebar (mantida) ---
    with st.sidebar:
        st.header("⚙️ Painel de Controle AI v2.1")
        # ... (Expanders e Sliders/Widgets como antes) ...
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
        if st.button("🔄 Atualizar Dados", type="primary", use_container_width=True, key='bt_upd'):
            st.cache_data.clear(); st.success("Cache limpo..."); st.rerun()


    # --- Carregamento e Processamento Principal ---
    master_data = load_and_process_data()
    if 'prices' not in master_data or master_data['prices'].empty:
        st.error("Erro crítico dados. Dashboard parado.")
        st.stop()

    # --- Previsão/Ação IA (Usa modelos carregados) ---
    current_lstm_prediction = None
    if lstm_model and lstm_scaler:
        current_lstm_prediction = predict_with_lstm(lstm_model, lstm_scaler, master_data['prices'], settings['lstm_window'])

    current_rl_action = None
    # ... (Lógica para simular ÚLTIMA ação RL - como antes) ...
    if rl_model and rl_scaler and rl_env_config:
         try:
             rl_feature_cols_norm = rl_env_config['feature_cols'] # Colunas normalizadas
             rl_feature_cols_base = [c.replace('_norm','') for c in rl_feature_cols_norm]
             df_rl_current = master_data['prices'].copy()
             # Verifica se todas as colunas base existem
             if not all(c in df_rl_current.columns for c in rl_feature_cols_base):
                  raise ValueError("Colunas base para RL não encontradas nos dados atuais.")
             # Remove linhas com NaN nas colunas base ANTES de escalar/simular
             df_rl_current.dropna(subset=rl_feature_cols_base, inplace=True)
             df_rl_current = df_rl_current.reset_index(drop=True)

             if not df_rl_current.empty:
                 env_sim = BitcoinTradingEnv(df_rl_current, rl_feature_cols_norm, rl_scaler, settings['rl_transaction_cost'])
                 obs, info = env_sim.reset(); terminated, truncated = False, False
                 sim_steps = 0; max_sim_steps = len(df_rl_current) # Limite de segurança
                 while not (terminated or truncated) and sim_steps < max_sim_steps:
                     action, _ = rl_model.predict(obs, deterministic=True)
                     # Guarda a ação ANTES do último step
                     if env_sim.current_step == len(df_rl_current) - 2:
                         current_rl_action = action
                     obs, _, terminated, truncated, info = env_sim.step(action)
                     sim_steps += 1
                 if current_rl_action is None and sim_steps > 0: # Se terminou no primeiro step, pega a primeira ação
                      action, _ = rl_model.predict(env_sim.reset()[0], deterministic=True)
                      current_rl_action = action

         except Exception as e: st.warning(f"Erro simulação RL: {e}")


    # --- Geração de Sinais Combinados ---
    signals, final_verdict, buy_count, sell_count = generate_signals_v2(
        master_data, settings, current_lstm_prediction, current_rl_action
    )

    # --- Busca Dados Adicionais (mantido) ---
    sentiment = get_market_sentiment()
    traditional_assets = get_traditional_assets()
    # ... (análise e filtro de notícias como antes) ...
    if sentiment_model and 'news' in master_data: analyzed_news = analyze_news_sentiment(master_data['news'], sentiment_model)
    else: analyzed_news = master_data.get('news', [])
    filtered_news = filter_news_by_confidence(analyzed_news, settings['min_confidence'])


    # --- Layout Principal (Métricas e Tabs - mantido) ---
    st.header("📊 Painel Integrado BTC AI Pro+ v2.1")
    # Métricas (mantidas)
    # ...
    # Tabs (mantidas - verificar se os nomes das tabs ainda fazem sentido com a mudança da Tab 5)
    tab_titles = ["📈 Mercado", "🆚 Comparativos", "🧪 Backtesting", "🌍 Cenários", "🤖 IA Training", "📉 Técnico", "📤 Exportar"]
    tabs = st.tabs(tab_titles)

    # Tab 1: Mercado (mantida - mostra gráficos e sinais atuais)
    # ...
    # Tab 2: Comparativos (mantida)
    # ...
    # Tab 3: Backtesting (mantida)
    # ...
    # Tab 4: Cenários (mantida)
    # ...
    # Tab 5: IA Training (mantida - foca no treino e status dos modelos)
    # ...
    # Tab 6: Técnico (mantida - gráficos detalhados)
    # ...
    # Tab 7: Exportar (mantida)
    # ...

    # Colar o conteúdo das abas aqui, adaptando se necessário para usar 'master_data'
    # (O conteúdo das abas é longo, omitido aqui para brevidade, mas use a versão
    # do código anterior, garantindo que as fontes de dados e variáveis estejam corretas)


if __name__ == "__main__":
    # Opcional: Limitar GPU para TensorFlow (como antes)
    # if TF_AVAILABLE:
    #    try: ... tf.config ...
    #    except: ...

    main()
