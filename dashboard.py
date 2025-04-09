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
import io # Necessário para exportar excel em memória
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
# Removidas importações de PyTorch não utilizadas diretamente no código principal
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader

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
    'ob_swing_length': 11, # Default ímpar
    'ob_show_bull': 3,
    'ob_show_bear': 3,
    'ob_use_body': True,
    'min_confidence': 0.7,
    'n_clusters': 5,
    'lstm_window': 60,
    'lstm_epochs': 50,
    'lstm_units': 50,
    'rl_episodes': 10000 # Aumentado default para timesteps
}

# ======================
# FUNÇÕES DE IA
# ======================

# CORREÇÃO: Classe BitcoinTradingEnv com indentação correta e compatível com Gymnasium
class BitcoinTradingEnv(gym.Env):
    """Ambiente customizado para trading de Bitcoin com Gymnasium."""
    metadata = {'render_modes': ['human'], 'render_fps': 30} # Metadados Gymnasium

    def __init__(self, df, initial_balance=10000, render_mode=None):
        super(BitcoinTradingEnv, self).__init__()

        # Verificar se o DataFrame tem as colunas necessárias
        required_cols = ['price', 'volume', f'RSI_{DEFAULT_SETTINGS["rsi_window"]}',
                         'MACD', 'MACD_Signal', f'BB_Upper_{DEFAULT_SETTINGS["bb_window"]}',
                         f'BB_Lower_{DEFAULT_SETTINGS["bb_window"]}'] # Adicione outras se usar no obs
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame de entrada para BitcoinTradingEnv não possui as colunas: {missing_cols}")

        self.df = df.dropna(subset=required_cols).reset_index(drop=True) # Remover NaNs nas colunas usadas e resetar índice
        if len(self.df) < 2:
             raise ValueError("DataFrame tem menos de 2 linhas válidas após dropna.")

        self.initial_balance = initial_balance
        self.current_step = 0
        self.render_mode = render_mode # Para renderização Gymnasium

        # Ações: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)

        # Espaço de Observação: Normalizado e com número fixo de features
        # O shape deve corresponder exatamente ao número de elementos retornados por _next_observation
        self.observation_shape = (10,) # Ajuste se mudar as features em _next_observation
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float32
        )

        # --- Inicialização de estado --- (movido para reset)
        # self.balance = 0
        # self.btc_held = 0
        # self.total_profit = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Necessário para compatibilidade Gymnasium (gerencia seed)

        # Inicializar estado do ambiente
        self.balance = self.initial_balance
        self.btc_held = 0
        self.current_step = 0 # Iniciar no primeiro passo válido
        self.total_profit = 0
        self.last_portfolio_value = self.initial_balance # Para cálculo de recompensa

        # Retornar a primeira observação e info (dicionário vazio por padrão)
        observation = self._next_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        # Gymnasium espera (observation, info) no reset
        return observation, info

    def _get_info(self):
        # Retorna informações adicionais sobre o estado (opcional)
        return {
            "step": self.current_step,
            "balance": self.balance,
            "btc_held": self.btc_held,
            "total_profit": self.total_profit,
            "portfolio_value": self.last_portfolio_value
        }

    # CORREÇÃO: Indentação correta para _next_observation ser um método da classe
    def _next_observation(self):
        # Garantir que current_step está dentro dos limites
        safe_step = min(self.current_step, len(self.df) - 1)
        current_data = self.df.iloc[safe_step]

        # Obter dados com segurança usando .get() ou acesso direto se garantido por dropna
        price = current_data['price']
        volume = current_data['volume']
        rsi = current_data[f'RSI_{DEFAULT_SETTINGS["rsi_window"]}']
        macd = current_data['MACD']
        macd_signal = current_data['MACD_Signal']
        bb_upper = current_data[f'BB_Upper_{DEFAULT_SETTINGS["bb_window"]}']
        bb_lower = current_data[f'BB_Lower_{DEFAULT_SETTINGS["bb_window"]}']

        # Normalização (exemplo - pode precisar de ajustes)
        # Usar médias ou valores fixos grandes para normalização pode ser problemático
        # Scaler pré-treinado seria ideal, mas simplificando:
        obs = np.array([
            np.log1p(price / (self.df['price'].mean() if self.df['price'].mean() else 1)), # Log do preço normalizado
            np.log1p(volume / (self.df['volume'].mean() if self.df['volume'].mean() else 1)), # Log do volume normalizado
            rsi / 100.0, # RSI normalizado 0-1
            macd / (price if price else 1), # MACD relativo ao preço
            macd_signal / (price if price else 1), # Sinal MACD relativo ao preço
            self.balance / self.initial_balance, # Saldo relativo
            (self.btc_held * price) / self.initial_balance, # Valor BTC relativo
            self.current_step / (len(self.df) -1 if len(self.df) > 1 else 1), # Progresso
            bb_upper / (price if price else 1), # BB Upper relativo ao preço
            bb_lower / (price if price else 1), # BB Lower relativo ao preço
        ], dtype=np.float32)

        # Tratamento final de NaN/Inf (embora dropna deva ter ajudado)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        # Verificar shape (importante para Stable Baselines)
        if obs.shape != self.observation_shape:
             # Pode acontecer se alguma coluna não foi calculada corretamente
             # Retornar um array de zeros ou lançar erro
             print(f"ALERTA RL: Shape da observação {obs.shape} diferente do esperado {self.observation_shape}")
             return np.zeros(self.observation_shape, dtype=np.float32)

        return obs

    # CORREÇÃO: Indentação correta para step ser um método da classe
    def step(self, action):
        # Verificar se o episódio já terminou em chamadas anteriores
        terminated = self.current_step >= len(self.df) - 1

        if terminated:
             # Se já terminou, retornar a última observação e zero reward
             # É importante não avançar mais o step
             obs = self._next_observation()
             return obs, 0, terminated, False, self._get_info() # truncated = False

        # Pegar preço atual
        current_price = self.df.iloc[self.current_step]['price']
        if pd.isna(current_price) or current_price <= 0:
            # Tentar usar preço anterior ou pular passo (melhor tratar NaNs antes)
             current_price = self.df.iloc[self.current_step-1]['price'] if self.current_step > 0 else self.initial_balance # Fallback

        # Executar ação
        if action == 1:  # Buy
            if self.balance > 10: # Comprar apenas se tiver saldo (deixar margem para taxas futuras)
                amount_to_buy = self.balance / current_price
                self.btc_held += amount_to_buy
                self.balance = 0
        elif action == 2:  # Sell
            if self.btc_held > 0.0001: # Vender apenas se tiver BTC
                self.balance += self.btc_held * current_price
                self.btc_held = 0

        # Avançar para o próximo passo
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1 # Verificar novamente se terminou
        truncated = False # Geralmente usado para limites de tempo, não aplicável aqui

        # Calcular valor atual do portfólio
        next_price = self.df.iloc[self.current_step]['price'] if not terminated else current_price
        if pd.isna(next_price) or next_price <= 0:
            next_price = current_price # Usar preço atual se o próximo for inválido

        current_portfolio_value = self.balance + (self.btc_held * next_price)

        # Calcular recompensa (ex: mudança no valor do portfólio)
        reward = (current_portfolio_value - self.last_portfolio_value) / self.initial_balance # Recompensa normalizada
        self.last_portfolio_value = current_portfolio_value # Atualizar para próximo passo
        self.total_profit = current_portfolio_value - self.initial_balance # Lucro total

        # Obter a próxima observação
        observation = self._next_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        # Gymnasium espera (obs, reward, terminated, truncated, info)
        return observation, reward, terminated, truncated, info

    # CORREÇÃO: Indentação correta para render ser um método da classe
    def render(self):
        # Implementação de renderização (opcional, pode ser via print ou visualização)
        if self.render_mode == "human":
             self._render_frame()

    def _render_frame(self):
         # Exemplo simples de renderização via print
         current_price = self.df.iloc[min(self.current_step, len(self.df)-1)]['price']
         portfolio_value = self.balance + self.btc_held * current_price
         print(f"Step: {self.current_step}, Price: {current_price:.2f}, Balance: {self.balance:.2f}, BTC: {self.btc_held:.6f}, Portfolio: {portfolio_value:.2f}, Profit: {self.total_profit:.2f}")


    def close(self):
        # Limpeza se necessário (ex: fechar janelas de visualização)
        pass

# --- Funções de Análise de Sentimento e LSTM ---

@st.cache_resource # Mantido para carregar o modelo uma vez
def load_sentiment_model():
    """Carrega o modelo de análise de sentimento."""
    try:
        return pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            framework="pt",
            device=-1 # Força CPU
        )
    except Exception as e:
        st.error(f"Erro ao carregar modelo de sentimentos: {str(e)}")
        # Tentar um fallback ou retornar None
        try:
            # Tentar um modelo diferente ou mais robusto se disponível
            # return pipeline("sentiment-analysis", model="...")
            return None # Retornar None se falhar
        except Exception as fallback_e:
             st.error(f"Erro no modelo de fallback de sentimentos: {fallback_e}")
             return None


def analyze_news_sentiment(news_list, _model):
    """Analisa o sentimento das notícias usando o modelo carregado."""
    if _model is None or not news_list or not isinstance(news_list, list):
        # Retorna a lista original com status de erro se o modelo falhou ou a lista é inválida
        return [{'title': n.get('title', 'N/A'), 'sentiment': 'MODEL_ERROR', 'sentiment_score': 0.0, **n} for n in news_list if isinstance(n, dict)]


    results = []
    # Extrair apenas títulos válidos para processamento em lote
    valid_news = [n for n in news_list if isinstance(n, dict) and 'title' in n and isinstance(n['title'], str)]
    texts = [n['title'] for n in valid_news]

    if not texts:
        return news_list # Retorna original se não houver títulos válidos

    try:
        # Processar textos válidos em lote
        predictions = _model(texts)

        # Mapear resultados de volta para as notícias válidas
        prediction_map = {text: pred for text, pred in zip(texts, predictions)}

        # Construir lista final mantendo a estrutura original
        for news_item in news_list:
            if isinstance(news_item, dict) and 'title' in news_item and news_item['title'] in prediction_map:
                result = prediction_map[news_item['title']]
                news_item['sentiment'] = result['label']
                news_item['sentiment_score'] = result['score']
            elif isinstance(news_item, dict): # Manter item mesmo que título falhe ou não esteja no map
                 news_item['sentiment'] = 'ANALYSIS_SKIPPED'
                 news_item['sentiment_score'] = 0.0
            # Ignorar itens que não são dicionários na lista original
            if isinstance(news_item, dict):
                results.append(news_item)

    except Exception as e:
        st.warning(f"Erro na análise de sentimento em lote: {e}. Marcando como erro.")
        # Marcar todos os itens como erro em caso de falha no lote
        results = []
        for news_item in news_list:
            if isinstance(news_item, dict):
                 news_item['sentiment'] = 'BATCH_ERROR'
                 news_item['sentiment_score'] = 0.0
                 results.append(news_item)

    return results


@st.cache_resource # Cache para o modelo LSTM
def create_lstm_model(input_shape, units=50):
    """Cria a arquitetura do modelo LSTM."""
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'), # Adicionar ativação
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def prepare_lstm_data(data, n_steps=60):
    """Prepara os dados para o LSTM (X, y) e retorna o scaler."""
    if data is None or data.empty or 'price' not in data.columns or len(data) <= n_steps:
        st.warning(f"Dados insuficientes ({len(data) if data is not None else 0}) ou inválidos para preparar LSTM (necessário > {n_steps}).")
        return None, None, None

    # Usar dropna para garantir que apenas dados válidos sejam usados
    price_data = data['price'].dropna().values.reshape(-1, 1)
    if len(price_data) <= n_steps:
         st.warning(f"Dados válidos insuficientes ({len(price_data)}) após dropna para preparar LSTM (necessário > {n_steps}).")
         return None, None, None

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(price_data)

    X, y = [], []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i-n_steps:i, 0])
        y.append(scaled_data[i, 0])

    if not X or not y:
        st.warning("Não foi possível gerar dados X, y para LSTM.")
        return None, None, scaler # Retorna scaler mesmo se X,y falhar

    return np.array(X), np.array(y), scaler


def train_lstm_model(data, epochs=50, batch_size=32, window=60):
    """Treina o modelo LSTM com os dados preparados."""
    X, y, scaler = prepare_lstm_data(data, window)
    if X is None or y is None or scaler is None or X.shape[0] == 0:
         st.error("Falha ao preparar dados ou dados insuficientes para treinamento LSTM.")
         return None, None # Não pode treinar

    # Reshape X para o formato esperado pelo LSTM [samples, time_steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Criar o modelo
    model = create_lstm_model((X.shape[1], 1), units=st.session_state.user_settings.get('lstm_units', 50))

    try:
        # Treinar o modelo
        history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.1) # Adicionar validação
        st.write(f"Treinamento LSTM Concluído. Perda final (validação): {history.history['val_loss'][-1]:.4f}")
    except Exception as e:
        st.error(f"Erro durante o treinamento do LSTM: {e}")
        return None, scaler # Retornar scaler mesmo com erro no treino

    return model, scaler


def predict_with_lstm(model, scaler, data, window=60):
    """Faz previsões para o próximo passo usando o modelo LSTM treinado."""
    if model is None or scaler is None or data is None or data.empty or 'price' not in data.columns:
        st.error("Modelo LSTM, scaler ou dados inválidos para previsão.")
        return np.nan

    # Pegar os últimos 'window' pontos válidos de preço
    last_window_data = data['price'].dropna().values[-window:]

    if len(last_window_data) < window:
        st.warning(f"Dados insuficientes ({len(last_window_data)}) para formar a janela LSTM ({window}). Previsão pode ser imprecisa.")
        # Tentar preencher com o último valor se houver algum dado
        if len(last_window_data) > 0:
            fill_value = last_window_data[-1]
            # Pad no início para manter a ordem temporal
            last_window_data = np.pad(last_window_data, (window - len(last_window_data), 0), 'constant', constant_values=fill_value)
        else:
            st.error("Não há dados válidos recentes para a janela LSTM. Impossível prever.")
            return np.nan # Retorna NaN se não há como formar a janela

    # Escalar a janela
    last_window_scaled = scaler.transform(last_window_data.reshape(-1,1))

    # Preparar para previsão (reshape)
    X_test = np.array([last_window_scaled[:,0]])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    try:
        # Fazer a previsão
        pred_scaled = model.predict(X_test, verbose=0)
        # Inverter a escala para obter o preço previsto
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
    except Exception as e:
        st.error(f"Erro durante a predição LSTM: {e}")
        return np.nan # Retorna NaN em caso de erro na previsão

    return pred_price


# ============================================
# FUNÇÕES AUXILIARES E DE CÁLCULO DE INDICADORES
# ============================================

def calculate_ema(series, window):
    """Calcula a Média Móvel Exponencial (EMA) de forma robusta."""
    if series is None or series.empty or series.isna().all():
        return pd.Series(dtype=float, index=series.index if series is not None else None)
    # Usar min_periods=window para garantir que só calcule com dados suficientes
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def calculate_rsi(series, window=14):
    """Calcula o Índice de Força Relativa (RSI) de forma robusta."""
    if series is None or len(series.dropna()) < window + 1:
        return pd.Series(np.nan, index=series.index if series is not None else None)

    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).fillna(0) # Preencher NaNs iniciais com 0
    loss = -delta.where(delta < 0, 0.0).fillna(0) # Preencher NaNs iniciais com 0

    # Usar EWM para cálculo mais estável do RSI
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

    # Evitar divisão por zero
    rs = avg_gain / avg_loss.replace(0, 1e-10) # Substituir 0 por valor pequeno

    rsi = 100.0 - (100.0 / (1.0 + rs))

    return pd.Series(rsi, index=series.index)


def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calcula o MACD, Linha de Sinal e Histograma de forma robusta."""
    if series is None or len(series.dropna()) < slow:
        nan_series = pd.Series(np.nan, index=series.index if series is not None else None)
        return nan_series, nan_series, nan_series # MACD, Signal, Hist

    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd = ema_fast - ema_slow

    # Calcular linha de sinal apenas se houver MACD suficiente
    if len(macd.dropna()) >= signal:
        signal_line = calculate_ema(macd, signal)
    else:
        signal_line = pd.Series(np.nan, index=series.index)

    histogram = macd - signal_line

    return macd, signal_line, histogram


def calculate_bollinger_bands(series, window=20, num_std=2):
    """Calcula as Bandas de Bollinger (Sup, Inf, Média) de forma robusta."""
    if series is None or len(series.dropna()) < window:
        nan_series = pd.Series(np.nan, index=series.index if series is not None else None)
        return nan_series, nan_series, nan_series # Upper, Lower, MA

    # Usar min_periods=window
    sma = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()

    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, lower, sma


def calculate_bollinger_bandwidth_pctb(price_series, upper_band, lower_band, middle_band):
    """Calcula Bollinger %B e Bandwidth."""
    # Garantir que os inputs são Series e alinhados
    if not all(isinstance(s, pd.Series) for s in [price_series, upper_band, lower_band, middle_band]):
         nan_series = pd.Series(np.nan, index=price_series.index if price_series is not None else None)
         return nan_series, nan_series # BW, PctB

    # Calcular largura como percentual da média
    bandwidth = ((upper_band - lower_band) / middle_band.replace(0, 1e-10)) * 100

    # Calcular %B (posição relativa do preço)
    band_range = (upper_band - lower_band).replace(0, 1e-10) # Evitar divisão por zero
    pct_b = (price_series - lower_band) / band_range

    # Tratar infinitos que podem surgir
    bandwidth = bandwidth.replace([np.inf, -np.inf], np.nan)
    pct_b = pct_b.replace([np.inf, -np.inf], np.nan)

    return bandwidth, pct_b


def calculate_obv(price_series, volume_series):
    """Calcula o On-Balance Volume (OBV) de forma robusta."""
    if price_series is None or volume_series is None or len(price_series) != len(volume_series) or len(price_series) < 2:
         return pd.Series(dtype=float, index=price_series.index if price_series is not None else None)

    # Trabalhar com cópias e remover NaNs alinhados
    df = pd.DataFrame({'price': price_series, 'volume': volume_series}).dropna()
    if len(df) < 2:
        return pd.Series(dtype=float, index=price_series.index) # Reindexar para original

    # Calcular diferença de preço
    price_diff = df['price'].diff()

    # Atribuir volume com sinal (+ se preço sobe, - se cai, 0 se igual)
    signed_volume = df['volume'] * np.sign(price_diff).fillna(0) # fillna(0) para o primeiro elemento

    # Calcular OBV cumulativo e reindexar
    obv = signed_volume.cumsum()
    return obv.reindex(price_series.index).ffill().fillna(0) # Preencher NaNs iniciais com ffill e depois 0


def calculate_stochastic(price_series, k_window=14, d_window=3):
    """Calcula o Stochastic Oscillator (%K, %D) de forma robusta."""
    if price_series is None or len(price_series.dropna()) < k_window:
        nan_series = pd.Series(np.nan, index=price_series.index if price_series is not None else None)
        return nan_series, nan_series # K, D

    # Usar min_periods
    low_min = price_series.rolling(k_window, min_periods=k_window).min()
    high_max = price_series.rolling(k_window, min_periods=k_window).max()

    # Calcular %K (Stoch cru)
    delta = (high_max - low_min).replace(0, np.nan) # Evitar divisão por zero
    stoch_raw = 100 * (price_series - low_min) / delta

    # Calcular %K (Stoch suavizado) e %D (média do %K)
    stoch_k = stoch_raw.rolling(d_window, min_periods=d_window).mean()
    stoch_d = stoch_k.rolling(d_window, min_periods=d_window).mean()

    return stoch_k, stoch_d


def calculate_gaussian_process(price_series, window=30, lookahead=5):
    """Calcula a Regressão de Processo Gaussiano (GP) para previsão."""
    # Aviso: GP é computacionalmente intensivo.
    if price_series is None or len(price_series.dropna()) < window + 1:
        return pd.Series(np.nan, index=price_series.index if price_series is not None else None)

    valid_prices = price_series.dropna()
    if len(valid_prices) < window + 1:
        return pd.Series(np.nan, index=price_series.index)

    # Kernel simples para GP
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True, random_state=0)

    predictions = pd.Series(np.nan, index=valid_prices.index) # Iniciar com NaNs

    # Iterar sobre os dados válidos para treinar e prever
    # Prever 'lookahead' passos à frente do final da janela de treino
    for i in range(window, len(valid_prices)):
        X_train = np.arange(window).reshape(-1, 1)
        y_train = valid_prices.iloc[i-window : i].values

        if np.isnan(y_train).any(): continue # Segurança extra

        try:
            gpr.fit(X_train, y_train)
            # Prever 'lookahead' passos à frente
            X_pred = np.arange(window, window + lookahead).reshape(-1, 1)
            y_pred, _ = gpr.predict(X_pred, return_std=True)
            # Atribuir a previsão do último passo do lookahead ao índice atual 'i'
            # Isso representa a previsão feita 'lookahead' dias atrás para o dia 'i'
            if i + lookahead -1 < len(predictions): # Verificar limites
                # O correto seria prever 1 passo e atribuir a i+1, mas vamos manter a lógica original por enquanto
                 predictions.iloc[i] = y_pred[-1] # Atribui previsão final ao índice atual
        except Exception as e:
            # st.warning(f"Erro no GPR no índice {valid_prices.index[i]}: {e}") # Log opcional
            predictions.iloc[i] = np.nan

    # Reindexar para o índice original
    return predictions.reindex(price_series.index)


def identify_order_blocks(df, swing_length=11, show_bull=3, show_bear=3, use_body=True):
    """Identifica Order Blocks (OB) e Breaker Blocks (BB) - Estilo LuxAlgo."""
    if df is None or df.empty or not all(col in df.columns for col in ['date', 'open', 'close', 'high', 'low']):
        return df, [] # Retorna df original e lista vazia

    df_copy = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
        df_copy['date'] = pd.to_datetime(df_copy['date'])

    # Garantir swing_length ímpar para center=True
    if swing_length % 2 == 0: swing_length += 1

    # Identificar swings (picos e vales)
    price_high_col = 'close' if use_body else 'high'
    price_low_col = 'close' if use_body else 'low'
    df_copy['swing_high'] = df_copy[price_high_col].rolling(swing_length, center=True, min_periods=swing_length // 2 + 1).max()
    df_copy['swing_low'] = df_copy[price_low_col].rolling(swing_length, center=True, min_periods=swing_length // 2 + 1).min()
    df_copy['is_swing_high'] = df_copy[price_high_col] == df_copy['swing_high']
    df_copy['is_swing_low'] = df_copy[price_low_col] == df_copy['swing_low']

    blocks = []
    processed_swing_indices = set()

    # Iterar de trás para frente para pegar os mais recentes
    for i in range(len(df_copy) - (swing_length // 2 + 1), swing_length // 2 -1, -1):
        # Bearish OB: Candle de alta *antes* de um swing low confirmado
        if df_copy['is_swing_low'].iloc[i] and i not in processed_swing_indices:
            # Procurar o último candle de alta antes deste swing low
            potential_ob_candle = None
            for j in range(i - 1, max(-1, i - swing_length // 2 - 1), -1): # Olhar para trás alguns candles
                candle = df_copy.iloc[j]
                if candle['close'] > candle['open']: # É um candle de alta
                    potential_ob_candle = candle
                    break # Achou o mais próximo
            if potential_ob_candle is not None and sum(b['type'] == 'bearish_ob' for b in blocks) < show_bear:
                 blocks.append({
                     'type': 'bearish_ob', 'date': potential_ob_candle['date'],
                     'high': potential_ob_candle['high'], 'low': potential_ob_candle['low'],
                     'trigger_price': df_copy['low'].iloc[i], # Preço do swing low
                     'broken': False, 'weight': INDICATOR_WEIGHTS.get('order_blocks', 2.0)
                 })
                 processed_swing_indices.add(i) # Marcar swing como processado

        # Bullish OB: Candle de baixa *antes* de um swing high confirmado
        if df_copy['is_swing_high'].iloc[i] and i not in processed_swing_indices:
             # Procurar o último candle de baixa antes deste swing high
            potential_ob_candle = None
            for j in range(i - 1, max(-1, i - swing_length // 2 - 1), -1):
                 candle = df_copy.iloc[j]
                 if candle['close'] < candle['open']: # É um candle de baixa
                     potential_ob_candle = candle
                     break
            if potential_ob_candle is not None and sum(b['type'] == 'bullish_ob' for b in blocks) < show_bull:
                 blocks.append({
                     'type': 'bullish_ob', 'date': potential_ob_candle['date'],
                     'high': potential_ob_candle['high'], 'low': potential_ob_candle['low'],
                     'trigger_price': df_copy['high'].iloc[i], # Preço do swing high
                     'broken': False, 'weight': INDICATOR_WEIGHTS.get('order_blocks', 2.0)
                 })
                 processed_swing_indices.add(i)

        # Parar se já encontrou o suficiente
        if sum(b['type'] == 'bullish_ob' for b in blocks) >= show_bull and \
           sum(b['type'] == 'bearish_ob' for b in blocks) >= show_bear:
            break

    # Verificar Breakers (se preço fechou além do range do OB)
    for block in blocks:
        subsequent_data = df_copy[df_copy['date'] > block['date']]
        if not subsequent_data.empty:
            if block['type'] == 'bullish_ob' and subsequent_data['close'].min() < block['low']:
                block['broken'] = True
                block['breaker_type'] = 'bullish_breaker' # Agora é Resistência
            elif block['type'] == 'bearish_ob' and subsequent_data['close'].max() > block['high']:
                block['broken'] = True
                block['breaker_type'] = 'bearish_breaker' # Agora é Suporte

        # Definir start/end date para plotagem (aproximado)
        block['start_date'] = block['date'] - pd.Timedelta(hours=10) # Ajustar visualmente
        block['end_date'] = block['date'] + pd.Timedelta(hours=10)

    return df_copy, sorted(blocks, key=lambda x: x['date'], reverse=True) # Retorna blocos mais recentes primeiro


def plot_order_blocks(fig, blocks, current_price):
    """Adiciona Order Blocks e Breaker Blocks ao gráfico Plotly."""
    if not blocks or fig is None:
        return fig

    for block in blocks:
        # Definir cores e estilos
        color_fill = "rgba(0, 0, 255, 0.1)" # Bullish OB
        color_line = "blue"
        line_dash = "solid"
        layer = "below"
        annotation_text = f"Bull OB"

        if block['type'] == 'bearish_ob':
            color_fill = "rgba(255, 165, 0, 0.1)" # Bearish OB
            color_line = "orange"
            annotation_text = f"Bear OB"

        if block['broken']:
            line_dash = "dot"
            layer = "above"
            if block['breaker_type'] == 'bullish_breaker':
                color_fill = "rgba(255, 0, 0, 0.1)" # Bullish Breaker (Resistance)
                color_line = "red"
                annotation_text = f"Bull Brkr"
            elif block['breaker_type'] == 'bearish_breaker':
                color_fill = "rgba(0, 255, 0, 0.1)" # Bearish Breaker (Support)
                color_line = "green"
                annotation_text = f"Bear Brkr"

        # Desenhar o retângulo do bloco (ajustar start/end date se necessário)
        # Usar datas fixas relativas à data do bloco para consistência
        start_plot = block['date'] - pd.Timedelta(hours=12) # Exemplo: meio dia antes
        end_plot = block['date'] + pd.Timedelta(hours=12)  # Exemplo: meio dia depois

        fig.add_shape(type="rect",
                        x0=start_plot, y0=block['low'],
                        x1=end_plot, y1=block['high'],
                        line=dict(color=color_line, width=1, dash=line_dash),
                        fillcolor=color_fill,
                        layer=layer)

        # Adicionar anotação no meio do bloco
        fig.add_annotation(
            x=block['date'], y=(block['high'] + block['low']) / 2,
            text=f"{annotation_text}<br>${block['low']:.0f}-${block['high']:.0f}", # Adicionar range de preço
            showarrow=False, yshift=0,
            font=dict(size=8, color=color_line),
            bgcolor="rgba(255,255,255,0.5)" # Fundo branco semi-transparente
        )

    return fig


def detect_support_resistance_clusters(prices, n_clusters=5):
    """Identifica zonas de Suporte/Resistência usando K-Means."""
    if prices is None or not isinstance(prices, np.ndarray) or prices.ndim != 1:
         # st.warning("Input para S/R deve ser um array numpy 1D.")
         return []

    # Remover NaNs e infinitos
    valid_prices = prices[np.isfinite(prices)]
    if len(valid_prices) < n_clusters:
        # st.warning(f"Dados insuficientes ({len(valid_prices)}) para {n_clusters} clusters S/R.")
        return sorted(list(np.unique(valid_prices))) # Retornar níveis únicos se poucos

    # Escalar dados (MinMaxScaler é bom para níveis de preço)
    scaler = MinMaxScaler()
    try:
        X_scaled = scaler.fit_transform(valid_prices.reshape(-1, 1))
    except ValueError: # Todos os preços iguais
         return [valid_prices[0]] if len(valid_prices) > 0 else []

    # Aplicar K-Means com n_init para robustez
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto') # 'auto' a partir do sklearn 1.4
    try:
         kmeans.fit(X_scaled)
    except Exception as e:
        st.warning(f"Erro durante K-Means para S/R: {e}")
        return []

    # Obter centros dos clusters e reverter escala
    if hasattr(kmeans, 'cluster_centers_'):
        clusters = scaler.inverse_transform(kmeans.cluster_centers_)
        # Agrupar clusters próximos (opcional)
        # Exemplo: se a diferença for menor que X% do preço médio, agrupar
        clusters = sorted([c[0] for c in clusters])
        merged_clusters = []
        if clusters:
             merged_clusters.append(clusters[0])
             for i in range(1, len(clusters)):
                  # Critério de merge (ex: < 0.5% de diferença)
                  if (clusters[i] - merged_clusters[-1]) / clusters[i] < 0.005:
                       # Média ponderada ou simples dos clusters agrupados
                       merged_clusters[-1] = (merged_clusters[-1] + clusters[i]) / 2
                  else:
                       merged_clusters.append(clusters[i])
        return merged_clusters
    else:
        return []


def detect_divergences(price_series, indicator_series, window=14):
    """Detecta divergências Bullish/Bearish (Regular) entre preço e indicador."""
    if price_series is None or indicator_series is None or len(price_series) != len(indicator_series):
        return pd.DataFrame({'divergence': 0}, index=price_series.index if price_series is not None else None)

    df = pd.DataFrame({'price': price_series, 'indicator': indicator_series}).dropna()
    if len(df) < window * 2: # Precisa de dados suficientes para comparar picos/vales
         return pd.DataFrame({'divergence': 0}, index=price_series.index).reindex(price_series.index).fillna(0)

    # Encontrar picos e vales (usando rolling max/min é simples, mas pode ser melhorado)
    local_max_window = window // 2 # Janela menor para máximos/mínimos locais
    price_roll = df['price'].rolling(window, center=True, min_periods=local_max_window)
    ind_roll = df['indicator'].rolling(window, center=True, min_periods=local_max_window)

    df['is_price_peak'] = (df['price'] == price_roll.max()) & (df['price'].shift(-local_max_window) < df['price']) # Confirmação futura
    df['is_price_valley'] = (df['price'] == price_roll.min()) & (df['price'].shift(-local_max_window) > df['price']) # Confirmação futura
    df['is_ind_peak'] = (df['indicator'] == ind_roll.max()) & (df['indicator'].shift(-local_max_window) < df['indicator'])
    df['is_ind_valley'] = (df['indicator'] == ind_roll.min()) & (df['indicator'].shift(-local_max_window) > df['indicator'])

    df['divergence'] = 0 # 1 = Bullish, -1 = Bearish

    # Identificar índices de picos e vales confirmados
    price_peaks_idx = df.index[df['is_price_peak']]
    price_valleys_idx = df.index[df['is_price_valley']]
    ind_peaks_idx = df.index[df['is_ind_peak']]
    ind_valleys_idx = df.index[df['is_ind_valley']]

    # --- Bearish Divergence (HH Price, LH Indicator) ---
    # Iterar sobre picos de preço
    for i in range(1, len(price_peaks_idx)):
        curr_p_peak_idx = price_peaks_idx[i]
        prev_p_peak_idx = price_peaks_idx[i-1]

        # Verificar se preço fez Higher High
        if df.loc[curr_p_peak_idx, 'price'] > df.loc[prev_p_peak_idx, 'price']:
            # Encontrar picos correspondentes do indicador entre os picos de preço
            ind_peaks_between = ind_peaks_idx[(ind_peaks_idx >= prev_p_peak_idx) & (ind_peaks_idx <= curr_p_peak_idx)]
            if len(ind_peaks_between) >= 2:
                # Pegar o último pico do indicador antes ou no pico atual do preço
                curr_i_peak_idx = ind_peaks_between[-1]
                # Pegar o último pico do indicador antes ou no pico anterior do preço
                prev_i_peak_idx = ind_peaks_between[0] if ind_peaks_between[0] >= prev_p_peak_idx else (ind_peaks_idx[ind_peaks_idx < prev_p_peak_idx].max() if any(ind_peaks_idx < prev_p_peak_idx) else None)

                if prev_i_peak_idx is not None and df.loc[curr_i_peak_idx, 'indicator'] < df.loc[prev_i_peak_idx, 'indicator']:
                    df.loc[curr_p_peak_idx, 'divergence'] = -1 # Bearish Divergence

    # --- Bullish Divergence (LL Price, HL Indicator) ---
    # Iterar sobre vales de preço
    for i in range(1, len(price_valleys_idx)):
        curr_p_valley_idx = price_valleys_idx[i]
        prev_p_valley_idx = price_valleys_idx[i-1]

        # Verificar se preço fez Lower Low
        if df.loc[curr_p_valley_idx, 'price'] < df.loc[prev_p_valley_idx, 'price']:
            # Encontrar vales correspondentes do indicador entre os vales de preço
            ind_valleys_between = ind_valleys_idx[(ind_valleys_idx >= prev_p_valley_idx) & (ind_valleys_idx <= curr_p_valley_idx)]
            if len(ind_valleys_between) >= 2:
                curr_i_valley_idx = ind_valleys_between[-1]
                prev_i_valley_idx = ind_valleys_between[0] if ind_valleys_between[0] >= prev_p_valley_idx else (ind_valleys_idx[ind_valleys_idx < prev_p_valley_idx].max() if any(ind_valleys_idx < prev_p_valley_idx) else None)

                if prev_i_valley_idx is not None and df.loc[curr_i_valley_idx, 'indicator'] > df.loc[prev_i_valley_idx, 'indicator']:
                     df.loc[curr_p_valley_idx, 'divergence'] = 1 # Bullish Divergence

    # Reindexar para garantir que o resultado corresponda ao índice original da série de preços
    return df[['divergence']].reindex(price_series.index).fillna(0)


def get_exchange_flows():
    """Retorna dados SIMULADOS de fluxo de exchanges."""
    # Idealmente, buscaria de APIs (Glassnode, CryptoQuant, etc.)
    exchanges = ["Binance", "Coinbase Pro", "Kraken", "Bybit", "OKX", "Huobi"]
    inflows = np.random.normal(2000, 1500, size=len(exchanges)).clip(0) # Média 2k, desvio 1.5k
    outflows = np.random.normal(2100, 1600, size=len(exchanges)).clip(0) # Média 2.1k, desvio 1.6k
    netflows = inflows - outflows
    # Adicionar reservas simuladas
    reserves = np.random.normal(300000, 150000, size=len(exchanges)).clip(50000)
    return pd.DataFrame({
        'Exchange': exchanges,
        'Entrada (BTC - 24h Sim.)': inflows,
        'Saída (BTC - 24h Sim.)': outflows,
        'Líquido (BTC - 24h Sim.)': netflows,
        'Reservas (BTC Sim.)': reserves
    })

def plot_hashrate_difficulty(data):
    """Cria gráfico combinado de hashrate e dificuldade."""
    if 'hashrate' not in data or 'difficulty' not in data or \
       data['hashrate'].empty or data['difficulty'].empty:
        return None

    fig = go.Figure()

    # Hashrate (TH/s)
    fig.add_trace(go.Scatter(
        x=data['hashrate']['date'], y=data['hashrate']['y'],
        name="Hashrate (TH/s)", mode='lines', line=dict(color='blue')
    ))

    # Dificuldade (Escala Log)
    fig.add_trace(go.Scatter(
        x=data['difficulty']['date'], y=data['difficulty']['y'],
        name="Dificuldade", yaxis="y2", mode='lines', line=dict(color='red')
    ))

    fig.update_layout(
        title="Hashrate vs Dificuldade",
        yaxis=dict(title="Hashrate (TH/s)", color='blue'),
        yaxis2=dict(title="Dificuldade", overlaying="y", side="right", color='red', type='log'), # Escala log
        hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig


def plot_whale_activity(data):
    """Mostra atividade SIMULADA de whales."""
    if 'whale_alert' not in data or data['whale_alert'].empty:
        return None

    df_whale = data['whale_alert'].copy()
    # Simular direção e tipo de forma mais realista
    df_whale['direction'] = np.random.choice(['inflow', 'outflow', 'transfer_wallet', 'transfer_exchange'],
                                             size=len(df_whale), p=[0.3, 0.3, 0.2, 0.2])
    df_whale['color'] = df_whale['direction'].map({'inflow': 'green', 'outflow': 'red',
                                                 'transfer_wallet': 'grey', 'transfer_exchange': 'orange'})

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_whale['date'], y=df_whale['amount'],
        name="Volume Transação Whale", marker_color=df_whale['color'],
        text=df_whale.apply(lambda r: f"{r['amount']} BTC ({r['direction']} - {r['exchange']})", axis=1),
        hoverinfo='text'
    ))

    fig.update_layout(
        title="Atividade Recente Simulada de Whales",
        xaxis_title="Data", yaxis_title="Quantidade (BTC)",
        hovermode="x", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def simulate_event(event, price_series):
    """Simula impacto de eventos no preço com base em fatores multiplicativos."""
    if not isinstance(price_series, pd.Series) or price_series.empty:
        st.warning("Série de preços inválida para simulação.")
        return price_series # Retorna original ou vazia

    clean_prices = price_series.dropna()
    if clean_prices.empty:
        return price_series

    simulated = clean_prices.copy()
    n_days = len(simulated)

    try:
        if event == "Halving (Pós)": # Aumento gradual pós-halving
             # Fator de crescimento logarítmico simulado ao longo do período
             growth_factor = 1 + 0.5 * np.log1p(np.linspace(0, 5, n_days)) / np.log1p(5) # Aumento ~50% log
             simulated = simulated * growth_factor
        elif event == "Crash (Súbito)": # Queda acentuada
             simulated = simulated * np.random.uniform(0.65, 0.80) # Queda 20-35%
        elif event == "ETF Approval (Impacto)": # Salto inicial
             simulated = simulated * np.random.uniform(1.15, 1.35) # Aumento 15-35%
        elif event == "Normal": # Sem evento, retorna original
            pass # Nenhuma mudança
        else:
            st.warning(f"Evento '{event}' não reconhecido para simulação.")
            return price_series.copy() # Retorna original

        # Reindexar para garantir o mesmo índice da entrada
        return simulated.reindex(price_series.index)

    except Exception as e:
        st.error(f"Erro na simulação do evento {event}: {str(e)}")
        return price_series.copy() # Retorna original em caso de erro


def get_market_sentiment():
    """Coleta dados de sentimento do Fear & Greed Index."""
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1&format=json", timeout=10)
        response.raise_for_status()
        data = response.json()
        if "data" in data and len(data["data"]) > 0:
            value = int(data["data"][0]["value"])
            sentiment = data["data"][0]["value_classification"]
            return {"value": value, "sentiment": sentiment}
        else:
            st.warning("Formato inesperado da API Fear & Greed.")
            return {"value": 50, "sentiment": "Neutral (API Format Error)"}
    except requests.exceptions.Timeout:
        st.warning("Timeout ao buscar Fear & Greed Index.")
        return {"value": 50, "sentiment": "Neutral (Timeout)"}
    except requests.exceptions.RequestException as e:
        st.warning(f"Erro na API Fear & Greed: {e}")
        return {"value": 50, "sentiment": f"Neutral (API Error)"}
    except Exception as e:
        st.warning(f"Erro ao processar Fear & Greed Index: {e}")
        return {"value": 50, "sentiment": "Neutral (Processing Error)"}


def get_traditional_assets():
    """Coleta dados de ativos tradicionais via Yahoo Finance."""
    assets = {
        "BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD",
        "S&P 500": "^GSPC", "NASDAQ": "^IXIC",
        "Ouro": "GC=F", "Petróleo WTI": "CL=F", "USD Index": "DX-Y.NYB"
    }
    dfs = []
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=120) # Pegar ~4 meses para garantir 90 dias úteis

    for name, ticker in assets.items():
        try:
            # Usar yf.download para mais robustez
            data = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
            if not data.empty:
                data = data.reset_index()
                # Garantir coluna 'Date' e renomear 'Close'
                if 'Date' not in data.columns and 'Datetime' in data.columns:
                     data = data.rename(columns={'Datetime': 'Date'})
                if 'Date' in data.columns and 'Close' in data.columns:
                    data['Date'] = pd.to_datetime(data['Date'])
                    # Filtrar pelo período exato após download
                    data = data[(data['Date'] >= pd.to_datetime(start_date)) & (data['Date'] < pd.to_datetime(end_date))]
                    data = data[['Date', 'Close']].rename(columns={'Close': 'value', 'Date': 'date'})
                    data['asset'] = name
                    data = data.dropna()
                    if not data.empty:
                        dfs.append(data)
                else:
                     st.warning(f"Colunas 'Date' ou 'Close' não encontradas para {name} ({ticker}).")

        except Exception as e:
            # Não mostrar aviso para cada falha, apenas logar talvez
            # st.warning(f"Falha ao buscar {name} ({ticker}): {e}")
            pass # Silencioso para não poluir

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def filter_news_by_confidence(analyzed_news_data, min_confidence=0.7):
    """Filtra notícias *já analisadas* pela confiança do sentimento."""
    if not analyzed_news_data or not isinstance(analyzed_news_data, list):
        return []

    return [
        news for news in analyzed_news_data
        if isinstance(news, dict) and # Garantir que é um dicionário
           news.get('sentiment') not in ['MODEL_ERROR', 'BATCH_ERROR', 'ANALYSIS_SKIPPED'] and # Excluir erros
           news.get('sentiment_score', 0.0) >= min_confidence # Filtrar pela confiança do sentimento
    ]


def calculate_daily_returns(df):
    """Calcula retornos diários e cumulativos."""
    if df is None or df.empty or 'price' not in df.columns:
        return df

    df_copy = df.copy()
    df_copy['daily_return'] = df_copy['price'].pct_change().fillna(0)
    df_copy['cumulative_return'] = (1 + df_copy['daily_return']).cumprod()
    return df_copy


def calculate_strategy_returns(df, signal_col='signal'):
    """Calcula retornos da estratégia baseados na coluna de sinal."""
    if df is None or df.empty or 'daily_return' not in df.columns or signal_col not in df.columns:
        if df is not None: # Adicionar colunas vazias se df existir mas faltar algo
             df['strategy_return'] = 0.0
             df['strategy_cumulative'] = 1.0
        return df

    df_copy = df.copy()
    # Sinal de hoje afeta retorno de amanhã; usar shift(1)
    df_copy['strategy_return'] = df_copy[signal_col].shift(1).fillna(0) * df_copy['daily_return']
    df_copy['strategy_return'] = df_copy['strategy_return'].fillna(0) # Garantir sem NaNs
    df_copy['strategy_cumulative'] = (1 + df_copy['strategy_return']).cumprod()
    return df_copy


# ============================================
# FUNÇÕES DE BACKTESTING E OTIMIZAÇÃO
# ============================================
# (Funções backtest_... mantidas como na resposta anterior - já robustas)

def backtest_rsi_strategy(df, rsi_window=14, overbought=70, oversold=30):
    """Estratégia RSI aprimorada com verificações robustas"""
    if df is None or df.empty or 'price' not in df.columns:
        return pd.DataFrame() # Retorna DataFrame vazio se dados inválidos

    df_res = df.copy()

    # Calcular ou verificar MAs necessárias (ex: MA30 para filtro de tendência)
    if 'MA30' not in df_res.columns:
        df_res['MA30'] = calculate_ema(df_res['price'], 30) # Usar EMA pode ser mais reativo

    # Calcular RSI
    rsi_col = f'RSI_{rsi_window}'
    df_res[rsi_col] = calculate_rsi(df_res['price'], rsi_window)

    # Inicializar sinal com 0 (Hold)
    df_res['signal'] = 0.0 # Usar float para pesos

    # Condições de Compra/Venda com pesos e filtro MA
    # Considerar usar EMA(30) para filtro
    buy_condition = (df_res[rsi_col] < oversold) #& (df_res['price'] > df_res['MA30']) # Filtro opcional
    sell_condition = (df_res[rsi_col] > overbought) #& (df_res['price'] < df_res['MA30']) # Filtro opcional

    weight = INDICATOR_WEIGHTS.get('rsi', 1.0)
    df_res.loc[buy_condition, 'signal'] = weight
    df_res.loc[sell_condition, 'signal'] = -weight

    # Calcular retornos
    df_res = calculate_daily_returns(df_res)
    df_res = calculate_strategy_returns(df_res)

    return df_res


def backtest_macd_strategy(df, fast=12, slow=26, signal=9):
    """Estratégia MACD com tratamento robusto (baseada em cruzamentos)."""
    if df is None or df.empty or 'price' not in df.columns:
        return pd.DataFrame()

    df_res = df.copy()
    # Usar função que retorna MACD, Sinal e Histograma
    df_res['MACD'], df_res['MACD_Signal'], df_res['MACD_Hist'] = calculate_macd(df_res['price'], fast, slow, signal)

    df_res['signal'] = 0.0 # Float para pesos
    weight = INDICATOR_WEIGHTS.get('macd', 1.0)

    # Sinal baseado no cruzamento da linha MACD com a linha de Sinal
    macd_cross_up = (df_res['MACD'] > df_res['MACD_Signal']) & (df_res['MACD'].shift(1) <= df_res['MACD_Signal'].shift(1))
    macd_cross_down = (df_res['MACD'] < df_res['MACD_Signal']) & (df_res['MACD'].shift(1) >= df_res['MACD_Signal'].shift(1))

    # Sinal baseado no cruzamento do Histograma com zero
    # hist_cross_up = (df_res['MACD_Hist'] > 0) & (df_res['MACD_Hist'].shift(1) <= 0)
    # hist_cross_down = (df_res['MACD_Hist'] < 0) & (df_res['MACD_Hist'].shift(1) >= 0)

    # Usar cruzamento MACD vs Sinal como gatilho principal
    df_res.loc[macd_cross_up, 'signal'] = weight
    df_res.loc[macd_cross_down, 'signal'] = -weight

    # Manter posição entre cruzamentos (opcional)
    # df_res['signal'] = df_res['signal'].replace(0, method='ffill').fillna(0)

    df_res = calculate_daily_returns(df_res)
    df_res = calculate_strategy_returns(df_res)
    return df_res


def backtest_bollinger_strategy(df, window=20, num_std=2):
    """Estratégia Bandas de Bollinger robusta"""
    if df is None or df.empty or 'price' not in df.columns:
        return pd.DataFrame()

    df_res = df.copy()
    bb_upper_col = f'BB_Upper_{window}'
    bb_lower_col = f'BB_Lower_{window}'
    bb_ma_col = f'BB_MA_{window}'

    # Calcular Bandas de Bollinger
    df_res[bb_upper_col], df_res[bb_lower_col], df_res[bb_ma_col] = calculate_bollinger_bands(df_res['price'], window, num_std)

    df_res['signal'] = 0.0 # Float para pesos
    weight = INDICATOR_WEIGHTS.get('bollinger', 1.0)

    # Condições de Compra/Venda no toque das bandas
    buy_condition = df_res['price'] < df_res[bb_lower_col]
    sell_condition = df_res['price'] > df_res[bb_upper_col]
    # Condição de saída (ex: voltar para a média)
    exit_condition = ((df_res['price'] > df_res[bb_ma_col]) & (df_res['signal'].shift(1) > 0)) | \
                     ((df_res['price'] < df_res[bb_ma_col]) & (df_res['signal'].shift(1) < 0))

    # Aplicar sinais
    df_res['signal'] = np.where(buy_condition, weight, df_res['signal']) # Compra sobrepõe outros?
    df_res['signal'] = np.where(sell_condition, -weight, df_res['signal']) # Venda sobrepõe?
    df_res['signal'] = np.where(exit_condition, 0.0, df_res['signal']) # Saída tem prioridade

    # Manter posição (opcional)
    # df_res['signal'] = df_res['signal'].replace(0, method='ffill').fillna(0)

    df_res = calculate_daily_returns(df_res)
    df_res = calculate_strategy_returns(df_res)
    return df_res


def backtest_ema_cross_strategy(df, short_window=9, long_window=21):
    """Estratégia EMA Cross com verificações"""
    if df is None or df.empty or 'price' not in df.columns:
        return pd.DataFrame()
    if short_window >= long_window: return pd.DataFrame() # Inválido

    df_res = df.copy()
    ema_short_col = f'EMA_{short_window}'
    ema_long_col = f'EMA_{long_window}'

    df_res[ema_short_col] = calculate_ema(df_res['price'], short_window)
    df_res[ema_long_col] = calculate_ema(df_res['price'], long_window)

    df_res['signal'] = 0.0 # Float para pesos
    weight = INDICATOR_WEIGHTS.get('ma_cross', 1.0)

    # Condições de Cruzamento
    cross_up = (df_res[ema_short_col] > df_res[ema_long_col]) & (df_res[ema_short_col].shift(1) <= df_res[ema_long_col].shift(1))
    cross_down = (df_res[ema_short_col] < df_res[ema_long_col]) & (df_res[ema_short_col].shift(1) >= df_res[ema_long_col].shift(1))

    df_res.loc[cross_up, 'signal'] = weight
    df_res.loc[cross_down, 'signal'] = -weight

    # Manter posição
    df_res['signal'] = df_res['signal'].replace(0, method='ffill').fillna(0)

    df_res = calculate_daily_returns(df_res)
    df_res = calculate_strategy_returns(df_res)
    return df_res


def backtest_volume_strategy(df, volume_window=20, threshold=1.5):
    """Estratégia baseada em volume"""
    if df is None or df.empty or 'price' not in df.columns or 'volume' not in df.columns:
        return pd.DataFrame()

    df_res = df.copy()
    vol_ma_col = f'Volume_MA{volume_window}'
    df_res[vol_ma_col] = df_res['volume'].rolling(volume_window, min_periods=volume_window).mean()

    # Evitar divisão por zero
    vol_ma_safe = df_res[vol_ma_col].replace(0, 1e-10) # Usar valor pequeno
    df_res['Volume_Ratio'] = df_res['volume'] / vol_ma_safe

    df_res['signal'] = 0.0
    weight = INDICATOR_WEIGHTS.get('volume', 1.0)
    price_change = df_res['price'].diff().fillna(0)

    # Condições: Volume alto E movimento de preço na mesma direção
    buy_condition = (df_res['Volume_Ratio'] > threshold) & (price_change > 0)
    sell_condition = (df_res['Volume_Ratio'] > threshold) & (price_change < 0)

    df_res.loc[buy_condition, 'signal'] = weight
    df_res.loc[sell_condition, 'signal'] = -weight

    # Sem manter posição, sinal apenas no dia do evento
    df_res = calculate_daily_returns(df_res)
    df_res = calculate_strategy_returns(df_res)
    return df_res


def backtest_obv_strategy(df, obv_window=20, price_window=30):
    """Estratégia baseada em OBV e sua MA, com filtro de MA de preço."""
    if df is None or df.empty or 'price' not in df.columns or 'volume' not in df.columns:
        return pd.DataFrame()

    df_res = df.copy()
    df_res['OBV'] = calculate_obv(df_res['price'], df_res['volume'])

    obv_ma_col = f'OBV_MA{obv_window}'
    price_ma_col = f'Price_EMA{price_window}' # Usar EMA para preço
    df_res[obv_ma_col] = df_res['OBV'].rolling(obv_window, min_periods=obv_window).mean()
    df_res[price_ma_col] = calculate_ema(df_res['price'], price_window)

    df_res['signal'] = 0.0
    weight = INDICATOR_WEIGHTS.get('obv', 1.0)

    # Condições: Alinhamento da tendência do OBV e do Preço
    buy_condition = (df_res['OBV'] > df_res[obv_ma_col]) & (df_res['price'] > df_res[price_ma_col])
    sell_condition = (df_res['OBV'] < df_res[obv_ma_col]) & (df_res['price'] < df_res[price_ma_col])

    # Aplicar sinais apenas quando as condições são verdadeiras
    df_res['signal'] = np.where(buy_condition, weight, 0.0) # Resetar para 0 se não for compra
    df_res['signal'] = np.where(sell_condition, -weight, df_res['signal']) # Venda sobrepõe compra se ambas verdadeiras?

    # Manter posição (opcional)
    df_res['signal'] = df_res['signal'].replace(0, method='ffill').fillna(0)

    df_res = calculate_daily_returns(df_res)
    df_res = calculate_strategy_returns(df_res)
    return df_res


def backtest_stochastic_strategy(df, k_window=14, d_window=3, overbought=80, oversold=20):
    """Estratégia baseada em Stochastic (cruzamentos K/D nas zonas)."""
    if df is None or df.empty or 'price' not in df.columns:
        return pd.DataFrame()

    df_res = df.copy()
    stoch_k_col = f'Stoch_K_{k_window}_{d_window}'
    stoch_d_col = f'Stoch_D_{k_window}_{d_window}'
    df_res[stoch_k_col], df_res[stoch_d_col] = calculate_stochastic(df_res['price'], k_window, d_window)

    df_res['signal'] = 0.0
    weight = INDICATOR_WEIGHTS.get('stochastic', 1.0)

    # Condições de Cruzamento K sobre D
    cross_up = (df_res[stoch_k_col] > df_res[stoch_d_col]) & (df_res[stoch_k_col].shift(1) <= df_res[stoch_d_col].shift(1))
    cross_down = (df_res[stoch_k_col] < df_res[stoch_d_col]) & (df_res[stoch_k_col].shift(1) >= df_res[stoch_d_col].shift(1))

    # Sinais de compra/venda apenas quando cruzamento ocorre *dentro* das zonas extremas
    buy_signal = cross_up & (df_res[stoch_k_col] < oversold) # Cruzou para cima abaixo de oversold
    sell_signal = cross_down & (df_res[stoch_k_col] > overbought) # Cruzou para baixo acima de overbought

    df_res.loc[buy_signal, 'signal'] = weight
    df_res.loc[sell_signal, 'signal'] = -weight

    # Manter posição
    df_res['signal'] = df_res['signal'].replace(0, method='ffill').fillna(0)

    df_res = calculate_daily_returns(df_res)
    df_res = calculate_strategy_returns(df_res)
    return df_res


def backtest_gp_strategy(df, window=30, lookahead=5, threshold=0.03):
    """Estratégia baseada em Regressão de Processo Gaussiano (GP)."""
    # Lento para backtesting longo.
    if df is None or df.empty or 'price' not in df.columns:
        return pd.DataFrame()

    df_res = df.copy()
    gp_pred_col = f'GP_Pred_{window}_{lookahead}'
    df_res[gp_pred_col] = calculate_gaussian_process(df_res['price'], window, lookahead)

    df_res['signal'] = 0.0
    weight = INDICATOR_WEIGHTS.get('gaussian_process', 1.0)

    # Usar previsão do dia anterior (shift(1)) para evitar lookahead bias
    buy_condition = df_res[gp_pred_col].shift(1) > df_res['price'] * (1 + threshold)
    sell_condition = df_res[gp_pred_col].shift(1) < df_res['price'] * (1 - threshold)

    df_res.loc[buy_condition, 'signal'] = weight
    df_res.loc[sell_condition, 'signal'] = -weight

    # Sem manter posição, sinal apenas no dia
    df_res = calculate_daily_returns(df_res)
    df_res = calculate_strategy_returns(df_res)
    return df_res


def backtest_order_block_strategy(df, swing_length=11, use_body=True):
    """Estratégia de backtesting para Order Blocks."""
    if df is None or df.empty or 'price' not in df.columns:
        return pd.DataFrame()

    df_res = df.copy().reset_index() # Resetar índice para merge/lookup fácil
    df_ob, blocks = identify_order_blocks(df_res, swing_length=swing_length, use_body=use_body,
                                          show_bull=50, show_bear=50) # Pegar muitos blocos para backtest

    df_res['signal'] = 0.0

    if not blocks: # Se nenhum bloco for encontrado
        df_res = calculate_daily_returns(df_res)
        df_res = calculate_strategy_returns(df_res)
        return df_res.set_index('date') if 'date' in df_res.columns else df_res

    # Iterar pelos dias do DataFrame para aplicar sinais
    last_signal = 0
    for i in range(1, len(df_res)): # Começar do segundo dia
        current_date = df_res.loc[i, 'date']
        current_low = df_res.loc[i, 'low']
        current_high = df_res.loc[i, 'high']
        current_signal = 0 # Sinal para o dia atual

        # Verificar blocos relevantes (formados antes do dia atual)
        relevant_blocks = [b for b in blocks if b['date'] < current_date]

        for block in relevant_blocks:
            # Verificar se o preço atual tocou o range do bloco
            touched = (current_low <= block['high']) and (current_high >= block['low'])

            if touched:
                if not block['broken']:
                    if block['type'] == 'bullish_ob': current_signal = max(current_signal, block['weight']) # Compra
                    elif block['type'] == 'bearish_ob': current_signal = min(current_signal, -block['weight']) # Venda
                else: # Breaker
                    if block['breaker_type'] == 'bullish_breaker': current_signal = min(current_signal, -block['weight']) # Venda (Resistência)
                    elif block['breaker_type'] == 'bearish_breaker': current_signal = max(current_signal, block['weight']) # Compra (Suporte)

        # Aplicar sinal (pode priorizar venda ou compra se ambos ocorrerem)
        # Manter posição do dia anterior se nenhum sinal novo
        df_res.loc[i, 'signal'] = current_signal if current_signal != 0 else last_signal
        last_signal = df_res.loc[i, 'signal']


    # Restaurar índice e calcular retornos
    if 'date' in df_res.columns: df_res = df_res.set_index('date')
    df_res = calculate_daily_returns(df_res)
    df_res = calculate_strategy_returns(df_res)
    return df_res


def calculate_metrics(df):
    """Calcula métricas de performance de forma robusta."""
    metrics = {
        'Retorno Estratégia': 0.0, 'Retorno Buy & Hold': 0.0,
        'Vol Estratégia': 0.0, 'Vol Buy & Hold': 0.0,
        'Sharpe Estratégia': 0.0, 'Sharpe Buy & Hold': 0.0,
        'Max Drawdown': 0.0, 'Taxa Acerto': 0.0, 'Num Trades': 0
    }

    if df is None or df.empty or 'strategy_cumulative' not in df.columns or \
       'cumulative_return' not in df.columns or 'strategy_return' not in df.columns or \
       'daily_return' not in df.columns:
        return metrics # Retorna defaults se dados incompletos

    # Usar iloc[-1] com dropna() para segurança
    last_strat_cum = df['strategy_cumulative'].dropna().iloc[-1] if not df['strategy_cumulative'].dropna().empty else 1.0
    last_bh_cum = df['cumulative_return'].dropna().iloc[-1] if not df['cumulative_return'].dropna().empty else 1.0

    metrics['Retorno Estratégia'] = last_strat_cum - 1
    metrics['Retorno Buy & Hold'] = last_bh_cum - 1

    returns_strat = df['strategy_return'].dropna()
    returns_bh = df['daily_return'].dropna()

    if len(returns_strat) > 1:
        metrics['Vol Estratégia'] = returns_strat.std() * np.sqrt(365)
        mean_return_ann = returns_strat.mean() * 365
        vol_ann = metrics['Vol Estratégia']
        metrics['Sharpe Estratégia'] = mean_return_ann / vol_ann if vol_ann != 0 else 0.0
        # Drawdown
        cum_returns_strat = (1 + returns_strat).cumprod()
        peak_strat = cum_returns_strat.expanding(min_periods=1).max()
        drawdown_strat = (cum_returns_strat / peak_strat) - 1 # Correção no cálculo
        metrics['Max Drawdown'] = drawdown_strat.min() if not drawdown_strat.empty else 0.0

    if len(returns_bh) > 1:
        metrics['Vol Buy & Hold'] = returns_bh.std() * np.sqrt(365)
        mean_bh_ann = returns_bh.mean() * 365
        vol_bh_ann = metrics['Vol Buy & Hold']
        metrics['Sharpe Buy & Hold'] = mean_bh_ann / vol_bh_ann if vol_bh_ann != 0 else 0.0

    # Taxa de Acerto por Trade
    if 'signal' in df.columns:
        # Identificar mudanças de posição (entrada/saída/reversão)
        signal_shifted = df['signal'].shift(1).fillna(0)
        trades = df[(df['signal'] != signal_shifted) & (signal_shifted != 0)] # Entradas/Saídas/Reversões
        entry_points = df[(df['signal'] != 0) & (signal_shifted == 0)] # Apenas entradas
        metrics['Num Trades'] = len(entry_points)

        # Calcular Win Rate baseado no retorno *após* a entrada (simplificado)
        # Uma métrica melhor exigiria rastrear cada trade individualmente
        if metrics['Num Trades'] > 0:
             # Verificar retorno no dia seguinte à entrada
             win_days_after_entry = df['strategy_return'].shift(-1).loc[entry_points.index] > 0
             metrics['Taxa Acerto'] = win_days_after_entry.sum() / metrics['Num Trades'] if metrics['Num Trades'] > 0 else 0.0

    # Arredondar
    for key in metrics:
        if isinstance(metrics[key], (float, np.floating)):
            metrics[key] = round(metrics[key], 4)

    return metrics


def optimize_strategy_parameters(data_dict, strategy_name, param_space):
    """Otimização de parâmetros de estratégia usando GridSearch."""
    best_sharpe = -np.inf
    best_params = None

    if not isinstance(data_dict, dict) or 'prices' not in data_dict or data_dict['prices'].empty:
        st.error("Dados inválidos para otimização.")
        return best_params, best_sharpe, None

    df_prices = data_dict['prices'] # Usar DataFrame passado no dicionário
    param_combinations = list(ParameterGrid(param_space))
    if not param_combinations: return best_params, best_sharpe, None

    total_combinations = len(param_combinations)
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_list = []

    for i, params in enumerate(param_combinations):
        df_test = None
        try:
            # Chamar backtest correspondente
            if strategy_name == 'RSI': df_test = backtest_rsi_strategy(df_prices, **params)
            elif strategy_name == 'MACD': df_test = backtest_macd_strategy(df_prices, **params)
            elif strategy_name == 'Bollinger': df_test = backtest_bollinger_strategy(df_prices, **params)
            elif strategy_name == 'EMA Cross':
                if params.get('short_window', 1) >= params.get('long_window', 0): continue
                df_test = backtest_ema_cross_strategy(df_prices, **params)
            elif strategy_name == 'Volume': df_test = backtest_volume_strategy(df_prices, **params)
            elif strategy_name == 'OBV': df_test = backtest_obv_strategy(df_prices, **params)
            elif strategy_name == 'Stochastic': df_test = backtest_stochastic_strategy(df_prices, **params)
            # elif strategy_name == 'Gaussian Process': df_test = backtest_gp_strategy(df_prices, **params) # Lento demais
            elif strategy_name == 'Order Blocks': df_test = backtest_order_block_strategy(df_prices, **params)
            else: continue

            if df_test is not None and not df_test.empty and 'strategy_return' in df_test.columns:
                metrics = calculate_metrics(df_test)
                sharpe = metrics.get('Sharpe Estratégia', -np.inf)
                results_list.append({'params': params, 'sharpe': sharpe, 'return': metrics.get('Retorno Estratégia', 0)})

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params

            # Atualizar UI de progresso
            if (i + 1) % max(1, total_combinations // 20) == 0 or (i + 1) == total_combinations: # Atualizar ~20 vezes
                 progress = (i + 1) / total_combinations
                 progress_bar.progress(progress)
                 status_text.text(f"Testando {i+1}/{total_combinations} | Melhor Sharpe: {best_sharpe:.3f}")

        except Exception as e:
            # st.warning(f"Erro otimizando {params}: {e}") # Log verboso opcional
            continue

    progress_bar.empty()
    status_text.empty()

    if best_params:
         st.success(f"Otimização Concluída. Melhor Sharpe: {best_sharpe:.3f}")
         # Opcional: Mostrar top resultados
         # top_results = sorted(results_list, key=lambda x: x['sharpe'], reverse=True)
         # st.dataframe(pd.DataFrame(top_results[:5]))
    else:
         st.warning("Nenhum parâmetro válido encontrado durante a otimização.")

    return best_params, best_sharpe, None # Não retornar DataFrame


# ============================================
# CARREGAMENTO DE DADOS E GERAÇÃO DE SINAIS
# ============================================

@st.cache_data(ttl=1800, show_spinner="Carregando dados de mercado...") # Cache de 30 min
def load_cached_data():
    """Carrega e processa dados da API, aplicando cache."""
    return load_data_from_api()

def load_data_from_api():
    """Busca e processa dados de mercado (OHLCV, Indicadores, etc.)."""
    data = {'prices': pd.DataFrame(), 'prices_full': pd.DataFrame()} # Iniciar com DFs vazios
    try:
        # 1. Buscar Histórico OHLCV do Yahoo Finance (Fonte Primária)
        ticker = "BTC-USD"
        hist = yf.download(ticker, period="2y", interval="1d", progress=False) # Pegar 2 anos

        if hist.empty:
            raise ValueError(f"Yahoo Finance não retornou dados para {ticker}.")

        hist = hist.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'price', 'Volume': 'volume', 'Adj Close': 'adj_close'
        })
        hist = hist[['open', 'high', 'low', 'price', 'volume']]
        hist = hist[hist.index.tz_localize(None) < datetime.now().replace(tzinfo=None)] # Remover dia atual incompleto, ignorar timezone
        hist['date'] = pd.to_datetime(hist.index) # Adicionar coluna 'date'

        # 2. Calcular Indicadores no Histórico Completo
        data['prices_full'] = hist.copy()
        price_full = data['prices_full']['price']
        vol_full = data['prices_full']['volume']

        # EMAs
        for window in [7, 14, 20, 30, 50, 100, 200]:
             data['prices_full'][f'MA{window}'] = calculate_ema(price_full, window)
        # RSI
        for window in [6, 12, 14, 24]:
             data['prices_full'][f'RSI_{window}'] = calculate_rsi(price_full, window)
        # MACD
        macd, macd_s, macd_h = calculate_macd(price_full)
        data['prices_full']['MACD'], data['prices_full']['MACD_Signal'], data['prices_full']['MACD_Hist'] = macd, macd_s, macd_h
        # Bollinger Bands
        bb_up, bb_low, bb_ma = calculate_bollinger_bands(price_full, window=20)
        data['prices_full']['BB_Upper_20'], data['prices_full']['BB_Lower_20'], data['prices_full']['BB_MA_20'] = bb_up, bb_low, bb_ma
        bb_bw, bb_pctb = calculate_bollinger_bandwidth_pctb(price_full, bb_up, bb_low, bb_ma)
        data['prices_full']['BB_Bandwidth_20'], data['prices_full']['BB_PctB_20'] = bb_bw, bb_pctb
        # OBV
        data['prices_full']['OBV'] = calculate_obv(price_full, vol_full)
        # Stochastic
        k, d = calculate_stochastic(price_full)
        data['prices_full']['Stoch_K'], data['prices_full']['Stoch_D'] = k, d

        # 3. Preparar Dados Recentes para UI e Cálculos Rápidos
        data['prices'] = data['prices_full'].tail(180).copy() # Pegar últimos ~6 meses para UI
        price_recent = data['prices']['price']

        # Calcular Indicadores que dependem de janela menor ou são caros (GP, Divergência, S/R) nos dados recentes
        data['prices']['GP_Prediction'] = calculate_gaussian_process(
            price_recent, window=DEFAULT_SETTINGS['gp_window'], lookahead=DEFAULT_SETTINGS['gp_lookahead']
        )
        # Reindexar RSI calculado no full para dados recentes
        rsi14_recent = data['prices_full']['RSI_14'].reindex(data['prices'].index)
        data['prices']['RSI_Divergence'] = detect_divergences(price_recent, rsi14_recent)

        # Calcular S/R Clusters nos últimos 90 dias
        support_resistance = detect_support_resistance_clusters(
            price_recent.tail(90).values, # Array numpy
            n_clusters=st.session_state.user_settings.get('n_clusters', 5)
        )
        data['support_resistance'] = support_resistance

        # Calcular Variação 24h (último dia vs penúltimo)
        if len(data['prices']) >= 2:
            last_price = data['prices']['price'].iloc[-1]
            prev_price = data['prices']['price'].iloc[-2]
            data['24h_change'] = ((last_price / prev_price) - 1) * 100 if prev_price else 0.0
        else: data['24h_change'] = 0.0

        # 4. Buscar/Simular Dados Externos/On-Chain
        # Hashrate e Dificuldade (API Blockchain.info)
        try:
            hr_response = requests.get("https://api.blockchain.info/charts/hash-rate?format=json&timespan=6months", timeout=10)
            hr_response.raise_for_status()
            hr_data = pd.DataFrame(hr_response.json()["values"])
            hr_data["date"] = pd.to_datetime(hr_data["x"], unit="s")
            hr_data['y'] = hr_data['y'] / 1e12 # TH/s
            data['hashrate'] = hr_data.dropna()
        except Exception: data['hashrate'] = pd.DataFrame()

        try:
            diff_response = requests.get("https://api.blockchain.info/charts/difficulty?timespan=1year&format=json", timeout=10)
            diff_response.raise_for_status()
            diff_data = pd.DataFrame(diff_response.json()["values"])
            diff_data["date"] = pd.to_datetime(diff_data["x"], unit="s")
            data['difficulty'] = diff_data.dropna()
        except Exception: data['difficulty'] = pd.DataFrame()

        # Fluxo Exchanges e Whales (Simulado)
        data['exchanges'] = get_exchange_flows() # Função simulada
        data['whale_alert'] = pd.DataFrame({ # Dados simulados
            "date": pd.date_range(end=datetime.now(), periods=15, freq='4H'),
            "amount": np.random.randint(50, 2000, 15),
            "exchange": np.random.choice(["Binance", "Coinbase", "Kraken", "Wallet", "Bybit", "OKX", "Unknown"], 15)
        })

        # Notícias (Simulado)
        data['news'] = [
             {"title": f"Bitcoin tenta romper {np.random.choice(['resistência', 'suporte'])} chave em ${np.random.randint(65000, 75000)}",
             "date": datetime.now() - timedelta(hours=np.random.randint(1, 6)), "source": "CryptoNews"},
             {"title": f"Análise On-Chain: {'Acúmulo por holders' if np.random.rand() > 0.4 else 'Fluxo para exchanges'} {'aumenta' if np.random.rand() > 0.3 else 'diminui'}",
             "date": datetime.now() - timedelta(hours=np.random.randint(6, 18)), "source": "Glassnode Insights"},
             {"title": f"Mercado de Opções mostra {'viés de alta' if np.random.rand() > 0.5 else 'cautela'} para BTC no curto prazo",
             "date": datetime.now() - timedelta(days=np.random.randint(0, 2)), "source": "Deribit Data"},
             {"title": f"{np.random.choice(['Regulador dos EUA', 'Banco Central Europeu'])} {'alerta sobre' if np.random.rand() > 0.6 else 'estuda regras para'} criptoativos",
             "date": datetime.now() - timedelta(days=np.random.randint(1, 4)), "source": "Reuters"},
        ]


    except requests.exceptions.RequestException as e:
        st.error(f"Erro na requisição à API externa: {e}")
        # Pode tentar retornar dados parcialmente carregados se o yfinance funcionou
        if data['prices'].empty and data['prices_full'].empty:
             data['message'] = f"Erro API: {e}" # Adiciona mensagem de erro
    except ValueError as e:
         st.error(f"Erro nos dados recebidos: {e}")
         data['message'] = f"Erro Dados: {e}"
    except Exception as e:
        st.error(f"Erro inesperado ao carregar/processar dados: {e}")
        import traceback
        st.error(traceback.format_exc()) # Log completo para depuração
        data['message'] = f"Erro Processamento: {e}"

    # Garantir que as chaves principais existem mesmo em caso de erro parcial
    for key in ['prices', 'prices_full', 'hashrate', 'difficulty', 'exchanges', 'whale_alert', 'news', 'support_resistance', '24h_change']:
        if key not in data:
             data[key] = pd.DataFrame() if 'price' in key else [] if key == 'news' or key == 'support_resistance' else None

    return data


def generate_signals(data, rsi_window=14, bb_window=20, ma_windows=[7, 30, 200]):
    """Gera sinais técnicos ponderados a partir dos dados processados."""
    signals = []
    buy_signals_count = 0
    sell_signals_count = 0

    if 'prices' not in data or data['prices'].empty:
        return signals, "➖ DADOS INDISP.", 0, 0

    df = data['prices'].copy() # Usar dados recentes
    if len(df) < 2: return signals, "➖ DADOS INSUF.", 0, 0 # Precisa de pelo menos 2 pontos

    last_row = df.iloc[-1]
    prev_row = df.iloc[-2] # Usado para cruzamentos e mudanças

    if pd.isna(last_row['price']): return signals, "➖ PREÇO INDISP.", 0, 0
    last_price = last_row['price']

    # --- Indicadores ---
    # Função auxiliar para adicionar sinal com segurança
    def add_signal(name, condition_buy, condition_sell, value_str, weight_key):
        signal = "NEUTRO"
        weight = INDICATOR_WEIGHTS.get(weight_key, 1.0)
        final_weight = 0.0
        if condition_buy:
             signal = "COMPRA"
             final_weight = weight
        elif condition_sell:
             signal = "VENDA"
             final_weight = -weight # Usar peso negativo para venda
        # Só adicionar se não for neutro? Ou sempre adicionar?
        if signal != "NEUTRO":
             signals.append((name, signal, value_str, final_weight)) # Guardar peso com sinal

    # 1. Médias Móveis (Preço vs MA)
    for window in ma_windows:
        ma_col = f'MA{window}'
        if ma_col in last_row and not pd.isna(last_row[ma_col]):
            ma_value = last_row[ma_col]
            add_signal(f"Preço vs EMA{window}", last_price > ma_value, last_price < ma_value,
                       f"{last_price/ma_value:.2f}x" if ma_value else "N/A", 'ma_cross')

    # 2. Cruzamento de MAs (Curta vs Longa)
    if len(ma_windows) >= 2:
        ma_short_col = f'MA{ma_windows[0]}'
        ma_long_col = f'MA{ma_windows[1]}'
        if all(c in last_row and not pd.isna(last_row[c]) for c in [ma_short_col, ma_long_col]):
             ma_short = last_row[ma_short_col]
             ma_long = last_row[ma_long_col]
             add_signal(f"EMA{ma_windows[0]} vs EMA{ma_windows[1]}", ma_short > ma_long, ma_short < ma_long,
                        f"{ma_short/ma_long:.3f}" if ma_long else "N/A", 'ma_cross')

    # 3. RSI
    rsi_col = f'RSI_{rsi_window}'
    if rsi_col in last_row and not pd.isna(last_row[rsi_col]):
        rsi = last_row[rsi_col]
        add_signal(f"RSI ({rsi_window})", rsi < 30, rsi > 70, f"{rsi:.1f}", 'rsi')

    # 4. MACD (Cruzamento Linha vs Sinal)
    if all(c in last_row and not pd.isna(last_row[c]) for c in ['MACD', 'MACD_Signal']) and \
       all(c in prev_row and not pd.isna(prev_row[c]) for c in ['MACD', 'MACD_Signal']):
        macd_c, sig_c = last_row['MACD'], last_row['MACD_Signal']
        macd_p, sig_p = prev_row['MACD'], prev_row['MACD_Signal']
        cross_up = macd_c > sig_c and macd_p <= sig_p
        cross_down = macd_c < sig_c and macd_p >= sig_p
        add_signal("MACD Cross", cross_up, cross_down, f"{macd_c:.1f} vs {sig_c:.1f}", 'macd')

    # 5. Bollinger Bands (Toque nas bandas)
    bb_up_col = f'BB_Upper_{bb_window}'
    bb_low_col = f'BB_Lower_{bb_window}'
    if all(c in last_row and not pd.isna(last_row[c]) for c in [bb_up_col, bb_low_col]):
        bb_up = last_row[bb_up_col]
        bb_low = last_row[bb_low_col]
        add_signal(f"Bollinger ({bb_window})", last_price < bb_low, last_price > bb_up,
                   f"Preço: ${last_price:,.0f}", 'bollinger')

    # 6. Volume Anormal
    vol_ma_col = 'Volume_MA20' # Assumindo MA20 calculada em load_data
    if 'volume' in last_row and vol_ma_col in df.columns and not pd.isna(df[vol_ma_col].iloc[-1]):
         last_vol = last_row['volume']
         vol_ma = df[vol_ma_col].iloc[-1]
         vol_ratio = last_vol / vol_ma if vol_ma else 0
         price_change = last_price - prev_row['price']
         vol_high = vol_ratio > 1.7 # Limiar mais alto para volume
         add_signal("Volume Spike", vol_high and price_change > 0, vol_high and price_change < 0,
                    f"{vol_ratio:.1f}x Média", 'volume')

    # 7. OBV (Tendência vs MA)
    obv_ma_col = 'OBV_MA20' # Assumindo MA20 calculada
    if 'OBV' in last_row and obv_ma_col in df.columns and not pd.isna(df[obv_ma_col].iloc[-1]):
        last_obv = last_row['OBV']
        obv_ma = df[obv_ma_col].iloc[-1]
        add_signal("OBV Trend", last_obv > obv_ma, last_obv < obv_ma,
                   f"OBV {'acima' if last_obv > obv_ma else 'abaixo'} MA", 'obv')

    # 8. Stochastic (Cruzamento K/D nas Zonas)
    if all(c in last_row and not pd.isna(last_row[c]) for c in ['Stoch_K', 'Stoch_D']) and \
       all(c in prev_row and not pd.isna(prev_row[c]) for c in ['Stoch_K', 'Stoch_D']):
        k_c, d_c = last_row['Stoch_K'], last_row['Stoch_D']
        k_p, d_p = prev_row['Stoch_K'], prev_row['Stoch_D']
        cross_up = k_c > d_c and k_p <= d_p
        cross_down = k_c < d_c and k_p >= d_p
        add_signal("Stochastic Cross", cross_up and k_c < 25, cross_down and k_c > 75, # Nas zonas
                   f"K:{k_c:.1f}, D:{d_c:.1f}", 'stochastic')

    # 9. Gaussian Process (Previsão vs Preço)
    if 'GP_Prediction' in last_row and not pd.isna(last_row['GP_Prediction']):
        gp_pred = last_row['GP_Prediction']
        threshold = 0.02 # Limiar menor para GP
        add_signal("Gaussian Process", gp_pred > last_price * (1 + threshold), gp_pred < last_price * (1 - threshold),
                   f"Prev: ${gp_pred:,.0f}", 'gaussian_process')

    # 10. Order Blocks (Preço Atual Tocando Bloco Válido)
    _, blocks = identify_order_blocks(df, **st.session_state.user_settings) # Usar config atual
    for block in blocks:
        is_touching = (last_price >= block['low']) and (last_price <= block['high'])
        if is_touching:
             block_id = f"{block['type']} ({block['date'].strftime('%m-%d')})"
             condition_buy = not block['broken'] and block['type'] == 'bullish_ob' or \
                             block['broken'] and block['breaker_type'] == 'bearish_breaker'
             condition_sell = not block['broken'] and block['type'] == 'bearish_ob' or \
                              block['broken'] and block['breaker_type'] == 'bullish_breaker'
             add_signal(block_id, condition_buy, condition_sell,
                        f"Zona: ${block['low']:,.0f}-${block['high']:,.0f}", 'order_blocks')

    # 11. Divergência RSI
    if 'RSI_Divergence' in last_row and last_row['RSI_Divergence'] != 0:
        div = last_row['RSI_Divergence']
        add_signal("Divergência RSI", div > 0, div < 0, "Detectada", 'rsi') # Reusar peso RSI?

    # --- Calcular Veredito Final Ponderado ---
    total_buy_weight = sum(s[3] for s in signals if s[3] > 0)
    total_sell_weight = abs(sum(s[3] for s in signals if s[3] < 0)) # Peso de venda é negativo
    buy_signals_count = sum(1 for s in signals if s[3] > 0)
    sell_signals_count = sum(1 for s in signals if s[3] < 0)

    final_verdict = "➖ NEUTRO"
    if total_buy_weight == 0 and total_sell_weight == 0: pass # Mantém Neutro
    elif total_buy_weight >= total_sell_weight * 1.8: final_verdict = "✅ FORTE COMPRA" # Limiar mais alto para "Forte"
    elif total_buy_weight > total_sell_weight: final_verdict = "📈 COMPRA"
    elif total_sell_weight >= total_buy_weight * 1.8: final_verdict = "❌ FORTE VENDA"
    elif total_sell_weight > total_buy_weight: final_verdict = "📉 VENDA"

    return signals, final_verdict, buy_signals_count, sell_signals_count


# ============================================
# GERAÇÃO DE RELATÓRIOS (CoinAnk e PDF)
# ============================================

def generate_coinank_style_report(data):
    """Gera um relatório de análise conciso no estilo CoinAnk."""
    report = {
        "status": "error", "message": "Dados insuficientes.",
        "current_price": "N/A", "24h_change": "N/A",
        "support_levels": [], "resistance_levels": [], "trend": "Indeterminado",
        "ma_summary": "N/A", "macd_summary": "N/A", "boll_summary": "N/A", "rsi_summary": "N/A",
        "volume_summary": "N/A",
        "suggestion": {"direction": "Manter Neutro", "entry": "N/A", "stop_loss": "N/A", "target": "N/A"}
    }

    if 'prices' not in data or data['prices'].empty: return report
    df = data['prices']
    last_row = df.iloc[-1] if not df.empty else None
    if last_row is None or pd.isna(last_row['price']): return report

    current_price = last_row['price']
    report["current_price"] = f"${current_price:,.2f} USDT"
    report["24h_change"] = f"{data.get('24h_change', 0.0):.2f}%" if '24h_change' in data else "N/A"

    # Níveis S/R (já calculados em data['support_resistance'])
    if 'support_resistance' in data and data['support_resistance']:
        levels = sorted(data['support_resistance'])
        report["support_levels"] = [f"${lvl:,.2f}" for lvl in levels if lvl < current_price][-3:] # Últimos 3 abaixo
        report["resistance_levels"] = [f"${lvl:,.2f}" for lvl in levels if lvl >= current_price][:3] # Primeiros 3 acima

    # Tendência (Baseada em EMAs curtas/médias e preço)
    ma_short = last_row.get('MA14', np.inf) # Usar EMA14 e EMA50
    ma_long = last_row.get('MA50', np.inf)
    if not np.isinf(ma_short) and not np.isinf(ma_long):
        if current_price > ma_short > ma_long: report["trend"] = "Claramente Altista"
        elif current_price < ma_short < ma_long: report["trend"] = "Claramente Baixista"
        elif current_price > ma_long: report["trend"] = "Potencialmente Altista"
        elif current_price < ma_long: report["trend"] = "Potencialmente Baixista"
        else: report["trend"] = "Lateral / Indefinido"
    else: report["trend"] = "Indeterminado (MAs ausentes)"

    # Resumo Indicadores
    mas = {f"EMA{w}": last_row.get(f'MA{w}', np.nan) for w in [7, 14, 30, 50, 100, 200]}
    report["ma_summary"] = ", ".join([f"{k}({v:.0f})" for k, v in mas.items() if not pd.isna(v)]) or "N/A"
    if 'MACD' in last_row and not pd.isna(last_row['MACD']):
        report["macd_summary"] = f"DIF:{last_row['MACD']:.1f}, DEA:{last_row.get('MACD_Signal', 0):.1f}, Hist:{last_row.get('MACD_Hist', 0):.1f}"
    if 'BB_MA_20' in last_row and not pd.isna(last_row['BB_MA_20']):
        report["boll_summary"] = f"Média:{last_row['BB_MA_20']:.0f}, %B:{last_row.get('BB_PctB_20', 0):.2f}, Largura:{last_row.get('BB_Bandwidth_20', 0):.1f}%"
    rsis = {f"RSI{w}": last_row.get(f'RSI_{w}', np.nan) for w in [6, 12, 14, 24]}
    report["rsi_summary"] = ", ".join([f"{k}:{v:.1f}" for k, v in rsis.items() if not pd.isna(v)]) or "N/A"
    if 'volume' in last_row and 'Volume_MA20' in df.columns and not pd.isna(df['Volume_MA20'].iloc[-1]):
        vol_ratio = last_row['volume'] / df['Volume_MA20'].iloc[-1] if df['Volume_MA20'].iloc[-1] else 0
        report["volume_summary"] = f"{'Alto' if vol_ratio > 1.7 else 'Baixo' if vol_ratio < 0.7 else 'Médio'} ({vol_ratio:.1f}x MA20)"
    else: report["volume_summary"] = "N/A"

    # Sugestão de Trade (Simplificada)
    if "Baixista" in report["trend"]:
        report["suggestion"]["direction"] = "Vender Short"
        entry = report["resistance_levels"][0].replace('$','').replace(',','') if report["resistance_levels"] else current_price * 1.005
        stop = float(entry) * 1.02 # Stop 2% acima
        target = report["support_levels"][-1].replace('$','').replace(',','') if report["support_levels"] else current_price * 0.97 # Target 3% abaixo
        report["suggestion"]["entry"] = f"~${float(entry):,.2f}"
        report["suggestion"]["stop_loss"] = f"~${stop:.2f}"
        report["suggestion"]["target"] = f"~${float(target):,.2f}"
    elif "Altista" in report["trend"]:
        report["suggestion"]["direction"] = "Comprar Long"
        entry = report["support_levels"][-1].replace('$','').replace(',','') if report["support_levels"] else current_price * 0.995
        stop = float(entry) * 0.98 # Stop 2% abaixo
        target = report["resistance_levels"][0].replace('$','').replace(',','') if report["resistance_levels"] else current_price * 1.03 # Target 3% acima
        report["suggestion"]["entry"] = f"~${float(entry):,.2f}"
        report["suggestion"]["stop_loss"] = f"~${stop:.2f}"
        report["suggestion"]["target"] = f"~${float(target):,.2f}"

    report["status"] = "success"
    report["message"] = "Relatório gerado."
    return report


def clean_text(text):
    """Limpa texto para PDF, removendo caracteres problemáticos."""
    if isinstance(text, (int, float, np.number)): return str(round(text, 2)) # Arredondar números
    if text is None: return ""
    text = str(text)
    # Remover emojis comuns e caracteres não-ASCII problemáticos para latin-1
    text = re.sub(r'[^\x00-\x7F]+', '', text) # Remover não-ASCII
    # Permitir básicos + moeda/pontuação
    text = re.sub(r'[^a-zA-Z0-9\s.,!$%*()+-/:]+', '', text)
    return text


def generate_pdf_report(data, signals, final_verdict, coinank_report):
    """Gera relatório PDF completo."""
    pdf = FPDF()
    pdf.add_page()
    # Adicionar fonte que suporte mais caracteres (se necessário e instalada)
    try:
        pdf.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
        main_font = 'DejaVu'
    except RuntimeError:
        main_font = 'Arial' # Fallback
        st.warning("Fonte DejaVu não encontrada, usando Arial (pode haver problemas com caracteres especiais).")

    pdf.set_font(main_font, size=10) # Usar fonte definida

    def write_pdf_cell(text, size=10, style='', align='L', ln=1):
        cleaned = clean_text(text)
        try:
            pdf.set_font(main_font, style=style, size=size)
            pdf.cell(0, 5, txt=cleaned, ln=ln, align=align, border=0)
        except Exception as e:
             print(f"FPDF Error: {e} with text: {cleaned}") # Debug
             pdf.set_font('Arial', style=style, size=size) # Tentar Arial como fallback
             pdf.cell(0, 5, txt="[Error displaying text]", ln=ln, align=align, border=0)

    # --- Cabeçalho ---
    write_pdf_cell("BTC AI Dashboard Pro+ - Relatório Completo", size=16, style='B', align='C')
    write_pdf_cell(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", size=8, align='C')
    pdf.ln(5)

    # --- Resumo CoinAnk ---
    write_pdf_cell("Resumo Rápido (Estilo CoinAnk)", size=12, style='B')
    if coinank_report["status"] == "success":
        cr = coinank_report # Alias
        write_pdf_cell(f"Preco Atual: {cr.get('current_price', 'N/A')} | Var 24h: {cr.get('24h_change', 'N/A')} | Tendencia: {cr.get('trend', 'N/A')}")
        write_pdf_cell(f"Suportes: {', '.join(cr.get('support_levels', [])) or 'N/A'}")
        write_pdf_cell(f"Resistencias: {', '.join(cr.get('resistance_levels', [])) or 'N/A'}")
        pdf.ln(2)
        write_pdf_cell("Indicadores:", style='B')
        write_pdf_cell(f"MA: {cr.get('ma_summary', 'N/A')}", size=9)
        write_pdf_cell(f"MACD: {cr.get('macd_summary', 'N/A')}", size=9)
        write_pdf_cell(f"BOLL: {cr.get('boll_summary', 'N/A')}", size=9)
        write_pdf_cell(f"RSI: {cr.get('rsi_summary', 'N/A')}", size=9)
        write_pdf_cell(f"Volume: {cr.get('volume_summary', 'N/A')}", size=9)
        pdf.ln(2)
        write_pdf_cell("Sugestao:", style='B')
        sg = cr.get('suggestion', {})
        write_pdf_cell(f"Direcao: {sg.get('direction', 'N/A')} | Entrada: {sg.get('entry', 'N/A')} | Stop: {sg.get('stop_loss', 'N/A')} | Alvo: {sg.get('target', 'N/A')}")
    else: write_pdf_cell(f"Erro: {coinank_report.get('message', 'N/A')}")
    pdf.ln(5)

    # --- Análise Ponderada ---
    write_pdf_cell("Analise Ponderada Detalhada", size=12, style='B')
    write_pdf_cell(f"Veredito Final: {final_verdict}", style='B')
    write_pdf_cell("Sinais Individuais:", style='B')
    for sig_name, sig_val, sig_detail, sig_weight in signals:
         weight_sign = '+' if sig_weight > 0 else '' if sig_weight == 0 else '-'
         write_pdf_cell(f"- {sig_name}: {sig_val} ({sig_detail}) [Peso: {weight_sign}{abs(sig_weight):.1f}]", size=9)
    pdf.ln(5)

    # --- Configurações ---
    write_pdf_cell("Configuracoes Utilizadas", size=12, style='B')
    settings = st.session_state.get('user_settings', DEFAULT_SETTINGS)
    # Listar apenas algumas configurações chave
    write_pdf_cell(f"- Indicadores: RSI({settings.get('rsi_window')}), BB({settings.get('bb_window')}), EMAs({settings.get('ma_windows')}), OB({settings.get('ob_swing_length')}, Body:{settings.get('ob_use_body')}), S/R({settings.get('n_clusters')})", size=9)
    write_pdf_cell(f"- IA: LSTM(W:{settings.get('lstm_window')},E:{settings.get('lstm_epochs')}), RL(T:{settings.get('rl_episodes')}), GP(W:{settings.get('gp_window')},L:{settings.get('gp_lookahead')}), Sentimento(Conf:{settings.get('min_confidence')})", size=9)
    pdf.ln(5)

    # --- Notícias ---
    write_pdf_cell("Noticias Relevantes (Analise de Sentimento)", size=12, style='B')
    sentiment_model = load_sentiment_model()
    if 'news' in data and data['news'] and sentiment_model:
        analyzed_news = analyze_news_sentiment(data['news'], sentiment_model)
        filtered_news = filter_news_by_confidence(analyzed_news, settings.get('min_confidence', 0.7))
        if filtered_news:
            for news in filtered_news[:5]: # Limitar a 5 notícias no PDF
                sentiment_label = news.get('sentiment', 'N/A')
                score = news.get('sentiment_score', 0)
                write_pdf_cell(f"- [{sentiment_label} ({score:.0%})] {news.get('title', 'N/A')}", size=9)
        else: write_pdf_cell("Nenhuma noticia filtrada encontrada.", size=9)
    else: write_pdf_cell("Dados de noticias ou modelo de sentimento indisponivel.", size=9)
    pdf.ln(5)

    # --- Disclaimer ---
    write_pdf_cell("Disclaimer: Esta analise e gerada por algoritmos e destina-se apenas a fins informativos. Nao constitui aconselhamento financeiro ou de investimento. Realize sua propria pesquisa.", size=8, style='I')

    # --- Salvar PDF ---
    try:
        pdf_bytes = pdf.output(dest='S').encode('latin-1') # Output como string binária
        return pdf_bytes
    except Exception as e:
        st.error(f"Erro ao gerar bytes do PDF: {e}")
        return None


# ============================================
# FUNÇÃO PRINCIPAL (main)
# ============================================

def main():
    # Inicializar estado da sessão se não existir
    if 'user_settings' not in st.session_state:
        st.session_state.user_settings = DEFAULT_SETTINGS.copy()
    if 'last_backtest_strategy' not in st.session_state:
         st.session_state.last_backtest_strategy = None
    if 'backtest_results' not in st.session_state:
         st.session_state.backtest_results = pd.DataFrame()
    if 'backtest_metrics' not in st.session_state:
         st.session_state.backtest_metrics = {}


    # === Sidebar ===
    st.sidebar.header("⚙️ Painel de Controle AI")
    # (Expanders e sliders/widgets do sidebar mantidos como na versão anterior)
    with st.sidebar.expander("🧠 Configurações de IA", expanded=False):
        st.session_state.user_settings['lstm_window'] = st.slider("Janela LSTM", 30, 120, st.session_state.user_settings.get('lstm_window', 60), 10, key='cfg_lstm_w')
        st.session_state.user_settings['lstm_epochs'] = st.slider("Épocas LSTM", 10, 100, st.session_state.user_settings.get('lstm_epochs', 50), 10, key='cfg_lstm_e')
        st.session_state.user_settings['lstm_units'] = st.slider("Unidades LSTM", 30, 100, st.session_state.user_settings.get('lstm_units', 50), 10, key='cfg_lstm_u')
        st.session_state.user_settings['rl_episodes'] = st.slider("Timesteps RL", 5000, 50000, st.session_state.user_settings.get('rl_episodes', 10000), 1000, key='cfg_rl_ts')

    with st.sidebar.expander("🔧 Parâmetros Técnicos", expanded=True):
        st.session_state.user_settings['rsi_window'] = st.slider("Período RSI", 7, 30, st.session_state.user_settings.get('rsi_window', 14), key='cfg_rsi_w')
        st.session_state.user_settings['bb_window'] = st.slider("Janela Bollinger", 10, 50, st.session_state.user_settings.get('bb_window', 20), key='cfg_bb_w')
        st.session_state.user_settings['ma_windows'] = st.multiselect("EMAs para Exibir", [7, 14, 20, 30, 50, 100, 200], st.session_state.user_settings.get('ma_windows', [7, 30, 200]), key='cfg_ma_wins')
        st.session_state.user_settings['gp_window'] = st.slider("Janela Gauss. Process", 10, 60, st.session_state.user_settings.get('gp_window', 30), key='cfg_gp_w')
        st.session_state.user_settings['gp_lookahead'] = st.slider("Previsão GP (dias)", 1, 10, st.session_state.user_settings.get('gp_lookahead', 5), key='cfg_gp_l')

    with st.sidebar.expander("📊 Order Blocks (LuxAlgo)", expanded=False):
        st.session_state.user_settings['ob_swing_length'] = st.slider("Swing Lookback (Ímpar)", 5, 21, st.session_state.user_settings.get('ob_swing_length', 11), 2, key='cfg_ob_sw')
        st.session_state.user_settings['ob_show_bull'] = st.slider("Mostrar Bullish OBs", 1, 5, st.session_state.user_settings.get('ob_show_bull', 3), key='cfg_ob_b')
        st.session_state.user_settings['ob_show_bear'] = st.slider("Mostrar Bearish OBs", 1, 5, st.session_state.user_settings.get('ob_show_bear', 3), key='cfg_ob_s')
        st.session_state.user_settings['ob_use_body'] = st.checkbox("Usar corpo candle OB", st.session_state.user_settings.get('ob_use_body', True), key='cfg_ob_body')

    with st.sidebar.expander("🔍 Clusterização K-Means", expanded=False):
        st.session_state.user_settings['n_clusters'] = st.slider("Clusters S/R", 3, 12, st.session_state.user_settings.get('n_clusters', 5), key='cfg_clus')

    with st.sidebar.expander("🔔 Alertas e Filtros", expanded=False):
        st.session_state.user_settings['email'] = st.text_input("E-mail (Não implementado)", st.session_state.user_settings.get('email', ''), key='cfg_email')
        st.session_state.user_settings['min_confidence'] = st.slider("Conf. Mín. Sentimento", 0.5, 1.0, st.session_state.user_settings.get('min_confidence', 0.7), 0.05, key='cfg_conf')

    col1, col2 = st.sidebar.columns(2)
    # Botão Salvar apenas atualiza a sessão, não salva permanentemente
    if col1.button("💾 Aplicar", key='btn_save'): st.sidebar.success("Config. aplicadas na sessão.")
    if col2.button("🔄 Resetar", key='btn_reset'):
        st.session_state.user_settings = DEFAULT_SETTINGS.copy()
        # Limpar estados derivados das configurações
        keys_to_clear = ['lstm_model', 'lstm_scaler', 'rl_model', 'backtest_results', 'backtest_metrics', 'last_backtest_strategy']
        for key in keys_to_clear:
             if key in st.session_state: del st.session_state[key]
        st.sidebar.success("Configurações resetadas!")
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("🔄 Atualização de Dados")
    if st.sidebar.button("🔄 Recarregar Dados Agora", key='btn_reload'):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    # === Carregar Dados ===
    data = load_cached_data()
    if not data or 'prices' not in data or data['prices'].empty:
        st.error("Falha crítica ao carregar dados. O dashboard não pode continuar.")
        st.stop()

    # === Gerar Análises e Relatórios ===
    signals, final_verdict, buy_signals, sell_signals = generate_signals(
        data, **st.session_state.user_settings # Passar configurações
    )
    coinank_report = generate_coinank_style_report(data)
    sentiment = get_market_sentiment()
    traditional_assets = get_traditional_assets()

    # Analisar sentimento das notícias (garante que modelo está carregado)
    sentiment_model = load_sentiment_model()
    analyzed_news = analyze_news_sentiment(data.get('news', []), sentiment_model)
    filtered_news = filter_news_by_confidence(analyzed_news, st.session_state.user_settings['min_confidence'])


    # === Interface Principal ===
    st.header("📊 Painel Integrado BTC AI Pro+")

    # Métricas Principais
    m1, m2, m3, m4, m5 = st.columns(5)
    price_txt = coinank_report.get('current_price', 'N/A')
    change_txt = coinank_report.get('24h_change', 'N/A')
    change_val = float(re.sub(r'[^\d.-]', '', change_txt)) if change_txt != 'N/A' else 0.0
    m1.metric("Preço BTC", price_txt, f"{change_val:.2f}%" if change_txt != 'N/A' else None)
    m2.metric("Sentimento F&G", f"{sentiment['value']}/100", sentiment['sentiment'])
    display_asset_metric(m3, "S&P 500", traditional_assets)
    display_asset_metric(m4, "Ouro", traditional_assets)
    m5.metric("Tendência Diária", coinank_report.get('trend', 'N/A')) # Usar tendência do relatório

    # Abas
    tab_titles = ["💡 Resumo", "📈 Mercado", "🆚 Comparativos", "🧪 Backtesting",
                  "🌍 Cenários", "🤖 IA", "📉 Técnico Det.", "📤 Exportar"]
    tabs = st.tabs(tab_titles)

    # Conteúdo das Abas (mantido como na resposta anterior, usando as novas funções e dados)
    # --- Aba 0: Resumo Rápido (CoinAnk Style) ---
    with tabs[0]:
        st.subheader("⚡ Análise Rápida do Mercado (Estilo CoinAnk)")
        if coinank_report["status"] == "success":
            st.markdown(f"**Preço Atual:** `{coinank_report.get('current_price', 'N/A')}` | **Variação 24h:** `{coinank_report.get('24h_change', 'N/A')}`")
            st.markdown(f"**Tendência Principal (Diário):** `{coinank_report.get('trend', 'N/A')}`")
            col_sr1, col_sr2 = st.columns(2)
            with col_sr1:
                st.markdown("**Níveis de Suporte Chave:**")
                st.markdown(f"`{', '.join(coinank_report.get('support_levels', ['N/A']))}`")
            with col_sr2:
                st.markdown("**Níveis de Resistência Chave:**")
                st.markdown(f"`{', '.join(coinank_report.get('resistance_levels', ['N/A']))}`")
            st.divider()
            st.markdown("**Resumo dos Indicadores Técnicos:**")
            st.markdown(f"- **Médias Móveis:** {coinank_report.get('ma_summary', 'N/A')}")
            st.markdown(f"- **MACD:** {coinank_report.get('macd_summary', 'N/A')}")
            st.markdown(f"- **Bollinger Bands:** {coinank_report.get('boll_summary', 'N/A')}")
            st.markdown(f"- **RSI:** {coinank_report.get('rsi_summary', 'N/A')}")
            st.markdown(f"- **Volume:** {coinank_report.get('volume_summary', 'N/A')}")
            st.divider()
            st.markdown("**Sugestão Baseada na Análise:**")
            suggestion = coinank_report.get('suggestion', {})
            st.markdown(f"- **Direção:** `{suggestion.get('direction', 'N/A')}` | **Entrada:** `{suggestion.get('entry', 'N/A')}`")
            st.markdown(f"- **Stop Loss:** `{suggestion.get('stop_loss', 'N/A')}` | **Alvo:** `{suggestion.get('target', 'N/A')}`")
            st.caption("Disclaimer: Análise gerada por algoritmos, não é conselho financeiro.")
        else: st.error(f"Não foi possível gerar o resumo: {coinank_report.get('message', 'Erro')}")

    # --- Aba 1: Mercado (Gráfico Principal e Sinais Ponderados) ---
    with tabs[1]:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Preço BTC, EMAs, Order Blocks e Zonas S/R")
            if 'prices' in data and not data['prices'].empty:
                df_plot = data['prices']
                ma_cols_plot = ['price'] + [f'MA{w}' for w in st.session_state.user_settings['ma_windows'] if f'MA{w}' in df_plot.columns]
                fig = px.line(df_plot, x="date", y=ma_cols_plot, title="Preço BTC e Indicadores Chave")
                # Adicionar Order Blocks
                _, blocks_plot = identify_order_blocks(df_plot, **st.session_state.user_settings)
                fig = plot_order_blocks(fig, blocks_plot, df_plot['price'].iloc[-1])
                # Adicionar Zonas S/R
                if 'support_resistance' in data and data['support_resistance']:
                    for level in data['support_resistance']:
                        fig.add_hline(y=level, line_dash="dot", line_color="grey", opacity=0.7, annotation_text=f"S/R: ${level:,.0f}", annotation_position="bottom right")
                # Adicionar Divergências
                if 'RSI_Divergence' in df_plot.columns:
                     divergences = df_plot[df_plot['RSI_Divergence'] != 0]
                     if not divergences.empty:
                          fig.add_trace(go.Scatter(
                              x=divergences['date'], y=divergences['price'] * divergences['RSI_Divergence'].apply(lambda x: 1.02 if x>0 else 0.98), # Offset
                              mode='markers', name='Divergência RSI',
                              marker=dict(symbol=divergences['RSI_Divergence'].apply(lambda x: 'triangle-up' if x > 0 else 'triangle-down'),
                                          color=divergences['RSI_Divergence'].apply(lambda x: 'green' if x > 0 else 'red'), size=10) ))
                fig.update_layout(hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
            else: st.warning("Dados de preços indisponíveis.")

            # Gráficos On-Chain/Mercado
            st.subheader("Dados On-Chain e Atividade de Mercado (Simulado)")
            col_oc1, col_oc2 = st.columns(2)
            with col_oc1:
                 hr_fig = plot_hashrate_difficulty(data)
                 if hr_fig: st.plotly_chart(hr_fig, use_container_width=True)
                 else: st.caption("Hashrate/Dificuldade indisponível.")
            with col_oc2:
                 wh_fig = plot_whale_activity(data)
                 if wh_fig: st.plotly_chart(wh_fig, use_container_width=True)
                 else: st.caption("Atividade Whale indisponível.")

        with col2:
            st.subheader("🔎 Análise Ponderada")
            # Veredito Final com cor
            if final_verdict == "✅ FORTE COMPRA": st.success(f"### {final_verdict}")
            elif final_verdict == "📈 COMPRA": st.info(f"### {final_verdict}")
            elif final_verdict == "📉 VENDA": st.warning(f"### {final_verdict}")
            elif final_verdict == "❌ FORTE VENDA": st.error(f"### {final_verdict}")
            else: st.markdown(f"### {final_verdict}")
            st.caption(f"Compra: {buy_signals} sinais | Venda: {sell_signals} sinais (Ponderados)")
            st.divider()
            # Sinais Individuais (Expander)
            with st.expander("Mostrar/Ocultar Sinais Detalhados", expanded=False):
                if signals:
                    for sig_name, sig_val, sig_detail, sig_weight in signals:
                        color = "green" if sig_weight > 0 else "red" if sig_weight < 0 else "grey"
                        st.markdown(f"<span style='color:{color};'>●</span> **{sig_name}**: {sig_val} ({sig_detail}) `[Peso: {sig_weight:+.1f}]`", unsafe_allow_html=True)
                else: st.warning("Nenhum sinal individual gerado.")
            st.divider()
            # Notícias e Sentimento
            st.subheader("📰 Notícias e Sentimento")
            st.metric("Índice Fear & Greed", f"{sentiment['value']}/100", sentiment['sentiment'])
            st.progress(sentiment['value'] / 100)
            st.markdown("**Notícias Recentes Filtradas:**")
            if filtered_news:
                 for news in filtered_news[:5]: # Mostrar top 5
                      senti_label = news.get('sentiment', 'N/A')
                      senti_score = news.get('sentiment_score', 0)
                      color = "green" if senti_label == 'POSITIVE' else "red" if senti_label == 'NEGATIVE' else "grey"
                      st.markdown(f"<span style='color:{color};'>●</span> [{news.get('source','N/A')}] **{news['title']}** <span style='font-size:smaller;'>({senti_label}: {senti_score:.0%})</span>", unsafe_allow_html=True)
            else: st.caption("Nenhuma notícia relevante filtrada.")


    # --- Aba 2: Comparativos ---
    with tabs[2]:
        st.subheader("📌 BTC vs Ativos Tradicionais (Base 100)")
        if traditional_assets is not None and not traditional_assets.empty:
            assets_to_plot = ["BTC-USD", "ETH-USD", "S&P 500", "NASDAQ", "Ouro", "USD Index"]
            df_comp = traditional_assets[traditional_assets['asset'].isin(assets_to_plot)]
            # Normalizar
            df_comp['value_norm'] = df_comp.groupby('asset')['value'].transform(lambda x: (x / x.iloc[0]) * 100 if not x.empty and x.iloc[0] != 0 else 100)
            fig_comp = px.line(df_comp, x="date", y="value_norm", color="asset", title="Desempenho Relativo Normalizado")
            fig_comp.update_layout(hovermode="x unified", legend_title_text='Ativo')
            st.plotly_chart(fig_comp, use_container_width=True)
        else: st.warning("Dados comparativos não disponíveis.")

        st.subheader("📊 Fluxo de Exchanges (Simulado)")
        exchange_flows = data.get('exchanges')
        if exchange_flows is not None and not exchange_flows.empty:
            # Formatar colunas numéricas
            format_dict = {col: '{:,.0f}' for col in exchange_flows.columns if 'BTC' in col}
            st.dataframe(exchange_flows.style.format(format_dict).background_gradient(cmap='RdYlGn', subset=['Líquido (BTC - 24h Sim.)']), use_container_width=True)
        else: st.caption("Dados de fluxo de exchange simulados não disponíveis.")

    # --- Aba 3: Backtesting ---
    with tabs[3]:
        st.subheader("🧪 Backtesting Avançado de Estratégias")
        if 'prices_full' not in data or data['prices_full'].empty:
            st.error("Dados históricos completos não disponíveis para backtesting.")
        else:
            df_backtest_data = data['prices_full'].copy()
            strategy = st.selectbox("Escolha a Estratégia:", ["RSI", "MACD", "Bollinger", "EMA Cross", "Volume", "OBV", "Stochastic", "Order Blocks"], key="bt_strat_sel")

            params_container = st.container()
            with params_container:
                st.markdown(f"**Configurar Parâmetros para {strategy}:**")
                params = {}
                # (Sliders/widgets de parâmetros mantidos como na resposta anterior)
                if strategy == "RSI":
                    params['rsi_window'] = st.slider("Período RSI", 7, 30, 14, key='p_rsi_w')
                    params['overbought'] = st.slider("Sobrecompra", 60, 90, 70, key='p_rsi_ob')
                    params['oversold'] = st.slider("Sobrevenda", 10, 40, 30, key='p_rsi_os')
                elif strategy == "MACD":
                    params['fast'] = st.slider("EMA Rápida", 5, 20, 12, key='p_macd_f')
                    params['slow'] = st.slider("EMA Lenta", 20, 50, 26, key='p_macd_s')
                    params['signal'] = st.slider("Linha de Sinal", 5, 20, 9, key='p_macd_sig')
                elif strategy == "Bollinger":
                    params['window'] = st.slider("Janela BB", 10, 50, 20, key='p_bb_w')
                    params['num_std'] = st.slider("Nº Desvios", 1.0, 3.0, 2.0, 0.1, key='p_bb_std')
                elif strategy == "EMA Cross":
                    params['short_window'] = st.slider("EMA Curta", 5, 50, 9, key='p_ema_s')
                    params['long_window'] = st.slider("EMA Longa", 10, 200, 21, key='p_ema_l')
                    if params['short_window'] >= params['long_window']: st.warning("EMA Curta deve ser menor que Longa.")
                elif strategy == "Volume":
                    params['volume_window'] = st.slider("Janela Média Vol", 10, 50, 20, key='p_vol_w')
                    params['threshold'] = st.slider("Limiar Vol", 1.1, 3.0, 1.7, 0.1, key='p_vol_t') # Aumentado limiar
                elif strategy == "OBV":
                    params['obv_window'] = st.slider("Janela MA OBV", 10, 50, 20, key='p_obv_w')
                    params['price_window'] = st.slider("Janela EMA Preço", 10, 50, 30, key='p_obv_p')
                elif strategy == "Stochastic":
                    params['k_window'] = st.slider("%K Período", 5, 30, 14, key='p_stoch_k')
                    params['d_window'] = st.slider("%K Suavização (%D)", 3, 9, 3, key='p_stoch_d')
                    params['overbought'] = st.slider("Sobrecompra Stoch", 70, 95, 80, key='p_stoch_ob')
                    params['oversold'] = st.slider("Sobrevenda Stoch", 5, 30, 20, key='p_stoch_os')
                elif strategy == "Order Blocks":
                    params['swing_length'] = st.slider("Swing OB (Ímpar)", 5, 21, 11, 2, key='p_ob_sw')
                    params['use_body'] = st.checkbox("Usar Corpo Candle OB", True, key='p_ob_body')

                if st.button(f"Executar Backtest {strategy}", key=f"run_{strategy}"):
                    valid_params = True
                    if strategy == 'EMA Cross' and params['short_window'] >= params['long_window']:
                         st.error("EMA Curta deve ser menor que Longa.")
                         valid_params = False

                    if valid_params:
                        with st.spinner(f"Executando backtest {strategy}..."):
                             # Chamar função de backtest
                             func_map = {
                                 "RSI": backtest_rsi_strategy, "MACD": backtest_macd_strategy,
                                 "Bollinger": backtest_bollinger_strategy, "EMA Cross": backtest_ema_cross_strategy,
                                 "Volume": backtest_volume_strategy, "OBV": backtest_obv_strategy,
                                 "Stochastic": backtest_stochastic_strategy, "Order Blocks": backtest_order_block_strategy
                             }
                             if strategy in func_map:
                                 df_results = func_map[strategy](df_backtest_data, **params)
                                 st.session_state.backtest_results = df_results
                                 st.session_state.backtest_metrics = calculate_metrics(df_results)
                                 st.session_state.last_backtest_strategy = strategy
                                 st.rerun() # Re-renderizar para mostrar resultados
                             else:
                                 st.error("Estratégia não implementada para backtest.")

            # Exibição dos Resultados
            if st.session_state.last_backtest_strategy == strategy and not st.session_state.backtest_results.empty:
                 st.divider()
                 st.subheader(f"📊 Resultados do Backtesting: {strategy}")
                 df_display = st.session_state.backtest_results
                 metrics = st.session_state.backtest_metrics

                 # Gráfico de Desempenho (Log Scale)
                 fig_bt = go.Figure()
                 fig_bt.add_trace(go.Scatter(x=df_display.index, y=df_display['strategy_cumulative'], name="Estratégia", line=dict(color='green')))
                 fig_bt.add_trace(go.Scatter(x=df_display.index, y=df_display['cumulative_return'], name="Buy & Hold", line=dict(color='blue', dash='dot')))
                 fig_bt.update_layout(title=f"Desempenho Cumulativo (Escala Log)", yaxis_type="log", hovermode="x unified")
                 st.plotly_chart(fig_bt, use_container_width=True)

                 # Métricas
                 st.subheader("📈 Métricas de Performance")
                 if metrics:
                     m_cols = st.columns(5)
                     m_cols[0].metric("Retorno Estratégia", f"{metrics.get('Retorno Estratégia', 0.0):.2%}")
                     m_cols[1].metric("Retorno Buy & Hold", f"{metrics.get('Retorno Buy & Hold', 0.0):.2%}")
                     m_cols[2].metric("Sharpe Ratio", f"{metrics.get('Sharpe Estratégia', 0.0):.2f}")
                     m_cols[3].metric("Max Drawdown", f"{metrics.get('Max Drawdown', 0.0):.2%}")
                     m_cols[4].metric("Trades", f"{metrics.get('Num Trades', 0)}")
                     # m_cols[4].metric("Taxa Acerto Trades", f"{metrics.get('Taxa Acerto', 0.0):.1%}")
                 else: st.warning("Métricas não calculadas.")

            # Otimização (Opcional)
            st.divider()
            st.subheader("⚙️ Otimização Automática (Experimental)")
            if st.checkbox("🔍 Ativar Otimização", key=f"opt_cb_{strategy}"):
                 st.warning("Pode demorar vários minutos!")
                 if st.button(f"Otimizar {strategy}", key=f"run_opt_{strategy}"):
                      param_space_opt = {} # Definir espaços para cada estratégia
                      if strategy == "RSI": param_space_opt = {'rsi_window': [10, 14, 20], 'oversold': [20, 25, 30], 'overbought': [70, 75, 80]}
                      elif strategy == "EMA Cross": param_space_opt = {'short_window': [7, 9, 12], 'long_window': [20, 26, 30, 50]}
                      # Adicionar outros...
                      elif strategy == "Order Blocks": param_space_opt = {'swing_length': [9, 11, 15], 'use_body': [True, False]}

                      if param_space_opt:
                           best_params_opt, best_sharpe_opt, _ = optimize_strategy_parameters(
                               {'prices': df_backtest_data}, strategy, param_space_opt
                           )
                           if best_params_opt:
                                st.success(f"🎯 Melhores Parâmetros (Sharpe: {best_sharpe_opt:.3f}):")
                                st.json(best_params_opt)
                                # Botão para aplicar (requereria re-rodar o backtest com estes params)
                                # if st.button("Aplicar Otimizados"): ...
                           else: st.warning("Não foi possível otimizar.")
                      else: st.error("Otimização não definida para esta estratégia.")


    # --- Aba 4: Cenários ---
    with tabs[4]:
        st.subheader("🌍 Simulação de Cenários de Mercado")
        event = st.selectbox("Selecione um Cenário:", ["Normal", "Halving (Pós)", "Crash (Súbito)", "ETF Approval (Impacto)"], key="scen_sel")
        if 'prices' in data and not data['prices'].empty:
            df_scenario_base = data['prices'].tail(120).copy() # Base para simulação
            simulated_prices = simulate_event(event, df_scenario_base['price'])
            if simulated_prices is not None:
                fig_scenario = go.Figure()
                fig_scenario.add_trace(go.Scatter(x=df_scenario_base['date'], y=df_scenario_base['price'], name="Preço Base", line=dict(color='grey')))
                fig_scenario.add_trace(go.Scatter(x=df_scenario_base['date'], y=simulated_prices, name=f"Projeção: {event}", line=dict(color='orange', dash='dash')))
                # Adicionar OBs atuais
                _, blocks_scen = identify_order_blocks(df_scenario_base, **st.session_state.user_settings)
                fig_scenario = plot_order_blocks(fig_scenario, blocks_scen, df_scenario_base['price'].iloc[-1])
                fig_scenario.update_layout(title=f"Simulação de Impacto: {event}", hovermode="x unified")
                st.plotly_chart(fig_scenario, use_container_width=True)
            else: st.error("Falha ao gerar simulação.")
        else: st.warning("Dados insuficientes para simulação.")

    # --- Aba 5: IA ---
    with tabs[5]:
        st.header("🤖 Módulos de Inteligência Artificial")
        # LSTM
        st.subheader("🔮 Previsão LSTM (Próximo Dia)")
        if st.button("Treinar/Retreinar Modelo LSTM", key="btn_lstm_train"):
             if 'prices_full' in data and len(data['prices_full']) > st.session_state.user_settings['lstm_window']:
                  with st.spinner("Treinando LSTM..."):
                       model, scaler = train_lstm_model(data['prices_full'], **st.session_state.user_settings)
                       if model and scaler:
                            st.session_state.lstm_model = model
                            st.session_state.lstm_scaler = scaler
                            st.success("Modelo LSTM treinado!")
                       else: st.error("Falha no treino LSTM.")
             else: st.error("Dados históricos insuficientes.")

        if 'lstm_model' in st.session_state and 'lstm_scaler' in st.session_state:
             with st.spinner("Gerando previsão LSTM..."):
                  pred_price = predict_with_lstm(st.session_state.lstm_model, st.session_state.lstm_scaler, data['prices'], st.session_state.user_settings['lstm_window'])
             if not pd.isna(pred_price):
                 current_price = data['prices']['price'].iloc[-1]
                 change_pct = ((pred_price / current_price) - 1) * 100 if current_price else 0
                 c1, c2 = st.columns(2)
                 c1.metric("Preço Atual", f"${current_price:,.2f}")
                 c2.metric("Previsão LSTM", f"${pred_price:,.2f}", f"{change_pct:.2f}%")
                 # Gráfico LSTM
                 fig_lstm = go.Figure()
                 df_plot_lstm = data['prices'].tail(90)
                 fig_lstm.add_trace(go.Scatter(x=df_plot_lstm['date'], y=df_plot_lstm['price'], name="Histórico"))
                 last_date = df_plot_lstm['date'].iloc[-1]
                 fig_lstm.add_trace(go.Scatter(x=[last_date, last_date + timedelta(days=1)], y=[current_price, pred_price], name="Previsão", line=dict(color='red', dash='dot')))
                 st.plotly_chart(fig_lstm, use_container_width=True)
             else: st.warning("Falha na previsão LSTM.")
        else: st.info("Treine o modelo LSTM para ver previsões.")
        st.divider()

        # Análise de Sentimento (BERT)
        st.subheader("📰 Análise de Sentimento (BERT)")
        if analyzed_news:
             st.caption(f"Notícias filtradas por confiança >= {st.session_state.user_settings['min_confidence']:.0%}")
             for news in filtered_news[:5]: # Mostrar 5
                  senti_label = news.get('sentiment', 'N/A')
                  score = news.get('sentiment_score', 0)
                  color = "green" if senti_label == 'POSITIVE' else "red" if senti_label == 'NEGATIVE' else "grey"
                  st.markdown(f"<small style='color:{color};'>● [{senti_label} {score:.0%}]</small> **{news['title']}** <small>({news.get('source', 'N/A')})</small>", unsafe_allow_html=True)
        else: st.info("Nenhuma notícia para analisar ou modelo indisponível.")
        st.divider()

        # Reinforcement Learning (RL)
        st.subheader("🤖 Trading com RL (Experimental)")
        if st.button("Treinar/Retreinar Agente RL", key="btn_rl_train"):
            if 'prices' in data and len(data['prices']) > 50: # Mínimo de dados para RL
                df_rl_train = data['prices'].copy()
                # Garantir colunas necessárias para o Env
                cols_ok = True
                for col in ['price', 'volume', f'RSI_{DEFAULT_SETTINGS["rsi_window"]}', 'MACD', 'MACD_Signal', f'BB_Upper_{DEFAULT_SETTINGS["bb_window"]}', f'BB_Lower_{DEFAULT_SETTINGS["bb_window"]}']:
                    if col not in df_rl_train.columns:
                         if col in data['prices_full'].columns:
                              df_rl_train[col] = data['prices_full'][col].reindex(df_rl_train.index)
                         else:
                              st.error(f"Coluna '{col}' necessária para RL não encontrada.")
                              cols_ok = False
                              break
                if cols_ok:
                     with st.spinner(f"Treinando RL ({st.session_state.user_settings['rl_episodes']} timesteps)..."):
                          try:
                               env = BitcoinTradingEnv(df_rl_train.dropna()) # Usar dados recentes limpos
                               vec_env = DummyVecEnv([lambda: env])
                               rl_model = PPO('MlpPolicy', vec_env, verbose=0, **{'learning_rate':1e-4, 'n_steps':1024, 'batch_size':64, 'n_epochs':4}) # Parâmetros padrão
                               rl_model.learn(total_timesteps=st.session_state.user_settings['rl_episodes'])
                               st.session_state.rl_model = rl_model
                               st.success("Agente RL treinado!")
                          except Exception as e:
                               st.error(f"Erro Treino RL: {e}")
                               st.error(traceback.format_exc())
            else: st.error("Dados insuficientes para treinar RL.")

        if 'rl_model' in st.session_state:
             st.subheader("Simulação RL")
             with st.spinner("Executando simulação RL..."):
                  try:
                      df_rl_sim = data['prices'].copy() # Usar mesmos dados recentes
                      cols_ok = True
                      for col in ['price', 'volume', f'RSI_{DEFAULT_SETTINGS["rsi_window"]}', 'MACD', 'MACD_Signal', f'BB_Upper_{DEFAULT_SETTINGS["bb_window"]}', f'BB_Lower_{DEFAULT_SETTINGS["bb_window"]}']:
                          if col not in df_rl_sim.columns:
                               if col in data['prices_full'].columns: df_rl_sim[col] = data['prices_full'][col].reindex(df_rl_sim.index)
                               else: cols_ok = False; break
                      if cols_ok:
                          sim_env = BitcoinTradingEnv(df_rl_sim.dropna())
                          obs, info = sim_env.reset()
                          terminated, truncated = False, False
                          log = {'date': [], 'price': [], 'action': [], 'reward': [], 'portfolio': []}
                          while not terminated and not truncated:
                              action, _ = st.session_state.rl_model.predict(obs, deterministic=True)
                              step_idx = sim_env.current_step # Pega step antes de avançar
                              obs, reward, terminated, truncated, info = sim_env.step(action)
                              # Log
                              if step_idx < len(sim_env.df): # Logar dados do step que acabou
                                  log['date'].append(sim_env.df.loc[step_idx, 'date'])
                                  log['price'].append(sim_env.df.loc[step_idx, 'price'])
                                  log['action'].append(action)
                                  log['reward'].append(reward)
                                  log['portfolio'].append(info.get('portfolio_value', 0))

                          sim_df = pd.DataFrame(log)
                          final_profit = info.get('total_profit', 0)
                          st.metric("Resultado Simulação RL", f"${final_profit:,.2f}", f"{final_profit/sim_env.initial_balance:.2%}")

                          # Gráficos RL
                          fig_rl_act = px.line(sim_df, x='date', y='price', title="Ações Agente RL")
                          fig_rl_act.add_trace(go.Scatter(x=sim_df['date'][sim_df['action']==1], y=sim_df['price'][sim_df['action']==1], mode='markers', name='Buy', marker=dict(color='green', symbol='triangle-up', size=8)))
                          fig_rl_act.add_trace(go.Scatter(x=sim_df['date'][sim_df['action']==2], y=sim_df['price'][sim_df['action']==2], mode='markers', name='Sell', marker=dict(color='red', symbol='triangle-down', size=8)))
                          st.plotly_chart(fig_rl_act, use_container_width=True)

                          fig_rl_port = px.line(sim_df, x='date', y='portfolio', title="Evolução Portfólio RL")
                          st.plotly_chart(fig_rl_port, use_container_width=True)
                      else: st.error("Faltam colunas para simulação RL.")
                  except Exception as e:
                       st.error(f"Erro Simulação RL: {e}")
                       st.error(traceback.format_exc())
        else: st.info("Treine o agente RL para ver a simulação.")


    # --- Aba 6: Técnico Detalhado ---
    with tabs[6]:
        st.subheader("📉 Gráficos Técnicos Detalhados")
        df_tech = data['prices']
        if df_tech is None or df_tech.empty: st.warning("Dados técnicos indisponíveis.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                # RSI
                rsi_col = f'RSI_{st.session_state.user_settings["rsi_window"]}'
                if rsi_col in df_tech:
                    fig = px.line(df_tech, x="date", y=rsi_col, title=f"RSI ({st.session_state.user_settings['rsi_window']})")
                    fig.add_hline(y=70, line_dash="dot", line_color="red"); fig.add_hline(y=30, line_dash="dot", line_color="green")
                    st.plotly_chart(fig, use_container_width=True)
                # MACD
                if 'MACD' in df_tech:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_tech['date'], y=df_tech['MACD'], name='MACD'))
                    fig.add_trace(go.Scatter(x=df_tech['date'], y=df_tech['MACD_Signal'], name='Signal'))
                    colors = ['green' if v >= 0 else 'red' for v in df_tech['MACD_Hist'].fillna(0)]
                    fig.add_trace(go.Bar(x=df_tech['date'], y=df_tech['MACD_Hist'], name='Hist.', marker_color=colors))
                    fig.update_layout(title="MACD (12, 26, 9)")
                    st.plotly_chart(fig, use_container_width=True)
                # OBV
                if 'OBV' in df_tech:
                     fig = px.line(df_tech, x='date', y='OBV', title='On-Balance Volume')
                     st.plotly_chart(fig, use_container_width=True)
                # Zonas S/R
                if 'support_resistance' in data and data['support_resistance']:
                    fig = go.Figure(go.Scatter(x=df_tech['date'], y=df_tech['price'], name='Preço', line=dict(color='lightgrey')))
                    for level in data['support_resistance']: fig.add_hline(y=level, line_dash="dot", annotation_text=f"${level:,.0f}")
                    fig.update_layout(title=f"Zonas S/R ({len(data['support_resistance'])} clusters)")
                    st.plotly_chart(fig, use_container_width=True)

            with c2:
                 # Bollinger
                 bb_w = st.session_state.user_settings["bb_window"]
                 bb_up, bb_low, bb_ma = f'BB_Upper_{bb_w}', f'BB_Lower_{bb_w}', f'BB_MA_{bb_w}'
                 if bb_up in df_tech:
                      fig = go.Figure()
                      fig.add_trace(go.Scatter(x=df_tech['date'], y=df_tech[bb_up], name='Sup', line=dict(width=1)))
                      fig.add_trace(go.Scatter(x=df_tech['date'], y=df_tech[bb_low], name='Inf', line=dict(width=1), fill='tonexty', fillcolor='rgba(0,100,80,0.1)'))
                      fig.add_trace(go.Scatter(x=df_tech['date'], y=df_tech[bb_ma], name='Média', line=dict(dash='dot')))
                      fig.add_trace(go.Scatter(x=df_tech['date'], y=df_tech['price'], name='Preço', line=dict(color='black')))
                      fig.update_layout(title=f"Bandas de Bollinger ({bb_w}, 2)")
                      st.plotly_chart(fig, use_container_width=True)
                 # Stochastic
                 if 'Stoch_K' in df_tech:
                      fig = go.Figure()
                      fig.add_trace(go.Scatter(x=df_tech['date'], y=df_tech['Stoch_K'], name='%K'))
                      fig.add_trace(go.Scatter(x=df_tech['date'], y=df_tech['Stoch_D'], name='%D'))
                      fig.add_hline(y=80, line_dash="dot", line_color="red"); fig.add_hline(y=20, line_dash="dot", line_color="green")
                      fig.update_layout(title="Stochastic (14, 3)")
                      st.plotly_chart(fig, use_container_width=True)
                 # Volume
                 if 'volume' in df_tech:
                      fig = px.bar(df_tech, x='date', y='volume', title='Volume Diário')
                      st.plotly_chart(fig, use_container_width=True)
                 # Order Blocks
                 fig = px.line(df_tech, x='date', y='price', title='Order Blocks Recentes')
                 _, blocks_tech = identify_order_blocks(df_tech, **st.session_state.user_settings)
                 fig = plot_order_blocks(fig, blocks_tech, df_tech['price'].iloc[-1])
                 st.plotly_chart(fig, use_container_width=True)


    # --- Aba 7: Exportar ---
    with tabs[7]:
        st.subheader("📤 Exportar Relatório e Dados")
        # PDF
        st.markdown("**Relatório Completo em PDF:**")
        if st.button("📄 Gerar Relatório PDF", key='btn_pdf'):
            with st.spinner("Gerando PDF..."):
                pdf_bytes = generate_pdf_report(data, signals, final_verdict, coinank_report)
                if pdf_bytes:
                    st.download_button(label="📥 Baixar PDF", data=pdf_bytes,
                                       file_name=f"BTC_AI_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                                       mime="application/pdf")
                else: st.error("Falha ao gerar PDF.")
        st.divider()
        # Excel
        st.markdown("**Dados Brutos e Processados em Excel:**")
        if st.button("💾 Exportar para Excel", key='btn_excel'):
            with st.spinner("Preparando Excel..."):
                try:
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        if 'prices' in data and not data['prices'].empty: data['prices'].reset_index().to_excel(writer, sheet_name="Dados Recentes", index=False)
                        if 'prices_full' in data and not data['prices_full'].empty: data['prices_full'].reset_index().to_excel(writer, sheet_name="Dados Históricos", index=False)
                        if traditional_assets is not None and not traditional_assets.empty: traditional_assets.to_excel(writer, sheet_name="Ativos Tradicionais", index=False)
                        # Adicionar outros DFs (onchain, OBs, S/R, notícias)
                        if 'hashrate' in data and not data['hashrate'].empty: data['hashrate'].to_excel(writer, sheet_name="Hashrate", index=False)
                        if 'difficulty' in data and not data['difficulty'].empty: data['difficulty'].to_excel(writer, sheet_name="Dificuldade", index=False)
                        _, blocks_ex = identify_order_blocks(data['prices'], **st.session_state.user_settings)
                        if blocks_ex: pd.DataFrame(blocks_ex).to_excel(writer, sheet_name="Order Blocks", index=False)
                        if 'support_resistance' in data and data['support_resistance']: pd.DataFrame({'Level': data['support_resistance']}).to_excel(writer, sheet_name="Suporte Resistencia", index=False)
                        if analyzed_news: pd.DataFrame(analyzed_news).to_excel(writer, sheet_name="Noticias Analisadas", index=False)

                    excel_data = output.getvalue()
                    st.download_button(label="📥 Baixar Excel", data=excel_data,
                                       file_name=f"BTC_AI_Data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                except Exception as e: st.error(f"Erro ao gerar Excel: {e}")


    # --- Legenda Sidebar ---
    st.sidebar.divider()
    st.sidebar.markdown("""
    **📌 Legenda & Fontes:**
    - **Sinais:** 🟢Compra, 🔴Venda, 🟡Neutro. ✅/❌ Forte.
    - **OB:** 🔵Bull, 🟠Bear. 🟢Breaker Suporte, 🔴Breaker Resist.
    - **Div:** 🔺Alta, 🔻Baixa.
    - **Fontes:** Preços (Yahoo Finance), Indicadores (Calculados), Sentimento (Alternative.me, BERT), OnChain (Blockchain.info - Exemplo), Outros (Simulado).
    - **IA:** LSTM, RL (PPO), GP, K-Means.
    """)


if __name__ == "__main__":
    main()
