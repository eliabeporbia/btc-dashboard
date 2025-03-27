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
from sklearn.model_selection import ParameterGrid, GridSearchCV, TimeSeriesSplit
from itertools import product
import re
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ======================
# CONFIGURAÇÕES INICIAIS
# ======================
st.set_page_config(layout="wide", page_title="BTC Super Dashboard Pro+ ML")
st.title("🚀 BTC Super Dashboard Pro+ - Edição Machine Learning")

# Configuração da API de Notícias
NEWS_API_KEY = "9962f95c7c4942279e538f4abc9c2f6b"
NEWS_API_URL = "https://newsapi.org/v2/everything"

# ======================
# NOVAS CONSTANTES
# ======================
INDICATOR_WEIGHTS = {
    'order_blocks': 2.0,
    'gaussian_process': 1.5,
    'rsi': 1.5,
    'macd': 1.3,
    'bollinger': 1.2,
    'volume': 1.1,
    'obv': 1.1,
    'stochastic': 1.1,
    'ma_cross': 1.0,
    'sentiment': 1.4,
    'news_impact': 1.3
}

# ======================
# FUNÇÕES DE ANÁLISE DE NOTÍCIAS E SENTIMENTO
# ======================

def fetch_news(query="Bitcoin", days=7):
    """Busca notícias usando a NewsAPI"""
    try:
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        params = {
            'q': query,
            'from': from_date,
            'sortBy': 'publishedAt',
            'language': 'en',
            'apiKey': NEWS_API_KEY,
            'pageSize': 50
        }
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        response.raise_for_status()
        news_data = response.json()
        return news_data.get('articles', [])
    except Exception as e:
        st.error(f"Erro ao buscar notícias: {str(e)}")
        return []

def analyze_sentiment_textblob(text):
    """Analisa sentimento usando TextBlob"""
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def analyze_sentiment_vader(text):
    """Analisa sentimento usando VADER"""
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)['compound']

def process_news(articles):
    """Processa notícias e calcula sentimentos"""
    processed = []
    for article in articles:
        try:
            title = article.get('title', '')
            description = article.get('description', '') or ''
            content = article.get('content', '') or ''
            
            # Combina título, descrição e conteúdo para análise
            full_text = f"{title}. {description}. {content}"
            
            # Calcula vários scores de sentimento
            tb_score = analyze_sentiment_textblob(full_text)
            vader_score = analyze_sentiment_vader(full_text)
            
            # Score combinado (média ponderada)
            combined_score = (tb_score + vader_score * 1.5) / 2.5
            
            processed.append({
                'title': title,
                'source': article.get('source', {}).get('name', ''),
                'date': pd.to_datetime(article.get('publishedAt')),
                'url': article.get('url', ''),
                'textblob': tb_score,
                'vader': vader_score,
                'sentiment': combined_score,
                'confidence': min(0.99, abs(combined_score) * 2),  # Converte para confiança 0-1
                'impact': classify_impact(combined_score, tb_score, vader_score)
            })
        except Exception as e:
            continue
    
    return pd.DataFrame(processed)

def classify_impact(combined_score, tb_score, vader_score):
    """Classifica o impacto da notícia baseado no sentimento"""
    if combined_score > 0.3 and tb_score > 0.2 and vader_score > 0.3:
        return "Alto Positivo"
    elif combined_score > 0.1:
        return "Positivo"
    elif combined_score < -0.3 and tb_score < -0.2 and vader_score < -0.3:
        return "Alto Negativo"
    elif combined_score < -0.1:
        return "Negativo"
    else:
        return "Neutro"

def detect_volatile_days(news_df, price_data):
    """Detecta dias voláteis baseado em notícias e movimentos de preço"""
    if news_df.empty or price_data.empty:
        return pd.DataFrame()
    
    # Agrupa notícias por dia e calcula média de sentimento
    news_daily = news_df.resample('D', on='date').agg({
        'sentiment': 'mean',
        'confidence': 'mean',
        'impact': lambda x: (x.isin(['Alto Positivo', 'Alto Negativo'])).sum()
    }).rename(columns={'impact': 'high_impact_news'})
    
    # Calcula volatilidade diária (range percentual)
    price_data['date_only'] = pd.to_datetime(price_data['date'].dt.date)
    daily_stats = price_data.groupby('date_only').agg({
        'price': ['min', 'max', 'first', 'last']
    })
    daily_stats.columns = ['_'.join(col).strip() for col in daily_stats.columns.values]
    daily_stats['volatility'] = (daily_stats['price_max'] - daily_stats['price_min']) / daily_stats['price_first']
    
    # Combina dados
    combined = news_daily.join(daily_stats, how='inner')
    
    # Classifica dias voláteis
    combined['is_volatile'] = (combined['volatility'] > combined['volatility'].quantile(0.75)) & \
                             ((combined['high_impact_news'] > 0) | (combined['confidence'] > 0.7))
    
    return combined.reset_index()

# ======================
# FUNÇÕES DE MACHINE LEARNING
# ======================

def prepare_ml_data(data, news_data):
    """Prepara dados para treinamento de modelos de ML"""
    if data.empty or 'prices' not in data or data['prices'].empty:
        return pd.DataFrame()
    
    df = data['prices'].copy()
    
    # Calcula retornos futuros (target)
    df['future_5d_return'] = df['price'].pct_change(5).shift(-5)
    df['target'] = (df['future_5d_return'] > 0).astype(int)
    
    # Adiciona indicadores técnicos como features
    df['rsi'] = calculate_rsi(df['price'], 14)
    df['macd'], df['macd_signal'] = calculate_macd(df['price'])
    df['bb_upper'], df['bb_lower'] = calculate_bollinger_bands(df['price'])
    df['obv'] = calculate_obv(df['price'], df['volume'])
    df['stoch_k'], df['stoch_d'] = calculate_stochastic(df['price'])
    
    # Adiciona médias móveis
    for window in [7, 14, 30, 50]:
        df[f'ma_{window}'] = df['price'].rolling(window).mean()
    
    # Adiciona dados de sentimento se disponíveis
    if not news_data.empty:
        news_daily = news_data.resample('D', on='date').agg({
            'sentiment': 'mean',
            'confidence': 'mean',
            'high_impact_news': 'sum'
        }).reset_index()
        
        df['date_only'] = pd.to_datetime(df['date'].dt.date)
        df = df.merge(news_daily, left_on='date_only', right_on='date', how='left')
        df.drop(columns=['date_only', 'date_y'], inplace=True)
        df.rename(columns={'date_x': 'date'}, inplace=True)
    
    # Remove linhas com valores faltantes
    df.dropna(inplace=True)
    
    return df

def train_signal_efficacy_model(X, y):
    """Treina modelo para prever eficácia dos sinais"""
    try:
        # Divide dados em treino e teste mantendo ordem temporal
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Modelo Random Forest com GridSearch para otimização
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10]
        }
        
        model = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        model.fit(X, y)
        
        return model.best_estimator_, model.best_params_
    except Exception as e:
        st.error(f"Erro ao treinar modelo: {str(e)}")
        return None, None

def optimize_indicator_weights(X, y):
    """Otimiza os pesos dos indicadores usando GridSearchCV"""
    try:
        # Normaliza os dados
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Define o espaço de parâmetros (pesos dos indicadores)
        param_grid = {
            'rsi_weight': [0.5, 1.0, 1.5],
            'macd_weight': [0.5, 1.0, 1.3],
            'bollinger_weight': [0.5, 1.0, 1.2],
            'volume_weight': [0.5, 1.0, 1.1],
            'sentiment_weight': [0.5, 1.0, 1.4]
        }
        
        # Modelo de regressão logística para otimização
        model = LogisticRegression(max_iter=1000)
        
        # TimeSeriesSplit para validação
        tscv = TimeSeriesSplit(n_splits=5)
        
        # GridSearchCV
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        grid.fit(X_scaled, y)
        
        return grid.best_params_
    except Exception as e:
        st.error(f"Erro ao otimizar pesos: {str(e)}")
        return None

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

def identify_order_blocks(df, swing_length=10, show_bull=3, show_bear=3, use_body=True):
    """
    Identifica Order Blocks e Breaker Blocks no estilo LuxAlgo
    """
    if df.empty:
        return df, []
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Identificar swings highs e lows
    if use_body:
        df['swing_high'] = df['close'].rolling(swing_length, center=True).max()
        df['swing_low'] = df['close'].rolling(swing_length, center=True).min()
    else:
        df['swing_high'] = df['high'].rolling(swing_length, center=True).max()
        df['swing_low'] = df['low'].rolling(swing_length, center=True).min()
    
    # Identificar Order Blocks
    blocks = []
    
    # Bullish Order Blocks (compra)
    bullish_blocks = df[df['close'] == df['swing_high']].copy()
    bullish_blocks = bullish_blocks.sort_values('date', ascending=False).head(show_bull)
    
    for idx, row in bullish_blocks.iterrows():
        block_start = row['date'] - pd.Timedelta(days=swing_length//2)
        block_end = row['date'] + pd.Timedelta(days=swing_length//2)
        
        block_df = df[(df['date'] >= block_start) & (df['date'] <= block_end)]
        
        if not block_df.empty:
            high = block_df['high'].max() if not use_body else block_df['close'].max()
            low = block_df['low'].min() if not use_body else block_df['close'].min()
            
            blocks.append({
                'type': 'bullish_ob',
                'start_date': block_start,
                'end_date': block_end,
                'high': high,
                'low': low,
                'trigger_price': row['close'],
                'broken': False,
                'weight': INDICATOR_WEIGHTS['order_blocks']
            })
    
    # Bearish Order Blocks (venda)
    bearish_blocks = df[df['close'] == df['swing_low']].copy()
    bearish_blocks = bearish_blocks.sort_values('date', ascending=False).head(show_bear)
    
    for idx, row in bearish_blocks.iterrows():
        block_start = row['date'] - pd.Timedelta(days=swing_length//2)
        block_end = row['date'] + pd.Timedelta(days=swing_length//2)
        
        block_df = df[(df['date'] >= block_start) & (df['date'] <= block_end)]
        
        if not block_df.empty:
            high = block_df['high'].max() if not use_body else block_df['close'].max()
            low = block_df['low'].min() if not use_body else block_df['close'].min()
            
            blocks.append({
                'type': 'bearish_ob',
                'start_date': block_start,
                'end_date': block_end,
                'high': high,
                'low': low,
                'trigger_price': row['close'],
                'broken': False,
                'weight': INDICATOR_WEIGHTS['order_blocks']
            })
    
    # Verificar Breaker Blocks
    for block in blocks:
        if block['type'] == 'bullish_ob':
            # Verificar se o preço fechou abaixo do bloco (tornando-se um breaker)
            subsequent_data = df[df['date'] > block['end_date']]
            if not subsequent_data.empty:
                if subsequent_data['close'].min() < block['low']:
                    block['broken'] = True
                    block['breaker_type'] = 'bullish_breaker'
        
        elif block['type'] == 'bearish_ob':
            # Verificar se o preço fechou acima do bloco (tornando-se um breaker)
            subsequent_data = df[df['date'] > block['end_date']]
            if not subsequent_data.empty:
                if subsequent_data['close'].max() > block['high']:
                    block['broken'] = True
                    block['breaker_type'] = 'bearish_breaker'
    
    return df, blocks

# ... (continua com as outras funções existentes como plot_order_blocks, detect_support_resistance_clusters, etc.)

# ======================
# CARREGAMENTO DE DADOS (ATUALIZADO)
# ======================

@st.cache_data(ttl=3600, show_spinner="Carregando dados do mercado...")
def load_data():
    data = {}
    try:
        # Carrega dados de preço do BTC
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=90"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        market_data = response.json()
        
        data['prices'] = pd.DataFrame(market_data["prices"], columns=["timestamp", "price"])
        data['prices']["date"] = pd.to_datetime(data['prices']["timestamp"], unit="ms")
        data['prices']['close'] = data['prices']['price']  # Para compatibilidade com Order Blocks
        data['prices']['high'] = data['prices']['price'] * 1.01  # Simulando high/low
        data['prices']['low'] = data['prices']['price'] * 0.99
        data['prices']['open'] = data['prices']['price'] * 1.005
        data['prices']['volume'] = np.random.randint(10000, 50000, size=len(data['prices']))
        
        # Carrega notícias e processa sentimentos
        news_articles = fetch_news(days=30)
        data['news_raw'] = news_articles
        data['news_processed'] = process_news(news_articles)
        
        # Detecta dias voláteis
        if not data['news_processed'].empty:
            data['volatile_days'] = detect_volatile_days(data['news_processed'], data['prices'])
        
        # Prepara dados para ML
        data['ml_data'] = prepare_ml_data(data, data['news_processed'] if 'news_processed' in data else pd.DataFrame())
        
        # Carrega outros dados (hashrate, dificuldade, etc.)
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
        
        # Dados simulados de exchanges e whales
        data['exchanges'] = {
            "binance": {"inflow": 1500, "outflow": 1200, "reserves": 500000},
            "coinbase": {"inflow": 800, "outflow": 750, "reserves": 350000},
            "kraken": {"inflow": 600, "outflow": 550, "reserves": 200000}
        }
        
        data['whale_alert'] = pd.DataFrame({
            "date": pd.date_range(end=datetime.now(), periods=5, freq='12H'),
            "amount": np.random.randint(100, 500, 5),
            "exchange": ["Binance", "Coinbase", "Kraken", "Unknown", "Binance"]
        })
        
    except requests.exceptions.RequestException as e:
        st.error(f"Erro na requisição à API: {str(e)}")
        data['prices'] = pd.DataFrame()
    except Exception as e:
        st.error(f"Erro ao processar dados: {str(e)}")
        data['prices'] = pd.DataFrame()
    
    return data

# ======================
# GERAÇÃO DE SINAIS (ATUALIZADA COM ML)
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
        
        # 1. Sinais Técnicos Tradicionais
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
            signals.append((name, signal, f"{change:.2%}", INDICATOR_WEIGHTS['ma_cross']))
        
        # RSI
        rsi_col = f'RSI_{rsi_window}'
        if rsi_col not in data['prices'].columns:
            data['prices'][rsi_col] = calculate_rsi(data['prices']['price'], rsi_window)
        
        if not data['prices'][rsi_col].isna().all():
            rsi = data['prices'][rsi_col].iloc[-1]
            rsi_signal = "COMPRA" if rsi < 30 else "VENDA" if rsi > 70 else "NEUTRO"
            signals.append((f"RSI ({rsi_window})", rsi_signal, f"{rsi:.2f}", INDICATOR_WEIGHTS['rsi']))
        
        # MACD
        if 'MACD' in data['prices'].columns and not data['prices']['MACD'].isna().all():
            macd = data['prices']['MACD'].iloc[-1]
            macd_signal = "COMPRA" if macd > 0 else "VENDA"
            signals.append(("MACD", macd_signal, f"{macd:.2f}", INDICATOR_WEIGHTS['macd']))
        
        # Bollinger Bands
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
            signals.append((f"Bollinger Bands ({bb_window})", bb_signal, f"Atual: ${last_price:,.0f}", INDICATOR_WEIGHTS['bollinger']))
        
        # Volume
        if 'volume' in data['prices'].columns:
            volume_ma = data['prices']['volume'].rolling(20).mean().iloc[-1]
            last_volume = data['prices']['volume'].iloc[-1]
            volume_ratio = last_volume / volume_ma
            volume_signal = "COMPRA" if volume_ratio > 1.5 and last_price > data['prices']['price'].iloc[-2] else "VENDA" if volume_ratio > 1.5 and last_price < data['prices']['price'].iloc[-2] else "NEUTRO"
            signals.append(("Volume (20MA)", volume_signal, f"{volume_ratio:.1f}x", INDICATOR_WEIGHTS['volume']))
        
        # OBV
        if 'OBV' in data['prices'].columns:
            obv_ma = data['prices']['OBV'].rolling(20).mean().iloc[-1]
            last_obv = data['prices']['OBV'].iloc[-1]
            obv_signal = "COMPRA" if last_obv > obv_ma and last_price > data['prices']['price'].iloc[-2] else "VENDA" if last_obv < obv_ma and last_price < data['prices']['price'].iloc[-2] else "NEUTRO"
            signals.append(("OBV (20MA)", obv_signal, f"{last_obv/1e6:.1f}M", INDICATOR_WEIGHTS['obv']))
        
        # Stochastic
        if 'Stoch_K' in data['prices'].columns and 'Stoch_D' in data['prices'].columns:
            stoch_k = data['prices']['Stoch_K'].iloc[-1]
            stoch_d = data['prices']['Stoch_D'].iloc[-1]
            stoch_signal = "COMPRA" if stoch_k < 20 and stoch_d < 20 else "VENDA" if stoch_k > 80 and stoch_d > 80 else "NEUTRO"
            signals.append(("Stochastic (14,3)", stoch_signal, f"K:{stoch_k:.1f}, D:{stoch_d:.1f}", INDICATOR_WEIGHTS['stochastic']))
        
        # Gaussian Process
        if 'GP_Prediction' in data['prices'].columns and not data['prices']['GP_Prediction'].isna().all():
            gp_pred = data['prices']['GP_Prediction'].iloc[-1]
            gp_signal = "COMPRA" if gp_pred > last_price * 1.03 else "VENDA" if gp_pred < last_price * 0.97 else "NEUTRO"
            signals.append(("Gaussian Process", gp_signal, f"Previsão: ${gp_pred:,.0f}", INDICATOR_WEIGHTS['gaussian_process']))
        
        # Order Blocks
        if 'prices' in data and not data['prices'].empty:
            _, blocks = identify_order_blocks(
                data['prices'],
                swing_length=st.session_state.user_settings['ob_swing_length'],
                show_bull=st.session_state.user_settings['ob_show_bull'],
                show_bear=st.session_state.user_settings['ob_show_bear'],
                use_body=st.session_state.user_settings['ob_use_body']
            )
            
            for block in blocks:
                if not block['broken']:
                    if block['type'] == 'bullish_ob':
                        if last_price >= block['low'] and last_price <= block['high']:
                            signals.append((f"Order Block (Compra)", "COMPRA", f"Zona: ${block['low']:,.0f}-${block['high']:,.0f}", block['weight']))
                    elif block['type'] == 'bearish_ob':
                        if last_price >= block['low'] and last_price <= block['high']:
                            signals.append((f"Order Block (Venda)", "VENDA", f"Zona: ${block['low']:,.0f}-${block['high']:,.0f}", block['weight']))
                else:
                    if block['breaker_type'] == 'bullish_breaker':
                        if last_price >= block['low'] * 0.99 and last_price <= block['high'] * 1.01:
                            signals.append((f"Breaker Block (Resistência)", "VENDA", f"Zona: ${block['low']:,.0f}-${block['high']:,.0f}", block['weight']))
                    elif block['breaker_type'] == 'bearish_breaker':
                        if last_price >= block['low'] * 0.99 and last_price <= block['high'] * 1.01:
                            signals.append((f"Breaker Block (Suporte)", "COMPRA", f"Zona: ${block['low']:,.0f}-${block['high']:,.0f}", block['weight']))
        
        # Divergências RSI
        if 'RSI_Divergence' in data['prices'].columns:
            last_div = data['prices']['RSI_Divergence'].iloc[-1]
            if last_div == 1:
                signals.append(("Divergência de Alta (RSI)", "COMPRA", "Preço caindo e RSI subindo", 1.2))
            elif last_div == -1:
                signals.append(("Divergência de Baixa (RSI)", "VENDA", "Preço subindo e RSI caindo", 1.2))
        
        # 2. Sinais de Sentimento e Notícias (se disponíveis)
        if 'news_processed' in data and not data['news_processed'].empty:
            last_news = data['news_processed'].iloc[-1]
            news_signal = "COMPRA" if last_news['sentiment'] > 0.2 else "VENDA" if last_news['sentiment'] < -0.2 else "NEUTRO"
            signals.append(("Sentimento de Notícias", news_signal, 
                          f"Score: {last_news['sentiment']:.2f}", 
                          INDICATOR_WEIGHTS['sentiment'] * last_news['confidence']))
            
            # Impacto de notícias de alto impacto
            high_impact_news = data['news_processed'][
                data['news_processed']['impact'].isin(['Alto Positivo', 'Alto Negativo'])
            ].tail(3)
            
            for _, news in high_impact_news.iterrows():
                news_signal = "COMPRA" if news['impact'] == 'Alto Positivo' else "VENDA"
                signals.append((f"Notícia: {news['title'][:30]}...", news_signal, 
                               f"Impacto: {news['impact']}", 
                               INDICATOR_WEIGHTS['news_impact'] * news['confidence']))
        
        # 3. Ajuste baseado em ML (se modelo disponível)
        if 'ml_model' in st.session_state and 'ml_data' in data and not data['ml_data'].empty:
            try:
                # Prepara features para predição
                last_data = data['ml_data'].iloc[-1]
                features = last_data.drop(['date', 'future_5d_return', 'target']).values.reshape(1, -1)
                
                # Faz predição
                prediction = st.session_state.ml_model.predict(features)[0]
                proba = st.session_state.ml_model.predict_proba(features)[0][1]
                
                ml_signal = "COMPRA" if prediction == 1 else "VENDA"
                signals.append(("Modelo ML (5 dias)", ml_signal, 
                              f"Probabilidade: {proba:.0%}", 
                              INDICATOR_WEIGHTS['gaussian_process'] * proba))
            except Exception as e:
                st.error(f"Erro na predição ML: {str(e)}")
    
    except Exception as e:
        st.error(f"Erro ao gerar sinais: {str(e)}")
        return signals, "➖ ERRO NA ANÁLISE", buy_signals, sell_signals
    
    # Calcular sinais ponderados
    weighted_buy = sum(s[3] for s in signals if s[1] == "COMPRA")
    weighted_sell = sum(s[3] for s in signals if s[1] == "VENDA")
    
    buy_signals = sum(1 for s in signals if s[1] == "COMPRA")
    sell_signals = sum(1 for s in signals if s[1] == "VENDA")
    
    if weighted_buy >= weighted_sell * 1.5:
        final_verdict = "✅ FORTE COMPRA"
    elif weighted_buy > weighted_sell:
        final_verdict = "📈 COMPRA"
    elif weighted_sell >= weighted_buy * 1.5:
        final_verdict = "❌ FORTE VENDA"
    elif weighted_sell > weighted_buy:
        final_verdict = "📉 VENDA"
    else:
        final_verdict = "➖ NEUTRO"
    
    return signals, final_verdict, buy_signals, sell_signals

# ======================
# INTERFACE DO USUÁRIO (ATUALIZADA)
# ======================

# Carrega dados
data = load_data()

# Configurações padrão
DEFAULT_SETTINGS = {
    'rsi_window': 14,
    'bb_window': 20,
    'ma_windows': [7, 30, 200],
    'email': '',
    'gp_window': 30,
    'gp_lookahead': 5,
    'ob_swing_length': 10,
    'ob_show_bull': 3,
    'ob_show_bear': 3,
    'ob_use_body': True,
    'min_confidence': 0.7,
    'n_clusters': 5,
    'news_days': 7,
    'ml_enabled': True
}

if 'user_settings' not in st.session_state:
    st.session_state.user_settings = DEFAULT_SETTINGS.copy()

# Sidebar
st.sidebar.header("⚙️ Painel de Controle")

# Seção de Configurações Técnicas
st.sidebar.subheader("🔧 Parâmetros Técnicos")
rsi_window = st.sidebar.slider("Período do RSI", 7, 21, st.session_state.user_settings['rsi_window'])
bb_window = st.sidebar.slider("Janela das Bandas de Bollinger", 10, 50, st.session_state.user_settings['bb_window'])
ma_windows = st.sidebar.multiselect("Médias Móveis para Exibir", [7, 20, 30, 50, 100, 200], st.session_state.user_settings['ma_windows'])

# Seção de Order Blocks
st.sidebar.subheader("📊 Order Blocks (LuxAlgo)")
ob_swing_length = st.sidebar.slider("Swing Lookback", 5, 20, st.session_state.user_settings['ob_swing_length'])
ob_show_bull = st.sidebar.slider("Mostrar últimos Bullish OBs", 1, 5, st.session_state.user_settings['ob_show_bull'])
ob_show_bear = st.sidebar.slider("Mostrar últimos Bearish OBs", 1, 5, st.session_state.user_settings['ob_show_bear'])
ob_use_body = st.sidebar.checkbox("Usar corpo do candle", st.session_state.user_settings['ob_use_body'])

# Seção de Notícias e Sentimento
st.sidebar.subheader("📰 Análise de Notícias")
news_days = st.sidebar.slider("Dias para buscar notícias", 1, 30, st.session_state.user_settings['news_days'])
min_confidence = st.sidebar.slider("Confiança Mínima para Notícias", 0.0, 1.0, st.session_state.user_settings['min_confidence'], 0.05)

# Seção de Machine Learning
st.sidebar.subheader("🤖 Machine Learning")
ml_enabled = st.sidebar.checkbox("Ativar Modelo de ML", st.session_state.user_settings['ml_enabled'])

# Botões de ação
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("💾 Salvar Configurações"):
        st.session_state.user_settings = {
            'rsi_window': rsi_window,
            'bb_window': bb_window,
            'ma_windows': ma_windows,
            'gp_window': st.session_state.user_settings['gp_window'],
            'gp_lookahead': st.session_state.user_settings['gp_lookahead'],
            'ob_swing_length': ob_swing_length,
            'ob_show_bull': ob_show_bull,
            'ob_show_bear': ob_show_bear,
            'ob_use_body': ob_use_body,
            'min_confidence': min_confidence,
            'n_clusters': st.session_state.user_settings['n_clusters'],
            'news_days': news_days,
            'ml_enabled': ml_enabled
        }
        st.sidebar.success("Configurações salvas com sucesso!")
        
with col2:
    if st.button("🔄 Resetar"):
        st.session_state.user_settings = DEFAULT_SETTINGS.copy()
        st.sidebar.success("Configurações resetadas para padrão!")
        st.rerun()

# Treinar modelo de ML se dados disponíveis
if ml_enabled and 'ml_data' in data and not data['ml_data'].empty:
    with st.spinner("Treinando modelo de ML..."):
        X = data['ml_data'].drop(columns=['date', 'future_5d_return', 'target'])
        y = data['ml_data']['target']
        
        model, best_params = train_signal_efficacy_model(X, y)
        if model:
            st.session_state.ml_model = model
            st.sidebar.success(f"Modelo treinado! Acurácia: {accuracy_score(y, model.predict(X)):.0%}")
        else:
            st.sidebar.warning("Não foi possível treinar o modelo")

# Otimizar pesos dos indicadores
if st.sidebar.button("🔍 Otimizar Pesos dos Indicadores"):
    if 'ml_data' in data and not data['ml_data'].empty:
        with st.spinner("Otimizando pesos dos indicadores..."):
            X = data['ml_data'][['rsi', 'macd', 'bb_upper', 'volume', 'sentiment']].dropna()
            y = data['ml_data'].loc[X.index, 'target']
            
            best_weights = optimize_indicator_weights(X, y)
            if best_weights:
                # Atualiza pesos globais com os otimizados
                INDICATOR_WEIGHTS['rsi'] = best_weights.get('rsi_weight', 1.5)
                INDICATOR_WEIGHTS['macd'] = best_weights.get('macd_weight', 1.3)
                INDICATOR_WEIGHTS['bollinger'] = best_weights.get('bollinger_weight', 1.2)
                INDICATOR_WEIGHTS['volume'] = best_weights.get('volume_weight', 1.1)
                INDICATOR_WEIGHTS['sentiment'] = best_weights.get('sentiment_weight', 1.4)
                
                st.sidebar.success("Pesos otimizados com sucesso!")
                st.sidebar.json({k: round(v, 2) for k, v in INDICATOR_WEIGHTS.items()})
            else:
                st.sidebar.warning("Não foi possível otimizar os pesos")
    else:
        st.sidebar.warning("Dados insuficientes para otimização")

# Gera sinais
signals, final_verdict, buy_signals, sell_signals = generate_signals(
    data, 
    rsi_window=st.session_state.user_settings['rsi_window'],
    bb_window=st.session_state.user_settings['bb_window']
)

# Layout principal
st.header("📊 Painel Integrado BTC Pro+ ML")

# Métricas rápidas
col1, col2, col3, col4, col5 = st.columns(5)
if 'prices' in data and not data['prices'].empty:
    col1.metric("Preço BTC", f"${data['prices']['price'].iloc[-1]:,.2f}")
else:
    col1.metric("Preço BTC", "N/A")

sentiment = get_market_sentiment()
col2.metric("Sentimento", f"{sentiment['value']}/100", sentiment['sentiment'])

if 'news_processed' in data and not data['news_processed'].empty:
    avg_sentiment = data['news_processed']['sentiment'].mean()
    col3.metric("Sentimento Notícias", f"{avg_sentiment:.2f}", 
               "Positivo" if avg_sentiment > 0 else "Negativo" if avg_sentiment < 0 else "Neutro")

if 'volatile_days' in data and not data['volatile_days'].empty:
    volatile_count = data['volatile_days']['is_volatile'].sum()
    col4.metric("Dias Voláteis (30d)", f"{volatile_count}", 
               "Alta Volatilidade" if volatile_count > 5 else "Normal")

col5.metric("Análise Final", final_verdict)

# Abas principais
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Mercado", "📰 Notícias", "🆚 Comparativos", "🧪 Backtesting", 
    "🌍 Cenários", "🤖 ML Insights", "📤 Exportar"
])

with tab1:
    # ... (mantém a mesma implementação da aba Mercado anterior)
    pass

with tab2:
    st.subheader("📰 Análise de Notícias em Tempo Real")
    
    if 'news_processed' in data and not data['news_processed'].empty:
        # Filtra notícias por confiança
        filtered_news = data['news_processed'][
            data['news_processed']['confidence'] >= st.session_state.user_settings['min_confidence']
        ].sort_values('date', ascending=False)
        
        # Mostra estatísticas de sentimento
        st.plotly_chart(px.line(
            filtered_news, 
            x='date', 
            y='sentiment', 
            color='impact',
            title='Evolução do Sentimento das Notícias',
            hover_data=['title']
        ), use_container_width=True)
        
        # Mostra notícias recentes
        st.subheader("📌 Notícias Recentes")
        for _, news in filtered_news.head(10).iterrows():
            with st.expander(f"{news['date'].strftime('%Y-%m-%d')} | {news['title']} ({news['source']})"):
                st.markdown(f"**Sentimento**: {news['sentiment']:.2f} ({news['impact']})")
                st.markdown(f"**Confiança**: {news['confidence']:.0%}")
                st.markdown(f"[Leia mais]({news['url']})")
    else:
        st.warning("Nenhuma notícia disponível ou não foi possível carregar notícias")

with tab3:
    # ... (mantém a mesma implementação da aba Comparativos anterior)
    pass

with tab4:
    # ... (mantém a mesma implementação da aba Backtesting anterior)
    pass

with tab5:
    # ... (mantém a mesma implementação da aba Cenários anterior)
    pass

with tab6:
    st.subheader("🤖 Insights de Machine Learning")
    
    if 'ml_model' in st.session_state and 'ml_data' in data and not data['ml_data'].empty:
        # Feature importance
        if hasattr(st.session_state.ml_model, 'feature_importances_'):
            importance = pd.DataFrame({
                'Feature': data['ml_data'].drop(columns=['date', 'future_5d_return', 'target']).columns,
                'Importance': st.session_state.ml_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.plotly_chart(px.bar(
                importance, 
                x='Feature', 
                y='Importance',
                title='Importância das Features no Modelo'
            ), use_container_width=True)
        
        # Performance do modelo
        X = data['ml_data'].drop(columns=['date', 'future_5d_return', 'target'])
        y = data['ml_data']['target']
        y_pred = st.session_state.ml_model.predict(X)
        
        st.write("**Relatório de Classificação:**")
        st.text(classification_report(y, y_pred))
        
        # Probabilidades de predição
        proba = st.session_state.ml_model.predict_proba(X)[:, 1]
        data['ml_data']['pred_proba'] = proba
        
        st.plotly_chart(px.line(
            data['ml_data'], 
            x='date', 
            y=['pred_proba', 'target'],
            title='Probabilidade de Predição vs Real',
            labels={'value': 'Probabilidade/Real'}
        ), use_container_width=True)
    else:
        st.warning("Nenhum modelo de ML treinado ou dados insuficientes")

with tab7:
    # ... (mantém a mesma implementação da aba Exportar anterior)
    pass

# Rodapé
st.sidebar.markdown("""
**📌 Legenda:**
- 🟢 **COMPRA**: Indicador positivo
- 🔴 **VENDA**: Indicador negativo
- 🟡 **NEUTRO**: Sem sinal claro
- ✅ **FORTE COMPRA**: 1.5x mais sinais ponderados
- ❌ **FORTE VENDA**: 1.5x mais sinais ponderados
- 💬 **NOTÍCIAS**: Impacto de notícias no sentimento
- 🤖 **ML**: Previsão baseada em machine learning
""")
