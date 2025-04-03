import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# ======================
# CONFIGURAÇÕES INICIAIS
# ======================
st.set_page_config(layout="wide", page_title="BTC Confluence Pro", page_icon="🚀")
st.title("💰 BTC Confluence Pro - 10 Indicadores Essenciais")

# ======================
# CONSTANTES
# ======================
INDICATOR_WEIGHTS = {
    'order_blocks': 1.8,
    'fibonacci': 1.5,
    'ichimoku': 1.7,
    'volume_profile': 1.4,
    'cumulative_delta': 1.6,
    'rsi': 1.3,
    'macd': 1.5,
    'bollinger': 1.2,
    'ma_200': 1.0,
    'divergence': 1.1
}

DEFAULT_SETTINGS = {
    'rsi_window': 14,
    'bb_window': 20,
    'fib_levels': ['0.236', '0.382', '0.5', '0.618', '0.786'],
    'ichimoku_params': {'tenkan': 9, 'kijun': 26, 'senkou': 52},
    'vp_bins': 20,
    'ob_swing_length': 10,
    'ob_show_bull': 3,
    'ob_show_bear': 3,
    'ob_use_body': True
}

# ======================
# FUNÇÕES PRINCIPAIS
# ======================
def load_bitcoin_data(days=180):
    """Carrega dados históricos do BTC"""
    data = yf.download("BTC-USD", period=f"{days}d", interval="1d")
    data = data.reset_index()
    data.rename(columns={
        'Date': 'date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)
    return data

def calculate_fibonacci_levels(high, low):
    """Calcula níveis de Fibonacci"""
    return {level: high - (high - low) * float(level) 
            for level in DEFAULT_SETTINGS['fib_levels']}

def calculate_ichimoku(df):
    """Calcula componentes do Ichimoku"""
    params = st.session_state.user_settings['ichimoku_params']
    df['tenkan_sen'] = (df['high'].rolling(params['tenkan']).max() + 
                        df['low'].rolling(params['tenkan']).min()) / 2
    df['kijun_sen'] = (df['high'].rolling(params['kijun']).max() + 
                       df['low'].rolling(params['kijun']).min()) / 2
    df['senkou_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(params['kijun'])
    df['senkou_b'] = ((df['high'].rolling(params['senkou']).max() + 
                       df['low'].rolling(params['senkou']).min()) / 2).shift(params['kijun'])
    return df

def calculate_volume_profile(df):
    """Identifica áreas de alto volume"""
    bins = st.session_state.user_settings['vp_bins']
    vp = []
    min_p, max_p = df['low'].min(), df['high'].max()
    bin_size = (max_p - min_p) / bins
    
    for i in range(bins):
        lower = min_p + i * bin_size
        upper = lower + bin_size
        mask = (df['close'] >= lower) & (df['close'] <= upper)
        vp.append({'price_range': (lower, upper), 
                  'volume': df[mask]['volume'].sum()})
    
    return pd.DataFrame(vp).sort_values('volume', ascending=False)

def calculate_cumulative_delta(df):
    """Calcula fluxo de ordens"""
    df['delta'] = np.where(df['close'] > df['open'], df['volume'], 
                          np.where(df['close'] < df['open'], -df['volume'], 0))
    df['cumulative_delta'] = df['delta'].cumsum()
    return df

def identify_order_blocks(df):
    """Identifica zonas de liquidez"""
    settings = st.session_state.user_settings
    df = df.copy()
    
    if settings['ob_use_body']:
        df['swing_high'] = df['close'].rolling(settings['ob_swing_length'], center=True).max()
        df['swing_low'] = df['close'].rolling(settings['ob_swing_length'], center=True).min()
    else:
        df['swing_high'] = df['high'].rolling(settings['ob_swing_length'], center=True).max()
        df['swing_low'] = df['low'].rolling(settings['ob_swing_length'], center=True).min()
    
    blocks = []
    
    # Bullish Order Blocks
    bullish_blocks = df[df['close'] == df['swing_high']].sort_values('date', ascending=False).head(settings['ob_show_bull'])
    for idx, row in bullish_blocks.iterrows():
        block_start = row['date'] - pd.Timedelta(days=settings['ob_swing_length']//2)
        block_end = row['date'] + pd.Timedelta(days=settings['ob_swing_length']//2)
        block_df = df[(df['date'] >= block_start) & (df['date'] <= block_end)]
        if not block_df.empty:
            high = block_df['high'].max() if not settings['ob_use_body'] else block_df['close'].max()
            low = block_df['low'].min() if not settings['ob_use_body'] else block_df['close'].min()
            blocks.append({
                'type': 'bullish_ob',
                'start_date': block_start,
                'end_date': block_end,
                'high': high,
                'low': low,
                'trigger_price': row['close']
            })
    
    # Bearish Order Blocks
    bearish_blocks = df[df['close'] == df['swing_low']].sort_values('date', ascending=False).head(settings['ob_show_bear'])
    for idx, row in bearish_blocks.iterrows():
        block_start = row['date'] - pd.Timedelta(days=settings['ob_swing_length']//2)
        block_end = row['date'] + pd.Timedelta(days=settings['ob_swing_length']//2)
        block_df = df[(df['date'] >= block_start) & (df['date'] <= block_end)]
        if not block_df.empty:
            high = block_df['high'].max() if not settings['ob_use_body'] else block_df['close'].max()
            low = block_df['low'].min() if not settings['ob_use_body'] else block_df['close'].min()
            blocks.append({
                'type': 'bearish_ob',
                'start_date': block_start,
                'end_date': block_end,
                'high': high,
                'low': low,
                'trigger_price': row['close']
            })
    
    return blocks

def calculate_rsi(series, window=14):
    """Calcula Relative Strength Index"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calcula MACD"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(series, window=20, num_std=2):
    """Calcula Bandas de Bollinger"""
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, lower

def detect_divergence(price, indicator):
    """Detecta divergências entre preço e indicador"""
    price_peaks = price.rolling(5, center=True).max() == price
    indicator_peaks = indicator.rolling(5, center=True).max() == indicator
    bearish_div = (price_peaks & (indicator.diff() < 0)).any()
    
    price_valleys = price.rolling(5, center=True).min() == price
    indicator_valleys = indicator.rolling(5, center=True).min() == indicator
    bullish_div = (price_valleys & (indicator.diff() > 0)).any()
    
    return bullish_div, bearish_div

# ======================
# GERADOR DE SINAIS
# ======================
def generate_signals(df):
    """Gera todos os sinais de confluência"""
    signals = []
    current_price = df['close'].iloc[-1]
    
    # 1. Order Blocks
    blocks = identify_order_blocks(df)
    for block in blocks:
        if block['low'] <= current_price <= block['high']:
            signal_type = "COMPRA" if block['type'] == 'bullish_ob' else "VENDA"
            signals.append((
                f"Order Block ({'Bull' if block['type'] == 'bullish_ob' else 'Bear'})",
                signal_type,
                f"Zona: ${block['low']:,.0f}-${block['high']:,.0f}",
                INDICATOR_WEIGHTS['order_blocks']
            ))
    
    # 2. Fibonacci
    fib_levels = calculate_fibonacci_levels(df['high'].max(), df['low'].min())
    for level in st.session_state.user_settings['fib_levels']:
        if abs(current_price - fib_levels[level]) < current_price * 0.005:
            signal = "COMPRA" if float(level) >= 0.5 else "VENDA"
            signals.append((
                f"Fibonacci {level}",
                signal,
                f"Preço: ${fib_levels[level]:,.0f}",
                INDICATOR_WEIGHTS['fibonacci']
            ))
    
    # 3. Ichimoku
    df = calculate_ichimoku(df)
    tenkan = df['tenkan_sen'].iloc[-1]
    kijun = df['kijun_sen'].iloc[-1]
    if tenkan > kijun and current_price > df['senkou_a'].iloc[-1]:
        signals.append(("Ichimoku", "COMPRA", "TK Cross + Acima da Nuvem", INDICATOR_WEIGHTS['ichimoku']))
    elif tenkan < kijun and current_price < df['senkou_b'].iloc[-1]:
        signals.append(("Ichimoku", "VENDA", "TK Cross + Abaixo da Nuvem", INDICATOR_WEIGHTS['ichimoku']))
    
    # 4. Volume Profile
    vp = calculate_volume_profile(df)
    poc = vp.iloc[0]['price_range']
    if poc[0] <= current_price <= poc[1]:
        signals.append(("Volume Profile", "COMPRA", f"POC: ${poc[0]:,.0f}-${poc[1]:,.0f}", INDICATOR_WEIGHTS['volume_profile']))
    
    # 5. Cumulative Delta
    df = calculate_cumulative_delta(df)
    if df['cumulative_delta'].iloc[-1] > df['cumulative_delta'].rolling(20).mean().iloc[-1]:
        signals.append(("Cumulative Delta", "COMPRA", "Fluxo de Compra", INDICATOR_WEIGHTS['cumulative_delta']))
    elif df['cumulative_delta'].iloc[-1] < df['cumulative_delta'].rolling(20).mean().iloc[-1]:
        signals.append(("Cumulative Delta", "VENDA", "Fluxo de Venda", INDICATOR_WEIGHTS['cumulative_delta']))
    
    # 6. RSI
    rsi = calculate_rsi(df['close'], st.session_state.user_settings['rsi_window']).iloc[-1]
    if rsi < 30:
        signals.append(("RSI", "COMPRA", f"Oversold: {rsi:.1f}", INDICATOR_WEIGHTS['rsi']))
    elif rsi > 70:
        signals.append(("RSI", "VENDA", f"Overbought: {rsi:.1f}", INDICATOR_WEIGHTS['rsi']))
    
    # 7. MACD
    macd, signal = calculate_macd(df['close'])
    if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
        signals.append(("MACD", "COMPRA", "Cruzamento de Alta", INDICATOR_WEIGHTS['macd']))
    elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
        signals.append(("MACD", "VENDA", "Cruzamento de Baixa", INDICATOR_WEIGHTS['macd']))
    
    # 8. Bollinger Bands
    upper, lower = calculate_bollinger_bands(df['close'], st.session_state.user_settings['bb_window'])
    if current_price <= lower.iloc[-1]:
        signals.append(("Bollinger", "COMPRA", "Banda Inferior", INDICATOR_WEIGHTS['bollinger']))
    elif current_price >= upper.iloc[-1]:
        signals.append(("Bollinger", "VENDA", "Banda Superior", INDICATOR_WEIGHTS['bollinger']))
    
    # 9. MA 200
    ma_200 = df['close'].rolling(200).mean().iloc[-1]
    if current_price > ma_200:
        signals.append(("MA 200", "COMPRA", f"Preço > ${ma_200:,.0f}", INDICATOR_WEIGHTS['ma_200']))
    else:
        signals.append(("MA 200", "VENDA", f"Preço < ${ma_200:,.0f}", INDICATOR_WEIGHTS['ma_200']))
    
    # 10. Divergência
    bullish_div, bearish_div = detect_divergence(df['close'], rsi)
    if bullish_div:
        signals.append(("Divergência", "COMPRA", "Bullish Hidden", INDICATOR_WEIGHTS['divergence']))
    elif bearish_div:
        signals.append(("Divergência", "VENDA", "Bearish Regular", INDICATOR_WEIGHTS['divergence']))
    
    return signals

# ======================
# INTERFACE GRÁFICA
# ======================
def main():
    # Configurações iniciais
    if 'user_settings' not in st.session_state:
        st.session_state.user_settings = DEFAULT_SETTINGS.copy()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configurações")
        st.session_state.user_settings['rsi_window'] = st.slider("Período RSI", 7, 21, 14)
        st.session_state.user_settings['bb_window'] = st.slider("Bandas de Bollinger", 10, 50, 20)
        st.session_state.user_settings['ob_swing_length'] = st.slider("Order Block Swing", 5, 20, 10)
        st.session_state.user_settings['vp_bins'] = st.slider("Volume Profile Bins", 10, 50, 20)
        st.session_state.user_settings['ob_use_body'] = st.checkbox("Usar corpo do candle (Order Blocks)", True)
    
    # Carregar dados
    df = load_bitcoin_data(180)
    
    # Gerar sinais
    signals = generate_signals(df)
    current_price = df['close'].iloc[-1]
    
    # Layout principal
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Gráfico principal
        fig = go.Figure()
        
        # Preço
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['close'],
            name="BTC/USD",
            line=dict(color='#00BFFF', width=2)
        ))
        
        # Order Blocks
        blocks = identify_order_blocks(df)
        for block in blocks:
            color = 'rgba(0, 100, 255, 0.2)' if block['type'] == 'bullish_ob' else 'rgba(255, 80, 0, 0.2)'
            fig.add_shape(type="rect",
                         x0=block['start_date'], y0=block['low'],
                         x1=block['end_date'], y1=block['high'],
                         fillcolor=color, line=dict(width=0))
        
        # Ichimoku Cloud
        df = calculate_ichimoku(df)
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['senkou_a'],
            fill=None, mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['senkou_b'],
            fill='tonexty', mode='lines',
            line=dict(width=0),
            fillcolor='rgba(100, 100, 255, 0.2)',
            name='Ichimoku Cloud'
        ))
        
        # Fibonacci (apenas últimos níveis relevantes)
        fib_levels = calculate_fibonacci_levels(df['high'].max(), df['low'].min())
        for level in ['0.382', '0.5', '0.618']:
            fig.add_hline(y=fib_levels[level], 
                         line=dict(color='gray', dash='dot'),
                         annotation_text=f"Fib {level}",
                         annotation_position="bottom right")
        
        fig.update_layout(
            title="BTC/USD - Análise de Confluência",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            height=700
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Painel de sinais
        st.subheader("🎯 Sinais Confluentes")
        
        total_weight = sum(s[3] for s in signals)
        buy_weight = sum(s[3] for s in signals if s[1] == "COMPRA")
        sell_weight = sum(s[3] for s in signals if s[1] == "VENDA")
        
        if buy_weight >= sell_weight * 1.5:
            st.success(f"✅ FORTE COMPRA ({buy_weight:.1f}/{total_weight:.1f})")
        elif buy_weight > sell_weight:
            st.info(f"📈 COMPRA ({buy_weight:.1f}/{total_weight:.1f})")
        elif sell_weight >= buy_weight * 1.5:
            st.error(f"❌ FORTE VENDA ({sell_weight:.1f}/{total_weight:.1f})")
        else:
            st.warning(f"📉 VENDA ({sell_weight:.1f}/{total_weight:.1f})")
        
        st.divider()
        
        for signal in signals:
            color = "#4CAF50" if signal[1] == "COMPRA" else "#F44336"
            emoji = "🟢" if signal[1] == "COMPRA" else "🔴"
            st.markdown(f"""
            <div style="background-color:{color}20; padding:10px; border-radius:5px; margin:5px 0;">
                <strong>{emoji} {signal[0]}</strong><br>
                <code>{signal[1]}</code> {signal[2]}<br>
                <small>Peso: {signal[3]:.1f}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Volume Profile
        st.divider()
        st.subheader("📊 Volume Profile")
        vp = calculate_volume_profile(df)
        st.dataframe(vp.head(), hide_index=True)

if __name__ == "__main__":
    main()
