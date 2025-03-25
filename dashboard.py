# ======================
# CONFIGURAÇÕES REATIVAS
# ======================
if 'config' not in st.session_state:
    st.session_state.config = {
        'rsi_window': 14,
        'bb_window': 20,
        'ma_windows': [7, 30, 200]
    }

def update_config():
    """Atualiza os parâmetros técnicos"""
    st.session_state.config = {
        'rsi_window': st.session_state.rsi_slider,
        'bb_window': st.session_state.bb_slider,
        'ma_windows': st.session_state.ma_multiselect
    }
    st.rerun()

# ======================
# PAINEL DE CONTROLE (ATUALIZADO)
# ======================
with st.sidebar:
    st.header("⚙️ Painel de Controle")
    
    with st.form("config_form"):
        st.subheader("🔧 Parâmetros Técnicos")
        
        # Widgets interativos
        rsi_window = st.slider(
            "Período do RSI",
            7, 21, st.session_state.config['rsi_window'],
            key="rsi_slider"
        )
        
        bb_window = st.slider(
            "Janela das Bandas de Bollinger",
            10, 50, st.session_state.config['bb_window'],
            key="bb_slider"
        )
        
        ma_windows = st.multiselect(
            "Médias Móveis para Exibir",
            [7, 20, 30, 50, 100, 200],
            default=st.session_state.config['ma_windows'],
            key="ma_multiselect"
        )
        
        # Botão para aplicar as configurações
        submitted = st.form_submit_button(
            "💾 Salvar Configurações", 
            on_click=update_config
        )
        if submitted:
            st.success("Configurações atualizadas!")

# ======================
# FUNÇÕES ATUALIZADAS (REATIVAS)
# ======================
def calculate_rsi(series, window=st.session_state.config['rsi_window']):
    """Versão reativa do RSI"""
    # ... (restante da função igual) ...

def calculate_bollinger_bands(series, window=st.session_state.config['bb_window']):
    """Versão reativa das Bandas de Bollinger"""
    # ... (restante da função igual) ...

@st.cache_data(ttl=3600, show_spinner="Recalculando indicadores...")
def load_data():
    """Carrega dados com os parâmetros atuais"""
    data = {}
    # ... (código original) ...
    
    # Aplica as configurações salvas
    price_series = data['prices']['price']
    data['prices']['RSI'] = calculate_rsi(price_series)
    data['prices']['BB_Upper'], data['prices']['BB_Lower'] = calculate_bollinger_bands(price_series)
    
    # Filtra as médias móveis selecionadas
    for window in st.session_state.config['ma_windows']:
        data['prices'][f'MA{window}'] = price_series.rolling(window).mean()
    
    return data

# ======================
# BACKTESTING REATIVO
# ======================
def backtest_strategy(data):
    """Versão reativa do backtesting"""
    df = data['prices'].copy()
    
    # Usa os parâmetros salvos
    rsi_window = st.session_state.config['rsi_window']
    ma_window = st.session_state.config['ma_windows'][1] if len(st.session_state.config['ma_windows']) > 1 else 30
    
    df['signal'] = np.where(
        (df['RSI'] < 30) & (df['price'] < df[f'MA{ma_window}']), 1,
        np.where(
            (df['RSI'] > 70) & (df['price'] > df[f'MA{ma_window}']), -1, 0
        )
    )
    
    # ... (restante do cálculo igual) ...
    return df

# ======================
# INTERFACE (MANTIDA)
# ======================
data = load_data()
signals, final_verdict, buy_signals, sell_signals = generate_signals(data)
bt_data = backtest_strategy(data)

# ... (restante do código original) ...
