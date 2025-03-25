with tab1:  # Mercado
    if not data['prices'].empty:
        fig = px.line(data['prices'], x="date", y=["price", "MA7", "MA30", "MA200"], 
                     title="Preço BTC e Médias Móveis")
        st.plotly_chart(fig, use_container_width=True)
    
    # Tabela de Sinais (ADICIONAR ESTA SEÇÃO)
    st.subheader("📊 Indicadores Técnicos")
    df_signals = pd.DataFrame(signals, columns=["Indicador", "Sinal", "Valor"])
    
    def color_signal(val):
        if "COMPRA" in val:
            return 'background-color: #4CAF50'
        elif "VENDA" in val:
            return 'background-color: #F44336'
        return 'background-color: #FFC107'
    
    st.dataframe(
        df_signals.style.applymap(color_signal, subset=["Sinal"]),
        hide_index=True,
        use_container_width=True,
        height=400
    )
    
    # Gráfico de Sentimento (mantido)
    st.subheader("📊 Sentimento do Mercado (Fear & Greed Index)")
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
