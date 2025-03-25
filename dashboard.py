import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Painel BTC On-Chain")
st.write("Dados simulados para teste!")

# Dados fictícios
dates = pd.date_range(start="2023-01-01", periods=90)
mvrv = [1.2 + 0.02*i for i in range(90)]
hash_rate = [200 + i*2 for i in range(90)]

# Gráficos
st.subheader("📈 MVRV Ratio (Simulado)")
fig_mvrv = px.line(x=dates, y=mvrv, labels={"y": "MVRV"})
fig_mvrv.add_hline(y=3.7, line_color="red")
st.plotly_chart(fig_mvrv)

st.subheader("⚡ Hash Rate (Simulado)")
fig_hash = px.line(x=dates, y=hash_rate, labels={"y": "TH/s"})
st.plotly_chart(fig_hash)
