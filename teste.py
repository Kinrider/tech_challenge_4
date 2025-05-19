import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime

# ========== Baixar modelo do GitHub ==========
@st.cache_resource
def carregar_modelo():
    url = "https://raw.githubusercontent.com/usuario/repositorio/main/modelo_xgb.joblib"  # Substitua com o seu link raw
    response = requests.get(url)
    with open("modelo_xgb.joblib", "wb") as f:
        f.write(response.content)
    modelo = joblib.load("modelo_xgb.joblib")
    return modelo

modelo = carregar_modelo()

# ========== Função para gerar os features ==========
def gerar_features(data):
    data = pd.to_datetime(data)
    df = pd.DataFrame({
        "ano": [data.year],
        "mes": [data.month],
        "dia": [data.day],
        "dia_semana": [data.weekday()],
        "dia_do_ano": [data.timetuple().tm_yday]
    })
    return df

# ========== Interface Streamlit ==========
st.title("Previsão do Preço do Petróleo Brent")

data_input = st.date_input(
    "Escolha a data para previsão:",
    min_value=datetime(2000, 1, 1),
    max_value=datetime(2100, 1, 1),
    value=datetime.today()
)

if st.button("Prever"):
    features = gerar_features(data_input)
    previsao = modelo.predict(features)[0]
    st.success(f"📈 Preço previsto para {data_input.strftime('%d/%m/%Y')}: **${previsao:.2f}**")
