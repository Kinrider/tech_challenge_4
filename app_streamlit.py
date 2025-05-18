# pip install streamlit pandas requests matplotlib prophet scikit-learn xgboost seaborn

import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np

st.set_page_config(page_title="Previsão Petróleo Brent", layout="centered")

st.title("Previsão do Preço do Petróleo")

# --- Baixar dados da API do IPEA ---
codigo_serie = 'EIA366_PBRENT366'
url = f"http://ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='{codigo_serie}')"

@st.cache_data
def carregar_dados():
    try:
        response = requests.get(url)
        response.raise_for_status()
        dados_json = response.json()
        dados = dados_json['value']
        df = pd.DataFrame(dados)
        df = df[['VALDATA', 'VALVALOR']]
        df.columns = ['Data', 'Valor']
        df['Data'] = pd.to_datetime(df['Data'], utc=True).dt.tz_localize(None)
        df = df.sort_values('Data')
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame(columns=['Data', 'Valor'])

# Carregar dados
df = carregar_dados()

# Mostrar tabela e linha do tempo
st.subheader("Histórico de Preços")
st.line_chart(df.set_index('Data')['Valor'])

# --- Gráfico com eventos históricos ---
st.subheader("Série Histórica com Fatos Destacados")
fig_eventos, ax = plt.subplots(figsize=(14, 6))
sns.lineplot(x="Data", y="Valor", data=df, ax=ax)

ax.axvline(pd.Timestamp("2003-03-20"), color="black", linestyle="--")
ax.text(pd.Timestamp("2003-03-20"), 95, "Invasão do Iraque", color="black")
ax.axvspan(pd.Timestamp("2008-07-01"), pd.Timestamp("2009-01-01"), color="red", alpha=0.2)
ax.text(pd.Timestamp("2008-07-01"), 140, "Crise de 2008", color="red")
ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-05-01"), color="orange", alpha=0.2)
ax.text(pd.Timestamp("2020-03-10"), 25, "COVID-19", color="orange")
ax.axvline(pd.Timestamp("2022-02-24"), color="purple", linestyle="--")
ax.text(pd.Timestamp("2022-02-25"), 115, "Invasão da Ucrânia", color="purple")
ax.set_title("Série Histórica com Fatos Destacados")
ax.grid(True)
st.pyplot(fig_eventos)

# --- Seletor de modelo ---
modelo = st.selectbox("Escolha o modelo de previsão:", ["Prophet", "XGBoost"])
dias = st.slider("Quantos dias para prever?", min_value=7, max_value=60, step=1, value=30)

# Função para avaliação
def avaliar_modelo(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    r2 = r2_score(y_true_clean, y_pred_clean)
    return mae, rmse, r2

# --- Previsão usando Prophet ---
if modelo == "Prophet":
    st.subheader(f"Previsão para os próximos {dias} dias com Prophet")
    df_prophet = df.rename(columns={"Data": "ds", "Valor": "y"})
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=dias)
    forecast = model.predict(future)

    # Gráfico
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    # Tabela com previsões
    st.subheader("Tabela de Previsão")
    previsoes = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(dias)
    st.write(previsoes)

    # Avaliação
    historico = forecast.iloc[:len(df_prophet)]
    mae, rmse, r2 = avaliar_modelo(df_prophet['y'], historico['yhat'])
    st.markdown(f"**MAE**: {mae:.2f} | **RMSE**: {rmse:.2f} | **R²**: {r2:.2f}")

    # Validação cruzada
    try:
        df_cv = cross_validation(model, initial='730 days', period='180 days', horizon=f'{dias} days')
        df_p = performance_metrics(df_cv)
        st.subheader("Validação Cruzada Prophet")
        st.write(df_p[['mae', 'rmse', 'r2']])
    except Exception as e:
        st.warning(f"Erro na validação cruzada: {e}")

# --- Previsão com XGBoost (modelo simples) ---
elif modelo == "XGBoost":
    st.subheader(f"Previsão para os próximos {dias} dias com XGBoost")

    df_xgb = df.copy()
    df_xgb['Data_ordinal'] = df_xgb['Data'].map(pd.Timestamp.toordinal)

    # Separar treino e teste
    X = df_xgb[['Data_ordinal']]
    y = df_xgb['Valor']
    X_train, y_train = X[:-dias], y[:-dias]
    X_test, y_test = X[-dias:], y[-dias:]

    model_xgb = XGBRegressor(n_estimators=100, random_state=42)
    model_xgb.fit(X_train, y_train)

    y_pred = model_xgb.predict(X_test)

    # Avaliação
    mae, rmse, r2 = avaliar_modelo(y_test, y_pred)
    st.markdown(f"**MAE**: {mae:.2f} | **RMSE**: {rmse:.2f} | **R²**: {r2:.2f}")

    # Datas futuras reais
    datas_futuras = df_xgb['Data'].iloc[-dias:]
    df_pred_real = pd.DataFrame({
        'Data': datas_futuras,
        'Valor Previsto': y_pred,
        'Valor Real': y_test.values
    })

    # Gráfico real vs previsto
    fig, ax = plt.subplots()
    ax.plot(df_pred_real['Data'], df_pred_real['Valor Real'], label='Valor Real')
    ax.plot(df_pred_real['Data'], df_pred_real['Valor Previsto'], label='Valor Previsto')
    ax.set_title("Previsão XGBoost vs Real")
    ax.legend()
    st.pyplot(fig)

    # Previsão futura
    ultima_data = df['Data'].max()
    novas_datas = pd.date_range(start=ultima_data + pd.Timedelta(days=1), periods=dias)
    X_future = pd.DataFrame({'Data_ordinal': novas_datas.map(pd.Timestamp.toordinal)})
    y_pred_future = model_xgb.predict(X_future)
    df_pred_futuro = pd.DataFrame({'Data': novas_datas, 'Valor Previsto': y_pred_future})

    st.subheader("Previsão Futura com XGBoost")
    st.write(df_pred_futuro)

# --- Download CSV ---
st.subheader("Download dos Dados")

csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Baixar Histórico em CSV",
    data=csv,
    file_name='dados_petroleo.csv',
    mime='text/csv'
)

if modelo == "Prophet":
    csv_forecast = previsoes.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Baixar Previsão em CSV",
        data=csv_forecast,
        file_name='previsao_prophet.csv',
        mime='text/csv'
    )
elif modelo == "XGBoost":
    csv_forecast = df_pred_futuro.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Baixar Previsão em CSV",
        data=csv_forecast,
        file_name='previsao_xgboost.csv',
        mime='text/csv'
    )
    
    #streamlit run "c:/Users/jdlma/Downloads/Nova pasta (2)/app_streamlit.py"

