import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
import seaborn as sns

st.set_page_config(layout="wide")
st.title("Analise e Previsao do Preco do Petroleo Brent")

@st.cache_data
def carregar_dados():
    url = "http://ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='EIA366_PBRENT366')"
    response = requests.get(url)
    dados_json = response.json()
    dados = dados_json["value"]
    df = pd.DataFrame(dados)[["VALDATA", "VALVALOR"]]
    df.columns = ["Data", "Valor"]
    df["Data"] = pd.to_datetime(df["Data"].str[:10])
    df = df.sort_values("Data").dropna()
    return df

df = carregar_dados()
data_min = df["Data"].min()
data_max = df["Data"].max()
data_inicio, data_fim = st.date_input("Selecione o intervalo de datas", [data_min, data_max], min_value=data_min, max_value=data_max)
df_filtrado = df[(df["Data"] >= pd.to_datetime(data_inicio)) & (df["Data"] <= pd.to_datetime(data_fim))]

st.subheader("Preco Historico do Petroleo Brent (USD)")
st.line_chart(df_filtrado.set_index("Data")["Valor"])

st.subheader("Serie Historica com Fatos Destacados")
fig, ax = plt.subplots(figsize=(14, 6))
sns.lineplot(x="Data", y="Valor", data=df, ax=ax)

# 01 - Invasão Iraque
ax.axvline(pd.Timestamp("2003-03-20"), color="black", linestyle="--")
ax.text(pd.Timestamp("2003-03-20"), df["Valor"].max()*0.9, "Invasão do Iraque", color="black")

# 02 - Crise 2008
ax.axvspan(pd.Timestamp("2008-07-01"), pd.Timestamp("2009-01-01"), color="red", alpha=0.2)
ax.text(pd.Timestamp("2008-07-01"), df["Valor"].max()*0.95, "Crise de 2008", color="red")

# 03 - COVID-19
ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-05-01"), color="orange", alpha=0.2)
ax.text(pd.Timestamp("2020-03-10"), df["Valor"].min()*1.05, "COVID-19", color="orange")

# 04 - Invasão Ucrânia
ax.axvline(pd.Timestamp("2022-02-24"), color="purple", linestyle="--")
ax.text(pd.Timestamp("2022-02-25"), df["Valor"].max()*0.85, "Invasão da Ucrânia", color="purple")

ax.set_title("Série Histórica com fatos destacados")
ax.grid(True)
st.pyplot(fig)

st.subheader("Previsao com Prophet")
df_prophet = df_filtrado.rename(columns={"Data": "ds", "Valor": "y"})
modelo = Prophet()
modelo.fit(df_prophet)
futuro = modelo.make_future_dataframe(periods=30)
forecast = modelo.predict(futuro)
fig1 = modelo.plot(forecast)
st.pyplot(fig1)

st.subheader("Previsao com XGBoost")
df_xgb = df_filtrado.set_index("Data").asfreq("D")
df_xgb["Valor"] = df_xgb["Valor"].interpolate()
df_xgb["lag1"] = df_xgb["Valor"].shift(1)
df_xgb["lag2"] = df_xgb["Valor"].shift(2)
df_xgb["lag7"] = df_xgb["Valor"].shift(7)
df_xgb.dropna(inplace=True)
X = df_xgb[["lag1", "lag2", "lag7"]]
y = df_xgb["Valor"]
X_train, X_test = X[:-30], X[-30:]
y_train, y_test = y[:-30], y[-30:]
modelo_xgb = XGBRegressor()
modelo_xgb.fit(X_train, y_train)
pred_xgb = modelo_xgb.predict(X_test)
fig2, ax2 = plt.subplots()
ax2.plot(y_test.index, y_test, label="Real")
ax2.plot(y_test.index, pred_xgb, label="Previsto")
ax2.set_title("Previsao com XGBoost")
ax2.legend()
st.pyplot(fig2)

st.subheader("Previsao com LSTM")
df_lstm = df_filtrado.set_index("Data").asfreq("D")
df_lstm["Valor"] = df_lstm["Valor"].interpolate()
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_lstm[["Valor"]])
X_lstm, y_lstm = [], []
for i in range(30, len(scaled_data)):
    X_lstm.append(scaled_data[i-30:i])
    y_lstm.append(scaled_data[i])
X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=False, input_shape=(X_lstm.shape[1], 1)))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mean_squared_error', optimizer='adam')
model_lstm.fit(X_lstm, y_lstm, epochs=10, batch_size=32, verbose=0)
last_30 = scaled_data[-30:].reshape(1, 30, 1)
forecast_lstm = []
for _ in range(30):
    pred = model_lstm.predict(last_30)[0]
    forecast_lstm.append(pred)
    last_30 = np.append(last_30[:,1:,:], [[pred]], axis=1)
forecast_lstm = scaler.inverse_transform(forecast_lstm)
futuras_datas = pd.date_range(df_lstm.index[-1] + pd.Timedelta(days=1), periods=30)
fig3, ax3 = plt.subplots()
ax3.plot(df_lstm.index[-60:], df_lstm["Valor"].values[-60:], label="Historico")
ax3.plot(futuras_datas, forecast_lstm, label="Previsao LSTM")
ax3.set_title("Previsao com LSTM")
ax3.legend()
st.pyplot(fig3)

st.subheader("Tabela de Resultados (XGBoost)")
mae_xgb = mean_absolute_error(y_test, pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, pred_xgb))
r2_xgb = r2_score(y_test, pred_xgb)
st.table(pd.DataFrame({
    "Modelo": ["XGBoost"],
    "MAE": [round(mae_xgb, 2)],
    "RMSE": [round(rmse_xgb, 2)],
    "R2": [round(r2_xgb, 2)]
}))


#streamlit run previsão_petroleo.py