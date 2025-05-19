import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# Pega o modelo no GitHub
# ===============================
@st.cache_resource
def carregar_modelo():
    url = "https://github.com/Kinrider/tech_challenge_4/raw/refs/heads/main/Modelo/xgb_petroleo_model.pkl"
    response = requests.get(url)
    from io import BytesIO
    modelo = joblib.load(BytesIO(response.content))
    return modelo

modelo = carregar_modelo()

# ===============================
# Captura dado no IPEA
# ===============================
@st.cache_data
def carregar_dados_ipea():
    codigo_serie = 'EIA366_PBRENT366'
    url = f"http://ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='{codigo_serie}')"
    response = requests.get(url)
    dados_json = response.json()
    dados = dados_json['value']
    df = pd.DataFrame(dados)[['VALDATA', 'VALVALOR']]
    df.columns = ['Data', 'Valor']
    df['Data'] = df['Data'].str[:10]
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values("Data")
    return df

df_dados = carregar_dados_ipea()

# ===============================
# criação de  feature
# ===============================
def preparar_dados_com_features(df):
    df = df.copy()
    df = df.sort_values("Data")
    df["lag1"] = df["Valor"].shift(1)
    df["lag2"] = df["Valor"].shift(2)
    df["lag7"] = df["Valor"].shift(7)
    df["roll_mean_3"] = df["Valor"].rolling(window=3).mean()
    df["roll_mean_7"] = df["Valor"].rolling(window=7).mean()
    df["day"] = df["Data"].dt.day
    df["month"] = df["Data"].dt.month
    df["weekday"] = df["Data"].dt.weekday
    df = df.dropna()
    return df

def gerar_features_para_data(df, data_input):
    df = df[df["Data"] < data_input]  # filtra até o dia anterior
    df = preparar_dados_com_features(df)
    ultima_linha = df.iloc[-1]
    features = pd.DataFrame([{
        "lag1": ultima_linha["lag1"],
        "lag2": ultima_linha["lag2"],
        "lag7": ultima_linha["lag7"],
        "roll_mean_3": ultima_linha["roll_mean_3"],
        "roll_mean_7": ultima_linha["roll_mean_7"],
        "day": data_input.day,
        "month": data_input.month,
        "weekday": data_input.weekday()
    }])
    return features

# ===============================
# Habilitando abas para exibição
# ===============================
tab1, tab2 , tab3 = st.tabs(["Insights e análises", "Modelagem", "Previsão" ])

# --- Aba 1: Visualização + Eventos ---
with tab1:
    st.title("📈 Visualização Histórica do Preço do Petróleo Brent")

    st.subheader("Preço Histórico do Petróleo Brent (USD)")
    st.line_chart(df_dados.set_index("Data")["Valor"])

    st.subheader("Série Histórica com Fatos Destacados")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(x="Data", y="Valor", data=df_dados, ax=ax)

    # Eventos destacados
    ax.axvline(pd.Timestamp("2003-03-20"), color="black", linestyle="--")
    ax.text(pd.Timestamp("2003-03-20"), df_dados["Valor"].max()*0.9, "Invasão do Iraque", color="black")

    ax.axvspan(pd.Timestamp("2008-07-01"), pd.Timestamp("2009-01-01"), color="red", alpha=0.2)
    ax.text(pd.Timestamp("2008-07-01"), df_dados["Valor"].max()*0.95, "Crise de 2008", color="red")

    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-05-01"), color="orange", alpha=0.2)
    ax.text(pd.Timestamp("2020-03-10"), df_dados["Valor"].min()*1.05, "COVID-19", color="orange")

    ax.axvline(pd.Timestamp("2022-02-24"), color="purple", linestyle="--")
    ax.text(pd.Timestamp("2022-02-25"), df_dados["Valor"].max()*0.85, "Invasão da Ucrânia", color="purple")

    ax.set_title("Série Histórica com fatos destacados")
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Análise exploratória - Insights e análises:")

    EDA = """
    1. Visão geral do comportamento do preço ao longo do tempo

    O preço do Brent apresentou oscilações marcantes ao longo dos últimos 30 anos, refletindo períodos de estabilidade, choques abruptos e tendências de longo prazo. Grandes picos e quedas estão associados a eventos econômicos globais, conflitos geopolíticos e transformações estruturais no mercado de petróleo.  
    Observando o gráfico de preço ao longo do tempo, é possível identificar tendências, picos e quedas.

    **Pontos de atenção:**  
    - O gráfico mostra que, entre 2003 e 2008, houve uma forte alta (associada à Guerra do Iraque e ao crescimento global), seguida de uma queda brusca em 2008-2009 (Crise Financeira Global).  
    - Entre 2014 e 2016, outra grande queda, relacionada à Revolução do Xisto nos EUA e ao Colapso dos Preços do Petróleo.  
    - Em 2020, a Pandemia de COVID-19 levou o preço a mínimas históricas, com posterior recuperação.

    O Brent alterna períodos de estabilidade com choques abruptos. Após grandes eventos, o preço raramente retorna rapidamente ao patamar anterior, estabelecendo novos níveis médios.

    2. Eventos com maior impacto no preço

    Com base nos dados do preço Brent durante o período de cada evento, os maiores impactos absolutos (diferença entre o preço máximo e mínimo durante o evento) foram:

    | Evento                           | Preço Mínimo | Preço Máximo | Impacto Real (US$) |
    |---------------------------------|--------------|--------------|-------------------|
    | Guerra do Iraque                 | 23,23        | 143,95       | 120,72            |
    | Revolução do Xisto nos EUA       | 26,01        | 115,19       | 89,18             |
    | Pandemia de COVID-19             | 9,12         | 85,76        | 76,64             |
    | Invasão da Ucrânia pela Rússia   | 61,57        | 133,18       | 71,61             |
    | Crise Financeira Global          | 33,73        | 102,09       | 68,36             |
    | Colapso dos Preços do Petróleo   | 26,01        | 79,62        | 53,61             |
    | Primavera Árabe                  | 88,69        | 128,14       | 39,45             |
    | Crise Energética na Europa       | 71,03        | 99,87        | 28,84             |
    | Guerra do Golfo                 | 17,68        | 41,45        | 23,77             |
    | Reabertura da China Pós-COVID    | 71,03        | 88,31        | 17,28             |

    Os maiores impactos absolutos coincidem com eventos prolongados e/ou de grande magnitude, como guerras, transformações estruturais e crises globais.

    3. Categorias de eventos mais impactantes – com base nos 10 eventos com maior impacto no preço

    Levando-se em consideração os 10 eventos com maior impacto (análise gerada no tópico anterior), a média do impacto real por categoria de evento é:

    | Categoria do Evento                              | Quantidade de Eventos | Impacto Médio (US$) |
    |-------------------------------------------------|----------------------|--------------------|
    | Crises Econômicas Globais ou Regionais           | 2                    | 72,50              |
    | Transformações Estruturais no Mercado de Petróleo| 2                    | 71,39              |
    | Conflitos Geopolíticos e Guerras                 | 4                    | 63,88              |
    | Fatores de Incerteza ou Choques Pontuais         | 2                    | 23,06              |

    Crises Econômicas e transformações estruturais são os eventos que mais movimentam o preço do petróleo Brent, seguidos por Conflitos Geopolíticos ou Guerras. Fatores de Incerteza, embora possam apresentar variações imediatas grandes e tenha um impacto também relevante, tendem a não prolongar o seu efeito.

    4. Relação entre duração do evento e impacto – avaliando todos os eventos do período.

    A análise mostra que eventos de longa duração tendem a apresentar maiores impactos absolutos, mas há exceções. Por exemplo:

    | Categoria de Evento                              | Duração Média (dias) | Impacto Médio (US$) |
    |-------------------------------------------------|---------------------|--------------------|
    | Conflitos Geopolíticos e Guerras (1)            | 1331                | 63,88              |
    | Transformações Estruturais no Mercado de Petróleo| 757                 | 71,39              |
    | Crises Econômicas Globais ou Regionais           | 377                 | 41,24              |
    | Fatores de Incerteza ou Choques Pontuais (2)     | 223                 | 23,06              |

    (1) Nesta categoria foram desconsiderados os eventos: Crise do Petróleo de 1973, Revolução Iraniana, Guerra do Irã – Iraque e Ataques à Saudi Aramco. Para os três primeiros casos, não temos os dados do preço do petróleo, já que nossa base se inicia em 1987, e os eventos foram anteriores, já os Ataques a Saudi Aramco, são considerados como duração de duração de 1 único dia, o que distorceria os resultados.  
    (2) Nesta categoria foi desconsiderado o evento dos ataques de 11 de setembro, pois são considerados como duração de 1 único dia, o que distorceria os resultados.

    Eventos mais longos, como guerras e transformações estruturais, tendem a gerar maiores oscilações acumuladas, enquanto choques pontuais têm impacto mais limitado, mas podem ser relevantes dependendo do contexto.

    **Pontos de Atenção:**  
    - Guerra do Iraque: duração longa (quase 9 anos), impacto absoluto muito alto.  
    - Pandemia de COVID-19: duração relativamente curta, mas impacto elevado.  
    - Reabertura da China Pós-COVID: curta duração, impacto menor.

    Não existe uma relação linear, a intensidade do evento é tão ou mais importante que a sua duração.

    **Conclusão:**  
    O comportamento do Brent é altamente sensível a eventos globais, com grandes oscilações associadas a guerras, transformações estruturais no mercado e crises econômicas. Transformações estruturais e conflitos geopolíticos são, em média, os eventos mais impactantes. O tempo de recuperação do preço após grandes choques é longo, frequentemente superior a uma década, indicando que choques desse porte alteram de forma duradoura o patamar de preço do Brent. A duração do evento está relacionada ao impacto acumulado, mas choques pontuais também podem provocar movimentos bruscos. O monitoramento contínuo do contexto global e das tendências estruturais é fundamental para antecipar movimentos do Brent.
    """

    st.markdown(EDA)
    
# --- Aba 2: Escolha do modelo ---    
with tab2:
    st.title("🔍 Modelagem")

    st.subheader("ETL + Modelo utilizado")
    
    st.markdown(
    '<a href="https://github.com/Kinrider/tech_challenge_4/blob/8692a219b911901de775aaff5a79d10b7034f577/Modelo/modelo_previsao_preco_petrole.ipynb" target="_blank">Clique aqui para visualizar o notebook do modelo</a>',
    unsafe_allow_html=True
    )   
    
    st.markdown("""
                                
    1. Coleta dos Dados (API)
    Capturamos os dados do preço do petróleo Brent direto da API do IPEA, os dados buscados são retornados em JSON, que posteriormente é organizado em um data frame do pandas, com as colunas de data e valor do petróleo.

    2. Tratamento dos Dados
    É feito o ordenamento pela data e eliminamos informações repetidas. Também ajustamos o formato das datas e deixamos os nomes das colunas padronizadas. Depois disso, fazemos uma análise para entender melhor o comportamento dos preços (média, variações).

    3. Testando os Modelos
    Nesta etapa foram realizados testes com alguns modelos pra tentar prever como o preço do petróleo pode se comportar no futuro.

        ✅ 1. Modelo Naive (baseline)
        Este modelo assume que o próximo valor vai ser igual ao último. Serve como comparação.
        
        ✅ 2. XGBoost
        Eficiente em problemas de regressão e capacidade de lidar com relações não lineares nos dados. Ele utiliza árvores de decisão com boosting, otimizando erros iterativamente.
       
        ✅ 3. LSTM (rede neural)
        rede neural recorrente, foi aplicada para explorar padrões temporais nos dados. Como o preço do petróleo tem dependência temporal, a LSTM poderia capturar melhor tendências de longo prazo, demonstrou ser consideravelmente mais lendo que os demais modelos na etapa de treinamento.
       
        ✅ 4. Prophet (Facebook)
        Desenvolvido para séries temporais, lida bem com sazonalidade e outliers. Ele foi testado por sua facilidade de uso e capacidade de decompor tendências, mas pode não ser tão eficiente quanto o XGBoost em conjuntos de dados menores ou menos complexos.


    4. Avaliação dos Modelos:

    Usamos métricas como o MAE (erro médio), o RMSE (erro quadrático médio) e o R² (que mostra o quanto o modelo explica os dados).

    | Modelo   | MAE    | RMSE   | R²      |
    |----------|--------|--------|---------|
    | Naive    | 2.477  | 3.084  | -0.254  |
    | XGBoost  | 0.457  | 0.567  | 0.958   |
    | LSTM     | 0.877  | 1.079  | 0.846   |
    | Prophet  | 37.266 | 37.558 | -66.102 |

    5. Modelo escolhido:
    
    O XGBoost foi selecionado por equilibrar desempenho e interpretabilidade, além de apresentar métricas superiores em comparação aos outros modelos testados. Sua eficiência computacional e adaptabilidade a diferentes cenários o tornaram a melhor opção
    """)

# --- Aba 3: Previsão ---
with tab3:
    st.title("📊 Previsão do Preço do Petróleo Brent")
    
    texto1 = """
    
    Obs:
    
    - O modelo está conectado diretamente aos dados do IPEA, e a previsão é feita para até 30 dias a partir do último dados divulgado.
    
    - Selecione no calendário abaixo a data para qual deseja obter a previsão do preço do petróleo.
    
    """ 
    st.markdown(texto1)   
    
    ultima_data = df_dados["Data"].max()
    max_data_permitida = ultima_data + timedelta(days=30)

    data_input = st.date_input(
        "Escolha a data para previsão:",
        min_value=ultima_data + timedelta(days=1),
        max_value=max_data_permitida,
        value=ultima_data + timedelta(days=1)
    )
    data_input = pd.to_datetime(data_input)

    if st.button("Prever"):
        if data_input <= ultima_data:
            st.error("❌ A data selecionada deve ser posterior à última data com dados disponíveis.")
        else:
            try:
                datas_para_prever = pd.date_range(start=ultima_data + timedelta(days=1), end=data_input)

                previsoes = []
                df_temp = df_dados.copy()

                for data_prev in datas_para_prever:
                    features = gerar_features_para_data(df_temp, data_prev)
                    pred = modelo.predict(features)[0]

                    previsoes.append({"Data": data_prev, "Preco_Previsto": pred})

                    nova_linha = {
                        "Data": data_prev,
                        "Valor": pred
                    }
                    df_temp = pd.concat([df_temp, pd.DataFrame([nova_linha])], ignore_index=True)

                df_previsoes = pd.DataFrame(previsoes)
                df_previsoes["Data"] = pd.to_datetime(df_previsoes["Data"])

                # --- Gráfico ---
                fig, ax = plt.subplots(figsize=(12, 5))

                # Dados reais - últimos 30 dias antes da última data de dados
                inicio_real = ultima_data - timedelta(days=29)
                df_reais_30 = df_dados[(df_dados["Data"] >= inicio_real) & (df_dados["Data"] <= ultima_data)]

                ax.plot(df_reais_30["Data"], df_reais_30["Valor"], label="Valores Reais", color="blue")

                # Dados previstos - do dia seguinte até data_input
                ax.plot(df_previsoes["Data"], df_previsoes["Preco_Previsto"], label="Previsão", color="orange")

                ax.set_title(f"Previsão do Petróleo Brent até {data_input.strftime('%d/%m/%Y')}")
                ax.set_xlabel("Data")
                ax.set_ylabel("Preço (USD)")
                ax.legend()
                ax.grid(True)
                plt.xticks(rotation=45)
                st.pyplot(fig)

                # Ajustar formato da data para exibir na tabela
                df_previsoes["Data"] = df_previsoes["Data"].dt.strftime('%d/%m/%Y')

                st.success(f"📈 Previsões geradas para {len(df_previsoes)} dias até {data_input.strftime('%d/%m/%Y')}")
                st.dataframe(df_previsoes)

            except Exception as e:
                st.error(f"Erro ao gerar previsão: {e}")