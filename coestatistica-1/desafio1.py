import pandas as pd

# carregando o dataset
df = pd.read_csv("desafio1.csv")

# visualizando os 5 primeiros registros para conhecermos nossos dados
df.head()

# verificando os tipos de dados encontrados
df.dtypes

# visualizando a variável que devemos realizar o desafio
df["pontuacao_credito"]

# visualizando os estados únicos do dataset
df["estado_residencia"].unique()

# Criando uma função que irá retornar todos os cálculos


def stats(df):
    return {"moda": df.mode()[0], "mediana": df.median(), "media": df.mean(), "desvio_padrao": df.std()}


# Agrupando para relacionar estado_residencia e pontuacao_credito
# Apply => recebe uma função como parâmetro que se aplica para cada valor
df_final = df.groupby("estado_residencia")[
    "pontuacao_credito"].apply(stats).unstack()

# Fazendo a conversão para .json
df_final.to_json("submission.json", orient="index")
