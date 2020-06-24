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
    return {"moda": df.mode(), "mediana": df.median(), "media": df.mean(), "desvio_padrao": df.std()}


# Agrupando para relacionar estado_residencia e pontuacao_credito
# Aggregate => podemos retornar todos os cálculos de uma só vez utilizando a função aggregate e armazendo em um novo Dataframe
df_final = df.groupby("estado_residencia")[
    "pontuacao_credito"].apply(stats).unstack()

# Precisamos renomear as colunas para ir de acordo com o desafio
print(df_final)

# Fazendo a conversão para .json
df_final.to_json("submission.json")
