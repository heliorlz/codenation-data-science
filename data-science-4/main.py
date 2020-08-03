#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[129]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import (
    OneHotEncoder, Binarizer, KBinsDiscretizer,
    MinMaxScaler, StandardScaler, PolynomialFeatures
)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# In[ ]:


# Algumas configurações para o matplotlib.
# %matplotlib inline

# from IPython.core.pylabtools import figsize


# figsize(12, 8)

# sns.set()


# In[69]:


countries = pd.read_csv("countries.csv")


# In[70]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[71]:


# Sua análise começa aqui.
countries.info()


# In[104]:


# separando as features numericas e convertendo para tipo float
num_features = new_column_names[2:]
countries.replace(',', '.',regex=True ,inplace=True)


# In[105]:


countries.head()


# In[106]:


# convertendo o tipo de dado das variáveis numéricas
countries[num_features] = countries[num_features].astype('float')


# In[107]:


countries.info()


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[108]:


def q1():
    # Retorne aqui o resultado da questão 1.

    # removendo os espaços antes e depois da string
    countries['Country'] = countries['Country'].str.strip()
    countries['Region'] = countries['Region'].str.strip()
    return list(countries['Region'].sort_values().unique())


# In[109]:


q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[ ]:


def q2():
    # Retorne aqui o resultado da questão 2.
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    discretizer.fit_transform(countries[['Pop_density']])
    percentil_090 = countries['Pop_density'].quantile(0.90)
    pop_density_percentil_090 = countries[countries['Pop_density'] > percentil_090]
    return pop_density_percentil_090['Country'].nunique()


# In[ ]:


q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[126]:


def q3():
    # Retorne aqui o resultado da questão 3.
    ohe = OneHotEncoder()
    region = ohe.fit_transform(countries[['Region']])

    count_region_ohe = region.shape[1]

    count_climate = len(countries['Climate'].unique())

    return (count_region_ohe + count_climate)
    


# In[127]:


q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[134]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[130]:


# gerando o Pipeline
pipeline = Pipeline(steps=[
                            ("imputar", SimpleImputer(strategy='median')),
                            ("padronizar", StandardScaler()),
                            ])


# In[132]:


# rodando o pipeline nas features numéricas
pipeline.fit_transform(countries[num_features])


# In[137]:


test_country_pipeline = pipeline.transform([test_country[2:]])
test_country_pipeline


# In[141]:


df_teste = pd.DataFrame(test_country_pipeline, columns=countries[num_features].columns)
df_teste


# In[168]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return float(df_teste['Arable'].values[0].round(3))


# In[169]:


q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[156]:


def q5():
    # Retorne aqui o resultado da questão 4.
    q1 = countries['Net_migration'].quantile(0.25)
    q3 = countries['Net_migration'].quantile(0.75)

    iqr = q3 - q1

    intervalo_inferior = (q1 - (1.5 * iqr))
    intervalo_superior = (q3 + (1.5 * iqr))

    outliers_abaixo = int((countries['Net_migration'] < intervalo_inferior).sum())
    outliers_acima = int((countries['Net_migration'] > intervalo_superior).sum())
    return (outliers_abaixo, outliers_acima, False)


# In[157]:


q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[158]:


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer


# In[159]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[170]:


def q6():
    # Retorne aqui o resultado da questão 4.
    count_vector = CountVectorizer()
    newsgroup_count = count_vector.fit_transform(newsgroup['data'])
    count_phone = count_vector.vocabulary_.get('phone')
    total_phone = newsgroup_count[:, count_phone].toarray().sum()
    return int(total_phone)


# In[171]:


q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[162]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[166]:


def q7():
    # Retorne aqui o resultado da questão 4.
    tfidf = TfidfVectorizer()
    newsgroup_tfidf = tfidf.fit_transform(newsgroup['data'])

    count_phone = tfidf.vocabulary_.get('phone')
    total_phone = newsgroup_tfidf[:, count_phone].toarray().sum()
    return float(total_phone.round(3))


# In[167]:


q7()


# In[ ]:




