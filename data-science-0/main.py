#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[ ]:


# !pip install sklearn


# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


black_friday.head()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[4]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return black_friday.shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[5]:


def q2():
    # Retorne aqui o resultado da questão 2.
    df = black_friday[(black_friday['Age'] == '26-35') & (black_friday['Gender'] == 'F')]
    return len(df)


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[6]:


def q3():
    # Retorne aqui o resultado da questão 3.
    n_unicos_id = black_friday["User_ID"].nunique()
    return n_unicos_id


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[7]:


def q4():
    # Retorne aqui o resultado da questão 4.
    tipos_dados = black_friday.dtypes.nunique()
    return tipos_dados


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[8]:


def q5():
    # Retorne aqui o resultado da questão 5.
    missing_values = max(black_friday.isna().sum(axis="rows")) / black_friday.shape[0]
    return missing_values


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[9]:


def q6():
    # Retorne aqui o resultado da questão 6.
    df_aux = int(max(black_friday.isna().sum()))
    return df_aux


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[10]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return float(black_friday["Product_Category_3"].dropna().mode())


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[11]:


def q8():
    # Retorne aqui o resultado da questão 8.
    scaler_norm = MinMaxScaler()
    scaled_data_norm = scaler_norm.fit_transform(black_friday[["Purchase"]])
    return float(scaled_data_norm.mean())


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[12]:


def q9():
    # Retorne aqui o resultado da questão 9.
    scaler_std = StandardScaler()
    scaled_data_std = scaler_std.fit_transform(black_friday[["Purchase"]])
    return int(((scaled_data_std > -1) & (scaled_data_std < 1)).sum()) 


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[20]:


def q10():
    # Retorne aqui o resultado da questão 10.
    is_na = black_friday["Product_Category_2"].isna() # Armazenando os registros com NaN
    filter_is_na_cat2 = black_friday["Product_Category_2"][is_na] # Filtrando Product Category 2 recebendo apenas os registros com NaN
    filter_is_na_cat3 = black_friday["Product_Category_3"][is_na]

    is_equal = filter_is_na_cat3.equals(filter_is_na_cat2) # Comparando os Product Category 2 e 3, levando em consideração os registros NaN

    return is_equal

q10()


# In[ ]:




