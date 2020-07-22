#!/usr/bin/env python
# coding: utf-8

# # Desafio 5
# 
# Neste desafio, vamos praticar sobre redução de dimensionalidade com PCA e seleção de variáveis com RFE. Utilizaremos o _data set_ [Fifa 2019](https://www.kaggle.com/karangadiya/fifa19), contendo originalmente 89 variáveis de mais de 18 mil jogadores do _game_ FIFA 2019.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats as st
from sklearn.decomposition import PCA

from loguru import logger


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

#from IPython.core.pylabtools import figsize


#figsize(12, 8)

#sns.set()


# In[4]:


fifa = pd.read_csv("fifa.csv")


# In[5]:


columns_to_drop = ["Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",
                   "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",
                   "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
                   "Body Type", "Real Face", "Position", "Jersey Number", "Joined",
                   "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",
                   "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
                   "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
                   "CB", "RCB", "RB", "Release Clause"
]

try:
    fifa.drop(columns_to_drop, axis=1, inplace=True)
except KeyError:
    logger.warning(f"Columns already dropped")


# ## Inicia sua análise a partir daqui

# In[6]:


# Sua análise começa aqui.
fifa.head()


# In[7]:


fifa.shape


# In[8]:


fifa.isna().sum()


# In[9]:


fifa.info()


# In[10]:


fifa.describe()


# In[13]:


df_fifa = fifa.dropna()


# ## Questão 1
# 
# Qual fração da variância consegue ser explicada pelo primeiro componente principal de `fifa`? Responda como um único float (entre 0 e 1) arredondado para três casas decimais.

# In[11]:


pca = PCA()


# In[24]:


pca.fit(df_fifa)
evr = pca.explained_variance_ratio_[0]


# In[26]:


evr.dtype


# In[69]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return float(evr.round(3))
    


# In[70]:


q1()


# ## Questão 2
# 
# Quantos componentes principais precisamos para explicar 95% da variância total? Responda como un único escalar inteiro.

# In[29]:


pca_095 = PCA(n_components=0.95)
df_fifa_reduced = pca_095.fit_transform(df_fifa)


# In[32]:


df_fifa_reduced.shape[1]


# In[33]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return int(df_fifa_reduced.shape[1])


# In[34]:


q2()


# ## Questão 3
# 
# Qual são as coordenadas (primeiro e segundo componentes principais) do ponto `x` abaixo? O vetor abaixo já está centralizado. Cuidado para __não__ centralizar o vetor novamente (por exemplo, invocando `PCA.transform()` nele). Responda como uma tupla de float arredondados para três casas decimais.

# In[35]:


x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
]


# In[36]:


pca_2 = PCA(n_components=2).fit(df_fifa)


# In[51]:


componente_1, componente_2 = pca_2.components_.dot(x).round(3)


# In[52]:


print(componente_1)
print(componente_2)


# In[53]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return tuple((componente_1, componente_2))


# In[54]:


q3()


# ## Questão 4
# 
# Realiza RFE com estimador de regressão linear para selecionar cinco variáveis, eliminando uma a uma. Quais são as variáveis selecionadas? Responda como uma lista de nomes de variáveis.

# In[55]:


from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE


# In[56]:


y = df_fifa['Overall']
y.head()


# In[58]:


X = df_fifa.drop(columns='Overall')
X.head()


# In[59]:


estimator = LinearRegression()
n_features = 5
step = 1


# In[60]:


rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=step)


# In[61]:


rfe_fifa = rfe.fit(X, y)


# In[66]:


variaveis_selecionadas = rfe_fifa.support_
variaveis_selecionadas


# In[67]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return list(X.columns[variaveis_selecionadas])


# In[68]:


q4()


# In[ ]:




