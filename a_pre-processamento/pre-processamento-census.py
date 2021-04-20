# -*- coding: utf-8 -*-

__author__ = "Willian J Campos Almeida"
__copyright__ = "Copyright 2021, The Cogent Project"
__credits__ = ["Willian J Campos Almeida"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Willian J Campos Almeida"
__email__ = "willianjunio93@gmail.com"
__status__ = "Production"

import pandas as pd
from os import path

base = pd.read_csv(path.join('arquivos', 'census.csv'))

'''É necessário fazer a conversão dos dados categóricos em numéricos, porque
alguns algoritmos de aprendizagem de máquina são baseados em cálculos de
equações e por essa razão, eles não conseguem lidar ou tratar com dados
categóricos sendo, por tanto, necessário essa transformação'''

# Fazer a divisão da base de dados entre atributos previsores e atributo classe
previsores = base.iloc[:,0:14].values
classe = base.iloc[:, 14].values

from sklearn.preprocessing import LabelEncoder
labelencoder_previsores = LabelEncoder()

# Só a título de curiosidade, este comando faz a conversão dos dados da coluna 1
# labels = labelencoder_previsores.fit_transform(previsores[:,1])
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = labelencoder_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = labelencoder_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 13] = labelencoder_previsores.fit_transform(previsores[:, 13])

# Transforma o atributo classe
labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

'''Existe uma falha relacionado ao método LabelEncoder, pois ele cria uma
sequência, fazendo com que cada atributo receba um valore iniciado em 0 e
autoincrementado, dando assim o aspécto de que determinado atributo seja do
tipo ordinal.
Ao verificar a base de dados do censo, o único atributo que podemos considerar
como Ordinal é o de escolaridade 'education', sendo os demais atributos
categóricos do tipo nominal e isso pode interferir nos resultados dos
algoritmos.

A solução para este tipo de problema é a criação de um variáveis do tipo
'Dummy', que transforma os valores dos atributos em colunas.
Ex.: 'race' ser dividida em várias colunas.'''
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Transforma os previsore em binário
onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')
# Aplica a transformação dos atributos em colunas
previsores = onehotencorder.fit_transform(previsores).toarray()


'''Executa o escalonamento dos atributos previsores'''
# 1) Padronização (Standardisation)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores_standard = scaler.fit_transform(previsores)

# 2) Normalização (Normalization)
from sklearn.preprocessing import Normalizer
scaler = Normalizer()
previsores_normalizer = scaler.fit_transform(previsores)

# 3) MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
previsores_min_max = scaler.fit_transform(previsores)

# 4) RobustScaler
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
previsores_robust = scaler.fit_transform(previsores)

# 5) QuantileTransformer
from sklearn.preprocessing import QuantileTransformer
scaler = QuantileTransformer()
previsores_quantile = scaler.fit_transform(previsores)

# 6) PowerTransformer
from sklearn.preprocessing import PowerTransformer
scaler = PowerTransformer()
previsores_power = scaler.fit_transform(previsores)




