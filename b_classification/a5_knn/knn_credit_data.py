# -*- coding: utf-8 -*-
"""
Descrição:
	O objetivo deste arquivo é realizar a classificação da base de dados
	'credit_data' utilizando o Algoritmo de Aprendizagem baseado em Instâncias
	chamado de Algoritmo KNN (K-Nearest Neighbors Algorithm) do pacote
	sklearn.neighbors, pertencente a biblioteca Sklearn.
===============================================================================
Base de dados:
	Na base de dados 'credit_data_orange.csv' os campos (colunas) ou atriutos
	são separados por vírgula. Além disso, no atributo 'clientid' foi
	adicionado o caractere 'i#' para ignorá-lo, e no atributo 'default', o
	caractere 'c#' para identificá-lo como atributo meta ou classe.

Atributos:
	-- clienteid: previsor que define o id do cliente (i#clientid).
	-- income: previsor que define o salário ou renda anual do cliente.
	-- age: previsor que define a idade do cliente que, apesar do tipo 'float'.
	-- loan: previsor que define o valor solicitado do emprestimo.
	
	-- default: classe que define qual cliente não pagou (0) ou pagou (1) o
	financiamento ou empréstimo (c#default).
"""

__author__ = "Willian J Campos Almeida"
__copyright__ = "Copyright 2021, Machine Learning Project"
__credits__ = ["Willian J Campos Almeida"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Willian J Campos Almeida"
__email__ = "willianjunio93@gmail.com"
__status__ = "Production"



import pandas as pd
from os import path

# Carregamento da base de dados
base = pd.read_csv(path.join('arquivos', 'credit_data.csv'))


# Tratamento de dados inconsistentes
base.loc[base.age < 0, 'age'] = base['age'][base.age > 0].mean()


# Divisão dos Atributos em Previsores e Classe
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values


# Tratamento de dados Faltantes
from sklearn.impute import SimpleImputer
import numpy as np
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])
del imputer


# Escalonamento dos valores
# 1) Padronização (Standardisation)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores_standard = scaler.fit_transform(previsores)
del scaler


# 2) Normalização (Normalization)
from sklearn.preprocessing import Normalizer
scaler = Normalizer()
previsores_normalizer = scaler.fit_transform(previsores)
del scaler


# 3) MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
previsores_min_max = scaler.fit_transform(previsores)
del scaler


# 4) RobustScaler
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
previsores_robust = scaler.fit_transform(previsores)
del scaler


# 5) QuantileTransformer
from sklearn.preprocessing import QuantileTransformer
scaler = QuantileTransformer()
previsores_quantile = scaler.fit_transform(previsores)
del scaler


# 6) PowerTransformer
from sklearn.preprocessing import PowerTransformer
scaler = PowerTransformer()
previsores_power = scaler.fit_transform(previsores)
del scaler


pr = [previsores, previsores_standard, previsores_normalizer,
	  previsores_min_max, previsores_robust, previsores_quantile,
	  previsores_power]


# Divisão da Base de Dados
from sklearn.model_selection import train_test_split
previsor = np.array(pr[1])  # Previsor selecionado (varia entre 0-6)
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(
    previsor, classe, test_size=0.25, random_state=0)


# Treinamento: KNN
from sklearn.neighbors import KNeighborsClassifier
# O parâmetro 'metric='minkowski'' e 'p=2' está relacionado a utilização da medida euclidiana
classificador = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)


# Resultado
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
precisao = accuracy_score(classe_teste, previsoes)
print(precisao)
matriz = confusion_matrix(classe_teste, previsoes)
cls_report = classification_report(classe_teste, previsoes)

from sklearn.metrics import average_precision_score
precisao_micro = average_precision_score(classe_teste, previsoes, average='micro')
precisao_samples = average_precision_score(classe_teste, previsoes, average='samples')
precisao_weighted = average_precision_score(classe_teste, previsoes, average='weighted')
precisao_macro = average_precision_score(classe_teste, previsoes, average='macro')
