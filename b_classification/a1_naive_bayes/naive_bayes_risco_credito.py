# -*- coding: utf-8 -*-


__author__ = "Willian J Campos Almeida"
__copyright__ = "Copyright 2021, The Cogent Project"
__credits__ = ["Willian J Campos Almeida"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Willian J Campos Almeida"
__email__ = "willianjunio93@gmail.com"
__status__ = "Production"


"""O objetivo deste arquivo é realizar a classificação da base de dados
'risco_credito.data' utilizando algoritmos de classificação distintos.
================
Descrição do arquivo 'risco_credito.data'
Cada campo (coluna) é separado por vírgula
- historia:
- divida:
- garantias:
- renda:

- risco:
================
Antes da realização do treinamento, a base de dados 'risco_credito.data'
passou por um pré-processamento relacionado a conversão dos dados categóricos
para dados numéricos. Pela base de dados ser pequena, optou-se por não criar
as variáveis do tipo 'Dummy'.
Com os dados tratados, realizou-se o processo de escalonamento utilizando os
métodos disponíveis da biblioteca Sklearn.

FLUXO UTILIZADO
- Pré-Processamento de dados
--- Divisão dos Atributos
--- Conversão dos dados categóricos
- Divisão da Base de Dados
- Treinamento (ALGORITMO)
- Resultados"""


import pandas as pd
from os import path

# Carrega a Base de Dados
base_dados = pd.read_table(path.join('arquivos', 'risco_credito.data'), delimiter=',')


# ---Divisão dos Atributos em Previsores e Classe---
previsores = base_dados.iloc[:, 0:4].values
classe = base_dados.iloc[:, 4].values


# ---Transforma dados Categóricos em Dados Numéricos---
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
previsores[:, 0] = label_encoder.fit_transform(previsores[:, 0])
previsores[:, 1] = label_encoder.fit_transform(previsores[:, 1])
previsores[:, 2] = label_encoder.fit_transform(previsores[:, 2])
previsores[:, 3] = label_encoder.fit_transform(previsores[:, 3])


# ---Treinamento---
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores, classe)


# ---Testes---
# História = (0) Boa, (1) Desconhecida, (3) Ruim
# Dívida = (0) Alta, (1) Baixa
# Garantias = (0) Adequada, (1) Nenhuma
# Renda = (0) < 15, (1) >= 15 <= 35, (2) > 35
'''
1) Ruim / Alta / Adequada / < 15
2) Desconhecida / Alta / Adequada / < 15
3) Desconhecida / Alta / Nenhuma / > 35
4) Boa / Alta / Adequada / >= 15 <= 35
5) Boa / Alta / Nenhuma / > 15
'''
resultado = classificador.predict([[3, 0, 0, 0],
								   [1, 0, 1, 0],
								   [1, 0, 1, 2],
								   [0, 0, 0, 1],
								   [0, 0, 1, 2]])


# Exibe as classes existentes
print(classificador.classes_)


# Exibe a quantidade de registros classificados como Alto, Moderado e Baixo
print(classificador.class_count_)


# Exibe as probabilidades a priori de cada um dos atributos
print(classificador.class_prior_)
