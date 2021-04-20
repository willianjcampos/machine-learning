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
--- Conversão dos dados categóricos"""


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


