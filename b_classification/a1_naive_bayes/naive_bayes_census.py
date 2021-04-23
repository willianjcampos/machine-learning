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
'census.csv', localizado no diretório 'arquivos', utilizando o Algoritmo Neive
Bayes da biblioteca Sklearn.

Antes da realização do treinamento, esta base de dados passou por um
pré-processamento, onde buscou verificar a existência de dados inconsistentes,
substituindo-os pela média.

Posteriormente, realizou-se a conversão dos dados categóricos para dados
numéricos e a criação de variváveis do tipo 'Dummy'.

Os dados faltantes, nesta base de dados, estavam com um sinal de interrogação
(?) não precisando de realizar tratamentos.

Com os dados tratados, realizou-se o processo de escalonamento utilizando os
métodos disponíveis da biblioteca Sklearn.

Após o escalonamento fez-se o treinamento com o Algoritmo Naive Bayes através
da classe GaussianNB() do pacote sklearn.naive_bayes. Para
isso, foi utilizado a classe train_test_split() da biblioteca sklearn do pacote
sklearn.model_selection. Esta divisão ficou entre 75% da base para treinamento
e 25% da base para teste.

Ao final do processo, foi utilizado o pacote sklearn.metrics para realizar
uma análise acerca do resultado.
===============================================================================
RESUMO DO FLUXO UTILIZADO
- Pré-Processamento de dados
--- Tratar dados inconsistentes
--- Conversão dos dados categóricos
------- Criação das variáveis do tipo 'Dummy'
------- Divisão dos Atributos
--- Escalonamento dos Atributos
------- StandardScaler()
------- Normalizer()
------- MinMaxScaler()
------- RobustScaler()
------- QuantileTransformer()
------- PowerTransformer()
- Divisão da Base de Dados
- Treinamento (NAIVE BAYES)
- Resultados
===============================================================================
Descrição do arquivo 'census.csv'
Cada campo (coluna) é separado por vírgula
- age: atributo previsor que define a idade do indivíduo
- workclass: atributo previsor que classifica o tipo de trabalho do indivíduo
- final-weight:
- education: atributo previsor que especifica a escolaridade do indivíduo
- education-num: atributo previsor que define a quantidade (em anos) em que o
indivíduo estudou.
- marital-status: atributo previsor que especifica o status social do indivíduo
- occupation: atributo previsor que define o cargo do indivíduo
- relationship: tipo de relação social do indivíduo.
- race: especifica a raça do indivíduo
- sex: especifica o sexo do indivíduo
- capital-gain:
- capital-loos:
- hour-per-week: horas de trabalho por semana
- native-country: pais de nacionalidade

- income: atributo classe ou meta que define a renda do indivíduo."""


import pandas as pd
from os import path


# Carrega os dados do arquivo 'census.csv' em um dataframe do pandas
base = pd.read_csv(path.join('arquivos', 'census.csv'))


# Divisão dos Atributos em Previsores e Classe
previsores = base.iloc[:,0:14].values
classe = base.iloc[:, 14].values


# Transforma os atributos Previsores do tipo Categórico em Numérico
from sklearn.preprocessing import LabelEncoder
labelencoder_previsores = LabelEncoder()
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = labelencoder_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = labelencoder_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 13] = labelencoder_previsores.fit_transform(previsores[:, 13])
del labelencoder_previsores

# Transforma o atributo classe em dados do tipo Numérico
# ESta transformação é necessária para realizar as análises no final.
labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)
del labelencoder_classe


# Cria as Variáveis do tipo 'Dummy'
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(),
												  [1, 3, 5, 6, 7, 8, 9, 13])],
								   remainder='passthrough')
previsores = onehotencorder.fit_transform(previsores).toarray()
del onehotencorder


# Escalonamento dos atributos previsores
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


# Divisão da Base de Dados entre Treinamento e Testes
from sklearn.model_selection import train_test_split
import numpy as np
previsor = np.array(pr[0])  # Previsor selecionado (varia entre 0-6)
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(
    previsor, classe, test_size=0.25, random_state=0)


# Treinamento: NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)
del classificador


# Resultados
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





