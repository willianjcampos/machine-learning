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
'credit_data' utilizando o Algoritmo Neive Bayes da biblioteca Sklearn.
================
Descrição do arquivo credit_data.csv
Cada campo (coluna) é separado por vírgula
- clienteid: atributo previsor que define o id do cliente
- income: atributo previsor que define o salário ou renda do cliente (por ano)
- age: atributo previsor que define a idade do cliente que, apesar do tipo
float, o importante será apenas a parte inteira
- loan: atributo previsor que define o valor solicitado do emprestimo

- default: atributo classe que define qual cliente não pagou (0) ou pagou (1)
o financiamento.
================
Antes da realização do treinamento, a base de dados 'credit_data' passou por um
pré-processamento, onde verificou-se a existência de dados (idades) da
categoria 'age' com valores negativos, substituindo-os pela média.
Como todos os dados base de dados 'credit_data' estão em valores do tipo
numérico, não foi necessário a conversão dos dados.
Após o tratamento desdes dados, realizou-se a substituição dos valores
faltantes (NaN) pelo valor médio de cada coluna.
Com os dados tratados, realizou-se o processo de escalonamento utilizando os
métodos disponíveis da biblioteca Sklearn.

FLUXO UTILIZADO
- Pré-Processamento de dados
--- Tratar dados inconsistentes
--- Tratar valores faltantes
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
--- GaussianNB()
--- MultinomialNB()
--- ComplementNB()
--- BernoulliNB()
--- CategoricalNB()
- Resultados"""


import pandas as pd
from os import path

# Carregamento dos dados
base = pd.read_csv(path.join('arquivos', 'credit_data.csv'))


# ---Tratamento de dados inconsistentes---
base.loc[base.age < 0, 'age'] = base['age'][base.age > 0].mean()


# ---Divisão dos Atributos em Previsores e Classe---
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values


# ---Tratamento de dados Faltantes---
from sklearn.impute import SimpleImputer
import numpy as np
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])


# ---Escalonamento dos valores---
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

pr = [previsores, previsores_standard, previsores_normalizer,
	  previsores_min_max, previsores_robust, previsores_quantile,
	  previsores_power]


# ---Divisão da Base de Dados---
from sklearn.model_selection import train_test_split
previsor = np.array(pr[3])  # Seleciona qual previsor será utilizado
previsores_treino, previsores_teste, classe_treinamento, classe_teste = train_test_split(
    previsor, classe, test_size=0.25, random_state=0)


# ---Treinamento: Baive Bayes---
prev = []
# 1) GaussianNB
from sklearn.naive_bayes import GaussianNB
clsGaussian = GaussianNB()
clsGaussian.fit(previsores_treino, classe_treinamento)
prev.append([clsGaussian.predict(previsores_teste), 'GaussianNB'])


# 2) MultinomialNB
from sklearn.naive_bayes import MultinomialNB
clsMultinomial = MultinomialNB()
clsMultinomial.fit(previsores_treino, classe_treinamento)
prev.append([clsMultinomial.predict(previsores_teste), 'MultinomialNB'])


# 3) ComplementNB
from sklearn.naive_bayes import ComplementNB
clsComplement = ComplementNB()
clsComplement.fit(previsores_treino, classe_treinamento)
prev.append([clsComplement.predict(previsores_teste), 'ComplementNB'])


# 4) BernoulliNB
from sklearn.naive_bayes import BernoulliNB
clsBernoulli = BernoulliNB()
clsBernoulli.fit(previsores_treino, classe_treinamento)
prev.append([clsBernoulli.predict(previsores_teste), 'BernoulliNB'])


# 5) CategoricalNB
from sklearn.naive_bayes import CategoricalNB
clsCategorical = CategoricalNB()
clsCategorical.fit(previsores_treino, classe_treinamento)
prev.append([clsCategorical.predict(previsores_teste), 'CategoricalNB'])

prev = np.asarray(prev)


# ---Resultado---
from sklearn.metrics import accuracy_score

acc_score = []

print('Accuracy Score')
for i in range(len(prev)):
	acc_score.append(accuracy_score(classe_teste, prev[i, 0]))
	print('%s: %.3f' % (prev[i, 1], acc_score[i]))



