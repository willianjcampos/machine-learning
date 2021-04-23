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
'credit_data', localizado no diretório 'arquivos', utilizando o Algoritmo Neive
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
- Treinamento (ALGORITMO)
- Resultados
===============================================================================
Descrição do arquivo 'credit_data.csv'
Cada campo (coluna) é separado por vírgula
- clienteid: atributo previsor que define o id do cliente
- income: atributo previsor que define o salário ou renda do cliente (por ano)
- age: atributo previsor que define a idade do cliente que, apesar do tipo
float, o importante será apenas a parte inteira
- loan: atributo previsor que define o valor solicitado do emprestimo

- default: atributo classe que define qual cliente não pagou (0) ou pagou (1)
o financiamento."""


import pandas as pd
from os import path

# Carregamento dos dados
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


# Treinamento: DECISION TREE
from sklearn.tree import DecisionTreeClassifier
classificador = DecisionTreeClassifier(criterion='entropy', random_state=0)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)
del classificador


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


