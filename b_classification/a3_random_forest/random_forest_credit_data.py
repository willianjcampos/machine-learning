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
'credit_data', localizado no diretório 'arquivos', utilizando o Algoritmo Random
Forest da biblioteca Sklearn.

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
da classe RandomForestClassifier() do pacote sklearn.ensemble. Para isso, foi
utilizado a classe train_test_split() da biblioteca sklearn do pacote
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
- Treinamento (RANDOM FOREST)
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
previsor = np.array(pr[0])  # Previsor selecionado (varia entre 0-6)
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(
    previsor, classe, test_size=0.25, random_state=0)


# Treinamento: RANDOM FOREST - DADOS ORIGINAIS
from sklearn.ensemble import RandomForestClassifier
classificador = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)
previsoes_probabilisticas = classificador.predict_proba(previsores_teste)
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








"""PLOTS"""
#>> ROC
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

# Score
ns_probs = [0 for _ in range(len(classe_teste))]  # Majority class
ns_auc = roc_auc_score(classe_teste, ns_probs)
rf_auc = roc_auc_score(classe_teste, previsoes_probabilisticas[:, 1])

# Calcula a curva ROC
ns_fpr, ns_tpr, _ = roc_curve(classe_teste, ns_probs)
rf_fpr, rf_tpr, _ = roc_curve(classe_teste, previsoes_probabilisticas[:, 1])

# Plot ROC
plt.figure(figsize=(16, 9))
plt.plot([0, 1], [1, 1], c='0.7')
plt.plot([0, 0], [0, 1], c='0.7')
plt.plot(ns_fpr, ns_tpr, label='No Skill (ROC AUC = %0.3f)' % ns_auc, linestyle='--', linewidth=2)
plt.plot(rf_fpr, rf_tpr, label='Random Forest (ROC AUC = %0.3f)' % rf_auc, marker='o', linewidth=2)
plt.ylabel("Verdadeiro Positivo", fontsize=10)
plt.xlabel("Falso Positivo", fontsize=12)
plt.legend(loc='lower right', fontsize=12)
plt.show()



#>> Recall
from sklearn.metrics import f1_score
from sklearn.metrics import auc

# Score
rf_precision, rf_recall, thresholds = precision_recall_curve(classe_teste, previsoes_probabilisticas[:, 1])
no_skill = len(classe_teste[classe_teste==1]) / len(classe_teste)

# Calcula a curva
rf_f1 = f1_score(classe_teste, previsoes)
rf_auc = auc(rf_recall, rf_precision)

# Plot Recall
plt.figure(figsize=(16, 9))
plt.plot([1, 1], [1, 0], c='0.7')
plt.plot([0, 1], [1, 1], c='0.7')
plt.plot(rf_recall, rf_precision, label='Random Forest (F1 = %0.3f | AUC = %0.3f)' % (rf_f1, rf_auc), marker='o', linewidth=2)
plt.plot([0, 1], [no_skill, no_skill], label='No Skill (ROC AUC = %0.3f)' % ns_auc, linestyle='--', linewidth=2)
plt.ylabel("Precisão", fontsize=12)
plt.xlabel("Recall", fontsize=12)
plt.legend(loc='lower right', fontsize=10, edgecolor='1')
plt.show()





