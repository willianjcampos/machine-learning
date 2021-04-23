# -*- coding: utf-8 -*-

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd
from os import path

base = pd.read_csv(path.join('arquivos', 'credit_data.csv'))
base.loc[base.age < 0, 'age'] = base['age'][base.age > 0].mean()
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])
del imputer
scaler = StandardScaler()
previsores_standard = scaler.fit_transform(previsores)
del scaler
scaler = Normalizer()
previsores_normalizer = scaler.fit_transform(previsores)
del scaler
scaler = MinMaxScaler()
previsores_min_max = scaler.fit_transform(previsores)
del scaler
scaler = RobustScaler()
previsores_robust = scaler.fit_transform(previsores)
del scaler
scaler = QuantileTransformer()
previsores_quantile = scaler.fit_transform(previsores)
del scaler
scaler = PowerTransformer()
previsores_power = scaler.fit_transform(previsores)
del scaler


pr = [previsores, previsores_standard, previsores_normalizer,
      previsores_min_max, previsores_robust, previsores_quantile,
      previsores_power]


# Divisão da Base de Dados
from sklearn.model_selection import train_test_split
t = np.arange(len(previsores))
i_treino, i_teste = train_test_split(t, test_size=0.25, random_state=0)
del t



"""===========================================================================
>> TREINAMENTO RANDOM FOREST
==========================================================================="""
from sklearn.ensemble import RandomForestClassifier
classificador = RandomForestClassifier(n_estimators=5, criterion='entropy', random_state=0)


def prever(classificador, previsores):
	prev = []
	prev_prob = []
	
	for i in range(len(previsores)):
		p = previsores[i]
		classificador.fit(p[i_treino], classe[i_treino])
		prev.append(classificador.predict(p[i_teste]))
		prev_prob.append(classificador.predict_proba(p[i_teste])[:, 1])
		
	return prev, prev_prob


prev, prev_prob = prever(classificador, pr)



"""===========================================================================
>> PRECISÃO (accuracy_score), AUC (roc_auc_score) e ROC (roc_curve)
==========================================================================="""
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


def calcular_precisao(classe, prev, acc):
	aux = []
	
	for i in range(len(prev)):
		aux.append(acc(classe, prev[i]))
		
	return aux


precisao = calcular_precisao(classe[i_teste], prev, accuracy_score)
rf_auc = calcular_precisao(classe[i_teste], prev_prob, roc_auc_score)
rf_roc = calcular_precisao(classe[i_teste], prev_prob, roc_curve)

# Majority learn
ns_prob = [0 for _ in range(len(classe[i_teste]))]
ns_auc = roc_auc_score(classe[i_teste], ns_prob)
ns_fpr, ns_tpr, _ = roc_curve(classe[i_teste], ns_prob)



"""===========================================================================
>> PLOT ROC
==========================================================================="""
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

n_estimator = 5
criterium = 'entropy'

obs = r'n_estimator: %d | criterium: %s' % (n_estimator, criterium)
valores = ['Sem Escalonamento',
		   'StandardScaler()',
		   'Normalizer()',
		   'MinMaxScaler()',
		   'RobustScaler()',
		   'QuantileTransformer()',
		   'PowerTransformer()']

def plotar(ns_fpr, ns_tpr, ns_auc, rf_roc, rf_auc, ax, obs, valores):
	# Estrutura basica
	ax.plot([0, 1], [1, 1], c='0.7')
	ax.plot([0, 0], [0, 1], c='0.7')
	ax.grid()
	ax.xaxis.set_minor_locator(AutoMinorLocator())
	ax.yaxis.set_minor_locator(AutoMinorLocator())
	ax.tick_params(which='major', direction='inout', length=7, width=2)
	ax.tick_params(which='minor', direction='out', length=4, grid_color='red', grid_alpha=1)
	
	# Majority
	ax.plot(ns_fpr, ns_tpr, label='Majority (AUC = %0.3f)' % ns_auc, linestyle='--', linewidth=2)
	
	for i in range(len(rf_roc)):
		aux = rf_roc[i]
		rf_fpr = aux[0]
		rf_tpr = aux[1]
	
		ax.plot(rf_fpr, rf_tpr, label='%s (AUC = %0.3f)' % (valores[i], rf_auc[i]), marker='o', linewidth=2)
	
	# Informações
	ax.set_title("Receiver Operating Characteristic (ROC): Random Forest", pad=20, fontsize=25, fontweight='bold')
	ax.set_ylabel("Verdadeiro Positivo", fontsize=15, labelpad=10)
	ax.set_xlabel("Falso Positivo", fontsize=15, labelpad=10)
	ax.legend(loc='lower right', fontsize=10, facecolor='white', framealpha=1)
	ax.text(0.75, 0.275, obs, transform=ax.transAxes, fontsize=10)
	

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
plotar(ns_fpr, ns_tpr, ns_auc, rf_roc, rf_auc, ax, obs, valores)
plt.show()


