# -*- coding: utf-8 -*-
"""
Descrição:
	O objetivo deste arquivo é realizar a classificação da base de dados
	'census_orange', preparada especificamente para a utilização do Orange,
	utilizando o Algoritmo de Aprendizagem por Regra chamado CN2Learner da
	biblioteca Orange.
===============================================================================
Base de dados:
	Na base de dados 'census_orange.csv' os campos (colunas) ou atriutos
	são separados por vírgula. Além disso, no atributo 'income' foi
	adicionado o caractere 'c#' para identificá-lo como atributo meta ou
	classe.

Atributos:
	-- age: previsor que define a idade do indivíduo
	-- workclass: previsor que classifica o tipo de trabalho do indivíduo
	-- final-weight:
	-- education: previsor que especifica a escolaridade do indivíduo
	-- education-num: previsor que define a quantidade (em anos) em que o
	indivíduo estudou.
	-- marital-status: previsor que especifica o status social do indivíduo
	-- occupation: previsor que define o cargo do indivíduo
	-- relationship: tipo de relação social do indivíduo.
	-- race: especifica a raça do indivíduo
	-- sex: especifica o sexo do indivíduo
	-- capital-gain:
	-- capital-loos:
	-- hour-per-week: horas de trabalho por semana
	-- native-country: pais de nacionalidade
	
	-- income: classe ou meta que define a renda do indivíduo.
"""

__author__ = "Willian J Campos Almeida"
__copyright__ = "Copyright 2021, Machine Learning Project"
__credits__ = ["Willian J Campos Almeida"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Willian J Campos Almeida"
__email__ = "willianjunio93@gmail.com"
__status__ = "Production"



from os import path
import Orange

base = Orange.data.Table(path.join('arquivos', 'census_orange.csv'))

# O simbolo | (barra) separa os previsores do atributo meta
base.domain  # Exibe o nome os atributos da base de dados


# Divide a base de dados em teste e treinamento, gerando uma tupla com duas tabelas
base_dividida = Orange.evaluation.testing.sample(base, n=0.25)
# A tabela 1 está relacionada a base de dados de treinamento
base_treinamento = base_dividida[1]
# A tabela 0 está relacionada a base de dados de teste
base_teste = base_dividida[0]

# Cria o learner
cn2_learner = Orange.classification.rules.CN2Learner()
# Gera as regras segundo o learner
# Dependendo do tamanho da base de dados, pode levar muito tempo para gera as regras
classificador_cn2 = cn2_learner(base_treinamento)


# Exibe as regras geradas
cont = 1
for regras in classificador_cn2.rule_list:
	print('Regra %d: %s' %(cont, regras))
	cont += 1
del cont


# Verificar o percentual de acerto
resultado = Orange.evaluation.testing.TestOnTestData(base_treinamento, base_teste, [lambda testdata: classificador_cn2])
precisao = Orange.evaluation.CA(resultado)
print(precisao)



