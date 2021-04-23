# -*- coding: utf-8 -*-
"""
Descrição:
	O objetivo deste arquivo é realizar a classificação da base de dados
	'credit_data_orange', preparada especificamente para a utilização do Orange,
	utilizando o Algoritmo de Aprendizagem por Regra chamado Majority Learner
	da biblioteca Orange.
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



from os import path
import Orange

# Carrega a base de dados
base = Orange.data.Table(path.join('arquivos', 'credit_data_orange.csv'))

# O simbolo | (barra) separa os previsores do atributo meta
base.domain  # Exibe o nome os atributos da base de dados


# Divide a base de dados em teste e treinamento, gerando uma tupla com duas tabelas
base_dividida = Orange.evaluation.testing.sample(base, n=0.25)
# A tabela 1 está relacionada a base de dados de treinamento
base_treinamento = base_dividida[1]
# A tabela 0 está relacionada a base de dados de teste
base_teste = base_dividida[0]

# Cria o learner
cn2_learner = Orange.classification.MajorityLearner()
# Gera as regras segundo o learner
classificador_majority = cn2_learner(base_treinamento)


# Verificar o percentual de acerto
resultado = Orange.evaluation.testing.TestOnTestData(base_treinamento, base_teste, [lambda testdata: classificador_majority])
precisao = Orange.evaluation.CA(resultado)
print(precisao)

# Uma outra forma de encontrar o Base Line Classifier
from collections import Counter
print(Counter(str(d.get_class()) for d in base_teste))

# 425 relacionado ao número de registros com resultado 0
contagem = 425 / len(base_teste)


