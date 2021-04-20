# -*- coding: utf-8 -*-
"""
DESCRIÇÃO:
	O objetivo deste arquivo é realizar o pré-processamento da base de dados
	'credit_data.csv', localizado no diretório 'arquivos'. Além disso, serve
	como uma referência básica acerca dos processos relacionados ao
	pré-processamento de dados, para que, posteriormente tais dados possam ser
	utilizados em um Algoritmo de Machine Learnin.
===============================================================================
BASE DE DADOS:
	Na base de dados 'credit_data.csv' os campos, colunas ou atriutos estão
	separados por vírgula. Esta base de dados possui 5 (cinco) atributos, sendo
	1 (um) o atributo meta ou classe, além de possuir 2000 (dois mil) registros.
	
	É importante notar que, os dados presentes nesta base de dados são todos do
	tipo numérico ('int' e 'float').
===============================================================================
ATRIBUTOS:
	-- clienteid: int
		Previsor que define um id para cada registro.
	-- income: float
		Previsor que quantifica a renda anual do cliente.
	-- age: float
		Previsor que define a idade do cliente.
	-- loan: float
		Previsor que define o valor solicitado do emprestimo ou financiamento.
	-- default: int
		Classe ou Meta que define qual cliente não pagou (0) ou pagou (1) o
		financiamento ou empréstimo.
===============================================================================	
PROCEDIMENTO:
	- Análise Exploratória dos Dados
	--- Tratar dados inconsistentes
	--- Tratar valores faltantes
	--- Divisão dos Atributos
	- Escalonamento dos Atributos
	--- StandardScaler()
	--- Normalizer()
	--- MinMaxScaler()
	--- RobustScaler()
	--- QuantileTransformer()
	--- PowerTransformer()
	- Salvar a base de dados arquivos.
"""

__author__ = "Willian J Campos Almeida"
__copyright__ = "Copyright 2021, The Cogent Project"
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


"""============================================================================
>> ANÁLISE PRELIMINAR

Após o carregamento de uma base de dados, é importante realizar algumas
verificações acerca dos dados desta base.
==========================================================================="""

# Informações básicas da base de dados
# Exibe o tamanho da base de dados
print(base.shape)  # 2000 registros e 5 atributos

# Exibe os 10 (dez) primeiros registros da base de dados (valor padrão 5)
print(base.head(n=10))

# Exibe os 10 (dez) ultimos registros da base de dados (valor padrão 5)
print(base.tail(n=10))

# Verifica, a quantidade de dados, valores nulos e seus tipos
"""A função info() exibe algumas informações acerca da quantidade e tipo de
dados de cada uma das colunas ou atributos:
- #: indica o index de cada uma das colunas ou atributo
- Column: nome da coluna ou atributo
- Not-Nul Count: quantidade de valores contáveis (não nulos)
- Dtype: tipo de atributo definido pelo pandas.

NOTA: Observe que possui 1997 valores não nulos na coluna ou atributo 'age', o
que indica a falta de 3 (três) valores."""
print(base.info())

# Dados estatísticos descritivos da base de dados
"""A função describe() é usada para gerar estatísticas descritivas que resumem
a tendência central, a dispersão e a forma da distribuição de um conjunto de
dados, excluindo os valores NaN
- count: quantidade de registros
- mean: média das colunas
- std: desvio padrão
- min: valor mínimo encontrado na coluna
- 25%: primeiro quartil (mostra a localização do quartil)
- 50%: segundo quartil ou mediana (mostra a localização do quartil ou mediana)
- 75%: terceiro quartil (mostra a localização do quartil)
- max: valor máximo encontrado na coluna"""
# NOTA: Atributo 'age' possui valores negativos, o que revela inconsistência nos dados
print(base.describe())

# Dados estatísticos descritivos da base de dados (transposta)
print(base.describe().T)

# Conta quantos registros possuem valores em 'age' abaixo de 0 (zero)
print(len(base.loc[base['age'] < 0]))

# Verifica se há valores faltantes (missing) na base de dados
# NOTA: Existe valores nulos no atributo 'age'
print(base.isna().any())
print(base.isnull().any())
# Exibe a quantidade de registros com valores em 'age' nulos ou faltantes
print(base['age'].isna().sum())
print(base['age'].isnull().sum())

# Verifica se existem registros duplicados na base de dados
# NOTA 1: O retorno foi 'Empty DataFrame', ou seja, não existem registros duplicados
# NOTA 2: Caso necessário, basta usar o comando base.drop_duplicates()
print(base.loc[base.duplicated() == True])


"""============================================================================
>> RESUMO DA ANÁLISE PRELIMINAR

Ao analisar os resultados dos dados apresentados acima, notou-se alguns
problemas acerca da base de dados:
	
1. Existem 3 (três) valores inconsistentes (negativos) no atributo 'age', o que
corresponde a cerca de 0,15% da base de dados.

2. Existem 3 (três) valores nulos (faltantes) no atributo 'age', o que
correspnde a cerca de 0,15% da base de dados.
==========================================================================="""



"""===========================================================================
>> 1. DADOS INCONSISTENTES

Como foi descrito na ANÁLISE PRELIMINAR dos dados, existem valores
inconsistentes do atributo 'age' (valores negativos).
==========================================================================="""

# Listar objetos que possuem cadastro com 'age' negativa
'''Ao verificar o resultado da função describe(), foi constatado que existem
registros com idades negativas. Com este comando, é possível listar os
registros com estes valores.'''
print(base.loc[base['age'] < 0])


"""Para o tratamento destes dados inconsistentes, tem-se 4 técnicas que podem
ser executadas:
1. Apagar a coluna
2. Apagar os registros com problemas
3. Preencher os valores manualmente
4. Substituir os valores com o valor médio do atributo. É importante frizar que
este método pode variar dependendo do tipo do atributo (numético ou
categórico). No caso dos categóricos, o valor seria substituido pelo valor mais
frequente (moda)."""

# 1. Apagar a Coluna ou Atributo
"""Não é uma solução interessante pelo fato de apagar TODO o atributo em
detrimento de alguns dados. Geralmente, esta solução é utilizada caso grande
parte dos dados do atributo encontram-se nulos. Para apagar um atributo da base
de dados, basta executar o seguinte comando:

>> base.drop(labels='age', axis=1, inplace=True)

- labels='age': coluna que será apagada
- axis=1: apagar a coluna
- inplace=True: usar a mesma variável e não retorna valor"""


# 2. Apagar somente os registros com problemas
"""Assim como o método anterior, não é uma solução interessante pelo fato de
haver a possibilidade de conter registros com campos importantes que podem
fazer diferença no resultado final (dados com relações interessantes para o
algoritmo). De forma geral, não é recomendável apagar registros a não ser que
o registro apresente uma grande quantidade de valores com problemas.

Para apagar uma linha específica, uma alternativa é utilizar o comando abaixo:

>> base.drop(labels=base[base.age < 0].index, axis=0, inplace=True)

Neste comando, é passado como labels, uma busca de valores inferiores a 0 (zero)
e retorna o indice destes valores. Basicamente é o mesmo comando do método
anterior, a diferença está relacionada a busca por registros específicos e a
modificação do parâmetro 'axis' que por padrão é 0 (zero), que representa o
index ou registro."""


# 3. Preencher os valores manualmente
"""Esta seria a melhor solução possível, porém, pode não ser viável por não
haver meios de entrar em contato com as pessoas. Além disso, pode se tornar
inviável em detrimento da quantidade de dados. Para executar uma substituição
de um valor, pode-se utilizar o comando abaixo:

>> base.loc[base['clientid'] == 16, 'age'] = 30

Neste exemplo de comando, o registro com o 'clientid'=16 terá o valor do
atributo 'age' substituido pelo valor 30. Para exibir a linha modificada, basta
utilizar o comando abaixo:
	
>> print(base['clientid'] == 16])

Caso queira exibir somente as colunas 'clientid'=16 e 'age', utiliza-se o
comando abaixo

>> print(base.loc[base['clientid'] == 16, 'age'])

Existem vários métodos para selecionar linhas e colunas especificas. Para mais
informações, acesse a documentação."""


# 4. Substituir os valores com a média da coluna
"""Esse será o método utilizado para o tratamento desses valores. Para isso, é
importante observar algumas questões importantes:
1. O atributo 'age' é do tipo numérico.
2. Deve-se levar em consideração que existem valores incorretos (negativos) e
para realizar a média corretamente, é importante utilizar apenas os valores
corretos (positivos).

Assim, não se deve utilizar o comando mean() simplesmente como é mostrado no
comando abaixo, pois apresenta uma média levando em conta os valores
incoerentes."""

# Exibe a média de todas as colunas ou atributos
print('Média: ', base.mean())

# Exibe a média apenas da coluna ou atributo 'age'
print('\n Média de idades (incorreta): ', base['age'].mean())

# Considera apenas os dados que possuem idades maiores do que 0
print('Média de idades (correta): ', base['age'][base.age > 0].mean())

# Substitui os valores negativos pela média correta
base.loc[base.age < 0, 'age'] = base['age'][base.age > 0].mean()

"""Exibe os valores negativos da coluna ou atributo 'age'. Observa-se que o
retorno ou saída foi 'Empty DataFrame', o que significa que não há valores
abaixo de 0 (zero) na base de dados."""
print(base.loc[base['age'] < 0])



"""===========================================================================
>> 2. DADOS NULOS OU FALTANTES

Como foi descrito na ANÁLISE PRELIMINAR dos dados, existem valores nulos ou
faltantes no atributo 'age' (valores negativos).

Estes valores devem ser tratados, pois podem gerar problemas na execução dos
algoritmos de Machine Learning ou pode encontrar um padrão falso Existem alguns
algoritmos que já tratam esses valores faltantes automaticamente, mas é
importante realizar todas essas análises.

Antes de prosseguir, é necessário fazer a divisão da base de dados em atributos
previsores e atributos classe pois é assim que funciona a maioria das
bibliotecas em Python. Dessa forma, deve-se realizar a divisão da base de
dados.
==========================================================================="""

# Verifica se há valores faltantes (missing) na base de dados
print(base.isna().any())
print(base.isnull().any())
# Exibe a quantidade de registros com valores em 'age' nulos ou faltantes
print(base['age'].isna().sum())
print(base['age'].isnull().sum())


# Divisão dos Atributos em Previsores e Classe
"""O comando iloc realizará a divisão. Dentro do colchetes, a primeira parte
antes da vírgula refere-se às linhas. Quando colocado os dois-pontos indica
todas as linhas. Após a vírgula, é o intervalo de colunas ou atributos. Neste
caso, pelo ID não ter relação direta com os dados, não sendo interessante
trabalhar com estes dados nos algoritmos de aprendizagem de máquina, o mesmo
será ignorado e, por essa razão, começou da coluna 1 até a coluna 3 (conjunto
aberto). Os algoritmso de aprendizagem de máquina trabalham com generalização
onde analiza-se um conjunto de dados e econtra-se padrões, e como os IDs dos
clientes são dados únicos (chave primária), o algoritmo não encontrará padrões
nesses dados.

Dica: Sempre que trabalhar com dados que envolva campos IDs, estes podem ser
removidos, pois não possuem utilidade.

Observação: O 'values' é utilizado porque se deseja passar somente os valores
para a variável. Se não for colocado, as colunas inteiras incluindo os rótulos
da coluna, serão passados à variável previsores, o que poderá causar um erro no
processamento mais a frente no código. Além disso, alguns algoritmos não
trabalham com dados do tipo DataFrame, então será copiado apenas os valores."""

previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

# Biblioteca responsável por tratar os valores faltantes (NaN)
from sklearn.impute import SimpleImputer
import numpy as np

# Localiza os valores faltantes (missing_values=np.nan) e substituindo pela
# média da coluna (strategy='mean')
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Realiza o cálculo para a realização do tratamento dos valores
imputer = imputer.fit(previsores[:, 0:3])

# Aplica os cálculos de acordo com os valores encontrados na variável imputer
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])

"""Por questões de organização foi apagada a variavel imputer por não ser mais
necessário durante o código."""
del imputer



"""===========================================================================
>> ESCALONAMENTO DOS ATRIBUTOS PREVISORES

ATENÇÃO: Antes de tudo, é importante ressaltar que o processo de escalonamento
de atributos só é possível ser realizado com atributos do tipo numérico.

Ao realizar a comparação entre os valores dos previsores, é notável que existe
uma grande diferença entre a escala dos valores principalmente se compararmos
a renda com a idade, pois existe uma diferença muito grande entre os valores.

O problema de utilizar valores não-escalonados é que, se utilizar algoritmos
baseados em Distância Euclidiana, como é o caso do KNN, estes algoritmos terá
a tendência de levar mais em consideração os valores maiores.

O escalonamento fará com que todos os atributos possuam um mesmo peso, uma
mesma relevância. Além disso, em algoritmos que não são baseados em distância
euclidiana, por ter feito o escalonamento, estes algoritmos costumam executar
mais rápido.

Existem várias técnicas de escalonamento de valores. O Sklearn possui 6 (seis)
métodos específicos para escalonamento. Todos estes métodos se emcontram no
pacote sklearn.prepocessing.

Vale ressaltar que não existe a necessidade de fazer a padronização dos
atributos classe.
==========================================================================="""


# 1. Padronização (Standardisation)
"""O StandardScaler age sobre as colunas, porém seu método subtrai do valor em
questão a média da coluna e divide o resultado pelo desvio padrão. No final
tem-se uma distribuição de dados com desvio padrão igual a 1 e variância de
também de 1.

Esse método trabalha melhor em dados com distribuição normal porém vale a
tentativa para outros tipos de distribuições, além disso podemos deixar como
dica que esse método resulta em ótimos frutos quando usado em conjunto com
algoritmos como Linear Regression e Logistic Regression."""
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores_standard = scaler.fit_transform(previsores)
del scaler


# 2. Normalização (Normalization)
"""O Normalizer age reescalando os dados por linhas ou registro e não por
colunas ou atributos, ou seja, o Normalizer levará em contas os atributos da
base de dados e reescala os valores com base nesses.

O Normalizer é uma boa escolha quando a distribuição dos dados não é
normal/gaussiana ou quando não sabe qual é o tipo de distribuição dos dados."""
from sklearn.preprocessing import Normalizer
scaler = Normalizer()
previsores_normalizer = scaler.fit_transform(previsores)
del scaler


# 3. MinMaxScaler
"""O MinMaxScaler age sobre a coluna, ou seja, o cálculo da reescala é
feito de forma independente entre cada coluna, de tal forma que a nova escala
se dará entre 0 e 1 ou -1 e 1 se houver valores negativos na base de dados. De
forma geral, o MinMaxScaler subtrai o valor em questão pelo menor valor da
coluna e então divide pela diferença entre o valor máximo e mínimo.

Importante ressaltar que essa técnica funciona melhor se a distribuição dos
dados não for normal e se o desvio padrão for pequeno, além disso o
MinMaxScaler não reduz de forma eficaz o impacto de outliers e também preserva
a distribuição original. """
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
previsores_min_max = scaler.fit_transform(previsores)
del scaler


# 4. RobustScaler
"""O RobustScaler atua sobre as colunas e o diferencial deste método é a
combinação com o uso de quartis o que garante um bom tratamento dos outliers.
Em seu método, o RobustScaler subtrai a média do valor em questão e então
divide o resultado pelo segundo quartil. Importante notar que os outliers ainda
estão presentes porém estão representados dentro de uma escala em que o seu
impacto negativo é reduzido."""
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
previsores_robust = scaler.fit_transform(previsores)
del scaler


# 5. QuantileTransformer
"""O QuantileTransformer atua sobre as colunas e trata os outliers com uso de
quartis. Este método transforma os valores de tal forma que a distribuição
tende a se aproximar de uma distribuição normal. Uma observação importante é
que essa tranformação pode distorcer as correlações lineares entre as colunas.
Neste método todos os valores serão reescalados em um intervalo de 0 a 1 de tal
forma que os outliers não poderão mais ser distinguidos logo ao contrário do
RobustScaler o impacto da ação em cima dos outilers será grande."""
from sklearn.preprocessing import QuantileTransformer
scaler = QuantileTransformer()
previsores_quantile = scaler.fit_transform(previsores)
del scaler


# 6. PowerTransformer
"""O PowerTransformer atua sobre as colunas e procura transformar os valores
em uma distribuição mais normal, sendo indicado em situações onde uma
distribuição normal é desejada para os dados, além disso esse método ainda
suporta os métodos de transformação Box-Cox (dataset com dados positivos) e
Yeo-Johnson (dataset com dados positivos e negativos)."""
from sklearn.preprocessing import PowerTransformer
scaler = PowerTransformer()
previsores_power = scaler.fit_transform(previsores)
del scaler



"""===========================================================================
>> SALVAR OS DADOS

Com os dados devidamente tratados, os mesmos serão salvos em arquivos para que,
posteriormente, possam passar por um algoritmo de Machine Learning.

Todos os previsores e as classes foram salvos em um dataframe com cada uma das
colunas relacionada a um tipo de processo de escalonamento.
==========================================================================="""

np.savetxt(path.join('a_pre-processamento', 'pre-processamento-credit_data-previsores.data'), previsores, fmt='%f', delimiter=',')
np.savetxt(path.join('a_pre-processamento', 'pre-processamento-credit_data-previsores_standard.data'), previsores_standard, fmt='%f', delimiter=',')
np.savetxt(path.join('a_pre-processamento', 'pre-processamento-credit_data-previsores_normalizer.data'), previsores_normalizer, fmt='%f', delimiter=',')
np.savetxt(path.join('a_pre-processamento', 'pre-processamento-credit_data-previsores_min_max.data'), previsores_min_max, fmt='%f', delimiter=',')
np.savetxt(path.join('a_pre-processamento', 'pre-processamento-credit_data-previsores_robust.data'), previsores_robust, fmt='%f', delimiter=',')
np.savetxt(path.join('a_pre-processamento', 'pre-processamento-credit_data-previsores_quantile.data'), previsores_quantile, fmt='%f', delimiter=',')
np.savetxt(path.join('a_pre-processamento', 'pre-processamento-credit_data-previsores_power.data'), previsores_power, fmt='%f', delimiter=',')

np.savetxt(path.join('a_pre-processamento', 'pre-processamento-credit_data-classe.data'), classe, fmt='%f', delimiter=',')




