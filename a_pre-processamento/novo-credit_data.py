# -*- coding: utf-8 -*-

__author__ = "Willian J Campos Almeida"
__copyright__ = "Copyright 2021, The Cogent Project"
__credits__ = ["Willian J Campos Almeida"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Willian J Campos Almeida"
__email__ = "willianjunio93@gmail.com"
__status__ = "Production"



import pandas as pd
import numpy as np
from os import path

base = pd.read_csv(path.join('arquivos', 'credit_data.csv'))

sex = np.random.uniform(low=0, high=2000, size=(2000, ))
sex_code = []

for i in range(len(sex)):
	if int(sex[i] % 2) == 0:
		sex_code.append('M')
	else:
		sex_code.append('F')

db_sex = pd.DataFrame(sex_code, columns=['sex'])

new_database = pd.concat([base, db_sex], axis=1,).reindex(base.index)
new_database = new_database[['clientid', 'income', 'age', 'sex', 'loan', 'default']]

new_database.to_csv(path.join('arquivos', 'credit-data.csv'), index=False)


