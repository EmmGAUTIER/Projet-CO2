#!/usr/bin/python3

from  lazypredict.Supervised   import LazyRegressor
from  sklearn.model_selection  import train_test_split
from  sklearn.svm              import LinearSVC

import pandas as pd

df = pd.read_csv('Emissions_CO2_FR.csv')
df = df[df.Emetteur_CO2 == 1]
#df = df[df.Type_carburant == 'Essence']
dfm = df[['Masse',
        'Empattement',
        'Larg_essieu_dir',
        #'Total_nvle_inscr',
        'Cylindree',
        'Puiss_moteur',
        'Emis_CO2_spe'
         ]]

#data['Surface_essieux'] = data.Empattement * data.Larg_essieu_dir
#data = data.drop(['Empattement', 'Larg_essieu_dir'], axis = 1)

dfm = dfm.drop_duplicates()
print ("Taille données : ", dfm.shape)

dfm.info()

data = dfm.drop('Emis_CO2_spe', axis = 1)
target = dfm.Emis_CO2_spe

print ("Taille données : ", data.shape)
print ("Colonnes essayées : ", data.columns)

#print (data.info())
#print (data.head(10))

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=.2, random_state =123)

#You can set `force_row_wise=true` to remove the overhead.
#And if memory is not enough, you can set `force_col_wise=true`.

#params = { "force_row_wise" : True }

#clfs =[LinearSVC()] 

reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)

models,predictions = reg.fit(X_train, X_test, y_train, y_test)

#--------------8<----------------------------------------
print(models)

