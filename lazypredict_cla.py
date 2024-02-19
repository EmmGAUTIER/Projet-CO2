#!/usr/bin/python3

from  lazypredict.Supervised   import LazyClassifier
from  sklearn.model_selection  import train_test_split
from  sklearn.svm              import LinearSVC
from  imblearn.under_sampling  import ClusterCentroids, RandomUnderSampler

import pandas as pd

df = pd.read_csv('Emissions_CO2_FR.csv')
#df = df.sample(frac = 0.99)
df = df[df.Emetteur_CO2 == 1]

df.info()

data = df[['Masse',
        'Empattement',
        'Larg_essieu_dir',
        'Total_nvle_inscr',
        'Cylindree',
        'Puiss_moteur',
        'Type_carburant',
        'Mode_carburant',
        #'Nap0',
        #'Nap_etat',
        'Nap_norme',
        #'Nap_n1',
        #'Nap_n2',
        'Hybride'
         ]]



target = df.Classe_CO2

#print(data.head(10))
print ("Taille : ", data.shape)
print ("Colonnes : ", data.columns)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=.2, random_state=123)
ccs = RandomUnderSampler(random_state=8421)
X_train, y_train = ccs.fit_resample(X_train, y_train)

#You can set `force_row_wise=true` to remove the overhead.
#And if memory is not enough, you can set `force_col_wise=true`.

params = { "force_row_wise" : True }

clfs =[LinearSVC()] 

clf = LazyClassifier(verbose=1, ignore_warnings=True, custom_metric=None)#, classifiers = clfs)

models,predictions = clf.fit(X_train, X_test, y_train, y_test)

print ("================================================================================")
print(models)

print ("================================================================================")
print (predictions)

