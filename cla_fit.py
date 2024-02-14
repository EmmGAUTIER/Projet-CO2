"""
cla_fit.py : Entraînement de modèles de classifications.

Ces entraînements servent à entraîner les modèles de classification
avec les meilleurs paramètres et d'autres prarmètres pour que les
feuilles stremlit permettent des affichages rapides avec les modèles
préalablement entraînés

La liste des modèles et les paramètres essayés sont dans le fichier cla.json
(JSON est un format simple et très courant pour stocker des données structurées.)

Il réalise :
  - la lecture des données ;
  - l'entraînement des modèles ;
  - l'enregistrement des modèles retenus ;
  - la réalisation de rapports en texte brut.

Le rapport en texte brut est destiné à faire des copier/coller
pour des rapports. Il contient, pour chaque modèle,
les meilleurs paramètres et les métriques de tests.

"""

import streamlit         as st
import pandas            as pd
import numpy             as np

import time
import json

from sklearn.model_selection   import GridSearchCV

#from sklearn.svm               import SVC
from sklearn.model_selection   import train_test_split
#from sklearn.linear_model      import LinearRegression
from sklearn.neighbors         import KNeighborsClassifier
from sklearn.linear_model      import LogisticRegression
#from sklearn.feature_selection import SelectKBest
#from sklearn.ensemble          import VotingClassifier
from sklearn.ensemble          import HistGradientBoostingClassifier
from sklearn.ensemble          import GradientBoostingClassifier
from sklearn.ensemble          import BaggingClassifier
from sklearn.ensemble          import ExtraTreesClassifier
#from sklearn.ensemble          import ExtraTreesRegressor
from sklearn.model_selection   import KFold
from sklearn.model_selection   import cross_validate
from sklearn.model_selection   import cross_val_score
from sklearn.metrics           import mean_squared_error, f1_score, r2_score, confusion_matrix, classification_report
#from premier_rapport           import premier_rapport
#from CO2_fcts                  import split_ap_type
from sklearn.feature_selection import f_regression
from sklearn.ensemble          import RandomForestClassifier
from sklearn.ensemble          import RandomForestRegressor
from sklearn.ensemble          import AdaBoostRegressor
from sklearn.ensemble          import ExtraTreesRegressor
from joblib import dump, load

##############################################################################
#  Initialisations                                                           #
##############################################################################

# Ouverture du fichier de résultats et effacement s'il existe
#fic_resultat = open("cla_res.txt", "w")
fic_resultat = open("tmp.txt", "w")

# Chargement du fichier de paramètres (format JSON)
file = open("cla_cfg.json", "r")
NVConfig = json.load(file)
file.close()
file = None
# print (NVConfig) # pour mise au point

resultats = {}

# Création de la liste des colonnes
# Toutes les colonnes apparaissent, elles sont commentées lorsqu'elles
# ne sont pas utilisées. Cela permet de modifier la liste si besoin.
cols = [
        #'Marque',
        #'Nom_fab',
        #'Numero_approb_type',
        #'Type',
        #'Variante',
        #'Version',
        #'Nom_commercial',
        #'Cat_vehicule',
        #'Total_nvle_inscr',
        #'Emis_CO2_spe',
        'Masse',
        'Empattement',
        'Larg_essieu_dir',
        #'Larg_essieu_aut',
        #'Type_carburant',
        #'Mode_carburant', 
        'Cylindree',
        'Puiss_moteur',
        #'Conso_elect',
        #'Nap_etat',
        'Nap_norme',#Cette var. doit ê dichotomisée puis suppr car elle fait planter
        #'Nap_n1',
        #'Nap_n2',
        #'Classe_CO2',
        'Carburant_Diesel',
        'Carburant_Diesel-Electrique',
        'Carburant_E85',
        #'Carburant_Electrique',
        'Carburant_Essence',
        'Carburant_Essence-Electrique',
        #'Carburant_Hydrogene',
        'Carburant_LPG',
        'Carburant_NG-Biomethane',
        #'Mode_carb_B',
        #'Mode_carb_F',
        #'Mode_carb_M',
        #'Hybride',
        #'Emetteur_CO2'
        ]

##############################################################################
#  Lecture du jeu de données et préparations                                  #
##############################################################################

# Lecture du jeu de données
df = pd.read_csv('Emissions_CO2_FR.csv')
# Suppression des modèles non émetteurs de CO2 (électrique et à H)
df = df[df.Emetteur_CO2 == 1]

print (df.info())
print (cols)

# Sélection des var. explicatives, sélection de la cible, création de vars.
data = pd.DataFrame(df[cols])
data = pd.concat([data, pd.get_dummies(data.Nap_norme, prefix = "Norme_")], axis = 1)
data = data.drop("Nap_norme", axis = 1)

print (data.info())
print (data.describe())
print (cols)
print ("Types de carburant :")
print (df.Type_carburant.value_counts())

print ("Nombre de doublons dans data : " , data.duplicated().sum())

exit(0)
target = df.Classe_CO2

# Création d'un jeu d'entraînement et d'un jeu de tests
X_train, X_test, y_train, y_test = train_test_split (data, target, test_size = 0.25, random_state = 421)

fic_resultat.write("variables explicatives utilisées pour les essais : ")
fic_resultat.write(str(data.columns))
fic_resultat.write("\n")

##############################################################################
#  Évaluation des modèles par GridSearchCV                                   #
##############################################################################

# Pour chaque modèle (de la liste dans le fichier json):
for cla in NVConfig["Modèles"] :
    fic_resultat.write ("------------ " + cla["Nom"] + " : " + cla["Libellé"] + " -----------------\n")
    fic_resultat.write ("\n")
    param_fixes = cla["Prm_fixes"]
    param_essayes = cla["Prm_essai"]
    fic_resultat.write ("Paramètres fixés                  : " + str(param_fixes)+"\n")
    fic_resultat.write ("Paramètres gérés par GridSearchCV : " + str(param_essayes)+"\n")

    # Création du classifier avec ses paramètres fixes (non gérés par GridSearchCV)
    # TODO Créer le classifieur directement avec le nom au lieu de ces if.
    model_nom = cla["Nom"]
    classifier = None # pour mise au point et tests
    if cla["Nom"] == "BaggingClassifier" :
        classifier = BaggingClassifier(**param_fixes)
    if cla["Nom"] == "ExtraTreesClassifier" :
        classifier = ExtraTreesClassifier(**param_fixes)
    if cla["Nom"] == "KNeighborsClassifier" :
        classifier = KNeighborsClassifier(**param_fixes)
    if cla["Nom"] == "RandomForestClassifier" :
        classifier = RandomForestClassifier(**param_fixes)
    if cla["Nom"] == "GradientBoostingClassifier" :
        classifier = GradientBoostingClassifier(**param_fixes)
    if cla["Nom"] == "HistGradientBoostingClassifier" :
        classifier = HistGradientBoostingClassifier(**param_fixes)
    #print ("Classifieur crée                  : ", classifier)
    #print (type(classifier))
    print ()
    print ("Entraînement : Recherche des meilleurs paramètres")
    grille_clf = GridSearchCV(classifier, param_grid = {}, cv = 5 )
    grille = grille_clf.fit (X_train, y_train)
    fic_resultat.write ("Paramètres retenus :")
    for it in grille.best_params_.items():
        fic_resultat.write (it[0] +  " : "+ it[1])
    print ("Entraînement : avec les meilleurs paramètres")
    model = grille_clf.best_estimator_
    model.fit(X_train, y_train)
    # Je tiens le meilleur ! je l'enregistre immédiatement !
    model_filename = f"model{model_nom}.pkl"
    dump(model, model_filename)

    # 
    y_pred = model.predict(X_test)
    fic_resultat.write ("Score sur le jeu d'entraînement : {:6.3f}\n".format(model.score(X_train, y_train)))
    fic_resultat.write ("Score sur le jeu de test        : {:6.3f}\n".format(model.score(X_test,  y_test)))
    fic_resultat.write ("Scores f1                       : \n")
    fic_resultat.write (str(f1_score(y_test, y_pred, average = None)))
    fic_resultat.write ("\n")
    fic_resultat.write (classification_report (y_test, y_pred))
    fic_resultat.write ("\n")
    #fic_resultat.write ("Score R2                        : {:s}\n".format(str(r2_score(y_test, y_pred))))
    fic_resultat.write ("")
    resultats


fic_resultat.close()
