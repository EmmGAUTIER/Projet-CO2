"""
cla_streamlit : Feuille streamlit de classifications

Cette affiche des résultats de classifications de modèles
préalablement sélectionnées avec GridSearchCV
entraînés avec de meilleurs paramètres et sauvegardés.

Elle utilise les classifications suivantes :
  - BaggingClassifier ;
  - ExtraTreesClassifier ;
  - KNeighborsClassifier ;
  - RandomForestClassifier ;
  - GradientBoostingClassifier ;
  - HistGradientBoostingClassifier.

Ella propose les choix suivants :  
  - L'algorithme de classification ;
  - Le choix du tye de carburant : tous, diesel eu essence
  - La norme 

Elle calcule :
    - la régression demandée avec les données choisies ;
    - des métriques d'évaluation.  

Elle affiche :
  - la classe prédite ;
  - des métriques.

Les types de carburant et les normes proposés sont restreintes 
aux modalités majoritaires. Les autres ne permettent pas d'avoir
suffisement d'observation pour faire des régressions fiables.

"""

import streamlit         as st
import pandas            as pd
#import numpy             as np

#import time
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
from CO2_fcts                  import split_ap_type, libelles_vars

def st_classification(df) :
    ##############################################################################
    #  Affichage de l'entête et des premières lignes                             #
    ##############################################################################

    st.header("Algorithmes de classification")
    st.write("Les algorithmes entraînés nous permettent de réaliser des prédictions"
            "de classe d'émission de CO2")
    st.write("Ces algorithmes sont destinés à prédire des classes, ici les   "
             " classes d'émissions de CO2")

    st.write("L'entraînement des modèles est en cours.")

    ##############################################################################
    #  Lecture du jeu de données et initialisations                              #
    ##############################################################################

    # Chargement du fichier de paramètres (format JSON)
    # Ce fichier contient la liste des modèles, les conditions de tests,
    # des métriques et les meiileurs résultats
    file = open("cla_mdl.json", "r")
    cla_res = json.load(file)
    file.close()
    file = None
    # print (NVConfig) # pour mise au point

    # Suppression des modèles non émetteurs de CO2 (électrique et à H)
    dfs = df[df.Emetteur_CO2 == 1]
    # Sélection des colonnes pour la classification



    cols = [
        'Masse',
        'Empattement',
        'Larg_essieu_dir',
        'Cylindree',
        'Puiss_moteur',
        'Nap_norme',#Cette var. doit ê dichotomisée puis suppr car elle fait planter
        'Classe_CO2',
        'Carburant_Diesel',
        'Carburant_Diesel-Electrique',
        'Carburant_E85',
        'Carburant_Essence',
        'Carburant_Essence-Electrique',
        'Carburant_LPG',
        'Carburant_NG-Biomethane',
        ]

       
    #  Sélection des var., suppression de doublons, sexplicatives, sélection de la cible, création de vars.
    dfs = df[cols]
    dfs = dfs.drop_duplicates()
    #df = df.sample(4000) # permet de réduire le temps pendant la mmise au point
    data = dfs.drop('Classe_CO2', axis = 1)
    data = pd.concat([data, pd.get_dummies(data.Nap_norme, prefix = "Norme_")], axis = 1)
    data = data.drop("Nap_norme", axis = 1)
    target = dfs.Classe_CO2

    # Préparation d'un jeu de données d'entraînement et d'un jeu de tests
    # Nous utilisons random_state = 421 pour avoir toujours le mêmes jeux
    # Nous utilisons 80% des données pour l'entraînenmt et 20% pour les test


    X_train, X_test, y_train, y_test = train_test_split (data,
                                                         target,
                                                         random_state = 421,
                                                         test_size = 0.2)

    normes_proposees = ["2001/116", "2007/46"]
    carb_proposes = ["Diesel", "Essence"]

    modeles       = cla_res["Modèles"]
    modeles_noms  = [] # Liste des libellé pour le choix du modèle

    ##########################################################################
    #  Chargement des modèles précédemment entraînés                         #
    ##########################################################################
   
    #st.text ("Nous avons essayé les modèles suivants : ")
    # st.text(str(modeles)) #Pour mise au point
    for mdl in modeles :
        nom = mdl["Libellé"]
        #st.text(f"  - {nom:s} ;")
        modeles_noms.append(mdl["Libellé"])
        # TODO : tenter de charger le modèle entraîné s'il a été enregistré
        mdl["Préparé"] = None
    #st.write(modeles)#

    ##############################################################################
    #  Création des curseurs et menus pour les choix.                            #
    ##############################################################################

    modele_choisi       = st.selectbox("Modèle de classification : ", modeles_noms)

    masse_choisie       = st.slider(libelles_vars["Masse"],
                                    min_value = int(df['Masse'].min()),
                                    max_value = int(df['Masse'].max()),
                                    value     = int(df['Masse'].median()),
                                    step=10)
    empattement_choisi  = st.slider(libelles_vars["Empattement"],
                                    min_value = int(df['Empattement'].min()),
                                    max_value = int(df['Empattement'].max()),
                                    value     = int(df['Empattement'].median()),
                                    step      = 10)

    larg_essieu_choisie = st.slider(libelles_vars["Larg_essieu_dir"],
                                    min_value = int(df['Larg_essieu_dir'].min()),
                                    max_value = int(df['Larg_essieu_dir'].max()),
                                    value     = int(df['Larg_essieu_dir'].median()),
                                    step      = 10)

    puissance_choisie   = st.slider(libelles_vars["Puiss_moteur"],
                                    min_value = int(df['Puiss_moteur'].min()),
                                    max_value = int(df['Puiss_moteur'].max()),
                                    value     = int(df['Puiss_moteur'].median()),
                                    step      = 10)

    cylindree_choisie   = st.slider(libelles_vars["Cylindree"],
                                    min_value = int(df['Cylindree'].min()),
                                    max_value = int(df['Cylindree'].max()),
                                    value     = int(df['Cylindree'].median()),
                                    step      = 10)

    norme_choisie       =  st.selectbox('Choix de la norme :', normes_proposees)

    type_carb_choisi    =  st.selectbox('Type d\'énergie :', df.Type_carburant.unique())

    ##############################################################################
    #  Prédiction et affichage du résultat                                       #
    ##############################################################################

    data_essai = pd.DataFrame({
            'Masse'                        : [masse_choisie],
            'Empattement'                  : [empattement_choisi],
            'Larg_essieu_dir'              : [larg_essieu_choisie],
            'Cylindree'                    : [cylindree_choisie],
            'Puiss_moteur'                 : [puissance_choisie],
            'Carburant_Diesel'             : [1 if type_carb_choisi == "Diesel" else 0],
            'Carburant_Diesel-Electrique'  : [1 if type_carb_choisi == "Diesel-Electrique" else 0],
            'Carburant_E85'                : [1 if type_carb_choisi == "E85" else 0],
            'Carburant_Essence'            : [1 if type_carb_choisi == "Essence" else 0],
            'Carburant_Essence-Electrique' : [1 if type_carb_choisi == "Essence-Electrique" else 0],
            'Carburant_LPG'                : [1 if type_carb_choisi == "LPG" else 0],
            'Carburant_NG-Biomethane'      : [1 if type_carb_choisi == "NG-Biomethane" else 0],
            'Norme__2001/116'              : [1 if norme_choisie    == "2001/116" else 0],
            'Norme__2007/46'               : [1 if norme_choisie    == "2007/46" else 0],
            'Norme__96/79'                 : [1 if norme_choisie    == "96/79" else 0],
            'Norme__98/14'                 : [1 if norme_choisie    == "98/14" else 0],
            'Norme__KS07/46'               : [1 if norme_choisie    == "KS07/46" else 0],
        })

    #st.write ("Num. du mdle : " + str(modeles_noms.index(modele_choisi)))

    modele = modeles[modeles_noms.index(modele_choisi)]
    #st.write(modele)
    nom  = modele["Nom"]
    prms = modele["Prm_fixes"]
    prms.update(modele["Prm_best"])
    #st.write(f"{nom:s} {str(prms):s}")
    #st.write(modele["Prm_best"])
    #st.write ("----->" + str(modele["Nom"]))
    #st.write ("----->" + str(modele["Préparé"]))
    if modele["Préparé"] == None :
        if nom == "BaggingClassifier" :
            modele["Préparé"] = BaggingClassifier(**prms)
        if nom == "ExtraTreesClassifier" :
            modele["Préparé"] = ExtraTreesClassifier(**prms)
        if nom == "KNeighborsClassifier" :
            modele["Préparé"] = KNeighborsClassifier(**prms)
        if nom == "RandomForestClassifier" :
            modele["Préparé"] = RandomForestClassifier(**prms)
        if nom == "GradientBoostingClassifier" :
            modele["Préparé"] = GradientBoostingClassifier(**prms)
        if nom == "HistGradientBoostingClassifier" :
            modele["Préparé"] = HistGradientBoostingClassifier(**prms)
        modeles[modeles_noms.index(modele_choisi)] = modele
        #st.write ("----->" + str(modele["Nom"]))
        #st.write ("----->" + str(modele["Préparé"]))
        #st.write ("Entraînement")
        modele["Préparé"].fit(X_train, y_train)
        #st.write("Entraîné")
    else :
        #st.write("Déjà préparé")
        pass

    #st.write ("Classifieur préparé :      : " +  str(modele["Préparé"]))
    y_essai = modele["Préparé"].predict(data_essai)
    res = y_essai
    st.write(f"Classe prédite : {res}")
