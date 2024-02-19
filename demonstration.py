"""
cla_streamlit : Feuille streamlit de présentation du projet CO2

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
  - l'algorithme de classification ;
  - le choix du tye de carburant : Tous, Diesel eu Essence
  - le choix d'une norme : "2007/46" et "2001/116"

Elle réalise les opérations suivantes :
  - prédiction de la classe ;
  - affichage de la classe.

Les types de carburant et les normes proposés sont restreintes 
aux modalités majoritaires. Les autres ne permettent pas d'avoir
suffisement d'observation pour faire des régressions fiables.

"""

import streamlit         as st
import pandas            as pd
import numpy             as np

import time
import json

from sklearn.preprocessing     import StandardScaler
from sklearn.model_selection   import GridSearchCV

#from sklearn.svm               import SVC
from sklearn.model_selection   import train_test_split
from sklearn.linear_model      import LinearRegression
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
from xgboost                   import XGBRegressor

##############################################################################
#  Affichage de l'entête et des premières lignes                             #
##############################################################################

st.header("Émission de CO2 des véhicules")
st.write("Les algorithmes entraînés nous permettent de réaliser des prédictions"
        "de classe d'émission de CO2")

##############################################################################
#  Lecture du jeu de données et initialisations                              #
##############################################################################

# Chargement du fichier de paramètres (format JSON)
file = open("cla_cfg.json", "r")
NVConfig = json.load(file)
file.close()
file = None
# print (NVConfig) # pour mise au point

# Lecture du jeu de données
# Ces données servent à obtenir les valeurs extrèmes pour les curseurs
df = pd.read_csv('Emissions_CO2_FR.csv')
# Suppression des modèles non émetteurs de CO2 (électrique et à H)
df = df[df.Emetteur_CO2 == 1]

# Lecture de la liste des modèles de classification
# TODO : utiliser un fichier généré lors l'enregistrement des modèles avec résultats.
file = open("cla_cfg.json", "r")
cla_cfg = json.load(file) 
file.close()

normes_proposees = ["2001/116", "2007/46"]
carb_proposes = ["Diesel", "Essence"]

modeles_noms  = [] # Noms des modèles préalablement entraînés
modeles       = [] # Modèles préalablements entraînés
param_essayes = [] # Listes (pour chaque modèle) des paramètres essayés
param_fixes   = [] 
param_best    = []

##############################################################################
#  Chargement des modèles précédemment entraînés                             #
##############################################################################

# Pour chaque modèle (de la liste dans le fichier json):
for cla in cla_cfg["Modèles"] :
    param_fixes   = cla["Prm_fixes"]
    param_essayes = cla["Prm_essai"]
    nom = cla["Nom"]
    modele = load(f"model{nom}.pkl")
    modeles.append(modele)
    modeles_noms.append(nom)

##############################################################################
#
# Feuille de régressions
# 
##############################################################################

def feuille_regressions() :
    pass

    # Création des régresseurs
    lr      = LinearRegression()
    rfr     = RandomForestRegressor()
    etr     = ExtraTreesRegressor()
    xgb_reg = XGBRegressor()

    # Création de la liste des variables quantitatives retennues pour la rég.
    df_reg = df[['Masse',
                'Empattement',
                'Larg_essieu_dir',
                'Cylindree',
                'Puiss_moteur',
                'Emis_CO2_spe']]

    chx_carb = ["Tous", "Diesel", "Essence"]
    #chx_norme = ["2001/116", "2001/46"]
    chx_reg   = ['LinearRegression',
                 'ExtraTreesRegressor',
                 'RandomForestRegressor',
                 'XGBRegressor']

    reg_choisie = st.selectbox("Choisissez la régression", chx_reg)

    col1, col2 = st.columns(2)

    with col1:
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
    with col2:
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

    # norme_choisie       =  st.selectbox('Choix de la norme :', normes_proposees)

        type_carb_choisi           =  st.selectbox('Type d\'énergie :', df.Type_carburant.unique())

    st.header("Choisissez les données et le modèle : ")
    carb_choisi = st.selectbox('Choix du type de carburant',
                               chx_carb)

    #norme_choisie = st.selectbox('Choix de la norme',
    #                             chx_norme)

    if reg_choisie == "LinearRegression" :
        model_reg = lr
    elif reg_choisie == "ExtraTreesRegressor" :
        model_reg = rfr
    elif reg_choisie == "RandomForestRegressor" :
        model_reg = etr
    else : #XGBRegressor
        model_reg = xgb_reg

    # Types de carburant proposés
    chx_carb = ["Essence", "Diesel"]

    # Séparation du jeu de données
    X = df_reg.drop("Emis_CO2_spe", axis = 1)
    y = df_reg["Emis_CO2_spe"]

    # Création d'un jeu d'entraînement et d'un jeu de tests
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardisation des données
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Entraînement du modèle
    model_reg.fit(X_train, y_train)

	  # Prédiction
    y_pred = model_reg.predict(X_test)

    # Calcul des métriques (RMSE et R^2)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Affichage des résultats
    st.write(f"RMSE : {rmse}")
    st.write(f"R²   : {r2}")

    vars_choisies = pd.DataFrame({'Masse'           : [masse_choisie],
                                  'Empattement'     : [empattement_choisi],
                                  'Larg_essieu_dir' : [larg_essieu_choisie],
                                  'Cylindree'       : [cylindree_choisie],
                                  'Puiss_moteur'    : [puissance_choisie],
	                                })

    # Attention : il faut normaliser les données à essayer par le modèle
    # en effet ce dernier  été entraîné avec des données normalisées.
    # il est alors possible de faire une prédiction et l'afficher.
    y_essai = model_reg.predict(scaler.transform(vars_choisies))
    st.write (f"Valeurs prédite {y_essai[0]:.0f} g/km")

##############################################################################
#
# Feuille de classification
# 
##############################################################################

def feuille_classifications() :
# Création des curseurs et menus pour les choix.
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

    type_carb_choisi           =  st.selectbox('Type d\'énergie :', df.Type_carburant.unique())

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

    # Pour mise au point
    #st.dataframe(data_essai)

    y_essai = modele.predict(data_essai)

    res = y_essai
    st.write(f"Classe prédite : {res}")

    #st.write(y_essai)

# Sidebar avec les différentes pages
st.sidebar.title("Sommaire")
pages = ["Contexte du projet",
         "Données et méthodologie",
         "Exploration et visualisation",
         "Régressions",
         "Classification",
         "Conclusion et perspective"]

page = st.sidebar.radio("Aller vers", pages)

# Condition pour afficher le contenu de la page sélectionnée
if page == "Classification":
    feuille_classifications()
if page == "Régressions":
    feuille_regressions()

st.text("Ici")
