"""
regressions.py : streamlit, calcul et affichage de régressions.

Il propose :
  - les régressions suivantes : linéaire et Extra Trees Regression ;
  - le choix du tye de carburant : tous, diesel eu essence
  - le choix d'une norme : "2007/46" et "2001/116"

Il calcule :
    - la régression demandée avec les données choisies ;
    - des métriques d'évaluation.
  
Il affiche :
    - un nuage de points TODO détailler ;
    - les résultats des métriques.

Les types de carburant et les normes proposés sont restreintes 
aux modalités majoritaires. Les autres ne permettent pas d'avoir
suffisement d'observation pour faire des régressions.
"""

import streamlit         as st
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import seaborn           as sns
import plotly.express    as px
import time

#from sklearn.model_selection   import GridSearchCV
from sklearn.model_selection   import train_test_split
from sklearn.linear_model      import LinearRegression
from sklearn.ensemble          import ExtraTreesRegressor
#from sklearn.model_selection   import KFold
from sklearn.model_selection   import cross_validate
from sklearn.model_selection   import cross_val_score
from sklearn.metrics           import mean_squared_error, f1_score, r2_score, confusion_matrix, classification_report
from sklearn.ensemble          import ExtraTreesRegressor

from CO2_fcts                  import split_ap_type  # Quelques fcts auxiliaires et un dictionnaire pour les libellés

st.title("Régressions linéaires")

###############################################################################
#  Lecture du fichier de données, quelques initialisation                     #
###############################################################################

df = pd.read_csv('Emissions_CO2_FR.csv',  low_memory=False)

# Sélection des seuls émetteurs de CO2 (modèles élec. et à H exclus)
df = df[df.Emetteur_CO2 == 1]

# Création de la liste des variables quantitatives retennues pour le rég.
data_vars = ['Masse',
             'Empattement',
             'Larg_essieu_dir',
             'Cylindree',
             'Puiss_moteur']
target_var = 'Emis_CO2_spe'

# Création des régresseurs
lr  = LinearRegression()
etr = ExtraTreesRegressor()

# Pour la mise au point l'instruction suivante réduit le temps de calcul
#df = df.sample(3000)
st.write ("Nombre d'observations : ", df.shape[0])

# Types de carburant proposés
chx_carb = ["Tous", "Essence", "Diesel"]

# Choix de la norme.
# Il y a deux modalités prépondérantes et quelques autres rares.
# Nous proposons toutes ou chacune de deux plus fréquentes.
chx_norme = ["Toutes", "2007/46", "2001/116"]

chx_reg = ["Régression linéaire", "Extra Trees Regressor"]

###############################################################################
#  Affichage des choix de graphiques                                          #
###############################################################################

st.header("Choisissez les données et le modèle : ")
carb_choisi = st.selectbox('Choix du type de carburant',
                           chx_carb)

norme_choisie = st.selectbox('Choix de la norme',
                             chx_norme)

reg_choisie = st.selectbox("Choix de la régression",
                           chx_reg)

#st.button("Diesel")

#st.write ("Carburant  : '", carb_choisi, "'")
#st.write ("Norme      : ", norme_choisie)
#st.write ("Régression : ", reg_choisie)

###############################################################################
#  Sélection des données, calculs et affichage                                #
###############################################################################

# Sélection des données choisies
# df contient le jeu de données complet et garde ce jeu pour éviter les relectures.
# dfs contient les données retenues
dfs = df

# Sélection des observation : type de carburant et norme
if carb_choisi != "Tous" :
    dfs = dfs[dfs.Type_carburant == carb_choisi]
if norme_choisie != "Toutes" :
    dfs = dfs[dfs.Nap_norme == norme_choisie]

# Séparation des variables explicatives et de la variable cibles
data   = dfs[data_vars]
target = dfs[target_var]

# Création d'un jeu d'entraînement et d'un jeu de tests
X_train, X_test, y_train, y_test = train_test_split (data, target, test_size = 0.2, random_state = 421)

# Calcul de la régression 
reg = lr if reg_choisie == "Régression linéaire" else etr
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

fig = plt.figure(figsize=(4, 4))
hue = 'Type_carburant' if carb_choisi == "Tous" else None

plt.scatter(y_test, y_pred, s = 2)
plt.xlabel('Observation')
plt.ylabel('Prédiction')
plt.title ("Régresseur ")
st.pyplot(fig)

# Régression linéaire 
st.text ("Colonnes essayées pour la régression linéaire :")
st.text (data_vars)
st.text ("")
st.text ("Score sur jeu d'entraînement : " + str(reg.score (X_train, y_train)))
st.text ("Score sur jeu de tests       : " + str(reg.score (X_test,  y_test )))
st.text ("")
