import streamlit         as st
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import seaborn           as sns
import plotly.express    as px
from sklearn.model_selection   import   train_test_split
from sklearn.model_selection   import   cross_val_score
from sklearn.model_selection   import   GridSearchCV
from sklearn.metrics           import   mean_squared_error
from sklearn.metrics           import   r2_score
from sklearn.ensemble          import   RandomForestRegressor
from sklearn.ensemble          import   ExtraTreesRegressor
import xgboost as xgb
import shap 
import lime
import lime.lime_tabular
#import tensorflow       as tf
#from tensorflow                import keras
#from tensorflow.keras.layers   import Input, Dense
#from tensorflow.keras.models   import Model

#
#Cette feuille necessite d'installer les modules suivants :
#TODO : Mettre une description en qq mots pour chacun
#  - shap :
#  - lime :
#
#

from CO2_fcts                  import libelles_vars

## from   sklearn.linear_model      import LinearRegression

## %matplotlib inline 
## !pip install xgboost==2.0.3

################################################################################
#  Quelques paramètres et réglages                                             #
################################################################################
vars_quant = ['Masse',
              'Empattement',
              'Larg_essieu_dir',
              'Cylindree',
              'Puiss_moteur',
              'Emis_CO2_spe']

################################################################################
#  Bonjour !                                                                   #
################################################################################

st.title("Notre projet CO2")
st.write("Bonjour, Nous présentons notre modélisation.")

################################################################################
#  Lecture et affichage des données                                            #
################################################################################

st.write("Bonjour, Nous présentons notre modélisation, voici les \
          premières lignes de notre jeu de données")
df = pd.read_csv('../data/Emissions_CO2_FR.csv',  low_memory=False)
df = df.sample(5000)
df = df[df.Emetteur_CO2 == 1]
st.dataframe(df.head(10))
st.write ("Nombre d'observations : ", df.shape[0])

################################################################################
#  Affichage des corrélations entre les variables quantitatives                #
################################################################################

st.header("Corrélation entre les variables quantitatives")
st.write("Une heatmap apporte des informations utilies pour comprendre \
les liens entre les variables quantitatives et nous aider à choisir les modèles.")

# TODO : mettre les libellés des variables qui sont dans le dictionnaire : libelles_vars
df_matcorr = df[vars_quant]
fig = plt.figure(figsize=(5, 5))
plot = sns.heatmap(df_matcorr .corr(), annot=True, cmap="RdBu_r")#, center=0);
plt.title("Corrélation entre variables quantit. modèles émettteurs CO2")
st.pyplot(fig)

################################################################################
# Modélisations                                                                #
################################################################################

st.header("Modélisations")

################################################################################
# Modélisation : RandomForestRegressor                                         #
################################################################################

st.subheader("Modélisation avec RandomForestRegressor")
df_model_rf = df[vars_quant]

X = df_model_rf.drop(['Emis_CO2_spe'], axis=1)
y = df_model_rf['Emis_CO2_spe']
st.write("Nous utilisons les variables explicatives suivantes : ")
st.write(str(X.columns))
st.text("Notre variable cible est la suivante : ")
# TODO corriger la ligne suivante pour afficher le nom de la variable cible :
#st.text(str(y.columns))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#----- Standardisation -----
st.write("Nous normalisons les données avec StandardScaler()")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#----- Entraînement ------
regressor = RandomForestRegressor(n_estimators = 10, 
                                  random_state = 0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

#----- Affichage des performances -----
# TODO : faire un tableau lisible pour mettre ces résultats :
#Affichage du R2
st.text(f"R2 pour les données de test : {regressor.score(X_test, y_test)}")
st.text(f"R2 pour les données d'entraînement : {regressor.score(X_train, y_train)}")

#Validation croisée
st.text(f"Score de validation croisée (moyenne) : {cross_val_score(regressor, X_train, y_train, cv=5).mean()}")

#RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
st.text(f"RMSE : {rmse}")

#----- Affichage d'un nuage de points ------
st.write ("Voyons sur un graphique à points :")
fig = plt.figure(figsize=(5, 5))
plt.scatter(y_test, y_pred, s=2)
plt.xlabel('Observation')
plt.ylabel('Prédiction')
plt.title("RandomForestRegressor, comparaison entre observation et prédiction")
st.pyplot(fig)
st.write("Malgré de bons scores les points ne sont pas encore assez alignés")

################################################################################
# Un petit coup de SHAP                                                        #
################################################################################

st.write ("Utilisons SHAP pour y voir plus clair.")
explainer = shap.TreeExplainer(regressor)
shap_values = explainer.shap_values(X_test)
fig = plt.figure(figsize=(5, 5))
shap.summary_plot(shap_values, X_test, plot_type='bar')
st.pyplot(fig)
st.write("Ah oui les émmisions de CO2 dépendent surtout de la cylindrée et de\
         Puissance qui sont très corrélées.")

################################################################################
# Utilisons ExtraTreesRegressor suggéré par lazypredict/lazyRegressor          #
################################################################################
st.subheader("ExtraTreesRegressor")
#----- Création et entraînement du modèle -----
st.write ("Création et entraînement du modèle")
etr = ExtraTreesRegressor()
etr.fit(X_train,y_train)
st.write ("Prédiction")
y_pred = etr.predict(X_test)

#----- Évaluations du modèle -----
# R2
st.text(f"R2 pour les données de test : {etr.score(X_test, y_test)}")
st.text(f"R2 pour les données d'entraînement : {etr.score(X_train, y_train)}")

# Validation croisée
st.text(f"Score de validation croisée (moyenne) : {cross_val_score(regressor, X_train, y_train, cv=5).mean()}")

# RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
st.write(f"RMSE : {rmse}")

#----- Affichage du nuage de points -----
# TODO à enlever st.pyplot(fig)
fig = plt.figure(figsize=(5, 5))
plt.scatter(y_test, y_pred, s=2)
plt.xlabel('Observation')
plt.ylabel('Prédiction')
plt.title("Comparaison entre valeurs prédites et observées sur jeu de tests")
st.pyplot(fig)

#----- Affichage de l'histogramme des résidus -----
fig = plt.figure(figsize=(5, 4))
residus = y_test - y_pred
plt.hist(residus, bins=20)
plt.xlabel('Résidus')
plt.ylabel('Fréquence')
plt.title('Histogramme des Résidus');
st.pyplot(fig)

#----- Affichage du nuage de points des résidus -----
fig = plt.figure(figsize=(5, 5))
plt.scatter(y_test, residus, color='#980a10', s=15)
plt.title('Dispersion des Résidus')
plt.xlabel('Valeurs Réelles')
plt.ylabel('Résidus');
st.pyplot(fig)

################################################################################
# XgBoost                                                                      #
################################################################################

st.subheader("XgBoost")
st.write("Nous allons utiliser XgBoost, ")

#----- Création de xxxxxxxxxxxxxxxxx et de ses paramètres ----- 
train = xgb.DMatrix(X_train, label=y_train)
test = xgb.DMatrix(X_test, label=y_test)
params = {'learning_rate': 0.1,'objective': 'reg:linear'}
num_boost_round = 100
evals = [(train, 'train'), (test, 'eval')]

st.write ("Entraînement de XgBoost")
#----- Entraînement -----
xgb2 = xgb.train(params=params,
                 dtrain=train,
                 num_boost_round=num_boost_round,
                 evals=evals)

################################################################################
# GridSearchCV et XGBRegressor                                                 #
################################################################################

param_grid = {
    'learning_rate': [0.3, 0.4],
    'max_depth': [8, 9],
    'n_estimators': [350, 400]
}

xgb_model = xgb.XGBRegressor(booster='gbtree', objective='reg:linear') 

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)
st.write ("Meilleurs paramètres:")
st.write(str(grid_search.best_params_))

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

#R2
st.text ("Score sur jeu de test        : " + str((best_model.score(X_test, y_test))))
st.text ("Score sur jeu d'entrainement : " + str(best_model.score(X_train, y_train)))

#validation croisée 
st.write ("Validation croisée : " )
st.text(str(cross_val_score(best_model,X_train,y_train).mean()))

#RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
st.text(f"RMSE : {rmse}")
"""
################################################################################
# Deep Learning                                                                #
################################################################################

st.header("Deep learning")
X = df.drop(['Emis_CO2_spe'], axis=1)
y = df['Emis_CO2_spe']

X = X.apply(lambda x : (x-x.min())/(x.max() - x.min()))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

st.text(f"Forme de X_train : {np.shape(X_train)}")
st.text(f"Forme de y_train : {np.shape(y_train)}")

inputs = Input(shape = (5))
dense = Dense(units = 1)
outputs = dense(inputs)
linear_model = Model(inputs = inputs, outputs = outputs)
st.text(str(linear_model.summary()))

#----- Compilation -----
linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.01),
    loss='mean_squared_error')

y_pred = linear_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
st.text(f"Le coefficient de détermination R^2 est : {r2}")

rmse = np.sqrt(450)
st.text(str(rmse))

"""
