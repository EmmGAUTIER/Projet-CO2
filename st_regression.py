"""
    st_ regressions.py : streamlit, calcul et affichage de régressions.

Cette feuille permet d'essayer des modèles de régression.

Elle propose et affiche les choix suivants :
 linéaire" :
            reg = linr
        elif reg_choisie == "Extra Trees Regressor" :
            reg = extr
        elif reg_choisie == "Random Forest Regressor" :
            reg = rmfr
        else : # "XG Boost"

  - Type de régressions : linéaire, Extra Trees , Forêt aléatoire et XGBoost ;
  - Type de carburant : tous, diesel eu essence ;
  - La norme : "2007/46" et "2001/116" ;

Elle calcule :
    - la régression demandée avec les données choisies ;
    - des métriques d'évaluation.

Elle affiche :
    - Les choix proposés
    - un nuage de points illustrant les dispertions de prédictions ;
    - La prédiction pour un véhicule avec des métriques.

Les types de carburant et les normes proposés sont restreintes 
aux modalités majoritaires. Les autres ne permettent pas d'avoir
suffisement d'observation pour faire des régressions.
"""

###############################################################################
#  Imports de librairies                                                      #
###############################################################################

import sys

#----- Affichage et interaction -----
import streamlit         as st
import matplotlib.pyplot as plt

#----- librairie à usage général -----
import pandas as pd

#----- préparation de données ----- 
from sklearn.model_selection   import train_test_split

#----- Modèles de régression et métriques -----
from sklearn.linear_model      import LinearRegression
from sklearn.ensemble          import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics           import r2_score
from xgboost                   import XGBRegressor

#----- Auxiliaires (libellés,...) -----
from st_aux                    import libelles_vars

def st_regression(df) :

    st.title("Régressions linéaires")
    st.write ("Les régressions sont adaptées "
              "à la prédiction de variables quantitatives. "
              "Les régressions linéaires sont, de plus, "
              " simples et rapides à calculer. "
              "Les régressions mettent en évidence les effets de l'augmentation des "
              "valeurs des variables explicatives sur les émissions de CO2.")
    
    st.write ("La régression linéaire montre ici ses faiblesses, "
              "les autres régressions prédisent nettement mieux. ")
    
    st.write("Les variables explicatives suivantes sont utilisées : "
             "masse, empattement, largeur d'essieu, cylindrée "
             "et puissance du moteur."
             "Le type de carburant et la norme d'approbation sont "
             "des variable qualitatives et ne peuvent être utilisé "
             "par les régressions; le type de carburant et, "
             "dans une moindre mesure, la norme d'approbation "
             "ont cepandant un impact sur l'émission de CO2. ")

    st.write("Les modèles de régression sont alors entraînés "
             "avec tous les modèles de véhicules ou sur une restriction "
             "à un type de carburant ou une norme. "
             "La restriction à un type de carburant améliore alors les scores.")

    st.write("Les scores montrent encore un surapprentissage trop important.")

    ###############################################################################
    #  Initialisations                                                            #
    ###############################################################################

    used_vars = ['Masse',
                 'Empattement',
                 'Larg_essieu_dir',
                 'Cylindree',
                 'Puiss_moteur',
                 'Type_carburant',
                 'Nap_norme',
                 'Emis_CO2_spe']

    dfs = df[used_vars]
    dfs = dfs.drop_duplicates()

    data_vars = ['Masse',
                 'Empattement',
                 'Larg_essieu_dir',
                 'Cylindree',
                 'Puiss_moteur']
    
    target_var = 'Emis_CO2_spe'

    # Création des régresseurs
    linr = LinearRegression()
    extr = ExtraTreesRegressor()
    rmfr = RandomForestRegressor()
    xgbr = XGBRegressor()

    # Types de carburant proposés
    # Les carburant Diesel et Essence sont largement prépondérants
    # Nous proposons tous les types ou ces deux types de carburant.
    chx_carb = ["Tous", "Essence", "Diesel"]

    # Choix de la norme.
    # Il y a deux modalités prépondérantes et quelques autres rares.
    # Nous proposons toutes ou chacune de deux plus fréquentes.
    chx_norme = ["Toutes", "2007/46", "2001/116"]

    chx_reg = ["Régression linéaire", "Extra Trees Regressor", "Random Forest Regressor", "XG Boost"]

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

    ###############################################################################
    #  Sélection des données, calculs et affichage                                #
    ###############################################################################

    # TODO : utiliser un décorateur pour éviter de recalculer
    if True :

        # Sélection des observation : type de carburant et norme
        if carb_choisi != "Tous" :
            dfs = dfs[dfs.Type_carburant == carb_choisi]
        if norme_choisie != "Toutes" :
            dfs = dfs[dfs.Nap_norme == norme_choisie]

        # Séparation des variables explicatives et de la variable cible
        data   = dfs[data_vars]
        target = dfs[target_var]

        # Création d'un jeu d'entraînement et d'un jeu de tests
        X_train, X_test, y_train, y_test = train_test_split (data, target, test_size = 0.2, random_state = 421)

        # Calcul de la régression 
        if reg_choisie == "Régression linéaire" :
            reg = linr
        elif reg_choisie == "Extra Trees Regressor" :
            reg = extr
        elif reg_choisie == "Random Forest Regressor" :
            reg = rmfr
        else : # "XG Boost"
            reg = xgbr

        print ("Entraînement", file = sys.stderr)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        #prec_carb_choisi   = carb_choisi
        #prec_norme_choisie = norme_choisie
        #prec_chx_reg   = reg_choisie

    print ("Affichage du nuage de points")
    # Affichage d'un nuage de points permettant de comparer
    # les valeurs prédites et les valeurs vraies
    fig = plt.figure(figsize=(2, 2))
    hue = 'Type_carburant' if carb_choisi == "Tous" else None
    plt.scatter(y_test, y_pred, s = 2)
    plt.xlim(0, 400)
    plt.ylim(0, 400)
    plt.xlabel('Observation')
    plt.ylabel('Prédiction')
    #plt.title ("Régresseur ")
    st.pyplot(fig)

    st.text ("Score sur jeu d'entraînement : " + str(reg.score (X_train, y_train)))
    st.text ("Score sur jeu de tests       : " + str(reg.score (X_test,  y_test )))

    st.header("Application sur un véhicule")
    st.text("Choisissez les caratéristiques d'un véhicule et "
            "voyez l'effet sur l'émission de CO2")

    col1, col2 = st.columns(2)

    with col1 :
        masse_choisie       = st.slider(libelles_vars["Masse"],
                                        min_value = int(df['Masse'].min()),
                                        max_value = int(df['Masse'].max()),
                                        value     = int(df['Masse'].median()),
                                        step=10)

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

    with col2:
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


    data_vars = ['Masse',
                 'Empattement',
                 'Larg_essieu_dir',
                 'Cylindree',
                 'Puiss_moteur']

    X_essaye = pd.DataFrame({
        'Masse'                        : [masse_choisie],
        'Empattement'                  : [empattement_choisi],
        'Larg_essieu_dir'              : [larg_essieu_choisie],
        'Cylindree'                    : [cylindree_choisie],
        'Puiss_moteur'                 : [puissance_choisie],
    })

    y_pred = reg.predict(X_essaye)

    # Affichage du taux de CO2 prédit
    st.write(f"Émission de CO2 prédite : {y_pred[0]:.1f} g/km")
 