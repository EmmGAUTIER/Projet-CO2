"""
st_main.py : Feuille streamlit principale du projet CO2

Cette feuille est la feuille principale.
Elle réalise les actions suivantes :
   - Affiche du menu à gauche ;
   - Import des librairies ;
"""

import streamlit         as st
import pandas            as pd
import numpy             as np

###########################################################################
# Import des librairies
###########################################################################

#----- Autres feuilles et fichier auxiliaires -----
from st_aux              import vars_quant        # Liste des var. quant.
from st_aux              import libelles_vars     # libellés pour graphes
from st_jeudonnees       import st_jeudonnees
from st_visualisation    import st_visualisation
from st_presentation     import st_presentation
from st_regression       import st_regression
from st_classification   import st_classification

###########################################################################
# Chargement de données                                                   #
###########################################################################

# Chargement du jeu de données préalablement préparé
df = pd.read_csv('Emissions_CO2_FR.csv',  low_memory=False)

# et sélection des seuls émetteurs de CO2.
df = df[df.Emetteur_CO2 == 1]

# Présentation du menu principal avec un sidebar
st.sidebar.title("Sommaire")
pages = ["Présentation du projet",
         "Le jeu de données",
         "Visualisations",
         "Régressions",
         "Classifications"]#,
         #"Conclusion et perspective"]

page = st.sidebar.radio("", pages)

st.sidebar.page_link("https://www.linkedin.com/in/emmanuel-gautier-098460105", label = ":blue[Emmanuel Gautier]")
st.sidebar.page_link("https://github.com/EmmGAUTIER/Projet-CO2", label = ":blue[Sources (github)]")

if page == "Présentation du projet" :
    st_presentation(df)
    pass

if page == "Le jeu de données" :
    st_jeudonnees(df)
    pass

if page == "Visualisations" :
    st_visualisation(df)
    pass

if page == "Régressions" :
    st_regression(df)
    pass

if page == "Classifications" :
    st_classification(df)
    pass
