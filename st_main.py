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

import time
import json

###########################################################################
# Import des librairies
###########################################################################

#----- Autres feuilles et fichier auxiliaires -----
#from st_aux                    import 
#from st_view                   import st_view
from st_presentation            import st_presentation
from st_regressions             import st_regressions
#from st_classifications        import st_classifications

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
         "Données et méthodologie",
         "Visualisations",
         "Régressions",
         "Classifications"]#,
         #"Conclusion et perspective"]

page = st.sidebar.radio("", pages)

if page == "Présentation du projet" :
    st_presentation()
    pass

if page == "Visualisations" :
    #st_view()
    pass

if page == "Régressions" :
    st_regressions(df)
    pass

if page == "Classifications" :
    #st_classifications()
    pass
