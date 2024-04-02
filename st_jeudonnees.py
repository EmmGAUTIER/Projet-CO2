import streamlit as st


def st_jeudonnees (df = None) :
    
    st.title("Le jeu de données")

    st.write("Le jeu de données provient de l’Union Européenne. Il est téléchargeable à l’adresse suivante: "
              "https://www.eea.europa.eu/en/datahub/datahubitem-view/fa8b1229-3db6-495d-b18e-9c9b3267c02b "
              "Nous téléchargeons le fichier au format CSV. Il contient une ligne d’entête et  442475 lignes d’observations.")

    st.write("Le jeu de données contient les modèles de véhicules "
             "de transport de passagers 'légers' enregistrés dans chaque pays "
             "de l'Union Européenne. "
             "Nous recherchons d’abord les données à écarter "
             "avec des statistiques exploratoires, "
             "et des affichages de quelques modèles. "
             "Nous vérifions alors que les modèles sont "
             "des véhicules de transport de passagers avec au plus 9 places. "
             "Nous recherchons les camions, motos, bateaux,... "
             "et vérifions les données. "
             "Nous constatons que le jeu de données comporte "
             "un nombre important de valeurs non renseignées, "
             "des valeurs aberrantes et des doublons. ")

    st.write("Le jeu de données des modèles enregistrés en France comporte "
             "une proportion bien moindre de ces données. "
             "Nous n'utiliserons que les données enregistrées par la France "
             "après avoir écarté quelques modèles et des doublons. "
             "Le jeu de données retenu contient 30427 modèles.")
