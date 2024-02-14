import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from CO2_fcts import split_ap_type, dict_renommages_cols

st.title("Projet CO2")
st.header("Préparation des données")
st.write("Préparation du jeu de données pour notre modélisation.")

df = pd.read_csv('CO2_passenger_cars_v8.csv', sep = '\t', low_memory=False)
df.info()
# Changement des noms des colonnes
st.write(str(dict_renommages_cols))
st.write(str(df.info()))
df = df.rename(columns = dict_renommages_cols)

# Restriction au données françaises
df = df[df.Etat_mem == 'FR']

vars_ecartees = ['Id',
                 'Contis_fab',
                 'Nom_fab',
                 'Nom_fab_enregMS',
                 'Make',
                 'Techno_innov',
                 'Reduc_emis_techno_innov',
                 'Etat_mem']

marques_ecartees = ["OUT OF SCOPE", "AA-IVA", "AA-NSS"]

st.header ("Suppression d'observations")

#----- Suppression des variables -----
st.subheader ("Suppression de variables : ")
st.text(vars_ecartees)
df = df.drop(vars_ecartees, axis = 1)
st.text(str(df.info()))

#----- Suppression des modèles de marques à écarter ----- 

st.text ("Nombre d'observations avant suppression :" + str(df.shape[0]))
st.subheader ("Suppression des marques à écarter : ")
st.text(df[df.Marque.isin(marques_ecartees)].Marque.value_counts())
df = df[~df.Marque.isin(marques_ecartees)]
st.text(df.Marque.isin(marques_ecartees).value_counts())
st.text ("Nombre d'observations après suppression :" + str(df.shape[0]))

#----- Suppression des modèles dont la puissance est inconnue -----

st.subheader("Suppression des modèles dont la puissance est nulle")
st.text ("Nombre d'observations avant suppression :" + str(df.shape[0]))
df = df[~df.Puiss_moteur.isnull()]
st.text ("Nombre d'observations après suppression :" + str(df.shape[0]))

#----- Suppression des doublons -----

st.subheader("Suppression des doublons")
st.text ("Nombre d'observations avant suppression :" + str(df.shape[0]))
df = df.drop_duplicates()
st.text ("Nombre d'observations après suppression :" + str(df.shape[0]))

st.text("Attention, il reste " + str(df.drop("Masse", axis = 1).duplicated().sum()) + " observations qui ne diffèrent que par a masse.")
st.text("Autrement dit, des modèles identiques avec des masses différentes")

