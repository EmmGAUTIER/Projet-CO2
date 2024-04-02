
import streamlit         as st

def st_presentation (df = None) :
    
     st.title("Émissions de CO2 par les véhicules")
     st.header("Modélisation d'émissions de CO2 de véhicules légers,")
     st.header("influence de caractéristiques techniques et réglementaires")

     st.write("L’impact des émissions de CO2 des véhicules est de plus en plus "
              "préoccupant pour les citoyens de l’Union Européenne et leurs "
              "dirigeants. La demande de véhicules moins polluants est croissante."
              "La volonté de réduire ces émissions a conduit les gouvernements et "
              "l’UE à adopter des règlement des plus en plus exigeant en matière "
              "de réduction des émissions polluantes. L’UE prévoit d’interdire "
              " la mise en circulation des véhicules émetteurs de CO2 en 2035."
              "L’identification et l’impact de caractéristiques techniques ou "
              "réglementaires permettent de mieux cibler les choix qui permettent "
              "réduire les émissions des nouveaux véhicules. Dans cette optique, "
              "notre projet vise à modéliser les émissions de CO2 en fonction de "
              "caractéristiques techniques et réglementaires.")

     st.write("Ce projet présente notre travail depuis la collecte des données "
              "jusqu'à la modélisation. Les modèles entraînés sont "
              "des régresseurs et des classifieurs.")
 
     st.write("Ce projet permet, de plus, de mettre en évidence "
              "l'impact de choix par les décideurs, utilisateurs, concepteurs "
              "ou régulateurs de caractéristiques techniques  "
              "de véhicules légers sur leurs émissions de CO2. "
              "Il a été choisi car nous sommes sensibles     "
              "à la protection de l’environnement.")
              