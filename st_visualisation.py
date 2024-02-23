import streamlit         as st
import matplotlib.pyplot as plt
import seaborn           as sns
import st_aux            as aux


def st_visualisation(df) :


    st.header("Visualisation des données")
    #st.text(df.corr())

    # TODO : À corriger provoque une erreur sur stremlit.io
    #fig = plt.figure(figsize=(4, 4))
    #sns.heatmap(df[aux.vars_quant].corr(), annot = True)
    #plt.title ("Corrélation entre variables quantatives")
    #st.pyplot(fig)

    var_quant_choisie = st.selectbox("Variable explicative quantitative", aux.vars_quant)

    fig = plt.figure(figsize=(4, 4))
    sns.scatterplot(data = df, x=var_quant_choisie, y="Emis_CO2_spe", hue = "Type_carburant", s=2)
    plt.title("Titre")
    st.pyplot(fig)

