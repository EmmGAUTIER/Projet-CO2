import streamlit         as st
import matplotlib.pyplot as plt
import seaborn           as sns
import st_aux            as aux

def st_visualisation(df) :

    st.title("Visualisation des données")
    st.header("quelques statistiques exploratoires")

    rep_norme     = df.Nap_norme.value_counts(normalize = True)
    rep_marque    = df.Marque.value_counts(normalize = True)

    #########################################################################
    #  Répartition des types de carburant                                   #
    #########################################################################

    st.header ("Types de carburant")
    st.write("Les deux types de carburant principaux sont l'essence et le Diesel. "
             "Les autres sont en proportions très faibles "
             "Le Diesel est prépondérant en France.")

    rep_carburant = df.Type_carburant.value_counts(normalize = True)
    fig = plt.figure (figsize =  (10,   6))
    #plt.subplot(1,2,1)
    plt.bar(rep_carburant.index, rep_carburant.values*100)
    plt.ylabel("Proportion (%)")
    plt.xticks(rep_carburant.index, rotation = 45)
    plt.title("Répartition des types de carburant");
    #plt.legend(loc = "center")
    st.pyplot(fig)

    #########################################################################
    #  Normes d'approbation de types                                        #
    #########################################################################
    
    st.header("Norme d'approbation de type")
    st.write("Le numéro d'approbation de type contient la référence "
             "de la norme appliquée et son année de mise an application "
             "Les deux principales sont celles de 2001 et 2007.")

    fig = plt.figure ()
    #plt.subplot(1,2,2)
    plt.bar(rep_norme.index, rep_norme.values*100)
    plt.ylabel("Proportion (%)")
    plt.xticks(rep_norme.index, rotation = 45)
    plt.title("Répartition des normes d'approbation")
    st.pyplot(fig)

    #########################################################################
    #  Marques représentées                                                 #
    #########################################################################

    st.header("Marques représentées")
    st.write("La répartition des marques est plutôt homogène. "
             "Les marques les plus représentées sont les marques allemendes.")

    fig = plt.figure (figsize = (6,4))
    rep_marque = rep_marque.head(20)
    plt.bar(rep_marque.index, rep_marque.values*100)
    plt.ylabel("Proportion (%)")
    plt.xticks(rep_marque.index, rotation = 90)
    plt.title("Répartition des marques");
    st.pyplot(fig)

    #########################################################################
    #  Répartitions des valeurs des variables quantitatives                 #
    #########################################################################

    st.header("Répartitions des variables quantitatives")

    fig = plt.figure(figsize = (10,10))
    plt.subplot(3,2,1)
    ax = sns.violinplot(data= df, y="Emis_CO2_spe")
    plt.ylabel("g/km")
    plt.title("Répartition des émission de CO2")

    plt.subplot(3,2,2)
    ax = sns.violinplot(data= df, y="Masse")
    plt.ylabel("Masse (kg)")
    plt.title("Répartition des masses")

    plt.subplot(3,2,3)
    ax = sns.violinplot(data= df, y="Puiss_moteur")
    plt.ylabel("Puissance (kW)")
    plt.title("Répartion des puissances")

    plt.subplot(3,2,4)
    ax = sns.violinplot(data= df, y="Cylindree")
    plt.ylabel("Cylindrée (cm3)")
    plt.title("Répartion des cylindrées")

    plt.subplot(3,2,5)
    ax = sns.violinplot(data= df, y="Empattement")
    plt.ylabel("Empattament (mm)")
    plt.title ("Répartition des empattements")

    plt.subplot(3,2,6)
    ax = sns.violinplot(data= df, y="Larg_essieu_dir")
    plt.ylabel("Voie (mm)")
    plt.title ("Répartion des largeurs d'essieux (voie)")

    st.pyplot(fig)

    #########################################################################
    #  Affichage de la matrice de corrélation des variables quatitatives.   #
    #########################################################################

    st.header("Corrélations entre variables quantitatives")

    fig = plt.figure(figsize=(4, 4))
    mat_corr = df[["Emis_CO2_spe", "Masse", "Puiss_moteur",
                   "Cylindree", "Empattement", "Larg_essieu_dir"]].corr()
    sns.heatmap (mat_corr,
                 annot = True,
                 cmap = sns.color_palette("viridis", as_cmap=True),
                 xticklabels = ["Émission CO2", "Masse", "Puiss_moteur",
                                "Cylindrée", "Empattement", "Larg. essieu_dir"],
                 yticklabels = ["Émission CO2", "Masse", "Puiss_moteur",
                                "Cylindrée", "Empattement", "Larg. essieu_dir"])
    st.pyplot(fig)

    #########################################################################
    # Affichage d'un nuage de points émission de CO2 ct un variable quant.  #
    #########################################################################

    st.header ("Lien entre la cible et chacune des variables quantitatives explicatives")
    
    var_quant_choisie = st.selectbox("Variable explicative quantitative",
                                      aux.vars_expl_quant)
    fig = plt.figure(figsize=(4, 4))
    sns.scatterplot(data = df,
                    x   = var_quant_choisie,
                    y   = "Emis_CO2_spe",
                    hue = "Type_carburant",
                    s   = 2)
    plt.legend(fontsize = "xx-small")
    plt.ylabel("Émission de CO2 (g/km)")
    plt.xlabel(aux.libelles_vars[var_quant_choisie])
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.title("Émission de CO2")
    st.pyplot(fig)
