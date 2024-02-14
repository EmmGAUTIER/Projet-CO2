def split_ap_type(at):
    """
    split_ap_type(at)

    return : list of fields separated by '*' in the string at.
    the returned list is for elements long.
    """
#    if type(at) == 'float' :
#        return ['', '', '', '']
    l = str(at).split ('*', maxsplit = 4)
    for i in range (len(l), 5):
        l.append("")
    l[0] = l[0].lower()
    return l

dict_renommages_cols = {
              'id'           : 'Id',
              'MS'           : 'Etat_mem',
              'MP'           : 'Contis_fab',
              'Mh'           : 'Marque',
              'MAN'          : 'Nom_fab',
              'MMS'          : 'Nom_fab_enregMS',
              'TAN'          : 'Numero_approb_type',
              'T'            : 'Type',
              'Va'           : 'Variante',
              'Ve'           : 'Version',
              'Mk'           : 'Make',
              'Cn'           : 'Nom_commercial',
              'Ct'           : 'Cat_vehicule',
              'r'            : 'Total_nvle_inscr',
              'e (g/km)'     : 'Emis_CO2_spe',
              'm (kg)'       : 'Masse',              
              'w (mm)'       : 'Empattement',
              'at1 (mm)'     : 'Larg_essieu_dir',
              'at2 (mm)'     : 'Larg_essieu_aut',
              'Ft'           : 'Type_carburant',
              'Fm'           : 'Mode_carburant',
              'ec (cm3)'     : 'Cylindree',
              'ep (KW)'      : 'Puiss_moteur',
              'z (Wh/km)'    : 'Conso_elect',             
              'IT'           : 'Techno_innov',
              'Er (g/km)'    : 'Reduc_emis_techno_innov'
            } 

dict_repl_carb = {'DIESEL'          : 'Diesel',
                  'diesel'          : 'Diesel',
                  'PETROL'          : 'Essence',
                  'PETROL '         : 'Essence',
                  'Petrol'          : 'Essence',
                  'petrol'          : 'Essence',
                  'PETROL-ELECTRIC' : 'Essence-Electrique',
                  'Petrol-Electric' : 'Essence-Electrique',
                  'Petrol-electric' : 'Essence-Electrique',
                  'petrol-electric' : 'Essence-Electrique',
                  'diesel-Electric' : 'Diesel-Electrique',
                  'Diesel-electric' : 'Diesel-Electrique',
                  'Diesel-Electric' : 'Diesel-Electrique',
                  'diesel-electric' : 'Diesel-Electrique',
                  'DIESEL-ELECTRIC' : 'Diesel-Electrique',
                  'hydrogen'        : 'Hydrogene',
                  'Hydrogen'        : 'Hydrogene',
                  'NG-BIOMETHANE'   : 'NG-Biomethane',
                  'NG-biomethane'   : 'NG-Biomethane',
                  'ELECTRIC'        : 'Electrique',
                  'Electric'        : 'Electrique',
                  'electric'        : 'Electrique',
                  ' '               : 'Inconnu'}

# Ce dictionnaire contient les libellés des colonnes à afficher sur les graphes, tableaux, etc.
# Ces libellés ne sont pas destinés à identifier les colonnes.
libelles_vars = {
'ID'                       :  'Identifiant',
'Etat_mem'                 :  'État membre',
'Contis_fab'               :  '',
'Marque'                   :  'Marque',
'Nom_fab'                  :  '',
'Nom_fab_enregMS'          :  '',
'Numéro_approb_type'       :  'N° d\'approbation de type',
'Type'                     :  '',
'Variante'                 :  'Variante',
'Version'                  :  'Version',
'Make'                     :  'Marque',
'Nom_comm'                 :  'Nom Commercial',
'Cat_vehicule'             :  'Catégorie de véhicule',
'Total_nvle_inscr'         :  'Total nouvelle inscription',
'Emission_CO2'             :  'Émission de CO2 (g/km)',
'Masse'                    :  'Masse',
'Empattement'              :  'Empattement (mm)',
'Larg_essieu_dir'          :  'Largeur essieu directeur (mm)',
'Larg_essieu_aut'          :  'Largeur autre essieu (mm)',
'Type_carburant'           :  'Carburant',
'Mode_carburant'           :  'Mode de carburant',
'Cylindree'                :  'Cylindrée (cm3)',
'Puiss_moteur'             :  'Puissance (KW)',
'Conso_energie_elect'      :  'Consommation électrique (Wh/km)',
'Techno_innov'             :  'Technologie innovante',
'Reduc_emis_techno_innov'  :  'Réduc. d\'émission par techno. innov.'
}
