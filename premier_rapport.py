#!/usr/bin/python3

import sys
import numpy as np
import pandas as pd


def nb_to_str(fmt, nb):
    return fmt.format(nb)


def premier_rapport(df, nb_multi_max=10, gd_libelles_col=None):
    nb_lig = df.shape[0]
    nb_col = df.shape[1]
    nb_cell = nb_lig * nb_col
    nb_dec = int(np.log10(nb_lig)) + 1
    fmt_int = "{:" + str(nb_dec) + "d}"
    fmt_pc = "{:" + str(nb_dec + 2) + "." + str(nb_dec - 2) + "f}"

    print("Nombre de lignes   : ", nb_lig)
    print("format nombre      : ", fmt_int)
    print("format pourcentage : ", fmt_pc)
    # exit(1)

    nb_col_nulls = {}
    # nb_null = df.isnull().sum()
    types_col = {}
    print()
    # print ("Nombre de lignes   : "+fmt_int.format(nb_lig))
    print("Nombre de lignes   : ", nb_to_str(fmt_int, nb_lig))
    print("Nombre de colonnes : ", nb_to_str(fmt_int, nb_col))
    print("Nombre de cellules : ", nb_to_str(fmt_int, nb_cell))
    # print ("Nombre de nulls    : {:6d}".format(nb_null))
    print()
    print("Colonne                    type            Nulls")
    print("                                      Nombre    proportion")
    for col in df.columns:
        nb_col_nulls[col] = df[col].isnull().sum()
        types_col[col] = df.dtypes[col]
        fmt = "{:25s}  {:8s}   " + fmt_int + "     {:7.3f}"
        print(fmt.format(col, str(types_col[col]), nb_col_nulls[col], (100. * nb_col_nulls[col]) / nb_lig))

    icol = 0
    for col in df.columns:
        nb_null = df[col].isnull().sum()
        nb_null_FR = df[col].isnull().sum()
        typecol = str(df.dtypes.iloc[icol])
        print(
            "======================================================================================================")

        print()
        print(">>>> '" + col + "' <<<<   : ")  # , gd_libelles_col[col] if gd_libelles_col != None else '')
        print()
        print("Type           : ", df.dtypes[col])
        fmt = "Valeurs nulles : " + fmt_int + "   soit " + fmt_pc + " %"
        print(fmt.format(nb_null, 100. * nb_null / nb_lig))
        print()

        if typecol == 'float64':
            print(df[col].describe())

        if typecol == 'int64':
            print("Nombre de valeurs uniques  : ", df.shape[0] - df[col].duplicated().sum())
            print("Nombre de valeurs répétées : ", df[col].duplicated().sum())
        if typecol == 'object':
            nb_cumul = nb_null
            nb_diff = len(df[col].value_counts().unique())
            vc = df[col].value_counts()
            # fmt = "Nombre de modalités (valeurs différentes) : " + fmt_int
            # print (fmt.format(nb_diff))
            print("->")
            lg_max = 6
            for i in range(nb_multi_max if nb_diff > nb_multi_max else nb_diff):
                lg_max = len(vc.index[i]) if len(vc.index[i]) > lg_max else lg_max

            nb_autres = nb_diff  # - nb_null
            fmt = "{:" + str(lg_max) + "s} : " + fmt_int + "  " + fmt_pc + "%"

            for i in range(nb_multi_max if nb_diff > nb_multi_max else nb_diff):
                nb_occ = vc.iloc[i]
                nb_cumul += nb_occ
                print(fmt.format(vc.index[i], nb_occ, 100. * nb_occ / nb_lig))
                nb_autres -= 1
            nb_autres = nb_lig - (nb_cumul + nb_null)
            print(fmt.format("Autres", nb_autres, 100. * nb_autres / nb_lig))
            print(fmt.format("Nuls ", nb_null, 100. * nb_null / nb_lig))
        icol += 1
        print()
        print()


if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1], sep=sys.argv[2], low_memory=False, encoding='latin-1')
    premier_rapport(df, nb_multi_max=50, gd_libelles_col=None)

    sys.exit()
