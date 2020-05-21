# Utilisé pour des tests de comparaison manuels

import os
import pandas as pd
import numpy as np


file_path = os.path.dirname(os.path.realpath(__file__))
proj_path = os.path.abspath(os.path.join(file_path,"../.."))
testdata_path = os.path.join(proj_path,"test/data")
gen_path = os.path.join(proj_path,"../../gatsby/trends/generated")
datagen_path = os.path.join(gen_path,"data")

# Utilitaire pour des fichier de sortie générés manuellement
ref_data = os.path.join(testdata_path,"dep_out_2020-05-20.csv")
ref = pd.read_csv(ref_data)

verif_data = os.path.join(datagen_path,"departements.csv")
verif = pd.read_csv(verif_data)


errCols = []
def checkString(col):
    ref[col].fillna("None", inplace=True)
    verif[col].fillna("None", inplace=True)
    test["%sErr"%col] = (ref[col] != verif[col])
    errCols.append("%sErr"%col)

def checkNumeric(col, safe):
    ref[col].fillna(safe, inplace=True)
    verif[col].fillna(safe, inplace=True)
    test["%sErr"%col] = (ref[col] != verif[col])
    errCols.append("%sErr"%col)

test = pd.DataFrame(index = ref.index)
test["dep_name"] = ref["dep_name"]


# Special test here because of rounding errors
test["timeToDoubleErr"] = (np.abs(ref["timeToDouble"] - verif["timeToDouble"]) > 1e-10)
errCols.append("timeToDoubleErr")

checkString("reg_start")
checkString("reg_end")
checkString("cont_start")
checkString("cont_end")
checkString("rate_date")

checkNumeric("hosp_rate_urgence", -1)
checkNumeric("hosp_rate_all", -1)
checkNumeric("trend_confidence", -1)


test["difference"] = test[errCols[0]]
for e in errCols[1:]:
    test["difference"] = np.logical_or(test["difference"], test[e])

differences = test[test["difference"]]

if len(differences) == 0:
    print("Success")
else:
    print(differences)
    print(ref[test["difference"]])
    print(verif[test["difference"]])


