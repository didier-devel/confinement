import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from datetime import datetime, date, timedelta
from sklearn.linear_model import LinearRegression
import scipy
import math
import sys


file_path = os.path.dirname(os.path.realpath(__file__))
proj_path = os.path.abspath(os.path.join(file_path,".."))
datagouv_path = os.path.join(proj_path,"datagouv")
gen_path = os.path.join(proj_path,"../../gatsby/trends/generated")
datagen_path = os.path.join(gen_path,"data")

def downloadIfNeeded(fileName):
    need_download = True
    if os.path.exists(fileName):
        today = date.today()
        last_modified_ts = os.path.getmtime(fileName)
        mtime = date.fromtimestamp(last_modified_ts)
        if (today-mtime).days <= 1:
            need_download = False
    if need_download:
        print("%s Needs a download"%fileName)
        if "department" in fileName:
            command = "/usr/bin/wget https://www.data.gouv.fr/fr/datasets/r/eceb9fb4-3ebc-4da3-828d-f5939712600a -O %s"%fileName
        elif "hospitalieres" in fileName:
            command = "/usr/bin/wget https://www.data.gouv.fr/fr/datasets/r/6fadff46-9efd-4c53-942a-54aca783c30c -O %s"%fileName
            
        os.system(command)
    else:
        print("%s est à jour"%fileName)


urgence_data = os.path.join(datagouv_path,"department_latest.csv")
downloadIfNeeded(urgence_data)    
urgence_df = pd.read_csv(urgence_data)

hosp_data = os.path.join(datagouv_path,"donnees_hospitalieres_latest.csv")
downloadIfNeeded(hosp_data)    
hosp_df = pd.read_csv(hosp_data, sep=';')

# Heure des données (wget garde le mtime du site web)
last_modified_ts = os.path.getmtime(urgence_data)
data_date = datetime.fromtimestamp(last_modified_ts)


#extraire les données toutes classe d'age
urgence_df = urgence_df[urgence_df["sursaud_cl_age_corona"] == "0"].copy()

# Lire le fichier des code département
depts = pd.read_csv(os.path.join(datagouv_path,"departement2020.csv"))
depts.set_index(depts.dep, inplace=True)
depts.drop("dep",axis=1, inplace=True)

# Lire le fichier des régions
regs = pd.read_csv(os.path.join(datagouv_path,"region2020.csv"))
#regs["reg"] = regs["reg"].apply(lambda x: str(x) if len(str(x)) > 1 else '0' + str(x))
regs.set_index(regs.reg,  inplace=True)
regs.drop("reg", axis=1, inplace=True)


# Ajouter nom de département, code région, nom région dans les données des urgences
urgence_df["dep_name"] = urgence_df["dep"].apply(lambda x: depts.loc[str(x)].libelle if pd.notnull(x) else None)
urgence_df["reg"] = urgence_df["dep"].apply(lambda x: depts.loc[x].reg if pd.notnull(x) else None)
urgence_df["reg_name"] = urgence_df["reg"].apply(lambda x: regs.loc[x].libelle if pd.notnull(x) else None)

# Ajouter nom de département, code région, nom région dans les données des hospitalières
hosp_df["dep_name"] = hosp_df["dep"].apply(lambda x: depts.loc[str(x)].libelle if pd.notnull(x) else None)
hosp_df["reg"] = hosp_df["dep"].apply(lambda x: depts.loc[x].reg if pd.notnull(x) else None)
hosp_df["reg_name"] = hosp_df["reg"].apply(lambda x: regs.loc[x].libelle if pd.notnull(x) else None)

# Afficher les dates au format jj/mm/yy et les mettre en index
def convertDate(isodate):
    l = isodate.split('-')
    return l[2]+"/"+l[1]+"/"+l[0][2:]


def addDays(df, duration):
    # Agrandissement du dataframe du nombre de jours spécifié
    d = df.index[-1]
    a = d.split("/")
    dd = int(a[0])
    mm = int(a[1])
    yy = 2000 + int(a[2])
    first = date(yy,mm,dd)+ timedelta(days=1)
    last = date(yy,mm,dd)+ timedelta(days=duration)

    current = first
    indexExtension  = []
    while current  <= last:
        ds = str(current.day)
        if len(ds) == 1:
            ds = '0'+ds
        ms = str(current.month)
        if len(ms) == 1:
            ms = '0'+ms
        ys = str(current.year)[2:]
        di = ds + '/' + ms + '/' + ys
        indexExtension.append(di)
        current += timedelta(days = 1)

    return df.reindex(index = df.index.append(pd.Index(indexExtension)))


# Calcul de l'intervalle de confiance de la prédiction
# Voir http://pageperso.lif.univ-mrs.fr/~alexis.nasr/Ens/IAAAM2/SlidesModStat_C1_print.pdf
def estimateSigma(reg, X, Y):
    Y_pred = reg.predict(X)
    err = (Y - Y_pred)**2
    return math.sqrt(err.sum() / (len(err) - 2))

def plot_non_zero(ax, logScale, df, col, label):
    col_draw = col
    if logScale:
        col_draw = "nnz_%s"%col 
        df[col_draw] = df[col]
        df.loc[df[col] == 0 ,col_draw] = np.nan
    
    ax.plot(df[col_draw], label=label)

def make_curve(urgence, urg_index, hosp, hosp_index, src_urgence, roll_urg, roll_hosp, file_radical, df_row, label, logScale):
    # Plot
    fig = plt.figure(figsize=(10,6))
    ax = plt.axes()


    has_reg = df_row["reg_start"] is not None
    

    # Ajout d'un échelle à droite pour meilleure lecture sur les telephones
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(labeltop=False, labelright=True)

    if src_urgence:
        #plot_non_zero(ax, logScale, urgence, "nbre_hospit_corona", "Nouvelles hospitalisations quotidiennes - données urgences")
        ax.bar(urgence.index, urgence["nbre_hospit_corona"], label = "Nouvelles hospitalisations quotidiennes - données urgences", alpha=0.3, color="blue")
        ax.plot(urgence[roll_urg], label="Nouvelles hospitalisations quotidiennes lissées - données urgences", color="orange")
        if has_reg:
            ax.plot(urgence["pred_hosp"], "--", label="Tendance hospitalisations quotidiennes -- données urgences", color="orange")
            ax.fill_between(urgence.index, urgence["pred_max"], urgence["pred_min"],color="orange",alpha=0.3, label="Intervalle de confiance")
            # En plus foncé sur la zone de prediction
            pred_index = urgence.index.difference(urg_index, sort=False)
            ax.fill_between(pred_index, urgence.loc[pred_index, "pred_max"], urgence.loc[pred_index, "pred_min"],color="orange",alpha=0.2)

        # Autres données (non utilsées pour la tendance)
        ax.plot(hosp[roll_hosp], label="Nouvelles hospitalisations quotidiennes lissées - données hôpitaux", color="red")        
    else:
        ax.bar(hosp.index, hosp["incid_hosp"], label = "Nouvelles hospitalisations quotidiennes - données hôpitaux", alpha=0.3, color="blue")
        ax.plot(hosp[roll_hosp], label="Nouvelles hospitalisations quotidiennes lissées - données hôpitaux", color="orange")
        if has_reg:
            ax.plot(hosp["pred_hosp"], "--", label="Tendance hospitalisations quotidiennes - données hôpitaux", color="orange")
            ax.fill_between(hosp.index, hosp["pred_max"], hosp["pred_min"],color="orange",alpha=0.3, label="Intervalle de confiance")
            # En plus foncé sur la zone de prediction
            pred_index = hosp.index.difference(hosp_index, sort=False)
            ax.fill_between(pred_index, hosp.loc[pred_index, "pred_max"], hosp.loc[pred_index,"pred_min"],color="orange",alpha=0.2)
        
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.legend()

    if src_urgence:
        # Pour utiliser cette limite pour les données hospitalières, il faudrait étendre l'index vers le 24 février.
        ax.set_xlim(left = "24/02/20", right=urgence.index[-1])

    if logScale:
        plt.yscale("log")
        # Same scale for log curves
        ax.set_ylim(0.1,3000)
    else:
        if has_reg:
            # Protection contre les prédiction trop divergeantes
            df_source = urgence if src_urgence else hosp
            hosp_col = "nbre_hospit_corona" if src_urgence else "incid_hosp"
            if df_source.loc[df_source.index[-1], "pred_max"] > df_source[hosp_col].max()*4:
                ax.set_ylim(0, df_source[hosp_col].max()*4)


    ax.set_title("Hospitalisations COVID-19 quotidiennes en %s - échelle %s"%(label,"logarithmique" if logScale else "linéaire"))

    
                
    file_name = file_radical + ("_log" if logScale else  "_lin") + ".png"
    plt.savefig(os.path.join(datagen_path,file_name))
    df_row["log_curve" if logScale else "lin_curve"] = file_name
    plt.close()
    

def aggregate(df_source, date_col):
    df_source = df_source.groupby([date_col]).agg('sum')
    # Convertir les dates maintenant que les tris sont faits
    df_source["date"] = df_source.index
    df_source["date"] = df_source["date"].apply(convertDate)
    df_source = df_source.set_index(["date"])
        
    return df_source
    
def make_rolling(df_source, col, recent_hist):
    roll_col = "rolling_%s"%col
    nnz_col = "nnz_%s"%col

    df_source[nnz_col] = df_source[col]
    df_source.loc[df_source[nnz_col]==0,nnz_col] = 0.1
    
    # Calculer la moyenne lissée géométrique
    df_source[roll_col] = df_source[nnz_col].rolling(7,center=True).aggregate(lambda x: x.prod()**(1./7))
    # Remplacer ce qui vaut 0.1 par 0
    df_source.loc[df_source[roll_col]<=0.101, roll_col] = 0

    return roll_col

def make_trend(df_source, hosp_col, roll_col, recent_hist):


    recent = df_source.iloc[-recent_hist:]
    nullVals = len(recent[recent[hosp_col] == 0])

    if nullVals == 0:
        reg_col = hosp_col
    else:
        # Remplacer les valeurs nulles par 0.1 (ou 0 si la moyenne glissante vaut 0)
        reg_col = "%s_patch"%hosp_col
        df_source[reg_col] = df_source[hosp_col]
        df_source.loc[df_source[reg_col] == 0, reg_col] = 0.1
        df_source.loc[df_source[roll_col] == 0, reg_col] = 0
        # Si plus de 2 valeurs nulles, on double aussi la période d'estimation
        if nullVals > 2:
            recent_hist *= 2
        else:
            recent_hist = int(recent_hist*1.5)
        # Update null vals
        recent = df_source.iloc[-recent_hist:]

        
    # Ajouter une colonne de numéro de jour
    df_source["num_jour"] = np.arange(len(df_source))


    
    for_regression = df_source.iloc[-recent_hist:]
    # Si pas assez de données ne pas générer de tendance
    if len(for_regression[for_regression[reg_col] > 0]) < recent_hist*0.5:
        return None, None, df_source


    # Enlever les valeurs nulles ou non définies
    for_regression = for_regression[for_regression[reg_col] > 0]

    reg = LinearRegression()
    X_train = for_regression.drop(columns = [c for c in for_regression.columns if c != "num_jour"])
    Y_train = np.log(for_regression[reg_col])
    reg.fit(X_train,Y_train)

    # Extraire la pente de la regression
    slope = reg.coef_[0]
    timeToDouble = math.log(2)/slope

    
    # Ajouter deux semaines de données et mettre a jour la colonne num_jour
    df_source = addDays(df_source, 15)

    df_source["num_jour"] = np.arange(len(df_source))

    # Ajouter la prédiction dans les données
    df_source["pred_hosp"]=np.nan

    # Plage de prédiction: dans la phase descendante - jusqu'à last_day
    predIndex = df_source[(df_source["num_jour"] >= X_train.iloc[0]["num_jour"])].index  
    X = df_source.loc[predIndex].drop(columns = [c for c in df_source.columns if c != "num_jour"])
    df_source.loc[predIndex,"pred_hosp"]=np.exp(reg.predict(X))

    # Intervalle de confiance
    sigma = estimateSigma(reg,X_train,Y_train)
    X_train_mean = X_train["num_jour"].mean()
    
    # Ajout de l'intervalle de confiance en log (alpha = 10% -- 1 - alpha/2 = 0.95)
    df_source["conf_log_mean"] = np.nan
    
    # Plage pour l'intervalle de confiance sur la moyennes: depuis les données utilisées pour la régerssion linéaire
    df_source.loc[predIndex,"conf_log_mean"] = np.sqrt(1./len(X_train) + \
                                                      (df_source["num_jour"]-X_train_mean)**2 / ((X_train["num_jour"]-X_train_mean)**2).sum()) * \
                                                      sigma*scipy.stats.t.ppf(0.95,len(X_train)-2)
    df_source["pred_max"] = df_source["pred_hosp"]*np.exp(df_source["conf_log_mean"])
    df_source["pred_min"] = df_source["pred_hosp"]/np.exp(df_source["conf_log_mean"])


    return for_regression.index, timeToDouble, df_source
    

def make_data(urgence, hosp, file_radical, df_row, label):

    urgence = aggregate(urgence, "date_de_passage")
    hosp = aggregate(hosp, "jour")
    
    recent_hist  = 15

    recent = urgence.loc[urgence.index[-recent_hist:]]
    # Si trop de valeurs sont nulles dans les données urgence,
    # on utilise les données hospitalières
    src_urgence = True if len(recent[recent["nbre_hospit_corona"] == 0]) <= recent_hist * 0.7 else False

    roll_urg = make_rolling(urgence, "nbre_hospit_corona", recent_hist)
    roll_hosp = make_rolling(hosp, "incid_hosp", recent_hist)

    # On utilise le dernier jour de la moyenne lissée pour indiquer le nombre d'hospitalisations par jour
    if src_urgence:
        df_row["hosp_rate_urgence"] =  urgence[urgence[roll_urg] > 0 ][roll_urg][-1] 
        df_row["hosp_rate_all"] = hosp[hosp[roll_hosp] > 0 ][roll_hosp][-1]
    else:
        df_row["hosp_rate_all"] = hosp[hosp[roll_hosp] > 0 ][roll_hosp][-1]

    # make_trend modifies the dataframe (it extends the index) so we need to update the df variables
    if src_urgence:
        urg_index, urg_timeToDouble, urgence = make_trend(urgence, "nbre_hospit_corona", roll_urg, recent_hist)
    else:
        # Python interpreter complains if the value is not assigned
        urg_index = None
        
    hosp_index, hosp_timeToDouble, hosp = make_trend(hosp, "incid_hosp", roll_hosp, recent_hist)


    
    
    if src_urgence: 
        df_row["reg_start"] = urg_index[0] if urg_index is not None else None
        df_row["reg_end"]=urg_index[-1] if urg_index is not None else None
        cont_end_loc = urgence.index.get_loc(urg_index[-1]) - 11 if urg_index is not None else None
        cont_start_loc = urgence.index.get_loc(urg_index[0]) - 11 if urg_index is not None else None
        df_row["cont_end"]=urgence.index[cont_end_loc] if urg_index is not None else None
        df_row["cont_start"]=urgence.index[cont_start_loc] if urg_index is not None else None
        df_row["timeToDouble"] = urg_timeToDouble
        if urgence["pred_max"][-1] > df_row["hosp_rate_urgence"]*2 and urgence["pred_min"][-1] < df_row["hosp_rate_urgence"]/2.:
            df_row["trend_confidence"] = 0
        else:
            df_row["trend_confidence"] = 1

    else:
        df_row["reg_start"] = hosp_index[0] if hosp_index is not None else None
        df_row["reg_end"]=hosp_index[-1] if hosp_index is not None else None
        cont_end_loc = hosp.index.get_loc(hosp_index[-1]) - 11 if hosp_index is not None else None
        cont_start_loc = hosp.index.get_loc(hosp_index[0]) - 11 if hosp_index is not None else None
        df_row["cont_end"]=hosp.index[cont_end_loc] if hosp_index is not None else None
        df_row["cont_start"]=hosp.index[cont_start_loc] if hosp_index is not None else None
        df_row["timeToDouble"] = hosp_timeToDouble
        if df_row["reg_start"] is not None:
            # Pas de tendance s'il n'y avait pas assez de données pour la calculer
            df_row["trend_confidence"] = 0
            if hosp["pred_max"][-1] > df_row["hosp_rate_all"]*2 and hosp["pred_min"][-1] < df_row["hosp_rate_all"]/2.:
                df_row["trend_confidence"] = 0
            else:
                df_row["trend_confidence"] = 1
        else:
            df_row["trend_confidence"] = 0




        
    make_curve(urgence, urg_index, hosp, hosp_index, src_urgence, roll_urg, roll_hosp, file_radical, df_row, label, True)
    make_curve(urgence, urg_index, hosp, hosp_index, src_urgence, roll_urg, roll_hosp, file_radical, df_row, label, False)
    
    

common_fields = ["log_curve", "lin_curve","timeToDouble", "reg_start", "reg_end", "cont_start", "cont_end", "hosp_rate_urgence", "hosp_rate_all", "trend_confidence"]

fr_summary = pd.DataFrame(index=["France"],columns=["data_date"] + common_fields)
fr_summary.loc["France","data_date"] = data_date.strftime("%d/%m/%Y %H:%M")
make_data(urgence_df, hosp_df, "france", fr_summary.loc["France"], "France")
fr_summary.to_csv(os.path.join(datagen_path, "france.csv"), index_label='id')


metropole = [r for r in regs.index if r > 10]
drom = [r for r in regs.index if r < 10]


reg_summary = pd.DataFrame(index = metropole+drom, columns=["reg_name"] + common_fields)
dep_summary = pd.DataFrame(index = depts.index, columns=["dep_name", "reg"] + common_fields)

for reg in metropole + drom:
    reg_name = regs.loc[reg]["libelle"]
    file_radical = code = "r_" + str(reg)
    print(reg, reg_name)
    reg_summary.loc[reg]["reg_name"] = reg_name
    make_data(urgence_df[urgence_df["reg"] == reg], hosp_df[hosp_df["reg"] == reg], file_radical, reg_summary.loc[reg], reg_name)
    reg_depts = depts[depts["reg"]==reg]
    for dept in reg_depts.index:
        dep_name = reg_depts.loc[dept,"libelle"]
        dep_summary.loc[dept,"reg"] = reg
        dep_summary.loc[dept,"dep_name"] = dep_name
        file_radical = code = "d_" + str(dept)
        print("\t%s %s"%(dept, dep_name))
        make_data(urgence_df[urgence_df["dep"] == dept], hosp_df[hosp_df["dep"] == dept], file_radical, dep_summary.loc[dept], dep_name)


reg_summary.to_csv(os.path.join(datagen_path, "regions.csv"), index_label="reg")
dep_summary.to_csv(os.path.join(datagen_path, "departements.csv"), index_label="dep")


