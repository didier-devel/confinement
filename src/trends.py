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

def make_curve(urgence, use_raw_urg, hosp, use_raw_hosp, src_urgence, roll_urg, roll_hosp, file_radical, df_row, label, logScale):
    # Plot
    fig = plt.figure(figsize=(10,6))
    ax = plt.axes()


    has_reg = df_row["reg_start"] is not None
    

    # Ajout d'un échelle à droite pour meilleure lecture sur les telephones
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(labeltop=False, labelright=True)

    if src_urgence:
        plot_non_zero(ax, logScale, urgence, "nbre_hospit_corona", "Nouvelles hospitalisations quotidiennes - données urgences")
        ax.plot(urgence[roll_urg], label="Nouvelles hospitalisations quotidiennes lissées")
        if has_reg:
            ax.plot(urgence["pred_hosp"], "--", label="Tendance hospitalisations quotidiennes")
            # Intervalle de confiance sur la courbe lissée s'il y a des valeurs nulles
            ax.fill_between(urgence.index, urgence["pred_max"], urgence["pred_min"],color="blue" if use_raw_urg else "orange",alpha=0.3)

        # Autres données (non utilsées pour la tendance)
        ax.plot(hosp[roll_hosp], label="Nouvelles hospitalisations quotidiennes lissées - données hôpitaux")        
    else:
        plot_non_zero(ax, logScale, hosp, "incid_hosp", "Nouvelles hospitalisations quotidiennes - données hôpitaux")
        ax.plot(hosp[roll_hosp], label="Nouvelles hospitalisations quotidiennes lissées")
        if has_reg:
            ax.plot(hosp["pred_hosp"], "--", label="Tendance hospitalisations quotidiennes")
            # Intervalle de confiance sur la courbe lissée s'il y a des valeurs nulles
            ax.fill_between(hosp.index, hosp["pred_max"], hosp["pred_min"],color="blue" if use_raw_hosp else "orange",alpha=0.3)
        
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
    recent = df_source.loc[df_source.index[-recent_hist:]]
    nullVals = len(recent[recent[col] == 0])
    roll_col = "rolling_%s"%col
    if nullVals == 0:
        # Remplacer les valeurs nulles non récentes par na (pour ne pas perturber la vue en echelle logarithmique)
        df_source.loc[df_source[col] == 0,col] = np.nan
        # Calculer la moyenne lissée
        df_source[roll_col] = df_source[col].rolling(7,center=True).aggregate(lambda x: x.prod()**(1./7))
    else:
        df_source[roll_col] = df_source[col].rolling(7,center=True).aggregate(lambda x: x.sum()/7.)

    return roll_col

def make_trend(df_source, hosp_col, roll_col, recent_hist):


    recent = df_source.iloc[-recent_hist:]
    nullVals = len(recent[recent[hosp_col] == 0])

    # On tolere 2 valeurs manquantes pour une regression sur les données brutes
    # (Note: cela ajoute de l'incertitude sur l'intervalle de confiance - choix à revisiter peut-être)
    missingTolerance = 2
    
    if nullVals <= missingTolerance:
        reg_col = hosp_col
    else:
        # Utiliser la moyenne glissante pour ne pas avoir de zéros
        reg_col = roll_col
        
    # Interpolation sur une période plus longue s'il y a trop de jours sans hospitalisation
    # (valeurs faibles donc plus de bruit)
    reg_hist = recent_hist if nullVals <= missingTolerance else int(recent_hist*2)
    
    # Ajouter une colonne de numéro de jour
    df_source["num_jour"] = np.arange(len(df_source))


    
    for_regression = df_source.iloc[-reg_hist:]
    # Si pas assez de données ne pas générer de tendance
    if len(for_regression[for_regression[reg_col] > 0]) < reg_hist*0.5:
        return None, None, df_source, None, None


    
        
    
    if reg_col == hosp_col:
        # Enlever les deux derniers jours - car tous les chiffres ne sont pas toujours remontés
        for_regression = for_regression.loc[for_regression.index[:-2]]

        
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
    df_source["conf_log_raw"] = np.nan
    
    # Plage pour l'intervalle de confiance: depuis les données utilisées pour la régerssion linéaire
    df_source.loc[predIndex,"conf_log_raw"] = np.sqrt(1 + 1./len(X_train) + \
                                                      (df_source["num_jour"]-X_train_mean)**2 / ((X_train["num_jour"]-X_train_mean)**2).sum()) * \
                                                      sigma*scipy.stats.t.ppf(0.95,len(X_train)-2)
    df_source["pred_max"] = df_source["pred_hosp"]*np.exp(df_source["conf_log_raw"])
    df_source["pred_min"] = df_source["pred_hosp"]/np.exp(df_source["conf_log_raw"])

    trend_trusted = None
    
    if reg_col == hosp_col:
        # Alors on regarde si la moyenne glissante est dans l'intervalle de confiance de la moyenne
        for_confidence_index_start = df_source.index.get_loc(for_regression.index[0])
        for_confidence_index_end = df_source.index.get_loc(for_regression.index[-1])
        for_confidence = df_source.iloc[for_confidence_index_start:for_confidence_index_end]
        mean_conf = np.sqrt(1./len(X_train) + \
                            (for_confidence["num_jour"]-X_train_mean)**2 / ((X_train["num_jour"]-X_train_mean)**2).sum()) * \
                            sigma*scipy.stats.t.ppf(0.95,len(X_train)-2)
        mean_conf = mean_conf.loc[for_confidence.index]
        pred_mean = df_source.loc[for_confidence.index, "pred_hosp"]
        verif_trend = pd.DataFrame(index=for_confidence.index)
        verif_trend["pred_mean"] = pred_mean
        verif_trend["mean"] = for_confidence[roll_col]
        verif_trend["mean_min"] = pred_mean/np.exp(mean_conf) 
        verif_trend["mean_max"] = pred_mean*np.exp(mean_conf) 
        verif_trend["roll_over"] = pred_mean*np.exp(mean_conf) < for_confidence[roll_col] 
        verif_trend["roll_under"] = pred_mean/np.exp(mean_conf) > for_confidence[roll_col]

        roll_out =  verif_trend["roll_over"].sum() + verif_trend["roll_under"].sum()

        trend_trusted = 0 if roll_out > 1 else 1
        if roll_out > 1:
            print("Regression sur %s" % reg_col)
            print(verif_trend)

    
    return for_regression.index, timeToDouble, df_source, reg_col == hosp_col, trend_trusted
    

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
        urg_index, urg_timeToDouble, urgence, use_raw_urg, trend_trusted = make_trend(urgence, "nbre_hospit_corona", roll_urg, recent_hist)
    else:
        # Python interpreter complains if the value is not assigned
        use_raw_urg = None
        
    hosp_index, hosp_timeToDouble, hosp, use_raw_hosp, hosp_trend_trusted = make_trend(hosp, "incid_hosp", roll_hosp, recent_hist)

    if not src_urgence:
        trend_trusted = hosp_trend_trusted

    
    
    if src_urgence: 
        df_row["reg_start"] = urg_index[0] if urg_index is not None else None
        df_row["reg_end"]=urg_index[-1] if urg_index is not None else None
        cont_end_loc = urgence.index.get_loc(urg_index[-1]) - 11 if urg_index is not None else None
        cont_start_loc = urgence.index.get_loc(urg_index[0]) - 11 if urg_index is not None else None
        df_row["cont_end"]=urgence.index[cont_end_loc] if urg_index is not None else None
        df_row["cont_start"]=urgence.index[cont_start_loc] if urg_index is not None else None
        df_row["timeToDouble"] = urg_timeToDouble
    else:
        df_row["reg_start"] = hosp_index[0] if hosp_index is not None else None
        df_row["reg_end"]=hosp_index[-1] if hosp_index is not None else None
        cont_end_loc = hosp.index.get_loc(hosp_index[-1]) - 11 if hosp_index is not None else None
        cont_start_loc = hosp.index.get_loc(hosp_index[0]) - 11 if hosp_index is not None else None
        df_row["cont_end"]=hosp.index[cont_end_loc] if hosp_index is not None else None
        df_row["cont_start"]=hosp.index[cont_start_loc] if hosp_index is not None else None
        df_row["timeToDouble"] = hosp_timeToDouble


    df_source = urgence if src_urgence else hosp

    if df_row["reg_start"] is not None:
        if trend_trusted is not None:
            # moyenne glissante en accord avec la tendance
            df_row["trend_confidence"] = trend_trusted
        else:
            # On donne la tendance si la borne max est petite - sinon, seulement si le ration entre min et max est  inférieur à 30
            df_row["trend_confidence"] = 1
            if df_source["pred_max"][-1]/max(1,df_source["pred_min"][-1]) > 30:
                df_row["trend_confidence"] = 0
                if df_source["pred_max"][df_row["reg_start"]]/max(1,df_source["pred_min"][df_row["reg_start"]]) > 30:
                    df_row["trend_confidence"] = 0
    else:
        # Pas de tendance s'il n'y avait pas assez de données pour la calculer
        df_row["trend_confidence"] = 0

        
    make_curve(urgence, use_raw_urg, hosp, use_raw_hosp, src_urgence, roll_urg, roll_hosp, file_radical, df_row, label, True)
    make_curve(urgence, use_raw_urg, hosp, use_raw_hosp, src_urgence, roll_urg, roll_hosp, file_radical, df_row, label, False)
    
    

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


