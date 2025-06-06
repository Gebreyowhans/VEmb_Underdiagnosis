import pandas as pd
import numpy as np
import math
import random as python_random
import io
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from IPython.display import clear_output
import warnings

from config_MIMIC import get_diseases, get_diseases_abbr, get_seeds, get_patient_groups


def fpr(df, d, c, category_name):

    pred_disease = "bi_" + d
    gt = df.loc[(df[d] == 0) & (df[category_name] == c), :]
    pred = df.loc[(df[pred_disease] == 1) & (
        df[d] == 0) & (df[category_name] == c), :]
    if len(gt) != 0:
        FPR = len(pred) / len(gt)
        return FPR
    else:
        # print("Disease", d, "in category", c, "has zero division error")
        return -1


def Subgroup_FPR(df, diseases, category, category_name, seed=19, fpr_npr_path="default"):

    # return FPR and FNR per subgroup and the unber of patients with 0 No-finding in test set.
    FP_total = []
    percentage_total = []
    FN_total = []

    if category_name == 'insurance':
        FPR_Ins = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'race':
        FPR_race = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'gender':
        FPR_sex = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'age_decile':
        FPR_age = pd.DataFrame(diseases, columns=["diseases"])

    print("FP in MIMIC====================================")

    for c in category:
        FP_y = []
        FN_y = []
        percentage_y = []

        for d in diseases:
            pred_disease = "bi_" + d
            
            gt_fp = df.loc[(df[d] == 0) & (df[category_name] == c), :]
            gt_fn = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            
            pred_fp = df.loc[(df[pred_disease] == 1) & (
                df[d] == 0) & (df[category_name] == c), :]
            pred_fn = df.loc[(df[pred_disease] == 0) & (
                df[d] == 1) & (df[category_name] == c), :]

            pi_gy = df.loc[(df[d] == 0) & (df[category_name] == c), :]

            if len(gt_fp) != 0:

                FPR = len(pred_fp) / len(gt_fp)
                Percentage = len(pi_gy)
                FP_y.append(round(FPR, 3))
                percentage_y.append(round(Percentage, 3))

                print(len(pred_fp), '--', len(gt_fp), '====', c)

            else:

                FP_y.append(np.NaN)
                percentage_y.append(0)

            if len(gt_fn) != 0:
                FNR = len(pred_fn) / len(gt_fn)
                FN_y.append(round(FNR, 3))

            else:
                FN_y.append(np.NaN)

            FP_total.append(FP_y)
            percentage_total.append(percentage_y)
            FN_total.append(FN_y)

        # print("False Positive Rate in " + category[c] + " for " + diseases[d] + " is: " + str(FPR))

    for i in range(len(FN_total)):

        if category_name == 'gender':
            if i == 0:
                Perc_S = pd.DataFrame(percentage_total[i], columns=["#M"])
                FPR_sex = pd.concat(
                    [FPR_sex, Perc_S.reindex(FPR_sex.index)], axis=1)

                FPR_S = pd.DataFrame(FP_total[i], columns=["FPR_M"])
                FPR_sex = pd.concat(
                    [FPR_sex, FPR_S.reindex(FPR_sex.index)], axis=1)
                FNR_S = pd.DataFrame(FN_total[i], columns=["FNR_M"])
                FPR_sex = pd.concat(
                    [FPR_sex, FNR_S.reindex(FPR_sex.index)], axis=1)

            if i == 1:
                Perc_S = pd.DataFrame(percentage_total[i], columns=["#F"])
                FPR_sex = pd.concat(
                    [FPR_sex, Perc_S.reindex(FPR_sex.index)], axis=1)

                FPR_S = pd.DataFrame(FP_total[i], columns=["FPR_F"])
                FPR_sex = pd.concat(
                    [FPR_sex, FPR_S.reindex(FPR_sex.index)], axis=1)

                FNR_S = pd.DataFrame(FN_total[i], columns=["FNR_F"])
                FPR_sex = pd.concat(
                    [FPR_sex, FNR_S.reindex(FPR_sex.index)], axis=1)
                FPR_sex.to_csv(fpr_npr_path+"run_" +
                               str(seed)+"FPR_FNR_NF_sex.csv")

        if category_name == 'age_decile':

            if i == 0:
                Perc_A = pd.DataFrame(percentage_total[i], columns=["#60-80"])
                FPR_age = pd.concat(
                    [FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)

                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_60-80"])
                FPR_age = pd.concat(
                    [FPR_age, FPR_A.reindex(FPR_age.index)], axis=1)

                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_60-80"])
                FPR_age = pd.concat(
                    [FPR_age, FNR_A.reindex(FPR_age.index)], axis=1)

            if i == 1:

                Perc_A = pd.DataFrame(percentage_total[i], columns=["#40-60"])
                FPR_age = pd.concat(
                    [FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)

                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_40-60"])
                FPR_age = pd.concat(
                    [FPR_age, FPR_A.reindex(FPR_age.index)], axis=1)

                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_40-60"])
                FPR_age = pd.concat(
                    [FPR_age, FNR_A.reindex(FPR_age.index)], axis=1)

            if i == 2:

                Perc_A = pd.DataFrame(percentage_total[i], columns=["#20-40"])
                FPR_age = pd.concat(
                    [FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)

                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_20-40"])
                FPR_age = pd.concat(
                    [FPR_age, FPR_A.reindex(FPR_age.index)], axis=1)
                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_20-40"])
                FPR_age = pd.concat(
                    [FPR_age, FNR_A.reindex(FPR_age.index)], axis=1)

            if i == 3:
                Perc_A = pd.DataFrame(percentage_total[i], columns=["#80+"])
                FPR_age = pd.concat(
                    [FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)
                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_80+"])
                FPR_age = pd.concat(
                    [FPR_age, FPR_A.reindex(FPR_age.index)], axis=1)
                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_80+"])
                FPR_age = pd.concat(
                    [FPR_age, FNR_A.reindex(FPR_age.index)], axis=1)

            if i == 4:

                Perc_A = pd.DataFrame(percentage_total[i], columns=["#0-20"])
                FPR_age = pd.concat(
                    [FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)

                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_0-20"])
                FPR_age = pd.concat(
                    [FPR_age, FPR_A.reindex(FPR_age.index)], axis=1)

                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_0-20"])
                FPR_age = pd.concat(
                    [FPR_age, FNR_A.reindex(FPR_age.index)], axis=1)
                FPR_age.to_csv(fpr_npr_path+"run_" +
                               str(seed)+"FPR_FNR_NF_age.csv")

        if category_name == 'insurance':

            if i == 0:
                Perc_A = pd.DataFrame(
                    percentage_total[i], columns=["#Medicare"])
                FPR_Ins = pd.concat(
                    [FPR_Ins, Perc_A.reindex(FPR_Ins.index)], axis=1)

                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_Medicare"])
                FPR_Ins = pd.concat(
                    [FPR_Ins, FPR_A.reindex(FPR_Ins.index)], axis=1)

                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_Medicare"])
                FPR_Ins = pd.concat(
                    [FPR_Ins, FNR_A.reindex(FPR_Ins.index)], axis=1)

            if i == 1:

                Perc_A = pd.DataFrame(percentage_total[i], columns=["#Other"])
                FPR_Ins = pd.concat(
                    [FPR_Ins, Perc_A.reindex(FPR_Ins.index)], axis=1)

                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_Other"])
                FPR_Ins = pd.concat(
                    [FPR_Ins, FPR_A.reindex(FPR_Ins.index)], axis=1)

                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_Other"])
                FPR_Ins = pd.concat(
                    [FPR_Ins, FNR_A.reindex(FPR_Ins.index)], axis=1)

            if i == 2:

                Perc_A = pd.DataFrame(
                    percentage_total[i], columns=["#Medicaid"])
                FPR_Ins = pd.concat(
                    [FPR_Ins, Perc_A.reindex(FPR_Ins.index)], axis=1)

                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_Medicaid"])
                FPR_Ins = pd.concat(
                    [FPR_Ins, FPR_A.reindex(FPR_Ins.index)], axis=1)

                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_Medicaid"])
                FPR_Ins = pd.concat(
                    [FPR_Ins, FNR_A.reindex(FPR_Ins.index)], axis=1)

                FPR_Ins.to_csv(fpr_npr_path+"run_"+str(seed) +
                               "FPR_FNR_NF_insurance.csv")

        if category_name == 'race':

            if i == 0:

                Perc_A = pd.DataFrame(percentage_total[i], columns=["#White"])
                FPR_race = pd.concat(
                    [FPR_race, Perc_A.reindex(FPR_race.index)], axis=1)

                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_White"])
                FPR_race = pd.concat(
                    [FPR_race, FPR_A.reindex(FPR_race.index)], axis=1)

                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_White"])
                FPR_race = pd.concat(
                    [FPR_race, FNR_A.reindex(FPR_race.index)], axis=1)

            if i == 1:
                Perc_A = pd.DataFrame(percentage_total[i], columns=["#Black"])
                FPR_race = pd.concat(
                    [FPR_race, Perc_A.reindex(FPR_race.index)], axis=1)

                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_Black"])
                FPR_race = pd.concat(
                    [FPR_race, FPR_A.reindex(FPR_race.index)], axis=1)

                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_Black"])
                FPR_race = pd.concat(
                    [FPR_race, FNR_A.reindex(FPR_race.index)], axis=1)

            if i == 2:
                Perc_A = pd.DataFrame(percentage_total[i], columns=["#Hisp"])
                FPR_race = pd.concat(
                    [FPR_race, Perc_A.reindex(FPR_race.index)], axis=1)

                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_Hisp"])
                FPR_race = pd.concat(
                    [FPR_race, FPR_A.reindex(FPR_race.index)], axis=1)

                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_Hisp"])
                FPR_race = pd.concat(
                    [FPR_race, FNR_A.reindex(FPR_race.index)], axis=1)

            if i == 3:
                Perc_A = pd.DataFrame(percentage_total[i], columns=["#Other"])
                FPR_race = pd.concat(
                    [FPR_race, Perc_A.reindex(FPR_race.index)], axis=1)

                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_Other"])
                FPR_race = pd.concat(
                    [FPR_race, FPR_A.reindex(FPR_race.index)], axis=1)

                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_Other"])
                FPR_race = pd.concat(
                    [FPR_race, FNR_A.reindex(FPR_race.index)], axis=1)

            if i == 4:
                Perc_A = pd.DataFrame(percentage_total[i], columns=["#Asian"])
                FPR_race = pd.concat(
                    [FPR_race, Perc_A.reindex(FPR_race.index)], axis=1)

                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_Asian"])
                FPR_race = pd.concat(
                    [FPR_race, FPR_A.reindex(FPR_race.index)], axis=1)

                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_Asian"])
                FPR_race = pd.concat(
                    [FPR_race, FNR_A.reindex(FPR_race.index)], axis=1)

            if i == 5:
                Perc_A = pd.DataFrame(
                    percentage_total[i], columns=["#American"])
                FPR_race = pd.concat(
                    [FPR_race, Perc_A.reindex(FPR_race.index)], axis=1)

                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_American"])
                FPR_race = pd.concat(
                    [FPR_race, FPR_A.reindex(FPR_race.index)], axis=1)

                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_American"])
                FPR_race = pd.concat(
                    [FPR_race, FNR_A.reindex(FPR_race.index)], axis=1)
                FPR_race.to_csv(fpr_npr_path+"run_" +
                                str(seed)+"FPR_FNR_NF_race.csv")


def Two_Group_FPR(df, diseases, category1, category_name1, category2,category_name2, seed=19, fpr_npr_path_2_group_intersection="default"):

    print(f'Disease : {diseases}')
    if (category_name1 == 'gender') & (category_name2 == 'insurance'):
        FP_InsSex = pd.DataFrame(category2, columns=["Insurance"])

    if (category_name1 == 'gender') & (category_name2 == 'race'):
        FP_RaceSex = pd.DataFrame(category2, columns=["race"])

    if (category_name1 == 'gender') & (category_name2 == 'age_decile'):
        FP_AgeSex = pd.DataFrame(category2, columns=["Age"])

    if (category_name1 == 'insurance') & (category_name2 == 'race'):
        FP_InsRace = pd.DataFrame(category2, columns=["race"])

    if (category_name1 == 'insurance') & (category_name2 == 'age_decile'):
        FP_InsAge = pd.DataFrame(category2, columns=["age"])

    if (category_name1 == 'race') & (category_name2 == 'age_decile'):
        FP_RaceAge = pd.DataFrame(category2, columns=["age"])

    print("==================================== Calculating FP in vector embedded mimic cxr====================================")

    i = 0

    for c1 in range(len(category1)):
        FPR_list = []
        FNR_list = []

        for c2 in range(len(category2)):

            for d in range(len(diseases)):

                pred_disease = "bi_" + diseases[d]
                gt_fp = df.loc[((df[diseases[d]] == 0) & (df[category_name1] == category1[c1]) &
                                (df[category_name2] == category2[c2])), :]

                gt_fn = df.loc[((df[diseases[d]] == 1) & (df[category_name1] == category1[c1]) &
                                (df[category_name2] == category2[c2])), :]

                pred_fp = df.loc[((df[pred_disease] == 1) & (df[diseases[d]] == 0) & (df[category_name1] == category1[c1]) &
                                  (df[category_name2] == category2[c2])), :]

                pred_fn = df.loc[((df[pred_disease] == 0) & (df[diseases[d]] == 1) & (df[category_name1] == category1[c1]) &
                                  (df[category_name2] == category2[c2])), :]

                if len(gt_fp) != 0:
                    FPR = len(pred_fp) / len(gt_fp)
                    print(len(pred_fp), '--', len(gt_fp))
                    print("False Positive Rate in " +
                          category1[c1] + "/" + category2[c2] + " for " + diseases[d] + " is: " + str(FPR))

                else:
                    FPR = np.NaN
                    print("False Positive Rate in " +
                          category1[c1] + "/" + category2[c2] + " for " + diseases[d] + " is: N\A")

                print(
                    '=======================================================================================================')

            FPR_list.append(round(FPR, 3))

        if (category_name1 == 'gender') & (category_name2 == 'age_decile'):

            if i == 0:
                FPR_SA = pd.DataFrame(FPR_list, columns=["FPR_M"])
                FP_AgeSex = pd.concat(
                    [FP_AgeSex, FPR_SA.reindex(FP_AgeSex.index)], axis=1)

            if i == 1:
                FPR_SA = pd.DataFrame(FPR_list, columns=["FPR_F"])
                FP_AgeSex = pd.concat(
                    [FP_AgeSex, FPR_SA.reindex(FP_AgeSex.index)], axis=1)

        if (category_name1 == 'gender') & (category_name2 == 'race'):

            if i == 0:
                FPR_SR = pd.DataFrame(FPR_list, columns=["FPR_M"])
                FP_RaceSex = pd.concat(
                    [FP_RaceSex, FPR_SR.reindex(FP_RaceSex.index)], axis=1)

            if i == 1:
                FPR_SR = pd.DataFrame(FPR_list, columns=["FPR_F"])
                FP_RaceSex = pd.concat(
                    [FP_RaceSex, FPR_SR.reindex(FP_RaceSex.index)], axis=1)

        if (category_name1 == 'gender') & (category_name2 == 'insurance'):

            if i == 0:
                FPR_SIn = pd.DataFrame(FPR_list, columns=["FPR_M"])
                FP_InsSex = pd.concat(
                    [FP_InsSex, FPR_SIn.reindex(FP_InsSex.index)], axis=1)

            if i == 1:
                FPR_SIn = pd.DataFrame(FPR_list, columns=["FPR_F"])
                FP_InsSex = pd.concat(
                    [FP_InsSex, FPR_SIn.reindex(FP_InsSex.index)], axis=1)

        if (category_name1 == 'insurance') & (category_name2 == 'race'):

            if i == 0:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["FPR_Medicare"])
                FP_InsRace = pd.concat(
                    [FP_InsRace, FPR_RIn.reindex(FP_InsRace.index)], axis=1)

            if i == 1:

                FPR_RIn = pd.DataFrame(FPR_list, columns=["FPR_Other"])
                FP_InsRace = pd.concat(
                    [FP_InsRace, FPR_RIn.reindex(FP_InsRace.index)], axis=1)

            if i == 2:

                FPR_RIn = pd.DataFrame(FPR_list, columns=["FPR_Medicaid"])
                FP_InsRace = pd.concat(
                    [FP_InsRace, FPR_RIn.reindex(FP_InsRace.index)], axis=1)

        if (category_name1 == 'insurance') & (category_name2 == 'age_decile'):

            if i == 0:

                FPR_AIn = pd.DataFrame(FPR_list, columns=["FPR_Medicare"])
                FP_InsAge = pd.concat(
                    [FP_InsAge, FPR_AIn.reindex(FP_InsAge.index)], axis=1)

            if i == 1:
                FPR_AIn = pd.DataFrame(FPR_list, columns=["FPR_Other"])
                FP_InsAge = pd.concat(
                    [FP_InsAge, FPR_AIn.reindex(FP_InsAge.index)], axis=1)

            if i == 2:
                FPR_AIn = pd.DataFrame(FPR_list, columns=["FPR_Medicaid"])
                FP_InsAge = pd.concat(
                    [FP_InsAge, FPR_AIn.reindex(FP_InsAge.index)], axis=1)

        if (category_name1 == 'race') & (category_name2 == 'age_decile'):

            if i == 0:

                FPR_RIn = pd.DataFrame(FPR_list, columns=["FPR_White"])
                FP_RaceAge = pd.concat(
                    [FP_RaceAge, FPR_RIn.reindex(FP_RaceAge.index)], axis=1)

            if i == 1:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["FPR_Black"])
                FP_RaceAge = pd.concat(
                    [FP_RaceAge, FPR_RIn.reindex(FP_RaceAge.index)], axis=1)

            if i == 2:

                FPR_RIn = pd.DataFrame(FPR_list, columns=["FPR_Hisp"])
                FP_RaceAge = pd.concat(
                    [FP_RaceAge, FPR_RIn.reindex(FP_RaceAge.index)], axis=1)

            if i == 3:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["FPR_Other"])
                FP_RaceAge = pd.concat(
                    [FP_RaceAge, FPR_RIn.reindex(FP_RaceAge.index)], axis=1)

            if i == 4:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["FPR_Asian"])
                FP_RaceAge = pd.concat(
                    [FP_RaceAge, FPR_RIn.reindex(FP_RaceAge.index)], axis=1)

            if i == 5:

                FPR_RIn = pd.DataFrame(FPR_list, columns=["FPR_American"])
                FP_RaceAge = pd.concat(
                    [FP_RaceAge, FPR_RIn.reindex(FP_RaceAge.index)], axis=1)

        i += 1

    if (category_name1 == 'gender') & (category_name2 == 'insurance'):
        FP_InsSex.to_csv(fpr_npr_path_2_group_intersection +
                         "run_"+str(seed)+"FP_InsSex.csv")

    if (category_name1 == 'gender') & (category_name2 == 'race'):
        FP_RaceSex.to_csv(fpr_npr_path_2_group_intersection +
                          "run_"+str(seed)+"FP_RaceSex.csv")

    if (category_name1 == 'insurance') & (category_name2 == 'race'):
        FP_InsRace.to_csv(fpr_npr_path_2_group_intersection +
                          "run_"+str(seed)+"FP_InsRace.csv")

    if (category_name1 == 'insurance') & (category_name2 == 'age_decile'):
        FP_InsAge.to_csv(fpr_npr_path_2_group_intersection +
                         "run_"+str(seed)+"FP_InsAge.csv")

    if (category_name1 == 'race') & (category_name2 == 'age_decile'):
        FP_RaceAge.to_csv(fpr_npr_path_2_group_intersection +
                          "run_"+str(seed)+"FP_RaceAge.csv")

    if (category_name1 == 'gender') & (category_name2 == 'age_decile'):
        FP_AgeSex.to_csv(fpr_npr_path_2_group_intersection +
                         "run_"+str(seed)+"FP_AgeSex.csv")


def Three_Group_FPR(df, diseases, category1, category_name1, category2,
                                  category_name2, category3, category_name3, seed=19, 
                                  three_group_intersection_path="default"):
    if (category_name1 == 'insurance') & (category_name2 == 'gender') & (category_name3 == 'race'):

        FP_RaceInsSex = pd.DataFrame(category3, columns=["race"])

    if (category_name1 == 'insurance') & (category_name2 == 'gender') & (category_name3 == 'age_decile'):
        FP_AgeInsSex = pd.DataFrame(category3, columns=["Age"])

    if (category_name1 == 'insurance') & (category_name2 == 'age_decile') & (category_name3 == 'race'):
        FP_RaceAgeIns = pd.DataFrame(category3, columns=["race"])

    if (category_name1 == 'gender') & (category_name2 == 'age_decile') & (category_name3 == 'race'):
        FP_RaceAgeSex = pd.DataFrame(category3, columns=["race"])

    print("==================================== Calculating FP in vector embedded mimic cxr====================================")

    i = 0

    for c1 in range(len(category1)):

        # FPR_list = []

        for c2 in range(len(category2)):
            FPR_list = []

            for c3 in range(len(category3)):

                for d in range(len(diseases)):

                    pred_disease = "bi_" + diseases[d]
                    gt_fp = df.loc[((df[diseases[d]] == 0) & (df[category_name1] == category1[c1]) &
                                    (df[category_name2] == category2[c2]) & (df[category_name3] == category3[c3])), :]

                    gt_fn = df.loc[((df[diseases[d]] == 1) & (df[category_name1] == category1[c1]) &
                                    (df[category_name2] == category2[c2]) & (df[category_name3] == category3[c3])), :]

                    pred_fp = df.loc[((df[pred_disease] == 1) & (df[diseases[d]] == 0) & (df[category_name1] == category1[c1]) &
                                      (df[category_name2] == category2[c2]) & (df[category_name3] == category3[c3])), :]

                    pred_fn = df.loc[((df[pred_disease] == 0) & (df[diseases[d]] == 1) & (df[category_name1] == category1[c1]) &
                                      (df[category_name2] == category2[c2]) & (df[category_name3] == category3[c3])), :]

                    if len(gt_fp) != 0:
                        FPR = len(pred_fp) / len(gt_fp)
                        print(len(pred_fp), '--', len(gt_fp))
                        print("False Positive Rate in " + category3[c3] + "/" + category1[c1] +
                              " and "+category2[c2] + " for " + diseases[d] + " is: " + str(FPR))

                    else:
                        FPR = np.NaN
                        # print("False Positive Rate in " + category1[c1] +"/" + category2[c2] + " and "+category3[c3]+" for " + diseases[d] + " is: N\A")

                        print(
                            '=======================================================================================================')

                FPR_list.append(round(FPR, 3))
                print(FPR_list)
                # break
            # break

            if (category_name1 == 'insurance') & (category_name2 == 'gender') & (category_name3 == 'race'):

                if i == 0:
                    fpr_male_medicare = pd.DataFrame(
                        FPR_list, columns=["FPR_M_Medicare"])
                    FP_RaceInsSex = pd.concat(
                        [FP_RaceInsSex, fpr_male_medicare.reindex(FP_RaceInsSex.index)], axis=1)

                if i == 1:
                    fpr_F_medicare = pd.DataFrame(
                        FPR_list, columns=["FPR_F_Medicare"])
                    FP_RaceInsSex = pd.concat(
                        [FP_RaceInsSex, fpr_F_medicare.reindex(FP_RaceInsSex.index)], axis=1)

                if i == 2:
                    fpr_male_other = pd.DataFrame(
                        FPR_list, columns=["FPR_M_Other"])
                    FP_RaceInsSex = pd.concat(
                        [FP_RaceInsSex, fpr_male_other.reindex(FP_RaceInsSex.index)], axis=1)

                if i == 3:
                    fpr_female_other = pd.DataFrame(
                        FPR_list, columns=["FPR_F_other"])
                    FP_RaceInsSex = pd.concat(
                        [FP_RaceInsSex, fpr_female_other.reindex(FP_RaceInsSex.index)], axis=1)

                if i == 4:
                    fpr_male_medicaid = pd.DataFrame(
                        FPR_list, columns=["FPR_M_Medicaid"])
                    FP_RaceInsSex = pd.concat(
                        [FP_RaceInsSex, fpr_male_medicaid.reindex(FP_RaceInsSex.index)], axis=1)

                if i == 5:
                    fpr_female_medicaid = pd.DataFrame(
                        FPR_list, columns=["FPR_F_Medicaid"])
                    FP_RaceInsSex = pd.concat(
                        [FP_RaceInsSex, fpr_female_medicaid.reindex(FP_RaceInsSex.index)], axis=1)

            if (category_name1 == 'insurance') & (category_name2 == 'gender') & (category_name3 == 'age_decile'):

                if i == 0:
                    fpr_male_medicare = pd.DataFrame(
                        FPR_list, columns=["FPR_M_Medicare"])
                    FP_AgeInsSex = pd.concat(
                        [FP_AgeInsSex, fpr_male_medicare.reindex(FP_AgeInsSex.index)], axis=1)

                if i == 1:
                    fpr_F_medicare = pd.DataFrame(
                        FPR_list, columns=["FPR_F_Medicare"])
                    FP_AgeInsSex = pd.concat(
                        [FP_AgeInsSex, fpr_F_medicare.reindex(FP_AgeInsSex.index)], axis=1)

                if i == 2:
                    fpr_male_other = pd.DataFrame(
                        FPR_list, columns=["FPR_M_Other"])
                    FP_AgeInsSex = pd.concat(
                        [FP_AgeInsSex, fpr_male_other.reindex(FP_AgeInsSex.index)], axis=1)

                if i == 3:
                    fpr_female_other = pd.DataFrame(
                        FPR_list, columns=["FPR_F_other"])
                    FP_AgeInsSex = pd.concat(
                        [FP_AgeInsSex, fpr_female_other.reindex(FP_AgeInsSex.index)], axis=1)

                if i == 4:
                    fpr_male_medicaid = pd.DataFrame(
                        FPR_list, columns=["FPR_M_Medicaid"])
                    FP_AgeInsSex = pd.concat(
                        [FP_AgeInsSex, fpr_male_medicaid.reindex(FP_AgeInsSex.index)], axis=1)

                if i == 5:
                    fpr_female_medicaid = pd.DataFrame(
                        FPR_list, columns=["FPR_F_Medicaid"])
                    FP_AgeInsSex = pd.concat(
                        [FP_AgeInsSex, fpr_female_medicaid.reindex(FP_AgeInsSex.index)], axis=1)

            if (category_name1 == 'insurance') & (category_name2 == 'age_decile') & (category_name3 == 'race'):

                if i == 0:
                    fpr_60_medicare = pd.DataFrame(
                        FPR_list, columns=["FPR_60_80_Medicare"])
                    FP_RaceAgeIns = pd.concat(
                        [FP_RaceAgeIns, fpr_60_medicare.reindex(FP_RaceAgeIns.index)], axis=1)

                if i == 1:
                    fpr_40_medicare = pd.DataFrame(
                        FPR_list, columns=["FPR_40_60_Medicare"])
                    FP_RaceAgeIns = pd.concat(
                        [FP_RaceAgeIns, fpr_40_medicare.reindex(FP_RaceAgeIns.index)], axis=1)

                if i == 2:
                    fpr_20_medicare = pd.DataFrame(
                        FPR_list, columns=["FPR_20_40_Medicare"])
                    FP_RaceAgeIns = pd.concat(
                        [FP_RaceAgeIns, fpr_20_medicare.reindex(FP_RaceAgeIns.index)], axis=1)

                if i == 3:
                    fpr_80_medicare = pd.DataFrame(
                        FPR_list, columns=["FPR_80+_Medicare"])
                    FP_RaceAgeIns = pd.concat(
                        [FP_RaceAgeIns, fpr_80_medicare.reindex(FP_RaceAgeIns.index)], axis=1)

                if i == 4:
                    fpr_0_medicare = pd.DataFrame(
                        FPR_list, columns=["FPR_0_20_Medicare"])
                    FP_RaceAgeIns = pd.concat(
                        [FP_RaceAgeIns, fpr_0_medicare.reindex(FP_RaceAgeIns.index)], axis=1)

                if i == 5:
                    fpr_60_other = pd.DataFrame(
                        FPR_list, columns=["FPR_60_80_Other"])
                    FP_RaceAgeIns = pd.concat(
                        [FP_RaceAgeIns, fpr_60_other.reindex(FP_RaceAgeIns.index)], axis=1)

                if i == 6:
                    fpr_40_other = pd.DataFrame(
                        FPR_list, columns=["FPR_40_60_Other"])
                    FP_RaceAgeIns = pd.concat(
                        [FP_RaceAgeIns, fpr_40_other.reindex(FP_RaceAgeIns.index)], axis=1)

                if i == 7:
                    fpr_20_other = pd.DataFrame(
                        FPR_list, columns=["FPR_20_40_Other"])
                    FP_RaceAgeIns = pd.concat(
                        [FP_RaceAgeIns, fpr_20_other.reindex(FP_RaceAgeIns.index)], axis=1)

                if i == 8:
                    fpr_80_other = pd.DataFrame(
                        FPR_list, columns=["FPR_80+_Other"])
                    FP_RaceAgeIns = pd.concat(
                        [FP_RaceAgeIns, fpr_80_other.reindex(FP_RaceAgeIns.index)], axis=1)

                if i == 9:
                    fpr_0_other = pd.DataFrame(
                        FPR_list, columns=["FPR_0_20_Other"])
                    FP_RaceAgeIns = pd.concat(
                        [FP_RaceAgeIns, fpr_0_other.reindex(FP_RaceAgeIns.index)], axis=1)

                if i == 10:
                    fpr_60_medicaid = pd.DataFrame(
                        FPR_list, columns=["FPR_60_80_medicaid"])
                    FP_RaceAgeIns = pd.concat(
                        [FP_RaceAgeIns, fpr_60_medicaid.reindex(FP_RaceAgeIns.index)], axis=1)

                if i == 11:
                    fpr_40_medicaid = pd.DataFrame(
                        FPR_list, columns=["FPR_40_60_medicaid"])
                    FP_RaceAgeIns = pd.concat(
                        [FP_RaceAgeIns, fpr_40_medicaid.reindex(FP_RaceAgeIns.index)], axis=1)

                if i == 12:
                    fpr_20_medicaid = pd.DataFrame(
                        FPR_list, columns=["FPR_20_40_medicaid"])
                    FP_RaceAgeIns = pd.concat(
                        [FP_RaceAgeIns, fpr_20_medicaid.reindex(FP_RaceAgeIns.index)], axis=1)

                if i == 13:
                    fpr_80_medicaid = pd.DataFrame(
                        FPR_list, columns=["FPR_80+_medicaid"])
                    FP_RaceAgeIns = pd.concat(
                        [FP_RaceAgeIns, fpr_80_medicaid.reindex(FP_RaceAgeIns.index)], axis=1)

                if i == 14:
                    fpr_0_medicaid = pd.DataFrame(
                        FPR_list, columns=["FPR_0_20_medicaid"])
                    FP_RaceAgeIns = pd.concat(
                        [FP_RaceAgeIns, fpr_0_medicaid.reindex(FP_RaceAgeIns.index)], axis=1)

            if (category_name1 == 'gender') & (category_name2 == 'age_decile') & (category_name3 == 'race'):

                if i == 0:
                    fpr_60_male = pd.DataFrame(
                        FPR_list, columns=["FPR_60_80_M"])
                    FP_RaceAgeSex = pd.concat(
                        [FP_RaceAgeSex, fpr_60_male.reindex(FP_RaceAgeSex.index)], axis=1)

                if i == 1:
                    fpr_40_male = pd.DataFrame(
                        FPR_list, columns=["FPR_40_60_M"])
                    FP_RaceAgeSex = pd.concat(
                        [FP_RaceAgeSex, fpr_40_male.reindex(FP_RaceAgeSex.index)], axis=1)

                if i == 2:
                    fpr_20_male = pd.DataFrame(
                        FPR_list, columns=["FPR_20_40_M"])
                    FP_RaceAgeSex = pd.concat(
                        [FP_RaceAgeSex, fpr_20_male.reindex(FP_RaceAgeSex.index)], axis=1)

                if i == 3:
                    fpr_80_male = pd.DataFrame(FPR_list, columns=["FPR_80+_M"])
                    FP_RaceAgeSex = pd.concat(
                        [FP_RaceAgeSex, fpr_80_male.reindex(FP_RaceAgeSex.index)], axis=1)

                if i == 4:
                    fpr_0_male = pd.DataFrame(FPR_list, columns=["FPR_0_20_M"])
                    FP_RaceAgeSex = pd.concat(
                        [FP_RaceAgeSex, fpr_0_male.reindex(FP_RaceAgeSex.index)], axis=1)

                if i == 5:
                    fpr_60_female = pd.DataFrame(
                        FPR_list, columns=["FPR_60_80_F"])
                    FP_RaceAgeSex = pd.concat(
                        [FP_RaceAgeSex, fpr_60_female.reindex(FP_RaceAgeSex.index)], axis=1)

                if i == 6:
                    fpr_40_female = pd.DataFrame(
                        FPR_list, columns=["FPR_40_60_F"])
                    FP_RaceAgeSex = pd.concat(
                        [FP_RaceAgeSex, fpr_40_female.reindex(FP_RaceAgeSex.index)], axis=1)

                if i == 7:
                    fpr_20_female = pd.DataFrame(
                        FPR_list, columns=["FPR_20_40_F"])
                    FP_RaceAgeSex = pd.concat(
                        [FP_RaceAgeSex, fpr_20_female.reindex(FP_RaceAgeSex.index)], axis=1)

                if i == 8:
                    fpr_80_female = pd.DataFrame(
                        FPR_list, columns=["FPR_80+_F"])
                    FP_RaceAgeSex = pd.concat(
                        [FP_RaceAgeSex, fpr_80_female.reindex(FP_RaceAgeSex.index)], axis=1)

                if i == 9:
                    fpr_0_female = pd.DataFrame(
                        FPR_list, columns=["FPR_0_20_F"])
                    FP_RaceAgeSex = pd.concat(
                        [FP_RaceAgeSex, fpr_0_female.reindex(FP_RaceAgeSex.index)], axis=1)

            i += 1

    if (category_name1 == 'insurance') & (category_name2 == 'gender') & (category_name3 == 'race'):
        FP_RaceInsSex.to_csv(three_group_intersection_path +
                             "run_"+str(seed)+"FP_RaceInsSex.csv")

    if (category_name1 == 'insurance') & (category_name2 == 'gender') & (category_name3 == 'age_decile'):
        FP_AgeInsSex.to_csv(three_group_intersection_path +
                            "run_"+str(seed)+"FP_AgeInsSex.csv")

    if (category_name1 == 'insurance') & (category_name2 == 'age_decile') & (category_name3 == 'race'):
        FP_RaceAgeIns.to_csv(three_group_intersection_path +
                             "run_"+str(seed)+"FP_RaceAgeIns.csv")

    if (category_name1 == 'gender') & (category_name2 == 'age_decile') & (category_name3 == 'race'):
        FP_RaceAgeSex.to_csv(three_group_intersection_path +
                             "run_"+str(seed)+"FP_RaceAgeSex.csv")


def main():

    diseases = get_diseases()
    diseases_abbr = get_diseases_abbr()

    patient_groups = get_patient_groups()

    gender = patient_groups["sex"]
    age_decile = patient_groups["age"]
    race = patient_groups["race"]
    insurance = patient_groups["insurance"]

    factor = [gender, age_decile, race, insurance]

    factor_str = ['gender', 'age_decile', 'race', 'insurance']

    seeds = get_seeds()

    for seed in seeds:
        np.random.seed(seed)
        python_random.seed(seed)

        prediction_results_path = "./Prediction_results/"

        fpr_npr_path = "./FPR/SubGroup_FPR/"
        os.makedirs(os.path.dirname(fpr_npr_path), exist_ok=True)

        df = pd.read_csv(f'{prediction_results_path}bipred_{seed}.csv').rename(
            columns={'age decile': 'age_decile'})

        '''' FPR Calculation '''

        fpr_gap = True

        for i in range(len(factor)):
            Subgroup_FPR(df, [str(diseases[13])], factor[i],
                           factor_str[i], seed, fpr_npr_path)

        '''' Two group FPR '''
        fpr_npr_path_2_group_intersection = "./FPR/Two_Group_Intersection_FPR/"
        os.makedirs(os.path.dirname(
            fpr_npr_path_2_group_intersection), exist_ok=True)

        Two_Group_FPR(
            df, [str(diseases[13])], gender, 'gender', race, 'race', seed, fpr_npr_path_2_group_intersection)
        Two_Group_FPR(df, [str(diseases[13])], gender, 'gender', age_decile,
                                    'age_decile', seed, fpr_npr_path_2_group_intersection)
        Two_Group_FPR(df, [str(diseases[13])], race, 'race', age_decile,
                                    'age_decile', seed, fpr_npr_path_2_group_intersection)
        Two_Group_FPR(df, [str(diseases[13])], insurance, 'insurance',
                                    age_decile, 'age_decile', seed, fpr_npr_path_2_group_intersection)
        Two_Group_FPR(df, [str(diseases[13])], insurance, 'insurance',
                                    race, 'race', seed, fpr_npr_path_2_group_intersection)
        Two_Group_FPR(df, [str(diseases[13])], gender, 'gender',
                                    insurance, 'insurance', seed, fpr_npr_path_2_group_intersection)

        '''' Three group FPR '''
        fpr_npr_path_3_group_intersection = "./FPR/Three_Group_Intersection_FPR/"
        os.makedirs(os.path.dirname(
            fpr_npr_path_3_group_intersection), exist_ok=True)

        Three_Group_FPR(df, [str(diseases[13])], gender, 'gender', age_decile,
                                      'age_decile', race, 'race', seed, fpr_npr_path_3_group_intersection)
        Three_Group_FPR(df, [str(diseases[13])], insurance, 'insurance', gender,
                                      'gender', race, 'race', seed, fpr_npr_path_3_group_intersection)
        Three_Group_FPR(df, [str(diseases[13])], insurance, 'insurance', gender,
                                      'gender', age_decile, 'age_decile', seed, fpr_npr_path_3_group_intersection)
        Three_Group_FPR(df, [str(diseases[13])], insurance, 'insurance', age_decile,
                                      'age_decile', race, 'race', seed, fpr_npr_path_3_group_intersection)

    print(f'SEED : {seed}')


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()
