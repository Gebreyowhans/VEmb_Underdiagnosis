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

number_of_runs = 5
signficance_level = 1.96


# FPR
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


def FiveRunSubgroup(factors, df_in, df_out):
    fpr = []
    fnr = []
    percent = []
    ci_fpr = []
    ci_fnr = []
    confI = signficance_level * df_in.loc['std'] / np.sqrt(number_of_runs)

    for fact in factors:
        percent.append(round(df_in.loc['mean']['#'+fact], 3))
        fpr.append(round(df_in.loc['mean']['FPR_'+fact], 3))
        ci_fpr.append(round(confI.loc['FPR_'+fact], 3))

    df_out['#'] = percent
    df_out['FPR'] = fpr
    df_out['CI_FPR'] = ci_fpr

    return df_out


def FiveRunTwoGroup(factors, output_df, df):
    for factor in factors:
        fpr = round(df[factor]['mean'], 3)
        confI = round(signficance_level * df[factor]["std"] / np.sqrt(5), 3)
        output_df[factor] = pd.DataFrame(fpr.values.tolist(), columns=[factor])
        output_df['CI_' +
                  factor] = pd.DataFrame(confI.values.tolist(), columns=['CI_'+factor])

    return output_df


def FP_NF_CheXpert(df, diseases, category, category_name, seed=19):

    # return FPR and FNR per subgroup and the unber of patients with 0 No-finding in test set.
    FP_total = []
    percentage_total = []

    if category_name == 'race':
        FPR_race = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'gender':
        FPR_sex = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'age_decile':
        FPR_age = pd.DataFrame(diseases, columns=["diseases"])

    print("FP in MIMIC====================================")

    for c in category:
        FP_y = []
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

            FP_total.append(FP_y)
            percentage_total.append(percentage_y)

        # print("False Positive Rate in " + category[c] + " for " + diseases[d] + " is: " + str(FPR))

    for i in range(len(FP_total)):

        if category_name == 'gender':
            if i == 0:
                Perc_S = pd.DataFrame(percentage_total[i], columns=["#M"])
                FPR_sex = pd.concat(
                    [FPR_sex, Perc_S.reindex(FPR_sex.index)], axis=1)

                FPR_S = pd.DataFrame(FP_total[i], columns=["FPR_M"])
                FPR_sex = pd.concat(
                    [FPR_sex, FPR_S.reindex(FPR_sex.index)], axis=1)

            if i == 1:
                Perc_S = pd.DataFrame(percentage_total[i], columns=["#F"])
                FPR_sex = pd.concat(
                    [FPR_sex, Perc_S.reindex(FPR_sex.index)], axis=1)

                FPR_S = pd.DataFrame(FP_total[i], columns=["FPR_F"])
                FPR_sex = pd.concat(
                    [FPR_sex, FPR_S.reindex(FPR_sex.index)], axis=1)

                print(f'============FPR for gender done ==============')

                return FPR_sex

        if category_name == 'age_decile':

            if i == 0:
                Perc_A = pd.DataFrame(percentage_total[i], columns=["#0-20"])
                FPR_age = pd.concat(
                    [FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)

                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_0-20"])
                FPR_age = pd.concat(
                    [FPR_age, FPR_A.reindex(FPR_age.index)], axis=1)

            if i == 1:

                Perc_A = pd.DataFrame(percentage_total[i], columns=["#20-40"])
                FPR_age = pd.concat(
                    [FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)

                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_20-40"])
                FPR_age = pd.concat(
                    [FPR_age, FPR_A.reindex(FPR_age.index)], axis=1)

            if i == 2:

                Perc_A = pd.DataFrame(percentage_total[i], columns=["#40-60"])
                FPR_age = pd.concat(
                    [FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)

                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_40-60"])
                FPR_age = pd.concat(
                    [FPR_age, FPR_A.reindex(FPR_age.index)], axis=1)

            if i == 3:
                Perc_A = pd.DataFrame(percentage_total[i], columns=["#60-80"])
                FPR_age = pd.concat(
                    [FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)
                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_60-80"])
                FPR_age = pd.concat(
                    [FPR_age, FPR_A.reindex(FPR_age.index)], axis=1)

            if i == 4:

                Perc_A = pd.DataFrame(percentage_total[i], columns=["#80+"])
                FPR_age = pd.concat(
                    [FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)

                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_80+"])
                FPR_age = pd.concat(
                    [FPR_age, FPR_A.reindex(FPR_age.index)], axis=1)

                # age_FPR[f"Seed_{seed}"] = FPR_age
                print(f'============FPR for age done ==============')
                return FPR_age

        if category_name == 'race':

            if i == 0:

                Perc_A = pd.DataFrame(percentage_total[i], columns=["#White"])
                FPR_race = pd.concat(
                    [FPR_race, Perc_A.reindex(FPR_race.index)], axis=1)

                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_White"])
                FPR_race = pd.concat(
                    [FPR_race, FPR_A.reindex(FPR_race.index)], axis=1)

            if i == 1:
                Perc_A = pd.DataFrame(percentage_total[i], columns=["#Black"])
                FPR_race = pd.concat(
                    [FPR_race, Perc_A.reindex(FPR_race.index)], axis=1)

                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_Black"])
                FPR_race = pd.concat(
                    [FPR_race, FPR_A.reindex(FPR_race.index)], axis=1)

            if i == 2:
                Perc_A = pd.DataFrame(percentage_total[i], columns=["#Hisp"])
                FPR_race = pd.concat(
                    [FPR_race, Perc_A.reindex(FPR_race.index)], axis=1)

                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_Hisp"])
                FPR_race = pd.concat(
                    [FPR_race, FPR_A.reindex(FPR_race.index)], axis=1)

            if i == 3:
                Perc_A = pd.DataFrame(percentage_total[i], columns=["#Other"])
                FPR_race = pd.concat(
                    [FPR_race, Perc_A.reindex(FPR_race.index)], axis=1)

                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_Other"])
                FPR_race = pd.concat(
                    [FPR_race, FPR_A.reindex(FPR_race.index)], axis=1)

            if i == 4:
                Perc_A = pd.DataFrame(percentage_total[i], columns=["#Asian"])
                FPR_race = pd.concat(
                    [FPR_race, Perc_A.reindex(FPR_race.index)], axis=1)

                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_Asian"])
                FPR_race = pd.concat(
                    [FPR_race, FPR_A.reindex(FPR_race.index)], axis=1)

            if i == 5:
                Perc_A = pd.DataFrame(
                    percentage_total[i], columns=["#American"])
                FPR_race = pd.concat(
                    [FPR_race, Perc_A.reindex(FPR_race.index)], axis=1)

                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_American"])
                FPR_race = pd.concat(
                    [FPR_race, FPR_A.reindex(FPR_race.index)], axis=1)

                # race_FPR[f"Seed_{seed}"] = FPR_race

                print(f'============FPR for race done ==============')
                return FPR_race


def FP_NF_CheXpert_Inter(df, diseases, category1, category_name1, category2, category_name2, seed=19):

    if (category_name1 == 'gender') & (category_name2 == 'race'):
        FP_RaceSex = pd.DataFrame(category2, columns=["race"])

    if (category_name1 == 'gender') & (category_name2 == 'age_decile'):
        FP_AgeSex = pd.DataFrame(category2, columns=["Age"])

    if (category_name1 == 'race') & (category_name2 == 'age_decile'):
        FP_RaceAge = pd.DataFrame(category2, columns=["age"])

    print("==================================== Calculating FP in vector embedded CXP====================================")

    i = 0

    for c1 in range(len(category1)):
        FPR_list = []

        for c2 in range(len(category2)):

            for d in range(len(diseases)):

                pred_disease = "bi_" + diseases[d]

                gt_fp = df.loc[((df[diseases[d]] == 0) & (df[category_name1] == category1[c1]) &
                                (df[category_name2] == category2[c2])), :]

                pred_fp = df.loc[((df[pred_disease] == 1) & (df[diseases[d]] == 0) & (df[category_name1] == category1[c1]) &
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

                return FP_AgeSex

        if (category_name1 == 'gender') & (category_name2 == 'race'):

            if i == 0:
                FPR_SR = pd.DataFrame(FPR_list, columns=["FPR_M"])
                FP_RaceSex = pd.concat(
                    [FP_RaceSex, FPR_SR.reindex(FP_RaceSex.index)], axis=1)

            if i == 1:
                FPR_SR = pd.DataFrame(FPR_list, columns=["FPR_F"])
                FP_RaceSex = pd.concat(
                    [FP_RaceSex, FPR_SR.reindex(FP_RaceSex.index)], axis=1)

                return FP_RaceSex

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

                return FP_RaceAge

        i += 1
