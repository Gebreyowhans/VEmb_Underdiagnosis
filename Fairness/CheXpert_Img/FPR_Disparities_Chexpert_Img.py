import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
# ensures true fonte types are embedded in the pdf
plt.rcParams['pdf.fonttype'] = 42
# Applies the same setting for post script output
plt.rcParams['ps.fonttype'] = 42


def get_Sex_FPR_Disparities(basepath, diseases, diseases_abbr, number_of_runs, significance_level,
                            height, font_size, rotation_degree):
    seed_19_sex = pd.read_csv(
        basepath+"Run_seed32_FPR_GAP_sex.csv", index_col=0)
    seed_31_sex = pd.read_csv(
        basepath+"Run_seed40_FPR_GAP_sex.csv", index_col=0)
    seed_38_sex = pd.read_csv(
        basepath+"Run_seed56_FPR_GAP_sex.csv", index_col=0)
    seed_47_sex = pd.read_csv(
        basepath+"Run_seed60_FPR_GAP_sex.csv", index_col=0)
    seed_77_sex = pd.read_csv(
        basepath+"Run_seed90_FPR_GAP_sex.csv", index_col=0)

    diseases = [disease for disease in diseases if disease != 'No Finding']

    sex_dataframes = [seed_19_sex, seed_31_sex,
                      seed_38_sex, seed_47_sex, seed_77_sex]
    result_sex = pd.concat(sex_dataframes)

    # group by diseases and calculate the mean and std for each disease
    result_sex_grouped = result_sex.groupby("diseases")
    result_sex_grouped_Stat = result_sex_grouped.describe()

    sex_df_male_mean = result_sex_grouped_Stat['#M']['mean']

    # get FNR GAPS for each disease subgroup
    sex_df_male_gap = result_sex_grouped_Stat['Gap_M']["mean"]
    sex_df_male_ci = (significance_level *
                      result_sex_grouped_Stat['Gap_M']["std"]) / np.sqrt(number_of_runs)

    sex_df_female_mean = result_sex_grouped_Stat['#F']['mean']
    sex_df_female_gap = result_sex_grouped_Stat['Gap_F']["mean"]
    sex_df_female_ci = (significance_level *
                        result_sex_grouped_Stat['Gap_F']["std"]) / np.sqrt(number_of_runs)

    percent_male_list = []
    ci_male_list = []
    gap_male_mean_list = []

    diseases_abbr_list = []
    distance_list = []

    percent_female_list = []
    ci_female_list = []
    percent_female_list = []
    gap_femal_mean_list = []

    for disease in diseases:
        percent_male_list.append(sex_df_male_mean[disease])
        gap_male_mean_list.append(sex_df_male_gap[disease])
        ci_male_list.append(sex_df_male_ci[disease])

        percent_female_list.append(sex_df_female_mean[disease])
        gap_femal_mean_list.append(sex_df_female_gap[disease])
        ci_female_list.append(sex_df_female_ci[disease])

        distance_list.append(np.absolute(
            sex_df_female_gap[disease] - sex_df_male_gap[disease]))
        diseases_abbr_list.append(diseases_abbr[disease])

    d = {'diseases': diseases, 'diseases_abbr': diseases_abbr_list, 'Distance': distance_list,
         "#M": percent_male_list, 'Gap_M_mean': gap_male_mean_list, 'CI_M': ci_male_list,
         "#F": percent_female_list, 'Gap_F_mean': gap_femal_mean_list, 'CI_F': ci_female_list
         }

    sex_fpr_disp_df = pd.DataFrame(d)
    sex_fpr_disp_df = sex_fpr_disp_df.sort_values(by='Distance')

    sex_fpr_disp_df.to_csv(basepath+"sex_fpr_disp.csv", index=False)

    # Plotting the results
    plot_Sex_FPR_GAPS(basepath, sex_fpr_disp_df, height,
                      font_size, rotation_degree)


def plot_Sex_FPR_GAPS(basepath, sex_fpr_disp_df,
                      height, font_size, rotation_degree):

    plt.ioff()

    plt.rcParams.update({'font.size': font_size})

    plt.figure(figsize=(16, height))
    plt.scatter(sex_fpr_disp_df['diseases_abbr'], sex_fpr_disp_df['Gap_M_mean'],
                s=np.multiply(sex_fpr_disp_df['#M'], 0.05), marker='o', color='blue', label="Male")
    plt.errorbar(sex_fpr_disp_df['diseases_abbr'], sex_fpr_disp_df['Gap_M_mean'],
                 yerr=sex_fpr_disp_df['CI_M'], fmt='o', mfc='blue')
    plt.scatter(sex_fpr_disp_df['diseases_abbr'], sex_fpr_disp_df['Gap_F_mean'],
                s=np.multiply(sex_fpr_disp_df['#F'], 0.05), marker='o', color='red', label="Female")
    plt.errorbar(sex_fpr_disp_df['diseases_abbr'], sex_fpr_disp_df['Gap_F_mean'],
                 yerr=sex_fpr_disp_df['CI_F'], fmt='o', mfc='red')

    plt.xticks(rotation=rotation_degree, fontsize=font_size,
               fontname='Times New Roman')
    plt.yticks(fontsize=font_size, fontname='Times New Roman')
    plt.ylabel("FPR SEX DISPARITY", fontsize=font_size,
               fontname='Times New Roman')
    plt.legend()
    plt.grid(True)
    plt.savefig(basepath+"FPR_Dis_SEX.pdf")
    plt.close()


def get_Age_FPR_Disparities(basepath, diseases, diseases_abbr, number_of_runs, significance_level,
                            height, font_size, rotation_degree):

    seed_19_age = pd.read_csv(
        basepath+"Run_seed32_FPR_GAP_Age.csv", index_col=0)
    seed_31_age = pd.read_csv(
        basepath+"Run_seed40_FPR_GAP_Age.csv", index_col=0)
    seed_38_age = pd.read_csv(
        basepath+"Run_seed56_FPR_GAP_Age.csv", index_col=0)
    seed_47_age = pd.read_csv(
        basepath+"Run_seed60_FPR_GAP_Age.csv", index_col=0)
    seed_77_age = pd.read_csv(
        basepath+"Run_seed90_FPR_GAP_Age.csv", index_col=0)

    diseases = [disease for disease in diseases if disease != 'No Finding']

    # group by diseases and calculate the mean and std for each disease
    age_dataframes = [seed_19_age, seed_31_age,
                      seed_38_age, seed_47_age, seed_77_age]
    result_age_df = pd.concat(age_dataframes)
    result_age_grouped = result_age_df.groupby("diseases")
    result_age_grouped_stat = result_age_grouped.describe()

    # get FNR GAPS for each disease subgroup
    age_df_40_mean = result_age_grouped_stat['#40-60']['mean']
    age_df_40_gap = result_age_grouped_stat['Gap_40-60']["mean"]
    age_df_40_ci = (significance_level *
                    result_age_grouped_stat['Gap_40-60']["std"])/np.sqrt(number_of_runs)

    age_df_60_mean = result_age_grouped_stat['#60-80']['mean']
    age_df_60_gap = result_age_grouped_stat['Gap_60-80']["mean"]
    age_df_60_ci = (significance_level *
                    result_age_grouped_stat['Gap_60-80']["std"])/np.sqrt(number_of_runs)

    age_df_20_mean = result_age_grouped_stat['#20-40']['mean']
    age_df_20_gap = result_age_grouped_stat['Gap_20-40']["mean"]
    age_df_20_ci = (significance_level *
                    result_age_grouped_stat['Gap_20-40']["std"])/np.sqrt(number_of_runs)

    age_df_80_mean = result_age_grouped_stat['#80-']['mean']
    age_df_80_gap = result_age_grouped_stat['Gap_80-']["mean"]
    age_df_80_ci = (significance_level *
                    result_age_grouped_stat['Gap_80-']["std"])/np.sqrt(number_of_runs)

    age_df_0_mean = result_age_grouped_stat['#0-20']['mean']
    age_df_0_gap = result_age_grouped_stat['Gap_0-20']["mean"]
    age_df_0_ci = (significance_level *
                   result_age_grouped_stat['Gap_0-20']["std"])/np.sqrt(number_of_runs)

    prcent_40_list = []
    ci_40_list = []
    gap_40_mean_list = []
    diseases_abbr_list = []
    distance_list = []

    prcent_60_list = []
    ci_60_list = []
    prcent_60_list = []
    gap_60_mean_list = []

    prcent_20_list = []
    ci_20_list = []
    prcent_20_list = []
    gap_20_mean_list = []

    prcent_80_list = []
    ci_80_list = []
    prcent_80_list = []
    gap_80_mean_list = []

    prcent_0_list = []
    ci_0_list = []
    prcent_0_list = []
    gap_0_mean_list = []
    mean_list = []

    for disease in diseases:

        mean_list = []
        cleaned_mean_gap_list = []
        prcent_40_list.append(age_df_40_mean[disease])
        gap_40_mean_list.append(age_df_40_gap[disease])
        ci_40_list.append(age_df_40_ci[disease])
        mean_list.append(age_df_40_gap[disease])

        prcent_60_list.append(age_df_60_mean[disease])
        gap_60_mean_list.append(age_df_60_gap[disease])
        ci_60_list.append(age_df_60_ci[disease])
        mean_list.append(age_df_60_gap[disease])

        prcent_20_list.append(age_df_20_mean[disease])
        gap_20_mean_list.append(age_df_20_gap[disease])
        ci_20_list.append(age_df_20_ci[disease])
        mean_list.append(age_df_20_gap[disease])

        prcent_80_list.append(age_df_80_mean[disease])
        gap_80_mean_list.append(age_df_80_gap[disease])
        ci_80_list.append(age_df_80_ci[disease])
        mean_list.append(age_df_80_gap[disease])

        prcent_0_list.append(age_df_0_mean[disease])
        gap_0_mean_list.append(age_df_0_gap[disease])
        ci_0_list.append(age_df_0_ci[disease])
        mean_list.append(age_df_0_gap[disease])

        cleaned_mean_gap_list = [x for x in mean_list if str(x) != 'nan']
        distance_list.append(np.max(cleaned_mean_gap_list) -
                             np.min(cleaned_mean_gap_list))
        diseases_abbr_list.append(diseases_abbr[disease])

    d = {'diseases': diseases, 'diseases_abbr': diseases_abbr_list, 'Distance': distance_list,
         "#40-60": prcent_40_list, 'Gap_40-60_mean': gap_40_mean_list, 'CI_40-60': ci_40_list,
         "#60-80": prcent_60_list, 'Gap_60-80_mean': gap_60_mean_list, 'CI_60-80': ci_60_list,
         "#20-40": prcent_20_list, 'Gap_20-40_mean': gap_20_mean_list, 'CI_20-40': ci_20_list,
         "#80-": prcent_80_list, 'Gap_80-_mean': gap_80_mean_list, 'CI_80-': ci_80_list,
         "#0-20": prcent_0_list, 'Gap_0-20_mean': gap_0_mean_list, 'CI_0-20': ci_0_list
         }

    # Creating a DataFrame from the dictionary
    age_fpr_disp_df = pd.DataFrame(d)

    age_fpr_disp_df = age_fpr_disp_df.sort_values(by='Distance')
    age_fpr_disp_df.to_csv(basepath+"age_fpr_disp.csv", index=False)

    # Plotting the results
    plot_age_FPR_GAPS(basepath, age_fpr_disp_df, height,
                      font_size, rotation_degree)


def plot_age_FPR_GAPS(basepath, age_fpr_disp_df, height, font_size, rotation_degree):

    plt.ioff()
    plt.rcParams.update({'font.size': font_size})
    plt.figure(figsize=(16, height))
    plt.scatter(age_fpr_disp_df['diseases_abbr'], age_fpr_disp_df['Gap_60-80_mean'],
                s=np.multiply(age_fpr_disp_df['#60-80'], 0.5), marker='o', color='blue', label="60-80")
    plt.errorbar(age_fpr_disp_df['diseases_abbr'], age_fpr_disp_df['Gap_60-80_mean'],
                 yerr=age_fpr_disp_df['CI_60-80'], fmt='o', mfc='blue')
    plt.scatter(age_fpr_disp_df['diseases_abbr'], age_fpr_disp_df['Gap_40-60_mean'],
                s=np.multiply(age_fpr_disp_df['#40-60'], 0.5), marker='o', color='orange', label="40-60")
    plt.errorbar(age_fpr_disp_df['diseases_abbr'], age_fpr_disp_df['Gap_40-60_mean'],
                 yerr=age_fpr_disp_df['CI_40-60'], fmt='o', mfc='orange')
    plt.scatter(age_fpr_disp_df['diseases_abbr'], age_fpr_disp_df['Gap_20-40_mean'],
                s=np.multiply(age_fpr_disp_df['#20-40'], 0.5), marker='o', color='green', label="20-40")
    plt.errorbar(age_fpr_disp_df['diseases_abbr'], age_fpr_disp_df['Gap_20-40_mean'],
                 yerr=age_fpr_disp_df['CI_20-40'], fmt='o', mfc='green')
    plt.scatter(age_fpr_disp_df['diseases_abbr'], age_fpr_disp_df['Gap_80-_mean'],
                s=np.multiply(age_fpr_disp_df['#80-'], 0.5), marker='o', color='red', label="80-")
    plt.errorbar(age_fpr_disp_df['diseases_abbr'], age_fpr_disp_df['Gap_80-_mean'],
                 yerr=age_fpr_disp_df['CI_80-'], fmt='o', mfc='red')
    plt.scatter(age_fpr_disp_df['diseases_abbr'], age_fpr_disp_df['Gap_0-20_mean'],
                s=np.multiply(age_fpr_disp_df['#0-20'], 0.5), marker='o', color='purple', label="0-20")
    plt.errorbar(age_fpr_disp_df['diseases_abbr'], age_fpr_disp_df['Gap_0-20_mean'],
                 yerr=age_fpr_disp_df['CI_0-20'], fmt='o', mfc='purple')

    plt.xticks(rotation=rotation_degree, fontsize=font_size,
               fontname='Times New Roman')
    plt.ylabel("FPR AGE DISPARITY", fontsize=font_size,
               fontname='Times New Roman')
    plt.yticks(fontsize=font_size, fontname='Times New Roman')
    plt.legend()
    plt.grid(True)

    # plt.savefig(basepath+"FNR_Dis_Age.pdf")

    plt.close()


def get_Race_FPR_Disparities(basepath, diseases, diseases_abbr, number_of_runs, significance_level,
                             height, font_size, rotation_degree):

    seed_19_race = pd.read_csv(basepath +
                               "Run_seed32_FPR_GAP_race.csv", index_col=0)
    seed_31_race = pd.read_csv(basepath +
                               "Run_seed40_FPR_GAP_race.csv", index_col=0)
    seed_38_race = pd.read_csv(
        basepath + "Run_seed56_FPR_GAP_race.csv", index_col=0)
    seed_47_race = pd.read_csv(
        basepath + "Run_seed60_FPR_GAP_race.csv", index_col=0)

    seed_77_race = pd.read_csv(
        basepath + "Run_seed90_FPR_GAP_race.csv", index_col=0)

    # group by diseases and calculate the mean and std for each disease
    race_dataframes = [seed_19_race, seed_31_race,
                       seed_38_race, seed_47_race, seed_77_race]
    result_race_df = pd.concat(race_dataframes)

    result_race_grouped = result_race_df.groupby("diseases")
    result_race_grouped_stat = result_race_grouped.describe()

    race_tpr_disp = pd.DataFrame(pd.DataFrame(diseases, columns=["diseases"]))

    # get FNR GAPS for each disease subgroup
    race_df_white_mean = result_race_grouped_stat['#White']['mean']
    race_df_white_gap = result_race_grouped_stat['Gap_White']["mean"]
    race_df_white_ci = (significance_level *
                        result_race_grouped_stat['Gap_White']["std"]) / np.sqrt(number_of_runs)

    race_df_black_mean = result_race_grouped_stat['#Black']['mean']
    race_df_black_gap = result_race_grouped_stat['Gap_Black']["mean"]
    race_df_black_ci = (
        significance_level * result_race_grouped_stat['Gap_Black']["std"]) / np.sqrt(number_of_runs)

    race_df_hisp_mean = result_race_grouped_stat['#Hisp']['mean']
    race_df_hisp_gap = result_race_grouped_stat['Gap_Hisp']["mean"]
    race_df_hisp_ci = (significance_level *
                       result_race_grouped_stat['Gap_Hisp']["std"]) / np.sqrt(number_of_runs)

    race_df_other_mean = result_race_grouped_stat['#Other']['mean']
    race_df_other_gap = result_race_grouped_stat['Gap_Other']["mean"]
    race_df_other_ci = (
        significance_level * result_race_grouped_stat['Gap_Other']["std"]) / np.sqrt(number_of_runs)

    race_df_asian_mean = result_race_grouped_stat['#Asian']['mean']
    race_df_asian_gap = result_race_grouped_stat['Gap_Asian']["mean"]
    race_df_asian_ci = (
        significance_level * result_race_grouped_stat['Gap_Asian']["std"]) / np.sqrt(number_of_runs)

    race_df_american_mean = result_race_grouped_stat['#American']['mean']
    race_df_american_gap = result_race_grouped_stat['Gap_American']["mean"]
    race_df_american_ci = (
        significance_level * result_race_grouped_stat['Gap_American']["std"]) / np.sqrt(number_of_runs)

    percent_asian_list = []
    ci_asian_list = []
    gap_asian_mean_list = []

    ci_american_list = []
    percent_american_list = []
    gap_american_mean_list = []

    percent_white_list = []
    ci_white_list = []
    gap_white_mean_list = []

    percent_black_list = []
    ci_black_list = []
    gap_black_mean_list = []

    percent_hisp_list = []
    ci_hisp_list = []
    gap_hisp_mean_list = []

    percent_other_list = []
    ci_other_list = []
    gap_other_mean_list = []

    diseases_abbr_list = []
    distance_list = []

    for disease in diseases:

        mean_list = []
        percent_black_list.append(race_df_black_mean[disease])
        gap_black_mean_list.append(race_df_black_gap[disease])
        ci_black_list.append(race_df_black_ci[disease])
        mean_list.append(race_df_black_gap[disease])

        percent_hisp_list.append(race_df_hisp_mean[disease])
        gap_hisp_mean_list.append(race_df_hisp_gap[disease])
        ci_hisp_list.append(race_df_hisp_ci[disease])
        mean_list.append(race_df_hisp_gap[disease])

        percent_other_list.append(race_df_other_mean[disease])
        gap_other_mean_list.append(race_df_other_gap[disease])
        ci_other_list.append(race_df_other_ci[disease])
        mean_list.append(race_df_other_gap[disease])

        percent_white_list.append(race_df_white_mean[disease])
        gap_white_mean_list.append(race_df_white_gap[disease])
        ci_white_list.append(race_df_white_ci[disease])
        mean_list.append(race_df_white_gap[disease])

        percent_asian_list.append(race_df_asian_mean[disease])
        gap_asian_mean_list.append(race_df_asian_gap[disease])
        ci_asian_list.append(race_df_asian_ci[disease])
        mean_list.append(race_df_asian_gap[disease])

        percent_american_list.append(race_df_american_mean[disease])
        gap_american_mean_list.append(race_df_american_gap[disease])
        ci_american_list.append(race_df_american_ci[disease])
        mean_list.append(race_df_american_gap[disease])

        cleaned_mean_list = [x for x in mean_list if str(x) != 'nan']

        distance_list.append(np.max(cleaned_mean_list) -
                             np.min(cleaned_mean_list))

        diseases_abbr_list.append(diseases_abbr[disease])

    d = {'diseases': diseases, 'diseases_abbr': diseases_abbr_list, 'Distance': distance_list,
         "#White": percent_white_list, 'Gap_W_mean': gap_white_mean_list, 'CI_W': ci_white_list,
         "#Black": percent_white_list, 'Gap_B_mean': gap_black_mean_list, 'CI_B': ci_black_list,
         "#Hisp": percent_hisp_list, 'Gap_H_mean': gap_hisp_mean_list, 'CI_H': ci_hisp_list,
         "#Other": percent_other_list, 'Gap_Ot_mean': gap_other_mean_list, 'CI_Ot': ci_other_list,
         "#Asian": percent_asian_list, 'Gap_As_mean': gap_asian_mean_list, 'CI_As': ci_asian_list,
         "#American": percent_american_list, 'Gap_Am_mean': gap_american_mean_list, 'CI_Am': ci_american_list
         }
    race_fpr_disp_df = pd.DataFrame(d)
    race_fpr_disp_df = race_fpr_disp_df.sort_values(by='Distance')
    race_fpr_disp_df.to_csv(basepath+"race_fpr_disp.csv", index=False)

    # Plotting the results
    plot_race_FPR_GAPS(basepath, race_fpr_disp_df, height,
                       font_size, rotation_degree)


def plot_race_FPR_GAPS(basepath, race_fpr_disp_df, height, font_size, rotation_degree):

    plt.ioff()
    plt.rcParams.update({'font.size': font_size})
    plt.figure(figsize=(16, height))

    plt.scatter(race_fpr_disp_df['diseases_abbr'], race_fpr_disp_df['Gap_W_mean'],
                s=np.multiply(race_fpr_disp_df['#White'], 0.095), marker='o', color='blue', label="WHITE")
    plt.errorbar(race_fpr_disp_df['diseases_abbr'], race_fpr_disp_df['Gap_W_mean'],
                 yerr=race_fpr_disp_df['CI_W'], fmt='o', mfc='blue')  # ecolor='blue'

    plt.scatter(race_fpr_disp_df['diseases_abbr'], race_fpr_disp_df['Gap_B_mean'],
                s=np.multiply(race_fpr_disp_df['#Black'], 0.095), marker='o', color='orange', label="BLACK")
    plt.errorbar(race_fpr_disp_df['diseases_abbr'], race_fpr_disp_df['Gap_B_mean'],
                 yerr=race_fpr_disp_df['CI_B'], fmt='o', mfc='orange')

    plt.scatter(race_fpr_disp_df['diseases_abbr'], race_fpr_disp_df['Gap_H_mean'],
                s=np.multiply(race_fpr_disp_df['#Hisp'], 0.095), marker='o', color='green', label="HISPANIC")
    plt.errorbar(race_fpr_disp_df['diseases_abbr'], race_fpr_disp_df['Gap_H_mean'],
                 yerr=race_fpr_disp_df['CI_H'], fmt='o', mfc='green')

    plt.scatter(race_fpr_disp_df['diseases_abbr'], race_fpr_disp_df['Gap_Ot_mean'],
                s=np.multiply(race_fpr_disp_df['#Other'], 0.095), marker='o', color='r', label="OTHER")
    plt.errorbar(race_fpr_disp_df['diseases_abbr'], race_fpr_disp_df['Gap_Ot_mean'],
                 yerr=race_fpr_disp_df['CI_Ot'], fmt='o', mfc='r')

    plt.scatter(race_fpr_disp_df['diseases_abbr'], race_fpr_disp_df['Gap_As_mean'],
                s=np.multiply(race_fpr_disp_df['#Asian'], 0.095), marker='o', color='m', label="ASIAN")
    plt.errorbar(race_fpr_disp_df['diseases_abbr'], race_fpr_disp_df['Gap_As_mean'],
                 yerr=race_fpr_disp_df['CI_As'], fmt='o', mfc='m')

    plt.scatter(race_fpr_disp_df['diseases_abbr'], race_fpr_disp_df['Gap_Am_mean'],
                s=np.multiply(race_fpr_disp_df['#American'], 0.095), marker='o', color='k', label="NATIVE")
    plt.errorbar(race_fpr_disp_df['diseases_abbr'], race_fpr_disp_df['Gap_Am_mean'],
                 yerr=race_fpr_disp_df['CI_Am'], fmt='o', mfc='k')

    plt.xticks(rotation=rotation_degree, fontsize=font_size,
               fontname='Times New Roman')
    plt.ylabel("FPR RACE DISPARITY", fontsize=font_size,
               fontname='Times New Roman')
    plt.yticks(fontsize=font_size, fontname='Times New Roman')
    plt.legend()
    plt.grid(True)
    plt.savefig(basepath+"FPR_Dis_RACE.pdf")
    plt.close()


def get_Insurance_FPR_Disparities(basepath, diseases, diseases_abbr, number_of_runs, significance_level,
                                  height, font_size, rotation_degree):
    seed_19_insurance = pd.read_csv(
        basepath+"Run_seed32_FPR_GAP_insurance.csv", index_col=0)
    seed_31_insurance = pd.read_csv(
        basepath+"Run_seed40_FPR_GAP_insurance.csv", index_col=0)
    seed_38_insurance = pd.read_csv(
        basepath+"Run_seed56_FPR_GAP_insurance.csv", index_col=0)
    seed_47_insurance = pd.read_csv(
        basepath+"Run_seed60_FPR_GAP_insurance.csv", index_col=0)
    seed_77_insurance = pd.read_csv(
        basepath+"Run_seed90_FPR_GAP_insurance.csv", index_col=0)

    insurance_dataframes = [seed_19_insurance, seed_31_insurance, seed_38_insurance,
                            seed_47_insurance, seed_77_insurance]

    # group by diseases and calculate the mean and std for each disease
    result_insurance_df = pd.concat(insurance_dataframes)
    result_insurance_grouped = result_insurance_df.groupby("diseases")
    result_insurance_grouped_stat = result_insurance_grouped.describe()

    insurance_tpr_disp = pd.DataFrame(
        pd.DataFrame(diseases, columns=["diseases"]))

    # get FNR GAPS for each disease subgroup
    insurance_df_medicare_mean = result_insurance_grouped_stat['#medicare']['mean']
    insurance_df_medicare_gap = result_insurance_grouped_stat['Gap_medicare']["mean"]
    insurance_df_medicare_ci = (
        significance_level * result_insurance_grouped_stat['Gap_medicare']["std"]) / np.sqrt(number_of_runs)

    insurance_df_other_mean = result_insurance_grouped_stat['#other']['mean']
    insurance_df_other_gap = result_insurance_grouped_stat['Gap_other']["mean"]
    insurance_df_other_ci = (
        significance_level * result_insurance_grouped_stat['Gap_other']["std"]) / np.sqrt(number_of_runs)

    insurance_df_medicaid_mean = result_insurance_grouped_stat['#medicaid']['mean']
    insurance_df_medicaid_gap = result_insurance_grouped_stat['Gap_medicaid']["mean"]
    insurance_df_medicaid_ci = (
        significance_level * result_insurance_grouped_stat['Gap_medicaid']["std"]) / np.sqrt(number_of_runs)

    percent_medicaid_list = []
    ci_medicaid_list = []
    gap_medicaid_mean_list = []

    percent_medicare_list = []
    ci_medicare_list = []
    gap_medicare_mean_list = []

    percent_other_list = []
    ci_other_list = []
    gap_other_mean_list = []

    diseases_abbr_list = []
    distance_list = []

    for disease in diseases:
        mean_list = []

        percent_other_list.append(insurance_df_other_mean[disease])
        gap_other_mean_list.append(insurance_df_other_gap[disease])
        ci_other_list.append(insurance_df_other_ci[disease])
        mean_list.append(insurance_df_other_ci[disease])

        percent_medicare_list.append(insurance_df_medicare_mean[disease])
        gap_medicare_mean_list.append(insurance_df_medicare_gap[disease])
        ci_medicare_list.append(insurance_df_medicare_ci[disease])
        mean_list.append(insurance_df_medicare_ci[disease])

        percent_medicaid_list.append(insurance_df_medicaid_mean[disease])
        gap_medicaid_mean_list.append(insurance_df_medicaid_gap[disease])
        ci_medicaid_list.append(insurance_df_medicaid_ci[disease])
        mean_list.append(insurance_df_medicare_ci[disease])

        cleaned_mean_list = [x for x in mean_list if str(x) != 'nan']
        distance_list.append(np.max(cleaned_mean_list) -
                             np.min(cleaned_mean_list))
        diseases_abbr_list.append(diseases_abbr[disease])

    d = {'diseases': diseases, 'diseases_abbr': diseases_abbr_list, 'Distance': distance_list,
         "#Medicare": percent_medicare_list, 'Gap_C_mean': gap_medicare_mean_list, 'CI_C': ci_medicare_list,
         "#Other": percent_other_list, 'Gap_O_mean': gap_other_mean_list, 'CI_O': ci_other_list,
         "#Medicaid": percent_medicaid_list, 'Gap_A_mean': gap_medicaid_mean_list, 'CI_A': ci_medicaid_list
         }
    insurance_fpr_disp_df = pd.DataFrame(d)
    insurance_fpr_disp_df = insurance_fpr_disp_df.sort_values(by='Distance')
    insurance_fpr_disp_df.to_csv(
        basepath+"insurance_fpr_disp.csv", index=False)

    # Plotting the results
    plot_insurance_FPR_GAPS(basepath, insurance_fpr_disp_df, height,
                            font_size, rotation_degree)


def plot_insurance_FPR_GAPS(basepath, insurance_fpr_disp_df, height, font_size, rotation_degree):

    plt.ioff()
    plt.rcParams.update({'font.size': font_size})
    plt.figure(figsize=(16, height))
    plt.scatter(insurance_fpr_disp_df['diseases_abbr'], insurance_fpr_disp_df['Gap_C_mean'],
                s=np.multiply(insurance_fpr_disp_df['#Medicare'], 0.095), marker='o', color='blue', label="MEDICARE")
    plt.errorbar(insurance_fpr_disp_df['diseases_abbr'], insurance_fpr_disp_df['Gap_C_mean'],
                 yerr=insurance_fpr_disp_df['CI_C'], fmt='o', mfc='blue')

    plt.scatter(insurance_fpr_disp_df['diseases_abbr'], insurance_fpr_disp_df['Gap_O_mean'],
                s=np.multiply(insurance_fpr_disp_df['#Other'], 0.095), marker='o', color='orange', label="OTHER")
    plt.errorbar(insurance_fpr_disp_df['diseases_abbr'], insurance_fpr_disp_df['Gap_O_mean'],
                 yerr=insurance_fpr_disp_df['CI_O'], fmt='o', mfc='orange')

    plt.scatter(insurance_fpr_disp_df['diseases_abbr'], insurance_fpr_disp_df['Gap_A_mean'],
                s=np.multiply(insurance_fpr_disp_df['#Medicaid'], 0.095), marker='o', color='green', label="MEDICAID")
    plt.errorbar(insurance_fpr_disp_df['diseases_abbr'], insurance_fpr_disp_df['Gap_A_mean'],
                 yerr=insurance_fpr_disp_df['CI_A'], fmt='o', mfc='green')

    plt.xticks(rotation=rotation_degree, fontsize=font_size,
               fontname='Times New Roman')
    plt.ylabel("FPR INSURANCE DISPARITY", fontsize=font_size,
               fontname='Times New Roman')
    plt.yticks(fontsize=font_size, fontname='Times New Roman')
    plt.legend()
    plt.grid(True)
    plt.savefig(basepath+"FPR_Dis_INSURANCE.pdf")
    plt.close()
