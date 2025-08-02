
from MIMIC_CXR_Emb_Training import training_and_evaluation, calculate_avg_aucs
import pandas as pd
import numpy as np
import tensorflow as tf
import random as python_random
import sys
import os
import pickle
from sklearn.utils import shuffle
import pdb

from Fairness.MIMIC_CXR_EMB.config_MIMIC import get_diseases,get_diseases_abbr,get_patient_groups,get_patient_groups_abbr,get_seeds,get_utility_variables

sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname('__file__'), os.pardir)))

root_dir = os.path.dirname(os.path.dirname('__file__'))

from BMC_EMB.MIMIC.FPR import FP_NF_MIMIC, FP_NF_MIMIC_Two_Group_Inter, FiveRunSubgroup, FiveRunTwoGroup

def main():

    debug = False

    df = pd.read_csv("./Fairness/MIMIC_CXR_EMB/Extracted_Embeddings/processed_mimic_df.csv")
    
    # Change this to the correct  dataset path
    df['path']="./Fairness/MIMIC_CXR_EMB/Extracted_Embeddings/generalized-image-embeddings-for-the-mimic-chest-x-ray-dataset-1.0/"+df['path']
    
    # Define the desired order of columns
    olumns_order = ['path', 'subject_id', 'study_id', 'gender', 'insurance',
                    'age_decile', 'race', 'split', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                    'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                    'Fracture', 'Support Devices', 'No Finding']
    df = df[olumns_order]
   
    df_train = df[df["split"] == "train"]
    df_validate = df[df["split"] == "validate"]
    df_test = df[df["split"] == "test"]

    

    SEEDS = [19, 31, 38, 47, 77]
    # SEEDS = [19]

    test_evals = {}
    test_evals[f"Test_Eval"] = pd.DataFrame()

    thersholds = {}
    bipreds = {}

    race_FPR = {}
    age_FPR = {}
    gender_FPR = {}
    insurance_FPR = {}

    FPR_SexRace = {}
    FPR_AgeSex = {}
    FPR_AgeRace = {}
    FPR_InsuAge = {}
    FPR_InsuRace = {}
    FPR_InsuSex = {}
    
    labels_Columns = get_diseases()
    diseases = labels_Columns
    
    diseases_abbr =get_diseases_abbr()
    
    patient_groups=get_patient_groups()
    
    gender = patient_groups["sex"]
    age_decile =patient_groups["age"]
    race =patient_groups["race"]
    insurance = patient_groups["insurance"]
    
    category = [gender, age_decile, race, insurance]
    category_name = ['gender', 'age_decile', 'race', 'insurance']

    for seed in SEEDS:

        # Predictions
        print(
            f'================================== Seed {seed}  started ============================')
        
        base_path="./Fairness/MIMIC_CXR_EMB/"
        test_eval, bipred = training_and_evaluation(
            seed,base_path,df_train, df_validate,
            df_test, labels_Columns, thersholds, debug,is_mimic=True)

        print("Test Eval Output:", test_eval)
        print("Bipred Output:", bipred)

        if seed == 19:
            test_evals[f"Test_Eval"]['Label'] = test_eval['label']
            test_evals[f"Test_Eval"][f'Seed_{seed}'] = test_eval['auc']
        else:
            test_evals[f"Test_Eval"][f'Seed_{seed}'] = test_eval['auc']

        bipreds[f"Seed_{seed}"] = bipred

        print(
            f'================================== Predictions for seed {seed} complete ============================')

        print(
            f'================================== FPR Seed {seed} ============================')

        pred_df = df_test.merge(
            bipreds[f"Seed_{seed}"], on="path", how="inner")

        for i in range(len(category)):

            if category_name[i] == "gender":
                print(
                    "=====================FPR for gender started ===================================")
                gender_FPR[f"Seed_{seed}"] = FP_NF_MIMIC(
                    pred_df, diseases, category[i], category_name[i], seed=seed)

            if category_name[i] == "age_decile":
                print(
                    "=====================FPR for age started ===================================")
                age_FPR[f"Seed_{seed}"] = FP_NF_MIMIC(
                    pred_df, diseases, category[i], category_name[i], seed=seed)

            if category_name[i] == "race":

                print(
                    "=====================FPR for race started ===================================")
                race_FPR[f"Seed_{seed}"] = FP_NF_MIMIC(
                    pred_df, diseases, category[i], category_name[i], seed=seed)

            if category_name[i] == "insurance":
                print(
                    "=====================FPR for insurance started ===================================")
                insurance_FPR[f"Seed_{seed}"] = FP_NF_MIMIC(
                    pred_df, diseases, category[i], category_name[i], seed=seed)

        print(
            f'================================== FPR for seed {seed} complete ============================')

        # ============================Two group FPR ===========================

        FPR_SexRace[f"Seed_{seed}"] = FP_NF_MIMIC_Two_Group_Inter(
            pred_df, diseases, gender, 'gender', race, 'race', seed)

        FPR_AgeSex[f"Seed_{seed}"] = FP_NF_MIMIC_Two_Group_Inter(
            pred_df, diseases, gender, 'gender', age_decile, 'age_decile')
        FPR_AgeRace[f"Seed_{seed}"] = FP_NF_MIMIC_Two_Group_Inter(
            pred_df, diseases, race, 'race', age_decile, 'age_decile', seed)
        FPR_InsuAge[f"Seed_{seed}"] = FP_NF_MIMIC_Two_Group_Inter(pred_df, diseases, insurance, 'insurance',
                                                                  age_decile, 'age_decile', seed)
        FPR_InsuRace[f"Seed_{seed}"] = FP_NF_MIMIC_Two_Group_Inter(
            pred_df, diseases, insurance, 'insurance', race, 'race', seed)
        FPR_InsuSex[f"Seed_{seed}"] = FP_NF_MIMIC_Two_Group_Inter(
            pred_df, diseases, gender, 'gender', insurance, 'insurance', seed)

        print(
            f'================================== Seed {seed}  Finished =========================================')

    # Calcualting AVG AUC and Confidence
    # Access and work with each DataFrame in the dictionary and write into a file
    with open(f"AverageAucs.txt", "w") as file:
        file.write("Label, AUC, CI\n")  # write header

        for key, df in test_evals.items():

            rate = key.split('_')[1]

            avg_auc, avg_ci, mean_auc_labels, mean_auc_ci = calculate_avg_aucs(
                key=key, df=df)

            # Iterate over the indices of the DataFrame
            for i in range(len(df['Label'])):
                label = df['Label'].iloc[i]
                mean_auc_label = mean_auc_labels[i]
                mean_auc_ci_value = mean_auc_ci[i]
                file.write(
                    f"{label}, {mean_auc_label}, {mean_auc_ci_value}\n")

            file.write(f"AVG,  {avg_auc}, {avg_ci}")
            file.write("\n")

    print("======== Average AUCS Calcualted ============================")

    print(f"=========================== Average FPR===========================================")

    print(FPR_SexRace["Seed_19"].head())
    # ======================================== Sub group FPR================================
    # Sex
    FPR_results_path = os.path.join(base_path,"FPR")
    os.makedirs(FPR_results_path, exist_ok=True)

    fpr_sex = pd.concat(gender_FPR.values(), axis=0)
    fpr_sex = fpr_sex.reset_index(drop=True)
    fpr_sex = fpr_sex.describe()

    fpr_sex_df = pd.DataFrame(gender, columns=["sex"])
    fpr_sex_df = FiveRunSubgroup(gender, fpr_sex, fpr_sex_df)

    fpr_sex_df.to_csv(
        FPR_results_path+'/Subgroup_FPR_Sex.csv', index=False)

    # Age
    fpr_age = pd.concat(age_FPR.values(), axis=0)
    fpr_age = fpr_age.reset_index(drop=True)
    fpr_age = fpr_age.describe()

    fpr_age_df = pd.DataFrame(age_decile, columns=["age"])
    fpr_age_df = FiveRunSubgroup(age_decile, fpr_age, fpr_age_df)

    fpr_age_df.to_csv(
        FPR_results_path+'/Subgroup_FPR_Age.csv', index=False)

    # Race
    fpr_race = pd.concat(race_FPR.values(), axis=0)
    fpr_race = fpr_race.reset_index(drop=True)
    fpr_race = fpr_race.describe()

    race = ['White', 'Black', 'Hisp', 'Other', 'Asian', 'American']

    fpr_race_df = pd.DataFrame(race, columns=["race"])
    fpr_race_df = FiveRunSubgroup(race, fpr_race, fpr_race_df)

    fpr_race_df.to_csv(
        FPR_results_path+'/Subgroup_FPR_Race.csv', index=False)

    # Insurance
    fpr_insurance = pd.concat(insurance_FPR.values(), axis=0)
    fpr_insurance = fpr_insurance.reset_index(drop=True)
    fpr_insurance = fpr_insurance.describe()

    fpr_insu_df = pd.DataFrame(insurance, columns=["insurance"])
    fpr_insu_df = FiveRunSubgroup(insurance, fpr_insurance, fpr_insu_df)

    fpr_insu_df.to_csv(
        FPR_results_path+'/Subgroup_FPR_Insu.csv', index=False)

    # ======================================== Two group FPR================================

    fpr_SexRace = pd.concat(FPR_SexRace.values(), axis=0)
    fpr_SexRace = fpr_SexRace.reset_index(drop=True)

    print(fpr_SexRace)

    fpr_SexRace = fpr_SexRace.groupby("race")
    fpr_SexRace = fpr_SexRace.describe()

    print(fpr_SexRace)
    factors = ['FPR_F', 'FPR_M']

    # pdb.set_trace()

    fpr_SexRace_df = pd.DataFrame(race, columns=["race"])
    fpr_SexRace_df = FiveRunTwoGroup(factors, fpr_SexRace_df, fpr_SexRace)

    fpr_SexRace_df.to_csv(
        FPR_results_path+'/FPR_SexRace.csv', index=False)

    # Age and Sex
    fpr_AgeSex = pd.concat(FPR_AgeSex.values(), axis=0)
    fpr_AgeSex = fpr_AgeSex.reset_index(drop=True)
    fpr_AgeSex = fpr_AgeSex.groupby("Age")
    fpr_AgeSex = fpr_AgeSex.describe()

    fpr_AgeSex_df = pd.DataFrame(age_decile, columns=["Age"])
    fpr_AgeSex_df = FiveRunTwoGroup(factors, fpr_AgeSex_df, fpr_AgeSex)

    fpr_AgeSex_df.to_csv(
        FPR_results_path+'/FPR_AgeSex.csv', index=False)

    # Age and race
    fpr_AgeRace = pd.concat(FPR_AgeRace.values(), axis=0)
    fpr_AgeRace = fpr_AgeRace.reset_index(drop=True)
    fpr_AgeRace = fpr_AgeRace.groupby("age")
    fpr_AgeRace = fpr_AgeRace.describe()

    factors = ['FPR_White', 'FPR_Black', 'FPR_Hisp',
               'FPR_Other', 'FPR_Asian', 'FPR_American']

    fpr_AgeRace_df = pd.DataFrame(age_decile, columns=["Age"])
    fpr_AgeRace_df = FiveRunTwoGroup(factors, fpr_AgeRace_df, fpr_AgeRace)

    fpr_AgeRace_df.to_csv(
        FPR_results_path+'/FPR_RaceAge.csv', index=False)

    # Insurance and Age
    fpr_InsuAge = pd.concat(FPR_InsuAge.values(), axis=0)
    fpr_InsuAge = fpr_InsuAge.reset_index(drop=True)
    fpr_InsuAge = fpr_InsuAge.groupby("age")
    fpr_InsuAge = fpr_InsuAge.describe()

    factors = ['FPR_Medicare', 'FPR_Other', 'FPR_Medicaid']

    fpr_InsuAge_df = pd.DataFrame(age_decile, columns=["Age"])
    fpr_InsuAge_df = FiveRunTwoGroup(factors, fpr_InsuAge_df, fpr_InsuAge)

    fpr_InsuAge_df.to_csv(
        FPR_results_path+'/FPR_AgeIns.csv', index=False)

    # Insurance and Race
    fpr_InsuRace = pd.concat(FPR_InsuRace.values(), axis=0)
    fpr_InsuRace = fpr_InsuRace.reset_index(drop=True)
    fpr_InsuRace = fpr_InsuRace.groupby("race")
    fpr_InsuRace = fpr_InsuRace.describe()

    fpr_InsuRace_df = pd.DataFrame(race, columns=["race"])
    fpr_InsuRace_df = FiveRunTwoGroup(factors, fpr_InsuRace_df, fpr_InsuRace)

    fpr_InsuRace_df.to_csv(
        FPR_results_path+'/FPR_RaceIns.csv', index=False)

    # Insurance and Sex
    fpr_InsuSex = pd.concat(FPR_InsuSex.values(), axis=0)
    fpr_InsuSex = fpr_InsuSex.reset_index(drop=True)
    fpr_InsuSex = fpr_InsuSex.groupby("Insurance")
    fpr_InsuSex = fpr_InsuSex.describe()

    factors = ['FPR_F', 'FPR_M']

    fpr_InsuSex_df = pd.DataFrame(insurance, columns=["Insurance"])
    fpr_InsuSex_df = FiveRunTwoGroup(factors, fpr_InsuSex_df, fpr_InsuSex)

    fpr_InsuSex_df.to_csv(
        FPR_results_path+'/FPR_SexIns.csv', index=False)


print("======= done =====")


if __name__ == '__main__':
    main()
