from FPR import FP_NF_CheXpert, FP_NF_CheXpert_Inter, FiveRunSubgroup, FiveRunTwoGroup
from CheXpert_Biomedclip_Training import training_and_evaluation, calculate_avg_aucs, make_dataset
import pandas as pd
import numpy as np
import tensorflow as tf
import random as python_random
import sys
import os
import pickle
from sklearn.utils import shuffle
import pdb


def main():

    debug = False

    sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname('__file__'), os.pardir)))
    root_dir = os.path.dirname(os.path.dirname('__file__'))

    medclip_embedding_folder = os.path.join(root_dir, "biomedclip-embedding")

    os.makedirs(medclip_embedding_folder, exist_ok=True)
    medclip_embedding_path = os.path.join(
        medclip_embedding_folder, "embedding_from_biomedclip.pkl")
    print(medclip_embedding_path)

    metadata_df = pd.read_csv("./dataframes/chexpert_df.csv")
    metadata_df['ids'] = range(len(metadata_df))

    print('=============================== Load embedding and target ======================================')
    if os.path.exists(medclip_embedding_path):
        with open(medclip_embedding_path, 'rb') as f:
            X, y, valid_id, ids = pickle.load(f)

    bioClipdf = pd.DataFrame({
        'ids': range(len(X)),        # Create an 'id' column starting from 0
        'emb': list(X),  # The 'emb' column holds the embeddings
    })
    bioClipdf.columns = ['ids', 'emb']

    bioClipdf = bioClipdf[~bioClipdf['emb'].apply(
        lambda x: np.all(np.array(x) == 0.0))]

    df = bioClipdf.merge(metadata_df, on='ids', how="inner")
    df = df[~df['emb'].apply(lambda x: np.all(np.array(x) == 0.0))]

    # Define the desired order of columns
    # Define the desired order of columns
    columns_order = ['Path', 'race', 'Sex', 'age_decile', 'Frontal/Lateral', 'AP/PA', 'emb',
                     'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
                     'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                     'Fracture', 'Support Devices', 'No Finding']
    # Reorder the columns in the merged DataFrame
    df = df[columns_order]

    labels_Columns = ['Enlarged Cardiomediastinum', 'Cardiomegaly',
                      'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                      'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                      'Fracture', 'Support Devices', 'No Finding']
    df[labels_Columns] = df[labels_Columns].astype("float32")

    df['path_splited'] = df['Path'].str.split('/')
    df['patientid'] = df['path_splited'].apply(lambda x: x[2])

    # Insert 'patient_id' column at the second position
    df.insert(1, 'patient_id', df['patientid'])

    # Drop the intermediate 'path_splited' column
    df.drop('path_splited', axis=1, inplace=True)
    df.drop('patientid', axis=1, inplace=True)
    df[labels_Columns] = df[labels_Columns].replace(-1.0, 0.0)

    unique_pathId = df.patient_id.unique()

    train_percent, valid_percent, test_percent = 0.80, 0.10, 0.10

    unique_path_id = shuffle(unique_pathId)
    value1 = (round(len(unique_path_id)*train_percent))
    value2 = (round(len(unique_path_id)*valid_percent))
    value3 = value1 + value2
    value4 = (round(len(unique_path_id)*test_percent))

    print("===================== Patients in training set: ============" + str(value1))
    print("====================== Patients in validation set: =============" + str(value2))
    print("============================= Patients in testing set: =================== " + str(value4))

    df["Sex"].replace({'Male': 'M', 'Female': 'F'}, inplace=True)
    df.rename(columns={'Sex': 'gender'}, inplace=True)
    df.rename(columns={'Path': 'path'}, inplace=True)

    df = shuffle(df)
    train_path_id = unique_path_id[:value1]
    validate_path_id = unique_path_id[value1:value3]
    test_path_id = unique_path_id[value3:]
    df.insert(7, "split", "none", True)

    df.loc[df.patient_id.isin(train_path_id), "split"] = "train"
    df.loc[df.patient_id.isin(validate_path_id), "split"] = "validate"
    df.loc[df.patient_id.isin(test_path_id), "split"] = "test"

    df_train = df[df["split"] == "train"]
    df_validate = df[df["split"] == "validate"]
    df_test = df[df["split"] == "test"]

    training_data, validation_data, test_dataset = make_dataset(
        df_train=df_train, df_validate=df_validate, df_test=df_test, labels_Columns=labels_Columns)

    df_validate.to_csv("./biomedclip-embedding/df_validate.csv", index=False)
    df_test.to_csv("./biomedclip-embedding/df_test.csv", index=False)

    SEEDS = [19, 31, 38, 47, 77]

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

    diseases = ['No Finding']
    age_decile = ['0-20', '20-40', '40-60', '60-80', '80+']
    gender = ['M', 'F']
    race = ['WHITE', 'BLACK/AFRICAN AMERICAN',
            'HISPANIC/LATINO', 'OTHER', 'ASIAN',
            'AMERICAN INDIAN/ALASKA NATIVE']

    category = [gender, age_decile, race,]
    category_name = ['gender', 'age_decile', 'race']

    for seed in SEEDS:

        # Predictions
        print(
            f'================================== Seed {seed}  started ============================')

        test_eval, bipred = training_and_evaluation(
            seed, medclip_embedding_folder, training_data, validation_data, test_dataset,
            df_test, labels_Columns, thersholds, debug)

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
                gender_FPR[f"Seed_{seed}"] = FP_NF_CheXpert(
                    pred_df, diseases, category[i], category_name[i], seed=seed)

            if category_name[i] == "age_decile":
                print(
                    "=====================FPR for age started ===================================")
                age_FPR[f"Seed_{seed}"] = FP_NF_CheXpert(
                    pred_df, diseases, category[i], category_name[i], seed=seed)

            if category_name[i] == "race":

                print(
                    "=====================FPR for race started ===================================")
                race_FPR[f"Seed_{seed}"] = FP_NF_CheXpert(
                    pred_df, diseases, category[i], category_name[i], seed=seed)

        # ============================Two group FPR ===========================

        FPR_SexRace[f"Seed_{seed}"] = FP_NF_CheXpert_Inter(
            pred_df, diseases, gender, 'gender', race, 'race', seed)
        FPR_AgeSex[f"Seed_{seed}"] = FP_NF_CheXpert_Inter(
            pred_df, diseases, gender, 'gender', age_decile, 'age_decile')
        FPR_AgeRace[f"Seed_{seed}"] = FP_NF_CheXpert_Inter(
            pred_df, diseases, race, 'race', age_decile, 'age_decile', seed)

        print(
            f'================================== FPR for seed {seed} complete ============================')

        print(
            f'================================== Seed {seed}  Finished =========================================')

    # Calcualting AVG AUC and Confidence
    # Access and work with each DataFrame in the dictionary and write into a file
    with open(f"{medclip_embedding_folder}/AverageAucs.txt", "w") as file:
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

    # Sex
    FPR_results_path = os.path.join("FPR")
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

    # ======================================== Two group FPR================================

    fpr_SexRace = pd.concat(FPR_SexRace.values(), axis=0)
    fpr_SexRace = fpr_SexRace.reset_index(drop=True)
    fpr_SexRace = fpr_SexRace.groupby("race")
    fpr_SexRace = fpr_SexRace.describe()
    factors = ['FPR_F', 'FPR_M']
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
    factors = ['FPR_White', 'FPR_Black', 'FPR_Hisp',
               'FPR_Other', 'FPR_Asian', 'FPR_American']

    fpr_AgeRace = pd.concat(FPR_AgeRace.values(), axis=0)
    fpr_AgeRace = fpr_AgeRace.reset_index(drop=True)
    fpr_AgeRace = fpr_AgeRace.groupby("age")
    fpr_AgeRace = fpr_AgeRace.describe()

    fpr_AgeRace_df = pd.DataFrame(age_decile, columns=["Age"])
    fpr_AgeRace_df = FiveRunTwoGroup(factors, fpr_AgeRace_df, fpr_AgeRace)

    fpr_AgeRace_df.to_csv(
        FPR_results_path+'/FPR_RaceAge.csv', index=False)

    print("======= done =====")


if __name__ == '__main__':
    main()
