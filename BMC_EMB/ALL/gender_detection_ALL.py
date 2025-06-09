
import pandas as pd
from IPython.display import clear_output
import io
import os
import sys
import glob
import numpy as np
import random as python_random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import pickle

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score, roc_curve, auc


def main():

    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname('__file__'), "..")))

    from MIMIC.MIMIC_BioMedClip_Race_Detection import get_optimal_hyperparameters
    from MIMIC.gender_detection_MIMIC import build_model

   # region ===============================MIMIC_BMC_EMB preprocessing ===========================================
    root_dir = os.path.dirname(os.path.dirname('__file__'))

    medclip_embedding_folder = os.path.join(root_dir, "biomedclip-embedding")
    os.makedirs(medclip_embedding_folder, exist_ok=True)
    medclip_embedding_path = os.path.join(
        medclip_embedding_folder, "embedding_from_biomedclip_MIMIC.pkl")
    print(medclip_embedding_path)

    metadata_df = pd.read_csv("../MIMIC/df_final_for_metadata_prediction.csv")
    metadata_df['ids'] = range(len(metadata_df))

    print('=============================== Load embedding and target ======================================')

    metadata_df['ids'] = range(len(metadata_df))
    metadata_df = metadata_df[['ids', 'path']]

    print('=============================== Load embedding and target ======================================')
    if os.path.exists(medclip_embedding_path):
        with open(medclip_embedding_path, 'rb') as f:
            X, y, valid_id, ids = pickle.load(f)

    # X is 2D numpy array of shape (200452, 512)
    bioClipdf = pd.DataFrame({
        'ids': range(len(X)),
        'emb': list(X),
    })

    bioClipdf.columns = ['ids', 'emb']
    df = bioClipdf.merge(metadata_df, on='ids', how="inner")
    df = df[~df['emb'].apply(lambda x: np.all(np.array(x) == 0.0))]

    processed_mimic_df = pd.read_csv("../MIMIC/processed_mimic_df.csv")

    df_MIMIC = df.merge(processed_mimic_df, on='path', how="inner")

    # Define the desired order of columns
    columns_order = ['subject_id', 'emb', 'gender']
    df_MIMIC = df_MIMIC[columns_order]
    df_MIMIC.rename(columns={"subject_id": "patient_id"}, inplace=True)

    # endregion

    # region =================================== CheXpert Preprocessing ==============================

    root_dir = os.path.dirname(os.path.dirname('__file__'))

    medclip_embedding_folder = os.path.join(root_dir, "biomedclip-embedding")
    os.makedirs(medclip_embedding_folder, exist_ok=True)
    medclip_embedding_path = os.path.join(
        medclip_embedding_folder, "embedding_from_biomedclip_CXP.pkl")
    print(medclip_embedding_path)

    metadata_df = pd.read_csv("../CheXpert/dataframes/chexpert_df.csv")
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

    df['path_splited'] = df['Path'].str.split('/')
    df['patientid'] = df['path_splited'].apply(lambda x: x[2])

    # Insert 'patient_id' column at the second position
    df.insert(1, 'patient_id', df['patientid'])

    # Drop the intermediate 'path_splited' column
    df.drop(['path_splited', 'patientid', 'Path'],
            axis=1, inplace=True)
    df.rename(columns={"Sex": "gender"}, inplace=True)

    # Define the desired order of columns
    columns_order = ['patient_id', 'emb', 'gender']
    df_Chexpert = df[columns_order]

    df_Chexpert['gender'] = df_Chexpert['gender'].replace({'Male': 'M',
                                                           'Female': 'F', })
    # endregion

    # region =============================== concatenation, Training,validation, test split ===========================

    df_MIMIC["patient_id"] = df_MIMIC["patient_id"].astype("object")

    df = pd.concat([df_MIMIC, df_Chexpert], ignore_index=True)
    df = df[df.gender .isin(['M', 'F'])]

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

    df = shuffle(df)
    train_path_id = unique_path_id[:value1]
    validate_path_id = unique_path_id[value1:value3]
    test_path_id = unique_path_id[value3:]

    df.insert(1, "split", "none", True)
    df.loc[df.patient_id.isin(train_path_id), "split"] = "train"
    df.loc[df.patient_id.isin(validate_path_id), "split"] = "validate"
    df.loc[df.patient_id.isin(test_path_id), "split"] = "test"

    # Define the class names (replace with your actual class names)
    class_names = df['gender'].unique()

    print(
        f'"============================ class names : {class_names} ======================')

    print(
        f'"============================ number of class : {len(class_names)} ======================')
    df.split.value_counts()
    df_train = df[df["split"] == "train"]
    df_validate = df[df["split"] == "validate"]
    df_test = df[df["split"] == "test"]

    # endregion

    # region ==================================== Preparing Embeddings Files for Model Training ====================
    # We want to detect race of a patient using the embeddings with this label.
    label_encoder = LabelEncoder()
    df_train['gender'] = label_encoder.fit_transform(df_train['gender'])
    df_validate['gender'] = label_encoder.fit_transform(df_validate['gender'])
    df_test['gender'] = label_encoder.fit_transform(df_test['gender'])

    y_train_encoded = label_encoder.fit_transform(df_train.gender)
    y_val_encoded = label_encoder.transform(df_validate.gender)
    y_test_encoded = label_encoder.transform(df_test.gender)

    num_classes = len(class_names)

    # Convert integer labels to one-hot encoding
    y_train_one_hot = to_categorical(y_train_encoded, num_classes=num_classes)
    y_val_one_hot = to_categorical(y_val_encoded, num_classes=num_classes)
    y_test_one_hot = to_categorical(y_test_encoded, num_classes=num_classes)

    # Create training and validation Datasets
    training_data = tf.data.Dataset.from_tensor_slices(
        (df_train.emb.values.tolist(), y_train_one_hot))

    validation_data = tf.data.Dataset.from_tensor_slices(
        (df_validate.emb.values.tolist(), y_val_one_hot))

    test_data = tf.data.Dataset.from_tensor_slices(
        (df_test.emb.values.tolist(), y_test_one_hot))

    # Early Stopping
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True)

    # Create and Train Model
    params = get_optimal_hyperparameters()
    model = build_model(params, num_classes=num_classes)

    # endregion

    gender = ['F', 'M']
    auc_df = pd.DataFrame({"gender": gender})  # Create DataFrame correctly

    # region ====================================== Training and evaluation ===============================
    SEEDs = [19, 31, 38, 47, 77]

    for seed in SEEDs:

        np.random.seed(seed)
        python_random.seed(seed)
        tf.random.set_seed(seed)

        # train the model
        history = model.fit(
            x=training_data.batch(params["batch_size"]).prefetch(
                tf.data.AUTOTUNE).cache(),
            validation_data=validation_data.batch(
                params["batch_size"]).cache(),
            callbacks=[early_stopping],
            epochs=params["epochs"],)

        print("===================== Prepare test dataset and evaluate model performance =====================")

        # Evaluate using ROC-AUC
        y_pred_prob = model.predict(test_data.batch(params["batch_size"]))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for class_idx in range(num_classes):
            # Extract true labels and predicted probabilities for the current class
            y_true = y_test_one_hot[:, class_idx]
            y_prob = y_pred_prob[:, class_idx]

            # Calculate ROC-AUC for the current class
            fpr[class_idx], tpr[class_idx], _ = roc_curve(y_true, y_prob)
            roc_auc[class_idx] = auc(fpr[class_idx], tpr[class_idx])

        auc_df[f'seed_{seed}'] = list(roc_auc.values())

    print(" ============   AUCS ===========================")

    print(f'AUCs df : {auc_df}')

    numeric_cols = auc_df.select_dtypes(include='number').iloc[:, 0:]
    print(f'Five run Aucs : {round(numeric_cols.mean(axis=0),3)}')

    print(" ============= mean of auce per disease over 5 run: ======= \n")
    print(round(numeric_cols.mean(axis=1), 3))

    print(" ====== confidence interval of auce per disease over 5 run: ==== \n")

    print(round(1.96 * numeric_cols.std(axis=1) / np.sqrt(5), 3))

    # endregion

    print("======= done =====")


if __name__ == '__main__':
    main()
