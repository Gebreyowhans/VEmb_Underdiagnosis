import pandas as pd
from IPython.display import clear_output
import io
import os
import glob
import zipfile
import shutil
import math

import numpy as np
import random as python_random
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.optimize import curve_fit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import seaborn as sns

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
import sklearn.metrics as sklm

from sklearn.metrics import roc_auc_score, roc_curve, auc

SEEDS = [19, 31, 38, 47, 77]

number_of_runs = 5
signficance_level = 1.96


def get_optimal_hyperparameters():
    # Optimal hyperparameters
    hyperparameters = {
        "embeddings_size": 512,
        "learning_rate": 0.0001,
        "batch_size": 48,
        "epochs": 50,
        "end_lr_factor": 1.0,
        "dropout": 0.3,
        "decay_steps": 1000,
        "loss_weights": None,
        "weight_decay": 0.00001,
        "hidden_layer_sizes": [768, 256]
    }
    return hyperparameters

# Function to cast elements to float64


def cast_to_float64(features, label):
    features_casted = tf.dtypes.cast(features, tf.float64)
    return features_casted, label


def make_dataset(df_train, df_validate, df_test, labels_Columns):
    # Create training Dataset
    training_data = tf.data.Dataset.from_tensor_slices(
        (df_train.emb.values.tolist(), df_train[labels_Columns].values))

    # Create validation Dataset
    validation_data = tf.data.Dataset.from_tensor_slices((df_validate.emb.values.tolist(),
                                                         df_validate[labels_Columns].values))

    # Create validation Dataset
    test_data = tf.data.Dataset.from_tensor_slices((df_test.emb.values.tolist(),
                                                    df_test[labels_Columns].values))

    return training_data, validation_data, test_data


def get_callbacks(patience=5, fold=1, save_model_dir="./biomedclip-embedding/"):
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                  mode="min",
                                                  patience=patience,
                                                  restore_best_weights=True)

    saved_model = tf.keras.callbacks.ModelCheckpoint(f'{save_model_dir}/model_{fold}.keras',
                                                     monitor='val_auc',
                                                     mode='max',
                                                     save_freq='epoch',
                                                     save_best_only=True,
                                                     save_weights_only=False,
                                                     verbose=1)
    callbacks_list = [early_stop, saved_model]

    return callbacks_list


def build_model(params, labels_Columns=[]):

    # Define the input layer
    inputs = Input(shape=(params["embeddings_size"],))

    # Build the model using the Functional API
    hidden = inputs

    hidden_layer_sizes = params["hidden_layer_sizes"]

    for size in hidden_layer_sizes:
        hidden = Dense(size,
                       activation='relu',
                       kernel_initializer=HeUniform())(hidden)

        hidden = BatchNormalization()(hidden)
        hidden = Dropout(params["dropout"])(hidden)

    output = Dense(len(labels_Columns), activation='sigmoid',
                   kernel_initializer=HeUniform())(hidden)

    # Create the model
    model = Model(inputs=inputs, outputs=output)
    # Compile the model with AUC as the metric
    model.compile(optimizer=Adam(learning_rate=params["learning_rate"]),
                  loss='binary_crossentropy',
                  metrics=[AUC(multi_label=True)])

    return model


def make_pred_multilabel(model, test_dataset, validation_data, test_df, seed_number=19, labels_Columns=[], thersholds={}):

    # Load  pre-trained Keras model
    PRED_LABELS = labels_Columns

    for mode in ["Threshold", "test"]:
        # Create empty dataframes
        pred_df = pd.DataFrame(columns=["path"])
        bi_pred_df = pd.DataFrame(columns=["path"])
        true_df = pd.DataFrame(columns=["path"])

        if mode == "Threshold":
            data = validation_data  # Use validation dataset for threshold mode
            Eval_df = pd.DataFrame(columns=["label", 'bestthr'])
            thrs = []

        if mode == "test":
            data = test_dataset  # Use test dataset for test mode
            index = 0
            TestEval_df = pd.DataFrame(columns=["label", 'auc'])

            # Load threshold values from the CSV
            Eval = pd.DataFrame()
            Eval = thersholds[f"Threshold_{seed_number}"]

            thrs = [Eval["bestthr"][Eval[Eval["label"] == label].index[0]]
                    for label in PRED_LABELS]

            print("thrs :", thrs)

        for inputs, labels in data:
            # Get predictions from the pre-trained model
            # inputs = tf.constant(inputs, dtype=tf.float32)
            true_labels = labels.numpy()
            model.set_weights(model.get_weights())

            """ perform inference
              ... use tf.expand_dims(inputs, axis=0) adds an extra dimension at axis 0,
              ... effectively creating a batch of size 1. """

            inputs_batched = tf.expand_dims(inputs, axis=0)
            outputs = model(inputs_batched, training=False)
            probs = outputs.numpy()

            thisrow = {}
            bi_thisrow = {}
            truerow = {}

            thisrow['path'] = 'path'
            truerow['path'] = 'path'

            if mode == "test":
                path = test_df.iloc[index]['path']
                bi_thisrow['path'] = path

            # Iterate over each entry in prediction vector; each corresponds to an individual label
            for j, _label in enumerate(PRED_LABELS):
                thisrow["prob_" + _label] = probs[0][j]
                truerow[_label] = true_labels[j]

                if mode == "test":
                    bi_thisrow["bi_" + _label] = probs[0][j] >= thrs[j]

            pred_df = pd.concat(
                [pred_df, pd.DataFrame([thisrow])], ignore_index=True)
            true_df = pd.concat(
                [true_df, pd.DataFrame([truerow])], ignore_index=True)

            if mode == "test":
                # Explicitly cast object-dtype columns with boolean values to bool
                bool_columns = [col_name for col_name in bi_pred_df.columns if col_name != 'path'
                                and bi_pred_df[col_name].dtype == 'object']
                bi_pred_df[bool_columns] = bi_pred_df[bool_columns].astype(
                    bool)

                # Append the dictionary to the DataFrame
                bi_pred_df = pd.concat(
                    [bi_pred_df, pd.DataFrame([bi_thisrow])], ignore_index=True)
                index = index+1

        # print(f'Last indext in thershold : {index}')
        for column in true_df:
            if column not in PRED_LABELS:
                continue
            actual = true_df[column]
            pred = pred_df["prob_" + column]

            thisrow = {}
            thisrow['label'] = column
            if mode == "test":
                bi_pred = bi_pred_df["bi_" + column]
                thisrow['auc'] = np.nan
                thisrow['accuracy'] = np.nan
                thisrow['auprc'] = np.nan
            else:
                thisrow['bestthr'] = np.nan

            try:
                if mode == "test":
                    # Calculate the AUC using the true labels and predicted probabilities
                    thisrow['auc'] = sklm.roc_auc_score(
                        actual.astype(int), pred)

                    thisrow['auprc'] = sklm.average_precision_score(
                        actual.astype(int), pred)

                    # Calculate accuracy
                    thisrow['accuracy'] = sklm.accuracy_score(
                        actual.astype(int), bi_pred)
                else:

                    p, r, t = sklm.precision_recall_curve(
                        actual.astype(int), pred)

                    # Calculate F1-score, handling division by zero
                    f1_scores = []
                    for precision, recall in zip(p, r):
                        if precision + recall == 0:
                            f1_scores.append(0.0)  # Handle division by zero
                        else:
                            f1_scores.append(
                                2 * (precision * recall) / (precision + recall))

                    # Find the threshold that maximizes F1-score
                    best_threshold = t[np.argmax(f1_scores)]

                    thrs.append(best_threshold)
                    thisrow['bestthr'] = best_threshold

            except BaseException as be:
                # Handle the exception
                print(
                    f'can not caclucalte AUC and Accuracy for  : {str(column)}, see the error : {str(be)}')

            if mode == "Threshold":
                Eval_df = pd.concat(
                    [Eval_df, pd.DataFrame([thisrow])], ignore_index=True)

            if mode == "test":
                TestEval_df = pd.concat(
                    [TestEval_df, pd.DataFrame([thisrow])], ignore_index=True)

        if mode == "Threshold":

            Eval_df = Eval_df.reset_index(drop=True)
            thersholds[f"Threshold_{seed_number}"] = Eval_df

        if mode == "test":
            TestEval_df = TestEval_df.reset_index(drop=True)

    avg_AUC = TestEval_df['auc'].sum()/14.0
    avg_accuracy = TestEval_df['accuracy'].sum() / 14.0

    print(f"Seed : {seed_number} AUC avg: {round(avg_AUC, 2)}")
    print(f" Seed : {seed_number} Accuracy avg: {round(avg_accuracy, 2)}")
    print("done")

    return TestEval_df, bi_pred_df


def calculate_avg_aucs(key, df):

    numeric_cols = df.select_dtypes(include='number').iloc[:, 0:]

    avg_auc = round(numeric_cols.mean(axis=0).mean(), 3)

    avg_ci = round(
        signficance_level * numeric_cols.mean(axis=0).std()/np.sqrt(number_of_runs), 3)

    print(" ================= mean of auce per disease over 5 run: =============")
    mean_auc_labels = round(numeric_cols.mean(axis=1), 3)

    print("confidence interval of auce per disease over 5 run:")
    mean_auc_ci = round(signficance_level * numeric_cols.std(axis=1) /
                        np.sqrt(number_of_runs), 3)

    return avg_auc, avg_ci, mean_auc_labels, mean_auc_ci


def training_and_evaluation(seed, base_path, df_train, df_validate, df_test, labels_Columns, thersholds, debug):
    np.random.seed(seed)
    python_random.seed(seed)
    tf.random.set_seed(seed)

    # prediction_results_path = os.path.join(base_path+"/predictions/")
    prediction_results_path = os.path.join(base_path, "predictions")
    os.makedirs(os.path.dirname(prediction_results_path), exist_ok=True)

    training_data, validation_data, test_dataset = make_dataset(
        df_train=df_train, df_validate=df_validate, df_test=df_test, labels_Columns=labels_Columns)

    df_validate.to_csv("./biomedclip-embedding/df_validate.csv", index=False)
    df_test.to_csv("./biomedclip-embedding/df_test.csv", index=False)

    # Create directory model weightes saving
    checkpoint_path = os.path.join(base_path, "checkpoint")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    params = get_optimal_hyperparameters()

    model = build_model(params, labels_Columns=labels_Columns)

    epochs = params["epochs"]
    if debug:
        epochs = 10

    # train the model
    history = model.fit(x=training_data.batch(params["batch_size"]).prefetch(tf.data.AUTOTUNE).cache(),
                        validation_data=validation_data.batch(
                            params["batch_size"]).cache(), callbacks=get_callbacks(patience=5, fold=1, save_model_dir=checkpoint_path), epochs=epochs)

    # Evaluate using ROC-AUC
    y_pred_prob = model.predict(test_dataset.batch(params["batch_size"]))

    # Test
    test_eval, bipred = make_pred_multilabel(model, test_dataset,
                                             validation_data, df_test, seed_number=seed, labels_Columns=labels_Columns, thersholds=thersholds)

    print("============================= Training complete      ===========================")

    return test_eval, bipred
