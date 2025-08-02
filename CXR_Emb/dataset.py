
import numpy as np
import tensorflow as tf
def make_chexpert_dataset(df_train, df_validate, df_test, labels_Columns):
    
    # Create training and validation Datasets
    
    train_features = np.stack(df_train["features"].to_numpy())
    train_labels = df_train[labels_Columns].to_numpy()

    training_data = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    for train in training_data.take(2):
        print(f"training dataset : {train}")

     # Create validation Dataset
     
    validation_features = np.stack(df_validate["features"].to_numpy())
    validation_labels = df_validate[labels_Columns].to_numpy()
    
    validation_data = tf.data.Dataset.from_tensor_slices((validation_features, validation_labels))
    
    for val in validation_data.take(2):
        print(f"validation dataset : {val}")
        
    # Create test Dataset
    test_features = np.stack(df_test["features"].to_numpy())
    test_labels = df_test[labels_Columns].to_numpy()
    
    test_data = tf.data.Dataset.from_tensor_slices((test_features, test_labels))
        
    for test in test_data.take(2):
        print(f"test dataset : {test}")

    return training_data, validation_data, test_data