#!/usr/bin/env python3

"""
Feature extraction from the trained placesCNN model
"""

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from preprocess_data import pre_process
import config

x_train, y_train, x_test, y_test = pre_process(config.train_path, config.val_path)

def extractor(x_train, y_train, x_test, y_test, pre_train=True):
    """
    Function to extract the features from the trained model

    Parameters
    ----------
    x_train, y_train, x_test, y_test: np arrays
        Preprocessed data
    pre_train:= bool
        True if pretrained model is loaded from the memory, else False.

    Returns
    -------
    None
    """
    if pre_train:
        m_path = config.model_dir / config.model_name
        pre_trained_model = tf.keras.models.load_model(m_path)
        pre_trained_model.summary()
    
    else:
        pre_trained_model = None
        # Use the random model directly after traininig instead of loading pretrained model
        # model = ....

    #Extracting the feature vector of size 4096 from the second last fully connected layer in the pretrained model
    model = Model(pre_trained_model.input, [pre_trained_model.get_layer('dense_1').output, pre_trained_model.output])
    features, labels = model.predict(x_train)
    print(len(features))
    #Labels can be ignored for the feature extraction part
    #labels = np.argmax(labels, axis=1)

    #Push the extracted features and labels(actual labels, not the predicted one) into CSV file 
    #to be used in the training of random forest classifier
    csv_instance = os.path.sep.join([config.csv_path,config.fname])
    csv = open(csv_instance, "w")
    for (label, feature) in zip(y_train, features):
        feature = ",".join([str(v) for v in feature])
        csv.write("{},{}\n".format(label, feature))
    csv.close()

    print("Features and labels saved to csv successfully")

extractor(x_train, y_train, x_test, y_test, True)