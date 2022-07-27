#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementing random forest classifier from the extracted features(.csv) from placesCNN
for place classification.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import metrics 
import numpy as np
import pandas as pd
import config

def convert_df_to_arrays(df):
    """Function to process the dataframe by converting it as arrays comprising both vectors and labels 
    for training random forest.

    Parameters
    ----------
    df: pd.Dataframe
        Input dataframe

    Returns
    -------
    vectors: np array
        2D array of shape (no_of_samples, 4096)
    labels: np array
        1D array of shape (no_of_labels_for_each_sample)
    """
    vectors = df.iloc[:,1:].values
    labels = df[[0]].to_numpy()
    labels = np.squeeze(labels)
    return vectors, labels

path = config.csv_path / config.fname
features_df = pd.read_csv(path, header=None)
data, labels = convert_df_to_arrays(features_df)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.20)

classifier = RandomForestClassifier(n_estimators = 100)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#Got an initial accuracy of 33% with just 150 train images
print("Model accuracy: ", metrics.accuracy_score(y_test, y_pred))

print("Classification report:",classification_report(y_test, y_pred, target_names=config.target_names))
