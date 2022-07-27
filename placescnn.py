#!/usr/bin/env python3

"""
PlacesCNN model definition and training for 3 classes

Note: PlacesCNN uses AlexNet architecture
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras import losses, metrics
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from preprocess_data import pre_process
import config

if not config.model_dir.exists():
    config.model_dir.mkdir(parents=True)

def create_placescnn_model(input_shape):
    """Function to create placesCNN model architecture using keras.

    Parameters
    ----------
    input_shape: tuple
         (227,227,3)

    Returns
    -------
    model
    """
    model = Sequential([
        #Layer 1
        Conv2D(96, kernel_size = (11,11),strides=(4,4), activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        MaxPool2D(pool_size=(3,3), strides=(2,2)),
        
        #Layer 2
        Conv2D(256, kernel_size=(5,5),strides=(1,1), activation='relu',padding="same"),
        tf.keras.layers.BatchNormalization(),
        MaxPool2D(pool_size=(3,3), strides=(2,2)),
        
        #Layer 3
        Conv2D(384, kernel_size=(3,3),strides=(1,1), activation='relu',padding="same"),
        tf.keras.layers.BatchNormalization(),
        
        #Layer 4
        Conv2D(384, kernel_size=(3,3),strides=(1,1), activation='relu',padding="same"),
        tf.keras.layers.BatchNormalization(),
        
        #Layer 5
        Conv2D(256, kernel_size=(3,3),strides=(1,1), activation='relu',padding="same"),
        tf.keras.layers.BatchNormalization(),
        MaxPool2D(pool_size=(3,3), strides=(2,2)),
        
        #Layer 6
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        
        #Layer 7
        Dense(4096, activation='relu'),
        Dropout(0.5),
        
        #Layer 8
        Dense(3, activation='softmax')
    ])
    
    return model

def plot_graph(X, y, format = '-', label=''):
    """Helper function to visualize the training paradigms

    Parameters
    ----------
    X: range
        X-axis values
    y: range
        Y-axis epoch values with training parameter values.
        Eg: Accuracy, loss etc

    Returns
    -------
    None  
    """
    plt.plot(X, y, format, label=label)
    plt.xlabel("Epochs")
    plt.ylabel("Values")
    plt.grid(True)

x_train, y_train, x_test, y_test = pre_process(config.train_path, config.val_path)
model = create_placescnn_model(config.input_shape)
#Can try with different optimizers (SGD, AdaDelta etc. if time permits)
model.compile(optimizer = tf.keras.optimizers.Adam(0.001), 
              loss=losses.SparseCategoricalCrossentropy(),
              metrics=[metrics.SparseCategoricalAccuracy()])
model.summary()

#Start training
history = model.fit(x_train, y_train, batch_size=config.BATCH_SIZE, epochs=config.EPOCHS, validation_data=(x_test, y_test), shuffle=False)

#Save the trained model to the respective directory. Change the directory based on the file system
model.save(config.model_dir / config.model_name)

#Plotting train accuracy vs validation accuracy
fig = plt.figure(figsize=(10, 6))
plot_graph(range(1, len(history.epoch)+1), history.history['sparse_categorical_accuracy'], label='Train Accuracy')
plot_graph(range(1, len(history.epoch)+1), history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
plt.legend()

#Plotting training loss vs validation loss
plt.figure(figsize=(10, 6))
plot_graph(range(1, len(history.epoch)+1), history.history['loss'], label='Train loss')
plot_graph(range(1, len(history.epoch)+1), history.history['val_loss'], label='Validation loss')
plt.legend()
