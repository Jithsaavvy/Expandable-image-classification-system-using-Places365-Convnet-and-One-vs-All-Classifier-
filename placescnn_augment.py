#!/usr/bin/env python3

"""
PlacesCNN model definition and training for 3 classes using data augmentation by means 
of ImageDataGenerator.

Note: Should perform better than the other model architecture(no data augmentation - placescnn.py).
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import losses, metrics
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config
from placescnn import create_placescnn_model, plot_graph

if not config.model_dir.exists():
    config.model_dir.mkdir(parents=True)

train_dir = Path(config.train_path)
val_dir = Path(config.val_path)
train_image_count = len(list(train_dir.glob('*/*.jpg')))
val_image_count = len(list(val_dir.glob('*/*.jpg')))
STEPS_PER_EPOCH_TRAIN = np.ceil(train_image_count/config.BATCH_SIZE)
STEPS_PER_EPOCH_VAL = np.ceil(val_image_count/config.BATCH_SIZE)
print('Train images:',train_image_count)
print('Validation images:',val_image_count)

def generator(train_path, val_path):
    """Function to convert the images into generators that facilitates in data augmentation

    Parameters
    ----------
    train_path: str
        Path for training data
    val_path: str
        Path for validation data

    Returns
    -------
    Generators: <iterator>
        Train, valid. It is also recommended for test images as well.
    """
    train_generator_instance = ImageDataGenerator(rescale=1./255, rotation_range=10, width_shift_range=0.1,
                                                  height_shift_range=0.1, shear_range=0.1,zoom_range=0.1)

    val_generator_instance = ImageDataGenerator(rescale=1./255)

    train_generator = train_generator_instance.flow_from_directory(train_path,
                                                        target_size=(227, 227),
                                                        batch_size=config.BATCH_SIZE,
                                                        color_mode="rgb",
                                                        class_mode="categorical",
                                                        seed=1,
                                                        shuffle=True
                                                        )

    val_generator = val_generator_instance.flow_from_directory(val_path,
                                                        target_size=(227, 227),
                                                        batch_size=config.BATCH_SIZE,
                                                        color_mode="rgb",
                                                        class_mode="categorical",
                                                        seed=1,
                                                        shuffle=True
                                                        )

    return train_generator, val_generator

train_generator, val_generator = generator(config.train_path, config.val_path)
model = create_placescnn_model(config.input_shape)
model.compile(optimizer = tf.keras.optimizers.Adam(0.001), 
              loss=losses.CategoricalCrossentropy(),
              metrics=[metrics.CategoricalAccuracy()])
model.summary()

history = model.fit(train_generator, epochs=config.EPOCHS, steps_per_epoch=STEPS_PER_EPOCH_TRAIN,
                    validation_data=val_generator, validation_steps=STEPS_PER_EPOCH_VAL, verbose=0)

#Save the trained model to the respective directory. Change the directory based on the file system
model.save(config.model_dir / config.model_name)

#Evaluate the model
loss, accuracy = model.evaluate(val_generator)
print("Accuracy:{:.2f}%".format(accuracy*100))

#Plotting train accuracy vs validation accuracy
fig = plt.figure(figsize=(10, 6))
plot_graph(range(1, len(history.epoch)+1), history.history['categorical_accuracy'], label='Train Accuracy')
plot_graph(range(1, len(history.epoch)+1), history.history['val_categorical_accuracy'], label='Validation Accuracy')
plt.legend()

#Plotting training loss vs validation loss
plt.figure(figsize=(10, 6))
plot_graph(range(1, len(history.epoch)+1), history.history['loss'], label='Train loss')
plot_graph(range(1, len(history.epoch)+1), history.history['val_loss'], label='Validation loss')
plt.legend()

