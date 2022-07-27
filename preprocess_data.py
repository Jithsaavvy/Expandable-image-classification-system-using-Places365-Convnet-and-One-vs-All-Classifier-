#!/usr/bin/env python3

"""
Data loading and preprocessing of image files
"""

import numpy as np
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path
import cv2
import os
from PIL import Image

def pre_process(train_path, val_path):
    """Function to preprocess the training and validation data.

    Parameters
    ----------
    train_path: str
        Path for training data
    val_path: str
        Path for validation data

    Returns
    -------
    Set of preprocessed data for PlacesCNN model: np array
        x_train, y_train, x_test, y_test
    """
    random_seed = 42
    classes = os.listdir(train_path)
    print("Class names: {0} \n Total no of classes: {1}".format(classes, len(classes)))

    train_data, train_labels = load_data(classes, train_path)
    print("Total no. of train images: {0} \n Total no. of train labels: {1}".format(len(train_data),len(train_labels)))
    visualize_images(train_data, train_labels)
    
    val_data, val_labels = load_data(classes, val_path)
    print("Total no. of val images: {0} \n Total no. of val labels: {1}".format(len(val_data),len(val_labels)))
    visualize_images(val_data, val_labels)

    x_train, y_train = process(train_data, train_labels, random_seed)
    x_test, y_test = process(val_data, val_labels, random_seed)
    print("X_train: {0}, y_train: {1}".format(x_train.shape, y_train.shape))
    print("X_test: {0}, y_test: {1}".format(x_test.shape, y_test.shape))

    return x_train, y_train, x_test, y_test


def load_data(classes, path):
    """Helper function that loads image data with its respective labels. Images are resized to (227,227,3)
    to be used in PlacesCNN network

    Parameters
    ---------
    classes: list 
        Respective image classes
    path: str
        Can be train/val path

    Returns
    -------
    np array of images and labels
    """
    images, labels = [],[]
    for i, category in enumerate(classes):
        for image_name in os.listdir(path+"/"+category):
            img = cv2.imread(path+"/"+category+"/"+image_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_array = Image.fromarray(img, 'RGB')
            #Resizing images to 227 x 227
            resized_img = img_array.resize((227, 227))
            images.append(np.array(resized_img))
            labels.append(i)
    return np.array(images), np.array(labels)

def visualize_images(images, labels):
    """Helper function to visualize some sample images.

    Parameters
    ---------
    images: np array 
        Train/val images to be visualized.
    labels: np array
        Respective image labels

    Returns
    -------
    None
    """
    plt.figure(1 , figsize = (19 , 10))
    n = 0
    #Visualizing only 6 random images from the set
    for i in range(6):
        n = n+1 
        r = np.random.randint(0 , images.shape[0] , 1)
        plt.subplot(3 , 3 , n)
        plt.subplots_adjust(hspace = 0.3 , wspace = 0.3)
        plt.imshow(images[r[0]])
        plt.title('Places Categories : {}'.format(labels[r[0]]))
        plt.xticks([])
        plt.yticks([])

    plt.show()
    
def process(images, labels, r_seed):
    """Helper function to shuffle, randomize and normalize the images.

    Parameters
    ---------
    images: np array 
        Train/val images to be visualized.
    labels: np array
        Respective image labels
    r_seed: int
        Seeding of random number generator

    Returns
    -------
    Preprocessed images: np array
        images, labels
    """
    n = np.arange(images.shape[0])
    np.random.seed(r_seed)
    np.random.shuffle(n)
    image_shuffled = images[n]
    labels_shuffled = labels[n]
    images = image_shuffled.astype(np.float32)
    labels = labels_shuffled.astype(np.int32)
    images = images/255
    return images, labels


