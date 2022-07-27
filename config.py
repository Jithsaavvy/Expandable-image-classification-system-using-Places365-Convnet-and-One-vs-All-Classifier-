#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pathlib
from pathlib import Path

# Change the directory based on the file system
train_path = "./dataset/train"
val_path = "./dataset/test"

method = "normal_training"
this_dir = Path.cwd()
model_dir = this_dir / "saved_models" / method
model_name = "model.h5"
input_shape = (227,227,3)
BATCH_SIZE = 32
#Epochs can be changed to any arbitary value. The more epochs we train, more better the model learns
EPOCHS = 10

csv_path = './results'
fname = 'extracted_features.csv'
target_names = ['class 0', 'class 1', 'class 2']



