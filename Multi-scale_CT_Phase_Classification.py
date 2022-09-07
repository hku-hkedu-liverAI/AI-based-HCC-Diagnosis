import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import multi_gpu_model
import keras
import cv2
from keras.optimizers import Adam, SGD
import random
from scipy import ndimage
import tensorflow as tf
import numpy as np
from keras.layers import Conv3D, Input, MaxPool3D, BatchNormalization, Dense, GlobalAveragePooling3D
from keras.models import Model


def input_data_reshape(input_x, expected_height, expected_width):
    N, _, _, D = input_x.shape
    resized_input = np.zeros((N, expected_height, expected_width, D))
    for i in range(N):
        temp = input_x[i]          # [H, W, D]
        temp_resize = np.zeros((expected_height, expected_width, D))
        for j in range(D):
            temp_d = temp[:, :, j]
            resized = cv2.resize(temp_d, (expected_height, expected_width))
            temp_resize[:, :, j] = resized
        resized_input[i] = temp_resize

    return resized_input


def multi_scale_classification_model(width, height, depth, num_class):
    inputs = Input((width, height, depth, 1))
    resized_inputs = input_data_reshape(inputs, expected_height=height//2, expected_width=width//2)

    x = Conv3D(filters=32, kernel_size=3, activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = MaxPool3D(pool_size=2)(x)
    x = Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPool3D(pool_size=2)(x)
    x = Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPool3D(pool_size=2)(x)
    # This convolutional is for testing the effectiveness of adding new layer
    x = Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPool3D(pool_size=2)(x)
    x = GlobalAveragePooling3D()(x)

    x1 = Conv3D(filters=32, kernel_size=3, activation="relu")(resized_inputs)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool3D(pool_size=2)(x1)
    x1 = Conv3D(filters=64, kernel_size=3, activation="relu")(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool3D(pool_size=2)(x1)
    x1 = Conv3D(filters=128, kernel_size=3, activation="relu")(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool3D(pool_size=2)(x1)
    # This convolutional is for testing the effectiveness of adding new layer
    x1 = Conv3D(filters=256, kernel_size=3, activation="relu")(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool3D(pool_size=2)(x1)
    x1 = GlobalAveragePooling3D()(x1)

    print(x.shape, x1.shape, "Line-65")
    x = Dense(units=256, activation="relu")(x)

    outputs = Dense(units=num_class, activation='softmax')(x)
    model = Model(inputs, outputs, name="3dcnn")

    return model


model = multi_scale_classification_model(width=256, height=256, depth=128, num_class=3)
model.summary()

