import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import multi_gpu_model
from CT_Scans_Data_Augmentation import train_preprocessing, validation_preprocessing
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import cv2
from keras.optimizers import Adam, SGD
# from Classification_Model3D import get_model_CABM
from MS3DCN_Utils import multi_scale_get_model_DCN
from scipy import ndimage


class CT3D_DataLoader_For_Clas(keras.utils.Sequence):
    def __init__(self, batch_size=16, image_depth=64, image_size=(128, 128), input_data_label_path=None,
                 num_class=2, training_or_testing=''):
        self.batch_size = batch_size
        self.image_depth = image_depth
        self.image_size = image_size
        self.input_data_label_path = input_data_label_path
        self.num_class = num_class
        self.training_or_testing = training_or_testing

    def __len__(self):
        return len(self.input_data_label_path)//self.batch_size

    def __getitem__(self, idx):
        self.indexes = np.arange(len(self.input_data_label_path))

        np.random.shuffle(self.indexes)
        self.labels_shuffled = self.input_data_label_path[self.indexes]

        i = idx * self.batch_size
        batch_label_original = self.labels_shuffled[i:i+self.batch_size]
        batch_label = [int(batch_label_original[i][1]) for i in range(len(batch_label_original))]

        if self.training_or_testing == 'training':
            batch_input_data = [] #[np.load(batch_label_original[i][0]) for i in range(len(batch_label_original))]
            for idx in range(len(batch_label_original)):
                tmp = np.load(batch_label_original[idx][0])
                tmp = (tmp - (np.min(tmp))) / (0.5 * (np.max(tmp) - np.min(tmp))) - 1.0
                # tmp = tmp / 127.5 - 1.0
                volume = resize_volume(tmp)
                tmp = np.expand_dims(np.transpose(volume, [2, 1, 0]), axis=0)
                # tmp_resize = input_data_reshape(tmp, expected_height=256, expected_width=256)
                batch_input_data.append(tmp)
        else:
            batch_input_data = [] # [np.load(batch_label_original[i][0]) for i in range(len(batch_label_original))]
            for idx in range(len(batch_label_original)):
                tmp = np.load(batch_label_original[idx][0])
                # tmp = tmp / 127.5 - 1.0
                tmp = (tmp - (np.min(tmp))) / (0.5 * (np.max(tmp) - np.min(tmp))) - 1.0
                volume = resize_volume(tmp)
                tmp = np.expand_dims(np.transpose(volume, [2, 1, 0]), axis=0)
                # tmp_resize = input_data_reshape(tmp, expected_height=256, expected_width=256)
                batch_input_data.append(tmp)

        batch_input_data = np.array(batch_input_data).reshape((len(batch_input_data), 128, 128, 128))
        # batch_input_data_resized = input_data_reshape(batch_input_data, expected_height=128, expected_width=128)

        if self.training_or_testing == 'training':
            x, y = train_preprocessing(volume=batch_input_data, label=batch_label)
        else:
            x, y = validation_preprocessing(volume=batch_input_data, label=batch_label)
        if self.num_class == 2:
            y = y
        elif self.num_class > 2:
            y = keras.utils.to_categorical(y, self.num_class)
        return x, y


def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 128
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    # img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


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


data_train_label = np.load(os.path.join('/home/ra1/original', 'P2_P3_RegCT_HCC_Train.npy'))
data_test_label = np.load(os.path.join('/home/ra1/original', 'P2_P3_RegCT_HCC_Test.npy'))


print("Number of samples in training and validation are: %d and %d" % (len(data_train_label), len(data_test_label)))
batch_size = 2
image_depth, image_rows, image_cols = 128, 256, 256
train_gen = CT3D_DataLoader_For_Clas(batch_size=batch_size, image_depth=image_depth, image_size=(image_rows, image_cols),
                                  input_data_label_path=data_train_label, num_class=2,
                                     training_or_testing='training')
val_gen = CT3D_DataLoader_For_Clas(batch_size=batch_size, image_depth=image_depth, image_size=(image_rows, image_cols),
                                   input_data_label_path=data_test_label, num_class=2,
                                   training_or_testing='validation or testing')

# Build model.
# model = get_model_CABM(width=256, height=256, depth=128, num_class=2)
model = multi_scale_get_model_DCN(width=128, height=128, depth=128, batch_size=1, factor=2, num_class=2)
try:
    model = multi_gpu_model(model, gpus=3, cpu_merge=True, cpu_relocation=False)
    print("Training using multiple GPUs..")
except:
    print("Training using single GPU or CPU..")


# Compile model.

# data_rootpath = '/home/wenming/GANModels/Models_Classification_Phase'
data_rootpath = '/home/ra1/Documents'
# model.load_weights(os.path.join(data_rootpath, 'AllData_Clas_Phase_Model/Phase2_3_RegCT_HCC_Attention_One_GPU.h5'))
# model.load_weights(os.path.join(data_rootpath, 'AllData_Clas_Phase_Model/Phase2_3_RegCT_HCC_Attention_MS3DCN_Single_GPU.h5'))
model.load_weights(
    os.path.join(data_rootpath,
                 'AllData_Clas_Phase_Model/Phase2_3_RegCT_HCC_Attention_MS3DCN_Single_GPU.h5'))
model.compile(loss="binary_crossentropy", optimizer=SGD(lr=0.01), metrics=["accuracy"])
if not os.path.exists(os.path.join(data_rootpath, 'AllData_Clas_Phase_Model')):
    os.makedirs(os.path.join(data_rootpath, 'AllData_Clas_Phase_Model'))

# Define callbacks.
checkpoint_cb = ModelCheckpoint(
    os.path.join(data_rootpath,
                 'AllData_Clas_Phase_Model/Phase2_3_RegCT_HCC_Attention_MS3DCN_Single_GPU.h5'),
monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
early_stopping_cb = EarlyStopping(monitor="val_loss", min_delta=0, patience=150, verbose=1, mode='auto')
callbacks_list = [checkpoint_cb, early_stopping_cb]
# model.load_weights(os.path.join(data_rootpath, 'AllData_Clas_Phase_Model/Phase3_RegCT_HCC_Att.h5'))
# Train the model, doing validation at the end of each epoch
epochs = 250

model.fit(train_gen, epochs=epochs, validation_data=val_gen, verbose=1, workers=4,
          callbacks=callbacks_list)
if not os.path.exists(os.path.join(data_rootpath, 'AllData_Clas_Phase_Model')):
    os.makedirs(os.path.join(data_rootpath, 'AllData_Clas_Phase_Model'))
model.save(os.path.join(data_rootpath, 'AllData_Clas_Phase_Model/RegCT_Phase2_3_HCC_Classification_Model_Att_MS3DCN_Single_GPU.h5'))



