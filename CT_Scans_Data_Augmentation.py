import random
from scipy import ndimage
import tensorflow as tf
import numpy as np


def data_augmentation(image3d):
    image_3d = image3d
    flip_num = np.random.randint(0, 8)
    if flip_num == 1:
        image_3d = np.flipud(image_3d)  # Flip array in the up/down direction.symmetric
    elif flip_num == 2:
        image_3d = np.fliplr(image_3d)  # # Flip array in the left/right direction.
    elif flip_num == 3:
        image_3d = np.rot90(image_3d, k=1, axes=(1, 2))  # 256, 256, 3
    elif flip_num == 4:
        image_3d = np.rot90(image_3d, k=3, axes=(1, 2))
    elif flip_num == 5:
        image_3d = np.fliplr(image_3d)
        image_3d = np.rot90(image_3d, k=1, axes=(1, 2))
    elif flip_num == 6:
        image_3d = np.fliplr(image_3d)
        image_3d = np.rot90(image_3d, k=3, axes=(1, 2))
    elif flip_num == 7:
        image_3d = np.flipud(image_3d)
        image_3d = np.fliplr(image_3d)
    return image_3d


def data_augmentation_lesion(data_volume, mask_volume, mask_volume_resize):
    image_3d, mask_3d, mask_3d_resize = data_volume, mask_volume, mask_volume_resize
    flip_num = np.random.randint(0, 8)
    if flip_num == 1:
        image_3d = np.flipud(image_3d)  # Flip array in the up/down direction.symmetric
        mask_3d = np.flipud(mask_3d)
        mask_3d_resize = np.flipud(mask_3d_resize)
    elif flip_num == 2:
        image_3d = np.fliplr(image_3d)  # # Flip array in the left/right direction.
        mask_3d = np.fliplr(mask_3d)
        mask_3d_resize = np.fliplr(mask_3d_resize)
    elif flip_num == 3:
        image_3d = np.rot90(image_3d, k=1, axes=(1, 2))  # 256, 256, 3
        mask_3d = np.rot90(mask_3d, k=1, axes=(1, 2))
        mask_3d_resize = np.rot90(mask_3d_resize, k=1, axes=(1, 2))
    elif flip_num == 4:
        image_3d = np.rot90(image_3d, k=3, axes=(1, 2))
        mask_3d = np.rot90(mask_3d, k=3, axes=(1, 2))
        mask_3d_resize = np.rot90(mask_3d_resize, k=3, axes=(1, 2))
    elif flip_num == 5:
        image_3d = np.fliplr(image_3d)
        image_3d = np.rot90(image_3d, k=1, axes=(1, 2))
        mask_3d = np.fliplr(mask_3d)
        mask_3d = np.rot90(mask_3d, k=1, axes=(1, 2))
        mask_3d_resize = np.fliplr(mask_3d_resize)
        mask_3d_resize = np.rot90(mask_3d_resize, k=1, axes=(1, 2))
    elif flip_num == 6:
        image_3d = np.fliplr(image_3d)
        image_3d = np.rot90(image_3d, k=3, axes=(1, 2))
        mask_3d = np.fliplr(mask_3d)
        mask_3d = np.rot90(mask_3d, k=3, axes=(1, 2))
        mask_3d_resize = np.fliplr(mask_3d_resize)
        mask_3d_resize = np.rot90(mask_3d_resize, k=3, axes=(1, 2))
    elif flip_num == 7:
        image_3d = np.flipud(image_3d)
        image_3d = np.fliplr(image_3d)
        mask_3d = np.flipud(mask_3d)
        mask_3d = np.fliplr(mask_3d)
        mask_3d_resize = np.flipud(mask_3d_resize)
        mask_3d_resize = np.fliplr(mask_3d_resize)

    return image_3d, mask_3d, mask_3d_resize


def scipy_rotate(volume):
    angles = [-90, -80, -70, -60, -50, -40, -30, -20, -10, -5, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    angle = random.choice(angles)
    volume = ndimage.rotate(volume, angle, reshape=False)
    volume[volume<0] = 0
    volume[volume>1] = 1
    return volume


def rotate_ct_scans(volume):
    def scipy_rotate(volume):
        angles = [-20, -10, -5, 5, 10, 20]
        angle = random.choice(angles)
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume<0] = 0
        volume[volume>1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    # volume = rotate_ct_scans(volume)
    # volume = scipy_rotate(volume)
    if len(volume.shape) == 4:
        volume = np.expand_dims(volume, axis=-1)
    else:
        volume = volume

    volume = data_augmentation(volume)

    return volume, label


def validation_preprocessing(volume, label):
    if len(volume.shape) == 4:
        volume = np.expand_dims(volume, axis=-1)
    else:
        volume = volume

    return volume, label


def train_preprocessing_lesion(volume, mask_volume, mask_volume_resize, label):
    # volume = rotate_ct_scans(volume)
    # volume = scipy_rotate(volume)
    if len(volume.shape) == 4:
        volume = np.expand_dims(volume, axis=-1)
    else:
        volume = volume

    if len(mask_volume.shape) == 4:
        mask_volume = np.expand_dims(mask_volume, axis=-1)
    else:
        mask_volume = mask_volume

    if len(mask_volume_resize.shape) == 4:
        mask_volume_resize = np.expand_dims(mask_volume_resize, axis=-1)
    else:
        mask_volume_resize = mask_volume_resize

    data_volume, mask_volume, mask_volume_resize = data_augmentation_lesion(data_volume=volume,
                                                                            mask_volume=mask_volume,
                                                                            mask_volume_resize=mask_volume_resize)

    return data_volume, mask_volume, mask_volume_resize, label


def validation_preprocessing_lesion(volume, mask_volume, mask_volume_resize, label):
    if len(volume.shape) == 4:
        volume = np.expand_dims(volume, axis=-1)
    else:
        volume = volume

    if len(mask_volume.shape) == 4:
        mask_volume = np.expand_dims(mask_volume, axis=-1)
    else:
        mask_volume = mask_volume

    if len(mask_volume_resize) == 4:
        mask_volume_resize = np.expand_dims(mask_volume_resize, axis=-1)
    else:
        mask_volume_resize = mask_volume_resize

    return volume, mask_volume, mask_volume_resize, label

