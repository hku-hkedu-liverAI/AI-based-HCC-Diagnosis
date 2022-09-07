from keras.layers import *
from keras import backend as K
import keras
from keras.models import Model
import tensorflow as tf
import numpy as np


def attention_module_2d(net, attention_module):
    if attention_module == 'se_block':  # SE_Block
        net = se_block_2d(net)
    elif attention_module == 'cbam_block':
        net = cbam_block_2d(net)
    elif attention_module == 'eca_block':
        net = eca_block_2d(net)
    else:
        raise Exception("'{}' is not supported attention module!".format(attention_module))

    return net


def se_block_2d(input_feature, ratio=8):
    # suppose input_features: (B, 128, 128, 32], --> channels = 32
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    channel = input_feature._keras_shape[channel_axis]
    se_feature = GlobalAveragePooling2D()(input_feature)  # (B, H, W, C] ---> (B, C) where C: channels
    se_feature = Reshape((1, 1, channel))(se_feature)  # (B, C) --> (B, 1, 1, C]
    assert se_feature._keras_shape[1:] == (1, 1, channel)

    se_feature = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature._keras_shape[1:] == (1, 1, channel // ratio)

    se_feature = Dense(channel, activation='sigmoid', kernel_initializer='he_normal', use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature._keras_shape[1:] == (1, 1, channel)

    se_feature = multiply([input_feature, se_feature])  # (B, H, W, C) * (B, 1, 1, C) ---> (B, H, W, C)
    print(se_feature)
    return se_feature


def cbam_block_2d(cbam_feature, ratio=8):
    # cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = eca_block_2d(cbam_feature)
    cbam_feature = spatial_attention_2d(cbam_feature)
    return cbam_feature


def eca_block_2d(input_feature):
    avg_pool = GlobalAveragePooling2D()(input_feature)  # [B, H, W, Channels] --> [B, channels]
    unsqueezed_avg_pool = Lambda(lambda x: keras.backend.expand_dims(x, 1))(avg_pool)  # ---> [B, 1, channels]
    permuted_avg_pool = Permute((2, 1))(unsqueezed_avg_pool)  # ---> [B, channels, 1]

    # The output of Conv1D is [B, channels, 1]
    conv_avg_pool = Conv1D(1, kernel_size=3, strides=1, padding='same', activation='sigmoid')(permuted_avg_pool)
    permuted_conv_output = Permute((2, 1))(conv_avg_pool)  # ---> [B, 1, channels]

    unsqueezed = Lambda(lambda x: keras.backend.expand_dims(x, axis=1))(
        permuted_conv_output)  # ---> [B, 1, 1, channels]
    output_feature = multiply([input_feature, unsqueezed])  # --> Output: [B, H, W, Channels], e.g., (B, 128, 128, 32]

    return output_feature


def spatial_attention_2d(input_feature):
    kernel_size = 3
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    avg_pool = Lambda(lambda x: K.mean(x, axis=channel_axis, keepdims=True))(input_feature)
    max_gool = Lambda(lambda x: K.max(x, axis=channel_axis, keepdims=True))(input_feature)
    concat = keras.layers.Concatenate(axis=channel_axis)([avg_pool, max_gool])
    cbam_feature = Conv2D(filters=1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid',
                          kernel_initializer='he_normal', use_bias=False)(concat)
    # input_feature: (B, H, W, C) * (B, H, W, 1) ===> (B, H, W, C)

    return multiply([input_feature, cbam_feature])


def attention_module_3d(net, attention_module):
    if attention_module == 'se_block':  # SE_Block
        net = se_block_3d(net)
    elif attention_module == 'cbam_block':
        net = cbam_block_3d(net)
    elif attention_module == 'eca_block':
        net = eca_block_3d(net)
    else:
        raise Exception("'{}' is not supported attention module!".format(attention_module))

    return net


def se_block_3d(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    channel = input_feature._keras_shape[channel_axis]
    # print(channel_axis, channel)
    se_feature = GlobalAveragePooling3D()(input_feature)
    se_feature = Reshape((1, 1, 1, channel))(se_feature)
    assert se_feature._keras_shape[1:] == (1, 1, 1, channel)

    se_feature = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature._keras_shape[1:] == (1, 1, 1, channel // ratio)

    se_feature = Dense(channel, activation='sigmoid', kernel_initializer='he_normal', use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature._keras_shape[1:] == (1, 1, 1, channel)

    se_feature = multiply([input_feature, se_feature])
    return se_feature


def cbam_block_3d(cbam_feature, ratio=8):
    # cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = eca_block_3d(cbam_feature)
    cbam_feature = spatial_attention_3d(cbam_feature)
    return cbam_feature


def channel_attention_3d(input_feature, ratio=4):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    channel = input_feature._keras_shape[channel_axis]

    shared_layer_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel, activation='sigmoid', kernel_initializer='he_normal', use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling3D()(input_feature)
    avg_pool = Reshape((1, 1, 1, channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, 1, channel)

    max_pool = GlobalMaxPooling3D()(input_feature)
    max_pool = Reshape((1, 1, 1, channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, 1, channel)

    cbam_feature = keras.layers.Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return multiply([input_feature, cbam_feature])


def spatial_attention_3d(input_feature):
    kernel_size = 3
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    avg_pool = Lambda(lambda x: K.mean(x, axis=channel_axis, keepdims=True))(input_feature)
    max_gool = Lambda(lambda x: K.max(x, axis=channel_axis, keepdims=True))(input_feature)
    concat = keras.layers.Concatenate(axis=channel_axis)([avg_pool, max_gool])
    cbam_feature = Conv3D(filters=1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid',
                          kernel_initializer='he_normal', use_bias=False)(concat)

    return multiply([input_feature, cbam_feature])


def eca_block_3d(input_feature):
    avg_pool = GlobalAveragePooling3D()(input_feature)  # [B, H, W, D, Channels] --> [B, channels]
    unsqueezed_avg_pool = Lambda(lambda x: keras.backend.expand_dims(x, 1))(avg_pool)  # ---> [B, 1, channels]
    permuted_avg_pool = Permute((2, 1))(unsqueezed_avg_pool)  # ---> [B, channels, 1]
    # squeezed_avg_pool_w = Lambda(lambda x: keras.backend.squeeze(x, 1))(squeezed_avg_pool_h)  # ---> [B, 1, channels]

    # The output of Conv1D is [B, channels, 1]
    conv_avg_pool = Conv1D(1, kernel_size=3, strides=1, padding='same', activation='sigmoid')(permuted_avg_pool)
    permuted_conv_output = Permute((2, 1))(conv_avg_pool)  # ---> [B, channels, 1]

    unsqueezed_w = Lambda(lambda x: keras.backend.expand_dims(x, axis=1))(
        permuted_conv_output)  # ---> [B, 1, channels, 1]

    unsqueezed_h = Lambda(lambda x: keras.backend.expand_dims(x, axis=1))(unsqueezed_w)  # ---> [B, 1, 1, channels, 1]

    output_feature = multiply([input_feature, unsqueezed_h])
    return output_feature


def reshape_3d_to_2d(inputs, depth):
    def get_slice(x, h1, h2):
        return x[:, :, :, h1:h2, :]

    # [B, H, W, D, 1] ---> [B*D, H, W, 1, 1]
    for i in range(depth):
        if i == 0:  # The first slice
            inputs_2d = Lambda(lambda x: get_slice(x, h1=0, h2=1))(inputs)
        else:
            inputs_slices = Lambda(lambda x: get_slice(x, h1=i, h2=i + 1))(inputs)
            inputs_2d = concatenate([inputs_2d, inputs_slices], axis=0)

    # [B*D, H, W, 1, 1]  ---> [B*D, H, W, 1]  after squeeze operation
    inputs_2d = Lambda(lambda x: keras.backend.squeeze(x, axis=-1))(inputs_2d)
    # print(inputs_2d.shape, "Line-188")
    return inputs_2d


def restore_2d_to_3d(inputs, depth, batch_size):
    for i in range(depth):
        if i == 0:
            inputs_retored = Lambda(lambda x: x[i:i + batch_size])(inputs)
        else:
            outputs_slice = Lambda(lambda x: x[i * batch_size: i * batch_size + batch_size])(inputs)
            inputs_retored = concatenate([inputs_retored, outputs_slice], axis=-1)

    return inputs_retored


def reshape_3d_to_p3d(inputs, depth):
    def get_slice(x, h1, h2):
        return x[:, :, :, h1:h2, :]

    # [B, H, W, D, 1] ---> [B*s, H, W, 4, 1], where = int((depth-slice_num)/slice_stride) + 1
    slice_num = 4
    slice_stride = 2
    num_batch = int((depth - slice_num) / slice_stride) + 1
    for i in range(0, num_batch):
        if i == 0:  # The first slice
            inputs_p3d = Lambda(lambda x: get_slice(x, h1=0, h2=4))(inputs)
        else:
            inputs_slices = Lambda(lambda x: get_slice(x, h1=i * 2, h2=i * 2 + 4))(inputs)
            inputs_p3d = concatenate([inputs_p3d, inputs_slices], axis=0)

    # [B*s, H, W, 1, 1]  ---> [B*D, H, W, 1]  after squeeze operation ---> (B*s, 128, 128, 4, 1)
    inputs_p3d = Lambda(lambda x: keras.backend.squeeze(x, axis=-1))(inputs_p3d)  # (B*s, 128, 128, 4)
    return inputs_p3d


def restore_p3d_to_3d(inputs, depth, batch_size):
    overlap_slice_num = 2  # this depends on the setting in reshape_3d_to_p3d
    for i in range(depth):
        if i == 0:
            inputs_retored = Lambda(lambda x: x[i:i + batch_size])(inputs)
        else:
            outputs_slice = Lambda(lambda x: x[i * batch_size + overlap_slice_num: i * batch_size + batch_size])(inputs)
            inputs_retored = concatenate([inputs_retored, outputs_slice], axis=-1)

    return inputs_retored


# The model of the encoding path of the first branch for (B, 128, 128, 128, 1): 3D inputs
def conv3d_module_from_3D_input(input, factor, attention_type='eca_block', stage=4, conv_block=2):
    for stage_iter in range(stage):  # stage_iter: 0, 1, 2, 3
        for conv_iter in range(conv_block):  # conv_iter: 0, 1
            if stage_iter == 0 and conv_iter == 0:
                inputs = input
            else:
                inputs = x_3d
            filter_num = 8 * factor * 2 ** (stage_iter)
            x_3d = Conv3D(filters=filter_num, kernel_size=3, strides=1)(inputs)
            x_3d = Activation('relu')(x_3d)
            x_a_3d = attention_module_3d(x_3d, attention_module=attention_type)
            x_3d = keras.layers.add([x_3d, x_a_3d])
            x_3d = BatchNormalization()(x_3d)
            if conv_block == 1:
                x_3d = MaxPool3D(pool_size=(2, 2, 2))(x_3d)

    x_3d_gap = GlobalAveragePooling3D()(x_3d)
    return x_3d_gap


def conv2D_module_from_3D_input(input, factor, attention_type='eca_block', stage=4, conv_block=3):
    for stage_iter in range(stage):  # stage_iter = 0, 1, 2, 3
        for conv_iter in range(conv_block):  # conv_iter = 0, 1, 2. In each stage, there are 3 conv2D
            if stage_iter == 0 and conv_iter == 0:
                inputs = input
            else:
                inputs = x_2d
            filter_num = 8 * factor * 2 ** (stage_iter)
            x_2d = Conv2D(filters=filter_num, kernel_size=3, strides=1)(inputs)
            x_2d = Activation('relu')(x_2d)
            x_a_2d = attention_module_2d(x_2d, attention_type)
            x_2d = keras.layers.add([x_2d, x_a_2d])
            x_2d = BatchNormalization()(x_2d)
            if conv_iter == conv_block - 1:
                if input.shape[1] == 128:
                    x_2d = MaxPool2D(pool_size=(2, 2))(x_2d)
                else:
                    x_2d = MaxPool2D(pool_size=(3, 3))(x_2d)

    return x_2d


def convp3d_module_from_3D_input(input, factor, attention_type='eca_block', stage=4, conv_block=2):
    for stage_iter in range(stage):
        for conv_iter in range(conv_block):
            if stage_iter == 0 and conv_iter == 0:
                inputs = input
            else:
                inputs = x_p3d

            filter_num = 8 * factor * 2 ** (stage_iter)
            x_p3d = Conv2D(filters=filter_num, kernel_size=3, strides=1)(inputs)
            x_p3d = Activation('relu')(x_p3d)
            x_a_p3d = attention_module_2d(x_p3d, attention_type)
            x_p3d = keras.layers.add([x_p3d, x_a_p3d])
            x_p3d = BatchNormalization()(x_p3d)
            if conv_iter == 1:
                x_p3d = MaxPool2D(pool_size=(2, 2))(x_p3d)

    return x_p3d


def Conv2d_Module_from_3Dto2D_Input_Classifier(width, height, factor, attention_type='eca_block',
                                               stage=4, conv_block=2, num_class=2, threshold=5):
        # 3D input: (B, 256/128, 256/128, 128, 1) ---> 2D input: (B*128, 256/128, 256/128, 1)
        # In this setting, the label should be stacked as slices, otherwise there will be undesirable results
        input_orig = Input(shape=(width, height, 1))
        for stage_iter in range(stage):
            for conv_iter in range(conv_block):
                if stage_iter == 0 and conv_iter == 0:
                    inputs = input_orig
                else:
                    inputs = x_2d
                filter_num = 8 * factor * 2 ** (stage_iter)
                x_2d = Conv2D(filters=filter_num, kernel_size=3, strides=1)(inputs)
                x_2d = Activation('relu')(x_2d)
                x_a_2d = attention_module_2d(x_2d, attention_type)
                x_2d = keras.layers.add([x_2d, x_a_2d])
                x_2d = BatchNormalization()(x_2d)
                if conv_iter == 1:
                    # ---> (B*128, H_o, W_o, Num_channels=Filters_in_Last_Conv_Layer]
                    x_2d = MaxPool2D(pool_size=(2, 2))(x_2d)

        # For 2D Classification at the slice level
        x_2d_flatten_reshape = Lambda(lambda x: keras.backend.reshape(x, shape=(-1, 128, x.shape[1]*x.shape[2]*x.shape[3])))(x_2d)
        x_2d_prediction_slice = []
        for idx in range(x_2d_flatten_reshape.shape[1]):
            x_2d_flatten_reshape_slice = x_2d_flatten_reshape[:, idx, :]
            x_2d_fcn1 = Dense(units=1024, activation="relu")(x_2d_flatten_reshape_slice)
            x_2d_fcn1_drop = Dropout(rate=0.25)(x_2d_fcn1)
            x_2d_fcn2 = Dense(units=1024, activation="relu")(x_2d_fcn1_drop)
            if num_class == 2:
                outputs_class = Dense(units=1, activation="sigmoid")(x_2d_fcn2)
            else:
                outputs_class = Dense(units=num_class, activation="softmax")(x_2d_fcn2)
            x_2d_prediction_slice.append(outputs_class)

        x_2d_prediction_slice_to_tensor = tf.compat.v1.convert_to_tensor(x_2d_prediction_slice)            # (128, ?, 1]
        x_2d_prediction_slice_to_tensor = keras.backend.squeeze(x_2d_prediction_slice_to_tensor, axis=-1)  # (128, ?)
        # x_2d_prediction_slice_to_tensor = tf.transpose(x_2d_prediction_slice_to_tensor, perm=[1, 0])     # (?, 128)
        x_2d_prediction_slice_to_tensor = Lambda(lambda x:
                                                 keras.backend.permute_dimensions(x, (1, 0)))(x_2d_prediction_slice_to_tensor)

        x_2d_prediction_slice_to_tensor_top = tf.compat.v1.nn.top_k(x_2d_prediction_slice_to_tensor, threshold)[0]
        print(x_2d_prediction_slice_to_tensor_top, "Line-343")
        # x_2d_prediction_case = tf.reduce_mean(x_2d_prediction_slice_to_tensor_top, axis=-1)
        x_2d_prediction_case = keras.backend.mean(x_2d_prediction_slice_to_tensor_top, axis=-1)

        x_2d_prediction_case_to_tensor = Lambda(lambda x: keras.backend.expand_dims(x, axis=-1))(x_2d_prediction_case)
        print(x_2d_prediction_case_to_tensor.shape, x_2d_prediction_slice_to_tensor.shape, "Line-349")

        # outputs = keras.layers.Concatenate(axis=-1)([x_2d_prediction_slice_to_tensor, x_2d_prediction_case_to_tensor])
        outputs = keras.layers.concatenate([x_2d_prediction_slice_to_tensor, x_2d_prediction_case_to_tensor],
                                           axis=-1)

        model = Model(inputs=input_orig, outputs=outputs)

        return model


def Convp3d_Module_from_3DtoP3D_Input_Classifier(width, height, factor, attention_type='eca_block',
                                                 stage=4, conv_block=2, num_class=2, threshold=5):
    input_orig = Input(shape=(width, height, 1))
    for stage_iter in range(stage):
        for conv_iter in range(conv_block):
            if stage_iter == 0 and conv_iter == 0:
                inputs = input_orig
            else:
                inputs = x_p3d

            filter_num = 8 * factor * 2 ** (stage_iter)
            x_p3d = Conv2D(filters=filter_num, kernel_size=3, strides=1)(inputs)
            x_p3d = Activation('relu')(x_p3d)
            x_a_p3d = attention_module_2d(x_p3d, attention_type)
            x_p3d = keras.layers.add([x_p3d, x_a_p3d])
            x_p3d = BatchNormalization()(x_p3d)
            if conv_iter == 1:
                x_p3d = MaxPool2D(pool_size=(2, 2))(x_p3d)

    print(x_p3d.shape, "line-392")
    x_p3d_flatten_reshape = Lambda(
        lambda x: keras.backend.reshape(x, shape=(-1, 63, x.shape[1] * x.shape[2] * x.shape[3])))(x_p3d)
    print(x_p3d_flatten_reshape.shape, "Line-396")
    x_p3d_prediction_slice = []
    for idx in range(x_p3d_flatten_reshape.shape[1]):
        x_2d_flatten_reshape_slice = x_p3d_flatten_reshape[:, idx, :]
        x_2d_fcn1 = Dense(units=1024, activation="relu")(x_2d_flatten_reshape_slice)
        x_2d_fcn1_drop = Dropout(rate=0.25)(x_2d_fcn1)
        x_2d_fcn2 = Dense(units=1024, activation="relu")(x_2d_fcn1_drop)
        if num_class == 2:
            outputs_class = Dense(units=1, activation="sigmoid")(x_2d_fcn2)
        else:
            outputs_class = Dense(units=num_class, activation="softmax")(x_2d_fcn2)
        x_p3d_prediction_slice.append(outputs_class)

    x_p3d_prediction_slice_to_tensor = tf.compat.v1.convert_to_tensor(x_p3d_prediction_slice)  # (63, ?, 1]
    x_p3d_prediction_slice_to_tensor = keras.backend.squeeze(x_p3d_prediction_slice_to_tensor, axis=-1)  # (63, ?)
    # x_2d_prediction_slice_to_tensor = tf.transpose(x_2d_prediction_slice_to_tensor, perm=[1, 0])     # (?, 63)
    x_p3d_prediction_slice_to_tensor = Lambda(lambda x:
                                             keras.backend.permute_dimensions(x, (1, 0)))(
        x_p3d_prediction_slice_to_tensor)

    x_p3d_prediction_slice_to_tensor_top = tf.compat.v1.nn.top_k(x_p3d_prediction_slice_to_tensor, threshold)[0]
    print(x_p3d_prediction_slice_to_tensor_top, "Line-417")
    # x_2d_prediction_case = tf.reduce_mean(x_2d_prediction_slice_to_tensor_top, axis=-1)
    x_p3d_prediction_case = keras.backend.mean(x_p3d_prediction_slice_to_tensor_top, axis=-1)

    x_p3d_prediction_case_to_tensor = Lambda(lambda x: keras.backend.expand_dims(x, axis=-1))(x_p3d_prediction_case)
    print(x_p3d_prediction_case_to_tensor.shape, x_p3d_prediction_slice_to_tensor.shape, "Line-349")

    # outputs = keras.layers.Concatenate(axis=-1)([x_2d_prediction_slice_to_tensor, x_2d_prediction_case_to_tensor])
    outputs = keras.layers.concatenate([x_p3d_prediction_slice_to_tensor, x_p3d_prediction_case_to_tensor],
                                       axis=-1)

    model = Model(inputs=input_orig, outputs=outputs)

    return model


model_resized = Convp3d_Module_from_3DtoP3D_Input_Classifier(height=128, width=128, factor=4, num_class=2)
model_original = Convp3d_Module_from_3DtoP3D_Input_Classifier(height=256, width=256, factor=4, num_class=2)

model_2d_classification = Model(inputs=[model_resized.inputs, model_original.inputs], 
                                outputs=[model_resized.outputs, model_original.outputs])
