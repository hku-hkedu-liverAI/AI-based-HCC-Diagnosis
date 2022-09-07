from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Add, concatenate, GlobalAveragePooling2D
from keras.layers import Flatten, Dense, BatchNormalization, Activation, ZeroPadding2D, AveragePooling2D
from keras.models import Model
from lib.custom_layers import Scale
from keras import backend as K
bn_axis = -1
Height, Width = 256, 256
K.common.set_image_dim_ordering('tf')


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x


def conv_block_res(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    shortcut = Conv2D(filters3, (1, 1), strides=strides,name=conv_name_base + '1')(input_tensor)

    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x


def FCN16(length=128, width=128, classes=1):
    inputs = Input(shape=(length, width, 1), name='inputs')

    # Block 1:
    block1_conv1 = Conv2D(8, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    block1_conv2 = Conv2D(8, (3, 3), activation='relu', padding='same', name='block1_conv2')(block1_conv1)
    block1_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(block1_conv2)

    # Block 2:
    block2_conv1 = Conv2D(16, (3, 3), activation='relu', padding='same', name='block2_conv1')(block1_pool)
    block2_conv2 = Conv2D(16, (3, 3), activation='relu', padding='same', name='block2_conv2')(block2_conv1)
    block2_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(block2_conv2)

    # Block 3:
    block3_conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='block3_conv1')(block2_pool)
    block3_conv2 = Conv2D(32, (3, 3), activation='relu', padding='same', name='block3_conv2')(block3_conv1)
    block3_conv3 = Conv2D(32, (3, 3), activation='relu', padding='same', name='block3_conv3')(block3_conv2)
    block3_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(block3_conv3)

    # Block 4:
    block4_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block4_conv1')(block3_pool)
    block4_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block4_conv2')(block4_conv1)
    block4_conv3 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block4_conv3')(block4_conv2)
    block4_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(block4_conv3)

    # Block 5:
    # block5_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(block4_pool)
    # block5_conv2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(block5_conv1)
    # block5_conv3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(block5_conv2)
    # block5_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(block5_conv3)

    # Fully convolutional layers: chang the number of filters 4096 to 1024, and the strides (7, 7) to (3, 3)
    # y = Conv2D(512, (7, 7), activation='relu', padding='same', name='fcn1')(block5_pool)
    # y = Conv2D(512, (7, 7), activation='relu', name='fcn1')(block2_pool)
    # fcn1 = Dropout(0.5)(fcn1)
    y = Flatten()(block4_pool)
    # y = Dense(units=1028, activation='relu')(y)
    y = Dense(units=512, activation='relu')(y)
    # y = Dense(units=128, activation='relu')(y)
    outputs = Dense(classes, activation='softmax', kernel_initializer='he_normal')(y)
    model = Model(input=inputs, output=outputs)

    return model


def ResNet_50_Clas(Length=128, Width=128, num_classes=1):
    factor = 1
    inputs = Input((Length, Width, 1), name='input_1')
    x = ZeroPadding2D((3, 3))(inputs)
    x = Conv2D(64//factor, (7, 7), strides=(2, 2), name='conv1')(x)
    x0 = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x0 = Activation('relu')(x0)
    x1 = MaxPooling2D(strides=(2, 2))(x0)

    conv2 = conv_block_res(x1, 3, [64//factor, 64//factor, 256//factor], stage=2, block='a', strides=(1, 1))
    iden2 = identity_block(conv2, 3, [64//factor, 64//factor, 256//factor], stage=2, block='b')
    iden2 = identity_block(iden2, 3, [64//factor, 64//factor, 256//factor], stage=2, block='c')

    conv3 = conv_block_res(iden2, 3, [128//factor, 128//factor, 512//factor], stage=3, block='a')
    iden3 = identity_block(conv3, 3, [128//factor, 128//factor, 512//factor], stage=3, block='b')
    iden3 = identity_block(iden3, 3, [128//factor, 128//factor, 512//factor], stage=3, block='c')
    iden3 = identity_block(iden3, 3, [128//factor, 128//factor, 512//factor], stage=3, block='d')

    conv4 = conv_block_res(iden3, 3, [256//factor, 256//factor, 1024//factor], stage=4, block='a')
    iden4 = identity_block(conv4, 3, [256//factor, 256//factor, 1024//factor], stage=4, block='b')
    iden4 = identity_block(iden4, 3, [256//factor, 256//factor, 1024//factor], stage=4, block='c')
    iden4 = identity_block(iden4, 3, [256//factor, 256//factor, 1024//factor], stage=4, block='d')
    iden4 = identity_block(iden4, 3, [256//factor, 256//factor, 1024//factor], stage=4, block='e')
    iden4 = identity_block(iden4, 3, [256//factor, 256//factor, 1024//factor], stage=4, block='f')

    conv5 = conv_block_res(iden4, 3, [512//factor, 512//factor, 2048//factor], stage=5, block='a')
    iden5 = identity_block(conv5, 3, [512//factor, 512//factor, 2048//factor], stage=5, block='b')
    iden5 = identity_block(iden5, 3, [512//factor, 512//factor, 2048//factor], stage=5, block='c')

    # without final layer FCN4096
    # with 64 as the number of filters
    # iden3, the training sample size is 800, the testing accuracy is 0.71845
    # iden4, the training sample size is 800, the testing accuracy is 0.70158
    # iden5, the training sample size is 800, the testing accuracy is 0.69685
    # with 32 as the number of filters
    # 5 iden3, training sample size is 800, testing accuracy is
    # 5 iden4, training sample size is 800, testing accuracy is 0.71667, with 3 ident4, the testing accuracy is 0.69455
    # with 1 iden4, testing accuracy is 0.71717
    # 5 iden5, training sample size is 800, testing accuracy is 0.68659,

    # without final layer FCN4096
    # with 16 as the number of filters,
    # with ident5, the training sample size is 900, testing accuracy is 0.70193
    # with 32 as the number of filters
    # with iden5, the training sample size is 900, testing accuracy is 0.71255
    # with 64 as the number of filters
    # with iden5, the training sample size is 900, testing accuracy is 0.71550
    # with iden4, the training sample size is 900, testing accuracy is 0.70445

    # with final layer FCN4096
    # with iden5, training samole is 800, testing accuracy is 0.51850
    # without final layer FCN1024,
    # with iden5, training sample is 800, testing accuracy is 0.67693
    # without

    y = AveragePooling2D(pool_size=8)(iden4)
    y = Flatten()(y)
    # y = Dense(1024, activation='softmax', kernel_initializer='he_normal')(y)
    if num_classes == 2:
        num_neurons = 1
        activation_function = 'sigmoid'
    else:
        num_neurons = num_classes
        activation_function = 'softmax'

    outputs = Dense(units=num_neurons, activation=activation_function, kernel_initializer='he_normal')(y)
    model = Model(inputs, outputs)
    return model


def DenseNet_Clas(nb_dense_block=4, growth_rate=32, nb_filter=64, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4,
              classes=1000):
    eps = 1.1e-5
    compression = 1.0 - reduction
    global concat_axis

    if K.common.image_dim_ordering() == 'tf':
        concat_axis = -1
        img_input = Input(shape=(Height, Width, 1), name='data')
    else:
        concat_axis = 1
        img_input = Input(shape=(1, Height, Width), name='data')

    nb_filter = nb_filter
    nb_layers = [6, 12, 24, 16]

    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(nb_filter, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block-1):
        stage = block_idx + 2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter*compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_block_bn')(x)
    x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_block_scale')(x)
    x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)
    x = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)

    # y = Dense(64, activation='softmax', kernel_initializer='he_normal')(y)
    if classes == 2:
        units = 1
        activation = 'sigmoid'
    else:
        units = classes
        activation = 'softmax'
    outputs = Dense(units=units, activation=activation, name='fc_final')(x)          # For binary classification only
    # DenseNet:
    # DenseNet_Clas(nb_dense_block=4, growth_rate=32, nb_filter=64, reduction=0, dropout_rate=0, classes=2)
    # training accuracy is 0.8814, testing accuracy is 0.71071
    # DenseNet_Clas(nb_dense_block=4, growth_rate=32, nb_filter=32, reduction=0, dropout_rate=0, classes=2)
    # training accuracy is 0.8820, testing accuracy is 0.72496
    # DenseNet_Clas(nb_dense_block=4, growth_rate=15, nb_filter=32, reduction=0, dropout_rate=0, classes=2)
    # training accuracy is 0.8866, testing accuracy is 0.74919
    model = Model(inputs=img_input, outputs=outputs)

    return model


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    concat_axis = -1

    concat_feat = x
    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = concatenate([concat_feat, x], axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1e-4):
    eps = 1.1e-4
    concat_axis = -1

    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Conv2D(int(nb_filter*compression), (1, 1), name=conv_name_base, use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def conv_block(x, stage, branch, nb_filter, drouput_rate=None, weight_decay=1e-4):
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)
    concat_axis = -1

    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Conv2D(inter_channel, (1, 1), name=conv_name_base+'_x1', use_bias=False)(x)

    if drouput_rate:
        x = Dropout(drouput_rate)(x)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Conv2D(nb_filter, (3, 3), name=conv_name_base+'_x2', use_bias=False)(x)

    if drouput_rate:
        x = Dropout(drouput_rate)(x)

    return x

# model = FCN16(length=256, width=256, classes=2)
# model = ResNet_50_Clas(Length=256, Width=256, num_classes=2)
# model = DenseNet_Clas(nb_dense_block=4, growth_rate=48, nb_filter=96, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, classes=2)
# model.summary()