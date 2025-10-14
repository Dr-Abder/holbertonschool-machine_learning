#!/usr/bin/env python3


from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    
    init = K.initializers.HeNormal(seed=0)
    X_input = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(64, (7, 7), (2, 2), padding='same',
                            kernel_initializer=init)(X_input)
    norm1 = K.layers.BatchNormalization(axis=3)(conv1)
    relu1 = K.layers.Activation('relu')(norm1)
    max_pool1 = K.layers.MaxPooling2D((3, 3), strides=(2, 2),
                                      padding='same')(relu1)

    conv2_x = projection_block(max_pool1, [64, 64, 256], s=1)
    conv2_xx = identity_block(conv2_x, [64, 64, 256,])
    conv2_xxx = identity_block(conv2_xx, [64, 64, 256])

    conv3_x = projection_block(conv2_xxx, [128, 128, 512], s=2)
    conv3_xx = identity_block(conv3_x, [128, 128, 512])
    conv3_xxx = identity_block(conv3_xx, [128, 128, 512])
    conv3_xxxx = identity_block(conv3_xxx, [128, 128, 512])

    conv4_x = projection_block(conv3_xxxx, [256, 256, 1024], s=2)
    conv4_xx = identity_block(conv4_x, [256, 256, 1024])
    conv4_xxx = identity_block(conv4_xx, [256, 256, 1024])
    conv4_xxxx = identity_block(conv4_xxx, [256, 256, 1024])
    conv4_xxxxx = identity_block(conv4_xxxx, [256, 256, 1024])
    conv4_xxxxxx = identity_block(conv4_xxxxx, [256, 256, 1024])

    conv5_x = projection_block(conv4_xxxxxx, [512, 512, 2048], s=2)
    conv5_xx = identity_block(conv5_x, [512, 512, 2048])
    conv5_xxx = identity_block(conv5_xx, [512, 512, 2048])

    avg_pool = K.layers.AveragePooling2D((7, 7))(conv5_xxx)
    X = K.layers.Dense(1000, activation='softmax')(avg_pool)
    model = K.models.Model(inputs=X_input, outputs=X)

    return model
