#!/usr/bin/env python3
"""
Version de LeNet-5 avec une architecture en utilisant tensorflow.
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def lenet5(x, y):


    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    conv1 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)(x)

    pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    conv2 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)(pool1)

    pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    flatten = tf.layers.Flatten()(pool2)

    fc1 = tf.layers.Dense(units=120, activation=tf.nn.relu,
                          kernel_initializer=initializer)(flatten)

    fc2 = tf.layers.Dense(units=84, activation=tf.nn.relu,
                          kernel_initializer=initializer)(fc1)

    logits = tf.layers.Dense(units=10, kernel_initializer=initializer)(fc2)
    y_pred = tf.nn.softmax(logits)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)

    train_op = tf.train.AdamOptimizer().minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return y_pred, train_op, loss, accuracy
