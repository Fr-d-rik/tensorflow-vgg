import inspect
import os
import numpy as np
import tensorflow as tf


class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print(path)

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        self.imagenet_mean = [123.68, 116.779, 103.939]  # in RGB format

    def build(self, rgb, rescale=255.0):

        rgb_scaled = tf.multiply(rgb, rescale, name='rgb_scaled')

        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)

        bgr_normed = tf.concat(axis=3, values=[blue - self.imagenet_mean[2],
                                               green - self.imagenet_mean[1],
                                               red - self.imagenet_mean[0]],
                               name='bgr_normed')

        conv1_1 = self.conv_layer(bgr_normed, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = max_pool(conv1_2, 'pool1')

        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = max_pool(conv2_2, 'pool2')

        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        pool3 = max_pool(conv3_3, 'pool3')

        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        pool4 = max_pool(conv4_3, 'pool4')

        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        pool5 = max_pool(conv5_3, 'pool5')

        fc6 = self.fc_layer(pool5, "fc6")
        assert fc6.get_shape().as_list()[1:] == [4096]
        relu6 = tf.nn.relu(fc6)

        fc7 = self.fc_layer(relu6, "fc7")
        relu7 = tf.nn.relu(fc7)

        fc8 = self.fc_layer(relu7, "fc8")

        tf.nn.softmax(fc8, name="prob")

        self.data_dict = None

    # noinspection PyUnresolvedReferences
    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = tf.constant(self.data_dict[name][0], name="filter")

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = tf.constant(self.data_dict[name][1], name="biases")
            conv_lin = tf.nn.bias_add(conv, conv_biases, name='lin')
            conv = tf.nn.relu(conv_lin, name='relu')
            return conv

    # noinspection PyUnresolvedReferences
    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = tf.constant(self.data_dict[name][0], name="weights")
            biases = tf.constant(self.data_dict[name][1], name="biases")

            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc


def max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
