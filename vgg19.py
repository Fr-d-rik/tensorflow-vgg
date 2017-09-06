import os
import inspect
import numpy as np
import tensorflow as tf


class Vgg19:
    def __init__(self, vgg19_npy_path=None):
        if vgg19_npy_path is None:
            path = inspect.getfile(Vgg19)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg19.npy")
            vgg19_npy_path = path
            
        self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        self.imagenet_mean = [123.68, 116.779, 103.939]  # in RGB format

    def build(self, rgb, rescale=1.0, pool_mode='max'):

        rgb_scaled = tf.multiply(rgb, rescale, name='rgb_scaled')

        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)

        bgr_normed = tf.concat(axis=3, values=[blue - self.imagenet_mean[2],
                                               green - self.imagenet_mean[1],
                                               red - self.imagenet_mean[0]],
                               name='bgr_normed')

        conv1_1 = self.conv_layer(bgr_normed, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.pool(conv1_2, 'pool1', pool_mode)

        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.pool(conv2_2, 'pool2', pool_mode)

        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        conv3_4 = self.conv_layer(conv3_3, "conv3_4")
        pool3 = self.pool(conv3_4, 'pool3', pool_mode)

        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        conv4_4 = self.conv_layer(conv4_3, "conv4_4")
        pool4 = self.pool(conv4_4, 'pool4', pool_mode)

        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        conv5_4 = self.conv_layer(conv5_3, "conv5_4")
        pool5 = self.pool(conv5_4, 'pool5', pool_mode)

        relu6, _ = self.fc_layer(pool5, "fc6")

        relu7, _ = self.fc_layer(relu6, "fc7")

        _, fc8 = self.fc_layer(relu7, "fc8")

        tf.nn.softmax(fc8, name="prob")

        self.data_dict = None

    def pool(self, bottom, name, mode):
        assert mode in ('max', 'avg')
        if mode == 'max':
            return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
        else:
            return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    # noinspection PyUnresolvedReferences
    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = tf.constant(self.data_dict[name][0], name="filter")

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = tf.constant(self.data_dict[name][1], name="biases")
            bias = tf.nn.bias_add(conv, conv_biases, name='lin')
            relu = tf.nn.relu(bias, name='relu')
            return relu

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
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases, name='lin')
            relu = tf.nn.relu(fc, name='relu')
            return relu, fc
