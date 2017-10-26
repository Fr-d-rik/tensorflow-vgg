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

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        self.imagenet_mean = [123.68, 116.779, 103.939]  # in RGB format
        self.names = ['input', 'rgb_scaled', 'bgr_normed',
                      'conv1_1/lin', 'conv1_1/relu', 'conv1_2/lin', 'conv1_2/relu', 'pool1',
                      'conv2_1/lin', 'conv2_1/relu', 'conv2_2/lin', 'conv2_2/relu', 'pool2',
                      'conv3_1/lin', 'conv3_1/relu', 'conv3_2/lin', 'conv3_2/relu', 'conv3_3/lin', 'conv3_3/relu',
                      'pool3',
                      'conv4_1/lin', 'conv4_1/relu', 'conv4_2/lin', 'conv4_2/relu', 'conv4_3/lin', 'conv4_3/relu',
                      'pool4',
                      'conv5_1/lin', 'conv5_1/relu', 'conv5_2/lin', 'conv5_2/relu', 'conv5_3/lin', 'conv5_3/relu',
                      'pool5',
                      'fc6/lin', 'fc6/relu',
                      'fc7/lin', 'fc7/relu',
                      'fc8/lin', 'softmax']

    def build(self, rgb, rescale=1.0):

        rgb_scaled = tf.multiply(rgb, rescale, name='rgb_scaled')

        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)

        bgr_normed = tf.concat(axis=3, values=[blue - self.imagenet_mean[2],
                                               green - self.imagenet_mean[1],
                                               red - self.imagenet_mean[0]],
                               name='bgr_normed')

        conv1_1, _ = self.conv_layer(bgr_normed, "conv1_1")
        conv1_2, _ = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'pool1')

        conv2_1, _ = self.conv_layer(pool1, "conv2_1")
        conv2_2, _ = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'pool2')

        conv3_1, _ = self.conv_layer(pool2, "conv3_1")
        conv3_2, _ = self.conv_layer(conv3_1, "conv3_2")
        conv3_3, _ = self.conv_layer(conv3_2, "conv3_3")
        pool3 = self.max_pool(conv3_3, 'pool3')

        conv4_1, _ = self.conv_layer(pool3, "conv4_1")
        conv4_2, _ = self.conv_layer(conv4_1, "conv4_2")
        conv4_3, _ = self.conv_layer(conv4_2, "conv4_3")
        pool4 = self.max_pool(conv4_3, 'pool4')

        conv5_1, _ = self.conv_layer(pool4, "conv5_1")
        conv5_2, _ = self.conv_layer(conv5_1, "conv5_2")
        conv5_3, _ = self.conv_layer(conv5_2, "conv5_3")
        pool5 = self.max_pool(conv5_3, 'pool5')

        relu6, _ = self.fc_layer(pool5, "fc6")

        relu7, _ = self.fc_layer(relu6, "fc7")

        _, fc8 = self.fc_layer(relu7, "fc8")

        tf.nn.softmax(fc8, name="softmax")

        self.data_dict = None

    # noinspection PyUnresolvedReferences
    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = tf.constant(self.data_dict[name][0], name="filter")

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = tf.constant(self.data_dict[name][1], name="biases")
            conv_lin = tf.nn.bias_add(conv, conv_biases, name='lin')
            conv = tf.nn.relu(conv_lin, name='relu')
            return conv, conv_lin

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

    @staticmethod
    def max_pool(bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def build_partial(self, in_tensor, input_name, output_name=None, rescale=1.0):

        if 'lin' in input_name:
            in_tensor = tf.nn.relu(in_tensor)
            input_name = input_name.replace('lin', 'relu')

        names_to_build = [n for n in self.names if ('lin' not in n or 'fc8' in n)]
        assert input_name in names_to_build

        output_name = output_name or 'softmax'
        lin_out_option = [0] * 16
        if 'lin' in output_name:
            assert output_name.startswith('conv')
            lin_ids = ['-', '1_1', '1_2', '2_1', '2_2', '3_1', '3_2', '3_3',
                       '4_1', '4_2', '4_3', '5_1', '5_2', '5_3']
            lin_idx = lin_ids.index(output_name[4:7])
            lin_out_option[lin_idx] = 1
            output_name = output_name.replace('lin', 'relu')

        assert output_name in names_to_build

        build_ops = list()
        build_ops.append(lambda x: tf.multiply(x, rescale, name='rgb_scaled'))

        def rgb2bgr(x):
            rgb_normed = x - self.imagenet_mean
            red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_normed)
            return tf.concat(axis=3, values=[blue, green, red], name='bgr_normed')

        build_ops.append(rgb2bgr)

        build_ops.append(lambda x: self.conv_layer(x, "conv1_1")[lin_out_option[1]])
        build_ops.append(lambda x: self.conv_layer(x, "conv1_2")[lin_out_option[2]])
        build_ops.append(lambda x: self.max_pool(x, 'pool1'))

        build_ops.append(lambda x: self.conv_layer(x, "conv2_1")[lin_out_option[3]])
        build_ops.append(lambda x: self.conv_layer(x, "conv2_2")[lin_out_option[4]])
        build_ops.append(lambda x: self.max_pool(x, 'pool2'))

        build_ops.append(lambda x: self.conv_layer(x, "conv3_1")[lin_out_option[5]])
        build_ops.append(lambda x: self.conv_layer(x, "conv3_2")[lin_out_option[6]])
        build_ops.append(lambda x: self.conv_layer(x, "conv3_3")[lin_out_option[7]])
        build_ops.append(lambda x: self.max_pool(x, 'pool3'))

        build_ops.append(lambda x: self.conv_layer(x, "conv4_1")[lin_out_option[8]])
        build_ops.append(lambda x: self.conv_layer(x, "conv4_2")[lin_out_option[9]])
        build_ops.append(lambda x: self.conv_layer(x, "conv4_3")[lin_out_option[10]])
        build_ops.append(lambda x: self.max_pool(x, 'pool4'))

        build_ops.append(lambda x: self.conv_layer(x, "conv5_1")[lin_out_option[11]])
        build_ops.append(lambda x: self.conv_layer(x, "conv5_2")[lin_out_option[12]])
        build_ops.append(lambda x: self.conv_layer(x, "conv5_3")[lin_out_option[13]])
        build_ops.append(lambda x: self.max_pool(x, 'pool5'))

        build_ops.append(lambda x: self.fc_layer(x, "fc6")[lin_out_option[14]])
        build_ops.append(lambda x: self.fc_layer(x, "fc7")[lin_out_option[15]])
        build_ops.append(lambda x: self.fc_layer(x, "fc8")[1])

        build_ops.append(lambda x: tf.nn.softmax(x, name="softmax"))

        start_idx = names_to_build.index(input_name)
        end_idx = names_to_build.index(output_name)
        build_ops = build_ops[start_idx:end_idx]
        # print('building partial vgg:', names_to_build[start_idx + 1:end_idx + 1])
        temp_tensor = in_tensor
        for op in build_ops:
            temp_tensor = op(temp_tensor)
        out_tensor = temp_tensor

        self.data_dict = None
        return out_tensor
