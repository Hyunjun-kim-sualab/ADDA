from contextlib import ExitStack
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from adda.models import register_model_fn


@register_model_fn('svhnnet_sua')
def svhnnet_sua(inputs, scope='svhnnet_sua', is_training=True, reuse=False):
    layers = OrderedDict()
    net = inputs
    with tf.variable_scope(scope, reuse=reuse):
        with ExitStack() as stack:
            stack.enter_context(
                slim.arg_scope(
                    [slim.fully_connected, slim.conv2d],
                    activation_fn=tf.nn.relu,
                    weights_regularizer=slim.l2_regularizer(2.5e-5)))
            stack.enter_context(
                slim.arg_scope([slim.max_pool2d, slim.conv2d],
                               padding='SAME'))
            net = slim.conv2d(net, 32, [3, 3], scope='conv1_1')
            net = slim.conv2d(net, 32, [3, 3], scope='conv1_2')
            net = slim.conv2d(net, 32, [3, 3], scope='conv1_3')
            net = slim.max_pool2d(net, 2, stride=1, scope='pool1')
            layers['pool1'] = net
            net = slim.conv2d(net, 64, [3, 3], scope='conv2_1')
            net = slim.conv2d(net, 64, [3, 3], scope='conv2_2')
            net = slim.conv2d(net, 64, [3, 3], scope='conv2_3')
            net = slim.max_pool2d(net, 2, stride=1, scope='pool2')
            layers['pool2'] = net
            net = slim.conv2d(net, 128, 5, scope='conv3_1')
            net = slim.conv2d(net, 128, 5, scope='conv3_2')
            net = slim.conv2d(net, 128, 5, scope='conv3_3')
            net = slim.max_pool2d(net, 2, stride=1, scope='pool3')
            layers['pool3'] = net
            net = tf.contrib.layers.flatten(net)

            with slim.arg_scope([slim.fully_connected], normalizer_fn=None):
                net = slim.fully_connected(net, 128, scope="fc1")
                net = slim.fully_connected(net, 10, activation_fn=None, scope="fc2")


    return net, layers
svhnnet_sua.default_image_size = 32
svhnnet_sua.num_channels = 3
svhnnet_sua.range = 255
svhnnet_sua.mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
svhnnet_sua.bgr = False