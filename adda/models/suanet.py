from contextlib import ExitStack
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from adda.models import register_model_fn


@register_model_fn('suanet')
def suanet(inputs, scope='suanet', is_training=True, reuse=False):
    layers = OrderedDict()
    net = inputs
    with tf.variable_scope(scope, reuse=reuse):
        with ExitStack() as stack:
            stack.enter_context(
                slim.arg_scope(
                    [slim.fully_connected, slim.conv2d],
                    activation_fn=tf.nn.relu,
                    weights_regularizer=slim.l2_regularizer(1.0e-4)))
            stack.enter_context(
                slim.arg_scope([slim.max_pool2d, slim.conv2d],
                               padding='SAME'))
            net = slim.conv2d(net, 96, [11, 11], stride=4,scope='conv1')
            net = slim.max_pool2d(net, 3, stride=2, scope='pool1')
            layers['pool1'] = net
            net = slim.conv2d(net, 256, [5, 5], stride=1, scope='conv2')
            net = slim.max_pool2d(net, 3, stride=2, scope='pool2')
            layers['pool2'] = net
            net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv3')
            net = slim.max_pool2d(net, 16, stride=16, scope='pool3')
            layers['pool3'] = net
            net = tf.contrib.layers.flatten(net)

            with slim.arg_scope([slim.fully_connected], normalizer_fn=None):
                net = slim.fully_connected(net, 2, activation_fn=None, scope="fc2")


    return net, layers
suanet.default_image_size = 256
suanet.num_channels = 1
# suanet.range = 255
# suanet.mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
suanet.bgr = False