#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-4-24 下午6:42
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : vgg16_based_fcn.py
# @IDE: PyCharm
"""
Implement VGG16 based fcn net for semantic segmentation
"""
import argparse
import collections
import cv2
import numpy as np

import tensorflow as tf
from numpy.ma import array

from config import global_config
from semantic_segmentation_zoo import cnn_basenet

import matplotlib.pyplot as plt

CFG = global_config.cfg


class VGG16FCN(cnn_basenet.CNNBaseModel):
    """
    VGG 16 based fcn net for semantic segmentation
    """

    def __init__(self, phase):

        """

        """
        super(VGG16FCN, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._net_intermediate_results = collections.OrderedDict()

    def _is_net_for_training(self):

        """
        if the net is used for training or not
        :return:
        """
        if isinstance(self._phase, tf.Tensor):
            """
            isinstance(object, classinfo)
            object -- 实例对象。
            classinfo -- 可以是直接或间接类名、基本类型或者由它们组成的元组。
            
            isinstance() 会认为子类是一种父类类型，考虑继承关系。
            """
            phase = self._phase
        else:
            phase = tf.constant(self._phase, dtype=tf.string)
            """Creates a constant tensor.

              The resulting tensor is populated with values of type `dtype`, as
              specified by arguments `value` and (optionally) `shape` (see examples
              below).

              Args:
                value:          A constant value (or list) of output type `dtype`.
                dtype:          The type of the elements of the resulting tensor.
                shape:          Optional dimensions of resulting tensor.
                name:           Optional name for the tensor.
                verify_shape:   Boolean that enables verification of a shape of values.

              Returns:
                A Constant Tensor.
              """

        return tf.equal(phase, tf.constant('train', dtype=tf.string))

    def _vgg16_conv_stage(self, input_tensor, k_size, out_dims, name,
                          stride=1, pad='SAME', need_layer_norm=True):

        """
        stack conv and activation in vgg16
        :param input_tensor:
        :param k_size: kernel size
        :param out_dims: output dimension
        :param name:
        :param stride:
        :param pad: padding SAME
        :param need_layer_norm:
        :return: tensor
        """

        with tf.variable_scope(name):
            conv = self.conv2d(  # 卷积结果
                inputdata=input_tensor, out_channel=out_dims,
                kernel_size=k_size, stride=stride,
                use_bias=False, padding=pad, name='conv'
            )

            if need_layer_norm:
                bn = self.layerbn(inputdata=conv, is_training=self._is_training, name='bn')

                relu = self.relu(inputdata=bn, name='relu')
            else:
                relu = self.relu(inputdata=conv, name='relu')

        return relu

    def _decode_block(self, input_tensor, previous_feats_tensor,
                      out_channels_nums, name, kernel_size=4,
                      stride=2, use_bias=False,
                      previous_kernel_size=4, need_activate=True):

        """

        :param input_tensor:
        :param previous_feats_tensor:
        :param out_channels_nums:
        :param kernel_size:
        :param previous_kernel_size:
        :param use_bias:
        :param stride:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            deconv_weights_stddev = tf.sqrt(    # deconv 权重标准差
                tf.divide(tf.constant(2.0, tf.float32),
                          tf.multiply(tf.cast(previous_kernel_size * previous_kernel_size, tf.float32),
                                      tf.cast(tf.shape(input_tensor)[3], tf.float32)))
            )
            """
            tf.sqrt(x, name=None)
                计算x元素的平方根. 即,(y = sqrt{x} = x^{1/2}).

                函数参数：
                    x：一个 Tensor 或 SparseTensor,必须是下列类型之一：half,float32,float64,complex64,complex128.
                    name：操作的名称(可选).
                函数返回值：
                    tf.sqrt函数返回 Tensor 或者 SparseTensor,与 x 具有相同的类型相同.
            ---------
            tf.multiply(x, y, name=None)
                参数:
                    x: 一个类型为:half, float32, float64, uint8, int8, uint16, int16, int32, int64, complex64, complex128的张量。
                    y: 一个类型跟张量x相同的张量。
                返回值： x * y element-wise.
                注意：
                    （1）multiply这个函数实现的是元素级别的相乘，也就是两个相乘的数元素各自相乘，而不是矩阵乘法，注意和tf.matmul区别。
                    （2）两个相乘的数必须有相同的数据类型，不然就会报错。
            ---------
            tf.cast(x, dtype, name=None)
                第一个参数 x:   待转换的数据（张量）
                第二个参数 dtype： 目标数据类型
                第三个参数 name： 可选参数，定义操作的名称
            """

            deconv_weights_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=deconv_weights_stddev)
            """
            从截断的正态分布中输出随机值.
            生成的值遵循具有指定平均值和标准偏差的正态分布,不同之处在于其平均值大于 2 个标准差的值将被丢弃并重新选择.
                shape：一维整数张量或 Python 数组,输出张量的形状.
                mean：dtype 类型的 0-D 张量或 Python 值,截断正态分布的均值.
                stddev：dtype 类型的 0-D 张量或 Python 值,截断前正态分布的标准偏差.
                dtype：输出的类型.
                seed：一个 Python 整数.用于为分发创建随机种子.查看tf.set_random_seed行为.
                name：操作的名称(可选).
            函数返回值：
                tf.truncated_normal函数返回指定形状的张量填充随机截断的正常值.
            """

            deconv = self.deconv2d(
                inputdata=input_tensor, out_channel=out_channels_nums, kernel_size=kernel_size,
                stride=stride, use_bias=use_bias, w_init=deconv_weights_init,
                name='deconv'
            )

            deconv = self.layerbn(inputdata=deconv, is_training=self._is_training, name='deconv_bn')

            deconv = self.relu(inputdata=deconv, name='deconv_relu')

            fuse_feats = tf.add(
                previous_feats_tensor, deconv, name='fuse_feats'
            )

            if need_activate:
                fuse_feats = self.layerbn(
                    inputdata=fuse_feats, is_training=self._is_training, name='fuse_gn'
                )

                fuse_feats = self.relu(inputdata=fuse_feats, name='fuse_relu')

        return fuse_feats

    def _vgg16_fcn_encode(self, input_tensor, name):

        """
        根据vgg16框架对输入的tensor进行编码
        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            # encode stage 1
            conv_1_1 = self._vgg16_conv_stage(
                input_tensor=input_tensor, k_size=3,
                out_dims=64, name='conv1_1',
                need_layer_norm=True
            )
            conv_1_2 = self._vgg16_conv_stage(
                input_tensor=conv_1_1, k_size=3,
                out_dims=64, name='conv1_2',
                need_layer_norm=True
            )

            self._net_intermediate_results['encode_stage_1_share'] = {
                'data': conv_1_2,
                'shape': conv_1_2.get_shape().as_list()
            }

            # encode stage 2
            pool1 = self.maxpooling(
                inputdata=conv_1_2, kernel_size=2,
                stride=2, name='pool1'
            )

            conv_2_1 = self._vgg16_conv_stage(
                input_tensor=pool1, k_size=3,
                out_dims=128, name='conv2_1',
                need_layer_norm=True
            )

            conv_2_2 = self._vgg16_conv_stage(
                input_tensor=conv_2_1, k_size=3,
                out_dims=128, name='conv2_2',
                need_layer_norm=True
            )

            self._net_intermediate_results['encode_stage_2_share'] = {
                'data': conv_2_2,
                'shape': conv_2_2.get_shape().as_list()
            }

            # encode stage 3
            pool2 = self.maxpooling(
                inputdata=conv_2_2, kernel_size=2,
                stride=2, name='pool2'
            )

            conv_3_1 = self._vgg16_conv_stage(
                input_tensor=pool2, k_size=3,
                out_dims=256, name='conv3_1',
                need_layer_norm=True
            )

            conv_3_2 = self._vgg16_conv_stage(
                input_tensor=conv_3_1, k_size=3,
                out_dims=256, name='conv3_2',
                need_layer_norm=True
            )

            conv_3_3 = self._vgg16_conv_stage(
                input_tensor=conv_3_2, k_size=3,
                out_dims=256, name='conv3_3',
                need_layer_norm=True
            )

            self._net_intermediate_results['encode_stage_3_share'] = {
                'data': conv_3_3,
                'shape': conv_3_3.get_shape().as_list()
            }

            # encode stage 4
            pool3 = self.maxpooling(
                inputdata=conv_3_3, kernel_size=2,
                stride=2, name='pool3'
            )

            conv_4_1 = self._vgg16_conv_stage(
                input_tensor=pool3, k_size=3,
                out_dims=512, name='conv4_1',
                need_layer_norm=True
            )

            conv_4_2 = self._vgg16_conv_stage(
                input_tensor=conv_4_1, k_size=3,
                out_dims=512, name='conv4_2',
                need_layer_norm=True
            )

            conv_4_3 = self._vgg16_conv_stage(
                input_tensor=conv_4_2, k_size=3,
                out_dims=512, name='conv4_3',
                need_layer_norm=True
            )

            self._net_intermediate_results['encode_stage_4_share'] = {
                'data': conv_4_3,
                'shape': conv_4_3.get_shape().as_list()
            }

            # encode stage 5 for binary segmentation
            pool4 = self.maxpooling(
                inputdata=conv_4_3, kernel_size=2,
                stride=2, name='pool4'
            )

            conv_5_1_binary = self._vgg16_conv_stage(
                input_tensor=pool4, k_size=3,
                out_dims=512, name='conv5_1_binary',
                need_layer_norm=True
            )

            conv_5_2_binary = self._vgg16_conv_stage(
                input_tensor=conv_5_1_binary, k_size=3,
                out_dims=512, name='conv5_2_binary',
                need_layer_norm=True
            )

            conv_5_3_binary = self._vgg16_conv_stage(
                input_tensor=conv_5_2_binary, k_size=3,
                out_dims=512, name='conv5_3_binary',
                need_layer_norm=True
            )

            self._net_intermediate_results['encode_stage_5_binary'] = {
                'data': conv_5_3_binary,
                'shape': conv_5_3_binary.get_shape().as_list()
            }


            # encode stage 5 for instance segmentation
            conv_5_1_instance = self._vgg16_conv_stage(
                input_tensor=pool4, k_size=3,
                out_dims=512, name='conv5_1_instance',
                need_layer_norm=True
            )
            conv_5_2_instance = self._vgg16_conv_stage(
                input_tensor=conv_5_1_instance, k_size=3,
                out_dims=512, name='conv5_2_instance',
                need_layer_norm=True
            )
            conv_5_3_instance = self._vgg16_conv_stage(
                input_tensor=conv_5_2_instance, k_size=3,
                out_dims=512, name='conv5_3_instance',
                need_layer_norm=True
            )
            self._net_intermediate_results['encode_stage_5_instance'] = {
                'data': conv_5_3_instance,
                'shape': conv_5_3_instance.get_shape().as_list()
            }

        return

    def _vgg16_fcn_decode(self, name):

        """

        :return:
        """
        with tf.variable_scope(name):
            # decode part for binary segmentation
            with tf.variable_scope(name_or_scope='binary_seg_decode'):
                decode_stage_5_binary = self._net_intermediate_results['encode_stage_5_binary']['data']

                decode_stage_4_fuse = self._decode_block(
                    input_tensor=decode_stage_5_binary,
                    previous_feats_tensor=self._net_intermediate_results['encode_stage_4_share']['data'],
                    name='decode_stage_4_fuse', out_channels_nums=512, previous_kernel_size=3
                )
                decode_stage_3_fuse = self._decode_block(
                    input_tensor=decode_stage_4_fuse,
                    previous_feats_tensor=self._net_intermediate_results['encode_stage_3_share']['data'],
                    name='decode_stage_3_fuse', out_channels_nums=256
                )
                decode_stage_2_fuse = self._decode_block(
                    input_tensor=decode_stage_3_fuse,
                    previous_feats_tensor=self._net_intermediate_results['encode_stage_2_share']['data'],
                    name='decode_stage_2_fuse', out_channels_nums=128
                )
                decode_stage_1_fuse = self._decode_block(
                    input_tensor=decode_stage_2_fuse,
                    previous_feats_tensor=self._net_intermediate_results['encode_stage_1_share']['data'],
                    name='decode_stage_1_fuse', out_channels_nums=64
                )
                binary_final_logits_conv_weights_stddev = tf.sqrt(
                    tf.divide(tf.constant(2.0, tf.float32),
                              tf.multiply(4.0 * 4.0,
                                          tf.cast(tf.shape(decode_stage_1_fuse)[3], tf.float32)))
                )
                binary_final_logits_conv_weights_init = tf.truncated_normal_initializer(
                    mean=0.0, stddev=binary_final_logits_conv_weights_stddev)

                binary_final_logits = self.conv2d(
                    inputdata=decode_stage_1_fuse, out_channel=CFG.TRAIN.CLASSES_NUMS,  # CLASSES_NUMS = 2
                    kernel_size=1, use_bias=False,
                    w_init=binary_final_logits_conv_weights_init,
                    name='binary_final_logits')

                self._net_intermediate_results['binary_segment_logits'] = {
                    'data': binary_final_logits,
                    'shape': binary_final_logits.get_shape().as_list()
                }


            with tf.variable_scope(name_or_scope='instance_seg_decode'):
                decode_stage_5_instance = self._net_intermediate_results['encode_stage_5_instance']['data']

                decode_stage_4_fuse = self._decode_block(
                    input_tensor=decode_stage_5_instance,
                    previous_feats_tensor=self._net_intermediate_results['encode_stage_4_share']['data'],
                    name='decode_stage_4_fuse', out_channels_nums=512, previous_kernel_size=3)

                decode_stage_3_fuse = self._decode_block(
                    input_tensor=decode_stage_4_fuse,
                    previous_feats_tensor=self._net_intermediate_results['encode_stage_3_share']['data'],
                    name='decode_stage_3_fuse', out_channels_nums=256)

                decode_stage_2_fuse = self._decode_block(
                    input_tensor=decode_stage_3_fuse,
                    previous_feats_tensor=self._net_intermediate_results['encode_stage_2_share']['data'],
                    name='decode_stage_2_fuse', out_channels_nums=128)

                decode_stage_1_fuse = self._decode_block(
                    input_tensor=decode_stage_2_fuse,
                    previous_feats_tensor=self._net_intermediate_results['encode_stage_1_share']['data'],
                    name='decode_stage_1_fuse', out_channels_nums=64, need_activate=False)

                self._net_intermediate_results['instance_segment_logits'] = {
                    'data': decode_stage_1_fuse,
                    'shape': decode_stage_1_fuse.get_shape().as_list()
                }

    def build_model(self, input_tensor, name, reuse=False):

        """
        :param input_tensor:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            # vgg16 fcn encode part
            self._vgg16_fcn_encode(input_tensor=input_tensor, name='vgg16_encode_module')
            # vgg16 fcn decode part
            # 使用转置卷积做上采样
            self._vgg16_fcn_decode(name='vgg16_decode_module')

        return self._net_intermediate_results

def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')

    return parser.parse_args()


if __name__ == '__main__':
    """
    test code
    """
    """
    test_in_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input')
    model = VGG16FCN(phase='train')
    ret = model.build_model(test_in_tensor, name='vgg16fcn')
    for layer_name, layer_info in ret.items():
        print('layer name: {:s} shape: {}'.format(layer_name, layer_info['shape']))
    """
    args = init_args()
    image = cv2.imread(args.image_path, cv2.IMREAD_COLOR)

    image_vis = image

    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)

    image = image / 127.5 - 1.0  # 归一化 (只归一未改变维数)

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    model = VGG16FCN(phase='test')
    model._vgg16_fcn_encode(input_tensor=input_tensor, name='vgg16_encode_module')
    # vgg16 fcn decode part
    # 使用转置卷积做上采样
    model._vgg16_fcn_decode(name='vgg16_decode_module')
    decode_binary = model._net_intermediate_results['binary_segment_logits']['data']
    decode_instance = model._net_intermediate_results['instance_segment_logits']['data']
    enbinary = model._net_intermediate_results['encode_stage_5_binary']['data']
    eninstance = model._net_intermediate_results['encode_stage_5_instance']['data']

    with tf.variable_scope(name_or_scope='vgg_backend', reuse=True):
        with tf.variable_scope(name_or_scope='binary_seg'):
            binary_seg_score = tf.nn.softmax(logits=enbinary)
            """
            归一化
            将一个含任意实数的K维向量“压缩”到另一个K维实向量中，使得每一个元素的范围都在之间，并且所有元素的和为1
            Returns:
            A `Tensor`. Has the same type and shape as `logits`.
            """

            binary_seg_prediction = tf.argmax(binary_seg_score, axis=-1)
            # 返回最大值索引号 axis：选取的坐标值


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        image = array(image).reshape(1, 256, 512, 3)
        encode_binary =sess.run(enbinary, feed_dict={input_tensor: image})
        encode_instance =sess.run(eninstance, feed_dict={input_tensor: image})

        decode_binary1 = sess.run(decode_binary, feed_dict={input_tensor: image})
        decode_instance1 =sess.run(decode_instance, feed_dict={input_tensor: image})
        binary = sess.run(tf.transpose(decode_binary1,[3,0,1,2]))
        instance = sess.run(tf.transpose(decode_instance1,[3,0,1,2]))
        encodebinary = sess.run(tf.transpose(encode_binary,[3,0,1,2]))
        encodeinstance = sess.run(tf.transpose(encode_instance,[3,0,1,2]))

        plt.figure('encode_binary')
        plt.imshow(encodebinary[0][0])
        plt.figure('encode_instance')
        plt.imshow(encodeinstance[0][0])
        plt.figure('decode_binary')
        plt.imshow(binary[0][0])
        plt.figure('decode_instance')
        plt.imshow(instance[0][0])

        plt.show()

        sess.close()
