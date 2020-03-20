#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
test LaneNet model on single image
"""
import argparse
import os.path as ops
import time

import cv2
import glog as log
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import global_config
from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess

CFG = global_config.cfg


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')

    return parser.parse_args()


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def test_lanenet(image_path, weights_path):
    """

    :param image_path: 测试图片地址
    :param weights_path: 训练模型地址
    :return:
    """
    assert ops.exists(image_path), '{:s} not exist'.format(image_path)

    log.info('Start reading image and preprocessing')
    t_start = time.time()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    """
    :param: cv2.IMREAD_COLOR:  It specifies to load a color image. Any transparency of image will be neglected. 
    """
    image_vis = image

    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    """
    :param: INTER_LINEAR: 双线性插值。
    """

    image = image / 127.5 - 1.0  # 归一化 (只归一未改变维数)
    log.info('Image load complete, cost time: {:.5f}s'.format(time.time() - t_start))

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    """
    在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。
    等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
    
    :param: dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
    :param: shape：数据形状。NHWC：[batch, in_height, in_width, in_channels] [参与训练的一批(batch)图像的数量，输入图片的高度，输入图片的宽度，输入图片的通道数]
    :param: name：名称。
    """

    print("start")

    print("net = lanenet.LaneNet(phase='test', net_flag='vgg')")
    net = lanenet.LaneNet(phase='test', net_flag='vgg')

    print("binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')")
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    print("postprocessor = lanenet_postprocess.LaneNetPostProcessor()")
    postprocessor = lanenet_postprocess.LaneNetPostProcessor()

    print("saver = tf.train.Saver()")
    saver = tf.train.Saver()
    # 加载预训练模型参数

    # Set session configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    # 限制 GPU 使用率

    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    # 动态申请显存

    sess_config.gpu_options.allocator_type = 'BFC'  # best fit with coalescing  内存管理算法
    # 内存分配类型
    print(" sess = tf.Session(config=sess_config)")
    sess = tf.Session(config=sess_config)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        t_start = time.time()

        print("binary_seg_image, instance_seg_image = sess.run(")
        binary_seg_image, instance_seg_image = sess.run(
            [binary_seg_ret, instance_seg_ret],
            feed_dict={input_tensor: [image]}
        )
        t_cost = time.time() - t_start
        log.info('Single imgae inference cost time: {:.5f}s'.format(t_cost))

        print("postprocess_result = postprocessor.postprocess(")
        postprocess_result = postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_vis
        )

        print("mask_image = postprocess_result['mask_image']")
        mask_image = postprocess_result['mask_image']

        for i in range(CFG.TRAIN.EMBEDDING_FEATS_DIMS):
            # __C.TRAIN.EMBEDDING_FEATS_DIMS = 4
            instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
            # 与 instance_seg_image[0][:, :, i] =
            # cv2.normalize(instance_seg_image[0][:, :, i], None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
            # 功能相同
            # 将bgr彩色矩阵归一化到0-255之间
        embedding_image = np.array(instance_seg_image[0], np.uint8)

        for op in tf.get_default_graph().get_operations():
            print(str(op.name))

        #print([n.name for n in tf.get_default_graph().as_graph_def().node])

        plt.figure('mask_image')
        plt.imshow(mask_image[:, :, (2, 1, 0)])
        plt.figure('src_image')
        plt.imshow(image_vis[:, :, (2, 1, 0)])
        plt.figure('instance_image')
        plt.imshow(embedding_image[:, :, (2, 1, 0)])
        plt.figure('binary_image')
        plt.imshow(binary_seg_image[0] * 255, cmap='gray')
        plt.show()

        cv2.imwrite('instance_mask_image.png', mask_image)
        cv2.imwrite('source_image.png', postprocess_result['source_image'])
        cv2.imwrite('binary_mask_image.png', binary_seg_image[0] * 255)

    sess.close()

    return


if __name__ == '__main__':
    """
    test code
    """
    # init args
    args = init_args()

    test_lanenet(args.image_path, args.weights_path)
