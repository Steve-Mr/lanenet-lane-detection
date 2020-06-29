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

from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder

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

    net = lanenet.LaneNet(phase='test', net_flag='vgg')

    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    # postprocessor = lanenet_postprocess.LaneNetPostProcessor()
    postprocessor = lanenet_postprocess.LaneNetPostProcessor_noremap()

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
    sess = tf.Session(config=sess_config)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        t_start = time.time()

        binary_seg_image, instance_seg_image = sess.run(
            [binary_seg_ret, instance_seg_ret],
            feed_dict={input_tensor: [image]}
        )
        t_cost = time.time() - t_start
        log.info('Single imgae inference cost time: {:.5f}s'.format(t_cost))

        """
        postprocess_result = postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_vis
        )"""

        """
        postprocess_result = postprocessor.postprocess_for_test(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_vis
        )"""

        postprocess_result = postprocessor.postprocess_noremap(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_vis
        )

        mask_image = postprocess_result['mask_image']

        for i in range(CFG.TRAIN.EMBEDDING_FEATS_DIMS):
            # __C.TRAIN.EMBEDDING_FEATS_DIMS = 4
            instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
            # 与 instance_seg_image[0][:, :, i] =
            # cv2.normalize(instance_seg_image[0][:, :, i], None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
            # 功能相同
            # 将bgr彩色矩阵归一化到0-255之间
        embedding_image = np.array(instance_seg_image[0], np.uint8)

        # for op in tf.get_default_graph().get_operations():
        #     print(str(op.name))

        #print([n.name for n in tf.get_default_graph().as_graph_def().node])

        plt.figure('mask_image')
        # plt.imshow(mask_image[:, :, (2, 1, 0)])
        plt.imshow(mask_image)
        plt.figure('src_image')
        plt.imshow(image_vis[:, :, (2, 1, 0)])
        plt.figure('instance_image')
        plt.imshow(embedding_image[:, :, (2, 1, 0)])
        plt.figure('binary_image')
        plt.imshow(binary_seg_image[0] * 255, cmap='gray')
        """"
        plt.figure("result")
        plt.imshow(postprocess_result['source_image'])
        """
        plt.show()

        cv2.imwrite('instance_mask_image.png', mask_image)
        cv2.imwrite('source_image.png', postprocess_result['source_image'])
        cv2.imwrite('binary_mask_image.png', binary_seg_image[0] * 255)

    sess.close()

    return


def test_lanenet_for_eval(image_path, weights_path):
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

    net = lanenet.LaneNet(phase='test', net_flag='vgg')

    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    postprocessor = lanenet_postprocess.LaneNetPostProcessor()

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
    sess = tf.Session(config=sess_config)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        binary_seg_image_, instance_seg_image_ = sess.run(
            [binary_seg_ret, instance_seg_ret],
            feed_dict={input_tensor: [image]}
        )

        profiler = model_analyzer.Profiler(graph=sess.graph)
        run_metadata = tf.RunMetadata()

        t_start = time.time()

        binary_seg_image, instance_seg_image = sess.run(
            [binary_seg_ret, instance_seg_ret],
            feed_dict={input_tensor: [image]},
            options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
            run_metadata=run_metadata
        )

        t_cost = time.time() - t_start
        log.info('Single imgae inference cost time: {:.5f}s'.format(t_cost))

        profiler.add_step(step=1, run_meta=run_metadata)

        profile_op_builder = option_builder.ProfileOptionBuilder()
        profile_op_builder.select(['micros', 'occurrence'])
        profile_op_builder.order_by('micros')
        profile_op_builder.with_max_depth(5)
        profile_op_builder.with_file_output(outfile="./op_profiler.txt")
        # profiler.profile_graph(profile_op_builder.build())
        profiler.profile_operations(profile_op_builder.build())

        profile_code_builder = option_builder.ProfileOptionBuilder()
        profile_code_builder.with_max_depth(1000)
        profile_code_builder.with_node_names(show_name_regexes=['cnn_basenet.py.*'])
        profile_code_builder.with_min_execution_time(min_micros=10)
        profile_code_builder.select(['micros'])
        profile_code_builder.order_by('min_micros')
        profile_code_builder.with_file_output(outfile="./code_profiler.txt")
        profiler.profile_python(profile_code_builder.build())

        profiler.advise(options=model_analyzer.ALL_ADVICE)

        postprocess_result = postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_vis
        )

        """
        postprocess_result = postprocessor.postprocess_for_test(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_vis
        )"""

        mask_image = postprocess_result['mask_image']

        for i in range(CFG.TRAIN.EMBEDDING_FEATS_DIMS):
            # __C.TRAIN.EMBEDDING_FEATS_DIMS = 4
            instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
            # 与 instance_seg_image[0][:, :, i] =
            # cv2.normalize(instance_seg_image[0][:, :, i], None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
            # 功能相同
            # 将bgr彩色矩阵归一化到0-255之间
        embedding_image = np.array(instance_seg_image[0], np.uint8)

        # for op in tf.get_default_graph().get_operations():
        #     print(str(op.name))

        # print([n.name for n in tf.get_default_graph().as_graph_def().node])

        plt.figure('mask_image')
        # plt.imshow(mask_image[:, :, (2, 1, 0)])
        plt.imshow(mask_image)
        plt.figure('src_image')
        plt.imshow(image_vis[:, :, (2, 1, 0)])
        plt.figure('instance_image')
        plt.imshow(embedding_image[:, :, (2, 1, 0)])
        plt.figure('binary_image')
        plt.imshow(binary_seg_image[0] * 255, cmap='gray')
        """"
        plt.figure("result")
        plt.imshow(postprocess_result['source_image'])
        """
        plt.show()

        cv2.imwrite('instance_mask_image.png', mask_image)
        cv2.imwrite('source_image.png', postprocess_result['source_image'])
        cv2.imwrite('binary_mask_image.png', binary_seg_image[0] * 255)

    sess.close()

    return


def test_lanenet_nontusimple(image_path, weights_path):
    """

    :param image_path: 测试图片地址
    :param weights_path: 训练模型地址
    :return:
    """
    assert ops.exists(image_path), '{:s} not exist'.format(image_path)

    log.info('Start reading image and preprocessing')
    t_start = time.time()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_vis = image
    # image_vis = cv2.imread('/media/stevemaary/新加卷/data/caltech/caltech-lanes/label_file/cordova1/label/f00028.png', cv2.IMREAD_COLOR)

    # image = image[30:345, 40:600] # caltech
    # image = image[:430]           # culane

    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)

    # plt.figure("before")
    # plt.imshow(image[:, :, (2, 1, 0)])

    image = image / 127.5 - 1.0  # 归一化 (只归一未改变维数)

    # plt.figure("after")
    # plt.imshow(image[:, :, (2, 1, 0)])
    # cv2.imwrite('after.png', image[:,:,(2,1,0)])

    log.info('Image load complete, cost time: {:.5f}s'.format(time.time() - t_start))

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    net = lanenet.LaneNet(phase='test', net_flag='vgg')
    # binary_seg_ret, instance_seg_ret, enbinary_, eninstance_, binary_, instance_ = net.inference(input_tensor=input_tensor, name='lanenet_model')
    binary_seg_ret, instance_seg_ret = net.inference(
        input_tensor=input_tensor, name='lanenet_model')
    postprocessor = lanenet_postprocess.LaneNetPostProcessor_noremap()

    saver = tf.train.Saver()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'  # best fit with coalescing  内存管理算法
    sess = tf.Session(config=sess_config)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        t_start = time.time()
        """
        binary_seg_image, instance_seg_image, encode_binary, encode_instance, decode_binary, decode_instance = sess.run(
            [binary_seg_ret, instance_seg_ret, enbinary_, eninstance_, binary_, instance_],
            feed_dict={input_tensor: [image]}
        )
        """

        binary_seg_image, instance_seg_image = sess.run(
            [binary_seg_ret, instance_seg_ret],
            feed_dict={input_tensor: [image]}
        )

        t_cost = time.time() - t_start
        log.info('Single imgae inference cost time: {:.5f}s'.format(t_cost))
        """
        binary = sess.run(tf.transpose(decode_binary, [3, 0, 1, 2]))
        instance = sess.run(tf.transpose(decode_instance, [3, 0, 1, 2]))
        enbinary = sess.run(tf.transpose(encode_binary,[3, 0, 1, 2]))
        eninstance = sess.run(tf.transpose(encode_instance, [3, 0, 1, 2]))

        plt.figure('decode_binary')
        plt.imshow(binary[0][0])
        plt.figure('decode_instance')
        plt.imshow(instance[0][0])
        plt.figure('encode_binary')
        plt.imshow(enbinary[0][0])
        plt.figure('encode_instance')
        plt.imshow(eninstance[0][0])
        """
        postprocess_result = postprocessor.postprocess_noremap(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_vis,
            # data_source='caltech'
        )

        mask_image = postprocess_result['mask_image']

        for i in range(CFG.TRAIN.EMBEDDING_FEATS_DIMS):
            # __C.TRAIN.EMBEDDING_FEATS_DIMS = 4
            instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
            # 与 instance_seg_image[0][:, :, i] =
            # cv2.normalize(instance_seg_image[0][:, :, i], None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
            # 功能相同
            # 将bgr彩色矩阵归一化到0-255之间
        embedding_image = np.array(instance_seg_image[0], np.uint8)

        plt.figure('mask_image')
        # mask_image = cv2.resize(mask_image, (640, 480), interpolation= cv2.INTER_LINEAR)
        plt.imshow(mask_image[:, :, (2, 1, 0)])
        plt.imshow(mask_image)
        plt.figure('src_image')
        plt.imshow(image_vis[:, :, (2, 1, 0)])
        plt.figure('instance_image')
        plt.imshow(embedding_image[:, :, (2, 1, 0)])
        # plt.imshow(instance_seg_ret[:, :, 0])
        """
        back = np.zeros(shape=(256, 512), dtype= np.uint8)
        idx = np.where(embedding_image[:, :, 0] == 255)
        lane_coord = np.vstack((idx[1], idx[0])).transpose()
        for coord in lane_coord:
            cv2.circle(back, (coord[0], coord[1]), 5, 255, -1)
        plt.figure("test")
        plt.imshow(back)
        """
        # print(embedding_image[:, :, 0])
        plt.figure('binary_image')
        plt.imshow(binary_seg_image[0] * 255, cmap='gray')
        # plt.figure("cropped")
        # plt.imshow(image)
        plt.show()

        cv2.imwrite('instance_mask_image.png', mask_image)
        cv2.imwrite('source_image.png', postprocess_result['source_image'])
        cv2.imwrite('binary_mask_image.png', binary_seg_image[0] * 255)
        cv2.imwrite('instance.png', embedding_image[:, :, (2, 1, 0)])

    sess.close()

    return


if __name__ == '__main__':
    """
    test code
    """
    # init args
    args = init_args()

    # test_lanenet(args.image_path, args.weights_path)
    # test_lanenet_for_eval(args.image_path, args.weights_path)
    test_lanenet_nontusimple(args.image_path, args.weights_path)
