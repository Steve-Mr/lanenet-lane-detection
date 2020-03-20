#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/11/5 下午4:53
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : freeze_lanenet_model.py.py
# @IDE: PyCharm
"""
Freeze Lanenet model into frozen pb file
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
from config import global_config


import tensorflow as tf

from lanenet_model import lanenet

import cv2
import matplotlib.pyplot as plt
from lanenet_model import lanenet_postprocess


MODEL_WEIGHTS_FILE_PATH = './test.ckpt'
OUTPUT_PB_FILE_PATH = './lanenet.pb'

CFG = global_config.cfg

def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights_path', default=MODEL_WEIGHTS_FILE_PATH)
    parser.add_argument('-s', '--save_path', default=OUTPUT_PB_FILE_PATH)

    return parser.parse_args()


def convert_ckpt_into_pb_file(ckpt_file_path, pb_file_path):
    """

    :param ckpt_file_path:
    :param pb_file_path:
    :return:
    """
    # construct compute graph
    with tf.variable_scope('lanenet'):
        input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    with tf.variable_scope('lanenet/'):
        binary_seg_ret = tf.cast(binary_seg_ret, dtype=tf.float32)
        binary_seg_ret = tf.identity(binary_seg_ret, name= 'final_binary_output')
        instance_seg_ret = tf.identity(instance_seg_ret, name='final_pixel_embedding_output')
        # binary_seg_ret = tf.squeeze(binary_seg_ret, axis=0, name='final_binary_output')  # 删除所有大小是 1 的维度
        # instance_seg_ret = tf.squeeze(instance_seg_ret, axis=0, name='final_pixel_embedding_output')

    # create a session
    saver = tf.train.Saver()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.85
    sess_config.gpu_options.allow_growth = False
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():
        saver.restore(sess, ckpt_file_path)

        converted_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            input_graph_def=sess.graph.as_graph_def(),
            output_node_names=[
                'lanenet/input_tensor',
                'lanenet/final_binary_output',
                'lanenet/final_pixel_embedding_output'
            ]
        )

        with tf.gfile.GFile(pb_file_path, "wb") as f:
            f.write(converted_graph_def.SerializeToString())


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def freeze_graph_test(pb_path, image_path):
    """
    :param pb_path: pb 文件路径
    :param image_path: image 文件路径
    """
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for op in sess.graph.get_all_collection_keys():
                print(str(op.name))

            input_tensor = sess.graph.get_tensor_by_name("lanenet/input_tensor:0")

            output_binary = sess.graph.get_tensor_by_name("lanenet/final_binary_output:0")
            output_instance = sess.graph.get_tensor_by_name("lanenet/final_pixel_embedding_output:0")

            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_vis = image
            image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
            image = image / 127.5 - 1.0  # 归一化 (只归一未改变维数)

            postprocessor = lanenet_postprocess.LaneNetPostProcessor()

            out = sess.run([output_binary, output_instance], feed_dict={input_tensor: [image]})

            postprocess_result = postprocessor.postprocess(
                binary_seg_result=out[0][0],
                instance_seg_result=out[1][0],
                source_image=image_vis
            )

            """
            plt.figure('input')
            plt.imshow(out[0][0])
            plt.figure('final_binary_output')
            plt.imshow(out[1])
            plt.figure('final_instance_output')
            plt.imshow(out[2])
            """

            plt.figure('final_binary_output')
            plt.imshow(out[0][0])
            plt.figure('final_instance_output')
            plt.imshow(out[1][0])
            plt.figure('src_image')
            plt.imshow(image_vis[:, :, (2, 1, 0)])
            plt.show()
            # plt.imshow(out[:, :, (2, 1, 0)])


def printTensors(pb_file):
    # read pb into graph_def
    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import graph_def
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    # print operations
    for op in graph.get_operations():
        print(op.name)


if __name__ == '__main__':
    """
    test code
    """
    """
    args = init_args()

    convert_ckpt_into_pb_file(
        ckpt_file_path=args.weights_path,
        pb_file_path=args.save_path
     )
    """

    image_path = "./data/tusimple_test_image/0.jpg"
    pb_path = "./mnn_project/lanenet.pb"
    printTensors(pb_path)
    freeze_graph_test(pb_path, image_path)
