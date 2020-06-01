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
import tqdm
import glog as log

import json
import os
import os.path as ops
import time

from config import global_config
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.tools.graph_transforms import TransformGraph

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

def get_all_files(path):
    image_list = []
    for root, dirs, files in os.walk(path):
        if len(dirs) != 0:
            for dir in dirs:
                for root, dirs, files in os.walk(ops.join(path, dir)):
                    for file in files:
                        image_list.append(ops.join(root, file))
                        print(ops.join(root, file))
        else:
            for file in files:
                image_list.append(ops.join(root, file))
                print(ops.join(root, file))

    return image_list


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

def optimize_model(model_path):
    inputGraph = tf.GraphDef()
    with tf.gfile.Open(model_path, 'rb') as model:
        data2read = model.read()
        inputGraph.ParseFromString(data2read)

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


    """outputGraph = optimize_for_inference_lib.optimize_for_inference(
        inputGraph,
        input_node_names=['lanenet/input_tensor'],
        output_node_names=[
                'lanenet/final_binary_output',
                'lanenet/final_pixel_embedding_output'],
        placeholder_type_enum=tf.int32.as_datatype_enum
    )"""

    outputGraph = TransformGraph(
        inputGraph,
        ['lanenet/input_tensor'],
        ['lanenet/final_binary_output',
         'lanenet/final_pixel_embedding_output'],
        ['remove_nodes(op=Identity, op=CheckNumerics)',
         'merge_duplicate_nodes',
         'strip_unused_nodes',
         'fold_constants(ignore_errors=true)',
         'fold_batch_norms',
         'fold_old_batch_norms',
         'quantize_weights',
         'quantize_nodes',
         'sort_by_execution_order']
    )

    new_name = model_path.split('/')[-2] + '/OptimizedGraph.pb'
    model = tf.gfile.FastGFile(new_name, 'w')
    model.write(outputGraph.SerializeToString())



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

            # for op in sess.graph.get_all_collection_keys():
            #     print(str(op.name))

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


            plt.figure('final_binary_output')
            plt.imshow(out[0][0])
            plt.figure('final_instance_output')
            plt.imshow(out[1][0])
            plt.figure('src_image')
            plt.imshow(image_vis[:, :, (2, 1, 0)])
            plt.show()
            # plt.imshow(out[:, :, (2, 1, 0)])

def generate_prediction_result(src_dir, dst_dir, weights_path):
    """
    generate prediction results for evaluate
    """
    postprocessor = lanenet_postprocess.LaneNetPostProcessor_noremap()

    with tf.device('/gpu:0'):
        with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()
            with open(weights_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                tf.import_graph_def(output_graph_def, name="")
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                task_file_path = ops.join(src_dir, "test_tasks.json")
                target_image_path = ops.join(dst_dir, "results")

                image_nums = len(os.listdir(target_image_path))

                task_list = []
                with open(task_file_path, 'r') as file:
                    for line in file:
                        test_task = json.loads(line)
                        task_list.append(test_task)

                avg_time_cost = []
                avg_full_cost = []

                with open(task_file_path, 'r') as file:
                    for line_index, line in tqdm.tqdm(enumerate(task_list), total=len(task_list)):

                        image_dir = ops.split(line['raw_file'])[0]
                        image_dir_split = image_dir.split('/')[1:]
                        image_dir_split.append(ops.split(line['raw_file'])[1])
                        image_path = ops.join(src_dir, line['raw_file'])
                        assert ops.exists(image_path), '{:s} not exist'.format(image_path)

                        h_samples = line['h_samples']
                        raw_file = line['raw_file']

                        image_name_new = '{:s}.png'.format('{:d}'.format(line_index + image_nums).zfill(4))

                        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                        image_vis = image
                        image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
                        image = image / 127.5 - 1.0  # 归一化 (只归一未改变维数)

                        t_start = time.time()

                        input_tensor = sess.graph.get_tensor_by_name("lanenet/input_tensor:0")

                        output_binary = sess.graph.get_tensor_by_name("lanenet/final_binary_output:0")
                        output_instance = sess.graph.get_tensor_by_name("lanenet/final_pixel_embedding_output:0")

                        out = sess.run([output_binary, output_instance], feed_dict={input_tensor: [image]})

                        avg_time_cost.append(time.time() - t_start)

                        postprocess_result = postprocessor.postprocess_noremap(
                            binary_seg_result=out[0][0],
                            instance_seg_result=out[1][0],
                            source_image=image_vis
                        )

                        avg_full_cost.append(time.time() - t_start)

                        dst_image_path = ops.join(target_image_path, image_name_new)

                        cv2.imwrite(dst_image_path, image_vis)

                        with open(ops.join(dst_dir, "result.json"), 'a') as result_file:
                            data = {
                                'h_samples': h_samples,
                                'lanes': postprocess_result['lane_pts'],
                                'raw_file': raw_file,
                                'result_file': ops.join(dst_image_path)
                            }
                            json.dump(data, result_file)
                            result_file.write('\n')

                        if line_index % 100 == 0:
                            log.info('Mean inference time every single image: {:.5f}s'.format(np.mean(avg_time_cost)))
                            log.info('Mean full inference time every single image: {:.5f}s'.format(np.mean(avg_full_cost)))
                            with open(ops.join(dst_dir, 'log'), 'a') as log_file:
                                print('Mean inference time every single image: {:.5f}s'.format(np.mean(avg_time_cost)),
                                      file=log_file)
                                print('Mean full inference time every single image: {:.5f}s'.format(np.mean(avg_full_cost)),
                                      file=log_file)
                            avg_time_cost.clear()
                            avg_full_cost.clear()
    return



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

    args = init_args()
    """
    convert_ckpt_into_pb_file(
        ckpt_file_path=args.weights_path,
        pb_file_path=args.save_path
     )"""

    """
    image_path = "./data/tusimple_test_image/0.jpg"
    pb_path = "./mnn_project/OptimizedGraph.pb"
    # printTensors(pb_path)
    freeze_graph_test(pb_path, image_path)
    """
    """
    generate_prediction_result('/media/stevemaary/68A0799BA0797104/Users/a1975/Documents/lanenet_related_files/',
                               '/home/stevemaary/data/pred',
                               './mnn_project/OptimizedGraph.pb')
    """
    optimize_model('./mnn_project/lanenet.pb')

