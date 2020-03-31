#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-5-16 下午6:26
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : evaluate_lanenet_on_tusimple.py
# @IDE: PyCharm
"""
Evaluate lanenet model on tusimple lane dataset
"""
import argparse
import glob
import os
import os.path as ops
import time

import cv2
import glog as log
import numpy as np
import tensorflow as tf

import tqdm

from config import global_config
from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess

from random import sample

CFG = global_config.cfg


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, help='The source tusimple lane test data dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--model_path', type=str, help='The tflite model path')
    parser.add_argument('--save_dir', type=str, help='The test output save root dir')

    return parser.parse_args()


def src_dir_picker(src_dir, num=5564):
    """
    从 src_dir 中选取 num 个元素
    :param src_dir: the source tusimple lane test data dir
    :param num: the size of list
    """
    assert ops.exists(src_dir), '{:s} not exist'.format(src_dir)
    image_list = glob.glob('{:s}/**/*.jpg'.format(src_dir), recursive=True)
    image_list = sample(image_list, num)

    return image_list


def test_lanenet_batch(image_list, weights_path, save_dir):
    """

    :param src_dir:
    :param weights_path:
    :param save_dir:
    :return:
    """
    # assert ops.exists(src_dir), '{:s} not exist'.format(src_dir)
    save_dir = ops.join(save_dir, "ckpt")
    os.makedirs(save_dir, exist_ok=True)

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    postprocessor = lanenet_postprocess.LaneNetPostProcessor()

    saver = tf.train.Saver()

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        # image_list = glob.glob('{:s}/**/*.jpg'.format(src_dir), recursive=True)
        # image_list = sample(image_list, 5564)
        # 返回所有匹配的文件路径列表
        avg_time_cost = []
        for index, image_path in tqdm.tqdm(enumerate(image_list), total=len(image_list)):
                # tqdm: 进度条
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_vis = image
            image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
            image = image / 127.5 - 1.0

            t_start = time.time()
            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]}
            )
            avg_time_cost.append(time.time() - t_start)

            postprocess_result = postprocessor.postprocess(
                binary_seg_result=binary_seg_image[0],
                instance_seg_result=instance_seg_image[0],
                source_image=image_vis
            )

            if index % 100 == 0:
                log.info('Mean inference time every single image: {:.5f}s'.format(np.mean(avg_time_cost)))
                with open(ops.join(save_dir, 'log'), 'a') as log_file:
                    print('Mean inference time every single image: {:.5f}s'.format(np.mean(avg_time_cost)),
                          file=log_file)
                avg_time_cost.clear()

            input_image_dir = ops.split(image_path.split('clips')[1])[0][1:]
            input_image_name = ops.split(image_path)[1]
            output_image_dir = ops.join(save_dir, input_image_dir)
            os.makedirs(output_image_dir, exist_ok=True)
            output_image_path = ops.join(output_image_dir, input_image_name)
            if ops.exists(output_image_path):
                continue

            cv2.imwrite(output_image_path, postprocess_result['source_image'])

    return


def test_lanenet_batch_tflite(image_list, model_path, save_dir):
    """
    测试 tflite 模型的处理速度
    """
    save_dir = ops.join(save_dir, "tflite")
    os.makedirs(save_dir, exist_ok=True)

    interpreter = tf.lite.Interpreter(model_path=model_path)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.allocate_tensors()

    postprocessor = lanenet_postprocess.LaneNetPostProcessor()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    avg_time_cost = []
    for index, image_path in tqdm.tqdm(enumerate(image_list), total=len(image_list)):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_vis = image
        image = cv2.resize(image, (512,256), interpolation=cv2.INTER_LINEAR)
        image = image / 127.5 - 1.0
        image = image.reshape(1, 256, 512, 3)
        image = image.astype((np.float32))

        t_start = time.time()
        interpreter.set_tensor(input_details[0]['index'], image)

        interpreter.invoke()

        final_binary_output = interpreter.get_tensor(output_details[0]['index'])
        final_embedding_output = interpreter.get_tensor(output_details[1]['index'])
        avg_time_cost.append(time.time() - t_start)
        """
        postprocess_result =  postprocessor.postprocess(
            binary_seg_result=final_binary_output[0],
            instance_seg_result=final_embedding_output[0],
            source_image=image_vis
        )
"""
        if index % 100 == 0:
            log.info('Mean inference time every single image: {:.5f}s'.format(np.mean(avg_time_cost)))
            with open(ops.join(save_dir, 'log'), 'a') as log_file:
                print('Mean inference time every single image: {:.5f}s'.format(np.mean(avg_time_cost)), file=log_file)
            avg_time_cost.clear()
        """
        input_image_dir = ops.split(image_path.split('clips')[1])[0][1:]
        input_image_name = ops.split(image_path)[1]
        output_image_dir = ops.join(save_dir, input_image_dir)
        os.makedirs(output_image_dir, exist_ok=True)
        output_image_path = ops.join(output_image_dir, input_image_name)
        if ops.exists(output_image_path):
            continue

        cv2.imwrite(output_image_path, postprocess_result['source_image'])
"""
    return


if __name__ == '__main__':
    """
    test code
    """
    # init args
    args = init_args()

    image_list = src_dir_picker(args.image_dir)

    test_lanenet_batch(
       image_list=image_list,
       weights_path=args.weights_path,
       save_dir=args.save_dir
    )

    # test_lanenet_batch_tflite(
    #    image_list=image_list,
    #    model_path=args.model_path,
    #    save_dir=args.save_dir
    #)
