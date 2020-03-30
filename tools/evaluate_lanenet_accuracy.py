import json
import argparse
import glob
import os
import os.path as ops
import time
import matplotlib.pyplot as plt

import cv2
import glog as log
import numpy as np
import tensorflow as tf

from tools import evaluate_model_utils
from data_provider import tf_io_pipline_tools

import tqdm

from config import global_config
from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess

CFG = global_config.cfg
np.set_printoptions(threshold=np.inf)

CROP_IMAGE_HEIGHT = CFG.TRAIN.IMG_HEIGHT  # 512
CROP_IMAGE_WIDTH = CFG.TRAIN.IMG_WIDTH  # 256
RESIZE_IMAGE_HEIGHT = CFG.TRAIN.IMG_HEIGHT + CFG.TRAIN.CROP_PAD_SIZE  # 256+32=288
RESIZE_IMAGE_WIDTH = CFG.TRAIN.IMG_WIDTH + CFG.TRAIN.CROP_PAD_SIZE  # 512+32=544


def load_image_path_from_json(file_path):
    image_list = []
    index = 1
    with open(file_path) as file:
        for json_obj in file:
            image_info = json.loads(json_obj)
            image_list.append(image_info)
            index += 1
            if index == 10: break
        print(image_list)
        print(image_list[3]['raw_file'])
    return


def evaluate_lanenet_accuracy_ckpt(data_path, weights_path):
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

    gt_image_path = ops.join(data_path, 'gt_image')
    gt_binary_path = ops.join(data_path, 'gt_binary_image')

    accuracy_sum = 0.0
    fp_sum = 0.0
    fn_sum = 0.0

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)
        image_path_list = os.listdir(gt_image_path)
        i = 0
        for image_path in image_path_list:
            image_path = ops.join(gt_image_path, image_path)
            print(image_path)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            # image_vis = image
            image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
            image = image / 127.5 - 1.0  # 归一化 (只归一未改变维数)

            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]}
            )

            image_name = ops.split(image_path)[1]
            gt_image = cv2.imread(ops.join(gt_binary_path, image_name), cv2.IMREAD_COLOR)

            image_vis = gt_image

            postprocess_result = postprocessor.postprocess_for_test(
                binary_seg_result=binary_seg_image[0],
                instance_seg_result=instance_seg_image[0],
                source_image=image_vis
            )

            plt.figure("result")
            plt.imshow(postprocess_result['mask_image'])
            plt.show()

            # binary_seg_image = binary_seg_image.astype((np.float32))
            # print(binary_seg_image.shape)

            # binary_seg_image = binary_seg_image.reshape(1, 256, 512, 1)
            # binary_seg_image = cv2.resize(postprocess_result['source_image'], (512, 256), interpolation=cv2.INTER_LINEAR)
            binary_seg_image = postprocess_result['source_image'].reshape(1, 720, 1280, 1)

            # gt_image = cv2.resize(gt_image, (512, 256), interpolation=cv2.INTER_LINEAR)

            gt_image = gt_image[:, :, 0].reshape(720, 1280, 1)

            binary_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 720, 1280, 1], name='binary_tensor')
            gt_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 720, 1280, 1], name='gt_tensor')

            accuracy, count, idx = evaluate_model_utils.calculate_model_precision_for_test(
                binary_tensor, gt_tensor
            )

            fn = evaluate_model_utils.calculate_model_fn_for_test(
                binary_tensor, gt_tensor
            )

            fp = evaluate_model_utils.calculate_model_fp_for_test(
                binary_tensor, gt_tensor
            )

            accuracy_result, count_result, idx = sess.run([accuracy, count, idx],
                                                          feed_dict={binary_tensor: binary_seg_image,
                                                                     gt_tensor: [gt_image]})

            fn_result = sess.run(fn,
                                feed_dict={binary_tensor: binary_seg_image,
                                           gt_tensor: [gt_image]}
                                )

            fp_result = sess.run(fp,
                                feed_dict={binary_tensor: binary_seg_image,
                                           gt_tensor:[gt_image]}
                                )

            print("accuracy:", accuracy_result, count_result, idx)
            print("fn:", fn_result)
            print("fp:", fp_result)

            i += 1
            accuracy_sum += accuracy_result
            fp_sum += fp_result
            fn_sum += fn_result
            if i == 100:
                print("average_accuracy: ", accuracy_sum / 100.0)
                print("average_fn: ", fn_sum / 100.0)
                print("average_fp: ", fp_sum / 100.0)
                break

    return


if __name__ == '__main__':
    # load_image_path_from_json('/home/stevemaary/data/test_baseline_sample.json')

    evaluate_lanenet_accuracy_ckpt('/home/stevemaary/data/testing/',
                                   './model/tusimple_lanenet/tusimple_lanenet_vgg.ckpt')
