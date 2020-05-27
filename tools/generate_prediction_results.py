import time

import tensorflow as tf
import argparse
import glob
import json
import os
import os.path as ops
import shutil
import matplotlib.pyplot as plt

import cv2
import tqdm
import numpy as np

from config import global_config
from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess

CFG = global_config.cfg


def generate_prediction_result(src_dir, dst_dir, weights_path):
    """
    generate prediction results for evaluate
    """
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    # postprocessor = lanenet_postprocess.LaneNetPostProcessor()
    postprocessor = lanenet_postprocess.LaneNetPostProcessor_for_nontusimple()

    saver = tf.train.Saver()

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    task_file_path = ops.join(src_dir, "test_tasks.json")
    target_image_path = ops.join(dst_dir, "results")

    image_nums = len(os.listdir(target_image_path))

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        avg_inference_time_cost = []
        avg_full_time_cost = []

        with open(task_file_path, 'r') as file:

            count = -1
            for count, line in enumerate(file):
                pass
            count += 1

        with open(task_file_path, 'r') as file:

            for line_index, line in tqdm.tqdm(enumerate(file),
                                              total= count):
                test_task = json.loads(line)

                image_dir = ops.split(test_task['raw_file'])[0]
                image_dir_split = image_dir.split('/')[1:]
                image_dir_split.append(ops.split(test_task['raw_file'])[1])
                image_path = ops.join(src_dir, test_task['raw_file'])
                assert ops.exists(image_path), '{:s} not exist'.format(image_path)

                h_samples = test_task['h_samples']
                raw_file = test_task['raw_file']

                image_name_new = '{:s}.png'.format('{:d}'.format(line_index + image_nums).zfill(4))

                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image_vis = image
                image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
                image = image / 127.5 - 1.0  # 归一化 (只归一未改变维数)

                t_start = time.time()

                binary_seg_image, instance_seg_image = sess.run(
                    [binary_seg_ret, instance_seg_ret],
                    feed_dict={input_tensor: [image]}
                )

                avg_inference_time_cost.append(time.time() - t_start)

                postprocess_result = postprocessor.postprocess_for_non_tusimple(
                    binary_seg_result=binary_seg_image[0],
                    instance_seg_result=instance_seg_image[0],
                    source_image=image_vis
                )

                avg_full_time_cost.append(time.time() - t_start)

                if line_index % 100 == 0:
                    with open(ops.join(dst_dir, 'log'), 'a') as log_file:
                        print('Mean inference time every single image: {:.5f}s, Mean postprocess time every single image: {:.5f}s, Mean full time every single image: {:.5f}s,'
                              .format(np.mean(avg_inference_time_cost),
                                      np.mean(avg_full_time_cost)-np.mean(avg_inference_time_cost),
                                      np.mean(avg_full_time_cost)),
                              file=log_file)
                    avg_inference_time_cost.clear()
                    avg_full_time_cost.clear()

                dst_image_path = ops.join(target_image_path, image_name_new)

                # cv2.imwrite(dst_image_path, image_vis)

                with open(ops.join(dst_dir, "result.json"), 'a') as result_file:
                    data = {
                        'h_samples': h_samples,
                        'lanes': postprocess_result['lane_pts'],
                        'raw_file': raw_file,
                        'result_file': ops.join(dst_image_path)
                    }
                    json.dump(data, result_file)
                    result_file.write('\n')
        postprocessor.compute_mean_time()
    return


if __name__ == '__main__':

    generate_prediction_result('/media/stevemaary/68A0799BA0797104/Users/a1975/Documents/lanenet_related_files/', '/home/stevemaary/data/pred', './model/tusimple_lanenet/tusimple_lanenet_vgg.ckpt')