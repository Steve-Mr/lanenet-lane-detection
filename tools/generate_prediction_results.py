import time

import tensorflow as tf
import argparse
import glob
import json
import os
import os.path as ops
import shutil
import matplotlib.pyplot as plt
from random import sample

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
    postprocessor = lanenet_postprocess.LaneNetPostProcessor_noremap()

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

                postprocess_result = postprocessor.postprocess_noremap(
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

def process_video(video_path, video_name, dst_path):
    count = 1
    video = cv2.VideoCapture(ops.join(video_path, video_name))
    if video.isOpened():
        ret, frame = video.read()
    else:
        ret = False

    gap = 30

    while ret:
        ret, frame = video.read()
        if count % gap == 0:
            path = dst_path + video_name.split('.')[0] + "/"
            if not ops.exists(path):
                os.makedirs(path)
            path = path + str(count) + ".png"
            cv2.imwrite(path, frame)
            print(count)

        count = count + 1
    video.release()
    print("finish")


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


def generate(src_path, pred_path, weights_path):
    image_list = get_all_files(src_path)
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    net = lanenet.LaneNet(phase='test', net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')
    postprocessor = lanenet_postprocess.LaneNetPostProcessor_noremap()

    saver = tf.train.Saver()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'  # best fit with coalescing  内存管理算法
    sess = tf.Session(config=sess_config)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)
        for image_path in image_list:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_vis = image
            # image = image[:][300:850]
            # image = cv2.resize(image, (1920, 1280), interpolation=cv2.INTER_LINEAR)
            image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
            image = image / 127.5 - 1.0  # 归一化 (只归一未改变维数)

            t_strat = time.time()
            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]}
            )

            t_inference = time.time() - t_strat

            postprocess_result = postprocessor.postprocess_noremap(
                binary_seg_result=binary_seg_image[0],
                instance_seg_result=instance_seg_image[0],
                source_image=image_vis,
                data_source='caltech'
            )
            t_cost = time.time() - t_strat
            print(t_inference, t_cost)

            dst_image_path = ops.join(pred_path, image_path.split('/')[-2])
            if not ops.exists(dst_image_path):
                os.makedirs(dst_image_path)
            dst_image_path = ops.join(dst_image_path, image_path.split('/')[-1])

            cv2.imwrite(dst_image_path, image_vis)    # postprocess_result['mask_image'])

    return


def generate_on_caltech(src_path, pred_path, weights_path):
    image_list = []
    with open(ops.join(src_path, 'list.txt'), 'r') as file:
        for line in file:
            image_list.append(ops.join(src_path, line.strip('\n')))

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    net = lanenet.LaneNet(phase='test', net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')
    postprocessor = lanenet_postprocess.LaneNetPostProcessor_noremap()

    saver = tf.train.Saver()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'  # best fit with coalescing  内存管理算法
    sess = tf.Session(config=sess_config)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        avg_inference_time_cost = []
        avg_full_time_cost = []

        for index, image_path in tqdm.tqdm(enumerate(image_list), total=len(image_list)):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_vis = image # np.zeros(shape=(480, 640, 3), dtype=np.uint8)
            image = image[10:430, 40:600]
            image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
            image = image / 127.5 - 1.0  # 归一化 (只归一未改变维数)

            # t_start = time.time()

            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]}
            )

            # avg_inference_time_cost.append(time.time() - t_start)

            postprocess_result = postprocessor.postprocess_noremap(
                binary_seg_result=binary_seg_image[0],
                instance_seg_result=instance_seg_image[0],
                source_image=image_vis,
                data_source='caltech'
            )

            # avg_full_time_cost.append(time.time() - t_start)

            dst_image_path = ops.join(pred_path, image_path.split('/')[-2])
            if not ops.exists(dst_image_path):
                os.makedirs(dst_image_path)
            dst_image_path = ops.join(dst_image_path, image_path.split('/')[-1])
            """
            dst_full_image_path = ops.join(pred_path, 'full' ,image_path.split('/')[-2])
            if not ops.exists(dst_full_image_path):
                os.makedirs(dst_full_image_path)
            dst_full_image_path = ops.join(dst_full_image_path, image_path.split('/')[-1])
            """

            """
            binary = lanenet_postprocess._morphological_process(binary_seg_image[0])

            mask = cv2.resize(binary, (560, 315), interpolation=cv2.INTER_LINEAR)
            back = np.zeros(shape=(480, 640), dtype=np.uint8)
            for i in range(560):
                for j in range(315):
                    back[j + 30, i + 40] = mask[j, i]
                    # back[j + 10, i + 40, 1] = mask[j, i, 1]
                    # back[j + 10, i + 40, 2] = mask[j, i, 2]


            cv2.imwrite(dst_image_path, back*255)    # postprocess_result['mask_image'])
            """
            cv2.imwrite(dst_image_path, postprocess_result['source_image'])
        """
        with open(ops.join(pred_path, 'log.txt'), 'a') as log_file:
            print(
                'Mean inference time every single image: {:.5f}s, Mean postprocess time every single image: {:.5f}s, Mean full time every single image: {:.5f}s,'
                .format(np.mean(avg_inference_time_cost),
                        np.mean(avg_full_time_cost) - np.mean(avg_inference_time_cost),
                        np.mean(avg_full_time_cost)),
                file=log_file)
        avg_inference_time_cost.clear()
        avg_full_time_cost.clear()
        """

    return


def generate_on_culane(src_path, list_file_name, pred_path, weights_path):
    image_list = []
    with open(ops.join(src_path,
                       'list/test_split',
                       list_file_name), 'r') as file:
        for line in file:
            image_list.append(ops.join(src_path, line.strip('\n')))

    image_list = sample(image_list, 300)

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    net = lanenet.LaneNet(phase='test', net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')
    postprocessor = lanenet_postprocess.LaneNetPostProcessor_noremap()

    saver = tf.train.Saver()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'  # best fit with coalescing  内存管理算法
    sess = tf.Session(config=sess_config)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        avg_inference_time_cost = []
        avg_full_time_cost = []

        for index, image_path in tqdm.tqdm(enumerate(image_list), total=len(image_list)):

            # if imghdr.what(image_path) == None: continue
            image_path = image_path.strip('\n')
            image_path = image_path.strip()

            label_path = image_path.rsplit('.', 1)[0] + '.lines.txt'
            with open(label_path, 'r') as file:
                label = np.zeros(shape=(590, 1640), dtype=np.uint8)
                for line in file:
                    pts = []
                    line = line.strip('\n')
                    raw_list = line.split()
                    for index in range(0, len(raw_list), 2):
                        pt = [int(float(raw_list[index])), int(float(raw_list[index+1]))]
                        pts.append(pt)
                    pts = np.array([pts], np.int64)
                    cv2.polylines(label, pts, isClosed=False, color=255, thickness=5)
            dst_label_path = ops.join(pred_path, list_file_name.split('.')[-2], 'label', image_path.split('/')[-2])
            if not ops.exists(dst_label_path):
                os.makedirs(dst_label_path)
            dst_label_path = ops.join(dst_label_path, image_path.split('/')[-1])
            cv2.imwrite(dst_label_path, label)

            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_vis = np.zeros(shape=(590, 1640, 3), dtype=np.uint8)
            # plt.figure('image')
            # plt.imshow(image)
            # plt.show()
            image = image[:430]
            image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
            image = image / 127.5 - 1.0  # 归一化 (只归一未改变维数)

            t_start = time.time()

            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]}
            )

            avg_inference_time_cost.append(time.time() - t_start)

            postprocess_result = postprocessor.postprocess_noremap(
                binary_seg_result=binary_seg_image[0],
                instance_seg_result=instance_seg_image[0],
                source_image=image_vis,
                data_source='culane'
            )

            avg_full_time_cost.append(time.time() - t_start)

            dst_image_path = ops.join(pred_path, list_file_name.split('.')[-2], image_path.split('/')[-2])
            if not ops.exists(dst_image_path):
                os.makedirs(dst_image_path)
            dst_image_path = ops.join(dst_image_path, image_path.split('/')[-1])

            dst_full_image_path = ops.join(pred_path, list_file_name.split('.')[-2], 'full' ,image_path.split('/')[-2])
            if not ops.exists(dst_full_image_path):
                os.makedirs(dst_full_image_path)
            dst_full_image_path = ops.join(dst_full_image_path, image_path.split('/')[-1])

            binary = lanenet_postprocess._morphological_process(binary_seg_image[0])

            mask = cv2.resize(binary, (1640, 430), interpolation=cv2.INTER_LINEAR)
            back = np.zeros(shape=(590, 1640), dtype=np.uint8)
            for i in range(1640):
                for j in range(430):
                    back[j, i] = mask[j, i]

            cv2.imwrite(dst_image_path, back*255)    # postprocess_result['mask_image'])
            cv2.imwrite(dst_full_image_path, postprocess_result['source_image'])

            with open(ops.join(pred_path, list_file_name.split('.')[-2], list_file_name), 'a') as pred_file:
                print(image_path, file=pred_file)

        with open(ops.join(pred_path, list_file_name.split('.')[-2], 'log.txt'), 'a') as log_file:
            print(
                'Mean inference time every single image: {:.5f}s, Mean postprocess time every single image: {:.5f}s, Mean full time every single image: {:.5f}s,'
                .format(np.mean(avg_inference_time_cost),
                        np.mean(avg_full_time_cost) - np.mean(avg_inference_time_cost),
                        np.mean(avg_full_time_cost)),
                file=log_file)
        avg_inference_time_cost.clear()
        avg_full_time_cost.clear()
    postprocessor.compute_mean_time()


    return

def generate_on_vpgnet(scr_path, pred_path, weights_path):
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(scr_path):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    net = lanenet.LaneNet(phase='test', net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')
    postprocessor = lanenet_postprocess.LaneNetPostProcessor_noremap()

    saver = tf.train.Saver()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'  # best fit with coalescing  内存管理算法
    sess = tf.Session(config=sess_config)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)
        for index, image_path in tqdm.tqdm(enumerate(listOfFiles), total=len(listOfFiles)):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_vis = image
            image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
            image = image / 127.5 - 1.0  # 归一化 (只归一未改变维数)

            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]}
            )

            postprocess_result = postprocessor.postprocess_noremap(
                binary_seg_result=binary_seg_image[0],
                instance_seg_result=instance_seg_image[0],
                source_image=image_vis,
                data_source='vpgnet'
            )

            dst_image_path = ops.join(pred_path, image_path.split('/')[-2])
            if not ops.exists(dst_image_path):
                os.makedirs(dst_image_path)
            dst_image_path = ops.join(dst_image_path, image_path.split('/')[-1])

            # cv2.imwrite(dst_image_path, postprocess_result['mask_image'])    # postprocess_result['mask_image'])
            cv2.imwrite(dst_image_path, image_vis)

    return


def generate_on_jiqing(src_path, pred_path, weights_path):
    image_list = get_all_files(src_path)
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    net = lanenet.LaneNet(phase='test', net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    saver = tf.train.Saver()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'  # best fit with coalescing  内存管理算法
    sess = tf.Session(config=sess_config)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)
        for index, image_path in tqdm.tqdm(enumerate(image_list), total=len(image_list)):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_vis = image
            image = image[540:840]
            image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
            image = image / 127.5 - 1.0  # 归一化 (只归一未改变维数)
            postprocessor = lanenet_postprocess.LaneNetPostProcessor_noremap()

            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]}
            )

            postprocess_result = postprocessor.postprocess_noremap(
                binary_seg_result=binary_seg_image[0],
                instance_seg_result=instance_seg_image[0],
                source_image=image_vis,
                data_source='jiqing'
            )

            # binary = lanenet_postprocess._morphological_process(binary_seg_image[0])
            mask = postprocess_result['mask_image']

            # binary = lanenet_postprocess._morphological_process(mask[0])
            binary = cv2.resize(mask, (1920, 300), interpolation=cv2.INTER_LINEAR)
            back = np.zeros(shape=(1080, 1920, 3), dtype=np.uint8)
            for i in range(1920):
                for j in range(300):
                    # back[j+540, i] = binary[j, i]
                    back[j + 540, i, 0] = binary[j, i, 0]
                    back[j + 540, i, 1] = binary[j, i, 1]
                    back[j + 540, i, 2] = binary[j, i, 2]
            dst_image_path = ops.join(pred_path, image_path.split('/')[-2])
            if not ops.exists(dst_image_path):
                os.makedirs(dst_image_path)
            dst_image_path = ops.join(dst_image_path, image_path.split('/')[-1])

            cv2.imwrite(dst_image_path, back)
    return


def generate_tusimple_sample(scr_path, pred_path, weights_path):
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(scr_path):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    print(listOfFiles)

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    net = lanenet.LaneNet(phase='test', net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')
    postprocessor = lanenet_postprocess.LaneNetPostProcessor_noremap()

    saver = tf.train.Saver()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'  # best fit with coalescing  内存管理算法
    sess = tf.Session(config=sess_config)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)
        for index, image_path in tqdm.tqdm(enumerate(listOfFiles), total=len(listOfFiles)):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_vis = image
            image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
            image = image / 127.5 - 1.0  # 归一化 (只归一未改变维数)

            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]}
            )

            postprocess_result = postprocessor.postprocess_noremap(
                binary_seg_result=binary_seg_image[0],
                instance_seg_result=instance_seg_image[0],
                source_image=image_vis,
            )

            dst_image_path = ops.join(pred_path, image_path.split('/')[-2])
            if not ops.exists(dst_image_path):
                os.makedirs(dst_image_path)
            dst_image_path = ops.join(dst_image_path, image_path.split('/')[-1])

            cv2.imwrite(dst_image_path, image_vis)

    return


if __name__ == '__main__':

    # generate_prediction_result('/media/stevemaary/68A0799BA0797104/Users/a1975/Documents/lanenet_related_files/', '/home/stevemaary/data/pred', './model/tusimple_lanenet/tusimple_lanenet_vgg.ckpt')

    """
        process_video('/media/stevemaary/新加卷/data/Jiqing Expressway Video/',
                      'IMG_0259.MOV',
                      '/media/stevemaary/新加卷/data/src/')
        """
    # get_all_files('/media/stevemaary/新加卷/data/src/')
    """
    generate('/media/stevemaary/新加卷/data/src/',
             '/media/stevemaary/新加卷/data/pred/',
             './model/tusimple_lanenet/tusimple_lanenet_vgg.ckpt')
    """
    """
    generate_on_caltech('/media/stevemaary/新加卷/data/caltech/caltech-lanes/cordova1',
                        '/media/stevemaary/新加卷/data/caltech/caltech-lanes/pred',
                        './model/tusimple_lanenet/tusimple_lanenet_vgg.ckpt')
    """
    """
    generate_on_culane('/media/stevemaary/68A0799BA0797104/Users/a1975/Documents/lanenet_related_files/culane',
                       'test8_night.txt',
                       '/media/stevemaary/新加卷/data/culane',
                       './model/tusimple_lanenet/tusimple_lanenet_vgg.ckpt')
    """
    """
    generate_on_vpgnet('/media/stevemaary/新加卷/data/src/VPGNet/scene_4',
                       '/media/stevemaary/新加卷/data/pred/VPGNet',
                       './model/tusimple_lanenet/tusimple_lanenet_vgg.ckpt')
    """

    """
    generate_on_jiqing('/media/stevemaary/新加卷/data/src/IMG_0259',
                       '/media/stevemaary/新加卷/data/pred/',
                       './model/tusimple_lanenet/tusimple_lanenet_vgg.ckpt')
    """

    generate_tusimple_sample('/media/stevemaary/68A0799BA0797104/Users/a1975/Documents/lanenet_related_files/clips/0530/1492629957554176874_0',
                             '/media/stevemaary/新加卷/data/pred/sample/',
                             './model/tusimple_lanenet/tusimple_lanenet_vgg.ckpt')

    """
    generate_on_caltech('/media/stevemaary/新加卷/data/caltech/caltech-lanes/cordova1',
                        '/media/stevemaary/新加卷/data/pred/sample',
                        './model/tusimple_lanenet/tusimple_lanenet_vgg.ckpt')
    """