import cv2
import os.path as ops
import os
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import tqdm
import numpy as np

from lanenet_model import lanenet, lanenet_postprocess
from config import global_config

CFG = global_config.cfg


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


def evaluate(src_path, pred_path, weights_path):
    image_list = get_all_files(src_path)
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    net = lanenet.LaneNet(phase='test', net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')
    postprocessor = lanenet_postprocess.LaneNetPostProcessor_for_nontusimple()

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

            postprocess_result = postprocessor.postprocess_for_non_tusimple(
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


def evaluate_on_caltech(src_path, pred_path, weights_path):
    image_list = []
    with open(ops.join(src_path, 'list.txt'), 'r') as file:
        for line in file:
            image_list.append(ops.join(src_path, line.strip('\n')))

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    net = lanenet.LaneNet(phase='test', net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')
    postprocessor = lanenet_postprocess.LaneNetPostProcessor_for_nontusimple()

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
            image = image[10:430, 40:600]
            image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
            image = image / 127.5 - 1.0  # 归一化 (只归一未改变维数)

            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]}
            )

            postprocess_result = postprocessor.postprocess_for_non_tusimple(
                binary_seg_result=binary_seg_image[0],
                instance_seg_result=instance_seg_image[0],
                source_image=image_vis,
                data_source='caltech'
            )

            dst_image_path = ops.join(pred_path, image_path.split('/')[-2])
            if not ops.exists(dst_image_path):
                os.makedirs(dst_image_path)
            dst_image_path = ops.join(dst_image_path, image_path.split('/')[-1])

            binary = lanenet_postprocess._morphological_process(binary_seg_image[0])
            print(binary.shape)
            print(postprocess_result['mask_image'].shape)

            mask = cv2.resize(binary, (560, 420), interpolation=cv2.INTER_LINEAR)
            back = np.zeros(shape=(480, 640), dtype=np.uint8)
            for i in range(560):
                for j in range(420):
                    back[j + 10, i + 40] = mask[j, i]
                    # back[j + 10, i + 40, 1] = mask[j, i, 1]
                    # back[j + 10, i + 40, 2] = mask[j, i, 2]

            # cv2.imwrite(dst_image_path, postprocess_result['mask_image'])    # postprocess_result['mask_image'])
            # cv2.imwrite(dst_image_path, image_vis)
            cv2.imwrite(dst_image_path, back*255)    # postprocess_result['mask_image'])


    return

def generate_on_vpgnet(scr_path, pred_path, weights_path):
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(scr_path):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    net = lanenet.LaneNet(phase='test', net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')
    postprocessor = lanenet_postprocess.LaneNetPostProcessor_for_nontusimple()

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

            postprocess_result = postprocessor.postprocess_for_non_tusimple(
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
            postprocessor = lanenet_postprocess.LaneNetPostProcessor_for_nontusimple()

            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]}
            )

            postprocess_result = postprocessor.postprocess_for_non_tusimple(
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



if __name__ == '__main__':
    """
    process_video('/media/stevemaary/新加卷/data/Jiqing Expressway Video/',
                  'IMG_0259.MOV',
                  '/media/stevemaary/新加卷/data/src/')
    """
    # get_all_files('/media/stevemaary/新加卷/data/src/')
    """
    evaluate('/media/stevemaary/新加卷/data/src/',
             '/media/stevemaary/新加卷/data/pred/',
             './model/tusimple_lanenet/tusimple_lanenet_vgg.ckpt')
    """

    """
    evaluate_on_caltech('/media/stevemaary/新加卷/data/caltech/caltech-lanes/cordova1',
                        '/media/stevemaary/新加卷/data/caltech/caltech-lanes/pred',
                        './model/tusimple_lanenet/tusimple_lanenet_vgg.ckpt')"""
    """
    generate_on_vpgnet('/media/stevemaary/新加卷/data/src/VPGNet/scene_4',
                       '/media/stevemaary/新加卷/data/pred/VPGNet',
                       './model/tusimple_lanenet/tusimple_lanenet_vgg.ckpt')
    """

    generate_on_jiqing('/media/stevemaary/新加卷/data/src/IMG_0259',
                       '/media/stevemaary/新加卷/data/pred/',
                       './model/tusimple_lanenet/tusimple_lanenet_vgg.ckpt')