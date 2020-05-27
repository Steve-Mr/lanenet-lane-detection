import os
import os.path as ops
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tqdm
import tensorflow as tf
from config import global_config
import math


from tools import evaluate_model_utils

CFG = global_config.cfg

def get_label_list(path):
    label_list = []
    for root, dirs, files in os.walk(path):
            for file in files:
                label_list.append(ops.join(path, file))
    return label_list

def get_label_list_culane(path):
    label_list = []
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            tmp_path = ops.join(root, dir)
            for root_, dirs_, files_ in os.walk(tmp_path):
                for file in files_:
                    label_list.append(ops.join(root_, file))
    return label_list


def draw_labeled_image(label_path, image):
    with open(label_path, 'r') as file:
        for line in file:
            pattern = re.compile(r'[(](.*?)[)]', re.S)
            pts = re.findall(pattern, line)
            pts_new = []
            for pt in pts:
                pt = [int(float(pt.split(',')[0])), int(float(pt.split(',')[1]))]
                pts_new.append(pt)
            pts_new = np.array([pts_new], np.int64)
            cv2.polylines(image, pts_new, isClosed=False, color=255, thickness=5)
    return image


def generate_label_image(src_path, label_path, dst_path):
    image_list = []
    with open(ops.join(src_path, 'list.txt'), 'r') as file:
        for line in file:
            image_list.append(ops.join(src_path, line.strip('\n')))
    label_list = get_label_list(label_path)
    for index,label_path in tqdm.tqdm(enumerate(label_list), total=len(label_list)):
        order = int((label_path.split('/')[-1]).split('.')[0])
        background = np.zeros(shape=(480, 640), dtype=np.uint8)
        # image = cv2.imread(image_list[order], cv2.IMREAD_COLOR)
        # image = draw_labeled_image(label_path, image)
        background = draw_labeled_image(label_path, background)
        # dst_img_path = dst_path + "/labeled_image/"
        dst_back_path = dst_path + "/label/"

        # if not ops.exists(dst_img_path):
        #    os.makedirs(dst_img_path)
        if not ops.exists(dst_back_path):
            os.makedirs(dst_back_path)

       #  dst_img_path = ops.join(dst_img_path, image_list[order].split('/')[-1])
        dst_back_path = ops.join(dst_back_path, image_list[order].split('/')[-1])
        # cv2.imwrite(dst_img_path, image)
        cv2.imwrite(dst_back_path,background)


def calculate_accuracy(pred_path, label_path):
    label_list = get_label_list_culane(label_path)
    accuracy_list = []
    accuracy_sum = 0
    a_list = []
    b_list = []

    for index, label_path in tqdm.tqdm(enumerate(label_list), total=len(label_list)):
        # pred_name = label_path.split('/')[-1]
        #fro caltech
        pred_name = label_path.split('/')[-2] + '/' + label_path.split('/')[-1]
        print(pred_name)
        print(label_path)
        pred = cv2.imread(ops.join(pred_path, pred_name), cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_COLOR)
        print(type(pred))
        if pred is None:
            accuracy_list.append(0.0)
            continue

        pred = pred.astype((np.float32))
        label = label.astype((np.float32))

        print(pred.shape)
        print(label.shape)

        accuracy = evaluate_model_utils.calculate_model_precision_for_test(pred, label)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.5
        sess_config.gpu_options.allow_growth = False
        sess_config.gpu_options.allocator_type = 'BFC'  # best fit with coalescing  内存管理算法
        sess_config.allow_soft_placement = True
        sess_config.log_device_placement = False
        sess = tf.Session(config=sess_config)

        with sess.as_default():
            # accuracy_result, fn_result, fp_result = sess.run([accuracy, fn, fp])
            accuracy_result, a, b = sess.run(accuracy)
            sess.close()
        tf.reset_default_graph()

        accuracy_list.append(accuracy_result)
        a_list.append(a)
        b_list.append(b)

        if b==0:
            accuracy_sum += 1
            continue

        if math.isnan(accuracy_result):
            accuracy_list.append('nan')
            continue

        accuracy_sum = accuracy_sum + accuracy_result

    print("accuracy ", accuracy_sum / len(label_list))
    print(accuracy_list)
    print(a_list)
    print(b_list)


def calculate_accuracy_jiqing(pred_path, label_path):
    label_list = get_label_list(label_path)
    pred_list = get_label_list(pred_path)
    accuracy_list = []
    # fp_list = []
    # fn_list = []
    accuracy_sum = 0
    # fp_sum = 0
    # fn_sum = 0
    a_list = []
    b_list = []

    for index, pred_path in tqdm.tqdm(enumerate(pred_list), total=len(pred_list)):
        label_name = pred_path.split('/')[-1]
        label = cv2.imread(ops.join(label_path, label_name), cv2.IMREAD_COLOR)
        pred = cv2.imread(pred_path, cv2.IMREAD_COLOR)

        pred = pred.astype((np.float32))
        label = label.astype((np.float32))

        pred = cv2.resize(pred, (512, 256), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (512, 256), interpolation=cv2.INTER_LINEAR)

        accuracy = evaluate_model_utils.calculate_model_precision_for_test(pred, label)
        fn = evaluate_model_utils.calculate_model_fn(pred, label)
        fp = evaluate_model_utils.calculate_model_fp(pred, label)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.5
        sess_config.gpu_options.allow_growth = False
        sess_config.gpu_options.allocator_type = 'BFC'  # best fit with coalescing  内存管理算法
        sess_config.allow_soft_placement = True
        sess_config.log_device_placement = False
        sess = tf.Session(config=sess_config)

        with sess.as_default():
            # accuracy_result, fn_result, fp_result = sess.run([accuracy, fn, fp])
            accuracy_result, a, b = sess.run(accuracy)
            sess.close()
        tf.reset_default_graph()

        accuracy_list.append(accuracy_result)
        a_list.append(a)
        b_list.append(b)
        # fp_list.append(fp_result)
        # fn_list.append(fn_result)

        accuracy_sum = accuracy_sum + accuracy_result
        # fp_sum = fp_sum + fp_result
        # fn_sum = fn_sum + fn_result

    print("accuracy ", accuracy_sum / len(label_list))
    # print("fp ", fp_sum/len(label_list))
    # print("fn ", fn_sum/len(label_list))
    print(accuracy_list)
    print(a_list)
    print(b_list)
    # print(fp_list)
    # print(fn_list)



if __name__ == '__main__':
    # generate_label_image('/media/stevemaary/新加卷/data/caltech/caltech-lanes/washington1',
    #                      '/media/stevemaary/新加卷/data/caltech/caltech-lanes/label/washington1',
    #                      '/media/stevemaary/新加卷/data/caltech/caltech-lanes/label_file/washington1')
    # label_list = get_label_list('/media/stevemaary/新加卷/data/caltech/caltech-lanes/label/washington1')
    # calculate_accuracy('/media/stevemaary/新加卷/data/caltech/caltech-lanes/pred/full/washington1/',
    #                    '/media/stevemaary/新加卷/data/caltech/caltech-lanes/label_file/washington1/label/')

    calculate_accuracy('/media/stevemaary/新加卷/data/culane/test8_night/full/',
                       '/media/stevemaary/新加卷/data/culane/test8_night/label/')

    # calculate_accuracy_jiqing('/media/stevemaary/新加卷/data/pred/IMG_0259/',
    #                    '/media/stevemaary/新加卷/data/Jiqing Expressway Video/label/0259/')
