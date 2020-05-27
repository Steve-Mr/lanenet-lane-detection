import tqdm

from lanenet_model import lanenet_postprocess
from config import global_config
import matplotlib.pyplot as plt
import cv2
import os.path as ops
import numpy as np
import re
import os

import tensorflow as tf
from tensorflow.python.platform import gfile

CFG = global_config.cfg


def get_label_list(path):
    label_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            label_list.append(ops.join(path, file))
    return label_list

def pred_sample(binary_path, instance_path, image_path):
    binary = cv2.resize(cv2.imread(binary_path),(512,256), interpolation=cv2.INTER_LINEAR)/127.5 - 1.0
    instance = cv2.resize(cv2.imread(instance_path),(512,256), interpolation=cv2.INTER_LINEAR)/1217.5 - 1.0
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    print(instance.shape)

    postprocessor = lanenet_postprocess.LaneNetPostProcessor()
    postprocess_result = postprocessor.postprocess_for_test(
        binary_seg_result=binary[:, :, 0],
        instance_seg_result=instance,
        source_image=image
    )

    mask_image = postprocess_result['mask_image']

    # plt.figure('mask_image')
    # plt.imshow(mask_image[:, :, (2, 1, 0)])
    # plt.imshow(mask_image)
    plt.figure('src_image')
    plt.imshow(image[:, :, (2, 1, 0)])
    """"
    plt.figure("result")
    plt.imshow(postprocess_result['source_image'])
    """
    plt.show()

def show_graph_architecture (model_path):
    graph = tf.get_default_graph()
    graphdef = graph.as_graph_def()
    _ = tf.train.import_meta_graph(model_path)
    summary_write = tf.summary.FileWriter("./", graph)

def process_video(video_path, video_name):
    video = cv2.VideoCapture(ops.join(video_path, video_name))
    if video.isOpened():
        for index in range(0, 5393):
            ret, frame = video.read()
            path = video_path + "/frames/" + str(index) + '.png'
            cv2.imwrite(path, frame)
            print(index)
        print("finished")

def gengrate_test(path, image_path):
    background = np.zeros(shape=(480, 640), dtype=np.uint8)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    with open(path, 'r') as file:
        for line in file:
            pattern = re.compile(r'[(](.*?)[)]', re.S)
            print(re.findall(pattern, line))
            pts = re.findall(pattern, line)
            coord_list = []
            for pt in pts:
                pt = [int(float(pt.split(',')[0])), int(float(pt.split(',')[1]))]
                coord_list.append(pt)
            coord_list = np.array([coord_list], np.int64)
            print(coord_list)
            cv2.polylines(background, coord_list, isClosed=False, color=255, thickness=5)
        plt.figure("background")
        plt.imshow(background)
        plt.show()

def generate_jiqing_label(src_path, dst_path):
    label_file_list = get_label_list(src_path)
    for index, label_file_path in tqdm.tqdm(enumerate(label_file_list), total = len(label_file_list)):
        background = np.zeros(shape=(1080, 1920), dtype=np.uint8)
        with open(label_file_path) as file:
            for line in file:
                coords = line.split(":")[-1]
                pattern = re.compile(r'[(](.*?)[)]', re.S)
                pts = re.findall(pattern, coords)
                coord_list = []
                for pt in pts:
                    pt = [int(float(pt.split(',')[0])), int(float(pt.split(',')[1]))]
                    coord_list.append(pt)
                coord_list = np.array([coord_list], np.int64)
                cv2.polylines(background, coord_list, isClosed=False, color=255, thickness=5)
            folder_path = ops.join(dst_path, src_path.split("/")[-1])
            if not ops.exists(folder_path):
                os.makedirs(folder_path)
            # file_path = folder_path + str(index) + '.png'
            file_path = ops.join(folder_path, str(index) + '.png')
            cv2.imwrite(file_path, background)


def compute_mean_time(src_path):
    inference_full = 0.0
    postprocess_full = 0.0
    full = 0.0
    with open(src_path, 'r') as file:
        pattern = re.compile(r"\d+\.?\d*", re.S)
        for index, line in enumerate(file):
            if index == 0: continue
            if index == 27:
                time_list = re.findall(pattern, line)
                inference_full += float(time_list[0])*0.81
                postprocess_full += float(time_list[1])*0.81
                full += float(time_list[2])*0.81
                continue
            print(re.findall(pattern, line))
            time_list = re.findall(pattern, line)
            inference_full+=float(time_list[0])
            postprocess_full+=float(time_list[1])
            full+=float(time_list[2])
        print("mean_inference: ", inference_full/index,
              "mean_postprocess: ", postprocess_full/index,
              "mean_full: ", full/index)


if __name__ == '__main__':
    # pred_sample("/home/stevemaary/projects/lanenet-lane-detection/data/tusimple_test_image/00091.png",
    #             "/home/stevemaary/projects/lanenet-lane-detection/data/tusimple_test_image/00092.png",
    #             "/home/stevemaary/projects/lanenet-lane-detection/data/tusimple_test_image/0009.png")

    # show_graph_architecture("./model/tusimple_lanenet/tusimple_lanenet_vgg.ckpt.meta")

    # process_video('/media/stevemaary/新加卷/data/Jiqing Expressway Video/', 'IMG_0291.MOV')
    # gengrate_test('/media/stevemaary/新加卷/data/caltech/caltech-lanes/label/cordova1/0.txt',
    #               '/media/stevemaary/新加卷/data/caltech/caltech-lanes/cordova1/f00000.png')

    # generate_jiqing_label('/media/stevemaary/新加卷/data/Jiqing Expressway Video/Lane_Parameters/0259',
    #                       '/media/stevemaary/新加卷/data/Jiqing Expressway Video/label')

    compute_mean_time("/home/stevemaary/data/pred/default/log")