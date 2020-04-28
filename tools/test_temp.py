from lanenet_model import lanenet_postprocess
from config import global_config
import matplotlib.pyplot as plt
import cv2
import os.path as ops
import numpy as np
import re

import tensorflow as tf
from tensorflow.python.platform import gfile

CFG = global_config.cfg


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
            pts_new = []
            for pt in pts:
                pt = [int(float(pt.split(',')[0])), int(float(pt.split(',')[1]))]
                pts_new.append(pt)

            pts_new = np.array([pts_new], np.int64)
            print(pts_new)
            cv2.polylines(background, pts_new, isClosed=False, color=255, thickness=5)
        plt.figure("background")
        plt.imshow(background)
        plt.show()


if __name__ == '__main__':
    # pred_sample("/home/stevemaary/projects/lanenet-lane-detection/data/tusimple_test_image/00091.png",
    #             "/home/stevemaary/projects/lanenet-lane-detection/data/tusimple_test_image/00092.png",
    #             "/home/stevemaary/projects/lanenet-lane-detection/data/tusimple_test_image/0009.png")

    # show_graph_architecture("./model/tusimple_lanenet/tusimple_lanenet_vgg.ckpt.meta")

    #process_video('/media/stevemaary/新加卷/data/Jiqing Expressway Video/', 'IMG_0291.MOV')
    gengrate_test('/media/stevemaary/新加卷/data/caltech/caltech-lanes/label/cordova1/0.txt',
                  '/media/stevemaary/新加卷/data/caltech/caltech-lanes/cordova1/f00000.png')