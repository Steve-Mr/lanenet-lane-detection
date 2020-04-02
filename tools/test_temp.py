from lanenet_model import lanenet_postprocess
from config import global_config
import matplotlib.pyplot as plt
import cv2

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


if __name__ == '__main__':
    # pred_sample("/home/stevemaary/projects/lanenet-lane-detection/data/tusimple_test_image/00091.png",
    #             "/home/stevemaary/projects/lanenet-lane-detection/data/tusimple_test_image/00092.png",
    #             "/home/stevemaary/projects/lanenet-lane-detection/data/tusimple_test_image/0009.png")

    show_graph_architecture("./model/tusimple_lanenet/tusimple_lanenet_vgg.ckpt.meta")