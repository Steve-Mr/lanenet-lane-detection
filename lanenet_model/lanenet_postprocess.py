#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-30 上午10:04
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_postprocess.py
# @IDE: PyCharm Community Edition
"""
LaneNet model post process
"""
import os.path as ops
import math
import time

import cv2
import glog as log
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from config import global_config

CFG = global_config.cfg
np.set_printoptions(threshold=np.inf)


def _morphological_process(image, kernel_size=5):
    """
    morphological process to fill the hole in the binary segmentation result
    :param image:
    :param kernel_size:
    :return:
    """
    if len(image.shape) == 3:
        raise ValueError('Binary segmentation result image should be a single channel image')

    if image.dtype is not np.uint8:
        image = np.array(image, np.uint8)

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))
    """
    cv2.getStructuringElement( ) 返回指定形状和尺寸的结构元素。
    这个函数的第一个参数表示内核的形状，有三种形状可以选择。
    
    矩形：MORPH_RECT;
    交叉形：MORPH_CROSS;
    椭圆形：MORPH_ELLIPSE;

    第二和第三个参数分别是内核的尺寸以及锚点的位置。一般在调用erode以及dilate函数之前，先定义一个Mat类型的变量来获得
    getStructuringElement函数的返回值: 对于锚点的位置，有默认值Point（-1,-1），表示锚点位于中心点。element形状唯一依赖锚点位置，其他情况下，锚点只是影响了形态学运算结果的偏移。
    """

    # close operation fill hole
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
    """
    cv2.MORPH_CLOSE: 闭运算：先膨胀后腐蚀（先膨胀会使白色的部分扩张，以至于消除“闭合”物体里面的小黑洞）
    """
    # opening = cv2.dilate(closing,kernel,iterations=1)

    return closing   # opening


def _connect_components_analysis(image):
    """
    connect components analysis to remove the small components
    :param image:
    :return:
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 颜色空间转换 BGR 转灰度图片
    else:
        gray_image = image

    return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)


class _LaneFeat(object):
    """

    """

    def __init__(self, feat, coord, class_id=-1):

        """
        lane feat object
        :param feat: lane embeddng feats [feature_1, feature_2, ...]
        :param coord: lane coordinates [x, y]
        :param class_id: lane class id
        """
        self._feat = feat
        self._coord = coord
        self._class_id = class_id

    @property
    def feat(self):

        """

        :return:
        """
        return self._feat

    @feat.setter
    def feat(self, value):

        """

        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float64)

        if value.dtype != np.float32:
            value = np.array(value, dtype=np.float64)

        self._feat = value

    @property
    def coord(self):

        """

        :return:
        """
        return self._coord

    @coord.setter
    def coord(self, value):

        """

        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value)

        if value.dtype != np.int32:
            value = np.array(value, dtype=np.int32)

        self._coord = value

    @property
    def class_id(self):

        """

        :return:
        """
        return self._class_id

    @class_id.setter
    def class_id(self, value):

        """

        :param value:
        :return:
        """
        if not isinstance(value, np.int64):
            raise ValueError('Class id must be integer')

        self._class_id = value


class _LaneNetCluster(object):
    """
     Instance segmentation result cluster
     实例分割聚类器
    """

    def __init__(self):

        """

        """
        self._color_map = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]

    @staticmethod
    def _embedding_feats_dbscan_cluster(embedding_image_feats, eps=0.35, min_samples=100):# 0.50 700 (2)200

        """
        dbscan cluster
        :param embedding_image_feats:
        :return:
        """
        # db = DBSCAN(eps=CFG.POSTPROCESS.DBSCAN_EPS, min_samples=CFG.POSTPROCESS.DBSCAN_MIN_SAMPLES)
        db = DBSCAN(eps=eps, min_samples=min_samples)  # CFG.POSTPROCESS.DBSCAN_MIN_SAMPLES)

        """
        eps: 扫描半径 (0.35)
        min_samples：作为核心点其邻域中的最小样本数（包括点本身）(1000)
        """
        try:
            features = StandardScaler().fit_transform(embedding_image_feats)
            # fit_transform 不仅计算训练数据的均值和方差，
            # 还会基于计算出来的均值和方差来转换训练数据，从而把数据转换成标准的正太分布

            db.fit(features)
            """
            Perform DBSCAN clustering from features or distance matrix.
            """

        except Exception as err:
            log.error(err)
            ret = {
                'origin_features': None,
                'cluster_nums': 0,
                'db_labels': None,
                'unique_labels': None,
                'cluster_center': None
            }
            return ret
        db_labels = db.labels_
        # 标记所有像素点
        unique_labels = np.unique(db_labels)

        num_clusters = len(unique_labels)
        cluster_centers = db.components_

        ret = {
            'origin_features': features,
            'cluster_nums': num_clusters,
            'db_labels': db_labels,
            'unique_labels': unique_labels,
            'cluster_center': cluster_centers
        }

        return ret

    @staticmethod
    def _get_lane_embedding_feats(binary_seg_ret, instance_seg_ret):

        """
        通过二值分割掩码图在实例分割图上获取所有车道线的特征向量
        get lane embedding features according the binary seg result
        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        idx = np.where(binary_seg_ret == 255)
        # 只有一个参数，输出满足条件的元素**坐标**
        lane_embedding_feats = instance_seg_ret[idx]

        lane_coordinate = np.vstack((idx[1], idx[0])).transpose()
        """
        vstack() 沿着竖直方向将矩阵堆叠起来。
        使用 transpose() ：
            将 vstack 结果 [[237 234 235 ... 502 503 504]
                            [ 87  88  88 ... 255 255 255]]
            转换为         [[237  87]
                             [234  88]
                             [235  88]
                             ...
                             [502 255]
                             [503 255]
                             [504 255]] 形式。
        """
        # only for test
        tmp = np.zeros(shape=(1080, 1920), dtype=np.uint8)
        for coord in lane_coordinate:
            cv2.circle(tmp, (coord[0], coord[1]), 5, 255, -1)
        # plt.figure("feat")
        # plt.imshow(tmp)

        assert lane_embedding_feats.shape[0] == lane_coordinate.shape[0]

        ret = {
            'lane_embedding_feats': lane_embedding_feats,
            'lane_coordinates': lane_coordinate
        }

        return ret

    def apply_lane_feats_cluster(self, binary_seg_result, instance_seg_result):

        """
        :param binary_seg_result:
        :param instance_seg_result:
        :return:
        """

        # t_start = time.time()

        get_lane_embedding_feats_result = self._get_lane_embedding_feats(
            binary_seg_ret=binary_seg_result,
            instance_seg_ret=instance_seg_result
        )

        # t_feat = time.time() - t_start
        # get embedding feats and coords
        """
        {   'lane_embedding_feats': lane_embedding_feats,
            'lane_coordinates': lane_coordinate
        }
        """

        # dbscan cluster
        dbscan_cluster_result = self._embedding_feats_dbscan_cluster(
            embedding_image_feats=get_lane_embedding_feats_result['lane_embedding_feats']
        )
        if dbscan_cluster_result['cluster_nums'] < 2:
            dbscan_cluster_result = self._embedding_feats_dbscan_cluster(
                embedding_image_feats=get_lane_embedding_feats_result['lane_embedding_feats'],
                eps=0.45,
                min_samples=500
            )
        """
        ret = {
            'origin_features': features,
            'cluster_nums': num_clusters,
            'db_labels': db_labels,
            'unique_labels': unique_labels,
            'cluster_center': cluster_centers
        }
        """

        # t_dbscan = time.time() - t_start

        mask = np.zeros(shape=[binary_seg_result.shape[0], binary_seg_result.shape[1], 3], dtype=np.uint8)
        # 将 mask 初始化为空白矩阵（图）
        """
        用法：zeros(shape, dtype=float, order='C')
        返回：返回来一个给定形状和类型的用0填充的数组；
        参数：shape:形状
              dtype:数据类型，可选参数，默认numpy.float64
            """
        db_labels = dbscan_cluster_result['db_labels']  # example: [-1 -1 -1 ...  2 -1 -1]
        unique_labels = dbscan_cluster_result['unique_labels']  # example: [-1  0  1  2  3  4]
        coord = get_lane_embedding_feats_result['lane_coordinates']

        if db_labels is None:
            return None, None

        lane_coords = []

        for index, label in enumerate(unique_labels.tolist()):
            if label == -1:  # -1 为背景像素点
                continue
            idx = np.where(db_labels == label)
            pix_coord_idx = tuple((coord[idx][:, 1], coord[idx][:, 0]))  # tuple 即元组 不可修改
            # 获得所有属于该 label 的点坐标
            # coord[idx]第1列全部，coord[idx]第0列全部
            mask[pix_coord_idx] = self._color_map[0]    # [index]
            # 给线条上色
            lane_coords.append(coord[idx])
        # t_end = time.time() - t_start
        # print("cluster ", t_feat, t_dbscan, t_end)
        return mask, lane_coords

    def apply_lane_feats_cluster_for_test(self, binary_seg_result, instance_seg_result):

        """

        :param binary_seg_result:
        :param instance_seg_result:
        :return:
        """

        get_lane_embedding_feats_result = self._get_lane_embedding_feats(
            binary_seg_ret=binary_seg_result,
            instance_seg_ret=instance_seg_result
        )
        # get embedding feats and coords
        """
        {   'lane_embedding_feats': lane_embedding_feats,
            'lane_coordinates': lane_coordinate
        }
        """

        # dbscan cluster
        dbscan_cluster_result = self._embedding_feats_dbscan_cluster(
            embedding_image_feats=get_lane_embedding_feats_result['lane_embedding_feats']
        )
        """
        ret = {
            'origin_features': features,
            'cluster_nums': num_clusters,
            'db_labels': db_labels,
            'unique_labels': unique_labels,
            'cluster_center': cluster_centers
        }
        """

        mask = np.zeros(shape=[binary_seg_result.shape[0], binary_seg_result.shape[1], 3], dtype=np.uint8)
        # 将 mask 初始化为空白矩阵（图）
        """
        用法：zeros(shape, dtype=float, order='C')
        返回：返回来一个给定形状和类型的用0填充的数组；
        参数：shape:形状
              dtype:数据类型，可选参数，默认numpy.float64
            """
        db_labels = dbscan_cluster_result['db_labels']  # example: [-1 -1 -1 ...  2 -1 -1]
        unique_labels = dbscan_cluster_result['unique_labels']  # example: [-1  0  1  2  3  4]
        coord = get_lane_embedding_feats_result['lane_coordinates']

        if db_labels is None:
            return None, None

        lane_coords = []

        for index, label in enumerate(unique_labels.tolist()):
            if label == -1:  # -1 为背景像素点
                continue
            idx = np.where(db_labels == label)
            pix_coord_idx = tuple((coord[idx][:, 1], coord[idx][:, 0]))  # tuple 即元组 不可修改
            # 获得所有属于该 label 的点坐标
            # coord[idx]第1列全部，coord[idx]第0列全部
            mask[pix_coord_idx] = 255  # self._color_map[index]
            # 给线条上色
            lane_coords.append(coord[idx])

        return mask, lane_coords


class LaneNetPostProcessor(object):
    """
    lanenet post process for lane generation
    """

    def __init__(self, ipm_remap_file_path='./data/tusimple_ipm_remap.yml'):

        """

        :param ipm_remap_file_path: ipm generate file path
        """
        assert ops.exists(ipm_remap_file_path), '{:s} not exist'.format(ipm_remap_file_path)

        self._cluster = _LaneNetCluster()
        self._ipm_remap_file_path = ipm_remap_file_path
        # 替代 H-NET 作用

        remap_file_load_ret = self._load_remap_matrix()
        # 读取预存 remap 矩阵
        self._remap_to_ipm_x = remap_file_load_ret['remap_to_ipm_x']
        self._remap_to_ipm_y = remap_file_load_ret['remap_to_ipm_y']

        self._color_map = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]

        self.pre = []
        self.cluster = []
        self.fit = []
        self.full = []

    def _load_remap_matrix(self):

        """

        :return:
        """
        fs = cv2.FileStorage(self._ipm_remap_file_path, cv2.FILE_STORAGE_READ)

        remap_to_ipm_x = fs.getNode('remap_ipm_x').mat()
        # 读取矩阵型节点
        remap_to_ipm_y = fs.getNode('remap_ipm_y').mat()

        ret = {
            'remap_to_ipm_x': remap_to_ipm_x,
            'remap_to_ipm_y': remap_to_ipm_y,
        }

        fs.release()

        return ret

    def postprocess(self, binary_seg_result, instance_seg_result=None,
                    min_area_threshold=100, source_image=None,
                    data_source='tusimple'):

        """

        :param binary_seg_result:
        :param instance_seg_result:
        :param min_area_threshold: 连通域分析阈值
        :param source_image:
        :param data_source:
        :return:
        """
        # convert binary_seg_result
        #
        binary_seg_result = np.array(binary_seg_result * 255, dtype=np.uint8)

        # 首先进行图像形态学运算 闭运算 先膨胀后腐蚀
        # apply image morphology operation to fill in the hold and reduce the small area
        morphological_ret = _morphological_process(binary_seg_result, kernel_size=5)

        # 进行连通域分析
        connect_components_analysis_ret = _connect_components_analysis(image=morphological_ret)

        # 排序连通域并删除过小的连通域
        labels = connect_components_analysis_ret[1]
        stats = connect_components_analysis_ret[2]
        """
        Get the results
        The first cell is the number of labels 
        The second cell is the label matrix
            (Labels is a matrix the size of the input image where each element has a value equal to its label).
        The third cell is the stat matrix 
            (Stats is a matrix of the stats that the function calculates. It has a length equal to the number of labels 
             and a width equal to the number of stats.)
            Statistics output for each label, including the background label, see below for available statistics. 
            Statistics are accessed via stats[label, COLUMN] where available columns are defined below.
                cv2.CC_STAT_LEFT The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction.
                cv2.CC_STAT_TOP The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction.
                cv2.CC_STAT_WIDTH The horizontal size of the bounding box
                cv2.CC_STAT_HEIGHT The vertical size of the bounding box
                cv2.CC_STAT_AREA The total area (in pixels) of the connected component
        """

        for index, stat in enumerate(stats):
            if stat[4] <= min_area_threshold:
                idx = np.where(labels == index)
                morphological_ret[idx] = 0

        # apply embedding features cluster
        mask_image, lane_coords = self._cluster.apply_lane_feats_cluster(
            binary_seg_result=morphological_ret,
            instance_seg_result=instance_seg_result
        )

        if mask_image is None:
            return {
                'mask_image': None,
                'fit_params': None,
                'source_image': None,
            }

        """
        lane line fit
        """

        fit_params = []
        src_lane_pts = []  # lane pts every single lane
        for lane_index, coords in enumerate(lane_coords):
            if data_source == 'tusimple':
                tmp_mask = np.zeros(shape=(720, 1280), dtype=np.uint8)
                tmp_mask[tuple((np.int_(coords[:, 1] * 720 / 256), np.int_(coords[:, 0] * 1280 / 512)))] = 255
                # 新建空白 tmp_mask 并根据 mask_image 已有坐标进行变换后改变对应坐标的值
            elif data_source == 'beec_ccd':
                tmp_mask = np.zeros(shape=(1350, 2448), dtype=np.uint8)
                tmp_mask[tuple((np.int_(coords[:, 1] * 1350 / 256), np.int_(coords[:, 0] * 2448 / 512)))] = 255
            else:
                raise ValueError('Wrong data source now only support tusimple and beec_ccd')
            tmp_ipm_mask = cv2.remap(
                tmp_mask,
                self._remap_to_ipm_x,
                self._remap_to_ipm_y,
                interpolation=cv2.INTER_NEAREST
            )
            # 使用预设 ipm 进行视角转换 INTER_NEAREST——最近邻插值
            nonzero_y = np.array(tmp_ipm_mask.nonzero()[0])
            nonzero_x = np.array(tmp_ipm_mask.nonzero()[1])

            fit_param = np.polyfit(nonzero_y, nonzero_x, 2)
            # 进行拟合，目标函数为二次函数
            fit_params.append(fit_param)

            [ipm_image_height, ipm_image_width] = tmp_ipm_mask.shape
            plot_y = np.linspace(10, ipm_image_height, ipm_image_height - 10)
            # linspace(start, stop, num) 生成从 start 到 stop num 个数的等差数列
            fit_x = fit_param[0] * plot_y ** 2 + fit_param[1] * plot_y + fit_param[2]

            lane_pts = []
            # lane points 车道线点坐标
            for index in range(0, plot_y.shape[0], 5):
                src_x = self._remap_to_ipm_x[
                    int(plot_y[index]), int(np.clip(fit_x[index], 0, ipm_image_width - 1))]
                # np.clip(a, a_min, a_max, out=None) 限制 a 在 a_min 和 a_max 之间，超出部分则设置为 a_min 和 a_max
                # src_x 为 remap ipm x 矩阵 index 对应 plot_y, fit_x 对应值
                if src_x <= 0:
                    continue
                src_y = self._remap_to_ipm_y[
                    int(plot_y[index]), int(np.clip(fit_x[index], 0, ipm_image_width - 1))]
                # src_y 为 remap ipm x 矩阵 index 对应 plot_y, fit_x 对应值
                src_y = src_y if src_y > 0 else 0

                lane_pts.append([src_x, src_y])

            src_lane_pts.append(lane_pts)

        # tusimple test data sample point along y axis every 10 pixels
        source_image_width = source_image.shape[1]
        for index, single_lane_pts in enumerate(src_lane_pts):
            single_lane_pt_x = np.array(single_lane_pts, dtype=np.float32)[:, 0]
            single_lane_pt_y = np.array(single_lane_pts, dtype=np.float32)[:, 1]
            if data_source == 'tusimple':
                start_plot_y = 240
                end_plot_y = 720
            elif data_source == 'beec_ccd':
                start_plot_y = 820
                end_plot_y = 1350
            else:
                raise ValueError('Wrong data source now only support tusimple and beec_ccd')
            step = int(math.floor((end_plot_y - start_plot_y) / 10))
            # math.floor 下舍去整
            for plot_y in np.linspace(start_plot_y, end_plot_y, step):
                diff = single_lane_pt_y - plot_y  # （推断）预测车道线点与 plot_y 纵向距离
                fake_diff_bigger_than_zero = diff.copy()
                fake_diff_smaller_than_zero = diff.copy()
                fake_diff_bigger_than_zero[np.where(diff <= 0)] = float('inf')  # 正无穷
                fake_diff_smaller_than_zero[np.where(diff > 0)] = float('-inf')  # 负无穷
                idx_low = np.argmax(fake_diff_smaller_than_zero)  # smaller than zero 中最大值
                idx_high = np.argmin(fake_diff_bigger_than_zero)  # bigger than zero 中最小值

                previous_src_pt_x = single_lane_pt_x[idx_low]
                previous_src_pt_y = single_lane_pt_y[idx_low]
                last_src_pt_x = single_lane_pt_x[idx_high]
                last_src_pt_y = single_lane_pt_y[idx_high]
                # 找到与 plot_y 最靠近的两个点

                if previous_src_pt_y < start_plot_y or last_src_pt_y < start_plot_y or \
                        fake_diff_smaller_than_zero[idx_low] == float('-inf') or \
                        fake_diff_bigger_than_zero[idx_high] == float('inf'):
                    continue
                    # 不符合要求的情况（plot_y 在点集范围边界）

                interpolation_src_pt_x = (abs(previous_src_pt_y - plot_y) * previous_src_pt_x +
                                          abs(last_src_pt_y - plot_y) * last_src_pt_x) / \
                                         (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))  # \ 在末尾时 续行符
                interpolation_src_pt_y = (abs(previous_src_pt_y - plot_y) * previous_src_pt_y +
                                          abs(last_src_pt_y - plot_y) * last_src_pt_y) / \
                                         (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))
                # 确定遮罩层车道线像素点坐标
                if interpolation_src_pt_x > source_image_width or interpolation_src_pt_x < 10:
                    continue

                lane_color = self._color_map[index].tolist()
                cv2.circle(source_image, (int(interpolation_src_pt_x),
                                          int(interpolation_src_pt_y)), 5, lane_color, -1)
                """
                circle(img, center, radius, color, thickness=None, lineType=None, shift=None)
                    img：在img上绘图;
                    center：圆心；例如：(0,0)
                    radius：半径；例如：20
                    color：线的颜色；例如：(0,255,0)(绿色)
                    thickness：线的粗细程度，例如：-1,1,2,3…
                """
        ret = {
            'mask_image': mask_image,
            'fit_params': fit_params,
            'source_image': source_image,
        }

        return ret

    def postprocess_for_test(self, binary_seg_result, instance_seg_result=None,
                             min_area_threshold=100, source_image=None,
                             data_source='tusimple'):

        """

        :param binary_seg_result:
        :param instance_seg_result:
        :param min_area_threshold: 连通域分析阈值
        :param source_image:
        :param data_source:
        :return:
        """


        t_start = time.time()

        # convert binary_seg_result
        #
        binary_seg_result = np.array(binary_seg_result * 255, dtype=np.uint8)

        # background_img = np.zeros(shape=(720, 1280), dtype=np.uint8)

        # 首先进行图像形态学运算 闭运算 先膨胀后腐蚀
        # apply image morphology operation to fill in the hold and reduce the small area
        morphological_ret = _morphological_process(binary_seg_result, kernel_size=5)

        # 进行连通域分析
        connect_components_analysis_ret = _connect_components_analysis(image=morphological_ret)

        # 排序连通域并删除过小的连通域
        labels = connect_components_analysis_ret[1]
        stats = connect_components_analysis_ret[2]
        """
        Get the results
        The first cell is the number of labels 
        标签是输入图像大小的矩阵,其中每个元素的值等于其标签.
        统计数据是函数计算的统计数据的矩阵.它的长度等于标签数量,宽度等于统计数量.它可以与OpenCV文档一起使用:
            Statistics output for each label, including the background label, see below for available statistics. 
            Statistics are accessed via stats[label, COLUMN] where available columns are defined below.
                cv2.CC_STAT_LEFT The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction.
                cv2.CC_STAT_TOP The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction.
                cv2.CC_STAT_WIDTH The horizontal size of the bounding box
                cv2.CC_STAT_HEIGHT The vertical size of the bounding box
                cv2.CC_STAT_AREA The total area (in pixels) of the connected component
        """

        for index, stat in enumerate(stats):
            if stat[4] <= min_area_threshold:
                idx = np.where(labels == index)
                morphological_ret[idx] = 0

        t_pre = time.time() - t_start
        self.pre.append(t_pre)

        # apply embedding features cluster
        mask_image, lane_coords = self._cluster.apply_lane_feats_cluster(
            binary_seg_result=morphological_ret,
            instance_seg_result=instance_seg_result
        )

        if mask_image is None:
            return {
                'mask_image': None,
                'fit_params': None,
                'source_image': None,
            }

        t_cluster = time.time() - t_start - t_pre
        self.cluster.append(t_cluster)

        """
        lane line fit
        """

        fit_params = []
        src_lane_pts = []  # lane pts every single lane
        for lane_index, coords in enumerate(lane_coords):
            if data_source == 'tusimple':
                tmp_mask = np.zeros(shape=(720, 1280), dtype=np.uint8)
                tmp_mask[tuple((np.int_(coords[:, 1] * 720 / 256), np.int_(coords[:, 0] * 1280 / 512)))] = 255
                # 新建空白 tmp_mask 并根据 mask_image 已有坐标进行变换后改变对应坐标的值
            elif data_source == 'beec_ccd':
                tmp_mask = np.zeros(shape=(1350, 2448), dtype=np.uint8)
                tmp_mask[tuple((np.int_(coords[:, 1] * 1350 / 256), np.int_(coords[:, 0] * 2448 / 512)))] = 255
            else:
                raise ValueError('Wrong data source now only support tusimple and beec_ccd')
            tmp_ipm_mask = cv2.remap(
                tmp_mask,
                self._remap_to_ipm_x,
                self._remap_to_ipm_y,
                interpolation=cv2.INTER_NEAREST
            )
            # 使用预设 ipm 进行视角转换 INTER_NEAREST——最近邻插值
            nonzero_y = np.array(tmp_ipm_mask.nonzero()[0])
            nonzero_x = np.array(tmp_ipm_mask.nonzero()[1])

            fit_param = np.polyfit(nonzero_y, nonzero_x, 3)
            # 进行拟合，目标函数为二次函数
            fit_params.append(fit_param)

            [ipm_image_height, ipm_image_width] = tmp_ipm_mask.shape
            plot_y = np.linspace(10, ipm_image_height, ipm_image_height - 10)
            # linspace(start, stop, num) 生成从 start 到 stop num 个数的等差数列
            # fit_x = fit_param[0] * plot_y ** 2 + fit_param[1] * plot_y + fit_param[2]
            fit_x = fit_param[0] * plot_y ** 3 + fit_param[1] * plot_y ** 2 + fit_param[2] * plot_y + fit_param[3]


            lane_pts = []
            # lane points 车道线点坐标
            for index in range(0, plot_y.shape[0], 5):
                src_x = self._remap_to_ipm_x[
                    int(plot_y[index]), int(np.clip(fit_x[index], 0, ipm_image_width - 1))]
                # np.clip(a, a_min, a_max, out=None) 限制 a 在 a_min 和 a_max 之间，超出部分则设置为 a_min 和 a_max
                # src_x 为 remap ipm x 矩阵 index 对应 plot_y, fit_x 对应值
                if src_x <= 0:
                    continue
                src_y = self._remap_to_ipm_y[
                    int(plot_y[index]), int(np.clip(fit_x[index], 0, ipm_image_width - 1))]
                # src_y 为 remap ipm x 矩阵 index 对应 plot_y, fit_x 对应值
                src_y = src_y if src_y > 0 else 0

                lane_pts.append([src_x, src_y])

            src_lane_pts.append(lane_pts)

        t_fit = time.time() - t_start - t_pre - t_cluster
        self.fit.append(t_fit)

        # tusimple test data sample point along y axis every 10 pixels
        source_image_width = source_image.shape[1]
        lane = []
        for index, single_lane_pts in enumerate(src_lane_pts):
            background_img = np.zeros(shape=(720, 1280), dtype=np.uint8)

            single_lane_pt_x = np.array(single_lane_pts, dtype=np.float32)[:, 0]
            single_lane_pt_y = np.array(single_lane_pts, dtype=np.float32)[:, 1]
            if data_source == 'tusimple':
                start_plot_y = 240
                end_plot_y = 720
            elif data_source == 'beec_ccd':
                start_plot_y = 820
                end_plot_y = 1350
            else:
                raise ValueError('Wrong data source now only support tusimple and beec_ccd')
            step = int(math.floor((end_plot_y - start_plot_y) / 10))
            # math.floor 下舍去整

            src_pt_x = []
            src_pt_y = []

            for plot_y in np.linspace(start_plot_y, end_plot_y, step):
                diff = single_lane_pt_y - plot_y  # （推断）预测车道线点与 plot_y 纵向距离
                fake_diff_bigger_than_zero = diff.copy()
                fake_diff_smaller_than_zero = diff.copy()
                fake_diff_bigger_than_zero[np.where(diff <= 0)] = float('inf')  # 正无穷
                fake_diff_smaller_than_zero[np.where(diff > 0)] = float('-inf')  # 负无穷
                idx_low = np.argmax(fake_diff_smaller_than_zero)  # smaller than zero 中最大值
                idx_high = np.argmin(fake_diff_bigger_than_zero)  # bigger than zero 中最小值

                previous_src_pt_x = single_lane_pt_x[idx_low]
                previous_src_pt_y = single_lane_pt_y[idx_low]
                last_src_pt_x = single_lane_pt_x[idx_high]
                last_src_pt_y = single_lane_pt_y[idx_high]
                # 找到与 plot_y 最靠近的两个点

                if previous_src_pt_y < start_plot_y or last_src_pt_y < start_plot_y or \
                        fake_diff_smaller_than_zero[idx_low] == float('-inf') or \
                        fake_diff_bigger_than_zero[idx_high] == float('inf'):
                    continue
                    # 不符合要求的情况（plot_y 在点集范围边界）

                interpolation_src_pt_x = (abs(previous_src_pt_y - plot_y) * previous_src_pt_x +
                                          abs(last_src_pt_y - plot_y) * last_src_pt_x) / \
                                         (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))  # \ 在末尾时 续行符
                interpolation_src_pt_y = (abs(previous_src_pt_y - plot_y) * previous_src_pt_y +
                                          abs(last_src_pt_y - plot_y) * last_src_pt_y) / \
                                         (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))
                # 确定遮罩层车道线像素点坐标
                if interpolation_src_pt_x > source_image_width or interpolation_src_pt_x < 10:
                    continue
                src_pt_x.append(interpolation_src_pt_x)
                src_pt_y.append(interpolation_src_pt_y)

            pred_lane_pts = np.vstack((src_pt_x, src_pt_y)).transpose()
            pred_lane_pts = np.array([pred_lane_pts], np.int64)

            lane_color = self._color_map[index].tolist()

            cv2.polylines(source_image, pred_lane_pts, isClosed=False, color=lane_color, thickness=5)
            cv2.polylines(background_img, pred_lane_pts, isClosed=False, color=255, thickness=2)

            lane_pred = []
            for plot_y_single_lane in np.arange(160, 720, 10):
                if np.count_nonzero(background_img[plot_y_single_lane]) == 0:
                    pred_dot_x = -2
                    lane_pred.append(pred_dot_x)
                    continue
                idx = np.where(np.equal(background_img[plot_y_single_lane], 255))
                pred_dot_x = (idx[0][0] + idx[0][-1]) / 2
                lane_pred.append(int(round(pred_dot_x)))
            lane.append(lane_pred)

        ret = {
            'mask_image': mask_image,
            'fit_params': fit_params,
            'source_image': source_image,
            'lane_pts': lane
        }

        t_full = time.time() - t_start
        self.full.append(t_full)

        return ret


    def compute_mean_time(self):
        print("pre: ", np.mean(self.pre),
              "cluster: ", np.mean(self.cluster),
              "fit: ", np.mean(self.fit),
              "full: ", np.mean(self.full))

class LaneNetPostProcessor_for_nontusimple(object):
    """
    lanenet post process for lane generation
    """

    def __init__(self, ipm_remap_file_path='./data/tusimple_ipm_remap.yml'):

        """

        :param ipm_remap_file_path: ipm generate file path
        """

        self._cluster = _LaneNetCluster()

        self._color_map = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]

        self.pre = []
        self.cluster = []
        self.fit = []
        self.full = []

    def postprocess_for_non_tusimple(self, binary_seg_result, instance_seg_result=None,
                                     min_area_threshold=100, source_image=None,
                                     data_source='tusimple'):

        """

        :param binary_seg_result:
        :param instance_seg_result:
        :param min_area_threshold: 连通域分析阈值
        :param source_image:
        :param data_source:
        :return:
        """

        t_start = time.time()

        # t_start = time.time()

        # convert binary_seg_result
        #
        binary_seg_result = np.array(binary_seg_result * 255, dtype=np.uint8)

        # 首先进行图像形态学运算 闭运算 先膨胀后腐蚀
        # apply image morphology operation to fill in the hold and reduce the small area
        morphological_ret = _morphological_process(binary_seg_result, kernel_size=5)

        # 进行连通域分析
        connect_components_analysis_ret = _connect_components_analysis(image=morphological_ret)

        # 排序连通域并删除过小的连通域
        labels = connect_components_analysis_ret[1]
        stats = connect_components_analysis_ret[2]
        """
        Get the results
        The first cell is the number of labels 
        标签是输入图像大小的矩阵,其中每个元素的值等于其标签.
        统计数据是函数计算的统计数据的矩阵.它的长度等于标签数量,宽度等于统计数量.它可以与OpenCV文档一起使用:
            Statistics output for each label, including the background label, see below for available statistics. 
            Statistics are accessed via stats[label, COLUMN] where available columns are defined below.
                cv2.CC_STAT_LEFT The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction.
                cv2.CC_STAT_TOP The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction.
                cv2.CC_STAT_WIDTH The horizontal size of the bounding box
                cv2.CC_STAT_HEIGHT The vertical size of the bounding box
                cv2.CC_STAT_AREA The total area (in pixels) of the connected component
        """

        for index, stat in enumerate(stats):
            if stat[4] <= min_area_threshold:
                idx = np.where(labels == index)
                morphological_ret[idx] = 0

        # t_pre = time.time() - t_start

        # if data_source == "jiqing":
          #   morphological_ret = morphological_ret[:215]
            # instance_seg_result= instance_seg_result[:215]
            # morphological_ret = cv2.resize(morphological_ret, (1024, 256), interpolation=cv2.INTER_LINEAR)
            # instance_seg_result = cv2.resize(instance_seg_result, (1024, 256), interpolation=cv2.INTER_LINEAR)
        # if data_source == 'caltech':
          #   morphological_ret = morphological_ret[:][:182]
            # instance_seg_result = instance_seg_result[:][:182]

        # t_resize = time.time() - t_start

        t_pre = time.time() - t_start
        self.pre.append(t_pre)

        # apply embedding features cluster
        mask_image, lane_coords = self._cluster.apply_lane_feats_cluster(
            binary_seg_result=morphological_ret,
            instance_seg_result=instance_seg_result
        )

        # t_clu = time.time() - t_start

        if mask_image is None:
            return {
                'mask_image': None,
                'fit_params': None,
                'source_image': None,
            }

        t_cluster = time.time() - t_start - t_pre
        self.cluster.append(t_cluster)

        # t_dbscan = time.time() - t_start

        """
        lane line fit
        """
        fit_params = []
        src_lane_pts = []  # lane pts every single lane
        for lane_index, coords in enumerate(lane_coords):
            if data_source == 'tusimple':
                tmp_mask = np.zeros(shape=(720, 1280), dtype=np.uint8)
                tmp_mask[tuple((np.int_(coords[:, 1] * 720 / 256), np.int_(coords[:, 0] * 1280 / 512)))] = 255
                # 新建空白 tmp_mask 并根据 mask_image 已有坐标进行变换后改变对应坐标的值
            elif data_source == 'jiqing':
                tmp_mask = np.zeros(shape=(1080, 1920), dtype=np.uint8)
                tmp_mask[tuple((np.int_(coords[:, 1] * 300 / 256 + 540), np.int_(coords[:, 0] * 1920 / 512)))] = 255
            elif data_source == 'caltech':
                tmp_mask = np.zeros(shape=(480, 640), dtype=np.uint8)
                # tmp_mask[tuple((np.int_(coords[:, 1] * 480 / 256), np.int_(coords[:, 0] * 640 / 512)))] = 255
                tmp_mask[tuple((np.int_(coords[:, 1] * 420 / 256 +10), np.int_(coords[:, 0] * 560 / 512 + 40)))] = 255
            elif data_source == 'vpgnet':
                tmp_mask = np.zeros(shape=(480, 640), dtype=np.uint8)
                tmp_mask[tuple((np.int_(coords[:, 1] * 480 / 256), np.int_(coords[:, 0] * 640 / 512)))] = 255
            elif data_source == 'culane':
                tmp_mask = np.zeros(shape=(590, 1640), dtype=np.uint8)
                tmp_mask[tuple((np.int_(coords[:, 1] * 430 / 256), np.int_(coords[:, 0] * 1640 / 512)))] = 255
                """
                plt.figure("tmp")
                plt.imshow(tmp_mask)
                plt.show()
                """
            else:
                raise ValueError('Wrong data source now only support tusimple and beec_ccd')

            nonzero_y = np.array(tmp_mask.nonzero()[0])
            nonzero_x = np.array(tmp_mask.nonzero()[1])

            fit_param = np.polyfit(nonzero_y, nonzero_x, 3) #3
            fit_params.append(fit_param)

            [ipm_image_height, ipm_image_width] = tmp_mask.shape
            # plot_y = np.linspace(10, 420, 410)# tmp_mask.nonzero()[0][-1], tmp_mask.nonzero()[0][-1] - 10)
            # plot_y = np.linspace(10, ipm_image_height, ipm_image_height - 10)
            # plot_y = np.linspace(10, tmp_mask.nonzero()[0][-1], tmp_mask.nonzero()[0][-1] - 10)
            plot_y = np.linspace(10, 590, 580)

            # linspace(start, stop, num) 生成从 start 到 stop num 个数的等差数列
            fit_x = fit_param[0] * plot_y ** 3 + fit_param[1] * plot_y ** 2 + fit_param[2] * plot_y + fit_param[3]
            # fit_x = fit_param[0] * plot_y ** 2 + fit_param[1] * plot_y + fit_param[2]

            lane_pts = []
            # lane points 车道线点坐标
            for index in range(0, plot_y.shape[0], 5):
                if fit_x[index] <= 0 or fit_x[index] >= ipm_image_width:
                    continue
                src_x = int(fit_x[index])
                src_y = int(plot_y[index])

                lane_pts.append([src_x, src_y])

            src_lane_pts.append(lane_pts)

        t_fit = time.time() - t_start - t_pre - t_cluster
        self.fit.append(t_fit)

        # tusimple test data sample point along y axis every 10 pixels
        source_image_width = source_image.shape[1]
        lane = []
        """
        if data_source == 'tusimple':
            background_img = np.zeros(shape=(720, 1280), dtype=np.uint8)
        elif data_source == 'jiqing':
            background_img = np.zeros(shape=(1080, 1920), dtype=np.uint8)
        elif data_source == 'caltech':
            background_img = np.zeros(shape=(480, 640), dtype=np.uint8)
        elif data_source == 'vpgnet':
            background_img = np.zeros(shape=(480, 640), dtype=np.uint8)
        """
        for index, single_lane_pts in enumerate(src_lane_pts):

            if data_source == 'tusimple':
                background_img = np.zeros(shape=(720, 1280), dtype=np.uint8)
            elif data_source == 'jiqing':
                background_img = np.zeros(shape=(1080, 1920), dtype=np.uint8)
            elif data_source == 'caltech':
                background_img = np.zeros(shape=(480, 640), dtype=np.uint8)
            elif data_source == 'culane':
                background_img = np.zeros(shape=(590, 1640), dtype=np.uint8)
            else:
                raise ValueError('Wrong data source now only support tusimple and beec_ccd')

            single_lane_pt_x = np.array(single_lane_pts, dtype=np.float32)[:, 0]
            single_lane_pt_y = np.array(single_lane_pts, dtype=np.float32)[:, 1]
            if data_source == 'tusimple':
                start_plot_y = 240
                end_plot_y = 720
            elif data_source == 'jiqing':
                start_plot_y = 600
                end_plot_y = 850
            elif data_source == 'caltech':
                start_plot_y = 190
                end_plot_y = 480
            elif data_source == 'vpgnet':
                start_plot_y = 210
                end_plot_y = 480
            elif data_source == 'culane':
                start_plot_y = 290
                end_plot_y = 570
            else:
                raise ValueError('Wrong data source now only support tusimple and beec_ccd')
            step = int(math.floor((end_plot_y - start_plot_y) / 10))
            # math.floor 下舍去整

            src_pt_x = []
            src_pt_y = []

            for plot_y in np.linspace(start_plot_y, end_plot_y, step):
                diff = single_lane_pt_y - plot_y  # （推断）预测车道线点与 plot_y 纵向距离
                fake_diff_bigger_than_zero = diff.copy()
                fake_diff_smaller_than_zero = diff.copy()
                fake_diff_bigger_than_zero[np.where(diff <= 0)] = float('inf')  # 正无穷
                fake_diff_smaller_than_zero[np.where(diff > 0)] = float('-inf')  # 负无穷
                idx_low = np.argmax(fake_diff_smaller_than_zero)  # smaller than zero 中最大值
                idx_high = np.argmin(fake_diff_bigger_than_zero)  # bigger than zero 中最小值

                previous_src_pt_x = single_lane_pt_x[idx_low]
                previous_src_pt_y = single_lane_pt_y[idx_low]
                last_src_pt_x = single_lane_pt_x[idx_high]
                last_src_pt_y = single_lane_pt_y[idx_high]
                # 找到与 plot_y 最靠近的两个点

                if previous_src_pt_y < start_plot_y or last_src_pt_y < start_plot_y or \
                        fake_diff_smaller_than_zero[idx_low] == float('-inf') or \
                        fake_diff_bigger_than_zero[idx_high] == float('inf'):
                    continue
                    # 不符合要求的情况（plot_y 在点集范围边界外）

                interpolation_src_pt_x = (abs(previous_src_pt_y - plot_y) * previous_src_pt_x +
                                          abs(last_src_pt_y - plot_y) * last_src_pt_x) / \
                                         (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))  # \ 在末尾时 续行符
                interpolation_src_pt_y = (abs(previous_src_pt_y - plot_y) * previous_src_pt_y +
                                          abs(last_src_pt_y - plot_y) * last_src_pt_y) / \
                                         (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))
                # 确定遮罩层车道线像素点坐标
                if interpolation_src_pt_x > source_image_width or interpolation_src_pt_x < 10:
                    continue
                src_pt_x.append(interpolation_src_pt_x)
                src_pt_y.append(interpolation_src_pt_y)

            pred_lane_pts = np.vstack((src_pt_x, src_pt_y)).transpose()
            pred_lane_pts = np.array([pred_lane_pts], np.int64)

            lane_color = self._color_map[index].tolist()

            cv2.polylines(source_image, pred_lane_pts, isClosed=False, color=(255,255,255), thickness=5)
            cv2.polylines(background_img, pred_lane_pts, isClosed=False, color=255, thickness=5) # thickness=2

            if data_source == 'tusimple':
                start_plot_y = 160
                end_plot_y = 720
            elif data_source == 'jiqing':
                start_plot_y = 610
                end_plot_y = 850
            elif data_source == 'caltech':
                start_plot_y = 190
                end_plot_y = 480
            elif data_source == 'vpgnet':
                start_plot_y = 210
                end_plot_y = 480
            elif data_source == 'culane':
                start_plot_y = 290
                end_plot_y = 570
            else:
                raise ValueError('Wrong data source now only support tusimple and beec_ccd')

            lane_pred = []
            for plot_y_single_lane in np.arange(start_plot_y, end_plot_y, 10):
                if np.count_nonzero(background_img[plot_y_single_lane]) == 0:
                    pred_dot_x = -2
                    lane_pred.append(pred_dot_x)
                    continue
                idx = np.where(np.equal(background_img[plot_y_single_lane], 255))
                pred_dot_x = (idx[0][0] + idx[0][-1]) / 2
                lane_pred.append(int(round(pred_dot_x)))
            lane.append(lane_pred)

        ret = {
            'mask_image': mask_image,
            'fit_params': fit_params,
            'source_image': source_image,
            'lane_pts': lane# ,
            # 'pred_result':background_img
        }

        t_full = time.time() - t_start
        self.full.append(t_full)
        # t_ploy = time.time() - t_start

        # print("post ",t_resize, t_clu)
        return ret


    def compute_mean_time(self):
        print("pre: ", np.mean(self.pre),
              "cluster: ", np.mean(self.cluster),
              "fit: ", np.mean(self.fit),
              "full: ", np.mean(self.full))
