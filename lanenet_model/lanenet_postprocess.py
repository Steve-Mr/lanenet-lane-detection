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

import cv2
import glog as log
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from config import global_config

CFG = global_config.cfg


def _morphological_process(image, kernel_size=5):
    print("lanenet postprocess morphological")

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

    return closing


def _connect_components_analysis(image):
    print("lanenet postprocess connect components analysis")

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
        print("lanenet postprocess lanefeat __init__")

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
        print("lanenet postprocess lanefeat feat1")

        """

        :return:
        """
        return self._feat

    @feat.setter
    def feat(self, value):
        print("lanenet postprocess lanefeat feat2")

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
        print("lanenet postprocess lanefeat coord1")

        """

        :return:
        """
        return self._coord

    @coord.setter
    def coord(self, value):
        print("lanenet postprocess lanefeat coord2")

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
        print("lanenet postprocess lanefeat classid1")

        """

        :return:
        """
        return self._class_id

    @class_id.setter
    def class_id(self, value):
        print("lanenet postprocess lanefeat classid2")

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
        print("lanenet postprocess cluster init")

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
    def _embedding_feats_dbscan_cluster(embedding_image_feats):
        print("lanenet postprocess cluster dbscan")

        """
        dbscan cluster
        :param embedding_image_feats:
        :return:
        """
        db = DBSCAN(eps=CFG.POSTPROCESS.DBSCAN_EPS, min_samples=CFG.POSTPROCESS.DBSCAN_MIN_SAMPLES)
        """
        eps: 扫描半径 (0.35)
        min_samples：作为核心点其邻域中的最小样本数（包括点本身）(1000)
        """
        try:
            features = StandardScaler().fit_transform(embedding_image_feats)
            print("embedding_image_feats")
            print(embedding_image_feats)
            print("features")
            print(features)
            # fit_transform 不仅计算训练数据的均值和方差，
            # 还会基于计算出来的均值和方差来转换训练数据，从而把数据转换成标准的正太分布
            db.fit(features)
            """
            fit(self, X, y=None, sample_weight=None):
            Perform DBSCAN clustering from features or distance matrix.

            Parameters
            ----------
            X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
                    array of shape (n_samples, n_samples)
                A feature array, or array of distances between samples if
                ``metric='precomputed'``.
            sample_weight : array, shape (n_samples,), optional
                Weight of each sample, such that a sample with a weight of at least
                ``min_samples`` is by itself a core sample; a sample with negative
                weight may inhibit its eps-neighbor from being core.
                Note that weights are absolute, and default to 1.

            y : Ignored

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
        print("lanenet postprocess cluster get lane embedding feats")

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

        assert lane_embedding_feats.shape[0] == lane_coordinate.shape[0]

        ret = {
            'lane_embedding_feats': lane_embedding_feats,
            'lane_coordinates': lane_coordinate
        }

        return ret

    def apply_lane_feats_cluster(self, binary_seg_result, instance_seg_result):
        print("lanenet postprocess cluster apply lane feats")

        """

        :param binary_seg_result:
        :param instance_seg_result:
        :return:
        """
        # get embedding feats and coords
        """
        {   'lane_embedding_feats': lane_embedding_feats,
            'lane_coordinates': lane_coordinate
        }
        """
        get_lane_embedding_feats_result = self._get_lane_embedding_feats(
            binary_seg_ret=binary_seg_result,
            instance_seg_ret=instance_seg_result
        )

        # dbscan cluster
        dbscan_cluster_result = self._embedding_feats_dbscan_cluster(
            embedding_image_feats=get_lane_embedding_feats_result['lane_embedding_feats']
        )

        mask = np.zeros(shape=[binary_seg_result.shape[0], binary_seg_result.shape[1], 3], dtype=np.uint8)
        db_labels = dbscan_cluster_result['db_labels']
        unique_labels = dbscan_cluster_result['unique_labels']
        coord = get_lane_embedding_feats_result['lane_coordinates']

        if db_labels is None:
            return None, None

        lane_coords = []

        for index, label in enumerate(unique_labels.tolist()):
            if label == -1:
                continue
            idx = np.where(db_labels == label)
            pix_coord_idx = tuple((coord[idx][:, 1], coord[idx][:, 0]))
            mask[pix_coord_idx] = self._color_map[index]
            lane_coords.append(coord[idx])

        return mask, lane_coords


class LaneNetPostProcessor(object):
    """
    lanenet post process for lane generation
    """
    def __init__(self, ipm_remap_file_path='./data/tusimple_ipm_remap.yml'):
        print("lanenet postprocess processor init")

        """

        :param ipm_remap_file_path: ipm generate file path
        """
        assert ops.exists(ipm_remap_file_path), '{:s} not exist'.format(ipm_remap_file_path)

        self._cluster = _LaneNetCluster()
        self._ipm_remap_file_path = ipm_remap_file_path
        # 替代 H-NET 作用

        remap_file_load_ret = self._load_remap_matrix()
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

    def _load_remap_matrix(self):
        print("lanenet postprocess processor load remap matrix")

        """

        :return:
        """
        fs = cv2.FileStorage(self._ipm_remap_file_path, cv2.FILE_STORAGE_READ)

        remap_to_ipm_x = fs.getNode('remap_ipm_x').mat()
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
        print("lanenet postprocess processor postprocess")

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

        print(connect_components_analysis_ret)

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
        print("stats")
        print(stats)

        for index, stat in enumerate(stats):
            print("stat4", stat[4])
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
            nonzero_y = np.array(tmp_ipm_mask.nonzero()[0])
            nonzero_x = np.array(tmp_ipm_mask.nonzero()[1])

            fit_param = np.polyfit(nonzero_y, nonzero_x, 2)
            fit_params.append(fit_param)

            [ipm_image_height, ipm_image_width] = tmp_ipm_mask.shape
            plot_y = np.linspace(10, ipm_image_height, ipm_image_height - 10)
            fit_x = fit_param[0] * plot_y ** 2 + fit_param[1] * plot_y + fit_param[2]
            # fit_x = fit_param[0] * plot_y ** 3 + fit_param[1] * plot_y ** 2 + fit_param[2] * plot_y + fit_param[3]

            lane_pts = []
            for index in range(0, plot_y.shape[0], 5):
                src_x = self._remap_to_ipm_x[
                    int(plot_y[index]), int(np.clip(fit_x[index], 0, ipm_image_width - 1))]
                if src_x <= 0:
                    continue
                src_y = self._remap_to_ipm_y[
                    int(plot_y[index]), int(np.clip(fit_x[index], 0, ipm_image_width - 1))]
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
            for plot_y in np.linspace(start_plot_y, end_plot_y, step):
                diff = single_lane_pt_y - plot_y
                fake_diff_bigger_than_zero = diff.copy()
                fake_diff_smaller_than_zero = diff.copy()
                fake_diff_bigger_than_zero[np.where(diff <= 0)] = float('inf')
                fake_diff_smaller_than_zero[np.where(diff > 0)] = float('-inf')
                idx_low = np.argmax(fake_diff_smaller_than_zero)
                idx_high = np.argmin(fake_diff_bigger_than_zero)

                previous_src_pt_x = single_lane_pt_x[idx_low]
                previous_src_pt_y = single_lane_pt_y[idx_low]
                last_src_pt_x = single_lane_pt_x[idx_high]
                last_src_pt_y = single_lane_pt_y[idx_high]

                if previous_src_pt_y < start_plot_y or last_src_pt_y < start_plot_y or \
                        fake_diff_smaller_than_zero[idx_low] == float('-inf') or \
                        fake_diff_bigger_than_zero[idx_high] == float('inf'):
                    continue

                interpolation_src_pt_x = (abs(previous_src_pt_y - plot_y) * previous_src_pt_x +
                                          abs(last_src_pt_y - plot_y) * last_src_pt_x) / \
                                         (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))
                interpolation_src_pt_y = (abs(previous_src_pt_y - plot_y) * previous_src_pt_y +
                                          abs(last_src_pt_y - plot_y) * last_src_pt_y) / \
                                         (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))

                if interpolation_src_pt_x > source_image_width or interpolation_src_pt_x < 10:
                    continue

                lane_color = self._color_map[index].tolist()
                cv2.circle(source_image, (int(interpolation_src_pt_x),
                                          int(interpolation_src_pt_y)), 5, lane_color, -1)
        ret = {
            'mask_image': mask_image,
            'fit_params': fit_params,
            'source_image': source_image,
        }

        return ret
