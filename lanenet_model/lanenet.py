#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-4-24 下午8:50
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet.py
# @IDE: PyCharm
"""
Implement LaneNet Model
"""
import tensorflow as tf

from config import global_config
from lanenet_model import lanenet_back_end
from lanenet_model import lanenet_front_end
from semantic_segmentation_zoo import cnn_basenet

CFG = global_config.cfg


class LaneNet(cnn_basenet.CNNBaseModel):
    """

    """
    def __init__(self, phase, net_flag='vgg', reuse=False):
        print("lanenet __init__")

        """
        Python 中的 self 等价于 C++ 中的 self 指针和 Java、C# 中的 this 参数。
        :param: reuse:  设置 reuse=False 时，函数 get_variable() 表示创建变量
                        设置 reuse=True  时，函数 get_variable() 表示获取变量
        """
        super(LaneNet, self).__init__()
        self._net_flag = net_flag
        self._reuse = reuse

        self._frontend = lanenet_front_end.LaneNetFrondEnd(
            phase=phase, net_flag=net_flag
        )

        self._backend = lanenet_back_end.LaneNetBackEnd(
            phase=phase
        )

    def inference(self, input_tensor, name):
        print("lanenet inference")
        """
        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=self._reuse):
            # first extract image features
            extract_feats_result = self._frontend.build_model(
                input_tensor=input_tensor,
                name='{:s}_frontend'.format(self._net_flag),
                reuse=self._reuse
            )

            # second apply backend process
            binary_seg_prediction, instance_seg_prediction = self._backend.inference(
                binary_seg_logits=extract_feats_result['binary_segment_logits']['data'],
                instance_seg_logits=extract_feats_result['instance_segment_logits']['data'],
                name='{:s}_backend'.format(self._net_flag),
                reuse=self._reuse
            )

            if not self._reuse:
                self._reuse = True

        return binary_seg_prediction, instance_seg_prediction

    def compute_loss(self, input_tensor, binary_label, instance_label, name):
        print("lanenet compute_loss")

        """
        calculate lanenet loss for training
        :param input_tensor:
        :param binary_label:
        :param instance_label:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=self._reuse):
            # first extract image features
            extract_feats_result = self._frontend.build_model(
                input_tensor=input_tensor,
                name='{:s}_frontend'.format(self._net_flag),
                reuse=self._reuse
            )

            # second apply backend process
            calculated_losses = self._backend.compute_loss(
                binary_seg_logits=extract_feats_result['binary_segment_logits']['data'],
                binary_label=binary_label,
                instance_seg_logits=extract_feats_result['instance_segment_logits']['data'],
                instance_label=instance_label,
                name='{:s}_backend'.format(self._net_flag),
                reuse=self._reuse
            )

            if not self._reuse:
                self._reuse = True

        return calculated_losses
