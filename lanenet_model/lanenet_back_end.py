#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-4-24 下午3:54
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_back_end.py
# @IDE: PyCharm
"""
LaneNet backend branch which is mainly used for binary and instance segmentation loss calculation
"""
import tensorflow as tf

from config import global_config
from lanenet_model import lanenet_discriminative_loss
from semantic_segmentation_zoo import cnn_basenet

CFG = global_config.cfg


class LaneNetBackEnd(cnn_basenet.CNNBaseModel):
    """
    LaneNet backend branch which is mainly used for binary and instance segmentation loss calculation
    """
    def __init__(self, phase):
        print("backend __init__")

        """
        init lanenet backend
        :param phase: train or test
        """
        super(LaneNetBackEnd, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()

    def _is_net_for_training(self):
        print("backend is for training")

        """
        if the net is used for training or not
        :return:
        """
        if isinstance(self._phase, tf.Tensor):
            phase = self._phase
        else:
            phase = tf.constant(self._phase, dtype=tf.string)

        return tf.equal(phase, tf.constant('train', dtype=tf.string))

    @classmethod
    def _compute_class_weighted_cross_entropy_loss(cls, onehot_labels, logits, classes_weights):
        print("backend compute class weighted cross entropy loss")

        """
        用来计算神经元之间的权值的交叉熵损失函数
        :param onehot_labels:
        :param logits:
        :param classes_weights:
        :return:
        """
        loss_weights = tf.reduce_sum(tf.multiply(onehot_labels, classes_weights), axis=3)

        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels,
            logits=logits,
            weights=loss_weights
        )
        # 返回与logits具有相同类型的加权损失Tensor

        return loss

    def compute_loss(self, binary_seg_logits, binary_label,
                     instance_seg_logits, instance_label,
                     name, reuse):
        print("backend compute_loss")

        """
        compute lanenet loss
        :param binary_seg_logits:
        :param binary_label:
        :param instance_seg_logits:
        :param instance_label:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            # calculate class weighted binary seg loss
            with tf.variable_scope(name_or_scope='binary_seg'):
                binary_label_onehot = tf.one_hot(
                    tf.reshape(
                        tf.cast(binary_label, tf.int32),
                        shape=[binary_label.get_shape().as_list()[0],
                               binary_label.get_shape().as_list()[1],
                               binary_label.get_shape().as_list()[2]]),
                    depth=CFG.TRAIN.CLASSES_NUMS,   # 2
                    axis=-1
                )
                """
                将input转化为one-hot类型数据输出，相当于将多个数值联合放在一起作为多个相同类型的向量，可用于表示各自的概率分布
                函数规定输入的元素indices从0开始，最大的元素值不能超过（depth - 1），因此能够表示depth个单位的输入。
                若输入的元素值超出范围，输出的编码均为 [0, 0 … 0, 0]。
                """

                binary_label_plain = tf.reshape(
                    binary_label,
                    shape=[binary_label.get_shape().as_list()[0] *
                           binary_label.get_shape().as_list()[1] *
                           binary_label.get_shape().as_list()[2] *
                           binary_label.get_shape().as_list()[3]])
                unique_labels, unique_id, counts = tf.unique_with_counts(binary_label_plain)
                """
                一个张量 y,该张量包含出现在 x 中的以相同顺序排序的 x 的所有的唯一元素.
                一个与 x 具有相同大小的张量 idx,包含唯一的输出 y 中 x 的每个值的索引.
                一个张量 count,其中包含 x 中 y 的每个元素的计数
                """
                counts = tf.cast(counts, tf.float32)
                inverse_weights = tf.divide(
                    1.0,
                    tf.log(tf.add(tf.divide(counts, tf.reduce_sum(counts)), tf.constant(1.02)))
                )   # bounded inverse class weight 1/log(counts/all_counts + 1.02)

                binary_segmenatation_loss = self._compute_class_weighted_cross_entropy_loss(
                    onehot_labels=binary_label_onehot,
                    logits=binary_seg_logits,
                    classes_weights=inverse_weights
                )

            # calculate class weighted instance seg loss
            with tf.variable_scope(name_or_scope='instance_seg'):

                pix_bn = self.layerbn(
                    inputdata=instance_seg_logits, is_training=self._is_training, name='pix_bn')
                pix_relu = self.relu(inputdata=pix_bn, name='pix_relu')
                pix_embedding = self.conv2d(
                    inputdata=pix_relu,
                    out_channel=CFG.TRAIN.EMBEDDING_FEATS_DIMS,
                    kernel_size=1,
                    use_bias=False,
                    name='pix_embedding_conv'
                )
                pix_image_shape = (pix_embedding.get_shape().as_list()[1], pix_embedding.get_shape().as_list()[2])
                instance_segmentation_loss, l_var, l_dist, l_reg = \
                    lanenet_discriminative_loss.discriminative_loss(
                        pix_embedding, instance_label, CFG.TRAIN.EMBEDDING_FEATS_DIMS,
                        pix_image_shape, 0.5, 3.0, 1.0, 1.0, 0.001
                    )

            l2_reg_loss = tf.constant(0.0, tf.float32)
            for vv in tf.trainable_variables():
                if 'bn' in vv.name or 'gn' in vv.name:
                    continue
                else:
                    l2_reg_loss = tf.add(l2_reg_loss, tf.nn.l2_loss(vv))
            l2_reg_loss *= 0.001
            total_loss = binary_segmenatation_loss + instance_segmentation_loss + l2_reg_loss

            ret = {
                'total_loss': total_loss,
                'binary_seg_logits': binary_seg_logits,
                'instance_seg_logits': pix_embedding,
                'binary_seg_loss': binary_segmenatation_loss,
                'discriminative_loss': instance_segmentation_loss
            }

        return ret

    def inference(self, binary_seg_logits, instance_seg_logits, name, reuse):
        print("backend inference")

        """

        :param binary_seg_logits:
        :param instance_seg_logits:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):

            with tf.variable_scope(name_or_scope='binary_seg'):
                binary_seg_score = tf.nn.softmax(logits=binary_seg_logits)
                """
                归一化
                将一个含任意实数的K维向量“压缩”到另一个K维实向量中，使得每一个元素的范围都在之间，并且所有元素的和为1
                Returns:
                A `Tensor`. Has the same type and shape as `logits`.
                """

                binary_seg_prediction = tf.argmax(binary_seg_score, axis=-1)
                # 返回最大值索引号 axis：选取的坐标值

            with tf.variable_scope(name_or_scope='instance_seg'):

                pix_bn = self.layerbn(
                    inputdata=instance_seg_logits, is_training=self._is_training, name='pix_bn')
                pix_relu = self.relu(inputdata=pix_bn, name='pix_relu')
                instance_seg_prediction = self.conv2d(
                    inputdata=pix_relu,
                    out_channel=CFG.TRAIN.EMBEDDING_FEATS_DIMS,
                    kernel_size=1,
                    use_bias=False,
                    name='pix_embedding_conv'
                )

        return binary_seg_prediction, instance_seg_prediction
