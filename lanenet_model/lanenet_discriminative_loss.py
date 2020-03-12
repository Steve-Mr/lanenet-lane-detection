#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-11 下午3:48
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_discriminative_loss.py
# @IDE: PyCharm Community Edition
"""
Discriminative Loss for instance segmentation
"""
import tensorflow as tf


def discriminative_loss_single(
        prediction,
        correct_label,
        feature_dim,
        label_shape,
        delta_v,
        delta_d,
        param_var,
        param_dist,
        param_reg):
    print("lanenet discriminative loss single")

    """
    discriminative loss
    :param prediction: inference of network
    :param correct_label: instance label
    :param feature_dim: feature dimension of prediction : 4
    :param label_shape: shape of label
    :param delta_v: cut off variance distance
    :param delta_d: cut off cluster distance
    :param param_var: weight for intra cluster variance
    :param param_dist: weight for inter cluster distances
    :param param_reg: weight regularization
    """
    correct_label = tf.reshape(
        correct_label, [label_shape[1] * label_shape[0]]
    )
    reshaped_pred = tf.reshape(
        prediction, [label_shape[1] * label_shape[0], feature_dim]
    )
    # 将像素对齐为一行

    # calculate instance nums
    unique_labels, unique_id, counts = tf.unique_with_counts(correct_label)
    """
    统计实例个数
        unique_labels：correct_label中一共有几种数值                         
        unique_id：correct_label中的每个数值是属于unique_labels中第几类      --- 
        counts：统计unique_labels中每个数值在correct_label中出现了几次       --- Nc：簇 c 中元素数
    """
    counts = tf.cast(counts, tf.float32)    # 每个实例占用的像素点量
    num_instances = tf.size(unique_labels)  # 实例数量 --- C:真实值中簇总数

    # 计算pixel embedding均值向量
    # segmented_sum是把reshaped_pred中对于GT里每个部分位置上的像素点相加
    # 比如unique_id[0, 0, 1, 1, 0],reshaped_pred[1, 2, 3, 4, 5]，最后等于[1+2+5,3+4],channel层不相加
    segmented_sum = tf.unsorted_segment_sum(
        reshaped_pred, unique_id, num_instances)
    # 除以每个类别的像素在gt中出现的次数，是每个类别像素的均值 (?, feature_dim)
    mu = tf.div(segmented_sum, tf.reshape(counts, (-1, 1)))
    # 然后再还原为原图的形式，现在mu_expand中的数据是与correct_label的分布一致，但是数值不一样
    mu_expand = tf.gather(mu, unique_id)
    # tf. gather: 按照指定的下标集合从axis=0中抽取子集，适合抽取不连续区域的子集

    # 计算公式的loss(var)
    distance = tf.norm(tf.subtract(mu_expand, reshaped_pred), axis=1, ord=1)    # ||MUc - Xi||
    distance = tf.subtract(distance, delta_v)                                   # ||Muc - Xi|| - DELTAv
    distance = tf.clip_by_value(distance, 0., distance)                         # []+
    distance = tf.square(distance)                                              # ^2

    l_var = tf.unsorted_segment_sum(distance, unique_id, num_instances)         # i=1 to i=Nc 相加
    l_var = tf.div(l_var, counts)                                               # 1/Nc
    l_var = tf.reduce_sum(l_var)                                                # 压缩为一维
    l_var = tf.divide(l_var, tf.cast(num_instances, tf.float32))                # 1/C

    # 计算公式的loss(dist)
    mu_interleaved_rep = tf.tile(mu, [num_instances, 1])                 # shape: num_instance*num_instance, feature_dim
    mu_band_rep = tf.tile(mu, [1, num_instances])                        #
    mu_band_rep = tf.reshape(
        mu_band_rep,
        (num_instances *
         num_instances,
         feature_dim))

    mu_diff = tf.subtract(mu_band_rep, mu_interleaved_rep)              # muCa - muCb

    # 去除掩模上的零点 ca != cb
    intermediate_tensor = tf.reduce_sum(tf.abs(mu_diff), axis=1)
    zero_vector = tf.zeros(1, dtype=tf.float32)
    bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
    mu_diff_bool = tf.boolean_mask(mu_diff, bool_mask)

    mu_norm = tf.norm(mu_diff_bool, axis=1, ord=1)                      # ||muCa - muCb||
    mu_norm = tf.subtract(2. * delta_d, mu_norm)                        # delta d - mu_norm
    mu_norm = tf.clip_by_value(mu_norm, 0., mu_norm)                    # []+
    mu_norm = tf.square(mu_norm)                                        # ^2

    l_dist = tf.reduce_mean(mu_norm)                                    # 均值

    # 计算原始Discriminative Loss论文中提到的正则项损失
    l_reg = tf.reduce_mean(tf.norm(mu, axis=1, ord=1))

    # 合并损失按照原始Discriminative Loss论文中提到的参数合并
    param_scale = 1.
    l_var = param_var * l_var
    l_dist = param_dist * l_dist
    l_reg = param_reg * l_reg

    loss = param_scale * (l_var + l_dist + l_reg)

    return loss, l_var, l_dist, l_reg


def discriminative_loss(prediction, correct_label, feature_dim, image_shape,
                        delta_v, delta_d, param_var, param_dist, param_reg):
    print("lanenet discriminateive loss")

    """

    :return: discriminative loss and its three components
    """

    def cond(label, batch, out_loss, out_var, out_dist, out_reg, i):
        return tf.less(i, tf.shape(batch)[0])
        # 以元素方式返回(x <y)的真值.

    def body(label, batch, out_loss, out_var, out_dist, out_reg, i):
        disc_loss, l_var, l_dist, l_reg = discriminative_loss_single(
            prediction[i], correct_label[i], feature_dim, image_shape, delta_v, delta_d, param_var, param_dist, param_reg)

        out_loss = out_loss.write(i, disc_loss)
        out_var = out_var.write(i, l_var)
        out_dist = out_dist.write(i, l_dist)
        out_reg = out_reg.write(i, l_reg)

        return label, batch, out_loss, out_var, out_dist, out_reg, i + 1

    # TensorArray is a data structure that support dynamic writing
    output_ta_loss = tf.TensorArray(
        dtype=tf.float32, size=0, dynamic_size=True)
    output_ta_var = tf.TensorArray(
        dtype=tf.float32, size=0, dynamic_size=True)
    output_ta_dist = tf.TensorArray(
        dtype=tf.float32, size=0, dynamic_size=True)
    output_ta_reg = tf.TensorArray(
        dtype=tf.float32, size=0, dynamic_size=True)

    _, _, out_loss_op, out_var_op, out_dist_op, out_reg_op, _ = tf.while_loop(
        cond, body, [
            correct_label, prediction, output_ta_loss, output_ta_var, output_ta_dist, output_ta_reg, 0])
    # cond 成立时执行 body
    out_loss_op = out_loss_op.stack()
    out_var_op = out_var_op.stack()
    out_dist_op = out_dist_op.stack()
    out_reg_op = out_reg_op.stack()

    disc_loss = tf.reduce_mean(out_loss_op)
    l_var = tf.reduce_mean(out_var_op)
    l_dist = tf.reduce_mean(out_dist_op)
    l_reg = tf.reduce_mean(out_reg_op)
    # 求平均

    return disc_loss, l_var, l_dist, l_reg
