import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from sklearn.cluster import KMeans
from tensorflow.keras.layers import Lambda


def load_box_prior(dataset, name, img_size, data_num, k_per_grid=3):
    box_prior_dir = (
        f"./data_chkr/{''.join(char for char in name if char.isalnum())}_box_prior.csv"
    )
    if not (os.path.exists(box_prior_dir)):
        box_prior = build_box_prior(dataset, img_size, data_num, k_per_grid)
        box_prior.to_csv(box_prior_dir, index=False, header=False)
    else:
        box_prior = pd.read_csv(box_prior_dir, header=None).to_numpy()
        box_prior = tf.cast(box_prior, dtype=tf.float32)

    return box_prior


def build_box_prior(dataset, img_size, data_num, k_per_grid):
    gt_hws = collect_boxes(dataset, data_num, img_size)
    hw_area = gt_hws[..., 0] * gt_hws[..., 1]
    hw1 = gt_hws[hw_area <= np.quantile(hw_area, 0.333333)]
    hw2 = gt_hws[
        np.logical_and(
            hw_area > np.quantile(hw_area, 0.333333),
            hw_area <= np.quantile(hw_area, 0.666666),
        )
    ]
    hw3 = gt_hws[hw_area > np.quantile(hw_area, 0.666666)]

    box_prior = []
    progress = tqdm(range(3))
    progress.set_description("Clustering boxes")
    for i in progress:
        cluster_sample = (hw1, hw2, hw3)[i]
        box_prior.append(k_means(cluster_sample, k_per_grid))
    box_prior = np.concatenate(box_prior)

    prior_df = pd.DataFrame(box_prior, columns=["height", "width"])
    prior_df.insert(2, "area", box_prior[..., 0] * box_prior[..., 1])
    prior_df = prior_df.sort_values(by=["area"], axis=0)
    prior_df = prior_df[["height", "width"]]

    return prior_df


def k_means(cluster_sample, k_per_grid):
    model = KMeans(n_clusters=k_per_grid, random_state=1)
    model.fit(cluster_sample)
    box_prior = model.cluster_centers_

    return box_prior


def collect_boxes(dataset, data_num, img_size):
    prior_samples = []
    progress = tqdm(range(data_num))
    progress.set_description("Collecting boxes extraced from gt_boxes")
    for _ in progress:
        image, gt_boxes, gt_labels = next(dataset)
        prior_sample = extract_boxes(gt_boxes, img_size)
        prior_samples.append(prior_sample)
    gt_hws = tf.concat(prior_samples, axis=0)

    return gt_hws


def extract_boxes(gt_boxes, img_size):
    y1, x1, y2, x2 = tf.split(gt_boxes, 4, axis=-1)
    gt_heights = (y2 - y1) * tf.constant(img_size[0], dtype=tf.float32)
    gt_widths = (x2 - x1) * tf.constant(img_size[1], dtype=tf.float32)
    prior_sample = tf.concat([gt_heights, gt_widths], axis=-1)
    prior_sample = tf.reshape(
        prior_sample, (tf.shape(prior_sample)[0] * tf.shape(prior_sample)[1], 2)
    )
    not_zero = tf.not_equal(prior_sample, 0)
    not_zero = tf.logical_and(
        Lambda(lambda x: x[..., 0])(not_zero), Lambda(lambda x: x[..., 1])(not_zero)
    )
    prior_sample = Lambda(lambda x: x[not_zero])(prior_sample)

    return prior_sample


def build_anchor_ops(img_size, box_priors):
    anchors_lst = []
    prior_grids_lst = []
    offset_grids_lst = []
    stride_grids_lst = []

    strides = (32, 16, 8)
    for num, stride in enumerate(strides):
        feature_map_shape = img_size[0] // stride
        grid_y_ctr, grid_x_ctr = build_grid(feature_map_shape)
        flat_grid_x_ctr = tf.reshape(grid_x_ctr, (-1,))
        flat_grid_y_ctr = tf.reshape(grid_y_ctr, (-1,))
        box_prior = box_priors[6-3*num:9-3*num] 

        anchor = build_anchor(flat_grid_y_ctr, flat_grid_x_ctr, box_prior / img_size[0])
        prior_grid = tf.tile(box_prior, (feature_map_shape * feature_map_shape, 1))
        offset_grid = build_offset(
            flat_grid_y_ctr,
            flat_grid_x_ctr,
            feature_map_shape
            )
        stride_grid = tf.constant(stride, dtype=tf.float32, shape=tf.shape(offset_grid))

        anchors_lst.append(anchor)
        prior_grids_lst.append(prior_grid)
        offset_grids_lst.append(offset_grid)
        stride_grids_lst.append(stride_grid)

    anchors = tf.concat(anchors_lst, axis=0)
    prior_grids = tf.concat(prior_grids_lst, axis=0)
    offset_grids = tf.concat(offset_grids_lst, axis=0)
    stride_grids = tf.concat(stride_grids_lst, axis=0)

    return anchors, prior_grids, offset_grids, stride_grids


def build_grid(feature_map_shape):
    grid_size = 1 / feature_map_shape
    grid_coords_ctr = tf.cast(
        tf.range(0, feature_map_shape) / feature_map_shape + grid_size / 2,
        dtype=tf.float32,
    )
    grid_y_ctr, grid_x_ctr = tf.meshgrid(grid_coords_ctr, grid_coords_ctr)

    return grid_y_ctr, grid_x_ctr


def build_offset(flat_grid_y_ctr, flat_grid_x_ctr, feature_map_shape):
    offset_y = tf.tile(
        tf.expand_dims(flat_grid_y_ctr * feature_map_shape - 0.5, axis=-1),
        multiples=[1,3]
        )
    offset_x = tf.tile(
        tf.expand_dims(flat_grid_x_ctr * feature_map_shape - 0.5, axis=-1),
        multiples=[1,3]
        )
    offset = tf.stack(
        [
        tf.reshape(offset_y, (-1)),
        tf.reshape(offset_x, (-1)),
        ]
        , axis=-1)

    return offset


def build_anchor(flat_grid_y_ctr, flat_grid_x_ctr, box_prior):
    grid_map = tf.stack(
        [flat_grid_y_ctr, flat_grid_x_ctr, flat_grid_y_ctr, flat_grid_x_ctr], axis=-1
    )
    h, w = tf.split(box_prior, 2, axis=-1)
    base_anchors = tf.concat([-h/2, -w/2, h/2, w/2], axis=-1)
    anchors = tf.reshape(base_anchors, (1, -1, 4)) + tf.reshape(tf.cast(grid_map, dtype=tf.float32), (-1, 1, 4))
    anchors = tf.reshape(anchors, (-1, 4))

    return anchors
