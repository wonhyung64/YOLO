#%%
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Lambda

from utils import (
    load_dataset,
    build_dataset,
    preprocess,
)
import tensorflow_datasets as tfds
#%%
data_dir = "D:/won/data/tfds"
img_size = (416, 416)
batch_size=4
name = "voc/2007"

datasets, labels, data_num = load_dataset(name=name, data_dir=data_dir)
train_set, valid_set, test_set = build_dataset(datasets, data_num, batch_size, img_size)
print(next(train_set))

# %%


def extract_boxes(gt_boxes, img_size):
    y1, x1, y2, x2 = tf.split(gt_boxes, 4, axis=-1)
    gt_heights = (y2 - y1) * tf.constant(img_size[0], dtype=tf.float32)
    gt_widths = (x2 - x1) * tf.constant(img_size[1], dtype=tf.float32)
    prior_sample = tf.concat([gt_heights, gt_widths], axis=-1)
    prior_sample = tf.reshape(prior_sample, (tf.shape(prior_sample)[0] * tf.shape(prior_sample)[1], 2))
    not_zero = tf.not_equal(prior_sample, 0)
    not_zero = tf.logical_and(Lambda(lambda x: x[..., 0])(not_zero), Lambda(lambda x: x[..., 1])(not_zero))
    prior_sample = Lambda(lambda x: x[not_zero])(prior_sample)

    return prior_sample


def k_means(cluster_sample, k_per_grid):
    model = KMeans(n_clusters = k_per_grid, random_state = 1)
    model.fit(cluster_sample)
    box_prior = model.cluster_centers_

    return box_prior

#%%
# k_per_grid = 3
# # def build_box_prior(dataset, img_size, k_per_grid):
# prior_samples = []
# # for _ in tqdm(range(data_num)):
# for _ in tqdm(range(10)):
#     image, gt_boxes, _ = next(train_set)
#     prior_sample = extract_boxes(gt_boxes, img_size)
#     prior_samples.append(prior_sample)
# gt_hws = tf.concat(prior_samples, axis=0)
# del prior_samples

# hw_area = gt_hws[...,0] * gt_hws[...,1]
# hw1 = gt_hws[hw_area <= np.quantile(hw_area, 0.333333)]
# hw2 = gt_hws[np.logical_and(
#     hw_area > np.quantile(hw_area, 0.333333),
#     hw_area <= np.quantile(hw_area, 0.666666)
#     )]
# hw3 = gt_hws[hw_area > np.quantile(hw_area, 0.666666)]

# box_prior = []
# for cluster_sample in (hw1, hw2, hw3):
#     box_prior.append(k_means(cluster_sample, k_per_grid))
# box_prior = np.concatenate(box_prior)

# prior_df = pd.DataFrame(box_prior, columns=["height", "width"] )
# prior_df.insert(2, "area", box_prior[..., 0] * box_prior[..., 1])
# prior_df = prior_df.sort_values(by=["area"], axis=0)
# name = ''.join(filter(str.isalnum, name)) 
# prior_df[["height", "width"]].to_csv(f'./box_prior_tmp{name}.csv', header=False, index=False)

# %%
