#%%
from sklearn.cluster import KMeans
import pandas as pd

from utils import (
    load_dataset,
    build_dataset,
)
#%%
data_dir = "D:/won/data/tfds"
img_size = (416, 416)
batch_size=4
name = "coco/2017"

datasets, labels, data_num = load_dataset(name=name, data_dir=data_dir)
train_set, valid_set, test_set = build_dataset(datasets, data_num, batch_size, img_size)

# %%
import tensorflow as tf
from tqdm import tqdm


def extract_boxes(gt_boxes, img_size):
    y1, x1, y2, x2 = tf.split(gt_boxes, 4, axis=-1)
    gt_heights = (y2 - y1) * img_size[0]
    gt_widths = (x2 - x1) * img_size[1]
    prior_sample = tf.concat([gt_heights, gt_widths], axis=-1)
    prior_sample = tf.reshape(prior_sample, (tf.shape(prior_sample)[0] * tf.shape(prior_sample)[1], 2))
    not_zero = tf.not_equal(prior_sample, 0)
    not_zero = tf.logical_and(not_zero[...,0], not_zero[...,1])
    prior_sample = prior_sample[not_zero]

    return prior_sample

prior_samples = []
for _ in tqdm(range(data_num)):
    image, gt_boxes, _ = next(train_set)
    prior_sample = extract_boxes(gt_boxes, img_size)
    prior_samples.append(prior_sample)

hw = tf.concat(prior_samples, axis=0)

k = 9
model = KMeans(n_clusters = k, random_state = 1)
model.fit(hw)
box_prior = model.cluster_centers_
pd.DataFrame(box_prior, columns=['height', 'width']).to_csv('C:/Users/USER/Documents/GitHub/YOLO/box_prior.csv')