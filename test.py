#%%
import os
import time
import tensorflow as tf
import numpy as np

from PIL import Image
from tqdm import tqdm

import model_utils, data_utils, utils, preprocessing_utils, postprocessing_utils, test_utils
import convert
import tensorflow_datasets as tfds
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Concatenate

#%%
hyper_params = utils.get_hyper_params()

hyper_params["batch_size"] = batch_size = 1
img_size = (hyper_params["img_size"], hyper_params["img_size"])
dataset_name = hyper_params["dataset_name"]

# if dataset_name == "ship":
#     import ship
#     dataset, labels = ship.fetch_dataset(dataset_name, "train", img_size, save_dir="/home1/wonhyung64")
#     dataset = dataset.map(lambda x, y, z, w: preprocessing_utils.preprocessing_ship(x, y, z, w))
# else:
#     import data_utils
#     dataset, labels = data_utils.fetch_dataset(dataset_name, "train", img_size)
#     dataset = dataset.map(lambda x, y, z: preprocessing_utils.preprocessing(x, y, z))

def preprocessing(sample):
    image = sample["image"] 
    image = tf.image.resize(image, (416, 416)) / 255
    gt_boxes = sample["objects"]["bbox"]
    gt_labels = tf.cast(sample["objects"]["label"], tf.int32)
    return image, gt_boxes, gt_labels

dataset, dataset_info = tfds.load(name="coco/2017", data_dir="D:/won/data/tfds", with_info=True)
dataset = dataset["validation"]
labels = dataset_info.features["objects"]["label"].names
dataset = dataset.map(lambda x : preprocessing(x))

dataset = dataset.repeat().batch(batch_size)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
dataset = iter(dataset)

#%%
box_prior = utils.get_box_prior()
box_prior = np.array(box_prior)
anchors1 = box_prior[6:9] # 13,13
anchors2 = box_prior[3:6] # 26, 26
anchors3 = box_prior[0:3] # 52, 52
anchors = [anchors1, anchors2, anchors3]

#%%
# yolo_model.save_weights(weights_dir)


weights_dir = os.getcwd() + "/yolo_atmp"
weights_dir = weights_dir + "/" + os.listdir(weights_dir)[-1]
# weights_dir = os.getcwd() + "/redmon_weights/weights"
input_shape = (416, 416, 3)
yolo_model = model_utils.yolo_v3(input_shape, hyper_params)
yolo_model.load_weights(weights_dir)

#%%
# save_dir = os.getcwd()
# save_dir = utils.generate_save_dir(save_dir, hyper_params)

total_time = []
mAP = []

save_num = 0
progress_bar = tqdm(range(20))
for _ in progress_bar:
    img, gt_boxes, gt_labels = next(dataset)
    start_time = time.time()
    yolo_outputs = yolo_model(img)
    final_bboxes, final_labels, final_scores = postprocessing_utils.Decode(yolo_outputs)
    time_ = float(time.time() - start_time)*1000
    AP = test_utils.calculate_AP(final_bboxes, final_labels, gt_boxes, gt_labels, hyper_params)
    # test_utils.draw_yolo_output(img, final_bboxes, labels, final_labels, final_scores, save_dir=r"C:/Users/USER/Documents/GitHub\YOLO/yolo_atmp/redmon_weights2", save_num=save_num)
    test_utils.draw_yolo_output(img, final_bboxes, labels, final_labels, final_scores)
    total_time.append(time_)
    mAP.append(AP)
    save_num += 1

print("mAP: %.2f" % (tf.reduce_mean(mAP)))
print("Time taken: %.2fms" % (tf.reduce_mean(total_time)))

#%%
def draw_custom_img(img_dir):
    image = Image.open(img_dir)
    image_ = np.array(image)
    image_ = tf.convert_to_tensor(image_)
    image_ = tf.image.resize(image_, img_size)/ 255
    img = tf.expand_dims(image_, axis=0)
    yolo_outputs = yolo_model(img)
    final_bboxes, final_labels, final_scores = postprocessing_utils.Decode(yolo_outputs)
    test_utils.draw_yolo_output(img, final_bboxes, labels, final_labels, final_scores)

draw_custom_img("C:/won/test.jpg")
# %%
img, gt_boxes, gt_labels = next(dataset)
tf.keras.preprocessing.image.array_to_img(tf.squeeze(img, axis=0))
yolo_outputs = yolo_model(img)
final_bboxes, final_labels, final_scores = postprocessing_utils.Decode(yolo_outputs)
test_utils.draw_yolo_output(img, final_bboxes, labels, final_labels, final_scores)

from PIL import ImageDraw
import matplotlib.pyplot as plt

image = tf.squeeze(img, axis=0)
image = tf.keras.preprocessing.image.array_to_img(image)
draw = ImageDraw.Draw(image)

y1 = final_bboxes[0][...,0]
x1 = final_bboxes[0][...,1]
y2 = final_bboxes[0][...,2]
x2 = final_bboxes[0][...,3]

denormalized_box = tf.round(tf.stack([y1, x1, y2, x2], axis=-1))

colors = tf.random.uniform((len(labels), 4), maxval=256, dtype=tf.int32)


    
for index, bbox in enumerate(denormalized_box):
    y1, x1, y2, x2 = tf.split(bbox, 4, axis = -1)
    width = x2 - x1
    height = y2 - y1

    final_labels_ = tf.reshape(final_labels[0], shape=(final_labels.shape[1],))
    final_scores_ = tf.reshape(final_scores[0], shape=(final_scores.shape[1],))
    label_index = int(final_labels_[index])
    color = tuple(colors[label_index].numpy())
    label_text = "{0} {1:0.3f}".format(labels[label_index], final_scores_[index])
    draw.text((x1 + 4, y1 + 2), label_text, fill=color)
    draw.rectangle((x1, y1, x2, y2), outline=color, width=3)


denormalized_gt = tf.round(tf.squeeze(gt_boxes, axis=0) * 416)
for index, bbox in enumerate(denormalized_gt):
    y1, x1, y2, x2 = tf.split(bbox, 4, axis=-1)
    draw.rectangle((x1, y1, x2, y2), outline=(255, 0 , 0), width=3)
    

plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.show()
#%%
gt_boxes[0]
hw = []

for sample in tqdm(range(5011)):
    _, gt_boxes, _ = next(dataset)
    for bbox in gt_boxes[0]:
        y1, x1, y2, x2 = tf.split(bbox, 4, axis=-1)
        hw.append(tf.concat([(y2 - y1) * 416, (x2 - x1) * 416], axis=0))

hw = np.array(hw)
#%%
'''box prior'''
from sklearn.cluster import KMeans
import pandas as pd

k = 3
model = KMeans(n_clusters = k, random_state = 1)
model.fit(hw_2)
box_prior2 = model.cluster_centers_
pd.DataFrame(box_prior, columns=['height', 'width']).to_csv('C:/Users/USER/Documents/GitHub/YOLO/box_prior.csv')
#%%
#%%
box_prior = tf.cast([
    [44.876007, 33.94767],
    [111.59584, 55.58249],
    [110.780045, 148.88736],
    [198.02014, 87.824524],
    [312.71027, 121.00673],
    [212.9126, 209.1222],
    [197.57387, 344.69098],
    [353.16132, 235.58124],
    [361.2718, 371.65466]
    
], dtype = tf.float32)

box_prior = tf.cast([
    [34.62897487, 21.52063238],
    [50.31485668, 56.76226806],
    [95.32252384, 30.32935004],
    [119.00188798, 74.40600956],
    [220.72284, 80.4646506],
    [114.71482119, 157.40415637],
    [297.71821858, 165.87902991],
    [193.59411315, 293.148731  ],
    [356.74360478, 343.19366774]
], dtype = tf.float32)
#%%
hw_ = tf.convert_to_tensor(hw)
h_, w_ = tf.split(hw_, 2, -1)
hw_area = np.array(h_ * w_)

h_1 = h_[hw_area <= np.quantile(hw_area, 0.333333)]
w_1 = w_[hw_area <= np.quantile(hw_area, 0.333333)]
hw_1 = tf.concat([tf.expand_dims(h_1, -1), tf.expand_dims(w_1, -1)], axis=-1)

h_2 = h_[np.logical_and(hw_area > np.quantile(hw_area, 0.333333), hw_area <= np.quantile(hw_area, 0.666666))]
w_2 = w_[np.logical_and(hw_area > np.quantile(hw_area, 0.333333), hw_area <= np.quantile(hw_area, 0.666666))]
hw_2 = tf.concat([tf.expand_dims(h_2, -1), tf.expand_dims(w_2, -1)], axis=-1)

h_3 = h_[hw_area > np.quantile(hw_area, 0.666666)]
w_3 = w_[hw_area > np.quantile(hw_area, 0.666666)]
hw_3 = tf.concat([tf.expand_dims(h_3, -1), tf.expand_dims(w_3, -1)], axis=-1)

box_prior1
box_prior1[:,0] * box_prior1[:,1]

box_prior2
box_prior2[:,0] * box_prior2[:,1]
box_prior3
box_prior3[:,0] * box_prior3[:,1]


#%%
#%%
# %%




#%%