#%%
import os
import time
import tensorflow as tf
import numpy as np

from PIL import Image
from tqdm import tqdm

import model_utils, data_utils, utils, preprocessing_utils, postprocessing_utils, test_utils

#%%
hyper_params = utils.get_hyper_params()
hyper_params["score_thresh"] = 0.1

hyper_params["batch_size"] = batch_size = 1
img_size = (hyper_params["img_size"], hyper_params["img_size"])
dataset_name = hyper_params["dataset_name"]

if dataset_name == "ship":
    import ship
    dataset, labels = ship.fetch_dataset(dataset_name, "train", img_size, save_dir="/home1/wonhyung64")
    dataset = dataset.map(lambda x, y, z, w: preprocessing_utils.preprocessing_ship(x, y, z, w))
else:
    import data_utils
    dataset, labels = data_utils.fetch_dataset(dataset_name, "test", img_size)
    dataset = dataset.map(lambda x, y, z: preprocessing_utils.preprocessing(x, y, z))

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
weights_dir = os.getcwd() + "/yolo_atmp"
weights_no_dir = weights_dir + "/" + os.listdir(weights_dir)[-2]
weights_dir = weights_dir + "/" + os.listdir(weights_dir)[-1]

input_shape = (416, 416, 3)
yolo_model = model_utils.yolo_v3(input_shape, hyper_params)
yolo_model.load_weights(weights_dir + '/yolo_weights/weights')

# yolo_model_no = model_utils.yolo_v3(input_shape, hyper_params)
# yolo_model_no.load_weights(weights_no_dir + "/yolo_weights/weights")

#%%
# save_dir = os.getcwd()
# save_dir = utils.generate_save_dir(save_dir, hyper_params)

total_time = []
mAP = []


try_num = 0
progress_bar = tqdm(range(100))
for _ in progress_bar:
    img, gt_boxes, gt_labels = next(dataset)
    start_time = time.time()
    yolo_outputs = yolo_model(img)
    # yolo_outputs = yolo_model_no(img)
    final_bboxes, final_labels, final_scores = postprocessing_utils.Decode(yolo_outputs)
    final_labels
    time_ = float(time.time() - start_time)*1000
    AP = test_utils.calculate_AP(final_bboxes, final_labels, gt_boxes, gt_labels, hyper_params)
    print(try_num)
    test_utils.draw_yolo_output(img, final_bboxes, labels, final_labels, final_scores)

    total_time.append(time_)
    mAP.append(AP)
    try_num += 1

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
#%%