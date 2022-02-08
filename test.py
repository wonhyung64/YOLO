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

hyper_params["batch_size"] = batch_size = 1
img_size = (hyper_params["img_size"], hyper_params["img_size"])
dataset_name = hyper_params["dataset_name"]

if dataset_name == "ship":
    import ship
    dataset, labels = ship.fetch_dataset(dataset_name, "train", img_size, save_dir="/home1/wonhyung64")
    dataset = dataset.map(lambda x, y, z, w: preprocessing_utils.preprocessing_ship(x, y, z, w))
else:
    import data_utils
    dataset, labels = data_utils.fetch_dataset(dataset_name, "train", img_size, save_dir="/home1/wonhyung64")
    dataset = dataset.map(lambda x, y, z: preprocessing_utils.preprocessing(x, y, z))

data_shapes = ([None, None, None], [None, None], [None])
padding_values = (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))
dataset = dataset.repeat().padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)
dataset = iter(dataset)

hyper_params["total_labels"] = len(labels)

box_prior = np.array([[10,13], [16,30], [33,23], [30,61], [62,45], [59,119], [116,90],  [156,198],  [373,326]], dtype = np.float32)
anchors1 = box_prior[6:9] # 13,13
anchors2 = box_prior[3:6] # 26, 26
anchors3 = box_prior[0:3] # 52, 52
anchors = [anchors3, anchors2, anchors1]

#%%
weights_dir = os.getcwd() + "/atmp"
weights_dir = weights_dir + "/" + os.listdir(weights_dir)[-1]

input_shape = (416, 416, 3)
yolo_model = model_utils.yolo_v3(input_shape, hyper_params, anchors)
yolo_model.load_weights(weights_dir + '/yolo_weights/weights')

#%%
# save_dir = os.getcwd()
# save_dir = utils.generate_save_dir(save_dir, hyper_params)

total_time = []
mAP = []

progress_bar = tqdm(range(hyper_params['attempts']))
for _ in progress_bar:
    img, gt_boxes, gt_labels = next(dataset)
    start_time = time.time()
    pred = yolo_model(img)
    final_bboxes, final_labels, final_scores = postprocessing_utils.Decode(pred, hyper_params)
    time_ = float(time.time() - start_time)*1000
    AP = test_utils.calculate_AP(final_bboxes, final_labels, gt_boxes, gt_labels, hyper_params)
    test_utils.draw_yolo_output(img, final_bboxes, labels, final_labels, final_scores)
    total_time.append(time_)
    mAP.append(AP)

print("mAP: %.2f" % (tf.reduce_mean(mAP)))
print("Time taken: %.2fms" % (tf.reduce_mean(total_time)))

#%%
def draw_custom_img(img_dir):
    image = Image.open(img_dir)
    image_ = np.array(image)
    image_ = tf.convert_to_tensor(image_)
    image_ = tf.image.resize(image_, img_size)/ 255
    img = tf.expand_dims(image_, axis=0)
    pred = yolo_model(img)
    final_bboxes, final_labels, final_scores = postprocessing_utils.Decode(pred, hyper_params)
    test_utils.draw_yolo_output(img, final_bboxes, labels, final_labels, final_scores)

test_utils.draw_custom_img("C:/won/test9.jpg")