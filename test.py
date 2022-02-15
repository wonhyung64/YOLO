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
weights_dir = weights_dir + "/" + os.listdir(weights_dir)[1]

input_shape = (416, 416, 3)
yolo_model = model_utils.yolo_v3(input_shape, hyper_params)
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
    yolo_outputs = yolo_model(img)
    final_bboxes, final_labels, final_scores = postprocessing_utils.Decode(yolo_outputs)
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
    yolo_outputs = yolo_model(img)
    final_bboxes, final_labels, final_scores = postprocessing_utils.Decode(yolo_outputs)
    test_utils.draw_yolo_output(img, final_bboxes, labels, final_labels, final_scores)

draw_custom_img("C:/won/test.jpg")
# %%

    img, gt_boxes, gt_labels = next(dataset)
    yolo_outputs = yolo_model(img)

    anchors = utils.get_box_prior()
    num_layers = len(anchors)//3

    hyper_params = utils.get_hyper_params()
    num_classes = hyper_params["total_labels"]
    max_boxes = hyper_params["nms_boxes_per_class"]
    score_thresh = hyper_params["score_thresh"]
    nms_thresh = hyper_params["nms_thresh"]

    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]

    input_shape = tf.cast(tf.shape(yolo_outputs[0])[1:3] * 32, tf.float32)


    box, obj, cls = [], [], []

    for l in range(num_layers):
        l = 0
        _, _, pred_yx, pred_hw, pred_obj, pred_cls = model_utils.yolo_head(yolo_outputs[l], tf.gather(anchors, anchor_mask[l]), num_classes, input_shape)

        pred_box = tf.concat([pred_yx, pred_hw], axis=-1)
        pred_box *= hyper_params["img_size"]
        pred_box = tf.reshape(pred_box, [pred_box.shape[0], pred_box.shape[1] * pred_box.shape[2] * pred_box.shape[3], 4])
        box.append(pred_box)

        pred_obj = tf.reshape(pred_obj, [pred_obj.shape[0], pred_obj.shape[1] * pred_obj.shape[2] * pred_obj.shape[3], 1])
        obj.append(pred_obj)

        pred_cls = tf.reshape(pred_cls, [pred_cls.shape[0], pred_cls.shape[1] * pred_cls.shape[2] * pred_cls.shape[3], num_classes])
        cls.append(pred_cls)
    
    box, obj, cls = tf.squeeze(tf.concat(box, axis=1), 0), tf.squeeze(tf.concat(obj, axis=1), 0), tf.squeeze(tf.concat(cls, axis=1), 0)
    box = bbox_utils.xywh_to_bbox(box)
    box = tf.clip_by_value(box, 0., input_shape[0])

    score = obj * cls

    max_boxes = tf.constant(max_boxes, dtype=tf.int32)

    mask = tf.greater_equal(score, tf.constant(score_thresh))

    box_lst, label_lst, score_lst = [], [], []
    for i in range(num_classes):
        filter_boxes = tf.boolean_mask(box, mask[...,i])
        filter_scores = tf.boolean_mask(score[...,i], mask[...,i])

        nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                scores=filter_scores,
                                                max_output_size=max_boxes,
                                                iou_threshold=nms_thresh)
        box_lst.append(tf.gather(filter_boxes, nms_indices))
        score_lst.append(tf.gather(filter_scores, nms_indices))
        label_lst.append(tf.ones_like(tf.gather(filter_scores, nms_indices), dtype=tf.int32) * i)

    final_bboxes = tf.expand_dims(tf.concat(box_lst, axis=0), axis=0)
    final_labels = tf.expand_dims(tf.concat(label_lst, axis=0), axis=0)
    final_scores = tf.expand_dims(tf.concat(score_lst, axis=0), axis=0)