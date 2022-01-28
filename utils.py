#%%
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import ImageDraw
#%%
def get_hyper_params():
    hyper_params = {
        "img_size" : 416,
        "total_class" : 80,
        "iterations" : 20000,
        "nms_boxes_per_class" : 50,
        "score_thresh" : 0.5,
        "nms_thresh" : 0.5,
        "coord_tune" : 5.,
        "noobj_tune" : .5
    }
    return hyper_params
#%%
def nms(pred, hyper_params):
    num_classes = hyper_params["total_class"]
    max_boxes = hyper_params["nms_boxes_per_class"]
    score_thresh = hyper_params["score_thresh"]
    nms_thresh = hyper_params["nms_thresh"]

    pred = tf.squeeze(pred, axis=0)
    box = pred[...,:4]
    x1 = box[...,0] - box[...,2]/2
    x2 = box[...,0] + box[...,2]/2
    y1 = box[...,1] - box[...,3]/2
    y2 = box[...,1] + box[...,3]/2
    box = tf.stack([x1, y1, x2, y2], axis=-1)
    box = tf.clip_by_value(box, 0, hyper_params["img_size"])

    obj = pred[...,4:5]
    cls = pred[...,5:]
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

    box = tf.concat(box_lst, axis=0)
    label = tf.concat(label_lst, axis=0)
    score = tf.concat(score_lst, axis=0)

    return box, score, label
#%%
def draw(images, box, score, label, hyper_params):
        image = tf.squeeze(images, axis=0)

        image = tf.keras.preprocessing.image.array_to_img(image)
        width, height = image.size
        draw = ImageDraw.Draw(image)
        # gt_boxes_ = tf.reshape(gt_boxes, (7,4))

        for i in range(box.shape[0]):
            tmp_box = box[i]
            y1, x1, y2, x2 = tf.split(tmp_box, 4, axis = -1)
            draw.rectangle((x1, y1, x2, y2), outline=50, width=3)

        # for j in range(gt_boxes_.shape[0]):
        #     tmp_box = gt_boxes_[j] * 416
        #     y1, x1, y2, x2 = tf.split(tmp_box, 4, axis = -1) 
        #     draw.rectangle((x1, y1, x2, y2), outline=color, width=3)

        plt.figure()
        plt.imshow(image)
        plt.show()