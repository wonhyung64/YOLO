#%%
import tensorflow as tf
import bbox_utils

#%%
def Decode(pred, hyper_params):
    num_classes = hyper_params["total_labels"]
    max_boxes = hyper_params["nms_boxes_per_class"]
    score_thresh = hyper_params["score_thresh"]
    nms_thresh = hyper_params["nms_thresh"]

    pred = tf.squeeze(pred, axis=0)
    box = pred[...,:4]
    box = bbox_utils.xywh_to_bbox(box)
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

    final_bboxes = tf.expand_dims(tf.concat(box_lst, axis=0), axis=0)
    final_labels = tf.expand_dims(tf.concat(label_lst, axis=0), axis=0)
    final_scores = tf.expand_dims(tf.concat(score_lst, axis=0), axis=0)

    return final_bboxes, final_labels, final_scores