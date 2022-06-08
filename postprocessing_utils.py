#%%
import tensorflow as tf
import utils, bbox_utils, model_utils

#%%
def Decode(yolo_outputs, score_thresh=0.7, nms_thresh=0.5):

    anchors = utils.get_box_prior()
    num_layers = len(anchors)//3

    hyper_params = utils.get_hyper_params()
    num_classes = hyper_params["total_labels"]
    max_boxes = hyper_params["nms_boxes_per_class"]

    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]

    input_shape = tf.cast(tf.shape(yolo_outputs[0])[1:3] * 32, tf.float32)


    box, obj, cls = [], [], []

    for l in range(num_layers):
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
    
    return final_bboxes, final_labels, final_scores