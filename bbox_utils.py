#%%
import tensorflow as tf

#%%
def bbox_iou_tmp(pred_boxes, valid_true_boxes):
    pred_box_yx = pred_boxes[..., 0:2]
    pred_box_hw = pred_boxes[..., 2:4]

    pred_box_yx = tf.expand_dims(pred_box_yx, -2)
    pred_box_hw = tf.expand_dims(pred_box_hw, -2)
    
    true_box_yx = valid_true_boxes[..., 0:2]
    true_box_hw = valid_true_boxes[..., 2:4]

    intersect_mins = tf.maximum(pred_box_yx - pred_box_hw / 2., true_box_yx - true_box_hw / 2.)
    intersect_maxs = tf.minimum(pred_box_yx + pred_box_hw / 2., true_box_yx + true_box_hw / 2.)
    intersect_hw = tf.maximum(intersect_maxs - intersect_mins, 0.)

    intersection = intersect_hw[...,0] * intersect_hw[...,1]

    pred_box_area = pred_box_hw[...,0] * pred_box_hw[...,1]
    
    true_box_area = true_box_hw[...,0] * true_box_hw[...,1]
    true_box_area = tf.expand_dims(true_box_area, axis=0)

    union = pred_box_area + true_box_area - intersection

    return intersection / union
    
#%%
def generate_iou(anchors, gt_boxes):
    bbox_y1, bbox_x1, bbox_y2, bbox_x2 = tf.split(anchors, 4, axis=-1) 
    gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(gt_boxes, 4, axis=-1) 
    
    bbox_area = tf.squeeze((bbox_y2 - bbox_y1) * (bbox_x2 - bbox_x1), axis=-1) 
    gt_area = tf.squeeze((gt_y2 - gt_y1) * (gt_x2 - gt_x1), axis = -1)
    
    x_top = tf.maximum(bbox_x1, tf.transpose(gt_x1, [0, 2, 1])) 
    y_top = tf.maximum(bbox_y1, tf.transpose(gt_y1, [0, 2, 1]))
    x_bottom = tf.minimum(bbox_x2, tf.transpose(gt_x2, [0, 2, 1]))
    y_bottom = tf.minimum(bbox_y2, tf.transpose(gt_y2, [0, 2, 1]))
    
    intersection_area = tf.maximum(x_bottom - x_top, 0) * tf.maximum(y_bottom - y_top, 0)
    
    union_area = (tf.expand_dims(bbox_area, -1) + tf.expand_dims(gt_area, 1) - intersection_area)
    
    return intersection_area / union_area 

#%%
def xywh_to_bbox(boxes):
    y_ctr, x_ctr, height, width = tf.split(boxes, [1,1,1,1], axis=-1)
    y1 = y_ctr - height/2
    x1 = x_ctr - width/2
    y2 = y_ctr + height/2
    x2 = x_ctr + width/2
    boxes = tf.concat([y1, x1, y2, x2], axis=-1)
    return boxes

#%%
def box_iou(b1, b2):
    b1 = tf.expand_dims(b1, -2)
    b1_yx = b1[..., :2]
    b1_hw = b1[..., 2:4]
    b1_hw_half = b1_hw / 2.
    b1_mins = b1_yx - b1_hw_half
    b1_maxes = b1_yx + b1_hw_half

    b2 = tf.expand_dims(b2, 0)
    b2_yx = b2[..., :2]
    b2_hw = b2[..., 2:4]
    b2_hw_half = b2_hw / 2.
    b2_mins = b2_yx - b2_hw_half
    b2_maxes = b2_yx + b2_hw_half

    intersect_mins = tf.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
    intersect_hw = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_hw[..., 0] * intersect_hw[..., 1]
    b1_area = b1_hw[..., 0] * b1_hw[..., 1]
    b2_area = b2_hw[..., 0] * b2_hw[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou

