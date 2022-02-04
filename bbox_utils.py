#%%
import tensorflow as tf

#%%
def bbox_iou(pred_boxes, valid_true_boxes):
    pred_box_xy = pred_boxes[..., 0:2]
    pred_box_wh = pred_boxes[..., 2:4]

    pred_box_xy = tf.expand_dims(pred_box_xy, -2)
    pred_box_wh = tf.expand_dims(pred_box_wh, -2)
    
    true_box_xy = valid_true_boxes[..., 0:2]
    true_box_wh = valid_true_boxes[..., 2:4]

    intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2., true_box_xy - true_box_wh / 2.)
    intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2., true_box_xy + true_box_wh / 2.)
    intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)

    intersection = intersect_wh[...,0] * intersect_wh[...,1]

    pred_box_area = pred_box_wh[...,0] * pred_box_wh[...,1]
    
    true_box_area = true_box_wh[...,0] * true_box_wh[...,1]
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
    x_ctr, y_ctr, width, height = tf.split(boxes, [1,1,1,1], axis=-1)
    x1 = x_ctr - width/2
    y1 = y_ctr - height/2
    x2 = x_ctr + width/2
    y2 = y_ctr + height/2
    boxes = tf.concat([y1, x1, y2, x2], axis=-1)
    return boxes