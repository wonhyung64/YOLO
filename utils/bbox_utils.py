import tensorflow as tf


def calculate_iou(anchors: tf.Tensor, gt_boxes: tf.Tensor) -> tf.Tensor:
    """
    calculate Intersection over Union

    Args:
        anchors (tf.Tensor): reference anchors
        gt_boxes (tf.Tensor): bbox to calculate IoU

    Returns:
        tf.Tensor: Intersection over Union
    """
    bbox_y1, bbox_x1, bbox_y2, bbox_x2 = tf.split(anchors, 4, axis=-1)
    gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(gt_boxes, 4, axis=-1)

    bbox_area = tf.squeeze((bbox_y2 - bbox_y1) * (bbox_x2 - bbox_x1), axis=-1)
    gt_area = tf.squeeze((gt_y2 - gt_y1) * (gt_x2 - gt_x1), axis=-1)

    x_top = tf.maximum(bbox_x1, tf.transpose(gt_x1, [0, 2, 1]))
    y_top = tf.maximum(bbox_y1, tf.transpose(gt_y1, [0, 2, 1]))
    x_bottom = tf.minimum(bbox_x2, tf.transpose(gt_x2, [0, 2, 1]))
    y_bottom = tf.minimum(bbox_y2, tf.transpose(gt_y2, [0, 2, 1]))

    intersection_area = tf.maximum(x_bottom - x_top, 0) * tf.maximum(
        y_bottom - y_top, 0
    )

    union_area = (
        tf.expand_dims(bbox_area, -1) + tf.expand_dims(gt_area, 1) - intersection_area
    )

    return intersection_area / union_area
