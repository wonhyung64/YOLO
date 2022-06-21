import tensorflow as tf
from .bbox_utils import calculate_iou


def build_pos_target(anchors, gt_boxes, gt_labels, labels):
    iou_map = calculate_iou(anchors, gt_boxes)
    max_indices_each_column = tf.argmax(iou_map, axis=1, output_type=tf.int32)
    valid_indices_cond = tf.not_equal(gt_labels, -1)
    valid_indices = tf.cast(tf.where(valid_indices_cond), tf.int32)
    valid_gt_boxes = gt_boxes[valid_indices_cond]
    valid_gt_labels = gt_labels[valid_indices_cond] + 1
    valid_max_indices = max_indices_each_column[valid_indices_cond]
    scatter_bbox_indices = tf.stack([valid_indices[..., 0], valid_max_indices], 1)

    pos_reg = tf.scatter_nd(
        indices=scatter_bbox_indices,
        updates=valid_gt_boxes,
        shape=tf.concat([tf.shape(iou_map)[:2], tf.constant([4], tf.int32)], axis=0),
    )

    pos_mask = tf.scatter_nd(
        indices=scatter_bbox_indices,
        updates=tf.fill(tf.shape(valid_indices)[0], True),
        shape=tf.shape(iou_map)[:2],
    )
    pos_obj = tf.where(
        pos_mask,
        tf.ones_like(pos_mask, dtype=tf.float32),
        tf.zeros_like(pos_mask, dtype=tf.float32),
    )
    pos_obj = tf.expand_dims(pos_obj, axis=-1)

    pos_cls = (
        tf.scatter_nd(
            indices=scatter_bbox_indices,
            updates=valid_gt_labels,
            shape=tf.shape(iou_map)[:2],
        )
        - 1
    )
    pos_cls = tf.one_hot(pos_cls, len(labels), dtype=tf.float32)

    return pos_reg, pos_obj, pos_cls
