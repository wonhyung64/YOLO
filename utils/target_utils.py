import tensorflow as tf
from .bbox_utils import calculate_iou, bbox_to_delta


def build_target(anchors, gt_boxes, gt_labels, labels, img_size, stride_grids, ignore_threshold=0.5):
    iou_map = calculate_iou(anchors, gt_boxes)
    valid_indices, valid_gt_boxes, valid_gt_labels, scatter_bbox_indices = extract_valid_boxes(iou_map, gt_boxes, gt_labels)

    true_yx, true_hw = build_true_reg(scatter_bbox_indices, valid_gt_boxes, iou_map, img_size, stride_grids)
    true_obj, pos_mask = build_true_obj(scatter_bbox_indices, valid_indices, iou_map)
    true_nobj = build_true_nobj(iou_map, ignore_threshold, pos_mask)
    true_cls = build_true_cls(scatter_bbox_indices, valid_gt_labels, iou_map, labels)

    return true_yx, true_hw, true_obj, true_nobj, true_cls


def extract_valid_boxes(iou_map, gt_boxes, gt_labels):
    max_indices_each_column = tf.argmax(iou_map, axis=1, output_type=tf.int32)
    valid_indices_cond = tf.not_equal(gt_labels, -1)
    valid_indices = tf.cast(tf.where(valid_indices_cond), tf.int32)
    valid_gt_boxes = gt_boxes[valid_indices_cond]
    valid_gt_labels = gt_labels[valid_indices_cond] + 1
    valid_max_indices = max_indices_each_column[valid_indices_cond]
    scatter_bbox_indices = tf.stack([valid_indices[..., 0], valid_max_indices], 1)

    return valid_indices, valid_gt_boxes, valid_gt_labels, scatter_bbox_indices


def build_true_reg(scatter_bbox_indices, valid_gt_boxes, iou_map, img_size, stride_grids):
    true_reg = tf.scatter_nd(
        indices=scatter_bbox_indices,
        updates=valid_gt_boxes,
        shape=tf.concat([tf.shape(iou_map)[:2], tf.constant([4], tf.int32)], axis=0),
    )
    true_yx, true_hw = bbox_to_delta(true_reg, img_size, stride_grids)

    return true_yx, true_hw


def build_true_obj(scatter_bbox_indices, valid_indices, iou_map):
    pos_mask = tf.scatter_nd(
        indices=scatter_bbox_indices,
        updates=tf.fill(tf.shape(valid_indices)[0], True),
        shape=tf.shape(iou_map)[:2],
    )
    true_obj = tf.where(
        pos_mask,
        tf.ones_like(pos_mask, dtype=tf.float32),
        tf.zeros_like(pos_mask, dtype=tf.float32),
    )
    true_obj = tf.expand_dims(true_obj, axis=-1)

    return true_obj, pos_mask


def build_true_nobj(iou_map, ignore_threshold, pos_mask):
    merged_iou_map = tf.reduce_max(iou_map, axis=2)
    neg_mask = tf.logical_and(
        tf.less(merged_iou_map, ignore_threshold), tf.logical_not(pos_mask))
    true_nobj = tf.where(
        neg_mask,
        tf.ones_like(pos_mask, dtype=tf.float32),
        tf.zeros_like(pos_mask, dtype=tf.float32),
    )
    true_nobj = tf.expand_dims(true_nobj, axis=-1)

    return true_nobj


def build_true_cls(scatter_bbox_indices, valid_gt_labels, iou_map, labels):
    true_cls = (
        tf.scatter_nd(
            indices=scatter_bbox_indices,
            updates=valid_gt_labels,
            shape=tf.shape(iou_map)[:2],
        )
        - 1
    )
    true_cls = tf.one_hot(true_cls, len(labels), dtype=tf.float32)

    return true_cls

