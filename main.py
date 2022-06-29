#%%
import os
import tensorflow as tf
from tqdm import tqdm
from utils import (
    load_dataset,
    build_dataset,
    load_box_prior,
    build_anchor_ops,
    build_target,
    yolo_v3,
    build_optimizer,
    forward_backward,
    build_args,
    build_lambda,
    plugin_neptune,
    record_train_loss,
    delta_to_bbox,
    decode_pred,
    draw_output,
    NEPTUNE_API_KEY,
    NEPTUNE_PROJECT,
)




#%%
if __name__ == "__main__":
    # os.makedirs("data_chkr", exist_ok=True)
    # args = build_args()
    # run = plugin_neptune(NEPTUNE_API_KEY, NEPTUNE_PROJECT, args)

    # lambda_lst = build_lambda(args)
    # datasets, labels, data_num = load_dataset(name=args.name, data_dir=args.data_dir)
    # train_set, valid_set, test_set = build_dataset(datasets, args.batch_size, args.img_size)
    # box_priors = load_box_prior(train_set, args.name, args.img_size, data_num)
    # anchors, prior_grids, offset_grids, stride_grids = build_anchor_ops(args.img_size, box_priors)
    # model = yolo_v3(args.img_size+[3], labels, offset_grids, prior_grids, fine_tunning=True)
    # optimizer = build_optimizer(args.batch_size, data_num)

    # for epoch in range(1, args.epochs+1):
    #     epoch_progress = tqdm(range(data_num//args.batch_size))
    #     for _ in epoch_progress:
    #         image, gt_boxes, gt_labels = next(train_set)
    #         true = build_target(anchors, gt_boxes, gt_labels, labels, args.img_size, stride_grids)
    #         loss, total_loss = forward_backward(image, true, model, optimizer, args.batch_size, lambda_lst)
    #         record_train_loss(run, loss, total_loss)
    #         epoch_progress.set_description(
    #             "Epoch {}/{} | yx {:.4f}, hw {:.4f}, obj {:.4f}, nobj {:.4f}, cls {:.4f}, total {:.4f}".format(
    #                 epoch,
    #                 args.epochs,
    #                 loss[0].numpy(),
    #                 loss[1].numpy(),
    #                 loss[2].numpy(),
    #                 loss[3].numpy(),
    #                 loss[4].numpy(),
    #                 total_loss.numpy(),
    #             )
    #         )

    datasets, labels, data_num = load_dataset(name="coco/2017", data_dir="D:/won/data/tfds")
    train_set, valid_set, test_set = build_dataset(datasets, 1, [416,416])
    box_priors = load_box_prior(train_set, "coco/2017", [416,416], data_num)
    anchors, prior_grids, offset_grids, stride_grids = build_anchor_ops([416,416], box_priors)
    model = tf.keras.models.load_model("tmp.h5")
    

    for _ in tqdm(range(100)):
        image, gt_boxes, gt_labels = next(test_set)
        final_bboxes, final_labels, final_scores = decode_pred(model(image))
        draw_output(image, final_bboxes, final_labels, final_scores, labels)

# %%

import numpy as np
import tensorflow as tf
from .bbox_utils import generate_iou


def calculate_PR(final_bbox, gt_box, mAP_threshold):
    bbox_num = final_bbox.shape[1]
    gt_num = gt_box.shape[1]

    true_pos = tf.Variable(tf.zeros(bbox_num))
    for i in range(bbox_num):
        bbox = tf.split(final_bbox, bbox_num, axis=1)[i]

        iou = generate_iou(bbox, gt_box)

        best_iou = tf.reduce_max(iou, axis=1)
        pos_num = tf.cast(tf.greater(best_iou, mAP_threshold), dtype=tf.float32)
        if tf.reduce_sum(pos_num) >= 1:
            gt_box = gt_box * tf.expand_dims(
                tf.cast(1 - pos_num, dtype=tf.float32), axis=-1
            )
            true_pos = tf.tensor_scatter_nd_update(true_pos, [[i]], [1])
    false_pos = 1.0 - true_pos
    true_pos = tf.math.cumsum(true_pos)
    false_pos = tf.math.cumsum(false_pos)

    recall = true_pos / gt_num
    precision = tf.math.divide(true_pos, true_pos + false_pos)

    return precision, recall


def calculate_AP_per_class(recall, precision):
    interp = tf.constant([i / 10 for i in range(0, 11)])
    AP = tf.reduce_max(
        [tf.where(interp <= recall[i], precision[i], 0.0) for i in range(len(recall))],
        axis=0,
    )
    AP = tf.reduce_sum(AP) / 11
    return AP


def calculate_AP_const(
    final_bboxes, final_labels, gt_boxes, gt_labels, hyper_params, mAP_threshold=0.5
):
    total_labels = hyper_params["total_labels"]
    AP = []
    for c in range(1, total_labels):
        if tf.math.reduce_any(final_labels == c) or tf.math.reduce_any(gt_labels == c):
            final_bbox = tf.expand_dims(final_bboxes[final_labels == c], axis=0)
            gt_box = tf.expand_dims(gt_boxes[gt_labels == c], axis=0)

            if final_bbox.shape[1] == 0 or gt_box.shape[1] == 0:
                ap = tf.constant(0.0)
            else:
                precision, recall = calculate_PR(final_bbox, gt_box, mAP_threshold)
                ap = calculate_AP_per_class(recall, precision)
            AP.append(ap)
    if AP == []:
        AP = 1.0
    else:
        AP = tf.reduce_mean(AP)
    return AP


def calculate_AP(final_bboxes, final_labels, gt_boxes, gt_labels, hyper_params):
    total_labels = hyper_params["total_labels"]
    mAP_threshold_lst = np.arange(0.5, 1.0, 0.05)
    APs = []
    for mAP_threshold in mAP_threshold_lst:
        AP = []
        for c in range(1, total_labels):
            if tf.math.reduce_any(final_labels == c) or tf.math.reduce_any(
                gt_labels == c
            ):
                final_bbox = tf.expand_dims(final_bboxes[final_labels == c], axis=0)
                gt_box = tf.expand_dims(gt_boxes[gt_labels == c], axis=0)

                if final_bbox.shape[1] == 0 or gt_box.shape[1] == 0:
                    ap = tf.constant(0.0)
                else:
                    precision, recall = calculate_PR(final_bbox, gt_box, mAP_threshold)
                    ap = calculate_AP_per_class(recall, precision)
                AP.append(ap)
        if AP == []:
            AP = 1.0
        else:
            AP = tf.reduce_mean(AP)
        APs.append(AP)
    return tf.reduce_mean(APs)
