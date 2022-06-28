#%%
import os
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
    NEPTUNE_API_KEY,
    NEPTUNE_PROJECT,
)

def decode_pred(
    pred,
    batch_size=1,
    max_total_size=200,
    iou_threshold=0.5,
    score_threshold=0.7,
):
    pred_yx, pred_hw, pred_obj, pred_cls = pred
    pred_bboxes = delta_to_bbox(pred_yx, pred_hw, stride_grids)

    pred_bboxes = tf.reshape(pred_bboxes, (batch_size, -1, 1, 4))
    pred_labels = pred_cls * pred_obj

    final_bboxes, final_scores, final_labels, _ = tf.image.combined_non_max_suppression(
        pred_bboxes,
        pred_labels,
        max_output_size_per_class=max_total_size,
        max_total_size=max_total_size,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )

    return final_bboxes, final_labels, final_scores

def draw_output(
    image,
    final_bboxes,
    final_labels,
    final_scores,
    labels,
):
    image = tf.squeeze(image, axis=0)
    image = tf.keras.preprocessing.image.array_to_img(image)
    width, height = image.size
    draw = ImageDraw.Draw(image)

    y1 = final_bboxes[0][..., 0] * height
    x1 = final_bboxes[0][..., 1] * width
    y2 = final_bboxes[0][..., 2] * height
    x2 = final_bboxes[0][..., 3] * width

    denormalized_box = tf.round(tf.stack([y1, x1, y2, x2], axis=-1))

    colors = tf.random.uniform((len(labels), 4), maxval=256, dtype=tf.int32)

    for index, bbox in enumerate(denormalized_box):
        y1, x1, y2, x2 = tf.split(bbox, 4, axis=-1)
        width = x2 - x1
        height = y2 - y1

        final_labels_ = tf.reshape(final_labels[0], shape=(200,))
        final_scores_ = tf.reshape(final_scores[0], shape=(200,))
        label_index = int(final_labels_[index])
        color = tuple(colors[label_index].numpy())
        label_text = "{0} {1:0.3f}".format(labels[label_index], final_scores_[index])
        draw.text((x1 + 4, y1 + 2), label_text, fill=color)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)

    return image


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
    model = yolo_v3([416,416]+[3], labels, offset_grids, prior_grids, fine_tunning=True)
    # model.load_weights("C:/Users/USER/Documents/GitHub/YOLO/yolov3_org_weights/weights")
    model.save("tmp.h5")
    # model.load("tmp.h5")
    model = tf.keras.models.load_model("tmp.h5")
    
import tensorflow as tf
from PIL import ImageDraw

    image, gt_boxes, gt_labels = next(train_set)
    final_bboxes, final_labels, final_scores = decode_pred(model(image))
    draw_output(image, final_bboxes, final_labels, final_scores, labels)

# %%
