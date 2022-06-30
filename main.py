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
    calculate_iou,
    calculate_ap_const,
    NEPTUNE_API_KEY,
    NEPTUNE_PROJECT,
)



def validation(valid_set, model):
    map = []
    validation_progress = tqdm(range(100))
    for _ in validation_progress:
        image, gt_boxes, gt_labels = next(valid_set)
        pred = model(image)
        pred = [tf.stack([pred[0][...,1], pred[0][...,0]], axis=-1), pred[1], pred[2], pred[3]]
        final_bboxes, final_labels, final_scores = decode_pred(pred, stride_grids)
        ap = calculate_ap_const(final_bboxes, final_labels, gt_boxes, gt_labels, labels)
        validation_progress.set_description("Validation | Average_Precision {:.4f}".format(ap))
        map.append(ap)

    map_res = tf.reduce_mean(map)

    return map_res





#%%
if __name__ == "__main__":
    os.makedirs("data_chkr", exist_ok=True)
    args = build_args()
    run = plugin_neptune(NEPTUNE_API_KEY, NEPTUNE_PROJECT, args)

    lambda_lst = build_lambda(args)
    datasets, labels, data_num = load_dataset(name=args.name, data_dir=args.data_dir)
    train_set, valid_set, test_set = build_dataset(datasets, args.batch_size, args.img_size)
    box_priors = load_box_prior(train_set, args.name, args.img_size, data_num)
    anchors, prior_grids, offset_grids, stride_grids = build_anchor_ops(args.img_size, box_priors)
    model = yolo_v3(args.img_size+[3], labels, offset_grids, prior_grids, fine_tunning=True)
    optimizer = build_optimizer(args.batch_size, data_num)

    best_map = 0
    for epoch in range(1, args.epochs+1):
        epoch_progress = tqdm(range(data_num//args.batch_size))
        for _ in epoch_progress:
            image, gt_boxes, gt_labels = next(train_set)
            true = build_target(anchors, gt_boxes, gt_labels, labels, args.img_size, stride_grids)
            loss, total_loss = forward_backward(image, true, model, optimizer, args.batch_size, lambda_lst)
            record_train_loss(run, loss, total_loss)
            epoch_progress.set_description(
                "Epoch {}/{} | yx {:.4f}, hw {:.4f}, obj {:.4f}, nobj {:.4f}, cls {:.4f}, total {:.4f}".format(
                    epoch,
                    args.epochs,
                    loss[0].numpy(),
                    loss[1].numpy(),
                    loss[2].numpy(),
                    loss[3].numpy(),
                    loss[4].numpy(),
                    total_loss.numpy(),
                )
            )
        map_res = validation(valid_set, model)

        run["validation/mAP"].log(map_res.numpy())

        if map_res.numpy() > best_map:
            best_map = map_res.numpy()
            ckpt_dir = f"model_ckpt/{dataset_name}/rpn_weights"
            rpn_model.save_weights(f"{ckpt_dir}/weights")
            ckpt = os.listdir(ckpt_dir)
            for i in range(len(ckpt)):
                run[f"{ckpt_dir}/{ckpt[i]}"].upload(f"{ckpt_dir}/{ckpt[i]}")


    datasets, labels, data_num = load_dataset(name="coco/2017", data_dir="D:/won/data/tfds")
    train_set, valid_set, test_set = build_dataset(datasets, 1, [416,416])
    box_priors = load_box_prior(train_set, "coco/2017", [416,416], data_num)
    anchors, prior_grids, offset_grids, stride_grids = build_anchor_ops([416,416], box_priors)
    model = yolo_v3([416,416,3], labels, offset_grids, prior_grids, fine_tunning=True)
    model.load_weights("C:/Users/USER/Documents/GitHub/YOLO/yolov3_org_weights/weights")
    

# %%