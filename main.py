import os
import time
import neptune.new as neptune
from tqdm import tqdm
import tensorflow as tf
from utils import (
    gpu_memory_growth,
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
    decode_pred,
    draw_output,
    calculate_ap_const,
    record_result,
    NEPTUNE_API_KEY,
    NEPTUNE_PROJECT,
)


def train(
    run,
    args,
    train_num,
    train_set,
    valid_set,
    labels,
    anchors,
    stride_grids,
    model,
    optimizer,
    lambda_lst,
    weights_dir,
    strategy
    ):
    best_mean_ap = 0
    start_time = time.time()
    for epoch in range(args.epochs):
        epoch_progress = tqdm(range(train_num//args.batch_size))
        for _ in epoch_progress:
            image, gt_boxes, gt_labels = next(train_set)
            true = strategy.run(build_target, args=(anchors, gt_boxes, gt_labels, labels, args.img_size, stride_grids))
            loss = strategy.run(forward_backward, args=(image, true, model, optimizer, args.batch_size, lambda_lst))
            loss = [strategy.reduce(tf.distribute.ReduceOp.MEAN, loss[i], axis=None) for i in range(len(loss))]
            total_loss = tf.reduce_sum(loss)
            record_train_loss(run, loss, total_loss)
            epoch_progress.set_description(
                "Epoch {}/{} | yx {:.4f}, hw {:.4f}, obj {:.4f}, nobj {:.4f}, cls {:.4f}, total {:.4f}".format(
                    epoch+1,
                    args.epochs,
                    loss[0].numpy(),
                    loss[1].numpy(),
                    loss[2].numpy(),
                    loss[3].numpy(),
                    loss[4].numpy(),
                    total_loss.numpy(),
                )
            )
        mean_ap = validation(valid_set, stride_grids, model, labels, strategy)

        run["validation/mAP"].log(mean_ap.numpy())

        if mean_ap.numpy() > best_mean_ap:
            best_mean_ap = mean_ap.numpy()
            model.save_weights(weights_dir)

    train_time = time.time() - start_time

    return train_time


def validation(valid_set, stride_grids, model, labels, strategy):
    aps = []
    validation_progress = tqdm(range(100))
    for _ in validation_progress:
        image, gt_boxes, gt_labels = next(valid_set)
        pred = model(image)
        final_bboxes, final_labels, final_scores = decode_pred(pred, stride_grids)
        ap = calculate_ap_const(final_bboxes, final_labels, gt_boxes, gt_labels, labels)
        validation_progress.set_description("Validation | Average_Precision {:.4f}".format(ap))
        aps.append(ap)

    mean_ap = tf.reduce_mean(aps)

    return mean_ap


def test(run, test_num, test_set, model, stride_grids, labels):
    test_times = []
    aps = []
    test_progress = tqdm(range(test_num))
    for step in test_progress:
        image, gt_boxes, gt_labels = next(test_set)
        start_time = time.time()
        pred = model(image)
        final_bboxes, final_labels, final_scores = decode_pred(pred, stride_grids)
        test_time = time.time() - start_time
        ap = calculate_ap_const(final_bboxes, final_labels, gt_boxes, gt_labels, labels)
        test_progress.set_description("Test | Average_Precision {:.4f}".format(ap))
        aps.append(ap)
        test_times.append(test_time)

        if step <= 20 == 0:
            run["outputs"].log(
                neptune.types.File.as_image(
                    draw_output(image, final_bboxes, final_labels, final_scores, labels)
                    )
            )

    mean_ap = tf.reduce_mean(aps)
    mean_test_time = tf.reduce_mean(test_times)

    return mean_ap, mean_test_time


def main():
    args = build_args()
    os.makedirs("./data_chkr", exist_ok=True)
    run = plugin_neptune(NEPTUNE_API_KEY, NEPTUNE_PROJECT, args)

    experiment_name = run.get_run_url().split("/")[-1].replace("-", "_")
    experiment_dir = "./model_weights/experiment"
    os.makedirs(experiment_dir, exist_ok=True)
    weights_dir = f"{experiment_dir}/{experiment_name}.h5"

    strategy = tf.distribute.MirroredStrategy()
    lambda_lst = build_lambda(args)
    datasets, labels, train_num, test_num = load_dataset(name=args.name, data_dir=args.data_dir)
    train_set, valid_set, test_set = build_dataset(datasets, args.batch_size, args.img_size, strategy)
    box_priors = load_box_prior(train_set, args.name, args.img_size, train_num)
    anchors, prior_grids, offset_grids, stride_grids = build_anchor_ops(args.img_size, box_priors)

    with strategy.scope():
        model = yolo_v3(args.img_size+[3], labels, offset_grids, prior_grids, args.data_dir, fine_tunning=True)
        optimizer = build_optimizer(args.batch_size, train_num)
    with strategy.scope():
        train_time = train(run, args, train_num, train_set, valid_set, labels, anchors, stride_grids, model, optimizer, lambda_lst, weights_dir, strategy)

    model.load_weights(weights_dir)
    mean_ap, mean_test_time = test(run, test_num, test_set, model, stride_grids, labels)

    record_result(run, weights_dir, mean_ap, train_time, mean_test_time)


if __name__ == "__main__":
    main()
