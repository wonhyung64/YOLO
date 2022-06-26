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
    NEPTUNE_API_KEY,
    NEPTUNE_PROJECT,
)
    
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
        
