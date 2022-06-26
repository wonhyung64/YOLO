#%%
import os
import argparse
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
)

try: import neptune.new as neptune
except:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "neptune-client"])
    import neptune.new as neptune

from utils.variable import (
    NEPTUNE_API_KEY,
    NEPTUNE_PROJECT,
)

def plugin_neptune(NEPTUNE_API_KEY, NEPTUNE_PROJECT, hyper_params):
    os.environ["NEPTUNE_API_TOKEN"] = NEPTUNE_API_KEY

    run = neptune.init(project=NEPTUNE_PROJECT, 
                        api_token=NEPTUNE_PROJECT,
                    mode="offline")

    return run
    
#%%
if __name__ == "__main__":
    os.makedirs("data_chkr", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = str)
    parser.add_argument("--dataset-name", type = str)
    
    args = parser.parse_args()
    print(args.base_model)
    print(args.dataset_name)


    hyper_params_dict = get_hyper_params()

    epochs = 160
    img_size = [416, 416]
    data_dir = "D:/won/data/tfds"
    batch_size = 16
    name = "voc/2007"
    lambda_dict = {
        "lambda_yx": 1e-1,
        "lambda_hw": 1e-5,
        "lambda_obj": 1e-1,
        "lambda_nobj": 1e-4,
        "lambda_cls": 1e-3,
        }

    lambda_lst = [tf.constant(lambda_value, dtype=tf.float32) for lambda_value in lambda_dict.values()]
    datasets, labels, data_num = load_dataset(name=name, data_dir=data_dir)
    train_set, valid_set, test_set = build_dataset(datasets, batch_size, img_size)
    box_priors = load_box_prior(train_set, name, img_size, data_num)
    anchors, prior_grids, offset_grids, stride_grids = build_anchor_ops(img_size, box_priors)
    model = yolo_v3(img_size+[3], labels, offset_grids, prior_grids, fine_tunning=True)
    optimizer = build_optimizer(batch_size, data_num)

    for epoch in range(1, epochs+1):
        epoch_progress = tqdm(range(data_num//batch_size))
        for _ in epoch_progress:
            image, gt_boxes, gt_labels = next(train_set)
            true = build_target(anchors, gt_boxes, gt_labels, labels, img_size, stride_grids)
            loss = forward_backward(image, true, model, optimizer, batch_size, lambda_lst)
            epoch_progress.set_description(
                "Epoch {}/{} | yx {:.4f}, hw {:.4f}, obj {:.4f}, nobj {:.4f}, cls {:.4f}, total {:.4f}".format(
                    epoch,
                    epochs,
                    loss[0].numpy(),
                    loss[1].numpy(),
                    loss[2].numpy(),
                    loss[3].numpy(),
                    loss[4].numpy(),
                    tf.reduce_sum(loss).numpy(),
                )
            )
        
