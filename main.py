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
)

#%%
if __name__ == "__main__":
    os.makedirs("data_chkr", exist_ok=True)
    data_dir = "D:/won/data/tfds"
    img_size = [416, 416]
    batch_size = 16
    name = "coco/2017"
    epochs = 160
    lambda_dict = {
        "lambda_yx": 1e-1,
        "lambda_hw": 1e-5,
        "lambda_obj": 1e-1,
        "lambda_nobj": 1e-4,
        "lambda_cls": 1e-3,
        }


    datasets, labels, data_num = load_dataset(name=name, data_dir=data_dir)
    train_set, valid_set, test_set = build_dataset(datasets, batch_size, img_size)
    box_priors = load_box_prior(train_set, name, img_size, data_num)
    anchors, prior_grids, offset_grids, stride_grids = build_anchor_ops(img_size, box_priors)
    model = yolo_v3(img_size+[3], labels, offset_grids, prior_grids, fine_tunning=True)
    optimizer = build_optimizer(batch_size, data_num)

    for _ in tqdm(range(data_num)):
        image, gt_boxes, gt_labels = next(train_set)
        true = build_target(anchors, gt_boxes, gt_labels, labels, img_size, stride_grids)
        loss = forward_backward(image, true, model, optimizer, batch_size, lambda_dict.values())
        print(loss)
        
        if _ == 100: break 
# %%
