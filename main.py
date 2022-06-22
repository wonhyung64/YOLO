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
    loss_fn,
)

#%%
if __name__ == "__main__":
    os.makedirs("data_chkr", exist_ok=True)
    data_dir = "D:/won/data/tfds"
    img_size = (416, 416)
    batch_size = 4
    name = "coco/2017"

    datasets, labels, data_num = load_dataset(name=name, data_dir=data_dir)
    train_set, valid_set, test_set = build_dataset(datasets, batch_size, img_size)
    box_priors = load_box_prior(train_set, name, img_size, data_num)
    anchors, prior_grids, offset_grids, stride_grids = build_anchor_ops(img_size, box_priors)
    model = yolo_v3((416,416,3), labels, offset_grids, prior_grids, fine_tunning=True)

    for _ in tqdm(range(data_num)):
        image, gt_boxes, gt_labels = next(train_set)
        pred_yx, pred_hw, pred_obj, pred_cls = model(image)
        true_yx, true_hw, true_obj, true_nobj, true_cls = build_target(anchors, gt_boxes, gt_labels, labels, img_size, stride_grids)
        yx_loss, hw_loss, obj_loss, nobj_loss, cls_loss = loss_fn(
            pred=[pred_yx, pred_hw, pred_obj, pred_cls],
            true=[true_yx, true_hw, true_obj, true_nobj, true_cls],
            batch_size=batch_size
        )
        print(yx_loss)
        print(hw_loss)
        print(obj_loss)
        print(nobj_loss)
        print(cls_loss)
        
        if _ == 100: break 
# %%
