#%%
import os
from utils import (
    load_dataset,
    build_dataset,
    load_box_prior,
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
    box_prior = load_box_prior(train_set, name, img_size, data_num)

#%%


# %%
