#%%
from utils import (
    load_dataset,
    build_dataset,
)
#%%
data_dir = "D:/won/data/tfds"
img_size = (416, 416)
batch_size=4
name = "voc/2007"

datasets, labels, data_num = load_dataset(name=name, data_dir=data_dir)
train_set, valid_set, test_set = build_dataset(datasets, data_num, batch_size, img_size)

next(train_set)

# %%
