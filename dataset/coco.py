#%%
import fiftyone as fo
import fiftyone.zoo as foz

train_dataset = foz.load_zoo_dataset("coco-2017", split="train", dataset_dir=r"C:\won\data\COCO\train2017")
val_dataset = foz.load_zoo_dataset("coco-2017", split="validation", dataset_dir=r"C:\won\data\COCO\val2017")
test_dataset = foz.load_zoo_dataset("coco-2017", split="test", dataset_dir = r"C:\won\data\COCO\test2017")
# %%
