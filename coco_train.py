#%%
import time
import csv
import data
import tensorflow as tf

# %%
file_dir = r"C:\Users\USER\Documents\GitHub\YOLO\coco_train_lst.txt"
f = open(file_dir, "rt", encoding='utf-8')
rdr = csv.reader(f, delimiter=",")
file_lst_ = list(rdr)[0]
file_lst = ["0"*(12 - len(file_lst_[i][:-5].lstrip())) + (file_lst_[i][:-5]).lstrip() + ".jpg" for i in range(len(file_lst_))]

# %%
data_dir = r"C:\won\data\coco"
train, validation, test = data.download_dataset("coco17", data_dir)
data_iter = iter(train)

# %%
save_dir = r"D:\won\coco_tfrecord"
name = "train_10000"

try_num = 0
start_time = time.time()
writer = tf.io.TFRecordWriter(f"{save_dir}/{name}.tfrecord".encode("utf-8"))
while len(file_lst) != 0:
    sample = next(data_iter)
    filename = str(sample["image/filename"].numpy())[2:-1]
    if filename in file_lst : 
        example = {"image":sample["image"]/255, "bbox":sample["objects"]["bbox"], "label":sample["objects"]["label"]}
        x = data.serialize_example(example)
        writer.write(x)
        try_num += 1
        print(try_num)
        file_lst.remove(filename)
print("total time :", time.time() - start_time)
print("try_num :" ,try_num)

# %%