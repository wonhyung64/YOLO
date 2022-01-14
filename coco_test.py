#%%
import time
import data
import tensorflow as tf

#%%
data_dir = r"C:\won\data\coco"
train, validation, test = data.download_dataset("coco17", data_dir)
data_iter = iter(train)

#%%
save_dir = r"D:\won\coco_tfrecord"
name = "test_100"

try_num = 0
start_time = time.time()
writer = tf.io.TFRecordWriter(f"{save_dir}/{name}.tfrecord".encode("utf-8"))
while try_num < 100:
    sample = next(data_iter)
    example = {"image":sample["image"]/255, "bbox":sample["objects"]["bbox"], "label":sample["objects"]["label"]}
    x = data.serialize_example(example)
    writer.write(x)
    try_num += 1
    print(try_num)
print("total time :", time.time() - start_time)
print("try_num :" ,try_num)