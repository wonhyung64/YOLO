#%%
import os
import time
import tensorflow as tf
import numpy as np

from tqdm import tqdm
import model_utils, data_utils, loss_utils, target_utils, utils, preprocessing_utils, postprocessing_utils, test_utils, utils

#%%
hyper_params = utils.get_hyper_params()

iters = hyper_params["iters"]
batch_size = hyper_params["batch_size"]
img_size = (hyper_params["img_size"], hyper_params["img_size"])
dataset_name = hyper_params["dataset_name"]

#%%
if dataset_name == "ship":
    import ship
    dataset, labels = ship.fetch_dataset(dataset_name, "train", img_size)
    dataset = dataset.map(lambda x, y, z, w: preprocessing_utils.preprocessing_ship(x, y, z, w))
else:
    import data_utils
    dataset, labels = data_utils.fetch_dataset(dataset_name, "train", img_size)
    dataset = dataset.map(lambda x, y, z: preprocessing_utils.preprocessing(x, y, z))
    # dataset = dataset.map(lambda x, y, z: target_utils.calculate_target(x, y, z))

dataset = dataset.shuffle(buffer_size=5050, reshuffle_each_iteration=True)
data_shapes = ([None, None, None], [None, None], [None])
padding_values = (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))
dataset = dataset.repeat().padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values, drop_remainder=True)
# dataset = dataset.repeat().batch(batch_size)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
dataset = iter(dataset)
#%%
box_prior = utils.get_box_prior()
box_prior = np.array(box_prior)
anchors1 = box_prior[6:9] # 13,13
anchors2 = box_prior[3:6] # 26, 26
anchors3 = box_prior[0:3] # 52, 52
anchors = [anchors1, anchors2, anchors3]

#%%
input_shape = (416, 416, 3)
yolo_model = model_utils.yolo_v3(input_shape, hyper_params)


optimizer = tf.keras.optimizers.Adam(learning_rate=hyper_params["lr"])

@tf.function
def train_step(img, true):
    with tf.GradientTape(persistent=True) as tape:
        yolo_outputs = yolo_model(img)
        box_loss, conf_loss, cls_loss = loss_utils.yolo_loss(yolo_outputs, true)
        # box_loss = loss_utils.box_loss_fn(pred, true, anchors, hyper_params)
        # box_loss = loss_utils.box_loss_fn(pred, true, hyper_params)
        # conf_loss = loss_utils.conf_loss_fn(pred, true, ignore_mask, hyper_params)
        # cls_loss = loss_utils.cls_loss_fn(pred, true)
        total_loss = box_loss + conf_loss + cls_loss
    grads = tape.gradient(total_loss, yolo_model.trainable_weights)
    optimizer.apply_gradients(zip(grads, yolo_model.trainable_weights))
    
    return box_loss, conf_loss, cls_loss, total_loss

# %%
atmp_dir = os.getcwd()
atmp_dir = utils.generate_save_dir(atmp_dir, hyper_params)

step = 0
progress_bar = tqdm(range(hyper_params['iters']))
progress_bar.set_description('iterations {}/{} | current loss ?'.format(step, hyper_params['iters']))
start_time = time.time()

for _ in progress_bar:
    img, gt_boxes, gt_labels = next(dataset)
    true = target_utils.generate_target(gt_boxes, gt_labels)
    box_loss, conf_loss, cls_loss, total_loss = train_step(img, true)
    print(box_loss, conf_loss, cls_loss, total_loss)

    step += 1

    progress_bar.set_description('iterations {}/{} | box_loss {:.4f}, conf_loss {:.4f}, cls_loss {:.4f}, loss {:.4f}'.format(
        step, hyper_params['iters'], 
        box_loss.numpy(), conf_loss.numpy(), cls_loss.numpy(), total_loss.numpy()
    )) 

    if step % 500 == 0 : 
        print(progress_bar.set_description('iterations {}/{} | box_loss {:.4f}, conf_loss {:.4f}, cls_loss {:.4f}, loss {:.4f}'.format(
            step, hyper_params['iters'], 
            box_loss.numpy(), conf_loss.numpy(), cls_loss.numpy(), total_loss.numpy()
        )))
    
    if step % 1000 == 0 : 
        yolo_model.save_weights(atmp_dir + '/yolo_weights/weights')
        print("Weights Saved")

print("Time taken: %.2fs" % (time.time() - start_time))
utils.save_dict_to_file(hyper_params, atmp_dir + '/hyper_params')

#%%test
if dataset_name == "ship":
    import ship
    dataset, labels = ship.fetch_dataset(dataset_name, "train", img_size, save_dir="/home1/wonhyung64")
    dataset = dataset.map(lambda x, y, z, w: preprocessing_utils.preprocessing_ship(x, y, z, w))
else:
    import data_utils
    dataset, labels = data_utils.fetch_dataset(dataset_name, "train", img_size)
    dataset = dataset.map(lambda x, y, z: preprocessing_utils.preprocessing(x, y, z))

dataset = dataset.repeat().batch(1)
dataset = iter(dataset)

input_shape = (416, 416, 3)
yolo_model = model_utils.yolo_v3(input_shape, hyper_params, anchors)
yolo_model.load_weights(atmp_dir + '/yolo_weights/weights')

total_time = []
mAP = []

img_num = 0
progress_bar = tqdm(range(hyper_params['attempts']))
for _ in progress_bar:
    img, gt_boxes, gt_labels = next(dataset)
    start_time = time.time()
    pred = yolo_model(img)

    final_bboxes, final_labels, final_scores = postprocessing_utils.Decode(pred, hyper_params)
    time_ = float(time.time() - start_time)*1000
    AP = test_utils.calculate_AP(final_bboxes, final_labels, gt_boxes, gt_labels, hyper_params)
    total_time.append(time_)
    mAP.append(AP)

    test_utils.draw_yolo_output(img, final_bboxes, labels, final_labels, final_scores, atmp_dir, img_num)
    img_num += 1

mAP_res = "%.2f" % (tf.reduce_mean(mAP))
total_time_res = "%.2fms" % (tf.reduce_mean(total_time))

result = {"mAP" : mAP_res,
          "total_time" : total_time_res}

utils.save_dict_to_file(result, atmp_dir + "/result")

#%%
