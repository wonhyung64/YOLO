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

dataset, labels = data_utils.fetch_dataset("coco17", "test", img_size)

dataset = dataset.map(lambda x, y, z: preprocessing_utils.preprocessing(x, y, z))
data_shapes = ([None, None, None], [None, None], [None])
padding_values = (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))
dataset = dataset.repeat().padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)
dataset = iter(dataset)

hyper_params["total_labels"] = len(labels)


box_prior = np.array([[10,13], [16,30], [33,23], [30,61], [62,45], [59,119], [116,90],  [156,198],  [373,326]], dtype = np.float32)
anchors1 = box_prior[6:9] # 13,13
anchors2 = box_prior[3:6] # 26, 26
anchors3 = box_prior[0:3] # 52, 52
anchors = [anchors3, anchors2, anchors1]

#%%
input_shape = (416, 416, 3)
yolo_model = model_utils.yolo_v3(input_shape, hyper_params, anchors)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

# @tf.function
def train_step(img, gt_boxes, gt_labels, box_prior, hyper_params):
    with tf.GradientTape(persistent=True) as tape:
        pred = yolo_model(img)
        true = target_utils.generate_target([gt_boxes, gt_labels], box_prior, hyper_params)

        ignore_mask = target_utils.generate_ignore_mask(true, pred)
        coord_tune = hyper_params["coord_tune"]
        noobj_tune = hyper_params["noobj_tune"]

        box_loss = loss_utils.box_loss(pred, true, coord_tune)
        obj_loss = loss_utils.obj_loss(pred, true)
        nobj_loss = loss_utils.nobj_loss(pred, true, ignore_mask, noobj_tune)
        cls_loss = loss_utils.cls_loss(pred, true)
        total_loss = box_loss + obj_loss + nobj_loss + cls_loss
    grads = tape.gradient(total_loss, yolo_model.trainable_weights)
    optimizer.apply_gradients(zip(grads, yolo_model.trainable_weights))
    return box_loss, obj_loss, nobj_loss, cls_loss, total_loss

# %%
atmp_dir = os.getcwd()
atmp_dir = utils.generate_save_dir(atmp_dir, hyper_params)

step = 0
progress_bar = tqdm(range(hyper_params['iters']))
progress_bar.set_description('iterations {}/{} | current loss ?'.format(step, hyper_params['iters']))
start_time = time.time()

for _ in progress_bar:
    img, gt_boxes, gt_labels = next(dataset)
    box_loss, obj_loss, nobj_loss, cls_loss, total_loss = train_step(img, gt_boxes, gt_labels, box_prior, hyper_params)

    step += 1

    progress_bar.set_description('iterations {}/{} | box_loss {:.4f}, obj_loss {:.4f}, nobj_loss {:.4f}, cls_loss {:.4f}, loss {:.4f}'.format(
        step, hyper_params['iters'], 
        box_loss.numpy(), obj_loss.numpy(), nobj_loss.numpy(), cls_loss.numpy(), total_loss.numpy()
    )) 
    
    if step % 500 == 0 : 
        print(progress_bar.set_description('iterations {}/{} | box_loss {:.4f}, obj_loss {:.4f}, nobj_loss {:.4f}, cls_loss {:.4f}, loss {:.4f}'.format(
            step, hyper_params['iters'], 
            box_loss.numpy(), obj_loss.numpy(), nobj_loss.numpy(), cls_loss.numpy(), total_loss.numpy()
        )))
    
    if step % 1000 == 0 : 
        yolo_model.save_weights(atmp_dir + '/yolo_weights/weights')
        print("Weights Saved")

print("Time taken: %.2fs" % (time.time() - start_time))
utils.save_dict_to_file(hyper_params, atmp_dir + '/hyper_params')

#%%test
hyper_params["batch_size"] = batch_size = 1

dataset, _ = data_utils.fetch_dataset("coco17", "test", img_size)

dataset = dataset.map(lambda x, y, z: preprocessing_utils.preprocessing(x, y, z))
dataset = dataset.repeat().padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)
dataset = iter(dataset)

input_shape = (416, 416, 3)
yolo_model = model_utils.yolo_v3(input_shape, hyper_params, anchors)
yolo_model.load_weights(atmp_dir + '/yolo_weights/weights')

total_time = []
mAP = []

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

mAP_res = "%.2f" % (tf.reduce_mean(mAP))
total_time_res = "%.2fms" % (tf.reduce_mean(total_time))

result = {"mAP" : mAP_res,
          "total_time" : total_time_res}

utils.save_dict_to_file(result, atmp_dir + "/result")