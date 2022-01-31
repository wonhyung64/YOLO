#%%
import os
import time
import tensorflow as tf
import numpy as np

from tqdm import tqdm
import model_utils, data_utils, loss_utils, target_utils, utils, preprocessing_utils, postprocessing_utils, test_utils

#%%
hyper_params = utils.get_hyper_params()

iters = hyper_params["iters"]
batch_size = hyper_params["batch_size"]
img_size = (hyper_params["img_size"], hyper_params["img_size"])

dataset, labels = data_utils.fetch_dataset("coco17", "train", img_size)

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

        batch_size = hyper_params["batch_size"]
        img_size = hyper_params["img_size"]
        total_labels = hyper_params["total_labels"]

        y_13_lst, y_26_lst, y_52_lst = [], [], []
        for j in range(batch_size):
            j=0
            gt_box = gt_boxes[0][j] * img_size[0]
            gt_label = gt_labels[1][j]

            gt_ctr = (gt_box[:2] + gt_box[2:])/2 # x y
            gt_size = gt_box[2:] - gt_box[:2] # w h

            y_true_13 = np.zeros((13, 13, 3, 5 + total_labels), np.float32)
            y_true_26 = np.zeros((26, 26, 3, 5 + total_labels), np.float32)
            y_true_52 = np.zeros((52, 52, 3, 5 + total_labels), np.float32)
            y_true = [y_true_13, y_true_26, y_true_52]

            gt_size = tf.expand_dims(gt_size, 0)

            mins = tf.maximum(-gt_size / 2, -box_prior / 2)
            maxs = tf.minimum(gt_size / 2, box_prior / 2)
            whs = maxs - mins
            intersection = whs[...,0] * whs[...,1]
            union = gt_size[...,0] * gt_size[...,1]  + box_prior[...,0] * box_prior[...,1] - intersection

            iou_map = intersection / union

            best_iou_idx = tf.argmax(iou_map, axis=0)

            anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]

            ratio_dict = {1.:8., 2.:16., 3.:32.}
            for i, idx in enumerate(best_iou_idx):
                feature_map_group = 2 - idx // 3
                ratio = ratio_dict[np.ceil((idx + 1) / 3)]
                x = int(np.floor(gt_ctr[i, 0] / ratio))
                y = int(np.floor(gt_ctr[i, 1] / ratio))
                k = anchors_mask[feature_map_group].index(idx)
                c = gt_label[i]

                y_true[feature_map_group][y, x, k, :2] = gt_ctr[i]      # x, y
                y_true[feature_map_group][y, x, k, 2:4] = gt_size[i]    # w, h
                y_true[feature_map_group][y, x, k, 4] = 1.              # obj (0 ,1)
                y_true[feature_map_group][y, x, k, 5+c] = 1.            # cls (one-hot)

            y_13_lst.append(tf.reshape(y_true[0], (1, 13, 13, 3, 5 + total_class)))
            y_26_lst.append(tf.reshape(y_true[1], (1, 26, 26, 3, 5 + total_class)))
            y_52_lst.append(tf.reshape(y_true[2], (1, 52, 52, 3, 5 + total_class)))

        y_true = [tf.concat(y_13_lst, axis=0), tf.concat(y_26_lst, axis=0), tf.concat(y_52_lst, axis=0)]

        true_1 = tf.reshape(y_true[0], (y_true[0].shape[0], y_true[0].shape[1] * y_true[0].shape[2] * y_true[0].shape[3], -1))
        true_2 = tf.reshape(y_true[1], (y_true[1].shape[0], y_true[1].shape[1] * y_true[1].shape[2] * y_true[1].shape[3], -1))
        true_3 = tf.reshape(y_true[2], (y_true[2].shape[0], y_true[2].shape[1] * y_true[2].shape[2] * y_true[2].shape[3], -1))

        return tf.concat([true_1, true_2, true_3], axis=1)

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