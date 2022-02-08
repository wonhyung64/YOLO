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

if dataset_name == "ship":
    import ship
    dataset, labels = ship.fetch_dataset(dataset_name, "train", img_size)
    dataset = dataset.map(lambda x, y, z, w: preprocessing_utils.preprocessing_ship(x, y, z, w))
else:
    import data_utils
    dataset, labels = data_utils.fetch_dataset(dataset_name, "train", img_size)
    dataset = dataset.map(lambda x, y, z: preprocessing_utils.preprocessing(x, y, z))
    dataset = dataset.map(lambda x, y, z: target_utils.calculate_target(x, y, z))

data_shapes = ([None, None, None], [None, None])
dataset = dataset.repeat().padded_batch(batch_size)
dataset = iter(dataset)

box_prior = utils.get_box_prior()
anchors1 = box_prior[6:9] # 13,13
anchors2 = box_prior[3:6] # 26, 26
anchors3 = box_prior[0:3] # 52, 52
anchors = [anchors3, anchors2, anchors1]

#%%
input_shape = (416, 416, 3)
yolo_model = model_utils.yolo_v3(input_shape, hyper_params, anchors)

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

# @tf.function
def train_step(img, gt_boxes, gt_labels, box_prior, hyper_params):
    with tf.GradientTape(persistent=True) as tape:
        pred = yolo_model(img)

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
hyper_params["iters"] = 1004
atmp_dir = os.getcwd()
atmp_dir = utils.generate_save_dir(atmp_dir, hyper_params)

step = 0
progress_bar = tqdm(range(hyper_params['iters']))
# progress_bar.set_description('iterations {}/{} | current loss ?'.format(step, hyper_params['iters']))
start_time = time.time()
img, true = next(dataset)

for _ in progress_bar:
    box_loss, obj_loss, nobj_loss, cls_loss, total_loss = train_step(img, gt_boxes, gt_labels, box_prior, hyper_params)
    # if total_loss.dtype != tf.float32 :  break

    step += 1

    # progress_bar.set_description('iterations {}/{} | box_loss {:.4f}, obj_loss {:.4f}, nobj_loss {:.4f}, cls_loss {:.4f}, loss {:.4f}'.format(
    #     step, hyper_params['iters'], 
    #     box_loss.numpy(), obj_loss.numpy(), nobj_loss.numpy(), cls_loss.numpy(), total_loss.numpy()
    # )) 
    
    print('iterations {}/{} | box_loss {:.4f}, obj_loss {:.4f}, nobj_loss {:.4f}, cls_loss {:.4f}, loss {:.4f}'.format(
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
if dataset_name == "ship":
    import ship
    dataset, labels = ship.fetch_dataset(dataset_name, "train", img_size, save_dir="/home1/wonhyung64")
    dataset = dataset.map(lambda x, y, z, w: preprocessing_utils.preprocessing_ship(x, y, z, w))
else:
    import data_utils
    dataset, labels = data_utils.fetch_dataset(dataset_name, "train", img_size, save_dir="/home1/wonhyung64")
    dataset = dataset.map(lambda x, y, z: preprocessing_utils.preprocessing(x, y, z))

dataset = dataset.repeat().padded_batch(1)
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
# %%

batch_size = hyper_params["batch_size"]
img_size = hyper_params["img_size"]
total_class = hyper_params["total_labels"]

gt_boxes = gt_boxes * img_size

y_13_lst, y_26_lst, y_52_lst = [], [], []

box_prior = tf.cast([[10,13], [16,30], [33,23], [30,61], [62,45], [59,119], [116,90],  [156,198],  [373,326]], dtype = np.float32)


gt_ctr = (gt_boxes[...,:2] + gt_boxes[...,2:])/2 # y x
gt_size = gt_boxes[...,2:] - gt_boxes[...,:2] # h w
gt_size = tf.expand_dims(gt_size, 1)


mins = tf.maximum(-gt_size / 2, -box_prior / 2)
maxs = tf.minimum(gt_size / 2, box_prior / 2)
whs = maxs - mins
intersection = whs[...,0] * whs[...,1]
union = gt_size[...,0] * gt_size[...,1]  + box_prior[...,0] * box_prior[...,1] - intersection

iou_map = intersection / union

best_iou_idx = tf.argmax(iou_map, axis=1) # box_prior idx(0~8)

anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]

ratio_dict = {1.:8., 2.:16., 3.:32.} # size_rank : strides

y_true_13 = tf.zeros((13, 13, 3, 5 + total_class), tf.float32)
y_true_26 = tf.zeros((26, 26, 3, 5 + total_class), tf.float32)
y_true_52 = tf.zeros((52, 52, 3, 5 + total_class), tf.float32)
y_true = [y_true_13, y_true_26, y_true_52]

gt_size = tf.squeeze(gt_size, axis=1)

for i, idx in enumerate(best_iou_idx):

    feature_map_group = 2 - idx // 3 # (0~2) 
    ratio = ratio_dict[tf.math.ceil((idx + 1) / 3).numpy()]
    y = tf.cast(tf.math.floor(gt_ctr[i, 0] / ratio), tf.int32).numpy()
    x = tf.cast(tf.math.floor(gt_ctr[i, 1] / ratio), tf.int32).numpy()
    k = anchors_mask[feature_map_group].index(idx) # box_idx in grid
    c = gt_labels[i].numpy()

    y_true[feature_map_group][y, x, k, :2] = gt_ctr[i]      # y, x

    y_true[feature_map_group] = tf.tensor_scatter_nd_update(y_true[feature_map_group], [[y,x,k,0], [y,x,k,1]], gt_ctr[i])

    y_true[feature_map_group] = tf.tensor_scatter_nd_update(y_true[feature_map_group], [[y,x,k,2], [y,x,k,3]], gt_size[i])
    
    y_true[feature_map_group] = tf.tensor_scatter_nd_update(y_true[feature_map_group], [[y,x,k,4]], tf.constant([1.]))

    y_true[feature_map_group] = tf.tensor_scatter_nd_update(y_true[feature_map_group], [[y,x,k,5+c]], tf.constant([1.]))
    
    y_true[feature_map_group][y,x,k]


y_true[feature_map_group][y,x,k,4]
y_true[feature_map_group][y,x,k,2:4]
    y_true[feature_map_group][y, x, k, 2:4] = gt_size[i]    # h, w
    y_true[feature_map_group][y, x, k, 4] = 1.              # obj (0 ,1)
    y_true[feature_map_group][y, x, k, 5+c] = 1.            # cls (one-hot)

y_true = tf.concat([tf.reshape(tmp, (tmp.shape[0] * tmp.shape[1], tmp.shape[2], tmp.shape[3])) for tmp in y_true], axis=0)
y_true = tf.reshape(y_true, (y_true.shape[0] * y_true.shape[1], y_true.shape[2]))
img, gt_boxes, gt_labels = next(iter(dataset))