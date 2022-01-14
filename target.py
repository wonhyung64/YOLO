#%%
import numpy as np
import tensorflow as tf

#%%
def generate_target(inputs, box_prior, hyper_params):
    batch_size = hyper_params["batch_size"]
    img_size = hyper_params["img_size"]
    total_class = hyper_params["total_class"]

    y_13_lst, y_26_lst, y_52_lst = [], [], []
    for j in range(batch_size):
        gt_box = inputs[0][j] * img_size
        gt_label = inputs[1][j]

        gt_ctr = (gt_box[:,:2] + gt_box[:,2:])/2 # x y
        gt_size = gt_box[:,2:] - gt_box[:,:2] # w h

        y_true_13 = np.zeros((13, 13, 3, 5 + total_class), np.float32)
        y_true_26 = np.zeros((26, 26, 3, 5 + total_class), np.float32)
        y_true_52 = np.zeros((52, 52, 3, 5 + total_class), np.float32)
        y_true = [y_true_13, y_true_26, y_true_52]

        gt_size = tf.expand_dims(gt_size, 1)

        mins = tf.maximum(-gt_size / 2, -box_prior / 2)
        maxs = tf.minimum(gt_size / 2, box_prior / 2)
        whs = maxs - mins
        intersection = whs[...,0] * whs[...,1]
        union = gt_size[...,0] * gt_size[...,1]  + box_prior[...,0] * box_prior[...,1] - intersection

        iou_map = intersection / union

        best_iou_idx = tf.argmax(iou_map, axis=1)

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