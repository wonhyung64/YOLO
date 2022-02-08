#%%
import tensorflow as tf
#%%
def box_loss(pred, true, tune_param):
    tune_param = tf.constant(tune_param)

    xy_pred = pred[...,:2]
    wh_pred = pred[...,2:4]
    xy_true = true[...,:2]
    wh_true = true[...,2:4]
    obj_mask = true[...,4]
    
    xy_loss = tf.reduce_mean(tf.clip_by_value(tf.square(xy_pred - xy_true), 1e-9, 1e+9), axis=-1)
    wh_loss = tf.reduce_mean(tf.clip_by_value(tf.square(wh_pred - wh_true), 1e-9, 1e+9), axis=-1)

    return tf.reduce_mean((xy_loss + wh_loss) * obj_mask) * tune_param

#%%
def obj_loss(pred, true):
    obj_pred = tf.clip_by_value(pred[..., 4], 1e-9, 1e+9)
    obj_true = true[..., 4]
    return -tf.reduce_mean(tf.reduce_sum((obj_true * tf.math.log(obj_pred) + (1 - obj_true) * tf.math.log(tf.clip_by_value(1 - obj_pred, 1e-9, 1e+9))) * obj_true, axis=1))

#%%
def nobj_loss(pred, true, ignore_mask, tune_param):
    tune_param = tf.constant(tune_param)

    obj_pred = tf.clip_by_value(pred[...,4], 1e-9, 1e+9)
    obj_true = true[..., 4]
    return -tf.reduce_mean(tf.reduce_sum((obj_true * tf.math.log(obj_pred) + ((1 - obj_true) * tf.math.log(tf.clip_by_value(1 - obj_pred, 1e-9, 1e+9)))) * ((1-obj_true) * ignore_mask), axis=1)) * tune_param

#%%
def cls_loss(pred, true):
    cls_pred = tf.clip_by_value(pred[...,5:], 1e-9, 1e+9)
    cls_true = true[..., 5:]
    obj_mask = true[..., 4]
    return -tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(cls_true * tf.math.log(cls_pred) + (1 - cls_true) * tf.math.log(tf.clip_by_value(1 - cls_pred, 1e-9, 1e+9)), axis=-1) * obj_mask, axis=1))
#%%



'''
obj loss, noobj loss, cls loss => cls sum => grid sum => batch mean
box loss tune param 으로만 loss 크기 수정
'''
