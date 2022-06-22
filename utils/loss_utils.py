#%%
import tensorflow as tf

def bce_fn(pred, true):
    bce_loss = -(true * tf.math.log(tf.clip_by_value(pred, 1e-9, 1.)) + (1-true) * tf.math.log(tf.clip_by_value(1 - pred, 1e-9, 1.)))

    return bce_loss


def focal_fn(pred, true, alpha=1.0, gamma=2.0):
    focal_loss = alpha * tf.pow(tf.abs(true - tf.sigmoid(pred)), gamma)

    return focal_loss


def loss_fn(pred, true, batch_size, lambda_reg=1., lambda_nobj=1.):
    pred_yx, pred_hw, pred_obj, pred_cls = pred
    true_yx, true_hw, true_obj, true_nobj, true_cls = true

    yx_loss = tf.reduce_sum(tf.square(pred_yx, true_yx) * true_obj) / batch_size * lambda_reg
    hw_loss = tf.reduce_sum(tf.square(pred_hw, true_hw) * true_obj) / batch_size * lambda_reg
    obj_loss = tf.reduce_sum(bce_fn(pred_obj, true_obj) * true_obj) / batch_size
    nobj_loss = tf.reduce_sum(bce_fn(1-pred_obj, true_nobj) * (true_nobj)) / batch_size * lambda_nobj
    cls_loss = tf.reduce_sum(bce_fn(pred_cls, true_cls) * true_obj) / batch_size

    return yx_loss, hw_loss, obj_loss, nobj_loss, cls_loss
