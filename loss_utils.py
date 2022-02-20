#%%
import tensorflow as tf
import model_utils, utils, bbox_utils
#%%
def box_loss_fn_tmp(pred, true, hyper_params):
    tune_param = tf.constant(hyper_params["coord_tune"])

    yx_pred, hw_pred = tf.split(pred[..., :4], [2, 2], axis=-1)
    yx_true, hw_true, obj_mask = tf.split(true[..., :5], [2, 2, 1], axis=-1)
    obj_mask = tf.squeeze(obj_mask, axis=-1)

    yx_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.square(yx_true - yx_pred), axis=-1) * obj_mask, axis=-1))
    hw_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.square(hw_true - hw_pred), axis=-1) * obj_mask, axis=-1))

    return (yx_loss + hw_loss) * tune_param

#%%
def conf_loss_fn(pred, true, ignore_mask, hyper_params):
    tune_param = hyper_params["noobj_tune"]
    focal = hyper_params["focal"]

    obj_pred , obj_true = pred[...,4], true[...,4]

    obj_loss_fn = bce_fn(obj_pred, obj_true)

    obj_loss = obj_loss_fn * obj_true

    nobj_loss = ignore_mask * (1-obj_true) * obj_loss_fn

    conf_loss = obj_loss + nobj_loss * tune_param

    if focal == True:
        focal_loss = focal_fn(obj_pred, obj_true)
        conf_loss *= focal_loss

    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=-1))

    return conf_loss

#%%
def cls_loss_fn(pred, true):
    cls_pred, cls_true = pred[..., 5:], true[..., 5:]
    obj_mask = true[...,4]

    cls_loss_fn = bce_fn(cls_pred, cls_true)
    cls_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(cls_loss_fn, axis=-1) * obj_mask, -1))

    return cls_loss
    
#%%
def bce_fn(pred, true):
    return -(true * tf.math.log(tf.clip_by_value(pred, 1e-9, 1.)) + (1-true) * tf.math.log(tf.clip_by_value(1 - pred, 1e-9, 1.)))

#%%
def focal_fn(pred, true, alpha=1.0, gamma=2.0):
    return alpha * tf.pow(tf.abs(true - tf.sigmoid(pred)), gamma)

#%%
def box_loss_fn(pred, true, anchors, hyper_params):
    batch_size = hyper_params["batch_size"]
    tune_param = tf.constant(hyper_params["coord_tune"])

    pred_13, pred_26, pred_52 = tf.split(pred[...,:5], [13*13*3, 26*26*3, 52*52*3], axis=1)
    true_13, true_26, true_52 = tf.split(true[...,:5], [13*13*3, 26*26*3, 52*52*3], axis=1)

    pred_13 = tf.reshape(pred_13, (batch_size, 13, 13, 3, 5))
    true_13 = tf.reshape(true_13, (batch_size, 13, 13, 3, 5))
    pred_13_hw = tf.math.log(tf.clip_by_value(pred_13[..., 2:4], 1e-9, 1e+9)) / anchors[0]
    true_13_hw = tf.math.log(tf.clip_by_value(true_13[..., 2:4], 1e-9, 1e+9)) / anchors[0]

    pred_13_yx = pred_13[..., :2]/32. - pred_13[..., :2] // 32.
    true_13_yx = true_13[..., :2]/32. - true_13[..., :2] // 32.

    loss_13_hw = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.square(true_13_hw - pred_13_hw), axis=-1) * true_13[...,-1], axis=[1,2,3]))
    loss_13_yx = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(bce_fn(pred_13_yx, true_13_yx), axis=-1) * true_13[...,-1], axis=[1,2,3]))
    loss_13 = loss_13_hw + loss_13_yx

    pred_26 = tf.reshape(pred_26, (batch_size, 26, 26, 3, 5))
    true_26 = tf.reshape(true_26, (batch_size, 26, 26, 3, 5))
    pred_26_hw = tf.math.log(tf.clip_by_value(pred_26[..., 2:4], 1e-9, 1e+9)) / anchors[1]
    true_26_hw = tf.math.log(tf.clip_by_value(true_26[..., 2:4], 1e-9, 1e+9)) / anchors[1]

    pred_26_yx = pred_26[..., :2]/16. - pred_26[..., :2] // 16.
    true_26_yx = true_26[..., :2]/16. - true_26[..., :2] // 16.

    loss_26_hw = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.square(true_26_hw - pred_26_hw), axis=-1) * true_26[...,-1], axis=[1,2,3]))
    loss_26_yx = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(bce_fn(pred_26_yx, true_26_yx), axis=-1) * true_26[...,-1], axis=[1,2,3]))
    loss_26 = loss_26_hw + loss_26_yx

    pred_52 = tf.reshape(pred_52, (batch_size, 52, 52, 3, 5))
    true_52 = tf.reshape(true_52, (batch_size, 52, 52, 3, 5))
    pred_52_hw = tf.math.log(tf.clip_by_value(pred_52[..., 2:4], 1e-9, 1e+9)) / anchors[2]
    true_52_hw = tf.math.log(tf.clip_by_value(true_52[..., 2:4], 1e-9, 1e+9)) / anchors[2]

    pred_52_yx = pred_52[..., :2]/8. - pred_52[..., :2] // 8.
    true_52_yx = true_52[..., :2]/8. - true_52[..., :2] // 8.

    loss_52_hw = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.square(true_52_hw - pred_52_hw), axis=-1) * true_52[...,-1], axis=[1,2,3]))
    loss_52_yx = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(bce_fn(pred_52_yx, true_52_yx), axis=-1) * true_52[...,-1], axis=[1,2,3]))
    loss_52 = loss_52_hw + loss_52_yx

    box_loss = (loss_13 + loss_26 + loss_52) * tune_param
    
    return box_loss

#%%
def yolo_loss(yolo_outputs, true):
    anchors = utils.get_box_prior()
    num_layers = len(anchors)//3

    hyper_params = utils.get_hyper_params()
    num_classes = hyper_params["total_labels"]

    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]

    input_shape = tf.cast(tf.shape(yolo_outputs[0])[1:3] * 32, tf.float32)

    grid_shapes = [tf.cast(tf.shape(yolo_outputs[l])[1:3], dtype=tf.float32) for l in range(num_layers)]

    total_yx_loss = tf.constant(0.)
    total_hw_loss = tf.constant(0.)
    total_conf_loss = tf.constant(0.)
    total_cls_loss = tf.constant(0.)

    m = tf.shape(yolo_outputs[0])[0]
    mf = tf.cast(m, tf.float32)

    for l in range(num_layers):
        object_mask = true[l][..., 4:5]
        true_class_probs = true[l][..., 5:]

        grid, raw_pred, pred_yx, pred_hw, _, _ = model_utils.yolo_head(yolo_outputs[l], tf.gather(anchors, anchor_mask[l]), num_classes, input_shape)

        pred_box = tf.concat([pred_yx, pred_hw], axis=-1)
    # true yxhw : 이미지 안에서 0~1
        raw_true_yx = true[l][..., :2] * grid_shapes[l][..., -1] - grid # 그리드에서 0~1
        raw_true_yx = tf.where(object_mask==tf.constant([1.]), raw_true_yx, tf.zeros_like(raw_true_yx)) 
        raw_true_hw = tf.math.log(true[l][..., 2:4]) / tf.gather(anchors, anchor_mask[l]) * input_shape[..., -1] # ?
        raw_true_hw = tf.where(object_mask==tf.constant([1.]), raw_true_hw, tf.zeros_like(raw_true_hw))
        
        box_loss_scale = 2 - true[l][..., 2:3] * true[l][..., 3:4]

        ignore_mask = tf.TensorArray(tf.float32, 1, dynamic_size=True)
        object_mask_bool = tf.cast(object_mask, "bool")

        def loop_cond(b, ignore_mask):
            return tf.less(b, tf.cast(m, tf.int32))
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou = bbox_utils.box_iou(pred_box[b], true_box)
            best_iou = tf.reduce_max(iou, axis=-1)

            ignore_mask = ignore_mask.write(b, tf.cast(best_iou < 0.5, dtype=tf.float32))
            return b+1, ignore_mask
        _, ignore_mask = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=[0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = tf.expand_dims(ignore_mask, -1)

        # yx_loss = object_mask * box_loss_scale * bce_fn(tf.sigmoid(raw_pred[...,:2]), raw_true_yx)
        yx_loss = object_mask * tf.square(raw_true_yx - tf.sigmoid(raw_pred[...,:2]))
        yx_loss = tf.reduce_sum(yx_loss) / mf
        # hw_loss = object_mask * box_loss_scale * 0.5 * tf.square(raw_true_hw - raw_pred[...,2:4])
        hw_loss = object_mask * 0.5 * tf.square(raw_true_hw - raw_pred[...,2:4])
        hw_loss = tf.reduce_sum(hw_loss) / mf
        total_yx_loss += yx_loss
        total_hw_loss += hw_loss

        obj_loss = object_mask * bce_fn(tf.sigmoid(raw_pred[..., 4:5]), object_mask) 
        noobj_loss = (1 - object_mask) * bce_fn(tf.sigmoid(raw_pred[..., 4:5]), object_mask) * ignore_mask
        conf_loss = tf.reduce_sum(obj_loss + noobj_loss) / mf
        total_conf_loss += conf_loss

        cls_loss = object_mask * bce_fn(tf.sigmoid(raw_pred[..., 5:]), true_class_probs)
        cls_loss = tf.reduce_sum(cls_loss) / mf
        total_cls_loss += cls_loss

    return total_yx_loss, total_hw_loss, total_conf_loss, total_cls_loss