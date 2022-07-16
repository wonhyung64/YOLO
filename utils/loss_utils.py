import tensorflow as tf


@tf.function
def bce_fn(pred, true):
    bce_loss = -(
        true * tf.math.log(tf.clip_by_value(pred, 1e-9, 1.0))
        + (1 - true) * tf.math.log(tf.clip_by_value(1 - pred, 1e-9, 1.0))
    )

    return bce_loss


def focal_fn(pred, true, alpha=1.0, gamma=2.0):
    focal_loss = alpha * tf.pow(tf.abs(true - tf.sigmoid(pred)), gamma)

    return focal_loss


@tf.function
def loss_fn(pred, true, batch_size, lambda_lst):
    pred_yx, pred_hw, pred_obj, pred_cls = pred
    true_yx, true_hw, true_obj, true_nobj, true_cls = true
    batch_size = tf.constant(batch_size, dtype=tf.float32)
    lambda_yx, lambda_hw, lambda_obj, lambda_nobj, lambda_cls = lambda_lst

    yx_loss = (
        tf.reduce_sum(tf.square(pred_yx - true_yx) * true_obj) / batch_size * lambda_yx
    )
    hw_loss = (
        tf.reduce_sum(tf.square(pred_hw - true_hw) * true_obj) / batch_size * lambda_hw
    )
    obj_loss = (
        tf.reduce_sum(bce_fn(pred_obj, true_obj) * true_obj) / batch_size * lambda_obj
    )
    nobj_loss = (
        tf.reduce_sum(bce_fn(pred_obj, true_obj) * (true_nobj))
        / batch_size
        * lambda_nobj
    )
    cls_loss = (
        tf.reduce_sum(bce_fn(pred_cls, true_cls) * true_obj) / batch_size * lambda_cls
    )

    return yx_loss, hw_loss, obj_loss, nobj_loss, cls_loss


def build_lambda(args):
    lambda_dict = {
        "lambda_yx": args.lambda_yx,
        "lambda_hw": args.lambda_hw,
        "lambda_obj": args.lambda_obj,
        "lambda_nobj": args.lambda_nobj,
        "lambda_cls": args.lambda_cls,
    }

    lambda_lst = [
        tf.constant(lambda_value, dtype=tf.float32)
        for lambda_value in lambda_dict.values()
    ]

    return lambda_lst
