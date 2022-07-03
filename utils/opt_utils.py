import sys
import subprocess
import tensorflow as tf
try: import tensorflow_addons as tfa
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow-addons"])
    import tensorflow_addons as tfa

from .loss_utils import loss_fn


def build_optimizer(batch_size, data_num):
    boundaries = [data_num // batch_size * epoch for epoch in (10, 60, 90)]
    # values = [1e-3, 1e-4, 1e-5, 1e-6]
    values = [1e-4, 1e-5, 1e-6, 1e-7]
    lr_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values
    )
    # optimizer = tfa.optimizers.SGDW(learning_rate=lr_fn, weight_decay=tf.constant(0.0005), momentum=tf.constant(0.9))
    optimizer = tfa.optimizers.AdamW(learning_rate=lr_fn, weight_decay=tf.constant(0.0005))

    return optimizer


@tf.function
def forward_backward(image, true, model, optimizer, batch_size, lambda_lst):
    with tf.GradientTape(persistent=True) as tape:
        pred = model(image)
        loss = loss_fn(pred=pred, true=true, batch_size=batch_size, lambda_lst=lambda_lst)
        total_loss = tf.reduce_sum(loss)

    grads = tape.gradient(total_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return loss
