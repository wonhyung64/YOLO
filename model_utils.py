#%%
import os
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, GlobalAveragePooling2D, Dense, Add, BatchNormalization, LeakyReLU, Input, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
#%%
class conv_block(Layer):
    def __init__(self, filters, kernel_size, strides=1, batch_norm=True, **kwargs):
        super(conv_block, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.batch_norm = batch_norm

    def build(self, input_shape):
        super(conv_block, self).build(input_shape)
        self.cv = Conv2D(self.filters, self.kernel_size, self.strides, padding=("same" if self.strides==1 else "valid"))
        self.bn = BatchNormalization()
        self.ac = LeakyReLU(alpha=0.1)

    def call(self, inputs):
        tensor = inputs
        if self.strides > 1:
            pad_total = self.kernel_size - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            tensor = tf.pad(tensor, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        tensor = self.cv(tensor)
        if self.batch_norm==True:
            tensor = self.bn(tensor)
            tensor = self.ac(tensor)
        return tensor

#%%
def res_block(inputs, num):
    for _ in range(num):
        x = conv_block(inputs.shape[-1]//2, 1)(inputs)
        x = conv_block(inputs.shape[-1], 3)(x)
        x = Add()([inputs, x])
    return x

#%%
def DarkNet53(include_top=True, input_shape=(None, None, 3)):

    input_x = Input(shape=input_shape)
    x = conv_block(32, 3, name="conv1")(input_x)
    x = conv_block(64, 3, 2, name="conv2")(x)
    x = res_block(x, 1)
    x = conv_block(128, 3, 2, name="conv3")(x)
    x = res_block(x, 2)
    x = conv_block(256, 3, 2, name="conv4")(x)
    x = c3 = res_block(x, 8) # 52
    x = conv_block(512, 3, 2, name="conv5")(x)
    x = c2 = res_block(x, 8) # 26
    x = conv_block(1024, 3, 2, name="conv6")(x)
    x = c1 = res_block(x, 4) # 13
    if include_top == False:
        return Model(inputs=input_x, outputs=[c3, c2, c1])
    x = GlobalAveragePooling2D(name="avgpooling")(x)
    output_x = Dense(20, activation="softmax", name="fc")(x)
    return Model(inputs=input_x, outputs=output_x)

#%%
def generate_grid_cell(x_grid_size, y_grid_size):
    grid_x = tf.range(x_grid_size, dtype=tf.int32)
    grid_y = tf.range(y_grid_size, dtype=tf.int32)
    grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
    flat_grid_x = tf.reshape(grid_x, (-1, 1))
    flat_grid_y = tf.reshape(grid_y, (-1, 1))
    flat_grid_cell = tf.concat([flat_grid_x, flat_grid_y], axis=-1)
    grid_cell = tf.cast(tf.reshape(flat_grid_cell, [x_grid_size, y_grid_size, 1, 2]), tf.float32)
    return grid_cell

#%%
def output_to_pred(head, anchors, hyper_params):
    batch_size = hyper_params["batch_size"]
    img_size = hyper_params["img_size"]
    total_labels = hyper_params["total_labels"]

    grid_size = [head.shape[1], head.shape[2]]
    grid_cell = generate_grid_cell(grid_size[0], grid_size[1])

    ratio = tf.cast(img_size / grid_size[0], tf.float32)
    scaled_anchors = [(anchor[0]/ratio, anchor[1]/ratio) for anchor in anchors]

    head = tf.reshape(head, [batch_size, grid_size[0], grid_size[1], 3, 5 + total_labels])
    box_ctr_, box_size_, box_obj_, box_cls_ = tf.split(head, [2, 2, 1, total_labels], axis=-1)

    box_ctr = tf.nn.sigmoid(box_ctr_)
    box_ctr = box_ctr + grid_cell
    box_ctr = box_ctr * ratio # rescale to img size
    box_size = tf.exp(tf.tanh(box_size_)) * scaled_anchors
    box_size = box_size * ratio
    box_coor = tf.concat([box_ctr, box_size], axis=-1) # x y w h

    box_obj = tf.nn.sigmoid(box_obj_)
    box_cls = tf.nn.sigmoid(box_cls_)

    # flatten
    box_coor = tf.reshape(box_coor, [batch_size, grid_size[0] * grid_size[1], 3, 4]) 
    box_obj = tf.reshape(box_obj, [batch_size, grid_size[0] * grid_size[1], 3, 1])
    box_cls = tf.reshape(box_cls, [batch_size, grid_size[0] * grid_size[1], 3, total_labels])

    return box_coor, box_obj, box_cls


#%%
class Head(Layer):
    def __init__(self, anchors, hyper_params, **kwargs):
        super(Head, self).__init__(**kwargs)
        self.hyper_params = hyper_params
        self.anchors = anchors
    # @tf.function
    def call(self, inputs):
        coor_lst, obj_lst, cls_lst = [], [], []
        for i in range(len(inputs)):
            box_coor, box_obj, box_cls = output_to_pred(inputs[i], self.anchors[i], self.hyper_params)
            coor_lst.append(box_coor)
            obj_lst.append(box_obj)
            cls_lst.append(box_cls)
            
        boxes = tf.concat(coor_lst, axis=1)
        boxes = tf.reshape(boxes, [boxes.shape[0], boxes.shape[1] * boxes.shape[2], -1])

        confs = tf.concat(obj_lst, axis=1)
        confs = tf.reshape(confs, [confs.shape[0], confs.shape[1] * confs.shape[2], -1])

        probs = tf.concat(cls_lst, axis=1)
        probs = tf.reshape(probs, [probs.shape[0], probs.shape[1] * probs.shape[2], -1])

        pred = tf.concat([boxes, confs, probs], axis=-1)
        return pred

#%%
def yolo_block(inputs, filters):
    x = conv_block(filters*1, 1)(inputs)
    x = conv_block(filters*2, 3)(x)
    x = conv_block(filters*1, 1)(x)
    x = conv_block(filters*2, 3)(x)
    fpn = conv_block(filters*1, 1)(x)
    head = conv_block(filters*2, 3)(fpn)
    return [head, fpn]

#%%
def yolo_v3(input_shape, hyper_params):
    base_model = DarkNet53(include_top=False, input_shape=input_shape)
    weights_dir = os.getcwd() + "/darknet"#
    base_model.load_weights(weights_dir + '/weights')#
    base_model.trainable=False

    # base_model = Model(inputs=backbone.input, outputs=backbone.get_layer("add_22").output)
    total_labels = hyper_params["total_labels"]

    inputs = base_model.input

    head1, p1 = yolo_block(base_model.output[2], 512)
    head1 = Conv2D(3 * (5 + total_labels), 1, 1, bias_initializer=tf.zeros_initializer)(head1) # 13
    p1 = conv_block(256, 1)(p1)
    p1 = UpSampling2D(2)(p1)
    concat1 = Concatenate()([p1, base_model.output[1]])
    head2, p2 = yolo_block(concat1, 256)
    head2 = Conv2D(3 * (5 + total_labels), 1, 1, bias_initializer=tf.zeros_initializer)(head2) # 26
    p2 = conv_block(128, 1)(p2)
    p2 = UpSampling2D(2)(p2)
    concat2 = Concatenate()([p2, base_model.output[0]])
    head3, _ = yolo_block(concat2, 128)
    head3 = Conv2D(3 * (5 + total_labels), 1, 1, bias_initializer=tf.zeros_initializer)(head3) # 53
    head = [head1, head2, head3] # 13 26 52
    # outputs = Head(anchors, hyper_params)(head)

    return Model(inputs=inputs, outputs=head)

#%%
def yolo_head(feats, anchors, num_classes, input_shape):
    num_anchors = len(anchors)
    anchors_tensor = tf.reshape(anchors, [1, 1, 1, num_anchors, 2])

    grid_shape = tf.shape(feats)[1:3]

    grid_y = tf.tile(tf.reshape(tf.range(0, grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
    grid_x = tf.tile(tf.reshape(tf.range(0, grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])

    grid = tf.concat([grid_y, grid_x], axis=-1)
    grid = tf.cast(grid, tf.float32)

    feats = tf.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    box_yx = (tf.nn.sigmoid(feats[...,:2]) + grid) / tf.cast(grid_shape[...,-1], tf.float32)
    box_hw = tf.exp(feats[..., 2:4]) * anchors_tensor / tf.cast(input_shape[..., -1], tf.float32)

    box_obj = tf.sigmoid(feats[..., 4:5])
    box_cls = tf.sigmoid(feats[..., 5:])

    return grid, feats, box_yx, box_hw, box_obj, box_cls

