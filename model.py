#%%
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
def output_to_pred(head, anchors, hyper_params):
    batch_size = hyper_params["batch_size"]
    img_size = hyper_params["img_size"]
    total_class = hyper_params["total_class"]

    grid_size = [head.shape[1], head.shape[2]]
    grid_cell = generate_grid_cell(grid_size[0], grid_size[1])

    ratio = tf.cast(img_size / grid_size[0], tf.float32)
    scaled_anchors = [(anchor[0]/ratio, anchor[1]/ratio) for anchor in anchors]

    head = tf.reshape(head, [batch_size, grid_size[0], grid_size[1], 3, 5 + total_class])
    box_ctr_, box_size_, box_obj_, box_cls_ = tf.split(head, [2, 2, 1, total_class], axis=-1)

    box_ctr = tf.nn.sigmoid(box_ctr_)

    box_ctr = box_ctr + grid_cell
    box_ctr = box_ctr * ratio # rescale to img size

    box_size = tf.exp(box_size_) * scaled_anchors
    box_size = box_size * ratio

    box_coor = tf.concat([box_ctr, box_size], axis=-1) # x y w h
    box_coor = tf.reshape(box_coor, [batch_size, grid_size[0] * grid_size[1], 3, 4])

    box_obj = tf.sigmoid(box_obj_)
    box_obj = tf.reshape(box_obj, [batch_size, grid_size[0] * grid_size[1], 3, 1])

    box_cls = tf.sigmoid(box_cls_)
    box_cls = tf.reshape(box_cls, [batch_size, grid_size[0] * grid_size[1], 3, total_class])
    return box_coor, box_obj, box_cls

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
class Head(Layer):
    def __init__(self, anchors, hyper_params, **kwargs):
        super(Head, self).__init__(**kwargs)
        self.hyper_params = hyper_params
        self.anchors = anchors

    def call(self, inputs):
        coor_lst, obj_lst, cls_lst = [], [], []
        for i in range(len(inputs)):
            box_coor, box_obj, box_cls = output_to_pred(inputs[i], self.anchors[i], self.hyper_params)
            coor_lst.append(box_coor)
            obj_lst.append(box_obj)
            cls_lst.append(box_cls)
            
        boxes = tf.concat(coor_lst, axis=1)
        boxes = tf.reshape(boxes, [boxes.shape[0], boxes.shape[1] * boxes.shape[2], -1])
        x_ctr, y_ctr, width, height = tf.split(boxes, [1,1,1,1], axis=-1)
        x1 = x_ctr - width/2
        y1 = y_ctr - height/2
        x2 = x_ctr + width/2
        y2 = y_ctr + height/2
        boxes = tf.concat([x1, y1, x2, y2], axis=-1)

        confs = tf.concat(obj_lst, axis=1)
        confs = tf.reshape(confs, [confs.shape[0], confs.shape[1] * confs.shape[2], -1])

        probs = tf.concat(cls_lst, axis=1)
        probs = tf.reshape(probs, [probs.shape[0], probs.shape[1] * probs.shape[2], -1])

        pred = tf.concat([boxes, confs, probs], axis=-1)
        return pred

#%%
def DarkNet53(include_top=True, input_shape=(None, None, 3)):

    input_x = Input(shape=input_shape)
    x = conv_block(32, 3, name="conv1")(input_x)
    x = conv_block(64, 3, 2, name="conv2")(x)
    x = res_block(x, 1)
    x = conv_block(128, 3, 2, name="conv3")(x)
    x = res_block(x, 2)
    x = conv_block(256, 3, 2, name="conv4")(x)
    x = c3 = res_block(x, 8)
    x = conv_block(512, 3, 2, name="conv5")(x)
    x = c2 = res_block(x, 8)
    x = conv_block(1024, 3, 2, name="conv6")(x)
    x = c1 = res_block(x, 4)
    if include_top == False:
        return Model(inputs=input_x, outputs=[c3, c2, c1])
    x = GlobalAveragePooling2D(name="avgpooling")(x)
    output_x = Dense(1000, activation="softmax", name="fc")(x)
    return Model(inputs=input_x, outputs=output_x)

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
def yolo_v3(input_shape, hyper_params, anchors):
    base_model = DarkNet53(include_top=False, input_shape=input_shape)

    inputs = base_model.input

    head1, p1 = yolo_block(base_model.output[2], 512)
    head1 = Conv2D(3 * (5 + 80), 1, 1, bias_initializer=tf.zeros_initializer)(head1)
    p1 = conv_block(256, 1)(p1)
    p1 = UpSampling2D(2)(p1)
    concat1 = Concatenate()([p1, base_model.output[1]])
    head2, p2 = yolo_block(concat1, 256)
    head2 = Conv2D(3 * (5 + 80), 1, 1, bias_initializer=tf.zeros_initializer)(head2)
    p2 = conv_block(128, 1)(p2)
    p2 = UpSampling2D(2)(p2)
    concat2 = Concatenate()([p2, base_model.output[0]])
    head3, _ = yolo_block(concat2, 128)
    head3 = Conv2D(3 * (5 + 80), 1, 1, bias_initializer=tf.zeros_initializer)(head3)
    head = [head1, head2, head3]
    outputs = Head(anchors, hyper_params)(head)

    return Model(inputs=inputs, outputs=outputs)
# model = yolo_v3((416, 416, 3))
# model.summary()
# model.output

#%%
# hyper_params = {
#     "total_anchors" : 3,
#     "total_class" : 80,
#     "img_size" : 416
# }
# class YOLOv3(Model):
#     def __init__(self, hyper_params):
#         super(YOLOv3, self).__init__()
#         self.hyper_params = hyper_params
#         self.base_model = DarkNet53(include_top=False, inputs_shape=(hyper_params["img_size"], hyper_params["img_size"], 3))
#         self.conv = Conv2D(3 * (5 + 80), 1, 1, bias_initializer=tf.zeros_initializer)
#         self.upsample = UpSampling2D(2)
#         self.concat = Concatenate()
#         yolo_block
#         conv_block

#     def call(self, inputs):
#         x = self.base_model(inputs)
        
#         head1, p1 = yolo_block(self.base_model.output[2], 512)
#         head1 = self.conv(head1)
#         p1 = conv_block(256, 1)(p1)
#         p1 = self.upsample(2)(p1)
#         concat1 = self.concat([p1, self.base_model.output[1]])
#         head2, p2 = yolo_block(concat1, 256)
#         head2 = self.conv(head2)
#         p2 = conv_block(128, 1)(p2)
#         p2 = self.upsample(p2)
#         concat2 = self.concat([p2, self.base_model.output[0]])
#         head3, _ = yolo_block(concat2, 128)
#         head3 = self.conv(head3)
#         return [head1, head2, head3]



