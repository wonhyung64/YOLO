#%%
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Layer, Conv2D, Input, GlobalAveragePooling2D, Dense

# %%
def res_block(inputs, filters):
    shortcut = inputs
    net = Conv2D(inputs, filters * 1, 1)
    net = Conv2D(net, filters * 2, 3)

    net = net + shortcut
    
    return net
#%%
input = Input(shape=(512, 512, 3))

net = Conv2D(filters=32, kernel_size=3, padding="same")(input)
net = Conv2D(filters=64, kernel_size=3, strides=2)(net)
net

net_ = Conv2D(filters=32*1, kernel_size=1, padding="same")(net)
net_ = Conv2D(filters=32*2, kernel_size=3, padding="same")(net_)
net_

net + net_

#%%
net = res_block(net, 32)


# %%
kernel_size = 3
pad_total = kernel_size - 1
pad_beg = pad_total // 2
pad_end = pad_total - pad_beg

inputs = net
padded_inputs = tf.pad(inputs, [[0,0], [pad_beg, pad_end], [pad_beg, pad_end], [0,0]], mode='CONSTANT')
strides = 2
print(Conv2D(64, kernel_size=3, strides=strides, padding = ("same" if strides == 1 else 'valid'))(padded_inputs))
#%%
class cus_cv2d(Layer):
    def __init__(self, filters, kernel_size, strides=1, **kwargs):
        super(cus_cv2d, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

    def call(self, inputs):
        tensor = inputs
        if self.strides > 1:
            pad_total = self.kernel_size - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            tensor = tf.pad(tensor, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        tensor = Conv2D(self.filters, self.kernel_size, self.strides, padding=('same' if self.strides==1 else 'valid'))(tensor)
        return tensor


# %%
#%%
class res_block(Layer):
    def __init__(self, filters, repeat, **kwargs):
        super(res_block, self).__init__(**kwargs)
        self.filters = filters
        self.repeat = repeat

    def call(self, inputs):
        tensor = inputs
        for i in range(self.repeat):
            tensor_ = cus_cv2d(self.filters*1, 1)(tensor)
            tensor_ = cus_cv2d(self.filters*2, 3)(tensor_)
            tensor = tensor + tensor_

        return tensor

# %%

input = Input(shape=(256, 256, 3))

net = cus_cv2d(32, 3, name = 'cv1')(input)
net = cus_cv2d(64, 3, 2)(net)

net = res_block(32, 1)(net)

net = cus_cv2d(128, 3, 2)(net)

net = res_block(64, 2)(net)

net = cus_cv2d(256, 3, 2)(net)

net = res_block(128, 8)(net)

net = cus_cv2d(512, 3, 2)(net)

net = res_block(256, 8)(net)

net = cus_cv2d(1024, 3, 2)(net)

net = res_block(512, 4)(net)


net = GlobalAveragePooling2D()(net)
net = Dense(1000, activation="softmax")(net)

#%%
input_shape = (None,256,256,3)
from tensorflow.keras.models import Model
class Darknet53(Model):
    def __init__(self):
        super(Darknet53, self).__init__()

        self.conv1 = cus_cv2d(32, 3, name="conv1")
        self.conv2 = cus_cv2d(64, 32, 3, name="conv2")
        self.res1 = res_block(32, 1, name="res1")
        self.conv3 = cus_cv2d(128, 3, 2, name="conv3")
        self.res2 = res_block(64, 2, name="res2")
        self.conv4 = cus_cv2d(256, 3, 2, name="conv4")
        self.res3 = res_block(128, 8, name="res3")
        self.conv5 = cus_cv2d(512, 3, 2, name="conv5")
        self.res4 = res_block(256, 8, name="res4")
        self.conv6 = cus_cv2d(1024, 3, 2, name="conv6")
        self.res5 = res_block(512, 4, name="res5")
        self.avgpool = GlobalAveragePooling2D(name="avgpool")
        self.fc = Dense(1000, activation="softmax", name="fc")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.conv3(x)
        x = self.res2(x)
        x = self.conv4(x)
        x = self.res3(x)
        x = self.conv5(x)
        x = self.res4(x)
        x = self.conv6(x)
        x = self.res5(x)
        x = self.avgpool(x)
        cls = self.fc(x)
        return cls
#%%
model = Darknet53()
input_shape = (1, 256, 256, 3)
model.build(input_shape)
model.summary()

model.get_layer("conv1").output


    
# %%
