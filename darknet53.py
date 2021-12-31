#%%
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Layer, Conv2D, Input, GlobalAveragePooling2D, Dense, Add, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model

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
class conv2d(Layer):
    def __init__(self, filters, kernel_size, strides=1, **kwargs):
        super(conv2d, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

    def build(self, input_shape):
        self.conv = Conv2D(self.filters, self.kernel_size, self.strides, padding=("same" if self.strides==1 else "valid"))
        super(conv2d, self).build(input_shape)

    def call(self, inputs):
        tensor = inputs
        if self.strides > 1:
            pad_total = self.kernel_size - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            tensor = tf.pad(tensor, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        tensor = self.conv(tensor)
        return tensor


#%%
class res_block(Layer):
    def __init__(self, filters, repeat, **kwargs):
        super(res_block, self).__init__(**kwargs)
        self.filters = filters
        self.repeat = repeat

    def build(self, input_shape):
        

    def call(self, inputs):
        tensor = inputs
        for i in range(self.repeat):
            tensor_ = cus_cv2d(self.filters*1, 1)(tensor)
            tensor_ = cus_cv2d(self.filters*2, 3)(tensor_)
            tensor = tensor + tensor_

        return tensor

# %%

input = Input(shape=(256, 256, 3))
#convolution
x = conv2d(32, 3, name="conv1_conv")(input)

x = BatchNormalization(name="conv1_bn")(x)

x = LeakyReLU(name="conv1_leakyrelu")(x)
#convolution
x = conv2d(64, 3, 2, name="conv2_conv")(x)

x = BatchNormalization(name="conv2_bn")(x)

x = LeakyReLU(name="conv2_leakyrelu")(x)
#res block 1
x_ = conv2d(32, 1, name="conv3_block1_1_conv")(x)

x_ = BatchNormalization(name="conv3_block1_1_bn")(x_)

x_ = LeakyReLU(name="conv3_block1_1_leakyrelu")(x_)

x_ = conv2d(64, 3, name="conv3_block1_2_conv")(x_)

x_ = BatchNormalization(name="conv3_block1_2_bn")(x_)

x_ = LeakyReLU(name="conv3_block1_2_leakyrelu")(x_)

x = Add(name="conv3_block1_add")([x, x_])
#convolution
x = conv2d(128, 3, 2, name="conv4_conv")(x)

x = BatchNormalization(name="conv4_bn")(x)

x = LeakyReLU(name="conv4_leakyrelu")(x)
#res block 2 1
x_ = conv2d(64, 1, name="conv5_block1_1_conv")(x)

x_ = BatchNormalization(name="conv5_block1_1_bn")(x_)

x_ = LeakyReLU(name="conv5_block1_1_leakyrelu")(x_)

x_ = conv2d(128, 3, name="conv5_block1_2_conv")(x_)

x_ = BatchNormalization(name="conv5_block1_2_bn")(x_)

x_ = LeakyReLU(name="conv5_block1_2_leakyrelu")(x_)

x = Add(name="conv5_block1_add")([x, x_])
#res block 2 2
x_ = conv2d(64, 1, name="conv5_block2_1_conv")(x)

x_ = BatchNormalization(name="conv5_block2_1_bn")(x_)

x_ = LeakyReLU(name="conv5_block2_1_leakyrelu")(x_)

x_ = conv2d(128, 3, name="conv5_block2_2_conv")(x_)

x_ = BatchNormalization(name="conv5_block2_2_bn")(x_)

x_ = LeakyReLU(name="conv5_block2_2_leakyrelu")(x_)

x = Add(name="conv5_block2_add")([x, x_])
#convolution
x = conv2d(256, 3, 2, name="conv6_conv")(x)

x = BatchNormalization(name="conv6_bn")(x)

x = LeakyReLU(name="conv6_leakyrelu")(x)
# res block 3 1
x_ = conv2d(128, 1, name="conv7_block1_1_conv")(x)

x_ = BatchNormalization(name="conv7_block1_1_bn")(x_)

x_ = LeakyReLU(name="conv7_block1_1_leakyrelu")(x_)

x_ = conv2d(256, 3, name="conv7_block1_2_conv")(x_)

x_ = BatchNormalization(name="conv7_block1_2_bn")(x_)

x_ = LeakyReLU(name="conv7_block1_2_leakyrelu")(x_)

x = Add(name="conv7_block1_add")([x, x_])
# res block 3 2
x_ = conv2d(128, 1, name="conv7_block2_1_conv")(x)

x_ = BatchNormalization(name="conv7_block2_1_bn")(x_)

x_ = LeakyReLU(name="conv7_block2_1_leakyrelu")(x_)

x_ = conv2d(256, 3, name="conv7_block2_2_conv")(x_)

x_ = BatchNormalization(name="conv7_block2_2_bn")(x_)

x_ = LeakyReLU(name="conv7_block2_2_leakyrelu")(x_)

x = Add(name="conv7_block2_add")([x, x_])
# res block 3 3
x_ = conv2d(128, 1, name="conv7_block3_1_conv")(x)

x_ = BatchNormalization(name="conv7_block3_1_bn")(x_)

x_ = LeakyReLU(name="conv7_block3_1_leakyrelu")(x_)

x_ = conv2d(256, 3, name="conv7_block3_2_conv")(x_)

x_ = BatchNormalization(name="conv7_block3_2_bn")(x_)

x_ = LeakyReLU(name="conv7_block3_2_leakyrelu")(x_)

x = Add(name="conv7_block3_add")([x, x_])
# res block 3 4
x_ = conv2d(128, 1, name="conv7_block4_1_conv")(x)

x_ = BatchNormalization(name="conv7_block4_1_bn")(x_)

x_ = LeakyReLU(name="conv7_block4_1_leakyrelu")(x_)

x_ = conv2d(256, 3, name="conv7_block4_2_conv")(x_)

x_ = BatchNormalization(name="conv7_block4_2_bn")(x_)

x_ = LeakyReLU(name="conv7_block4_2_leakyrelu")(x_)

x = Add(name="conv7_block4_add")([x, x_])
# res block 3 5
x_ = conv2d(128, 1, name="conv7_block5_1_conv")(x)

x_ = BatchNormalization(name="conv7_block5_1_bn")(x_)

x_ = LeakyReLU(name="conv7_block5_1_leakyrelu")(x_)

x_ = conv2d(256, 3, name="conv7_block5_2_conv")(x_)

x_ = BatchNormalization(name="conv7_block5_2_bn")(x_)

x_ = LeakyReLU(name="conv7_block5_2_leakyrelu")(x_)

x = Add(name="conv7_block5_add")([x, x_])
# res block 3 6
x_ = conv2d(128, 1, name="conv7_block6_1_conv")(x)

x_ = BatchNormalization(name="conv7_block6_1_bn")(x_)

x_ = LeakyReLU(name="conv7_block6_1_leakyrelu")(x_)

x_ = conv2d(256, 3, name="conv7_block6_2_conv")(x_)

x_ = BatchNormalization(name="conv7_block6_2_bn")(x_)

x_ = LeakyReLU(name="conv7_block6_2_leakyrelu")(x_)

x = Add(name="conv7_block6_add")([x, x_])
# res block 3 7
x_ = conv2d(128, 1, name="conv7_block7_1_conv")(x)

x_ = BatchNormalization(name="conv7_block7_1_bn")(x_)

x_ = LeakyReLU(name="conv7_block7_1_leakyrelu")(x_)

x_ = conv2d(256, 3, name="conv7_block7_2_conv")(x_)

x_ = BatchNormalization(name="conv7_block7_2_bn")(x_)

x_ = LeakyReLU(name="conv7_block7_2_leakyrelu")(x_)

x = Add(name="conv7_block7_add")([x, x_])
# res block 3 8
x_ = conv2d(128, 1, name="conv7_block8_1_conv")(x)

x_ = BatchNormalization(name="conv7_block8_1_bn")(x_)

x_ = LeakyReLU(name="conv7_block8_1_leakyrelu")(x_)

x_ = conv2d(256, 3, name="conv7_block8_2_conv")(x_)

x_ = BatchNormalization(name="conv7_block8_2_bn")(x_)

x_ = LeakyReLU(name="conv7_block8_2_leakyrelu")(x_)

x = Add(name="conv7_block8_add")([x, x_])
#convolution
x = conv2d(512, 3, 2, name="conv8_conv")(x)

x = BatchNormalization(name="conv8_bn")(x)

x = LeakyReLU(name="conv8_leakyrelu")(x)
# res block 4 1
x_ = conv2d(256, 1, name="conv9_block1_1_conv")(x)

x_ = BatchNormalization(name="conv9_block1_1_bn")(x_)

x_ = LeakyReLU(name="conv9_block1_1_leakyrelu")(x_)

x_ = conv2d(512, 3, name="conv9_block1_2_conv")(x_)

x_ = BatchNormalization(name="conv9_block1_2_bn")(x_)

x_ = LeakyReLU(name="conv9_block1_2_leakyrelu")(x_)

x = Add(name="conv9_block1_add")([x, x_])
# res block 4 2
x_ = conv2d(256, 1, name="conv9_block2_1_conv")(x)

x_ = BatchNormalization(name="conv9_block2_1_bn")(x_)

x_ = LeakyReLU(name="conv9_block2_1_leakyrelu")(x_)

x_ = conv2d(512, 3, name="conv9_block2_2_conv")(x_)

x_ = BatchNormalization(name="conv9_block2_2_bn")(x_)

x_ = LeakyReLU(name="conv9_block2_2_leakyrelu")(x_)

x = Add(name="conv9_block2_add")([x, x_])
# res block 4 3
x_ = conv2d(256, 1, name="conv9_block3_1_conv")(x)

x_ = BatchNormalization(name="conv9_block3_1_bn")(x_)

x_ = LeakyReLU(name="conv9_block3_1_leakyrelu")(x_)

x_ = conv2d(512, 3, name="conv9_block3_2_conv")(x_)

x_ = BatchNormalization(name="conv9_block3_2_bn")(x_)

x_ = LeakyReLU(name="conv9_block3_2_leakyrelu")(x_)

x = Add(name="conv9_block3_add")([x, x_])
# res block 4 4
x_ = conv2d(256, 1, name="conv9_block4_1_conv")(x)

x_ = BatchNormalization(name="conv9_block4_1_bn")(x_)

x_ = LeakyReLU(name="conv9_block4_1_leakyrelu")(x_)

x_ = conv2d(512, 3, name="conv9_block4_2_conv")(x_)

x_ = BatchNormalization(name="conv9_block4_2_bn")(x_)

x_ = LeakyReLU(name="conv9_block4_2_leakyrelu")(x_)

x = Add(name="conv9_block4_add")([x, x_])
# res block 4 5
x_ = conv2d(256, 1, name="conv9_block5_1_conv")(x)

x_ = BatchNormalization(name="conv9_block5_1_bn")(x_)

x_ = LeakyReLU(name="conv9_block5_1_leakyrelu")(x_)

x_ = conv2d(512, 3, name="conv9_block5_2_conv")(x_)

x_ = BatchNormalization(name="conv9_block5_2_bn")(x_)

x_ = LeakyReLU(name="conv9_block5_2_leakyrelu")(x_)

x = Add(name="conv9_block5_add")([x, x_])
# res block 4 6
x_ = conv2d(256, 1, name="conv9_block6_1_conv")(x)

x_ = BatchNormalization(name="conv9_block6_1_bn")(x_)

x_ = LeakyReLU(name="conv9_block6_1_leakyrelu")(x_)

x_ = conv2d(512, 3, name="conv9_block6_2_conv")(x_)

x_ = BatchNormalization(name="conv9_block6_2_bn")(x_)

x_ = LeakyReLU(name="conv9_block6_2_leakyrelu")(x_)

x = Add(name="conv9_block6_add")([x, x_])
# res block 4 7
x_ = conv2d(256, 1, name="conv9_block7_1_conv")(x)

x_ = BatchNormalization(name="conv9_block7_1_bn")(x_)

x_ = LeakyReLU(name="conv9_block7_1_leakyrelu")(x_)

x_ = conv2d(512, 3, name="conv9_block7_2_conv")(x_)

x_ = BatchNormalization(name="conv9_block7_2_bn")(x_)

x_ = LeakyReLU(name="conv9_block7_2_leakyrelu")(x_)

x = Add(name="conv9_block7_add")([x, x_])
# res block 4 8
x_ = conv2d(256, 1, name="conv9_block8_1_conv")(x)

x_ = BatchNormalization(name="conv9_block8_1_bn")(x_)

x_ = LeakyReLU(name="conv9_block8_1_leakyrelu")(x_)

x_ = conv2d(512, 3, name="conv9_block8_2_conv")(x_)

x_ = BatchNormalization(name="conv9_block8_2_bn")(x_)

x_ = LeakyReLU(name="conv9_block8_2_leakyrelu")(x_)

x = Add(name="conv9_block8_add")([x, x_])
#convolution
x = conv2d(1024, 3, 2, name="conv10_conv")(x)

x = BatchNormalization(name="conv10_bn")(x)

x = LeakyReLU(name="conv10_leakyrelu")(x)
# res block 5 1
x_ = conv2d(512, 1, name="conv11_block1_1_conv")(x)

x_ = BatchNormalization(name="conv11_block1_1_bn")(x_)

x_ = LeakyReLU(name="conv11_block1_1_leakyrelu")(x_)

x_ = conv2d(1024, 3, name="conv11_block1_2_conv")(x_)

x_ = BatchNormalization(name="conv11_block1_2_bn")(x_)

x_ = LeakyReLU(name="conv11_block1_2_leakyrelu")(x_)

x = Add(name="conv11_block1_add")([x, x_])
# res block 5 2
x_ = conv2d(512, 1, name="conv11_block2_1_conv")(x)

x_ = BatchNormalization(name="conv11_block2_1_bn")(x_)

x_ = LeakyReLU(name="conv11_block2_1_leakyrelu")(x_)

x_ = conv2d(1024, 3, name="conv11_block2_2_conv")(x_)

x_ = BatchNormalization(name="conv11_block2_2_bn")(x_)

x_ = LeakyReLU(name="conv11_block2_2_leakyrelu")(x_)

x = Add(name="conv11_block2_add")([x, x_])
# res block 5 3
x_ = conv2d(512, 1, name="conv11_block3_1_conv")(x)

x_ = BatchNormalization(name="conv11_block3_1_bn")(x_)

x_ = LeakyReLU(name="conv11_block3_1_leakyrelu")(x_)

x_ = conv2d(1024, 3, name="conv11_block3_2_conv")(x_)

x_ = BatchNormalization(name="conv11_block3_2_bn")(x_)

x_ = LeakyReLU(name="conv11_block3_2_leakyrelu")(x_)

x = Add(name="conv11_block3_add")([x, x_])
# res block 5 4
x_ = conv2d(512, 1, name="conv11_block4_1_conv")(x)

x_ = BatchNormalization(name="conv11_block4_1_bn")(x_)

x_ = LeakyReLU(name="conv11_block4_1_leakyrelu")(x_)

x_ = conv2d(1024, 3, name="conv11_block4_2_conv")(x_)

x_ = BatchNormalization(name="conv11_block4_2_bn")(x_)

x_ = LeakyReLU(name="conv11_block4_2_leakyrelu")(x_)

x = Add(name="conv11_block4_add")([x, x_])

x = GlobalAveragePooling2D(name="avg_pool")(x)
output = Dense(1000, activation="softmax", name="predictions")(x)

model = Model(inputs=input, outputs=output)
model.summary()
#%%
class Darknet53(Model):
    def __init__(self):
        super(Darknet53, self).__init__()

        #convolution
        self.cv1_cv = conv2d(32, 3, name="conv1_conv")(input)
        self.cv1_bn = BatchNormalization(name="conv1_bn")(x)
        self.cv1_ac = LeakyReLU(name="conv1_leakyrelu")(x)
        #convolution
        self.cv2_cv = conv2d(64, 3, 2, name="conv2_conv")(x)
        self.cv2_bn = BatchNormalization(name="conv2_bn")(x)
        self.cv2_ac = LeakyReLU(name="conv2_leakyrelu")(x)
        #res block 1
        self.cv3_b1_1_cv = conv2d(32, 1, name="conv3_block1_1_conv")(x)
        self.cv3_b1_1_bn = BatchNormalization(name="conv3_block1_1_bn")(x_)
        self.cv3_b1_1_ac = LeakyReLU(name="conv3_block1_1_leakyrelu")(x_)
        self.cv3_b1_2_cv = conv2d(64, 3, name="conv3_block1_2_conv")(x_)
        self.cv3_b1_2_bn = BatchNormalization(name="conv3_block1_2_bn")(x_)
        self.cv3_b1_2_ac = LeakyReLU(name="conv3_block1_2_leakyrelu")(x_)
        self.cv3_b1_add = Add(name="conv3_block1_add")([x, x_])
        #convolution
        self.cv4_cv = conv2d(128, 3, 2, name="conv4_conv")(x)
        self.cv4_bn = BatchNormalization(name="conv4_bn")(x)
        self.cv4_ac = LeakyReLU(name="conv4_leakyrelu")(x)
        #res block 2 1
        self.cv5_b1_1_cv = conv2d(64, 1, name="conv5_block1_1_conv")(x)
        self.cv5_b1_1_bn = BatchNormalization(name="conv5_block1_1_bn")(x_)
        self.cv5_b1_1_ac = LeakyReLU(name="conv5_block1_1_leakyrelu")(x_)
        self.cv5_b1_2_cv = conv2d(128, 3, name="conv5_block1_2_conv")(x_)
        self.cv5.b1_2_bn = BatchNormalization(name="conv5_block1_2_bn")(x_)
        self.cv5.b1_2_ac = LeakyReLU(name="conv5_block1_2_leakyrelu")(x_)
        self.cv5_b1_add = Add(name="conv5_block1_add")([x, x_])
        #res block 2 2
        self.cv5_b2_1_cv = conv2d(64, 1, name="conv5_block2_1_conv")(x)
        self.cv5_b2_1_bn = BatchNormalization(name="conv5_block2_1_bn")(x_)
        self.cv5_b2_1_ac = LeakyReLU(name="conv5_block2_1_leakyrelu")(x_)
        self.cv5_b2_2_cv = conv2d(128, 3, name="conv5_block2_2_conv")(x_)
        self.cv5_b2_2_bn = BatchNormalization(name="conv5_block2_2_bn")(x_)
        self.cv5_b2_2_ac = LeakyReLU(name="conv5_block2_2_leakyrelu")(x_)
        self.cv5_b2_add = Add(name="conv5_block2_add")([x, x_])
        #convolution
        self.cv6_cv = conv2d(256, 3, 2, name="conv6_conv")(x)
        self.cv6_bn = BatchNormalization(name="conv6_bn")(x)
        self.cv6_ac = LeakyReLU(name="conv6_leakyrelu")(x)
        # res block 3 1
        self.cv7_b1_1_cv = conv2d(128, 1, name="conv7_block1_1_conv")(x)
        self.cv7_b1_1_bn = BatchNormalization(name="conv7_block1_1_bn")(x_)
        self.cv7_b1_1_ac = LeakyReLU(name="conv7_block1_1_leakyrelu")(x_)
        self.cv7_b1_2_cv = conv2d(256, 3, name="conv7_block1_2_conv")(x_)
        self.cv7.b1_2_bn = BatchNormalization(name="conv7_block1_2_bn")(x_)
        self.cv7.b1_2_ac = LeakyReLU(name="conv7_block1_2_leakyrelu")(x_)
        self.cv7_b1_add = Add(name="conv7_block1_add")([x, x_])
        # res block 3 2
        self.cv7_b2_1_cv = conv2d(128, 1, name="conv7_block2_1_conv")(x)
        self.cv7_b2_1_bn = BatchNormalization(name="conv7_block2_1_bn")(x_)
        self.cv7_b2_1_ac = LeakyReLU(name="conv7_block2_1_leakyrelu")(x_)
        self.cv7_b2_2_cv = conv2d(256, 3, name="conv7_block2_2_conv")(x_)
        self.cv7_b2_2_bn = BatchNormalization(name="conv7_block2_2_bn")(x_)
        self.cv7_b2_2_ac = LeakyReLU(name="conv7_block2_2_leakyrelu")(x_)
        self.cv7_b2_add = Add(name="conv7_block2_add")([x, x_])
        # res block 3 3
        self.cv7_b3_1_cv = conv2d(128, 1, name="conv7_block3_1_conv")(x)
        self.cv7_b3_1_bn = BatchNormalization(name="conv7_block3_1_bn")(x_)
        self.cv7_b3_1_ac = LeakyReLU(name="conv7_block3_1_leakyrelu")(x_)
        self.cv7_b3_2_cv = conv2d(256, 3, name="conv7_block3_2_conv")(x_)
        self.cv7_b3_2_bn = BatchNormalization(name="conv7_block3_2_bn")(x_)
        self.cv7_b3_2_ac = LeakyReLU(name="conv7_block3_2_leakyrelu")(x_)
        self.cv7_b3_add = Add(name="conv7_block3_add")([x, x_])
        # res block 3 4
        self.cv7_b4_1_cv = conv2d(128, 1, name="conv7_block4_1_conv")(x)
        self.cv7_b4_1_bn = BatchNormalization(name="conv7_block4_1_bn")(x_)
        self.cv7_b4_1_ac = LeakyReLU(name="conv7_block4_1_leakyrelu")(x_)
        self.cv7_b4_2_cv = conv2d(256, 3, name="conv7_block4_2_conv")(x_)
        self.cv7_b4_2_bn = BatchNormalization(name="conv7_block4_2_bn")(x_)
        self.cv7_b4_2_ac = LeakyReLU(name="conv7_block4_2_leakyrelu")(x_)
        self.cv7_b4_add = Add(name="conv7_block4_add")([x, x_])
        # res block 3 5
        self.cv7_b5_1_cv = conv2d(128, 1, name="conv7_block5_1_conv")(x)
        self.cv7_b5_1_bn = BatchNormalization(name="conv7_block5_1_bn")(x_)
        self.cv7_b5_1_ac = LeakyReLU(name="conv7_block5_1_leakyrelu")(x_)
        self.cv7_b5_2_cv = conv2d(256, 3, name="conv7_block5_2_conv")(x_)
        self.cv7_b5_2_bn = BatchNormalization(name="conv7_block5_2_bn")(x_)
        self.cv7_b5_2_ac = LeakyReLU(name="conv7_block5_2_leakyrelu")(x_)
        self.cv7_b5_add = Add(name="conv7_block5_add")([x, x_])
        # res block 3 6
        self.cv7_b6_1_cv = conv2d(128, 1, name="conv7_block6_1_conv")(x)
        self.cv7_b6_1_bn = BatchNormalization(name="conv7_block6_1_bn")(x_)
        self.cv7_b6_1_ac = LeakyReLU(name="conv7_block6_1_leakyrelu")(x_)
        self.cv7_b6_2_cv = conv2d(256, 3, name="conv7_block6_2_conv")(x_)
        self.cv7_b6_2_bn = BatchNormalization(name="conv7_block6_2_bn")(x_)
        self.cv7_b6_2_ac = LeakyReLU(name="conv7_block6_2_leakyrelu")(x_)
        self.cv7_b6_add = Add(name="conv7_block6_add")([x, x_])
        # res block 3 7
        self.cv7_b7_1_cv = conv2d(128, 1, name="conv7_block7_1_conv")(x)
        self.cv7_b7_1_bn = BatchNormalization(name="conv7_block7_1_bn")(x_)
        self.cv7_b7_1_ac = LeakyReLU(name="conv7_block7_1_leakyrelu")(x_)
        self.cv7_b7_2_cv = conv2d(256, 3, name="conv7_block7_2_conv")(x_)
        self.cv7_b7_2_bn = BatchNormalization(name="conv7_block7_2_bn")(x_)
        self.cv7_b7_2_ac = LeakyReLU(name="conv7_block7_2_leakyrelu")(x_)
        self.cv7_b7_add = Add(name="conv7_block7_add")([x, x_])
        # res block 3 8
        self.cv7_b8_1_cv = conv2d(128, 1, name="conv7_block8_1_conv")(x)
        self.cv7_b8_1_bn = BatchNormalization(name="conv7_block8_1_bn")(x_)
        self.cv7_b8_1_ac = LeakyReLU(name="conv7_block8_1_leakyrelu")(x_)
        self.cv7_b8_2_cv = conv2d(256, 3, name="conv7_block8_2_conv")(x_)
        self.cv7_b8_2_bn = BatchNormalization(name="conv7_block8_2_bn")(x_)
        self.cv7_b8_2_ac = LeakyReLU(name="conv7_block8_2_leakyrelu")(x_)
        self.cv7_b8_add = Add(name="conv7_block8_add")([x, x_])
        #convolution
        self.cv8_cv = conv2d(512, 3, 2, name="conv8_conv")(x)
        self.cv8_bn = BatchNormalization(name="conv8_bn")(x)
        self.cv8_ac = LeakyReLU(name="conv8_leakyrelu")(x)
        # res block 4 1
        self.cv9_b1_1_cv = conv2d(256, 1, name="conv9_block1_1_conv")(x)
        self.cv9_b1_1_bn = BatchNormalization(name="conv9_block1_1_bn")(x_)
        self.cv9_b1_1_ac = LeakyReLU(name="conv9_block1_1_leakyrelu")(x_)
        self.cv9_b1_2_cv = conv2d(512, 3, name="conv9_block1_2_conv")(x_)
        self.cv9_b1_2_bn = BatchNormalization(name="conv9_block1_2_bn")(x_)
        self.cv9_b1_2_ac = LeakyReLU(name="conv9_block1_2_leakyrelu")(x_)
        self.cv9_b1_add = Add(name="conv9_block1_add")([x, x_])
        # res block 4 2
        self.cv9_b2_1_cv = conv2d(256, 1, name="conv9_block2_1_conv")(x)
        self.cv9_b2_1_bn = BatchNormalization(name="conv9_block2_1_bn")(x_)
        self.cv9_b2_1_ac = LeakyReLU(name="conv9_block2_1_leakyrelu")(x_)
        self.cv9_b2_2_cv = conv2d(512, 3, name="conv9_block2_2_conv")(x_)
        self.cv9_b2_2_bn = BatchNormalization(name="conv9_block2_2_bn")(x_)
        self.cv9_b2_2_ac = LeakyReLU(name="conv9_block2_2_leakyrelu")(x_)
        self.cv9_b2_add = Add(name="conv9_block2_add")([x, x_])
        # res block 4 3
        self.cv9_b3_1_cv = conv2d(256, 1, name="conv9_block3_1_conv")(x)
        self.cv9_b3_1_bn = BatchNormalization(name="conv9_block3_1_bn")(x_)
        self.cv9_b3_1_ac = LeakyReLU(name="conv9_block3_1_leakyrelu")(x_)
        self.cv9_b3_2_cv = conv2d(512, 3, name="conv9_block3_2_conv")(x_)
        self.cv9_b3_2_bn = BatchNormalization(name="conv9_block3_2_bn")(x_)
        self.cv9_b3_2_ac = LeakyReLU(name="conv9_block3_2_leakyrelu")(x_)
        self.cv9_b3_add = Add(name="conv9_block3_add")([x, x_])
        # res block 4 4
        self.cv9_b4_1_cv = conv2d(256, 1, name="conv9_block4_1_conv")(x)
        self.cv9_b4_1_bn = BatchNormalization(name="conv9_block4_1_bn")(x_)
        self.cv9_b4_1_ac = LeakyReLU(name="conv9_block4_1_leakyrelu")(x_)
        self.cv9_b4_2_cv = conv2d(512, 3, name="conv9_block4_2_conv")(x_)
        self.cv9_b4_2_bn = BatchNormalization(name="conv9_block4_2_bn")(x_)
        self.cv9_b4_2_ac = LeakyReLU(name="conv9_block4_2_leakyrelu")(x_)
        self.cv9_b4_add = Add(name="conv9_block4_add")([x, x_])
        # res block 4 5
        self.cv9_b5_1_cv = conv2d(256, 1, name="conv9_block5_1_conv")(x)
        self.cv9_b5_1_bn = BatchNormalization(name="conv9_block5_1_bn")(x_)
        self.cv9_b5_1_ac = LeakyReLU(name="conv9_block5_1_leakyrelu")(x_)
        self.cv9_b5_2_cv = conv2d(512, 3, name="conv9_block5_2_conv")(x_)
        self.cv9_b5_2_bn = BatchNormalization(name="conv9_block5_2_bn")(x_)
        self.cv9_b5_2_ac = LeakyReLU(name="conv9_block5_2_leakyrelu")(x_)
        self.cv9_b5_add = Add(name="conv9_block5_add")([x, x_])
        # res block 4 6
        self.cv9_b6_1_cv = conv2d(256, 1, name="conv9_block6_1_conv")(x)
        self.cv9_b6_1_bn = BatchNormalization(name="conv9_block6_1_bn")(x_)
        self.cv9_b6_1_ac = LeakyReLU(name="conv9_block6_1_leakyrelu")(x_)
        self.cv9_b6_2_cv = conv2d(512, 3, name="conv9_block6_2_conv")(x_)
        self.cv9_b6_2_bn = BatchNormalization(name="conv9_block6_2_bn")(x_)
        self.cv9_b6_2_ac = LeakyReLU(name="conv9_block6_2_leakyrelu")(x_)
        self.cv9_b6_add = Add(name="conv9_block6_add")([x, x_])
        # res block 4 7
        self.cv9_b7_1_cv = conv2d(256, 1, name="conv9_block7_1_conv")(x)
        self.cv9_b7_1_bn = BatchNormalization(name="conv9_block7_1_bn")(x_)
        self.cv9_b7_1_ac = LeakyReLU(name="conv9_block7_1_leakyrelu")(x_)
        self.cv9_b7_2_cv = conv2d(512, 3, name="conv9_block7_2_conv")(x_)
        self.cv9_b7_2_bn = BatchNormalization(name="conv9_block7_2_bn")(x_)
        self.cv9_b7_2_ac = LeakyReLU(name="conv9_block7_2_leakyrelu")(x_)
        self.cv9_b7_add = Add(name="conv9_block7_add")([x, x_])
        # res block 4 8
        self.cv9_b8_1_cv = conv2d(256, 1, name="conv9_block8_1_conv")(x)
        self.cv9_b8_1_bn = BatchNormalization(name="conv9_block8_1_bn")(x_)
        self.cv9_b8_1_ac = LeakyReLU(name="conv9_block8_1_leakyrelu")(x_)
        self.cv9_b8_2_cv = conv2d(512, 3, name="conv9_block8_2_conv")(x_)
        self.cv9_b8_2_bn = BatchNormalization(name="conv9_block8_2_bn")(x_)
        self.cv9_b8_2_ac = LeakyReLU(name="conv9_block8_2_leakyrelu")(x_)
        self.cv9_b8_add = Add(name="conv9_block8_add")([x, x_])
        #convolution
        self.cv10_cv = conv2d(1024, 3, 2, name="conv10_conv")(x)
        self.cv10_bn = BatchNormalization(name="conv10_bn")(x)
        self.cv10_ac = LeakyReLU(name="conv10_leakyrelu")(x)
        # res block 5 1
        self.cv11_b1_1_cv = conv2d(512, 1, name="conv11_block1_1_conv")(x)
        self.cv11_b1_1_bn = BatchNormalization(name="conv11_block1_1_bn")(x_)
        self.cv11_b1_1_ac = LeakyReLU(name="conv11_block1_1_leakyrelu")(x_)
        self.cv11_b1_2_cv = conv2d(1024, 3, name="conv11_block1_2_conv")(x_)
        self.cv11_b1_2_bn = BatchNormalization(name="conv11_block1_2_bn")(x_)
        self.cv11_b1_2_ac = LeakyReLU(name="conv11_block1_2_leakyrelu")(x_)
        self.cv11_b1_add = Add(name="conv11_block1_add")([x, x_])
        # res block 5 2
        self.cv11_b2_1_cv = conv2d(512, 1, name="conv11_block2_1_conv")(x)
        self.cv11_b2_1_bn = BatchNormalization(name="conv11_block2_1_bn")(x_)
        self.cv11_b2_1_ac = LeakyReLU(name="conv11_block2_1_leakyrelu")(x_)
        self.cv11_b2_2_cv = conv2d(1024, 3, name="conv11_block2_2_conv")(x_)
        self.cv11_b2_2_bn = BatchNormalization(name="conv11_block2_2_bn")(x_)
        self.cv11_b2_2_ac = LeakyReLU(name="conv11_block2_2_leakyrelu")(x_)
        self.cv11_b2_add = Add(name="conv11_block2_add")([x, x_])
        # res block 5 3
        self.cv11_b3_1_cv = conv2d(512, 1, name="conv11_block3_1_conv")(x)
        self.cv11_b3_1_bn = BatchNormalization(name="conv11_block3_1_bn")(x_)
        self.cv11_b3_1_ac = LeakyReLU(name="conv11_block3_1_leakyrelu")(x_)
        self.cv11_b3_2_cv = conv2d(1024, 3, name="conv11_block3_2_conv")(x_)
        self.cv11_b3_2_bn = BatchNormalization(name="conv11_block3_2_bn")(x_)
        self.cv11_b3_2_ac = LeakyReLU(name="conv11_block3_2_leakyrelu")(x_)
        self.cv11_b3_add = Add(name="conv11_block3_add")([x, x_])
        # res block 5 4
        self.cv11_b4_1_cv = conv2d(512, 1, name="conv11_block4_1_conv")(x)
        self.cv11_b4_1_bn = BatchNormalization(name="conv11_block4_1_bn")(x_)
        self.cv11_b4_1_ac = LeakyReLU(name="conv11_block4_1_leakyrelu")(x_)
        self.cv11_b4_2_cv = conv2d(1024, 3, name="conv11_block4_2_conv")(x_)
        self.cv11_b4_2_bn = BatchNormalization(name="conv11_block4_2_bn")(x_)
        self.cv11_b4_2_ac = LeakyReLU(name="conv11_block4_2_leakyrelu")(x_)
        self.cv11_b4_add = Add(name="conv11_block4_add")([x, x_])
        #
        self.avg_pool = GlobalAveragePooling2D(name="avg_pool")(x)
        self.dense = Dense(1000, activation="softmax", name="predictions")(x)
            

    def call(self, inputs):

        #convolution
        x = self.cv1_cv(inputs)
        x = self.cv1_bn(x)
        x = self.cv1_ac(x)
        #convolution
        x = self.cv2_cv(x)
        x = self.cv2_bn(x)
        x = self.cv2_ac(x)
        #res block 1
        self.cv3_b1_1_cv(x)
        self.cv3_b1_1_bn(x_)
        self.cv3_b1_1_ac(x_)
        self.cv3_b1_2_cv(x_)
        self.cv3_b1_2_bn(x_)
        self.cv3_b1_2_ac(x_)
        x = self.cv3_b1_add([x, x_])
        #convolution
        x = self.cv4_cv(x)
        x = self.cv4_bn(x)
        x = self.cv4_ac(x)
        #res block 2 1
        self.cv5_b1_1_cv(x)
        self.cv5_b1_1_bn(x_)
        self.cv5_b1_1_ac(x_)
        self.cv5_b1_2_cv(x_)
        self.cv5.b1_2_bn(x_)
        self.cv5.b1_2_ac(x_)
        x = self.cv5_b1_add([x, x_])
        #res block 2 2
        self.cv5_b2_1_cv = conv2d(64, 1, name="conv5_block2_1_conv")(x)
        self.cv5_b2_1_bn = BatchNormalization(name="conv5_block2_1_bn")(x_)
        self.cv5_b2_1_ac = LeakyReLU(name="conv5_block2_1_leakyrelu")(x_)
        self.cv5_b2_2_cv = conv2d(128, 3, name="conv5_block2_2_conv")(x_)
        self.cv5_b2_2_bn = BatchNormalization(name="conv5_block2_2_bn")(x_)
        self.cv5_b2_2_ac = LeakyReLU(name="conv5_block2_2_leakyrelu")(x_)
        x = self.cv5_b2_add = Add(name="conv5_block2_add")([x, x_])
        #convolution
        x = self.cv6_cv = conv2d(256, 3, 2, name="conv6_conv")(x)
        x = self.cv6_bn = BatchNormalization(name="conv6_bn")(x)
        x = self.cv6_ac = LeakyReLU(name="conv6_leakyrelu")(x)
        # res block 3 1
        self.cv7_b1_1_cv = conv2d(128, 1, name="conv7_block1_1_conv")(x)
        self.cv7_b1_1_bn = BatchNormalization(name="conv7_block1_1_bn")(x_)
        self.cv7_b1_1_ac = LeakyReLU(name="conv7_block1_1_leakyrelu")(x_)
        self.cv7_b1_2_cv = conv2d(256, 3, name="conv7_block1_2_conv")(x_)
        self.cv7.b1_2_bn = BatchNormalization(name="conv7_block1_2_bn")(x_)
        self.cv7.b1_2_ac = LeakyReLU(name="conv7_block1_2_leakyrelu")(x_)
        x = self.cv7_b1_add = Add(name="conv7_block1_add")([x, x_])
        # res block 3 2
        self.cv7_b2_1_cv = conv2d(128, 1, name="conv7_block2_1_conv")(x)
        self.cv7_b2_1_bn = BatchNormalization(name="conv7_block2_1_bn")(x_)
        self.cv7_b2_1_ac = LeakyReLU(name="conv7_block2_1_leakyrelu")(x_)
        self.cv7_b2_2_cv = conv2d(256, 3, name="conv7_block2_2_conv")(x_)
        self.cv7_b2_2_bn = BatchNormalization(name="conv7_block2_2_bn")(x_)
        self.cv7_b2_2_ac = LeakyReLU(name="conv7_block2_2_leakyrelu")(x_)
        x = self.cv7_b2_add = Add(name="conv7_block2_add")([x, x_])
        # res block 3 3
        self.cv7_b3_1_cv = conv2d(128, 1, name="conv7_block3_1_conv")(x)
        self.cv7_b3_1_bn = BatchNormalization(name="conv7_block3_1_bn")(x_)
        self.cv7_b3_1_ac = LeakyReLU(name="conv7_block3_1_leakyrelu")(x_)
        self.cv7_b3_2_cv = conv2d(256, 3, name="conv7_block3_2_conv")(x_)
        self.cv7_b3_2_bn = BatchNormalization(name="conv7_block3_2_bn")(x_)
        self.cv7_b3_2_ac = LeakyReLU(name="conv7_block3_2_leakyrelu")(x_)
        x = self.cv7_b3_add = Add(name="conv7_block3_add")([x, x_])
        # res block 3 4
        self.cv7_b4_1_cv = conv2d(128, 1, name="conv7_block4_1_conv")(x)
        self.cv7_b4_1_bn = BatchNormalization(name="conv7_block4_1_bn")(x_)
        self.cv7_b4_1_ac = LeakyReLU(name="conv7_block4_1_leakyrelu")(x_)
        self.cv7_b4_2_cv = conv2d(256, 3, name="conv7_block4_2_conv")(x_)
        self.cv7_b4_2_bn = BatchNormalization(name="conv7_block4_2_bn")(x_)
        self.cv7_b4_2_ac = LeakyReLU(name="conv7_block4_2_leakyrelu")(x_)
        x = self.cv7_b4_add = Add(name="conv7_block4_add")([x, x_])
        # res block 3 5
        self.cv7_b5_1_cv = conv2d(128, 1, name="conv7_block5_1_conv")(x)
        self.cv7_b5_1_bn = BatchNormalization(name="conv7_block5_1_bn")(x_)
        self.cv7_b5_1_ac = LeakyReLU(name="conv7_block5_1_leakyrelu")(x_)
        self.cv7_b5_2_cv = conv2d(256, 3, name="conv7_block5_2_conv")(x_)
        self.cv7_b5_2_bn = BatchNormalization(name="conv7_block5_2_bn")(x_)
        self.cv7_b5_2_ac = LeakyReLU(name="conv7_block5_2_leakyrelu")(x_)
        x = self.cv7_b5_add = Add(name="conv7_block5_add")([x, x_])
        # res block 3 6
        self.cv7_b6_1_cv = conv2d(128, 1, name="conv7_block6_1_conv")(x)
        self.cv7_b6_1_bn = BatchNormalization(name="conv7_block6_1_bn")(x_)
        self.cv7_b6_1_ac = LeakyReLU(name="conv7_block6_1_leakyrelu")(x_)
        self.cv7_b6_2_cv = conv2d(256, 3, name="conv7_block6_2_conv")(x_)
        self.cv7_b6_2_bn = BatchNormalization(name="conv7_block6_2_bn")(x_)
        self.cv7_b6_2_ac = LeakyReLU(name="conv7_block6_2_leakyrelu")(x_)
        x = self.cv7_b6_add = Add(name="conv7_block6_add")([x, x_])
        # res block 3 7
        self.cv7_b7_1_cv = conv2d(128, 1, name="conv7_block7_1_conv")(x)
        self.cv7_b7_1_bn = BatchNormalization(name="conv7_block7_1_bn")(x_)
        self.cv7_b7_1_ac = LeakyReLU(name="conv7_block7_1_leakyrelu")(x_)
        self.cv7_b7_2_cv = conv2d(256, 3, name="conv7_block7_2_conv")(x_)
        self.cv7_b7_2_bn = BatchNormalization(name="conv7_block7_2_bn")(x_)
        self.cv7_b7_2_ac = LeakyReLU(name="conv7_block7_2_leakyrelu")(x_)
        x = self.cv7_b7_add = Add(name="conv7_block7_add")([x, x_])
        # res block 3 8
        self.cv7_b8_1_cv = conv2d(128, 1, name="conv7_block8_1_conv")(x)
        self.cv7_b8_1_bn = BatchNormalization(name="conv7_block8_1_bn")(x_)
        self.cv7_b8_1_ac = LeakyReLU(name="conv7_block8_1_leakyrelu")(x_)
        self.cv7_b8_2_cv = conv2d(256, 3, name="conv7_block8_2_conv")(x_)
        self.cv7_b8_2_bn = BatchNormalization(name="conv7_block8_2_bn")(x_)
        self.cv7_b8_2_ac = LeakyReLU(name="conv7_block8_2_leakyrelu")(x_)
        x = self.cv7_b8_add = Add(name="conv7_block8_add")([x, x_])
        #convolution
        x = self.cv8_cv = conv2d(512, 3, 2, name="conv8_conv")(x)
        x = self.cv8_bn = BatchNormalization(name="conv8_bn")(x)
        x = self.cv8_ac = LeakyReLU(name="conv8_leakyrelu")(x)
        # res block 4 1
        self.cv9_b1_1_cv = conv2d(256, 1, name="conv9_block1_1_conv")(x)
        self.cv9_b1_1_bn = BatchNormalization(name="conv9_block1_1_bn")(x_)
        self.cv9_b1_1_ac = LeakyReLU(name="conv9_block1_1_leakyrelu")(x_)
        self.cv9_b1_2_cv = conv2d(512, 3, name="conv9_block1_2_conv")(x_)
        self.cv9_b1_2_bn = BatchNormalization(name="conv9_block1_2_bn")(x_)
        self.cv9_b1_2_ac = LeakyReLU(name="conv9_block1_2_leakyrelu")(x_)
        x = self.cv9_b1_add = Add(name="conv9_block1_add")([x, x_])
        # res block 4 2
        self.cv9_b2_1_cv = conv2d(256, 1, name="conv9_block2_1_conv")(x)
        self.cv9_b2_1_bn = BatchNormalization(name="conv9_block2_1_bn")(x_)
        self.cv9_b2_1_ac = LeakyReLU(name="conv9_block2_1_leakyrelu")(x_)
        self.cv9_b2_2_cv = conv2d(512, 3, name="conv9_block2_2_conv")(x_)
        self.cv9_b2_2_bn = BatchNormalization(name="conv9_block2_2_bn")(x_)
        self.cv9_b2_2_ac = LeakyReLU(name="conv9_block2_2_leakyrelu")(x_)
        x = self.cv9_b2_add = Add(name="conv9_block2_add")([x, x_])
        # res block 4 3
        self.cv9_b3_1_cv = conv2d(256, 1, name="conv9_block3_1_conv")(x)
        self.cv9_b3_1_bn = BatchNormalization(name="conv9_block3_1_bn")(x_)
        self.cv9_b3_1_ac = LeakyReLU(name="conv9_block3_1_leakyrelu")(x_)
        self.cv9_b3_2_cv = conv2d(512, 3, name="conv9_block3_2_conv")(x_)
        self.cv9_b3_2_bn = BatchNormalization(name="conv9_block3_2_bn")(x_)
        self.cv9_b3_2_ac = LeakyReLU(name="conv9_block3_2_leakyrelu")(x_)
        self.cv9_b3_add = Add(name="conv9_block3_add")([x, x_])
        # res block 4 4
        self.cv9_b4_1_cv = conv2d(256, 1, name="conv9_block4_1_conv")(x)
        self.cv9_b4_1_bn = BatchNormalization(name="conv9_block4_1_bn")(x_)
        self.cv9_b4_1_ac = LeakyReLU(name="conv9_block4_1_leakyrelu")(x_)
        self.cv9_b4_2_cv = conv2d(512, 3, name="conv9_block4_2_conv")(x_)
        self.cv9_b4_2_bn = BatchNormalization(name="conv9_block4_2_bn")(x_)
        self.cv9_b4_2_ac = LeakyReLU(name="conv9_block4_2_leakyrelu")(x_)
        self.cv9_b4_add = Add(name="conv9_block4_add")([x, x_])
        # res block 4 5
        self.cv9_b5_1_cv = conv2d(256, 1, name="conv9_block5_1_conv")(x)
        self.cv9_b5_1_bn = BatchNormalization(name="conv9_block5_1_bn")(x_)
        self.cv9_b5_1_ac = LeakyReLU(name="conv9_block5_1_leakyrelu")(x_)
        self.cv9_b5_2_cv = conv2d(512, 3, name="conv9_block5_2_conv")(x_)
        self.cv9_b5_2_bn = BatchNormalization(name="conv9_block5_2_bn")(x_)
        self.cv9_b5_2_ac = LeakyReLU(name="conv9_block5_2_leakyrelu")(x_)
        self.cv9_b5_add = Add(name="conv9_block5_add")([x, x_])
        # res block 4 6
        self.cv9_b6_1_cv = conv2d(256, 1, name="conv9_block6_1_conv")(x)
        self.cv9_b6_1_bn = BatchNormalization(name="conv9_block6_1_bn")(x_)
        self.cv9_b6_1_ac = LeakyReLU(name="conv9_block6_1_leakyrelu")(x_)
        self.cv9_b6_2_cv = conv2d(512, 3, name="conv9_block6_2_conv")(x_)
        self.cv9_b6_2_bn = BatchNormalization(name="conv9_block6_2_bn")(x_)
        self.cv9_b6_2_ac = LeakyReLU(name="conv9_block6_2_leakyrelu")(x_)
        self.cv9_b6_add = Add(name="conv9_block6_add")([x, x_])
        # res block 4 7
        self.cv9_b7_1_cv = conv2d(256, 1, name="conv9_block7_1_conv")(x)
        self.cv9_b7_1_bn = BatchNormalization(name="conv9_block7_1_bn")(x_)
        self.cv9_b7_1_ac = LeakyReLU(name="conv9_block7_1_leakyrelu")(x_)
        self.cv9_b7_2_cv = conv2d(512, 3, name="conv9_block7_2_conv")(x_)
        self.cv9_b7_2_bn = BatchNormalization(name="conv9_block7_2_bn")(x_)
        self.cv9_b7_2_ac = LeakyReLU(name="conv9_block7_2_leakyrelu")(x_)
        self.cv9_b7_add = Add(name="conv9_block7_add")([x, x_])
        # res block 4 8
        self.cv9_b8_1_cv = conv2d(256, 1, name="conv9_block8_1_conv")(x)
        self.cv9_b8_1_bn = BatchNormalization(name="conv9_block8_1_bn")(x_)
        self.cv9_b8_1_ac = LeakyReLU(name="conv9_block8_1_leakyrelu")(x_)
        self.cv9_b8_2_cv = conv2d(512, 3, name="conv9_block8_2_conv")(x_)
        self.cv9_b8_2_bn = BatchNormalization(name="conv9_block8_2_bn")(x_)
        self.cv9_b8_2_ac = LeakyReLU(name="conv9_block8_2_leakyrelu")(x_)
        self.cv9_b8_add = Add(name="conv9_block8_add")([x, x_])
        #convolution
        self.cv10_cv = conv2d(1024, 3, 2, name="conv10_conv")(x)
        self.cv10_bn = BatchNormalization(name="conv10_bn")(x)
        self.cv10_ac = LeakyReLU(name="conv10_leakyrelu")(x)
        # res block 5 1
        self.cv11_b1_1_cv = conv2d(512, 1, name="conv11_block1_1_conv")(x)
        self.cv11_b1_1_bn = BatchNormalization(name="conv11_block1_1_bn")(x_)
        self.cv11_b1_1_ac = LeakyReLU(name="conv11_block1_1_leakyrelu")(x_)
        self.cv11_b1_2_cv = conv2d(1024, 3, name="conv11_block1_2_conv")(x_)
        self.cv11_b1_2_bn = BatchNormalization(name="conv11_block1_2_bn")(x_)
        self.cv11_b1_2_ac = LeakyReLU(name="conv11_block1_2_leakyrelu")(x_)
        self.cv11_b1_add = Add(name="conv11_block1_add")([x, x_])
        # res block 5 2
        self.cv11_b2_1_cv = conv2d(512, 1, name="conv11_block2_1_conv")(x)
        self.cv11_b2_1_bn = BatchNormalization(name="conv11_block2_1_bn")(x_)
        self.cv11_b2_1_ac = LeakyReLU(name="conv11_block2_1_leakyrelu")(x_)
        self.cv11_b2_2_cv = conv2d(1024, 3, name="conv11_block2_2_conv")(x_)
        self.cv11_b2_2_bn = BatchNormalization(name="conv11_block2_2_bn")(x_)
        self.cv11_b2_2_ac = LeakyReLU(name="conv11_block2_2_leakyrelu")(x_)
        self.cv11_b2_add = Add(name="conv11_block2_add")([x, x_])
        # res block 5 3
        self.cv11_b3_1_cv = conv2d(512, 1, name="conv11_block3_1_conv")(x)
        self.cv11_b3_1_bn = BatchNormalization(name="conv11_block3_1_bn")(x_)
        self.cv11_b3_1_ac = LeakyReLU(name="conv11_block3_1_leakyrelu")(x_)
        self.cv11_b3_2_cv = conv2d(1024, 3, name="conv11_block3_2_conv")(x_)
        self.cv11_b3_2_bn = BatchNormalization(name="conv11_block3_2_bn")(x_)
        self.cv11_b3_2_ac = LeakyReLU(name="conv11_block3_2_leakyrelu")(x_)
        self.cv11_b3_add = Add(name="conv11_block3_add")([x, x_])
        # res block 5 4
        self.cv11_b4_1_cv = conv2d(512, 1, name="conv11_block4_1_conv")(x)
        self.cv11_b4_1_bn = BatchNormalization(name="conv11_block4_1_bn")(x_)
        self.cv11_b4_1_ac = LeakyReLU(name="conv11_block4_1_leakyrelu")(x_)
        self.cv11_b4_2_cv = conv2d(1024, 3, name="conv11_block4_2_conv")(x_)
        self.cv11_b4_2_bn = BatchNormalization(name="conv11_block4_2_bn")(x_)
        self.cv11_b4_2_ac = LeakyReLU(name="conv11_block4_2_leakyrelu")(x_)
        self.cv11_b4_add = Add(name="conv11_block4_add")([x, x_])
        #
        self.avg_pool = GlobalAveragePooling2D(name="avg_pool")(x)
        self.dense = Dense(1000, activation="softmax", name="predictions")(x)
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

# %%

# %%

# %%

# %%

# %%
from tensorflow.keras.applications.vgg19 import VGG19
VGG19().summary()
# %%
from tensorflow.keras.applications.resnet50 import ResNet50
ResNet50().summary()
from tensorflow.keras.applications.resnet import ResNet101, ResNet152
ResNet101().summary()
ResNet152().summary()
