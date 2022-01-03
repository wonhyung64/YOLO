#%%
import tensorflow as tf
from tensorflow.keras.layers import Input, UpSampling2D, Conv2D, Concatenate
from darknet53 import Darknet53
from tensorflow.keras.models import Model
#%%
model = Darknet53()
inputs = Input(shape=(416,416,3))
input_shape = (None, 416, 416, 3)
model.build(input_shape)
model.call(inputs)

x1 = model.get_layer("conv11_block4_add").output
x2 = model.get_layer("conv9_block8_add").output
x3 = model.get_layer("conv7_block8_add").output

x3_ = UpSampling2D(2)(x2)
x3_ = Conv2D(256, 1)(x3_)
x3 = Concatenate()([x3, x3_])

x2_ = UpSampling2D(2)(x1)
x2_ = Conv2D(512, 1)(x2_)
x2 = Concatenate()([x2, x2_])

fpn = Model(inputs=model.get_layer("conv1_conv").input, outputs=[x1, x2, x3])

fpn.summary()

#%%
class FPN(Model):
    def __init__(self):
        super(FPN, self).__init__(self)
        self.model = Darknet53()
        self.img_shape = (None, 416, 416, 3)

    def call(self, inputs):
        self.model.build(self.img_shape)
        self.model.call(inputs)
        model_input = self.model.get_layer("conv1_conv").input
        model_output = [self.model.get_layer("conv7_block8_add").output, self.model.get_layer("conv9_block8_add").output, self.model.get_layer("conv11_block4_add").output]
        return Model(inputs=model_input, outputs=model_output)
# %%

model = FPN()
model.summary()
model.build((None, 416, 416, 3))
model.call(inputs)
# %%
