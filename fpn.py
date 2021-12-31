#%%
import tensorflow as tf
from tensorflow.keras.layers import Input
from darknet53 import Darknet53
# %%
model = Darknet53()
inputs = Input(shape=(256,256,3))
input_shape = (None, 256, 256, 3)
model.build(input_shape)
model.call(inputs)
model.summary()

# %%
model.get_layer("conv7_block8_add").output
model.get_layer("conv9_block8_add").output
model.get_layer("conv11_block4_add").output
#%%