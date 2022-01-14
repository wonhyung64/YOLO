#%%
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import cm
from PIL import Image, ImageDraw
import numpy as np

# %%
image_dir = r"C:\Users\USER\Desktop\000000033897.jpg"
save_dir = r"C:\Users\USER\Desktop\000000033897_2.jpg"
image = Image.open(image_dir)
imgArray = np.array(image)
img = tf.convert_to_tensor(imgArray)
img = tf.image.resize(img, (416, 416))

image = tf.keras.preprocessing.image.array_to_img(img)
#%%
# Draw some lines
draw = ImageDraw.Draw(image)
y_start = 0
y_end = image.height
step_size = int(image.width / 13)
for x in range(0, image.width, step_size):
    line = ((x, y_start), (x, y_end))
    draw.line(line, fill=(255, 255, 255))
x_start = 0
x_end = image.width
for y in range(0, image.height, step_size):
    line = ((x_start, y), (x_end, y))
    draw.line(line, fill=(255,255,255), )
del draw
plt.imshow(image)
plt.savefig(save_dir)
# %%
