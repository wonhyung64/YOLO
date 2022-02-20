#%%
import os
import time
import tensorflow as tf
import numpy as np

from PIL import Image
from tqdm import tqdm

import model_utils, data_utils, utils, preprocessing_utils, postprocessing_utils, test_utils, data_utils

from PIL import ImageFont
from PIL import ImageDraw
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#%%
hyper_params = utils.get_hyper_params()

hyper_params["batch_size"] = batch_size = 1
img_size = (hyper_params["img_size"], hyper_params["img_size"])
dataset_name = hyper_params["dataset_name"]

box_prior = utils.get_box_prior()
box_prior = np.array(box_prior)
anchors1 = box_prior[6:9] # 13,13
anchors2 = box_prior[3:6] # 26, 26
anchors3 = box_prior[0:3] # 52, 52
anchors = [anchors1, anchors2, anchors3]

#%%
weights_dir = os.getcwd() + "/yolo_atmp"
# weights_no_dir = weights_dir + "/" + os.listdir(weights_dir)[-2]
weights_dir = weights_dir + "/" + os.listdir(weights_dir)[-1]

input_shape = (416, 416, 3)
yolo_model = model_utils.yolo_v3(input_shape, hyper_params)
yolo_model.load_weights(weights_dir + '/yolo_weights/weights')

# yolo_model_no = model_utils.yolo_v3(input_shape, hyper_params)
# yolo_model_no.load_weights(weights_no_dir + "/yolo_weights/weights")


#%%
dataset, labels = data_utils.fetch_dataset(dataset_name, "test", img_size)
dataset = dataset.map(lambda x, y, z: preprocessing_utils.preprocessing(x, y, z))
dataset = dataset.repeat().batch(batch_size)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
dataset = iter(dataset)

total_time = []
mAP = []
hyper_params

fnt = ImageFont.truetype(r"C:\Users\USER\AppData\Local\Microsoft\Windows\Fonts\FiraCode-Regular.ttf", 15)
# colors= tf.random.uniform((21, 4), maxval=256, dtype=tf.int32)
colors = tf.convert_to_tensor(
    [[ 95,  12, 160,  57],
       [169, 116, 190,  20],
       [178, 157, 165, 108],
       [ 29,  82, 190, 207],
       [ 49,  62, 157, 124],
       [  8, 220,  93, 225],
       [217, 113, 227, 255],
       [155, 109,  91, 197],
       [154, 178,  86,  75],
       [ 72, 198, 101, 127],
       [ 42, 143,  90,  95],
       [ 32, 173,  82, 121],
       [209, 132, 166, 212],
       [210, 213, 125,  16],
       [166, 189,  10, 178],
       [ 20, 195, 249,  86],
       [203, 183, 140, 233],
       [134,  45, 240, 228],
       [125, 112, 228,  88],
       [ 22, 208, 110,  96],
       [232, 174,  36, 179]])

fig = plt.figure(figsize=(42,168))
gs = gridspec.GridSpec(1,4)
gs.update(wspace=0.05, hspace=0.05)
j = 0
try_num = 0
progress_bar = tqdm(range(100))
for _ in progress_bar:
    image, gt_boxes, gt_labels = next(dataset)
    if try_num in [3, 6, 9, 21]:
    # if try_num == 35:
        start_time = time.time()
        yolo_outputs = yolo_model(image)
        # yolo_outputs = yolo_model_no(image)
        final_bboxes, final_labels, final_scores = postprocessing_utils.Decode(yolo_outputs)
        time_ = float(time.time() - start_time)*1000
        AP = test_utils.calculate_AP(final_bboxes, final_labels, gt_boxes, gt_labels, hyper_params)
        print(try_num)
        image = tf.squeeze(image, axis=0)
        image = tf.keras.preprocessing.image.array_to_img(image)
        width, height = image.size
        draw = ImageDraw.Draw(image)

        y1 = final_bboxes[0][...,0]
        x1 = final_bboxes[0][...,1]
        y2 = final_bboxes[0][...,2]
        x2 = final_bboxes[0][...,3]

        denormalized_box = tf.round(tf.stack([y1, x1, y2, x2], axis=-1))


        for index, bbox in enumerate(denormalized_box):
            y1, x1, y2, x2 = tf.split(bbox, 4, axis = -1)

            final_labels_ = tf.reshape(final_labels[0], shape=(final_labels.shape[1],))
            final_scores_ = tf.reshape(final_scores[0], shape=(final_labels.shape[1],))
            label_index = int(final_labels_[index])
            color = tuple(colors[label_index].numpy())
            label_text = "{0} {1:0.3f}".format(labels[label_index], final_scores_[index])
            draw.text((x1 + 4, y1 + 4), label_text, fill=color, font=fnt)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=5)


        ax = plt.subplot(gs[0, j])
        ax.imshow(image)
        ax.set_xticks([]), ax.set_yticks([])
        # ax.set_aspect("equal")
        j += 1

        total_time.append(time_)
        mAP.append(AP)
    try_num += 1
plt.show()

print("mAP: %.2f" % (tf.reduce_mean(mAP)))
print("Time taken: %.2fms" % (tf.reduce_mean(total_time)))
#%%
def draw_custom_img(img_dir):
    image = Image.open(img_dir)
    image_ = np.array(image)
    image_ = tf.convert_to_tensor(image_)
    image_ = tf.image.resize(image_, img_size)/ 255
    img = tf.expand_dims(image_, axis=0)
    yolo_outputs = yolo_model(img)
    final_bboxes, final_labels, final_scores = postprocessing_utils.Decode(yolo_outputs)
    test_utils.draw_yolo_output(img, final_bboxes, labels, final_labels, final_scores)
#%%

colors_violet = tf.constant([[222, 1, 189, 59]])
colors_pink = tf.constant([[214,70, 119, 170]])
colors_skyblue = tf.constant([[54, 177, 235, 190]])
colors_kaki = tf.constant([[101, 158, 42, 42]])
colors = tf.concat([colors_skyblue, colors_pink, colors_violet, colors_kaki], axis=0)
colos_white = tf.constant([193, 181, 198, 136])

# colors= tf.random.uniform((4, 4), maxval=256, dtype=tf.int32)
fig = plt.figure(figsize=(76,44))
gs = gridspec.GridSpec(4, 4)
gs.update(wspace=0.05, hspace=0.05)
i = 0
j = 0

while True:
    image, gt_boxes, gt_labels = next(dataset)
    if try_num in file_num_dark: 

        image = tf.squeeze(image, axis=0)
        image = tf.keras.preprocessing.image.array_to_img(image)
        width, height = image.size
        draw = ImageDraw.Draw(image)

        y1 = gt_boxes[0][...,0] * height
        x1 = gt_boxes[0][...,1] * width
        y2 = gt_boxes[0][...,2] * height
        x2 = gt_boxes[0][...,3] * width

        gt_boxes = tf.round(tf.stack([y1, x1, y2, x2], axis=-1))

        y1 = final_bboxes[0][...,0] * height
        x1 = final_bboxes[0][...,1] * width
        y2 = final_bboxes[0][...,2] * height
        x2 = final_bboxes[0][...,3] * width

        denormalized_box = tf.round(tf.stack([y1, x1, y2, x2], axis=-1))

        gt_color = tf.constant([75,142,80,72], dtype=tf.int32)
        fnt = ImageFont.truetype(r"C:\Users\USER\AppData\Local\Microsoft\Windows\Fonts\FiraCode-Regular.ttf", 50)

        # for index, bbox in enumerate(gt_boxes):
        #     color = tf.constant([193, 181, 198, 136])
        #     y1, x1, y2, x2 = tf.split(bbox, 4, axis = -1)

        #     gt_labels_ = tf.reshape(gt_labels[0], shape=(gt_labels.shape[1]))
        #     label_index = int(gt_labels_[index])
        #     color = tuple(gt_color.numpy())
        #     label_text = labels[label_index]
        #     draw.text((x1 + 4, y1 - 60), label_text, fill=tuple(tf.constant([54, 177, 235, 190]).numpy()), font=fnt)
        #     draw.rectangle((x1, y1, x2, y2), outline=tuple(tf.constant([54, 177, 235, 190]).numpy()), width=10)


        for index, bbox in enumerate(denormalized_box):
            y1, x1, y2, x2 = tf.split(bbox, 4, axis = -1)

            final_labels_ = tf.reshape(final_labels[0], shape=(200,))
            final_scores_ = tf.reshape(final_scores[0], shape=(200,))
            label_index = int(final_labels_[index])
            color = tuple(colors[label_index].numpy())
            label_text = "{0} {1:0.3f}".format(labels[label_index], final_scores_[index])
            draw.text((x1 + 4, y1 - 60), label_text, fill=color, font=fnt)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=10)

        ax = plt.subplot(gs[i, j])
        ax.imshow(image)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_aspect("equal")

        if i == 0: 
            if j == 0 : j += 1
            else:
                i+=1
                j = 0
        else: j += 1

    try_num += 1

plt.show()

#%% custom img result

colors= tf.random.uniform((21, 4), maxval=256, dtype=tf.int32)

custom_dir = "C:/won"
file_lst = ["sample2", "sample4", "sample5", "sample7", "sample1", "sample3", "sample8"]
file_lst1 = ["sample4", "sample5", "sample8"]
fnt = ImageFont.truetype(r"C:\Users\USER\AppData\Local\Microsoft\Windows\Fonts\FiraCode-Regular.ttf", 40)

fig = plt.figure(figsize=(100,100))
gs = gridspec.GridSpec(1,2)
gs.update(wspace=0.05, hspace=0.05)
j = 0
for i in range(len(file_lst1)):
        

    image = tf.squeeze(img, axis=0)
    image = tf.keras.preprocessing.image.array_to_img(image)
    width, height = image.size
    draw = ImageDraw.Draw(image)

    y1 = final_bboxes[0][...,0] * height
    x1 = final_bboxes[0][...,1] * width
    y2 = final_bboxes[0][...,2] * height
    x2 = final_bboxes[0][...,3] * width

    denormalized_box = tf.round(tf.stack([y1, x1, y2, x2], axis=-1))


    for index, bbox in enumerate(denormalized_box):
        y1, x1, y2, x2 = tf.split(bbox, 4, axis = -1)

        final_labels_ = tf.reshape(final_labels[0], shape=(final_labels.shape[1],))
        final_scores_ = tf.reshape(final_scores[0], shape=(final_labels.shape[1],))
        label_index = int(final_labels_[index])
        color = tuple(colors[label_index].numpy())
        label_text = "{0} {1:0.3f}".format(labels[label_index], final_scores_[index])
        draw.text((x1 + 4, y1 - 60), label_text, fill=color, font=fnt)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=10)


    ax = plt.subplot(gs[0, j])
    ax.imshow(image)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_aspect("equal")
    j += 1

plt.show()
#%%
file_lst = ["sample2", "sample4", "sample5", "sample7", "sample1", "sample3", "sample8"]

file_lst[0]

custom_dir = "C:/won"
file_lst = ["sample2", "sample4", "sample5", "sample7", "sample1", "sample3", "sample8"]
file_lst1 = ["sample4", "sample5", "sample8"]
fnt = ImageFont.truetype(r"C:\Users\USER\AppData\Local\Microsoft\Windows\Fonts\FiraCode-Regular.ttf", 40)

custom_img = f"{custom_dir}/sample2.jpg"
image = Image.open(custom_img)
plt.figure(figsize=(20, 20))
plt.xticks([])
plt.yticks([])
plt.imshow(image)

image = np.array(image)
image_ = tf.convert_to_tensor(image)
image_ = tf.image.resize(image_, (500,500))/ 255
img = tf.expand_dims(image_, axis=0)

rpn_reg_output, rpn_cls_output, feature_map = rpn_model(img)
roi_bboxes, _ = postprocessing_utils.RoIBBox(rpn_reg_output, rpn_cls_output, anchors, hyper_params)
pooled_roi = postprocessing_utils.RoIAlign(roi_bboxes, feature_map, hyper_params)
dtn_reg_output, dtn_cls_output = dtn_model(pooled_roi)
final_bboxes, final_labels, final_scores = postprocessing_utils.Decode(dtn_reg_output, dtn_cls_output, roi_bboxes, hyper_params)

image = tf.keras.preprocessing.image.array_to_img(image)
width, height = image.size
draw = ImageDraw.Draw(image)


y1 = final_bboxes[0][...,0] * height
x1 = final_bboxes[0][...,1] * width
y2 = final_bboxes[0][...,2] * height
x2 = final_bboxes[0][...,3] * width

denormalized_box = tf.round(tf.stack([y1, x1, y2, x2], axis=-1))


for index, bbox in enumerate(denormalized_box):
    y1, x1, y2, x2 = tf.split(bbox, 4, axis = -1)

    final_labels_ = tf.reshape(final_labels[0], shape=(200,))
    final_scores_ = tf.reshape(final_scores[0], shape=(200,))
    label_index = int(final_labels_[index])
    color = tuple(colors[label_index].numpy())
    label_text = "{0} {1:0.3f}".format(labels[label_index], final_scores_[index])
    draw.text((x1 + 4, y1 - 60), label_text, fill=color, font=fnt)
    draw.rectangle((x1, y1, x2, y2), outline=color, width=10)

plt.figure(figsize=(20, 20))
plt.xticks([])
plt.yticks([])
plt.imshow(image)
# %%

def draw_custom_img(img_dir):
    image = Image.open(img_dir)
    image_ = np.array(image)
    image_ = tf.convert_to_tensor(image_)
    image_ = tf.image.resize(image_, img_size)/ 255
    img = tf.expand_dims(image_, axis=0)
    yolo_outputs = yolo_model(img)
    final_bboxes, final_labels, final_scores = postprocessing_utils.Decode(yolo_outputs)
    test_utils.draw_yolo_output(img, final_bboxes, labels, final_labels, final_scores)

img_dir = "C:/won/sample2.jpg"