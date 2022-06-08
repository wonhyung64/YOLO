#%%

import os
import tensorflow as tf
#%%
def get_hyper_params():
    hyper_params = {
        "img_size" : 416,
        "nms_boxes_per_class" : 50,
        "coord_tune" : .5,
        "noobj_tune" : 5.,
        "batch_size": 4,
        "iters" : 200000,
        "attempts" : 100,
        "mAP_threshold" : 0.5,
        "dataset_name" : "voc07",
        "total_labels" : 20,
        "focal" : False,
    }
    return hyper_params

#%%
def save_dict_to_file(dic,dict_dir):
    f = open(dict_dir + '.txt', 'w')
    f.write(str(dic))
    f.close()

#%%
def generate_save_dir(atmp_dir, hyper_params):
    atmp_dir = atmp_dir + '/yolo_atmp'

    i = 1
    tmp = True
    while tmp :
        if os.path.isdir(atmp_dir + '/' + str(i)) : 
            i+= 1
        else: 
            os.makedirs(atmp_dir + '/' + str(i))
            print("Generated atmp" + str(i))
            tmp = False
    atmp_dir = atmp_dir + '/' + str(i)

    os.makedirs(atmp_dir + '/yolo_weights')
    os.makedirs(atmp_dir + '/yolo_output')

    return atmp_dir

#%%
# def get_box_prior():
#     box_prior = tf.cast([[10,13], [16,30], [33,23], [30,61], [62,45], [59,119], [116,90],  [156,198],  [373,326]], dtype = tf.float32)
#     prior1, prior2 = tf.split(box_prior, 2, -1)
#     box_prior = tf.concat([prior2, prior1], axis=-1)
#     return box_prior
def get_box_prior():
    # box_prior = tf.cast([[10,13], [16,30], [33,23], [30,61], [62,45], [59,119], [116,90],  [156,198],  [373,326]], dtype = tf.float32)
#     box_prior = tf.cast([
#     [44.876007, 33.94767],
#     [111.59584, 55.58249],
#     [110.780045, 148.88736],
#     [198.02014, 87.824524],
#     [312.71027, 121.00673],
#     [212.9126, 209.1222],
#     [197.57387, 344.69098],
#     [353.16132, 235.58124],
#     [361.2718, 371.65466]
# ], dtype = tf.float32)
    box_prior = tf.cast([
    [34.62897487, 21.52063238],
    [50.31485668, 56.76226806],
    [95.32252384, 30.32935004],
    [119.00188798, 74.40600956],
    [220.72284, 80.4646506],
    [114.71482119, 157.40415637],
    [297.71821858, 165.87902991],
    [193.59411315, 293.148731  ],
    [356.74360478, 343.19366774]
], dtype = tf.float32)
    return box_prior