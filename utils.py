#%%

import os
import tensorflow as tf
#%%
def get_hyper_params():
    hyper_params = {
        "img_size" : 416,
        "nms_boxes_per_class" : 50,
        "score_thresh" : 0.5,
        "nms_thresh" : 0.5,
        "coord_tune" : .5,
        "noobj_tune" : 5.,
        "batch_size": 2,
        "iters" : 400000,
        "attempts" : 100,
        "mAP_threshold" : 0.5,
        "dataset_name" : "voc07",
        "total_labels" : 20,
        "focal" : False,
        "lr": 1e-5
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
def get_box_prior():
    box_prior = tf.cast([[10,13], [16,30], [33,23], [30,61], [62,45], [59,119], [116,90],  [156,198],  [373,326]], dtype = tf.float32)
    return box_prior