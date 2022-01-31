#%%
import os

#%%
def get_hyper_params():
    hyper_params = {
        "img_size" : 416,
        "nms_boxes_per_class" : 50,
        "score_thresh" : 0.5,
        "nms_thresh" : 0.5,
        "coord_tune" : 5.,
        "noobj_tune" : .5,
        "batch_size": 8,
        "iters" : 20000,
        "attempts" : 100,
        "mAP_threshold" : 0.5,
    }
    return hyper_params

#%%
def save_dict_to_file(dic,dict_dir):
    f = open(dict_dir + '.txt', 'w')
    f.write(str(dic))
    f.close()

#%%
def generate_save_dir(atmp_dir, hyper_params):
    atmp_dir = atmp_dir + '/atmp'

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

    os.makedirs(atmp_dir + '/rpn_weights')
    os.makedirs(atmp_dir + '/dtn_weights')

    return atmp_dir