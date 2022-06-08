#%%
import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from src.yolov3 import model_utils, post_processing_utils
import src

#%%
flags.DEFINE_string("classes", "C:/Users/USER/Documents/GitHub/video_detection/labels/coco.names", "path to classes file")
flags.DEFINE_integer("size", 416, "resize images to")
flags.DEFINE_string("output", None, "path to output video")
flags.DEFINE_string("output_format", "XVID", "codec used in VideoWriter when saving video to file")
flags.DEFINE_integer("num_classes", 80, "number of classes in the model")

#%%
# def main(_argv):
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = 







