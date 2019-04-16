import tensorflow as tf
from keras import backend as K
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dot, Flatten, Input, Concatenate
import numpy
from PIL import Image, ImageDraw
import time
import data
import utils

# constants
conv_feature_maps = 112
dense_size = 384
patch_size = 9
max_disp = 60
image_name = "pattern1"
match_th = 0.2
error_threshold = 12
cpu_only = True

# fix random seed for reproducibility
numpy.random.seed(7)

# set GPU use
if cpu_only:
    utils.set_gpu(False)

# load convolved images
left = numpy.load('np_data/' + image_name + '_left_conv.npy')
right = numpy.load('np_data/' + image_name + '_right_conv.npy')

# compute disparity map
data.disp_map_from_conv_fst(left, right, patch_size, max_disp, match_th, conv_feature_maps, image_name + "_disp_fst")

##############################################################################################
#left, right = data.get_random_sample("../samples/cones/", 11, 8, 12, 4, 0, 5, 68)
