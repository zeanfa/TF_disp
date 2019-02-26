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
patch_size = 11
max_disp = 50
image_name = "kron_big"
error_threshold = 12

# fix random seed for reproducibility
numpy.random.seed(7)

# use GPU with CPU
utils.set_gpu(True)

# create full model
left_input = Input(shape=(patch_size, patch_size, 1, ))
left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (left_input)
left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (left_conv)
left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (left_conv)
left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (left_conv)
left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (left_conv)
left_flatten = Flatten(name = "left_flatten_layer")(left_conv)

right_input = Input(shape=(patch_size, patch_size, 1, ))
right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (right_input)
right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (right_conv)
right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (right_conv)
right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (right_conv)
right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (right_conv)
right_flatten = Flatten(name = "right_flatten_layer")(right_conv)

conc_layer = Concatenate(name = "d1")([left_flatten, right_flatten])
dense_layer = Dense(dense_size, activation="relu", name = "d2")(conc_layer)
dense_layer = Dense(dense_size, activation="relu", name = "d3")(dense_layer)
output_layer = Dense(1, activation="sigmoid", name = "d4")(dense_layer)

model = Model(inputs=[left_input, right_input], outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.load_weights("weights/acc1_weights1.h5")
model.save_weights("weights/acc1_weights1_partial.h5")

left_name = 'left_flatten_layer'
right_name = 'right_flatten_layer'

# create convolutional part model
conv_model = Model(inputs=model.input, outputs=[model.get_layer(left_name).output, model.get_layer(right_name).output])

# convolve images
left, right = data.convolve_images("../samples/" + image_name + "/", patch_size, conv_feature_maps, conv_model)
numpy.save('np_data/' + image_name + '_left_conv', left)
numpy.save('np_data/' + image_name + '_right_conv', right)

# switch to CPU only
utils.set_gpu(False)

# create dense part model
dense_input1 = Input(shape=(1, conv_feature_maps ))
dense_input2 = Input(shape=(1, conv_feature_maps ))
conc_layer = Concatenate(name = "d1")([dense_input1, dense_input2])
dense_layer = Dense(dense_size, activation="relu", name = "d2")(conc_layer)
dense_layer = Dense(dense_size, activation="relu", name = "d3")(dense_layer)
dense_output = Dense(1, activation="sigmoid", name = "d4")(dense_layer)
dense_model = Model(inputs = [dense_input1, dense_input2], outputs = dense_output)
dense_model.load_weights("weights/acc1_weights1_partial.h5", by_name = True)

# compute disparity map
#left = numpy.load('np_data/' + image_name + '_left_conv.npy')
#right = numpy.load('np_data/' + image_name + '_right_conv.npy')
data.disp_map_from_conv(left, right, patch_size, max_disp, conv_feature_maps, dense_model, image_name + "_disp")

##############################################################################################
#data.comp_error_in_area("../samples/teddy/disp0", "../samples/teddy/disp_gpu", patch_size, max_disp, error_threshold)
#left, right = data.get_random_sample("../samples/cones/", 11, 8, 12, 4, 0, 5, 68)
