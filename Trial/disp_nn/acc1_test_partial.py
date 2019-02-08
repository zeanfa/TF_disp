from keras import backend as K
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dot, Flatten, Input, Concatenate
import numpy
import data
from PIL import Image, ImageDraw
import time

numpy.set_printoptions(threshold=numpy.inf)


# too long

def get_batch_from_image(folder_name, patch_size, max_disp, image_name, model):
    left_pic = Image.open(folder_name + "im0.png").convert("L")
    right_pic = Image.open(folder_name + "im1.png").convert("L")  
    left_pix = numpy.atleast_3d(left_pic)
    right_pix = numpy.atleast_3d(right_pic)
    width, height = left_pic.size
    print(width, height)
    disp_pix = numpy.zeros((height,width))
    left_f = lambda x: (x - left_pix.mean())/left_pix.std()
    norm_left = left_f(left_pix)
    right_f = lambda x: (x - right_pix.mean())/right_pix.std()
    norm_right = right_f(right_pix)
    
    best_disp_num = 0
    timestamp = time.time()
    
    for i in range(int(patch_size/2), height - int(patch_size/2)):
        pos_num = 0
        best_disp_num = 0
        left_disp_batch = []
        right_disp_batch = []
        for j in range(max_disp + int(patch_size/2), width - int(patch_size/2)):
            for d in range(0, max_disp):
                left_patch = norm_left[(i - int(patch_size/2)) : (i + int(patch_size/2) + 1),
                                       (j - int(patch_size/2)) : (j + int(patch_size/2) + 1)]
                right_patch = norm_right[(i - int(patch_size/2)) : (i + int(patch_size/2) + 1),
                                         (j - int(patch_size/2) - d) : (j + int(patch_size/2) - d + 1)]
                left_disp_batch.append(left_patch)
                right_disp_batch.append(right_patch)
                
            pos_num += 1
        prediction = model.predict([numpy.array(left_disp_batch),numpy.array(right_disp_batch)])
        best_disp_num = numpy.argmax(numpy.squeeze(prediction).reshape(max_disp, pos_num), axis = 0)
        disp_pix[i, max_disp + int(patch_size/2) : width - int(patch_size/2)] = best_disp_num*255/max_disp
        print("\rtime ", time.time()-timestamp)
        print("\ri", i)

    # Creates PIL image
    img = Image.fromarray(disp_pix.astype('uint8'), mode = 'L')
    img.save(image_name + ".png", "PNG")

# convolves images using upper model
def convolve_images(folder_name, patch_size, conv_feature_maps, conv_model):
    print("begin convolution")
    height = 375
    width = 450
    left_patches, right_patches = data.get_image_in_patches(folder_name, patch_size)
    conv_left_patches = numpy.zeros((height,width, conv_feature_maps))
    conv_right_patches = numpy.zeros((height,width, conv_feature_maps))
    timestamp = time.time()
    for i in range(int(patch_size/2), height - int(patch_size/2)):
        for j in range(int(patch_size/2), width - int(patch_size/2)):
            prediction = conv_model.predict([[left_patches[i, j]],[right_patches[i, j]]])
            conv_left_patches[i, j, ::] = prediction[0]
            conv_right_patches[i, j, ::] = prediction[1]
        print("\rtime ", time.time()-timestamp, " row ", i, end = "\r")
    print("total time ", time.time()-timestamp)
    return conv_left_patches, conv_right_patches

#computes disparity using convolved pictures and dense layers, quick
def disp_map_from_conv(left_conv, right_conv, patch_size, max_disp, dense_model, image_name):
    print("begin disparity computation")
    height = 375
    width = 450
    disp_pix = numpy.zeros((height,width))
    timestamp = time.time()
    for i in range(int(patch_size/2), height - int(patch_size/2)):
        for j in range(max_disp + int(patch_size/2), width - int(patch_size/2)):
            right_conv_pos = numpy.expand_dims(right_conv[i, j - max_disp : j], axis = 1)
            left_conv_pos = numpy.array([numpy.expand_dims(left_conv[i,j], axis = 0)]*max_disp)
            dense_predictions = dense_model.predict([left_conv_pos, right_conv_pos])
            disp_pix[i,j] = (255*(max_disp - numpy.argmax(numpy.squeeze(dense_predictions), axis = 0)))/max_disp
        print("\rtime ", time.time()-timestamp, " i ", i, end = "\r")
    print("total time ", time.time()-timestamp)
    # Creates PIL image
    img = Image.fromarray(disp_pix.astype('uint8'), mode = 'L')
    img.save(image_name + ".png", "PNG")
    
# constants
training_size = 288000
conv_feature_maps = 112
patch_size = 11
max_disp = 63

# fix random seed for reproducibility
numpy.random.seed(7)

# create model
left_input = Input(shape=(11, 11, 1, ))
left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (left_input)
left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (left_conv)
left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (left_conv)
left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (left_conv)
left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (left_conv)
left_flatten = Flatten(name = "left_flatten_layer")(left_conv)

right_input = Input(shape=(11, 11, 1, ))
right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (right_input)
right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (right_conv)
right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (right_conv)
right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (right_conv)
right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (right_conv)
right_flatten = Flatten(name = "right_flatten_layer")(right_conv)

conc_layer = Concatenate(name = "d1")([left_flatten, right_flatten])
dense_layer = Dense(384, activation="relu", name = "d2")(conc_layer)
dense_layer = Dense(384, activation="relu", name = "d3")(dense_layer)
output_layer = Dense(1, activation="sigmoid", name = "d4")(dense_layer)

model = Model(inputs=[left_input, right_input], outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.load_weights("weights/acc1_weights1.h5")
model.save_weights("weights/acc1_weights1_partial.h5")

left_name = 'left_flatten_layer'
right_name = 'right_flatten_layer'

conv_model = Model(inputs=model.input, outputs=[model.get_layer(left_name).output, model.get_layer(right_name).output])

dense_input1 = Input(shape=(1, conv_feature_maps ))
dense_input2 = Input(shape=(1, conv_feature_maps ))
conc_layer = Concatenate(name = "d1")([dense_input1, dense_input2])
dense_layer = Dense(384, activation="relu", name = "d2")(conc_layer)
dense_layer = Dense(384, activation="relu", name = "d3")(dense_layer)
dense_output = Dense(1, activation="sigmoid", name = "d4")(dense_layer)
dense_model = Model(inputs = [dense_input1, dense_input2], outputs = dense_output)
dense_model.load_weights("weights/acc1_weights1_partial.h5", by_name = True)

# test

left = numpy.load('np_data/left_conv.npy')
right = numpy.load('np_data/right_conv.npy')
disp_map_from_conv(left, right, patch_size, max_disp, dense_model, "disp2")

##############################################################################################
#left, right = convolve_images("../samples/cones/", patch_size, conv_feature_maps, conv_model)
#numpy.save('left_conv', left)
#numpy.save('right_conv', right)
#left = numpy.zeros((375, 450, 112))
#right = numpy.zeros((375, 450, 112))
#print(left.shape)

#left, right = data.get_random_sample("../samples/cones/", 11, 8, 12, 4, 0, 5, 68)
#conv_result = conv_model.predict([[left], [right]])
#print(dense_model.predict([[conv_result[0]],[conv_result[1]]]))

#plot_model(model, show_shapes=True, to_file='model.png')

#left, right = data.get_random_sample("../samples/cones/", 11, 8, 10, 4, 1, 120, 100)


