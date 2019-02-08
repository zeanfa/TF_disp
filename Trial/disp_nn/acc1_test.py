from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dot, Flatten, Input, Concatenate
import numpy
import data
from PIL import Image, ImageDraw
import time

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
    
    for i in range(int(patch_size/2), height - int(patch_size/2)):
        pos_num = 0
        best_disp_num = 0
        left_disp_batch = []
        right_disp_batch = []
        timestamp = time.time()
        for j in range(max_disp + int(patch_size/2), width - int(patch_size/2)):
            for d in range(0, max_disp):
                left_patch = norm_left[(i - int(patch_size/2)) : (i + int(patch_size/2) + 1),
                                       (j - int(patch_size/2)) : (j + int(patch_size/2) + 1)]
                right_patch = norm_right[(i - int(patch_size/2)) : (i + int(patch_size/2) + 1),
                                         (j - int(patch_size/2) - d) : (j + int(patch_size/2) - d + 1)]
                left_disp_batch.append(left_patch)
                right_disp_batch.append(right_patch)
                
            j += 1
            pos_num += 1
        prediction = model.predict([numpy.array(left_disp_batch),numpy.array(right_disp_batch)])
        #print(len(left_disp_batch))
        #print(j, pos_num)
        #print(numpy.argmax(numpy.squeeze(prediction).reshape(max_disp, pos_num), axis = 0).shape)
        best_disp_num = numpy.argmax(numpy.squeeze(prediction).reshape(max_disp, pos_num), axis = 0)
        #print(best_disp_num)
        disp_pix[i, max_disp + int(patch_size/2) : width - int(patch_size/2)] = best_disp_num*255/max_disp
        i += 1
        print("time ", time.time()-timestamp)
        print("i", i)

    # Creates PIL image
    img = Image.fromarray(disp_pix, 'L')
    img.save(image_name + ".png", "PNG")
    
# constants
training_size = 288000
conv_feature_maps = 112

# fix random seed for reproducibility
numpy.random.seed(7)

# create model
left_input = Input(shape=(11, 11, 1, ))
left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (left_input)
left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (left_conv)
left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (left_conv)
left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (left_conv)
left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (left_conv)
left_flatten = Flatten()(left_conv)

right_input = Input(shape=(11, 11, 1, ))
right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (right_input)
right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (right_conv)
right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (right_conv)
right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (right_conv)
right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (right_conv)
right_flatten = Flatten()(right_conv)

conc_layer = Concatenate()([left_flatten, right_flatten])
dense_layer = Dense(384, activation="relu")(conc_layer)
dense_layer = Dense(384, activation="relu")(dense_layer)
output_layer = Dense(1, activation="sigmoid")(dense_layer)

model = Model(inputs=[left_input, right_input], outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.load_weights("weights/acc1_weights1.h5")

get_batch_from_image("../samples/cones/", 11, 63, "new_disp", model)




#plot_model(model, show_shapes=True, to_file='model.png')

# test
#left, right = data.get_random_sample("../samples/cones/", 11, 8, 10, 4, 1, 120, 100)

#predictions = model.predict([[left], [right]])
#print(predictions)

########################################################################################

