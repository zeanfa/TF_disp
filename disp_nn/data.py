import numpy
import random
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dot, Flatten, Input, Concatenate
from PIL import Image, ImageDraw
import time

def get_batch(folder_name, patch_size, neg_low, neg_high, scale):
    left_pic = Image.open(folder_name + "im0.png").convert("L")
    right_pic = Image.open(folder_name + "im1.png").convert("L")
    disp0_pic = Image.open(folder_name + "disp0.png")
    
    left_pix = numpy.atleast_3d(left_pic)
    right_pix = numpy.atleast_3d(right_pic)
    disp0_pix = numpy.array(disp0_pic)
    width, height = left_pic.size
    
    left_f = lambda x: (x - left_pix.mean())/left_pix.std()
    norm_left = left_f(left_pix)
    right_f = lambda x: (x - right_pix.mean())/right_pix.std()
    norm_right = right_f(right_pix)

    left_patches = []
    right_patches = []
    outputs = []
    dataset_size = 0

    for i in range(0, height):
        for j in range(0, width):
            if disp0_pix[i, j] > 0 and j >= (int(disp0_pix[i, j]/scale) + int(patch_size/2)) and j < (width - int(patch_size/2)):
                if i >= int(patch_size/2) and i < (height - int(patch_size/2)):

                    # positive sample
                    left_patch = norm_left[(i - int(patch_size/2)) : (i + int(patch_size/2) + 1),
                                           (j - int(patch_size/2)) : (j + int(patch_size/2) + 1)]
                    disp = int(disp0_pix[i, j]/scale)
                    right_patch = norm_right[(i - int(patch_size/2)) : (i + int(patch_size/2) + 1),
                                             (j - int(patch_size/2) - disp) : (j + int(patch_size/2) - disp + 1)]
                    left_patches.append(left_patch)
                    right_patches.append(right_patch)
                    outputs.append(1)

                    # negative sample
                    offset = random.randint(neg_low, neg_high)
                    left_patch = norm_left[(i - int(patch_size/2)) : (i + int(patch_size/2) + 1),
                                           (j - int(patch_size/2)) : (j + int(patch_size/2) + 1)]
                    right_patch = norm_right[(i - int(patch_size/2)) : (i + int(patch_size/2) + 1),
                                             (j - int(patch_size/2) - disp + offset) : (j + int(patch_size/2) - disp + offset + 1)]
                    left_patches.append(left_patch)
                    right_patches.append(right_patch)
                    outputs.append(0)
                    
                    dataset_size += 2
        
    return numpy.array(left_patches), numpy.array(right_patches), numpy.array(outputs)

def get_random_sample(folder_name, patch_size, neg_low, neg_high, scale, pos_or_neg, i, j):
    left_pic = Image.open(folder_name + "im0.png").convert("L")
    right_pic = Image.open(folder_name + "im1.png").convert("L")
    disp0_pic = Image.open(folder_name + "disp0.png")
    
    left_pix = numpy.atleast_3d(left_pic)
    right_pix = numpy.atleast_3d(right_pic)
    disp0_pix = numpy.array(disp0_pic)
    width, height = left_pic.size
    
    left_f = lambda x: (x - left_pix.mean())/left_pix.std()
    norm_left = left_f(left_pix)
    right_f = lambda x: (x - right_pix.mean())/right_pix.std()
    norm_right = right_f(right_pix)

    if disp0_pix[i, j] > 0:
        left_patch = []
        right_patch = []
        disp = int(disp0_pix[i, j]/scale)
        if pos_or_neg == 1:
            left_patch = norm_left[(i - int(patch_size/2)) : (i + int(patch_size/2) + 1),
                                   (j - int(patch_size/2)) : (j + int(patch_size/2) + 1)]
            right_patch = norm_right[(i - int(patch_size/2)) : (i + int(patch_size/2) + 1),
                                     (j - int(patch_size/2) - disp) : (j + int(patch_size/2) - disp + 1)]
        else:
            offset = random.randint(neg_low, neg_high)
            left_patch = norm_left[(i - int(patch_size/2)) : (i + int(patch_size/2) + 1),
                                   (j - int(patch_size/2)) : (j + int(patch_size/2) + 1)]
            right_patch = norm_right[(i - int(patch_size/2)) : (i + int(patch_size/2) + 1),
                                     (j - int(patch_size/2) - disp + offset) : (j + int(patch_size/2) - disp + offset + 1)]
        
        return numpy.array(left_patch), numpy.array(right_patch)
    else:
        print("disparity at this point equals 0")

def get_image_in_patches(folder_name, patch_size):
    left_pic = Image.open(folder_name + "im0.png").convert("L")
    right_pic = Image.open(folder_name + "im1.png").convert("L")
    
    left_pix = numpy.atleast_3d(left_pic)
    right_pix = numpy.atleast_3d(right_pic)
    width, height = left_pic.size
    
    left_f = lambda x: (x - left_pix.mean())/left_pix.std()
    norm_left = left_f(left_pix)
    right_f = lambda x: (x - right_pix.mean())/right_pix.std()
    norm_right = right_f(right_pix)

    left_patches = numpy.zeros((height,width, patch_size, patch_size, 1))
    right_patches = numpy.zeros((height,width, patch_size, patch_size, 1))
    dataset_size = 0

    for i in range(int(patch_size/2), height - int(patch_size/2)):
        for j in range(int(patch_size/2), width - int(patch_size/2)):

            left_patches[i, j, ::, ::, ::] = norm_left[(i - int(patch_size/2)) : (i + int(patch_size/2) + 1),
                                                       (j - int(patch_size/2)) : (j + int(patch_size/2) + 1)]
            right_patches[i, j, ::, ::, ::] = norm_right[(i - int(patch_size/2)) : (i + int(patch_size/2) + 1),
                                                         (j - int(patch_size/2)) : (j + int(patch_size/2) + 1)]
        
    return left_patches, right_patches

# convolves images using upper model
def convolve_images_ctc(folder_name, patch_size, conv_feature_maps, lctc_model, rctc_model):
    print("begin convolution")
    left_pic = Image.open(folder_name + "im0.png").convert("L")
    right_pic = Image.open(folder_name + "im1.png").convert("L")
    border = int(patch_size/2)
    
    left_pix = numpy.atleast_3d(left_pic)
    right_pix = numpy.atleast_3d(right_pic)
    width, height = left_pic.size
    
    left_f = lambda x: (x - left_pix.mean())/left_pix.std()
    norm_left = left_f(left_pix)
    right_f = lambda x: (x - right_pix.mean())/right_pix.std()
    norm_right = right_f(right_pix)
    
    timestamp = time.time()
    l_prediction = lctc_model.predict([[norm_left]])
    r_prediction = rctc_model.predict([[norm_right]])
    conv_left_patches = l_prediction.reshape((height - 2*border, width - 2*border, conv_feature_maps))
    conv_right_patches = r_prediction.reshape((height - 2*border, width - 2*border, conv_feature_maps))
    print("total time ", round(time.time()-timestamp, 3))
    return conv_left_patches, conv_right_patches

def convolve_images(folder_name, patch_size, conv_feature_maps, ctc_model):
    print("begin convolution")
    pic = Image.open(folder_name + "im0.png")
    width, height = pic.size
    left_patches, right_patches = get_image_in_patches(folder_name, patch_size)
    left_patches = left_patches.reshape((height*width, patch_size, patch_size, 1))
    right_patches = right_patches.reshape((height*width, patch_size, patch_size, 1))
    timestamp = time.time()
    prediction = conv_model.predict([left_patches,right_patches])
    conv_left_patches = prediction[0]
    conv_right_patches = prediction[1]
    conv_left_patches = conv_left_patches.reshape((height, width, conv_feature_maps))
    conv_right_patches = conv_right_patches.reshape((height, width, conv_feature_maps))
    print("total time ", round(time.time()-timestamp, 3))
    return conv_left_patches, conv_right_patches

# computes disparity using convolved pictures and dense layers
def disp_map_from_conv(left_conv, right_conv, patch_size, max_disp, conv_feature_maps, dense_model, image_name):
    print("begin disparity computation")
    height = left_conv.shape[0]
    width = right_conv.shape[1]
    size = max_disp*(width - 2*int(patch_size/2) - max_disp)
    right_conv_pos = numpy.zeros((size, conv_feature_maps))
    left_conv_pos = numpy.zeros((size, conv_feature_maps))
    disp_pix = numpy.zeros((height,width))
    timestamp = time.time()
    
    for i in range(int(patch_size/2), height - int(patch_size/2)):
        left_conv_pos = numpy.repeat(left_conv[i,max_disp + int(patch_size/2) :
                                               width - int(patch_size/2)], max_disp, axis = 0).reshape(((width - max_disp - 2*int(patch_size/2))*max_disp, conv_feature_maps))
        for j in range(int(patch_size/2), width - int(patch_size/2) - max_disp):
            right_conv_pos[(j-int(patch_size/2))*max_disp :
                           (j-int(patch_size/2))*max_disp + max_disp,] = right_conv[i, j : (max_disp+j)]
            
        dense_predictions = dense_model.predict([numpy.expand_dims(left_conv_pos, axis = 1), numpy.expand_dims(right_conv_pos, axis = 1)])
        disp_pix[i, int(patch_size/2) + max_disp : width - int(patch_size/2)] = (
            255*(max_disp - numpy.argmax(numpy.squeeze(dense_predictions).reshape((width - 2*int(patch_size/2) - max_disp, max_disp)), axis = 1)))/max_disp
        print("\rtime ", "%.2f" % (time.time()-timestamp), " progress ", "%.2f" % (100*(i - int(patch_size/2))/(height - 2*int(patch_size/2))), "%", end = "\r")
    print("\rtime ", "%.2f" % (time.time()-timestamp), " progress ", "%.2f" % 100, "%", end = "\r")
    print("\ntotal time ", "%.2f" % (time.time()-timestamp))
    img = Image.fromarray(disp_pix.astype('uint8'), mode = 'L')
    img.save(image_name + ".png", "PNG")

# high improvement
def disp_map_from_conv_dtc(left_conv, right_conv, patch_size, max_disp, conv_feature_maps, dtc_model, image_name):
    print("begin disparity computation")
    height = left_conv.shape[0]
    width = right_conv.shape[1]
    disp_pix = numpy.zeros((height,width))
    timestamp = time.time()
    dtc_predictions = dtc_model.predict([numpy.expand_dims(numpy.concatenate((left_conv[::, max_disp:width,::],
                                                                                    right_conv[::,0:width-max_disp,::]), axis = 2), axis = 0)])
    dtc_predictions = numpy.squeeze(dtc_predictions, axis=0)
    for i in range(1, max_disp):
        prediction = dtc_model.predict([numpy.expand_dims(numpy.concatenate((left_conv[::, max_disp:width,::],
                                                                                    right_conv[::,i:width-max_disp+i,::]), axis = 2), axis = 0)])
        dtc_predictions = numpy.concatenate((dtc_predictions, numpy.squeeze(prediction, axis=0)), axis=2)
        print("\rtime ", "%.2f" % (time.time()-timestamp), " progress ", "%.2f" % (100*(i+1)/max_disp), "%", end = "\r")
    print("\rtime ", "%.2f" % (time.time()-timestamp), " progress ", "%.2f" % 100, "%", end = "\r")
    disp_pix[::, max_disp:width] = (255*(max_disp - numpy.argmax(dtc_predictions, axis = 2)))/max_disp
    print("\ntotal time ", "%.2f" % (time.time()-timestamp))
    img = Image.fromarray(disp_pix.astype('uint8'), mode = 'L')
    img.save(image_name + ".png", "PNG")

# computes disparity with single pass through dense layers
# no significant improvement
def disp_map_from_conv_single(left_conv, right_conv, patch_size, max_disp, conv_feature_maps, dense_model, image_name):
    print("begin disparity computation")
    height = left_conv.shape[0]
    width = right_conv.shape[1]
    disp_pix = numpy.zeros((height,width))
    timestamp = time.time()
    for i in range(max_disp):
        dense_predictions = dense_model.predict([numpy.expand_dims(left_conv[::, max_disp:width,::].reshape((height*(width-max_disp),conv_feature_maps)), axis = 1),
                                                 numpy.expand_dims(right_conv[::,i:width-max_disp+i,::].reshape((height*(width-max_disp),conv_feature_maps)), axis = 1)])
        #disp_pix[i, int(patch_size/2) + max_disp : width - int(patch_size/2)] = (
            #255*(max_disp - numpy.argmax(numpy.squeeze(dense_predictions).reshape((width - 2*int(patch_size/2) - max_disp, max_disp)), axis = 1)))/max_disp
        print("\rtime ", "%.2f" % (time.time()-timestamp), " progress ", "%.2f" % (100*(i+1)/max_disp), "%", end = "\r")
    print("\rtime ", "%.2f" % (time.time()-timestamp), " progress ", "%.2f" % 100, "%", end = "\r")
    print("\ntotal time ", "%.2f" % (time.time()-timestamp))
    img = Image.fromarray(disp_pix.astype('uint8'), mode = 'L')
    img.save(image_name + ".png", "PNG")

def comp_error_in_area(name1, name2, patch_size, max_disp, error_threshold):
    disp_ref = Image.open(name1 + ".png")
    disp = Image.open(name2 + ".png")
    width, height = disp.size
    disp_ref_pix = numpy.atleast_1d(disp_ref)
    disp_pix = numpy.atleast_1d(disp)
    filtered_pix = numpy.zeros((height,width))
    
    pix_num = (height - patch_size + 1) * (width - patch_size - max_disp + 1)
    error_num = 0
    not_recognized = 0

    for i in range(int(patch_size/2), height - int(patch_size/2)):
        for j in range(max_disp + int(patch_size/2), width - int(patch_size/2)):
            if int(disp_ref_pix[i,j]) == 0:
                not_recognized += 1
            elif abs(int(disp_ref_pix[i,j]) - int(disp_pix[i,j])) > error_threshold:
                error_num += 1
            else:
                filtered_pix[i, j] = disp_pix[i, j]
    print("error rate ", round(error_num*100/(pix_num - not_recognized), 2))
    print("not recognized", not_recognized)
    print("num of pixels", pix_num)
    print("mum of errors", error_num)
    img = Image.fromarray(filtered_pix.astype('uint8'), mode = 'L')
    img.save("filtered_disp_" + str(error_threshold) + ".png", "PNG")
