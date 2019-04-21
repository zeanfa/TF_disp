import numpy
import random
from PIL import Image, ImageDraw
from scipy import spatial, ndimage, misc
import time
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image as skim
import os


'''
This function creates image descriptors using the Sobel filter
as in ELAS algorithm.
'''
def sobel_desc(patch_size, folder_name):
    left_pic = Image.open(folder_name + "im0.png").convert("L")
    right_pic = Image.open(folder_name + "im1.png").convert("L")
    width, height = left_pic.size
    left_pix = numpy.atleast_1d(left_pic)
    right_pix = numpy.atleast_1d(right_pic)
    border = int(patch_size/2)

    right_f = lambda x: (x - right_pix.min())/(right_pix.max()-right_pix.min())
    left_f = lambda x: (x - left_pix.min())/(left_pix.max()-left_pix.min())
    norm_left = left_f(left_pix)
    norm_right = right_f(right_pix)

    ls0 = ndimage.sobel(norm_left, axis=0)
    ls1 = ndimage.sobel(norm_left,axis=1)
    ls0_f = lambda x: (x - ls0.min())/(ls0.max()-ls0.min())
    ls0 = ls0_f(ls0)
    ls1_f = lambda x: (x - ls1.min())/(ls1.max()-ls1.min())
    ls1 = ls1_f(ls1)
    ls0_patches = skim.extract_patches_2d(ls0, (patch_size, patch_size)).reshape((height-2*border,width-2*border,patch_size**2))
    ls1_patches = skim.extract_patches_2d(ls1, (patch_size, patch_size)).reshape((height-2*border,width-2*border,patch_size**2))
    ls_desc = numpy.concatenate((ls0_patches, ls1_patches), axis=2)

    rs0 = ndimage.sobel(norm_right, axis=0)
    rs1 = ndimage.sobel(norm_right,axis=1)
    rs0_f = lambda x: (x - rs0.min())/(rs0.max()-rs0.min())
    rs0 = rs0_f(rs0)
    rs1_f = lambda x: (x - rs1.min())/(rs1.max()-rs1.min())
    rs1 = rs1_f(rs1)
    rs0_patches = skim.extract_patches_2d(rs0, (patch_size, patch_size)).reshape((height-2*border,width-2*border,patch_size**2))
    rs1_patches = skim.extract_patches_2d(rs1, (patch_size, patch_size)).reshape((height-2*border,width-2*border,patch_size**2))
    rs_desc = numpy.concatenate((rs0_patches, rs1_patches), axis=2)

    return ls_desc, rs_desc

'''
This function makes stereo matching of support points.
'''

def compute_support_points(ls_desc, rs_desc, max_disp, image_name):
    timestamp_sp = time.time()
    width = ls_desc.shape[1]
    height = ls_desc.shape[0]
    sad_arr =  numpy.linalg.norm(ls_desc[::, max_disp:width,::] - rs_desc[::, 0:width-max_disp,::], axis=-1, ord=1)
    sad_arr = numpy.expand_dims(sad_arr, axis=2)
    for i in range(1, max_disp):
        sad =  numpy.linalg.norm(ls_desc[::, max_disp:width,::] - rs_desc[::,i:width-max_disp+i,::], axis=-1, ord=1)
        sad = numpy.expand_dims(sad, axis=2)
        sad_arr = numpy.concatenate((sad_arr, sad), axis=2)
        print("\rsad_arr time ", "%.2f" % (time.time()-timestamp_sp), " progress ", "%.2f" % (100*(i+1)/max_disp), "%", end = "\r")
    numpy.save("np_data/sad_" + image_name, sad_arr)
    return sad_arr

'''
This function computes support matching points grid.
'''
def compute_support_points_grid(metric, match_th, max_disp, patch_size, grid_step,folder_name, image_name):
    win = 4
    timestamp_sp = time.time()
    border = int(patch_size/2)
    left_pic = Image.open(folder_name + "im0.png")
    width = left_pic.size[0] - 2*border
    height = left_pic.size[1] - 2*border
    sp_arr = numpy.zeros((height,width-max_disp))
    sp_grid = numpy.zeros((int(height/grid_step),int((width-max_disp)/grid_step)))
    
    if metric == "sobel":
        exists = os.path.isfile("np_data/sad_" + image_name + ".npy")
        if exists:
            metric_arr = numpy.load("np_data/sad_" + image_name + ".npy")
        else:
            ls_desc, rs_desc = sobel_desc(patch_size, folder_name)
            metric_arr = compute_support_points(ls_desc, rs_desc, max_disp, image_name)
    elif metric == "nn_cos":
        exists = os.path.isfile("np_data/" + image_name + "_disp_fst_cosine.npy")
        if exists:
            metric_arr = 1 - numpy.load("np_data/" + image_name + "_disp_fst_cosine.npy")[border:height+border,border:width-max_disp+border]
        else:
            print("Metric file not found")
            exit()
    sorted_metric = numpy.sort(metric_arr)
    sp_num = 0
    timestamp_sparr = time.time()
    for i in range(height):
        print("\rsp_arr time ", "%.2f" % (time.time()-timestamp_sparr), " progress ", "%.2f" % (100*(i+1)/height), "%", end = "\r")
        for j in range(width-max_disp):
            if abs(sorted_metric[i,j,0] - sorted_metric[i,j,1]) < match_th:
                sp_arr[i,j] = 0
            else:
                sp_arr[i,j] = max_disp - numpy.argmin(metric_arr[i,j])
                sp_num += 1
    print("\rsp_arr time ", "%.2f" % (time.time()-timestamp_sparr), " progress ", "%.2f" % 100, "%", end = "\r")
    print("\nnum of robust support points ", sp_num)
    timestamp_spgrid = time.time()
    for i in range(1, int(height/grid_step)-1):
        print("\rsp_grid time ", "%.2f" % (time.time()-timestamp_spgrid), " progress ", "%.2f" % (100*(i+1)/int(height/grid_step)), "%", end = "\r")
        for j in range(1, int((width-max_disp)/grid_step)-1):
            nonzero = numpy.count_nonzero(sp_arr[i*grid_step-win:i*grid_step+win+1,j*grid_step-win:j*grid_step+win+1])
            if nonzero>0:
                sp_grid[i,j] = int(numpy.sum(sp_arr[i*grid_step-win:i*grid_step+win+1,j*grid_step-win:j*grid_step+win+1])/nonzero)
    print("\nsp_grid time ", "%.2f" % (time.time()-timestamp_spgrid), " progress ", "%.2f" % 100, "%", end = "\r")
    print("grid points found: ", numpy.count_nonzero(sp_grid), "(", round(numpy.count_nonzero(sp_grid)/sp_grid.size, 3), "%)")
    raw_disp_pix = numpy.zeros((height+2*border,width+2*border))
    raw_disp_pix[border:height+border, border+max_disp:width+border] = (255*(max_disp - numpy.argmin(metric_arr, axis = 2)))/max_disp
    return sp_grid, raw_disp_pix

'''
This function filters support points grid using STD filter.
'''
def grid_filter_std(sp_grid, filter_w, std_th, raw_disp_pix, grid_step, border, max_disp):
    filtered = 0
    for i in range(sp_grid.shape[0]):
        for j in range(sp_grid.shape[1]):
            if not sp_grid[i,j] == 0:
                area = int((filter_w-1)/2)
                d_i = i*grid_step+border
                d_j = j*grid_step+max_disp+border
                std = raw_disp_pix[d_i-area:d_i+area+1,d_j-area:d_j+area+1].std()
                if std > std_th:
                    sp_grid[i,j] = 0
    print("left after filtration ", numpy.count_nonzero(sp_grid), "(", round(numpy.count_nonzero(sp_grid)/sp_grid.size, 3), "%)")
    return sp_grid

'''
This function interpolates the support points grid.
'''

def interpolate_grid(sp_grid):
    timestamp_int = time.time()
    l_grid = sp_grid.copy()
    r_grid = sp_grid.copy()
    d_grid = sp_grid.copy()
    u_grid = sp_grid.copy()

    for i in range(1, sp_grid.shape[0]-1):
        for j in range(1, sp_grid.shape[1]-1):
            if l_grid[i,j] == 0:
                l_grid[i,j] = l_grid[i,j-1]
    for i in range(1, sp_grid.shape[0]-1):
        for j in range(sp_grid.shape[1]-2,0):
            if r_grid[i,j] == 0:
                r_grid[i,j] = r_grid[i,j+1]
    for j in range(1, sp_grid.shape[1]-1):
        for i in range(sp_grid.shape[0]-2,0):
            if u_grid[i,j] == 0:
                u_grid[i,j] = u_grid[i+1,j]
    for j in range(1, sp_grid.shape[1]-1):
        for i in range(1,sp_grid.shape[0]-1):
            if d_grid[i,j] == 0:
                d_grid[i,j] = d_grid[i-1,j]
    l_grid = numpy.expand_dims(l_grid, axis=2)
    r_grid = numpy.expand_dims(r_grid, axis=2)
    d_grid = numpy.expand_dims(d_grid, axis=2)
    u_grid = numpy.expand_dims(u_grid, axis=2)
    int_grid = numpy.concatenate((l_grid,r_grid,d_grid,u_grid), axis=2)
    
    for i in range(1, sp_grid.shape[0]-1):
        for j in range(1, sp_grid.shape[1]-1):
            if sp_grid[i,j] == 0:
                if numpy.count_nonzero(int_grid[i,j])>0:
                    sp_grid[i,j] = int_grid[i,j].sum()/numpy.count_nonzero(int_grid[i,j])
    print("\ngrid interpolation time ", "%.2f" % (time.time()-timestamp_int))
    return sp_grid

'''
This function interpolates the support points grid by windows.
'''

def interpolate_grid_window(sp_grid):
    timestamp_int = time.time()
    l_grid = sp_grid.copy()
    r_grid = sp_grid.copy()
    d_grid = sp_grid.copy()
    u_grid = sp_grid.copy()

    for i in range(1, sp_grid.shape[0]-1):
        for j in range(1, sp_grid.shape[1]-1):
            if l_grid[i,j] == 0:
                if numpy.count_nonzero(l_grid[i-1:i+2,j-1:j+2])>0:
                    l_grid[i,j] = l_grid[i-1:i+2,j-1:j+2].sum()/numpy.count_nonzero(l_grid[i-1:i+2,j-1:j+2])
    for i in range(1, sp_grid.shape[0]-1):
        for j in range(sp_grid.shape[1]-2,0):
            if r_grid[i,j] == 0:
                if numpy.count_nonzero(r_grid[i-1:i+2,j-1:j+2])>0:
                    r_grid[i,j] = r_grid[i-1:i+2,j-1:j+2].sum()/numpy.count_nonzero(r_grid[i-1:i+2,j-1:j+2])
    for j in range(1, sp_grid.shape[1]-1):
        for i in range(sp_grid.shape[0]-2,0):
            if u_grid[i,j] == 0:
                if numpy.count_nonzero(u_grid[i-1:i+2,j-1:j+2])>0:
                    u_grid[i,j] = u_grid[i-1:i+2,j-1:j+2].sum()/numpy.count_nonzero(u_grid[i-1:i+2,j-1:j+2])
    for j in range(1, sp_grid.shape[1]-1):
        for i in range(1,sp_grid.shape[0]-1):
            if d_grid[i,j] == 0:
                if numpy.count_nonzero(d_grid[i-1:i+2,j-1:j+2])>0:
                    d_grid[i,j] = d_grid[i-1:i+2,j-1:j+2].sum()/numpy.count_nonzero(d_grid[i-1:i+2,j-1:j+2])
    l_grid = numpy.expand_dims(l_grid, axis=2)
    r_grid = numpy.expand_dims(r_grid, axis=2)
    d_grid = numpy.expand_dims(d_grid, axis=2)
    u_grid = numpy.expand_dims(u_grid, axis=2)
    int_grid = numpy.concatenate((l_grid,r_grid,d_grid,u_grid), axis=2)
    
    for i in range(1, sp_grid.shape[0]-1):
        for j in range(1, sp_grid.shape[1]-1):
            if sp_grid[i,j] == 0:
                if numpy.count_nonzero(int_grid[i,j])>0:
                    sp_grid[i,j] = int_grid[i,j].sum()/numpy.count_nonzero(int_grid[i,j])
    print("\ngrid interpolation time ", "%.2f" % (time.time()-timestamp_int))
    return sp_grid

'''
This function computes error of the support points grid.
'''
def comp_grid_error(sp_grid,grid_pix,disp_ref_pix,error_th,grid_step,border,max_disp,grid_name):
    error_pix=0
    for i in range(sp_grid.shape[0]):
        for j in range(sp_grid.shape[1]):
            if not sp_grid[i,j] == 0:
                grid_pix[i*grid_step+border,j*grid_step+max_disp+border] = numpy.array([0,0,int(sp_grid[i,j])])
                if abs(int(sp_grid[i,j])-int(disp_ref_pix[i*grid_step+border,j*grid_step+max_disp+border]))>error_th:
                    error_pix += 1
                    grid_pix[i*grid_step+border,j*grid_step+max_disp+border] = numpy.array([int(sp_grid[i,j]),0,0])
            else:
                grid_pix[i*grid_step+border,j*grid_step+max_disp+border] = numpy.array([0,255,0])
    print("error rate ", round(error_pix/numpy.count_nonzero(sp_grid),2))
    img = Image.fromarray(grid_pix.astype('uint8'))
    img.save("work/" + grid_name + "_grid.png", "PNG")

'''
This function creates disparity map using raw disparity and support points grid
'''
def disp_sp(raw_disp, sp_grid):
    return
    

'''
This function implements EasyELAS algorithm using predictions of the net.
'''
def disp_map_easyELAS(patch_size, max_disp, match_th, conv_feature_maps, folder_name, image_name):
    timestamp = time.time()
    match_th = 0.28 # 0.28 for fst_metric, 0.9 for sad_metric
    grid_step = 5
    border = int(patch_size/2)
    timestamp = time.time()
    #print("begin disparity computation")
    sp_grid, raw_disp_pix = compute_support_points_grid("nn_cos", match_th, max_disp, patch_size, grid_step, folder_name, image_name)
    
    left_pic = Image.open(folder_name + "im0.png")
    width, height = left_pic.size
    left_pix = numpy.atleast_3d(left_pic)
    sp_grid = 255*sp_grid/max_disp

    disp_ref = Image.open("work/norm_disp.png")
    disp_ref_pix = numpy.atleast_1d(disp_ref)
    grid_pix = numpy.atleast_3d(left_pic.convert("L").convert("RGB")).copy()
    error_th = 12
    filter_w = 3
    std_th = 5
    sp_grid = grid_filter_std(sp_grid, filter_w, std_th, raw_disp_pix, grid_step, border, max_disp)
    comp_grid_error(sp_grid,grid_pix,disp_ref_pix,error_th,grid_step,border,max_disp,"filtered")

    sp_grid = interpolate_grid_window(sp_grid)
    comp_grid_error(sp_grid,grid_pix,disp_ref_pix,error_th,grid_step,border,max_disp,"interpolated")

    sp_grid = grid_filter_std(sp_grid, filter_w, std_th, raw_disp_pix, grid_step, border, max_disp)
    comp_grid_error(sp_grid,grid_pix,disp_ref_pix,error_th,grid_step,border,max_disp,"filtered2")

    #fig = plt.figure()
    #plt.imshow(grid_pix)
    #plt.show()
    #img = Image.fromarray(raw_disp_pix.astype('uint8'), mode = 'L')
    #img.save("work/sobel_disp.png", "PNG")
    print("\ntotal time ", "%.2f" % (time.time()-timestamp))
