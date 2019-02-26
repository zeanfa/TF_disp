import numpy
import data
from PIL import Image, ImageDraw
import time

patch_size = 2
max_disp = 2
height = 10
width = 20
br = 150
disp_pix = numpy.zeros((height,width))

for i in range(int(patch_size/2), height - int(patch_size/2)):
    for j in range(max_disp + int(patch_size/2), width - int(patch_size/2)):
        disp_pix[i,j] = br
            

# Creates PIL image
print(disp_pix)
img = Image.fromarray(disp_pix.astype('uint8'), mode = 'L')
img.save("sample.png", "PNG")
