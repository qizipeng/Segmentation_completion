import os
from PIL import Image
import numpy as np

img_path = "imgs/img.png"
mask_path = "imgs/random_mask.png"

carla = [
   [70 ,70 ,70],
   [150 , 60,  45],
   [180 ,130 , 70],
   [232 , 35 ,244],
   [ 35 ,142 ,107],
   [100 ,170 ,145],
   [160 ,190 ,110],
   [153 ,153 ,153],
   [80 ,90 ,55],
   [ 50 ,120 ,170],
   [128 , 64 ,128],
   [ 50 ,234 ,157],
   [142 ,  0  , 0],
   [  0 ,220 ,220],
   [ 30 ,170 ,250],
   [156 ,102 ,102],
   [ 40 , 40 ,100],
   [81  ,0 ,81],
   [140 ,150 ,230],
   [100 ,100 ,150]
]
carla = np.array(carla)
img = np.array(Image.open(img_path))
mask = np.array(Image.open(mask_path))
shape = img.shape

maskcolor = np.ones_like(img)
for x in range(shape[0]):
    for y in range(shape[1]):
        if mask[x,y]!=0:
            maskcolor[x,y,:] = carla[img[x,y,0],:]
        else:
            maskcolor[x, y, :] = [0,0,0]

Image.fromarray(maskcolor).save(mask_path.split(".")[0]+"_color.png")
