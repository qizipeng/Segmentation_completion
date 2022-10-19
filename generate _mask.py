import os
from PIL import Image
import numpy as np

img_path = "./img.png"
img = np.array(Image.open(img_path))

shape = img.shape
mask = np.ones((shape[0],shape[1]))

# print(np.sum(mask))


# mode: random row patch
###random mode
ratio = 0.2
num = int(ratio * shape[0]*shape[1])
position_random = np.random.randint(0,shape[0]*shape[1],size = num)
position_random = list(position_random)
# print(position_random)
mask_random = mask.reshape(-1).copy()
mask_random[position_random] = 0
Image.fromarray(mask_random.reshape(shape[0],shape[1])).convert("L").save("./random_mask.png")
# print(mask_random.shape)
# print(np.sum(mask), np.sum(mask_random))
del mask_random

###row mode
ratio = 0.2
num = int(ratio * shape[0])
x_random = np.random.randint(0,shape[0],size = num)
x_random = list(x_random)
# print(position_random)
mask_random = mask.copy()
mask_random[x_random,:] = 0
Image.fromarray(mask_random).convert("L").save("./row_mask.png")
# print(mask_random.shape)
# print(np.sum(mask), np.sum(mask_random))
del mask_random




