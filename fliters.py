"""
Created on Wed May 17 18:28:07 2017

@author: hzy
"""
import os
from skimage import data,filters,feature
from skimage.io import imread ,imshow ,imsave
import matplotlib.pyplot as plt
#%%
filename = os.path.join('C:\\Users\\hzy\\Desktop\\1','am.png')
img = imread(filename)
#edges1 = filters.gaussian_filter(img,sigma=0.1)   #sigma=0.4
#edges2 = filters.gaussian_filter(img,sigma=2)   #sigma=5

edges1 = feature.canny(img)   #sigma=1
edges2 = feature.canny(img,sigma=3)   #sigma=3

plt.figure('gaussian',figsize=(8,8))
plt.subplot(121)
plt.imshow(edges1,plt.cm.gray)  

plt.subplot(122)
plt.imshow(edges2,plt.cm.gray)

plt.show()
#%%
import numpy as np
from skimage import data, filter
import matplotlib.pyplot as plt
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
filename = os.path.join('C:\\Users\\hzy\\Desktop\\1','am.png')
img = imread(filename)
img = rgb2gray(img)
#img = Image.open(filename)
hsobel_text = filter.gabor(img)

plt.figure(figsize=(12, 3))

plt.subplot(121)
plt.imshow(img, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(122)
plt.imshow(hsobel_text, cmap='jet', interpolation='nearest')
plt.axis('off')
plt.tight_layout()
plt.show()
#%%
from skimage import data,filters
import matplotlib.pyplot as plt
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
filename = os.path.join('C:\\Users\\hzy\\Desktop\\1','am.png')
img = imread(filename)
img = rgb2gray(img)
filt_real, filt_imag = filters.gabor(img,frequency=0.6)   

plt.figure('gabor',figsize=(8,8))

plt.subplot(121)
plt.title('filt_real')
plt.imshow(filt_real,plt.cm.gray)  

plt.subplot(122)
plt.title('filt-imag')
plt.imshow(filt_imag,plt.cm.gray)

plt.show()
#%%
from skimage import data, exposure
import matplotlib.pyplot as plt

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
filename = os.path.join('C:\\Users\\hzy\\Desktop\\1','am.png')
img = imread(filename)
img = rgb2gray(img)/255
camera_equalized =exposure.equalize_adapthist(img) 



plt.figure(figsize=(7, 3))

plt.subplot(121)
plt.imshow(img, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(122)
plt.imshow(camera_equalized, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.tight_layout()
plt.show()