# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 20:11:34 2018

@author: dongg
"""


import pandas as pd
import os
import glob
import cv2
import math
import seaborn as sns
import json
import sys

import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)


import cv2
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from os.path import join


TRAIN_PATH = 'C:/Users/dongg/Desktop/deeplearning/project/stage1_train/stage1_train/'
TEST_PATH = 'C:/Users/dongg/Desktop/deeplearning/project/stage1_train/stage1_test/'

RANDOM_SEED=99

OUTPUT_PATH = './'
CONTOUR_EXTRACT_MODE = cv2.RETR_TREE
train_ids = [x for x in os.listdir(TRAIN_PATH) if os.path.isdir(TRAIN_PATH+x)]
test_ids = [x for x in os.listdir(TEST_PATH) if os.path.isdir(TEST_PATH+x)]

df = pd.DataFrame({'id':train_ids,'train_or_test':'train'})
df = df.append(pd.DataFrame({'id':test_ids,'train_or_test':'test'}))

df.groupby(['train_or_test']).count()

train_ids = os.listdir(TRAIN_PATH)
test_ids = os.listdir(TEST_PATH)

#image visulization
test_paths = [glob.glob(join(TEST_PATH, test_id, "images", "*"))[0] for test_id in test_ids]
tmp_path = np.random.choice(test_paths)
image = cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE)
plt.imshow(image)


from sklearn.cluster import KMeans

#create a histgram to check the disbtribution of pixel
def centroid_histogram(clt):
    
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

#collect image information for visulization of image and mask
def get_image_info(path, clusters=2):
    image = cv2.imread(path)
    height,width,_ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters = clusters)
    clt.fit(image)
    hist = centroid_histogram(clt)
    
    bg_idx, fg_idx = 0, clusters-1
    if hist[bg_idx] < hist[fg_idx]:
        bg_idx, fg_idx = clusters-1, 0
    
    bg_red, bg_green, bg_blue = clt.cluster_centers_[bg_idx]
    fg_red, fg_green, fg_blue = clt.cluster_centers_[fg_idx]
    
    bg_color = sum(clt.cluster_centers_[bg_idx])/3
    fg_color = sum(clt.cluster_centers_[fg_idx])/3
    
    return (pd.Series([height,width,
                       bg_red, bg_green, bg_blue, bg_color,
                       fg_red, fg_green, fg_blue, fg_color,
                       hist[bg_idx],hist[fg_idx],
                       fg_color < bg_color]))
    
    
#collect image data and mask data

mask_data = []
img_data = []

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')
    imgs=[]
    imgs.append(img)
    img_height = img.shape[0]
    img_width = img.shape[1]
    img_area = img_width * img_height
    
    nucleus_count = 1
    
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        masks = []
        mask = imread(path + '/masks/' + mask_file)
        masks.append(mask)
        mask_height = mask.shape[0]
        mask_width = mask.shape[1]
        mask_area = mask_width * img_height
        nucleus_area = (np.sum(mask) / 255)
        mask_to_img_ratio = nucleus_area / mask_area  
        mask_data.append([n, mask_height, mask_width, mask_area, nucleus_area, mask_to_img_ratio])
        nucleus_count = nucleus_count + 1
    
    img_data.append([img_height, img_width, img_area, nucleus_count])
    
#image and mask width, height, dimension statistics                      
df_img = pd.DataFrame(img_data, columns=['height', 'width', 'area', 'nuclei'])           
fig, ax = plt.subplots(1, 3, figsize=(10,5))
width_plt = sns.distplot(df_img['width'].values, ax=ax[0])
width_plt.set(xlabel='width (px)')
width_plt.set(ylim=(0, 0.01))
height_plt = sns.distplot(df_img['height'].values, ax=ax[1])
height_plt.set(xlabel='height (px)')
height_plt.set(ylim=(0, 0.015))
area_plt = sns.distplot(df_img['area'].values)
area_plt.set(xlabel="area (px)")
fig.show()
plt.tight_layout()