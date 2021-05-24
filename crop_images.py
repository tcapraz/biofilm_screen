import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
from itertools import product
import string

def crop(input_dir, filename, out_dir):
    img = cv2.imread(os.path.join(input_dir,filename))
    
    original_img = img.copy()
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    # check for high values in all 3 channels (colors close to white) 
    # 220 is a bit arbitrary and if there are very light pixels on the borders of the image the cropping will fail
    mask = img > 220
    # apply mask to get only grid
    img[mask] = 255
    img[np.invert( mask)] = 0
    # get consesus over all 3 color channels
    mask2 = np.logical_and((img[:,:,0]==img[:,:,1]), (img[:,:,1]==img[:,:,2]))
    # apply consesus mask
    img[np.invert(mask2)] =0
    
    # get edges of grid
    grid = np.where(img[:,:,0] > 0)
    upborder = min(grid[0])
    lowborder = max(grid[0])
    lborder = min(grid[1])
    rborder = max(grid[1])
    
    # crop to size of grid
    crop_img = original_img[upborder:lowborder, lborder:rborder]
        
    # crop each image into 384 equally sized squares
    h,w = crop_img[:,:,0].shape
    d_w = round(w/24)
    d_h = round(h/16)
    rows = list(string.ascii_uppercase)[0:16]
    cols = [str(i) for i in range(1,25)]
    wellname = list(product(rows,cols))
    
    img = Image.fromarray(np.uint8(crop_img)).convert('RGB')
    grid = list(product(range(0, h-h%d_h, d_h), range(0, w-w%d_w, d_w)))
    for c, g in enumerate(grid):
        i,j = g
        box = (j, i, j+d_h, i+d_w)
        # filename starts with name of well
        out = os.path.join(out_dir,"".join(wellname[c])+"_"+filename)
        img.crop(box).save(out)
        

# set input dir which contains images to crop
# make sure it only contains images with grids to crop
input_dir = "raw_data/PreScreen5_day1_30deg/"
# set output dir
out_dir = "cropped_images/"
# get all files in input dir
input_files = os.listdir(input_dir)
# run cropping for all files
for file in input_files:
    crop(input_dir, file, out_dir)
    
    

