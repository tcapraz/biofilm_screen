# import the necessary packages
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
files = os.listdir("in")

notfound =0
ambiguous=0
for f in files:
    xstart = 320
    ystart = 566
    image = cv2.imread(os.path.join("in",f))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = image[320:3120,566:4694].copy()
    imagesmall = image[320:3120,566:4694].copy()
    gray = cv2.cvtColor(imagesmall, cv2.COLOR_BGR2GRAY)
        
    edge = cv2.Canny(gray,50,50)
    
    margin = 50
    offset =200
    
    xspace = np.linspace(0,gray.shape[0], 17, dtype=int)
    yspace = np.linspace(0,gray.shape[1], 25, dtype=int)
    for i in range(len(xspace)-1):
        for j in range(len(yspace)-1): 
            x = xspace[i]
            y = yspace[j]
            window = imagesmall[x:x+offset,y:y+offset,:].copy()
            window_edge = edge[x:x+offset,y:y+offset].copy()
            circles = cv2.HoughCircles(window_edge, cv2.HOUGH_GRADIENT,3,120, minRadius = 30, maxRadius=120)
            if circles is not None:
                cx = int(circles[0,0,1]) +xstart + x
                cy = int(circles[0,0,0]) + ystart + y
                margin= 80
                crop = image[cx-margin:cx+margin,cy-margin:cy+margin]
                cropgray =  cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                ax0 = np.var(cropgray, axis=0)
                ax1 =np.var(cropgray, axis=1)
                edge_var = np.array([np.sum(ax0[0:5]), np.sum(ax0[-5:]), np.sum(ax1[0:5]), np.sum(ax1[-5:])])



                im = Image.fromarray(crop).convert('RGB')
                name,_ = f.split(".")
                # if circles.shape[1] > 1:
                #     fname = name + "_x" + str(j+1) + "_y" + str(i+1) +"_ambiguous.jpg"
                if any(edge_var > 100):
                    fname = name + "_x" + str(j+1) + "_y" + str(i+1) +"_ambiguous.jpg"
                    ambiguous +=1
                    im.save(os.path.join("out_new2/bad",fname))

                else:
                    fname = name + "_x" + str(j+1) + "_y" + str(i+1) +".jpg"
                    im.save(os.path.join("out_new2/clean",fname))

            else:
                notfound +=1
                continue
            
