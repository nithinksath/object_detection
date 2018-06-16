import cv2
import numpy as np
import os
import glob
import time
from features import Features
from features import ColorHistFeatures, HogImageFeatures
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import *
from features import *
from boxes import *
from images import *
from detection import *
from search import *
from ipywidgets import interact, interactive, fixed
from matplotlib import cm
from scipy import ndimage

timages, tfiles = load_test_images()
arr2img(cv2.cvtColor(timages[0], cv2.COLOR_BGR2RGB))
pimages, pframes = load_test_video(file_name='/home/sakthi/Nithin/seventhtuesday/project_video.mp4')

car_images, not_car_images = load_car_not_car_images()
print(len(car_images),len(not_car_images))
cars = car_images
notcars = not_car_images
colorspace = 'YCrCb' # Can be BGR, HSV, LUV, HLS, YUV, YCrCb
orient = 32
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
t=time.time()
car_features = extract_hog_features(cars, cspace=colorspace, orient=orient,
pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
hog_channel=hog_channel)
notcar_features = extract_hog_features(notcars, cspace=colorspace, orient=orient,
pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
hog_channel=hog_channel)
X = np.vstack((car_features, notcar_features)).astype(np.float64)
X_scaler = RobustScaler().fit(X)
scaled_X = X_scaler.transform(X)
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
scaled_X, y, test_size=0.2, random_state=rand_state)
print('Using:',hog_channel,'hog channel',orient,'orientations',pix_per_cell,
'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
svc = LinearSVC()
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
print('Train Accuracy of SVC = ', round(svc.score(X_train, y_train), 4))
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
wb = WindowBoxes(height=car_images[0].shape[0], width=car_images[0].shape[1])
total=0
for size, windows in wb.window_boxes_dict.items():
    count=len(windows)
    total = total + count
    print("{} count {}".format(size, count))

print("total", total)
        


# initialise an object to contain common parameters
color_space = 'YCrCb' # Can be BGR, HSV, LUV, HLS, YUV, YCrCb
orient = 32  # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
hist_range = (0,256)
spatial_feat = False # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = True # HOG features on or off
search_params = SearchParams(color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size,
hist_bins, hist_range, spatial_feat, hist_feat, hog_feat, svc, X_scaler)


loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

rimages=[]
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

vehicle_detection = VehicleDetection(search_params,timages[0].shape[0], timages[0].shape[1], loop)

@interact
def hot_windows_test(images=fixed(pimages), i:(0,len(pimages)-1)=0):

#     vehicle_detection = VehicleDetection(search_params,images[i].shape[0], images[i].shape[1], loop)
    vehicle_detection.image = images[i]
    
    heatmap = vehicle_detection.heatmap
    heatmap_history = vehicle_detection.heatmap_history
    labels=vehicle_detection.labels
#     print("history count", vehicle_detection.heatmap_history_count)
    print("label car count", labels[1], labels[0].shape, labels[0][0].shape)
    print("box variance", np.around(vehicle_detection.box_variance,decimals=3))
    print("labelled boxes", vehicle_detection.labelled_boxes)
    print("planes", vehicle_detection.label_box_planes)
    
    for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            #print(car_number)
            nonzeroz = np.array(nonzero[0])
            nonzeroy = np.array(nonzero[1])
            nonzerox = np.array(nonzero[2])
            #print(nonzeroy, nonzerox, nonzeroz)
    

    nplots=vehicle_detection.heatmap_history_count
    fig = plt.figure(figsize=(15,vehicle_detection.heatmap_history_count*5))
    
    for i in range(0,vehicle_detection.heatmap_history_count):
        plt.subplot(1, nplots, i+1)
        
        heatmap=heatmap_history[i]
        plt.imshow(heatmap,cmap='inferno')
        plt.xticks([]), plt.yticks([])
    fig.tight_layout()    
    
    
#     fig = plt.figure(figsize=(12,10))
#     plt.subplot(131)
    
#     print(labels[1], 'cars found')
#     plt.imshow(labels[0][0], cmap='gray')
#     plt.subplot(132)
#     plt.imshow(heatmap)
    
#     f, ax = plt.subplots(133,figsize=(6, 6))
#     nonzero = (labels[0] > 0).nonzero()
#     nonzeroz = np.array(nonzero[0])
#     nonzeroy = np.array(nonzero[1])
#     nonzerox = np.array(nonzero[2])
  
#     ax = Axes3D(f)
#     ax.scatter(nonzerox,nonzeroy,nonzeroz)
    #return arr2img(labels[0])
    
#     return arr2img(vehicle_detection.heatmap_decorated)
    return arr2img(cv2.cvtColor(cv2.resize(vehicle_detection.result,None,fx=.5,fy=.5),cv2.COLOR_BGR2RGB))

def do_project_realtime(frame_r):
    #height,width= (720,1280)
    height=frame_r.shape[0]
    width=frame_r.shape[1]
    #print(height,width)
    count=0
    font=cv2.FONT_HERSHEY_SIMPLEX
    detection= VehicleDetection(search_params, height,width,loop)
    

    detection.image=cv2.cvtColor(frame_r,cv2.COLOR_RGB2BGR)
    result=cv2.cvtColor(detection.result,cv2.COLOR_BGR2RGB)
    result=cv2.putText(result,'%4d' % count,(5,700), font, 1,(255,255,255),2,cv2.LINE_AA)
    count+=1
    cv2.imshow('DETECTED',result)
    cv2.waitKey(3)
    #rimages.append(result)
    #return result
#f_test=cv2.imread('/home/sakthi/Nithin/seventhtuesday/test_images/test1.jpg')
#do_project_realtime(f_test)
#pimages, pframes = load_test_video(file_name='/home/sakthi/Nithin/seventhtuesday/project_video.mp4')
#hot_windows_test()
cap=cv2.VideoCapture(0)
while(cap.isOpened()):
    ret,frame=cap.read()
    while ret:
        ret,frame_s=cap.read()
        
        cv2.imshow('ORIGINAL',frame_s)
        cv2.waitKey(3)
        #gray=cv2.cvtColor(frame_s,cv2.COLOR_BGR2GRAY)
        
        do_project_realtime(frame_s)
        
cap.release()
out.release()
cv2.destroyAllWindows
