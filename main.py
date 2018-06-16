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
def do_project_video(file_name, output_name):
    height, width = (720,1280)
    
    count=0
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    detection = VehicleDetection(search_params, height, width, loop)
    
    def process_image(image):
        nonlocal count
        
        # process the lane image
        detection.image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        result = cv2.cvtColor(detection.result, cv2.COLOR_BGR2RGB)

        result = cv2.putText(result,'%4d' % count,(5,700), font, 1,(255,255,255),2,cv2.LINE_AA)
        count +=1
        
        rimages.append(result)
        return result

    clip1 = VideoFileClip(file_name)
    lane_clip = clip1.fl_image(process_image) 
    lane_clip.write_videofile(output_name, audio=False)
    
    return lane_clip

# detection_clip = do_project_video("test_video.mp4","test_video_detection.mp4")
detection_clip = do_project_video("project_video.mp4","project_video_detection.mp4")

