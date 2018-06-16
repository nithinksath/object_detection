import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
import pickle
from extract_features import  *
from scipy.ndimage.measurements import label

def check_box_size(box,sizex=15,sizey=15):
    if((abs(box[0][0] - box[1][0]) < sizex) or (abs(box[1][1] - box[0][1]) < sizey)):
        return False
    else:
        return True

#creates a heat map images for given bounding box
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        if check_box_size(box) == True:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes

#returns the sum of heat map images in the heatmap list
def add_heat_overFrames(heatmap_list):
    image = np.zeros_like(heatmap_list[0])
    for heatmap in heatmap_list:
        image = cv2.add(image,heatmap)
    return image

#apply thresholds to heatmap images
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

# draws bounding box over the different segments of the label
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        # Draw the box on the image
        if check_box_size(bbox,64,64) == True:
            cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 3)
    # Return the image
    return img


# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


#  A function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat=True,
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #4) Extract features for that window using construct_features_single()
        features = single_img_features(test_img, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

#load the trained model
filename = 'model_2.dat'
svc = pickle.load(open(filename, 'rb'))
print("model loaded")

#loads the scaler
filename_scaler = 'scaler_2.dat'
X_scaler = pickle.load(open(filename_scaler, 'rb'))
print("model loaded")

#slides a list of windows on top of the image
def slide_multiple(image,x_start_stop_list,y_start_stop,windows_list,xy_overlap_list):
    params = get_features_parameters()
    box_list = []
    for i in range(len(x_start_stop_list)):
        windows = slide_window(image, x_start_stop=x_start_stop_list[i], y_start_stop=y_start_stop[i],
                             xy_window= windows_list[i], xy_overlap=xy_overlap_list[i])

        hot_windows = search_windows(image,windows, svc, X_scaler, color_space=params['color_space'],
                                   spatial_size=params['spatial_size'], hist_bins=params['hist_bins'],
                                   orient=params['orient'], pix_per_cell=params['pix_per_cell'],
                                   cell_per_block=params['cell_per_block'],
                                   hog_channel=params['hog_channel'], spatial_feat=params['spatial_feat'],
                                   hist_feat=params['hist_feat'], hog_feat=params['hog_feat'])
        box_list = box_list + hot_windows
    return box_list

#pipeline that process the video
def play_video(filename,output,thres,thres2):
    cap = cv2.VideoCapture(filename)
    first_frame = False
    #define the last n frames for which heat map will be stored.
    max_continuous_frames = 5

    #a list that stores heat map of last five images
    heatmap_list = []

    #counter to track heatmap_list
    counter = 0
    #count total number of frames inv ideo
    total_frames = 0

    #out video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output,fourcc, 25.0, (1280,720))

    while (cap.isOpened()):

        ret, frame = cap.read()

        if frame == None:
            break
        image = frame #cv2.imread("test_images/test1.jpg")

        # single heat map
        heat = np.zeros_like(image[:, :, 0]).astype(np.float)
        draw_image = np.copy(image)

        #starting for search window 1 32 X 32


        y_start_stop_list = [[400, 700],[400,700]]
        x_start_stop_list = [[200, None],[200, None]]
        windows_list = [(64, 64),(128, 128)]
        xy_overlap_list = [(0.8, 0.8),(0.8, 0.8)]


        box_list = slide_multiple(image, x_start_stop_list, y_start_stop_list, windows_list, xy_overlap_list)

        # Add heat to each box in box list
        heat = add_heat(heat, box_list)

        # Apply threshold to help remove false positives for the given heat image
        heat = apply_threshold(heat, thres)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        #check to prevent indexing into empty list
        if(total_frames < 6):
            heatmap_list.append(heatmap)
        else:
            heatmap_list[counter] = heatmap

        #reset the counter to zero, to replace the oldest frame in last five frames
        if(counter > max_continuous_frames-1):
            counter = 0
        else:
            counter = counter + 1

        #combine heat map over given n frames
        combined_heat = add_heat_overFrames(heatmap_list)
        #apply threshold
        combined_heat = apply_threshold(combined_heat, thres2)
        #clip to 0,255
        combined_heat = np.clip(combined_heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(combined_heat)
        draw_img = draw_labeled_bboxes(np.copy(image), labels)
        cv2.imshow("original", draw_img)

        cv2.imshow("combined heat map", heat)

        out.write(draw_img)
        total_frames = total_frames + 1
        print(total_frames)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


