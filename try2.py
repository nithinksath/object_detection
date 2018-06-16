import cv2
import numpy as np
import os
import glob
#for filename in os.listdir('/home/sakthi/Nithin/seventhtuesday/non-vehicles/GTI'):
    #if filename.endswith("image1.png"):
	#mat=cv2.imread(os.path.join('/home/sakthi/Nithin/seventhtuesday/non-vehicles/GTI', filename))
	#x = np.arange(20).reshape((4,5))
	#np.savetxt("foo.csv", mat, delimiter=",")
        #print(os.path.join('/home/sakthi/Nithin/seventhtuesday/non-vehicles/GTI', filename))
        #continue
    #else:
        #continue

#non_car=glob.glob('/home/sakthi/Nithin/seventhtuesday/non-vehicles/GTI/*.png')
#car=glob.glob('/home/sakthi/Nithin/seventhtuesday/vehicles/KITTI_extracted/*.png')
#print(len(car),len(non_car))
#for i in range(0,len(car)):
	#img=cv2.imread(car[i])
	#print(img)	
#def readImages(dir, pattern):
    
    #images = []
    #for dirpath, dirnames, filenames in os.walk(dir):
        #for dirname in dirnames:
            #images.append(glob.glob(dir + '/' + dirname + '/' + pattern))
    #flatten = [item for sublist in images for item in sublist]
    #return list(map(lambda img: cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB), flatten))

#vehicles = readImages('/home/sakthi/Nithin/vehicles/KITTI_extracted', '*.png')
#non_vehicles = readImages('/home/sakthi/Nithin/non-vehicles/GTI', '*.png')
#images=[]
#new1=os.listdir('/home/sakthi/Nithin/seventhtuesday/non-vehicles/GTI')
#new2=os.listdir('/home/sakthi/Nithin/seventhtuesday/vehicles/KITTI_extracted')
#for filename in new1:
	#if filename.endswith(".png"):
        	#images.append(filename)
#flatten = [item for sublist in images for item in sublist]
#list(map(lambda img: cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB), flatten))
#print(images)




