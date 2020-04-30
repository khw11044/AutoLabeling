import os
import argparse
import cv2
import numpy as np
import sys
import glob
import importlib.util
import time
import shutil

min_confidence = 0.1
margin = 30
#file_name = "drone3.jpg"

# Load Yolo
net = cv2.dnn.readNet("yolo/yolo-drone.weights", "yolo/yolo-drone.cfg")
classes = []
with open("yolo/drone.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
print(classes)
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

parser = argparse.ArgumentParser()
parser.add_argument('--imagedir', help='Name of the folder containing images to perform detection on. Folder must contain only images.',
                    default=None)
parser.add_argument('--failimagedir', help='Name of the folder containing images to perform fail detection on.',
                    default=None)
parser.add_argument('--xmldir', help='Name of the xml folder containing images xml.',
                    default=None)

args = parser.parse_args()

# Parse input image name and directory. 
IM_DIR = args.imagedir
FA_DIR = args.failimagedir
XML_DIR = args.xmldir
    
# Get path to current working directory
CWD_PATH = os.getcwd()

# Define path to images and grab all image filenames
if IM_DIR:
    PATH_TO_IMAGES = os.path.join(CWD_PATH,IM_DIR)
    images = glob.glob(PATH_TO_IMAGES + '/*')

if FA_DIR:
    PATH_TO_FAIMAGES = os.path.join(CWD_PATH,FA_DIR)
    failimages = glob.glob(PATH_TO_FAIMAGES)[0]
    
if XML_DIR:
    PATH_TO_XML = os.path.join(CWD_PATH,XML_DIR)
    xmlfolder = glob.glob(PATH_TO_XML)[0]
    

for image_path in images:
    filename = image_path.split('/')[-1]
    filename2 = filename.split('.')[0]
    print('imagefolder',PATH_TO_IMAGES)
    print('failfolder',failimages)
    print('image_path',image_path)
    print('image file name',filename)
    print('image file name',filename2)
    # Loading image
    start_time = time.time()
    img = cv2.imread(image_path)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # Filter only 'drone'
            if class_id == 0 and confidence > min_confidence:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    color = (0, 255, 0)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = '{:,.2%}'.format(confidences[i])
            print(i, label)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            shutil.move(PATH_TO_IMAGES+'/'+filename, xmlfolder+'/'+filename)
            print("succese")
        
    #No Drone --> relocate        
    if len(indexes) == 0:
        print("fail")
        try:
            shutil.move(PATH_TO_IMAGES+'/'+filename, failimages+'/'+filename)
        except FileNotFoundError :
            print('error')
        
    end_time = time.time()
    process_time = end_time - start_time
    print("=== A frame took {:.3f} seconds".format(process_time))
   

        
        
    
           



