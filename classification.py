'''
드론 구별 코드 이미지데이터셋 폴더에 드론이있는지 없는지 인식하고 
있으면 successdir 성공폴더로 이미지 이동
없으면 faildir 실패폴더로 이미지를 이동하여 분리한다.
구별한 전체 이미지들은 images폴더에 넣으면 된다.
실행 방법 예 : python classification.py --modeldir=model --imagedir=images --successdir=success --faildir=fail
'''
# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import glob
import importlib.util
import time
import shutil

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.7)

parser.add_argument('--imagedir', help='Name of the folder containing images to perform detection on. Folder must contain only images.',
                    default=images)
parser.add_argument('--successdir', help='Name of the xml folder containing images xml.',
                    default=success)
parser.add_argument('--faildir', help='Name of the folder containing images to perform fail detection on.',
                    default=fail)
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu

# Parse input image name and directory. 
IM_DIR = args.imagedir
FA_DIR = args.faildir
SU_DIR = args.successdir

CWD_PATH = os.getcwd()

if IM_DIR:
    PATH_TO_IMAGES = os.path.join(CWD_PATH,IM_DIR)
    images = glob.glob(PATH_TO_IMAGES + '/*')

if FA_DIR:
    PATH_TO_FAIMAGES = os.path.join(CWD_PATH,FA_DIR)
    failimages = glob.glob(PATH_TO_FAIMAGES)[0]
    
if SU_DIR:
    PATH_TO_SUCCESS = os.path.join(CWD_PATH,SU_DIR)
    successimages = glob.glob(PATH_TO_SUCCESS)[0]



# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'


# Get path to current working directory
CWD_PATH = os.getcwd()

# Define path to images and grab all image filenames


# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details() # 입력 텐서 정보
output_details = interpreter.get_output_details() # 출력 텐서 정보 
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

failcount = 0
successcount = 0
processingtime = 0

# Loop over every image and perform detection
for image_path in images:
    start_time = time.time()
    # Load image and resize to expected shape [1xHxWx3]
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # bgr -> rgb
    imH, imW, _ = image.shape 
    image_resized = cv2.resize(image_rgb, (width, height)) # 원본 이미지 -> (width, height)
    input_data = np.expand_dims(image_resized, axis=0)
    
    filename = image_path.split('\\')[-1]
    print('imagefile : ',filename)
    
    
    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    
    num = []
    
    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            
            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            
            if scores[i] > 0.5:
                num.append(scores[i])

   
    allimagecount = failcount + successcount 
    if len(num) == 0:
        failcount += 1
        print('*******fail: {} proccessing: {}'.format(failcount, allimagecount))
        time.sleep(0.1)
        shutil.move(PATH_TO_IMAGES+'\\'+filename, failimages+'\\'+filename)
        
    if len(num) !=0:
        successcount += 1
        print('success: {} fail: {} proccessing: {}'.format(successcount, failcount ,allimagecount)) 
        shutil.move(PATH_TO_IMAGES+'\\'+filename, successimages+'\\'+filename)
    # All the results have been drawn on the image, now display the image
    #cv2.imshow('filename', image)
    
    end_time = time.time()
    process_time = end_time - start_time
    processingtime = processingtime + process_time
    print("======== A frame took {:.3f}seconds========Whole time : {:.3f}seconds ({:.1f}min)===========".format(process_time,processingtime,processingtime/60))   
    # Press any key to continue to next image, or press 'q' to quit
    if cv2.waitKey(0) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()

print('End--------------------------------------------------------------------------')
print('Fail Number of Image : ', failcount)
print('Success Number of Image : ', successcount)
print('All Number of Image : ', allimagecount)


