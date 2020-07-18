
'''
자동라벨링코드이다. 
먼저 classification.py를 통해 분리된 확실히 드론이라고 인식된 이미지들이 모여있는 success폴더에
있는 이미지들에 객체 라벨링을 한다.
스페이스를 누르면 진행된다. 스페이스를 누르면 자동라벨링이되고 이상한 곳에 라벨링을 했다던가 라벨크기가 너무 크면
d 를 누르면 된다. 하다 지치거나 중단하고 싶으면 q를 누르면 된다.
실행 코드는 python autolabel.py --modeldir=model --imagedir=success --successdir=successfolder --faildir=fail 
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
                    default=0.8)
parser.add_argument('--imagedir', help='Name of the folder containing images to perform detection on. Folder must contain only images.',
                    default=None)
parser.add_argument('--successdir', help='Name of the xml folder containing images xml.',
                    default=None)
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
    successfolder = glob.glob(PATH_TO_SUCCESS)[0]



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
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

def generatefakefile(paths):
    f = open(paths+'\\'+"fail.xml", 'w')
    f.write('fail')
    f.close

input_mean = 127.5
input_std = 127.5
print('First Press Anykey')
# Loop over every image and perform detection
for image_path in images:
    start_time = time.time()
    # Load image and resize to expected shape [1xHxWx3]
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape 
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)
    
    filename = image_path.split('\\')[-1]
    filename2 = str(filename.split('.')[0])
    filename3 = filename2 + '.xml'
    print('---------------------------------start--------------------------------------------------')
    print('image file name',filename)
    print('image file name',filename2)
    print('image file name',filename3)
    
    
    
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
            
            print("xmin,ymin,xmax,ymax",xmin,ymin,xmax,ymax)
            
            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            
            if scores[i] > 0.5:
                num.append(scores[i])

            label = '%s: %d%%' % ('Drone', int(scores[i]*100)) # Example: 'person: 72%' #object_name
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

    
    
    if len(num) == 0:
        cv2.putText(image,'NO DRONE ',(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
        cv2.putText(image,'JUST PRESS ',(40,180),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        cv2.putText(image,'"'"SPACE"'" key',(30,220),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)       
        print('fail------------------')
        time.sleep(0.1)
        shutil.move(PATH_TO_IMAGES+'/'+filename, failimages+'/'+filename)
        
    if len(num) !=0:
        print(len(num))
        print('success')
        f = open(successfolder+'\\'+filename2+".xml", 'w')
        time.sleep(0.1)
        f.write('<annotation>\n\t<folder>'+PATH_TO_IMAGES+'</folder>\n')
        f.write('\t<filename>'+filename+'</filename>\n<path>'+filename2+'.jpg</path>\n')
        f.write('\t<source>\n\t\t<database>Unknown</database>\n\t</source>\n')
        f.write('\t<size>\n\t\t<width>300</width>\n\t\t<height>300</height>\n')
        f.write('\t\t<depth>3</depth>\n\t</size>\n\t<segmented>0</segmented>\n')
        for i in num:
            f.write('\t<object>\n\t\t<name>Drone</name>\n\t\t<pose>Unspecified</pose>\n')
            f.write('\t\t<truncated>0</truncated>\n\t\t<difficult>0</difficult>\n')
            f.write('\t\t<bndbox>\n\t\t\t<xmin>'+str(xmin)+'</xmin>\n')
            f.write('\t\t\t<ymin>'+str(ymin)+'</ymin>\n')
            f.write('\t\t\t<xmax>'+str(xmax)+'</xmax>\n')
            f.write('\t\t\t<ymax>'+str(ymax)+'</ymax>\n')
            f.write('\t\t</bndbox>\n\t</object>\n')
        f.write('</annotation>')
        time.sleep(0.1)
        f.close
        time.sleep(0.05)
        shutil.move(PATH_TO_IMAGES+'\\'+filename, successfolder+'\\'+filename)
        time.sleep(0.1)
        print('createXML')
        # All the results have been drawn on the image, now display the image
    cv2.imshow('filename ', image)
    
    end_time = time.time()
    process_time = end_time - start_time
    print("======== A frame took  {:.3f}seconds=================".format(process_time))
    print('Choose Success Or Fail')

    if cv2.waitKey(0) == ord('d'):
        f = open(successfolder+'\\'+"fail.xml", 'w')
        f.write('fail')
        f.close
        print('you pressed d')
        shutil.move(successfolder+'\\'+filename, failimages+'\\'+filename)
        time.sleep(0.1)
        print("----------------------------sendtofailfoder------------------------------")
        os.remove(successfolder+'\\'+filename3)
        print('delete')
        time.sleep(0.1)
        print("fail")
        cv2.destroyAllWindows()
        
    # Press any key to continue to next image, or press 'q' to quit
    if cv2.waitKey(0) == ord('q'):
        f = open(successfolder+'\\'+"fail.xml", 'w')
        f.write('fail')
        f.close
        print('End of proccessing ')
        shutil.move(successfolder+'\\'+filename, PATH_TO_IMAGES+'\\'+filename)
        time.sleep(0.1)
        os.remove(successfolder+'\\'+filename3)
        time.sleep(0.1)
        generatefakefile(failimages)
        break

# Clean up
cv2.destroyAllWindows()
f = open(failimages+'\\'+"fail.xml", 'w')
f.write('fail')
f.close
os.remove(successfolder+'\\'+'fail.xml')
print('all end')




