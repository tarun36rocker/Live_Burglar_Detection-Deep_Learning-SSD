from __future__ import print_function, division
from builtins import range, input
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util

# Define the video stream
cap = cv2.VideoCapture(0)

# Note: you may need to update your version of future
# sudo pip install -U future

import os, sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import imageio
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
 
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
 
sys.path.append("..")
from object_detection.utils import ops as utils_ops
 
from object_detection.utils import label_map_util
utils_ops.tf = tf.compat.v1
from object_detection.utils import visualization_utils as vis_util
 

if tf.__version__ < '1.4.0':
  raise ImportError(
    'Please upgrade your tensorflow installation to v1.4.* or later!'
  )


# change this to wherever you cloned the tensorflow models repo
# which I assume you've already downloaded from:
# https://github.com/tensorflow/models
RESEARCH_PATH = 'C:/Users/Tarun/Desktop/comp/Computer vision course/object detection/models-master/research'
MODELS_PATH = 'C:/Users/Tarun/Desktop/comp/Computer vision course/object detection/models-master/research/object_detection'
sys.path.append(RESEARCH_PATH)
sys.path.append(MODELS_PATH)

# import local modules
import object_detection
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# I've assumed you already ran the notebook and downloaded the model
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
 
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
 
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
 
NUM_CLASSES = 90
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())
 

# load the model into memory
    
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')




# load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
print("categories:")
print(categories)
person_sign_class_id = 1
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# Detection
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while True:

            # Read frame from camera
            ret, image_np = cap.read()
            from datetime import datetime
            #print(datetime.now())
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image_np,str(datetime.now()),(0,40), font, .5,(255,255,255),2,cv2.LINE_AA)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            cv2.putText(image_np,"CCTV",(380,80), font, .5,(255,255,255),2,cv2.LINE_AA)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Extract image tensor
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Extract detection boxes
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Extract detection scores
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # Extract detection classes
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # Extract number of detectionsd
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            count=0
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index ,
                count,
                use_normalized_coordinates=True,
                line_thickness=8)
            '''vis_util.abc50(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index ,
                count,
                use_normalized_coordinates=True,
                line_thickness=8)'''

      
            # Display output
            cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
'''print('Total occurences:',count2)'''