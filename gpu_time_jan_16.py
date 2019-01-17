import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# from utils import label_map_util

# from utils import visualization_utils as vis_util

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

print("model")
# What model to download.
MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

print("urllib")
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())

print("graph/")
'''
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()

    
with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)

        tf.import_graph_def(od_graph_def, name='')
'''

# category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = os.path.join(PATH_TO_TEST_IMAGES_DIR)

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
 
                start = time.time()
                ops = tf.get_default_graph().get_operations()
                end = time.time()
                print(f'tf.get_default_graph().get_operations(): {end - start}')
                all_tensor_names = {}

                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes'#, 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name
                            )
                '''
                if 'detection_masks' in tensor_dict:
                    start = time.time()
                    # the following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                    end = time.time()
                    print(f'if detection_masks in tensor_dict:: {end - start}')
                '''

                start = time.time()
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
                end = time.time()
                print(f'tf.get_default_graph().get_tensor_by_name {end - start}')
                # up to here can be put outside no need to be repeated

sess = tf.Session() 
def run_inference_for_single_image(image):
   # with tf.device('/device:GPU:'):
                # Get handles to input and output tensors
                              # Run inference
                start = time.time()
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: np.expand_dims(image, 0)})
                end = time.time()
                print(f'sess.run {end - start}')

                # all outputs are float32 numpy arrays, so convert types as appropriate
                start = time.time()
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                end = time.time()
                print(f'output_dict {end - start}')

                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
                return output_dict



'''
def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            start = time.time()
            ops = tf.get_default_graph().get_operations()
            end = time.time()
            print(f'tf.get_default_graph().get_operations(): {end - start}')
            all_tensor_names = {}

            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                start = time.time()
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
                end = time.time()
                print(f'if detection_masks in tensor_dict:: {end - start}')

            start = time.time()
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            end = time.time()
            print(f'tf.get_default_graph().get_tensor_by_name {end - start}')

            # Run inference
            start = time.time()
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})
            end = time.time()
            print(f'sess.run {end - start}')

            # all outputs are float32 numpy arrays, so convert types as appropriate
            start = time.time()
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            end = time.time()
            print(f'output_dict {end - start}')

            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

'''

import json

PATH_DIR = '/dvmm-filer2/projects/Hearst/keyframes/'

import time




print("hi")
image_dict = {}
counter = 0
flag = False
for root, dirs, files in os.walk(PATH_DIR):

    if flag == True:
        break
    for file in [f for f in files if f.endswith(".png")]:
        counter += 1
        print(counter)
        if counter == 5:
            flag = True
            break
        print(file)

        start = time.time()
        image = Image.open(os.path.join(root, file))
        print(f'image size - {image.size}')
        end = time.time()
        print(f'Image.open: {end - start}')

        # resize the image
        image = image.resize((600, 600), Image.ANTIALIAS)
        print(f'after resize image size - {image.size}')
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.

        start = time.time()
        image_np = load_image_into_numpy_array(image)
        end = time.time()
        print(f'load_image_into_numpy_array: {end - start}')

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]

        start = time.time()
        image_np_expanded = np.expand_dims(image_np, axis=0)
        end = time.time()
        print(f'np.expand_dims: {end - start}')

        # Actual detection.
        start = time.time()
        output_dict = run_inference_for_single_image(image_np)
        end = time.time()
        print(f'run_inference_for_single_image: {end - start}')

        start = time.time()
        image_dict.update({os.path.join(root, file): output_dict})
        end = time.time()
        print(f'image_dict.update: {end - start}')
        # save the results

        ''' 
        # Visualization of the results of a detection.
        print("new IMAGE image should show - 1 starting 0.5")
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            min_score_thresh=0.5,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
        plt.figure(figsize=IMAGE_SIZE)
        plt.imshow(image_np)
        '''


