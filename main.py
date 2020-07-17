from flask import Flask, request, make_response, render_template, redirect, url_for, send_file
from PIL import Image
import tensorflow as tf
import numpy as np

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import os
ON_HEROKU = os.environ.get('ON_HEROKU')

app = Flask(__name__)

#set paths to upload folder
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['IMAGE_UPLOADS'] = os.path.join(APP_ROOT, 'static')

app.config['MODEL'] = os.path.join(APP_ROOT, 'model')

# path to the frozen graph:
PATH_TO_FROZEN_GRAPH = os.path.join(app.config['MODEL'], 'frozen_inference_graph.pb')

# path to the label map
PATH_TO_LABEL_MAP = os.path.join(app.config['MODEL'], 'label_map.pbtxt')

# number of classes
NUM_CLASSES = 1

# reads the frozen graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABEL_MAP)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

@app.route("/")
def hello():
    return "Hello, World!"

@app.route("/landing_page",methods=["GET","POST"])
def landing_page():
    return render_template("upload.html", image_path = 'landing_page_pic.jpg')

@app.route("/detect", methods=["GET", "POST"])
def detect():
    # read and upload resized files to folder
    image = request.files['input_file']
    filename = image.filename
    file_path = os.path.join(app.config["IMAGE_UPLOADS"], filename)
    image_pil = Image.open(image)
    image_np = np.array(image_pil)

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Extract image tensor
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Extract detection boxes
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Extract detection scores
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    # Extract detection classes
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    # Extract number of detections
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')
    # Actual detection.
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=3,
            )

    image_processed = Image.fromarray(image_np)
    image_processed.thumbnail((1200, 600), Image.ANTIALIAS)
    image_processed.save(file_path)

    return render_template("upload.html", image_path = filename)

if __name__ == '__main__':

    if ON_HEROKU == 'true':
        # get the heroku port
        print("Running on Heroku")
        expose_port = int(os.environ.get('PORT', 8000))  # as per OP comments default is 17995
    else:
        print("Running locally")
        expose_port = 8000
    app.run(host='0.0.0.0', port=expose_port)