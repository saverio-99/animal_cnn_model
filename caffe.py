import cv2
import numpy as np
import caffe

# Load the pre-trained model and labels
model_path = 'C:/Users/sandor/Documents/animals/caffe_model.prototxt'
weights_path = 'C:/Users/sandor/Documents/animals/caffe_model.caffemodel'
labels_path = 'C:/Users/sandor/Documents/animals/labels.txt'

# Load the model and set it to GPU mode if available
net = caffe.Net(model_path, weights_path, caffe.TEST)
if caffe.cuda.is_available():
    caffe.cuda.set_device(0)
    net.set_mode_gpu()

# Load the class labels
with open(labels_path, 'r') as f:
    labels = f.read().splitlines()

# Define the classes you want to detect
classes = ['bear', 'wolf', 'fox']

# Function to perform object detection and classification
def detect_objects(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Preprocess the image
    blob = caffe.io.blobFromImage(image, mean=np.array([104, 117, 123]), swapRB=True)

    # Set the image as input to the network
    net.blobs['data'].reshape(1, 3, image.shape[0], image.shape[1])
    net.blobs['data'].data[...] = blob

    # Forward pass through the network
    detections = net.forward()['detection_out']

    # Process the detections
    for detection in detections[0, 0]:
        confidence = detection[2]
        class_id = int(detection[1])

        if confidence > 0.5 and labels[class_id] in classes:
            class_name = labels[class_id]
            # Raise an alarm or perform any desired action
            print("Alarm! Detected:", class_name)

    # Release resources
    cv2.destroyAllWindows()

# Call the function with the path to the image
detect_objects("urs1.jpg")
