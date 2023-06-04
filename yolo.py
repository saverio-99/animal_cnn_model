import cv2
import darknet

# Load the pre-trained model
net = darknet.load_net("yolo.cfg", "yolo.weights")
meta = darknet.load_meta("coco.data")

# Define the classes you want to detect
classes = ['bear', 'wolf', 'fox']

# Function to perform object detection and classification
def detect_objects(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to a format suitable for YOLO
    darknet_image = darknet.make_image(darknet.network_width(net), darknet.network_height(net), 3)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    darknet.copy_image_from_bytes(darknet_image, image_rgb.tobytes())

    # Perform object detection
    detections = darknet.detect_image(net, meta, darknet_image)

    # Process the detections
    for detection in detections:
        class_name = detection[0].decode()
        if class_name in classes:
            # Raise an alarm or perform any desired action
            print("Alarm! Detected:", class_name)

    # Release resources
    cv2.destroyAllWindows()
    darknet.free_image(darknet_image)

# Call the function with the path to the image
detect_objects("urs1.jpg")
