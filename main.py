# Import necessary libraries
import numpy as np
import cv2

# Path to the image and the pre-trained MobileNet-SSD model files
image_path = 'image1.jpg'  # You can change this to the path of the image you want to test
prototxt_path = 'models/MobileNetSSD_deploy.prototxt'  # Path to the network architecture file
model_path = 'models/MobileNetSSD_deploy.caffemodel'  # Path to the pre-trained model weights

# Minimum confidence threshold for filtering weak object detections
min_confidence = 0.2

# List of class labels MobileNet-SSD was trained to detect (21 classes)
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

# Generate random colors for each class to use in bounding boxes and text
np.random.seed(543210)  # Set a seed for reproducibility
colors = np.random.uniform(0, 255, size=(len(classes), 3))  # Generate random RGB colors for each class

# Load the pre-trained MobileNet-SSD model from disk
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Initialize video capture object to access the webcam (use 0 for default camera)
cap = cv2.VideoCapture(0)

# Main loop to process video frames
while True:
    # Capture the current frame from the webcam
    _, image = cap.read()

    # Optionally, you can load a single image from disk using cv2.imread
    # image = cv2.imread(image_path)

    # Get the dimensions of the image (height and width)
    height, width = image.shape[0], image.shape[1]

    # Preprocess the image to create an input blob for the neural network
    # Resize the image to 300x300, normalize it, and subtract the mean value
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007, (300, 300), 130)

    # Set the blob as input to the network and perform forward pass to get detections
    net.setInput(blob)
    detected_objects = net.forward()

    # Print the detection output for debugging (optional)
    print(detected_objects[0][0][0])

    # Loop over the detected objects
    for i in range(detected_objects.shape[2]):
        # Extract the confidence (probability) of the current detection
        confidence = detected_objects[0][0][i][2]

        # Filter out weak detections by ensuring the confidence is above the minimum threshold
        if confidence > min_confidence:
            # Extract the index of the detected class label
            class_index = int(detected_objects[0, 0, i, 1])

            # Compute the (x, y) coordinates of the bounding box for the detected object
            upper_left_x = int(detected_objects[0, 0, i, 3] * width)
            upper_left_y = int(detected_objects[0, 0, i, 4] * height)
            lower_right_x = int(detected_objects[0, 0, i, 5] * width)
            lower_right_y = int(detected_objects[0, 0, i, 6] * height)

            # Create the label text including the class name and confidence score
            prediction_text = f'{classes[class_index]}: {confidence:.2f}%'

            # Draw the bounding box around the detected object
            cv2.rectangle(image, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), colors[class_index], 3)

            # Put the class label and confidence score above the bounding box
            cv2.putText(image, prediction_text, 
                        (upper_left_x, upper_left_y - 15 if upper_left_y > 30 else upper_left_y + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[class_index], 2)

    # Display the image with detected objects
    cv2.imshow("Detected Objects", image)

    # Wait for a short period (5 ms) and check if the user presses a key to exit
    cv2.waitKey(5)

# Release the video capture object and close all OpenCV windows
cv2.destroyAllWindows()
cap.release()
