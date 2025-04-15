# src/detection/detector.py
import cv2
import numpy as np
import os
import sys

class FaceDetector:
    """
    Handles face detection using OpenCV's DNN module with a pre-trained model.
    """
    def __init__(self, prototxt_path, model_path, confidence_threshold=0.5,
                 input_width=300, input_height=300, mean_subtraction=(104.0, 177.0, 123.0)):
        """
        Initializes the face detector.

        Args:
            prototxt_path (str): Path to the .prototxt file defining the network architecture.
            model_path (str): Path to the .caffemodel (or other format) file with weights.
            confidence_threshold (float): Minimum confidence score to consider a detection valid.
            input_width (int): Width the input image is resized to for the network.
            input_height (int): Height the input image is resized to for the network.
            mean_subtraction (tuple): Mean values (B, G, R) to subtract from the image.
        """
        self.prototxt_path = prototxt_path
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.input_size = (input_width, input_height)
        self.mean_subtraction = mean_subtraction
        self.net = self._load_model()

    def _load_model(self):
        """Loads the DNN model from the specified files."""
        if not os.path.exists(self.prototxt_path):
            raise FileNotFoundError(f"Detector prototxt file not found at: {self.prototxt_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Detector model weights file not found at: {self.model_path}")

        print("Loading face detector model...")
        try:
            # Load model based on file extensions (Caffe in this case)
            # Add checks here if supporting TF, ONNX etc.
            net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.model_path)
            print("Face detector model loaded successfully.")
            return net
        except cv2.error as e:
            print(f"Error loading DNN model: {e}", file=sys.stderr)
            print("Ensure OpenCV was built with DNN support and model files are correct.", file=sys.stderr)
            raise

    def detect_faces(self, frame):
        """
        Detects faces in a given frame.

        Args:
            frame (numpy.ndarray): The input image frame (BGR format).

        Returns:
            list: A list of tuples, where each tuple represents a bounding box
                  in the format (x1, y1, x2, y2) for a detected face.
                  Coordinates are relative to the original frame dimensions.
        """
        if self.net is None:
            print("Error: Face detector network not loaded.", file=sys.stderr)
            return []

        # Get frame dimensions
        (h, w) = frame.shape[:2]

        # Create a blob from the image - this preprocesses the image for the network
        # 1. Resizes to self.input_size (e.g., 300x300)
        # 2. Applies scaling factor (1.0 here - no change)
        # 3. Subtracts mean values (self.mean_subtraction)
        # 4. swapRB is False as OpenCV loads images in BGR by default
        blob = cv2.dnn.blobFromImage(
            image=frame,
            scalefactor=1.0,
            size=self.input_size,
            mean=self.mean_subtraction,
            swapRB=False,
            crop=False # Do not crop after resize
        )

        # Pass the blob through the network
        self.net.setInput(blob)
        detections = self.net.forward() # Shape depends on model, often [1, 1, N, 7] for SSD

        detected_boxes = []

        # Loop over the detections
        # The shape of 'detections' for SSD models is typically [1, 1, num_detections, 7]
        # where the last dimension contains [batchId, classId, confidence, x1, y1, x2, y2]
        # (Coordinates are normalized between 0 and 1 relative to blob size)
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections
            if confidence > self.confidence_threshold:
                # Extract the coordinates of the bounding box and scale them
                # back to the original frame dimensions
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Ensure the bounding box coordinates are within the frame boundaries
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w - 1, endX)
                endY = min(h - 1, endY)

                # Ensure width and height are positive
                if endX > startX and endY > startY:
                    detected_boxes.append((startX, startY, endX, endY))

        return detected_boxes