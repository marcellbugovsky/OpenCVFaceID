# src/recognition/encoder.py
import cv2
import numpy as np
import os
import sys

class FaceEncoder:
    """
    Handles loading a face recognition model (ONNX format assumed for now)
    and generating face embeddings using OpenCV's DNN module.
    """
    def __init__(self, model_path, input_width=112, input_height=112,
                 scale_factor=1.0/255.0, swap_rb=False):
        """
        Initializes the face encoder.

        Args:
            model_path (str): Path to the ONNX model file.
            input_width (int): Width the input face image is resized to.
            input_height (int): Height the input face image is resized to.
            scale_factor (float): Factor to scale pixel values by.
            swap_rb (bool): Whether to swap Red and Blue channels (e.g., for RGB input).
        """
        self.model_path = model_path
        self.input_size = (input_width, input_height)
        self.scale_factor = scale_factor
        self.swap_rb = swap_rb
        self.net = self._load_model()

    def _load_model(self):
        """Loads the DNN model from the specified ONNX file."""
        # Construct absolute path based on script location for robustness, similar to detector fix
        abs_model_path = os.path.abspath(self.model_path)
        if not os.path.exists(abs_model_path):
             # Fallback check in case relative path was needed from specific working dir
             if not os.path.exists(self.model_path):
                 raise FileNotFoundError(f"Recognizer model file not found at: {self.model_path} or {abs_model_path}")
             else:
                 abs_model_path = self.model_path # Use the relative path if it exists

        print(f"Loading face recognizer model from: {abs_model_path}...")
        try:
            # Assuming ONNX format based on config example
            net = cv2.dnn.readNetFromONNX(abs_model_path)
            print("Face recognizer model loaded successfully.")
            return net
        except cv2.error as e:
            print(f"Error loading ONNX model '{abs_model_path}': {e}", file=sys.stderr)
            print("Ensure OpenCV was built with ONNX support and the model file is valid.", file=sys.stderr)
            raise

    def generate_embedding(self, face_image):
        """
        Generates a face embedding for the given face image.

        Args:
            face_image (numpy.ndarray): The cropped face image (BGR format).

        Returns:
            numpy.ndarray | None: The generated embedding vector (e.g., 128-d or 512-d),
                                  or None if an error occurs. The embedding is L2 normalized.
        """
        if self.net is None:
            print("Error: Face recognizer network not loaded.", file=sys.stderr)
            return None
        if face_image is None or face_image.size == 0:
             print("Error: Cannot generate embedding from empty face image.", file=sys.stderr)
             return None

        try:
            # 1. Create blob from the cropped face image
            #    - Resize to the model's expected input size (e.g., 112x112)
            #    - Apply scaling factor (e.g., 1.0/255.0 to scale pixels to 0-1)
            #    - Mean subtraction is often (0,0,0) for these models, but check model specifics
            #    - swapRB depends on whether the model expects BGR (False) or RGB (True)
            blob = cv2.dnn.blobFromImage(
                image=face_image,
                scalefactor=float(self.scale_factor),
                # --------------------------------
                size=self.input_size,
                mean=(0, 0, 0),
                swapRB=self.swap_rb,
                crop=False
            )

            # 2. Set the blob as input and perform forward pass
            self.net.setInput(blob)
            embedding = self.net.forward() # Output is the feature vector

            # 3. L2 Normalize the embedding (critical for distance/similarity comparisons)
            #    The output shape might be (1, N), flatten it first.
            embedding = self.net.forward()
            embedding = embedding.flatten()
            norm = np.linalg.norm(embedding)
            if norm == 0:
                print("Warning: Embedding norm is zero.", file=sys.stderr)
                return None
            normalized_embedding = embedding / norm

            return normalized_embedding

        except Exception as e:
            print(f"Error during embedding generation: {e}", file=sys.stderr)
            # Optional: print traceback
            # import traceback
            # traceback.print_exc()
            return None