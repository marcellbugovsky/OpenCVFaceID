# src/camera/capture.py
import cv2
import time
import sys

class Camera:
    """Handles camera initialization and frame capturing using OpenCV."""

    def __init__(self, camera_index=0):
        """
        Initializes the camera.

        Args:
            camera_index (int): The index of the camera to use (e.g., 0, 1).
        """
        self.camera_index = camera_index
        self.cap = None
        self._initialize_camera()

    def _initialize_camera(self):
        """Initializes the VideoCapture object."""
        print(f"Initializing camera with index: {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)

        # Allow camera some time to initialize - essential for some webcams
        time.sleep(1.0)

        if not self.cap.isOpened():
            print(f"Error: Could not open camera stream with index {self.camera_index}.", file=sys.stderr)
            # Provide a more helpful message, possibly suggesting trying other indices
            print("Please check if the camera is connected and not in use by another application.", file=sys.stderr)
            print(f"You might need to change 'camera_index' in config/config.yaml to 1, 2, etc.", file=sys.stderr)
            raise IOError(f"Cannot open camera stream with index {self.camera_index}")
        else:
            # Optional: Read camera properties
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = self.cap.get(cv2.CAP_PROP_FPS) # Might not always be accurate
            print(f"Camera {self.camera_index} opened successfully.")
            print(f"Resolution: {int(width)}x{int(height)}, Target FPS: {fps:.2f}")

    def read_frame(self):
        """
        Reads a single frame from the camera.

        Returns:
            tuple: (bool, numpy.ndarray | None):
                     - success (bool): True if a frame was read successfully, False otherwise.
                     - frame (numpy.ndarray | None): The captured frame, or None if failed.
        """
        if self.cap is None or not self.cap.isOpened():
            print("Error: Camera not initialized or already released.", file=sys.stderr)
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            # Don't print warning continuously if stream just ends
            # print("Warning: Failed to grab frame.")
            return False, None
        return True, frame

    def release(self):
        """Releases the camera hardware resource."""
        if self.cap is not None and self.cap.isOpened():
            print(f"Releasing camera {self.camera_index}...")
            self.cap.release()
            self.cap = None
            print("Camera released.")

    # --- Context Manager Protocol (Recommended) ---
    def __enter__(self):
        """Allows using the Camera object with 'with' statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensures camera is released when exiting 'with' block."""
        self.release()