# OpenCVFaceID/run_live_recognition.py
import cv2
import sys
import os
import time

# Ensure the script can find modules in the 'src' directory
# This adjusts the Python path dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from utils.config_loader import load_config
from camera.capture import Camera

def main():
    """Main function to capture and display camera feed."""
    try:
        # Assumes config is in the default location relative to project root
        config = load_config("../config/config.yaml")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Ensure 'config/config.yaml' exists in the project root.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # Safely get config values with defaults
    camera_settings = config.get('camera_settings', {})
    camera_index = camera_settings.get('camera_index', 0)

    ui_settings = config.get('ui_settings', {})
    window_name = ui_settings.get('display_window_name', 'Live Feed')

    try:
        # Use the Camera class with a 'with' statement for automatic release
        with Camera(camera_index=camera_index) as camera:
            print("\n--> Starting live feed. Press 'q' in the window to quit. <--")
            while True:
                success, frame = camera.read_frame()

                # If reading the frame failed, break the loop
                if not success or frame is None:
                    print("Info: End of video stream or cannot read frame.", file=sys.stderr)
                    time.sleep(0.5) # Prevent high CPU usage if stream ends suddenly
                    break

                # --- Placeholder for future processing ---
                # 1. Detect faces
                # 2. Encode detected faces
                # 3. Match encodings against database
                # 4. Draw results
                # --- End Placeholder ---

                # Display the current frame
                cv2.imshow(window_name, frame)

                # Wait for 1ms for a key press and check if it's 'q'
                key = cv2.waitKey(1) & 0xFF # Use mask for cross-platform compatibility
                if key == ord('q'):
                    print("'q' pressed, exiting gracefully.")
                    break

    except IOError as e:
        # Camera initialization errors are caught here
        print(f"\nCamera Error: {e}", file=sys.stderr)
        print("Please check camera connection and configuration.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nCtrl+C detected. Exiting...")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        # Optional: print traceback for debugging
        # import traceback
        # traceback.print_exc()
        sys.exit(1)
    finally:
        # Ensure all OpenCV windows are closed cleanly
        cv2.destroyAllWindows()
        print("Cleaned up resources. Exiting application.")


if __name__ == "__main__":
    # This ensures the main function runs only when the script is executed directly
    main()