# OpenCVFaceID/run_live_recognition.py
import cv2
import sys
import os
import time
import numpy as np
import pickle

# --- Determine Project Root and Add src to Python Path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = script_dir
src_path = os.path.join(PROJECT_ROOT, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- Imports ---
from utils.config_loader import load_config
from camera.capture import Camera
from detection.detector import FaceDetector
from recognition.encoder import FaceEncoder

def load_known_faces(config):
    """Loads the known face encodings database."""
    db_dir_rel = config.get('database_dir', 'database')
    db_filename = config.get('db_file', 'known_face_encodings.pkl')
    db_file_abs = os.path.join(PROJECT_ROOT, db_dir_rel, db_filename)

    if not os.path.exists(db_file_abs):
        print(f"Error: Database file not found at: {db_file_abs}", file=sys.stderr)
        print("Please run 'build_database.py' first.", file=sys.stderr)
        return None, None # Indicate failure

    print(f"Loading known faces database from: {db_file_abs}...")
    try:
        with open(db_file_abs, "rb") as f:
            data = pickle.load(f)
        # It's good practice to convert encodings to a NumPy array here if they aren't already
        known_encodings = np.array(data["encodings"], dtype=np.float32)
        known_names = data["names"]
        print(f"Database loaded successfully: {len(known_names)} encodings.")
        if len(known_names) == 0:
             print("Warning: Loaded database contains no known faces.", file=sys.stderr)
        return known_encodings, known_names
    except Exception as e:
        print(f"Error loading database file '{db_file_abs}': {e}", file=sys.stderr)
        return None, None # Indicate failure


def main():
    """Main function to capture, detect, recognize faces, and display."""
    try:
        config = load_config()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Load Known Faces Database ---
    known_encodings, known_names = load_known_faces(config)
    if known_encodings is None: # Check if loading failed
        sys.exit(1)
    has_known_faces = len(known_names) > 0
    # -----------------------------------

    # --- Camera Settings ---
    camera_settings = config.get('camera_settings', {})
    camera_index = camera_settings.get('camera_index', 0)

    # --- Detector Settings ---
    detector_config = config.get('detector', {})
    prototxt_rel_path = detector_config.get('prototxt_path')
    model_weights_rel_path = detector_config.get('model_path')
    det_confidence = detector_config.get('confidence_threshold', 0.5)
    if not prototxt_rel_path or not model_weights_rel_path: sys.exit("Detector paths missing in config") # Simplified error exit
    prototxt_abs_path = os.path.join(PROJECT_ROOT, prototxt_rel_path)
    model_weights_abs_path = os.path.join(PROJECT_ROOT, model_weights_rel_path)
    if not os.path.exists(prototxt_abs_path) or not os.path.exists(model_weights_abs_path): sys.exit("Detector model files not found") # Simplified error exit

    # --- Recognizer Settings ---
    recognizer_config = config.get('recognizer', {})
    rec_model_rel_path = recognizer_config.get('model_path')
    rec_input_w = recognizer_config.get('input_width', 112)
    rec_input_h = recognizer_config.get('input_height', 112)
    rec_scale = float(recognizer_config.get('scale_factor', 1.0/255.0)) # Ensure float
    rec_swap_rb = recognizer_config.get('swap_rb', False)
    if not rec_model_rel_path: sys.exit("Recognizer path missing in config") # Simplified error exit
    rec_model_abs_path = os.path.join(PROJECT_ROOT, rec_model_rel_path)
    if not os.path.exists(rec_model_abs_path): sys.exit(f"Recognizer model file not found: {rec_model_abs_path}") # Simplified error exit
    recognition_threshold = config.get('recognition_threshold_distance', 0.6) # Get threshold

    # --- UI Settings --- (Keep as before)
    ui_settings = config.get('ui_settings', {})
    window_name = ui_settings.get('display_window_name', 'Live Feed')
    box_color = tuple(ui_settings.get('box_color', [0, 255, 0]))
    unknown_label = ui_settings.get('unknown_person_label', 'Unknown')
    text_color = (255, 255, 255) # White text

    try:
        # Initialize Detector
        face_detector = FaceDetector(
            prototxt_path=prototxt_abs_path, model_path=model_weights_abs_path,
            confidence_threshold=det_confidence
            # Pass other detector params if needed
        )

        # Initialize Recognizer
        face_encoder = FaceEncoder(
            model_path=rec_model_abs_path, input_width=rec_input_w, input_height=rec_input_h,
            scale_factor=rec_scale, swap_rb=rec_swap_rb
        )

        # Initialize Camera
        with Camera(camera_index=camera_index) as camera:
            print(f"\n--> Starting live recognition. Threshold={recognition_threshold}. Press 'q' to quit. <--")
            while True:
                start_time = time.time()

                success, frame = camera.read_frame()
                if not success or frame is None:
                    time.sleep(0.5)
                    break

                # --- Face Detection ---
                detected_boxes = face_detector.detect_faces(frame)

                # --- Process Detected Faces (Encoding & Matching) ---
                current_names = [] # Store names found in this frame
                current_boxes = [] # Store corresponding boxes

                for (x1, y1, x2, y2) in detected_boxes:
                    # Crop face ROI
                    if x1 >= x2 or y1 >= y2: continue # Skip invalid boxes
                    face_roi = frame[y1:y2, x1:x2]

                    # Generate embedding
                    live_embedding = face_encoder.generate_embedding(face_roi)

                    name = unknown_label # Default to unknown
                    if live_embedding is not None and has_known_faces:
                        # Calculate Euclidean distances between live embedding and all known embeddings
                        distances = np.linalg.norm(known_encodings - live_embedding, axis=1)

                        # Find the index of the best match (minimum distance)
                        best_match_index = np.argmin(distances)
                        min_distance = distances[best_match_index]

                        # Compare distance with threshold
                        if min_distance < recognition_threshold:
                            name = known_names[best_match_index] # Found a match!

                    # Store the determined name and box for drawing later
                    current_names.append(name)
                    current_boxes.append((x1, y1, x2, y2))
                # --- End Processing Detected Faces ---


                # --- Draw Results ---
                for i, (x1, y1, x2, y2) in enumerate(current_boxes):
                    name = current_names[i] # Get the name for this box

                    # Draw the bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    # Prepare text properties
                    text = name
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 1

                    # Calculate text size to draw a background rectangle
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                    # Put text above the box, handle boundary conditions
                    text_y = y1 - 10 if y1 - 10 > 10 else y1 + text_height + 10
                    # Draw a filled rectangle (background for text)
                    cv2.rectangle(frame, (x1, text_y - text_height - baseline), (x1 + text_width, text_y + baseline), box_color, -1) # cv2.FILLED is -1
                    # Draw the text
                    cv2.putText(frame, text, (x1, text_y), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

                # --- End Draw Results ---

                # Calculate and display FPS
                end_time = time.time()
                fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("'q' pressed, exiting gracefully.")
                    break

    # --- Error Handling & Cleanup ---
    except (IOError, FileNotFoundError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting...")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()
        print("Cleaned up resources. Exiting application.")


if __name__ == "__main__":
    main()