# OpenCVFaceID/build_database.py
import cv2
import sys
import os
import pickle # To save the database
import numpy as np

# --- Determine Project Root and Add src to Python Path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = script_dir
src_path = os.path.join(PROJECT_ROOT, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Now import local modules
from utils.config_loader import load_config
from detection.detector import FaceDetector
from recognition.encoder import FaceEncoder

def find_largest_face(boxes):
    """Finds the bounding box with the largest area."""
    if not boxes:
        return None
    largest_box = None
    max_area = 0
    for (x1, y1, x2, y2) in boxes:
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            largest_box = (x1, y1, x2, y2)
    return largest_box

def build_known_faces_database(config):
    """
    Detects faces, generates embeddings for images in the known_faces directory,
    and saves them to a database file.
    """
    # --- Load Paths and Settings from Config ---
    known_faces_rel_path = config.get('known_faces_dir', 'known_faces')
    db_dir_rel_path = config.get('database_dir', 'database')
    db_filename = config.get('db_file', 'known_face_encodings.pkl')

    known_faces_abs_path = os.path.join(PROJECT_ROOT, known_faces_rel_path)
    db_dir_abs_path = os.path.join(PROJECT_ROOT, db_dir_rel_path)
    db_file_abs_path = os.path.join(db_dir_abs_path, db_filename)

    if not os.path.isdir(known_faces_abs_path):
        print(f"Error: Known faces directory not found at: {known_faces_abs_path}", file=sys.stderr)
        print("Please create it and add subdirectories with images for each known person.", file=sys.stderr)
        return

    # --- Initialize Detector ---
    detector_config = config.get('detector', {})
    prototxt_rel = detector_config.get('prototxt_path')
    model_weights_rel = detector_config.get('model_path')
    det_confidence = detector_config.get('confidence_threshold', 0.5)
    # (Add other detector params if needed, ensure paths are handled)
    if not prototxt_rel or not model_weights_rel:
        print("Error: Detector model paths not found in config.", file=sys.stderr)
        return
    # Construct absolute paths robustly (assuming paths in config are relative to PROJECT_ROOT)
    prototxt_abs = os.path.join(PROJECT_ROOT, prototxt_rel)
    model_weights_abs = os.path.join(PROJECT_ROOT, model_weights_rel)
    if not os.path.exists(prototxt_abs) or not os.path.exists(model_weights_abs):
        print(f"Error: Cannot find detector model files needed for database building.", file=sys.stderr)
        return
    try:
        face_detector = FaceDetector(
            prototxt_path=prototxt_abs,
            model_path=model_weights_abs,
            confidence_threshold=det_confidence
            # Add other params like input size, mean if needed by your FaceDetector init
        )
    except Exception as e:
        print(f"Error initializing face detector: {e}", file=sys.stderr)
        return


    # --- Initialize Recognizer ---
    recognizer_config = config.get('recognizer', {})
    rec_model_rel = recognizer_config.get('model_path')
    # (Add other recognizer params)
    if not rec_model_rel:
        print("Error: Recognizer model path not found in config.", file=sys.stderr)
        return
    rec_model_abs = os.path.join(PROJECT_ROOT, rec_model_rel)
    if not os.path.exists(rec_model_abs):
         print(f"Error: Cannot find recognizer model file needed for database building: {rec_model_abs}", file=sys.stderr)
         return
    try:
        face_encoder = FaceEncoder(
             model_path=rec_model_abs,
             # Pass other necessary params from config
             input_width=recognizer_config.get('input_width', 112),
             input_height=recognizer_config.get('input_height', 112),
             scale_factor=float(recognizer_config.get('scale_factor', 1.0/255.0)), # Ensure float
             swap_rb=recognizer_config.get('swap_rb', False)
        )
    except Exception as e:
        print(f"Error initializing face encoder: {e}", file=sys.stderr)
        return

    # --- Process Known Faces ---
    print(f"Processing images in '{known_faces_abs_path}'...")
    known_encodings = []
    known_names = []
    processed_image_count = 0
    encoded_face_count = 0

    # Iterate through each person's directory
    for person_name in os.listdir(known_faces_abs_path):
        person_dir = os.path.join(known_faces_abs_path, person_name)
        if not os.path.isdir(person_dir):
            continue # Skip files directly inside known_faces_dir

        print(f"Processing person: {person_name}")
        image_count_for_person = 0

        # Iterate through images for the current person
        for filename in os.listdir(person_dir):
            file_path = os.path.join(person_dir, filename)
            # Basic check for image file extensions
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                print(f"  Skipping non-image file: {filename}")
                continue

            print(f"  Processing image: {filename}...")
            processed_image_count += 1
            image = cv2.imread(file_path)

            if image is None:
                print(f"  Warning: Failed to read image {filename}. Skipping.")
                continue

            # Detect faces in the image
            boxes = face_detector.detect_faces(image)

            if not boxes:
                print(f"  Warning: No faces detected in {filename}. Skipping.")
                continue

            # Assume the largest face is the person of interest
            largest_box = find_largest_face(boxes)
            if largest_box is None: # Should not happen if boxes is not empty, but safety check
                 continue

            (x1, y1, x2, y2) = largest_box

            # Crop the face ROI
            if x1 < x2 and y1 < y2:
                face_roi = image[y1:y2, x1:x2]

                # Generate embedding
                embedding = face_encoder.generate_embedding(face_roi)

                if embedding is not None:
                    known_encodings.append(embedding)
                    known_names.append(person_name)
                    encoded_face_count += 1
                    image_count_for_person += 1
                else:
                    print(f"  Warning: Failed to generate embedding for face in {filename}. Skipping.")
            else:
                 print(f"  Warning: Invalid face box dimensions in {filename}. Skipping.")


        if image_count_for_person == 0:
            print(f"Warning: No usable faces found for person '{person_name}' in directory {person_dir}")

    # --- Save the Database ---
    if not known_encodings:
        print("\nError: No face encodings were generated. Database not saved.", file=sys.stderr)
        print("Please check images in known_faces directory and model configurations.", file=sys.stderr)
        return

    print(f"\nProcessed {processed_image_count} images.")
    print(f"Generated {encoded_face_count} face encodings for {len(set(known_names))} unique people.")

    # Ensure the database directory exists
    os.makedirs(db_dir_abs_path, exist_ok=True)

    # Package data (using lists is simple and often sufficient)
    data = {"encodings": known_encodings, "names": known_names}

    print(f"Saving database to: {db_file_abs_path}")
    try:
        with open(db_file_abs_path, "wb") as f: # Use 'wb' for binary writing
            pickle.dump(data, f)
        print("Database built and saved successfully.")
    except Exception as e:
        print(f"Error saving database file: {e}", file=sys.stderr)


if __name__ == "__main__":
    print("--- Building Face Recognition Database ---")
    try:
        app_config = load_config()
        build_known_faces_database(app_config)
    except FileNotFoundError as e:
         print(f"Configuration Error: {e}", file=sys.stderr)
    except Exception as e:
         print(f"An unexpected error occurred during database building: {e}", file=sys.stderr)
         # import traceback
         # traceback.print_exc()

    print("--- Database Building Script Finished ---")