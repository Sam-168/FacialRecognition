import cv2
import numpy as np
import face_recognition
import pickle
import os
from typing import List, Tuple, Optional

class FaceRecognitionService:
    
    
    def __init__(self, encodings_dir: str = "data/encodings", photos_dir: str = "data/photos"):
        self.encodings_dir = encodings_dir
        self.photos_dir = photos_dir
        
    
        os.makedirs(encodings_dir, exist_ok=True)
        os.makedirs(photos_dir, exist_ok=True)

    def generate_encoding_from_image(self, image_array: np.ndarray) -> Tuple[bool, Optional[np.ndarray], str]:
        
        try:
            
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            
            img_small = cv2.resize(image_bgr, (0, 0), fx=0.50, fy=0.50)
            
            
            face_locations = face_recognition.face_locations(img_small)
            
            if len(face_locations) == 0:
                return False, None, "No face detected in the image"
            
            if len(face_locations) > 1:
                return False, None, f"Multiple faces detected ({len(face_locations)}). Please provide image with single face"
            
            
            encodings = face_recognition.face_encodings(img_small, face_locations)
            
            if len(encodings) == 0:
                return False, None, "Could not generate face encoding"
            
            encoding = encodings[0]
            
            return True, encoding, "Face encoding generated successfully"
            
        except Exception as e:
            return False, None, f"Error processing image: {str(e)}"

    def save_student_encoding(self, student_id: int, encoding: np.ndarray, photo_array: np.ndarray) -> Tuple[str, str]:
        
        
        encoding_path = os.path.join(self.encodings_dir, f"student_{student_id}.pkl")
        with open(encoding_path, 'wb') as f:
            pickle.dump(encoding, f)
        
        
        photo_path = os.path.join(self.photos_dir, f"student_{student_id}.jpg")
        
        photo_bgr = cv2.cvtColor(photo_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(photo_path, photo_bgr)
        
        return encoding_path, photo_path

    def load_encoding(self, encoding_path: str) -> Optional[np.ndarray]:
        
        try:
            if not os.path.exists(encoding_path):
                return None
            
            with open(encoding_path, 'rb') as f:
                encoding = pickle.load(f)
            return encoding
            
        except Exception as e:
            print(f"Error loading encoding from {encoding_path}: {e}")
            return None

    def recognize_face(self, image_array: np.ndarray, known_encodings: List[Tuple[int, np.ndarray]]) -> Tuple[bool, Optional[int], float, str]:
        
        try:
           
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            
            face_locations = face_recognition.face_locations(image_bgr)
            
            if len(face_locations) == 0:
                return False, None, 0.0, "No face detected in image"
            
            
            encode_current_frame = face_recognition.face_encodings(image_bgr, face_locations)
            
            if len(encode_current_frame) == 0:
                return False, None, 0.0, "Could not encode detected face"
            
            encode_face = encode_current_frame[0]
            
            student_ids = [item[0] for item in known_encodings]
            encodings_only = [item[1] for item in known_encodings]
            
            matches = face_recognition.compare_faces(encodings_only, encode_face, tolerance=0.6)
            face_distances = face_recognition.face_distance(encodings_only, encode_face)
            
            print(f"Face distances: {face_distances}")  # debugging line
            
            match_index = np.argmin(face_distances)
            
            if matches[match_index]:
                matched_student_id = student_ids[match_index]
                
                confidence = 1 - face_distances[match_index]
                
                return True, matched_student_id, float(confidence), "Face recognized"
            else:
                return True, None, 0.0, "Face detected but not recognized"
                
        except Exception as e:
            return False, None, 0.0, f"Error during recognition: {str(e)}"

    