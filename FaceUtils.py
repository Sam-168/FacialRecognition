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
        
        # Create directories if they don't exist
        os.makedirs(encodings_dir, exist_ok=True)
        os.makedirs(photos_dir, exist_ok=True)

    def generate_encoding_from_image(self, image_array: np.ndarray) -> Tuple[bool, Optional[np.ndarray], str]:
        
        try:
            # Convert to BGR for OpenCV operations
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            # Resize to improve performance
            img_small = cv2.resize(image_bgr, (0, 0), fx=0.50, fy=0.50)
            
            # Find face locations
            face_locations = face_recognition.face_locations(img_small)
            
            if len(face_locations) == 0:
                return False, None, "No face detected in the image"
            
            if len(face_locations) > 1:
                return False, None, f"Multiple faces detected ({len(face_locations)}). Please provide image with single face"
            
            # Generate encoding
            encodings = face_recognition.face_encodings(img_small, face_locations)
            
            if len(encodings) == 0:
                return False, None, "Could not generate face encoding"
            
            encoding = encodings[0]
            
            return True, encoding, "Face encoding generated successfully"
            
        except Exception as e:
            return False, None, f"Error processing image: {str(e)}"

    def save_student_encoding(self, student_id: int, encoding: np.ndarray, photo_array: np.ndarray) -> Tuple[str, str]:
        
        # Save encoding as pickle file
        encoding_path = os.path.join(self.encodings_dir, f"student_{student_id}.pkl")
        with open(encoding_path, 'wb') as f:
            pickle.dump(encoding, f)
        
        # Save photo as JPG
        photo_path = os.path.join(self.photos_dir, f"student_{student_id}.jpg")
        # Convert RGB to BGR for OpenCV
        photo_bgr = cv2.cvtColor(photo_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(photo_path, photo_bgr)
        
        return encoding_path, photo_path