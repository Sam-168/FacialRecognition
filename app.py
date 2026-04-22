from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import cv2

from face_utils import FaceRecognitionService

app = FastAPI(
    title="Face Recognition Service",
    description="Face recognition API for student attendance system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #spring backend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

face_service = FaceRecognitionService()

class RegisterFaceRequest(BaseModel):
    studentId: int
    imageBase64: str

class RegisterFaceResponse(BaseModel):
    success:bool
    studentId: int
    encodingPath: Optional[str] = None
    photoPath: Optional[str] = None
    message: str
    error: Optional[str] = None

class KnownStudent(BaseModel):
    studentId: int
    encodingPath: str

class RecognitionFaceRequest(BaseModel):
    imageBase64: str
    knownEncodings: List[KnownStudent]

class RecognizeFaceResponse(BaseModel):
    success: bool
    faceDetected: bool
    matched: bool
    studentId: Optional[int] = None
    message: str

def base64_to_image(base64_string: str) -> np.ndarray:
    
    try:
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_string)
        
        # Convert bytes to PIL Image
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert PIL Image to numpy array
        image_array = np.array(image)
        
        return image_array
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

@app.get("/")
def root():
    """
    Root endpoint
    """
    return {
        "service": "Face Recognition API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "UP",
        "message": "Face recognition service is running",
        "services": {
            "opencv": "UP",
            "face_recognition": "UP"
        }
    }