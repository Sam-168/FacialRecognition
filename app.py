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

@app.post("/register-face", response_model=RegisterFaceResponse)
async def register_face(request: RegisterFaceRequest):
    """
    Register a student's face
    
    This endpoint is supposed to:
    1. Receives student photo as base64
    2. Generates face encoding
    3. Saves encoding and photo to disk
    4. Returns file paths
    """
    try:
        
        image_array = base64_to_image(request.imageBase64)
        
       
        success, encoding, message = face_service.generate_encoding_from_image(image_array)
        
        if not success:
            return RegisterFaceResponse(
                success=False,
                studentId=request.studentId,
                message=message,
                error=message
            )
        
        
        encoding_path, photo_path = face_service.save_student_encoding(
            request.studentId,
            encoding,
            image_array
        )
        
        return RegisterFaceResponse(
            success=True,
            studentId=request.studentId,
            encodingPath=encoding_path,
            photoPath=photo_path,
            message="Face registered successfully"
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        return RegisterFaceResponse(
            success=False,
            studentId=request.studentId,
            message=f"Error processing request: {str(e)}",
            error=str(e)
        )

@app.post("/recognize-face", response_model=RecognizeFaceResponse)
async def recognize_face(request: RecognizeFaceRequest):
    """
    Recognize a face against known students
    
    This endpoint is supposed to:
    1. Receives photo as base64
    2. Receives list of known student encodings
    3. Compares photo against known encodings
    4. Returns matched student ID if found
    """
    try:
        # Convert base64 to image array
        image_array = base64_to_image(request.imageBase64)
        
        # Load all known encodings
        known_encodings = []
        
        for known_student in request.knownEncodings:
            encoding = face_service.load_encoding(known_student.encodingPath)
            
            if encoding is not None:
                known_encodings.append((known_student.studentId, encoding))
        
        if len(known_encodings) == 0:
            return RecognizeFaceResponse(
                success=False,
                faceDetected=False,
                matched=False,
                message="No valid encodings loaded"
            )
        
        # Recognize face
        face_detected, matched_student_id, confidence, message = face_service.recognize_face(
            image_array,
            known_encodings
        )
        
        if not face_detected:
            return RecognizeFaceResponse(
                success=True,
                faceDetected=False,
                matched=False,
                message=message
            )
        
        if matched_student_id is not None:
            return RecognizeFaceResponse(
                success=True,
                faceDetected=True,
                matched=True,
                studentId=matched_student_id,
                confidence=confidence,
                message=message
            )
        else:
            return RecognizeFaceResponse(
                success=True,
                faceDetected=True,
                matched=False,
                message=message
            )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during recognition: {str(e)}")