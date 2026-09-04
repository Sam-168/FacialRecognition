from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import logging

from FaceUtils import FaceRecognitionService

# ----------------------------
# Logging setup
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face-service")

app = FastAPI(
    title="Face Recognition Service",
    description="Face recognition API for student attendance system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

face_service = FaceRecognitionService()


# ----------------------------
# Models
# ----------------------------
class RegisterFaceRequest(BaseModel):
    studentId: int
    imageBase64: str


class RegisterFaceResponse(BaseModel):
    success: bool
    studentId: int
    encodingPath: Optional[str] = None
    photoPath: Optional[str] = None
    message: str
    error: Optional[str] = None


class KnownStudent(BaseModel):
    studentId: int
    encodingPath: str


class RecognizeFaceRequest(BaseModel):
    imageBase64: str
    knownEncodings: List[KnownStudent]


class RecognizeFaceResponse(BaseModel):
    success: bool
    faceDetected: bool
    matched: bool
    studentId: Optional[int] = None
    message: str


# ----------------------------
# Helpers
# ----------------------------
def base64_to_image(base64_string: str) -> np.ndarray:
    try:
        logger.info(f"Decoding base64 image. Length: {len(base64_string)}")

        image_bytes = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_bytes))

        logger.info(f"Decoded image mode: {image.mode}")

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image_array = np.array(image)

        logger.info(f"Image converted to array. Shape: {image_array.shape}")

        return image_array

    except Exception as e:
        logger.error(f"Base64 decode failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return {
        "service": "Face Recognition API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
def health_check():
    logger.info("Health check called")
    try:
        face_service.check_storage()
        return {
            "status": "UP",
            "message": "Face recognition service and biometric storage are available"
        }
    except Exception as error:
        logger.error("Biometric storage health check failed: %s", error)
        return JSONResponse(
            status_code=503,
            content={
                "status": "DOWN",
                "message": "Biometric storage is unavailable"
            },
        )


# ----------------------------
# REGISTER FACE
# ----------------------------
@app.post("/register-face", response_model=RegisterFaceResponse)
async def register_face(request: RegisterFaceRequest):
    logger.info(f"REGISTER FACE REQUEST - Student ID: {request.studentId}")

    try:
        image_array = base64_to_image(request.imageBase64)

        logger.info("Generating face encoding...")

        success, encoding, message = face_service.generate_encoding_from_image(image_array)

        logger.info(f"Encoding result: success={success}, message={message}")

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

        logger.info(f"Saved encoding path: {encoding_path}")
        logger.info(f"Saved photo path: {photo_path}")

        return RegisterFaceResponse(
            success=True,
            studentId=request.studentId,
            encodingPath=encoding_path,
            photoPath=photo_path,
            message="Face registered successfully"
        )

    except HTTPException as e:
        logger.error(f"HTTPException: {e.detail}")
        raise e

    except Exception as e:
        logger.error(f"Unexpected error in register-face: {str(e)}")
        return RegisterFaceResponse(
            success=False,
            studentId=request.studentId,
            message=f"Error processing request: {str(e)}",
            error=str(e)
        )


# ----------------------------
# RECOGNIZE FACE
# ----------------------------
@app.post("/recognize-face", response_model=RecognizeFaceResponse)
async def recognize_face(request: RecognizeFaceRequest):
    logger.info("RECOGNIZE FACE REQUEST RECEIVED")

    try:
        image_array = base64_to_image(request.imageBase64)

        logger.info(f"Known encodings received: {len(request.knownEncodings)}")

        known_encodings = []

        for ks in request.knownEncodings:
            encoding = face_service.load_student_encoding(ks.studentId)

            if encoding is not None:
                known_encodings.append((ks.studentId, encoding))

        logger.info(f"Valid encodings loaded: {len(known_encodings)}")

        if len(known_encodings) == 0:
            logger.warning("No valid encodings found")
            return RecognizeFaceResponse(
                success=False,
                faceDetected=False,
                matched=False,
                message="No valid encodings loaded"
            )

        face_detected, matched_student_id, confidence, message = face_service.recognize_face(
            image_array,
            known_encodings
        )

        logger.info(f"Face detected: {face_detected}")
        logger.info(f"Match result: {matched_student_id}")

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
                message=message
            )

        return RecognizeFaceResponse(
            success=True,
            faceDetected=True,
            matched=False,
            message=message
        )

    except HTTPException as e:
        logger.error(f"HTTPException: {e.detail}")
        raise e

    except Exception as e:
        logger.error(f"Unexpected error in recognize-face: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
