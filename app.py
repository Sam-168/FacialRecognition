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