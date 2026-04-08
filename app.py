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