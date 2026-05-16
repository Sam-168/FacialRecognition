# RaySense Face Recognition Service

A FastAPI-based face recognition microservice for the RaySense smart attendance platform.

## Overview

This microservice handles:

- Face detection
- Face encoding generation
- Face verification
- Image preprocessing
- Attendance verification

The service receives images as Base64 strings from the Spring Boot backend and processes them using computer vision libraries.

---

## Architecture

Vue.js Frontend
↓
Spring Boot Backend
↓
FastAPI Face Recognition Service
↓
Face Encoding Generation

---

## Features

- Base64 image processing
- Face detection
- Face encoding generation
- Face matching
- OpenCV image processing
- FastAPI REST APIs
- Railway deployment

---

## Tech Stack

- Python
- FastAPI
- OpenCV
- face_recognition
- NumPy
- Pillow (PIL)
- Uvicorn
- Railway

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/raysense-face-service.git
```

Navigate into the project:

```bash
cd raysense-face-service
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\\Scripts\\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
uvicorn main:app --host 0.0.0.0 --port 5000
```

---

## API Endpoints

### Health Check

```text
GET /health
```

### Register Face

```text
POST /register-face
```

### Verify Face

```text
POST /verify-face
```

---

## Base64 Image Processing

Example:

```python
image_bytes = base64.b64decode(base64_string)
image = Image.open(BytesIO(image_bytes))
image_array = np.array(image)
```

---

## Face Encoding Pipeline

1. Receive Base64 image
2. Decode image
3. Detect face location
4. Generate face encoding
5. Save encoding
6. Return encoding metadata

---

## Deployment

The service is deployed using Railway.

### Deployment Steps

1. Push repository to GitHub
2. Connect repository to Railway
3. Configure environment variables
4. Deploy

---

## Challenges Solved

- Base64 image decoding
- OpenCV integration
- Face encoding generation
- Multi-service communication
- Railway deployment debugging
- Image preprocessing
- Production troubleshooting

---

## Future Improvements

- GPU acceleration
- Liveness detection
- Anti-spoofing
- Higher accuracy models
- Cloud storage integration
- Batch processing

---

## Important Note

Railway file systems are ephemeral.

For production-grade persistence consider:

- AWS S3
- Cloudinary
- Supabase Storage
- Railway Volumes

---

## License

This project is for educational and portfolio purposes.
