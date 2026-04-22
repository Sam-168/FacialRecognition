
import requests
import base64
import json

def test_register_face():
    
    with open("Images/samukelo.jpg", "rb") as image_file:  
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Send to API
    response = requests.post(
        "http://localhost:8000/register-face",
        json={
            "studentId": 1,
            "imageBase64": image_base64
        }
    )
    
    print("Status Code:", response.status_code)
    print("Response:", json.dumps(response.json(), indent=2))

def test_recognize_face():
    """
    Test recognizing a face
    """
    
    
    with open("Images/myPic.png", "rb") as image_file: 
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    
    
    response = requests.post(
        "http://localhost:8000/recognize-face",
        json={
            "imageBase64": image_base64,
            "knownEncodings": [
                {
                    "studentId": 1,
                    "encodingPath": "data/encodings/student_1.pkl"
                }
            ]
        }
    )
    
    print("Status Code:", response.status_code)
    print("Response:", json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    print("=== Testing Face Registration ===")
    test_register_face()
    
    print("\n=== Testing Face Recognition ===")
    test_recognize_face()