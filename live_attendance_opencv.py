import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime

# --- CONFIGURATION ---
IMAGE_PATH = 'images'
ATTENDANCE_FILE = 'attendance.csv'
CONFIDENCE_THRESHOLD = 50  # Lower is stricter for LBPH (0 is perfect match)

# Ensure images directory exists
if not os.path.exists(IMAGE_PATH):
    print(f"❌ Error: '{IMAGE_PATH}' directory not found.")
    exit()

def get_images_and_labels(path):
    """
    Loads images from the directory, detects faces, and prepares training data.
    """
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    face_samples = []
    ids = []
    names = {} # Map ID to Name
    
    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("🔄 Training Face Recognition Model (this might take a few seconds)...")
    
    for image_path in image_paths:
        try:
            # Read image in grayscale
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Get name from filename
            filename = os.path.split(image_path)[-1]
            # Remove extension
            name_base = os.path.splitext(filename)[0]
            # Remove trailing numbers/dots/underscores (e.g. Prem.1 -> Prem, Prem_02 -> Prem)
            import re
            name = re.sub(r'[\._]?[0-9]+$', '', name_base)
            
            # Assign an ID to this name if new
            if name not in names.values():
                new_id = len(names)
                names[new_id] = name
                this_id = new_id
            else:
                # Find the existing ID for this name
                for k, v in names.items():
                    if v == name:
                        this_id = k
                        break
            
            # Detect face in the training image
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
            
            for (x, y, w, h) in faces:
                face_samples.append(img[y:y+h, x:x+w])
                ids.append(this_id)
                
        except Exception as e:
            print(f"⚠️ Could not process {image_path}: {e}")

    return face_samples, ids, names

# --- MAIN ---

# 1. Train the Recognizer
faces, ids, names = get_images_and_labels(IMAGE_PATH)

if len(faces) == 0:
    print("❌ No faces found in local images! Add clear face images to 'images/' folder.")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(ids))
print("✅ Model Trained Successfully!")

# 2. Setup Live Video
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def mark_attendance(name):
    """Marks attendance in the CSV file."""
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        df.to_csv(ATTENDANCE_FILE, index=False)
        
    df = pd.read_csv(ATTENDANCE_FILE)
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Check duplicate for today
    if df.empty or len(df[(df['Name'] == name) & (df['Date'] == today)]) == 0:
        now = datetime.now()
        new_record = pd.DataFrame({
            'Name': [name], 
            'Date': [today], 
            'Time': [now.strftime('%H:%M:%S')]
        })
        df = pd.concat([df, new_record], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        print(f"📌 Attendance Marked: {name}")

print("🎥 Starting Live Recognition... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    for (x, y, w, h) in faces:
        # Recognize
        id_pred, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        
        # Check confidence (Lower is better for LBPH)
        # 0 is perfect match.
        # < 50 is a very strong match.
        # < 70 is decent.
        # > 80 is usually noise/unknown.
        
        STRICT_THRESHOLD = 55
        
        if confidence < STRICT_THRESHOLD:
            name = names.get(id_pred, "Unknown")
            confidence_text = f"  {round(100 - confidence)}%"
        else:
            name = "Unknown"
            confidence_text = f"  {round(100 - confidence)}%"
        
        if name != "Unknown":
             mark_attendance(name)
        
        # Draw Output
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, str(name), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, str(confidence_text), (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.imshow('Face Attendance (OpenCV LBPH)', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
