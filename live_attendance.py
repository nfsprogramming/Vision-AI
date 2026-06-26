import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
import pyttsx3
import threading

# --- CONFIGURATION ---
# --- CONFIGURATION ---
IMAGE_PATH = 'images'
ATTENDANCE_FILE = 'attendance.csv'
FRAME_RES = (640, 480) 

if not os.path.exists(IMAGE_PATH):
    print(f"❌ Error: '{IMAGE_PATH}' directory not found.")
    exit()

def get_features(img_gray):
    # USE SIFT (More robust than ORB)
    # nfeatures=0 means 'detect all the best ones'
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img_gray, None)
    return kp, des

def load_known_faces(path):
    known_data = [] # List of (name, descriptor) tuples
    
    if not os.path.exists(path):
         return []

    print(f"🔄 Processing known faces (SIFT Mode)...")
    
    files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for file in files:
        try:
            image_path = os.path.join(path, file)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            
            # Detect face
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(img, 1.1, 5)
            
            if len(faces) == 0:
                face_img = img
            else:
                (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3]) # Largest face
                face_img = img[y:y+h, x:x+w]

            kp, des = get_features(face_img)
            
            if des is not None and len(des) > 0:
                filename = os.path.splitext(file)[0]
                import re
                name = re.sub(r'[\._]?[0-9]*$', '', filename)
                
                # Store (Name, Descriptors, Total_Features_Count, Keypoints)
                known_data.append((name, des, len(des), kp))
                print(f"   Loaded: {name} ({len(des)} features)")
            
        except Exception as e:
            print(f"⚠️ process error {file}: {e}")
            
    return known_data

def speak_message(message):
    def run():
        try:
            engine = pyttsx3.init()
            engine.say(message)
            engine.runAndWait()
        except Exception: pass
    threading.Thread(target=run, daemon=True).start()

# --- ATTENDANCE UTILS ---
def mark_attendance(name):
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        df.to_csv(ATTENDANCE_FILE, index=False)
        
    try:
        df = pd.read_csv(ATTENDANCE_FILE)
        name = name.strip().title()
        today = datetime.now().strftime('%Y-%m-%d')
        
        if not df.empty and 'Name' in df.columns and 'Date' in df.columns:
            existing = df[(df['Name'].str.title() == name) & (df['Date'] == today)]
            if not existing.empty:
                return False 

        now = datetime.now()
        new_record = pd.DataFrame({
            'Name': [name], 'Date': [today], 'Time': [now.strftime('%H:%M:%S')]
        })
        
        if df.empty:
            df = new_record
        else:
            df = pd.concat([df, new_record], ignore_index=True)
            
        df.to_csv(ATTENDANCE_FILE, index=False, mode='w')
        print(f"✅ Attendance Marked: {name}")
        return True
    except Exception as e:
        print(f"Marking Error: {e}")
        return False

# --- MAIN ---
known_data = load_known_faces(IMAGE_PATH)

if len(known_data) == 0:
    print("❌ No images found or features extraction failed!")
    exit()

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Feature Matcher (KNN) uses NORM_L2 for SIFT
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

print("🎥 LIVE SYSTEM STARTED (High Accuracy SIFT Mode). Press 'q' to quit.")

last_marked_name = "None"
last_marked_time = ""

while True:
    ret, frame = cap.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
    
    # Header
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (200, 200, 200), -1)
    cv2.putText(frame, f"Last Marked: {last_marked_name} {last_marked_time}", (10, 28), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    for (x, y, w, h) in faces:
        live_face = gray[y:y+h, x:x+w]
        
        # Live Features
        live_kp, live_des = get_features(live_face)
        
        best_name = "Unknown"
        best_score = 0
        
        if live_des is not None and len(known_data) > 0 and len(live_des) > 0:
            for name, known_des, total_feats, known_kp in known_data:
                
                # KNN Match with k=2
                try:
                    matches = bf.knnMatch(live_des, known_des, k=2)
                    
                    # Apply Lowe's Ratio Test
                    good_matches = []
                    for m, n in matches:
                        if m.distance < 0.70 * n.distance:
                            good_matches.append(m)
                    
                    # --- GEOMETRIC VERIFICATION (RANSAC) ---
                    # This ensures the points form a valid geometric transformation (a real face match).
                    score = 0
                    if len(good_matches) >= 6:
                        src_pts = np.float32([live_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([known_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        
                        try:
                            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                            if mask is not None:
                                score = np.sum(mask) # Count of inliers
                        except Exception:
                            pass
                    
                    if score > best_score:
                        best_score = score
                        best_name = name
                        
                except Exception:
                    continue
        
        # --- THRESHOLDS ---
        # With RANSAC, even 6-8 inliers is a very strong mathematical proof of a match.
        SCORE_THRESHOLD = 8
        
        if best_score < SCORE_THRESHOLD:
            best_name = "Unknown"
        
        match_found = best_name != "Unknown"
        
        box_color = (0, 0, 255) # Red for Unknown
        message = f"Score: {int(best_score)}"
        
        if match_found:
             if mark_attendance(best_name):
                last_marked_name = best_name
                last_marked_time = datetime.now().strftime('%H:%M:%S')
                speak_message(f"Welcome {best_name}")
                box_color = (0, 255, 0) # Green for Marked
                message = "MARKED"
             else:
                box_color = (0, 255, 255) # Yellow for Already Present
                message = "PRESENT"
        else:
             box_color = (0, 0, 255)

        # Draw
        cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
        
        label = best_name if best_name != "Unknown" else "Unknown"
        
        # Name Label
        cv2.rectangle(frame, (x, y-40), (x+w, y), box_color, -1)
        cv2.putText(frame, label, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
        # Score Label
        cv2.putText(frame, f"Verified: {int(best_score)}", (x, y+h+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    cv2.imshow("Smart Attendance (SIFT + RANSAC)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
