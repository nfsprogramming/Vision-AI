import os
import json
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import base64
import re

app = Flask(__name__)

CONFIG_FILE = 'config.json'
def get_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {"MATCH_THRESHOLD": 8}

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)


# --- CONFIGURATION ---
IMAGE_PATH = 'images'
ATTENDANCE_FILE = 'attendance.csv'

# Make sure images and CSV exists
if not os.path.exists(IMAGE_PATH):
    os.makedirs(IMAGE_PATH)

def get_orb_features(img_gray):
    orb = cv2.ORB_create(nfeatures=2500)
    kp, des = orb.detectAndCompute(img_gray, None)
    return kp, des

def load_known_faces(path):
    known_data = []
    
    if not os.path.exists(path):
         return []

    print("🔄 Processing known faces for Web App...")
    
    files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    for file in files:
        try:
            image_path = os.path.join(path, file)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            
            faces = face_cascade.detectMultiScale(img, 1.1, 5)
            
            if len(faces) == 0:
                face_img = img
            else:
                (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])
                face_img = img[y:y+h, x:x+w]

            kp, des = get_orb_features(face_img)
            
            if des is not None:
                filename = os.path.splitext(file)[0]
                name = re.sub(r'[\._]?[0-9]*$', '', filename)
                known_data.append((name, des))
                print(f"   Loaded: {name} ({len(des)} features)")
            
        except Exception as e:
            print(f"⚠️ process error {file}: {e}")
            
    return known_data

# Load once on startup
KNOWN_DATA = load_known_faces(IMAGE_PATH)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Track the last marked person globally (or via sessions, but global is fine for a kiosk)
last_marked_state = {}

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    global last_marked_state
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
            
        # Decode base64 image from the frontend
        image_data = data['image'].split(',')[1]
        decoded_data = base64.b64decode(image_data)
        np_arr = np.frombuffer(decoded_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(30, 30))
        
        results = []
        
        for (x, y, w, h) in faces:
            live_face = gray[y:y+h, x:x+w]
            _, live_des = get_orb_features(live_face)
            
            best_name = "Unknown"
            max_good_matches = 0
            
            if live_des is not None and len(KNOWN_DATA) > 0:
                for name, known_des in KNOWN_DATA:
                    if known_des is None: continue
                    try:
                        matches = bf.knnMatch(live_des, known_des, k=2)
                        good_matches = []
                        for m, n in matches:
                            if m.distance < 0.85 * n.distance:
                                good_matches.append(m)
                                
                        count = len(good_matches)
                        if count > max_good_matches:
                            max_good_matches = count
                            best_name = name
                    except Exception:
                        continue
            
            MATCH_THRESHOLD = get_config().get('MATCH_THRESHOLD', 8)
            
            if max_good_matches < MATCH_THRESHOLD:
                best_name = "Unknown"
            
            action = "None"
            message = ""
            
            if best_name != "Unknown":
                # Check if we should mark attendance
                 if mark_attendance(best_name):
                    action = "MARKED"
                    message = f"Welcome {best_name}"
                    last_marked_state[best_name] = datetime.now().strftime('%H:%M:%S')
                 else:
                    action = "PRESENT"
                    message = f"{best_name} is already present"
                    
            results.append({
                "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "name": best_name,
                "score": max_good_matches,
                "action": action,
                "message": message
            })

        return jsonify({"status": "success", "results": results, "faces_detected": len(faces)})

    except Exception as e:
        print(f"Error in recognition: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/attendance')
def get_attendance():
    if not os.path.exists(ATTENDANCE_FILE):
        return jsonify([])
    try:
        df = pd.read_csv(ATTENDANCE_FILE)
        if df.empty:
            return jsonify([])
        df = df.dropna(subset=['Name', 'Date'])
        df = df.fillna('')
        return jsonify(df.to_dict('records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/cameras')
def cameras():
    return render_template('cameras.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/api/analytics')
def api_analytics():
    if not os.path.exists(ATTENDANCE_FILE):
        return jsonify({"labels": [], "data": [], "total_today": 0, "total_all": 0})
    try:
        df = pd.read_csv(ATTENDANCE_FILE)
        if df.empty:
            return jsonify({"labels": [], "data": [], "total_today": 0, "total_all": 0})
            
        counts = df.groupby('Date').size().reset_index(name='Count')
        counts = counts.tail(7)
        
        today_str = datetime.now().strftime('%Y-%m-%d')
        today_count = int(counts[counts['Date'] == today_str]['Count'].sum())
        
        return jsonify({
            "labels": counts['Date'].tolist(),
            "data": counts['Count'].tolist(),
            "total_today": today_count,
            "total_all": int(len(df))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    if request.method == 'POST':
        data = request.json
        config = get_config()
        if 'MATCH_THRESHOLD' in data:
            config['MATCH_THRESHOLD'] = int(data['MATCH_THRESHOLD'])
        save_config(config)
        return jsonify({"status": "success"})
    return jsonify(get_config())

if __name__ == '__main__':
    # Production should use Gunicorn, this is for local testing
    app.run(host='0.0.0.0', port=5000, debug=True)
