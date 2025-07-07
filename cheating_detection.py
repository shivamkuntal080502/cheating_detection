import sys
import os
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import csv

# ── Initial Setup ──
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Enable iris landmarks and multiple faces
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=5)
hands_detector = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=1)

engine = pyttsx3.init()
engine.setProperty('rate', 150)

# ── Logging Setup ──
OUTPUT_DIR = r"C:\Users\ASUS\Desktop\opencv\logs"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
log_path = os.path.join(OUTPUT_DIR, "events_log.csv")
if not os.path.exists(log_path):
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(["timestamp","event_type","description","image_path"])
    try: os.chmod(log_path, 0o666)
    except: pass

# Thresholds & indices
LEFT_IRIS = list(range(474, 478))
LEFT_INNER = 133
LEFT_OUTER = 33
MOUTH_UP = 13
MOUTH_DOWN = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

eyelids = {
    'left': [33, 160, 158, 133, 153, 144],
    'right': [263, 385, 387, 362, 380, 373]
}
EAR_T = 0.25
EAR_FRAMES = 3
MAR_T = 0.5
GAZE_T = 2

blink_count = 0
mouth_count = 0
suspicious_score = 0
start_time = time.time()
last_alert = dict.fromkeys(['faces','gaze','blink','mouth','hands'], 0)
ALERT_CD = 3

# Utility functions

def save_event(frame, box, etype, desc):
    ts = time.strftime("%Y%m%d_%H%M%S")
    fn = f"{etype}_{ts}.jpg"
    p = os.path.join(OUTPUT_DIR, fn)
    x, y, w, h = box
    crop = frame[y:y+h, x:x+w]
    cv2.imwrite(p, crop)
    row = [ts, etype, desc, p]
    for _ in range(2):
        try:
            with open(log_path, 'a', newline='') as f:
                csv.writer(f).writerow(row)
            break
        except PermissionError:
            try: os.chmod(log_path, 0o666)
            except: pass
            time.sleep(0.1)


def compute_EAR(lm, idxs, dims):
    h, w = dims
    pts = [(lm[i].x*w, lm[i].y*h) for i in idxs]
    A = np.linalg.norm(np.array(pts[1]) - pts[5])
    B = np.linalg.norm(np.array(pts[2]) - pts[4])
    C = np.linalg.norm(np.array(pts[0]) - pts[3])
    return (A+B)/(2*C) if C else 0


def compute_MAR(lm, dims):
    h, w = dims
    up = np.array([lm[MOUTH_UP].x*w, lm[MOUTH_UP].y*h])
    down = np.array([lm[MOUTH_DOWN].x*w, lm[MOUTH_DOWN].y*h])
    left = np.array([lm[MOUTH_LEFT].x*w, lm[MOUTH_LEFT].y*h])
    right = np.array([lm[MOUTH_RIGHT].x*w, lm[MOUTH_RIGHT].y*h])
    vertical = np.linalg.norm(up - down)
    horizontal = np.linalg.norm(left - right)
    return vertical/horizontal if horizontal else 0

# ── Main Loop ──
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces_res = face_mesh.process(rgb)
    hands_res = hands_detector.process(rgb)
    now = time.time()

    # Multiple faces detection
    if faces_res.multi_face_landmarks and len(faces_res.multi_face_landmarks) > 1:
        desc = f"Multiple faces detected: {len(faces_res.multi_face_landmarks)}"
        # full-frame crop
        save_event(frame, (0, 0, w, h), 'faces', desc)
        cv2.putText(frame, desc, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if now - last_alert['faces'] > ALERT_CD:
            engine.say("Multiple faces detected. Only one candidate allowed.")
            engine.runAndWait()
            last_alert['faces'] = now
        suspicious_score += 1

    # Process first face for other metrics
    if faces_res.multi_face_landmarks:
        face = faces_res.multi_face_landmarks[0]
        # draw meshes
        mp_drawing.draw_landmarks(frame, face, mp_face_mesh.FACEMESH_TESSELATION,
                                  landmark_drawing_spec=None,
                                  connection_drawing_spec=mp_drawing.DrawingSpec((0,255,0),1,1))
        mp_drawing.draw_landmarks(frame, face, mp_face_mesh.FACEMESH_IRISES,
                                  landmark_drawing_spec=None,
                                  connection_drawing_spec=mp_drawing.DrawingSpec((0,255,255),1,1))
        lm = face.landmark
        # Gaze
        iris = np.mean([(lm[i].x*w, lm[i].y*h) for i in LEFT_IRIS], axis=0)
        inner = np.array([lm[LEFT_INNER].x*w, lm[LEFT_INNER].y*h])
        eye_w = np.linalg.norm(np.array([lm[LEFT_OUTER].x*w, lm[LEFT_OUTER].y*h]) - inner)
        gaze = (iris[0] - inner[0]) / (eye_w if eye_w else 1)
        cv2.putText(frame, f"Gaze: {gaze:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        if abs(gaze) > GAZE_T:
            desc = f"Gaze off-screen ({gaze:.2f})"
            box = (int(inner[0]-eye_w/2), int(inner[1]-eye_w/2), int(eye_w), int(eye_w))
            save_event(frame, box, 'gaze', desc)
            cv2.putText(frame, desc, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            if now - last_alert['gaze'] > ALERT_CD:
                engine.say("Please look at the screen")
                engine.runAndWait()
                last_alert['gaze'] = now
            suspicious_score += 1
        # Blink
        ear = (compute_EAR(lm, eyelids['left'], (h,w)) + compute_EAR(lm, eyelids['right'], (h,w))) / 2
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        if ear < EAR_T:
            blink_count += 1
        else:
            if blink_count >= EAR_FRAMES:
                desc = "Excessive blinking"
                box = (int(inner[0]-eye_w), int(inner[1]-eye_w/2), int(2*eye_w), int(eye_w))
                save_event(frame, box, 'blink', desc)
                cv2.putText(frame, desc, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                if now - last_alert['blink'] > ALERT_CD:
                    engine.say("Excessive blinking detected")
                    engine.runAndWait()
                    last_alert['blink'] = now
                suspicious_score += 1
            blink_count = 0
        # Mouth movement
        mar = compute_MAR(lm, (h,w))
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        if mar > MAR_T:
            mouth_count += 1
        else:
            if mouth_count > 0:
                desc = f"Mouth movement (MAR {mar:.2f})"
                up_pt = (int(lm[MOUTH_UP].x*w), int(lm[MOUTH_UP].y*h))
                box = (up_pt[0]-40, up_pt[1]-20, 80, 40)
                save_event(frame, box, 'mouth', desc)
                cv2.putText(frame, desc, (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                if now - last_alert['mouth'] > ALERT_CD:
                    engine.say("Please refrain from talking")
                    engine.runAndWait()
                    last_alert['mouth'] = now
                suspicious_score += 1
            mouth_count = 0

    # Hand detection
    if hands_res.multi_hand_landmarks:
        for hand in hands_res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            pts = [(int(l.x*w), int(l.y*h)) for l in hand.landmark]
            xs, ys = zip(*pts)
            box = (min(xs)-5, min(ys)-5, max(xs)-min(xs)+10, max(ys)-min(ys)+10)
            save_event(frame, box, 'hands', 'Hands detected')
            cv2.putText(frame, "Hands detected", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            if now - last_alert['hands'] > ALERT_CD:
                engine.say("Hands detected, please keep your hands off screen")
                engine.runAndWait()
                last_alert['hands'] = now
            suspicious_score += 1

    # Overlay status & display
    elapsed = int(now - start_time)
    cv2.putText(frame, f"Time: {elapsed}s  Suspicious: {suspicious_score}", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.imshow("Cheating Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
