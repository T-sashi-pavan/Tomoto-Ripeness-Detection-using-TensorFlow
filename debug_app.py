from flask import Flask, Response, jsonify
import cv2
import numpy as np
import threading
import time

app = Flask(__name__)

# ===================== CONFIG =====================
ESP32_URL = "http://192.168.29.126:81/stream"

latest_result = {"color": "none", "result": "No Tomato"}
latest_frame = None
debug_frame = None
frame_lock = threading.Lock()

# fallback frame
default_frame = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(default_frame, "Waiting for ESP32 Camera...", (40, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# ===================== IMAGE ENHANCEMENT =====================
def enhance_image(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    return enhanced

# ===================== TOMATO SHAPE CHECK =====================
def is_real_tomato(contour):
    area = cv2.contourArea(contour)
    if area < 1200 or area > 200000:
        return False

    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return False

    circularity = (4 * np.pi * area) / (peri * peri)
    if circularity < 0.60:
        return False

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return False

    solidity = area / hull_area
    if solidity < 0.82:
        return False

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h
    if aspect_ratio < 0.75 or aspect_ratio > 1.3:
        return False

    return True

# ===================== DETECTION FUNCTION =====================
def detect_tomato(frame):
    frame = cv2.resize(frame, (640, 480))
    frame = enhance_image(frame)

    debug_vis = frame.copy()
    result_vis = frame.copy()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ✅ STRICT tomato HSV ranges
    ranges = {
        "red1": (np.array([0, 70, 70]), np.array([10, 255, 255])),
        "red2": (np.array([170, 70, 70]), np.array([180, 255, 255])),
        "orange": (np.array([10, 80, 80]), np.array([20, 255, 255])),
        "yellow": (np.array([20, 90, 90]), np.array([32, 255, 255])),
        "green": (np.array([36, 60, 60]), np.array([80, 255, 200]))
    }

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for r in ranges.values():
        mask |= cv2.inRange(hsv, r[0], r[1])

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(debug_vis, contours, -1, (255, 255, 255), 2)

    result = "No Tomato Detected"
    color = "none"

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:5]:
        if not is_real_tomato(cnt):
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        roi = mask[y:y+h, x:x+w]

        color_pixels = np.sum(roi > 0)
        coverage = color_pixels / (w * h)

        if coverage < 0.20:
            continue

        hsv_roi = hsv[y:y+h, x:x+w]

        counts = {}
        for k, r in ranges.items():
            m = cv2.inRange(hsv_roi, r[0], r[1])
            counts[k] = np.sum(m > 0)

        dominant = max(counts, key=counts.get)

        if "red" in dominant:
            result = "🔴 READY TO USE"
            color = "red"
            box_color = (0, 0, 255)
        elif dominant == "orange":
            result = "🟠 NEED FEW DAYS"
            color = "orange"
            box_color = (0, 140, 255)
        elif dominant == "yellow":
            result = "🟡 NEED MORE DAYS"
            color = "yellow"
            box_color = (0, 255, 255)
        else:
            result = "🟢 UNRIPE"
            color = "green"
            box_color = (0, 255, 0)

        cv2.rectangle(result_vis, (x, y), (x+w, y+h), box_color, 3)
        cv2.putText(result_vis, result, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
        break

    return result_vis, debug_vis, result, color

# ===================== STREAM THREAD =====================
def process_stream():
    global latest_frame, debug_frame, latest_result

    while True:
        cap = cv2.VideoCapture(ESP32_URL)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            time.sleep(2)
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                break

            try:
                out, dbg, res, col = detect_tomato(frame)

                with frame_lock:
                    latest_frame = out.copy()
                    debug_frame = dbg.copy()
                    latest_result = {"result": res, "color": col}

                time.sleep(0.04)

            except Exception as e:
                print("Processing error:", e)
                time.sleep(0.1)

        time.sleep(1)

# ===================== FLASK ROUTES =====================
@app.route('/')
def index():
    return '''
    <html>
    <body style="background:black;color:white;text-align:center">
    <h1>🍅 Tomato Detection</h1>
    <img src="/video_feed" width="640"><br><br>
    <img src="/debug_feed" width="640"><br><br>
    <h2 id="status">Loading...</h2>
    <script>
    setInterval(()=>{
      fetch('/predict').then(r=>r.json()).then(d=>{
        document.getElementById("status").innerHTML=d.result;
      })
    },1000);
    </script>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            with frame_lock:
                frame = latest_frame if latest_frame is not None else default_frame
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   jpeg.tobytes() + b'\r\n')
            time.sleep(0.03)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/debug_feed')
def debug_feed():
    def gen():
        while True:
            with frame_lock:
                frame = debug_frame if debug_frame is not None else default_frame
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   jpeg.tobytes() + b'\r\n')
            time.sleep(0.05)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict')
def predict():
    return jsonify(latest_result)

# ===================== MAIN =====================
if __name__ == '__main__':
    threading.Thread(target=process_stream, daemon=True).start()
    print("✅ Tomato detection server running")
    app.run(host='0.0.0.0', port=5000, threaded=True)
