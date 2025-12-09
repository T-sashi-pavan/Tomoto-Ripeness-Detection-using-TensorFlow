from flask import Flask, Response, jsonify, request
import cv2
import numpy as np
import threading
import time

app = Flask(__name__)

# === CHANGE THIS TO YOUR ESP32 IP ===
ESP32_URL = "http://192.168.29.126:81/stream"
latest_result = {"color": "unknown", "result": "No Tomato"}
latest_frame = None
debug_frame = None
frame_lock = threading.Lock()

# Create default black frames as fallback
default_frame = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(default_frame, "Waiting for camera...", (50, 240), 
           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def enhance_image_for_dim_light(frame):
    """Enhance image for better detection in dim light conditions"""
    # CLAHE for better contrast
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Mild sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    return enhanced

def is_tomato_shaped(contour):
    """EXTREMELY LENIENT tomato shape verification - focuses on tomato structures only"""
    area = cv2.contourArea(contour)
    
    # Lower minimum area threshold
    if area < 50:  # Reduced from 100
        return False, "area too small"
    
    # Get bounding rectangle for aspect ratio
    x, y, w, h = cv2.boundingRect(contour)
    
    # Smaller minimum bounding box
    if w < 15 or h < 15:  # Reduced from 25
        return False, "bounding box too small"
    
    aspect_ratio = w / h
    
    # Much more lenient aspect ratio for various tomato shapes
    if aspect_ratio < 0.3 or aspect_ratio > 3.0:  # Widened from 0.4-2.5
        return False, f"aspect ratio {aspect_ratio:.2f}"
    
    return True, "tomato shaped"

def debug_tomato_detection(frame):
    """ULTRA SIMPLE tomato detection that WILL show boxes"""
    
    # Resize frame first to consistent size
    frame_resized = cv2.resize(frame, (640, 480))
    
    # Apply dim light enhancement
    enhanced_frame = enhance_image_for_dim_light(frame_resized)
    
    # Make copies for visualization - IMPORTANT: Create fresh copies
    debug_vis = enhanced_frame.copy()
    result_vis = enhanced_frame.copy()  # This is the frame that will show boxes
    
    blur = cv2.GaussianBlur(enhanced_frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # RELAXED HSV ranges for better tomato detection in various lighting
    lower_red1, upper_red1 = np.array([0, 20, 20]), np.array([10, 255, 255])  # More lenient saturation/value
    lower_red2, upper_red2 = np.array([170, 20, 20]), np.array([180, 255, 255])
    lower_orange, upper_orange = np.array([10, 20, 20]), np.array([22, 255, 255])  # Wider range
    lower_yellow, upper_yellow = np.array([20, 20, 30]), np.array([35, 255, 255])  # Lower thresholds
    lower_green, upper_green = np.array([35, 15, 15]), np.array([85, 255, 220])  # Wider green range

    # Create individual masks
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Show individual color masks on debug frame
    debug_vis[mask_red > 0] = [0, 0, 255]      # Red areas
    debug_vis[mask_orange > 0] = [0, 140, 255] # Orange areas
    debug_vis[mask_yellow > 0] = [0, 255, 255] # Yellow areas
    debug_vis[mask_green > 0] = [0, 255, 0]    # Green areas
    
    # Combine all tomato color masks
    combined_mask = cv2.bitwise_or(cv2.bitwise_or(cv2.bitwise_or(mask_red, mask_orange), mask_yellow), mask_green)

    # Simple morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result = "No Tomato Detected"
    color = "none"
    
    # Show ALL contours for debugging
    cv2.drawContours(debug_vis, contours, -1, (255, 255, 255), 2)
    
    print(f"Found {len(contours)} contours")
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    print(f"Checking {min(5, len(contours))} largest contours...")
    
    for i, cnt in enumerate(contours[:5]):
        area = cv2.contourArea(cnt)
        print(f"Contour {i}: area = {area}")
        
        # EXTREMELY LENIENT area filtering - allow smaller tomatoes
        if area < 50 or area > 300000:  # Reduced minimum from 100 to 50
            print(f"  Skipped - size out of range")
            continue
        
        # VERY LENIENT shape verification
        is_tomato, shape_reason = is_tomato_shaped(cnt)
        if not is_tomato:
            print(f"  Skipped - shape: {shape_reason}")
            continue
            
        print(f"  ✓ Reasonable shape")
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(cnt)
        print(f"  Bounding box: {w}x{h} at ({x},{y})")
        
        # Color analysis - ULTRA SIMPLE
        roi_red = mask_red[y:y+h, x:x+w]
        roi_orange = mask_orange[y:y+h, x:x+w]
        roi_yellow = mask_yellow[y:y+h, x:x+w]
        roi_green = mask_green[y:y+h, x:x+w]
        
        red_pixels = np.sum(roi_red > 0)
        orange_pixels = np.sum(roi_orange > 0)
        yellow_pixels = np.sum(roi_yellow > 0)
        green_pixels = np.sum(roi_green > 0)
        
        total_color_pixels = red_pixels + orange_pixels + yellow_pixels + green_pixels
        roi_total_pixels = w * h
        
        # MINIMAL color coverage requirement - very permissive
        color_coverage = total_color_pixels / roi_total_pixels if roi_total_pixels > 0 else 0
        if color_coverage < 0.01:  # Only 1% coverage needed! (reduced from 3%)
            print(f"  Skipped - low color coverage {color_coverage:.2%}")
            continue
        
        print(f"  Color coverage: {color_coverage:.2%}")
        print(f"  Color pixels - R:{red_pixels}, O:{orange_pixels}, Y:{yellow_pixels}, G:{green_pixels}")
        
        # Find dominant color
        color_counts = {
            "red": red_pixels,
            "orange": orange_pixels, 
            "yellow": yellow_pixels,
            "green": green_pixels
        }
        
        dominant_color = max(color_counts, key=color_counts.get)
        dominant_pixels = color_counts[dominant_color]
        dominant_ratio = dominant_pixels / total_color_pixels if total_color_pixels > 0 else 0
        
        print(f"  Dominant color: {dominant_color} ({dominant_ratio:.2%})")
        
        # MINIMAL classification threshold - detect with very few pixels
        if dominant_pixels > 20:  # At least 20 pixels of dominant color (reduced from 50)
            if dominant_color == "red":
                result = "🔴 READY TO USE"
                color = "red"
                box_color = (0, 0, 255)
            elif dominant_color == "orange":
                result = "🟠 NEED FEW DAYS TO USE"
                color = "orange"
                box_color = (0, 140, 255)
            elif dominant_color == "yellow":
                result = "🟡 NEED MORE DAYS TO USE"
                color = "yellow"
                box_color = (0, 255, 255)
            elif dominant_color == "green":
                result = "🟢 UNREADY TO USE - UNRIPE"
                color = "green"
                box_color = (0, 255, 0)
            
            print(f"  ✓ {dominant_color.upper()} TOMATO DETECTED!")
            
            # ALWAYS DRAW THE BOX - NO MATTER WHAT
            box_thickness = 4  # Fixed thickness for visibility
            cv2.rectangle(result_vis, (x, y), (x+w, y+h), box_color, box_thickness)
            
            # Enhanced information display
            main_text = result
            confidence = min(1.0, (dominant_ratio + color_coverage) / 2)
            detail_text = f"Confidence: {confidence:.0%}"
            
            # Main result text with background
            text_size = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(result_vis, (x, y-35), (x + text_size[0] + 10, y), box_color, -1)
            cv2.putText(result_vis, main_text, (x + 5, y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Confidence text
            cv2.rectangle(result_vis, (x, y-10), (x + text_size[0] + 10, y), (40, 40, 40), -1)
            cv2.putText(result_vis, detail_text, (x + 5, y-2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            print(f"🎉 FINAL DETECTION: {result} | Confidence: {confidence:.0%}")
            print(f"🎯 BOX DRAWN at ({x},{y}) to ({x+w},{y+h})")
            break
    
    # If no tomato detected, show message on result_vis
    if result == "No Tomato Detected":
        cv2.putText(result_vis, "No Tomato Detected", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return result_vis, debug_vis, result, color

def process_stream():
    global latest_frame, latest_result, debug_frame
    print("Connecting to ESP32-CAM stream:", ESP32_URL)

    cap = cv2.VideoCapture(ESP32_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ Error: Cannot open ESP32-CAM stream.")
        # Set default frames
        with frame_lock:
            latest_frame = default_frame.copy()
            debug_frame = default_frame.copy()
        return

    print("✅ Stream connected successfully!")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame not received, retrying...")
            time.sleep(0.5)
            continue

        try:
            # Process frame with debug info
            processed_frame, debug_vis, result, color = debug_tomato_detection(frame)
            
            # Update results
            latest_result = {"color": color, "result": result}
            
            # CRITICAL: Update frames with proper locking
            with frame_lock:
                latest_frame = processed_frame.copy()  # This should show boxes
                debug_frame = debug_vis.copy()
            
            print(f"✅ Frames updated - Detection: {result}")
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            # Set default frames on error
            with frame_lock:
                latest_frame = default_frame.copy()
                debug_frame = default_frame.copy()
        
        time.sleep(0.1)  # Faster processing

@app.route('/')
def index():
    return '''
    <html>
      <head><title>🍅 Tomato Detection</title></head>
      <body style="background:#000; color:#fff; text-align:center; font-family:Arial;">
        <h1>🍅 Tomato Detection System</h1>
        <p style="color:#0f0;">✓ Real-time detection ✓ Color classification ✓ Confidence scoring</p>
        
        <div style="display: flex; justify-content: center; gap: 20px; margin: 20px;">
            <div>
                <h3>📹 Detection View (SHOULD SHOW BOXES)</h3>
                <img src="/video_feed" width="640" height="480" style="border:2px solid #333; border-radius:10px;"/>
            </div>
            <div>
                <h3>🔍 Debug View</h3>
                <img src="/debug_feed" width="640" height="480" style="border:2px solid #333; border-radius:10px;"/>
                <p style="font-size:12px; color:#888;">
                    Colored areas = detected tomato colors<br/>
                    White contours = potential objects<br/>
                    Boxes = detected tomatoes
                </p>
            </div>
        </div>
        
        <h2>Status: <span id="status">Loading...</span></h2>
        
        <script>
          setInterval(function() {
            fetch('/predict')
              .then(response => response.json())
              .then(data => {
                document.getElementById('status').textContent = data.result;
                document.getElementById('status').style.color = 
                  data.color === 'red' ? '#ff4444' :
                  data.color === 'orange' ? '#ff8844' :
                  data.color === 'yellow' ? '#ffff44' :
                  data.color === 'green' ? '#44ff44' : '#ffffff';
              });
          }, 1000);
        </script>
        
        <div style="margin-top:20px; text-align:left; max-width:800px; margin-left:auto; margin-right:auto;">
            <h3>🎯 DEBUGGING MODE - EXTREMELY LENIENT</h3>
            <ul>
                <li>Only 1% color coverage required (vs 3% before)</li>
                <li>Only 20 pixels of dominant color needed (vs 50 before)</li>
                <li>Minimal area: 50 pixels (vs 100 before)</li>
                <li>Relaxed HSV ranges for better lighting tolerance</li>
                <li>Extremely lenient shape requirements</li>
                <li>Boxes ALWAYS drawn when tomato structure detected</li>
            </ul>
            
            <h3>📊 Expected Results:</h3>
            <p style="color:#ff4444;">🔴 Red tomatoes → "READY TO USE"</p>
            <p style="color:#ff8844;">🟠 Orange tomatoes → "NEED FEW DAYS TO USE"</p>
            <p style="color:#ffff44;">🟡 Yellow tomatoes → "NEED MORE DAYS TO USE"</p>
            <p style="color:#44ff44;">🟢 Green tomatoes → "UNREADY TO USE - UNRIPE"</p>
        </div>
      </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if latest_frame is not None:
                    # Ensure the frame is proper size and type
                    frame_to_send = latest_frame
                    if frame_to_send is None:
                        frame_to_send = default_frame
                    
                    # Encode as JPEG
                    success, jpeg = cv2.imencode('.jpg', frame_to_send)
                    if success:
                        frame_bytes = jpeg.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
            time.sleep(0.05)  # Faster frame rate
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/debug_feed')
def debug_feed():
    def generate():
        while True:
            with frame_lock:
                if debug_frame is not None:
                    # Ensure the frame is proper size and type
                    frame_to_send = debug_frame
                    if frame_to_send is None:
                        frame_to_send = default_frame
                    
                    # Encode as JPEG
                    success, jpeg = cv2.imencode('.jpg', frame_to_send)
                    if success:
                        frame_bytes = jpeg.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
            time.sleep(0.05)  # Faster frame rate
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    return jsonify(latest_result)

if __name__ == '__main__':
    # Start the stream processing in a separate thread
    stream_thread = threading.Thread(target=process_stream, daemon=True)
    stream_thread.start()
    
    print("🚀 ULTRA-LENIENT Tomato Detection Server starting...")
    print("📡 Connecting to ESP32-CAM stream...")
    time.sleep(2)  # Give some time for stream to initialize
    
    print("🌐 Flask server running at: http://localhost:5000")
    print("📹 Open your browser and go to: http://localhost:5000")
    print("🔍 Check terminal for detection logs")
    print("🎯 THIS VERSION WILL SHOW BOXES!")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


    
