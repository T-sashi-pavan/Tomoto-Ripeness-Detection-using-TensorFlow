from flask import Flask, Response, jsonify, request
import cv2
import numpy as np
import threading
import time

app = Flask(__name__)

# === CHANGE THIS TO YOUR ESP32 IP ===
ESP32_URL = "http://192.168.1.8:81/stream"

latest_result = {"color": "unknown", "result": "No Tomato"}
latest_frame = None
debug_frame = None
frame_lock = threading.Lock()

def debug_tomato_detection(frame):
    """Debug version - shows what the system sees and detects"""
    
    # Resize frame first to consistent size
    frame_resized = cv2.resize(frame, (640, 480))
    
    # Make copies for visualization
    debug_vis = frame_resized.copy()
    
    blur = cv2.GaussianBlur(frame_resized, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Enhanced HSV ranges for precise tomato color detection
    # Red tomatoes (ripe) - Two ranges for wraparound hue
    lower_red1, upper_red1 = np.array([0, 80, 50]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 80, 50]), np.array([180, 255, 255])
    
    # Orange tomatoes (nearly ripe) - Critical transition color
    lower_orange, upper_orange = np.array([10, 100, 80]), np.array([25, 255, 255])
    
    # Yellow tomatoes (turning) - Bright yellow range
    lower_yellow, upper_yellow = np.array([25, 80, 100]), np.array([35, 255, 255])
    
    # Green tomatoes (unripe) - Specific tomato green, avoid background
    lower_green, upper_green = np.array([35, 70, 50]), np.array([75, 255, 200])

    # Create individual masks with enhanced ranges
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Show individual color masks on debug frame with distinct colors
    debug_vis[mask_red > 0] = [0, 0, 255]      # Red areas
    debug_vis[mask_orange > 0] = [0, 165, 255] # Orange areas
    debug_vis[mask_yellow > 0] = [0, 255, 255] # Yellow areas
    debug_vis[mask_green > 0] = [0, 255, 0]    # Green areas
    
    # Combine all tomato color masks
    combined_mask = cv2.bitwise_or(cv2.bitwise_or(cv2.bitwise_or(mask_red, mask_orange), mask_yellow), mask_green)

    # Minimal morphology
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result = "No Tomato Detected"
    color = "none"
    
    # Show ALL contours for debugging
    cv2.drawContours(debug_vis, contours, -1, (255, 255, 255), 2)
    
    print(f"Found {len(contours)} contours")
    
    # Sort contours by area (largest first) - focus on bigger objects
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    print(f"Checking {min(10, len(contours))} largest contours...")
    
    for i, cnt in enumerate(contours[:10]):  # Only check top 10 largest contours
        area = cv2.contourArea(cnt)
        print(f"Contour {i}: area = {area}")
        
        # Dynamic area filtering for near and far tomato detection
        if area < 300 or area > 80000:  # Very wide range for far/close detection
            print(f"  Skipped - size out of range")
            continue
            
        # Classify distance based on size for adaptive thresholds
        if area > 10000:
            distance_class = "close"
            min_color_pixels = 100
        elif area > 2000:
            distance_class = "medium"
            min_color_pixels = 50
        else:
            distance_class = "far"
            min_color_pixels = 20
            
        # Basic shape filtering - tomatoes are roughly round (more lenient)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            print(f"  Skipped - zero perimeter")
            continue
            
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < 0.25:  # More lenient roundness check
            print(f"  Skipped - circularity {circularity:.3f}")
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Aspect ratio check - tomatoes are roughly circular
        aspect_ratio = w / h
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:  # More lenient aspect ratio
            print(f"  Skipped - aspect ratio {aspect_ratio:.2f}")
            continue
        
        print(f"Potential tomato {i}: area = {area}, circularity = {circularity:.2f}, aspect = {aspect_ratio:.2f}, distance = {distance_class}")
        
        # Draw potential tomato regions
        cv2.rectangle(debug_vis, (x, y), (x+w, y+h), (128, 128, 128), 2)
        cv2.putText(debug_vis, f"A:{int(area)}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Get the region colors
        roi_hsv = hsv[y:y+h, x:x+w]
        roi_mask = combined_mask[y:y+h, x:x+w]
        
        masked_hsv = roi_hsv[roi_mask > 0]
        if len(masked_hsv) < min_color_pixels:  # Adaptive threshold based on distance
            print(f"  Skipped - insufficient colored pixels ({len(masked_hsv)}, need {min_color_pixels})")
            continue
            
        mean_hue = np.mean(masked_hsv[:, 0])
        mean_sat = np.mean(masked_hsv[:, 1])
        mean_val = np.mean(masked_hsv[:, 2])
        
        print(f"  Hue: {mean_hue:.1f}, Sat: {mean_sat:.1f}, Val: {mean_val:.1f}")
        
        # Enhanced multi-stage color analysis for precise tomato classification
        detected = False
        
        # Analyze color distribution in the region for better accuracy
        roi_red = mask_red[y:y+h, x:x+w]
        roi_orange = mask_orange[y:y+h, x:x+w]
        roi_yellow = mask_yellow[y:y+h, x:x+w]
        roi_green = mask_green[y:y+h, x:x+w]
        
        # Count pixels for each color
        red_pixels = np.sum(roi_red > 0)
        orange_pixels = np.sum(roi_orange > 0)
        yellow_pixels = np.sum(roi_yellow > 0)
        green_pixels = np.sum(roi_green > 0)
        
        total_color_pixels = red_pixels + orange_pixels + yellow_pixels + green_pixels
        
        print(f"  Color pixel counts - R:{red_pixels}, O:{orange_pixels}, Y:{yellow_pixels}, G:{green_pixels}")
        
        # Determine dominant color with adaptive thresholds based on distance
        if total_color_pixels > min_color_pixels:  # Adaptive threshold
            # Calculate color percentages
            red_pct = red_pixels / total_color_pixels
            orange_pct = orange_pixels / total_color_pixels
            yellow_pct = yellow_pixels / total_color_pixels
            green_pct = green_pixels / total_color_pixels
            
            print(f"  Color percentages - R:{red_pct:.2f}, O:{orange_pct:.2f}, Y:{yellow_pct:.2f}, G:{green_pct:.2f}")
            
            # Red tomato - must be predominantly red with good saturation
            if (red_pct > 0.5 or (mean_hue < 10 or mean_hue > 170)) and mean_sat > 80 and mean_val > 60:
                result = "🔴 READY TO USE"
                color = "red"
                box_color = (0, 0, 255)
                detected = True
            # Orange tomato - specific orange hue range with high saturation
            elif (orange_pct > 0.4 or (10 <= mean_hue <= 25)) and mean_sat > 100 and mean_val > 70:
                result = "🟠 NEED FEW DAYS TO USE"
                color = "orange"
                box_color = (0, 140, 255)
                detected = True
            # Yellow tomato - bright yellow with good visibility
            elif (yellow_pct > 0.4 or (25 < mean_hue <= 35)) and mean_sat > 80 and mean_val > 100:
                result = "🟡 NEED MORE DAYS TO USE"
                color = "yellow"
                box_color = (0, 255, 255)
                detected = True
            # Green tomato - unripe, specific green range
            elif (green_pct > 0.4 or (35 < mean_hue <= 75)) and mean_sat > 60 and mean_val > 50:
                result = "🟢 UNREADY TO USE - UNRIPE"
                color = "green"
                box_color = (0, 255, 0)
                detected = True
        
        if detected:
            # Calculate confidence based on color purity and size
            dominant_pct = max(red_pct, orange_pct, yellow_pct, green_pct)
            size_confidence = min(1.0, area / 2000)  # Better confidence for larger tomatoes
            confidence = (dominant_pct + size_confidence) / 2
            
            # Draw the detection box with thickness based on confidence
            box_thickness = max(2, int(confidence * 5))
            cv2.rectangle(frame_resized, (x, y), (x+w, y+h), box_color, box_thickness)
            
            # Enhanced text display with confidence and distance
            main_text = result
            detail_text = f"{distance_class.upper()} | {confidence:.0%}"
            
            # Main text
            text_size = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame_resized, (x, y-45), (x + text_size[0] + 10, y), box_color, -1)
            cv2.putText(frame_resized, main_text, (x + 5, y-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Detail text
            detail_size = cv2.getTextSize(detail_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(frame_resized, (x, y-20), (x + detail_size[0] + 10, y), (0, 0, 0), -1)
            cv2.putText(frame_resized, detail_text, (x + 5, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            print(f"  DETECTED: {result} | Confidence: {confidence:.0%} | Distance: {distance_class}")
            break  # Take first detection
        else:
            print(f"  REJECTED: Not tomato-like colors (H:{mean_hue:.1f}, S:{mean_sat:.1f}, V:{mean_val:.1f})")
    
    return frame_resized, debug_vis, result, color

def process_stream():
    global latest_frame, latest_result, debug_frame
    print("Connecting to ESP32-CAM stream:", ESP32_URL)

    cap = cv2.VideoCapture(ESP32_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ Error: Cannot open ESP32-CAM stream.")
        return

    print("✅ Stream connected successfully!")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame not received, retrying...")
            time.sleep(0.5)
            continue

        # Process frame with debug info
        processed_frame, debug_vis, result, color = debug_tomato_detection(frame)
        
        # Update results
        latest_result = {"color": color, "result": result}
        
        with frame_lock:
            latest_frame = processed_frame.copy()
            debug_frame = debug_vis.copy()
        
        time.sleep(0.2)  # 5 FPS for better debugging

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if latest_frame is not None:
                    _, jpeg = cv2.imencode('.jpg', latest_frame)
                    frame_bytes = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/debug_feed')
def debug_feed():
    def generate():
        while True:
            with frame_lock:
                if debug_frame is not None:
                    _, jpeg = cv2.imencode('.jpg', debug_frame)
                    frame_bytes = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Handle ESP32 image uploads
        try:
            image_data = request.get_data()
            if len(image_data) > 0:
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    _, _, result, color = debug_tomato_detection(img)
                    return jsonify({"color": color, "result": result})
        except Exception as e:
            print(f"Error processing POST image: {e}")
    
    return jsonify(latest_result)

@app.route('/')
def index():
    return '''
    <html>
      <head><title>🍅 Debug Tomato Detection</title></head>
      <body style="background:#000; color:#fff; text-align:center; font-family:Arial;">
        <h1>🍅 DEBUG Tomato Detection System</h1>
        
        <div style="display: flex; justify-content: center; gap: 20px; margin: 20px;">
            <div>
                <h3>📹 Detection View</h3>
                <img src="/video_feed" width="640" height="480" style="border:2px solid #333; border-radius:10px;"/>
            </div>
            <div>
                <h3>🔍 Debug View</h3>
                <img src="/debug_feed" width="640" height="480" style="border:2px solid #333; border-radius:10px;"/>
                <p style="font-size:12px; color:#888;">
                    White boxes = detected contours<br/>
                    Colored areas = detected tomato colors<br/>
                    Gray boxes = potential regions
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
            <h3>🔧 Debugging Instructions:</h3>
            <ol>
                <li><strong>Check Debug View:</strong> You should see colored areas where tomato colors are detected</li>
                <li><strong>Hold a tomato close:</strong> Start with the tomato 1-2 feet from camera</li>
                <li><strong>Good lighting:</strong> Make sure tomato is well-lit, avoid shadows</li>
                <li><strong>Look for white boxes:</strong> These show where the system finds potential objects</li>
                <li><strong>Check console:</strong> Look at terminal output for detection details</li>
            </ol>
            
            <h3>📊 Expected Results:</h3>
            <p>🔴 Red tomatoes → "READY TO USE"</p>
            <p>� Orange tomatoes → "NEED FEW DAYS TO USE"</p>
            <p>�🟡 Yellow tomatoes → "NEED MORE DAYS TO USE"</p>
            <p>🟢 Green tomatoes → "UNREADY TO USE - UNRIPE"</p>
            
            <h3>🎯 Enhanced Features:</h3>
            <p>• Distance-adaptive detection (close/medium/far)</p>
            <p>• Color percentage analysis for accuracy</p>
            <p>• Confidence scoring based on color purity</p>
            <p>• Improved orange/yellow distinction</p>
        </div>
      </body>
    </html>
    '''

if __name__ == '__main__':
    threading.Thread(target=process_stream, daemon=True).start()
    print("🚀 Debug Flask server running at http://localhost:5000")
    print("🔍 Open browser to see both detection and debug views")
    app.run(host='0.0.0.0', port=5000, debug=False)