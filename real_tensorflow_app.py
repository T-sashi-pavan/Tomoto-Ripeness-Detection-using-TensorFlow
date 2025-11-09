"""
🍅 REAL TensorFlow Tomato Detection System
=========================================
Uses actual trained model OR creates a functional detection system
"""

from flask import Flask, Response, jsonify, request
import cv2
import numpy as np
import threading
import time
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
import os
import pickle

app = Flask(__name__)

# === CHANGE THIS TO YOUR ESP32 IP ===
ESP32_URL = "http://192.168.1.8:81/stream"

# Global variables
latest_result = {"color": "unknown", "result": "No Tomato"}
latest_frame = None
debug_frame = None
frame_lock = threading.Lock()

class RealTomatoDetector:
    def __init__(self):
        """Initialize REAL tomato detection system"""
        print("🍅 Initializing REAL Tomato Detection System...")
        
        # Try to load existing trained model
        self.model = None
        self.use_opencv_fallback = True
        
        model_path = "trained_tomato_model.h5"
        
        if os.path.exists(model_path):
            try:
                print("📁 Loading existing trained model...")
                self.model = tf.keras.models.load_model(model_path)
                self.use_opencv_fallback = False
                print("✅ Trained TensorFlow model loaded!")
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
                print("🔄 Using OpenCV-based intelligent system instead...")
        else:
            print("📝 No trained model found. Using enhanced OpenCV system...")
        
        # Tomato ripeness classes
        self.ripeness_classes = {
            0: {"name": "green", "label": "🟢 UNREADY TO USE - UNRIPE", "color": (0, 255, 0)},
            1: {"name": "yellow", "label": "🟡 NEED MORE DAYS TO USE", "color": (0, 255, 255)},
            2: {"name": "orange", "label": "🟠 NEED FEW DAYS TO USE", "color": (0, 140, 255)},
            3: {"name": "red", "label": "🔴 READY TO USE", "color": (0, 0, 255)}
        }
        
        # Enhanced feature extraction parameters
        self.color_ranges = {
            'red': ([0, 80, 50], [15, 255, 255]),
            'red2': ([165, 80, 50], [180, 255, 255]),
            'orange': ([10, 100, 80], [25, 255, 255]),
            'yellow': ([25, 80, 100], [35, 255, 255]),
            'green': ([35, 70, 50], [75, 255, 200])
        }
        
        print("✅ REAL Detection System Ready!")
    
    def extract_features(self, img_region):
        """Extract comprehensive features from tomato region"""
        features = {}
        
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(img_region, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img_region, cv2.COLOR_BGR2LAB)
        
        # 1. Color Distribution Analysis
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        
        # Find dominant hue
        dominant_hue = np.argmax(hist_h)
        dominant_sat = np.argmax(hist_s)
        dominant_val = np.argmax(hist_v)
        
        features['dominant_hue'] = dominant_hue
        features['dominant_sat'] = dominant_sat
        features['dominant_val'] = dominant_val
        
        # 2. Color Purity Analysis
        mean_hsv = np.mean(hsv.reshape(-1, 3), axis=0)
        std_hsv = np.std(hsv.reshape(-1, 3), axis=0)
        
        features['mean_hue'] = mean_hsv[0]
        features['mean_sat'] = mean_hsv[1]
        features['mean_val'] = mean_hsv[2]
        features['hue_consistency'] = 180 - std_hsv[0]  # Higher = more consistent
        features['sat_consistency'] = 255 - std_hsv[1]
        
        # 3. Texture Analysis (Simple)
        gray = cv2.cvtColor(img_region, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features['texture_variance'] = np.var(laplacian)
        
        # 4. Shape Regularity
        h, w = img_region.shape[:2]
        features['aspect_ratio'] = w / h if h > 0 else 1.0
        features['area_ratio'] = (w * h) / (img_region.size / 3)  # Normalized area
        
        return features
    
    def intelligent_classification(self, features):
        """Intelligent rule-based classification using extracted features"""
        hue = features['mean_hue']
        sat = features['mean_sat']
        val = features['mean_val']
        consistency = features['hue_consistency']
        
        # Multi-factor scoring system
        scores = {'red': 0, 'orange': 0, 'yellow': 0, 'green': 0}
        
        # Hue-based scoring
        if hue < 15 or hue > 160:
            scores['red'] += 0.6
        elif 10 <= hue <= 25:
            scores['orange'] += 0.7
            scores['red'] += 0.3
        elif 25 < hue <= 35:
            scores['yellow'] += 0.7
            scores['orange'] += 0.2
        elif 35 < hue <= 75:
            scores['green'] += 0.8
            if hue < 50:
                scores['yellow'] += 0.1
        
        # Saturation weighting
        sat_factor = min(1.0, sat / 150.0)  # Normalize saturation
        for key in scores:
            scores[key] *= sat_factor
        
        # Value (brightness) adjustment
        if val < 50:  # Too dark
            for key in scores:
                scores[key] *= 0.5
        elif val > 200:  # Very bright
            scores['yellow'] += 0.2
        
        # Consistency bonus
        consistency_bonus = min(0.3, consistency / 180.0)
        for key in scores:
            scores[key] += consistency_bonus
        
        # Find best classification
        best_class = max(scores, key=scores.get)
        confidence = scores[best_class]
        
        # Map to class index
        class_mapping = {'green': 0, 'yellow': 1, 'orange': 2, 'red': 3}
        class_idx = class_mapping[best_class]
        
        return {
            'class': class_idx,
            'confidence': min(1.0, confidence),
            'ripeness': best_class,
            'scores': scores,
            'features': features
        }
    
    def predict_ripeness(self, img_region):
        """Predict tomato ripeness using available method"""
        try:
            if self.model is not None and not self.use_opencv_fallback:
                # Use trained TensorFlow model
                return self.tensorflow_predict(img_region)
            else:
                # Use intelligent OpenCV-based system
                return self.opencv_predict(img_region)
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def tensorflow_predict(self, img_region):
        """TensorFlow prediction (if model available)"""
        # Preprocess for model
        img_resized = cv2.resize(img_region, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_array = np.expand_dims(img_rgb, axis=0) / 255.0
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        class_info = self.ripeness_classes[predicted_class]
        
        return {
            "class": predicted_class,
            "confidence": confidence,
            "ripeness": class_info["name"],
            "label": class_info["label"],
            "color": class_info["color"],
            "method": "TensorFlow",
            "all_predictions": predictions[0].tolist()
        }
    
    def opencv_predict(self, img_region):
        """Enhanced OpenCV-based prediction"""
        # Extract comprehensive features
        features = self.extract_features(img_region)
        
        # Intelligent classification
        result = self.intelligent_classification(features)
        
        class_info = self.ripeness_classes[result['class']]
        
        return {
            "class": result['class'],
            "confidence": result['confidence'],
            "ripeness": result['ripeness'],
            "label": class_info["label"],
            "color": class_info["color"],
            "method": "Enhanced OpenCV",
            "scores": result['scores'],
            "features": result['features']
        }

# Initialize detector
print("🚀 Starting REAL Tomato Detection System...")
detector = RealTomatoDetector()

def advanced_tomato_detection(frame):
    """Advanced detection with real analysis"""
    frame_resized = cv2.resize(frame, (640, 480))
    debug_vis = frame_resized.copy()
    
    # Enhanced preprocessing
    blur = cv2.GaussianBlur(frame_resized, (7, 7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    # Create comprehensive mask for all tomato colors
    masks = []
    for color_name, (lower, upper) in detector.color_ranges.items():
        if color_name != 'red2':
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            masks.append(mask)
    
    # Red requires two ranges
    red_mask1 = cv2.inRange(hsv, np.array(detector.color_ranges['red'][0]), 
                           np.array(detector.color_ranges['red'][1]))
    red_mask2 = cv2.inRange(hsv, np.array(detector.color_ranges['red2'][0]), 
                           np.array(detector.color_ranges['red2'][1]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    masks.append(red_mask)
    
    # Combine all masks
    combined_mask = masks[0]
    for mask in masks[1:]:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Advanced morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    result = "No Tomato Detected"
    color = "none"
    
    print(f"🔍 Analyzing {len(contours)} potential objects...")
    
    for i, cnt in enumerate(contours[:3]):  # Check top 3 candidates
        area = cv2.contourArea(cnt)
        
        if area < 1000 or area > 100000:
            continue
        
        # Enhanced shape analysis
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
        # Fit ellipse for better shape analysis
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (center, axes, angle) = ellipse
            major_axis, minor_axis = max(axes), min(axes)
            eccentricity = np.sqrt(1 - (minor_axis/major_axis)**2) if major_axis > 0 else 1
        else:
            continue
        
        # Multiple shape criteria
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h
        
        # Shape filtering
        if (circularity > 0.3 and eccentricity < 0.8 and 
            0.5 < aspect_ratio < 2.0):
            
            print(f"🎯 Analyzing candidate {i}: area={area:.0f}, circ={circularity:.3f}, ecc={eccentricity:.3f}")
            
            # Extract region for analysis
            roi = frame_resized[y:y+h, x:x+w]
            
            # Use detector for classification
            prediction = detector.predict_ripeness(roi)
            
            if prediction and prediction["confidence"] > 0.4:
                
                box_color = prediction["color"]
                confidence_pct = int(prediction["confidence"] * 100)
                method = prediction["method"]
                
                # Draw enhanced detection box
                thickness = max(2, int(prediction["confidence"] * 6))
                cv2.rectangle(frame_resized, (x, y), (x+w, y+h), box_color, thickness)
                
                # Enhanced labeling
                main_text = prediction["label"]
                detail_text = f"{method} | {confidence_pct}% | Area: {int(area)}"
                
                # Background for text
                text_size = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame_resized, (x, y-70), (x + max(text_size[0], 400), y), box_color, -1)
                
                # Main label
                cv2.putText(frame_resized, main_text, (x + 5, y-45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Detail info
                cv2.putText(frame_resized, detail_text, (x + 5, y-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Additional analysis info
                if "scores" in prediction:
                    score_text = f"R:{prediction['scores']['red']:.2f} O:{prediction['scores']['orange']:.2f} Y:{prediction['scores']['yellow']:.2f} G:{prediction['scores']['green']:.2f}"
                    cv2.putText(frame_resized, score_text, (x + 5, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                
                # Debug visualization
                cv2.rectangle(debug_vis, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(debug_vis, f"{method} {confidence_pct}%", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                result = prediction["label"]
                color = prediction["ripeness"]
                
                print(f"🍅 REAL Detection: {result} ({method}, {confidence_pct}%)")
                if "scores" in prediction:
                    print(f"   Scores: {prediction['scores']}")
                
                break
    
    return frame_resized, debug_vis, result, color

def process_stream():
    global latest_frame, latest_result, debug_frame
    print("🌐 Connecting to ESP32-CAM stream:", ESP32_URL)

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

        # Process with real detection system
        processed_frame, debug_vis, result, color = advanced_tomato_detection(frame)
        
        latest_result = {"color": color, "result": result}
        
        with frame_lock:
            latest_frame = processed_frame.copy()
            debug_frame = debug_vis.copy()
        
        time.sleep(0.4)  # Reasonable processing rate

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
        try:
            image_data = request.get_data()
            if len(image_data) > 0:
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    _, _, result, color = advanced_tomato_detection(img)
                    return jsonify({"color": color, "result": result})
        except Exception as e:
            print(f"Error processing POST image: {e}")
    
    return jsonify(latest_result)

@app.route('/')
def index():
    method = "Enhanced OpenCV" if detector.use_opencv_fallback else "TensorFlow"
    return f'''
    <html>
      <head><title>🍅 REAL Tomato Detection</title></head>
      <body style="background:#000; color:#fff; text-align:center; font-family:Arial;">
        <h1>🍅 REAL Working Tomato Detection</h1>
        <h2>🤖 Method: {method}</h2>
        
        <div style="display: flex; justify-content: center; gap: 20px; margin: 20px;">
            <div>
                <h3>📹 Detection Results</h3>
                <img src="/video_feed" width="640" height="480" style="border:2px solid #333; border-radius:10px;"/>
            </div>
            <div>
                <h3>🔍 Analysis Debug</h3>
                <img src="/debug_feed" width="640" height="480" style="border:2px solid #333; border-radius:10px;"/>
            </div>
        </div>
        
        <h2>Status: <span id="status">Loading...</span></h2>
        
        <div style="background:#111; padding:20px; margin:20px; border-radius:10px;">
            <h3>✅ REAL Detection Features:</h3>
            <p>🔬 Multi-factor Feature Extraction</p>
            <p>🎨 Advanced Color Space Analysis</p>
            <p>📐 Shape Regularity Assessment</p>
            <p>🧮 Intelligent Rule-based Classification</p>
            <p>⚡ Fast Real-time Processing</p>
            <p>📊 Confidence Scoring System</p>
        </div>
        
        <script>
          setInterval(function() {{
            fetch('/predict')
              .then(response => response.json())
              .then(data => {{
                document.getElementById('status').textContent = data.result;
                document.getElementById('status').style.color = 
                  data.color === 'red' ? '#ff4444' :
                  data.color === 'orange' ? '#ff8844' :
                  data.color === 'yellow' ? '#ffff44' :
                  data.color === 'green' ? '#44ff44' : '#ffffff';
              }});
          }}, 1500);
        </script>
      </body>
    </html>
    '''

if __name__ == '__main__':
    threading.Thread(target=process_stream, daemon=True).start()
    print("🚀 REAL Detection Server running at http://localhost:5000")
    print("✅ Using intelligent feature-based classification!")
    app.run(host='0.0.0.0', port=5000, debug=False)