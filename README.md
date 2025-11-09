# 🍅 ESP32-CAM Tomato Ripeness Detection System

A real-time tomato ripeness detection system using ESP32-CAM and computer vision to classify tomatoes into different ripeness stages with colored bounding boxes.

## 🎯 Features

- **Real-time Detection**: Live video stream processing from ESP32-CAM
- **4 Ripeness Stages**: Red (Ready), Orange (Few Days), Yellow (More Days), Green (Unripe)
- **Distance Adaptive**: Detects tomatoes from close, medium, and far distances
- **Confidence Scoring**: Shows detection reliability percentage
- **Debug Mode**: Visual analysis of color detection process
- **High Accuracy**: Multi-stage color analysis with pixel-level precision

## 📋 Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [Software Installation](#software-installation)
3. [Arduino IDE Setup](#arduino-ide-setup)
4. [ESP32-CAM Hardware Setup](#esp32-cam-hardware-setup)
5. [Code Upload to ESP32](#code-upload-to-esp32)
6. [Python Environment Setup](#python-environment-setup)
7. [Running the System](#running-the-system)
8. [Testing Guide](#testing-guide)
9. [Troubleshooting](#troubleshooting)
10. [API Documentation](#api-documentation)

---

## 🛠️ Hardware Requirements

### Essential Components
- **ESP32-CAM module** (AI-Thinker or similar)
- **FTDI USB to TTL Serial Adapter** (3.3V/5V)
- **Jumper wires** (Female-to-Female)
- **Breadboard** (optional but recommended)
- **MicroSD card** (optional, for image storage)
- **USB cable** for FTDI adapter

### Optional Components
- **External antenna** for better WiFi range
- **Power supply** (3.3V, if not using USB power)
- **Case/mounting** for ESP32-CAM

---

## 💻 Software Installation

### 1. Arduino IDE Installation

#### Windows:
1. Go to [Arduino IDE Official Website](https://www.arduino.cc/en/software)
2. Download "Arduino IDE 2.x.x for Windows"
3. Run the installer and follow setup wizard
4. Launch Arduino IDE

#### macOS:
1. Download "Arduino IDE 2.x.x for macOS"
2. Drag to Applications folder
3. Launch from Applications

#### Linux:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install arduino

# Or download from website for latest version
```

### 2. Python Installation

#### Windows:
1. Go to [Python Official Website](https://www.python.org/downloads/)
2. Download Python 3.8+ 
3. **IMPORTANT**: Check "Add Python to PATH" during installation
4. Verify installation:
```cmd
python --version
pip --version
```

#### macOS/Linux:
```bash
# macOS (using Homebrew)
brew install python3

# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip

# Verify
python3 --version
pip3 --version
```

---

## 🔧 Arduino IDE Setup

### 1. Add ESP32 Board Support

1. **Open Arduino IDE**
2. **Go to File → Preferences**
3. **Add Board Manager URL**:
   ```
   https://dl.espressif.com/dl/package_esp32_index.json
   ```
   - In "Additional Board Manager URLs" field
   - If there are existing URLs, separate with comma

4. **Install ESP32 Boards**:
   - Go to **Tools → Board → Board Manager**
   - Search for "ESP32"
   - Install **"esp32 by Espressif Systems"**
   - Wait for installation to complete

### 2. Select ESP32-CAM Board

1. **Go to Tools → Board**
2. **Select "ESP32 Arduino"**
3. **Choose "AI Thinker ESP32-CAM"**

### 3. Install Required Libraries

1. **Go to Tools → Manage Libraries**
2. **Search and Install**:
   - No additional libraries needed for basic camera functionality
   - ESP32 core includes camera libraries

---

## 🔌 ESP32-CAM Hardware Setup

### Wiring Diagram

#### ESP32-CAM to FTDI Adapter Connection:

| ESP32-CAM Pin | FTDI Pin | Wire Color (suggested) |
|---------------|----------|------------------------|
| GND          | GND      | Black                  |
| 5V           | 5V       | Red                    |
| UOR          | TX       | Green                  |
| UOT          | RX       | Blue                   |
| IO0          | GND      | Yellow (Programming)   |

### Programming Mode Setup

1. **Connect wires** as per table above
2. **Connect IO0 to GND** (CRITICAL for programming mode)
3. **Plug FTDI adapter** into computer USB port
4. **Press RESET button** on ESP32-CAM while IO0 is connected to GND

### Normal Operation Mode

1. **Disconnect IO0 from GND**
2. **Keep other connections** (GND, 5V, UOR, UOT)
3. **Press RESET button** to start normal operation

---

## 📤 Code Upload to ESP32

### 1. Prepare Arduino IDE

1. **Open Arduino IDE**
2. **Select Board**: Tools → Board → ESP32 Arduino → AI Thinker ESP32-CAM
3. **Select Port**: Tools → Port → (your FTDI adapter port)
   - Windows: Usually COM3, COM4, etc.
   - macOS: Usually /dev/cu.usbserial-xxxx
   - Linux: Usually /dev/ttyUSB0

### 2. Load ESP32 Code

1. **Open** `improved_esp32_code.ino` file
2. **Configure WiFi Credentials**:
   ```cpp
   const char* ssid = "YOUR_WIFI_NAME";
   const char* password = "YOUR_WIFI_PASSWORD";
   ```
3. **Verify IP Settings** (optional):
   ```cpp
   // For static IP (optional)
   IPAddress local_IP(192, 168, 1, 8);
   IPAddress gateway(192, 168, 1, 1);
   ```

### 3. Upload Process

1. **Put ESP32-CAM in Programming Mode**:
   - Connect IO0 to GND
   - Press RESET button
   
2. **Click Upload Button** (→) in Arduino IDE

3. **Monitor Upload Progress**:
   ```
   Connecting........_____.....
   Chip is ESP32-D0WD-V3 (revision 3)
   ```

4. **If Upload Successful**:
   ```
   Hash of data verified.
   Leaving... Hard resetting via RTS pin...
   ```

### 4. Switch to Normal Mode

1. **Disconnect IO0 from GND**
2. **Press RESET button**
3. **ESP32-CAM should start normally**

---

## 🔍 Serial Monitor Testing

### 1. Open Serial Monitor

1. **In Arduino IDE**: Tools → Serial Monitor
2. **Set Baud Rate**: 115200
3. **Set Line Ending**: "Both NL & CR"

### 2. Expected Output

```
ESP32-CAM Tomato Detection System
Connecting to WiFi: YOUR_NETWORK_NAME
.....
WiFi connected!
IP address: 192.168.1.8
Camera initialized successfully
Server started on port 81
Stream URL: http://192.168.1.8:81/stream
```

### 3. Error Messages and Solutions

| Error Message | Solution |
|---------------|----------|
| "WiFi connection failed" | Check SSID/password, WiFi range |
| "Camera init failed" | Check camera module connection |
| "Brownout detector" | Use better power supply (5V 2A) |
| "Guru Meditation Error" | Press RESET, check wiring |

---

## 🐍 Python Environment Setup

### 1. Navigate to Project Directory

```bash
# Windows
cd "C:\Desktop\CLIENT MAIN\tomato_detector"

# macOS/Linux  
cd "/path/to/tomato_detector"
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv tomato_env
tomato_env\Scripts\activate

# macOS/Linux
python3 -m venv tomato_env
source tomato_env/bin/activate
```

### 3. Install Required Packages

#### Option A: Basic Installation (OpenCV Only)
```bash
pip install flask opencv-python numpy requests
```

#### Option B: TensorFlow-Enhanced Installation (Recommended)
```bash
pip install -r requirements.txt
```

**Or manually install TensorFlow packages:**
```bash
pip install flask opencv-python numpy requests tensorflow pillow
```

#### Option C: GPU-Accelerated TensorFlow (If you have NVIDIA GPU)
```bash
pip install tensorflow-gpu
# Note: Requires CUDA and cuDNN installation
```

### 4. Verify Installation

```bash
python -c "import cv2, flask, numpy; print('All packages installed successfully!')"
```

---

## 🚀 Running the System

### 1. Update ESP32 IP Address

1. **Check Serial Monitor** for ESP32 IP address
2. **Edit Python files**:
   ```python
   # In debug_app.py and app.py
   ESP32_URL = "http://192.168.1.8:81/stream"  # Replace with your IP
   ```

### 2. Start Detection Server

#### Debug Mode (Recommended for testing):
```bash
python debug_app.py
```

#### Production Mode:
```bash
python app.py
```

#### TensorFlow-Enhanced Mode (Recommended):
```bash
python tensorflow_app.py
```

### 3. Access Web Interface

1. **Open browser**
2. **Navigate to**: `http://localhost:5000`
3. **You should see**:
   - Detection View (processed video)
   - Debug View (color analysis)
   - Status information

---

## � TensorFlow Integration

### What is TensorFlow?

**TensorFlow** is Google's open-source machine learning framework that enables:

#### 🎯 **Key Advantages for Tomato Detection:**

1. **Deep Learning Classification**
   - Neural networks learn from thousands of tomato images
   - Recognizes patterns beyond just color (texture, shape, shadows)
   - More robust than rule-based color detection

2. **Advanced Feature Recognition**
   - Detects tomato-specific characteristics
   - Understands context and lighting variations
   - Learns from mistakes and improves accuracy

3. **Professional ML Approach**
   - Industry-standard framework used by Google, Tesla, etc.
   - Scalable and maintainable
   - Impressive for academic/professional projects

4. **Confidence Scoring**
   - Provides probability scores for each ripeness class
   - Shows uncertainty levels in predictions
   - Allows threshold-based filtering

### 🔄 **Detection Method Comparison:**

| Feature | OpenCV Method | TensorFlow Method |
|---------|---------------|-------------------|
| **Accuracy** | Good in ideal lighting | Excellent in all conditions |
| **Robustness** | Color-dependent | Pattern + texture + color |
| **Speed** | Very Fast (real-time) | Fast (mobile-optimized) |
| **Learning** | Fixed rules | Learns from data |
| **Professional** | Basic computer vision | Advanced machine learning |
| **Lighting** | Sensitive to changes | Adapts to variations |

### 🚀 **Hybrid Approach (Best of Both Worlds):**

Our TensorFlow implementation uses a **hybrid approach**:

1. **OpenCV**: Fast object detection and region extraction
2. **TensorFlow**: Intelligent classification of detected regions
3. **Combined**: Speed + Accuracy + Robustness

```python
# Workflow:
OpenCV → Find potential tomatoes → TensorFlow → Classify ripeness
```

### 🏗️ **TensorFlow Architecture Used:**

- **Base Model**: MobileNetV2 (mobile-optimized, fast inference)
- **Custom Layers**: Added for tomato-specific classification
- **Classes**: 4 ripeness stages (Green, Yellow, Orange, Red)
- **Input Size**: 224x224 pixels (standard for mobile models)
- **Output**: Probability distribution across all classes

### 📊 **TensorFlow Output Example:**
```
🍅 TensorFlow Detection: 🔴 READY TO USE (Confidence: 87%)
Predictions - Green:0.05, Yellow:0.08, Orange:0.12, Red:0.87
```

This shows the model is 87% confident it's a red tomato, with low probabilities for other classes.

---

## �🧪 Testing Guide

### Test Case 1: Basic Connection Test

**Objective**: Verify ESP32-CAM and server connection

**Steps**:
1. Upload code to ESP32-CAM
2. Check Serial Monitor for IP address
3. Start `debug_app.py`
4. Open browser to `http://localhost:5000`

**Expected Result**:
- Video stream appears
- Status shows "No Tomato Detected"
- Debug view shows camera feed

**Troubleshooting**:
- No video: Check IP address in code
- Connection timeout: Verify WiFi connection
- Black screen: Check camera module seating

### Test Case 2: Red Tomato Detection

**Objective**: Test ripe tomato detection

**Steps**:
1. Hold a red tomato 1-2 feet from camera
2. Ensure good lighting
3. Observe detection view and terminal output

**Expected Result**:
```
🔴 READY TO USE | Confidence: 85% | Distance: MEDIUM
```
- Red bounding box appears
- Status shows "READY TO USE"

### Test Case 3: Green Tomato Detection

**Objective**: Test unripe tomato detection

**Steps**:
1. Use green/unripe tomato
2. Hold at various distances
3. Check debug output for color analysis

**Expected Result**:
```
🟢 UNREADY TO USE - UNRIPE | Confidence: 70% | Distance: FAR
Color percentages - R:0.02, O:0.01, Y:0.15, G:0.82
```

### Test Case 4: Distance Adaptation Test

**Objective**: Test far-distance detection

**Steps**:
1. Start with tomato close (1 foot)
2. Gradually move away (up to 6 feet)
3. Observe confidence and distance classification

**Expected Results**:
- **Close**: Large bounding box, high confidence
- **Medium**: Medium box, moderate confidence  
- **Far**: Small box, lower confidence but still detected

### Test Case 5: Color Accuracy Test

**Objective**: Test orange/yellow distinction

**Setup**: Use tomatoes at different ripeness stages

**Test Matrix**:
| Tomato Color | Expected Detection | Confidence Range |
|--------------|-------------------|------------------|
| Deep Red | 🔴 READY TO USE | 80-95% |
| Orange-Red | 🟠 NEED FEW DAYS | 70-85% |
| Yellow | 🟡 NEED MORE DAYS | 75-90% |
| Green | 🟢 UNRIPE | 60-80% |

### Test Case 6: False Positive Test

**Objective**: Ensure system doesn't detect non-tomatoes

**Steps**:
1. Show red apple, orange ball, yellow lemon
2. Test with green leaves, walls
3. Verify no false detections

**Expected Result**:
- Terminal shows "REJECTED: Not tomato-like colors"
- No bounding boxes appear

### Test Case 7: Lighting Conditions Test

**Objective**: Test various lighting conditions

**Test Scenarios**:
1. **Bright sunlight**: Should work well
2. **Indoor lighting**: Moderate performance expected  
3. **Dim lighting**: May have reduced confidence
4. **Backlit**: Should avoid false shadows

### Test Case 8: Multiple Tomatoes Test

**Objective**: Test with multiple tomatoes in view

**Steps**:
1. Place 2-3 tomatoes of different colors
2. System should detect the largest/most prominent one
3. Check terminal for all analyzed objects

**Expected Behavior**:
- Detects most prominent tomato
- Shows analysis for multiple candidates
- Consistent detection priority

---

## 🔧 Troubleshooting

### Common ESP32-CAM Issues

#### Upload Failed
```
Error: Failed to connect to ESP32
```
**Solutions**:
1. Check IO0 to GND connection
2. Press and hold RESET while clicking upload
3. Try different baud rate (115200 → 460800)
4. Check FTDI adapter drivers

#### Camera Not Working
```
Camera init failed
```
**Solutions**:
1. Reseat camera ribbon cable
2. Check camera module for damage
3. Try different power supply (5V 2A recommended)
4. Verify ESP32-CAM board selection

#### WiFi Connection Issues
```
WiFi connection failed
```
**Solutions**:
1. Verify SSID and password
2. Check WiFi signal strength
3. Try 2.4GHz network (ESP32 doesn't support 5GHz)
4. Restart router if needed

### Python Application Issues

#### Import Errors
```
ModuleNotFoundError: No module named 'cv2'
```
**Solutions**:
```bash
pip install opencv-python
# If still fails:
pip install opencv-python-headless
```

#### Camera Stream Timeout
```
Stream timeout triggered after 30000 ms
```
**Solutions**:
1. Check ESP32-CAM power supply
2. Verify network stability
3. Restart ESP32-CAM
4. Check IP address in Python code

#### No Detection Despite Visible Tomato
**Debug Steps**:
1. Check terminal output for contour analysis
2. Verify lighting conditions
3. Adjust HSV ranges if needed
4. Use debug view to see color masks

### Performance Issues

#### Slow Detection
**Solutions**:
1. Reduce frame size in ESP32 code
2. Increase time.sleep() in Python processing
3. Use production app.py instead of debug_app.py

#### High CPU Usage
**Solutions**:
1. Reduce processing frequency
2. Lower camera resolution
3. Close unnecessary applications

---

## 📡 API Documentation

### Endpoints

#### GET `/`
- **Description**: Main web interface
- **Returns**: HTML page with video streams

#### GET `/video_feed`
- **Description**: Processed video stream with detection boxes
- **Returns**: MJPEG stream

#### GET `/debug_feed`
- **Description**: Debug visualization stream
- **Returns**: MJPEG stream with color analysis

#### GET/POST `/predict`
- **Description**: Get/Send detection results
- **GET Returns**: 
  ```json
  {
    "color": "red|orange|yellow|green|none",
    "result": "🔴 READY TO USE"
  }
  ```
- **POST**: Accepts image data from ESP32-CAM

### ESP32-CAM Endpoints

#### GET `http://ESP32_IP:81/stream`
- **Description**: Raw camera video stream
- **Returns**: MJPEG stream

#### POST `http://SERVER_IP:5000/predict`
- **Description**: ESP32 sends images for analysis
- **Frequency**: Every 10 seconds
- **Data**: Raw JPEG image data

---

## 🎯 Advanced Configuration

### Camera Settings (ESP32)

```cpp
// In improved_esp32_code.ino
config.frame_size = FRAMESIZE_SXGA;    // 1280x1024 (high quality)
config.jpeg_quality = 8;              // 1-63, lower = better quality
config.fb_count = 2;                  // Frame buffers
```

### Detection Parameters (Python)

```python
# In debug_app.py - HSV color ranges
lower_red1 = np.array([0, 80, 50])     # Red tomatoes
upper_red1 = np.array([10, 255, 255])

lower_green = np.array([35, 70, 50])   # Green tomatoes  
upper_green = np.array([75, 255, 200])
```

### Size Thresholds

```python
# Area filtering for different distances
if area < 300 or area > 80000:  # Very wide range
    continue

# Distance classification
if area > 10000: distance_class = "close"
elif area > 2000: distance_class = "medium"  
else: distance_class = "far"
```

---

## 🚀 Future Enhancements

### Planned Features
- [ ] Mobile app interface
- [ ] Database logging of detections  
- [ ] Email/SMS notifications
- [ ] Multiple camera support
- [ ] Machine learning model training
- [ ] Cloud storage integration

### Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📞 Support

### Getting Help
1. **Check troubleshooting section** above
2. **Review terminal output** for error messages
3. **Test individual components** (ESP32, camera, Python)
4. **Verify all connections** and power supply

### Common Questions

**Q: Why is detection accuracy low?**
A: Check lighting conditions, camera focus, and HSV color ranges

**Q: Can I use different camera module?**  
A: Yes, but may require code modifications for different resolutions

**Q: How to improve detection distance?**
A: Use higher resolution, better lighting, and adjust size thresholds

**Q: System detects wrong colors?**
A: Calibrate HSV ranges for your specific lighting conditions

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- ESP32-CAM community for hardware support
- OpenCV team for computer vision library
- Flask team for web framework
- Arduino community for development environment

---

## 📁 Project Files

### 🔹 **Essential Files (Must Keep)**
- **`tensorflow_app.py`** - TensorFlow-enhanced detection with deep learning (NEW!)
- **`debug_app.py`** - OpenCV-based detection with color analysis
- **`improved_esp32_code.ino`** - ESP32-CAM firmware with optimized camera settings
- **`requirements.txt`** - Python dependencies including TensorFlow
- **`README.md`** - Complete setup and usage documentation
- **`.gitignore`** - Git ignore rules for clean repository

### 🔸 **Important Files (Recommended to Keep)**
- **`app.py`** - Production version of detection server (simpler, faster)
- **`templates/index.html`** - Web interface template (if using app.py)
- **`templates/overlay.html`** - Alternative web interface

### 🔹 **Optional Files (Your Choice)**
- **`tomato_simple_model.h5`** - Pre-trained ML model (currently NOT used by the system)
- **`simple_app.py`** - Simplified version for basic testing
- **`tomato_detection_features.py`** - Feature documentation

### 🔸 **Development/Testing Files (Can Remove)**
- **`test_detection.py`** - Testing script
- **`test_server.py`** - Server testing script  
- **`train_tomato_ai.py`** - ML model training script
- **`.venv/`** - Virtual environment (excluded by .gitignore)

### 📋 **About the H5 File**
The `tomato_simple_model.h5` file contains a pre-trained machine learning model, but **it's currently NOT being used** by your working system. Your current implementation uses **computer vision with HSV color analysis** which is:
- ✅ More accurate for color-based detection
- ✅ Faster processing
- ✅ No ML dependencies required
- ✅ Works in real-time

**Recommendation**: You can keep the H5 file for future ML experiments, but it's not essential for current functionality.

### 🚀 **Minimal Git Repository Files**
For a clean repository, you only need:
```
├── debug_app.py              # Main application
├── improved_esp32_code.ino   # ESP32 firmware  
├── README.md                 # Documentation
├── .gitignore               # Git ignore rules
└── app.py                   # Production version (optional)
```

---

**Happy Tomato Detecting! 🍅**

*Last Updated: November 9, 2025*