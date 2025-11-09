#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_timer.h"
#include "img_converters.h"
#include "fb_gfx.h"
#include "esp_http_server.h"

const char* ssid = "Sharvani Setty";
const char* password = "21385003";

String serverName = "http://192.168.1.7:5000/predict";

httpd_handle_t stream_httpd = NULL;
camera_config_t config;

unsigned long lastPredictionTime = 0;
const unsigned long predictionInterval = 10000; // Send prediction request every 10 seconds instead of 5

void startCameraServer();

void setup() {
  Serial.begin(115200);
  Serial.println("Starting ESP32-CAM...");
  
  WiFi.begin(ssid, password);
  Serial.println("Connecting to WiFi...");
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected!");
    Serial.print("ESP32 IP: ");
    Serial.println(WiFi.localIP());
    Serial.print("Stream URL: http://");
    Serial.print(WiFi.localIP());
    Serial.println(":81/stream");
  } else {
    Serial.println("\nFailed to connect to WiFi!");
    ESP.restart();
  }

  // Camera configuration
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = 5;
  config.pin_d1 = 18;
  config.pin_d2 = 19;
  config.pin_d3 = 21;
  config.pin_d4 = 36;
  config.pin_d5 = 39;
  config.pin_d6 = 34;
  config.pin_d7 = 35;
  config.pin_xclk = 0;
  config.pin_pclk = 22;
  config.pin_vsync = 25;
  config.pin_href = 23;
  config.pin_sscb_sda = 26;
  config.pin_sscb_scl = 27;
  config.pin_pwdn = 32;
  config.pin_reset = -1;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_SXGA; // 1280x1024 for high quality
  config.jpeg_quality = 8; // Lower number = higher quality (1-63)
  config.fb_count = 2;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    ESP.restart();
  }

  // Enhanced camera settings for high-quality tomato detection
  sensor_t * s = esp_camera_sensor_get();
  if (s != NULL) {
    s->set_brightness(s, 0);     // -2 to 2 (optimal for tomato colors)
    s->set_contrast(s, 2);       // -2 to 2 (increased for better definition)
    s->set_saturation(s, 2);     // -2 to 2 (enhanced for better color detection)
    s->set_special_effect(s, 0); // 0-No Effect for natural colors
    s->set_whitebal(s, 1);       // Enable white balance
    s->set_awb_gain(s, 1);       // Enable auto white balance gain
    s->set_wb_mode(s, 0);        // Auto white balance
    s->set_exposure_ctrl(s, 1);  // Enable exposure control
    s->set_aec2(s, 0);           // Disable AEC2 for manual control
    s->set_ae_level(s, 0);       // Auto exposure level
    s->set_aec_value(s, 400);    // Exposure value (increased for better clarity)
    s->set_gain_ctrl(s, 1);      // Enable gain control
    s->set_agc_gain(s, 2);       // Auto gain (slight increase for far detection)
    s->set_gainceiling(s, (gainceiling_t)2);  // Gain ceiling for noise reduction
    s->set_bpc(s, 1);            // Enable bad pixel correction
    s->set_wpc(s, 1);            // Enable white pixel correction
    s->set_raw_gma(s, 1);        // Enable gamma correction
    s->set_lenc(s, 1);           // Enable lens correction
    s->set_hmirror(s, 0);        // No horizontal mirror
    s->set_vflip(s, 0);          // No vertical flip
    s->set_dcw(s, 1);            // Enable downsize
    s->set_colorbar(s, 0);       // Disable color bar
    
    // Additional settings for better focus and clarity
    s->set_sharpness(s, 2);      // Increase sharpness for better edge detection
    s->set_denoise(s, 1);        // Enable noise reduction
  }

  startCameraServer();
  Serial.println("Camera stream ready!");
  Serial.println("System initialized successfully!");
}

static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  char part_buf[64];
  
  res = httpd_resp_set_type(req, "multipart/x-mixed-replace;boundary=frame");
  if (res != ESP_OK) return res;
  
  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      res = ESP_FAIL;
    } else {
      httpd_resp_send_chunk(req, "--frame\r\n", strlen("--frame\r\n"));
      sprintf(part_buf, "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", fb->len);
      httpd_resp_send_chunk(req, part_buf, strlen(part_buf));
      httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
      httpd_resp_send_chunk(req, "\r\n", 2);
      esp_camera_fb_return(fb);
    }
    if (res != ESP_OK) break;
    delay(100); // Control frame rate
  }
  return res;
}

void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 81;
  config.stack_size = 8192; // Increase stack size

  httpd_uri_t stream_uri = {
    .uri = "/stream",
    .method = HTTP_GET,
    .handler = stream_handler,
    .user_ctx = NULL
  };

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    Serial.println("HTTP server started successfully");
  } else {
    Serial.println("Failed to start HTTP server");
  }
}

void sendPredictionRequest() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected, skipping prediction request");
    return;
  }

  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed for prediction");
    return;
  }

  HTTPClient http;
  http.begin(serverName);
  http.addHeader("Content-Type", "image/jpeg");
  http.setTimeout(5000); // 5 second timeout
  
  int httpResponseCode = http.POST(fb->buf, fb->len);
  
  if (httpResponseCode > 0) {
    if (httpResponseCode == 200) {
      String response = http.getString();
      Serial.println("Prediction response: " + response);
    } else {
      Serial.printf("HTTP Error: %d\n", httpResponseCode);
    }
  } else {
    Serial.printf("Connection failed: %d\n", httpResponseCode);
  }
  
  http.end();
  esp_camera_fb_return(fb);
}

void loop() {
  unsigned long currentTime = millis();
  
  // Check WiFi connection
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected, attempting reconnection...");
    WiFi.begin(ssid, password);
    delay(5000);
    return;
  }
  
  // Send prediction request at intervals
  if (currentTime - lastPredictionTime >= predictionInterval) {
    Serial.println("Sending prediction request...");
    sendPredictionRequest();
    lastPredictionTime = currentTime;
  }
  
  delay(1000); // Main loop delay
}