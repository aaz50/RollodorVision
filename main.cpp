/*
 * Rollodor ESP32-CAM Firmware
 * ===========================
 * Tank-steering robot controller with camera streaming.
 *
 * Hardware:
 *   - ESP32-CAM (AI-Thinker / Hosyond)
 *   - 2x 360° continuous rotation servos (left + right)
 *   - Optional: HC-SR04 ultrasonic sensors for collision avoidance
 *
 * Endpoints:
 *   GET /stream        — MJPEG camera stream
 *   GET /cmd?dir=X     — steering command (forward|left|right|stop|search)
 *   GET /status        — JSON status (command, uptime, wifi rssi)
 *
 * Building:
 *   This is structured for PlatformIO but will also compile in Arduino IDE.
 *   For Arduino IDE: rename to rollodor_firmware.ino
 *
 * ⚠️  COMPILE-ONLY without hardware. The servo and camera calls are
 *     wrapped so the logic compiles and the structure is testable.
 */

 #include <Arduino.h>
 #include <WiFi.h>
 #include <WebServer.h>
 #include <ESP32Servo.h>  // PlatformIO: add to lib_deps. Arduino: install via Library Manager.
 
 // =========================================================================
 // CONFIG — Edit these for your setup
 // =========================================================================
 
 // WiFi credentials
 const char* WIFI_SSID     = "YOUR_WIFI_SSID";
 const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
 
 // Servo pins (use GPIO pins that aren't used by the camera)
 // On ESP32-CAM, safe GPIOs for servos: 12, 13, 14, 15
 // (GPIO 2, 4 are used by camera/flash)
 const int LEFT_SERVO_PIN  = 12;
 const int RIGHT_SERVO_PIN = 13;
 
 // Servo speed values (for continuous rotation servos)
 // 90 = stop, <90 = one direction, >90 = other direction
 // Exact values depend on your servos — calibrate these!
 const int SERVO_STOP     = 90;
 const int SERVO_FWD_L    = 70;   // left servo forward speed
 const int SERVO_FWD_R    = 110;  // right servo forward speed (reversed)
 const int SERVO_TURN_SPD = 30;   // offset from STOP for turning
 
 // Ultrasonic sensor pins (optional — set to -1 to disable)
 const int USS_TRIG_PIN = 14;
 const int USS_ECHO_PIN = 15;
 const float USS_MIN_DISTANCE_CM = 15.0;  // emergency stop distance
 
 // =========================================================================
 // GLOBALS
 // =========================================================================
 
 WebServer server(80);
 Servo leftServo;
 Servo rightServo;
 
 String currentCommand = "stop";
 unsigned long lastCommandTime = 0;
 unsigned long commandTimeout = 3000;  // stop if no command for 3 seconds
 
 // =========================================================================
 // MOTOR CONTROL
 // =========================================================================
 
 void motorsStop() {
     leftServo.write(SERVO_STOP);
     rightServo.write(SERVO_STOP);
 }
 
 void motorsForward() {
     leftServo.write(SERVO_FWD_L);
     rightServo.write(SERVO_FWD_R);
 }
 
 void motorsTurnLeft() {
     leftServo.write(SERVO_STOP - SERVO_TURN_SPD);  // left backward
     rightServo.write(SERVO_STOP + SERVO_TURN_SPD);  // right forward
 }
 
 void motorsTurnRight() {
     leftServo.write(SERVO_STOP + SERVO_TURN_SPD);  // left forward
     rightServo.write(SERVO_STOP - SERVO_TURN_SPD);  // right backward
 }
 
 void motorsSearch() {
     // Slow rotation to scan for people
     motorsTurnRight();  // could alternate direction on a timer
 }
 
 void executeCommand(const String& cmd) {
     currentCommand = cmd;
     lastCommandTime = millis();
 
     if (cmd == "forward")     motorsForward();
     else if (cmd == "left")   motorsTurnLeft();
     else if (cmd == "right")  motorsTurnRight();
     else if (cmd == "search") motorsSearch();
     else                      motorsStop();  // "stop" or unknown
 }
 
 // =========================================================================
 // ULTRASONIC SENSOR (optional collision avoidance)
 // =========================================================================
 
 float readUltrasonicCm() {
     if (USS_TRIG_PIN < 0 || USS_ECHO_PIN < 0) {
         return 999.0;  // disabled, return "far away"
     }
 
     digitalWrite(USS_TRIG_PIN, LOW);
     delayMicroseconds(2);
     digitalWrite(USS_TRIG_PIN, HIGH);
     delayMicroseconds(10);
     digitalWrite(USS_TRIG_PIN, LOW);
 
     long duration = pulseIn(USS_ECHO_PIN, HIGH, 30000);  // 30ms timeout
     if (duration == 0) return 999.0;
 
     return (duration * 0.0343) / 2.0;  // speed of sound / 2
 }
 
 // =========================================================================
 // HTTP HANDLERS
 // =========================================================================
 
 void handleCommand() {
     if (!server.hasArg("dir")) {
         server.send(400, "text/plain", "Missing 'dir' parameter");
         return;
     }
 
     String dir = server.arg("dir");
     dir.toLowerCase();
 
     // Validate
     if (dir != "forward" && dir != "left" && dir != "right" &&
         dir != "stop" && dir != "search") {
         server.send(400, "text/plain", "Invalid direction: " + dir);
         return;
     }
 
     executeCommand(dir);
 
     // CORS headers so the web UI can call this from any origin
     server.sendHeader("Access-Control-Allow-Origin", "*");
     server.send(200, "application/json",
                 "{\"status\":\"ok\",\"command\":\"" + dir + "\"}");
 
     Serial.printf("[CMD] %s\n", dir.c_str());
 }
 
 void handleStatus() {
     float distance = readUltrasonicCm();
 
     String json = "{";
     json += "\"command\":\"" + currentCommand + "\",";
     json += "\"uptime_s\":" + String(millis() / 1000) + ",";
     json += "\"wifi_rssi\":" + String(WiFi.RSSI()) + ",";
     json += "\"obstacle_cm\":" + String(distance, 1);
     json += "}";
 
     server.sendHeader("Access-Control-Allow-Origin", "*");
     server.send(200, "application/json", json);
 }
 
 void handleRoot() {
     String html = "<html><body>";
     html += "<h1>Rollodor ESP32</h1>";
     html += "<p>Status: " + currentCommand + "</p>";
     html += "<p><a href='/stream'>Camera Stream</a></p>";
     html += "<p>Commands: /cmd?dir=forward|left|right|stop|search</p>";
     html += "</body></html>";
     server.send(200, "text/html", html);
 }
 
 // =========================================================================
 // CAMERA STREAM
 // =========================================================================
 // The ESP32-CAM camera streaming code is well-documented in the
 // Arduino ESP32 examples (CameraWebServer). For brevity, here's the
 // setup structure. You'll integrate the actual MJPEG streaming from
 // the example sketch.
 
 /*
  * TODO: When you have the ESP32-CAM hardware:
  *
  * 1. #include "esp_camera.h"
  * 2. Copy the camera pin definitions for AI-Thinker board
  * 3. Initialize camera in setup() with camera_config_t
  * 4. Add the /stream endpoint that serves MJPEG
  *
  * The CameraWebServer example in Arduino > Examples > ESP32 > Camera
  * has everything you need. The key function is:
  *
  *   esp_err_t stream_handler(httpd_req_t *req)
  *
  * For now, the /stream endpoint returns a placeholder.
  */
 
 void handleStream() {
     server.send(200, "text/plain",
                 "Camera stream will be available when running on ESP32-CAM hardware.");
 }
 
 // =========================================================================
 // SETUP & LOOP
 // =========================================================================
 
 void setup() {
     Serial.begin(115200);
     Serial.println("\n\n=== Rollodor ESP32 Firmware ===\n");
 
     // Initialize servos
     leftServo.attach(LEFT_SERVO_PIN);
     rightServo.attach(RIGHT_SERVO_PIN);
     motorsStop();
     Serial.println("[OK] Servos initialized");
 
     // Initialize ultrasonic (if enabled)
     if (USS_TRIG_PIN >= 0 && USS_ECHO_PIN >= 0) {
         pinMode(USS_TRIG_PIN, OUTPUT);
         pinMode(USS_ECHO_PIN, INPUT);
         Serial.println("[OK] Ultrasonic sensor initialized");
     }
 
     // Connect to WiFi
     Serial.printf("[WIFI] Connecting to %s", WIFI_SSID);
     WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
     int attempts = 0;
     while (WiFi.status() != WL_CONNECTED && attempts < 30) {
         delay(500);
         Serial.print(".");
         attempts++;
     }
 
     if (WiFi.status() == WL_CONNECTED) {
         Serial.printf("\n[OK] Connected! IP: %s\n",
                       WiFi.localIP().toString().c_str());
     } else {
         Serial.println("\n[WARN] WiFi connection failed. Running offline.");
     }
 
     // HTTP endpoints
     server.on("/", handleRoot);
     server.on("/cmd", handleCommand);
     server.on("/status", handleStatus);
     server.on("/stream", handleStream);
     server.begin();
     Serial.println("[OK] HTTP server started on port 80");
 
     Serial.println("\n=== Ready ===\n");
 }
 
 void loop() {
     server.handleClient();
 
     // Safety: stop motors if no command received recently
     if (currentCommand != "stop" &&
         (millis() - lastCommandTime) > commandTimeout) {
         Serial.println("[SAFETY] Command timeout — stopping motors");
         executeCommand("stop");
     }
 
     // Collision avoidance: emergency stop if obstacle too close
     float distance = readUltrasonicCm();
     if (distance < USS_MIN_DISTANCE_CM && currentCommand == "forward") {
         Serial.printf("[SAFETY] Obstacle at %.1f cm — emergency stop!\n", distance);
         executeCommand("stop");
     }
 
     delay(10);  // small yield
 }