// DEPRECATED: This project now targets ESP32.
// Use: esp32/vision_servo/vision_servo.ino

#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <Servo.h>

// --- Configuration ---
const char* ssid = "EdNet";
const char* password = "Huawei@123";

const char* mqtt_server = "10.12.75.96"; 
const int mqtt_port = 1883;
const char* client_id = "esp8266_dragonfly";
const char* topic_movement = "vision/dragonfly/movement";
const char* topic_heartbeat = "vision/dragonfly/heartbeat";

// Servo Configuration
Servo myServo;
const int servoPin = D5; 

// Tuning constants
const int SERVO_MIN_ANGLE = 0;
const int SERVO_MAX_ANGLE = 180;
const int TRACK_STEP_DEG = 3;   // How much to move per MOVE_LEFT/RIGHT
const int SWEEP_STEP_DEG = 2;   // How fast to sweep while searching

int currentAngle = 90;   

// --- Search Mode Variables ---
bool isSearching = true;         // Start in search mode by default
unsigned long lastSweepTime = 0;
int sweepStep = SWEEP_STEP_DEG;       

// --- Watchdog Timer Variables ---
unsigned long lastFaceDetectTime = 0;
const unsigned long FACE_TIMEOUT = 2000; // 2 seconds without a face triggers a search

WiFiClient espClient;
PubSubClient client(espClient);

void setup_wifi() {
  delay(10);
  Serial.println("\nConnecting to WiFi...");
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
}

void moveServo(int delta) {
  currentAngle += delta;
  if (currentAngle < SERVO_MIN_ANGLE) currentAngle = SERVO_MIN_ANGLE;
  if (currentAngle > SERVO_MAX_ANGLE) currentAngle = SERVO_MAX_ANGLE;
  myServo.write(currentAngle);
}

void callback(char* topic, byte* payload, unsigned int length) {
  String message = "";
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  
  // Parse the commands and update the Watchdog Timer
  if (message.indexOf("MOVE_LEFT") >= 0) {
    isSearching = false; 
    lastFaceDetectTime = millis(); // Reset the timer!
    moveServo(-TRACK_STEP_DEG);       
  } 
  else if (message.indexOf("MOVE_RIGHT") >= 0) {
    isSearching = false; 
    lastFaceDetectTime = millis(); // Reset the timer!
    moveServo(TRACK_STEP_DEG);        
  } 
  else if (message.indexOf("CENTERED") >= 0) {
    isSearching = false; 
    lastFaceDetectTime = millis(); // Reset the timer!
  } 
  else if (message.indexOf("NO_FACE") >= 0) {
    isSearching = true;  // Explicit command to start searching
  }
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    if (client.connect(client_id)) {
      Serial.println("Connected!");
      client.subscribe(topic_movement); 
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" trying again in 5s");
      delay(5000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  myServo.attach(servoPin);
  myServo.write(currentAngle); 

  setup_wifi();
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  unsigned long now = millis();

  // --- WATCHDOG TIMER ---
  // If we aren't currently searching, but it's been more than 2 seconds 
  // since we last saw a face, force the system back into search mode.
  if (!isSearching && (now - lastFaceDetectTime > FACE_TIMEOUT)) {
    Serial.println("Face lost! Watchdog triggered. Starting search...");
    isSearching = true;
  }

  // --- NON-BLOCKING SEARCH SWEEP ---
  if (isSearching) {
    if (now - lastSweepTime > 50) { 
      lastSweepTime = now;
      currentAngle += sweepStep;

      if (currentAngle >= SERVO_MAX_ANGLE) {
        currentAngle = SERVO_MAX_ANGLE;
        sweepStep = -SWEEP_STEP_DEG; 
      } else if (currentAngle <= SERVO_MIN_ANGLE) {
        currentAngle = SERVO_MIN_ANGLE;
        sweepStep = SWEEP_STEP_DEG;  
      }
      myServo.write(currentAngle);
    }
  }

  // --- SYSTEM HEARTBEAT ---
  static unsigned long lastHeartbeat = 0;
  if (now - lastHeartbeat > 5000) {
    lastHeartbeat = now;
    String heartbeat = "{\"node\": \"esp8266\", \"status\": \"ONLINE\"}";
    client.publish(topic_heartbeat, heartbeat.c_str());
  }
}