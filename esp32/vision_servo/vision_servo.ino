/*
 * vision_servo.ino — ESP32 face-tracking servo controller
 *
 * Sweep only on explicit NO_FACE from vision node — not on MQTT silence.
 * Power: separate 5V for servo; common GND with ESP32.
 */

#include <WiFi.h>
#include <PubSubClient.h>
#include <ESP32Servo.h>

const char* ssid = "EdNet";
const char* password = "Huawei@123";

const char* mqtt_server = "157.173.101.159";
const int mqtt_port = 1883;
const char* topic_movement = "vision/dragonfly/movement";
const char* topic_heartbeat = "vision/dragonfly/heartbeat";

char client_id[50];

const int servoPin = 18;
Servo myServo;

const int SERVO_MIN_ANGLE = 0;
const int SERVO_MAX_ANGLE = 180;
const int TRACK_STEP_DEG = 1;
const int SWEEP_STEP_DEG = 1;
const int MAX_PENDING_TRACK = 3;

int currentAngle = 90;
int pendingTrack = 0;
bool isSearching = true;

unsigned long lastSweepTime = 0;
unsigned long lastTrackTime = 0;
int sweepStep = SWEEP_STEP_DEG;

unsigned long lastReconnectAttempt = 0;
unsigned long lastWifiWaitLog = 0;
const unsigned long RECONNECT_INTERVAL_MS = 5000;
const unsigned long SWEEP_INTERVAL_MS = 55;
const unsigned long TRACK_INTERVAL_MS = 30;

WiFiClient espClient;
PubSubClient client(espClient);

void setup_wifi() {
  Serial.println("\nConnecting to WiFi...");
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.setSleep(false);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.println(WiFi.localIP());
}

void moveServo(int delta) {
  currentAngle += delta;
  if (currentAngle < SERVO_MIN_ANGLE) currentAngle = SERVO_MIN_ANGLE;
  if (currentAngle > SERVO_MAX_ANGLE) currentAngle = SERVO_MAX_ANGLE;
  myServo.write(currentAngle);
}

void callback(char* topic, byte* payload, unsigned int length) {
  if (length > 256) return;

  char message[257];
  unsigned int copyLen = length < 256 ? length : 256;
  for (unsigned int i = 0; i < copyLen; i++) {
    message[i] = (char)payload[i];
  }
  message[copyLen] = '\0';

  // Only explicit NO_FACE starts sweep — CENTERED/MOVE_* keep tracking
  if (strstr(message, "NO_FACE") != NULL) {
    isSearching = true;
    pendingTrack = 0;
    return;
  }

  isSearching = false;

  if (strstr(message, "MOVE_LEFT") != NULL) {
    pendingTrack = max(pendingTrack - 1, -MAX_PENDING_TRACK);
  } else if (strstr(message, "MOVE_RIGHT") != NULL) {
    pendingTrack = min(pendingTrack + 1, MAX_PENDING_TRACK);
  } else if (strstr(message, "CENTERED") != NULL) {
    pendingTrack = 0;
  }
}

bool mqtt_connect() {
  Serial.print("Attempting MQTT connection...");
  if (client.connect(client_id)) {
    Serial.println("Connected!");
    client.subscribe(topic_movement);
    return true;
  }
  Serial.print(" failed, rc=");
  Serial.println(client.state());
  return false;
}

void setup() {
  Serial.begin(115200);
  delay(500);

  uint64_t chipid = ESP.getEfuseMac();
  snprintf(client_id, sizeof(client_id), "esp32_dragonfly_%08X%08X",
           (uint32_t)(chipid >> 32), (uint32_t)chipid);

  myServo.attach(servoPin, 500, 2400);
  myServo.write(currentAngle);

  setup_wifi();

  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
  client.setBufferSize(512);
  client.setKeepAlive(30);
  client.setSocketTimeout(15);

  mqtt_connect();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    if (millis() - lastWifiWaitLog >= 10000) {
      lastWifiWaitLog = millis();
      Serial.println("WiFi disconnected — auto-reconnecting...");
    }
    delay(100);
    return;
  }

  if (!client.connected()) {
    if (millis() - lastReconnectAttempt >= RECONNECT_INTERVAL_MS) {
      lastReconnectAttempt = millis();
      mqtt_connect();
    }
  } else {
    client.loop();
  }

  unsigned long now = millis();

  if (isSearching) {
    if (now - lastSweepTime >= SWEEP_INTERVAL_MS) {
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
  } else if (pendingTrack != 0 && now - lastTrackTime >= TRACK_INTERVAL_MS) {
    lastTrackTime = now;
    int step = (pendingTrack > 0) ? 1 : -1;
    pendingTrack -= step;
    moveServo(step * TRACK_STEP_DEG);
  }

  static unsigned long lastHeartbeat = 0;
  if (client.connected() && now - lastHeartbeat >= 5000) {
    lastHeartbeat = now;
    client.publish(topic_heartbeat, "{\"node\":\"esp32\",\"status\":\"ONLINE\"}");
  }
}
