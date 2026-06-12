# MicroPython alternative for ESP32 (optional — primary firmware is vision_servo.ino)
from machine import Pin, PWM
from umqtt.simple import MQTTClient
import time
import ujson

MQTT_BROKER = "157.173.101.159"
CLIENT_ID = "esp32_dragonfly"
TOPIC_SUB = b"vision/dragonfly/movement"
TOPIC_PUB = b"vision/dragonfly/heartbeat"

# ESP32: GPIO 18 (or 14 if you wired like ESP8266 NodeMCU D5)
SERVO_PIN = 18
servo = PWM(Pin(SERVO_PIN), freq=50)

MIN_DUTY = 40
MAX_DUTY = 115
CENTER_DUTY = 77

current_duty = CENTER_DUTY


def set_servo(duty):
    global current_duty
    if duty < MIN_DUTY:
        duty = MIN_DUTY
    if duty > MAX_DUTY:
        duty = MAX_DUTY
    servo.duty(duty)
    current_duty = duty
    print("Servo Duty:", duty)


def sub_cb(topic, msg):
    global current_duty
    print((topic, msg))
    try:
        data = ujson.loads(msg)
        status = data.get("status", "")
        step = 5
        if status == "MOVE_LEFT":
            set_servo(current_duty + step)
        elif status == "MOVE_RIGHT":
            set_servo(current_duty - step)
        elif status == "CENTERED":
            pass
    except Exception as e:
        print("Error parsing JSON:", e)


def main():
    print("Starting ESP32 MQTT Client...")
    set_servo(CENTER_DUTY)
    try:
        client = MQTTClient(CLIENT_ID, MQTT_BROKER)
        client.set_callback(sub_cb)
        client.connect()
        print("Connected to MQTT Broker:", MQTT_BROKER)
        client.subscribe(TOPIC_SUB)
        last_heartbeat = 0
        while True:
            client.check_msg()
            now = time.time()
            if now - last_heartbeat > 10:
                payload = ujson.dumps({"node": "esp32", "status": "ONLINE", "uptime": now})
                client.publish(TOPIC_PUB, payload)
                last_heartbeat = now
            time.sleep(0.1)
    except Exception as e:
        print("Error:", e)
        time.sleep(5)
        import machine
        machine.reset()


if __name__ == "__main__":
    main()
