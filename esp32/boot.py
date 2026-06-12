# boot.py — ESP32 MicroPython WiFi setup (optional)
import network
import time
from machine import Pin

# Onboard LED on many ESP32 dev boards is GPIO 2
led = Pin(2, Pin.OUT)
led.value(0)


def connect_wifi(ssid, password):
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        print("Connecting to network...")
        wlan.connect(ssid, password)
        while not wlan.isconnected():
            led.value(1)
            time.sleep(0.1)
            led.value(0)
            time.sleep(0.1)
    print("Network config:", wlan.ifconfig())
    led.value(1)


SSID = "EdNet"
PASSWORD = "Huawei@123"

try:
    connect_wifi(SSID, PASSWORD)
except Exception as e:
    print("WiFi Connection Failed:", e)
