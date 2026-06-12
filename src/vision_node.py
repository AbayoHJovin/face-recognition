"""
vision_node.py
Simulated Vision Node for Distributed Vision-Control System.
Tracks face and publishes movement commands via MQTT.
Topic: vision/team313/movement
"""

import time
import argparse
import cv2
import json
import numpy as np
try:
    import paho.mqtt.client as mqtt
except Exception as e:
    mqtt = None
    _MQTT_IMPORT_ERROR = e
from pathlib import Path
import sys
import ssl
import os
import base64
from typing import Optional

# Add src to path if needed
sys.path.append(str(Path(__file__).parent.parent))

# Import Face Locking modules
from src.haar_5pt import Haar5ptDetector
from src.recognize import ArcFaceEmbedderONNX, FaceDBMatcher, load_db_npz
from src.face_locking import FaceLockSystem, LockState

DEFAULT_BROKER = os.environ.get("MQTT_HOST", "157.173.101.159")
DEFAULT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
TEAM_ID = "dragonfly"
TOPIC_MOVEMENT = f"vision/{TEAM_ID}/movement"
TOPIC_SNAPSHOT = f"vision/{TEAM_ID}/snapshot"
TOPIC_HEARTBEAT = f"vision/{TEAM_ID}/heartbeat"

# Webcam is fixed (doesn't rotate with the servo), so continuous MOVE_* commands
# never converge — the servo just keeps stepping. Once locked, hold the servo still.
HOLD_WHEN_LOCKED = True


def _looks_like_ip(host: str) -> bool:
    parts = host.split(".")
    if len(parts) != 4:
        return False
    try:
        return all(0 <= int(p) <= 255 for p in parts)
    except ValueError:
        return False


def _open_any_camera(indices=(0, 1, 2, 3)) -> cv2.VideoCapture:
    """Try several camera indices and return the first that opens and reads a frame."""
    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        ok, _ = cap.read()
        if ok:
            return cap
        cap.release()
    raise RuntimeError(
        f"Failed to open camera. Tried indices: {list(indices)}. "
        "Check that a webcam is connected and not in use by another app."
    )

class VisionNode:
    def __init__(self, broker, port, target_name, username=None, password=None, use_tls=None):
        if mqtt is None:
            raise RuntimeError(
                f"paho-mqtt import failed: {_MQTT_IMPORT_ERROR}\n"
                "Install dependencies with: pip install -r requirements.txt"
            )
        if use_tls is None:
            use_tls = port == 8883

        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"{TEAM_ID}_vision_node",
        )
        self.client.on_connect = self.on_connect
        if username:
            self.client.username_pw_set(username, password or "")
        if use_tls:
            self.client.tls_set(tls_version=ssl.PROTOCOL_TLS_CLIENT)
        try:
            print(f"Connecting to MQTT {broker}:{port} (TLS={use_tls})...")
            self.client.connect(broker, port, 60)
        except Exception as e:
            print(f"ERROR: MQTT connect failed: {e}")
            if use_tls and _looks_like_ip(broker):
                print(
                    "\nTLS certificates are not valid for IP addresses.\n"
                    f"  Use: python src/vision_node.py --broker {broker} --name Jovin\n"
                    "  (plain MQTT on port 1883)"
                )
            sys.exit(1)
        self.client.loop_start()
        
        # Face Recognition & Locking Setup
        print("Initializing Face Recognition...")
        self.det = Haar5ptDetector(min_size=(50, 50))
        self.embedder = ArcFaceEmbedderONNX(input_size=(112, 112))
        
        # Load Database
        db_path = Path(__file__).parent.parent / "data/db/face_db.npz"
        if not db_path.exists():
            print(f"ERROR: Face DB not found at {db_path}. Run enroll.py first!")
            sys.exit(1)
            
        db = load_db_npz(db_path)
        if target_name not in db:
            print(f"WARNING: Target '{target_name}' not in database. Available: {list(db.keys())}")
        
        self.matcher = FaceDBMatcher(db, dist_thresh=0.60)
        self.system = FaceLockSystem(target_name, self.matcher, self.det)
        
        self.running = True
        self.last_heartbeat = 0
        self.last_publish_time = 0
        self.mqtt_topic = TOPIC_MOVEMENT
        self.snapshot_sent = False  # Track if we've sent the face snapshot
        # Remember last non-NO_FACE status while locked so we can hold position
        self.last_status = "CENTERED"
        self.last_published_status: Optional[str] = None
        self._motor_centered = True  # hysteresis: stops jitter at deadband edge
        self.occlusion_grace_frames = 90  # match face_locking MAX_LOST_FRAMES

        self.system.log_event("SESSION_START", details="vision node online", dedupe_seconds=0)

    def _movement_from_cx(self, cx_norm: float) -> str:
        """Deadband with hysteresis — avoids stop→move→stop while face stays centered."""
        enter_left, enter_right = 0.35, 0.65
        exit_left, exit_right = 0.33, 0.67

        if self._motor_centered:
            if cx_norm < exit_left:
                self._motor_centered = False
                return "MOVE_LEFT"
            if cx_norm > exit_right:
                self._motor_centered = False
                return "MOVE_RIGHT"
            return "CENTERED"

        if cx_norm < enter_left:
            return "MOVE_LEFT"
        if cx_norm > enter_right:
            return "MOVE_RIGHT"
        self._motor_centered = True
        return "CENTERED"

    def on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code != 0:
            print(f"ERROR: MQTT broker rejected connection (rc={reason_code})")
            return
        print(f"Connected to MQTT Broker with result code {reason_code}")
        self.publish_heartbeat()

    def publish_movement(self, status, confidence=1.0, target=None, locked=False):
        """Small payload for ESP32 — never includes face images."""
        payload = {
            "status": status,
            "confidence": confidence,
            "target": target,
            "locked": locked,
            "timestamp": time.time(),
        }
        self.client.publish(self.mqtt_topic, json.dumps(payload))
        print(f"Published: {status}")

    def publish_snapshot(self, face_image, confidence=1.0, target=None):
        """Large snapshot on a separate topic so ESP32 buffer is not overwhelmed."""
        _, buffer = cv2.imencode(".jpg", face_image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        payload = {
            "target": target,
            "confidence": confidence,
            "timestamp": time.time(),
            "face_image": base64.b64encode(buffer).decode("utf-8"),
        }
        self.client.publish(TOPIC_SNAPSHOT, json.dumps(payload))
        print("Published: face snapshot")

    def publish_heartbeat(self):
        payload = {
            "node": "pc_vision",
            "status": "ONLINE",
            "timestamp": time.time()
        }
        self.client.publish(TOPIC_HEARTBEAT, json.dumps(payload))

    def run(self):
        # Try multiple indices so it works across different machines
        cap = _open_any_camera()
        
        print(f"Vision Node Started. Tracking target: {self.system.target_name}")
        print(f"Publishing to {TOPIC_MOVEMENT}")
        
        while self.running:
            ret, frame = cap.read()
            if not ret: break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            H, W = frame.shape[:2]
            
            # Process Frame using FaceLockSystem
            # process_frame returns (vis_frame, target_face_obj, lock_state)
            vis, target_face, lock_state, target_sim = self.system.process_frame(
                frame, self.embedder
            )

            status = "NO_FACE"
            face_crop = None
            confidence = target_sim if target_sim > 0 else 0.0

            if lock_state == LockState.SEARCHING:
                # Explicitly searching for the target -> tell ESP to sweep
                status = "NO_FACE"
                if self.snapshot_sent:
                    self.snapshot_sent = False
                    print("🔓 Target lost - snapshot flag reset")
            elif lock_state == LockState.LOCKED:
                if target_face:
                    f = target_face

                    if not self.snapshot_sent:
                        x1, y1, x2, y2 = int(f.x1), int(f.y1), int(f.x2), int(f.y2)
                        pad = 20
                        x1 = max(0, x1 - pad)
                        y1 = max(0, y1 - pad)
                        x2 = min(W, x2 + pad)
                        y2 = min(H, y2 + pad)
                        face_crop = frame[y1:y2, x1:x2]
                        self.snapshot_sent = True
                        print("📸 Face snapshot captured and will be sent")

                    cx_norm = (f.x1 + f.x2) / 2.0 / W
                    if HOLD_WHEN_LOCKED:
                        # Face captured -> stop the servo and hold position
                        status = "CENTERED"
                    else:
                        status = self._movement_from_cx(cx_norm)
                    self.last_status = status
                elif self.system.lost_frames <= self.occlusion_grace_frames:
                    # Detection blip — hold servo still, stay locked
                    status = "CENTERED"
                    confidence = (
                        self.system._last_track.similarity
                        if self.system._last_track
                        else confidence
                    )
                else:
                    status = "NO_FACE"
                    self.last_status = "NO_FACE"
                    self._motor_centered = True
            
            # --- RATE LIMITING (10Hz) ---
            current_time = time.time()
            if current_time - self.last_publish_time >= 0.05:
                is_locked = lock_state == LockState.LOCKED and status != "NO_FACE"
                self.publish_movement(
                    status,
                    confidence=confidence,
                    target=self.system.target_name,
                    locked=is_locked,
                )
                if face_crop is not None:
                    self.publish_snapshot(
                        face_crop,
                        confidence=confidence,
                        target=self.system.target_name,
                    )
                if status != self.last_published_status:
                    self.system.log_motor_command(status, confidence=confidence)
                    self.last_published_status = status
                self.last_publish_time = current_time
            
            # Heartbeat every 5s
            if time.time() - self.last_heartbeat > 5:
                self.publish_heartbeat()
                self.last_heartbeat = time.time()
            
            cv2.imshow("Vision Node (Locked)", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.client.loop_stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", type=str, default=DEFAULT_BROKER)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--username", type=str, default=None)
    parser.add_argument("--password", type=str, default=None)
    parser.add_argument("--tls", action="store_true", help="Enable TLS (port 8883)")
    parser.add_argument("--name", type=str, default="Jovin")
    args = parser.parse_args()

    use_tls = args.tls or args.port == 8883
    node = VisionNode(
        args.broker, args.port, args.name,
        username=args.username, password=args.password, use_tls=use_tls,
    )
    node.run()
