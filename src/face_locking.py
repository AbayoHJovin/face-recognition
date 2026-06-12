"""
face_locking.py
"""
import json
import time
import argparse
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from enum import Enum
import mediapipe as mp

# Import existing modules
# We need to ensure we can import from . if run as a module or direct
try:
    from .haar_5pt import Haar5ptDetector, align_face_5pt, _bbox_from_5pt, _clip_box_xyxy
    from .recognize import ArcFaceEmbedderONNX, FaceDBMatcher, load_db_npz
except ImportError:
    # If run directly: python src/face_locking.py
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from src.haar_5pt import Haar5ptDetector, align_face_5pt, _bbox_from_5pt, _clip_box_xyxy
    from src.recognize import ArcFaceEmbedderONNX, FaceDBMatcher, load_db_npz

# ---------------------------------------------------------
# Action Logic
# ---------------------------------------------------------
@dataclass
class FaceAction:
    timestamp: float
    action_type: str
    details: str

class FaceActionDetector:
    def __init__(self):
        # MediaPipe Landmark Indices
        # Left Eye (for EAR)
        self.P_LEFT_EYE = [33, 160, 158, 133, 153, 144] 
        # Right Eye (for EAR)
        self.P_RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        # Mouth (for SMILE/MAR) - 61=left corner, 291=right corner, 0=upper lip, 17=lower lip
        self.P_MOUTH = [61, 291, 0, 17]
        # Nose for pose
        self.P_NOSE_TIP = 1
        
        # Thresholds
        self.EAR_THRESH = 0.22  # Below this -> closed
        self.MAR_THRESH = 0.45  # Above this -> smile/open (simplified smile detection)
        # Smile can also be detected by mouth corner width relative to face width

        self.last_blink_time = 0.0
        self.blink_cooldown = 0.3
        
        self.last_nose_x = None

    def _ear(self, lm, idxs):
        # eye aspect ratio
        # vertical dists
        v1 = np.linalg.norm(lm[idxs[1]] - lm[idxs[5]])
        v2 = np.linalg.norm(lm[idxs[2]] - lm[idxs[4]])
        # horizontal
        h = np.linalg.norm(lm[idxs[0]] - lm[idxs[3]])
        return (v1 + v2) / (2.0 * h + 1e-6)

    def detect(self, mp_landmarks, frame_w, frame_h) -> List[Tuple[str, str]]:
        """
        Input: mp_landmarks (list of normalized x,y,z) from MediaPipe
        Returns: list of (ActionType, Description)
        """
        actions = []
        now = time.time()
        
        # Convert necessary landmarks to np arrays for calculation
        coords = np.array([[p.x, p.y] for p in mp_landmarks])
        
        # 1. Blink Detection
        left_ear = self._ear(coords, self.P_LEFT_EYE)
        right_ear = self._ear(coords, self.P_RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0
        
        if avg_ear < self.EAR_THRESH:
            if (now - self.last_blink_time) > self.blink_cooldown:
                actions.append(("BLINK", f"EAR={avg_ear:.2f}"))
                self.last_blink_time = now

        # 2. Smile Detection (Simple width checks or mouth alignment)
        # Check if mouth corners are 'wide' or mouth is open
        # Better simple smile: check if corners (61, 291) are higher than usual relative to upper lip (0)?
        # Or just use mouth width / jaw width ratio?
        # Let's use simple aspect ratio of mouth for "laugh/smile" (open mouth)
        # and maybe specific corner comparison for closed smile.
        # Simplest: Mouth width (61-291) vs Face Width (234-454 for cheeks)
        left_cheek = coords[234]
        right_cheek = coords[454]
        face_width = np.linalg.norm(right_cheek - left_cheek)
        
        mouth_l = coords[61]
        mouth_r = coords[291]
        mouth_width = np.linalg.norm(mouth_r - mouth_l)
        
        ratio = mouth_width / (face_width + 1e-6)
        if ratio > 0.45: # Tweak this
             actions.append(("SMILE", f"ratio={ratio:.2f}"))

        # 3. Head Movement (Left/Right)
        # Check nose x relative to frame center (0.5 in normalized coords)
        nose = coords[self.P_NOSE_TIP]
        if nose[0] < 0.50:
             actions.append(("HEAD_TURN_LEFT", f"nose_x={nose[0]:.2f}"))
        elif nose[0] > 0.60:
             actions.append(("HEAD_TURN_RIGHT", f"nose_x={nose[0]:.2f}"))

        return actions

# ---------------------------------------------------------
# Face Locking System
# ---------------------------------------------------------
class LockState(Enum):
    SEARCHING = 0
    LOCKED = 1
    # Could add LOST_RECOVERING state if we want hysteresis

# Frame-position bands (speaker moving across the camera view)
FRAME_LEFT = 0.40
FRAME_RIGHT = 0.60


def _bbox_iou(a, b) -> float:
    """Intersection-over-union for two xyxy face boxes."""
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(1, (a.x2 - a.x1) * (a.y2 - a.y1))
    area_b = max(1, (b.x2 - b.x1) * (b.y2 - b.y1))
    return inter / float(area_a + area_b - inter)


@dataclass
class _TrackedBBox:
    x1: float
    y1: float
    x2: float
    y2: float
    cx_norm: float
    similarity: float


class FaceLockSystem:
    def __init__(self, target_name: str, matcher: FaceDBMatcher, detector: Haar5ptDetector):
        self.target_name = target_name
        self.matcher = matcher
        self.det = detector
        self.state = LockState.SEARCHING

        self.action_det = FaceActionDetector()
        self.history: List[FaceAction] = []

        self.locked_frames = 0
        self.lost_frames = 0
        self.MAX_LOST_FRAMES = 90  # ~3 s at 30 fps before unlock

        self._last_track: Optional[_TrackedBBox] = None
        self.tracking_cx_norm: Optional[float] = None
        self.is_tracking_hold = False

        self._last_frame_pos: Optional[str] = None
        self._out_of_frame_logged = False
        self._last_motor_command: Optional[str] = None

        ts = time.strftime("%Y%m%d%H%M%S")
        safe_name = "".join(c for c in target_name if c.isalnum())
        log_dir = Path(__file__).resolve().parent.parent / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = log_dir / f"{safe_name}_history_{ts}.txt"
        self.evidence_file = log_dir / f"{safe_name}_evidence_{ts}.jsonl"
        self._log_write_failed = False

        print(f"[FaceLock] Initialized. Target: {target_name}")
        print(f"[FaceLock] History: {self.history_file}")
        print(f"[FaceLock] Evidence: {self.evidence_file}")

    def _iso_timestamp(self, ts: Optional[float] = None) -> str:
        ts = time.time() if ts is None else ts
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

    def _append_log_file(self, path: Path, line: str) -> None:
        """Append one line to a log file; never crash the vision loop on I/O errors."""
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError as e:
            if not self._log_write_failed:
                self._log_write_failed = True
                print(
                    f"[FaceLock] WARNING: cannot write to {path.name} ({e}). "
                    "Close the file if it is open in another app. Logging to console only."
                )

    def log_event(
        self,
        event_type: str,
        details: str = "",
        confidence: Optional[float] = None,
        motor_command: Optional[str] = None,
        dedupe_seconds: float = 1.0,
    ) -> None:
        """Write a timestamped history line and structured evidence record."""
        now = time.time()

        if dedupe_seconds > 0 and self.history:
            last = self.history[-1]
            if (
                last.action_type == event_type
                and (now - last.timestamp) < dedupe_seconds
                and motor_command is None
            ):
                return

        conf_str = f"{confidence:.2f}" if confidence is not None else "n/a"
        motor_str = motor_command or "n/a"

        act = FaceAction(timestamp=now, action_type=event_type, details=details)
        self.history.append(act)

        history_line = (
            f"{self._iso_timestamp(now)} | {event_type} | "
            f"speaker={self.target_name} | confidence={conf_str} | "
            f"motor={motor_str} | {details}\n"
        )
        self._append_log_file(self.history_file, history_line)

        evidence = {
            "timestamp": self._iso_timestamp(now),
            "unix_ts": now,
            "event": event_type,
            "speaker_id": self.target_name,
            "confidence": confidence,
            "motor_command": motor_command,
            "details": details,
        }
        self._append_log_file(self.evidence_file, json.dumps(evidence) + "\n")

        print(f">> {event_type} | speaker={self.target_name} | conf={conf_str} | {details}")

    def log_action(self, atype: str, details: str, confidence: Optional[float] = None):
        """Backward-compatible wrapper for action logging."""
        self.log_event(atype, details=details, confidence=confidence)

    def log_motor_command(self, command: str, confidence: Optional[float] = None):
        """Log a published motor/MQTT movement command (vision node)."""
        if command == self._last_motor_command:
            return
        self._last_motor_command = command
        self.log_event(
            "MOTOR_COMMAND",
            details=f"published={command}",
            confidence=confidence,
            motor_command=command,
            dedupe_seconds=0,
        )

    def _frame_position(self, cx_norm: float) -> str:
        if cx_norm < FRAME_LEFT:
            return "LEFT"
        if cx_norm > FRAME_RIGHT:
            return "RIGHT"
        return "CENTER"

    def _log_frame_movement(self, cx_norm: float, confidence: float) -> None:
        pos = self._frame_position(cx_norm)
        if pos == self._last_frame_pos:
            return

        if pos == "LEFT":
            event = "MOVED_LEFT"
        elif pos == "RIGHT":
            event = "MOVED_RIGHT"
        else:
            event = "STOPPED"

        self.log_event(
            event,
            details=f"cx_norm={cx_norm:.2f}",
            confidence=confidence,
            dedupe_seconds=0,
        )
        self._last_frame_pos = pos

    def _remember_target(self, f, target_sim: float, frame_w: int) -> None:
        cx_norm = ((f.x1 + f.x2) / 2.0) / frame_w
        self._last_track = _TrackedBBox(
            x1=f.x1, y1=f.y1, x2=f.x2, y2=f.y2,
            cx_norm=cx_norm, similarity=target_sim,
        )
        self.tracking_cx_norm = cx_norm
        self.is_tracking_hold = False

    def _spatial_reacquire(self, faces, frame_w: int):
        """When locked, re-use a face in the same region if recognition flickers."""
        if not self._last_track or not faces:
            return None, 0.0

        best_f = None
        best_iou = 0.0
        for f in faces:
            iou = _bbox_iou(f, self._last_track)
            if iou > best_iou:
                best_iou = iou
                best_f = f

        if best_f is None or best_iou < 0.20:
            return None, 0.0

        cx_norm = ((best_f.x1 + best_f.x2) / 2.0) / frame_w
        self._last_track = _TrackedBBox(
            x1=best_f.x1, y1=best_f.y1, x2=best_f.x2, y2=best_f.y2,
            cx_norm=cx_norm, similarity=self._last_track.similarity,
        )
        self.tracking_cx_norm = cx_norm
        return best_f, self._last_track.similarity

    def process_frame(
        self, frame: np.ndarray, embedder: ArcFaceEmbedderONNX
    ) -> Tuple[np.ndarray, Optional[object], LockState, float]:
        vis = frame.copy()
        H, W = vis.shape[:2]

        faces, mp_res = self.det.detect_with_mesh(frame, max_faces=5)

        # 1. Process all faces to find matches
        # We want to identify everyone, but only "lock" on the target.
        target_face = None
        target_sim = 0.0

        for f in faces:
            aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
            emb = embedder.embed(aligned)
            mr = self.matcher.match(emb)

            if mr.accepted:
                is_target = (mr.name == self.target_name)

                if is_target:
                    if mr.similarity > target_sim:
                        target_sim = mr.similarity
                        target_face = f
                else:
                    cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (255, 200, 0), 2)
                    cv2.putText(
                        vis,
                        f"{mr.name} ({mr.similarity:.0%})",
                        (f.x1, f.y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 200, 0),
                        2,
                    )
            else:
                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 0, 255), 2)
                cv2.putText(
                    vis,
                    f"Unknown ({mr.similarity:.0%})",
                    (f.x1, f.y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

        # Spatial re-acquire while locked (recognition flicker, face still visible)
        if self.state == LockState.LOCKED and target_face is None and faces:
            target_face, target_sim = self._spatial_reacquire(faces, W)

        # 2. State Machine Logic for Target
        # Handle state transitions based on whether target was found this frame
        if self.state == LockState.SEARCHING:
            cv2.putText(
                vis,
                f"SEARCHING: {self.target_name}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2
            )

            if target_face is not None:
                self.state = LockState.LOCKED
                self.lost_frames = 0
                self._out_of_frame_logged = False
                self.log_event(
                    "LOCK_ACQUIRED",
                    details="target entered frame",
                    confidence=target_sim,
                    dedupe_seconds=0,
                )

        if self.state == LockState.LOCKED:
            if target_face is not None:
                self.lost_frames = 0
                self.is_tracking_hold = False

                if self._out_of_frame_logged:
                    self.log_event(
                        "BACK_IN_FRAME",
                        details="target visible again after occlusion",
                        confidence=target_sim,
                        dedupe_seconds=0,
                    )
                    self._out_of_frame_logged = False

                f = target_face
                cx_norm = ((f.x1 + f.x2) / 2.0) / W
                if not self.is_tracking_hold:
                    self._remember_target(f, target_sim, W)

                hold_label = " (hold)" if self.is_tracking_hold else ""
                box_color = (0, 200, 255) if self.is_tracking_hold else (0, 255, 0)

                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), box_color, 3)
                cv2.putText(
                    vis,
                    f"TARGET: {self.target_name}{hold_label}",
                    (f.x1, max(0, f.y1 - 34)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    vis,
                    f"confidence: {target_sim:.0%} ({target_sim:.2f})",
                    (f.x1, max(0, f.y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

                self._log_frame_movement(cx_norm, target_sim)

                cv2.putText(
                    vis,
                    f"LOCKED: {self.target_name} | conf={target_sim:.0%}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    box_color,
                    2,
                )
                
                # Action Detection on Target (skip during spatial hold)
                if not self.is_tracking_hold and mp_res and mp_res.multi_face_landmarks:
                    fw_x, fw_y = (f.x1 + f.x2) / 2, (f.y1 + f.y2) / 2
                    best_lm = None
                    min_dist = float("inf")

                    for lm_list in mp_res.multi_face_landmarks:
                        nose = lm_list.landmark[1]
                        nx, ny = nose.x * W, nose.y * H
                        dist = ((nx - fw_x) ** 2 + (ny - fw_y) ** 2) ** 0.5

                        if dist < min_dist:
                            min_dist = dist
                            best_lm = lm_list.landmark

                    if best_lm and min_dist < max(f.x2 - f.x1, f.y2 - f.y1):
                        actions = self.action_det.detect(best_lm, W, H)
                        for atype, desc in actions:
                            self.log_event(atype, details=desc, confidence=target_sim)
                            cv2.putText(
                                vis,
                                f"ACT: {atype}",
                                (10, H - 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 255),
                                2,
                            )
            else:
                self.lost_frames += 1
                self.is_tracking_hold = True
                if self._last_track is not None:
                    self.tracking_cx_norm = self._last_track.cx_norm

                cv2.putText(
                    vis,
                    f"LOCKED: {self.target_name} | OCCLUDED",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    vis,
                    f"OUT OF FRAME ({self.lost_frames}/{self.MAX_LOST_FRAMES})",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

                if self.lost_frames == 1 and not self._out_of_frame_logged:
                    self.log_event(
                        "OUT_OF_FRAME",
                        details=f"lost_frames={self.lost_frames}",
                        dedupe_seconds=0,
                    )
                    self._out_of_frame_logged = True
                    self._last_frame_pos = None

                if self.lost_frames > self.MAX_LOST_FRAMES:
                    self.state = LockState.SEARCHING
                    self._last_frame_pos = None
                    self._out_of_frame_logged = False
                    self._last_track = None
                    self.tracking_cx_norm = None
                    self.is_tracking_hold = False
                    self.log_event(
                        "LOCK_LOST",
                        details="target left frame beyond tolerance",
                        dedupe_seconds=0,
                    )

        return vis, target_face, self.state, target_sim


def _open_any_camera(indices=(0, 1, 2, 3)) -> cv2.VideoCapture:
    """Try several camera indices; return the first that opens and reads a frame."""
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


def main():
    cfg = argparse.ArgumentParser()
    cfg.add_argument("--name", type=str, default="Jovin", help="Target identity to lock onto")
    args = cfg.parse_args()
    
    # Init
    db_path = Path("data/db/face_db.npz")
    if not db_path.exists():
        print("No database found! Please run enroll.py first.")
        return

    det = Haar5ptDetector(min_size=(70, 70), debug=False)
    embedder = ArcFaceEmbedderONNX(input_size=(112, 112))
    
    db = load_db_npz(db_path)
    if args.name not in db:
        print(f"Warning: '{args.name}' not in database. Available: {list(db.keys())}")
        # Proceed anyway? No, impossible to lock.
        # But let's allow it to start scanning so user can see failures.
    
    matcher = FaceDBMatcher(db, dist_thresh=0.60)
    
    system = FaceLockSystem(args.name, matcher, det)

    try:
        cap = _open_any_camera()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return

    print("Face Locking System Started. Press 'q' to quit.")
    system.log_event("SESSION_START", details="camera opened", dedupe_seconds=0)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERROR: Camera stopped delivering frames.")
            break

        frame = cv2.flip(frame, 1)

        vis, _, _, _ = system.process_frame(frame, embedder)

        cv2.imshow("Face Locking", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
