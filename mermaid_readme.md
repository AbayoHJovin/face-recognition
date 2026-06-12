# Face-Locking System — Mermaid Diagrams

Architecture and data-flow diagrams for the distributed face recognition + servo tracking system.

**Broker:** HiveMQ Cloud (`olivequeen-db8548cb.a03.euc1.aws.hivemq.cloud:8883`, TLS)  
**Team ID:** `dragonfly`  
**MQTT prefix:** `vision/dragonfly/`

---

## 1. System Architecture (High Level)

```mermaid
flowchart TB
    subgraph PC["PC (Developer Machine)"]
        CAM[Webcam]
        VN["Vision Node<br/>src/vision_node.py"]
        FL["Face Lock System<br/>src/face_locking.py"]
        DET["Haar + 5pt Detector<br/>src/haar_5pt.py"]
        ARC["ArcFace Embedder<br/>models/embedder_arcface.onnx"]
        DB[("Face DB<br/>data/db/face_db.npz")]
        BE["Backend Relay<br/>backend/src/server.js"]
        DASH["Web Dashboard<br/>dashboard/index.html"]
    end

    subgraph CLOUD["HiveMQ Cloud (MQTT Broker)"]
        MQTT["TLS :8883<br/>Pub/Sub Hub"]
    end

    subgraph EDGE["ESP32 Edge Device"]
        FW["vision_servo.ino"]
        SERVO["Servo Motor<br/>GPIO 18"]
    end

    CAM --> VN
    VN --> FL
    FL --> DET
    FL --> ARC
    ARC --> DB
    DET --> FL

    VN -->|"publish movement,<br/>snapshot, heartbeat"| MQTT
    BE -->|"subscribe movement,<br/>snapshot"| MQTT
    BE -->|"WebSocket :9002"| DASH
    BE -->|"HTTP :8080"| DASH

    MQTT -->|"subscribe movement"| FW
    FW -->|"publish heartbeat"| MQTT
    FW --> SERVO

    style CLOUD fill:#1a1a2e,stroke:#00d4ff,color:#fff
    style PC fill:#13131f,stroke:#00ff9d,color:#fff
    style EDGE fill:#13131f,stroke:#ff0055,color:#fff
```

---

## 2. Network & Deployment Topology

```mermaid
flowchart LR
    subgraph WiFi["Local WiFi (e.g. EdNet)"]
        PC["PC<br/>10.x.x.x"]
        ESP["ESP32<br/>10.x.x.x"]
    end

    subgraph Internet["Internet"]
        HIVE["HiveMQ Cloud<br/>*.aws.hivemq.cloud<br/>Port 8883 TLS"]
    end

    PC -->|"mqtts:// + TLS"| HIVE
    ESP -->|"WiFiClientSecure<br/>mqtts:// + TLS"| HIVE

    PC -->|"ws://localhost:9002"| PC
    PC -->|"http://localhost:8080"| PC

    note1["PC and ESP32 never need<br/>direct LAN connection.<br/>All coordination via cloud broker."]
```

---

## 3. End-to-End Message Flow (Sequence)

```mermaid
sequenceDiagram
    autonumber
    participant Cam as Webcam
    participant VN as Vision Node (PC)
    participant MQ as HiveMQ Cloud
    participant ESP as ESP32
    participant BE as Backend
    participant UI as Dashboard

    Note over VN,MQ: Startup — all clients connect with TLS + credentials

    VN->>MQ: CONNECT (TLS :8883, user: jovin)
    ESP->>MQ: CONNECT (TLS :8883)
    BE->>MQ: CONNECT (mqtts://)
    BE->>UI: WebSocket OPEN (port 9002)
    UI->>UI: Show CONNECTED

    loop Every frame (~30 fps)
        Cam->>VN: Video frame
        VN->>VN: Detect faces → ArcFace match → Lock state
    end

    loop Every 100 ms (10 Hz)
        VN->>MQ: PUBLISH vision/dragonfly/movement<br/>{status, target, locked, confidence}
        MQ->>ESP: DELIVER movement (small JSON)
        MQ->>BE: DELIVER movement
        BE->>UI: WebSocket broadcast
        ESP->>ESP: MOVE_LEFT / MOVE_RIGHT / CENTERED / sweep
    end

    opt First lock acquired
        VN->>MQ: PUBLISH vision/dragonfly/snapshot<br/>{face_image base64}
        MQ->>BE: DELIVER snapshot
        BE->>UI: WebSocket broadcast (face photo)
    end

    loop Every 5 s
        VN->>MQ: PUBLISH vision/dragonfly/heartbeat
        ESP->>MQ: PUBLISH vision/dragonfly/heartbeat
    end
```

---

## 4. Vision Node Processing Pipeline

```mermaid
flowchart TD
    START([Camera frame]) --> FLIP[Mirror flip]
    FLIP --> DETECT[Haar detect + 5pt landmarks]
    DETECT --> EMBED[ArcFace embed each face]
    EMBED --> MATCH{Match enrolled<br/>identity?}
    MATCH -->|No| UNKNOWN[Draw unknown / other labels]
    MATCH -->|Yes, target| LOCK{Lock state?}

    LOCK -->|SEARCHING| FOUND{Target in frame?}
    FOUND -->|Yes| ACQUIRE[LOCK_ACQUIRED]
    ACQUIRE --> LOCKED[State: LOCKED]

    FOUND -->|No| PUB_NO[Publish NO_FACE]

    LOCK -->|LOCKED| VISIBLE{Target visible<br/>this frame?}
    VISIBLE -->|Yes| POS[Compute cx_norm<br/>face center in frame]
    POS --> BAND{cx_norm position}
    BAND -->|< 0.35| ML[MOVE_LEFT]
    BAND -->|> 0.65| MR[MOVE_RIGHT]
    BAND -->|else| CEN[CENTERED]
    ML --> PUB_MOVE[Publish movement @ 10 Hz]
    MR --> PUB_MOVE
    CEN --> PUB_MOVE

    VISIBLE -->|No, brief| HOLD[Hold last command<br/>occlusion grace ~20 frames]
    HOLD --> PUB_MOVE
    VISIBLE -->|No, prolonged| LOST[lost_frames > 30]
    LOST --> SEARCH[State: SEARCHING]
    SEARCH --> PUB_NO

    PUB_MOVE --> MQTT_OUT[(HiveMQ)]
    PUB_NO --> MQTT_OUT
    UNKNOWN --> DISPLAY[Show annotated frame]
    DISPLAY --> END([Next frame])
```

---

## 5. Face Locking State Machine

```mermaid
stateDiagram-v2
    [*] --> SEARCHING

    SEARCHING --> LOCKED : Target face recognized<br/>(LOCK_ACQUIRED)
    LOCKED --> SEARCHING : Target lost > 30 frames<br/>(LOCK_LOST)

    state SEARCHING {
        [*] --> Scanning
        Scanning --> Scanning : NO_FACE published<br/>ESP sweeps 0°–180°
    }

    state LOCKED {
        [*] --> Tracking
        Tracking --> Occluded : 1+ frames without detection
        Occluded --> Tracking : BACK_IN_FRAME
        Tracking --> Tracking : MOVE_LEFT / MOVE_RIGHT / CENTERED
        Occluded --> Tracking : Hold last motor cmd<br/>(grace period)
    }

    note right of LOCKED
        Action detection (optional):
        BLINK, SMILE, HEAD_TURN
        Logged to data/logs/
    end note
```

---

## 6. MQTT Topics & Publishers

```mermaid
flowchart LR
    subgraph Publishers
        VN1["Vision Node"]
        ESP1["ESP32"]
    end

    subgraph Topics["HiveMQ Topics"]
        T1["vision/dragonfly/movement"]
        T2["vision/dragonfly/snapshot"]
        T3["vision/dragonfly/heartbeat"]
    end

    subgraph Subscribers
        ESP2["ESP32"]
        BE["Backend"]
    end

    VN1 -->|publish ~10 Hz| T1
    VN1 -->|publish once per lock| T2
    VN1 -->|publish every 5 s| T3
    ESP1 -->|publish every 5 s| T3

    T1 --> ESP2
    T1 --> BE
    T2 --> BE
    T3 --> BE

    style T1 fill:#2d5016,stroke:#00ff9d,color:#fff
    style T2 fill:#503316,stroke:#ffaa00,color:#fff
    style T3 fill:#1a3352,stroke:#00d4ff,color:#fff
```

### Movement payload (small — ESP32 safe)

```json
{
  "status": "MOVE_LEFT | MOVE_RIGHT | CENTERED | NO_FACE",
  "confidence": 0.85,
  "target": "Jovin",
  "locked": true,
  "timestamp": 1781270502.65
}
```

### Snapshot payload (large — dashboard only)

```json
{
  "target": "Jovin",
  "confidence": 0.85,
  "timestamp": 1781270502.65,
  "face_image": "<base64 JPEG>"
}
```

---

## 7. ESP32 Servo Control Logic

```mermaid
flowchart TD
    MSG[MQTT message received<br/>vision/dragonfly/movement] --> PARSE{Parse status}

    PARSE -->|NO_FACE| SEARCH[isSearching = true<br/>pendingTrack = 0]
    PARSE -->|MOVE_LEFT| TRACK_L[pendingTrack -= 1]
    PARSE -->|MOVE_RIGHT| TRACK_R[pendingTrack += 1]
    PARSE -->|CENTERED| HOLD[pendingTrack = 0]

    SEARCH --> SWEEP[Sweep servo 0° ↔ 180°<br/>1° every 55 ms]

    TRACK_L --> SMOOTH[Apply 1–2° steps<br/>every 45 ms]
    TRACK_R --> SMOOTH
    HOLD --> IDLE[Hold current angle]

    SMOOTH --> SERVO[(Servo GPIO 18)]

    WATCH{No face cmd<br/>for 5 s?} -->|Yes| SEARCH
    TRACK_L --> WATCH
    TRACK_R --> WATCH
    HOLD --> WATCH
```

---

## 8. Dashboard Relay Path

```mermaid
flowchart LR
    MQ[(HiveMQ Cloud)]

    subgraph Backend["backend/src/server.js"]
        MQTTC[MQTT Client<br/>mqtts://]
        WSS[WebSocket Server<br/>:9002]
        HTTP[HTTP Server<br/>:8080]
    end

    subgraph Browser
        HTML[dashboard/index.html]
        WS[WebSocket client]
        UI[Face picture<br/>Tracking status<br/>Event log]
    end

    MQ -->|movement + snapshot| MQTTC
    MQTTC -->|broadcast JSON string| WSS
    HTTP -->|serve index.html| HTML
    HTML --> WS
    WSS --> WS
    WS --> UI

    note["Dashboard must use same broker as vision node.<br/>Open http://localhost:8080 on the PC running backend."]
```

---

## 9. Component File Map

```mermaid
flowchart TB
    subgraph src["src/"]
        vision_node["vision_node.py<br/>Camera + MQTT publisher"]
        face_locking["face_locking.py<br/>Lock FSM + logging"]
        enroll["enroll.py<br/>Build face_db.npz"]
        haar["haar_5pt.py"]
        recognize["recognize.py"]
    end

    subgraph backend["backend/"]
        server["src/server.js"]
    end

    subgraph esp32["esp32/vision_servo/"]
        ino["vision_servo.ino"]
    end

    subgraph data["data/"]
        npz["db/face_db.npz"]
        logs["logs/*.txt / *.jsonl"]
    end

    enroll --> npz
    vision_node --> face_locking
    face_locking --> haar
    face_locking --> recognize
    recognize --> npz
    vision_node --> logs
    server --> dashboard["dashboard/index.html"]
    ino --> servo["Physical servo"]
```

---

## 10. Typical Run Order

```mermaid
flowchart TD
    A[1. Enroll face<br/>python -m src.enroll --name Jovin] --> B[2. Flash ESP32<br/>vision_servo.ino]
    B --> C[3. Start backend<br/>cd backend && pnpm start]
    C --> D[4. Open dashboard<br/>http://localhost:8080]
    D --> E[5. Start vision node<br/>python src/vision_node.py --name Jovin]
    E --> F{All connected?}
    F -->|ESP32 Serial: Connected!| G[6. Stand in front of camera]
    F -->|Dashboard: MQTT LINKED| G
    G --> H[System tracks face<br/>Servo follows<br/>Dashboard updates]
```

---

## Viewing these diagrams

- **GitHub / GitLab** — renders Mermaid in `.md` files automatically.
- **VS Code / Cursor** — install a Mermaid preview extension, or use Markdown preview.
- **Online** — paste any block into [https://mermaid.live](https://mermaid.live).
