const mqtt = require('mqtt');
const WebSocket = require('ws');
const http = require('http');
const fs = require('fs');
const path = require('path');

// Local Mosquitto on this PC. ESP32 uses PC WiFi IP in the sketch.
const MQTT_BROKER = process.env.MQTT_BROKER || 'mqtt://127.0.0.1';
const TARGET_NAME = (process.env.TARGET_NAME || 'Jovin').toLowerCase();
const TEAM_ID = 'dragonfly';
const MQTT_TOPIC_VS = `vision/${TEAM_ID}/movement`;
const MQTT_TOPIC_SNAPSHOT = `vision/${TEAM_ID}/snapshot`;
const WS_PORT = 9002;
const HTTP_PORT = 8080;

let mqttConnected = false;
let lastMqttMessageAt = 0;

// --- MQTT Client ---
console.log(`Connecting to MQTT Broker: ${MQTT_BROKER}...`);
const mqttClient = mqtt.connect(MQTT_BROKER, {
    reconnectPeriod: 3000,
    connectTimeout: 10000,
});

mqttClient.on('connect', () => {
    mqttConnected = true;
    console.log('Connected to MQTT Broker.');
    mqttClient.subscribe([MQTT_TOPIC_VS, MQTT_TOPIC_SNAPSHOT], (err) => {
        if (!err) {
            console.log(`Subscribed to: ${MQTT_TOPIC_VS}, ${MQTT_TOPIC_SNAPSHOT}`);
        } else {
            console.error('MQTT Subscription Error:', err);
        }
    });
});

mqttClient.on('reconnect', () => {
    console.log('MQTT reconnecting...');
});

mqttClient.on('offline', () => {
    mqttConnected = false;
    console.warn('MQTT offline');
});

mqttClient.on('error', (err) => {
    console.error('MQTT error:', err.message);
});

mqttClient.on('message', (topic, message) => {
    lastMqttMessageAt = Date.now();
    const msgString = message.toString();

    try {
        const parsed = JSON.parse(msgString);
        if (parsed.target && String(parsed.target).toLowerCase() !== TARGET_NAME) {
            console.log(`MQTT IGNORED [${topic}]: foreign target "${parsed.target}"`);
            return;
        }
    } catch (_) {
        // non-JSON payloads pass through
    }

    if (topic === MQTT_TOPIC_SNAPSHOT) {
        console.log(`MQTT IN [${topic}]: (snapshot, ${msgString.length} bytes)`);
    } else {
        console.log(`MQTT IN [${topic}]: ${msgString}`);
    }

    broadcast(msgString);
});

// --- WebSocket Server (large maxPayload for face snapshots) ---
const wss = new WebSocket.Server({ port: WS_PORT, maxPayload: 16 * 1024 * 1024 });

console.log(`WebSocket Server started on port ${WS_PORT}`);

function sendStatus(ws) {
    ws.send(JSON.stringify({
        type: 'STATUS',
        message: 'Connected to Vision Backend',
        mqttConnected,
        broker: MQTT_BROKER,
        lastMqttMessageAt,
    }));
}

wss.on('connection', (ws) => {
    console.log('New WebSocket Client connected');
    sendStatus(ws);

    ws.on('close', () => {
        console.log('Client disconnected');
    });
});

// Keep dashboard informed of MQTT link state
setInterval(() => {
    const status = JSON.stringify({
        type: 'STATUS',
        message: mqttConnected ? 'MQTT broker linked' : 'MQTT broker offline',
        mqttConnected,
        broker: MQTT_BROKER,
        lastMqttMessageAt,
    });
    wss.clients.forEach((client) => {
        if (client.readyState === WebSocket.OPEN) {
            client.send(status);
        }
    });
}, 5000);

function broadcast(data) {
    wss.clients.forEach((client) => {
        if (client.readyState === WebSocket.OPEN) {
            client.send(data);
        }
    });
}

// --- HTTP Server for Dashboard ---
const server = http.createServer((req, res) => {
    if (req.url === '/' || req.url === '/index.html') {
        fs.readFile(path.join(__dirname, '../../dashboard/index.html'), (err, data) => {
            if (err) {
                res.writeHead(500);
                res.end('Error loading dashboard');
                return;
            }
            res.writeHead(200, { 'Content-Type': 'text/html' });
            res.end(data);
        });
    } else {
        res.writeHead(404);
        res.end('Not Found');
    }
});

server.listen(HTTP_PORT, () => {
    console.log(`HTTP Dashboard running on http://localhost:${HTTP_PORT}`);
});
