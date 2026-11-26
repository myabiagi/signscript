/**
 * SignScript browser demo.
 * MediaPipe Hands in-page; POST landmarks only to /api/predict.
 *
 * Preview is mirrored (selfie). Landmarks sent to the API are converted into
 * the same mirrored camera space used by `python dev.py start` after cv2.flip.
 */

import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const startBtn = document.getElementById("start-btn");
const cameraStatus = document.getElementById("camera-status");
const letterEl = document.getElementById("letter");
const confEl = document.getElementById("confidence");
const confBar = document.getElementById("conf-bar");
const handEl = document.getElementById("hand");
const statusEl = document.getElementById("status");

const PREDICT_EVERY_MS = 120;
const STABLE_FRAMES = 4;

let handLandmarker = null;
let running = false;
let lastVideoTime = -1;
let lastPredictAt = 0;
let pendingPredict = false;
let recentLetters = [];

const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

function setStatus(text) {
  statusEl.textContent = text;
}

function setCameraStatus(text) {
  cameraStatus.textContent = text;
  cameraStatus.hidden = !text;
}

function updateReadout(letter, confidence, handedness) {
  letterEl.textContent = letter || "—";
  const pct = Math.round((confidence || 0) * 100);
  confEl.textContent = `Confidence ${pct}%`;
  confBar.style.width = `${pct}%`;
  if (pct >= 65) confBar.style.background = "var(--good)";
  else if (pct >= 45) confBar.style.background = "var(--warn)";
  else confBar.style.background = "var(--bad)";
  handEl.textContent = handedness ? `Hand ${handedness}` : "Hand —";
}

function swapHandedness(label) {
  if (label === "Left") return "Right";
  if (label === "Right") return "Left";
  return label;
}

async function createLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  return HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 1,
  });
}

async function enableCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "user",
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
}

function clearOverlay() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawHand(landmarks) {
  const w = canvas.width;
  const h = canvas.height;
  clearOverlay();
  ctx.strokeStyle = "#e08a3c";
  ctx.fillStyle = "#f9e867";
  ctx.lineWidth = 3;

  // Landmark coords are from the unflipped video. The canvas is CSS-mirrored
  // with the video, so drawing at (x, y) keeps the skeleton on the hand.
  for (const [a, b] of CONNECTIONS) {
    const pa = landmarks[a];
    const pb = landmarks[b];
    ctx.beginPath();
    ctx.moveTo(pa.x * w, pa.y * h);
    ctx.lineTo(pb.x * w, pb.y * h);
    ctx.stroke();
  }

  for (const p of landmarks) {
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

function stableLetter(letter) {
  recentLetters.push(letter);
  if (recentLetters.length > STABLE_FRAMES) recentLetters.shift();
  if (recentLetters.length < STABLE_FRAMES) return null;
  if (recentLetters.every((x) => x === letter)) return letter;
  return null;
}

async function predict(landmarks, handednessRaw) {
  if (pendingPredict) return;
  pendingPredict = true;
  try {
    // Match OpenCV selfie flip: mirror x, swap handedness label.
    const payload = {
      landmarks: landmarks.map((p) => [1 - p.x, p.y, p.z]),
      handedness: swapHandedness(handednessRaw),
    };
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      const detail = err.detail;
      const message =
        typeof detail === "string"
          ? detail
          : Array.isArray(detail)
            ? detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
            : res.statusText;
      throw new Error(message);
    }
    const data = await res.json();
    const stable = stableLetter(data.letter);
    if (stable) {
      updateReadout(data.letter, data.confidence, data.handedness);
      setStatus("Tracking");
    } else {
      setStatus("Stabilizing…");
    }
  } catch (err) {
    setStatus(`Predict error: ${err.message}`);
  } finally {
    pendingPredict = false;
  }
}

function loop() {
  if (!running) return;
  const now = performance.now();

  if (handLandmarker && video.readyState >= 2) {
    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      const result = handLandmarker.detectForVideo(video, now);

      if (result.landmarks && result.landmarks.length > 0) {
        const landmarks = result.landmarks[0];
        drawHand(landmarks);
        const handedness =
          result.handedness?.[0]?.[0]?.categoryName || "Right";

        if (now - lastPredictAt >= PREDICT_EVERY_MS) {
          lastPredictAt = now;
          predict(landmarks, handedness);
        }
        setCameraStatus("");
      } else {
        recentLetters = [];
        clearOverlay();
        updateReadout("—", 0, null);
        setStatus("Waiting for hand…");
        setCameraStatus("No hand detected");
      }
    }
  }

  requestAnimationFrame(loop);
}

async function start() {
  startBtn.disabled = true;
  setCameraStatus("Loading hand model…");
  setStatus("Starting…");
  try {
    handLandmarker = await createLandmarker();
    await enableCamera();
    startBtn.hidden = true;
    running = true;
    setCameraStatus("");
    setStatus("Show a static letter");
    requestAnimationFrame(loop);
  } catch (err) {
    startBtn.disabled = false;
    setCameraStatus("Camera access denied or unavailable.");
    setStatus(String(err.message || err));
  }
}

startBtn.addEventListener("click", start);

fetch("/api/health")
  .then((r) => r.json())
  .then((h) => {
    if (!h.ok) setStatus("API up, but static model file is missing.");
  })
  .catch(() => setStatus("Cannot reach API — is the server running?"));
