/** Browser demo: camera + MediaPipe, posts landmarks to /api/*. */

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
const alphabetGrid = document.getElementById("alphabet-grid");

/** Defaults; overwritten by GET /api/config. */
const cfg = {
  predictEveryMs: 120,
  stableFrames: 4,
  moveSpeed: 0.022,
  idleSpeed: 0.012,
  speedEma: 0.35,
  maxBufferFrames: 90,
  minStrokeFrames: 4,
  dynamicHoldMs: 1800,
  tipIndices: [4, 8, 12, 16, 20],
  wristIndex: 0,
  indexTip: 8,
};

let handLandmarker = null;
let running = false;
let lastVideoTime = -1;
let lastPredictAt = 0;
let pendingPredict = false;
let recentLetters = [];

let previousSelfie = null;
let smoothSpeed = 0;
let moving = false;
let moveFrames = [];
let moveHandedness = "Right";
let moveHandednessDisplay = "Right";
let dynamicHoldUntil = 0;
let pendingDynamic = false;

const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const MOTION_LETTERS = new Set(["J", "Z"]);

function setStatus(text) {
  statusEl.textContent = text;
}

function setCameraStatus(text) {
  cameraStatus.textContent = text;
  cameraStatus.hidden = !text;
}

function updateReadout(letter, confidence, handedness, kind = "static") {
  letterEl.textContent = letter || "—";
  const pct = Math.round((confidence || 0) * 100);
  confEl.textContent = `Confidence ${pct}%${kind === "dynamic" ? " · J/Z" : ""}`;
  confBar.style.width = `${pct}%`;
  if (pct >= 65) confBar.style.background = kind === "dynamic" ? "#00c8dc" : "var(--good)";
  else if (pct >= 45) confBar.style.background = "var(--warn)";
  else confBar.style.background = "var(--bad)";
  handEl.textContent = handedness ? `Hand ${handedness}` : "Hand —";
  highlightAlphabet(letter);
}

function highlightAlphabet(letter) {
  if (!alphabetGrid) return;
  for (const tile of alphabetGrid.querySelectorAll(".asl-tile")) {
    tile.classList.toggle("is-active", letter && tile.dataset.letter === letter);
  }
}

function buildAlphabetGrid() {
  if (!alphabetGrid) return;
  alphabetGrid.innerHTML = "";
  for (const letter of ALPHABET) {
    const tile = document.createElement("figure");
    tile.className = "asl-tile";
    tile.dataset.letter = letter;
    if (MOTION_LETTERS.has(letter)) tile.classList.add("is-motion");

    const img = document.createElement("img");
    img.src = `/static/asl/${letter}.svg`;
    img.alt = `ASL handshape for ${letter}`;
    img.loading = "lazy";
    img.width = 96;
    img.height = 96;

    const caption = document.createElement("figcaption");
    caption.textContent = letter;
    if (MOTION_LETTERS.has(letter)) {
      const tag = document.createElement("span");
      tag.className = "asl-tag";
      tag.textContent = "motion";
      caption.appendChild(tag);
    }

    tile.append(img, caption);
    alphabetGrid.appendChild(tile);
  }
}

/** Flip x into OpenCV selfie space. */
function toSelfieFrame(landmarks) {
  return landmarks.map((p) => [1 - p.x, p.y, p.z]);
}

/** Mirror x so the overlay matches the CSS-flipped video. */
function displayX(rawX) {
  return (1 - rawX) * canvas.width;
}

function displayY(rawY) {
  return rawY * canvas.height;
}

function handSpeed(prev, curr) {
  if (!prev || !curr) return 0;
  const indices = cfg.tipIndices.concat([cfg.wristIndex]);
  let total = 0;
  for (const i of indices) {
    const dx = curr[i][0] - prev[i][0];
    const dy = curr[i][1] - prev[i][1];
    total += Math.hypot(dx, dy);
  }
  return total / indices.length;
}

function apiErrorMessage(err, fallback) {
  const detail = err?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
  }
  return fallback;
}

async function loadConfig() {
  const res = await fetch("/api/config");
  if (!res.ok) return;
  const data = await res.json();
  Object.assign(cfg, data);
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
  ctx.strokeStyle = "#e08a3c";
  ctx.fillStyle = "#f9e867";
  ctx.lineWidth = 3;
  ctx.globalAlpha = 1;

  for (const [a, b] of CONNECTIONS) {
    const pa = landmarks[a];
    const pb = landmarks[b];
    ctx.beginPath();
    ctx.moveTo(displayX(pa.x), displayY(pa.y));
    ctx.lineTo(displayX(pb.x), displayY(pb.y));
    ctx.stroke();
  }

  for (const p of landmarks) {
    ctx.beginPath();
    ctx.arc(displayX(p.x), displayY(p.y), 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawOverlay(landmarks) {
  clearOverlay();
  if (landmarks) drawHand(landmarks);
}

function stableLetter(letter) {
  recentLetters.push(letter);
  if (recentLetters.length > cfg.stableFrames) recentLetters.shift();
  if (recentLetters.length < cfg.stableFrames) return null;
  if (recentLetters.every((x) => x === letter)) return letter;
  return null;
}

async function predictStatic(landmarks, handednessRaw) {
  if (pendingPredict || pendingDynamic) return;
  if (performance.now() < dynamicHoldUntil) return;

  pendingPredict = true;
  try {
    // Selfie-flip coords; keep MediaPipe anatomical Left/Right.
    const payload = {
      landmarks: toSelfieFrame(landmarks),
      handedness: handednessRaw,
    };
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(apiErrorMessage(err, res.statusText));
    }
    const data = await res.json();
    const stable = stableLetter(data.letter);
    if (stable) {
      updateReadout(data.letter, data.confidence, handednessRaw, "static");
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

async function predictDynamic(frames, handedness, displayHandedness) {
  if (pendingDynamic) return;
  pendingDynamic = true;
  setStatus("Classifying J/Z…");
  try {
    const res = await fetch("/api/predict-dynamic", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frames, handedness }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(apiErrorMessage(err, res.statusText));
    }
    const data = await res.json();
    recentLetters = [];
    updateReadout(
      data.letter,
      data.confidence,
      displayHandedness || data.handedness,
      "dynamic"
    );
    dynamicHoldUntil = performance.now() + cfg.dynamicHoldMs;
    setStatus("Holding J/Z result");
  } catch (err) {
    setStatus(`Stroke skipped: ${err.message}`);
  } finally {
    pendingDynamic = false;
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
        drawOverlay(landmarks);
        const handednessRaw =
          result.handedness?.[0]?.[0]?.categoryName || "Right";
        const selfie = toSelfieFrame(landmarks);

        const instant = handSpeed(previousSelfie, selfie);
        previousSelfie = selfie;
        smoothSpeed =
          (1.0 - cfg.speedEma) * smoothSpeed + cfg.speedEma * instant;

        const holdingDynamic = now < dynamicHoldUntil;

        if (holdingDynamic) {
          setStatus("Holding J/Z result");
          setCameraStatus("");
        } else if (smoothSpeed >= cfg.moveSpeed) {
          if (!moving) {
            moving = true;
            moveFrames = [];
            moveHandedness = handednessRaw;
            moveHandednessDisplay = handednessRaw;
            recentLetters = [];
          }
          if (moveFrames.length < cfg.maxBufferFrames) {
            moveFrames.push(selfie);
          }
          setStatus("Moving — letter frozen");
          setCameraStatus("");
        } else if (moving) {
          const frames = moveFrames.slice();
          moving = false;
          moveFrames = [];
          if (frames.length >= cfg.minStrokeFrames) {
            predictDynamic(frames, moveHandedness, moveHandednessDisplay);
          } else {
            setStatus("Motion too short for J/Z");
          }
          previousSelfie = selfie;
        } else if (smoothSpeed < cfg.idleSpeed) {
          if (now - lastPredictAt >= cfg.predictEveryMs) {
            lastPredictAt = now;
            predictStatic(landmarks, handednessRaw);
          }
          setCameraStatus("");
        } else {
          setStatus("Settling…");
          setCameraStatus("");
        }
      } else {
        recentLetters = [];
        previousSelfie = null;
        smoothSpeed = 0;
        moving = false;
        moveFrames = [];
        clearOverlay();
        if (performance.now() >= dynamicHoldUntil) {
          updateReadout("—", 0, null);
          setStatus("Waiting for hand…");
        }
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
    await loadConfig();
    handLandmarker = await createLandmarker();
    await enableCamera();
    startBtn.hidden = true;
    running = true;
    setCameraStatus("");
    setStatus("Show a letter — draw J or Z with motion");
    requestAnimationFrame(loop);
  } catch (err) {
    startBtn.disabled = false;
    setCameraStatus("Camera access denied or unavailable.");
    setStatus(String(err.message || err));
  }
}

buildAlphabetGrid();
startBtn.addEventListener("click", start);

fetch("/api/health")
  .then((r) => r.json())
  .then((h) => {
    if (!h.ok) setStatus("API up, but static model file is missing.");
    else if (!h.dynamic) setStatus("Static ready — dynamic J/Z model missing.");
  })
  .catch(() => setStatus("Cannot reach API — is the server running?"));
