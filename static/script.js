const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const liveLabel = document.getElementById("live-label");
const liveConf = document.getElementById("live-conf");
const sentenceBox = document.getElementById("sentence-box");
const progressFill = document.getElementById("progress-fill");

const holdSlider = document.getElementById("hold-threshold");
const holdVal = document.getElementById("hold-threshold-val");
const confSlider = document.getElementById("conf-threshold");
const confVal = document.getElementById("conf-threshold-val");

const audioPlayer = document.getElementById("audio-player");

let sentence = "";
let currentLabel = null;
let holdCount = 0;
let lastAddedLabel = null; // mencegah duplikasi langsung berturut-turut
let busy = false; // hindari overlap request /predict

const CAPTURE_INTERVAL_MS = 200; // ~5 fps, cukup untuk landmark statis & hemat request

holdSlider.addEventListener("input", () => (holdVal.textContent = holdSlider.value));
confSlider.addEventListener("input", () => (confVal.textContent = confSlider.value + "%"));

// ---------------------------------------------------------------------------
// Setup webcam
// ---------------------------------------------------------------------------
async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 480, height: 360, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  return new Promise((resolve) => {
    video.onloadedmetadata = () => resolve(video);
  });
}

function captureFrameAsBase64() {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.save();
  ctx.scale(-1, 1); // cocokkan dengan video yang di-mirror di CSS
  ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
  ctx.restore();
  return canvas.toDataURL("image/jpeg", 0.8);
}

// ---------------------------------------------------------------------------
// Loop prediksi
// ---------------------------------------------------------------------------
let frameCounter = 0;

async function predictLoop() {
  if (!busy) {
    busy = true;
    try {
      const imageData = captureFrameAsBase64();
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData }),
      });
      const data = await res.json();
      frameCounter += 1;
      if (frameCounter % 20 === 0) {
        console.log(`[debug] frame #${frameCounter}, video: ${video.videoWidth}x${video.videoHeight}, response:`, data);
      }
      if (data.error) {
        console.error("Server error dari /predict:", data.error);
      }
      handlePrediction(data);
    } catch (err) {
      console.error("Predict error:", err);
    } finally {
      busy = false;
    }
  }
  setTimeout(predictLoop, CAPTURE_INTERVAL_MS);
}

function handlePrediction(data) {
  const holdThreshold = parseInt(holdSlider.value, 10);
  const confThreshold = parseInt(confSlider.value, 10) / 100;

  if (!data.detected || !data.label || data.confidence < confThreshold) {
    liveLabel.textContent = data.detected ? `${data.label ?? "-"}` : "-";
    liveConf.textContent = data.detected ? `${(data.confidence * 100).toFixed(0)}%` : "";
    currentLabel = null;
    holdCount = 0;
    updateProgress(0);
    return;
  }

  liveLabel.textContent = data.label;
  liveConf.textContent = `${(data.confidence * 100).toFixed(0)}%`;

  if (data.label === currentLabel) {
    holdCount += 1;
  } else {
    currentLabel = data.label;
    holdCount = 1;
  }

  updateProgress(Math.min(holdCount / holdThreshold, 1));

  if (holdCount >= holdThreshold) {
    if (currentLabel !== lastAddedLabel) {
      addSymbol(currentLabel);
      lastAddedLabel = currentLabel;
    }
    holdCount = 0;
    updateProgress(0);
  }
}

function updateProgress(fraction) {
  progressFill.style.width = `${fraction * 100}%`;
}

// ---------------------------------------------------------------------------
// Penyusun kalimat
// ---------------------------------------------------------------------------
function renderSentence() {
  sentenceBox.textContent = sentence.length ? sentence : "\u00A0";
}

function addSymbol(symbol) {
  sentence += symbol;
  renderSentence();
}

document.getElementById("btn-space").addEventListener("click", () => {
  sentence += " ";
  lastAddedLabel = null;
  renderSentence();
});

document.getElementById("btn-backspace").addEventListener("click", () => {
  sentence = sentence.slice(0, -1);
  lastAddedLabel = null;
  renderSentence();
});

document.getElementById("btn-clear").addEventListener("click", () => {
  sentence = "";
  lastAddedLabel = null;
  renderSentence();
});

document.getElementById("btn-speak").addEventListener("click", async () => {
  const text = sentence.trim();
  if (!text) return;
  try {
    const res = await fetch("/speak", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (data.audio_base64) {
      audioPlayer.src = `data:${data.mime};base64,${data.audio_base64}`;
      audioPlayer.play();
    }
  } catch (err) {
    console.error("Speak error:", err);
  }
});

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
(async function init() {
  try {
    await setupCamera();
    renderSentence();
    predictLoop();
  } catch (err) {
    alert("Tidak bisa mengakses webcam: " + err.message);
  }
})();
