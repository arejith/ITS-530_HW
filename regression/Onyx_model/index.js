// -----------------------------------------------------------
//  CONFIGURATION
// -----------------------------------------------------------

// Add ALL your ONNX models here. (Kept your originals; just added ModelF for XGB.)
const MODEL_PATHS = {
  modelA: "./regression_MLP_net.onnx?v=" + Date.now(),
  ModelB: "./regression_DL_net.onnx?v=" + Date.now(),
  ModelC: "./regression_My_resnet.onnx?v=" + Date.now(),
  ModelD: "./regression_NMLP.onnx?v=" + Date.now(),
  ModelE: "./regression_no_act.onnx?v=" + Date.now(),
  // 👉 XGBoost regressor:
  ModelF: "./regression_XGB.onnx?v=" + Date.now(),
};

// Keep your existing names/counts
const INPUT_NAME = "input1";
const OUTPUT_NAME = "output1";
const NUM_FEATURES = 14;


const MODEL_DISPLAY_NAMES = {
  modelA: "MLP_net",
  ModelB: "DL_net",
  ModelC: "ResNet",
  ModelD: "NMLP",
  ModelE: "No-Act",
  ModelF: "XGBoost_Regressor",
};

// If a model uses a different INPUT name, override here (needed for XGBoost).
const MODEL_INPUT_OVERRIDE = {
  ModelF: "float_input" // common from skl2onnx / onnxmltools exports
};

// -----------------------------------------------------------
//  INTERNAL STATE
// -----------------------------------------------------------
const sessions = {};
let modelsReady = false;

// -----------------------------------------------------------
//  HELPERS (minimal and safe)
// -----------------------------------------------------------
function setStatus(msg) {
  const el = document.getElementById("status");
  if (el) el.textContent = msg;
}

function readInputs() {
  const x = new Float32Array(NUM_FEATURES);
  for (let i = 0; i < NUM_FEATURES; i++) {
    const v = parseFloat(document.getElementById(`box${i}c1`).value);
    x[i] = Number.isFinite(v) ? v : 0;
  }
  return new ort.Tensor("float32", x, [1, NUM_FEATURES]);
}

// Return first scalar from a tensor (any shape).
function toScalar(t) {
  if (!t || !t.data || t.data.length === 0) return null;
  return Number(t.data[0]);
}

// Pick output tensor by your OUTPUT_NAME, otherwise first available.
function pickOutputTensor(result) {
  if (result[OUTPUT_NAME]) return result[OUTPUT_NAME];
  const keys = Object.keys(result);
  for (const k of keys) {
    const v = result[k];
    if (v && v.data && typeof v.data.length === "number" && v.data.length >= 1) {
      return v;
    }
  }
  return null;
}

function renderTable(outputs) {
  const rows = outputs
    .map(({ name, val }) => {
      const display = val == null ? "-" : Number(val).toFixed(4);
      return `<tr><td>${MODEL_DISPLAY_NAMES[name] || name}</td><td>${display}</td></tr>`;
    })
    .join("");

  document.getElementById("predictions1").innerHTML = `
    <table>
      <tr><th>Architecture</th><th>Prediction</th></tr>
      ${rows}
    </table>
  `;
}

// -----------------------------------------------------------
/* LOAD MODELS — unchanged behavior, just loops over configured models */
// -----------------------------------------------------------
async function loadModels() {
  try {
    setStatus("Loading models…");
    const keys = Object.keys(MODEL_PATHS);
    if (keys.length === 0) {
      setStatus("No models configured.");
      return;
    }

    for (const [key, path] of Object.entries(MODEL_PATHS)) {
      sessions[key] = await ort.InferenceSession.create(path);
    }

    modelsReady = true;
    const btn = document.getElementById("evalBtn");
    if (btn) btn.disabled = false;
    setStatus("Models loaded. Ready.");
  } catch (e) {
    console.error("Model load error:", e);
    setStatus("Failed to load model(s): " + e.message);
  }
}

// -----------------------------------------------------------
//  RUN INFERENCE (all models) — unchanged UI; just robust I/O
// -----------------------------------------------------------
async function runExample1() {
  if (!modelsReady) {
    alert("Models are still loading…");
    return;
  }

  const x = readInputs();
  const outputs = [];

  try {
    for (const [name, session] of Object.entries(sessions)) {
      // Input name override only where needed (XGBoost)
      const inName = MODEL_INPUT_OVERRIDE[name] || INPUT_NAME;
      const feeds = { [inName]: x };

      const result = await session.run(feeds);
      const outT = pickOutputTensor(result);
      const val = toScalar(outT);

      outputs.push({ name, val });
    }
  } catch (e) {
    console.error("Inference error:", e);
    alert("Error running inference: " + e.message);
    return;
  }

  renderTable(outputs);
}

// -----------------------------------------------------------
//  INITIALIZE (DOMContentLoaded ensures DOM exists)
// -----------------------------------------------------------
window.addEventListener("DOMContentLoaded", () => {
  loadModels();

  const btn = document.getElementById("evalBtn");
  btn.addEventListener("click", runExample1);
});
