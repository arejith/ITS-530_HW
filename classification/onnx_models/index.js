// -----------------------------------------------------------
//  CONFIGURATION
// -----------------------------------------------------------

// Add ALL your ONNX models here.
const MODEL_PATHS = {
  modelA: "./class_MLP_net.onnx?v=" + Date.now(),
  ModelB: "./class_DL_net.onnx?v=" + Date.now(),
  ModelC: "./class_My_resnet.onnx?v=" + Date.now(),
  ModelD: "./class_NMLP.onnx?v=" + Date.now(),
  ModelE: "./class_no_act.onnx?v=" + Date.now(),
  ModelF: "./class_XGB.onnx?v=" + Date.now(),
  //ModelB: "./regression_DL_net.onnx?v=" + Date.now()
  // modelB: "./model2.onnx?v=" + Date.now(),
  // modelC: "./model3.onnx?v=" + Date.now(),
};

const MODEL_DISPLAY_NAMES = {
  modelA: "MLP_net",
  ModelB: "DL_net",
  ModelC: "ResNet",
  ModelD: "NMLP",
  ModelE: "No-Act",
  ModelF: "XGB_Classifier"
};

// 'Good':3, 'Hazardous':0, 'Moderate':2, 'Poor':1
const CLASS_LABELS = {
  0: "Hazardous",
  1: "Poor",
  2: "Moderate",
  3: "Good"
};

const INPUT_NAME = "input1";
const OUTPUT_NAME = "output1";
const NUM_FEATURES = 9;

// If some models use a different INPUT name, list them here.
const MODEL_INPUT_OVERRIDE = {
  ModelF: "float_input", // XGBoost exported by skl2onnx typically uses this
  // add more if needed
};

// NEW: per-model OUTPUT overrides (probabilities and/or labels)
const MODEL_OUTPUT_OVERRIDE = {
  // Try these in order for ModelF; the first one found will be used
  ModelF: ["output_probability", "probabilities", "probability"] // common names
};
const MODEL_LABEL_OVERRIDE = {
  ModelF: ["output_label", "label"] // if only labels are present
};

// -----------------------------------------------------------
//  INTERNAL STATE
// -----------------------------------------------------------
const sessions = {};
let modelsReady = false;

// -----------------------------------------------------------
//  HELPER FUNCTIONS
// -----------------------------------------------------------
function readInputs() {
  const x = new Float32Array(NUM_FEATURES);
  for (let i = 0; i < NUM_FEATURES; i++) {
    const v = parseFloat(document.getElementById(`box${i}c1`).value);
    x[i] = Number.isFinite(v) ? v : 0;
  }
  return new ort.Tensor("float32", x, [1, NUM_FEATURES]);
}

function setStatus(msg) {
  const el = document.getElementById("status");
  if (el) el.textContent = msg;
}

// NEW: tiny helpers for argmax & softmax fallback
function argmax(arr) {
  let idx = 0, max = arr[0];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > max) { max = arr[i]; idx = i; }
  }
  return idx;
}
function softmaxFallback(logits) {
  const maxLogit = Math.max(...logits);
  const exps = logits.map(v => Math.exp(v - maxLogit));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}
function isProbVector(p) {
  if (!p || p.length === 0) return false;
  const s = p.reduce((a, b) => a + b, 0);
  return Number.isFinite(s) && Math.abs(s - 1) < 1e-3 &&
         p.every(v => v >= -1e-6 && v <= 1 + 1e-6);
}

// UPDATED: richer table renderer with minimal change footprint
function renderTable(rows) {
  const htmlRows = rows.map(r => {
    const probsStr = r.probs.length ? r.probs.map(v => v.toFixed(4)).join(", ") : "-";
    const label = r.pred === "-" ? "Class -" : (CLASS_LABELS[r.pred] || `Class ${r.pred}`);
    const conf = r.confidence != null ? r.confidence.toFixed(4) : "-";
    return `
      <tr>
        <td>${r.model}</td>
        <td>${label}</td>
        <td>${r.pred}</td>
        <td>${conf}</td>
        <td>${probsStr}</td>
      </tr>
    `;
  }).join("");

  document.getElementById("predictions1").innerHTML = `
    <table>
      <tr>
        <th>Model</th>
        <th>Air Quality</th>
        <th>Predicted Class (argmax)</th>
        <th>Confidence</th>
        <th>Probabilities [C]</th>
      </tr>
      ${htmlRows}
    </table>
  `;
}

// NEW: pick an output tensor by trying known names then falling back
function pickOutputTensor(name, result) {
  // 1) default "output1"
  if (result[OUTPUT_NAME]) return result[OUTPUT_NAME];

  // 2) per-model override list
  const overrideList = MODEL_OUTPUT_OVERRIDE[name];
  if (overrideList) {
    for (const k of overrideList) if (result[k]) return result[k];
  }

  // 3) Heuristic: find the first Float tensor with length > 1 (likely probs)
  for (const v of Object.values(result)) {
    if (v && v.data && v.data.length > 1 && typeof v.data[0] === "number") {
      return v;
    }
  }

  return null;
}

// NEW: pick a label output if no probabilities available
function pickLabelTensor(name, result) {
  const overrideList = MODEL_LABEL_OVERRIDE[name];
  if (overrideList) {
    for (const k of overrideList) if (result[k]) return result[k];
  }
  // common fallbacks
  if (result["label"]) return result["label"];
  if (result["output_label"]) return result["output_label"];
  return null;
}

// -----------------------------------------------------------
//  LOAD MODELS
// -----------------------------------------------------------
async function loadModels() {
  try {
    setStatus("Loading models…");
    for (const [key, path] of Object.entries(MODEL_PATHS)) {
      sessions[key] = await ort.InferenceSession.create(path);
    }
    modelsReady = true;
    document.getElementById("evalBtn").disabled = false;
    setStatus("Models loaded. Ready.");
  } catch (e) {
    console.error("Model load error:", e);
    setStatus("Failed to load model(s): " + e.message);
  }
}

// -----------------------------------------------------------
//  RUN INFERENCE (all models)
// -----------------------------------------------------------
async function runExample1() {
  if (!modelsReady) {
    alert("Models are still loading…");
    return;
  }

  const tensorX = readInputs();
  const rows = [];

  try {
    for (const [name, session] of Object.entries(sessions)) {
      const realInputName = MODEL_INPUT_OVERRIDE[name] || INPUT_NAME;
      const feeds = { [realInputName]: tensorX };
      const result = await session.run(feeds);

      // 1) try to get probabilities
      const probTensor = pickOutputTensor(name, result);

      if (probTensor && probTensor.data) {
        let probs = Array.from(probTensor.data);
        // Some exporters return [1, C]; we just keep the flat view
        if (!isProbVector(probs)) probs = softmaxFallback(probs);

        const idx = argmax(probs);
        const conf = probs[idx] ?? 0;

        rows.push({
          model: MODEL_DISPLAY_NAMES[name] || name,
          pred: idx,
          confidence: conf,
          probs
        });
        continue;
      }

      // 2) no probs → try label only
      const labelTensor = pickLabelTensor(name, result);
      if (labelTensor && labelTensor.data) {
        const pred = Array.isArray(labelTensor.data) ? labelTensor.data[0] : labelTensor.data;
        rows.push({
          model: MODEL_DISPLAY_NAMES[name] || name,
          pred: Number(pred),
          confidence: null,
          probs: []
        });
        continue;
      }

      // 3) nothing found
      rows.push({
        model: MODEL_DISPLAY_NAMES[name] || name,
        pred: "-",
        confidence: 0,
        probs: []
      });
    }
  } catch (e) {
    console.error("Inference error:", e);
    alert("Error running inference: " + e.message);
    return;
  }

  renderTable(rows);
}

// -----------------------------------------------------------
//  INITIALIZE
// -----------------------------------------------------------
window.addEventListener("DOMContentLoaded", () => {
  loadModels();
  const btn = document.getElementById("evalBtn");
  btn.addEventListener("click", runExample1);
});
