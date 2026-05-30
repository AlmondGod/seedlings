import http from "node:http";

const port = Number(process.env.LLM_PORT ?? 8787);
const ollamaUrl = process.env.OLLAMA_URL ?? "http://127.0.0.1:11434/api/generate";
const model = process.env.OLLAMA_MODEL ?? "qwen2.5:0.5b-instruct";

const validActions = new Set(["test_reagent", "write_formula", "teach", "wander"]);
const validTargets = new Set(["emberglass", "moonsalt", "verdigris", "lab", "archive", "furnace", "solvent", "peer"]);

function sendJson(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS, GET",
    "Access-Control-Allow-Headers": "Content-Type",
    "Content-Type": "application/json",
    "Content-Length": Buffer.byteLength(body)
  });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk;
      if (body.length > 32_000) {
        reject(new Error("request too large"));
        req.destroy();
      }
    });
    req.on("end", () => resolve(body));
    req.on("error", reject);
  });
}

function buildPrompt(state) {
  return `Scenario:
A society of AI agents is inside a sealed alchemy laboratory. They independently develop a chemistry by testing reagents, observing reaction outcomes, writing hypotheses, and teaching compact claims to one another.

You are the shared local policy-and-communication backbone for one autonomous alchemy agent.
Return only strict JSON, no markdown, no commentary.

Allowed actions: test_reagent, write_formula, teach, wander.
Allowed targets: emberglass, moonsalt, verdigris, lab, archive, furnace, solvent, peer.

Goal: maximize prediction of hidden chemistry laws and communicate useful compact beliefs.
Inputs include current observations, retrieved-token memory, and incoming communication. Treat all of them as conditioning.
Avoid repeating the same action/message as recent LLM proposals unless new observations strongly support it.
Prefer actions that test uncertain rules or teach high-utility claims.

Agent:
${JSON.stringify(state, null, 2)}

Required JSON:
{"action":"one allowed action","target":"one allowed target","message":"short compact claim based on the input","memory_write":"short memory token"}`;
}

function parseJson(text) {
  const trimmed = text.trim();
  try {
    return JSON.parse(trimmed);
  } catch {
    const start = trimmed.indexOf("{");
    const end = trimmed.lastIndexOf("}");
    if (start >= 0 && end > start) return JSON.parse(trimmed.slice(start, end + 1));
    throw new Error("model did not return JSON");
  }
}

function validateProposal(value) {
  const action = validActions.has(value?.action) ? value.action : "wander";
  const target = validTargets.has(value?.target) ? value.target : "lab";
  const message = String(value?.message ?? "").slice(0, 96) || "testing formula";
  const memoryWrite = String(value?.memory_write ?? "").slice(0, 96) || `${action}|${target}`;
  return { action, target, message, memory_write: memoryWrite };
}

async function queryOllama(state) {
  const response = await fetch(ollamaUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      prompt: buildPrompt(state),
      stream: false,
      options: {
        temperature: 0.55,
        num_predict: 90
      }
    })
  });
  if (!response.ok) throw new Error(`ollama ${response.status}`);
  const data = await response.json();
  return validateProposal(parseJson(data.response ?? ""));
}

const server = http.createServer(async (req, res) => {
  if (req.method === "OPTIONS") {
    sendJson(res, 204, {});
    return;
  }
  if (req.method === "GET" && req.url === "/health") {
    sendJson(res, 200, { ok: true, model });
    return;
  }
  if (req.method !== "POST" || req.url !== "/agent-action") {
    sendJson(res, 404, { error: "not found" });
    return;
  }

  try {
    const state = JSON.parse(await readBody(req));
    const proposal = await queryOllama(state);
    sendJson(res, 200, proposal);
  } catch (error) {
    sendJson(res, 200, {
      action: "wander",
      target: "lab",
      message: "fallback formula",
      memory_write: "fallback|local_policy",
      error: error.message
    });
  }
});

server.listen(port, "127.0.0.1", () => {
  console.log(`LLM sidecar listening on http://127.0.0.1:${port}`);
  console.log(`Using Ollama model ${model}`);
});
