const canvas = document.querySelector("#world");
const ctx = canvas.getContext("2d");
const W = canvas.width;
const H = canvas.height;
const tile = 32;
const cols = W / tile;
const rows = H / tile;

const ui = {
  day: document.querySelector("#day"),
  season: document.querySelector("#season"),
  weather: document.querySelector("#weather"),
  beliefs: document.querySelector("#beliefs"),
  communications: document.querySelector("#communications"),
  domains: document.querySelector("#domains"),
  tree: document.querySelector("#tree"),
  consensus: document.querySelector("#consensus"),
  overall: document.querySelector("#overall"),
  reward: document.querySelector("#reward"),
  memeToggle: document.querySelector("#memeToggle"),
  llmToggle: document.querySelector("#llmToggle"),
  llmStatus: document.querySelector("#llmStatus")
};

const rand = (min, max) => min + Math.random() * (max - min);
const pick = (items) => items[Math.floor(Math.random() * items.length)];
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const sigmoid = (value) => 1 / (1 + Math.exp(-value));
const neuralInputs = 10;
const neuralHidden = 5;

const world = {
  tick: 0,
  day: 1,
  seasonIndex: 0,
  weather: "neutral",
  nextWeatherTick: 0,
  log: [],
  discoveries: [],
  observations: [],
  ruleStats: new Map(),
  minedRules: [],
  communications: [],
  memeVisualization: false,
  llmEnabled: false,
  llmOnline: false,
  llmRequests: 0,
  llmFailures: 0,
  memeEdges: [],
  history: [],
  lastHistoryTick: -1,
  reward: 0,
  predictions: 0
};

ui.memeToggle?.addEventListener("change", () => {
  world.memeVisualization = ui.memeToggle.checked;
});

ui.llmToggle?.addEventListener("change", () => {
  world.llmEnabled = ui.llmToggle.checked;
  updateLlmStatus();
});

const seasons = ["Dawn", "Noon", "Dusk", "Midnight"];
const weatherTypes = ["neutral", "neutral", "warmth", "acid", "acid", "vapor"];
const clusterPalette = ["#65a969", "#5d8edb", "#de8a4c", "#a27bc2", "#d45f73", "#d8b34d"];
const predictorPool = {
  emberglass: ["weather=warmth", "terrain=furnace", "terrain=bench", "nearWater=false"],
  moonsalt: ["weather=vapor", "season=Midnight", "terrain=bench", "terrain=archive"],
  verdigris: ["weather=acid", "terrain=channel", "crowded=true", "nearBuilding=true"]
};

const hiddenRules = Object.fromEntries(
  Object.entries(predictorPool).map(([species, predictors]) => [species, pick(predictors)])
);

const outcomeDefs = [
  { id: "reaction", label: "reaction success", color: "#65a969" },
  { id: "social", label: "listener prediction gain", color: "#a27bc2" }
];

const memeColors = {
  green: "#47c86a",
  blue: "#4e8fea",
  red: "#e34f50",
  black: "#252330"
};

function createBrain() {
  return {
    inputHidden: Array.from({ length: neuralHidden }, () =>
      Array.from({ length: neuralInputs }, () => rand(-0.45, 0.45))
    ),
    hiddenBias: Array.from({ length: neuralHidden }, () => rand(-0.12, 0.12)),
    hiddenOutput: Array.from({ length: neuralHidden }, () => rand(-0.45, 0.45)),
    outputBias: rand(-0.12, 0.12)
  };
}

function runBrain(brain, inputs) {
  const hidden = brain.inputHidden.map((weights, index) => {
    const sum = weights.reduce((total, weight, inputIndex) => total + weight * inputs[inputIndex], brain.hiddenBias[index]);
    return sigmoid(sum);
  });
  const outputSum = hidden.reduce((total, value, index) => total + value * brain.hiddenOutput[index], brain.outputBias);
  return { hidden, output: sigmoid(outputSum) };
}

function trainBrain(brain, inputs, target, rate = 0.18) {
  const { hidden, output } = runBrain(brain, inputs);
  const outputDelta = (target - output) * output * (1 - output);
  const oldHiddenOutput = [...brain.hiddenOutput];

  for (let i = 0; i < neuralHidden; i += 1) {
    brain.hiddenOutput[i] += rate * outputDelta * hidden[i];
  }
  brain.outputBias += rate * outputDelta;

  for (let i = 0; i < neuralHidden; i += 1) {
    const hiddenDelta = outputDelta * oldHiddenOutput[i] * hidden[i] * (1 - hidden[i]);
    for (let j = 0; j < neuralInputs; j += 1) {
      brain.inputHidden[i][j] += rate * hiddenDelta * inputs[j];
    }
    brain.hiddenBias[i] += rate * hiddenDelta;
  }

  return output;
}

function observationFeatures(plant) {
  const facts = plantFacts(plant);
  const hour = (world.tick % 1440) / 1440;
  return [
    facts.includes("species=emberglass") ? 1 : 0,
    facts.includes("species=moonsalt") ? 1 : 0,
    facts.includes("species=verdigris") ? 1 : 0,
    facts.includes("weather=acid") ? 1 : 0,
    facts.includes("weather=neutral") ? 1 : 0,
    facts.includes("weather=vapor") ? 1 : 0,
    facts.includes("terrain=channel") || facts.includes("terrain=archive") ? 1 : 0,
    facts.includes("terrain=bench") || facts.includes("terrain=furnace") ? 1 : 0,
    facts.includes("nearWater=true") ? 1 : 0,
    hour
  ];
}

function conceptFeatures(seed) {
  const index = Math.abs(hashString(seed)) % neuralInputs;
  return Array.from({ length: neuralInputs }, (_, inputIndex) => {
    if (inputIndex === index) return 1;
    if (inputIndex === 9) return (world.tick % 1440) / 1440;
    return 0;
  });
}

function hashString(text) {
  return [...text].reduce((hash, char) => ((hash << 5) - hash + char.charCodeAt(0)) | 0, 0);
}

function plantFacts(plant) {
  const terrainType = terrain[Math.floor(plant.y / tile)]?.[Math.floor(plant.x / tile)] ?? "unknown";
  const nearbyAgents = agents?.filter((agent) => distance(agent, plant) < 80).length ?? 0;
  const baseFacts = [
    `species=${plant.species}`,
    `weather=${world.weather}`,
    `terrain=${terrainType}`,
    `season=${seasons[world.seasonIndex]}`,
    `nearWater=${["channel", "basin"].includes(terrainType)}`,
    `crowded=${nearbyAgents >= 3}`,
    `nearBuilding=${isNearBuilding(plant.x, plant.y)}`
  ];
  const species = baseFacts[0];
  return [
    ...baseFacts,
    ...baseFacts.slice(1).map((fact) => `${species}&${fact}`)
  ];
}

function isNearBuilding(x, y) {
  return [...houses, ...landmarks].some((site) => {
    const cx = (site.x + site.w / 2) * tile;
    const cy = (site.y + site.h / 2) * tile;
    return Math.hypot(cx - x, cy - y) < 120;
  });
}

const houses = [];

const landmarks = [
  { type: "biotech", name: "BIOTECH LAB", x: 2, y: 1, w: 19, h: 9 },
  { type: "library", name: "DATA LIBRARY", x: 39, y: 1, w: 19, h: 9 },
  { type: "robotics", name: "ROBOTICS LAB", x: 2, y: 11, w: 17, h: 9 },
  { type: "energy", name: "ENERGY ANALYSIS", x: 41, y: 11, w: 17, h: 9 },
  { type: "synthesis", name: "ADVANCED SYNTHESIS", x: 2, y: 24, w: 18, h: 11 },
  { type: "materials", name: "MATERIALS RESEARCH", x: 40, y: 24, w: 18, h: 11 },
  { type: "hub", name: "COLLABORATION HUB", x: 22, y: 12, w: 16, h: 10 },
  { type: "gate", name: "", x: 24, y: 1, w: 12, h: 8 },
  { type: "garden", name: "", x: 23, y: 25, w: 14, h: 12 }
];

const terrain = Array.from({ length: rows }, (_, y) =>
  Array.from({ length: cols }, (_, x) => {
    const edge = x < 1 || y < 1 || x > cols - 2 || y > rows - 2;
    if (edge) return "wall";
    if ((x - 52) ** 2 + (y - 10) ** 2 < 12 || (x - 5) ** 2 + (y - 28) ** 2 < 10) return "basin";
    if ((x >= 5 && x <= 54 && y === 6) || (x === 50 && y >= 6 && y <= 33)) return "channel";
    if (x >= 3 && x <= 10 && y >= 10 && y <= 17) return "bench";
    if (x >= 9 && x <= 17 && y >= 27 && y <= 34) return "bench";
    if (x >= 15 && x <= 29 && y >= 6 && y <= 17) return "furnace";
    if (x >= 33 && x <= 54 && y >= 5 && y <= 17) return "archive";
    if (x >= 41 && x <= 56 && y >= 25 && y <= 35) return "archive";
    if ((x + y) % 17 === 0 && (
      (x >= 2 && x <= 12 && y >= 3 && y <= 18) ||
      (x >= 21 && x <= 38 && y >= 23 && y <= 36)
    )) return "crystals";
    if (
      (x >= 2 && x <= 12 && y >= 3 && y <= 18) ||
      (x >= 14 && x <= 30 && y >= 4 && y <= 18) ||
      (x >= 32 && x <= 55 && y >= 3 && y <= 18) ||
      (x >= 2 && x <= 19 && y >= 23 && y <= 36) ||
      (x >= 21 && x <= 38 && y >= 23 && y <= 36) ||
      (x >= 40 && x <= 57 && y >= 23 && y <= 36)
    ) return "floor";
    if (
      (x >= 6 && x <= 53 && y >= 19 && y <= 21) ||
      (x >= 20 && x <= 23 && y >= 6 && y <= 36) ||
      (x >= 38 && x <= 41 && y >= 6 && y <= 36)
    ) return "path";
    return "wall";
  })
);

const plants = [];
for (let i = 0; i < 90; i += 1) {
  const inGarden = i < 48;
  const gardenZone = i % 2 === 0 ? { x1: 3.2, x2: 9.7, y1: 12.4, y2: 17.4 } : { x1: 10.2, x2: 15.8, y1: 27.2, y2: 32.2 };
  let x = inGarden ? rand(gardenZone.x1, gardenZone.x2) * tile : rand(2, cols - 3) * tile;
  let y = inGarden ? rand(gardenZone.y1, gardenZone.y2) * tile : rand(2, rows - 3) * tile;
  let guard = 0;
  while (!inGarden && ["channel", "basin", "wall"].includes(terrain[Math.floor(y / tile)]?.[Math.floor(x / tile)]) && guard < 20) {
    x = rand(2, cols - 3) * tile;
    y = rand(2, rows - 3) * tile;
    guard += 1;
  }
  plants.push({
    x,
    y,
    species: pick(["emberglass", "moonsalt", "verdigris"]),
    age: rand(0, 1),
    bloom: false,
    observedRain: 0
  });
}

const capPalettes = ["#d85d63", "#e0b84f", "#6fbf75", "#7aa5d8", "#a37ac0", "#ef8f65", "#cfd66b", "#70b7a8"];
const skinPalettes = ["#f4d6a3", "#d9b47f", "#b7d48b", "#98cfa7", "#d7c7a5", "#b9d5c5", "#e8c0a6", "#a9c98d"];
const clothingPalettes = ["#476c55", "#5b8f6b", "#7a6aa8", "#b66b56", "#4f7f9f", "#a08248", "#8c5f7f", "#52786c"];

const agents = [
  ["Mika", "glass scribe"],
  ["Taro", "reagent keeper"],
  ["Nori", "skeptic"],
  ["Piko", "formula singer"],
  ["Sumi", "dream theorist"],
  ["Ren", "teacher"],
  ["Kiko", "sample collector"],
  ["Bo", "apparatus tuner"],
  ["Yui", "child"],
  ["Aki", "elder"],
  ["Mori", "chemist"],
  ["Nana", "catalyst watcher"],
  ["Tobu", "furnace builder"],
  ["Lumi", "archivist"],
  ["Fenn", "distiller"],
  ["Riri", "listener"],
  ["Momo", "student"],
  ["Sora", "lab mapper"],
  ["Beni", "bench keeper"],
  ["Iro", "inventor"],
  ["Koma", "reagent keeper"],
  ["Mugi", "distiller"],
  ["Hana", "teacher"],
  ["Toki", "catalyst watcher"],
  ["Nemu", "dream theorist"],
  ["Raku", "furnace builder"],
  ["Sasa", "listener"],
  ["Pomu", "student"],
  ["Kiri", "archivist"],
  ["Maro", "chemist"],
  ["Fuyu", "skeptic"],
  ["Tama", "bench keeper"],
  ["Mina", "glass scribe"],
  ["Kumo", "inventor"],
  ["Roko", "sample collector"],
  ["Niko", "child"],
  ["Suzu", "lab mapper"],
  ["Pipi", "formula singer"],
  ["Eno", "apparatus tuner"],
  ["Matsu", "elder"]
].map(([name, role], index) => ({
  id: index,
  name,
  role,
  color: clothingPalettes[index % clothingPalettes.length],
  skin: skinPalettes[index % skinPalettes.length],
  cap: capPalettes[index % capPalettes.length],
  capShape: pick(["round", "wide", "point", "flat"]),
  spots: 2 + (index % 4),
  x: (10 + (index % 10) * 4.2) * tile + 16,
  y: (18 + Math.floor(index / 10) * 4.2) * tile + 20,
  tx: rand(4, 27) * tile,
  ty: rand(3, 17) * tile,
  speed: rand(0.58, 0.86),
  direction: "down",
  mood: pick(["curious", "cozy", "restless", "focused"]),
  action: "waking",
  speech: "",
  speechUntil: 0,
  notebook: [],
  memory: [],
  retrievedTokens: [],
  models: Object.fromEntries(outcomeDefs.map((outcome) => [outcome.id, {
    ...outcome,
    confidence: rand(0.08, 0.18),
    evidence: 0,
    source: "hunch",
    brain: createBrain(),
    neural: 0.5
  }])),
  policy: {
    observing: rand(0.8, 1.25),
    talking: rand(0.65, 1.05),
    researching: rand(0.45, 0.85),
    wandering: rand(0.45, 0.9)
  },
  memoryPolicy: {
    writeThreshold: rand(0.35, 0.7),
    readBias: rand(0.1, 0.35)
  },
  lastAction: "wandering",
  llmProposal: null,
  llmPending: false,
  nextLlmTick: 0,
  currentMeme: {
    nature: "black",
    color: memeColors.black,
    until: 0,
    label: "quiet"
  },
  outfit: index % 3 === 0 ? "blue" : "plain",
  homeCharm: 0
}));

const artifacts = [
  { type: "blackboard", x: 30.4 * tile, y: 15.1 * tile, text: "?", idea: "questions" },
  { type: "map", x: 46.5 * tile, y: 5.8 * tile, text: "db", idea: "library" },
  { type: "specimen", x: 7.6 * tile, y: 5.4 * tile, text: "jar", idea: "biology" },
  { type: "bell", x: 29.8 * tile, y: 19.2 * tile, text: "hub", idea: "teaching" },
  { type: "chip", x: 9.2 * tile, y: 15.1 * tile, text: "bot", idea: "robotics" }
];

function addLog(text) {
  world.log.unshift(`Day ${world.day}: ${text}`);
  world.log = world.log.slice(0, 34);
  renderPanels();
}

function remember(agent, event) {
  agent.memory.unshift({
    day: world.day,
    tick: world.tick,
    ...event
  });
  agent.memory = agent.memory.slice(0, 24);
  agent.retrievedTokens.unshift(memoryToken(event));
  agent.retrievedTokens = agent.retrievedTokens.filter(Boolean).slice(0, 32);
}

function memoryToken(event) {
  if (!event) return "";
  if (event.type === "observation") {
    const cue = event.facts?.[0]?.replace("species=", "re:") ?? event.outcome;
    return `${cue}|${event.outcome}|${Math.round((event.reward ?? 0) * 100)}`;
  }
  if (event.type === "heard" || event.type === "said") return `${event.type}|${event.claim}|${Math.round((event.confidence ?? 0) * 100)}`;
  return `${event.type ?? "memory"}|${event.claim ?? event.outcome ?? "lab"}`;
}

function addCommunication(from, to, belief, delta) {
  const nature = memeNature(belief.confidence, delta);
  const color = memeColors[nature];
  world.communications.unshift({
    from: from.name,
    to: to.name,
    claim: belief.claim,
    confidence: belief.confidence,
    delta,
    nature,
    day: world.day
  });
  world.communications = world.communications.slice(0, 10);

  const pulse = {
    nature,
    color,
    until: world.tick + 280,
    label: belief.claim
  };
  from.currentMeme = pulse;
  to.currentMeme = { ...pulse, until: world.tick + 340 };
  world.memeEdges.unshift({
    fromId: from.id,
    toId: to.id,
    claim: belief.claim,
    confidence: belief.confidence,
    delta,
    nature,
    color,
    tick: world.tick
  });
  world.memeEdges = world.memeEdges.slice(0, 180);
}

function memeNature(confidence, delta) {
  if (delta < -0.012 || confidence < 0.28) return "red";
  if (confidence > 0.66 && delta >= 0) return "green";
  if (delta > 0.018 || confidence > 0.48) return "blue";
  return "black";
}

function addPredictionReward(prediction, target) {
  const reward = 1 - Math.abs(target - prediction);
  world.reward += reward;
  world.predictions += 1;
  return reward;
}

function recordObservation(agent, outcome, facts, prediction, target, source = "field") {
  const reward = addPredictionReward(prediction, target);
  const record = { tick: world.tick, day: world.day, agent: agent.name, outcome, facts, prediction, target, reward, source };
  world.observations.push(record);
  world.observations = world.observations.slice(-2500);
  if (outcome === "reaction") {
    for (const fact of facts) updateRuleStat(fact, outcome, target);
    mineRules();
  }
  remember(agent, { type: "observation", outcome, facts: facts.slice(0, 4), prediction, target, reward });
  updatePolicy(agent, reward);
  return reward;
}

function updateRuleStat(feature, outcome, target) {
  const key = `${feature}->${outcome}`;
  const stat = world.ruleStats.get(key) ?? { feature, outcome, seen: 0, hits: 0, absentSeen: 0, absentHits: 0 };
  stat.seen += 1;
  stat.hits += target ? 1 : 0;
  world.ruleStats.set(key, stat);

  for (const otherKey of world.ruleStats.keys()) {
    const other = world.ruleStats.get(otherKey);
    if (other.outcome === outcome && other.feature !== feature) {
      other.absentSeen += 1;
      other.absentHits += target ? 1 : 0;
    }
  }
}

function mineRules() {
  const mined = [];
  for (const stat of world.ruleStats.values()) {
    if (stat.feature.startsWith("heard=")) continue;
    if (stat.seen < 6) continue;
    const p = stat.hits / stat.seen;
    const base = stat.absentSeen ? stat.absentHits / stat.absentSeen : 0.5;
    const lift = p - base;
    const confidence = clamp(0.4 * p + 0.6 * Math.max(0, lift), 0, 1);
    if (lift > 0.16 && p > 0.54) {
      mined.push({
        id: `${stat.feature}->${stat.outcome}`,
        feature: stat.feature,
        outcome: stat.outcome,
        claim: `${labelFeature(stat.feature)} predicts ${labelOutcome(stat.outcome)}.`,
        cluster: clusterForFeature(stat.feature),
        confidence,
        neural: p,
        evidence: stat.seen,
        lift,
        parent: stat.feature.includes("&") ? "combined cues" : stat.feature.split("=")[0]
      });
    }
  }
  world.minedRules = mined.sort((a, b) => b.confidence - a.confidence).slice(0, 28);
  for (const rule of world.minedRules) {
    if (rule.confidence > 0.56 && !world.discoveries.includes(rule.id)) {
      world.discoveries.push(rule.id);
      addLog(`A rule emerged: ${rule.claim}`);
      artifacts.push({
        type: "note",
        x: rand(22.5, 27.4) * tile,
        y: rand(12.2, 17) * tile,
        text: rule.cluster.slice(0, 3),
        idea: rule.claim
      });
    }
  }
}

function labelFeature(feature) {
  if (feature.includes("&")) {
    return feature.split("&").map(labelFeature).join(" + ");
  }
  return feature
    .replace("emberglass", "emberglass")
    .replace("moonsalt", "moon salt")
    .replace("verdigris", "verdigris")
    .replace("species=", "")
    .replace("weather=", "ambient ")
    .replace("terrain=", "")
    .replace("season=", "")
    .replace("nearWater=true", "solvent adjacency")
    .replace("nearWater=false", "dry station")
    .replace("crowded=true", "crowding")
    .replace("nearBuilding=true", "apparatus proximity");
}

function labelOutcome(outcome) {
  return outcomeDefs.find((item) => item.id === outcome)?.label ?? outcome;
}

function clusterForFeature(feature) {
  if (feature.includes("&")) return clusterForFeature(feature.split("&")[1]);
  if (feature.startsWith("species=") || feature === "terrain=bench" || feature === "nearWater=true") return "reagents";
  if (feature.startsWith("weather=") || feature.startsWith("season=") || feature.includes("water")) return "catalysts";
  if (feature.includes("crowded") || feature.includes("Building") || feature.includes("building") || feature.includes("archive") || feature.includes("path")) return "apparatus";
  return "models";
}

function titleCase(text) {
  return text.replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function hiddenRuleMatchesPlant(plant) {
  return plantFacts(plant).includes(hiddenRules[plant.species]);
}

function updatePolicy(agent, reward) {
  const key = agent.lastAction ?? "wandering";
  agent.policy[key] = clamp(agent.policy[key] * 0.96 + reward * 0.12, 0.08, 2.4);
  if (agent.memory.length) {
    agent.memoryPolicy.writeThreshold = clamp(agent.memoryPolicy.writeThreshold + (reward > 0.65 ? 0.005 : -0.01), 0.2, 0.85);
    agent.memoryPolicy.readBias = clamp(agent.memoryPolicy.readBias + (reward > 0.6 ? 0.004 : -0.006), 0.02, 0.6);
  }
}

function setWeather() {
  world.weather = pick(weatherTypes);
  world.nextWeatherTick = world.tick + 720;
  if (world.weather === "acid") {
    addLog("The lab air turned acidic. Alchemists began retesting reagent reactions.");
  }
}

function chooseWeightedAction(agent) {
  const weights = {
    observing: agent.policy.observing,
    talking: agent.policy.talking + agent.memoryPolicy.readBias,
    researching: agent.policy.researching,
    wandering: agent.policy.wandering
  };
  const total = Object.values(weights).reduce((sum, value) => sum + value, 0);
  let roll = Math.random() * total;
  for (const [action, weight] of Object.entries(weights)) {
    roll -= weight;
    if (roll <= 0) return action;
  }
  return "wandering";
}

function chooseTarget(agent) {
  const hour = (world.tick % 1440) / 60;
  if (hour < 6 || hour > 21) {
    const restZones = landmarks.filter((site) => ["library", "hub", "synthesis", "materials"].includes(site.type));
    const zone = restZones[agent.id % restZones.length];
    setTarget(agent, (zone.x + zone.w / 2) * tile, (zone.y + zone.h / 2) * tile, "resting between lab shifts");
    return;
  }

  maybeRequestLlmProposal(agent);
  if (agent.llmProposal && applyLlmProposal(agent, agent.llmProposal)) {
    agent.llmProposal = null;
    return;
  }

  const action = chooseWeightedAction(agent);
  agent.lastAction = action;
  if (action === "observing") {
    const plant = pick(plants);
    setTarget(agent, plant.x + rand(-16, 16), plant.y + rand(-16, 16), "testing reagents");
  } else if (action === "researching") {
    const site = pick(landmarks.filter((landmark) => ["lab", "archive", "townHall", "observatory"].includes(landmark.type)));
    setTarget(agent, (site.x + site.w / 2) * tile, (site.y + site.h + 0.35) * tile, "writing formula");
  } else if (action === "talking") {
    const friend = pick(agents.filter((a) => a !== agent));
    setTarget(agent, friend.x + rand(-28, 28), friend.y + rand(-28, 28), "teaching formula");
  } else {
    const site = pick([
      { x: 4.4, y: 7.4, action: "checking solvent bath" },
      { x: 6.2, y: 11.3, action: "tuning catalyst" },
      { x: 5.5, y: 15.8, action: "blowing glass" },
      { x: 36, y: 11.8, action: "sharing lab notes" },
      { x: 47, y: 16.3, action: "studying recipes" },
      { x: 34, y: 34.3, action: "trading reagents" },
      { x: 49, y: 32.3, action: "giving lecture" },
      { x: rand(3, cols - 4), y: rand(2, rows - 4), action: pick(["wandering", "calibrating", "collecting"]) }
    ]);
    setTarget(agent, site.x * tile, site.y * tile, site.action);
  }
}

function maybeRequestLlmProposal(agent) {
  if (!world.llmEnabled || agent.llmPending || world.tick < agent.nextLlmTick) return;
  if (world.tick % 240 !== (agent.id * 17) % 240) return;
  agent.llmPending = true;
  agent.nextLlmTick = world.tick + 900;
  world.llmRequests += 1;
  updateLlmStatus();

  fetch("/agent-action", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(agentLlmState(agent))
  })
    .then((response) => response.json())
    .then((proposal) => {
      if (proposal.error) {
        agent.llmProposal = null;
        world.llmFailures += 1;
        world.llmOnline = false;
      } else {
        agent.llmProposal = validateLlmProposal(proposal);
        world.llmOnline = true;
      }
    })
    .catch(() => {
      world.llmFailures += 1;
      world.llmOnline = false;
    })
    .finally(() => {
      agent.llmPending = false;
      updateLlmStatus();
    });
}

function agentLlmState(agent) {
  const place = terrain[Math.floor(agent.y / tile)]?.[Math.floor(agent.x / tile)] ?? "unknown";
  const incoming = world.communications
    .filter((item) => item.to === agent.name)
    .slice(0, 6)
    .map((item) => ({
      from: item.from,
      claim: item.claim,
      confidence: Number(item.confidence.toFixed(2)),
      utility: item.nature,
      delta: Number(item.delta.toFixed(3))
    }));
  const recentObservations = agent.memory
    .filter((item) => item.type === "observation")
    .slice(0, 8)
    .map((item) => ({
      facts: item.facts,
      predicted: Number(item.prediction.toFixed(2)),
      target: item.target,
      reward: Number(item.reward.toFixed(2))
    }));
  const nearbyReagents = plants
    .filter((plant) => distance(agent, plant) < 180)
    .slice(0, 8)
    .map((plant) => ({
      reagent: plant.species,
      room: terrain[Math.floor(plant.y / tile)]?.[Math.floor(plant.x / tile)] ?? "unknown",
      reacted: plant.bloom
    }));
  const recentLlm = agent.retrievedTokens.filter((token) => token.startsWith("llm|")).slice(0, 5);
  return {
    scenario: "Agents in a sealed alchemy lab learn hidden reagent/catalyst/apparatus laws through experiments and teaching.",
    name: agent.name,
    role: agent.role,
    day: world.day,
    cycle: seasons[world.seasonIndex],
    ambient: world.weather,
    action: agent.action,
    place,
    current_policy_weights: Object.fromEntries(Object.entries(agent.policy).map(([key, value]) => [key, Number(value.toFixed(2))])),
    current_predictors: {
      reaction: {
        confidence: Number(agent.models.reaction.confidence.toFixed(2)),
        neural: Number(agent.models.reaction.neural.toFixed(2)),
        evidence: agent.models.reaction.evidence
      },
      social: {
        confidence: Number(agent.models.social.confidence.toFixed(2)),
        neural: Number(agent.models.social.neural.toFixed(2)),
        evidence: agent.models.social.evidence
      }
    },
    recent_observations: recentObservations,
    incoming_communication: incoming,
    nearby_reagents: nearbyReagents,
    memory_tokens: agent.retrievedTokens.slice(0, 12),
    recent_llm_proposals: recentLlm,
    top_rules: world.minedRules.slice(0, 5).map((rule) => ({
      claim: rule.claim,
      confidence: Number(rule.confidence.toFixed(2)),
      evidence: rule.evidence
    })),
    allowed_actions: ["test_reagent", "write_formula", "teach", "wander"],
    allowed_targets: ["emberglass", "moonsalt", "verdigris", "lab", "archive", "furnace", "solvent", "peer"]
  };
}

function validateLlmProposal(proposal) {
  const validActions = ["test_reagent", "write_formula", "teach", "wander"];
  const validTargets = ["emberglass", "moonsalt", "verdigris", "lab", "archive", "furnace", "solvent", "peer"];
  return {
    action: validActions.includes(proposal?.action) ? proposal.action : "wander",
    target: validTargets.includes(proposal?.target) ? proposal.target : "lab",
    message: String(proposal?.message ?? "testing formula").slice(0, 96),
    memory_write: String(proposal?.memory_write ?? "llm|formula").slice(0, 96)
  };
}

function applyLlmProposal(agent, proposal) {
  agent.retrievedTokens.unshift(`llm|${proposal.memory_write}`);
  agent.retrievedTokens = agent.retrievedTokens.slice(0, 32);
  if (proposal.message) speak(agent, proposal.message);

  if (proposal.action === "test_reagent") {
    const plant = nearestReagent(proposal.target, agent) ?? pick(plants);
    setTarget(agent, plant.x + rand(-18, 18), plant.y + rand(-18, 18), `LLM testing ${labelFeature(`species=${plant.species}`)}`);
    agent.lastAction = "observing";
    return true;
  }
  if (proposal.action === "write_formula") {
    const site = proposal.target === "archive" ? landmarks.find((item) => item.type === "archive") : landmarks.find((item) => item.type === "lab");
    setTarget(agent, (site.x + site.w / 2) * tile, (site.y + site.h + 0.35) * tile, "LLM writing formula");
    agent.lastAction = "researching";
    return true;
  }
  if (proposal.action === "teach") {
    const friend = nearestAgent(agent) ?? pick(agents.filter((candidate) => candidate !== agent));
    setTarget(agent, friend.x + rand(-28, 28), friend.y + rand(-28, 28), "LLM teaching formula");
    agent.lastAction = "talking";
    return true;
  }
  return false;
}

function nearestReagent(species, agent) {
  const candidates = plants.filter((plant) => plant.species === species);
  return candidates.sort((a, b) => distance(agent, a) - distance(agent, b))[0];
}

function nearestAgent(agent) {
  return agents
    .filter((candidate) => candidate !== agent)
    .sort((a, b) => distance(agent, a) - distance(agent, b))[0];
}

function updateLlmStatus() {
  if (!ui.llmStatus) return;
  if (!world.llmEnabled) {
    ui.llmStatus.textContent = "offline";
    return;
  }
  ui.llmStatus.textContent = world.llmOnline ? "online" : world.llmRequests ? "fallback" : "waiting";
}

function setTarget(agent, x, y, action) {
  const point = nearestWalkablePoint(x, y);
  agent.tx = point.x;
  agent.ty = point.y;
  agent.action = action;
}

function nearestWalkablePoint(x, y) {
  const start = { x: clamp(x, tile + 4, W - tile - 4), y: clamp(y, tile + 4, H - tile - 4) };
  if (isWalkablePixel(start.x, start.y)) return start;

  for (let radius = 8; radius <= tile * 5; radius += 8) {
    for (let angle = 0; angle < Math.PI * 2; angle += Math.PI / 8) {
      const candidate = {
        x: clamp(start.x + Math.cos(angle) * radius, tile + 4, W - tile - 4),
        y: clamp(start.y + Math.sin(angle) * radius, tile + 4, H - tile - 4)
      };
      if (isWalkablePixel(candidate.x, candidate.y)) return candidate;
    }
  }

  return { x: 16 * tile, y: 20 * tile };
}

function isWalkablePixel(x, y) {
  if (x < tile || y < tile || x > W - tile || y > H - tile) return false;
  const terrainType = terrain[Math.floor(y / tile)]?.[Math.floor(x / tile)];
  return !["channel", "basin", "wall"].includes(terrainType);
}

function isDockPixel(x, y) {
  return false;
}

function isEntrancePixel(x, y) {
  return false;
}

function isBuildingShellPixel(x, y) {
  return false;
}

function moveAgent(agent) {
  const dx = agent.tx - agent.x;
  const dy = agent.ty - agent.y;
  const d = Math.hypot(dx, dy);
  if (d < 4) {
    chooseTarget(agent);
    return;
  }
  if (Math.abs(dx) > 3) {
    const nextX = agent.x + Math.sign(dx) * agent.speed;
    if (isWalkablePixel(nextX, agent.y)) {
      agent.x = nextX;
    } else {
      chooseTarget(agent);
    }
    agent.direction = dx > 0 ? "right" : "left";
  } else if (Math.abs(dy) > 3) {
    const nextY = agent.y + Math.sign(dy) * agent.speed;
    if (isWalkablePixel(agent.x, nextY)) {
      agent.y = nextY;
    } else {
      chooseTarget(agent);
    }
    agent.direction = dy > 0 ? "down" : "up";
  }
}

function observe(agent) {
  if (world.tick % 32 !== agent.id * 5 % 32) return;
  const plant = plants.find((p) => distance(agent, p) < 70);
  if (!plant) return;

  const facts = plantFacts(plant);
  const model = agent.models.reaction;
  const inputs = observationFeatures(plant);
  const target = hiddenRuleMatchesPlant(plant) ? 1 : 0;
  const prediction = trainBrain(model.brain, inputs, target);
  model.evidence += 1;
  model.neural = prediction;
  model.confidence = clamp(model.confidence * 0.78 + prediction * 0.16 + target * 0.05, 0, 0.98);
  model.source = "field note";
  const reward = recordObservation(agent, "reaction", facts, prediction, target);

  agent.notebook.unshift(`${labelFeature(`species=${plant.species}`)} ${target ? "reacted" : "stayed inert"} ${world.weather} (${Math.round(prediction * 100)}%)`);
  agent.notebook = agent.notebook.slice(0, 6);

  if (Math.abs(target - prediction) > agent.memoryPolicy.writeThreshold && Math.random() < 0.36) {
    speak(agent, pick(["lab note!", "odd result!", "tiny proof!", "hmm!"]));
  }
}

function talk() {
  for (const a of agents) {
    if (world.tick % 34 !== (a.id * 7) % 34) continue;
    const b = agents
      .filter((candidate) => candidate !== a && distance(a, candidate) < 96)
      .sort((left, right) => distance(a, left) - distance(a, right))[0];
    if (!b) continue;

    const rule = world.minedRules.find((item) => item.outcome === "reaction" && world.discoveries.includes(item.id)) ??
      world.minedRules.find((item) => item.outcome === "reaction") ??
      world.minedRules.find((item) => world.discoveries.includes(item.id)) ??
      world.minedRules[0];
    const memory = a.retrievedTokens[0] ?? a.memory.find((item) => item.type === "observation");
    const symbol = rule ? compactRule(rule) : compactMemory(memory);
    const confidence = rule?.confidence ?? memoryConfidence(memory);
    const model = b.models.social;
    const trust = 0.024 + (a.role === "teacher" || b.role === "child" ? 0.026 : 0);
    const target = confidence > 0.58 ? 1 : 0;
    const socialPrediction = trainBrain(model.brain, conceptFeatures(symbol), target, 0.07);
    const before = model.confidence;
    model.neural = socialPrediction;
    model.confidence = clamp(model.confidence * 0.84 + socialPrediction * 0.1 + confidence * trust, 0, 0.98);
    model.source = a.name;
    model.evidence += 1;
    const delta = model.confidence - before;
    recordObservation(b, "social", [`speaker=${a.role}`, `place=${terrain[Math.floor(b.y / tile)]?.[Math.floor(b.x / tile)] ?? "unknown"}`], socialPrediction, target, "communication");
    addCommunication(a, b, { claim: symbol, confidence }, delta);
    remember(a, { type: "said", claim: symbol, to: b.name, confidence });
    remember(b, { type: "heard", claim: symbol, from: a.name, confidence, delta });
    speak(a, `${symbol}?`);
  }
}

function compactRule(rule) {
  const feature = rule.feature
    .replace("weather=", "cat:")
    .replace("terrain=", "app:")
    .replace("species=", "re:")
    .replace("season=", "cycle:")
    .replace("nearWater=true", "solvent")
    .replace("nearWater=false", "dry")
    .replace("crowded=true", "crowd")
    .replace("nearBuilding=true", "apparatus");
  return `${feature}->${rule.outcome}`;
}

function compactMemory(memory) {
  if (!memory) return "formula?";
  if (typeof memory === "string") return memory.split("|").slice(0, 2).join("->");
  const cue = memory.facts?.[0]?.replace("species=", "re:") ?? "note";
  return `${cue}->${memory.outcome ?? "chemistry"}`;
}

function memoryConfidence(memory) {
  if (!memory) return 0.5;
  if (typeof memory === "string") {
    const score = Number(memory.split("|")[2]);
    return Number.isFinite(score) ? score / 100 : 0.5;
  }
  return memory.reward ?? memory.confidence ?? 0.5;
}

function speak(agent, text) {
  agent.speech = text;
  agent.speechUntil = world.tick + 120;
}

function updatePlants() {
  if (world.tick % 50 !== 0) return;
  for (const plant of plants) {
    const boost = hiddenRuleMatchesPlant(plant) ? 0.055 : 0.012;
    plant.age = clamp(plant.age + boost, 0, 1);
    plant.bloom = plant.age > 0.68;
    if (plant.age >= 1 && Math.random() < 0.12) plant.age = rand(0.18, 0.42);
  }
}

function update() {
  world.tick += 1;
  if (world.tick >= world.nextWeatherTick) setWeather();
  if (world.tick % 1440 === 0) {
    world.day += 1;
    world.seasonIndex = Math.floor((world.day - 1) / 7) % seasons.length;
    addLog("Night formulas became fresh ablations for morning.");
  }

  agents.forEach((agent) => {
    moveAgent(agent);
    observe(agent);
  });
  talk();
  updatePlants();
}

function drawTile(x, y, type) {
  const px = x * tile;
  const py = y * tile;
  const palette = {
    floor: "#e8edf0",
    crystals: "#d8f7ff",
    bench: "#d6dee7",
    path: "#edf2f5",
    furnace: "#e4e8ed",
    channel: "#aeeaff",
    basin: "#9ce7ff",
    archive: "#e6ebf1",
    wall: "#eef2f7"
  };
  ctx.fillStyle = palette[type] ?? "#e8edf0";
  ctx.fillRect(px, py, tile, tile);
  if (type === "wall") drawWallTile(px, py, x, y);
  else drawFloorTile(px, py, x, y, type);
}

function drawFloorTile(px, py, x, y, type) {
  ctx.fillStyle = "rgba(143, 161, 176, 0.28)";
  ctx.fillRect(px, py, tile, 1);
  ctx.fillRect(px, py, 1, tile);
  ctx.fillStyle = "rgba(33, 196, 232, 0.18)";
  if ((x + y * 3) % 11 === 0) ctx.fillRect(px + 10, py + 15, 12, 2);
  if (type === "path") {
    ctx.fillStyle = "rgba(28, 221, 246, 0.38)";
    if ((x + y) % 4 === 0) ctx.fillRect(px + 12, py + 2, 3, 26);
  }
  if (type === "channel" || type === "basin") {
    ctx.fillStyle = "#38cfff";
    ctx.fillRect(px + 3, py + 5, 26, 20);
    ctx.fillStyle = "rgba(255,255,255,0.68)";
    ctx.fillRect(px + ((world.tick + x * 7) % 18), py + 10, 8, 2);
  }
}

function drawWallTile(px, py, x, y) {
  ctx.fillStyle = "#d6dce5";
  ctx.fillRect(px, py, tile, tile);
  ctx.fillStyle = "#f9fbff";
  ctx.fillRect(px, py, tile, 8);
  ctx.fillStyle = "#b7c0cc";
  ctx.fillRect(px, py + 24, tile, 8);
  ctx.fillStyle = "#8f9aa8";
  ctx.fillRect(px, py + 30, tile, 2);
  ctx.fillStyle = "rgba(0, 198, 255, 0.7)";
  if ((x + y) % 5 === 0) ctx.fillRect(px + 11, py + 9, 10, 3);
}

function drawLandmark(site) {
  const x = site.x * tile;
  const y = site.y * tile;
  const w = site.w * tile;
  const h = site.h * tile;
  if (site.type === "gate") {
    drawCentralGate(x, y, w, h);
    return;
  }
  if (site.type === "garden") {
    drawAtrium(x, y, w, h);
    return;
  }
  drawLabRoom(site, x, y, w, h);
}

function drawLabRoom(site, x, y, w, h) {
  drawRoomShell(x, y, w, h, site.name);
  if (site.type === "biotech") {
    drawReactionTank(x + 26, y + 34, "#8b5cff", 72);
    drawReactionTank(x + 76, y + 34, "#20df82", 72);
    drawConsoleWall(x + 136, y + 34, 5);
    drawHoloBoard(x + 148, y + 34, 92, "leaf");
    drawBenchCluster(x + 124, y + 126, "#20df82");
    drawSmallDevices(x + 22, y + 132, 5);
    return;
  }
  if (site.type === "library") {
    drawServerWall(x + 22, y + 36, 6);
    drawConsoleWall(x + 160, y + 34, 5);
    drawLongConsole(x + 120, y + 124, 146);
    drawHoloGlobe(x + w - 76, y + 86, "#35d7ff", 34);
    return;
  }
  if (site.type === "robotics") {
    drawRobotArm(x + 48, y + 84);
    drawRobotArm(x + 116, y + 84);
    drawSmallDevices(x + 24, y + 38, 6);
    drawLongConsole(x + 70, y + 150, 156);
    drawDrone(x + w - 74, y + 118);
    return;
  }
  if (site.type === "energy") {
    drawReactionTank(x + 30, y + 44, "#9b68ff", 82);
    drawHoloBoard(x + 112, y + 40, 118, "wave");
    drawReactionTank(x + w - 58, y + 44, "#23e77e", 82);
    drawConsoleWall(x + 94, y + 118, 4);
    return;
  }
  if (site.type === "synthesis") {
    drawReactionTank(x + 28, y + 54, "#a654ff", 86);
    drawReagentShelves(x + 132, y + 40);
    drawLongConsole(x + 96, y + 150, 150);
    drawSmallDevices(x + 34, y + h - 64, 6);
    return;
  }
  if (site.type === "materials") {
    drawCrystalAnalyzer(x + 118, y + 54);
    drawReagentShelves(x + w - 96, y + 42);
    drawLongConsole(x + 124, y + 154, 144);
    drawSmallDevices(x + 28, y + 58, 5);
    return;
  }
  if (site.type === "hub") {
    drawHubTable(x + w / 2 - 70, y + h / 2 - 44);
    drawConsolePod(x + 24, y + 96);
    drawConsolePod(x + w - 64, y + 96);
    drawSign(x + w / 2 - 72, y + 10, 144, "COLLABORATION HUB");
  }
}

function drawRoomShell(x, y, w, h, label) {
  ctx.fillStyle = "rgba(75, 92, 110, 0.28)";
  ctx.fillRect(x + 8, y + h - 2, w - 8, 9);
  ctx.fillStyle = "#dfe5ec";
  ctx.fillRect(x, y, w, h);
  ctx.fillStyle = "#f8fbff";
  ctx.fillRect(x + 8, y + 8, w - 16, h - 16);
  ctx.fillStyle = "#c8d0da";
  ctx.fillRect(x, y, w, 8);
  ctx.fillRect(x, y + h - 8, w, 8);
  ctx.fillRect(x, y, 8, h);
  ctx.fillRect(x + w - 8, y, 8, h);
  ctx.fillStyle = "#9aa6b4";
  ctx.fillRect(x + 8, y + h - 13, w - 16, 5);
  ctx.fillStyle = "rgba(11, 205, 245, 0.8)";
  ctx.fillRect(x + 20, y + 13, w - 40, 3);
  ctx.fillRect(x + 18, y + h - 19, w - 36, 2);
  for (let px = x + 24; px < x + w - 16; px += 48) drawWallLight(px, y + 14);
  if (label) drawSign(x + w / 2 - 88, y + 18, 176, label);
}

function drawSign(x, y, w, label) {
  ctx.fillStyle = "#06151f";
  ctx.fillRect(x, y, w, 28);
  ctx.strokeStyle = "#0de8f4";
  ctx.lineWidth = 2;
  ctx.strokeRect(x + 2, y + 2, w - 4, 24);
  ctx.fillStyle = "#16f4ef";
  ctx.font = "15px monospace";
  ctx.textAlign = "center";
  ctx.fillText(label.slice(0, 22), x + w / 2, y + 19);
}

function drawWallLight(x, y) {
  ctx.fillStyle = "#b7c7d5";
  ctx.fillRect(x, y, 26, 10);
  ctx.fillStyle = "#31e3ff";
  ctx.fillRect(x + 4, y + 3, 18, 4);
}

function drawCentralGate(x, y, w, h) {
  ctx.fillStyle = "#edf2f6";
  ctx.fillRect(x, y, w, h);
  ctx.fillStyle = "#c9d1dc";
  ctx.fillRect(x + 16, y + 16, w - 32, h - 32);
  ctx.fillStyle = "#f8fbff";
  ctx.fillRect(x + 32, y + 22, w - 64, h - 44);
  ctx.fillStyle = "#778596";
  ctx.fillRect(x + w / 2 - 32, y + 34, 64, 76);
  ctx.fillStyle = "#dfe7ef";
  ctx.fillRect(x + w / 2 - 24, y + 40, 48, 64);
  ctx.strokeStyle = "#18dbff";
  ctx.lineWidth = 3;
  ctx.strokeRect(x + w / 2 - 20, y + 44, 40, 56);
  ctx.strokeStyle = "#18dbff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.ellipse(x + w / 2, y + 74, 18, 7, 0, 0, Math.PI * 2);
  ctx.ellipse(x + w / 2, y + 74, 18, 7, Math.PI / 3, 0, Math.PI * 2);
  ctx.ellipse(x + w / 2, y + 74, 18, 7, -Math.PI / 3, 0, Math.PI * 2);
  ctx.stroke();
  ctx.fillStyle = "#18dbff";
  ctx.fillRect(x + w / 2 - 3, y + 71, 6, 6);
}

function drawAtrium(x, y, w, h) {
  ctx.fillStyle = "#eef4f2";
  ctx.fillRect(x, y, w, h);
  ctx.fillStyle = "#d6dde6";
  ctx.fillRect(x + 16, y + 12, w - 32, h - 24);
  ctx.fillStyle = "#f8fbff";
  ctx.fillRect(x + 36, y + 20, w - 72, h - 40);
  drawPlanterBox(x + 18, y + 38);
  drawPlanterBox(x + w - 54, y + 38);
  drawWaterPanel(x + 28, y + h - 56);
  drawWaterPanel(x + w - 60, y + h - 56);
  ctx.fillStyle = "#748290";
  ctx.fillRect(x + w / 2 - 34, y + 50, 68, 74);
  ctx.fillStyle = "#dfe7ee";
  ctx.fillRect(x + w / 2 - 28, y + 56, 56, 62);
  ctx.strokeStyle = "#19dfff";
  ctx.lineWidth = 2;
  ctx.strokeRect(x + w / 2 - 22, y + 62, 44, 50);
}

function drawPlanterBox(x, y) {
  ctx.fillStyle = "#d8dee7";
  ctx.fillRect(x, y, 36, 58);
  ctx.fillStyle = "#5cab60";
  for (let i = 0; i < 5; i += 1) {
    ctx.fillRect(x + 8 + (i % 2) * 9, y + 12 + i * 7, 13, 9);
  }
}

function drawWaterPanel(x, y) {
  ctx.fillStyle = "#d8dee7";
  ctx.fillRect(x, y, 32, 46);
  ctx.fillStyle = "#55d8ff";
  ctx.fillRect(x + 6, y + 7, 20, 32);
  ctx.fillStyle = "rgba(255,255,255,0.5)";
  ctx.fillRect(x + 10, y + 12, 12, 2);
}

function drawConsoleWall(x, y, count) {
  for (let i = 0; i < count; i += 1) {
    drawScreen(x + i * 34, y + (i % 2) * 4, 28, 34, i % 3 === 0 ? "#21e48a" : "#12cfff");
  }
}

function drawServerWall(x, y, count) {
  for (let i = 0; i < count; i += 1) {
    const px = x + i * 31;
    ctx.fillStyle = "#233142";
    ctx.fillRect(px, y, 24, 72);
    ctx.fillStyle = "#0b1b29";
    ctx.fillRect(px + 4, y + 5, 16, 62);
    for (let j = 0; j < 7; j += 1) {
      ctx.fillStyle = j % 2 ? "#21e4ff" : "#447bff";
      ctx.fillRect(px + 7, y + 10 + j * 8, 10, 2);
    }
  }
}

function drawScreen(x, y, w, h, color) {
  ctx.fillStyle = "#1f2e3d";
  ctx.fillRect(x, y, w, h);
  ctx.fillStyle = "#07151f";
  ctx.fillRect(x + 3, y + 4, w - 6, h - 8);
  ctx.fillStyle = color;
  ctx.fillRect(x + 7, y + 8, w - 14, 3);
  ctx.fillRect(x + 7, y + 15, 4, 9);
  ctx.fillRect(x + 14, y + 18, w - 22, 2);
}

function drawReactionTank(x, y, color, height = 64) {
  ctx.fillStyle = "#7d8998";
  ctx.fillRect(x - 8, y - 4, 42, height + 12);
  ctx.fillStyle = "#273544";
  ctx.fillRect(x - 4, y, 34, height + 4);
  ctx.fillStyle = "rgba(220, 250, 255, 0.62)";
  ctx.fillRect(x + 2, y + 7, 22, height - 8);
  ctx.fillStyle = color;
  ctx.fillRect(x + 5, y + height * 0.42, 16, height * 0.48);
  ctx.fillStyle = "rgba(255,255,255,0.7)";
  ctx.fillRect(x + 8, y + 11, 4, height - 16);
  ctx.fillStyle = "#10dfff";
  ctx.fillRect(x, y - 7, 26, 5);
  ctx.fillRect(x, y + height + 4, 26, 5);
}

function drawLongConsole(x, y, w) {
  ctx.fillStyle = "#8f99a8";
  ctx.fillRect(x, y + 26, w, 18);
  ctx.fillStyle = "#f4f7fb";
  ctx.fillRect(x + 6, y + 6, w - 12, 30);
  ctx.fillStyle = "#273747";
  ctx.fillRect(x + 10, y + 10, w - 20, 20);
  for (let i = 0; i < Math.floor(w / 32); i += 1) {
    ctx.fillStyle = i % 2 ? "#18dfff" : "#815fff";
    ctx.fillRect(x + 18 + i * 28, y + 15, 18, 4);
  }
}

function drawBenchCluster(x, y, color) {
  drawLongConsole(x, y, 116);
  drawScreen(x + 10, y - 42, 34, 38, color);
  drawHoloTable(x + 64, y - 20, "#2fd8ff");
}

function drawSmallDevices(x, y, count) {
  for (let i = 0; i < count; i += 1) {
    const px = x + (i % 3) * 32;
    const py = y + Math.floor(i / 3) * 34;
    ctx.fillStyle = "#9aa5b2";
    ctx.fillRect(px, py, 22, 24);
    ctx.fillStyle = "#1e2d3b";
    ctx.fillRect(px + 4, py + 4, 14, 14);
    ctx.fillStyle = i % 2 ? "#24e6ff" : "#f5b947";
    ctx.fillRect(px + 8, py + 8, 6, 6);
  }
}

function drawRobotArm(x, y) {
  ctx.strokeStyle = "#f4ae2f";
  ctx.lineWidth = 8;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(x, y + 42);
  ctx.lineTo(x + 22, y + 18);
  ctx.lineTo(x + 48, y + 28);
  ctx.stroke();
  ctx.fillStyle = "#5b6675";
  ctx.fillRect(x - 12, y + 40, 24, 18);
  ctx.fillStyle = "#24dfff";
  ctx.beginPath();
  ctx.arc(x + 52, y + 30, 9, 0, Math.PI * 2);
  ctx.fill();
}

function drawDrone(x, y) {
  ctx.fillStyle = "#dce6ef";
  ctx.fillRect(x, y, 44, 18);
  ctx.fillStyle = "#2bdcff";
  ctx.fillRect(x + 9, y + 5, 26, 8);
  ctx.fillStyle = "#6f7a88";
  ctx.fillRect(x - 8, y + 4, 8, 8);
  ctx.fillRect(x + 44, y + 4, 8, 8);
}

function drawCrystalAnalyzer(x, y) {
  drawScreen(x, y, 72, 58, "#20dfff");
  ctx.fillStyle = "#53e4ff";
  ctx.beginPath();
  ctx.moveTo(x + 36, y + 8);
  ctx.lineTo(x + 54, y + 30);
  ctx.lineTo(x + 36, y + 50);
  ctx.lineTo(x + 18, y + 30);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = "rgba(255,255,255,0.62)";
  ctx.fillRect(x + 33, y + 16, 5, 26);
}

function drawReagentShelves(x, y) {
  ctx.fillStyle = "#7c8796";
  ctx.fillRect(x, y, 82, 80);
  ctx.fillStyle = "#f7f9fc";
  ctx.fillRect(x + 5, y + 5, 72, 70);
  for (let row = 0; row < 3; row += 1) {
    for (let col = 0; col < 5; col += 1) {
      ctx.fillStyle = ["#ff8f5a", "#5be7ff", "#22e38b", "#8d67ff", "#f6c44a"][(row + col) % 5];
      ctx.fillRect(x + 12 + col * 12, y + 14 + row * 18, 7, 13);
    }
  }
}

function drawHubTable(x, y) {
  ctx.fillStyle = "#7d8998";
  ctx.fillRect(x, y + 56, 140, 34);
  ctx.fillStyle = "#f5f8fb";
  ctx.fillRect(x + 8, y + 24, 124, 48);
  ctx.strokeStyle = "#19dfff";
  ctx.lineWidth = 3;
  ctx.strokeRect(x + 18, y + 32, 104, 30);
  ctx.fillStyle = "rgba(29, 209, 255, 0.44)";
  ctx.beginPath();
  ctx.arc(x + 70, y + 34, 30, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#19dfff";
  ctx.fillRect(x + 64, y + 2, 12, 48);
}

function drawConsolePod(x, y) {
  drawScreen(x, y, 40, 48, "#15dfff");
  ctx.fillStyle = "#8d98a7";
  ctx.fillRect(x - 6, y + 48, 52, 10);
}

function drawHouse() {}

function drawConsoleBank(x, y, count) {
  for (let i = 0; i < count; i += 1) {
    const px = x + i * 30;
    ctx.fillStyle = "#101923";
    ctx.fillRect(px, y, 22, 28);
    ctx.fillStyle = "#263747";
    ctx.fillRect(px + 2, y + 3, 18, 22);
    ctx.fillStyle = "#0fd0ff";
    ctx.fillRect(px + 5, y + 6, 12, 4);
    ctx.fillStyle = "#28f0a1";
    ctx.fillRect(px + 5, y + 14, 5, 3);
    ctx.fillStyle = "#8b66ff";
    ctx.fillRect(px + 12, y + 14, 5, 3);
  }
}

function drawServerRack(x, y) {
  ctx.fillStyle = "#0f1720";
  ctx.fillRect(x, y, 18, 46);
  ctx.fillStyle = "#273444";
  ctx.fillRect(x + 2, y + 2, 14, 42);
  for (let i = 0; i < 5; i += 1) {
    ctx.fillStyle = i % 2 ? "#19e8f4" : "#326bff";
    ctx.fillRect(x + 5, y + 6 + i * 7, 8, 2);
  }
}

function drawHoloTable(x, y, color) {
  ctx.fillStyle = "#101923";
  ctx.fillRect(x - 4, y + 12, 48, 24);
  ctx.fillStyle = "#2c4052";
  ctx.fillRect(x, y + 8, 40, 22);
  ctx.fillStyle = color;
  ctx.fillRect(x + 10, y + 2, 20, 6);
  ctx.fillStyle = "rgba(42, 221, 255, 0.36)";
  ctx.fillRect(x + 7, y - 8, 26, 16);
}

function drawHoloBoard(x, y, w) {
  ctx.fillStyle = "#07151d";
  ctx.fillRect(x, y, w, 22);
  ctx.strokeStyle = "#12e1ee";
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y, w, 22);
  ctx.fillStyle = "#13e7f0";
  for (let i = 0; i < 5; i += 1) ctx.fillRect(x + 10 + i * 22, y + 7 + (i % 2) * 5, 14, 2);
}

function drawHoloGlobe(x, y, color) {
  ctx.fillStyle = "#101923";
  ctx.fillRect(x - 14, y + 12, 48, 22);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x + 10, y + 8, 18, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "rgba(255,255,255,0.45)";
  ctx.fillRect(x + 3, y - 2, 16, 3);
  ctx.fillRect(x - 2, y + 9, 24, 2);
}

function drawReagentCrate(x, y, color) {
  ctx.fillStyle = "#171e27";
  ctx.fillRect(x, y, 24, 24);
  ctx.fillStyle = "#3a4553";
  ctx.fillRect(x + 3, y + 3, 18, 18);
  ctx.fillStyle = color;
  ctx.fillRect(x + 7, y + 7, 10, 10);
}

function drawPlant(plant) {
  const x = Math.round(plant.x);
  const y = Math.round(plant.y);
  const liquid = {
    emberglass: plant.bloom ? "#ff8f5a" : "#ff623f",
    moonsalt: plant.bloom ? "#dfe9ff" : "#8daeff",
    verdigris: plant.bloom ? "#43e98e" : "#27b976"
  }[plant.species];
  ctx.fillStyle = "rgba(73, 92, 112, 0.28)";
  ctx.fillRect(x - 10, y + 10, 20, 5);
  ctx.fillStyle = "#73808f";
  ctx.fillRect(x - 8, y - 10, 16, 24);
  ctx.fillStyle = "#e7fbff";
  ctx.fillRect(x - 5, y - 7, 10, 18);
  ctx.fillStyle = liquid;
  ctx.fillRect(x - 4, y + 3 - Math.floor(plant.age * 12), 8, 11 + Math.floor(plant.age * 7));
  ctx.fillStyle = "rgba(255,255,255,0.72)";
  ctx.fillRect(x - 2, y - 5, 2, 11);
  ctx.fillStyle = "#4d5a68";
  ctx.fillRect(x - 6, y - 14, 12, 5);
  if (plant.bloom) {
    ctx.fillStyle = liquid;
    ctx.fillRect(x - 12, y - 21, 4, 4);
    ctx.fillRect(x + 8, y - 18, 3, 3);
    ctx.fillRect(x - 1, y - 25, 2, 2);
  }
}

function drawArtifact(artifact) {
  const x = Math.round(artifact.x);
  const y = Math.round(artifact.y);
  ctx.fillStyle = "#203040";
  ctx.fillRect(x - 8, y - 8, 16, 16);
  ctx.fillStyle = artifact.type === "note" ? "#1de1ff" : "#8b6cff";
  ctx.fillRect(x - 5, y - 5, 10, 10);
  ctx.fillStyle = "rgba(255,255,255,0.65)";
  ctx.fillRect(x - 3, y - 3, 4, 2);
}

function drawAgent(agent) {
  const x = Math.round(agent.x);
  const y = Math.round(agent.y);
  const hair = agent.cap;
  ctx.fillStyle = "rgba(67, 82, 96, 0.32)";
  ctx.fillRect(x - 10, y + 13, 20, 5);
  ctx.fillStyle = "#263446";
  ctx.fillRect(x - 8, y + 4, 6, 11);
  ctx.fillRect(x + 2, y + 4, 6, 11);
  ctx.fillStyle = "#1e2b3c";
  ctx.fillRect(x - 8, y + 14, 6, 3);
  ctx.fillRect(x + 2, y + 14, 6, 3);
  ctx.fillStyle = "#f7fbff";
  ctx.fillRect(x - 9, y - 3, 18, 16);
  ctx.fillStyle = "#dce8ef";
  ctx.fillRect(x - 7, y + 8, 5, 6);
  ctx.fillRect(x + 2, y + 8, 5, 6);
  ctx.fillStyle = agent.color;
  ctx.fillRect(x - 5, y + 1, 10, 4);
  ctx.fillStyle = agent.skin;
  ctx.fillRect(x - 7, y - 16, 14, 13);
  ctx.fillStyle = shade(agent.skin, -18);
  ctx.fillRect(x - 7, y - 6, 14, 2);
  drawScientistHair(agent, x, y, hair);
  ctx.fillStyle = "#243044";
  if (agent.direction === "left") {
    ctx.fillRect(x - 5, y - 12, 2, 2);
  } else if (agent.direction === "right") {
    ctx.fillRect(x + 3, y - 12, 2, 2);
  } else if (agent.direction === "up") {
    ctx.fillRect(x - 5, y - 14, 10, 3);
  } else {
    ctx.fillRect(x - 4, y - 12, 2, 2);
    ctx.fillRect(x + 3, y - 12, 2, 2);
  }
  ctx.fillStyle = "#35dfff";
  if (agent.role.includes("scribe")) ctx.fillRect(x + 8, y - 6, 7, 9);
  if (agent.role.includes("watcher")) ctx.fillRect(x - 14, y - 4, 6, 11);
  if (agent.speech && world.tick < agent.speechUntil) drawSpeech(x, y - 28, agent.speech);
}

function drawScientistHair(agent, x, y, hair) {
  ctx.fillStyle = "#1f2939";
  ctx.fillRect(x - 8, y - 20, 16, 7);
  ctx.fillStyle = hair;
  const style = agent.id % 5;
  if (style === 0) {
    ctx.fillRect(x - 8, y - 20, 16, 6);
    ctx.fillRect(x - 10, y - 17, 6, 8);
  } else if (style === 1) {
    ctx.fillRect(x - 9, y - 22, 18, 8);
    ctx.fillRect(x + 5, y - 17, 5, 9);
  } else if (style === 2) {
    ctx.fillRect(x - 7, y - 23, 14, 7);
    ctx.fillRect(x - 4, y - 25, 8, 4);
  } else if (style === 3) {
    ctx.fillRect(x - 8, y - 21, 16, 8);
    ctx.fillRect(x - 6, y - 24, 12, 4);
  } else {
    ctx.fillRect(x - 10, y - 20, 20, 6);
    ctx.fillRect(x - 8, y - 16, 5, 7);
    ctx.fillRect(x + 3, y - 16, 5, 7);
  }
  ctx.fillStyle = "rgba(255,255,255,0.45)";
  ctx.fillRect(x - 4, y - 21, 4, 2);
}

function shade(hex, amount) {
  const value = Number.parseInt(hex.slice(1), 16);
  const r = clamp((value >> 16) + amount, 0, 255);
  const g = clamp(((value >> 8) & 255) + amount, 0, 255);
  const b = clamp((value & 255) + amount, 0, 255);
  return `rgb(${r}, ${g}, ${b})`;
}

function drawSpeech(x, y, text) {
  const short = text.length > 18 ? `${text.slice(0, 16)}...` : text;
  const width = Math.max(42, short.length * 6 + 12);
  ctx.fillStyle = "#fffdf5";
  ctx.fillRect(x - width / 2, y - 18, width, 15);
  ctx.fillStyle = "#2c2740";
  ctx.font = "10px monospace";
  ctx.textAlign = "center";
  ctx.fillText(short, x, y - 7);
}

function drawWeather() {
  if (world.weather === "acid") {
    ctx.fillStyle = "rgba(113, 211, 120, 0.28)";
    for (let i = 0; i < 80; i += 1) {
      const x = (i * 53 + world.tick * 2) % W;
      const y = (i * 97 + world.tick * 5) % H;
      ctx.fillRect(x, y, 3, 7);
    }
  }
  if (world.weather === "vapor") {
    ctx.fillStyle = "rgba(255,255,255,0.24)";
    for (let i = 0; i < 5; i += 1) {
      ctx.fillRect((world.tick + i * 220) % W - 120, 80 + i * 86, 180, 16);
    }
  }
  if (world.weather === "warmth") {
    ctx.fillStyle = "rgba(255, 193, 97, 0.16)";
    ctx.fillRect(0, 0, W, H);
  }
}

function drawMemeVisualization() {
  if (!world.memeVisualization) return;
  drawMemeEdges();
  drawMemeOrbs();
}

function drawMemeEdges() {
  const maxAge = 7200;
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (const edge of world.memeEdges) {
    const age = world.tick - edge.tick;
    if (age < 0 || age > maxAge) continue;
    const from = agents[edge.fromId];
    const to = agents[edge.toId];
    if (!from || !to) continue;
    const alpha = clamp(1 - age / maxAge, 0.24, 1);
    const pulse = Math.sin((world.tick - edge.tick) * 0.1) * 0.5 + 0.5;
    const start = memeOrbPoint(from);
    const end = memeOrbPoint(to);
    const mx = (start.x + end.x) / 2;
    const my = (start.y + end.y) / 2 - clamp(distance(start, end) * 0.12, 18, 80);
    const stroke = edge.nature === "black" ? "#f6efd2" : edge.color;

    ctx.globalAlpha = alpha * 0.42;
    ctx.strokeStyle = "rgba(255, 252, 229, 0.95)";
    ctx.lineWidth = 24 + pulse * 6;
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.quadraticCurveTo(mx, my, end.x, end.y);
    ctx.stroke();

    ctx.globalAlpha = alpha * 0.92;
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 9 + pulse * 3;
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.quadraticCurveTo(mx, my, end.x, end.y);
    ctx.stroke();

    ctx.globalAlpha = alpha * 0.95;
    drawMemeNode(start.x, start.y, edge.color, 10 + pulse * 2, false);
    drawMemeNode(end.x, end.y, edge.color, 10 + pulse * 2, false);
  }
  ctx.restore();
}

function drawMemeOrbs() {
  for (const agent of agents) {
    const active = world.tick < agent.currentMeme.until;
    const color = active ? agent.currentMeme.color : memeColors.black;
    const point = memeOrbPoint(agent);
    const pulse = active ? Math.sin(world.tick * 0.12 + agent.id) * 1.5 : 0;
    drawMemeNode(point.x, point.y, color, active ? 11 + pulse : 8, true);
  }
}

function memeOrbPoint(agent) {
  return { x: agent.x, y: agent.y - 44 };
}

function drawMemeNode(x, y, color, radius, outlined) {
  ctx.save();
  ctx.fillStyle = "rgba(255,255,245,0.72)";
  ctx.beginPath();
  ctx.arc(x, y, radius + 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#252330";
  ctx.beginPath();
  ctx.arc(x, y, radius + 2, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "rgba(255,255,255,0.65)";
  ctx.beginPath();
  ctx.arc(x - radius * 0.32, y - radius * 0.35, Math.max(2, radius * 0.22), 0, Math.PI * 2);
  ctx.fill();
  if (outlined) {
    ctx.strokeStyle = "rgba(255,255,245,0.78)";
    ctx.lineWidth = 2;
    ctx.stroke();
  }
  ctx.restore();
}

function draw() {
  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < cols; x += 1) drawTile(x, y, terrain[y][x]);
  }

  landmarks.forEach(drawLandmark);
  houses.forEach(drawHouse);
  artifacts.forEach(drawArtifact);
  plants.forEach(drawPlant);
  agents
    .slice()
    .sort((a, b) => a.y - b.y)
    .forEach(drawAgent);
  drawWeather();

  const hour = (world.tick % 1440) / 60;
  if (hour < 5 || hour > 20) {
    ctx.fillStyle = "rgba(35,43,83,0.28)";
    ctx.fillRect(0, 0, W, H);
  }

  drawMemeVisualization();
}

function aggregateBeliefs() {
  if (world.minedRules.length) {
    return world.minedRules.map((rule) => ({
      id: rule.id,
      claim: rule.claim,
      domain: rule.cluster,
      parent: rule.parent,
      confidence: rule.confidence,
      neural: rule.neural,
      evidence: rule.evidence,
      sources: [labelFeature(rule.feature), labelOutcome(rule.outcome)],
      feature: rule.feature,
      outcome: rule.outcome,
      lift: rule.lift
    })).sort((a, b) => b.confidence - a.confidence);
  }

  const candidates = [...world.ruleStats.values()]
    .filter((stat) => stat.seen >= 2)
    .map((stat) => {
      const p = stat.hits / stat.seen;
      const base = stat.absentSeen ? stat.absentHits / stat.absentSeen : 0.5;
      return {
        id: `${stat.feature}->${stat.outcome}`,
        claim: `${labelFeature(stat.feature)} may predict ${labelOutcome(stat.outcome)}.`,
        domain: clusterForFeature(stat.feature),
        parent: stat.feature.split("=")[0],
        confidence: clamp(0.22 + Math.max(0, p - base), 0, 0.58),
        neural: p,
        evidence: stat.seen,
        sources: ["early notes"],
        feature: stat.feature,
        outcome: stat.outcome,
        lift: p - base
      };
    })
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 10);

  if (candidates.length) return candidates;

  return outcomeDefs.map((outcome, index) => {
    const models = agents.map((agent) => agent.models[outcome.id]);
    return {
      id: `seed-${outcome.id}`,
      claim: `Agents are gathering first ${outcome.label} predictions.`,
      domain: index === 0 ? "reagents" : "communication",
      parent: "seed",
      confidence: models.reduce((sum, model) => sum + model.confidence, 0) / models.length,
      neural: models.reduce((sum, model) => sum + model.neural, 0) / models.length,
      evidence: models.reduce((sum, model) => sum + model.evidence, 0),
      sources: ["unclustered"],
      feature: "seed",
      outcome: outcome.id,
      lift: 0
    };
  });
}

function dynamicDomains(beliefs) {
  const ids = [...new Set(beliefs.map((belief) => belief.domain))];
  const fallback = ids.length ? ids : ["reagents", "catalysts", "apparatus", "communication"];
  return fallback.map((id, index) => {
    const claims = beliefs.filter((belief) => belief.domain === id);
    const accepted = claims.filter((belief) => world.discoveries.includes(belief.id)).length;
    const confidence = claims.length ? claims.reduce((sum, belief) => sum + belief.confidence, 0) / claims.length : 0;
    const evidence = claims.reduce((sum, belief) => sum + belief.evidence, 0);
    const percent = Math.round(clamp(confidence * 72 + accepted * 12 + Math.min(evidence, 60) * 0.32, 0, 100));
    return {
      id,
      name: titleCase(id),
      color: clusterPalette[index % clusterPalette.length],
      accepted,
      confidence,
      evidence,
      goal: Math.max(1, claims.length),
      percent
    };
  });
}

function renderDiscoveryTree(beliefs, domains) {
  const rootX = 18;
  const groupX = 168;
  const ruleX = 390;
  const rowHeight = 96;
  const ruleHeight = 82;
  const edges = [];
  const nodes = [];
  let cursorY = 20;

  for (const domain of domains) {
    const domainRules = beliefs.filter((belief) => belief.domain === domain.id);
    const groups = [...new Set(domainRules.map((belief) => belief.parent ?? "pattern"))]
      .map((parent) => {
        const rules = domainRules
          .filter((belief) => (belief.parent ?? "pattern") === parent)
          .sort((a, b) => b.confidence - a.confidence);
        return { parent, rules, height: Math.max(rowHeight, rules.length * ruleHeight) };
      });
    const blockHeight = Math.max(rowHeight, groups.reduce((sum, block) => sum + block.height, 0));
    const rootY = cursorY + blockHeight / 2 - 29;

    nodes.push(`
      <article class="tree-node root" style="left:${rootX}px; top:${rootY}px">
        <strong>${domain.name}</strong>
        <div class="tags"><span class="tag">${domain.percent}%</span></div>
      </article>
    `);

    let blockY = cursorY;
    for (const group of groups) {
      const y = blockY + group.height / 2 - 32;
      const topRule = group.rules[0];
      const groupAccepted = group.rules.some((rule) => world.discoveries.includes(rule.id));
      const edgeColor = groupAccepted ? "#5d9b62" : "#c8a25b";
      edges.push(`<path d="M ${rootX + 122} ${rootY + 29} C ${rootX + 148} ${rootY + 29}, ${groupX - 32} ${y + 29}, ${groupX} ${y + 29}" stroke="${edgeColor}" stroke-width="3" fill="none" stroke-linecap="round" />`);
      nodes.push(`
        <article class="tree-node ${groupAccepted ? "discovered" : "hypothesis"}" style="left:${groupX}px; top:${y}px">
          <strong>${titleCase(group.parent)}</strong>
          <small>${group.rules.length} pattern${group.rules.length === 1 ? "" : "s"} under test</small>
          <div class="meter"><span style="width:${Math.round((topRule?.confidence ?? 0) * 100)}%; background:${groupAccepted ? "#65a969" : "#c9a96e"}"></span></div>
          <div class="tags">
            <span class="tag">${Math.round((topRule?.confidence ?? 0) * 100)}% conf</span>
            <span class="tag">${group.rules.reduce((sum, rule) => sum + rule.evidence, 0)} notes</span>
          </div>
        </article>
      `);

      group.rules.forEach((rule, ruleIndex) => {
        const accepted = world.discoveries.includes(rule.id);
        const ruleY = blockY + ruleIndex * ruleHeight + 2;
        const childEdgeColor = accepted ? "#5d9b62" : "#c8a25b";
        edges.push(`<path d="M ${groupX + 178} ${y + 29} C ${groupX + 210} ${y + 29}, ${ruleX - 34} ${ruleY + 29}, ${ruleX} ${ruleY + 29}" stroke="${childEdgeColor}" stroke-width="3" fill="none" stroke-linecap="round" />`);
        nodes.push(`
          <article class="tree-node ${accepted ? "discovered" : "hypothesis"}" style="left:${ruleX}px; top:${ruleY}px">
            <strong>${accepted ? "Discovered" : "Hypothesis"}</strong>
            <small>${rule.claim}</small>
            <div class="meter"><span style="width:${Math.round(rule.confidence * 100)}%; background:${accepted ? "#65a969" : "#c9a96e"}"></span></div>
            <div class="tags">
              <span class="tag">${Math.round(rule.confidence * 100)}% conf</span>
              <span class="tag">${Math.round(rule.neural * 100)}% net</span>
            </div>
          </article>
        `);
      });

      blockY += group.height;
    }

    cursorY += blockHeight + 18;
  }

  return `
    <div class="tree-canvas" style="height:${cursorY + 20}px">
      <svg viewBox="0 0 610 ${cursorY + 20}" style="height:${cursorY + 20}px" aria-hidden="true">
        ${edges.join("")}
      </svg>
      ${nodes.join("")}
    </div>
  `;
}

function recordUnderstandingHistory(domains, overall) {
  if (world.lastHistoryTick === world.tick) return;
  world.lastHistoryTick = world.tick;
  world.history.push({
    tick: world.tick,
    day: world.day,
    overall,
    domains: Object.fromEntries(domains.map((domain) => [domain.id, domain.percent]))
  });
}

function renderUnderstandingGraph(domains, overall) {
  recordUnderstandingHistory(domains, overall);
  const width = 620;
  const height = 220;
  const pad = { left: 34, right: 12, top: 14, bottom: 26 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const history = world.history.length > 1 ? world.history : [
    { tick: 0, day: world.day, overall: 0, domains: Object.fromEntries(domains.map((domain) => [domain.id, 0])) },
    world.history[0] ?? { tick: world.tick, day: world.day, overall, domains: Object.fromEntries(domains.map((domain) => [domain.id, domain.percent])) }
  ];
  const maxTick = Math.max(1, world.tick, ...history.map((entry) => entry.tick));
  const xFor = (tick) => pad.left + (tick / maxTick) * plotW;
  const yFor = (value) => pad.top + plotH - (value / 100) * plotH;
  const grid = [0, 25, 50, 75, 100].map((value) => `
    <line x1="${pad.left}" y1="${yFor(value)}" x2="${width - pad.right}" y2="${yFor(value)}" stroke="#dfcea5" stroke-width="1" />
    <text x="6" y="${yFor(value) + 4}" fill="#7a6f80" font-size="10">${value}</text>
  `).join("");
  const paths = domains.map((domain) => {
    const points = history.map((entry) => `${xFor(entry.tick)},${yFor(entry.domains[domain.id] ?? 0)}`).join(" ");
    return `<polyline points="${points}" fill="none" stroke="${domain.color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />`;
  }).join("");
  const overallPoints = history.map((entry) => `${xFor(entry.tick)},${yFor(entry.overall)}`).join(" ");
  const legend = domains.map((domain) => `
    <span class="legend-item"><span class="legend-swatch" style="background:${domain.color}"></span>${domain.name} ${domain.percent}%</span>
  `).join("");

  return `
    <svg class="understanding-graph" viewBox="0 0 ${width} ${height}" role="img" aria-label="World understanding over time">
      ${grid}
      <line x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}" stroke="#bfa66e" stroke-width="2" />
      <line x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}" stroke="#bfa66e" stroke-width="2" />
      ${paths}
      <polyline points="${overallPoints}" fill="none" stroke="#2c2740" stroke-width="2" stroke-dasharray="5 5" stroke-linejoin="round" stroke-linecap="round" />
      <text x="${pad.left}" y="${height - 7}" fill="#7a6f80" font-size="11">0</text>
      <text x="${width - 62}" y="${height - 7}" fill="#7a6f80" font-size="11">Day ${world.day}</text>
    </svg>
    <div class="domain-legend">${legend}<span class="legend-item"><span class="legend-swatch" style="background:#2c2740"></span>Overall ${overall}%</span></div>
  `;
}

function renderPanels() {
  const beliefs = aggregateBeliefs();
  const domains = dynamicDomains(beliefs);
  const overall = Math.round(domains.reduce((sum, domain) => sum + domain.percent, 0) / domains.length);

  ui.overall.textContent = `${overall}% charted`;
  ui.domains.innerHTML = renderUnderstandingGraph(domains, overall);

  ui.tree.innerHTML = renderDiscoveryTree(beliefs, domains);

  ui.beliefs.innerHTML = beliefs.slice(0, 5).map((belief) => `
    <article class="belief">
      <strong>${belief.claim}</strong>
      <div class="meter"><span style="width:${Math.round(belief.confidence * 100)}%"></span></div>
      <div class="tags">
        <span class="tag">${titleCase(belief.domain)}</span>
        <span class="tag">${Math.round(belief.confidence * 100)}% consensus</span>
        <span class="tag">${Math.round(belief.neural * 100)}% neural</span>
        <span class="tag">${belief.evidence} notes</span>
        <span class="tag">${belief.sources.join(", ")}</span>
      </div>
    </article>
  `).join("");

  const averageReward = world.predictions ? world.reward / world.predictions : 0;
  ui.reward.textContent = `${averageReward.toFixed(2)} reward`;
  ui.communications.innerHTML = world.communications.map((item) => `
    <article class="communication">
      <strong>${item.from} -> ${item.to}</strong>
      <div class="tags">
        <span class="tag">${item.claim}</span>
        <span class="tag">${Math.round(item.confidence * 100)}% said</span>
        <span class="tag">+${Math.round(Math.max(0, item.delta) * 100)} listener</span>
      </div>
    </article>
  `).join("");

  const top = beliefs[0];
  ui.consensus.textContent = top ? `${Math.round(top.confidence * 100)}%: ${top.claim}` : "forming first questions";
}

function renderHud() {
  ui.day.textContent = `Day ${world.day}`;
  ui.season.textContent = seasons[world.seasonIndex];
  ui.weather.textContent = world.weather;
}

function loop() {
  for (let i = 0; i < 2; i += 1) update();
  draw();
  if (world.tick % 30 === 0) renderHud();
  if (world.tick % 120 === 0) renderPanels();
  requestAnimationFrame(loop);
}

function setInitialWorldScroll() {
  const wrap = document.querySelector(".canvas-wrap");
  if (!wrap) return;
  wrap.scrollLeft = Math.max(0, 420 - wrap.clientWidth * 0.08);
  wrap.scrollTop = Math.max(0, 250 - wrap.clientHeight * 0.08);
}

setWeather();
addLog("The alchemy commons opened its benches and began asking which mixtures react.");
agents.forEach(chooseTarget);
renderHud();
renderPanels();
draw();
setInitialWorldScroll();
loop();
