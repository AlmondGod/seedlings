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

const houses = [
  { id: "mika", x: 13, y: 13, w: 4, h: 3, roof: "#d96f5d", wall: "#ffe2a8", owner: "Mika" },
  { id: "taro", x: 18, y: 13, w: 4, h: 3, roof: "#6fa765", wall: "#f6dba6", owner: "Taro" },
  { id: "nori", x: 13, y: 17, w: 4, h: 3, roof: "#5d8edb", wall: "#f7e4bb", owner: "Nori" },
  { id: "piko", x: 18, y: 17, w: 4, h: 3, roof: "#e4b44e", wall: "#ffe1b5", owner: "Piko" },
  { id: "sumi", x: 23, y: 17, w: 4, h: 3, roof: "#9f80bd", wall: "#f4d8bb", owner: "Sumi" },
  { id: "ren", x: 28, y: 17, w: 4, h: 3, roof: "#58aeb8", wall: "#f3ddb8", owner: "Ren" },
  { id: "kiko", x: 33, y: 23, w: 4, h: 3, roof: "#e080a4", wall: "#f6ddbd", owner: "Kiko" },
  { id: "bo", x: 38, y: 23, w: 4, h: 3, roof: "#96714b", wall: "#efd7ad", owner: "Bo" },
  { id: "yui", x: 43, y: 23, w: 4, h: 3, roof: "#f08f54", wall: "#f5dfb9", owner: "Yui" },
  { id: "aki", x: 48, y: 23, w: 4, h: 3, roof: "#74b765", wall: "#efe0b9", owner: "Aki" }
];

const landmarks = [
  { type: "townHall", name: "Commons Hall", x: 13, y: 8, w: 5, h: 4, roof: "#8f6fd6", wall: "#efe0c6" },
  { type: "lab", name: "Synthesis Lab", x: 24, y: 12, w: 6, h: 4, roof: "#5f8fc9", wall: "#e8ece4" },
  { type: "archive", name: "Formula Archive", x: 23, y: 8, w: 5, h: 3, roof: "#a979bd", wall: "#efe2c7" },
  { type: "workshop", name: "Glassworks", x: 4, y: 13, w: 3, h: 2, roof: "#d89b4a", wall: "#ead0a3" },
  { type: "observatory", name: "Fume Tower", x: 49, y: 4, w: 2, h: 2, roof: "#4f6fb4", wall: "#d7e3ec" },
  { type: "shrine", name: "Catalyst Shrine", x: 5, y: 8, w: 2, h: 3, roof: "#d85d63", wall: "#f4cc72" },
  { type: "dock", name: "Solvent Bath", x: 3, y: 5, w: 3, h: 2, roof: "#5aaec8", wall: "#c99b62" },
  { type: "cafe", name: "Memory Refectory", x: 34, y: 9, w: 5, h: 3, roof: "#d28a4d", wall: "#f0d09e" },
  { type: "school", name: "Apprentice Hall", x: 45, y: 12, w: 5, h: 4, roof: "#6aa86c", wall: "#e8dfbc" },
  { type: "greenhouse", name: "Distillery", x: 10, y: 27, w: 6, h: 4, roof: "#6fbfb2", wall: "#d8eee2" },
  { type: "market", name: "Reagent Exchange", x: 31, y: 30, w: 6, h: 4, roof: "#d8b34d", wall: "#efddb3" },
  { type: "theater", name: "Lecture Theater", x: 45, y: 28, w: 6, h: 4, roof: "#cc6d8e", wall: "#ead5c5" }
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
  { type: "blackboard", x: 25.4 * tile, y: 14.2 * tile, text: "?", idea: "questions" },
  { type: "map", x: 14.5 * tile, y: 9.8 * tile, text: "map", idea: "paths" },
  { type: "specimen", x: 24.6 * tile, y: 13.2 * tile, text: "jar", idea: "biology" },
  { type: "bell", x: 14.5 * tile, y: 8.8 * tile, text: "bell", idea: "council" },
  { type: "rod", x: 4.2 * tile, y: 6.1 * tile, text: "fish", idea: "river" }
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
    const home = houses[agent.id % houses.length];
    setTarget(agent, (home.x + home.w / 2) * tile, (home.y + home.h + 0.12) * tile, "sleepwalking home");
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
  if (isDockPixel(x, y)) return true;
  if (isEntrancePixel(x, y)) return true;
  if (isBuildingShellPixel(x, y)) return false;

  const terrainType = terrain[Math.floor(y / tile)]?.[Math.floor(x / tile)];
  return !["channel", "basin", "wall"].includes(terrainType);
}

function isDockPixel(x, y) {
  const dock = landmarks.find((site) => site.type === "dock");
  if (!dock) return false;
  const left = dock.x * tile - 8;
  const top = dock.y * tile + 5;
  const right = (dock.x + dock.w) * tile + 8;
  const bottom = dock.y * tile + 36;
  return x >= left && x <= right && y >= top && y <= bottom;
}

function isEntrancePixel(x, y) {
  return [...houses, ...landmarks.filter((site) => !["dock", "shrine", "observatory"].includes(site.type))]
    .some((site) => {
      const center = (site.x + site.w / 2) * tile;
      const left = center - 18;
      const right = center + 18;
      const top = (site.y + site.h) * tile - 24;
      const bottom = (site.y + site.h) * tile + 7;
      return x >= left && x <= right && y >= top && y <= bottom;
    });
}

function isBuildingShellPixel(x, y) {
  return [...houses, ...landmarks.filter((site) => site.type !== "dock")]
    .some((site) => {
      const left = site.x * tile - 2;
      const top = site.y * tile - 8;
      const right = (site.x + site.w) * tile + 2;
      const bottom = (site.y + site.h) * tile + 3;
      return x >= left && x <= right && y >= top && y <= bottom;
    });
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
    floor: "#b9b3a5",
    crystals: "#c7b4d8",
    bench: "#9a7651",
    path: "#d8bf7a",
    furnace: "#b77c57",
    channel: "#5faed0",
    basin: "#5aaec8",
    archive: "#cdb579",
    wall: "#675d68"
  };
  ctx.fillStyle = palette[type];
  ctx.fillRect(px, py, tile, tile);

  ctx.fillStyle = "rgba(255,255,255,0.16)";
  if ((x * 3 + y + world.day) % 4 === 0) ctx.fillRect(px + 5, py + 6, 4, 4);
  ctx.fillStyle = "rgba(58, 93, 48, 0.12)";
  if ((x + y * 2) % 5 === 0) ctx.fillRect(px + 20, py + 18, 6, 3);

  if (type === "floor" || type === "crystals") {
    ctx.fillStyle = "#8f8791";
    if ((x + y) % 2 === 0) {
      ctx.fillRect(px + 8, py + 22, 8, 2);
      ctx.fillRect(px + 12, py + 14, 2, 8);
    }
    if ((x * 7 + y) % 5 === 0) {
      ctx.fillStyle = "#7a5ed9";
      ctx.fillRect(px + 23, py + 9, 3, 3);
      ctx.fillStyle = "#f6d765";
      ctx.fillRect(px + 26, py + 12, 2, 2);
    }
  }

  if (type === "wall") {
    ctx.fillStyle = "#3f3543";
    ctx.fillRect(px, py, tile, tile);
    ctx.fillStyle = "#6d6270";
    ctx.fillRect(px + 2, py + 2, 28, 12);
    ctx.fillStyle = "#9c8f9f";
    ctx.fillRect(px + 4, py + 4, 11, 4);
    ctx.fillRect(px + 18, py + 7, 9, 3);
    ctx.fillStyle = "#514656";
    ctx.fillRect(px, py + 23, tile, 9);
    ctx.fillStyle = "rgba(255,255,255,0.18)";
    ctx.fillRect(px + 5, py + 3, 5, 3);
  }
  if (type === "path") {
    ctx.fillStyle = "#caa96d";
    ctx.fillRect(px, py + 11, tile, 10);
    ctx.fillRect(px + 11, py, 10, tile);
    ctx.fillStyle = "#ead79b";
    if ((x + y) % 2 === 0) ctx.fillRect(px + 4, py + 14, 7, 3);
    if ((x + y) % 3 === 0) ctx.fillRect(px + 18, py + 6, 3, 7);
  }
  if (type === "archive") {
    ctx.fillStyle = "#bfa266";
    ctx.fillRect(px, py, tile, tile);
    ctx.fillStyle = "#dbc489";
    ctx.fillRect(px + 2, py + 2, 12, 12);
    ctx.fillRect(px + 18, py + 18, 12, 12);
    ctx.fillStyle = "#c8ac70";
    ctx.fillRect(px + 17, py + 2, 13, 12);
    ctx.fillRect(px + 2, py + 18, 12, 12);
  }
  if (type === "bench") {
    ctx.fillStyle = "#7d5b3e";
    for (let i = 0; i < 4; i += 1) ctx.fillRect(px, py + i * 8 + 3, tile, 2);
    ctx.fillStyle = "#af8b5d";
    if ((x + y) % 2 === 0) ctx.fillRect(px + 6, py + 7, 20, 3);
  }
  if (type === "furnace") {
    ctx.fillStyle = "#9e6348";
    ctx.fillRect(px, py, tile, tile);
    ctx.fillStyle = "#d69a62";
    ctx.fillRect(px + 2, py + 2, 12, 12);
    ctx.fillRect(px + 18, py + 18, 12, 12);
    ctx.fillStyle = "rgba(255, 226, 126, 0.42)";
    if ((x + y + Math.floor(world.tick / 20)) % 3 === 0) ctx.fillRect(px + 11, py + 11, 10, 10);
  }
  if (type === "channel" || type === "basin") {
    ctx.fillStyle = type === "channel" ? "#4b98bd" : "#438fab";
    ctx.fillRect(px, py, tile, tile);
    ctx.fillStyle = "#438fab";
    ctx.fillRect(px, py + 24, tile, 8);
    ctx.fillStyle = "#7ed3dc";
    if ((x + y + Math.floor(world.tick / 30)) % 3 === 0) ctx.fillRect(px + 5, py + 10, 18, 3);
    if (type === "channel") {
      ctx.fillStyle = "#8ee2e6";
      ctx.fillRect(px + ((world.tick + x * 9) % 18), py + 17, 10, 2);
      ctx.fillStyle = "#2f806f";
      if ((x + y) % 2 === 0) ctx.fillRect(px + 1, py + 1, 5, 9);
    }
    ctx.fillStyle = "#3f8052";
    if ((x + y) % 4 === 0) ctx.fillRect(px + 2, py + 24, 5, 8);
  }
}

function drawLandmark(site) {
  if (site.type === "shrine") {
    drawShrine(site);
    return;
  }
  if (site.type === "dock") {
    drawDock(site);
    return;
  }
  if (site.type === "observatory") {
    drawObservatory(site);
    return;
  }
  drawInteriorBuilding(site);
}

function drawInteriorBuilding(site) {
  const x = site.x * tile;
  const y = site.y * tile;
  const w = site.w * tile;
  const h = site.h * tile;
  ctx.fillStyle = "rgba(5, 12, 22, 0.34)";
  ctx.fillRect(x + 6, y + h - 1, w, 10);
  ctx.fillStyle = "#151b25";
  ctx.fillRect(x, y, w, h);
  ctx.fillStyle = "#4f5868";
  ctx.fillRect(x, y, w, 8);
  ctx.fillRect(x, y + h - 8, w, 8);
  ctx.fillRect(x, y, 8, h);
  ctx.fillRect(x + w - 8, y, 8, h);
  ctx.fillStyle = "#232b37";
  ctx.fillRect(x + 8, y + 8, w - 16, h - 16);
  ctx.fillStyle = "#303948";
  for (let px = x + 16; px < x + w - 14; px += 24) ctx.fillRect(px, y + 13, 2, h - 26);
  for (let py = y + 18; py < y + h - 14; py += 24) ctx.fillRect(x + 13, py, w - 26, 2);
  ctx.fillStyle = "rgba(18, 225, 238, 0.82)";
  ctx.fillRect(x + 12, y + 10, w - 24, 2);
  ctx.fillRect(x + 12, y + h - 12, w - 24, 2);
  ctx.fillStyle = "#07151d";
  ctx.fillRect(x + w / 2 - 56, y + 9, 112, 18);
  ctx.strokeStyle = "#12e1ee";
  ctx.lineWidth = 2;
  ctx.strokeRect(x + w / 2 - 56, y + 9, 112, 18);
  ctx.fillStyle = "#19f3ef";
  ctx.font = "10px monospace";
  ctx.textAlign = "center";
  ctx.fillText(site.name.toUpperCase().slice(0, 18), x + w / 2, y + 22);

  drawFurniture(site, x, y, w, h);
  ctx.fillStyle = "#111822";
  ctx.fillRect(x + w / 2 - 10, y + h - 11, 20, 10);
  ctx.fillStyle = "#22cff0";
  ctx.fillRect(x + w / 2 - 8, y + h - 10, 16, 2);
}

function drawFurniture(site, x, y, w, h) {
  if (site.type === "lab") {
    drawConsoleBank(x + 18, y + 34, 4);
    drawReactionTank(x + w - 58, y + 36, "#25f2ad");
    drawHoloTable(x + w / 2 - 18, y + h - 48, "#32d7ff");
    return;
  }
  if (site.type === "townHall") {
    drawHoloTable(x + w / 2 - 20, y + 44, "#36e5ff");
    drawConsoleBank(x + 18, y + h - 44, 3);
    return;
  }
  if (site.type === "archive") {
    for (let i = 0; i < 5; i += 1) drawServerRack(x + 16 + i * 28, y + 34);
    drawHoloGlobe(x + w - 46, y + h - 52, "#31cfff");
    return;
  }
  if (site.type === "cafe") {
    drawConsoleBank(x + 18, y + 34, 3);
    drawHoloTable(x + w - 54, y + 54, "#27e8a6");
    return;
  }
  if (site.type === "school") {
    drawConsoleBank(x + 18, y + 36, 4);
    drawHoloBoard(x + 20, y + 28, w - 40);
    return;
  }
  if (site.type === "greenhouse") {
    for (let i = 0; i < 3; i += 1) drawReactionTank(x + 22 + i * 46, y + 36, i % 2 ? "#8d6cff" : "#25f2ad");
    drawConsoleBank(x + 20, y + h - 42, 4);
    return;
  }
  if (site.type === "market") {
    for (let i = 0; i < 4; i += 1) drawReagentCrate(x + 18 + i * 34, y + 42, i % 2 ? "#ff965f" : "#62e2ff");
    drawConsoleBank(x + 18, y + h - 40, 4);
    return;
  }
  if (site.type === "theater") {
    drawHoloBoard(x + 18, y + 30, w - 36);
    drawBenchRow(x + 22, y + h - 42, 5);
    drawBenchRow(x + 22, y + h - 26, 5);
    return;
  }
  drawConsoleBank(x + 18, y + 34, 3);
}

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

function drawReactionTank(x, y, color) {
  ctx.fillStyle = "#0d1420";
  ctx.fillRect(x - 5, y + 4, 34, 48);
  ctx.fillStyle = "#526071";
  ctx.fillRect(x, y, 24, 54);
  ctx.fillStyle = "rgba(210, 247, 255, 0.35)";
  ctx.fillRect(x + 4, y + 6, 16, 40);
  ctx.fillStyle = color;
  ctx.fillRect(x + 6, y + 18, 12, 25);
  ctx.fillStyle = "rgba(255,255,255,0.55)";
  ctx.fillRect(x + 8, y + 9, 4, 29);
  ctx.fillStyle = "#18e4ff";
  ctx.fillRect(x + 2, y - 4, 20, 4);
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

function drawCounters(x, y, w, h, color) {
  ctx.fillStyle = "#5c4638";
  ctx.fillRect(x + 14, y + 24, w - 28, 9);
  ctx.fillStyle = color;
  ctx.fillRect(x + 16, y + 22, w - 32, 8);
  ctx.fillStyle = "#f7e7ad";
  ctx.fillRect(x + 26, y + 20, 14, 5);
}

function drawTable(x, y, top, leg) {
  ctx.fillStyle = "#5c4638";
  ctx.fillRect(x - 2, y + 2, 32, 18);
  ctx.fillStyle = top;
  ctx.fillRect(x, y, 28, 16);
  ctx.fillStyle = leg;
  ctx.fillRect(x + 5, y + 16, 4, 5);
  ctx.fillRect(x + 20, y + 16, 4, 5);
}

function drawBenchRow(x, y, count) {
  for (let i = 0; i < count; i += 1) {
    ctx.fillStyle = "#76553c";
    ctx.fillRect(x + i * 26, y, 18, 7);
    ctx.fillStyle = "#b88752";
    ctx.fillRect(x + i * 26, y - 2, 18, 5);
  }
}

function drawShelf(x, y, color) {
  ctx.fillStyle = "#5c4638";
  ctx.fillRect(x - 2, y - 2, 16, 36);
  ctx.fillStyle = color;
  ctx.fillRect(x, y, 12, 32);
  ctx.fillStyle = "#f4d25b";
  ctx.fillRect(x + 2, y + 5, 8, 3);
  ctx.fillStyle = "#d85d63";
  ctx.fillRect(x + 2, y + 14, 8, 3);
  ctx.fillStyle = "#6aa86c";
  ctx.fillRect(x + 2, y + 23, 8, 3);
}

function drawPlanter(x, y) {
  ctx.fillStyle = "#79523c";
  ctx.fillRect(x, y + 9, 28, 8);
  ctx.fillStyle = "#6aa86c";
  ctx.fillRect(x + 4, y + 4, 4, 7);
  ctx.fillRect(x + 12, y + 1, 4, 10);
  ctx.fillRect(x + 20, y + 5, 4, 6);
}

function drawStall(x, y, color) {
  ctx.fillStyle = "#5c4638";
  ctx.fillRect(x, y, 26, 28);
  ctx.fillStyle = color;
  ctx.fillRect(x + 2, y, 22, 8);
  ctx.fillStyle = "#d8b36d";
  ctx.fillRect(x + 3, y + 14, 20, 10);
}

function drawDock(site) {
  const x = site.x * tile;
  const y = site.y * tile;
  ctx.fillStyle = "#6e4b35";
  ctx.fillRect(x - 4, y + 20, site.w * tile + 8, 11);
  ctx.fillStyle = "#b88752";
  for (let i = 0; i < site.w; i += 1) ctx.fillRect(x + i * tile + 2, y + 18, 25, 13);
  ctx.fillStyle = "#6e4b35";
  ctx.fillRect(x + 8, y + 7, 5, 28);
  ctx.fillRect(x + site.w * tile - 14, y + 7, 5, 28);
  ctx.fillStyle = "#e8e0bb";
  ctx.fillRect(x + 34, y + 7, 2, 18);
  ctx.fillStyle = "#2c2740";
  ctx.fillRect(x + 35, y + 23, 8, 2);
}

function drawShrine(site) {
  const x = site.x * tile;
  const y = site.y * tile;
  ctx.fillStyle = "rgba(48,42,44,0.18)";
  ctx.fillRect(x + 4, y + 74, 60, 8);
  ctx.fillStyle = "#6b3f35";
  ctx.fillRect(x + 4, y + 20, 56, 10);
  ctx.fillStyle = site.roof;
  ctx.fillRect(x, y + 13, 64, 10);
  ctx.fillStyle = "#7a4936";
  ctx.fillRect(x + 8, y + 31, 8, 43);
  ctx.fillRect(x + 48, y + 31, 8, 43);
  ctx.fillStyle = site.wall;
  ctx.fillRect(x + 19, y + 38, 26, 34);
  ctx.fillStyle = "#d85d63";
  ctx.fillRect(x + 15, y + 34, 34, 8);
  ctx.fillStyle = "#f4ead0";
  ctx.fillRect(x + 28, y + 47, 8, 16);
  ctx.fillStyle = "#73a85d";
  ctx.fillRect(x + 24, y + 72, 16, 5);
}

function drawObservatory(site) {
  const x = site.x * tile;
  const y = site.y * tile;
  ctx.fillStyle = "rgba(48,42,44,0.2)";
  ctx.fillRect(x + 3, y + 60, 60, 7);
  ctx.fillStyle = "#4d5660";
  ctx.fillRect(x + 9, y + 29, 46, 34);
  ctx.fillStyle = site.wall;
  ctx.fillRect(x + 12, y + 32, 40, 31);
  ctx.fillStyle = site.roof;
  ctx.beginPath();
  ctx.arc(x + 32, y + 31, 22, Math.PI, 0);
  ctx.lineTo(x + 54, y + 31);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = "#d9f0f3";
  ctx.fillRect(x + 25, y + 17, 14, 9);
  ctx.fillStyle = "#5b4635";
  ctx.fillRect(x + 27, y + 48, 10, 15);
}

function drawHouse(house) {
  const x = house.x * tile;
  const y = house.y * tile;
  const w = house.w * tile;
  const h = house.h * tile;

  ctx.fillStyle = "rgba(48,42,44,0.18)";
  ctx.fillRect(x + 6, y + h - 2, w - 4, 8);
  ctx.fillStyle = "#5b4635";
  ctx.fillRect(x, y, w, h);
  ctx.fillStyle = house.wall;
  ctx.fillRect(x + 4, y + 4, w - 8, h - 8);
  ctx.fillStyle = "#d8ad72";
  ctx.fillRect(x + 9, y + 14, w - 18, h - 24);
  ctx.fillStyle = house.roof;
  ctx.fillRect(x - 2, y - 6, w + 4, 12);
  ctx.fillStyle = shade(house.roof, -22);
  ctx.fillRect(x - 2, y + 3, w + 4, 4);
  ctx.fillStyle = "#7fb0c4";
  ctx.fillRect(x + 10, y + 20, 10, 9);
  ctx.fillStyle = "#f6f0c9";
  ctx.fillRect(x + 12, y + 22, 3, 3);
  drawTable(x + 43, y + 34, "#f0d38d", "#7b573d");
  drawShelf(x + w - 26, y + 24, "#b88752");
  ctx.fillStyle = "#8b6544";
  ctx.fillRect(x + w / 2 - 18, y + h - 18, 36, 16);
  ctx.fillStyle = "#d7aa72";
  ctx.fillRect(x + w / 2 - 13, y + h - 15, 26, 13);
  ctx.fillStyle = "#f5d56c";
  ctx.fillRect(x + w / 2 + 10, y + h - 10, 3, 3);
}

function drawPlant(plant) {
  const x = Math.round(plant.x);
  const y = Math.round(plant.y);
  const colors = {
    emberglass: plant.bloom ? "#ff915f" : "#c94d45",
    moonsalt: plant.bloom ? "#dfe7ff" : "#8da1cf",
    verdigris: plant.bloom ? "#58d08c" : "#4e9f75"
  };
  ctx.fillStyle = "rgba(35, 31, 42, 0.24)";
  ctx.fillRect(x - 9, y + 8, 18, 5);
  ctx.fillStyle = "#3d3442";
  ctx.fillRect(x - 7, y - 8, 14, 20);
  ctx.fillStyle = "#d7f4f2";
  ctx.fillRect(x - 5, y - 6, 10, 16);
  ctx.fillStyle = colors[plant.species];
  const size = 5 + Math.floor(plant.age * 7);
  ctx.fillRect(x - 4, y + 4 - size, 8, size);
  ctx.fillStyle = "rgba(255,255,255,0.52)";
  ctx.fillRect(x - 3, y - 4, 2, 8);
  ctx.fillStyle = "#3d3442";
  ctx.fillRect(x - 6, y - 11, 12, 4);
  ctx.fillStyle = "rgba(255,255,255,0.45)";
  if (plant.bloom) {
    ctx.fillRect(x - 10, y - 18, 4, 4);
    ctx.fillRect(x + 7, y - 16, 3, 3);
    ctx.fillRect(x - 1, y - 21, 2, 2);
  }
}

function drawArtifact(artifact) {
  const x = Math.round(artifact.x);
  const y = Math.round(artifact.y);
  if (artifact.type === "blackboard") {
    ctx.fillStyle = "#5f4b3b";
    ctx.fillRect(x - 18, y - 18, 36, 26);
    ctx.fillStyle = "#345d4d";
    ctx.fillRect(x - 15, y - 15, 30, 18);
    ctx.fillStyle = "#f8f1ca";
    ctx.fillRect(x - 6, y - 9, 12, 3);
  } else if (artifact.type === "map") {
    ctx.fillStyle = "#f4df9d";
    ctx.fillRect(x - 13, y - 10, 26, 20);
    ctx.fillStyle = "#9f7b43";
    ctx.fillRect(x - 7, y - 2, 14, 2);
  } else if (artifact.type === "shrine") {
    ctx.fillStyle = "#ce5f58";
    ctx.fillRect(x - 13, y - 14, 26, 8);
    ctx.fillStyle = "#f3d06a";
    ctx.fillRect(x - 8, y - 5, 16, 15);
  } else {
    ctx.fillStyle = artifact.type === "charm" ? "#e56f76" : "#fff0b7";
    ctx.fillRect(x - 7, y - 8, 14, 14);
    ctx.fillStyle = "#69513b";
    ctx.fillRect(x - 3, y - 3, 6, 2);
  }
}

function drawAgent(agent) {
  const x = Math.round(agent.x);
  const y = Math.round(agent.y);
  const outfit = agent.color;
  ctx.fillStyle = "rgba(48,42,44,0.25)";
  ctx.fillRect(x - 10, y + 12, 20, 5);
  ctx.fillStyle = "#2d2632";
  ctx.fillRect(x - 9, y - 4, 18, 18);
  ctx.fillStyle = outfit;
  ctx.fillRect(x - 7, y - 2, 14, 15);
  ctx.fillStyle = shade(outfit, 18);
  ctx.fillRect(x - 5, y, 10, 4);
  ctx.fillStyle = shade(outfit, -24);
  ctx.fillRect(x - 7, y + 8, 5, 6);
  ctx.fillRect(x + 2, y + 8, 5, 6);
  ctx.fillStyle = "#3f3140";
  ctx.fillRect(x - 6, y + 13, 5, 3);
  ctx.fillRect(x + 1, y + 13, 5, 3);
  ctx.fillStyle = "#2d2632";
  ctx.fillRect(x - 7, y - 15, 14, 13);
  ctx.fillStyle = agent.skin;
  ctx.fillRect(x - 6, y - 15, 12, 12);
  ctx.fillStyle = shade(agent.skin, -18);
  ctx.fillRect(x - 6, y - 5, 12, 2);

  ctx.fillStyle = "#3f3140";
  if (agent.direction === "left") {
    ctx.fillRect(x - 5, y - 11, 2, 2);
    ctx.fillStyle = shade(agent.skin, -18);
    ctx.fillRect(x - 7, y - 8, 2, 3);
  } else if (agent.direction === "right") {
    ctx.fillRect(x + 3, y - 11, 2, 2);
    ctx.fillStyle = shade(agent.skin, -18);
    ctx.fillRect(x + 5, y - 8, 2, 3);
  } else if (agent.direction === "up") {
    ctx.fillStyle = shade(agent.skin, -24);
    ctx.fillRect(x - 5, y - 12, 10, 3);
  } else {
    ctx.fillRect(x - 4, y - 11, 2, 2);
    ctx.fillRect(x + 3, y - 11, 2, 2);
    ctx.fillStyle = "#b85d63";
    ctx.fillRect(x - 1, y - 7, 2, 1);
  }

  drawMushroomCap(agent, x, y);

  if (agent.role === "glass scribe") {
    ctx.fillStyle = "#f7e6a5";
    ctx.fillRect(x + 8, y - 8, 7, 9);
  }
  if (agent.role === "formula singer") {
    ctx.fillStyle = "#2c2740";
    ctx.fillRect(x + 9, y - 14, 3, 8);
    ctx.fillRect(x + 12, y - 14, 5, 3);
  }

  if (agent.speech && world.tick < agent.speechUntil) drawSpeech(x, y - 24, agent.speech);
}

function drawMushroomCap(agent, x, y) {
  ctx.fillStyle = "#2d2632";
  if (agent.capShape === "wide") {
    ctx.fillRect(x - 13, y - 23, 26, 10);
    ctx.fillRect(x - 9, y - 27, 18, 5);
  } else if (agent.capShape === "point") {
    ctx.beginPath();
    ctx.moveTo(x - 12, y - 13);
    ctx.lineTo(x, y - 30);
    ctx.lineTo(x + 12, y - 13);
    ctx.closePath();
    ctx.fill();
  } else if (agent.capShape === "flat") {
    ctx.fillRect(x - 11, y - 23, 22, 8);
    ctx.fillRect(x - 7, y - 27, 14, 4);
  } else {
    ctx.fillRect(x - 11, y - 23, 22, 9);
    ctx.fillRect(x - 8, y - 27, 16, 6);
  }

  ctx.fillStyle = agent.cap;
  if (agent.capShape === "wide") {
    ctx.fillRect(x - 12, y - 22, 24, 8);
    ctx.fillRect(x - 8, y - 26, 16, 4);
  } else if (agent.capShape === "point") {
    ctx.beginPath();
    ctx.moveTo(x - 10, y - 14);
    ctx.lineTo(x, y - 28);
    ctx.lineTo(x + 10, y - 14);
    ctx.closePath();
    ctx.fill();
  } else if (agent.capShape === "flat") {
    ctx.fillRect(x - 10, y - 22, 20, 6);
    ctx.fillRect(x - 6, y - 25, 12, 3);
  } else {
    ctx.fillRect(x - 10, y - 22, 20, 7);
    ctx.fillRect(x - 7, y - 26, 14, 5);
  }

  ctx.fillStyle = shade(agent.cap, 34);
  ctx.fillRect(x - 6, y - 23, 5, 3);
  ctx.fillStyle = "#f7edcf";
  for (let i = 0; i < agent.spots; i += 1) {
    const sx = x - 8 + ((i * 6 + agent.id * 3) % 17);
    const sy = y - 23 + ((i * 5 + agent.id) % 7);
    ctx.fillRect(sx, sy, 3, 3);
  }
  ctx.fillStyle = "#4d8b62";
  if (agent.id % 3 === 0) {
    ctx.fillRect(x - 1, y - 30, 2, 5);
    ctx.fillRect(x + 1, y - 31, 5, 2);
  }
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
