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
  memeToggle: document.querySelector("#memeToggle")
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
  weather: "clear",
  nextWeatherTick: 0,
  log: [],
  discoveries: [],
  observations: [],
  ruleStats: new Map(),
  minedRules: [],
  communications: [],
  memeVisualization: false,
  memeEdges: [],
  history: [],
  lastHistoryTick: -1,
  reward: 0,
  predictions: 0
};

ui.memeToggle?.addEventListener("change", () => {
  world.memeVisualization = ui.memeToggle.checked;
});

const seasons = ["Sprout", "Cicada", "Maple", "Snowbell"];
const weatherTypes = ["clear", "clear", "breezy", "rain", "rain", "mist"];
const clusterPalette = ["#65a969", "#5d8edb", "#de8a4c", "#a27bc2", "#d45f73", "#d8b34d"];
const predictorPool = {
  bluecap: ["weather=rain", "weather=mist", "terrain=garden", "nearWater=true"],
  sunbean: ["weather=clear", "season=Cicada", "terrain=garden", "terrain=plaza"],
  puffroot: ["terrain=path", "terrain=plaza", "crowded=true", "nearBuilding=true"]
};

const hiddenRules = Object.fromEntries(
  Object.entries(predictorPool).map(([species, predictors]) => [species, pick(predictors)])
);

const outcomeDefs = [
  { id: "bloom", label: "plant bloom", color: "#65a969" },
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
    facts.includes("species=bluecap") ? 1 : 0,
    facts.includes("species=sunbean") ? 1 : 0,
    facts.includes("species=puffroot") ? 1 : 0,
    facts.includes("weather=rain") ? 1 : 0,
    facts.includes("weather=clear") ? 1 : 0,
    facts.includes("weather=mist") ? 1 : 0,
    facts.includes("terrain=path") || facts.includes("terrain=plaza") ? 1 : 0,
    facts.includes("terrain=garden") ? 1 : 0,
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
  return [
    `species=${plant.species}`,
    `weather=${world.weather}`,
    `terrain=${terrainType}`,
    `season=${seasons[world.seasonIndex]}`,
    `nearWater=${["river", "pond"].includes(terrainType)}`,
    `crowded=${nearbyAgents >= 3}`,
    `nearBuilding=${isNearBuilding(plant.x, plant.y)}`
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
  { type: "townHall", name: "Town Hall", x: 13, y: 8, w: 5, h: 4, roof: "#c95f55", wall: "#f4dca9" },
  { type: "lab", name: "Experiment Lab", x: 24, y: 12, w: 6, h: 4, roof: "#5f8fc9", wall: "#e8ece4" },
  { type: "archive", name: "Archive", x: 23, y: 8, w: 5, h: 3, roof: "#a979bd", wall: "#efe2c7" },
  { type: "workshop", name: "Tool Workshop", x: 4, y: 13, w: 3, h: 2, roof: "#d89b4a", wall: "#ead0a3" },
  { type: "observatory", name: "Sky Hut", x: 49, y: 4, w: 2, h: 2, roof: "#4f6fb4", wall: "#d7e3ec" },
  { type: "shrine", name: "Moss Shrine", x: 5, y: 8, w: 2, h: 3, roof: "#d85d63", wall: "#f4cc72" },
  { type: "dock", name: "Fishing Dock", x: 3, y: 5, w: 3, h: 2, roof: "#9a7147", wall: "#c99b62" },
  { type: "cafe", name: "Memory Cafe", x: 34, y: 9, w: 5, h: 3, roof: "#d28a4d", wall: "#f0d09e" },
  { type: "school", name: "Sprout School", x: 45, y: 12, w: 5, h: 4, roof: "#6aa86c", wall: "#e8dfbc" },
  { type: "greenhouse", name: "Greenhouse", x: 10, y: 27, w: 6, h: 4, roof: "#6fbfb2", wall: "#d8eee2" },
  { type: "market", name: "Market Hall", x: 31, y: 30, w: 6, h: 4, roof: "#d8b34d", wall: "#efddb3" },
  { type: "theater", name: "Song Theater", x: 45, y: 28, w: 6, h: 4, roof: "#cc6d8e", wall: "#ead5c5" }
];

const terrain = Array.from({ length: rows }, (_, y) =>
  Array.from({ length: cols }, (_, x) => {
    const edge = x < 1 || y < 1 || x > cols - 2 || y > rows - 2;
    const riverY = 6 + Math.round(Math.sin(x * 0.34) * 2.2);
    if (edge) return "trees";
    if (x > 1 && x < cols - 2 && Math.abs(y - riverY) <= 1) return "river";
    if ((x - 52) ** 2 + (y - 8) ** 2 < 18) return "pond";
    if ((x - 50) ** 2 + (y - 32) ** 2 < 11) return "pond";
    if (x > 2 && x < 10 && y > 12 && y < 18) return "garden";
    if (x > 9 && x < 17 && y > 26 && y < 33) return "garden";
    if (x > 12 && x < 31 && y > 8 && y < 19) return "plaza";
    if (x > 30 && x < 52 && y > 22 && y < 35) return "plaza";
    if (
      (x > 6 && x < 53 && (y === 20 || y === 21)) ||
      (x > 10 && x < 55 && y === 12) ||
      (x === 22 && y > 7 && y < 36) ||
      (x === 40 && y > 8 && y < 36) ||
      (x === 7 && y > 7 && y < 22) ||
      (y > 6 && y < 10 && x > 2 && x < 8)
    ) return "path";
    if ((x + y) % 17 === 0) return "flowers";
    return "grass";
  })
);

const plants = [];
for (let i = 0; i < 90; i += 1) {
  const inGarden = i < 48;
  const gardenZone = i % 2 === 0 ? { x1: 3.2, x2: 9.7, y1: 12.4, y2: 17.4 } : { x1: 10.2, x2: 15.8, y1: 27.2, y2: 32.2 };
  let x = inGarden ? rand(gardenZone.x1, gardenZone.x2) * tile : rand(2, cols - 3) * tile;
  let y = inGarden ? rand(gardenZone.y1, gardenZone.y2) * tile : rand(2, rows - 3) * tile;
  let guard = 0;
  while (!inGarden && ["river", "pond", "trees"].includes(terrain[Math.floor(y / tile)]?.[Math.floor(x / tile)]) && guard < 20) {
    x = rand(2, cols - 3) * tile;
    y = rand(2, rows - 3) * tile;
    guard += 1;
  }
  plants.push({
    x,
    y,
    species: pick(["bluecap", "sunbean", "puffroot"]),
    age: rand(0, 1),
    bloom: false,
    observedRain: 0
  });
}

const capPalettes = ["#d85d63", "#e0b84f", "#6fbf75", "#7aa5d8", "#a37ac0", "#ef8f65", "#cfd66b", "#70b7a8"];
const skinPalettes = ["#f4d6a3", "#d9b47f", "#b7d48b", "#98cfa7", "#d7c7a5", "#b9d5c5", "#e8c0a6", "#a9c98d"];
const clothingPalettes = ["#476c55", "#5b8f6b", "#7a6aa8", "#b66b56", "#4f7f9f", "#a08248", "#8c5f7f", "#52786c"];

const agents = [
  ["Mika", "cartographer"],
  ["Taro", "gardener"],
  ["Nori", "skeptic"],
  ["Piko", "songmaker"],
  ["Sumi", "dreamer"],
  ["Ren", "teacher"],
  ["Kiko", "collector"],
  ["Bo", "tinkerer"],
  ["Yui", "child"],
  ["Aki", "elder"],
  ["Mori", "naturalist"],
  ["Nana", "weatherwatcher"],
  ["Tobu", "builder"],
  ["Lumi", "archivist"],
  ["Fenn", "forager"],
  ["Riri", "listener"],
  ["Momo", "student"],
  ["Sora", "mapper"],
  ["Beni", "caretaker"],
  ["Iro", "inventor"],
  ["Koma", "gardener"],
  ["Mugi", "forager"],
  ["Hana", "teacher"],
  ["Toki", "weatherwatcher"],
  ["Nemu", "dreamer"],
  ["Raku", "builder"],
  ["Sasa", "listener"],
  ["Pomu", "student"],
  ["Kiri", "archivist"],
  ["Maro", "naturalist"],
  ["Fuyu", "skeptic"],
  ["Tama", "caretaker"],
  ["Mina", "cartographer"],
  ["Kumo", "inventor"],
  ["Roko", "collector"],
  ["Niko", "child"],
  ["Suzu", "mapper"],
  ["Pipi", "songmaker"],
  ["Eno", "tinkerer"],
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
  world.memeEdges = world.memeEdges.slice(0, 96);
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
  if (outcome === "bloom") {
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
    if (stat.seen < 8) continue;
    const p = stat.hits / stat.seen;
    const base = stat.absentSeen ? stat.absentHits / stat.absentSeen : 0.5;
    const lift = p - base;
    const confidence = clamp(0.4 * p + 0.6 * Math.max(0, lift), 0, 1);
    if (lift > 0.22 && p > 0.58) {
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
        parent: stat.feature.split("=")[0]
      });
    }
  }
  world.minedRules = mined.sort((a, b) => b.confidence - a.confidence).slice(0, 18);
  for (const rule of world.minedRules) {
    if (rule.confidence > 0.62 && !world.discoveries.includes(rule.id)) {
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
  return feature
    .replace("species=", "")
    .replace("weather=", "")
    .replace("terrain=", "")
    .replace("season=", "")
    .replace("nearWater=true", "water adjacency")
    .replace("crowded=true", "crowding")
    .replace("nearBuilding=true", "building proximity");
}

function labelOutcome(outcome) {
  return outcomeDefs.find((item) => item.id === outcome)?.label ?? outcome;
}

function clusterForFeature(feature) {
  if (feature.startsWith("species=") || feature === "terrain=garden" || feature === "nearWater=true") return "ecology";
  if (feature.startsWith("weather=") || feature.startsWith("season=") || feature.includes("water")) return "weather";
  if (feature.includes("crowded") || feature.includes("Building") || feature.includes("building") || feature.includes("plaza") || feature.includes("path")) return "social space";
  return "prediction";
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
  if (world.weather === "rain") {
    addLog("Rain softened the garden. The villagers began watching which plants respond.");
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

  const action = chooseWeightedAction(agent);
  agent.lastAction = action;
  if (action === "observing") {
    const plant = pick(plants);
    setTarget(agent, plant.x + rand(-16, 16), plant.y + rand(-16, 16), "observing plants");
  } else if (action === "researching") {
    const site = pick(landmarks.filter((landmark) => ["lab", "archive", "townHall", "observatory"].includes(landmark.type)));
    setTarget(agent, (site.x + site.w / 2) * tile, (site.y + site.h + 0.35) * tile, "writing theory");
  } else if (action === "talking") {
    const friend = pick(agents.filter((a) => a !== agent));
    setTarget(agent, friend.x + rand(-28, 28), friend.y + rand(-28, 28), "seeking talk");
  } else {
    const site = pick([
      { x: 4.4, y: 7.4, action: "fishing" },
      { x: 6.2, y: 11.3, action: "visiting shrine" },
      { x: 5.5, y: 15.8, action: "tinkering" },
      { x: 36, y: 11.8, action: "sharing coffee notes" },
      { x: 47, y: 16.3, action: "studying at school" },
      { x: 34, y: 34.3, action: "market gossip" },
      { x: 49, y: 32.3, action: "performing song" },
      { x: rand(3, cols - 4), y: rand(2, rows - 4), action: pick(["wandering", "humming", "collecting"]) }
    ]);
    setTarget(agent, site.x * tile, site.y * tile, site.action);
  }
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
  return !["river", "pond", "trees"].includes(terrainType);
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
  if (world.tick % 52 !== agent.id * 7 % 52) return;
  const plant = plants.find((p) => distance(agent, p) < 42);
  if (!plant) return;

  const facts = plantFacts(plant);
  const model = agent.models.bloom;
  const inputs = observationFeatures(plant);
  const target = plant.bloom ? 1 : 0;
  const prediction = trainBrain(model.brain, inputs, target);
  model.evidence += 1;
  model.neural = prediction;
  model.confidence = clamp(model.confidence * 0.78 + prediction * 0.16 + target * 0.05, 0, 0.98);
  model.source = "field note";
  const reward = recordObservation(agent, "bloom", facts, prediction, target);

  agent.notebook.unshift(`${plant.species} ${plant.bloom ? "bloomed" : "waited"} ${world.weather} (${Math.round(prediction * 100)}%)`);
  agent.notebook = agent.notebook.slice(0, 6);

  if (Math.abs(target - prediction) > agent.memoryPolicy.writeThreshold && Math.random() < 0.36) {
    speak(agent, pick(["field note!", "odd result!", "tiny proof!", "hmm!"]));
  }
}

function talk() {
  for (const a of agents) {
    if (world.tick % 55 !== (a.id * 11) % 55) continue;
    const b = agents.find((candidate) => candidate !== a && distance(a, candidate) < 44);
    if (!b) continue;

    const rule = world.minedRules.find((item) => item.outcome === "bloom" && world.discoveries.includes(item.id)) ??
      world.minedRules.find((item) => item.outcome === "bloom") ??
      world.minedRules.find((item) => world.discoveries.includes(item.id)) ??
      world.minedRules[0];
    const memory = a.memory.find((item) => item.type === "observation");
    const symbol = rule ? compactRule(rule) : compactMemory(memory);
    const confidence = rule?.confidence ?? (memory ? memory.reward : 0.5);
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
    .replace("weather=", "w:")
    .replace("terrain=", "t:")
    .replace("species=", "s:")
    .replace("season=", "se:")
    .replace("nearWater=true", "water")
    .replace("crowded=true", "crowd")
    .replace("nearBuilding=true", "house");
  return `${feature}->${rule.outcome}`;
}

function compactMemory(memory) {
  if (!memory) return "notebook?";
  const cue = memory.facts?.[0]?.replace("species=", "s:") ?? "note";
  return `${cue}->${memory.outcome ?? "world"}`;
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
    addLog("Night notebooks became fresh questions for morning.");
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
    grass: "#91cf73",
    flowers: "#9edb78",
    garden: "#9a7651",
    path: "#d8bf7a",
    plaza: "#cdb579",
    river: "#5faed0",
    pond: "#5aaec8",
    trees: "#4c8b55"
  };
  ctx.fillStyle = palette[type];
  ctx.fillRect(px, py, tile, tile);

  ctx.fillStyle = "rgba(255,255,255,0.16)";
  if ((x * 3 + y + world.day) % 4 === 0) ctx.fillRect(px + 5, py + 6, 4, 4);
  ctx.fillStyle = "rgba(58, 93, 48, 0.12)";
  if ((x + y * 2) % 5 === 0) ctx.fillRect(px + 20, py + 18, 6, 3);

  if (type === "grass" || type === "flowers") {
    ctx.fillStyle = "#5ea55a";
    if ((x + y) % 2 === 0) {
      ctx.fillRect(px + 8, py + 22, 3, 5);
      ctx.fillRect(px + 12, py + 19, 3, 8);
    }
    if ((x * 7 + y) % 5 === 0) {
      ctx.fillStyle = "#d95e7a";
      ctx.fillRect(px + 23, py + 9, 3, 3);
      ctx.fillStyle = "#f6d765";
      ctx.fillRect(px + 26, py + 12, 2, 2);
    }
  }

  if (type === "trees") {
    ctx.fillStyle = "#2f653f";
    ctx.fillRect(px + 7, py + 7, 18, 19);
    ctx.fillStyle = "#4f9a59";
    ctx.fillRect(px + 5, py + 4, 22, 14);
    ctx.fillStyle = "#6fb867";
    ctx.fillRect(px + 10, py + 2, 12, 8);
    ctx.fillStyle = "#725134";
    ctx.fillRect(px + 13, py + 20, 6, 10);
    ctx.fillStyle = "rgba(255,255,255,0.18)";
    ctx.fillRect(px + 9, py + 7, 5, 3);
  }
  if (type === "path") {
    ctx.fillStyle = "#caa96d";
    ctx.fillRect(px, py + 11, tile, 10);
    ctx.fillRect(px + 11, py, 10, tile);
    ctx.fillStyle = "#ead79b";
    if ((x + y) % 2 === 0) ctx.fillRect(px + 4, py + 14, 7, 3);
    if ((x + y) % 3 === 0) ctx.fillRect(px + 18, py + 6, 3, 7);
  }
  if (type === "plaza") {
    ctx.fillStyle = "#bfa266";
    ctx.fillRect(px, py, tile, tile);
    ctx.fillStyle = "#dbc489";
    ctx.fillRect(px + 2, py + 2, 12, 12);
    ctx.fillRect(px + 18, py + 18, 12, 12);
    ctx.fillStyle = "#c8ac70";
    ctx.fillRect(px + 17, py + 2, 13, 12);
    ctx.fillRect(px + 2, py + 18, 12, 12);
  }
  if (type === "garden") {
    ctx.fillStyle = "#7d5b3e";
    for (let i = 0; i < 4; i += 1) ctx.fillRect(px, py + i * 8 + 3, tile, 2);
    ctx.fillStyle = "#af8b5d";
    if ((x + y) % 2 === 0) ctx.fillRect(px + 6, py + 7, 20, 3);
  }
  if (type === "river" || type === "pond") {
    ctx.fillStyle = type === "river" ? "#4b98bd" : "#438fab";
    ctx.fillRect(px, py, tile, tile);
    ctx.fillStyle = "#438fab";
    ctx.fillRect(px, py + 24, tile, 8);
    ctx.fillStyle = "#7ed3dc";
    if ((x + y + Math.floor(world.tick / 30)) % 3 === 0) ctx.fillRect(px + 5, py + 10, 18, 3);
    if (type === "river") {
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
  const floor = {
    townHall: "#d7aa72",
    lab: "#d6e0df",
    archive: "#cba97a",
    workshop: "#c79a62",
    cafe: "#dcb26f",
    school: "#d7bf7a",
    greenhouse: "#bfe4cf",
    market: "#d6b96b",
    theater: "#c99abb"
  }[site.type] ?? "#d6b275";

  ctx.fillStyle = "rgba(48,42,44,0.22)";
  ctx.fillRect(x + 7, y + h - 2, w - 2, 9);
  ctx.fillStyle = "#5a4738";
  ctx.fillRect(x, y, w, h);
  ctx.fillStyle = site.wall;
  ctx.fillRect(x + 4, y + 4, w - 8, h - 8);
  ctx.fillStyle = floor;
  ctx.fillRect(x + 10, y + 14, w - 20, h - 24);

  ctx.fillStyle = "rgba(92, 62, 42, 0.18)";
  for (let px = x + 16; px < x + w - 12; px += 16) ctx.fillRect(px, y + 16, 2, h - 28);
  for (let py = y + 22; py < y + h - 12; py += 16) ctx.fillRect(x + 12, py, w - 24, 2);

  ctx.fillStyle = site.roof;
  ctx.fillRect(x - 2, y - 7, w + 4, 12);
  ctx.fillStyle = shade(site.roof, -24);
  ctx.fillRect(x - 2, y + 2, w + 4, 5);
  ctx.fillStyle = "#f6f1d2";
  ctx.font = "10px monospace";
  ctx.textAlign = "center";
  ctx.fillText(site.name.slice(0, 14), x + w / 2, y + 2);

  drawFurniture(site, x, y, w, h);
  ctx.fillStyle = "#6b4b37";
  ctx.fillRect(x + w / 2 - 8, y + h - 12, 16, 10);
}

function drawFurniture(site, x, y, w, h) {
  if (site.type === "lab") {
    drawCounters(x, y, w, h, "#88a9c9");
    for (let i = 0; i < 4; i += 1) drawTable(x + 26 + i * 34, y + 45, "#dce8ec", "#68a7b8");
    drawShelf(x + w - 36, y + 28, "#9fb2be");
    return;
  }
  if (site.type === "townHall") {
    drawTable(x + w / 2 - 18, y + 42, "#eed084", "#9d6a44");
    drawBenchRow(x + 18, y + h - 42, 4);
    drawBenchRow(x + 18, y + h - 28, 4);
    return;
  }
  if (site.type === "archive") {
    for (let i = 0; i < 5; i += 1) drawShelf(x + 14 + i * 26, y + 26, "#8e6348");
    drawTable(x + w - 48, y + h - 42, "#f1dfaa", "#8b6544");
    return;
  }
  if (site.type === "cafe") {
    drawCounters(x, y, w, h, "#b57b4b");
    for (let i = 0; i < 4; i += 1) drawTable(x + 22 + i * 30, y + 55, "#f0d38d", "#7b573d");
    return;
  }
  if (site.type === "school") {
    ctx.fillStyle = "#4f7e55";
    ctx.fillRect(x + 14, y + 22, w - 28, 12);
    for (let row = 0; row < 3; row += 1) drawBenchRow(x + 20, y + 48 + row * 22, 4);
    return;
  }
  if (site.type === "greenhouse") {
    ctx.fillStyle = "rgba(225,255,247,0.45)";
    for (let i = 0; i < 5; i += 1) ctx.fillRect(x + 14 + i * 28, y + 18, 5, h - 32);
    for (let i = 0; i < 9; i += 1) drawPlanter(x + 18 + (i % 3) * 42, y + 36 + Math.floor(i / 3) * 24);
    return;
  }
  if (site.type === "market") {
    for (let i = 0; i < 4; i += 1) drawStall(x + 16 + i * 34, y + 36, i % 2 ? "#d85d63" : "#6aa86c");
    drawCounters(x, y + 54, w, h - 54, "#b88752");
    return;
  }
  if (site.type === "theater") {
    ctx.fillStyle = "#74485e";
    ctx.fillRect(x + 14, y + 22, w - 28, 24);
    ctx.fillStyle = "#e8c15f";
    ctx.fillRect(x + 23, y + 28, w - 46, 4);
    for (let row = 0; row < 3; row += 1) drawBenchRow(x + 22, y + 58 + row * 18, 5);
    return;
  }
  drawCounters(x, y, w, h, "#b88752");
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
    bluecap: plant.bloom ? "#5d8edb" : "#78a95e",
    sunbean: plant.bloom ? "#e9c849" : "#85b85d",
    puffroot: plant.bloom ? "#f0e4d7" : "#7fb36a"
  };
  ctx.fillStyle = "#2f653f";
  ctx.fillRect(x - 3, y + 3, 6, 9);
  ctx.fillStyle = "#427a43";
  ctx.fillRect(x - 2, y + 4, 4, 8);
  ctx.fillRect(x - 6, y + 7, 4, 2);
  ctx.fillRect(x + 2, y + 5, 5, 2);
  ctx.fillStyle = "#324437";
  ctx.fillRect(x - 5, y - 5, 10, 10);
  ctx.fillStyle = colors[plant.species];
  const size = 4 + Math.floor(plant.age * 6);
  ctx.fillRect(x - size / 2, y - size / 2, size, size);
  ctx.fillStyle = "rgba(255,255,255,0.45)";
  if (plant.bloom) ctx.fillRect(x - 1, y - 1, 2, 2);
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

  if (agent.role === "cartographer") {
    ctx.fillStyle = "#f7e6a5";
    ctx.fillRect(x + 8, y - 8, 7, 9);
  }
  if (agent.role === "songmaker") {
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
  if (world.weather === "rain") {
    ctx.fillStyle = "rgba(75,111,164,0.38)";
    for (let i = 0; i < 80; i += 1) {
      const x = (i * 53 + world.tick * 2) % W;
      const y = (i * 97 + world.tick * 5) % H;
      ctx.fillRect(x, y, 2, 9);
    }
  }
  if (world.weather === "mist") {
    ctx.fillStyle = "rgba(255,255,255,0.18)";
    for (let i = 0; i < 5; i += 1) {
      ctx.fillRect((world.tick + i * 220) % W - 120, 80 + i * 86, 180, 16);
    }
  }
}

function drawMemeVisualization() {
  if (!world.memeVisualization) return;
  drawMemeEdges();
  drawMemeOrbs();
}

function drawMemeEdges() {
  const maxAge = 2400;
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (const edge of world.memeEdges) {
    const age = world.tick - edge.tick;
    if (age < 0 || age > maxAge) continue;
    const from = agents[edge.fromId];
    const to = agents[edge.toId];
    if (!from || !to) continue;
    const alpha = clamp(1 - age / maxAge, 0.12, 1);
    const pulse = Math.sin((world.tick - edge.tick) * 0.1) * 0.5 + 0.5;
    const start = memeOrbPoint(from);
    const end = memeOrbPoint(to);
    const mx = (start.x + end.x) / 2;
    const my = (start.y + end.y) / 2 - clamp(distance(start, end) * 0.12, 18, 80);

    ctx.globalAlpha = alpha * 0.22;
    ctx.strokeStyle = edge.color;
    ctx.lineWidth = 16 + pulse * 5;
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.quadraticCurveTo(mx, my, end.x, end.y);
    ctx.stroke();

    ctx.globalAlpha = alpha * 0.78;
    ctx.strokeStyle = edge.color;
    ctx.lineWidth = 5 + pulse * 2;
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.quadraticCurveTo(mx, my, end.x, end.y);
    ctx.stroke();

    ctx.globalAlpha = alpha * 0.95;
    drawMemeNode(start.x, start.y, edge.color, 8 + pulse * 2, false);
    drawMemeNode(end.x, end.y, edge.color, 8 + pulse * 2, false);
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
      domain: index === 0 ? "ecology" : "communication",
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
  const fallback = ids.length ? ids : ["ecology", "weather", "social space", "communication"];
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
addLog("The village woke inside a small green world and began asking why plants change.");
agents.forEach(chooseTarget);
renderHud();
renderPanels();
draw();
setInitialWorldScroll();
loop();
