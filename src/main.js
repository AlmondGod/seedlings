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
  reward: document.querySelector("#reward")
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
  communications: [],
  history: [],
  lastHistoryTick: -1,
  reward: 0,
  predictions: 0
};

const seasons = ["Sprout", "Cicada", "Maple", "Snowbell"];
const weatherTypes = ["clear", "clear", "breezy", "rain", "rain", "mist"];
const domainDefs = [
  { id: "biology", name: "Biology", goal: 6, color: "#65a969" },
  { id: "physics", name: "Physics", goal: 4, color: "#5d8edb" },
  { id: "sociology", name: "Sociology", goal: 2, color: "#de8a4c" },
  { id: "psychology", name: "Psychology", goal: 2, color: "#a27bc2" }
];

const claimCatalog = [
  { claim: "Bluecap blooms after rain.", domain: "biology", parent: "biology", threshold: 0.62 },
  { claim: "Sunbean follows warm light.", domain: "biology", parent: "biology", threshold: 0.62 },
  { claim: "Puffroot grows near crowded paths.", domain: "sociology", parent: "sociology", threshold: 0.62 },
  { claim: "Rain returns in seasonal clusters.", domain: "physics", parent: "physics", threshold: 0.78 },
  { claim: "River reeds grow near moving water.", domain: "biology", parent: "biology", threshold: 0.7 },
  { claim: "Mist gathers around ponds.", domain: "physics", parent: "physics", threshold: 0.72 },
  { claim: "Shared observations improve predictions.", domain: "psychology", parent: "psychology", threshold: 0.76 },
  { claim: "Rain predicts the next bluecap bloom patch.", domain: "biology", parent: "Bluecap blooms after rain.", threshold: 0.72 },
  { claim: "Warm mornings predict sunbean harvest windows.", domain: "biology", parent: "Sunbean follows warm light.", threshold: 0.72 },
  { claim: "Busy paths predict future puffroot spread.", domain: "sociology", parent: "Puffroot grows near crowded paths.", threshold: 0.72 },
  { claim: "Season clusters forecast rain one morning ahead.", domain: "physics", parent: "Rain returns in seasonal clusters.", threshold: 0.82 },
  { claim: "Pond mist predicts cooler morning fields.", domain: "physics", parent: "Mist gathers around ponds.", threshold: 0.78 },
  { claim: "River speed predicts reed density.", domain: "biology", parent: "River reeds grow near moving water.", threshold: 0.76 },
  { claim: "Accurate reports reduce listener prediction error.", domain: "psychology", parent: "Shared observations improve predictions.", threshold: 0.82 }
];

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
  const terrainType = terrain[Math.floor(plant.y / tile)]?.[Math.floor(plant.x / tile)];
  const hour = (world.tick % 1440) / 1440;
  return [
    plant.species === "bluecap" ? 1 : 0,
    plant.species === "sunbean" ? 1 : 0,
    plant.species === "puffroot" ? 1 : 0,
    world.weather === "rain" ? 1 : 0,
    world.weather === "clear" ? 1 : 0,
    world.weather === "mist" ? 1 : 0,
    terrainType === "path" || terrainType === "plaza" ? 1 : 0,
    terrainType === "garden" ? 1 : 0,
    plant.bloom ? 1 : 0,
    hour
  ];
}

function conceptFeatures(claim) {
  const index = claimCatalog.findIndex((item) => item.claim === claim);
  return Array.from({ length: neuralInputs }, (_, inputIndex) => {
    if (inputIndex === index % neuralInputs) return 1;
    if (inputIndex === 9) return (world.tick % 1440) / 1440;
    return 0;
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
  beliefs: claimCatalog.map((item, beliefIndex) => ({
    ...item,
    confidence: beliefIndex < 3 ? rand(0.08, 0.24) : rand(0.02, 0.1),
    evidence: 0,
    source: "hunch",
    brain: createBrain(),
    neural: 0.5
  })),
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
  world.communications.unshift({
    from: from.name,
    to: to.name,
    claim: belief.claim,
    confidence: belief.confidence,
    delta,
    day: world.day
  });
  world.communications = world.communications.slice(0, 10);
}

function addPredictionReward(prediction, target) {
  const reward = 1 - Math.abs(target - prediction);
  world.reward += reward;
  world.predictions += 1;
  return reward;
}

function addEvidence(claim, amount, source) {
  const inputs = conceptFeatures(claim);
  for (const agent of agents) {
    const belief = agent.beliefs.find((item) => item.claim === claim);
    if (!belief) continue;
    const prediction = trainBrain(belief.brain, inputs, 1, 0.1);
    belief.neural = prediction;
    belief.evidence += 1;
    belief.confidence = clamp(belief.confidence * 0.72 + prediction * 0.22 + amount * rand(0.7, 1.25), 0, 0.98);
    belief.source = source;
  }
  checkDiscovery(claim);
}

function checkDiscovery(claim) {
  const item = claimCatalog.find((entry) => entry.claim === claim);
  if (!item || world.discoveries.includes(claim)) return;
  if (!isClaimUnlocked(item)) return;
  const belief = aggregateBeliefs().find((entry) => entry.claim === claim);
  if (!belief || belief.confidence < item.threshold) return;
  world.discoveries.push(claim);
  addLog(`The village accepted a new ${domainName(item.domain).toLowerCase()} discovery: ${claim}`);
  artifacts.push({
    type: "note",
    x: rand(22.5, 27.4) * tile,
    y: rand(12.2, 17) * tile,
    text: item.domain.slice(0, 3),
    idea: claim
  });
  for (const child of claimCatalog.filter((entry) => entry.parent === claim)) {
    addEvidence(child.claim, 0.035, "derived from parent discovery");
  }
}

function domainName(id) {
  return domainDefs.find((domain) => domain.id === id)?.name ?? id;
}

function isClaimUnlocked(item) {
  return domainDefs.some((domain) => domain.id === item.parent) || world.discoveries.includes(item.parent);
}

function setWeather() {
  world.weather = pick(weatherTypes);
  world.nextWeatherTick = world.tick + 720;
  if (world.weather === "rain") {
    addLog("Rain softened the garden. The villagers began watching bluecap closely.");
    addEvidence("Rain returns in seasonal clusters.", 0.025, "weather diary");
  }
}

function chooseTarget(agent) {
  const hour = (world.tick % 1440) / 60;
  if (hour < 6 || hour > 21) {
    const home = houses[agent.id % houses.length];
    setTarget(agent, (home.x + home.w / 2) * tile, (home.y + home.h + 0.12) * tile, "sleepwalking home");
    return;
  }

  const roll = Math.random();
  if (roll < 0.38) {
    const plant = pick(plants);
    setTarget(agent, plant.x + rand(-16, 16), plant.y + rand(-16, 16), "observing plants");
  } else if (roll < 0.56) {
    const site = pick(landmarks.filter((landmark) => ["lab", "archive", "townHall", "observatory"].includes(landmark.type)));
    setTarget(agent, (site.x + site.w / 2) * tile, (site.y + site.h + 0.35) * tile, "writing theory");
  } else if (roll < 0.78) {
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

  const belief = agent.beliefs.find((b) => b.claim.includes(
    plant.species === "bluecap" ? "Bluecap" : plant.species === "sunbean" ? "Sunbean" : "Puffroot"
  ));
  if (!belief) return;

  const matchesRule =
    (plant.species === "bluecap" && world.weather === "rain") ||
    (plant.species === "sunbean" && world.weather === "clear") ||
    (plant.species === "puffroot" && ["path", "plaza"].includes(terrain[Math.floor(plant.y / tile)]?.[Math.floor(plant.x / tile)]));
  const inputs = observationFeatures(plant);
  const target = matchesRule ? 1 : 0;
  const prediction = trainBrain(belief.brain, inputs, target);

  belief.evidence += 1;
  belief.neural = prediction;
  belief.confidence = clamp(belief.confidence * 0.64 + prediction * 0.32 + (matchesRule ? 0.08 : 0.012), 0, 0.98);
  belief.source = "field note";
  const reward = addPredictionReward(prediction, target);
  agent.notebook.unshift(`${plant.species} ${matchesRule ? "matched" : "confused"} ${world.weather} (${Math.round(prediction * 100)}%)`);
  agent.notebook = agent.notebook.slice(0, 6);
  remember(agent, {
    type: "observation",
    claim: belief.claim,
    prediction,
    target,
    reward,
    place: terrain[Math.floor(agent.y / tile)]?.[Math.floor(agent.x / tile)] ?? "unknown"
  });

  if (matchesRule && Math.random() < 0.24) {
    speak(agent, pick(["I saw it!", "field note!", "tiny proof!", "hmm!"]));
  }

  if (belief.confidence > belief.threshold && !world.discoveries.includes(belief.claim)) {
    world.discoveries.push(belief.claim);
    addLog(`${agent.name} proposed a ${domainName(belief.domain).toLowerCase()} discovery: ${belief.claim}`);
    artifacts.push({
      type: "note",
      x: rand(22.5, 27.4) * tile,
      y: rand(12.2, 17) * tile,
      text: plant.species.slice(0, 4),
      idea: belief.claim
    });
  }
}

function talk() {
  for (const a of agents) {
    if (world.tick % 55 !== (a.id * 11) % 55) continue;
    const b = agents.find((candidate) => candidate !== a && distance(a, candidate) < 44);
    if (!b) continue;

    const belief = pick(a.beliefs);
    const other = b.beliefs.find((item) => item.claim === belief.claim);
    const trust = 0.024 + (a.role === "teacher" || b.role === "child" ? 0.026 : 0);
    const socialPrediction = trainBrain(other.brain, conceptFeatures(belief.claim), belief.confidence, 0.07);
    other.neural = socialPrediction;
    const before = other.confidence;
    other.confidence = clamp(other.confidence * 0.86 + socialPrediction * 0.1 + belief.confidence * trust, 0, 0.98);
    other.source = a.name;
    other.evidence += 1;
    const delta = other.confidence - before;
    addCommunication(a, b, belief, delta);
    remember(a, { type: "said", claim: belief.claim, to: b.name, confidence: belief.confidence });
    remember(b, { type: "heard", claim: belief.claim, from: a.name, confidence: belief.confidence, delta });
    addEvidence("Shared observations improve predictions.", Math.max(0.002, delta * 0.08), "communication");
    speak(a, belief.claim.split(" ").slice(0, 3).join(" ") + "?");
    checkDiscovery(belief.claim);
  }
}

function speak(agent, text) {
  agent.speech = text;
  agent.speechUntil = world.tick + 120;
}

function updatePlants() {
  if (world.tick % 50 !== 0) return;
  for (const plant of plants) {
    const boost =
      (plant.species === "bluecap" && world.weather === "rain") ||
      (plant.species === "sunbean" && world.weather === "clear")
        ? 0.05
        : 0.015;
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
}

function aggregateBeliefs() {
  return claimCatalog.map((catalogItem) => {
    const claim = catalogItem.claim;
    const entries = agents.map((agent) => agent.beliefs.find((b) => b.claim === claim));
    return {
      ...catalogItem,
      claim,
      confidence: entries.reduce((sum, item) => sum + item.confidence, 0) / entries.length,
      neural: entries.reduce((sum, item) => sum + item.neural, 0) / entries.length,
      evidence: entries.reduce((sum, item) => sum + item.evidence, 0),
      sources: [...new Set(entries.map((item) => item.source))].slice(0, 3)
    };
  }).sort((a, b) => b.confidence - a.confidence);
}

function renderDiscoveryTree(beliefs, domains) {
  const rootX = 18;
  const claimX = 168;
  const childX = 390;
  const rowHeight = 104;
  const childHeight = 86;
  const edges = [];
  const nodes = [];
  let cursorY = 20;

  for (const domain of domainDefs) {
    const rootClaims = beliefs
      .filter((belief) => belief.domain === domain.id && belief.parent === domain.id)
      .sort((a, b) => claimCatalog.findIndex((item) => item.claim === a.claim) - claimCatalog.findIndex((item) => item.claim === b.claim));
    const claimBlocks = rootClaims.map((root) => {
      const children = beliefs
        .filter((belief) => belief.parent === root.claim)
        .sort((a, b) => claimCatalog.findIndex((item) => item.claim === a.claim) - claimCatalog.findIndex((item) => item.claim === b.claim));
      return { root, children, height: Math.max(rowHeight, children.length * childHeight) };
    });
    const blockHeight = Math.max(rowHeight, claimBlocks.reduce((sum, block) => sum + block.height, 0));
    const rootY = cursorY + blockHeight / 2 - 29;
    const domainProgress = domains.find((item) => item.id === domain.id);

    nodes.push(`
      <article class="tree-node root" style="left:${rootX}px; top:${rootY}px">
        <strong>${domain.name}</strong>
        <div class="tags"><span class="tag">${domainProgress.percent}%</span></div>
      </article>
    `);

    let blockY = cursorY;
    for (const block of claimBlocks) {
      const belief = block.root;
      const accepted = world.discoveries.includes(belief.claim);
      const y = blockY + block.height / 2 - 32;
      const edgeColor = accepted ? "#5d9b62" : "#c8a25b";
      edges.push(`<path d="M ${rootX + 122} ${rootY + 29} C ${rootX + 148} ${rootY + 29}, ${claimX - 32} ${y + 29}, ${claimX} ${y + 29}" stroke="${edgeColor}" stroke-width="3" fill="none" stroke-linecap="round" />`);
      nodes.push(`
        <article class="tree-node ${accepted ? "discovered" : "hypothesis"}" style="left:${claimX}px; top:${y}px">
          <strong>${accepted ? "Discovered" : "Hypothesis"}</strong>
          <small>${belief.claim}</small>
          <div class="meter"><span style="width:${Math.round(belief.confidence * 100)}%; background:${accepted ? "#65a969" : "#c9a96e"}"></span></div>
          <div class="tags">
            <span class="tag">${Math.round(belief.confidence * 100)}% conf</span>
            <span class="tag">${Math.round(belief.neural * 100)}% net</span>
          </div>
        </article>
      `);

      block.children.forEach((child, childIndex) => {
        const childAccepted = world.discoveries.includes(child.claim);
        const unlocked = isClaimUnlocked(child);
        const childY = blockY + childIndex * childHeight + 2;
        const childEdgeColor = childAccepted ? "#5d9b62" : unlocked ? "#c8a25b" : "#b9ad95";
        edges.push(`<path d="M ${claimX + 178} ${y + 29} C ${claimX + 210} ${y + 29}, ${childX - 34} ${childY + 29}, ${childX} ${childY + 29}" stroke="${childEdgeColor}" stroke-width="3" fill="none" stroke-linecap="round" />`);
        nodes.push(`
          <article class="tree-node ${childAccepted ? "discovered" : unlocked ? "hypothesis" : "locked"}" style="left:${childX}px; top:${childY}px">
            <strong>${childAccepted ? "Discovered" : unlocked ? "Derived" : "Locked"}</strong>
            <small>${child.claim}</small>
            <div class="meter"><span style="width:${Math.round(child.confidence * 100)}%; background:${childAccepted ? "#65a969" : unlocked ? "#c9a96e" : "#aaa08c"}"></span></div>
            <div class="tags">
              <span class="tag">${Math.round(child.confidence * 100)}% conf</span>
              <span class="tag">${Math.round(child.neural * 100)}% net</span>
            </div>
          </article>
        `);
      });

      blockY += block.height;
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
  world.history = world.history.slice(-80);
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
  const xFor = (index) => pad.left + (history.length === 1 ? plotW : (index / (history.length - 1)) * plotW);
  const yFor = (value) => pad.top + plotH - (value / 100) * plotH;
  const grid = [0, 25, 50, 75, 100].map((value) => `
    <line x1="${pad.left}" y1="${yFor(value)}" x2="${width - pad.right}" y2="${yFor(value)}" stroke="#dfcea5" stroke-width="1" />
    <text x="6" y="${yFor(value) + 4}" fill="#7a6f80" font-size="10">${value}</text>
  `).join("");
  const paths = domains.map((domain) => {
    const points = history.map((entry, index) => `${xFor(index)},${yFor(entry.domains[domain.id] ?? 0)}`).join(" ");
    return `<polyline points="${points}" fill="none" stroke="${domain.color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />`;
  }).join("");
  const overallPoints = history.map((entry, index) => `${xFor(index)},${yFor(entry.overall)}`).join(" ");
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
      <text x="${pad.left}" y="${height - 7}" fill="#7a6f80" font-size="11">Day ${history[0].day}</text>
      <text x="${width - 62}" y="${height - 7}" fill="#7a6f80" font-size="11">Day ${world.day}</text>
    </svg>
    <div class="domain-legend">${legend}<span class="legend-item"><span class="legend-swatch" style="background:#2c2740"></span>Overall ${overall}%</span></div>
  `;
}

function renderPanels() {
  const beliefs = aggregateBeliefs();
  const domains = domainDefs.map((domain) => {
    const claims = beliefs.filter((belief) => belief.domain === domain.id);
    const accepted = claims.filter((belief) => world.discoveries.includes(belief.claim)).length;
    const confidence = claims.reduce((sum, belief) => sum + belief.confidence, 0) / claims.length;
    const percent = Math.round(((accepted / domain.goal) * 0.7 + confidence * 0.3) * 100);
    return { ...domain, accepted, confidence, percent: clamp(percent, 0, 100) };
  });
  const overall = Math.round(domains.reduce((sum, domain) => sum + domain.percent, 0) / domains.length);

  ui.overall.textContent = `${overall}% charted`;
  ui.domains.innerHTML = renderUnderstandingGraph(domains, overall);

  ui.tree.innerHTML = renderDiscoveryTree(beliefs, domains);

  ui.beliefs.innerHTML = beliefs.slice(0, 5).map((belief) => `
    <article class="belief">
      <strong>${belief.claim}</strong>
      <div class="meter"><span style="width:${Math.round(belief.confidence * 100)}%"></span></div>
      <div class="tags">
        <span class="tag">${domainName(belief.domain)}</span>
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

setWeather();
addLog("The village woke inside a small green world and began asking why plants change.");
agents.forEach(chooseTarget);
renderHud();
renderPanels();
loop();
