// ============================================================
// NeuroNav
// A*, Dijkstra, and Q-learning racing on generated mazes.
// ============================================================

const N = 16;
const MAX_STEPS = 700;
const TRAIN_BATCH = 30;
const FRAME_STEP_MS = 65;

const DIRECTIONS = [
  [0, -1], // up
  [1, 0],  // right
  [0, 1],  // down
  [-1, 0]  // left
];

const COLORS = {
  background: '#080d18',
  grid: '#162039',
  wall: '#9eabc2',
  astar: '#61a5ff',
  dijkstra: '#42d6a4',
  neural: '#ffb454',
  goal: '#ff6b9d'
};

const canvas = document.getElementById('maze');
const ctx = canvas.getContext('2d');

const ui = {
  train: document.getElementById('train'),
  race: document.getElementById('race'),
  reset: document.getElementById('reset'),
  status: document.getElementById('status'),
  aStat: document.getElementById('aStat'),
  dStat: document.getElementById('dStat'),
  qStat: document.getElementById('qStat'),
  episodes: document.getElementById('episodes'),
  epsilon: document.getElementById('epsilon'),
  states: document.getElementById('states'),
  wins: document.getElementById('wins')
};

// 256 local wall/visit patterns × 9 goal directions × 4 actions.
const qTable = new Float32Array(256 * 9 * 4);
const seenStates = new Uint8Array(256 * 9);

let episodes = 0;
let epsilon = 0.35;
let learnedStates = 0;
let trainingWins = 0;
let training = false;

let race = null;
let lastFrame = 0;

// ------------------------------------------------------------
// Utilities
// ------------------------------------------------------------

const randomInt = max => Math.floor(Math.random() * max);

function neighbour(cell, direction) {
  return cell +
    DIRECTIONS[direction][1] * N +
    DIRECTIONS[direction][0];
}

function openDirections(walls, cell) {
  const result = [];

  for (let direction = 0; direction < 4; direction++) {
    if (!(walls[cell] & (1 << direction))) {
      result.push(direction);
    }
  }

  return result;
}

// ------------------------------------------------------------
// Maze generation
// ------------------------------------------------------------

function generateMaze() {
  const walls = new Uint8Array(N * N).fill(15);
  const visited = new Uint8Array(N * N);
  const stack = [randomInt(N * N)];

  visited[stack[0]] = 1;

  while (stack.length) {
    const cell = stack[stack.length - 1];
    const x = cell % N;
    const y = Math.floor(cell / N);
    const options = [];

    for (let direction = 0; direction < 4; direction++) {
      const nx = x + DIRECTIONS[direction][0];
      const ny = y + DIRECTIONS[direction][1];

      if (
        nx >= 0 &&
        ny >= 0 &&
        nx < N &&
        ny < N &&
        !visited[ny * N + nx]
      ) {
        options.push(direction);
      }
    }

    if (!options.length) {
      stack.pop();
      continue;
    }

    const direction = options[randomInt(options.length)];
    const next = neighbour(cell, direction);

    walls[cell] &= ~(1 << direction);
    walls[next] &= ~(1 << ((direction + 2) % 4));

    visited[next] = 1;
    stack.push(next);
  }

  return walls;
}

// ------------------------------------------------------------
// Shortest path
// ------------------------------------------------------------

function shortestDistance(walls, start, goal) {
  const distance = new Int16Array(N * N).fill(-1);
  const queue = [start];

  distance[start] = 0;

  for (let head = 0; head < queue.length; head++) {
    const cell = queue[head];

    if (cell === goal) {
      return distance[cell];
    }

    for (const direction of openDirections(walls, cell)) {
      const next = neighbour(cell, direction);

      if (distance[next] < 0) {
        distance[next] = distance[cell] + 1;
        queue.push(next);
      }
    }
  }

  return -1;
}

// ------------------------------------------------------------
// Q-learning environment
// ------------------------------------------------------------

function createEnvironment(maxSteps = MAX_STEPS) {
  const walls = generateMaze();

  let start;
  let goal;

  do {
    start = randomInt(N * N);
    goal = randomInt(N * N);
  } while (
    Math.abs(start % N - goal % N) +
    Math.abs(
      Math.floor(start / N) -
      Math.floor(goal / N)
    ) < N / 2
  );

  const visits = new Uint8Array(N * N);
  visits[start] = 1;

  return {
    walls,
    position: start,
    goal,
    visits,
    steps: 0,
    done: false,
    won: false,
    maxSteps,
    optimal: shortestDistance(walls, start, goal),
    trail: [start]
  };
}

function getObservation(environment) {
  let key = 0;
  const cell = environment.position;

  for (let direction = 0; direction < 4; direction++) {
    const next = neighbour(cell, direction);

    const state =
      (environment.walls[cell] >> direction) & 1
        ? 0
        : 1 + Math.min(environment.visits[next], 2);

    key += state << (2 * direction);
  }

  const goalX =
    Math.sign(
      (environment.goal % N) -
      (cell % N)
    ) + 1;

  const goalY =
    Math.sign(
      Math.floor(environment.goal / N) -
      Math.floor(cell / N)
    ) + 1;

  return key * 9 + goalX * 3 + goalY;
}

function chooseAction(
  environment,
  observation,
  explorationRate
) {
  const available = openDirections(
    environment.walls,
    environment.position
  );

  if (
    Math.random() < explorationRate
  ) {
    return available[
      randomInt(available.length)
    ];
  }

  let bestValue = -Infinity;
  let bestActions = [];

  for (const action of available) {
    const value =
      qTable[observation * 4 + action];

    if (value > bestValue + 1e-8) {
      bestValue = value;
      bestActions = [action];
    } else if (
      Math.abs(value - bestValue) < 1e-8
    ) {
      bestActions.push(action);
    }
  }

  return bestActions[
    randomInt(bestActions.length)
  ];
}

function moveAgent(environment, action) {
  const next = neighbour(
    environment.position,
    action
  );

  let reward = -0.04;

  if (environment.visits[next]) {
    reward -= 0.12;
  }

  environment.position = next;
  environment.visits[next]++;
  environment.steps++;
  environment.trail.push(next);

  if (next === environment.goal) {
    environment.done = true;
    environment.won = true;
    reward = 10;
  } else if (
    environment.steps >= environment.maxSteps
  ) {
    environment.done = true;
  }

  return reward;
}

// ------------------------------------------------------------
// Q-learning
// ------------------------------------------------------------

function trainEpisode() {
  const environment = createEnvironment();

  const learningRate = 0.18;
  const discount = 0.96;

  epsilon = Math.max(
    0.03,
    0.35 * Math.exp(-episodes / 1800)
  );

  let observation =
    getObservation(environment);

  while (!environment.done) {
    const action = chooseAction(
      environment,
      observation,
      epsilon
    );

    const reward =
      moveAgent(environment, action);

    const nextObservation =
      getObservation(environment);

    if (!seenStates[observation]) {
      seenStates[observation] = 1;
      learnedStates++;
    }

    let target = reward;

    if (!environment.done) {
      let bestNext = -Infinity;

      for (
        const nextAction of openDirections(
          environment.walls,
          environment.position
        )
      ) {
        bestNext = Math.max(
          bestNext,
          qTable[
            nextObservation * 4 +
            nextAction
          ]
        );
      }

      target += discount * bestNext;
    }

    const index =
      observation * 4 + action;

    qTable[index] +=
      learningRate *
      (target - qTable[index]);

    observation = nextObservation;
  }

  episodes++;

  if (environment.won) {
    trainingWins++;
  }
}

// ------------------------------------------------------------
// A* / Dijkstra
// ------------------------------------------------------------

function searchPath(
  walls,
  start,
  goal,
  heuristic
) {
  const size = N * N;
  const INF = 1e9;

  const distance =
    new Float64Array(size);

  const previous =
    new Int16Array(size).fill(-1);

  const queue = [];
  const closed =
    new Uint8Array(size);

  distance.fill(INF);
  distance[start] = 0;

  queue.push({
    cell: start,
    priority: heuristic(start)
  });

  while (queue.length) {
    queue.sort(
      (a, b) =>
        a.priority - b.priority
    );

    const current =
      queue.shift().cell;

    if (closed[current]) {
      continue;
    }

    closed[current] = 1;

    if (current === goal) {
      break;
    }

    for (
      const direction of openDirections(
        walls,
        current
      )
    ) {
      const next =
        neighbour(
          current,
          direction
        );

      const newDistance =
        distance[current] + 1;

      if (
        newDistance <
        distance[next]
      ) {
        distance[next] =
          newDistance;

        previous[next] =
          current;

        queue.push({
          cell: next,
          priority:
            newDistance +
            heuristic(next)
        });
      }
    }
  }

  if (distance[goal] === INF) {
    return [];
  }

  const path = [];

  for (
    let cell = goal;
    cell !== -1;
    cell = previous[cell]
  ) {
    path.push(cell);
  }

  return path.reverse();
}

function dijkstra(walls, start, goal) {
  return searchPath(
    walls,
    start,
    goal,
    () => 0
  );
}

function aStar(walls, start, goal) {
  const goalX = goal % N;
  const goalY = Math.floor(goal / N);

  return searchPath(
    walls,
    start,
    goal,
    cell =>
      Math.abs(
        cell % N - goalX
      ) +
      Math.abs(
        Math.floor(cell / N) -
        goalY
      )
  );
}

// ------------------------------------------------------------
// Race
// ------------------------------------------------------------

function startRace() {
  const walls = generateMaze();

  let start = randomInt(N * N);
  let goal = randomInt(N * N);

  while (start === goal) {
    goal = randomInt(N * N);
  }

  const astarPath =
    aStar(walls, start, goal);

  const dijkstraPath =
    dijkstra(walls, start, goal);

  const neural = {
    walls,
    position: start,
    goal,
    visits: new Uint8Array(N * N),
    trail: [start],
    steps: 0,
    done: false,
    won: false,
    maxSteps: MAX_STEPS
  };

  neural.visits[start] = 1;

  race = {
    walls,
    start,
    goal,

    astarPath,
    dijkstraPath,

    astarIndex: 0,
    dijkstraIndex: 0,

    neural,

    accumulator: 0,

    // IMPORTANT:
    // This remains true after completion.
    // We do NOT set race = null.
    finished: false
  };

  updateRaceStats();

  ui.status.textContent =
    'New maze loaded. All three agents are racing from the same start to the same goal.';
}

function updateRaceStats() {
  if (!race) return;

  ui.aStat.textContent =
    `${race.astarIndex} steps`;

  ui.dStat.textContent =
    `${race.dijkstraIndex} steps`;

  ui.qStat.textContent =
    `${race.neural.steps} steps`;
}

function isRaceFinished() {
  return (
    race.astarIndex >=
      race.astarPath.length - 1 &&

    race.dijkstraIndex >=
      race.dijkstraPath.length - 1 &&

    race.neural.done
  );
}

function advanceRace() {
  if (!race || race.finished) {
    return;
  }

  if (
    race.astarIndex <
    race.astarPath.length - 1
  ) {
    race.astarIndex++;
  }

  if (
    race.dijkstraIndex <
    race.dijkstraPath.length - 1
  ) {
    race.dijkstraIndex++;
  }

  if (!race.neural.done) {
    const observation =
      getObservation(race.neural);

    const action =
      chooseAction(
        race.neural,
        observation,
        0
      );

    moveAgent(
      race.neural,
      action
    );
  }

  updateRaceStats();

  if (isRaceFinished()) {
    race.finished = true;

    // DO NOT clear the race here.
    // The completed maze remains rendered.
    ui.status.textContent =
      'Race complete. Final paths and agent positions remain visible.';
  }
}

// ------------------------------------------------------------
// Drawing
// ------------------------------------------------------------

function drawPath(
  path,
  endIndex,
  color,
  width
) {
  if (path.length < 2) {
    return;
  }

  const size =
    canvas.width / N;

  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  ctx.beginPath();

  for (
    let i = 0;
    i <= endIndex &&
    i < path.length;
    i++
  ) {
    const cell = path[i];

    const x =
      (cell % N + 0.5) *
      size;

    const y =
      (Math.floor(cell / N) + 0.5) *
      size;

    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }

  ctx.stroke();
}

function drawAgent(
  cell,
  color,
  radius
) {
  const size =
    canvas.width / N;

  const x =
    (cell % N + 0.5) *
    size;

  const y =
    (Math.floor(cell / N) + 0.5) *
    size;

  ctx.fillStyle = color;

  ctx.beginPath();
  ctx.arc(
    x,
    y,
    radius,
    0,
    Math.PI * 2
  );
  ctx.fill();

  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function drawMaze() {
  ctx.clearRect(
    0,
    0,
    canvas.width,
    canvas.height
  );

  ctx.fillStyle =
    COLORS.background;

  ctx.fillRect(
    0,
    0,
    canvas.width,
    canvas.height
  );

  if (!race) {
    return;
  }

  const size =
    canvas.width / N;

  // Grid
  ctx.strokeStyle =
    COLORS.grid;

  ctx.lineWidth = 1;

  for (let i = 1; i < N; i++) {
    ctx.beginPath();
    ctx.moveTo(
      i * size,
      0
    );
    ctx.lineTo(
      i * size,
      canvas.height
    );
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(
      0,
      i * size
    );
    ctx.lineTo(
      canvas.width,
      i * size
    );
    ctx.stroke();
  }

  // NeuroNav trail
  drawPath(
    race.neural.trail,
    race.neural.trail.length - 1,
    COLORS.neural,
    3
  );

  // Dijkstra
  drawPath(
    race.dijkstraPath,
    race.dijkstraIndex,
    COLORS.dijkstra,
    3
  );

  // A*
  drawPath(
    race.astarPath,
    race.astarIndex,
    COLORS.astar,
    4
  );

  // Goal
  const goalX =
    (race.goal % N + 0.5) *
    size;

  const goalY =
    (Math.floor(race.goal / N) + 0.5) *
    size;

  ctx.strokeStyle =
    COLORS.goal;

  ctx.lineWidth = 4;

  ctx.beginPath();

  ctx.arc(
    goalX,
    goalY,
    size * 0.23,
    0,
    Math.PI * 2
  );

  ctx.stroke();

  // Maze walls
  ctx.strokeStyle =
    COLORS.wall;

  ctx.lineWidth = 2;

  for (
    let cell = 0;
    cell < N * N;
    cell++
  ) {
    const x =
      (cell % N) * size;

    const y =
      Math.floor(cell / N) *
      size;

    const walls =
      race.walls[cell];

    ctx.beginPath();

    if (walls & 1) {
      ctx.moveTo(x, y);
      ctx.lineTo(
        x + size,
        y
      );
    }

    if (walls & 2) {
      ctx.moveTo(
        x + size,
        y
      );

      ctx.lineTo(
        x + size,
        y + size
      );
    }

    if (walls & 4) {
      ctx.moveTo(
        x,
        y + size
      );

      ctx.lineTo(
        x + size,
        y + size
      );
    }

    if (walls & 8) {
      ctx.moveTo(x, y);

      ctx.lineTo(
        x,
        y + size
      );
    }

    ctx.stroke();
  }

  // Agents are drawn last.
  // This keeps the final positions visible
  // after the race has completed.

  drawAgent(
    race.astarPath[
      race.astarIndex
    ],
    COLORS.astar,
    7
  );

  drawAgent(
    race.dijkstraPath[
      race.dijkstraIndex
    ],
    COLORS.dijkstra,
    7
  );

  drawAgent(
    race.neural.position,
    COLORS.neural,
    7
  );
}

// ------------------------------------------------------------
// UI
// ------------------------------------------------------------

function updateLearningStats() {
  ui.episodes.textContent =
    episodes.toLocaleString();

  ui.epsilon.textContent =
    `${Math.round(
      epsilon * 100
    )}%`;

  ui.states.textContent =
    learnedStates.toLocaleString();

  ui.wins.textContent =
    episodes
      ? `${Math.round(
          (trainingWins / episodes) *
          100
        )}%`
      : '0%';
}

ui.train.addEventListener(
  'click',
  () => {
    training = !training;

    ui.train.textContent =
      training
        ? 'Pause training'
        : 'Start training';

    ui.status.textContent =
      training
        ? 'Training NeuroNav across random mazes…'
        : 'Training paused.';
  }
);

ui.race.addEventListener(
  'click',
  () => {
    training = false;

    ui.train.textContent =
      'Start training';

    startRace();
  }
);

ui.reset.addEventListener(
  'click',
  () => {
    qTable.fill(0);
    seenStates.fill(0);

    episodes = 0;
    epsilon = 0.35;
    learnedStates = 0;
    trainingWins = 0;
    training = false;

    ui.train.textContent =
      'Start training';

    updateLearningStats();

    ui.status.textContent =
      'Brain reset. NeuroNav has no learned experience.';
  }
);

// ------------------------------------------------------------
// Animation loop
// ------------------------------------------------------------

function animationLoop(timestamp) {
  const delta =
    timestamp - lastFrame;

  lastFrame = timestamp;

  if (training) {
    for (
      let i = 0;
      i < TRAIN_BATCH;
      i++
    ) {
      trainEpisode();
    }

    updateLearningStats();
  }

  // Once race.finished becomes true,
  // advanceRace() stops, but drawMaze()
  // continues. This is what keeps the
  // completed maze on screen.

  if (
    race &&
    !race.finished
  ) {
    race.accumulator += delta;

    if (
      race.accumulator >=
      FRAME_STEP_MS
    ) {
      race.accumulator = 0;
      advanceRace();
    }
  }

  drawMaze();

  requestAnimationFrame(
    animationLoop
  );
}

// ------------------------------------------------------------
// Start
// ------------------------------------------------------------

updateLearningStats();
startRace();

requestAnimationFrame(
  animationLoop
);