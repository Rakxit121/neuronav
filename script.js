// ===== Settings =====
const N = 10;                 // the maze is N x N cells
const MAX_TRAIN = 300;        // give up on a practice maze after this many steps
const MAX_TEST = 400;         // same, for the test maze
const D = [[0,-1],[1,0],[0,1],[-1,0]];   // directions: up, right, down, left

// ===== Grab things from the page =====
const $ = id => document.getElementById(id);
const cv = $('maze'), cx = cv.getContext('2d');
const ch = $('chart'), cc = ch.getContext('2d');

// ===== The robot's brain: a table of scores =====
// 256 possible wall/visited patterns x 9 goal directions x 4 possible moves
const Q = new Float32Array(256 * 9 * 4);
const seen = new Uint8Array(256 * 9);
let eps = 0, nSeen = 0, ex = 0.35, training = false, hist = [];
let batch = { s: 0, n: 0, eff: 0 }, demoOn = false, acc = 0, last = 0;
const rnd = n => Math.floor(Math.random() * n);

// ===== 1. Making a random maze =====
// Start in a random cell, keep carving into unvisited neighbours,
// and back up when stuck. Each cell stores which of its 4 walls exist.
function genMaze() {
  const w = new Uint8Array(N * N).fill(15);   // 15 = all four walls present
  const vis = new Uint8Array(N * N);
  const st = [rnd(N * N)];
  vis[st[0]] = 1;
  while (st.length) {
    const c = st[st.length - 1], x = c % N, y = Math.floor(c / N), opts = [];
    D.forEach((d, i) => {
      const nx = x + d[0], ny = y + d[1];
      if (nx >= 0 && ny >= 0 && nx < N && ny < N && !vis[ny * N + nx]) opts.push(i);
    });
    if (!opts.length) { st.pop(); continue; }
    const i = opts[rnd(opts.length)];
    const n = (y + D[i][1]) * N + x + D[i][0];
    w[c] &= ~(1 << i);                 // knock down the wall here...
    w[n] &= ~(1 << ((i + 2) % 4));     // ...and the matching wall next door
    vis[n] = 1;
    st.push(n);
  }
  return w;
}

// Shortest possible path (breadth-first search), used to judge the robot
function shortest(w, a, b) {
  const d = new Int16Array(N * N).fill(-1), q = [a];
  d[a] = 0;
  for (let h = 0; h < q.length; h++) {
    const c = q[h];
    if (c === b) return d[c];
    for (let i = 0; i < 4; i++) {
      if (!((w[c] >> i) & 1)) {
        const n = c + D[i][1] * N + D[i][0];
        if (d[n] < 0) { d[n] = d[c] + 1; q.push(n); }
      }
    }
  }
  return -1;
}

// ===== 2. One "episode": a maze, a start, a goal =====
function newEnv(max) {
  const w = genMaze();
  let s, g;
  do { s = rnd(N * N); g = rnd(N * N); }
  while (Math.abs(s % N - g % N) + Math.abs(Math.floor(s / N) - Math.floor(g / N)) < 8);
  const v = new Uint8Array(N * N);   // how many times each cell was visited
  v[s] = 1;
  return { w, pos: s, g, v, steps: 0, done: false, won: false, max, opt: shortest(w, s, g), trail: [s] };
}

// ===== 3. What the robot can sense =====
// For each side: wall (0), new cell (1), visited once (2), visited a lot (3).
// Plus: is the goal left/right and up/down? All squashed into one number.
function obs(e) {
  const c = e.pos;
  let k = 0;
  for (let i = 0; i < 4; i++) {
    const s = ((e.w[c] >> i) & 1) ? 0 : 1 + Math.min(e.v[c + D[i][1] * N + D[i][0]], 2);
    k += s << (2 * i);
  }
  const gx = Math.sign((e.g % N) - (c % N)) + 1;                        // 0, 1 or 2
  const gy = Math.sign(Math.floor(e.g / N) - Math.floor(c / N)) + 1;    // 0, 1 or 2
  return k * 9 + gx * 3 + gy;
}

function open(e) {   // which directions have no wall?
  const o = [];
  for (let i = 0; i < 4; i++) if (!((e.w[e.pos] >> i) & 1)) o.push(i);
  return o;
}

// ===== 4. Choosing a move =====
// Sometimes random (to explore), otherwise the move with the best score.
function act(e, s, epsilon) {
  const ok = open(e);
  if (Math.random() < epsilon) return ok[rnd(ok.length)];
  let best = -1e9, bs = [];
  for (const i of ok) {
    const q = Q[s * 4 + i];
    if (q > best + 1e-9) { best = q; bs = [i]; }
    else if (Math.abs(q - best) <= 1e-9) bs.push(i);
  }
  return bs[rnd(bs.length)];
}

// Make the move and return a reward
function step(e, a) {
  const n = e.pos + D[a][1] * N + D[a][0];
  let r = -0.05;                    // small cost for every step
  if (e.v[n] > 0) r -= 0.15;        // extra cost for going back to a visited cell
  e.pos = n; e.v[n]++; e.steps++; e.trail.push(n);
  if (n === e.g) { e.done = e.won = true; r = 10; }   // big reward for the goal
  else if (e.steps >= e.max) e.done = true;
  return r;
}

// ===== 5. Learning =====
// After each move: nudge the score of that move toward
// (reward now + best score available in the next situation).
function trainEp() {
  const e = newEnv(MAX_TRAIN), al = 0.2, ga = 0.95;
  ex = Math.max(0.05, 0.35 * Math.exp(-eps / 1500));   // explore less over time
  let s = obs(e);
  while (!e.done) {
    const a = act(e, s, ex), r = step(e, a), s2 = obs(e);
    if (!seen[s]) { seen[s] = 1; nSeen++; }
    let t = r;
    if (!e.done) {
      let m = -1e9;
      for (const i of open(e)) m = Math.max(m, Q[s2 * 4 + i]);
      t += ga * m;
    }
    Q[s * 4 + a] += al * (t - Q[s * 4 + a]);
    s = s2;
  }
  eps++; batch.n++;
  if (e.won) { batch.s++; batch.eff += e.opt / e.steps; }
  if (batch.n === 100) {   // every 100 mazes, add a point to the chart
    hist.push([batch.s / 100, batch.s ? batch.eff / batch.s : 0]);
    batch = { s: 0, n: 0, eff: 0 };
  }
}

// ===== 6. Testing on a brand-new maze =====
let demo = newEnv(MAX_TEST);
const msg = t => { $('msg').textContent = t; };

function startTest() {
  demo = newEnv(MAX_TEST);
  demoOn = true;
  acc = 0;
  msg('Testing on a brand-new maze...');
}
function report() {
  const e = demo;
  msg(e.won
    ? `Reached the goal in ${e.steps} steps. Shortest possible: ${e.opt}.`
    : `Stuck. Hit the ${e.max}-step limit without reaching the goal.`);
}
function stats() {
  $('sEp').textContent = eps.toLocaleString();
  $('sEx').textContent = Math.round(ex * 100) + '%';
  $('sSeen').textContent = nSeen;
}

// ===== 7. Drawing =====
function draw() {
  const css = getComputedStyle(document.documentElement);
  const c = n => css.getPropertyValue(n).trim();
  const W = cv.width, S = W / N, e = demo;
  cx.clearRect(0, 0, W, W);

  // blue shading on visited cells
  for (let i = 0; i < N * N; i++) {
    if (e.v[i]) {
      cx.fillStyle = `rgba(${c('--heat')},${Math.min(0.12 * e.v[i], 0.4)})`;
      cx.fillRect((i % N) * S, Math.floor(i / N) * S, S, S);
    }
  }
  // goal (two rings)
  const gx = (e.g % N + 0.5) * S, gy = (Math.floor(e.g / N) + 0.5) * S;
  cx.strokeStyle = c('--goal'); cx.lineWidth = 4;
  [0.3, 0.15].forEach(r => { cx.beginPath(); cx.arc(gx, gy, r * S, 0, 7); cx.stroke(); });
  // trail
  cx.strokeStyle = c('--trail'); cx.lineWidth = 4; cx.lineJoin = 'round';
  cx.beginPath();
  e.trail.forEach((t, i) => {
    const x = (t % N + 0.5) * S, y = (Math.floor(t / N) + 0.5) * S;
    i ? cx.lineTo(x, y) : cx.moveTo(x, y);
  });
  cx.stroke();
  // walls
  cx.strokeStyle = c('--ink'); cx.lineWidth = 5; cx.lineCap = 'round';
  cx.beginPath();
  for (let i = 0; i < N * N; i++) {
    const x = (i % N) * S, y = Math.floor(i / N) * S, w = e.w[i];
    if (w & 1) { cx.moveTo(x, y);         cx.lineTo(x + S, y); }
    if (w & 2) { cx.moveTo(x + S, y);     cx.lineTo(x + S, y + S); }
    if (w & 4) { cx.moveTo(x, y + S);     cx.lineTo(x + S, y + S); }
    if (w & 8) { cx.moveTo(x, y);         cx.lineTo(x, y + S); }
  }
  cx.stroke();
  // robot: a rounded square with a small tick toward each open side
  const rx = (e.pos % N + 0.5) * S, ry = (Math.floor(e.pos / N) + 0.5) * S;
  cx.lineWidth = 3;
  for (const i of open(e)) {
    cx.beginPath();
    cx.moveTo(rx + D[i][0] * 0.2 * S, ry + D[i][1] * 0.2 * S);
    cx.lineTo(rx + D[i][0] * 0.4 * S, ry + D[i][1] * 0.4 * S);
    cx.stroke();
  }
  cx.fillStyle = c('--robot');
  cx.beginPath(); cx.roundRect(rx - 0.2 * S, ry - 0.2 * S, 0.4 * S, 0.4 * S, 8);
  cx.fill(); cx.stroke();
}

function drawChart() {
  const css = getComputedStyle(document.documentElement);
  const c = n => css.getPropertyValue(n).trim();
  const W = ch.width, H = ch.height;
  cc.clearRect(0, 0, W, H);
  cc.strokeStyle = c('--line'); cc.lineWidth = 1;
  [0, 0.5, 1].forEach(f => {
    const y = H - 6 - (H - 12) * f;
    cc.beginPath(); cc.moveTo(6, y); cc.lineTo(W - 6, y); cc.stroke();
  });
  [[0, '--trail'], [1, '--goal']].forEach(([k, col]) => {
    cc.strokeStyle = c(col); cc.lineWidth = 2; cc.beginPath();
    hist.forEach((h, i) => {
      const x = 6 + (W - 12) * i / Math.max(hist.length - 1, 19);
      const y = H - 6 - (H - 12) * h[k];
      i ? cc.lineTo(x, y) : cc.moveTo(x, y);
    });
    cc.stroke();
  });
}

// ===== 8. The main loop: runs about 60 times per second =====
function loop(t) {
  const dt = t - last; last = t;
  if (training) {
    for (let i = 0; i < 20; i++) trainEp();   // 20 practice mazes per frame
    stats(); drawChart();
  }
  if (demoOn && !demo.done && (acc += dt) > 70) {   // one test step every 70 ms
    acc = 0;
    step(demo, act(demo, obs(demo), 0));
    if (demo.done) { demoOn = false; report(); }
  }
  draw();
  requestAnimationFrame(loop);
}

// ===== 9. Buttons =====
$('btnTrain').onclick = () => {
  training = !training;
  $('btnTrain').textContent = training ? 'Pause training' : 'Resume training';
  if (training) msg('Practising on random mazes. Test it whenever you like.');
};
$('btnTest').onclick = startTest;
$('btnReset').onclick = () => {
  Q.fill(0); seen.fill(0);
  eps = nSeen = 0; ex = 0.35; hist = [];
  batch = { s: 0, n: 0, eff: 0 }; training = false;
  $('btnTrain').textContent = 'Start training';
  stats(); drawChart();
  msg('Brain wiped. Test it now to see an untrained robot wander.');
};

stats(); drawChart();
requestAnimationFrame(loop);