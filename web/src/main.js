import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import './style.css';

const root = document.querySelector('#root');
root.innerHTML = `<header><div class="brand"><span class="mark">✳</span><div><b>FLYSLOP</b><small>REPLAY LAB / NEUROMECHFLY</small></div></div><div class="runmeta"><span class="live-dot"></span><span id="run-label">Loading replay…</span><span id="mode-label" class="pill">—</span></div><label class="source-pick">SOURCE <select id="source-select" aria-label="Replay source"><option value="kinematic">Scripted · kinematic</option></select></label><button id="reload" class="quiet">↻ Reload replay</button></header>
<main><section class="stage"><div class="stage-top"><div><span class="eyebrow">SIMULATION VIEW</span><h1>Contact replay</h1></div><div class="cam-hint">LEFT DRAG ORBIT <i>·</i> RIGHT DRAG PAN <i>·</i> SCROLL ZOOM</div></div>
<div id="scene" class="viewport"><div class="view-tag">3D SCENE <span>●</span></div><div class="cam-buttons"><button data-view="follow" class="active">Follow fly</button><button data-view="close">Close-up</button><button data-view="laptop">Laptop</button></div><div class="scene-help" id="scene-help">NEUROMECHFLY v2 BODY · IK-POSED LEGS · LAPTOP KEYBOARD</div></div>
<div class="transport"><div class="transport-actions"><button id="play" class="play">Ⅱ Pause</button><button id="step">Step →</button><button id="restart">↺ Replay</button><label>Speed <select id="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option><option value="8">8×</option><option value="16">16×</option><option value="32">32×</option></select></label></div><div class="scrub"><span id="tick-label">TICK 0</span><input id="timeline" type="range" min="0" max="0" value="0"><span id="tick-total">0</span></div></div>
<div class="event-focus"><span class="focus-icon">⌖</span><div><b id="focus-title">Waiting for replay</b><small id="focus-detail">The selected event’s pose and output appear here.</small></div><span class="focus-coord" id="pose">POSE —</span></div></section>
<aside class="side"><section class="panel target-panel"><div class="panel-head"><span class="eyebrow">TASK TARGET</span><select id="target-select" aria-label="Select replay target"></select></div><pre id="target-code">Waiting for replay…</pre></section>
<section class="panel output-panel"><div class="panel-head"><span class="eyebrow">MONITOR OUTPUT</span><span class="output-state"><span class="live-dot"></span> LIVE</span></div><pre id="output-code"></pre><div class="output-foot"><span>CHARACTERS <b id="char-count">0</b></span><span id="match-state">IN PROGRESS</span></div><div class="validation" id="validation">Validation · waiting</div></section>
<section class="panel keyboard-panel"><div class="panel-head"><span class="eyebrow">KEY CONTACTS</span><span class="key-count" id="key-count">0 EVENTS</span></div><div id="keyboard" class="keyboard"></div><div id="history" class="history"></div></section>
<section class="panel brain-panel"><div class="panel-head"><span class="eyebrow" id="brain-title">SAMPLED BRAIN ACTIVITY</span><span class="pill modeled" id="brain-pill">SYNTHETIC MODEL</span></div><div class="brain-wrap"><div id="brain" class="brain-canvas"></div><div class="brain-legend"><span><i class="legend-dot active"></i> Activity</span><span id="brain-legend-cells"><i class="legend-dot quiet-dot"></i> Sampled cells</span></div></div><div class="brain-foot" id="brain-readout"></div><div class="brain-foot" id="brain-foot">No connectome topology loaded · scripted policy · leg-state proxy activity</div></section></aside></main>
<footer><span>FLYSLOP / INTERACTIVE REPLAY</span><span>REPLAY DATA IS AUTHORITATIVE · VIEWER DOES NOT GENERATE KEY EVENTS</span></footer>`;

const $ = s => document.querySelector(s);
const LEGS = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH'];

let data, frames = [], field = {}, tickHz = 30;
const params = new URLSearchParams(location.search);
let playing = true, time = 0, lastTime = 0, cursor = 0, selectedTarget = params.get('target') || 'fly_demo';
let selectedSource = params.get('source') || 'kinematic';
let scene, camera, renderer, controls, simRoot, keyMeshes = new Map(), keyboardGroup;
let monitorCanvas, monitorCtx, monitorTexture, lastMonitorText = null;
let fly = null, flyModel = null, cameraMode = params.get('view') || 'follow';
let brainScene, brainRenderer, brainCamera, brainPoints = [];
let layoutInfo = { width: 15, height: 5.6, travel: 0.075, deck_depth: 0.04 };

// ---------------------------------------------------------------------------
// Scene: simulation frame is X right, Y toward the screen, Z up, 1 unit = one
// key pitch. simRoot maps it into three.js' Y-up frame.
function initScene() {
  const host = $('#scene');
  scene = new THREE.Scene();
  scene.background = new THREE.Color('#0f1618');
  scene.fog = new THREE.Fog('#0f1618', 60, 140);
  camera = new THREE.PerspectiveCamera(35, host.clientWidth / host.clientHeight, 0.05, 400);
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(host.clientWidth, host.clientHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  host.appendChild(renderer.domElement);
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.maxDistance = 80;
  controls.minDistance = 1.2;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.addEventListener('start', () => { if (cameraMode === 'laptop') setView(null); });

  scene.add(new THREE.HemisphereLight(0xdfeaf0, 0x2a2622, 0.6));
  const sun = new THREE.DirectionalLight(0xfff1dc, 1.8);
  sun.position.set(-14, 30, 16);
  sun.castShadow = true;
  sun.shadow.mapSize.set(4096, 4096);
  Object.assign(sun.shadow.camera, { left: -16, right: 16, top: 16, bottom: -16, near: 1, far: 80 });
  sun.shadow.bias = -0.0004;
  sun.shadow.normalBias = 0.02;
  scene.add(sun);

  simRoot = new THREE.Group();
  simRoot.rotation.x = -Math.PI / 2;
  scene.add(simRoot);

  const desk = new THREE.Mesh(new THREE.BoxGeometry(90, 60, 1.5), new THREE.MeshStandardMaterial({ color: '#4a3526', roughness: 0.75, map: woodTexture() }));
  desk.position.set(0, -2, -0.84 - 0.75);
  desk.receiveShadow = true;
  simRoot.add(desk);

  window.addEventListener('resize', () => {
    camera.aspect = host.clientWidth / host.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(host.clientWidth, host.clientHeight);
  });
  document.querySelectorAll('.cam-buttons button').forEach(b => b.addEventListener('click', () => setView(b.dataset.view)));
  animate();
}

function woodTexture() {
  const c = document.createElement('canvas'); c.width = 512; c.height = 512;
  const x = c.getContext('2d');
  x.fillStyle = '#6b4a33'; x.fillRect(0, 0, 512, 512);
  for (let i = 0; i < 140; i++) {
    x.strokeStyle = `rgba(${40 + Math.random() * 30},${24 + Math.random() * 18},${14},${0.08 + Math.random() * 0.12})`;
    x.lineWidth = 1 + Math.random() * 3;
    x.beginPath(); const y = Math.random() * 512; x.moveTo(0, y);
    for (let px = 0; px <= 512; px += 32) x.lineTo(px, y + Math.sin(px / 60 + i) * 6);
    x.stroke();
  }
  const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(3, 2); t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// ---------------------------------------------------------------------------
// Laptop: aluminium unibody, recessed black keys, trackpad, hinged display.
function buildLaptop(layout) {
  if (keyboardGroup) simRoot.remove(keyboardGroup);
  keyMeshes.clear();
  keyboardGroup = new THREE.Group();
  simRoot.add(keyboardGroup);
  const W = layout.width, H = layout.height, deck = -layout.deck_depth;
  const baseW = 17.4, baseD = 16.2, thick = 0.8;   // deep palm rest: the fly's hind legs stand on it
  const backY = H / 2 + 0.75, frontY = backY - baseD;
  const alu = new THREE.MeshStandardMaterial({ color: '#b9bcc0', metalness: 0.85, roughness: 0.32 });
  const base = new THREE.Mesh(new RoundedBoxGeometry(baseW, baseD, thick, 4, 0.35), alu);
  base.position.set(0, (backY + frontY) / 2, deck - thick / 2);
  base.castShadow = base.receiveShadow = true;
  keyboardGroup.add(base);
  const well = new THREE.Mesh(new THREE.PlaneGeometry(W + 0.12, H + 0.12), new THREE.MeshStandardMaterial({ color: '#141516', roughness: 0.8 }));
  well.position.set(0, 0, deck + 0.002);
  well.receiveShadow = true;
  keyboardGroup.add(well);
  const pad = new THREE.Mesh(new RoundedBoxGeometry(6.6, 4.1, 0.02, 2, 0.01), new THREE.MeshStandardMaterial({ color: '#a9adb1', metalness: 0.6, roughness: 0.22 }));
  pad.position.set(0, -H / 2 - 0.55 - 2.05, deck + 0.001);
  pad.receiveShadow = true;
  keyboardGroup.add(pad);

  const capMat = new THREE.MeshStandardMaterial({ color: '#1b1c1e', roughness: 0.55, metalness: 0.05 });
  for (const k of layout.keys) {
    const cx = k.x + k.width / 2 - W / 2, cy = H / 2 - (k.y + k.height / 2), tall = 0.09;
    const top = new THREE.MeshStandardMaterial({ color: '#ffffff', roughness: 0.55, map: keyLabel(k), emissive: '#000000' });
    const mesh = new THREE.Mesh(new RoundedBoxGeometry(k.width, k.height, tall, 2, 0.035), [capMat, capMat, capMat, capMat, top, capMat]);
    fixRoundedBoxGroups(mesh.geometry);
    mesh.position.set(cx, cy, -tall / 2);
    mesh.castShadow = false; mesh.receiveShadow = true;
    mesh.userData = { key: k, rest: -tall / 2, top };
    keyboardGroup.add(mesh);
    keyMeshes.set(k.id, mesh);
  }

  // Display lid, hinged at the back edge and opened ~110°.
  const hinge = new THREE.Group();
  hinge.position.set(0, backY - 0.1, deck + 0.05);
  hinge.rotation.x = THREE.MathUtils.degToRad(-20);
  keyboardGroup.add(hinge);
  const lidH = 11.4;
  const lid = new THREE.Mesh(new RoundedBoxGeometry(baseW, 0.3, lidH, 4, 0.14), alu);
  lid.position.set(0, 0.15, lidH / 2);
  lid.castShadow = true;
  hinge.add(lid);
  const bezel = new THREE.Mesh(new THREE.PlaneGeometry(baseW - 0.4, lidH - 0.4), new THREE.MeshStandardMaterial({ color: '#07080a', roughness: 0.3 }));
  bezel.rotation.x = Math.PI / 2; bezel.position.set(0, -0.005, lidH / 2);
  hinge.add(bezel);
  monitorCanvas = document.createElement('canvas'); monitorCanvas.width = 1600; monitorCanvas.height = 1040;
  monitorCtx = monitorCanvas.getContext('2d');
  monitorTexture = new THREE.CanvasTexture(monitorCanvas);
  monitorTexture.colorSpace = THREE.SRGBColorSpace;
  monitorTexture.anisotropy = 8;
  const screen = new THREE.Mesh(new THREE.PlaneGeometry(baseW - 1.0, (baseW - 1.0) * 1040 / 1600), new THREE.MeshBasicMaterial({ map: monitorTexture, toneMapped: false }));
  screen.rotation.x = Math.PI / 2; screen.position.set(0, -0.01, lidH / 2 + 0.2);
  hinge.add(screen);
  lastMonitorText = null;
  drawMonitor('');
}

// RoundedBoxGeometry has one group; split faces by normal so the top gets the label.
function fixRoundedBoxGroups(geo) {
  const n = geo.attributes.normal, idx = geo.index, count = idx ? idx.count : n.count;
  geo.clearGroups();
  const tris = [];
  for (let i = 0; i < count; i += 3) {
    const a = idx ? idx.getX(i) : i;
    tris.push(n.getZ(a) > 0.9 ? 4 : 0);
  }
  let start = 0;
  for (let t = 1; t <= tris.length; t++) {
    if (t === tris.length || tris[t] !== tris[start]) { geo.addGroup(start * 3, (t - start) * 3, tris[start]); start = t; }
  }
  // Planar UVs across the top so the label is not distorted by the bevel.
  geo.computeBoundingBox();
  const bb = geo.boundingBox, p = geo.attributes.position, uv = geo.attributes.uv;
  for (let i = 0; i < p.count; i++) uv.setXY(i, (p.getX(i) - bb.min.x) / (bb.max.x - bb.min.x), (p.getY(i) - bb.min.y) / (bb.max.y - bb.min.y));
  uv.needsUpdate = true;
}

function keyLabel(k) {
  const px = 96, c = document.createElement('canvas');
  c.width = Math.round(px * k.width); c.height = Math.round(px * k.height);
  const x = c.getContext('2d');
  x.fillStyle = '#1b1c1e'; x.fillRect(0, 0, c.width, c.height);
  x.fillStyle = '#e8e8e8';
  const alt = k.shift_output && k.shift_output !== k.output && !/[a-z]/.test(k.output);
  const word = k.label.length > 2;
  x.font = `${word ? 17 : alt ? 26 : 32}px -apple-system, "Helvetica Neue", Arial, sans-serif`;
  if (alt) {
    x.textAlign = 'center';
    x.fillText(k.shift_output, c.width / 2, c.height * 0.42);
    x.fillText(k.output, c.width / 2, c.height * 0.8);
  } else if (word) {
    const right = ['Backspace', 'Enter', 'ShiftRight', 'MetaRight', 'AltRight'].includes(k.id);
    x.textAlign = right ? 'right' : 'left';
    x.fillText(k.label, right ? c.width - 10 : 10, c.height - 12);
  } else {
    x.textAlign = 'center'; x.textBaseline = 'middle';
    x.fillText(k.label, c.width / 2, c.height / 2 + 1);
  }
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4;
  return t;
}

const SV_KEYWORDS = /\b(module|endmodule|input|output|logic|wire|reg|assign|always_ff|always_comb|always|begin|end|if|else|case|endcase|posedge|negedge|parameter|localparam|typedef|enum|struct|packed|default)\b/g;
// Cursor/selection come from the last key event (C4 cursor_after / selection_after); missing fields mean the cursor sits at the end.
function editorView(text, ev) {
  const n = (text || '').length, ok = v => Number.isInteger(v) && v >= 0 && v <= n;
  const cur = ev && ok(ev.cursor_after) ? ev.cursor_after : n;
  const s = ev?.selection_after;
  // selection_after is [start, end] | null; the older {anchor, head} form is still accepted.
  const [a, b] = Array.isArray(s) ? s : s ? [s.anchor, s.head] : [];
  const sel = ok(a) && ok(b) && a !== b ? [Math.min(a, b), Math.max(a, b)] : null;
  return { cur, sel };
}
function drawMonitor(text, view = editorView(text)) {
  const stamp = `${view.cur}|${view.sel}|${text}`;
  if (!monitorCtx || stamp === lastMonitorText) return;
  lastMonitorText = stamp;
  const { cur, sel } = view;
  const x = monitorCtx, W = monitorCanvas.width, H = monitorCanvas.height;
  x.fillStyle = '#1e1f24'; x.fillRect(0, 0, W, H);
  x.fillStyle = '#2a2c33'; x.fillRect(0, 0, W, 64);
  ['#ff5f57', '#febc2e', '#28c840'].forEach((c, i) => { x.fillStyle = c; x.beginPath(); x.arc(34 + i * 30, 32, 9, 0, 7); x.fill(); });
  x.fillStyle = '#1e1f24'; x.fillRect(150, 14, 300, 50);
  x.fillStyle = '#c9ccd4'; x.font = '24px ui-monospace, Menlo, monospace';
  x.fillText(`${data?.metadata?.target_id || 'untitled'}.sv`, 172, 48);
  const lines = (text || '').split('\n'), lh = 56, top = 130, visible = Math.floor((H - top - 40) / lh);
  x.font = '42px ui-monospace, Menlo, monospace';
  const charW = x.measureText('M').width;
  let curLine = 0, curCol = 0, lineStart = 0;
  lines.forEach((line, k) => { if (cur >= lineStart && cur <= lineStart + line.length) { curLine = k; curCol = cur - lineStart; } lineStart += line.length + 1; });
  const first = Math.max(0, curLine - visible + 1);
  let offset = lines.slice(0, first).reduce((a, l) => a + l.length + 1, 0);
  lines.slice(first, first + visible).forEach((line, i) => {
    const y = top + i * lh, lineOffset = offset;
    offset += line.length + 1;
    if (sel && sel[0] <= lineOffset + line.length && sel[1] > lineOffset) {
      const a = Math.max(sel[0], lineOffset) - lineOffset, b = Math.min(sel[1], lineOffset + line.length + 1) - lineOffset;
      x.fillStyle = '#264f78'; x.fillRect(130 + a * charW, y - 40, Math.max(b - a, 0) * charW + (sel[1] > lineOffset + line.length ? charW * 0.4 : 0), lh - 6);
    }
    x.fillStyle = '#5c6070'; x.textAlign = 'right'; x.fillText(String(first + i + 1), 90, y); x.textAlign = 'left';
    let px = 130;
    for (const part of line.split(/(\b\w+\b|\/\/.*$|'[a-z]\w*|\d+)/)) {
      if (!part) continue;
      x.fillStyle = part.startsWith('//') ? '#6a9955' : SV_KEYWORDS.test(part) ? '#c586c0' : /^\d|^'/.test(part) ? '#b5cea8' : '#d4d4d4';
      SV_KEYWORDS.lastIndex = 0;
      x.fillText(part, px, y); px += x.measureText(part).width;
    }
    if (first + i === curLine) { x.fillStyle = '#aeafad'; x.fillRect(130 + curCol * charW - 2, y - 38, 4, 48); }
  });
  x.fillStyle = '#007acc'; x.fillRect(0, H - 36, W, 36);
  x.fillStyle = '#ffffff'; x.font = '20px -apple-system, Arial, sans-serif';
  x.fillText(`SystemVerilog  ·  Ln ${curLine + 1}, Col ${curCol + 1}  ·  ${[...(text || '')].length} chars`, 20, H - 11);
  monitorTexture.needsUpdate = true;
}

// ---------------------------------------------------------------------------
// NeuroMechFly body: same kinematic tree the backend poses.
async function loadFlyModel() {
  const [model, bin] = await Promise.all([
    fetch('/api/fly/model.json').then(r => r.json()),
    fetch('/api/fly/meshes.bin').then(r => r.arrayBuffer()),
  ]);
  const geometries = model.meshes.map(m => {
    const pos = new Float32Array(bin, m.vertex_offset, m.vertex_count * 3);
    const idx = new Uint32Array(bin.slice(m.face_offset, m.face_offset + m.face_count * 12));
    if (signedVolume(pos, idx) < 0) for (let i = 0; i < idx.length; i += 3) [idx[i + 1], idx[i + 2]] = [idx[i + 2], idx[i + 1]];
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    g.computeVertexNormals();
    return g;
  });
  return { model, geometries };
}

function signedVolume(p, idx) {
  let v = 0;
  for (let i = 0; i < idx.length; i += 3) {
    const a = idx[i] * 3, b = idx[i + 1] * 3, c = idx[i + 2] * 3;
    v += p[a] * (p[b + 1] * p[c + 2] - p[b + 2] * p[c + 1]) - p[a + 1] * (p[b] * p[c + 2] - p[b + 2] * p[c]) + p[a + 2] * (p[b] * p[c + 1] - p[b + 1] * p[c]);
  }
  return v;
}

function flyMaterial(name) {
  const std = (color, extra = {}) => new THREE.MeshPhysicalMaterial({ color, roughness: 0.62, metalness: 0, clearcoat: 0.15, clearcoatRoughness: 0.6, envMapIntensity: 0.5, ...extra });
  if (/Eye/.test(name)) return std('#8c1812', { roughness: 0.35, clearcoat: 0.8, sheen: 1, sheenColor: '#ff5a3c' });
  if (/Wing/.test(name)) return new THREE.MeshPhysicalMaterial({ color: '#7d878c', transparent: true, opacity: 0.13, roughness: 0.2, iridescence: 0.8, iridescenceIOR: 1.5, envMapIntensity: 0.4, side: THREE.DoubleSide, depthWrite: false });
  if (/Haltere/.test(name)) return std('#d8c49a');
  if (/Arista/.test(name)) return std('#2a1f18');
  if (/A[3-6]/.test(name)) return std(/A[35]/.test(name) ? '#5e4330' : '#3a2a20');
  if (/A1A2/.test(name)) return std('#a07c52');
  if (/Thorax/.test(name)) return std('#7d5e40', { clearcoat: 0.55 });
  if (/Head|Pedicel|Funiculus/.test(name)) return std('#a6784b');
  if (/Rostrum|Haustellum/.test(name)) return std('#b38b62');
  if (/Coxa|Femur/.test(name)) return std('#8a6644');
  return std('#9c7650');
}

function buildFly({ model, geometries }) {
  const bodies = new Map(model.bodies.map(b => [b.name, b]));
  const children = new Map();
  for (const b of model.bodies) { if (!children.has(b.parent)) children.set(b.parent, []); children.get(b.parent).push(b); }
  const joints = new Map();
  const materials = new Map();
  const build = (body, isRoot) => {
    const group = new THREE.Group();
    if (!isRoot) {
      group.position.fromArray(body.pos);
      group.quaternion.set(body.quat[1], body.quat[2], body.quat[3], body.quat[0]);
    }
    let content = group;
    for (const j of body.joints) {
      const node = new THREE.Group();
      node.userData.axis = new THREE.Vector3().fromArray(j.axis);
      node.quaternion.setFromAxisAngle(node.userData.axis, j.rest);
      joints.set(j.name, node);
      content.add(node);
      content = node;
    }
    for (const geom of body.geoms) {
      const mname = model.meshes[geom.mesh].name;
      if (!materials.has(mname)) materials.set(mname, flyMaterial(mname));
      const mesh = new THREE.Mesh(geometries[geom.mesh], materials.get(mname));
      mesh.position.fromArray(geom.pos);
      mesh.quaternion.set(geom.quat[1], geom.quat[2], geom.quat[3], geom.quat[0]);
      mesh.castShadow = !/Wing/.test(mname);
      mesh.receiveShadow = true;
      content.add(mesh);
    }
    for (const child of children.get(body.name) || []) content.add(build(child, false));
    return group;
  };
  const rootGroup = new THREE.Group();
  rootGroup.add(build(bodies.get('Thorax'), true));
  const dofNodes = LEGS.map(leg => model.leg_dofs.map(dof => joints.get(`joint_${leg}${dof}`)));
  return { group: rootGroup, dofNodes };
}

// ---------------------------------------------------------------------------
// Playback. Frames carry thorax pose, 42 leg joint angles, and posed tips.
function frameValue(row, name) { return row[field[name]]; }

function applyFrame(t) {
  if (!frames.length) return null;
  const i = Math.max(0, Math.min(frames.length - 1, Math.floor(t)));
  const j = Math.min(frames.length - 1, i + 1), a = Math.max(0, Math.min(1, t - i));
  const A = frames[i], B = frames[j];
  const lerp = k => A[k] + (B[k] - A[k]) * a;
  const pose = { x: lerp(1), y: lerp(2), z: lerp(3), pitch: lerp(5), roll: lerp(6) };
  let dyaw = B[4] - A[4]; dyaw = Math.atan2(Math.sin(dyaw), Math.cos(dyaw));
  pose.yaw = A[4] + dyaw * a;
  if (fly) {
    fly.group.position.set(pose.x, pose.y, pose.z);
    fly.group.rotation.set(pose.roll, pose.pitch, pose.yaw, 'ZYX');
    const q0 = field['LF.Coxa_yaw'];
    fly.dofNodes.forEach((nodes, leg) => nodes.forEach((node, d) => node.quaternion.setFromAxisAngle(node.userData.axis, lerp(q0 + leg * 7 + d))));
  }
  // Keycaps follow the claw tips that are below the cap surface.
  const t0 = field['LF.tip.x'];
  const depth = new Map();
  for (let leg = 0; leg < 6; leg++) {
    const x = lerp(t0 + leg * 3), y = lerp(t0 + leg * 3 + 1), z = lerp(t0 + leg * 3 + 2);
    if (z >= 0) continue;
    const lx = x + layoutInfo.width / 2, ly = layoutInfo.height / 2 - y;
    for (const [id, mesh] of keyMeshes) {
      const k = mesh.userData.key;
      if (lx >= k.x && lx <= k.x + k.width && ly >= k.y && ly <= k.y + k.height) depth.set(id, Math.min(layoutInfo.travel, -z));
    }
  }
  keyMeshes.forEach((mesh, id) => { mesh.position.z = mesh.userData.rest - (depth.get(id) || 0); });
  return pose;
}

function setView(mode) {
  cameraMode = mode;
  document.querySelectorAll('.cam-buttons button').forEach(b => b.classList.toggle('active', b.dataset.view === mode));
  if (!mode || !camera) return;
  const p = fly ? fly.group.position : new THREE.Vector3();
  if (mode === 'laptop') { controls.target.copy(simToThree(0, -1.0, 3.5)); camera.position.copy(simToThree(0, -34, 24)); }
  else { controls.target.copy(cameraGoal(mode, p)); camera.position.copy(controls.target).add(CAMERA_OFFSET[mode] || CAMERA_OFFSET.follow); }
}

function laptopTag(replay) {
  const scale = replay.layout?.laptop_scale;
  return scale ? `LAPTOP 1/${Math.round(1 / scale)}` : 'LAPTOP';
}

// Cameras look over the (large) fly at the keys under its forelegs.
const CAMERA_OFFSET = { follow: new THREE.Vector3(0, 20, 15), close: new THREE.Vector3(6, 8, 6) };
function cameraGoal(mode, pose) {
  return mode === 'close' ? simToThree(pose.x, pose.y + 2.6, 0) : simToThree(pose.x * 0.7, pose.y + 2.4, 1.5);
}

function simToThree(x, y, z) { return new THREE.Vector3(x, z, -y); }

function animate(now = 0) {
  requestAnimationFrame(animate);
  if (playing && frames.length) {
    if (!lastTime) lastTime = now;
    time += (now - lastTime) / 1000 * tickHz * Number($('#speed').value);
    lastTime = now;
    if (time >= frames.length - 1) { time = frames.length - 1; playing = false; $('#play').textContent = '▶ Play'; }
    syncCursor();
  } else lastTime = now;
  const pose = applyFrame(time);
  if (pose && (cameraMode === 'follow' || cameraMode === 'close')) {
    const goal = cameraGoal(cameraMode, pose);
    const delta = goal.sub(controls.target).multiplyScalar(0.06);
    controls.target.add(delta); camera.position.add(delta);
  }
  controls?.update();
  renderer?.render(scene, camera);
  if (brainRenderer) brainRenderer.render(brainScene, brainCamera);
}

function syncCursor() {
  const tick = Math.floor(time);
  let next = 0;
  while (next < data.events.length && data.events[next].tick <= tick) next++;
  if (next !== cursor || tick !== Number($('#timeline').value)) { cursor = next; renderState(); }
}

// ---------------------------------------------------------------------------
function initBrain() {
  const host = $('#brain');
  brainScene = new THREE.Scene();
  brainScene.background = new THREE.Color('#111a1c');
  brainCamera = new THREE.PerspectiveCamera(45, host.clientWidth / host.clientHeight, 0.1, 100);
  brainCamera.position.set(0, 0, 9);
  brainRenderer = new THREE.WebGLRenderer({ antialias: true });
  brainRenderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  brainRenderer.setSize(host.clientWidth, host.clientHeight);
  host.appendChild(brainRenderer.domElement);
  initBrainPoints();
}

function initBrainPoints() {
  const geo = new THREE.BufferGeometry(), positions = [], colors = [];
  // A sampled ellipsoid is a visualization only; supplied activity samples modulate its colors.
  for (let i = 0; i < 460; i++) {
    const u = Math.random() * Math.PI * 2, v = Math.acos(2 * Math.random() - 1);
    positions.push(2.55 * Math.sin(v) * Math.cos(u), 1.8 * Math.cos(v), 1.45 * Math.sin(v) * Math.sin(u));
    colors.push(0.16, 0.42, 0.42);
  }
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  brainScene.add(new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.055, vertexColors: true, sizeAttenuation: true })));
  brainPoints = [geo];
}

const GROUP_TINT = { DN: [0.95, 0.66, 0.36], SN: [0.45, 0.78, 0.95], IN: [0.62, 0.86, 0.62], MN: [0.95, 0.45, 0.55] };

// Real circuit: MaleCNS soma positions; colour = group tint scaled by modelled activity.
function setBrainCircuit(connectome) {
  const geo = brainPoints[0];
  const brainObj = brainScene.children.find(c => c.isPoints);
  if (!connectome) {
    $('#brain-title').textContent = 'SAMPLED BRAIN ACTIVITY';
    $('#brain-pill').textContent = 'SYNTHETIC MODEL';
    $('#brain-legend-cells').innerHTML = '<i class="legend-dot quiet-dot"></i> Sampled cells';
    $('#brain-foot').textContent = 'No connectome topology loaded · leg-state proxy activity, not neural data';
    $('#brain-readout').textContent = '';
    if (brainObj && brainObj.userData.circuit) { brainScene.remove(brainObj); initBrainPoints(); }
    return;
  }
  const pos = connectome.positions_um || [];
  if (!pos.length || !connectome.groups) { setBrainCircuit(null); return; }
  const span = Math.max(...pos.flat().map(Math.abs)) || 1;
  const positions = pos.flatMap(([x, y, z]) => [2.4 * x / span, -2.4 * y / span, 2.4 * z / span]);
  if (brainObj) brainScene.remove(brainObj);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(new Array(positions.length).fill(0.3), 3));
  const points = new THREE.Points(g, new THREE.PointsMaterial({ size: 0.11, vertexColors: true, sizeAttenuation: true }));
  points.userData.circuit = true;
  brainScene.add(points);
  brainPoints = [g];
  const counts = connectome.groups.reduce((a, k) => (a[k] = (a[k] || 0) + 1, a), {});
  $('#brain-title').textContent = 'CONNECTOME ACTOR ACTIVITY';
  $('#brain-pill').textContent = 'MODEL ACTIVITY · SIMULATED';
  $('#brain-legend-cells').innerHTML = Object.entries(counts).map(([k, v]) => `<span style="color:rgb(${GROUP_TINT[k].map(c => c * 255).join(',')})">${k} ${v}</span>`).join(' ');
  const schematic = connectome.positions_kind === 'schematic_layout';
  $('#brain-foot').textContent = `Wiring: MaleCNS v1.0 connectome (measured) · activity: model, simulated${schematic ? ' · layout is schematic, not soma positions' : ''}${connectome.label ? ' · ' + connectome.label : ''}`;
  $('#brain-readout').textContent = '';
}

// Per-frame readout: population mean |rate| and the most active cells, all modelled (missing fields are skipped).
function updateBrainReadout(c, t) {
  const el = $('#brain-readout');
  const pm = c.population_mean_abs_rate;
  const parts = [];
  if (pm && typeof pm === 'object') {
    for (const [pop, series] of Object.entries(pm)) {
      const v = Array.isArray(series) ? series[Math.min(series.length - 1, t)] : undefined;
      if (Number.isFinite(v)) parts.push(`${pop} ${v.toFixed(2)}`);
    }
  }
  const ids = Array.isArray(c.top_ids) ? c.top_ids[Math.min(c.top_ids.length - 1, t)] : null;
  const text = (parts.length ? `model mean |rate| ${parts.join(' · ')}` : '') + (ids && ids.length ? `${parts.length ? ' · ' : ''}top ${ids.slice(0, 5).join(',')}` : '');
  if (el && el.textContent !== text) el.textContent = text;
}

function updateCircuit(activity = [], topIds = null) {
  const color = brainPoints[0].attributes.color;
  const groups = data.connectome.groups;
  for (let i = 0; i < color.count; i++) {
    const a = Math.min(1, Math.abs(activity[i] || 0));
    const [r, g, b] = GROUP_TINT[groups[i]] || [0.6, 0.6, 0.6];
    color.setXYZ(i, 0.1 + r * a, 0.12 + g * a, 0.14 + b * a);
  }
  if (topIds) for (const i of topIds) if (i >= 0 && i < color.count) color.setXYZ(i, 1, 1, 1);
  color.needsUpdate = true;
}

function updateBrain(activity = []) {
  if (!brainPoints.length) return;
  const color = brainPoints[0].attributes.color;
  for (let i = 0; i < color.count; i++) {
    const glow = Math.max(0, Math.min(1, activity.length ? activity[i % activity.length] : 0));
    color.setXYZ(i, 0.12 + 0.88 * glow, 0.34 + 0.38 * (1 - glow), 0.42 + 0.46 * glow);
  }
  color.needsUpdate = true;
}

function isContactEvent(e) { return e.type === 'key' || e.type === 'contact_onset' || e.type === 'contact_offset'; }
function eventCharacter(e) { return e.type === 'key' ? (typeof e.char === 'string' ? e.char : '') : ''; }
function charLabel(c) { return c === ' ' ? '␣' : c === '\n' ? '⏎' : c === '' ? 'DEL' : c; }

function renderState() {
  if (!data) return;
  const tick = Math.floor(time);
  const event = cursor > 0 ? data.events[cursor - 1] : undefined;
  const reached = data.events.slice(0, cursor);
  const keyEvents = reached.filter(e => e.type === 'key');
  const contactEvents = reached.filter(isContactEvent);
  const text = keyEvents.length ? keyEvents.at(-1).text : '';
  $('#output-code').textContent = text;
  drawMonitor(text, editorView(text, keyEvents.at(-1)));
  $('#char-count').textContent = String([...text].length);
  $('#timeline').value = tick;
  $('#tick-label').textContent = `TICK ${tick}`;
  $('#tick-total').textContent = `${frames.length - 1} TICKS`;
  $('#key-count').textContent = `${keyEvents.length} KEYS`;

  const down = new Map();
  for (const e of reached) {
    if (e.type === 'contact_onset') down.set(e.foot, e.key_id);
    else if (e.type === 'contact_offset') down.delete(e.foot);
  }
  const downKeys = new Set(down.values());
  document.querySelectorAll('.keycap').forEach(k => {
    k.classList.toggle('pressed', downKeys.has(k.dataset.id));
    k.classList.toggle('latest', keyEvents.at(-1)?.key_id === k.dataset.id);
  });
  keyMeshes.forEach((mesh, id) => {
    const active = downKeys.has(id);
    mesh.userData.top.emissive.set(active ? '#2f9e66' : '#000000');
    mesh.userData.top.emissiveIntensity = active ? 0.9 : 0;
  });

  const shown = contactEvents.slice(-6).reverse();
  $('#history').innerHTML = shown.map(e => `<button class="history-row ${e === contactEvents.at(-1) ? 'selected' : ''}" data-tick="${e.tick}"><span class="hist-tick">${e.tick}</span><b>${escapeHtml(e.type === 'key' ? charLabel(eventCharacter(e)) : e.type === 'contact_onset' ? '↓' : '↑')}</b><span>${escapeHtml(e.key_id || 'event')}</span><small>${e.foot ? escapeHtml(footName(e.foot)) : ''}</small></button>`).join('') || '<div class="empty-history">No contacts recorded yet.</div>';
  document.querySelectorAll('.history-row').forEach(b => b.addEventListener('click', () => seek(Number(b.dataset.tick))));

  const row = frames[Math.min(frames.length - 1, tick)];
  if (row) {
    $('#pose').textContent = `THORAX ${n(row[1])}, ${n(row[2])}, ${n(row[3])} · YAW ${Math.round(row[4] * 180 / Math.PI)}°`;
    if (data.connectome) {
      const c = data.connectome;
      const shape = Array.isArray(c.activity_shape) ? c.activity_shape : [];
      const cells = shape.length >= 2 ? shape[1] : (c.groups ? c.groups.length : 0);
      const steps = shape.length >= 2 ? shape[0] : (cells && c.activity ? Math.floor(c.activity.length / cells) : 0);
      if (steps && cells && c.activity) {
        const t = Math.min(steps - 1, tick);
        const topIds = Array.isArray(c.top_ids) ? c.top_ids[Math.min(c.top_ids.length - 1, t)] : null;
        updateCircuit(Array.from(c.activity.subarray(t * cells, (t + 1) * cells), v => v * (c.activity_scale || 1 / 127)), topIds);
        updateBrainReadout(c, t);
      }
    }
    else updateBrain(row.slice(field['activity.0'], field['activity.0'] + 16));
  }
  $('#focus-title').textContent = event ? `${eventTitle(event)}` : 'Walking';
  $('#focus-detail').textContent = event ? `Tick ${event.tick} · ${footName(event.foot)} claw · output buffer ${[...text].length} chars` : 'Step through the replay to inspect a contact and body pose.';
  const done = tick >= frames.length - 1;
  $('#match-state').textContent = done ? (text === data.final_text && text === data.target ? 'TARGET MATCH' : 'TEXT MISMATCH') : 'IN PROGRESS';
  $('#match-state').classList.toggle('matched', done && text === data.target);
  const v = data.validation || {};
  const r = x => x?.passed === true ? 'PASS' : x?.passed === false ? 'FAIL' : '—';
  $('#validation').textContent = `Exact ${r(v.exact)} · Syntax ${r(v.syntax)} · Elaboration ${r(v.elaboration)}`;
}

function eventTitle(e) {
  if (e.type === 'key') return `Key ${e.key_id} → ${charLabel(eventCharacter(e))}`;
  if (e.type === 'contact_onset') return `Contact onset · ${e.key_id}`;
  if (e.type === 'contact_offset') return `Contact released · ${e.key_id}`;
  if (e.type === 'modifier') return `Shift latched · ${e.key_id}`;
  return e.type;
}
function footName(f) { return ({ LF: 'left foreleg', RF: 'right foreleg', LM: 'left midleg', RM: 'right midleg', LH: 'left hindleg', RH: 'right hindleg' })[f] || f || 'body'; }
function n(v) { return Number.isFinite(v) ? Number(v).toFixed(2) : '0.00'; }
function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }

const PANEL_LABELS = { control: 'ctrl', option: 'opt', command: 'cmd', 'caps lock': 'caps', delete: 'del', return: 'ret' };
function renderKeyboard(layout) {
  const rowPx = 27;
  $('#keyboard').style.height = `${layout.height * rowPx + 6}px`;
  $('#keyboard').innerHTML = layout.keys.map(k => `<div class="keycap" data-id="${escapeHtml(k.id)}" style="left:${(k.x / layout.width * 100).toFixed(3)}%;top:${(3 + k.y * rowPx).toFixed(1)}px;width:${(k.width / layout.width * 100).toFixed(3)}%;height:${(k.height * rowPx).toFixed(1)}px">${escapeHtml(PANEL_LABELS[k.label] ?? k.label)}</div>`).join('');
}

function seek(tick) {
  playing = false; $('#play').textContent = '▶ Play';
  time = Math.max(0, Math.min(frames.length - 1, Number.isFinite(tick) ? tick : 0));
  cursor = -1; syncCursor();
}

function load(replay) {
  data = replay;
  frames = replay.frames.rows;
  field = Object.fromEntries(replay.frames.fields.map((f, i) => [f, i]));
  tickHz = replay.metadata?.tick_hz || 30;
  layoutInfo = replay.layout;
  const presses = data.events.filter(e => e.type === 'key').length;
  $('#run-label').textContent = `${frames.length} ticks · ${presses} keypresses · max IK error ${(replay.metadata.max_ik_error * replay.metadata.key_pitch_mm).toFixed(2)} mm`;
  $('#mode-label').textContent = data.metadata?.mode || 'REPLAY';
  $('#target-select').value = data.metadata?.target_id || selectedTarget;
  $('#target-code').textContent = data.target || 'No target supplied';
  $('#timeline').max = String(frames.length - 1);
  const physics = String(replay.metadata.body_model || '').includes('MuJoCo');
  $('#scene-help').textContent = physics
    ? `NEUROMECHFLY v2 ×${replay.metadata.fly_scale} · ${laptopTag(replay)} · MUJOCO LEGS + SPRING KEYS · THORAX CARRIED`
    : `NEUROMECHFLY v2 BODY ×${replay.metadata.fly_scale} · ${laptopTag(replay)} · IK-POSED LEGS · NO PHYSICS`;
  renderKeyboard(layoutInfo);
  if (!scene) { initScene(); initBrain(); }
  if (replay.connectome && !replay.connectome.activity && replay.connectome.activity_int8_b64) {
    const bytes = Uint8Array.from(atob(replay.connectome.activity_int8_b64), c => c.charCodeAt(0));
    replay.connectome.activity = new Int8Array(bytes.buffer);
  }
  setBrainCircuit(replay.connectome);
  buildLaptop(layoutInfo);
  if (fly) fly.group.scale.setScalar(replay.metadata.fly_scale / replay.metadata.key_pitch_mm);
  time = 0; cursor = -1; playing = true; lastTime = 0;
  $('#play').textContent = 'Ⅱ Pause';
  applyFrame(0);
  if (cameraMode) setView(cameraMode);
  syncCursor();
}

async function fetchReplay() {
  try {
    $('#run-label').textContent = 'Generating replay…';
    const r = await fetch(`/api/replay?target_id=${encodeURIComponent(selectedTarget)}&source=${encodeURIComponent(selectedSource)}`);
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `Replay API returned ${r.status}`);
    const replay = await r.json();
    if (!flyModel) {
      flyModel = await loadFlyModel();
      fly = buildFly(flyModel);
    }
    load(replay);
    if (!fly.group.parent) simRoot.add(fly.group);
    fly.group.scale.setScalar(replay.metadata.fly_scale / replay.metadata.key_pitch_mm);
    setView(cameraMode || 'follow');
    if (params.has('tick')) { seek(Number(params.get('tick'))); setView(cameraMode); }
  } catch (e) {
    console.error(e);
    $('#run-label').textContent = 'Replay unavailable';
    $('#mode-label').textContent = 'OFFLINE';
    $('#target-code').textContent = 'Start the backend to load a recorded replay.';
    $('#focus-detail').textContent = e.message;
  }
}

$('#play').onclick = () => {
  if (!data) return;
  playing = !playing;
  if (playing && time >= frames.length - 1) { time = 0; cursor = -1; }
  lastTime = 0;
  $('#play').textContent = playing ? 'Ⅱ Pause' : '▶ Play';
};
$('#step').onclick = () => { if (data) seek(Math.floor(time) + 1); };
$('#restart').onclick = () => { if (!data) return; time = 0; cursor = -1; playing = true; lastTime = 0; $('#play').textContent = 'Ⅱ Pause'; syncCursor(); };
$('#timeline').oninput = e => seek(Number(e.target.value));
$('#reload').onclick = fetchReplay;
$('#target-select').onchange = e => { selectedTarget = e.target.value; fetchReplay(); };
$('#source-select').onchange = e => { selectedSource = e.target.value; fetchReplay(); };
const SOURCE_LABEL = { kinematic: 'Scripted · kinematic', physics: 'Scripted · MuJoCo physics', policy: 'Learned policy · physics' };
async function fetchSources() {
  try {
    const r = await fetch('/api/sources');
    if (!r.ok) throw new Error();
    const sources = await r.json();
    $('#source-select').innerHTML = sources.map(s => `<option value="${s}">${SOURCE_LABEL[s] || s}</option>`).join('');
    $('#source-select').value = sources.includes(selectedSource) ? selectedSource : sources[0];
    selectedSource = $('#source-select').value;
  } catch { /* offline */ }
}

async function fetchTargets() {
  try {
    const r = await fetch('/api/targets');
    if (!r.ok) throw new Error();
    const targets = await r.json();
    $('#target-select').innerHTML = targets.map(t => `<option value="${escapeHtml(t.id)}">${escapeHtml(t.title || t.id)}</option>`).join('');
    $('#target-select').value = selectedTarget;
  } catch { /* offline */ }
}
fetchTargets();
fetchSources().then(fetchReplay);
