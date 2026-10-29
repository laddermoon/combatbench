import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

const el = id => document.getElementById(id);
const api = (p, d) => fetch(p, {method: 'POST', headers: {'Content-Type': 'application/json'},
  body: JSON.stringify(d || {})}).then(r => r.json());

// ---------- three scene ----------
const view = el('view');
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setSize(innerWidth, innerHeight);
view.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14161a);
const cam = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.01, 100);
cam.position.set(1.8, 1.4, 2.2);
const ctl = new OrbitControls(cam, renderer.domElement);
ctl.target.set(0, 0, 0.9);
scene.add(new THREE.AmbientLight(0xffffff, 0.75));
const dl = new THREE.DirectionalLight(0xffffff, 1.1); dl.position.set(2, 3, 4); scene.add(dl);
const grid = new THREE.GridHelper(8, 40, 0x335, 0x222630); grid.rotation.x = Math.PI / 2; scene.add(grid);
scene.add(new THREE.AxesHelper(0.3));

const bodyGroup = new THREE.Group(); scene.add(bodyGroup);   // robot pose
const ptGroup = new THREE.Group(); scene.add(ptGroup);       // p5 spheres

let state = null, vis = {}, meshes = {}, spheres = {}, names = [];

const _Y2Z = new THREE.Quaternion().setFromAxisAngle(
  new THREE.Vector3(1, 0, 0), Math.PI / 2);   // three capsules sit on Y
function mkBodyMeshes() {
  for (const c of bodyGroup.children.slice()) bodyGroup.remove(c);
  meshes = {};
  for (const [n, geoms] of Object.entries(vis)) {
    const g = new THREE.Group();
    for (const v of geoms) {
      const col = new THREE.Color(v.rgba[0], v.rgba[1], v.rgba[2]);
      const mat = new THREE.MeshStandardMaterial(
        {color: col, roughness: 0.7, transparent: v.rgba[3] < 1,
         opacity: v.rgba[3]});
      let geo = null;
      const s = v.size;
      if (v.type === 'sphere') geo = new THREE.SphereGeometry(s[0], 20, 14);
      else if (v.type === 'capsule')
        geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 4, 12);
      else if (v.type === 'cylinder')
        geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 14);
      else if (v.type === 'box')
        geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
      else if (v.type === 'ellipsoid') {
        geo = new THREE.SphereGeometry(1, 16, 12);
        geo.scale(s[0], s[1], s[2]);
      }
      if (!geo) continue;
      const m = new THREE.Mesh(geo, mat);
      m.position.set(...v.pos);
      const q = new THREE.Quaternion(v.quat[1], v.quat[2], v.quat[3], v.quat[0]);
      if (v.type === 'capsule' || v.type === 'cylinder')
        q.multiply(_Y2Z);
      m.quaternion.copy(q);
      g.add(m);
    }
    bodyGroup.add(g); meshes[n] = g;
  }
}

function setPose(bodies) {
  for (const [n, b] of Object.entries(bodies)) {
    const g = meshes[n]; if (!g) continue;
    g.position.set(...b.p);
    g.quaternion.set(b.q[0], b.q[1], b.q[2], b.q[3]);
  }
}

function mkPoints() {
  spheres = {};
  for (const n of names) {
    const s = new THREE.Mesh(new THREE.SphereGeometry(0.045, 20, 14),
      new THREE.MeshStandardMaterial({
        color: new THREE.Color(state.point_colors[n]), roughness: 0.4,
        transparent: true, opacity: 0.95}));
    s.userData.point = n;
    ptGroup.add(s); spheres[n] = s;
  }
}

function setPoints(p5) { names.forEach((n, i) => spheres[n].position.set(...p5[i])); }

// ---------- status ----------
function showRes(r, extra = '') {
  const pe = r.point_errors.map(e => e.toFixed(3)).join(' ');
  el('status').textContent =
    `${r.satisfied ? 'SAT ' : 'FAIL ' + (r.failed_stage || '')}${extra}\n` +
    `errs [${names.join(',')}]\n[${pe}]  E=${r.energy.toFixed(2)}`;
}

// ---------- drag ----------
const ray = new THREE.Raycaster(), ptr = new THREE.Vector2();
let drag = null, dragPlane = new THREE.Plane(), lastMove = 0;
function p2v(e) {
  const r = renderer.domElement.getBoundingClientRect();
  ptr.set(((e.clientX - r.left) / r.width) * 2 - 1,
          -((e.clientY - r.top) / r.height) * 2 + 1);
}
renderer.domElement.addEventListener('pointerdown', e => {
  p2v(e); ray.setFromCamera(ptr, cam);
  const hit = ray.intersectObjects(Object.values(spheres))[0];
  if (!hit) return;
  const n = hit.object.userData.point;
  drag = {name: n, obj: hit.object};
  ctl.enabled = false;
  const nrm = cam.getWorldDirection(new THREE.Vector3());
  dragPlane.setFromNormalAndCoplanarPoint(nrm, hit.object.position);
  renderer.domElement.setPointerCapture(e.pointerId);
});
renderer.domElement.addEventListener('pointermove', e => {
  if (!drag) return;
  p2v(e); ray.setFromCamera(ptr, cam);
  const p = new THREE.Vector3();
  if (!ray.ray.intersectPlane(dragPlane, p)) return;
  drag.obj.position.copy(p);
  const now = performance.now();
  if (now - lastMove > 50) {
    lastMove = now;
    api('/api/drag', {point: drag.name, pos: p.toArray(), phase: 'move'})
      .then(r => {
        drag.obj.material.color.set(r.ok === false ? 0xff2222 : state.point_colors[drag.name]);
        if (r.res) { setPose(r.res.bodies); showRes(r.res, r.ms ? ` ${r.ms}ms` : ''); }
      });
  }
});
addEventListener('pointerup', () => {
  if (!drag) return;
  const d = drag;
  api('/api/drag', {point: d.name, pos: d.obj.position.toArray(), phase: 'end'})
    .then(r => {
      if (r.res) { setPose(r.res.bodies); showRes(r.res, ` end ${r.ms ?? '?'}ms`); }
      if (r.ok === false && r.res && r.res.bodies)
        api('/api/init').then(s => { if (s.res) setPoints(s.p5); });
      else api('/api/clip/commit');
    });
  drag = null; ctl.enabled = true;
});

// ---------- clip ----------
function refreshClip() {
  const L = el('kflist'); L.innerHTML = '';
  (state.clip || []).forEach((k, i) => {
    const d = document.createElement('div');
    d.className = 'kf' + (i === state.sel ? ' sel' : '');
    const dur = k.transfer ? k.transfer.dur : '';
    const tr = k.transfer ? ` dx${k.transfer.dx} dy${k.transfer.dy} dyaw${k.transfer.dyaw}` : '';
    d.innerHTML = `#${i} t=<input value="${dur}" style="width:45px"><span class="dim">${tr} del</span>`;
    d.querySelector('input').onchange = ev =>
      api('/api/clip/time', {i, t: parseFloat(ev.target.value)});
    d.onclick = ev => { if (ev.target.tagName !== 'INPUT') api('/api/clip/sel', {i}).then(applySel); };
    d.querySelector('.dim').onclick = ev => { ev.stopPropagation(); api('/api/clip/del', {i}).then(r => { state.clip = r.clip; state.sel = r.sel; refreshClip(); }); };
    L.appendChild(d);
  });
}
function applySel(r) {
  state.sel = r.sel;
  if (r.res) { setPose(r.res.bodies); showRes(r.res); }
  api('/api/init').then(s => { setPoints(s.p5); });
  refreshClip();
}
el('badd').onclick = () => api('/api/clip/add').then(r => { state.clip = r.clip; state.sel = r.sel; refreshClip(); });
el('bcom').onclick = () => api('/api/clip/commit').then(r => { state.clip = r.clip; refreshClip(); });

el('bsave').onclick = () => { const n = el('fname').value || 'clip.json'; api('/api/save', {name: n}).then(refreshFiles); };

function refreshFiles() {
  fetch('/api/files').then(r => r.json()).then(fs => {
    const L = el('flist'); L.innerHTML = '';
    fs.forEach(f => {
      const d = document.createElement('div');
      d.className = 'file'; d.textContent = f.name;
      d.onclick = () => api('/api/load', {name: f.name}).then(r => location.reload());
      L.appendChild(d);
    });
  });
}

// ---------- boot ----------
api_boot();
async function api_boot() {
  const s = await fetch('/api/init').then(r => r.json());
  state = s; vis = s.vis; names = s.point_names;
  el('wdir').textContent = s.workdir;
  mkBodyMeshes(); mkPoints(); setPoints(s.p5);
  if (s.res) { setPose(s.res.bodies); showRes(s.res); }

  refreshClip(); refreshFiles();
  (function loop() { requestAnimationFrame(loop); ctl.update(); renderer.render(scene, cam); })();
}
addEventListener('resize', () => {
  cam.aspect = innerWidth / innerHeight; cam.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
