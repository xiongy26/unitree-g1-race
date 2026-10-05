// three.js 场景: 程序化赛道 + 基于 mjv 渲染管线的机器人可视化。
// 世界坐标系与 MuJoCo 一致: z 轴向上。
//
// 机器人渲染走 mjv_updateScene 管线(与官方 demo / g1-boxing-wasm 相同):
// 每个 geom 的最终世界位姿由 MuJoCo 计算好(pos + 3x3 mat), 自动包含
// mesh 顶点坐标系对齐, 避免手工组合 body×geom×mesh 变换导致的"散开"。
// 注意: 本 @mujoco/mujoco WASM 构建中 mjvGeom.dataid 对 mesh geom 会返回
// 损坏值(真实值的 2 倍), 需要用 geom_dataid[objid] 还原真实 mesh id。

import * as THREE from 'three';
import { FENCE_INNER_Y, FENCE_CENTER_X, FENCE_HALF_LEN, FENCE_HALF_T, makeRng } from './sim.js';

export const FINISH_X = 25;      // 终点线
export const LANE_WIDTH = 1.35;  // 道宽
export const TRACK_W = 11;       // 赛道总宽(含缓冲)
export const TRACK_X0 = -4;      // 赛道纹理覆盖范围
export const TRACK_X1 = 38;

export const TEAM_COLORS = ['#ff4d4d', '#ffb400', '#37c871', '#3a9bff', '#c85bff', '#ff6fb0'];

export const MAX_LANES = 6;
export function laneY(i) { return (i - (MAX_LANES - 1) / 2) * LANE_WIDTH; }

// ---------- 体育场竞赛跑道：尺寸与物理车道保持一致 ----------
export function buildTrack(scene) {
  const cv = document.createElement('canvas');
  cv.width = 4096; cv.height = 1072;
  const g = cv.getContext('2d'), sx = cv.width / 42, sy = cv.height / TRACK_W;
  const X = x => (x - TRACK_X0) * sx, Y = y => (TRACK_W / 2 - y) * sy;
  g.fillStyle = '#567eaa'; g.fillRect(0, 0, cv.width, cv.height);
  const rng = makeRng(9021);
  // Fine rubber granules, rather than a perfectly flat color.
  for (let i = 0; i < 180000; i++) {
    g.fillStyle = rng() > .5 ? 'rgba(255,255,255,.045)' : 'rgba(12,32,65,.055)';
    g.fillRect(rng()*cv.width, rng()*cv.height, 1.5, 1.5);
  }
  const half = MAX_LANES * LANE_WIDTH / 2;
  g.fillStyle = '#f4f0df';
  for (let i = 0; i <= MAX_LANES; i++) g.fillRect(0, Y(-half+i*LANE_WIDTH)-2, cv.width, 4);
  for (const x of [0, FINISH_X]) g.fillRect(X(x)-3, Y(half), 6, 2*half*sy);
  // Finish judging lines and restrained orange timing marks.
  g.fillRect(X(FINISH_X+.18)-1, Y(half), 2, 2*half*sy);
  g.fillStyle = '#dc9d70';
  for (const x of [5, 15, 30]) for (let i=0;i<MAX_LANES;i++)
    g.fillRect(X(x)-2, Y(laneY(i))-.18*sy, 4, .36*sy);
  for (let i=0;i<MAX_LANES;i++) {
    g.save(); g.translate(X(-.85),Y(laneY(i))); g.rotate(-Math.PI/2);
    g.fillStyle='#f4f0df'; g.font=`bold ${.78*sy}px Arial`; g.textAlign='center';
    g.textBaseline='middle'; g.fillText(String(i+1),0,0); g.restore();
  }
  const tex = new THREE.CanvasTexture(cv); tex.colorSpace=THREE.SRGBColorSpace; tex.anisotropy=8;
  const track = new THREE.Mesh(new THREE.PlaneGeometry(42,TRACK_W), new THREE.MeshStandardMaterial({map:tex,roughness:.91}));
  track.position.set(17,0,.005); track.receiveShadow=true; scene.add(track);
  box(scene, [180,140,.12], [14,0,-.08], 0x62885b);
  box(scene, [46,9,.015], [17,10,-.005], 0x4c968d);
  // Low white advertising walls coincide with the existing collision walls.
  for (const side of [-1,1]) {
    box(scene,[FENCE_HALF_LEN*2,FENCE_HALF_T*2,.6],[FENCE_CENTER_X,side*(FENCE_INNER_Y+FENCE_HALF_T),.3],0xe5e7e3);
    for (let x=-.5;x<31;x+=3.15) {
      const board = textPanel('ROBOT GAMES   /   2026',3.08,.46, x%2>0?'#b64d52':'#244f70','#f4f4ef');
      board.rotation.x=Math.PI/2; board.position.set(x,side*(FENCE_INNER_Y-.006),.32);
      if(side<0) board.rotation.x=-Math.PI/2;
      scene.add(board);
    }
  }
  for(let i=0;i<MAX_LANES;i++) {
    const marker=textPanel(String(i+1),.38,.5,'#242b32','#f4f2e8');
    marker.rotation.y=-Math.PI/2; marker.position.set(-1.9,laneY(i),.28); scene.add(marker);
    box(scene,[.4,.44,.08],[-1.9,laneY(i),.04],0xdadbd5);
  }
}

// ---------- mjv 共享状态 ----------
let sharedOption = null, sharedPerturb = null, sharedCamera = null;
const meshGeoCache = new Map(); // meshId -> BufferGeometry(全机器人共享)
const primGeoCache = new Map(); // 图元 key -> BufferGeometry

function ensureShared(mj) {
  if (sharedOption) return;
  sharedOption = new mj.MjvOption();
  for (let i = 0; i < 6; i++) {
    sharedOption.geomgroup[i] = 0;
    sharedOption.sitegroup[i] = 0;
  }
  sharedOption.geomgroup[1] = 1; // 本模型: group 1 = 可视网格; group 0 = 碰撞网格(隐藏)
  sharedPerturb = new mj.MjvPerturb();
  sharedCamera = new mj.MjvCamera();
}

function getMeshGeometry(model, meshId, cacheKey = '') {
  const key = cacheKey + meshId;
  let g = meshGeoCache.get(key);
  if (g) return g;
  const va = model.mesh_vertadr[meshId], vn = model.mesh_vertnum[meshId];
  const fa = model.mesh_faceadr[meshId], fn = model.mesh_facenum[meshId];
  const positions = new Float32Array(vn * 3);
  for (let i = 0; i < vn * 3; i++) positions[i] = model.mesh_vert[3 * va + i];
  const indices = new Uint32Array(fn * 3);
  for (let i = 0; i < fn * 3; i++) indices[i] = model.mesh_face[3 * fa + i];
  g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setIndex(new THREE.BufferAttribute(indices, 1));
  g.computeVertexNormals();
  meshGeoCache.set(key, g);
  return g;
}

function getPrimGeometry(mj, type, size) {
  const key = type + ':' + size.join(',');
  let g = primGeoCache.get(key);
  if (g) return g;
  const G = mj.mjtGeom;
  if (type === G.mjGEOM_SPHERE.value) g = new THREE.SphereGeometry(size[0], 24, 16);
  else if (type === G.mjGEOM_ELLIPSOID.value) {
    g = new THREE.SphereGeometry(1, 24, 16);
    g.scale(size[0], size[1], size[2]);
  } else if (type === G.mjGEOM_CAPSULE.value) {
    g = new THREE.CapsuleGeometry(size[0], 2 * size[1], 8, 16);
    g.rotateX(Math.PI / 2); // mujoco capsule 沿 z
  } else if (type === G.mjGEOM_CYLINDER.value) {
    g = new THREE.CylinderGeometry(size[0], size[0], 2 * size[1], 28);
    g.rotateX(Math.PI / 2);
  } else if (type === G.mjGEOM_BOX.value) {
    g = new THREE.BoxGeometry(2 * size[0], 2 * size[1], 2 * size[2]);
  } else if (type === G.mjGEOM_PLANE.value) {
    g = new THREE.PlaneGeometry(size[0] ? 2 * size[0] : 12, size[1] ? 2 * size[1] : 12);
  } else {
    g = new THREE.SphereGeometry(0.02, 8, 6);
  }
  primGeoCache.set(key, g);
  return g;
}

// ---------- 机器人 ----------
export class RobotVisual {
  constructor(mj, model, teamColor, label, labelH = 1.25, cacheKey = '', visGroups = [1]) {
    this.mj = mj;
    this.model = model;
    ensureShared(mj);
    this.mjvScene = new mj.MjvScene(model, 20000);
    // 每物种可见 geom 组(官方 MJCF 的视觉网格组各不相同: G1/T1=1, PM01=2)
    this.option = new mj.MjvOption();
    for (let i = 0; i < 6; i++) this.option.geomgroup[i] = 0;
    for (let i = 0; i < 6; i++) this.option.sitegroup[i] = 0;
    for (const g of visGroups) this.option.geomgroup[g] = 1;
    this.group = new THREE.Group();
    this.meshes = []; // mjv geom 槽位 -> three mesh
    this.tint = new THREE.Color(teamColor);
    this.labelH = labelH;
    this.cacheKey = cacheKey;

    // 头顶编号牌
    const cv = document.createElement('canvas');
    cv.width = 128; cv.height = 128;
    const c = cv.getContext('2d');
    c.beginPath(); c.arc(64, 64, 56, 0, Math.PI * 2);
    c.fillStyle = teamColor; c.fill();
    c.lineWidth = 6; c.strokeStyle = '#fff'; c.stroke();
    c.fillStyle = '#fff'; c.font = 'bold 72px sans-serif';
    c.textAlign = 'center'; c.textBaseline = 'middle';
    c.fillText(String(label), 64, 70);
    const tex = new THREE.CanvasTexture(cv);
    this.sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: true }));
    this.sprite.scale.set(0.42, 0.42, 1);
    this.group.add(this.sprite);
  }

  // 由 MjData 更新位姿
  update(data) {
    const mj = this.mj;
    const model = this.model;
    mj.mjv_updateScene(model, data, this.option, sharedPerturb, sharedCamera,
      mj.mjtCatBit.mjCAT_ALL.value, this.mjvScene);
    const geoms = this.mjvScene.geoms;
    const n = geoms.size();
    try {
      for (let i = 0; i < n; i++) {
        const gm = geoms.get(i);
        const type = Number(gm.type);
        const objtype = Number(gm.objtype), objid = Number(gm.objid);
        if (objtype !== mj.mjtObj.mjOBJ_GEOM.value) { gm.delete(); continue; }
        const size = Array.from(gm.size).map(Number);
        const rgba = Array.from(gm.rgba);
        const mat9 = Array.from(gm.mat);
        const pos = Array.from(gm.pos);
        gm.delete();

        // 修复该 WASM 构建的 dataid 损坏问题: 用 objid 查真实 mesh id
        let meshId = -1;
        if (type === mj.mjtGeom.mjGEOM_MESH.value) {
          meshId = (objid >= 0 && objid < model.ngeom) ? Number(model.geom_dataid[objid]) : Number.NaN;
          if (!Number.isFinite(meshId)) continue;
        }
        // The shared floor and race barriers are rendered once by buildTrack.
        const geomName = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM.value, objid);
        if (type === mj.mjtGeom.mjGEOM_PLANE.value || geomName?.startsWith('race_fence')) continue;
        const isMesh = type === mj.mjtGeom.mjGEOM_MESH.value;
        const geo = isMesh
          ? getMeshGeometry(model, meshId, this.cacheKey)
          : getPrimGeometry(mj, type, size);
        const key = isMesh ? `m${this.cacheKey}${meshId}` : `p${type}:${size.join(',')}`;

        let mesh = this.meshes[i];
        if (!mesh || mesh.userData.key !== key) {
          if (mesh) this.group.remove(mesh);
          mesh = new THREE.Mesh(geo, new THREE.MeshPhongMaterial({ shininess: 24, specular: 0x222222 }));
          mesh.castShadow = true;
          mesh.userData.key = key;
          this.meshes[i] = mesh;
          this.group.add(mesh);
        }

        // 白色/浅色部件染队伍色, 深色部件保持原色
        const lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2];
        if (lum > 0.25) {
          mesh.material.color.setRGB(
            rgba[0] * 0.55 + this.tint.r * 0.45,
            rgba[1] * 0.55 + this.tint.g * 0.45,
            rgba[2] * 0.55 + this.tint.b * 0.45);
        } else {
          mesh.material.color.setRGB(rgba[0], rgba[1], rgba[2]);
        }

        mesh.matrixAutoUpdate = false;
        mesh.matrix.set(
          mat9[0], mat9[1], mat9[2], pos[0],
          mat9[3], mat9[4], mat9[5], pos[1],
          mat9[6], mat9[7], mat9[8], pos[2],
          0, 0, 0, 1);
        mesh.matrixWorldNeedsUpdate = true;
        mesh.visible = true;
      }
    } finally {
      geoms.delete();
    }
    for (let i = n; i < this.meshes.length; i++) {
      if (this.meshes[i]) this.meshes[i].visible = false;
    }
    // 编号牌跟随 pelvis(body 1), 高度随物种
    const xpos = data.xpos;
    this.sprite.position.set(xpos[3], xpos[4], xpos[5] + this.labelH);
  }

  dispose() {
    try { this.mjvScene.delete(); } catch (e) { /* 已释放 */ }
    this.group.traverse((o) => { if (o.material) o.material.dispose(); });
  }
}

// ---------- Outdoor athletics stadium ----------
function box(scene, size, pos, color, roughness=.75) {
  const mesh=new THREE.Mesh(new THREE.BoxGeometry(...size),new THREE.MeshStandardMaterial({color,roughness}));
  mesh.position.set(...pos); mesh.castShadow=true; mesh.receiveShadow=true; scene.add(mesh); return mesh;
}
function textPanel(text,w,h,ink='#edf4fb',background='#19364d') {
  const cv=document.createElement('canvas'); cv.width=1024; cv.height=256;
  const c=cv.getContext('2d'); c.fillStyle=background; c.fillRect(0,0,1024,256);
  c.fillStyle=ink; c.font='bold 64px Arial'; c.textAlign='center'; c.textBaseline='middle';
  c.fillText(text,512,128,970);
  const map=new THREE.CanvasTexture(cv); map.colorSpace=THREE.SRGBColorSpace;
  return new THREE.Mesh(new THREE.PlaneGeometry(w,h),new THREE.MeshStandardMaterial({map,roughness:.8,side:THREE.DoubleSide}));
}
export function addLights(scene) {
  scene.add(new THREE.HemisphereLight(0xe7f0ff,0x687c52,1.8));
  const key=new THREE.DirectionalLight(0xfff5e7,3.1);
  key.position.set(-10,-8,20); key.castShadow=true; key.shadow.mapSize.set(2048,2048);
  Object.assign(key.shadow.camera,{near:1,far:65,left:-22,right:22,top:15,bottom:-15});
  key.shadow.bias=-.0003; key.shadow.normalBias=.025;
  scene.add(key,key.target);
  const fill=new THREE.DirectionalLight(0xdceaff,1.0); fill.position.set(20,12,14); scene.add(fill);
  return key;
}
export function updateScenery() {} // Stadium props are static.

function person(scene,x,y,rng,seated=false) {
  const group=new THREE.Group(); group.position.set(x,y,0); scene.add(group);
  const skin=[0xb58b70,0xd4af91,0x94715b][Math.floor(rng()*3)];
  const shirt=[0x25323d,0x5c6973,0x303c50,0x908a81,0x455854][Math.floor(rng()*5)];
  const z=seated?.9:1.2;
  const torso=new THREE.Mesh(new THREE.CapsuleGeometry(.16,.27,5,10),new THREE.MeshStandardMaterial({color:shirt,roughness:.95}));
  torso.rotation.x=Math.PI/2; torso.scale.z=.72; torso.position.z=z; torso.castShadow=true; group.add(torso);
  const head=new THREE.Mesh(new THREE.SphereGeometry(.12,12,10),new THREE.MeshStandardMaterial({color:skin,roughness:.9}));
  head.position.set(0,0,z+.4); head.scale.z=1.16; head.castShadow=true; group.add(head);
  for(const side of [-1,1]) {
    box(group,[.115,.14,seated?.38:.9],[side*.1,0,seated?.48:.48],0x252c32);
    const arm=box(group,[.09,.12,.44],[side*.23,-.015,z-.02],shirt); arm.rotation.y=side*.12;
    box(group,[.12,.23,.09],[side*.1,-.05,seated?.25:.045],0x1b2229);
  }
  group.rotation.z=(rng()-.5)*.5;
}
function cameraRig(scene,x,y) {
  for(const dx of [-.22,.22]) {
    const leg=box(scene,[.035,.04,1.16],[x+dx,y,.58],0x282e34); leg.rotation.y=dx>0?.22:-.22;
  }
  box(scene,[.04,.035,1.2],[x,y+.2,.58],0x282e34);
  box(scene,[.32,.2,.2],[x,y,1.23],0x202b33);
  box(scene,[.12,.13,.12],[x,y-.16,1.23],0x121c24);
}
export function buildEnvironment(scene) {
  const rng=makeRng(0xbeef01);
  // Open-air stadium: grass infield, tiered stands and floodlight towers.
  box(scene,[76,54,.12],[14,0,-.17],0x54815b);
  box(scene,[46,9,.015],[17,10,-.005],0x4c968d);
  const skyGeo=new THREE.SphereGeometry(220,32,20);
  const positions=skyGeo.attributes.position, colors=[];
  const horizon=new THREE.Color(0xe0ebee), zenith=new THREE.Color(0x6eabe0), tint=new THREE.Color();
  for(let i=0;i<positions.count;i++) {
    tint.copy(horizon).lerp(zenith,Math.max(0,positions.getZ(i)/170));
    colors.push(tint.r,tint.g,tint.b);
  }
  skyGeo.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));
  const sky=new THREE.Mesh(skyGeo,new THREE.MeshBasicMaterial({vertexColors:true,side:THREE.BackSide,fog:false}));
  scene.add(sky);
  const seats=new THREE.InstancedMesh(new THREE.BoxGeometry(.44,.43,.12),new THREE.MeshStandardMaterial({roughness:.85}),1584);
  const backs=new THREE.InstancedMesh(new THREE.BoxGeometry(.44,.08,.36),seats.material,1584);
  const seatMatrix=new THREE.Matrix4(); let seatCount=0;
  seats.castShadow=backs.castShadow=true; seats.receiveShadow=backs.receiveShadow=true;
  for(const side of [-1,1]) {
    for(let row=0;row<9;row++) {
      const y=side*(17+row*.95), height=.65+row*.48;
      box(scene,[70,.95,height],[14,y,height/2],0xa2aaa9);
      // Rows of stadium seats, separated by regular stair aisles.
      for(let n=0;n<88;n++) {
        if(n%22<2) continue;
        const x=-20+n*.78;
        const color=(Math.floor(n/22)+row)%3===0?0xc4d4df:0x416d99;
        seatMatrix.makeTranslation(x,y,height+.09); seats.setMatrixAt(seatCount,seatMatrix);
        seatMatrix.makeTranslation(x,y+side*.18,height+.28); backs.setMatrixAt(seatCount,seatMatrix);
        seats.setColorAt(seatCount,new THREE.Color(color)); backs.setColorAt(seatCount,new THREE.Color(color)); seatCount++;
      }
    }
    box(scene,[71,.1,.12],[14,side*16.4,.95],0x8b989f);
  }
  seats.count=backs.count=seatCount; scene.add(seats,backs);
  for(const x of [-16,44]) for(const y of [-24,24]) {
    box(scene,[.28,.28,21],[x,y,10.5],0xa9b4b9);
    box(scene,[3.6,.4,1.5],[x,y,20.8],0x626f77);
    for(let n=0;n<6;n++) {
      const lamp=box(scene,[.46,.16,.5],[x-1.4+n*.56,y-.25,20.8],0xe7eff0);
      lamp.material.emissive.set(0xd8e1e4); lamp.material.emissiveIntensity=.4;
    }
  }
  for(let x=-17;x<50;x+=6) {
    const banner=textPanel('ROBOT ATHLETICS',4.2,.8); banner.rotation.x=Math.PI/2;
    banner.position.set(x,16.4,.55); scene.add(banner);
  }
  // Technical benches: monitors, flight cases, chairs and timing equipment.
  for(let x=1;x<31;x+=3.8) {
    box(scene,[2.9,.85,.09],[x,6.7,.77],0xd4d7d5);
    for(const dx of [-1.25,1.25]) box(scene,[.055,.65,.73],[x+dx,6.7,.36],0x58636c);
    box(scene,[.48,.08,.32],[x,6.9,1.03],0x202b36);
    const screen=textPanel('LIVE  /  TIMING',.43,.25,'#88c4db','#152b3b');
    screen.rotation.x=Math.PI/2; screen.position.set(x,6.85,1.03); scene.add(screen);
    box(scene,[.65,.44,.56],[x+1,7.8,.28],0x35434e);
    box(scene,[.69,.46,.045],[x+1,7.8,.56],0xadb8bd);
    person(scene,x-.6,7.5,rng);
  }
  for(const x of [-2,5,14,24,30]) {
    cameraRig(scene,x,-5.8); person(scene,x+.55,-6.2,rng);
  }
  // Crowd behind the press rail, with individual seats and aisle gaps.
  for(const side of [-1,1]) {
    for(let row=0;row<3;row++) {
      const y=side*(10.5+row*1.1), z=row*.36;
      box(scene,[37,1.1,.36+z],[15,y,(.36+z)/2],0x7c858c);
      for(let n=0;n<48;n++) {
        if(n%16===0) continue;
        const x=-3+n*.76;
        box(scene,[.42,.4,.08],[x,y,.43+z],0x344f65);
        if(rng()>.18) {
          const holder=new THREE.Group(); holder.position.z=z; scene.add(holder);
          person(holder,x,y,rng,true);
        }
      }
    }
    box(scene,[38,.055,.055],[15,side*9.5,1.02],0xa5b0b6);
    for(let x=-4;x<35;x+=2) box(scene,[.045,.045,1],[x,side*9.5,.5],0x939fa7);
  }
  // Start official and orange timing pedestal, as in a meet broadcast.
  box(scene,[.44,.44,1.15],[-.3,5.3,.575],0xdf7249);
  const clock=textPanel('READY',.4,.26,'#e1f6fb','#152935');
  clock.rotation.x=Math.PI/2; clock.position.set(-.3,5.065,.88); scene.add(clock);
  person(scene,.4,5.5,rng);
  const event=textPanel('ROBOT GAMES 2026   /   25m SPRINT',12,1.35);
  event.rotation.x=Math.PI/2; event.position.set(19,16.4,1.8); scene.add(event);
}
