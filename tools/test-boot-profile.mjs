// 启动阶段性能剖析: 用带时间戳的日志量化每段耗时(与浏览器同速的 WASM 编译)。
// 浏览器里另有 ~160MB 资产 + 32MB WASM 的 localhost 传输(数秒级), 此处只测计算侧。
// 运行: node tools/test-boot-profile.mjs
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url))); // 脚本在 tools/ 下, ROOT 指向仓库根
const imp = (p) => import(pathToFileURL(path.join(ROOT, p)).href);

const t0 = performance.now();
const stamp = (label) => console.log(`+${((performance.now() - t0) / 1000).toFixed(1)}s  ${label}`);

const ortSrc = fs.readFileSync(path.join(ROOT, 'vendor/ort/ort.min.js'), 'utf8');
const require_ = createRequire(import.meta.url);
const ort = new Function('require', ortSrc + '\n;return ort;')(require_);
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = pathToFileURL(path.join(ROOT, 'vendor/ort')).href + '/';

stamp('开始加载 MuJoCo WASM(10MB 编译)...');
const loadMujoco = (await imp('vendor/mujoco/mujoco.js')).default;
const mj = await loadMujoco();
stamp('MuJoCo WASM 就绪');

const { SPECIES } = await imp('src/robots.js');
const { Sim } = await imp('src/sim.js');
const { PolicySession } = await imp('src/policy.js');

// 与页面 fetchSpeciesAssets 相同, 但从磁盘直读(模拟 fetch 完成后的状态)
async function readSpeciesAssets(spec) {
  const extraFiles = new Map();
  for (const f of spec.extraFiles ?? []) {
    extraFiles.set(f, fs.readFileSync(path.join(ROOT, 'assets', spec.id, f), 'utf8'));
  }
  const xml = fs.readFileSync(path.join(ROOT, spec.xmlFile.replace(/^\.\//, '')), 'utf8');
  const allXml = [xml, ...extraFiles.values()].join('\n');
  const meshFiles = [...new Set([...allXml.matchAll(/file="([^"]+\.(?:STL|stl|obj))"/g)].map((m) => m[1].split('/').pop()))];
  const meshes = new Map();
  for (const name of meshFiles) {
    meshes.set(name, new Uint8Array(fs.readFileSync(path.join(ROOT, 'assets', spec.id, 'meshes', name))));
  }
  return { xml, extraFiles, meshes };
}

const g1Xml = fs.readFileSync(path.join(ROOT, 'assets/g1_29dof.xml'), 'utf8');
const g1Meshes = new Map();
for (const name of [...new Set([...g1Xml.matchAll(/file="([^"]+\.STL)"/g)].map((m) => m[1]))]) {
  g1Meshes.set(name, new Uint8Array(fs.readFileSync(path.join(ROOT, 'assets/meshes', name))));
}
const assets = { g1: { xml: g1Xml, meshes: g1Meshes }, species: new Map() };
for (const sp of SPECIES) {
  if (sp.id === 'g1') continue;
  const s = performance.now();
  assets.species.set(sp.id, await readSpeciesAssets(sp));
  stamp(`${sp.name} 资产读取(浏览器里为 fetch): ${((performance.now() - s) / 1000).toFixed(2)}s, ${sp.id === 'pm01' ? '' : ''}${[...assets.species.get(sp.id).meshes.values()].reduce((n, b) => n + b.length, 0) / 1048576 | 0}MB 网格`);
}

stamp('=== 开始模型编译(Sim.load, 全部同步阻塞) ===');
await Sim.load(mj, assets, (msg) => stamp(`  ${msg}`));

stamp('=== 开始策略会话创建(ONNX) ===');
for (const sp of SPECIES) {
  const file = path.join(ROOT, sp.policyFile.replace(/^\.\//, ''));
  if (!fs.existsSync(file)) { stamp(`${sp.name}: 策略缺失, 跳过`); continue; }
  const s = performance.now();
  const buf = new Uint8Array(fs.readFileSync(file));
  const session = new PolicySession(ort, sp.contract);
  await session.load(buf);
  stamp(`${sp.name} 策略会话: ${((performance.now() - s) / 1000).toFixed(2)}s`);
}
stamp('=== 剖析结束(浏览器总启动 = 以上计算 + ~190MB localhost 传输) ===');
