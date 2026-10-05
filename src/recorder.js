// 比赛录制: 把 WebGL 仿真画面与页面面板(标题/状态栏/排名/控制条/倒计时/结算)
// 合成进同一路视频, 停止后自动下载 —— Chrome/Edge/Safari 直接出 H.264 MP4,
// 其余浏览器回退 WebM(文件后缀跟随实际容器)。
//
// 原理:
//   1. 离屏合成画布: 每帧先画 WebGL 画布。onFrame 必须在 renderer.render 之后
//      的同一任务里调用, 此时 WebGL 绘图缓冲仍有效, 无需 preserveDrawingBuffer;
//      画布按物理像素(×devicePixelRatio, 宽度上限 2560)创建 —— 按 CSS 像素录制
//      会把高分屏画面先降采样再编码, 缩放开 125%/150% 时细节不可逆丢失;
//   2. 面板层: 把页面面板克隆进 SVG foreignObject(内嵌页面 <style> + 根节点补上
//      CSS 变量与 body 字体), 用 <img> 光栅化到透明画布后叠加在 3D 画面上。
//      面板按 CSS 像素布局, 整体 scale 放大到物理像素输出, 文字按设备分辨率光栅化。
//      MutationObserver 标脏 + 50ms 节流, 面板静止时零开销, 不拖累 RTF;
//   3. 合成画布 captureStream(60) -> MediaRecorder 编码, 停止时 Blob 自动下载。
//      码率按物理像素数给足(1080p60 ≈ 16Mbps, 上限 24Mbps), 避免 60fps 下每帧
//      码率减半导致的压缩糊化。
// 控制条「面板」勾选框(动态读取 #rec-panels)决定是否合成面板层, 录制中可实时切换;
// 带 data-norec 属性的元素(录制徽标/提示条/勾选框本身)不会进入视频。

const PANEL_SELECTOR = '#hdr, #statusbar, #board, #ctrl, #countdown, #hint, #results';
const MAX_DPR = 2;             // 录制像素比上限, 与渲染器 setPixelRatio 一致(不超过 3D 画布实际分辨率)
const MAX_PHYS_WIDTH = 2560;   // 合成画布物理像素宽度上限: 2K/4K 屏压到 1440p 级别实时编码
const OVERLAY_MIN_MS = 50;     // 面板层重绘最小间隔, 约 20Hz(排名板本身 20Hz 刷新)
const MIN_BPS = 10e6;          // 码率下限
const MAX_BPS = 24e6;          // 码率上限(1440p60 级别)
const BITS_PER_PX = 8;         // 每物理像素的比特预算(60fps 下 H.264 不发糊的经验值)
const CSS_VARS = ['--bg', '--panel', '--line', '--line-bright', '--text', '--dim', '--amber', '--red', '--green'];

// MP4 优先逐个探测, 全部不支持才回退 WebM
const MIME_CANDIDATES = [
  ['video/mp4;codecs=avc1.640028', 'mp4'],
  ['video/mp4;codecs=avc1.4D0028', 'mp4'],
  ['video/mp4;codecs=avc1.42E01E', 'mp4'],
  ['video/mp4;codecs=avc1', 'mp4'],
  ['video/mp4', 'mp4'],
  ['video/webm;codecs=vp9', 'webm'],
  ['video/webm;codecs=vp8', 'webm'],
  ['video/webm', 'webm'],
];

export class RaceRecorder {
  constructor({ getSceneCanvas }) {
    this.getSceneCanvas = getSceneCanvas;
    this.recording = false;
    this.includePanels = true; // 是否把面板层合成进视频, 勾选框与 main.js 实时改写
    this.lastInfo = null;

    this.comp = document.createElement('canvas');
    this.compCtx = this.comp.getContext('2d');
    this.overlay = document.createElement('canvas');   // 面板层(CSS 像素尺寸, 透明底)
    this.overlayCtx = this.overlay.getContext('2d');
    this.overlayReady = false;
    this.overlayFailed = false;

    this.dirty = false;
    this.rasterBusy = false;
    this.lastRasterT = 0;

    this._rec = null;
    this._chunks = [];
    this._startedAt = 0;
    this._timer = 0;
    this._toastT = 0;
    this._onResize = null;
    this._obs = new MutationObserver((records) => {
      for (const r of records) {
        const t = r.target;
        if (t instanceof Element && t.closest('[data-norec]')) continue;
        this.dirty = true;
        break;
      }
    });
  }

  toggle() { this.recording ? this.stop() : this.start(); }

  start() {
    if (this.recording) return;
    const glCanvas = this.getSceneCanvas();
    if (!glCanvas || !glCanvas.width) { this._toast('3D 画布尚未就绪, 稍后再试', true); return; }
    if (typeof MediaRecorder === 'undefined' || !glCanvas.captureStream) {
      this._toast('当前浏览器不支持视频录制(推荐 Chrome/Edge)', true);
      return;
    }
    const picked = MIME_CANDIDATES.find(([m]) => MediaRecorder.isTypeSupported(m));
    if (!picked) { this._toast('当前浏览器没有可用的视频编码器', true); return; }
    const [mime, ext] = picked;
    this._ext = ext;
    this.includePanels = document.getElementById('rec-panels')?.checked ?? true;

    // 按物理像素定合成画布尺寸(显示缩放 125%/150% 下 innerWidth 只是 CSS 像素)
    const cssW = window.innerWidth, cssH = window.innerHeight;
    const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
    const scale = dpr * Math.min(1, MAX_PHYS_WIDTH / (cssW * dpr));
    this.comp.width = evenize(Math.round(cssW * scale));
    this.comp.height = evenize(Math.round(cssH * scale));
    this.overlay.width = this.comp.width;
    this.overlay.height = this.comp.height;
    this.overlayReady = false;
    this.overlayFailed = false;

    let rec;
    try {
      const px = this.comp.width * this.comp.height;
      const bps = Math.round(Math.min(MAX_BPS, Math.max(MIN_BPS, px * BITS_PER_PX)));
      rec = new MediaRecorder(this.comp.captureStream(60), { mimeType: mime, videoBitsPerSecond: bps });
    } catch (e) {
      this._toast('创建编码器失败: ' + e.message, true);
      return;
    }
    this._chunks = [];
    rec.ondataavailable = (e) => { if (e.data && e.data.size) this._chunks.push(e.data); };
    rec.onerror = (e) => { this._toast('录制出错: ' + (e.error?.message || '未知错误'), true); this.stop(); };
    rec.onstop = () => this._finalize(mime, ext);
    rec.start(1000);
    this._rec = rec;

    this._startedAt = performance.now();
    this.recording = true;
    this.dirty = true;
    this._obs.observe(document.body, { subtree: true, childList: true, characterData: true, attributes: true });
    this._onResize = () => {
      // 码流分辨率中途不可变, 合成画布保持起始尺寸; 只按当前视口重光栅面板层
      this.overlay.width = this.comp.width;
      this.overlay.height = this.comp.height;
      this.overlayReady = false;
      this.dirty = true;
    };
    window.addEventListener('resize', this._onResize);
    document.getElementById('rec')?.classList.add('show');
    const btn = document.getElementById('btn-rec');
    if (btn) { btn.classList.add('rec'); btn.textContent = '■ 停止录制'; }
    this._timer = setInterval(() => this._tickBadge(), 250);
    this._tickBadge();
    window.__recState = 'recording';
    this._toast((mime.includes('mp4') ? '开始录制 MP4' : `开始录制(${ext} 容器, 当前浏览器不支持 MP4 直录)`)
      + (this.includePanels ? ', 再点一次或按 R 结束' : '(纯 3D 画面), 再点一次或按 R 结束'));
  }

  stop() {
    if (!this.recording) return;
    this.recording = false;
    this._obs.disconnect();
    clearInterval(this._timer);
    window.removeEventListener('resize', this._onResize);
    document.getElementById('rec')?.classList.remove('show');
    const btn = document.getElementById('btn-rec');
    if (btn) {
      btn.classList.remove('rec');
      btn.textContent = '● 录制 ' + (this._ext === 'mp4' ? 'MP4' : (this._ext || 'MP4').toUpperCase());
    }
    window.__recState = 'stopping';
    try { if (this._rec && this._rec.state !== 'inactive') this._rec.stop(); } catch (e) { /* 已停止 */ }
  }

  // 每帧调用; 必须紧跟 renderer.render 之后(同一任务), 见文件头注释
  onFrame() {
    if (!this.recording) return;
    const gl = this.getSceneCanvas();
    if (gl && gl.width && gl.height) this.compCtx.drawImage(gl, 0, 0, this.comp.width, this.comp.height);
    else this.compCtx.clearRect(0, 0, this.comp.width, this.comp.height);
    if (this.includePanels && this.overlayReady) this.compCtx.drawImage(this.overlay, 0, 0, this.comp.width, this.comp.height);
    const now = performance.now();
    if (this.includePanels && this.dirty && !this.rasterBusy && !this.overlayFailed && now - this.lastRasterT >= OVERLAY_MIN_MS) {
      this._rasterize();
    }
  }

  // 自动化测试/诊断: 采样合成画布与面板层, 确认 3D 与面板都真实入镜
  debugSample() {
    const sample = (cv) => {
      if (!cv.width) return null;
      const s = document.createElement('canvas');
      s.width = 1; s.height = 1;
      const g = s.getContext('2d');
      const pick = (fx, fy) => {
        g.clearRect(0, 0, 1, 1);
        g.drawImage(cv, Math.max(0, (cv.width * fx) | 0), Math.max(0, (cv.height * fy) | 0), 2, 2, 0, 0, 1, 1);
        const d = g.getImageData(0, 0, 1, 1).data;
        return [d[0], d[1], d[2]];
      };
      const t = document.createElement('canvas');
      t.width = 96; t.height = 54;
      const tg = t.getContext('2d');
      tg.drawImage(cv, 0, 0, 96, 54);
      const d = tg.getImageData(0, 0, 96, 54).data;
      const colors = new Set();
      for (let i = 0; i < d.length; i += 4) colors.add(((d[i] >> 4) << 8) | ((d[i + 1] >> 4) << 4) | (d[i + 2] >> 4));
      return { headerPx: pick(0.02, 0.025), boardPx: pick(0.98, 0.13), centerPx: pick(0.5, 0.5), uniqueColors: colors.size };
    };
    return { comp: sample(this.comp), overlay: sample(this.overlay) };
  }

  // ---------- 内部 ----------
  _tickBadge() {
    const s = Math.max(0, Math.floor((performance.now() - this._startedAt) / 1000));
    const el = document.getElementById('rec-t');
    if (el) el.textContent = `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
  }

  async _finalize(mime, ext) {
    const blob = new Blob(this._chunks, { type: mime.split(';')[0] });
    this._chunks = [];
    const d = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    const name = `race_${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}.${ext}`;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);

    // 记录容器魔数供自动化测试校验(mp4: 第 4~8 字节为 "ftyp")
    let headHex = '';
    try {
      const head = new Uint8Array(await blob.slice(0, 12).arrayBuffer());
      headHex = [...head].map((b) => b.toString(16).padStart(2, '0')).join('');
    } catch (e) { /* 采样失败不影响保存 */ }

    const dur = (performance.now() - this._startedAt) / 1000;
    this.lastInfo = { name, mime, ext, size: blob.size, durationS: +dur.toFixed(1), headHex };
    window.__recInfo = this.lastInfo;
    window.__recState = 'idle';
    this._toast(`已保存 ${name}(${(blob.size / 1048576).toFixed(1)}MB)`);
  }

  _rasterize() {
    this.dirty = false;
    this.rasterBusy = true;
    this.lastRasterT = performance.now();
    const w = this.overlay.width, h = this.overlay.height;
    let url;
    try {
      url = this._buildOverlaySvg(w, h);
    } catch (e) {
      this.rasterBusy = false;
      this.overlayFailed = true;
      console.error('面板层序列化失败', e);
      this._toast('面板层序列化失败, 视频将只含 3D 画面', true);
      return;
    }
    const img = new Image();
    img.onload = () => {
      try {
        this.overlayCtx.clearRect(0, 0, w, h);
        this.overlayCtx.drawImage(img, 0, 0);
        this.overlayReady = true;
      } catch (e) {
        this.overlayFailed = true;
        console.error('面板层光栅化失败', e);
        this._toast('面板层光栅化失败, 视频将只含 3D 画面', true);
      } finally {
        this.rasterBusy = false;
      }
    };
    img.onerror = () => {
      this.rasterBusy = false;
      this.overlayFailed = true;
      console.error('面板层图像加载失败');
      this._toast('面板层光栅化失败, 视频将只含 3D 画面', true);
    };
    img.src = url;
  }

  // 把面板克隆进 foreignObject: 内嵌页面样式表, 根节点补齐 CSS 变量与 body 字体
  // (样式选择器 #hdr/#board 等直接生效; fixed 定位改 absolute —— wrapper 与视口同尺寸同原点)
  // 面板按 CSS 像素布局, 整体 transform 放大到物理像素输出 —— 否则高分屏下面板
  // 先按 CSS 尺寸光栅化再拉伸进物理像素画布, 文字会糊
  _buildOverlaySvg(outW, outH) {
    const cssW = window.innerWidth, cssH = window.innerHeight;
    const scale = cssW ? outW / cssW : 1;
    const wrapper = document.createElement('div');
    wrapper.setAttribute('xmlns', 'http://www.w3.org/1999/xhtml');
    const rs = getComputedStyle(document.documentElement);
    for (const v of CSS_VARS) {
      const val = rs.getPropertyValue(v).trim();
      if (val) wrapper.style.setProperty(v, val);
    }
    const bs = getComputedStyle(document.body);
    wrapper.style.fontFamily = bs.fontFamily;
    wrapper.style.fontSize = bs.fontSize;
    wrapper.style.lineHeight = bs.lineHeight;
    wrapper.style.color = bs.color;
    wrapper.style.width = cssW + 'px';
    wrapper.style.height = cssH + 'px';
    wrapper.style.position = 'relative';
    wrapper.style.overflow = 'hidden';
    wrapper.style.background = 'transparent';
    wrapper.style.transformOrigin = '0 0';
    wrapper.style.transform = 'scale(' + scale + ')';

    const styleEl = document.createElement('style');
    styleEl.textContent = collectPageCss();
    wrapper.appendChild(styleEl);

    for (const src of document.querySelectorAll(PANEL_SELECTOR)) {
      const clone = src.cloneNode(true);
      for (const el of clone.querySelectorAll('[data-norec]')) el.remove();
      const srcInputs = src.querySelectorAll('input');
      const dstInputs = clone.querySelectorAll('input');
      for (let i = 0; i < srcInputs.length && i < dstInputs.length; i++) {
        dstInputs[i].setAttribute('value', srcInputs[i].value); // cloneNode 不带动态 value
      }
      clone.style.position = 'absolute';
      wrapper.appendChild(clone);
    }

    const xhtml = new XMLSerializer().serializeToString(wrapper);
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${outW}" height="${outH}">`
      + `<foreignObject width="100%" height="100%">${xhtml}</foreignObject></svg>`;
    return 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
  }

  _toast(msg, isErr = false) {
    let t = document.getElementById('rec-toast');
    if (!t) {
      t = document.createElement('div');
      t.id = 'rec-toast';
      t.setAttribute('data-norec', '');
      document.body.appendChild(t);
    }
    t.textContent = msg;
    t.className = isErr ? 'show err' : 'show';
    clearTimeout(this._toastT);
    this._toastT = setTimeout(() => { t.className = ''; }, 4200);
  }
}

function collectPageCss() {
  let css = '';
  for (const sheet of document.styleSheets) {
    if (sheet.ownerNode?.tagName !== 'STYLE') continue; // 只带走内联样式表(跨域的拿不到)
    try {
      for (const rule of sheet.cssRules) css += rule.cssText + '\n';
    } catch (e) { /* 安全模式下逐条读取失败则跳过 */ }
  }
  return css;
}

// H.264 编码要求宽高为偶数
function evenize(n) { return Math.max(2, Math.round(n / 2) * 2); }
