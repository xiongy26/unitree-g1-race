// 极简静态文件服务器(node server.js [port])
// 缓存策略: ETag 协商缓存(no-cache) —— 每次刷新都带 If-None-Match 回来验证,
// 未变更的文件返回 304(约 160MB 资产的二次刷新只传几十 KB 的校验请求)。
const http = require('http');
const fs = require('fs');
const path = require('path');

const root = __dirname;
const port = parseInt(process.argv[2] || '8137', 10);
const mime = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.wasm': 'application/wasm', '.onnx': 'application/octet-stream', '.stl': 'application/octet-stream',
  '.xml': 'application/xml', '.json': 'application/json', '.css': 'text/css',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.map': 'application/json',
};

http.createServer((req, res) => {
  let p;
  try {
    p = decodeURIComponent(req.url.split('?')[0]);
  } catch (e) { // 畸形 URL(如未闭合的 % 序列)
    res.writeHead(400); res.end(); return;
  }
  if (p.endsWith('/')) p += 'index.html';
  const file = path.normalize(path.join(root, p));
  // 前缀校验必须带路径分隔符: 否则 /root-evil 兄弟目录可绕过 startsWith(root)
  if (file !== root && !file.startsWith(root + path.sep)) { res.writeHead(403); res.end(); return; }
  fs.stat(file, (err, st) => {
    if (err || !st.isFile()) { res.writeHead(404); res.end('404'); return; }
    const etag = `"${st.size.toString(16)}-${Math.round(st.mtimeMs).toString(16)}"`;
    const headers = {
      'Content-Type': mime[path.extname(file).toLowerCase()] || 'application/octet-stream',
      'Cache-Control': 'no-cache',
      ETag: etag,
    };
    if (req.headers['if-none-match'] === etag) {
      res.writeHead(304, headers);
      res.end();
      return;
    }
    res.writeHead(200, { ...headers, 'Content-Length': st.size });
    fs.createReadStream(file).pipe(res);
  });
}).listen(port, '127.0.0.1', () => console.log(`serving ${root} at http://127.0.0.1:${port}`));
