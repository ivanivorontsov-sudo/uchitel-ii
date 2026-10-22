// «Учитель ИИ» — оболочка Electron. Ставит встроенный Python в ASCII-папку и запускает локальный сервер.
const { app, BrowserWindow, ipcMain, dialog, shell, Menu, safeStorage } = require('electron');
const gh = require('./github');
const fs = require('fs'), path = require('path'), https = require('https'), cp = require('child_process'), net = require('net');
const PY_VER = '3.12.10';
const PY_URL = `https://www.python.org/ftp/python/${PY_VER}/python-${PY_VER}-embed-amd64.zip`;
const GETPIP = 'https://bootstrap.pypa.io/get-pip.py';
let win, server, port, logFile;

function pickHome() {
  if (process.env.UCHITEL_HOME) return process.env.UCHITEL_HOME;
  const c = process.platform === 'win32' ? ['C:\\ProgramData\\UchitelII', 'C:\\UchitelII'] : [path.join(require('os').homedir(), 'uchitel-data')];
  for (const d of c) { try { fs.mkdirSync(d, { recursive: true }); fs.writeFileSync(path.join(d, '.w'), ''); fs.rmSync(path.join(d, '.w')); return d } catch {} }
  return c[0];
}
const HOME = pickHome();
const PYDIR = path.join(HOME, 'python');
const PY = process.platform === 'win32' ? path.join(PYDIR, 'python.exe') : (process.env.UCHITEL_PY || 'python3');
const APPDIR = path.join(HOME, 'app');
const res = p => app.isPackaged ? path.join(process.resourcesPath, p) : path.join(__dirname, '..', p);
function log(t) { try { fs.mkdirSync(path.join(HOME, 'logs'), { recursive: true }); fs.appendFileSync(path.join(HOME, 'logs', 'shell.log'), new Date().toISOString() + ' ' + t + '\n') } catch {} }
const send = (ch, d) => win && !win.isDestroyed() && win.webContents.send(ch, d);   // окно может быть уже закрыто

function download(url, dest, onp, tries = 3) {
  return new Promise((ok, bad) => {
    const part = dest + '.part'; let have = fs.existsSync(part) ? fs.statSync(part).size : 0;
    const go = (u, n) => https.get(u, { headers: { 'User-Agent': 'UchitelII', ...(have ? { Range: `bytes=${have}-` } : {}) }, timeout: 60000 }, r => {
      if ([301, 302, 303, 307, 308].includes(r.statusCode)) { r.resume(); return go(new URL(r.headers.location, u).href, n) }
      if (r.statusCode === 416) { r.resume(); fs.renameSync(part, dest); return ok(dest) }
      if (r.statusCode !== 200 && r.statusCode !== 206) { r.resume(); return retry(new Error('HTTP ' + r.statusCode), n) }
      if (have && r.statusCode === 200) have = 0;
      const total = +(r.headers['content-length'] || 0) + have; let done = have;
      const f = fs.createWriteStream(part, { flags: have ? 'a' : 'w' });
      r.on('data', b => { done += b.length; onp && onp(done, total) }); r.pipe(f);
      f.on('finish', () => { if (total && fs.statSync(part).size < total) return retry(new Error('недокачано'), n); fs.renameSync(part, dest); ok(dest) });
      r.on('error', e => retry(e, n));
    }).on('error', e => retry(e, n)).on('timeout', function () { this.destroy(new Error('таймаут')) });
    const retry = (e, n) => { if (n >= tries) return bad(e); have = fs.existsSync(part) ? fs.statSync(part).size : 0; setTimeout(() => go(url, n + 1), 3000 * n) };
    go(url, 1);
  });
}
function unzip(file, dir) {   // встроенная распаковка ZIP (без tar/PowerShell)
  const zlib = require('zlib'), b = fs.readFileSync(file);
  let e = b.length - 22; while (e >= 0 && b.readUInt32LE(e) !== 0x06054b50) e--;
  if (e < 0) throw new Error('повреждённый ZIP');
  let n = b.readUInt16LE(e + 10), o = b.readUInt32LE(e + 16);
  while (n--) {
    const m = b.readUInt16LE(o + 10), cs = b.readUInt32LE(o + 20), fl = b.readUInt16LE(o + 28), xl = b.readUInt16LE(o + 30), cl = b.readUInt16LE(o + 32), lo = b.readUInt32LE(o + 42);
    const name = b.toString('utf8', o + 46, o + 46 + fl); o += 46 + fl + xl + cl;
    const out = path.join(dir, name); if (!out.startsWith(path.resolve(dir))) continue;
    if (name.endsWith('/')) { fs.mkdirSync(out, { recursive: true }); continue }
    const ds = lo + 30 + b.readUInt16LE(lo + 26) + b.readUInt16LE(lo + 28), raw = b.subarray(ds, ds + cs);
    fs.mkdirSync(path.dirname(out), { recursive: true }); fs.writeFileSync(out, m === 0 ? raw : zlib.inflateRawSync(raw));
  }
}
function runP(cmd, args, opts = {}) {
  return new Promise(ok => { const p = cp.spawn(cmd, args, { windowsHide: true, ...opts }); let out = '';
    p.stdout?.on('data', d => { out += d; send('boot-log', String(d)) }); p.stderr?.on('data', d => { out += d; send('boot-log', String(d)) });
    p.on('error', e => ok({ code: -1, out: String(e) })); p.on('close', code => ok({ code, out })) });
}
function syncBackend() {   // копируем код сервера в ASCII-папку (путь установки может содержать кириллицу/пробелы)
  const src = res('backend'), dst = path.join(APPDIR, 'backend');
  fs.mkdirSync(dst, { recursive: true });
  for (const f of fs.readdirSync(src)) if (f.endsWith('.py')) fs.copyFileSync(path.join(src, f), path.join(dst, f));
  fs.cpSync(res('app/ui'), path.join(APPDIR, 'ui'), { recursive: true });
}
async function bootstrap() {
  if (process.platform !== 'win32') return true;
  const step = (t, p) => send('boot', { t, p });
  if (!fs.existsSync(PY)) {
    step('Скачиваю встроенный Python ' + PY_VER + '…', 0);
    const z = path.join(HOME, 'downloads', `python-${PY_VER}-embed-amd64.zip`); fs.mkdirSync(path.dirname(z), { recursive: true });
    if (!fs.existsSync(z)) await download(PY_URL, z, (d, t) => step(`Скачиваю Python: ${(d / 1e6).toFixed(1)} из ${(t / 1e6).toFixed(1)} МБ`, t ? d / t : 0));
    fs.mkdirSync(PYDIR, { recursive: true });
    try { unzip(z, path.resolve(PYDIR)) } catch (e) { fs.rmSync(z, { force: true }); throw new Error('Не удалось распаковать Python (архив удалён, будет скачан заново): ' + e.message) }
    const pth = fs.readdirSync(PYDIR).find(f => f.endsWith('._pth'));
    fs.writeFileSync(path.join(PYDIR, pth), `python312.zip\n.\nLib\\site-packages\n${path.join(APPDIR, 'backend')}\nimport site\n`);
  }
  if (!fs.existsSync(path.join(PYDIR, 'Lib', 'site-packages', 'pip'))) {
    step('Ставлю pip…', 0.5);
    const gp = path.join(HOME, 'downloads', 'get-pip.py'); if (!fs.existsSync(gp)) await download(GETPIP, gp);
    const r = await runP(PY, [gp, '--no-warn-script-location'], { env: pyEnv() });
    if (r.code !== 0) throw new Error('pip не установился: ' + r.out.slice(-400));
  }
  return true;
}
function pyEnv() { const t = path.join(HOME, 'tmp'); fs.mkdirSync(t, { recursive: true });
  return { ...process.env, UCHITEL_HOME: HOME, UCHITEL_UI: path.join(APPDIR, 'ui'), PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8', TMP: t, TEMP: t, HF_HOME: path.join(HOME, 'hf'), PYTHONNOUSERSITE: '1', ...(getSecret('hf') ? { HF_TOKEN: getSecret('hf') } : {}) } }
function freePort() { return new Promise(ok => { const s = net.createServer(); s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => ok(p)) }) }) }
async function startServer() {
  port = await freePort();
  server = cp.spawn(PY, ['-u', path.join(APPDIR, 'backend', 'server.py'), String(port)], { env: pyEnv(), windowsHide: true });
  return new Promise((ok, bad) => {
    let buf = ''; const t = setTimeout(() => bad(new Error('Сервер не ответил за 60 секунд.\n' + buf.slice(-600))), 60000);
    const on = d => { buf += d; log('srv: ' + String(d).trim()); if (buf.includes('"ready"')) { clearTimeout(t); ok() } };
    server.stdout.on('data', on); server.stderr.on('data', on);
    server.on('exit', c => { clearTimeout(t); if (!app.isQuitting) bad(new Error('Сервер завершился (код ' + c + ').\n' + buf.slice(-600))) });
  });
}
async function main() {
  Menu.setApplicationMenu(null);
  win = new BrowserWindow({ width: 1280, height: 860, title: 'Учитель ИИ', icon: path.join(__dirname, 'icon.png'),
    webPreferences: { preload: path.join(__dirname, 'preload.js'), contextIsolation: true } });
  win.on('closed', () => { win = null });
  await win.loadFile(path.join(__dirname, 'boot.html'));
  try {
    send('boot', { t: 'Готовлю папку ' + HOME + '…', p: 0 }); syncBackend();
    await bootstrap(); send('boot', { t: 'Запускаю…', p: 1 }); await startServer();
    win && win.loadURL(`http://127.0.0.1:${port}/`);
  } catch (e) { log('boot error: ' + e.stack); send('boot-error', String(e.message || e)) }
}
ipcMain.handle('pick', async (_, kind) => { const r = await dialog.showOpenDialog(win, { properties: kind === 'dir' ? ['openDirectory', 'multiSelections'] : ['openFile', 'multiSelections'],
  filters: kind === 'dir' ? [] : kind === 'weights' ? [{ name: 'Веса Ivan GPT', extensions: ['safetensors', 'zip', 'apk', 'bin'] }, { name: 'Все', extensions: ['*'] }] : [{ name: 'Данные', extensions: ['txt', 'md', 'pdf', 'docx', 'csv', 'tsv', 'json', 'jsonl', 'zip', 'html'] }, { name: 'Все', extensions: ['*'] }] }); return r.canceled ? [] : r.filePaths });
ipcMain.handle('open', (_, u) => /^https:\/\//.test(u) && shell.openExternal(u));
ipcMain.handle('reveal', (_, p) => shell.showItemInFolder(p));
// ---- секреты: GitHub и Hugging Face токены, зашифрованы safeStorage (DPAPI в Windows), лежат только на этом компьютере ----
const secretFile = k => path.join(app.getPath('userData'), k + '.secret');
function getSecret(k) { try { return safeStorage.decryptString(fs.readFileSync(secretFile(k))) } catch { return '' } }
ipcMain.handle('secret-set', (_, k, v) => { if (!['github', 'hf'].includes(k)) return false;
  if (!v) { fs.rmSync(secretFile(k), { force: true }); return true }
  if (!safeStorage.isEncryptionAvailable()) throw new Error('Шифрование Windows недоступно — токен не сохранён.');
  fs.writeFileSync(secretFile(k), safeStorage.encryptString(String(v).trim())); return true });
ipcMain.handle('secret-has', (_, k) => !!getSecret(k));
ipcMain.handle('ivan-releases', async () => { const t = getSecret('github'); if (!t) throw new Error('Введите GitHub-токен в настройках.'); return gh.releases(t) });
ipcMain.handle('ivan-fetch', async (_, tag) => { const t = getSecret('github'); if (!t) throw new Error('Введите GitHub-токен в настройках.');
  return gh.fetchIvan(t, tag, HOME, (text, p) => send('ivan-progress', { text, p })) });
ipcMain.handle('retry', () => { app.relaunch(); app.exit(0) });
app.setAppUserModelId('ru.vorontsov.uchitel');
if (!app.requestSingleInstanceLock()) app.quit();
else { app.on('second-instance', () => win && (win.restore(), win.focus())); app.whenReady().then(main) }
app.on('before-quit', () => { app.isQuitting = true; try { if (server) process.platform === 'win32' ? cp.spawnSync('taskkill', ['/PID', String(server.pid), '/T', '/F'], { windowsHide: true }) : server.kill() } catch {} });
app.on('window-all-closed', () => app.quit());
process.on('uncaughtException', e => log('uncaught: ' + e.stack));
