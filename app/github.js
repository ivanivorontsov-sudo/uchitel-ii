// Скачивание Ivan GPT из приватного репозитория по GitHub-токену. Токен живёт только в главном процессе Electron
// (зашифрован safeStorage), в интерфейс и в Python не передаётся. Авторизация снимается при переходе на другой хост (S3).
const https = require('https'), fs = require('fs'), path = require('path');
const REPO = 'ivanivorontsov-sudo/ivan-gpt';
function req(url, token, accept, redirects = 5) {
  return new Promise((ok, bad) => {
    const u = new URL(url), gh = u.hostname === 'api.github.com';
    https.get(u, { headers: { 'User-Agent': 'UchitelII', Accept: accept || 'application/vnd.github+json', ...(gh && token ? { Authorization: 'Bearer ' + token } : {}), 'X-GitHub-Api-Version': '2022-11-28' }, timeout: 60000 }, r => {
      if ([301, 302, 303, 307, 308].includes(r.statusCode) && redirects) { r.resume(); return ok(req(new URL(r.headers.location, url).href, gh && new URL(r.headers.location, url).hostname === 'api.github.com' ? token : null, accept, redirects - 1)) }
      ok(r);
    }).on('error', bad).on('timeout', function () { this.destroy(new Error('таймаут')) });
  });
}
async function json(url, token) {
  const r = await req(url, token); let b = ''; r.setEncoding('utf8'); for await (const c of r) b += c;
  if (r.statusCode === 401) throw new Error('GitHub не принял токен (401). Проверьте токен в настройках.');
  if (r.statusCode === 404) throw new Error('Репозиторий не найден (404): у токена нет доступа к ' + REPO + '.');
  if (r.statusCode >= 400) throw new Error('GitHub ответил ' + r.statusCode + ': ' + b.slice(0, 200));
  return JSON.parse(b);
}
async function releases(token) {
  const rs = await json(`https://api.github.com/repos/${REPO}/releases?per_page=20`, token);
  return rs.map(r => ({ tag: r.tag_name, name: r.name, date: r.published_at, latest: false, assets: r.assets.map(a => ({ name: a.name, size: a.size, url: a.url })) }))
    .map((r, i) => ({ ...r, latest: i === 0 }));
}
function pickAssets(assets) {   // сначала чистые веса, иначе APK (в нём web/model.bin — те же веса в fp16)
  const w = assets.find(a => /model\.safetensors$/i.test(a.name)) || assets.find(a => /weights.*\.zip$/i.test(a.name)) || assets.find(a => /\.apk$/i.test(a.name));
  return w ? [w] : [];
}
async function file(url, token, dest, accept, onp) {
  fs.mkdirSync(path.dirname(dest), { recursive: true }); const part = dest + '.part';
  for (let n = 1; n <= 3; n++) {
    try {
      const r = await req(url, token, accept);
      if (r.statusCode !== 200) { r.resume(); throw new Error('HTTP ' + r.statusCode) }
      const total = +(r.headers['content-length'] || 0); let done = 0; const f = fs.createWriteStream(part);
      await new Promise((ok, bad) => { r.on('data', b => { done += b.length; onp && onp(done, total) }); r.pipe(f); f.on('finish', ok); r.on('error', bad); f.on('error', bad) });
      if (total && fs.statSync(part).size < total) throw new Error('недокачано');
      fs.renameSync(part, dest); return dest;
    } catch (e) { if (n === 3) throw e; await new Promise(z => setTimeout(z, 2000 * n)) }
  }
}
async function fetchIvan(token, tag, home, onp) {
  const rel = (await releases(token)).find(r => r.tag === tag); if (!rel) throw new Error('Нет релиза ' + tag);
  const a = pickAssets(rel.assets); if (!a.length) throw new Error('В релизе ' + tag + ' нет файлов с весами.');
  const dir = path.join(home, 'downloads', 'ivan-gpt', tag.replace(/[^\w.-]/g, '_'));
  const src = path.join(dir, a[0].name);
  if (!fs.existsSync(src)) await file(a[0].url, token, src, 'application/octet-stream', (d, t) => onp && onp(`${a[0].name}: ${(d / 1e6).toFixed(1)} из ${(t / 1e6).toFixed(1)} МБ`, t ? d / t : 0));
  const tok = path.join(dir, 'ivan.model');
  if (!fs.existsSync(tok)) await file(`https://api.github.com/repos/${REPO}/contents/tok/ivan.model?ref=${encodeURIComponent(tag)}`, token, tok, 'application/vnd.github.raw');
  return { src, tok, tag, asset: a[0].name };
}
module.exports = { releases, fetchIvan, pickAssets, REPO };
