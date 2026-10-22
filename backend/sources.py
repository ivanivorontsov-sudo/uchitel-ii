# Источники данных для предобучения: потоково, с докачкой, очисткой, дедупликацией, фильтром языка, лимитом и
# токенизацией сразу в .bin (uint16/uint32 memmap). Только легальные открытые источники; лицензия показывается в каталоге.
import os, re, io, json, time, hashlib, bz2, lzma, zipfile, tarfile, urllib.request, urllib.parse, urllib.robotparser, html
from common import DIRS, free_gb

CATALOG = [
    {'id': 'wikimedia', 'name': 'Википедия / Викисловарь / Викитека (дамп Wikimedia)', 'license': 'CC BY-SA 4.0 и GFDL (указывать авторство, распространять на тех же условиях)',
     'url': 'https://dumps.wikimedia.org/legal.html', 'params': {'lang': 'ru', 'project': 'wiki'}},
    {'id': 'hf', 'preset': 'fineweb2_ru', 'name': 'FineWeb-2 (русский веб, очищенный)', 'license': 'ODC-By 1.0 (+ условия исходных сайтов, Common Crawl ToU)',
     'url': 'https://huggingface.co/datasets/HuggingFaceFW/fineweb-2', 'params': {'repo': 'HuggingFaceFW/fineweb-2', 'config': 'rus_Cyrl', 'field': 'text'}},
    {'id': 'hf', 'preset': 'openwebmath', 'name': 'OpenWebMath (математика)', 'license': 'ODC-By 1.0',
     'url': 'https://huggingface.co/datasets/open-web-math/open-web-math', 'params': {'repo': 'open-web-math/open-web-math', 'field': 'text'}},
    {'id': 'hf', 'preset': 'starcoderdata', 'name': 'StarCoderData (код, нужен HF-токен и согласие на странице набора)', 'license': 'Пермиссивные лицензии исходных репозиториев; условия The Stack (gated)',
     'url': 'https://huggingface.co/datasets/bigcode/starcoderdata', 'params': {'repo': 'bigcode/starcoderdata', 'data_dir': 'python', 'field': 'content'}, 'gated': True},
    {'id': 'hf', 'preset': 'stack_v2_smol', 'name': 'The Stack v2 (smol-ids; содержимое файлов скачивается из Software Heritage)', 'license': 'Условия The Stack v2 (gated), лицензии исходных репозиториев',
     'url': 'https://huggingface.co/datasets/bigcode/the-stack-v2-train-smol-ids', 'params': {'repo': 'bigcode/the-stack-v2-train-smol-ids'}, 'gated': True,
     'note': 'В наборе только идентификаторы файлов, сам код лежит в Software Heritage (AWS S3). Для обучения проще StarCoderData.'},
    {'id': 'hf', 'preset': 'gutenberg', 'name': 'Книги Project Gutenberg (через набор на Hugging Face)', 'license': 'Общественное достояние в США; товарный знак Project Gutenberg удалён',
     'url': 'https://huggingface.co/datasets/manu/project_gutenberg', 'params': {'repo': 'manu/project_gutenberg', 'split': 'ru', 'field': 'text'}},
    {'id': 'hf', 'name': 'Любой набор Hugging Face по имени (потоково)', 'license': 'Смотрите карточку набора', 'params': {'repo': '', 'field': 'text'}},
    {'id': 'gutenberg', 'name': 'Книга Project Gutenberg по номеру', 'license': 'Общественное достояние в США', 'url': 'https://www.gutenberg.org/policy/robot_access.html', 'params': {'ids': ''}},
    {'id': 'github', 'name': 'Репозиторий GitHub по ссылке (фильтр языков и лицензий)', 'license': 'Берутся только репозитории с разрешённой лицензией (MIT, Apache-2.0, BSD, ISC, MPL, Unlicense, CC0…)', 'params': {'url': '', 'langs': 'py,js,ts,cs,java,cpp,c,go,rs'}},
    {'id': 'url', 'name': 'Файл или архив по ссылке (zip, tar, bz2, xz, zst, jsonl, parquet, txt)', 'license': 'Ответственность пользователя: используйте только данные, которые разрешено использовать', 'params': {'url': ''}},
    {'id': 'site', 'name': 'Сайт по ссылке (учитывает robots.txt, не чаще 1 запроса в секунду)', 'license': 'Условия сайта; соблюдается robots.txt', 'params': {'url': '', 'max_pages': 200}},
]
OK_LICENSES = ('mit', 'apache', 'bsd', 'isc', 'mpl', 'unlicense', 'cc0', 'zlib', 'boost', 'mozilla public', 'cc-by-4', 'creative commons attribution 4')
UA = 'UchitelII/1.1 (+https://github.com/ivanivorontsov-sudo/uchitel-ii)'

def _open(url, headers=None, timeout=60):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': UA, **(headers or {})}), timeout=timeout)

def download_resumable(url, dest, emit, headers=None, attempts=3):
    """Скачивание с докачкой (.part)."""
    part = dest + '.part'
    for a in range(1, attempts + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        try:
            with _open(url, {**(headers or {}), **({'Range': f'bytes={have}-'} if have else {})}) as r:
                if have and r.status != 206: have = 0
                total = int(r.headers.get('Content-Length') or 0) + have
                with open(part, 'ab' if have else 'wb') as f:
                    done, last = have, 0
                    while True:
                        b = r.read(1 << 20)
                        if not b: break
                        f.write(b); done += len(b)
                        if time.time() - last > 1: last = time.time(); emit(progress=done / total if total else 0, text=f'{done/1e6:.0f} из {total/1e6:.0f} МБ')
            os.replace(part, dest); return dest
        except Exception as e:
            if a == attempts: raise SystemExit(f'Не удалось скачать {url}: {e}. Недокачанный файл сохранён — нажмите ещё раз, чтобы продолжить.')
            time.sleep(3 * a)

# ---------- очистка и фильтры ----------
def clean(t):
    t = html.unescape(t).replace('\r\n', '\n').replace('\r', '\n').replace('\x00', '')
    t = re.sub(r'[ \t\u00a0]+', ' ', t); t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()
def lang_ok(t, lang):
    if not lang or lang == 'any': return True
    s = t[:2000]; letters = [c for c in s if c.isalpha()]
    if len(letters) < 20: return False
    cyr = sum('\u0400' <= c <= '\u04ff' for c in letters) / len(letters)
    return cyr > 0.5 if lang == 'ru' else cyr < 0.1 if lang in ('en', 'code') else True

# ---------- генераторы текстов ----------
def wiki_texts(p, emit):
    """Потоковое чтение дампа pages-articles.xml.bz2 прямо из сети (без хранения на диске)."""
    lang, proj = p.get('lang', 'ru'), p.get('project', 'wiki')
    url = p.get('url') or f'https://dumps.wikimedia.org/{lang}{proj}/latest/{lang}{proj}-latest-pages-articles.xml.bz2'
    import xml.etree.ElementTree as ET
    emit(log=f'Читаю дамп потоково: {url}')
    r = _open(url, timeout=120); dec = bz2.BZ2File(r)
    for ev, el in ET.iterparse(dec, events=('end',)):
        if el.tag.endswith('}page'):
            ns = el.find('{*}ns'); txt = el.find('.//{*}text')
            if ns is not None and ns.text == '0' and txt is not None and txt.text and not txt.text.lower().startswith('#redirect') and not txt.text.startswith('#перенаправление'):
                t = strip_wiki(txt.text); title = el.find('{*}title').text or ''
                if len(t) > 200: yield (title + '\n\n' + t)
            el.clear()
def strip_wiki(t):
    t = re.sub(r'<!--.*?-->', '', t, flags=re.S); t = re.sub(r'<ref[^>]*/>', '', t); t = re.sub(r'<ref.*?</ref>', '', t, flags=re.S)
    for _ in range(3): t = re.sub(r'\{\{[^{}]*\}\}', '', t)
    t = re.sub(r'\{\|.*?\|\}', '', t, flags=re.S); t = re.sub(r'\[\[(?:Файл|Изображение|File|Image|Категория|Category):[^\]]*\]\]', '', t)
    t = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]]*)\]\]', r'\1', t); t = re.sub(r'\[https?://\S+ ([^\]]*)\]', r'\1', t)
    t = re.sub(r"'{2,}", '', t); t = re.sub(r'<[^>]+>', '', t); t = re.sub(r'^=+\s*(.*?)\s*=+$', r'\1', t, flags=re.M)
    t = t.replace('\u0301', ''); t = re.sub(r'\(\s*[,;]?\s*\)', '', t)
    return clean(t)

def hf_texts(p, emit):
    from datasets import load_dataset
    kw = {k: p[k] for k in ('data_dir',) if p.get(k)}
    tok = p.get('hf_token') or None
    ds = load_dataset(p['repo'], p.get('config') or None, split=p.get('split') or 'train', streaming=True, token=tok, **kw)
    f = p.get('field') or 'text'
    for row in ds:
        v = row.get(f) or row.get('text') or row.get('content')
        if isinstance(v, str): yield v

def gutenberg_texts(p, emit):
    rp = urllib.robotparser.RobotFileParser('https://www.gutenberg.org/robots.txt')
    try: rp.read()
    except Exception: pass
    for i in re.findall(r'\d+', str(p.get('ids', ''))):
        u = f'https://www.gutenberg.org/cache/epub/{i}/pg{i}.txt'
        if not rp.can_fetch(UA, u): emit(log=f'robots.txt запрещает {u}'); continue
        t = _open(u).read().decode('utf-8', 'replace')
        a = re.search(r'\*\*\* ?START OF.*?\*\*\*', t); b = re.search(r'\*\*\* ?END OF', t)
        yield t[a.end() if a else 0: b.start() if b else len(t)]; time.sleep(2)

def github_texts(p, emit):
    m = re.match(r'https?://github\.com/([^/]+)/([^/#?]+)', p['url'].strip())
    if not m: raise SystemExit('Нужна ссылка вида https://github.com/владелец/репозиторий')
    o, r = m.group(1), m.group(2).removesuffix('.git'); langs = {'.' + x.strip().lstrip('.') for x in p.get('langs', '').split(',') if x.strip()}
    dest = os.path.join(DIRS['downloads'], f'gh_{o}_{r}.zip')
    download_resumable(f'https://codeload.github.com/{o}/{r}/zip/HEAD', dest, emit)
    z = zipfile.ZipFile(dest); lic = [n for n in z.namelist() if re.search(r'^[^/]+/(LICEN[CS]E|COPYING)(\.\w+)?$', n, re.I)]
    lt = z.read(lic[0]).decode('utf-8', 'replace').lower()[:3000] if lic else ''
    if not any(k in lt for k in OK_LICENSES) and not p.get('allow_any_license'):
        raise SystemExit(f'У репозитория {o}/{r} нет разрешённой открытой лицензии (MIT, Apache, BSD…) — пропускаю.')
    emit(log=f'Лицензия найдена: {lic[0].split("/",1)[1] if lic else "?"}')
    for n in z.namelist():
        if n.endswith('/') or (langs and os.path.splitext(n)[1].lower() not in langs): continue
        if z.getinfo(n).file_size > 1_000_000: continue
        yield f'# {n.split("/",1)[1]}\n' + z.read(n).decode('utf-8', 'replace')
    os.remove(dest)

def _lines_texts(fobj, name):
    name = name.lower()
    if name.endswith('.jsonl') or name.endswith('.json'):
        for l in io.TextIOWrapper(fobj, encoding='utf-8', errors='replace'):
            try: d = json.loads(l)
            except Exception: continue
            v = d.get('text') or d.get('content') if isinstance(d, dict) else None
            if isinstance(v, str): yield v
    elif name.endswith('.parquet'):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(io.BytesIO(fobj.read()))
        for b in pf.iter_batches(columns=None):
            col = next((c for c in ('text', 'content') if c in b.schema.names), None)
            if col:
                for v in b.column(col).to_pylist():
                    if isinstance(v, str): yield v
    elif name.endswith(('.txt', '.md', '.text')) or '.' not in os.path.basename(name):
        buf = []
        for l in io.TextIOWrapper(fobj, encoding='utf-8', errors='replace'):
            if not l.strip() and buf: yield ''.join(buf); buf = []
            else: buf.append(l)
        if buf: yield ''.join(buf)

def url_texts(p, emit):
    u = p['url'].strip(); name = urllib.parse.urlparse(u).path.lower()
    dest = os.path.join(DIRS['downloads'], re.sub(r'[^A-Za-z0-9._-]', '_', os.path.basename(name) or 'file'))
    download_resumable(u, dest, emit, headers={'Authorization': 'Bearer ' + p['hf_token']} if p.get('hf_token') and 'huggingface.co' in u else None)
    def unwrap(path, nm):
        if nm.endswith('.zip'):
            z = zipfile.ZipFile(path)
            for n in z.namelist():
                if not n.endswith('/'): yield from _lines_texts(z.open(n), n)
        elif re.search(r'\.(tar|tar\.gz|tgz|tar\.bz2|tar\.xz)$', nm):
            with tarfile.open(path, 'r:*') as t:
                for m in t:
                    if m.isfile(): yield from _lines_texts(t.extractfile(m), m.name)
        elif nm.endswith('.bz2'): yield from _lines_texts(bz2.open(path), nm[:-4])
        elif nm.endswith('.xz'): yield from _lines_texts(lzma.open(path), nm[:-3])
        elif nm.endswith('.gz'):
            import gzip; yield from _lines_texts(gzip.open(path), nm[:-3])
        elif nm.endswith('.zst'):
            import zstandard; yield from _lines_texts(zstandard.ZstdDecompressor().stream_reader(open(path, 'rb')), nm[:-4])
        else: yield from _lines_texts(open(path, 'rb'), nm)
    yield from unwrap(dest, name)
    if not p.get('keep_file'): os.remove(dest)

def site_texts(p, emit):
    start = p['url'].strip(); host = urllib.parse.urlparse(start).netloc
    rp = urllib.robotparser.RobotFileParser(f'{urllib.parse.urlparse(start).scheme}://{host}/robots.txt')
    try: rp.read()
    except Exception: pass
    delay = max(1.0, float(rp.crawl_delay(UA) or 0), float(p.get('delay', 1)))
    q, seen, n = [start], {start}, 0
    while q and n < int(p.get('max_pages', 200)):
        u = q.pop(0)
        if not rp.can_fetch(UA, u): continue
        try:
            with _open(u, timeout=30) as r:
                if 'text/html' not in r.headers.get('Content-Type', ''): continue
                h = r.read(3_000_000).decode('utf-8', 'replace')
        except Exception: continue
        n += 1; emit(progress=n / int(p.get('max_pages', 200)), text=f'страниц {n}')
        for a in re.findall(r'href=["\']([^"\'#]+)', h):
            v = urllib.parse.urljoin(u, a)
            if urllib.parse.urlparse(v).netloc == host and v not in seen and not re.search(r'\.(jpg|png|gif|pdf|zip|css|js)$', v, re.I): seen.add(v); q.append(v)
        body = re.sub(r'(?is)<(script|style|nav|header|footer|noscript)[^>]*>.*?</\1>', ' ', h)
        t = re.sub(r'(?s)<[^>]+>', '\n', body); t = clean(re.sub(r'\n\s*\n+', '\n\n', t))
        if len(t) > 200: yield t
        time.sleep(delay)

GEN = {'wikimedia': wiki_texts, 'hf': hf_texts, 'gutenberg': gutenberg_texts, 'github': github_texts, 'url': url_texts, 'site': site_texts}

# ---------- токенизация в .bin ----------
class BinWriter:
    """Пишет токены в train.bin/val.bin (каждый 100-й документ — в проверку). SentencePiece — в несколько процессов."""
    def __init__(s, out, model_dir, workers=None):
        os.makedirs(out, exist_ok=True); s.out = out; s.n = {'train': 0, 'val': 0}; s.docs = 0
        tm = os.path.join(model_dir, 'tokenizer.model')
        from transformers import AutoTokenizer
        s.tok = AutoTokenizer.from_pretrained(model_dir); s.eos = s.tok.eos_token_id
        V = max(len(s.tok), s.tok.vocab_size); s.dt = 'uint16' if V < 65535 else 'uint32'
        s.sp = tm if os.path.exists(tm) else None; s.pool = None
        if s.sp:
            import multiprocessing as mp
            s.pool = mp.get_context('spawn').Pool(workers or max(1, min(8, (os.cpu_count() or 2) - 1)), initializer=_sp_init, initargs=(s.sp,))
        s.f = {k: open(os.path.join(out, f'{k}.bin'), 'wb') for k in ('train', 'val')}; s.buf = []
    def add(s, text):
        s.buf.append(text)
        if len(s.buf) >= 2000: s.flush()
    def flush(s):
        import numpy as np
        if not s.buf: return
        ids = s.pool.map(_sp_encode, s.buf, chunksize=64) if s.pool else s.tok(s.buf, add_special_tokens=False)['input_ids']
        for x in ids:
            k = 'val' if s.docs % 100 == 99 else 'train'; s.docs += 1
            a = np.array(x + [s.eos], dtype=s.dt); a.tofile(s.f[k]); s.n[k] += len(a)
        s.buf = []
    def close(s):
        s.flush(); [f.close() for f in s.f.values()]
        if s.pool: s.pool.close()
        meta = {'dtype': s.dt, 'train_tokens': s.n['train'], 'val_tokens': s.n['val'], 'docs': s.docs, 'eos': s.eos}
        json.dump(meta, open(os.path.join(s.out, 'meta.json'), 'w')); return meta
_SP = None
def _sp_init(p):
    global _SP; import sentencepiece as spm; _SP = spm.SentencePieceProcessor(model_file=p)
def _sp_encode(t): return _SP.encode(t)

def collect(p, emit):
    """Общий конвейер: источник → очистка → язык → дедупликация → лимит → jsonl (по желанию) и .bin."""
    name = re.sub(r'[^A-Za-z0-9._-]', '_', p.get('name') or p['source'])
    out = os.path.join(DIRS['datasets'], name); os.makedirs(out, exist_ok=True)
    limit = float(p.get('limit_mb', 500)) * 1e6; lang = p.get('lang_filter', 'any'); minlen = int(p.get('minlen', 100))
    bw = BinWriter(os.path.join(out, 'bin'), p['tokenize_with']) if p.get('tokenize_with') else None
    jf = open(os.path.join(out, 'train.jsonl'), 'w', encoding='utf-8') if p.get('keep_text', True) else None
    seen = set(); st = {'texts': 0, 'dupes': 0, 'short': 0, 'lang': 0, 'chats': 0}; size = 0
    for t in GEN[p['source']](p, emit):
        t = clean(t)
        if len(t) < minlen: st['short'] += 1; continue
        if not lang_ok(t, lang): st['lang'] += 1; continue
        h = hashlib.sha1(re.sub(r'\W+', '', t.lower()).encode()).digest()[:12]
        if h in seen: st['dupes'] += 1; continue
        seen.add(h); st['texts'] += 1; size += len(t.encode())
        if jf: jf.write(json.dumps({'text': t}, ensure_ascii=False) + '\n')
        if bw: bw.add(t)
        if st['texts'] % 200 == 0:
            emit(progress=min(1, size / limit), text=f"{size/1e6:.0f} из {limit/1e6:.0f} МБ · текстов {st['texts']}")
            if free_gb(out) < 2: emit(log='На диске меньше 2 ГБ — останавливаюсь.'); break
        if size >= limit: emit(log='Достигнут лимит размера.'); break
    if jf: jf.close()
    meta = bw.close() if bw else None
    info = {'stats': st, 'sources': [p['source']], 'license': next((c['license'] for c in CATALOG if c['id'] == p['source'] and (c.get('preset') == p.get('preset'))), ''), 'bytes': size, 'bin': meta}
    json.dump(info, open(os.path.join(out, 'info.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    return {'path': os.path.join(out, 'train.jsonl'), 'dir': out, 'stats': st, 'bin': os.path.join(out, 'bin') if bw else None, 'tokens': meta}

def tokenize_jsonl(paths, model_dir, out, emit):
    """Готовые наборы (вкладка «Данные») → .bin; диалоги переводятся в формат «Человек:/ИИ:» шаблоном модели."""
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(model_dir); bw = BinWriter(out, model_dir); n = 0
    for pth in paths:
        for l in open(pth, encoding='utf-8'):
            d = json.loads(l)
            if d.get('messages'):
                try: t = tk.apply_chat_template(d['messages'], tokenize=False)
                except Exception: t = '\n'.join(f"{m['role']}: {m['content']}" for m in d['messages'])
            else: t = d.get('text') or ''
            if t: bw.add(t); n += 1
            if n % 5000 == 0: emit(text=f'документов {n}')
    return bw.close()
