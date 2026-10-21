# Фоновые задачи «Учитель ИИ». Запуск: python tasks.py <вид> <json-файл параметров>.
# Пишут в stdout строки JSON: {"log": "..."} / {"progress": 0..1, "text": "..."} / {"result": {...}}
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, time, hashlib, zipfile, re, io, glob, shutil, csv, unicodedata, subprocess, urllib.request, urllib.parse
from common import DIRS, free_gb, human, child_env, decode

def emit(**k):
    print(json.dumps(k, ensure_ascii=False), flush=True)

def log(t): emit(log=t)

# ---------- скачивание с докачкой ----------
def http_download(url, dest, sha256=None, attempts=3):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + '.part'
    for a in range(1, attempts + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(url, headers={'User-Agent': 'UchitelII/1.0', **({'Range': f'bytes={have}-'} if have else {})})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:
                    log('Сервер не поддерживает докачку — начинаю заново'); have = 0
                total = int(r.headers.get('Content-Length') or 0) + have
                if total and free_gb(os.path.dirname(dest)) * 1e9 < (total - have) * 1.05 + 1e9:
                    raise SystemExit(f'Мало места на диске: нужно {human(total - have)}, свободно {free_gb(os.path.dirname(dest)):.1f} ГБ')
                log(f'Скачиваю {url} ({human(total) if total else "размер неизвестен"}){", продолжаю с " + human(have) if have else ""}')
                with open(part, 'ab' if have else 'wb') as f:
                    done, t0, last = have, time.time(), 0
                    while True:
                        b = r.read(1 << 20)
                        if not b: break
                        f.write(b); done += len(b)
                        if time.time() - last > 1:
                            last = time.time(); sp = (done - have) / max(1e-3, last - t0)
                            emit(progress=done / total if total else 0, text=f'{human(done)} из {human(total) if total else "?"}, {human(sp)}/с')
            if total and os.path.getsize(part) < total:
                raise IOError('файл скачался не полностью')
            break
        except SystemExit:
            raise
        except Exception as e:
            log(f'Ошибка скачивания (попытка {a} из {attempts}): {e}')
            if a == attempts:
                raise SystemExit('Не удалось скачать. Файл недокачан сохранён, можно продолжить позже той же кнопкой.')
            time.sleep(3 * a)
    if sha256:
        log('Проверяю контрольную сумму SHA-256…')
        h = hashlib.sha256()
        with open(part, 'rb') as f:
            for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
        if h.hexdigest().lower() != sha256.strip().lower():
            os.remove(part); raise SystemExit('Контрольная сумма не совпала — файл испорчен и удалён.')
        log('Контрольная сумма совпала.')
    os.replace(part, dest)
    return dest

def t_download(p):
    url = p['url'].strip()
    name = p.get('name') or os.path.basename(urllib.parse.urlparse(url).path) or 'download.bin'
    name = re.sub(r'[^A-Za-z0-9._-]', '_', name)
    dest = http_download(url, os.path.join(DIRS['downloads'], name), p.get('sha256'))
    if zipfile.is_zipfile(dest) and p.get('extract', True):
        out = dest + '_files'; log('Распаковываю архив…')
        with zipfile.ZipFile(dest) as z:
            z.extractall(out)
        dest = out
    elif re.search(r'\.(tar(\.gz|\.xz|\.bz2)?|tgz)$', dest) and p.get('extract', True):
        import tarfile; out = dest + '_files'; log('Распаковываю архив…')
        with tarfile.open(dest) as t: t.extractall(out, filter='data')
        dest = out
    emit(result={'path': dest}); log(f'Готово: {dest}')

def t_hfdata(p):
    from datasets import load_dataset
    rid, split, limit = p['repo'].strip(), p.get('split') or 'train', int(p.get('limit') or 0)
    log(f'Открываю набор {rid} (потоково, без скачивания целиком)…')
    ds = load_dataset(rid, p.get('config') or None, split=split, streaming=True)
    out = os.path.join(DIRS['downloads'], re.sub(r'[^A-Za-z0-9._-]', '_', rid) + f'_{split}.jsonl')
    n = 0
    with open(out, 'w', encoding='utf-8') as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + '\n'); n += 1
            if n % 500 == 0:
                emit(progress=n / limit if limit else 0, text=f'{n} записей')
                if free_gb() < 2: log('На диске меньше 2 ГБ — останавливаюсь.'); break
            if limit and n >= limit: break
    log(f'Сохранено {n} записей в {out}'); emit(result={'path': out, 'rows': n})

# ---------- подготовка данных ----------
TEXT_EXT = {'.txt', '.md', '.markdown', '.rst', '.html', '.htm', '.py', '.js', '.cs', '.pon', '.log'}
def read_text_file(path):
    raw = open(path, 'rb').read()
    return decode(raw)

def extract(path):
    """Возвращает список ('text', str) или ('chat', [messages])."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in TEXT_EXT:
            t = read_text_file(path)
            if ext in ('.html', '.htm'): t = re.sub(r'<script.*?</script>|<style.*?</style>|<[^>]+>', ' ', t, flags=re.S)
            return [('text', t)]
        if ext == '.pdf':
            from pypdf import PdfReader
            return [('text', '\n'.join((pg.extract_text() or '') for pg in PdfReader(path).pages))]
        if ext == '.docx':
            import docx
            return [('text', '\n'.join(p.text for p in docx.Document(path).paragraphs))]
        if ext in ('.json', '.jsonl'):
            rows = []
            txt = read_text_file(path)
            if ext == '.jsonl' or not txt.lstrip().startswith(('[', '{')):
                for l in txt.splitlines():
                    l = l.strip()
                    if l:
                        try: rows.append(json.loads(l))
                        except Exception: pass
            else:
                d = json.loads(txt); rows = d if isinstance(d, list) else d.get('data', [d]) if isinstance(d, dict) else []
            return [r for r in (row_to_item(x) for x in rows) if r]
        if ext in ('.csv', '.tsv'):
            txt = read_text_file(path)
            rd = csv.DictReader(io.StringIO(txt), delimiter='\t' if ext == '.tsv' else ',')
            return [r for r in (row_to_item(x) for x in rd) if r]
        if ext == '.zip':
            out = os.path.join(DIRS['tmp'], 'unz_' + hashlib.md5(path.encode()).hexdigest()[:8])
            with zipfile.ZipFile(path) as z: z.extractall(out)
            return [x for f in walk(out) for x in extract(f)]
    except Exception as e:
        log(f'Не удалось прочитать {os.path.basename(path)}: {e}')
    return []

PAIRS = [('instruction', 'output'), ('prompt', 'response'), ('question', 'answer'), ('input', 'output'), ('prompt', 'completion'), ('query', 'response'), ('вопрос', 'ответ')]
def row_to_item(r):
    if not isinstance(r, dict):
        return ('text', str(r)) if r else None
    low = {str(k).lower(): v for k, v in r.items()}
    for key in ('messages', 'conversations', 'conversation'):
        m = low.get(key)
        if isinstance(m, list) and m and isinstance(m[0], dict):
            msgs = []
            for x in m:
                role = x.get('role') or x.get('from') or 'user'; content = x.get('content') or x.get('value') or ''
                role = {'human': 'user', 'gpt': 'assistant', 'bot': 'assistant', 'model': 'assistant'}.get(role, role)
                msgs.append({'role': role, 'content': str(content)})
            return ('chat', msgs)
    for a, b in PAIRS:
        if low.get(a) and low.get(b):
            q = str(low[a]); 
            if a == 'instruction' and low.get('input'): q += '\n\n' + str(low['input'])
            return ('chat', [{'role': 'user', 'content': q}, {'role': 'assistant', 'content': str(low[b])}])
    for k in ('text', 'content', 'body', 'текст'):
        if low.get(k): return ('text', str(low[k]))
    vals = [str(v) for v in r.values() if isinstance(v, str)]
    return ('text', '\n'.join(vals)) if vals else None

def walk(p):
    if os.path.isfile(p): yield p; return
    for root, _, files in os.walk(p):
        for f in sorted(files): yield os.path.join(root, f)

def clean(t):
    t = unicodedata.normalize('NFC', t).replace('\r\n', '\n').replace('\u00a0', ' ')
    t = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd]', '', t)
    t = re.sub(r'[ \t]+', ' ', t); t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()

def chunks(t, size, overlap):
    paras, cur, out = re.split(r'\n\s*\n', t), '', []
    for para in paras:
        while len(para) > size:
            out.append((cur + '\n\n' + para[:size]).strip() if cur else para[:size]); cur = ''; para = para[size - overlap:]
        if len(cur) + len(para) + 2 > size and cur:
            out.append(cur); cur = cur[-overlap:] + '\n\n' + para if overlap else para
        else:
            cur = (cur + '\n\n' + para) if cur else para
    if cur.strip(): out.append(cur)
    return out

def t_prepare(p):
    sources = [s for s in p['sources'] if s.strip()]
    size, overlap, minlen = int(p.get('chunk', 1500)), int(p.get('overlap', 150)), int(p.get('min_len', 40))
    mode = p.get('mode', 'auto')  # auto | text | chat
    name = re.sub(r'[^A-Za-z0-9._-]', '_', p.get('name') or 'dataset')
    out_dir = os.path.join(DIRS['datasets'], name); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'train.jsonl')
    files = [f for s in sources for f in walk(s.strip().strip('"'))]
    log(f'Найдено файлов: {len(files)}')
    seen, stats = set(), {'files': len(files), 'texts': 0, 'chats': 0, 'dupes': 0, 'short': 0}
    with open(out, 'w', encoding='utf-8') as f:
        for i, path in enumerate(files):
            emit(progress=i / max(1, len(files)), text=os.path.basename(path))
            for kind, val in extract(path):
                if kind == 'text':
                    pieces = chunks(clean(val), size, overlap)
                    if mode == 'chat':
                        log('Обычный текст нельзя превратить в диалог без вопросов — сохраняю как текст.')
                    items = [{'text': c} for c in pieces]
                else:
                    msgs = [{'role': m['role'], 'content': clean(m['content'])} for m in val]
                    items = [{'messages': msgs}] if mode != 'text' else [{'text': '\n\n'.join(m['content'] for m in msgs)}]
                for it in items:
                    s = it.get('text') or ' '.join(m['content'] for m in it['messages'])
                    if len(s) < minlen: stats['short'] += 1; continue
                    h = hashlib.sha1(re.sub(r'\W+', '', s.lower()).encode()).hexdigest()
                    if h in seen: stats['dupes'] += 1; continue
                    seen.add(h); f.write(json.dumps(it, ensure_ascii=False) + '\n')
                    stats['texts' if 'text' in it else 'chats'] += 1
    preview = [json.loads(l) for _, l in zip(range(5), open(out, encoding='utf-8'))]
    json.dump({'stats': stats, 'sources': sources, 'chunk': size}, open(os.path.join(out_dir, 'info.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    log(f"Готово: примеров текста {stats['texts']}, диалогов {stats['chats']}, повторов убрано {stats['dupes']}, коротких отброшено {stats['short']}")
    emit(result={'path': out, 'stats': stats, 'preview': preview})

# ---------- llama.cpp ----------
def gh_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'UchitelII'}), timeout=60) as r:
        return json.load(r)

def t_install_llama(p):
    d = os.path.join(DIRS['tools'], 'llama.cpp'); os.makedirs(d, exist_ok=True)
    base = 'https://github.com/ggml-org/llama.cpp/releases'
    try:   # без API GitHub (у него лимит запросов): тег ночной сборки лежит файлом в последнем релизе
        with urllib.request.urlopen(urllib.request.Request(base + '/latest/download/nightly-tag.txt', headers={'User-Agent': 'UchitelII'}), timeout=60) as r:
            tag = r.read().decode().strip()
    except Exception as e:
        raise SystemExit(f'Не удалось узнать версию llama.cpp на GitHub: {e}')
    log(f'Версия llama.cpp: {tag}')
    an = f'llama-{tag}-bin-win-cpu-x64.zip' if os.name == 'nt' else f'llama-{tag}-bin-ubuntu-x64.tar.gz'
    z = http_download(f'{base}/download/{tag}/{an}', os.path.join(DIRS['downloads'], an))
    if an.endswith('.zip'):
        with zipfile.ZipFile(z) as zz: zz.extractall(os.path.join(d, 'bin'))
    else:
        import tarfile
        with tarfile.open(z) as t: t.extractall(os.path.join(d, 'bin'), filter='tar')
    if os.name != 'nt':
        for f in glob.glob(os.path.join(d, 'bin', '**', 'llama-*'), recursive=True): os.chmod(f, 0o755)
    src = http_download(f'https://codeload.github.com/ggml-org/llama.cpp/zip/refs/tags/{tag}', os.path.join(DIRS['downloads'], f'llama.cpp-src-{tag}.zip'))
    with zipfile.ZipFile(src) as zz:
        for m in zz.namelist():   # нужны только конвертер и gguf-py
            if re.search(r'/(convert_hf_to_gguf\.py|convert_lora_to_gguf\.py|gguf-py/.*|conversion/.*)$', m):
                zz.extract(m, os.path.join(d, 'src'))
    log('llama.cpp установлен.'); emit(result={'tag': tag})

# ---------- результат ----------
def t_merge(p):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    run = p['run']; cfg = json.load(open(os.path.join(run, 'config.json'), encoding='utf-8'))
    out = os.path.join(run, 'merged')
    adapter = os.path.join(run, 'adapter')
    if cfg['method'] == 'full':
        log('Полное обучение: веса уже целиком в папке full.'); emit(result={'path': os.path.join(run, 'full')}); return
    log('Загружаю базовую модель в полной точности (для слияния нужна оперативка ≈ 2× размера модели)…')
    base = AutoModelForCausalLM.from_pretrained(cfg['model'], dtype=torch.float16 if torch.cuda.is_available() else torch.float32, low_cpu_mem_usage=True)
    log('Вливаю LoRA в веса…'); emit(progress=0.5, text='слияние')
    m = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    m.save_pretrained(out, safe_serialization=True, max_shard_size='2GB')
    AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
    log(f'Объединённая модель сохранена: {out}'); emit(result={'path': out})

def t_gguf(p):
    from envcheck import llama_tools
    q, conv = llama_tools()
    if not conv: raise SystemExit('Нет llama.cpp. Откройте «Окружение» и нажмите «Установить» у llama.cpp.')
    src, qt = p['model_dir'], p.get('quant', 'Q4_K_M')
    name = re.sub(r'[^A-Za-z0-9._-]', '_', p.get('name') or os.path.basename(os.path.dirname(src.rstrip('/\\'))))
    os.makedirs(DIRS['exports'], exist_ok=True)
    f16 = os.path.join(DIRS['exports'], f'{name}-f16.gguf')
    env = child_env(); env['PYTHONPATH'] = os.path.join(os.path.dirname(conv), 'gguf-py') + os.pathsep + env.get('PYTHONPATH', '')
    def sh(cmd, what):
        log(what)
        pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        for line in pr.stdout:
            l = decode(line).rstrip()
            if l: emit(log='  ' + l[-300:])
        if pr.wait() != 0: raise SystemExit(f'{what} — ошибка (код {pr.returncode}). Подробности выше.')
    # встроенный Python не читает PYTHONPATH — подключаем gguf-py и conversion из llama.cpp явно
    boot = ('import sys,runpy;c=sys.argv[1];d=__import__("os").path.dirname(c);sys.path[:0]=[d,d+"/gguf-py"];'
            'sys.argv=sys.argv[1:];runpy.run_path(c,run_name="__main__")')
    sh([sys.executable, '-c', boot, conv, src, '--outfile', f16, '--outtype', 'f16'], 'Конвертирую в GGUF (f16)…')
    emit(progress=0.6, text='сжатие')
    final = f16
    if qt and qt.upper() != 'F16':
        if not q: raise SystemExit('Нет llama-quantize — переустановите llama.cpp в «Окружении».')
        final = os.path.join(DIRS['exports'], f'{name}-{qt}.gguf')
        sh([q, f16, final, qt], f'Сжимаю до {qt}…'); os.remove(f16)
    log(f'Готово: {final} ({human(os.path.getsize(final))}). Этот файл можно открыть в приложении «мини-ИИ» на телефоне.')
    emit(result={'path': final, 'size': os.path.getsize(final)})

def t_pip(p):
    for cmd in p['cmds']:
        log('> ' + ' '.join(cmd[3:]))
        for attempt in (1, 2):   # одна повторная попытка, без бесконечных циклов
            pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=child_env(), creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            for line in pr.stdout:
                l = decode(line).rstrip()
                if l: emit(log='  ' + l[-300:])
            if pr.wait() == 0: break
            if attempt == 2: raise SystemExit('Установка не удалась (подробности выше). Проверьте интернет/антивирус и нажмите ещё раз.')
            log('Ошибка, пробую ещё раз через 5 секунд…'); time.sleep(5)
    log('Установлено. Нажмите «Проверить снова».'); emit(result={})

if __name__ == '__main__':
    kind, pf = sys.argv[1], sys.argv[2]
    try:
        globals()['t_' + kind](json.load(open(pf, encoding='utf-8')))
    except SystemExit as e:
        if e.code not in (None, 0): emit(error=str(e.code)); sys.exit(1)
    except Exception as e:
        import traceback; emit(log=traceback.format_exc()[-1500:]); emit(error=f'{type(e).__name__}: {e}'); sys.exit(1)
