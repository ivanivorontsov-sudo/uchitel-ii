# Локальный сервер «Учитель ИИ»: API + интерфейс. Слушает только 127.0.0.1.
import os, sys, json, time, uuid, threading, subprocess, tempfile, glob, shutil, mimetypes
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DIRS, HOME, child_env, decode, free_gb
import hardware, envcheck

HERE = os.path.dirname(os.path.abspath(__file__))
UI = os.environ.get('UCHITEL_UI') or os.path.join(HERE, '..', 'app', 'ui')
JOBS = {}

class Job:
    def __init__(s, kind, cmd, title, run=None):
        s.id = uuid.uuid4().hex[:8]; s.kind = kind; s.title = title; s.run = run
        s.logs = []; s.points = []; s.state = 'running'; s.progress = 0; s.text = ''; s.result = None; s.error = None; s.t0 = time.time(); s.last = {}
        s.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=child_env(), cwd=HERE,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        mf = os.path.join(run, 'metrics.jsonl') if run else ''
        if mf and os.path.exists(mf):   # продолжение: показываем и прошлую часть графика
            for l in open(mf, encoding='utf-8'):
                try: s.points.append(json.loads(l))
                except Exception: pass
            s.skip_points = len(s.points)
        JOBS[s.id] = s; threading.Thread(target=s.read, daemon=True).start()
    def add(s, t):
        s.logs.append(time.strftime('%H:%M:%S ') + t); s.logs = s.logs[-3000:]
        with open(os.path.join(DIRS['logs'], 'uchitel.log'), 'a', encoding='utf-8') as f: f.write(f'[{s.kind}] {t}\n')
    def read(s):
        for raw in s.p.stdout:
          for line in decode(raw).split('\r'):
            line = line.strip()
            if not line: continue
            try: m = json.loads(line)
            except Exception:
                if 'it/s]' in line or 's/it]' in line or line.startswith(('{\'loss', '{\'train')): continue
                s.add(line[-400:]); continue
            if 'log' in m: s.add(m['log'])
            if 'progress' in m: s.progress = m['progress']; s.text = m.get('text', s.text)
            if m.get('loss') is not None or m.get('eval_loss') is not None:
                p = {'step': m.get('step'), 'loss': m.get('loss'), 'eval_loss': m.get('eval_loss')}
                if p not in s.points: s.points.append(p)
            if 'step' in m: s.last = m
            if 'result' in m: s.result = m['result']
            if m.get('paused'): s.state = 'paused'
            if 'error' in m: s.error = m['error']; s.add('ОШИБКА: ' + m['error'])
        code = s.p.wait()
        if getattr(s, 'killed', False): s.state = 'stopped'
        elif s.state != 'paused':
            s.state = 'done' if code == 0 and not s.error else 'error'
            if s.state == 'error' and not s.error: s.error = f'процесс завершился с кодом {code}'
        s.add({'done': 'Готово.', 'error': 'Задача завершилась с ошибкой.', 'stopped': 'Остановлено.', 'paused': 'На паузе.'}.get(s.state, s.state))
    def info(s, full=False):
        d = dict(id=s.id, kind=s.kind, title=s.title, state=s.state, progress=s.progress, text=s.text, result=s.result, error=s.error,
                 run=s.run, last=s.last, elapsed=int(time.time() - s.t0))
        if full: d.update(logs=s.logs[-400:], points=s.points)
        return d

def task(kind, params, title):
    pf = os.path.join(DIRS['tmp'], f'{kind}_{uuid.uuid4().hex[:6]}.json')
    json.dump(params, open(pf, 'w', encoding='utf-8'), ensure_ascii=False)
    return Job(kind, [sys.executable, '-u', os.path.join(HERE, 'tasks.py'), kind, pf], title)

def start_train(cfg):
    run = cfg.get('run') or os.path.join(DIRS['runs'], time.strftime('%Y%m%d-%H%M%S'))
    os.makedirs(run, exist_ok=True)
    try: os.remove(os.path.join(run, 'PAUSE'))
    except OSError: pass
    cfg['run'] = run; json.dump(cfg, open(os.path.join(run, 'config.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    script = os.path.join(HERE, 'train.py')
    n = int(cfg.get('gpus', 1) or 1)
    cmd = [sys.executable, '-m', 'accelerate.commands.launch', '--num_processes', str(n), script, run] if n > 1 else [sys.executable, '-u', script, run]
    return Job('train', cmd, 'Обучение ' + cfg['model'], run)

# ---------- чат ----------
CHAT = {}
LOCK = threading.Lock()
def chat(model_path, messages, max_new=200, temp=0.7):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    with LOCK:
        if model_path not in CHAT:
            if len(CHAT) >= 2: CHAT.pop(next(iter(CHAT)))
            ad = os.path.join(model_path, 'adapter_config.json')
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'; dt = torch.float16 if dev == 'cuda' else torch.float32
            if os.path.exists(ad):
                from peft import AutoPeftModelForCausalLM
                m = AutoPeftModelForCausalLM.from_pretrained(model_path, dtype=dt).to(dev)
            else:
                m = AutoModelForCausalLM.from_pretrained(model_path, dtype=dt).to(dev)
            CHAT[model_path] = (m.eval(), AutoTokenizer.from_pretrained(model_path))
        m, tok = CHAT[model_path]
        try: ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt', return_dict=True)
        except Exception: ids = tok('\n'.join(x['content'] for x in messages), return_tensors='pt')
        ids = {k: v.to(m.device) for k, v in ids.items()}
        with torch.no_grad():
            out = m.generate(**ids, max_new_tokens=int(max_new), do_sample=temp > 0, temperature=max(temp, 0.01), top_p=0.9, repetition_penalty=1.1, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()

def list_runs():
    res = []
    for r in sorted(glob.glob(os.path.join(DIRS['runs'], '*')), reverse=True):
        try: c = json.load(open(os.path.join(r, 'config.json'), encoding='utf-8'))
        except Exception: continue
        res.append({'run': r, 'model': c.get('model'), 'method': c.get('method'), 'dataset': c.get('dataset'),
                    'adapter': os.path.isdir(os.path.join(r, 'adapter')), 'merged': os.path.isdir(os.path.join(r, 'merged')),
                    'full': os.path.isdir(os.path.join(r, 'full')), 'checkpoints': len(glob.glob(os.path.join(r, 'checkpoints', 'checkpoint-*')))})
    return res

class H(BaseHTTPRequestHandler):
    def log_message(s, *a): pass
    def send(s, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode(); s.send_response(code)
        s.send_header('Content-Type', 'application/json; charset=utf-8'); s.send_header('Content-Length', str(len(b))); s.end_headers(); s.wfile.write(b)
    def do_GET(s):
        u = urlparse(s.path); q = parse_qs(u.query); p = u.path
        try:
            if p == '/api/hw': return s.send(hardware.detect())
            if p == '/api/env': return s.send(envcheck.check())
            if p == '/api/jobs': return s.send([j.info() for j in JOBS.values()])
            if p == '/api/job': return s.send(JOBS[q['id'][0]].info(True))
            if p == '/api/runs': return s.send(list_runs())
            if p == '/api/datasets':
                return s.send([{'path': os.path.join(d, 'train.jsonl'), 'name': os.path.basename(d), 'info': json.load(open(os.path.join(d, 'info.json'), encoding='utf-8')) if os.path.exists(os.path.join(d, 'info.json')) else {}}
                               for d in sorted(glob.glob(os.path.join(DIRS['datasets'], '*'))) if os.path.exists(os.path.join(d, 'train.jsonl'))])
            if p == '/api/exports':
                return s.send([{'path': f, 'size': os.path.getsize(f)} for f in sorted(glob.glob(os.path.join(DIRS['exports'], '*.gguf')))])
            if p == '/api/preview':
                f = q['path'][0]; return s.send([json.loads(l) for _, l in zip(range(int(q.get('n', ['8'])[0])), open(f, encoding='utf-8'))])
            if p == '/api/info': return s.send({'home': HOME, 'dirs': DIRS, 'free_gb': round(free_gb(), 1), 'python': sys.executable})
            # статика интерфейса
            f = os.path.normpath(os.path.join(UI, (p.lstrip('/') or 'index.html')))
            if not f.startswith(os.path.normpath(UI)) or not os.path.isfile(f): f = os.path.join(UI, 'index.html')
            b = open(f, 'rb').read(); s.send_response(200)
            s.send_header('Content-Type', (mimetypes.guess_type(f)[0] or 'text/plain') + '; charset=utf-8'); s.send_header('Content-Length', str(len(b))); s.end_headers(); s.wfile.write(b)
        except Exception as e:
            s.send({'error': f'{type(e).__name__}: {e}'}, 500)
    def do_POST(s):
        p = urlparse(s.path).path
        try:
            d = json.loads(s.rfile.read(int(s.headers.get('Content-Length') or 0)) or b'{}')
            if p == '/api/install':
                what = d['what']
                if what == 'llama.cpp': return s.send(task('install_llama', {}, 'Установка llama.cpp').info())
                if what == 'driver': return s.send({'open': 'https://www.nvidia.com/ru-ru/drivers/'})
                st = envcheck.check(); cmds = envcheck.install_cmds(what, st['gpu'])
                if what == 'all' and not any(i['id'] == 'llama.cpp' and i['ok'] for i in st['items']): task('install_llama', {}, 'Установка llama.cpp')
                if not cmds: return s.send({'msg': 'Всё уже установлено'})
                return s.send(task('pip', {'cmds': cmds}, 'Установка: ' + what).info())
            if p == '/api/download': return s.send(task('download', d, 'Скачивание ' + d['url'][:60]).info())
            if p == '/api/hfdata': return s.send(task('hfdata', d, 'Набор HF ' + d['repo']).info())
            if p == '/api/prepare': return s.send(task('prepare', d, 'Подготовка данных ' + d.get('name', '')).info())
            if p == '/api/train': return s.send(start_train(d).info())
            if p == '/api/pause':
                j = JOBS[d['id']]; open(os.path.join(j.run, 'PAUSE'), 'w').close(); return s.send({'ok': True})
            if p == '/api/resume':
                cfg = json.load(open(os.path.join(d['run'], 'config.json'), encoding='utf-8')); cfg['resume'] = True; return s.send(start_train(cfg).info())
            if p == '/api/stop':
                j = JOBS[d['id']]; j.killed = True
                if os.name == 'nt': subprocess.run(['taskkill', '/PID', str(j.p.pid), '/T', '/F'], capture_output=True, creationflags=0x08000000)
                else: j.p.kill()
                return s.send({'ok': True})
            if p == '/api/merge': return s.send(task('merge', d, 'Слияние LoRA').info())
            if p == '/api/gguf': return s.send(task('gguf', d, 'Экспорт GGUF').info())
            if p == '/api/chat': return s.send({'text': chat(d['model'], d['messages'], d.get('max_new', 200), float(d.get('temp', 0.7)))})
            if p == '/api/plan':   # команда для мощного компьютера / нескольких GPU
                run = d.get('run') or '<папка>'
                return s.send({'cmd': f'accelerate launch --num_processes {d.get("gpus", 2)} train.py {run}'})
            s.send({'error': 'нет такого действия'}, 404)
        except Exception as e:
            s.send({'error': f'{type(e).__name__}: {e}'}, 500)

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    srv = ThreadingHTTPServer(('127.0.0.1', port), H)
    print(json.dumps({'ready': port}), flush=True)
    srv.serve_forever()
