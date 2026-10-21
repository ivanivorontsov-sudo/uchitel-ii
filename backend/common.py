# Общие пути и утилиты «Учитель ИИ». Все рабочие папки — ASCII без пробелов.
import os, sys, json, shutil, platform, subprocess, re

def default_home():
    if os.environ.get('UCHITEL_HOME'):
        return os.environ['UCHITEL_HOME']
    if os.name == 'nt':
        for c in (r'C:\ProgramData\UchitelII', r'C:\UchitelII'):
            try:
                os.makedirs(c, exist_ok=True); t = os.path.join(c, '.w'); open(t, 'w').close(); os.remove(t); return c
            except OSError:
                continue
    return os.path.join(os.path.expanduser('~'), 'uchitel-data')

HOME = default_home()
DIRS = {k: os.path.join(HOME, k) for k in ('hf', 'downloads', 'datasets', 'runs', 'exports', 'tools', 'tmp', 'logs')}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

def child_env():
    e = dict(os.environ)
    e.update(HF_HOME=DIRS['hf'], TMP=DIRS['tmp'], TEMP=DIRS['tmp'], TMPDIR=DIRS['tmp'],
             PYTHONIOENCODING='utf-8', PYTHONUTF8='1', HF_HUB_DISABLE_SYMLINKS_WARNING='1',
             PIP_DISABLE_PIP_VERSION_CHECK='1', TOKENIZERS_PARALLELISM='false')
    return e

def decode(b):
    """Вывод чужих программ: сначала UTF-8, потом cp866 (консоль Windows), потом cp1251."""
    if isinstance(b, str):
        return b
    for enc in ('utf-8', 'cp866', 'cp1251'):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            pass
    return b.decode('utf-8', 'replace')

def run(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, env=child_env(),
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        return r.returncode, decode(r.stdout) + decode(r.stderr)
    except FileNotFoundError:
        return -1, 'не найдено'
    except subprocess.TimeoutExpired:
        return -2, 'превышено время ожидания'

def free_gb(path=HOME):
    return shutil.disk_usage(path).free / 1e9

def human(n):
    for u in ('Б', 'КБ', 'МБ', 'ГБ', 'ТБ'):
        if n < 1024: return f'{n:.1f} {u}'
        n /= 1024
    return f'{n:.1f} ПБ'
