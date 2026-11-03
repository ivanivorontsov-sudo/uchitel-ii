# Определение железа и рекомендации по моделям/методу обучения.
import os, sys, platform, ctypes, shutil, re
from common import run, HOME

def ram_gb():
    try:
        if os.name == 'nt':
            class M(ctypes.Structure):
                _fields_ = [('l', ctypes.c_ulong), ('load', ctypes.c_ulong), ('total', ctypes.c_ulonglong), ('avail', ctypes.c_ulonglong),
                            ('tp', ctypes.c_ulonglong), ('ap', ctypes.c_ulonglong), ('tv', ctypes.c_ulonglong), ('av', ctypes.c_ulonglong), ('x', ctypes.c_ulonglong)]
            m = M(); m.l = ctypes.sizeof(M); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.total / 2**30, m.avail / 2**30
        d = dict(l.split(':', 1) for l in open('/proc/meminfo'))
        return int(d['MemTotal'].split()[0]) / 2**20, int(d['MemAvailable'].split()[0]) / 2**20
    except Exception:
        return 0, 0

def gpus():
    code, out = run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version,compute_cap', '--format=csv,noheader,nounits'])
    res = []
    if code == 0:
        for line in out.strip().splitlines():
            p = [x.strip() for x in line.split(',')]
            if len(p) >= 4:
                res.append({'name': p[0], 'vram_gb': round(float(p[1]) / 1024, 1), 'free_gb': round(float(p[2]) / 1024, 1),
                            'driver': p[3], 'cc': p[4] if len(p) > 4 else ''})
    cuda = ''
    if code == 0:
        _, o2 = run(['nvidia-smi'])
        m = re.search(r'CUDA Version:\s*([\d.]+)', o2); cuda = m.group(1) if m else ''
    return res, cuda

def detect():
    tot, av = ram_gb()
    g, cuda = gpus()
    du = shutil.disk_usage(HOME)
    return {'os': f'{platform.system()} {platform.release()} ({platform.version()})', 'cpu': platform.processor() or platform.machine(),
            'cores': os.cpu_count(), 'ram_gb': round(tot, 1), 'ram_free_gb': round(av, 1), 'gpus': g, 'cuda_driver': cuda,
            'disk_free_gb': round(du.free / 1e9, 1), 'home': HOME, 'recommend': recommend(g, av)}

CATALOG = [  # id, параметров (млрд), описание
    ('Qwen/Qwen2.5-0.5B-Instruct', 0.5, 'Крошечная, понимает русский. Учится даже на процессоре.'),
    ('HuggingFaceTB/SmolLM2-360M-Instruct', 0.36, 'Самая маленькая, для проверки и слабых ПК (английский).'),
    ('Qwen/Qwen2.5-1.5B-Instruct', 1.5, 'Лёгкая, хорошо для телефона.'),
    ('Qwen/Qwen2.5-3B-Instruct', 3, 'Средняя для телефона, заметно умнее.'),
    ('meta-llama/Llama-3.2-3B-Instruct', 3.2, 'Llama 3.2 3B (нужен доступ на Hugging Face).'),
    ('Qwen/Qwen2.5-7B-Instruct', 7.6, 'Мощная 7B, лучший русский среди 7B.'),
    ('mistralai/Mistral-7B-Instruct-v0.3', 7.2, 'Mistral 7B.'),
    ('meta-llama/Llama-3.1-8B-Instruct', 8, 'Llama 3.1 8B (нужен доступ на Hugging Face).'),
    ('Qwen/Qwen2.5-14B-Instruct', 14.7, 'Большая 14B — для 16+ ГБ видеопамяти.'),
    ('Qwen/Qwen2.5-32B-Instruct', 32.5, 'Очень большая 32B — 24+ ГБ видеопамяти.'),
    ('Qwen/Qwen2.5-72B-Instruct', 72.7, 'Огромная 72B — серверные видеокарты или несколько GPU.'),
    ('meta-llama/Llama-3.1-70B-Instruct', 70.6, 'Llama 70B — несколько GPU / 48+ ГБ.'),
]

def need_gb(b, method):
    """Грубая оценка памяти (ГБ) на модель с b млрд параметров."""
    return {'full': b * 16 + 2, 'lora': b * 2.2 + 2, 'qlora': b * 0.7 + 2, 'cpu': b * 5 + 2}[method]

def pick(b, vram, ram):
    # TrainingArguments/Trainer currently use DDP, not model sharding:
    # every GPU must hold a complete copy of the model. Use the least-free GPU,
    # never the sum of VRAM across cards, to avoid impossible recommendations.
    if vram > 0:
        for m in ('full', 'lora', 'qlora'):
            if need_gb(b, m) <= vram * 0.80:
                return m
    # RAM is the currently available memory, not installed physical RAM.
    if b <= 1.6 and ram > 0 and need_gb(b, 'cpu') <= ram * 0.65:
        return 'cpu'
    return None

METHODS = {'full': 'Полное обучение (меняются все веса)', 'lora': 'LoRA (быстрые «насадки» к весам)',
           'qlora': 'QLoRA 4 бита (экономит видеопамять)', 'cpu': 'На процессоре (медленно, только маленькие)'}

def recommend(g, ram):
    # DDP replicates the model on every GPU. The weakest/free-memory-lowest
    # device determines whether a configuration fits.
    per_gpu = min((x.get('free_gb', x.get('vram_gb', 0)) for x in g), default=0)
    total = sum(x.get('vram_gb', 0) for x in g)
    out = []
    for mid, b, desc in CATALOG:
        m = pick(b, per_gpu, ram)
        out.append({'id': mid, 'params_b': b, 'desc': desc, 'method': m, 'method_ru': METHODS.get(m, 'Не хватит памяти на этом ПК — нужен мощнее'),
                    'need': {k: round(need_gb(b, k), 1) for k in METHODS}})
    return {'vram_total_gb': total, 'vram_per_gpu_free_gb': round(per_gpu, 1), 'models': out, 'multi_gpu': len(g) > 1}
