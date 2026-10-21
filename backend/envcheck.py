# Проверка окружения: что есть, чего не хватает, как поставить.
import importlib.util, sys, os, json, glob
from common import run, DIRS
from hardware import gpus

PKGS = [  # модуль, pip-имя, зачем, обязателен
    ('torch', 'torch', 'PyTorch — движок нейросетей', True),
    ('transformers', 'transformers', 'Transformers — загрузка и обучение моделей', True),
    ('peft', 'peft', 'PEFT — LoRA / QLoRA', True),
    ('datasets', 'datasets', 'Datasets — наборы данных Hugging Face', True),
    ('accelerate', 'accelerate', 'Accelerate — обучение на нескольких GPU', True),
    ('safetensors', 'safetensors', 'Safetensors — формат весов', True),
    ('sentencepiece', 'sentencepiece', 'SentencePiece — токенизаторы', True),
    ('gguf', 'gguf', 'GGUF — экспорт для телефона', True),
    ('pypdf', 'pypdf', 'Чтение PDF', True),
    ('docx', 'python-docx', 'Чтение DOCX', True),
    ('google.protobuf', 'protobuf', 'Protobuf — нужен некоторым токенизаторам', True),
    ('trl', 'trl', 'TRL — дополнительные методы обучения', False),
    ('bitsandbytes', 'bitsandbytes', 'bitsandbytes — 4-битный QLoRA (только с NVIDIA)', False),
]

def has(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False

def llama_tools():
    d = os.path.join(DIRS['tools'], 'llama.cpp')
    q = glob.glob(os.path.join(d, '**', 'llama-quantize*'), recursive=True)
    c = glob.glob(os.path.join(d, '**', 'convert_hf_to_gguf.py'), recursive=True)
    return (q[0] if q else os.environ.get('UCHITEL_QUANTIZE', '')), (c[0] if c else os.environ.get('UCHITEL_CONVERT', ''))

def check():
    items = [{'id': 'python', 'name': f'Python {sys.version.split()[0]}', 'ok': sys.version_info >= (3, 10), 'required': True,
              'hint': sys.executable}]
    g, cuda = gpus()
    items.append({'id': 'driver', 'name': 'Драйвер NVIDIA', 'ok': bool(g), 'required': False,
                  'hint': (f"{g[0]['name']}, драйвер {g[0]['driver']}, CUDA {cuda}" if g else 'Видеокарта NVIDIA не найдена: обучение пойдёт на процессоре. Драйвер: nvidia.com/drivers'),
                  'install': None if g else 'driver'})
    for mod, pipn, why, req in PKGS:
        if mod == 'bitsandbytes' and not g:
            req = False
        items.append({'id': pipn, 'name': why, 'ok': has(mod), 'required': req, 'install': pipn})
    if g:
        try: cv = float(cuda)
        except ValueError: cv = 0
        if cv and cv < 12.6:
            items.append({'id': 'driver-old', 'name': 'Драйвер NVIDIA устарел', 'ok': False, 'required': True,
                          'hint': f'Драйвер поддерживает CUDA {cuda}, а нужна 12.6 или новее. Обновите драйвер.', 'install': 'driver'})
    if has('torch'):
        try:
            import torch
            ok = torch.cuda.is_available()
            items.append({'id': 'torch-cuda', 'name': 'PyTorch видит видеокарту (CUDA)', 'ok': ok, 'required': bool(g),
                          'hint': f'torch {torch.__version__}' + (f', {torch.cuda.get_device_name(0)}' if ok else (' — стоит версия без CUDA, нажмите «Установить»' if g else '')),
                          'install': 'torch' if g and not ok else None})
        except Exception as e:
            items.append({'id': 'torch-cuda', 'name': 'PyTorch', 'ok': False, 'required': True, 'hint': str(e)[:200], 'install': 'torch'})
    q, c = llama_tools()
    items.append({'id': 'llama.cpp', 'name': 'llama.cpp — конвертер и сжатие в GGUF', 'ok': bool(q and c), 'required': True,
                  'hint': q or 'не установлен', 'install': 'llama.cpp'})
    return {'items': items, 'missing': [i for i in items if not i['ok'] and i['required']], 'gpu': bool(g)}

TORCH = '2.14.1'
PINS = {'transformers': '5.20.2', 'peft': '0.21.2', 'datasets': '5.1.0', 'accelerate': '1.15.0', 'safetensors': '0.8.0', 'sentencepiece': '0.2.2',
        'gguf': '0.19.0', 'pypdf': '6.21.1', 'python-docx': '1.2.0', 'protobuf': '7.36.2', 'trl': '1.16.0'}  # проверенные вместе версии

def torch_index(gpu):
    if not gpu: return 'https://download.pytorch.org/whl/cpu'
    _, cuda = gpus()
    try: v = float(cuda)
    except ValueError: v = 0
    # cu130 нужен для RTX 50xx и требует драйвер с CUDA 13; cu126 — для драйверов с CUDA 12.6+
    return 'https://download.pytorch.org/whl/' + ('cu130' if v >= 13.0 else 'cu126')

def pin(p): return f'{p}=={PINS[p]}' if p in PINS else p

def install_cmds(what, gpu):
    """Команды для установки; выполняются задачей с журналом (одна попытка + 1 повтор, без бесконечных циклов)."""
    py = [sys.executable, '-m', 'pip', 'install', '--no-warn-script-location', '--progress-bar', 'off']
    if what == 'torch':
        return [py + ['--upgrade', '--force-reinstall', 'torch==' + TORCH, '--index-url', torch_index(gpu)]]
    if what == 'all':
        cmds = []
        st = check()
        if any(i['id'] in ('torch', 'torch-cuda') and not i['ok'] for i in st['items']):
            cmds += install_cmds('torch', gpu)
        rest = [p for m, p, _, req in PKGS if p != 'torch' and not has(m) and (req or (p == 'bitsandbytes' and gpu))]
        if rest:
            cmds.append(py + [pin(p) for p in rest])
        return cmds
    return [py + [pin(what)]]
