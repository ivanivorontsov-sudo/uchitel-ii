# Быстрое полное обучение / предобучение в стиле nanoGPT (без HF Trainer).
# Запуск: python pretrain.py <папка запуска>   ·   несколько GPU/процессов: torchrun --nproc_per_node N pretrain.py <папка>
# В папке запуска config.json: model (папка HF), data (папка с train.bin/val.bin/meta.json), шаги, батч, lr и т.д.
import os, sys, json, time, math, glob, contextlib
RUN = sys.argv[1]; CFG = json.load(open(os.path.join(RUN, 'config.json'), encoding='utf-8'))
import numpy as np, torch, torch.nn.functional as F
WORLD = int(os.environ.get('WORLD_SIZE', '1')); RANK = int(os.environ.get('RANK', '0')); LOCAL = int(os.environ.get('LOCAL_RANK', '0'))
R0 = RANK == 0
def emit(**k):
    if R0: print(json.dumps(k, ensure_ascii=False), flush=True)
def log(t): emit(log=t)
cuda = torch.cuda.is_available() and CFG.get('device', 'auto') != 'cpu'
if WORLD > 1:
    import torch.distributed as dist
    dist.init_process_group('nccl' if cuda and os.name != 'nt' else 'gloo')
dev = torch.device(f'cuda:{LOCAL}' if cuda else 'cpu')
if cuda: torch.cuda.set_device(dev); torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
bf16 = cuda and torch.cuda.is_bf16_supported(); fp16 = cuda and not bf16
amp = torch.autocast('cuda', dtype=torch.bfloat16 if bf16 else torch.float16) if cuda else contextlib.nullcontext()
torch.manual_seed(int(CFG.get('seed', 42)) + RANK)

# ---------- данные ----------
D = os.path.abspath(CFG.get('data', ''))
if not D or not os.path.isdir(D):
    raise SystemExit(f'Папка токенизированных данных не найдена: {D!r}. Сначала подготовьте набор данных.')
meta_path = os.path.join(D, 'meta.json')
if not os.path.isfile(meta_path):
    raise SystemExit(f'Не найден {meta_path}. Пересоздайте токенизированный набор.')
with open(meta_path, encoding='utf-8') as mf:
    meta = json.load(mf)
if meta.get('dtype') not in ('uint16', 'uint32'):
    raise SystemExit(f"Неизвестный тип токенов в meta.json: {meta.get('dtype')!r}.")
dt = np.uint16 if meta['dtype'] == 'uint16' else np.uint32
train_path = os.path.join(D, 'train.bin')
val_path = os.path.join(D, 'val.bin')
if not os.path.isfile(train_path) or os.path.getsize(train_path) < np.dtype(dt).itemsize:
    raise SystemExit('train.bin отсутствует или пуст. Вернитесь во вкладку «Данные» и подготовьте набор заново.')
tr = np.memmap(train_path, dtype=dt, mode='r')
T = max(8, int(CFG.get('max_len', 512)))
if len(tr) < T + 2:
    raise SystemExit(f'В обучающем наборе только {len(tr)} токенов, а для контекста {T} нужно минимум {T + 2}. Уменьшите длину контекста или добавьте данные.')
# Пустой/слишком короткий val.bin не должен обрушивать обучение: используем train.bin для оценки.
if os.path.isfile(val_path) and os.path.getsize(val_path) >= (T + 2) * np.dtype(dt).itemsize:
    va = np.memmap(val_path, dtype=dt, mode='r')
else:
    va = tr
    if os.path.exists(val_path):
        log('Проверочная часть пуста или короче контекста; для оценки временно использую обучающие данные.')
def batch(src, B, gen):
    max_start = len(src) - T - 1
    if max_start < 0:
        raise ValueError(f'Недостаточно токенов ({len(src)}) для окна {T + 1}.')
    # randint не включает верхнюю границу; +1 позволяет брать и последнее полное окно.
    ix = torch.randint(0, max_start + 1, (B,), generator=gen)
    x = torch.from_numpy(np.stack([src[i:i + T + 1].astype(np.int64) for i in ix.tolist()]))
    if cuda: x = x.pin_memory().to(dev, non_blocking=True)
    return x

# ---------- модель ----------
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
mc = AutoConfig.from_pretrained(CFG['model'])
if not cuda:
    # Full pretraining keeps FP32 weights, gradients and Adam states in RAM.
    # Estimate model size on the meta device before allocating real tensors.
    log('Проверяю, поместится ли полное обучение в оперативную память…')
    try:
        import psutil
        with torch.device('meta'):
            probe = AutoModelForCausalLM.from_config(mc)
        params = sum(p.numel() for p in probe.parameters())
        del probe
        available = psutil.virtual_memory().available
        required = int(params * 20)  # weights + gradients + Adam states + 25% headroom
        log(f'Оценка: {params/1e6:.1f} млн параметров; свободно {available/2**30:.1f} ГБ; нужно около {required/2**30:.1f} ГБ.')
        if required > available * 0.80:
            raise SystemExit('Полное предобучение не запускаю: недостаточно свободной ОЗУ. Выберите меньшую модель или используйте LoRA во вкладке «Обучение».')
    except SystemExit:
        raise
    except Exception as e:
        raise SystemExit(f'Не удалось безопасно оценить память модели ({type(e).__name__}: {e}). Полное предобучение на CPU отменено; выберите меньшую модель или LoRA.')
log(f"Загружаю модель {mc.model_type}…")
UM = os.path.join(CFG['model'], 'uchitel.json'); LAZY = os.path.exists(UM) and json.load(open(UM, encoding='utf-8')).get('lazy_init') and not os.path.exists(os.path.join(RUN, 'ckpt', 'state.pt'))
if os.path.exists(UM) and json.load(open(UM, encoding='utf-8')).get('lazy_init'):
    mc._attn_implementation = 'sdpa'
    if WORLD > 1 and CFG.get('strategy') == 'fsdp':
        with torch.device('meta'): model = AutoModelForCausalLM.from_config(mc)   # без выделения памяти
    else:
        torch.manual_seed(int(CFG.get('seed', 42))); model = AutoModelForCausalLM.from_config(mc)
    log('Модель создаётся из описания (огромный размер, веса случайные).')
else:
    model = AutoModelForCausalLM.from_pretrained(CFG['model'], dtype=torch.float32, attn_implementation='sdpa')
moe = hasattr(mc, 'num_experts_per_tok')
if moe: model.config.output_router_logits = True
NP = sum(p.numel() for p in model.parameters())
if CFG.get('grad_ckpt'): model.gradient_checkpointing_enable(); model.config.use_cache = False
model.config.use_cache = False
if not any(p.is_meta for p in model.parameters()): model.to(dev)
strategy = CFG.get('strategy', 'ddp')
if WORLD > 1:
    if strategy == 'fsdp':
        from torch.distributed.fsdp import fully_shard, CPUOffloadPolicy   # FSDP2: шардирование весов, градиентов и Adam
        kw = {'offload_policy': CPUOffloadPolicy()} if CFG.get('offload') else {}
        for layer in model.model.layers: fully_shard(layer, **kw)
        fully_shard(model, **kw)
        if any(p.is_meta for p in model.parameters()):   # веса прямо на своих устройствах, по частям
            torch.manual_seed(int(CFG.get('seed', 42)))
            model.to_empty(device=dev)
            with torch.no_grad():
                for mod in model.modules():
                    if hasattr(mod, 'inv_freq'):
                        f = 1.0 / (mod.config.rope_parameters.get('rope_theta', 10000.0) ** (torch.arange(0, mod.inv_freq.numel() * 2, 2, dtype=torch.float32, device=dev) / (mod.inv_freq.numel() * 2)))
                        mod.inv_freq.copy_(f)
                        if hasattr(mod, 'original_inv_freq'): mod.original_inv_freq = f.clone()
                    else:
                        for n, p_ in mod.named_parameters(recurse=False):
                            if 'norm' in type(mod).__name__.lower() or n == 'bias': p_.fill_(1.0 if 'norm' in type(mod).__name__.lower() else 0.0)
                            else: p_.normal_(0, 0.02)
            model.tie_weights() if hasattr(model, 'tie_weights') else None
    else:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[LOCAL] if cuda else None)
raw = model.module if hasattr(model, 'module') else model
FSDP_ON = WORLD > 1 and strategy == 'fsdp'
tok = AutoTokenizer.from_pretrained(CFG['model'])
log(f"Параметров: {NP/1e6:.1f} млн · устройство: {torch.cuda.get_device_name(dev) if cuda else 'процессор'} · процессов: {WORLD} · {'bf16' if bf16 else 'fp16' if fp16 else 'fp32'}")

# ---------- оптимизатор ----------
dec = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2 and 'embed' not in n]
nodec = [p for n, p in model.named_parameters() if p.requires_grad and not (p.dim() >= 2 and 'embed' not in n)]
groups = [{'params': dec, 'weight_decay': float(CFG.get('weight_decay', 0.1))}, {'params': nodec, 'weight_decay': 0.0}]
LR = float(CFG.get('lr', 6e-4))
if CFG.get('adam8') and cuda:
    import bitsandbytes as bnb; opt = bnb.optim.AdamW8bit(groups, lr=LR, betas=(0.9, 0.95)); log('Оптимизатор: 8-битный AdamW')
else:
    opt = torch.optim.AdamW(groups, lr=LR, betas=(0.9, 0.95), fused=cuda)
scaler = torch.amp.GradScaler('cuda', enabled=fp16)

def try_compile(m):
    if not CFG.get('compile', 'auto') or not cuda or WORLD > 1: return m, False
    try:
        import triton  # noqa — на Windows нужен triton-windows
        mm = torch.compile(m)
        with amp: mm(torch.zeros(1, 8, dtype=torch.long, device=dev)).logits.sum().backward()
        model.zero_grad(set_to_none=True); return mm, True
    except Exception as e:
        log(f'torch.compile недоступен ({type(e).__name__}) — работаю без него.'); return m, False
fwd, compiled = try_compile(model)
if compiled: log('torch.compile включён.')

def lm_loss(x, m=None):
    out = (m or fwd)(input_ids=x[:, :-1])
    l = F.cross_entropy(out.logits.float().reshape(-1, out.logits.size(-1)), x[:, 1:].reshape(-1))
    aux = getattr(out, 'aux_loss', None)
    return l, (aux if aux is not None else None)
# ---------- батч ----------
B = int(CFG.get('batch', 0) or 0); TOK_STEP = int(CFG.get('tokens_per_step', 0) or 0)
def autobatch():
    # CPU memory is shared with the OS; start conservatively instead of
    # allocating a large batch and forcing the machine into swap.
    if not cuda: return max(1, int(CFG.get('batch', 1) or 1))
    b = 64
    while b > 1:
        try:
            x = batch(tr, b, torch.Generator().manual_seed(0))
            with amp: l = lm_loss(x)[0]
            l.backward(); model.zero_grad(set_to_none=True); del x, l; torch.cuda.empty_cache()
            return max(1, int(b * 0.85))   # запас 15%
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True); torch.cuda.empty_cache(); b //= 2
    return 1
if not B: B = autobatch(); log(f'Подобран размер пачки: {B} × {T} токенов')
ACC = int(CFG.get('grad_accum', 0) or 0) or max(1, (TOK_STEP or 131072) // (B * T * WORLD)) if cuda else int(CFG.get('grad_accum', 1) or 1)
STEPS = int(CFG.get('max_steps', 0) or 0)
if STEPS <= 0: STEPS = max(1, int(float(CFG.get('epochs', 1)) * len(tr) / (B * T * ACC * WORLD)))
WARM = max(1, int(STEPS * float(CFG.get('warmup', 0.02))))
def lr_at(s):
    if s < WARM: return LR * (s + 1) / WARM
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, (s - WARM) / max(1, STEPS - WARM)))))

# ---------- продолжение ----------
CK = os.path.join(RUN, 'ckpt'); step = 0; seen = 0
def state_dicts():
    if FSDP_ON:
        from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions
        return get_state_dict(model, opt, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    return raw.state_dict(), opt.state_dict()
def save_ckpt():
    m, o = state_dicts()
    if R0:
        os.makedirs(CK, exist_ok=True); tmp = os.path.join(CK, 'state.pt.tmp')
        torch.save({'model': m, 'opt': o, 'step': step, 'seen': seen, 'B': B, 'ACC': ACC, 'STEPS': STEPS}, tmp); os.replace(tmp, os.path.join(CK, 'state.pt'))
        log(f'Контрольная точка сохранена (шаг {step}).')
if os.path.exists(os.path.join(CK, 'state.pt')) and CFG.get('resume', True):
    st = torch.load(os.path.join(CK, 'state.pt'), map_location='cpu', weights_only=False)
    if FSDP_ON:
        from torch.distributed.checkpoint.state_dict import set_state_dict, StateDictOptions
        set_state_dict(model, opt, model_state_dict=st['model'], optim_state_dict=st['opt'], options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=False))
    else:
        raw.load_state_dict(st['model']); opt.load_state_dict(st['opt'])
    step, seen, B, ACC, STEPS = st['step'], st['seen'], st['B'], st['ACC'], st['STEPS']; del st
    log(f'Продолжаю с шага {step} из {STEPS}.')

@torch.no_grad()
def sample(prompt=None, n=40):
    if FSDP_ON: return ''
    raw.eval(); p = prompt or CFG.get('sample_prompt') or ''
    ids = tok(p, return_tensors='pt')['input_ids'].to(dev) if p else torch.tensor([[tok.bos_token_id or tok.eos_token_id]], device=dev)
    for _ in range(n):
        with amp: lg = raw(input_ids=ids[:, -T:]).logits[0, -1].float()
        nx = torch.multinomial(F.softmax(lg / 0.8, -1), 1)
        if int(nx) == tok.eos_token_id: break
        ids = torch.cat([ids, nx[None]], 1)
    raw.train(); return tok.decode(ids[0], skip_special_tokens=True)

def evaluate(n=8):
    model.eval(); g = torch.Generator().manual_seed(123); ls = []
    with torch.no_grad():
        for _ in range(n):
            x = batch(va, min(B, 8), g)
            with amp: ls.append(float(lm_loss(x, model if FSDP_ON else raw)[0]))
    model.train(); return sum(ls) / len(ls)

# ---------- цикл ----------
model.train(); gen = torch.Generator().manual_seed(1000 + RANK + step)
total_tok = len(tr); per_step = B * T * ACC * WORLD
log(f'Обучение: {STEPS} шагов по {per_step} токенов (пачка {B} × накопление {ACC} × процессов {WORLD}), данных {total_tok/1e6:.1f} млн токенов.')
if step == 0 and (R0 or FSDP_ON):
    l0 = evaluate()
if step == 0 and R0:
    emit(step=0, max_steps=STEPS, eval_loss=round(l0, 4), loss=None); log(f'Loss до обучения: {l0:.4f}')
    with open(os.path.join(RUN, 'metrics.jsonl'), 'a') as f: f.write(json.dumps({'step': 0, 'eval_loss': l0}) + '\n')
SAVE = int(CFG.get('save_steps', 200)); EVAL = int(CFG.get('eval_steps', SAVE)); SAMPLE = int(CFG.get('sample_steps', SAVE)); oom_retry = False
t0 = time.time(); tok0 = seen; tl = time.time()
while step < STEPS:
    for g in opt.param_groups: g['lr'] = lr_at(step)
    try:
        lsum = 0.0
        for a in range(ACC):
            x = batch(tr, B, gen)
            sync = contextlib.nullcontext() if (WORLD == 1 or a == ACC - 1 or strategy == 'fsdp') else model.no_sync()
            with sync:
                with amp:
                    l, aux = lm_loss(x)
                    tot = l + (float(getattr(mc, 'router_aux_loss_coef', 0.01)) * aux if aux is not None else 0)
                scaler.scale(tot / ACC).backward(); lsum += l.item() / ACC
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    except torch.cuda.OutOfMemoryError:
        opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        if oom_retry or B == 1: emit(error='Не хватило видеопамяти даже после уменьшения пачки. Выберите модель меньше, включите 8-битный Adam или контрольные точки градиента.'); sys.exit(1)
        B = max(1, B // 2); ACC *= 2; oom_retry = True
        log(f'Не хватило видеопамяти — уменьшаю пачку до {B} (накопление {ACC}), общий размер шага тот же. Продолжаю.'); continue
    step += 1; seen += per_step
    if R0 and (step % int(CFG.get('log_steps', 5)) == 0 or step == STEPS):
        dtm = time.time() - t0; tps = (seen - tok0) / max(1e-6, dtm); eta = (STEPS - step) * per_step / max(1, tps)
        emit(progress=step / STEPS, step=step, max_steps=STEPS, loss=round(lsum, 4), lr=lr_at(step), tok_s=int(tps), seen=seen,
             eaten=round(seen / total_tok, 4), eta=int(eta), text=f'шаг {step}/{STEPS} · {int(tps):,} ток/с · съедено {seen/total_tok*100:.1f}% данных'.replace(',', ' '))
        with open(os.path.join(RUN, 'metrics.jsonl'), 'a') as f: f.write(json.dumps({'step': step, 'loss': lsum, 'tok_s': int(tps)}) + '\n')
    if step % EVAL == 0 or step == STEPS:
        el = evaluate() if (R0 or FSDP_ON) else 0
        if R0:
            emit(step=step, max_steps=STEPS, eval_loss=round(el, 4))
            with open(os.path.join(RUN, 'metrics.jsonl'), 'a') as f: f.write(json.dumps({'step': step, 'eval_loss': el}) + '\n')
    if R0 and SAMPLE and step % SAMPLE == 0: log('Пример текста: ' + sample().replace('\n', ' ⏎ ')[:300])
    if step % SAVE == 0 and step < STEPS: save_ckpt()
    if os.path.exists(os.path.join(RUN, 'PAUSE')):
        save_ckpt(); emit(paused=True); log('Пауза. «Продолжить» начнёт с этого места.'); sys.exit(0)
save_ckpt()
m, _ = state_dicts()
if R0:
    out = os.path.join(RUN, 'full'); raw_cpu = AutoModelForCausalLM.from_config(mc) if not os.path.exists(os.path.join(CFG['model'], 'model.safetensors')) and not glob.glob(os.path.join(CFG['model'], '*.safetensors')) else AutoModelForCausalLM.from_pretrained(CFG['model'], dtype=torch.float32)
    raw_cpu.load_state_dict(m); raw_cpu.save_pretrained(out, safe_serialization=True); tok.save_pretrained(out)
    for f in ('tokenizer.model', 'uchitel.json'):
        if os.path.exists(os.path.join(CFG['model'], f)):
            import shutil; shutil.copy(os.path.join(CFG['model'], f), out)
    log('Пример текста: ' + sample().replace('\n', ' ⏎ ')[:300])
    log(f'Обучение завершено. Модель: {out}'); emit(done=True, progress=1.0, result={'path': out})
if WORLD > 1: dist.barrier(); dist.destroy_process_group()
