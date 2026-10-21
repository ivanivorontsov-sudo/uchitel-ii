# Дообучение модели. Запуск: python train.py <папка запуска> (там config.json).
# Многокарточно: accelerate launch train.py <папка>  (сформирует программа).
import sys, os, json, time, math, glob
RUN = sys.argv[1]
CFG = json.load(open(os.path.join(RUN, 'config.json'), encoding='utf-8'))
RANK0 = int(os.environ.get('LOCAL_RANK', '0')) == 0
def emit(**k):
    if RANK0: print(json.dumps(k, ensure_ascii=False), flush=True)
def log(t): emit(log=t)

import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback, DataCollatorForLanguageModeling)

method = CFG['method']; cuda = torch.cuda.is_available()
if method != 'cpu' and not cuda:
    log('Видеокарта недоступна PyTorch — переключаюсь на обучение на процессоре.'); method = 'cpu'
bf16 = cuda and torch.cuda.is_bf16_supported(); fp16 = cuda and not bf16
dtype = torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.float32
log(f"Модель: {CFG['model']}, метод: {method}, устройство: {torch.cuda.get_device_name(0) if cuda else 'процессор'}")

tok = AutoTokenizer.from_pretrained(CFG['model'], trust_remote_code=CFG.get('trust_remote_code', False))
if tok.pad_token is None: tok.pad_token = tok.eos_token
kw = dict(dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=CFG.get('trust_remote_code', False))
if method == 'qlora':
    from transformers import BitsAndBytesConfig
    kw['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    kw['device_map'] = {'': int(os.environ.get('LOCAL_RANK', 0))}
log('Загружаю модель (первый раз скачивается с Hugging Face)…')
model = AutoModelForCausalLM.from_pretrained(CFG['model'], **kw)
model.config.use_cache = False
if method in ('lora', 'qlora', 'cpu'):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if method == 'qlora': model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=CFG.get('grad_ckpt', True))
    lc = LoraConfig(r=int(CFG.get('lora_r', 16)), lora_alpha=int(CFG.get('lora_alpha', 32)), lora_dropout=float(CFG.get('lora_dropout', 0.05)),
                    target_modules=CFG.get('target_modules') or 'all-linear', task_type='CAUSAL_LM')
    model = get_peft_model(model, lc)
    tp = sum(p.numel() for p in model.parameters() if p.requires_grad); ap = sum(p.numel() for p in model.parameters())
    log(f'Обучаемых параметров: {tp/1e6:.2f} млн из {ap/1e6:.0f} млн ({100*tp/ap:.2f}%)')

ML = int(CFG.get('max_len', 512))
def fmt(ex):
    if ex.get('messages'):
        try:
            return tok.apply_chat_template(ex['messages'], tokenize=False)
        except Exception:
            return '\n'.join(f"{m['role']}: {m['content']}" for m in ex['messages']) + tok.eos_token
    return (ex.get('text') or '') + tok.eos_token
ds = load_dataset('json', data_files=CFG['dataset'], split='train')
if CFG.get('max_examples'): ds = ds.select(range(min(len(ds), int(CFG['max_examples']))))
ds = ds.map(lambda b: tok([fmt(dict(zip(b, v))) for v in zip(*b.values())], truncation=True, max_length=ML), batched=True, remove_columns=ds.column_names)
split = ds.train_test_split(test_size=min(0.05, max(1, len(ds)//20)/len(ds)), seed=42) if len(ds) >= 20 else None
log(f'Примеров для обучения: {len(ds)}, длина до {ML} токенов')

class Progress(TrainerCallback):
    def __init__(s): s.t0 = time.time()
    def on_log(s, args, state, control, logs=None, **k):
        logs = logs or {}; done = state.global_step / max(1, state.max_steps)
        el = time.time() - s.t0; eta = el / max(1e-6, done) * (1 - done) if done > 0 else 0
        if RANK0 and (logs.get('loss') is not None or logs.get('eval_loss') is not None):
            with open(os.path.join(RUN, 'metrics.jsonl'), 'a', encoding='utf-8') as f:
                f.write(json.dumps({'step': state.global_step, 'loss': logs.get('loss'), 'eval_loss': logs.get('eval_loss')}) + '\n')
        emit(progress=done, step=state.global_step, max_steps=state.max_steps, epoch=round(state.epoch or 0, 3),
             loss=logs.get('loss'), eval_loss=logs.get('eval_loss'), lr=logs.get('learning_rate'), eta=int(eta),
             text=f'шаг {state.global_step} из {state.max_steps}')
    def on_step_end(s, args, state, control, **k):
        if os.path.exists(os.path.join(RUN, 'PAUSE')):
            control.should_save = True; control.should_training_stop = True
            log('Пауза: сохраняю контрольную точку…')
    def on_save(s, args, state, control, **k):
        log(f'Контрольная точка сохранена (шаг {state.global_step}).')

import inspect
_tr = split['train'] if split else ds
_steps = int(CFG.get('max_steps', -1) or -1)
if _steps <= 0:
    _steps = math.ceil(len(_tr) / max(1, int(CFG.get('batch', 1)) * int(CFG.get('grad_accum', 4)))) * float(CFG.get('epochs', 1))
A = dict(
    output_dir=os.path.join(RUN, 'checkpoints'), num_train_epochs=float(CFG.get('epochs', 1)), max_steps=int(CFG.get('max_steps', -1) or -1),
    per_device_train_batch_size=int(CFG.get('batch', 1)), gradient_accumulation_steps=int(CFG.get('grad_accum', 4)),
    learning_rate=float(CFG.get('lr', 2e-4)), lr_scheduler_type=CFG.get('scheduler', 'cosine'), warmup_steps=int(_steps * float(CFG.get('warmup', 0.03))),
    weight_decay=float(CFG.get('weight_decay', 0.0)), logging_steps=int(CFG.get('log_steps', 1)), save_steps=int(CFG.get('save_steps', 50)),
    save_total_limit=3, bf16=bf16, fp16=fp16, use_cpu=not cuda, gradient_checkpointing=bool(CFG.get('grad_ckpt', cuda)) and method != 'cpu',
    eval_strategy='steps' if split else 'no', eval_steps=int(CFG.get('save_steps', 50)), report_to=[], seed=int(CFG.get('seed', 42)),
    optim='paged_adamw_8bit' if method == 'qlora' else 'adamw_torch', dataloader_num_workers=0, disable_tqdm=True, remove_unused_columns=False,
    deepspeed=CFG.get('deepspeed') or None)
_ok = set(inspect.signature(TrainingArguments.__init__).parameters)
args = TrainingArguments(**{k: v for k, v in A.items() if k in _ok})
tr = Trainer(model=model, args=args, train_dataset=split['train'] if split else ds, eval_dataset=split['test'] if split else None,
             data_collator=DataCollatorForLanguageModeling(tok, mlm=False), callbacks=[Progress()])
ck = sorted(glob.glob(os.path.join(RUN, 'checkpoints', 'checkpoint-*')), key=lambda p: int(p.rsplit('-', 1)[1]))
if ck and CFG.get('resume', True):
    log(f'Продолжаю с контрольной точки {os.path.basename(ck[-1])}')
log('Обучение началось.')
tr.train(resume_from_checkpoint=ck[-1] if ck and CFG.get('resume', True) else None)
if os.path.exists(os.path.join(RUN, 'PAUSE')):
    emit(paused=True); log('Обучение на паузе. Нажмите «Продолжить», чтобы продолжить с этого места.'); sys.exit(0)
if RANK0:
    dst = os.path.join(RUN, 'adapter' if method != 'full' else 'full')
    tr.save_model(dst); tok.save_pretrained(dst)
    log(f'Обучение завершено. Результат: {dst}'); emit(done=True, progress=1.0, result={'path': dst})
