# Ivan GPT и свои модели: загрузка, перенос без потерь в HF (Llama / Qwen2), новые модели (Llama / Qwen2 / Qwen3),
# рост (слои, ширина, FFN, контекст) с сохранением функции, экспорт обратно в формат Ivan GPT, оценка памяти/времени.
import os, json, math, zipfile, io, shutil, glob

ARCHS = {
    'llama': {'name': 'Llama', 'hf': 'LlamaForCausalLM', 'ivan_lossless': True, 'gguf': True,
              'desc': 'Классика: RMSNorm, SwiGLU, RoPE. Ivan GPT устроена именно так — перенос весов точный. Лучший выбор, чтобы растить Ivan GPT.',
              'rec_grow': True},
    'qwen2': {'name': 'Qwen2', 'hf': 'Qwen2ForCausalLM', 'ivan_lossless': True, 'gguf': True,
              'desc': 'Как Llama, плюс смещения (bias) в Q/K/V и GQA. Ivan GPT переносится без потерь (смещения = 0).'},
    'qwen3': {'name': 'Qwen3', 'hf': 'Qwen3ForCausalLM', 'ivan_lossless': False, 'gguf': True,
              'desc': 'Современная: QK-нормализация (устойчивее обучение с нуля) и GQA. Только новая модель с нуля — знания Ivan GPT не переносятся.',
              'rec_new': True},
    'mixtral': {'name': 'Mixtral (MoE)', 'hf': 'MixtralForCausalLM', 'ivan_lossless': False, 'gguf': True, 'moe': True, 'upcycle': True,
                'desc': 'Смесь экспертов на основе Llama: в каждом слое несколько FFN-«экспертов», на каждый токен работают top-k. Можно «апсайклить» Ivan GPT и любую Llama-модель: FFN копируется в каждого эксперта — функция сохраняется.'},
    'qwen3_moe': {'name': 'Qwen3-MoE', 'hf': 'Qwen3MoeForCausalLM', 'ivan_lossless': False, 'gguf': True, 'moe': True,
                  'desc': 'Смесь экспертов в стиле Qwen3 (QK-нормализация, много маленьких экспертов). Только с нуля.'},
}
# пресеты размера: (подпись, слои, ширина, FFN, головы, kv-головы); head_dim = 64 как у Ivan GPT
SIZES = [('10M', 6, 320, 864, 5, 5), ('25M', 8, 448, 1216, 7, 7), ('50M', 12, 512, 1408, 8, 8), ('125M', 16, 768, 2048, 12, 12),
         ('350M', 24, 1024, 2816, 16, 16), ('1B', 20, 2048, 5632, 32, 32)]

def count_params(L, D, F, H, KV, V, hd=64, arch='llama', experts=0, topk=0, active=False):
    attn = D * H * hd * 2 + D * KV * hd * 2 + (2 * KV * hd + H * hd if arch == 'qwen2' else 0) + (2 * hd if arch in ('qwen3', 'qwen3_moe') else 0)
    ffn = 3 * D * F * ((topk if active else experts) if experts else 1) + (D * experts if experts else 0)
    return V * D + L * (attn + ffn + 2 * D) + D

# ---------- железо и оценки ----------
GPU_TFLOPS = [('5090', 209), ('5080', 112), ('5070 ti', 88), ('5070', 62), ('5060 ti', 47), ('5060', 38), ('4090', 165), ('4080', 97),
              ('4070 ti', 80), ('4070', 58), ('4060 ti', 44), ('4060', 30), ('3090', 71), ('3080', 59), ('3070', 40), ('3060', 25),
              ('2080', 40), ('2070', 30), ('2060', 20), ('a100', 312), ('h100', 700), ('l4', 120), ('t4', 65)]
def gpu_tflops(name, vram):
    n = (name or '').lower()
    for k, v in GPU_TFLOPS:
        if k in n: return v
    return 20 + 3 * vram
def hw_budget(hw):
    g = hw.get('gpus') or []
    if g:
        x = max(g, key=lambda a: a['vram_gb'])
        return {'device': 'gpu', 'name': x['name'], 'mem_gb': x['vram_gb'], 'tflops': gpu_tflops(x['name'], x['vram_gb']), 'mfu': 0.35}
    cores = hw.get('cores') or 4
    return {'device': 'cpu', 'name': hw.get('cpu') or 'процессор', 'mem_gb': hw.get('ram_gb', 8), 'tflops': cores * 0.12, 'mfu': 0.5}

def train_mem_gb(N, L, D, V, T=512, B=8, device='gpu', ckpt=True, adam8=False):
    st = N * ((4 + 4 + (2 if adam8 else 8)) if device == 'gpu' else 16)   # веса fp32 + градиенты + Adam
    a = 2 if device == 'gpu' else 4
    act = B * T * D * a * (2 * L + 34) if ckpt else B * T * D * a * 34 * L
    logits = B * T * V * 4 * 2
    return (st + act + logits) / 1e9 + (0.8 if device == 'gpu' else 1.5)

def plan(hw, vocab=8000, base=None, tokens=None):
    """Варианты размеров в пределах железа (запас 15%), с оценкой памяти и времени."""
    b = hw_budget(hw); limit = b['mem_gb'] * (0.85 if b['device'] == 'gpu' else 0.6)
    disk = hw.get('disk_free_gb', 50); out = []
    for lab, L, D, F, H, KV in SIZES:
        if base and (L < base['layers'] or D < base['dim'] or F < base['hidden'] or H < base['heads']): continue
        N = count_params(L, D, F, H, KV, vocab)
        mem = train_mem_gb(N, L, D, vocab, device=b['device']); mem8 = train_mem_gb(N, L, D, vocab, device=b['device'], adam8=True)
        tok = tokens or 20 * N                         # «по Шиншилле» ≈ 20 токенов на параметр
        sec = 6 * N * tok / (b['tflops'] * 1e12 * b['mfu'])
        tps = b['tflops'] * 1e12 * b['mfu'] / (6 * N)
        fits = mem <= limit or (b['device'] == 'gpu' and mem8 <= limit)
        out.append({'label': lab, 'layers': L, 'dim': D, 'hidden': F, 'heads': H, 'kv_heads': KV, 'params': N, 'mem_gb': round(mem, 1),
                    'mem8_gb': round(mem8, 1), 'fits': fits, 'adam8': mem > limit and fits, 'tok_per_s': int(tps), 'hours': round(sec / 3600, 1),
                    'tokens': int(tok), 'disk_gb': round(N * 4 * 3 / 1e9 + 0.2, 1)})
    ok = [o for o in out if o['fits']]
    for o in out:
        o['tag'] = '' if not o['fits'] else 'на пределе' if o is ok[-1] and len(ok) > 1 else 'быстро' if o is ok[0] else 'оптимально'
        if not o['fits']: o['warn'] = f"Не влезет: нужно ≈{o['mem8_gb'] if b['device']=='gpu' else o['mem_gb']} ГБ, доступно ≈{limit:.1f} ГБ"
    return {'budget': b, 'limit_gb': round(limit, 1), 'sizes': out, 'max': ok[-1]['label'] if ok else None, 'disk_free_gb': disk}

# ---------- чтение Ivan GPT ----------
def _ivan_cfg(c):
    return {'vocab': c.get('vocab', 8000), 'dim': c.get('dim', 320), 'layers': c.get('layers', 6), 'heads': c.get('heads', 5),
            'ctx': c.get('ctx', 256), 'hidden': c.get('hidden', 864), 'eps': c.get('eps', 1e-5)}
def load_ivan(src, tok_path=None, work=None):
    """src: папка / model.safetensors / zip / apk / model.bin. Возвращает (state_dict fp32, config, путь к ivan.model или None)."""
    import torch, numpy as np
    from safetensors.torch import load_file, load as st_load
    tok = None; cfg = None; sd = None
    def from_bin(raw, c):
        c = _ivan_cfg(c); sd = {}; a = np.frombuffer(raw, dtype=np.float16); o = 0
        for name, shape in json.loads(json.dumps(c_full['tensors'])):
            n = int(np.prod(shape)); sd[name] = torch.from_numpy(a[o:o + n].astype(np.float32).reshape(shape)); o += n
        if o != a.size: raise ValueError('model.bin не совпадает с config.json')
        return sd, c
    if os.path.isdir(src):
        for p in ('model.safetensors', 'out/model.safetensors'):
            if os.path.exists(os.path.join(src, p)): sd = load_file(os.path.join(src, p)); break
        cj = next((os.path.join(src, p) for p in ('config.json', 'web/config.json') if os.path.exists(os.path.join(src, p))), None)
        if sd is None and os.path.exists(os.path.join(src, 'web', 'model.bin')):
            c_full = json.load(open(os.path.join(src, 'web', 'config.json'))); sd, cfg = from_bin(open(os.path.join(src, 'web', 'model.bin'), 'rb').read(), c_full)
        if cj and cfg is None: cfg = _ivan_cfg(json.load(open(cj)))
        t = glob.glob(os.path.join(src, '**', '*.model'), recursive=True); tok = t[0] if t else None
    elif src.endswith('.safetensors'):
        sd = load_file(src); d = os.path.dirname(src)
        if os.path.exists(os.path.join(d, 'config.json')): cfg = _ivan_cfg(json.load(open(os.path.join(d, 'config.json'))))
        t = glob.glob(os.path.join(d, '**', 'ivan.model'), recursive=True); tok = t[0] if t else None
    elif src.lower().endswith(('.zip', '.apk')):
        z = zipfile.ZipFile(src); names = z.namelist()
        st = [n for n in names if n.endswith('model.safetensors')]
        if st: sd = st_load(z.read(st[0]))
        else:
            mb = [n for n in names if n.endswith('model.bin')]
            if not mb: raise ValueError('В архиве нет весов Ivan GPT (model.safetensors или web/model.bin)')
            c_full = json.loads(z.read(mb[0][:-9] + 'config.json')); sd, cfg = from_bin(z.read(mb[0]), c_full)
        cj = [n for n in names if n.endswith('config.json') and 'web/' not in n]
        if cj and cfg is None: cfg = _ivan_cfg(json.loads(z.read(cj[0])))
        tm = [n for n in names if n.endswith('.model')]
        if tm:
            import tempfile; tok = os.path.join(work or tempfile.mkdtemp(), 'ivan.model')
            open(tok, 'wb').write(z.read(tm[0]))
    else: raise ValueError('Неизвестный формат: ' + src)
    tok = tok_path or tok
    sd = {k: v.float() for k, v in sd.items() if k not in ('head.weight', 'cos', 'sin')}
    if cfg is None:   # конфиг по форме тензоров (как в model.py, head_dim 64)
        V, D = sd['emb.weight'].shape; L = len({k.split('.')[1] for k in sd if k.startswith('blocks.')})
        cfg = _ivan_cfg({'vocab': V, 'dim': D, 'layers': L, 'heads': D // 64, 'hidden': sd['blocks.0.w1.weight'].shape[0]})
    return sd, cfg, tok

def _perm(hd):   # чередующиеся пары RoPE (x[::2], x[1::2]) -> половины (rotate_half в HF)
    return list(range(0, hd, 2)) + list(range(1, hd, 2))
def _inv(p):
    r = [0] * len(p)
    for i, j in enumerate(p): r[j] = i
    return r

def hf_config(arch, V, D, L, H, KV, F, ctx, eps=1e-5, theta=10000.0, bos=1, eos=2, pad=None, experts=8, topk=2, aux=0.01):
    from transformers import AutoConfig
    kw = dict(vocab_size=V, hidden_size=D, num_hidden_layers=L, num_attention_heads=H, num_key_value_heads=KV, intermediate_size=F,
              max_position_embeddings=ctx, rms_norm_eps=eps, rope_theta=theta, tie_word_embeddings=True, hidden_act='silu',
              bos_token_id=bos, eos_token_id=eos, pad_token_id=pad, head_dim=D // H, architectures=[ARCHS[arch]['hf']])
    if arch == 'llama': kw.update(attention_bias=False, mlp_bias=False)
    if arch == 'mixtral': kw.update(num_local_experts=experts, num_experts_per_tok=topk, router_aux_loss_coef=aux, output_router_logits=True)
    if arch == 'qwen3_moe': kw.update(num_experts=experts, num_experts_per_tok=topk, moe_intermediate_size=F, decoder_sparse_step=1, mlp_only_layers=[],
                                     norm_topk_prob=True, router_aux_loss_coef=aux, output_router_logits=True)
    c = AutoConfig.for_model(arch, **kw)
    try: c.rope_parameters = {'rope_type': 'default', 'rope_theta': theta}   # transformers 5
    except Exception: pass
    return c

def ivan_to_hf_sd(sd, c, arch):
    D, H = c['dim'], c['heads']; hd = D // H; p = _perm(hd); idx = [h * hd + j for h in range(H) for j in p]
    out = {'model.embed_tokens.weight': sd['emb.weight'], 'model.norm.weight': sd['norm.w'], 'lm_head.weight': sd['emb.weight']}
    for l in range(c['layers']):
        b = f'blocks.{l}.'; m = f'model.layers.{l}.'; qkv = sd[b + 'qkv.weight']
        q, k, v = qkv[:D][idx], qkv[D:2 * D][idx], qkv[2 * D:]
        out.update({m + 'self_attn.q_proj.weight': q, m + 'self_attn.k_proj.weight': k, m + 'self_attn.v_proj.weight': v,
                    m + 'self_attn.o_proj.weight': sd[b + 'o.weight'], m + 'input_layernorm.weight': sd[b + 'n1.w'],
                    m + 'post_attention_layernorm.weight': sd[b + 'n2.w'], m + 'mlp.gate_proj.weight': sd[b + 'w1.weight'],
                    m + 'mlp.up_proj.weight': sd[b + 'w3.weight'], m + 'mlp.down_proj.weight': sd[b + 'w2.weight']})
        if arch == 'qwen2':
            import torch
            for n, w in (('q', q), ('k', k), ('v', v)): out[m + f'self_attn.{n}_proj.bias'] = torch.zeros(w.shape[0])
    return out

CHAT_TEMPLATE = ("{% for m in messages %}{% if m['role'] == 'user' %}Человек: {{ m['content'] }}\n{% elif m['role'] == 'assistant' %}"
                 "ИИ: {{ m['content'] }}{{ eos_token }}\n{% else %}{{ m['content'] }}\n{% endif %}{% endfor %}{% if add_generation_prompt %}ИИ:{% endif %}")

def save_tokenizer(spm_path, out):
    """SentencePiece -> токенизатор HF (Llama, Metaspace prepend=first). Совпадает с sp.encode, кроме текста, начинающегося с пробела."""
    from transformers import AutoTokenizer
    os.makedirs(out, exist_ok=True); tm = os.path.join(out, 'tokenizer.model')
    if os.path.abspath(spm_path) != os.path.abspath(tm): shutil.copy(spm_path, tm)
    json.dump({'tokenizer_class': 'LlamaTokenizer', 'add_bos_token': False, 'add_eos_token': False, 'legacy': False,
               'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '</s>', 'model_max_length': 1 << 20},
              open(os.path.join(out, 'tokenizer_config.json'), 'w', encoding='utf-8'))
    t = AutoTokenizer.from_pretrained(out); t.chat_template = CHAT_TEMPLATE; t.save_pretrained(out)
    if not os.path.exists(tm): shutil.copy(spm_path, tm)
    return t

def save_hf(sd_hf, cfg, out, spm_path=None, tok_src=None, meta=None):
    import torch
    from transformers import AutoModelForCausalLM
    os.makedirs(out, exist_ok=True)
    with torch.device('meta'): m = AutoModelForCausalLM.from_config(cfg)
    m.load_state_dict(sd_hf, strict=True, assign=True); m.tie_weights()
    m.save_pretrained(out, safe_serialization=True)
    if spm_path: save_tokenizer(spm_path, out)
    elif tok_src:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(tok_src).save_pretrained(out)
    json.dump(meta or {}, open(os.path.join(out, 'uchitel.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return out

def convert_ivan(src, out, arch='llama', ctx=None, tok_path=None):
    if not ARCHS[arch]['ivan_lossless']: raise SystemExit(f'Перенос Ivan GPT в {ARCHS[arch]["name"]} без потерь невозможен — создайте новую модель этой архитектуры.')
    sd, c, tok = load_ivan(src, tok_path)
    if not tok: raise SystemExit('Не найден токенизатор ivan.model рядом с весами.')
    cfg = hf_config(arch, c['vocab'], c['dim'], c['layers'], c['heads'], c['heads'], c['hidden'], ctx or c['ctx'], c['eps'])
    save_hf(ivan_to_hf_sd(sd, c, arch), cfg, out, spm_path=tok, meta={'from': 'ivan-gpt', 'arch': arch, 'ivan_cfg': c})
    return sd, c, tok

# ---------- проверка логитов против исходного model.py ----------
IVAN_MODEL_PY = None
def ivan_reference(sd, c):
    """Точная копия forward из ivan-gpt/model.py (RoPE чередующимися парами, RMSNorm eps=1e-5, SwiGLU, общие веса)."""
    import torch, torch.nn.functional as F
    D, H = c['dim'], c['heads']; hd = D // H; eps = c.get('eps', 1e-5)
    f = 1.0 / 10000.0 ** (torch.arange(0, hd, 2).float() / hd)
    def rms(x, w): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
    def rope(x, cos, sin):
        x1, x2 = x[..., ::2], x[..., 1::2]; return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)
    def fwd(idx):
        B, T = idx.shape; a = torch.outer(torch.arange(T).float(), f); cos, sin = a.cos(), a.sin(); x = sd['emb.weight'][idx]
        for l in range(c['layers']):
            b = f'blocks.{l}.'
            q, k, v = (rms(x, sd[b + 'n1.w']) @ sd[b + 'qkv.weight'].T).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
            q, k = rope(q, cos, sin), rope(k, cos, sin)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D)
            x = x + y @ sd[b + 'o.weight'].T; h = rms(x, sd[b + 'n2.w'])
            x = x + (F.silu(h @ sd[b + 'w1.weight'].T) * (h @ sd[b + 'w3.weight'].T)) @ sd[b + 'w2.weight'].T
        return rms(x, sd['norm.w']) @ sd['emb.weight'].T
    return fwd

def hf_logits(model_dir, ids):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, attn_implementation='eager').eval()
    with torch.no_grad(): return m(torch.tensor(ids)).logits

def compare(a, b):
    d = (a - b).abs(); return {'max_abs': float(d.max()), 'mean_abs': float(d.mean()), 'argmax_same': float((a.argmax(-1) == b.argmax(-1)).float().mean())}

# ---------- рост ----------
def grow(src_dir, out, layers=None, dim=None, hidden=None, heads=None, ctx=None, seed=0, std=0.02):
    """Сохраняет функцию модели: новые слои — тождественные (o_proj=down_proj=0); ширина — новые размерности
    остатка нулевые, RMSNorm пересчитан (вес·√(D/D'), eps·D/D'); новые головы и нейроны FFN входят с нулевым выходом."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    g = torch.Generator().manual_seed(seed)
    c = AutoConfig.from_pretrained(src_dir); arch = c.model_type
    m = AutoModelForCausalLM.from_pretrained(src_dir, dtype=torch.float32); sd = m.state_dict()
    L, D, F, H, KV = c.num_hidden_layers, c.hidden_size, c.intermediate_size, c.num_attention_heads, c.num_key_value_heads
    hd = getattr(c, 'head_dim', None) or D // H
    L2, D2, F2 = layers or L, dim or D, hidden or F
    H2 = heads or max(H, D2 // hd); KV2 = H2 if KV == H else max(KV, KV * H2 // H)
    if L2 < L or D2 < D or F2 < F or H2 < H: raise SystemExit('Уменьшать модель нельзя — только увеличивать.')
    if H2 % KV2: raise SystemExit('Число голов должно делиться на число kv-голов.')
    if KV != H and H2 != H and (H2 // KV2) != (H // KV): raise SystemExit('Для GQA рост голов поддерживается только с тем же размером группы.')
    r = lambda *s: torch.randn(*s, generator=g) * std
    def pad(w, rows, cols, fill_rows='rand', fill_cols='rand'):
        o = torch.zeros(rows, cols); o[:w.shape[0], :w.shape[1]] = w
        if rows > w.shape[0] and fill_rows == 'rand': o[w.shape[0]:, :] = r(rows - w.shape[0], cols)
        if cols > w.shape[1] and fill_cols == 'rand': o[:w.shape[0], w.shape[1]:] = r(w.shape[0], cols - w.shape[1])
        return o
    s = math.sqrt(D / D2); eps2 = c.rms_norm_eps * D / D2
    def norm(w): o = torch.full((D2,), float(w.mean()) * s); o[:D] = w * s; return o
    new = {}
    emb = pad(sd['model.embed_tokens.weight'], sd['model.embed_tokens.weight'].shape[0], D2, fill_cols='zero')
    new['model.embed_tokens.weight'] = emb; new['lm_head.weight'] = emb; new['model.norm.weight'] = norm(sd['model.norm.weight'])
    def headpad(w, nh_old, nh_new, cols):   # строки [nh*hd, D] -> [nh2*hd, D2]; новые головы случайные
        return pad(w, nh_new * hd, cols)
    def block(i, src=None):
        P = f'model.layers.{i}.'; o = {}
        if src is None:   # новый тождественный слой
            o[P + 'input_layernorm.weight'] = torch.ones(D2) * s; o[P + 'post_attention_layernorm.weight'] = torch.ones(D2) * s
            o[P + 'self_attn.q_proj.weight'] = r(H2 * hd, D2); o[P + 'self_attn.k_proj.weight'] = r(KV2 * hd, D2); o[P + 'self_attn.v_proj.weight'] = r(KV2 * hd, D2)
            o[P + 'self_attn.o_proj.weight'] = torch.zeros(D2, H2 * hd)
            o[P + 'mlp.gate_proj.weight'] = r(F2, D2); o[P + 'mlp.up_proj.weight'] = r(F2, D2); o[P + 'mlp.down_proj.weight'] = torch.zeros(D2, F2)
            if arch == 'qwen2':
                for n, k in (('q', H2), ('k', KV2), ('v', KV2)): o[P + f'self_attn.{n}_proj.bias'] = torch.zeros(k * hd)
            if arch == 'qwen3': o[P + 'self_attn.q_norm.weight'] = torch.ones(hd); o[P + 'self_attn.k_norm.weight'] = torch.ones(hd)
            return o
        S = f'model.layers.{src}.'
        o[P + 'input_layernorm.weight'] = norm(sd[S + 'input_layernorm.weight']); o[P + 'post_attention_layernorm.weight'] = norm(sd[S + 'post_attention_layernorm.weight'])
        o[P + 'self_attn.q_proj.weight'] = pad(sd[S + 'self_attn.q_proj.weight'], H2 * hd, D2)
        o[P + 'self_attn.k_proj.weight'] = pad(sd[S + 'self_attn.k_proj.weight'], KV2 * hd, D2)
        o[P + 'self_attn.v_proj.weight'] = pad(sd[S + 'self_attn.v_proj.weight'], KV2 * hd, D2)
        o[P + 'self_attn.o_proj.weight'] = pad(sd[S + 'self_attn.o_proj.weight'], D2, H2 * hd, fill_rows='zero', fill_cols='zero')
        o[P + 'mlp.gate_proj.weight'] = pad(sd[S + 'mlp.gate_proj.weight'], F2, D2); o[P + 'mlp.up_proj.weight'] = pad(sd[S + 'mlp.up_proj.weight'], F2, D2)
        o[P + 'mlp.down_proj.weight'] = pad(sd[S + 'mlp.down_proj.weight'], D2, F2, fill_rows='zero', fill_cols='zero')
        if arch == 'qwen2':
            for n, k in (('q', H2), ('k', KV2), ('v', KV2)):
                b = torch.zeros(k * hd); ob = sd[S + f'self_attn.{n}_proj.bias']; b[:ob.shape[0]] = ob; o[P + f'self_attn.{n}_proj.bias'] = b
        if arch == 'qwen3':
            o[P + 'self_attn.q_norm.weight'] = sd[S + 'self_attn.q_norm.weight']; o[P + 'self_attn.k_norm.weight'] = sd[S + 'self_attn.k_norm.weight']
        return o
    # равномерно вставляем новые слои после старых
    order = []; extra = L2 - L
    for i in range(L):
        order.append(i)
        for _ in range(extra // L + (1 if i >= L - extra % L else 0)): order.append(None)
    for j, src in enumerate(order): new.update(block(j, src))
    c2 = hf_config(arch, c.vocab_size, D2, L2, H2, KV2, F2, ctx or c.max_position_embeddings, eps2,
                   bos=c.bos_token_id, eos=c.eos_token_id, pad=c.pad_token_id)
    if c2.head_dim != hd: c2.head_dim = hd
    meta = json.load(open(os.path.join(src_dir, 'uchitel.json'), encoding='utf-8')) if os.path.exists(os.path.join(src_dir, 'uchitel.json')) else {}
    meta.setdefault('history', []).append({'grow_from': [L, D, F, H], 'to': [L2, D2, F2, H2]}); meta['arch'] = arch
    save_hf(new, c2, out, tok_src=src_dir, meta=meta)
    return {'params_before': sum(v.numel() for k, v in sd.items() if k != 'lm_head.weight'),
            'params_after': sum(v.numel() for k, v in new.items() if k != 'lm_head.weight'), 'layers': L2, 'dim': D2, 'hidden': F2, 'heads': H2}

# ---------- новая модель с нуля ----------
def new_model(out, arch, L, D, F, H, KV, ctx, tokenizer_dir, seed=0, experts=8, topk=2, lazy=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    V = max(len(tok), tok.vocab_size); V = (V + 63) // 64 * 64
    eos = tok.eos_token_id; bos = tok.bos_token_id if tok.bos_token_id is not None else eos
    cfg = hf_config(arch, V, D, L, H, KV, F, ctx, 1e-6 if arch == 'qwen3' else 1e-5, bos=bos, eos=eos, pad=tok.pad_token_id if tok.pad_token_id is not None else eos, experts=experts, topk=topk)
    cfg.initializer_range = 0.02
    N = count_params(L, D, F, H, KV, V, D // H, arch, experts if ARCHS[arch].get('moe') else 0, topk)
    if lazy is None:
        try: import psutil; avail = psutil.virtual_memory().available
        except Exception: avail = 8e9
        lazy = N * 4 > 0.4 * avail
    if lazy:   # огромная модель: только описание; веса создаются при обучении сразу по частям на всех GPU (FSDP)
        os.makedirs(out, exist_ok=True); cfg.save_pretrained(out)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        if not tok.chat_template: tok.chat_template = CHAT_TEMPLATE
        tok.save_pretrained(out)
        if os.path.exists(os.path.join(tokenizer_dir, 'tokenizer.model')): shutil.copy(os.path.join(tokenizer_dir, 'tokenizer.model'), out)
        json.dump({'arch': arch, 'from': 'scratch', 'lazy_init': True}, open(os.path.join(out, 'uchitel.json'), 'w', encoding='utf-8'))
        return N
    torch.manual_seed(seed); m = AutoModelForCausalLM.from_config(cfg)
    with torch.no_grad():   # масштаб выходных проекций как в GPT-2/nanoGPT
        for n, p in m.named_parameters():
            if n.endswith(('o_proj.weight', 'down_proj.weight')): p.normal_(0, 0.02 / math.sqrt(2 * L))
    os.makedirs(out, exist_ok=True); m.save_pretrained(out, safe_serialization=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    if not tok.chat_template: tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(out)
    for f in ('tokenizer.model',):
        if os.path.exists(os.path.join(tokenizer_dir, f)): shutil.copy(os.path.join(tokenizer_dir, f), out)
    json.dump({'arch': arch, 'from': 'scratch'}, open(os.path.join(out, 'uchitel.json'), 'w', encoding='utf-8'))
    return sum(p.numel() for p in m.parameters())

def train_tokenizer(texts_iter, out, vocab=16000, max_chars=200_000_000):
    """Быстрый SentencePiece BPE (как у Ivan GPT: byte_fallback, цифры по одной) на тексте пользователя."""
    import sentencepiece as spm
    os.makedirs(out, exist_ok=True); tmp = os.path.join(out, '_sample.txt'); n = 0
    with open(tmp, 'w', encoding='utf-8') as f:
        for t in texts_iter:
            f.write(t.replace('\x00', '') + '\n'); n += len(t)
            if n > max_chars: break
    spm.SentencePieceTrainer.train(input=tmp, model_prefix=os.path.join(out, 'sp'), vocab_size=vocab, model_type='bpe', character_coverage=0.9995,
                                   byte_fallback=True, split_digits=True, input_sentence_size=2_000_000, shuffle_input_sentence=True,
                                   num_threads=os.cpu_count() or 4, user_defined_symbols=['\n'], remove_extra_whitespaces=False,
                                   normalization_rule_name='identity', pad_id=-1, unk_id=0, bos_id=1, eos_id=2, hard_vocab_limit=False)
    os.remove(tmp); save_tokenizer(os.path.join(out, 'sp.model'), out); return out

# ---------- экспорт обратно в формат Ivan GPT ----------
def export_ivan(model_dir, out):
    import torch, numpy as np
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForCausalLM
    c = AutoConfig.from_pretrained(model_dir)
    if c.model_type not in ('llama', 'qwen2'): raise SystemExit('В формат Ivan GPT можно сохранить только модели Llama/Qwen2 (у Qwen3 есть QK-нормализация, которой нет в движке Ivan GPT).')
    if c.num_key_value_heads != c.num_attention_heads: raise SystemExit('Движок Ivan GPT не поддерживает GQA.')
    sd = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).state_dict()
    if any(k.endswith('_proj.bias') and sd[k].abs().max() > 0 for k in sd): raise SystemExit('У модели ненулевые bias в Q/K/V — в формат Ivan GPT без потерь не переносится.')
    D, H, L = c.hidden_size, c.num_attention_heads, c.num_hidden_layers; hd = D // H; inv = _inv(_perm(hd)); idx = [h * hd + j for h in range(H) for j in inv]
    o = {'emb.weight': sd['model.embed_tokens.weight'], 'norm.w': sd['model.norm.weight']}
    for l in range(L):
        P = f'model.layers.{l}.'; b = f'blocks.{l}.'
        o[b + 'n1.w'] = sd[P + 'input_layernorm.weight']; o[b + 'n2.w'] = sd[P + 'post_attention_layernorm.weight']
        o[b + 'qkv.weight'] = torch.cat([sd[P + 'self_attn.q_proj.weight'][idx], sd[P + 'self_attn.k_proj.weight'][idx], sd[P + 'self_attn.v_proj.weight']])
        o[b + 'o.weight'] = sd[P + 'self_attn.o_proj.weight']; o[b + 'w1.weight'] = sd[P + 'mlp.gate_proj.weight']
        o[b + 'w3.weight'] = sd[P + 'mlp.up_proj.weight']; o[b + 'w2.weight'] = sd[P + 'mlp.down_proj.weight']
    o = {k: v.contiguous() for k, v in o.items()}
    os.makedirs(os.path.join(out, 'web'), exist_ok=True); os.makedirs(os.path.join(out, 'tok'), exist_ok=True)
    save_file({k: v.half() for k, v in o.items()}, os.path.join(out, 'model.safetensors'))
    cfg = {'vocab': c.vocab_size, 'dim': D, 'layers': L, 'heads': H, 'ctx': c.max_position_embeddings, 'hidden': c.intermediate_size, 'eps': c.rms_norm_eps}
    json.dump(cfg, open(os.path.join(out, 'config.json'), 'w'))
    order = ['emb.weight'] + [f'blocks.{l}.{n}' for l in range(L) for n in ['n1.w', 'qkv.weight', 'o.weight', 'n2.w', 'w1.weight', 'w3.weight', 'w2.weight']] + ['norm.w']
    with open(os.path.join(out, 'web', 'model.bin'), 'wb') as f:
        for k in order: o[k].half().numpy().tofile(f)
    json.dump({**cfg, 'tensors': [[k, list(o[k].shape)] for k in order]}, open(os.path.join(out, 'web', 'config.json'), 'w'))
    tm = os.path.join(model_dir, 'tokenizer.model')
    if os.path.exists(tm):
        import sentencepiece as spm
        shutil.copy(tm, os.path.join(out, 'tok', 'ivan.model')); sp = spm.SentencePieceProcessor(model_file=tm)
        json.dump({'pieces': [sp.id_to_piece(i) for i in range(sp.get_piece_size())], 'scores': [sp.get_score(i) for i in range(sp.get_piece_size())]},
                  open(os.path.join(out, 'web', 'vocab.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    std = (D, L, H, c.max_position_embeddings, c.intermediate_size, c.vocab_size) == (320, 6, 5, 256, 864, 8000)
    return {'path': out, 'config': cfg, 'standard_size': std, 'eps_note': abs(c.rms_norm_eps - 1e-5) > 1e-12}

# ---------- апсайклинг плотной модели в MoE (Mixtral) ----------
def upcycle(src_dir, out, experts=8, topk=2, noise=0.0, seed=0):
    """FFN плотного слоя копируется в каждого эксперта; роутер случайный. При noise=0 модель считает то же самое
    (Mixtral нормирует веса top-k, их сумма = 1). Небольшой шум разводит экспертов, чтобы они учились разному."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    c = AutoConfig.from_pretrained(src_dir)
    if c.model_type != 'llama': raise SystemExit('Апсайклинг в Mixtral — из модели Llama (Ivan GPT сначала переносится в Llama).')
    g = torch.Generator().manual_seed(seed); sd = AutoModelForCausalLM.from_pretrained(src_dir, dtype=torch.float32).state_dict(); new = {}
    for k, v in sd.items():
        if '.mlp.' not in k: new[k] = v
    for l in range(c.num_hidden_layers):
        P = f'model.layers.{l}.mlp.'
        gu = torch.cat([sd[P + 'gate_proj.weight'], sd[P + 'up_proj.weight']])
        dn = sd[P + 'down_proj.weight']
        ns = lambda w: w + (torch.randn(w.shape, generator=g) * noise * w.std() if noise else 0)
        new[P + 'experts.gate_up_proj'] = torch.stack([ns(gu) for _ in range(experts)])
        new[P + 'experts.down_proj'] = torch.stack([ns(dn) for _ in range(experts)])
        new[P + 'gate.weight'] = torch.randn(experts, c.hidden_size, generator=g) * 0.02
    c2 = hf_config('mixtral', c.vocab_size, c.hidden_size, c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads, c.intermediate_size,
                   c.max_position_embeddings, c.rms_norm_eps, bos=c.bos_token_id, eos=c.eos_token_id, pad=c.pad_token_id, experts=experts, topk=topk)
    c2.head_dim = getattr(c, 'head_dim', None) or c.hidden_size // c.num_attention_heads
    meta = json.load(open(os.path.join(src_dir, 'uchitel.json'), encoding='utf-8')) if os.path.exists(os.path.join(src_dir, 'uchitel.json')) else {}
    meta.setdefault('history', []).append({'upcycle': [experts, topk, noise]}); meta['arch'] = 'mixtral'
    save_hf(new, c2, out, tok_src=src_dir, meta=meta)
    tot = sum(v.numel() for k, v in new.items() if k != 'lm_head.weight')
    act = tot - c.num_hidden_layers * 3 * c.hidden_size * c.intermediate_size * (experts - topk)
    return {'params_total': tot, 'params_active': act}
