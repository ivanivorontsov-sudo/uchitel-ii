// «Своя модель»: Ivan GPT, рост, MoE, новая модель с нуля, источники данных, быстрое полное обучение.
let CAT=null,MODELS=[],PLAN=null,NEW={arch:'qwen3',size:null,tok:'ivan'};
const fmtN=n=>n>=1e12?(n/1e12).toFixed(2)+' трлн':n>=1e9?(n/1e9).toFixed(2)+' млрд':(n/1e6).toFixed(1)+' млн';
const fmtH=h=>h<1?Math.max(1,Math.round(h*60))+' мин':h<48?h.toFixed(1)+' ч':(h/24).toFixed(1)+' дн';
const U=window.uchitel;
async function waitJob(id,onTick){for(;;){const j=await api('job?id='+id);onTick&&onTick(j);if(j.state!=='running')return j;await new Promise(r=>setTimeout(r,1200))}}
async function runTask(kind,params,stEl){const j=await api('ivan/task',{kind,...params});const r=await waitJob(j.id,x=>{if(stEl)$(stEl).textContent=(x.logs.slice(-1)[0]||'')+(x.text?' · '+x.text:'')});
 if(r.state!=='done')throw new Error(r.error||'задача не выполнена (подробности в «Журнале»)');return r.result}
async function loadOwn(){if(!CAT)CAT=await api('catalog');
 if(U){$('o-gh-st').textContent=(await U.hasSecret('github'))?'✓ GitHub-токен сохранён':'нет токена';$('o-hf-st').textContent=(await U.hasSecret('hf'))?'✓ HF-токен сохранён':'нет токена'}
 else{$('o-gh-st').textContent='Ключи и скачивание Ivan GPT работают в программе (не в браузере).'}
 await loadModels();renderNew();renderSources()}
async function saveSecrets(){if(!U)return say('Только в программе');const g=$('o-gh').value.trim(),h=$('o-hf').value.trim();
 try{if(g)await U.setSecret('github',g);if(h)await U.setSecret('hf',h)}catch(e){return say(e.message)}$('o-gh').value=$('o-hf').value='';loadOwn();if(h)say('HF-токен сохранён. Он начнёт действовать после перезапуска программы.')}
async function clearSecrets(){if(!U||!confirm('Удалить сохранённые ключи?'))return;await U.setSecret('github','');await U.setSecret('hf','');loadOwn()}
async function ivanList(){if(!U)return say('Только в программе');try{const r=await U.ivanReleases();$('o-tag').innerHTML=r.map(x=>`<option value="${esc(x.tag)}">${esc(x.tag)} — ${esc(x.name||'')}${x.latest?' (последний)':''}</option>`).join('')}catch(e){say(e.message)}}
async function ivanFetch(){const tag=$('o-tag').value;if(!tag)return say('Сначала «Обновить список» и выберите релиз');U.onIvanProgress(d=>$('o-ivan-st').textContent='Скачиваю '+d.text);
 try{const f=await U.ivanFetch(tag);await ivanImport(f.src,f.tok,tag)}catch(e){$('o-ivan-st').textContent='';say('Ошибка: '+e.message)}}
async function ivanLocal(kind){if(!U)return say('Только в программе');const p=await U.pick(kind);if(!p?.length)return;let tok=null;
 if(!confirm('Токенизатор ivan.model лежит рядом с весами или внутри архива? (Отмена — выбрать его файл)')){const t=await U.pick('files');tok=t?.[0]||null}
 await ivanImport(p[0],tok,'local')}
async function ivanImport(src,tok,tag){try{const r=await runTask('ivan_import',{src,tok,tag,arch:$('o-iarch').value},'o-ivan-st');
 $('o-ivan-st').textContent=`✓ Готово: разница логитов ${r.check.max_abs.toExponential(1)}, лучший токен совпадает в ${Math.round(r.check.argmax_same*100)}%`;loadModels()}catch(e){say('Ошибка: '+e.message)}}

async function loadModels(){MODELS=await api('models');$('o-models').innerHTML=MODELS.length?MODELS.map((m,i)=>`<div class="card" style="margin:8px 0"><b>${esc(m.name)}</b> · ${esc(m.arch)} · ${fmtN(m.params)} параметров${m.experts?` (активных ${fmtN(m.active)}, экспертов ${m.experts}, top-${m.topk})`:''}
 <br><small>${m.layers} слоёв · ширина ${m.dim} · FFN ${m.hidden} · голов ${m.heads} · контекст ${m.ctx} · словарь ${m.vocab}${m.from==='ivan-gpt'?' · из Ivan GPT':''}</small>
 <div class="row"><button onclick="growPanel(${i})">📈 Увеличить</button>${m.arch==='llama'?`<button onclick="moePanel(${i})">🧩 В смесь экспертов (MoE)</button>`:''}<button class="big" onclick="trainPanel(${i})">🏋️ Обучить</button>
 <button onclick="exportGguf(${i})">📱 GGUF</button>${['llama','qwen2'].includes(m.arch)&&m.kv_heads===m.heads?`<button onclick="exportIvan(${i})">💾 В формат Ivan GPT</button>`:''}</div></div>`).join(''):'<p class="hint">Пока нет. Скачайте Ivan GPT или создайте новую модель.</p>'}

function sizeCards(plan,sel,onpick){return `<p class="hint">${plan.budget.device==='gpu'?`Видеокарта ${esc(plan.budget.name)}, ${plan.budget.mem_gb} ГБ`:`Без видеокарты: оценка по процессору и ОЗУ (${plan.budget.mem_gb} ГБ)`}. Предел этого компьютера с запасом 15%: <b>${plan.max||'—'}</b>.</p><div class="row" style="flex-wrap:wrap">`+
 plan.sizes.map((s,i)=>`<div class="model ${s.fits?'fit':'nofit'}" style="display:block;min-width:190px;cursor:pointer;${sel===i?'outline:2px solid #2f7d4f':''}" onclick="${onpick}(${i})"><div class="n">${s.label} ${s.tag?`<small>· ${s.tag}</small>`:''}</div>
 <small>${fmtN(s.params)} · ${s.layers} сл. × ${s.dim}<br>память ≈ ${s.mem_gb} ГБ${s.adam8?` (с 8-бит Adam ${s.mem8_gb})`:''}<br>${s.tok_per_s.toLocaleString('ru')} ток/с · ${fmtH(s.hours)} на ${fmtN(s.tokens).replace(' ','\u00a0')} ток.</small>${s.warn?`<br><small class="bad">${esc(s.warn)}</small>`:''}</div>`).join('')+'</div>'}
function advFields(pfx,s,moe){return `<details><summary>Для опытных: ручная настройка (любой размер)</summary><div class="grid2">
 <label>Слоёв <input id="${pfx}-L" type="number" value="${s.layers}"></label><label>Ширина <input id="${pfx}-D" type="number" step="64" value="${s.dim}"></label>
 <label>FFN <input id="${pfx}-F" type="number" value="${s.hidden}"></label><label>Голов внимания <input id="${pfx}-H" type="number" value="${s.heads}"></label>
 <label>Контекст (токенов) <input id="${pfx}-C" type="number" value="${s.ctx||1024}"></label>${moe?`<label>Экспертов <input id="${pfx}-E" type="number" value="8"></label><label>Активных (top-k) <input id="${pfx}-K" type="number" value="2"></label>`:''}
 </div><button onclick="checkCustom('${pfx}')">Оценить</button> <span id="${pfx}-est"></span></details>`}
function readAdv(p){const v=k=>+($(p+'-'+k)?.value||0);return{layers:v('L'),dim:v('D'),hidden:v('F'),heads:v('H'),kv_heads:v('H'),ctx:v('C'),experts:v('E'),topk:v('K')}}
async function checkCustom(p){const c=readAdv(p);const arch=p==='g'?GROW.m.arch:NEW.arch;const r=await api('plan2',{custom:c,arch,experts:c.experts,topk:c.topk,vocab:p==='g'?GROW.m.vocab:8000});const x=r.custom;
 $(p+'-est').innerHTML=`${fmtN(x.params)} параметров${x.active!==x.params?` (активных ${fmtN(x.active)})`:''} · память для полного обучения ≈ ${x.mem_gb} ГБ · ≈${fmtH(x.hours)} на ${fmtN(x.tokens)} токенов · ОЗУ при оффлоаде ≈ ${x.ram_gb_offload} ГБ · диск ≈ ${x.disk_gb} ГБ`+
 (x.fits?' <span class="ok">✓ влезет</span>':` <b class="bad">Не влезет в этот компьютер (предел ${r.limit_gb} ГБ)${x.gpus_needed?`, нужно ≈${x.gpus_needed} таких видеокарт (FSDP)`:''}.</b>`);return r}

let GROW=null;
async function growPanel(i){const m=MODELS[i];PLAN=await api('plan2',{base:{layers:m.layers,dim:m.dim,hidden:m.hidden,heads:m.heads},vocab:m.vocab});GROW={m,sel:null};
 $('o-panel').innerHTML=`<div class="card"><h3>📈 Увеличить «${esc(m.name)}» (сейчас ${fmtN(m.params)})</h3><p class="hint">Модель растёт без потери знаний: новые слои сначала «прозрачные» (ничего не меняют), новые нейроны и головы начинают с нулевым вкладом. Ответы сразу после роста те же — потом нужно полное обучение, чтобы новые части научились.</p>
 ${sizeCards(PLAN,null,'growPick')}${advFields('g',{...m,ctx:m.ctx},false)}<label>Новое имя <input id="g-name" value="${esc(m.name.replace(/\W+/g,'-'))}-big"></label><label><input type="checkbox" id="g-ok"> Я понимаю, что большая модель может не влезть и учиться долго</label>
 <button class="big" onclick="growGo()">Увеличить</button> <span id="g-st"></span></div>`;$('o-panel').scrollIntoView()}
function growPick(i){GROW.sel=i;const s=PLAN.sizes[i];for(const[k,v]of[['L',s.layers],['D',s.dim],['F',s.hidden],['H',s.heads]])$('g-'+k).value=v;
 document.querySelectorAll('#o-panel .model').forEach((e,j)=>e.style.outline=j===i?'2px solid #2f7d4f':'')}
async function growGo(){const c=readAdv('g');const r=await checkCustom('g');if(!r.custom.fits&&!$('g-ok').checked)return say('Этот размер больше, чем потянет компьютер. Отметьте «Я понимаю», чтобы всё равно продолжить.');
 try{const x=await runTask('grow',{src:GROW.m.path,name:$('g-name').value,layers:c.layers,dim:c.dim,hidden:c.hidden,heads:c.heads,ctx:c.ctx},'g-st');
 $('g-st').innerHTML=`✓ ${fmtN(x.params_before)} → ${fmtN(x.params_after)}. Loss до ${x.check.loss_before.toFixed(4)}, после ${x.check.loss_after.toFixed(4)} — знания сохранены. Теперь «Обучить» (полное обучение).`;loadModels()}catch(e){say(e.message)}}
function moePanel(i){const m=MODELS[i];$('o-panel').innerHTML=`<div class="card"><h3>🧩 Смесь экспертов из «${esc(m.name)}»</h3><p class="hint">FFN каждого слоя копируется в N экспертов (с небольшим шумом, чтобы они учились разному). На каждый токен работают top-k экспертов: параметров больше, а скорость почти как у исходной. Знания сохраняются; потом — полное обучение.</p>
 <div class="grid2"><label>Экспертов <input id="m-E" type="number" value="8"></label><label>Активных (top-k) <input id="m-K" type="number" value="2"></label><label>Шум <input id="m-N" value="0.01"></label><label>Имя <input id="m-name" value="${esc(m.name.replace(/\W+/g,'-'))}-moe"></label></div>
 <button class="big" onclick="moeGo('${esc(m.path.replace(/\\/g,'\\\\'))}')">Создать MoE</button> <span id="m-st"></span></div>`;$('o-panel').scrollIntoView()}
async function moeGo(src){try{const x=await runTask('upcycle',{src,name:$('m-name').value,experts:+$('m-E').value,topk:+$('m-K').value,noise:+$('m-N').value},'m-st');
 $('m-st').textContent=`✓ Всего ${fmtN(x.params_total)}, активных ${fmtN(x.params_active)}. Loss до ${x.check.loss_before.toFixed(4)}, после ${x.check.loss_after.toFixed(4)}.`;loadModels()}catch(e){say(e.message)}}

async function trainPanel(i){const m=MODELS[i];const d=await api('datasets');const hw=HW||await api('hw');const gpus=Math.max(1,hw.gpus.length);
 $('o-panel').innerHTML=`<div class="card"><h3>🏋️ Обучение «${esc(m.name)}» (${fmtN(m.params)})</h3>
 <label>Режим <select id="tr-mode"><option value="full">Полное обучение (все веса, быстрый движок) — после роста и для новой модели</option><option value="pretrain">Продолжение предобучения на текстах (Википедия, книги)</option><option value="lora">LoRA (небольшие «насадки», обычная вкладка «Обучение»)</option></select></label>
 <label>Наборы данных (Ctrl — несколько) <select id="tr-ds" multiple size="5">${d.map(x=>`<option value="${esc(x.path)}">${esc(x.name)}${x.info?.bin?` · ${(x.info.bin.train_tokens/1e6).toFixed(1)} млн ток.`:''}</option>`).join('')}</select></label>
 <div class="grid2"><label>Шагов (0 = 1 проход по данным) <input id="tr-steps" type="number" value="0"></label><label>Скорость обучения <input id="tr-lr" value="3e-4"></label>
 <label>Длина куска (токенов) <input id="tr-len" type="number" value="${Math.min(m.ctx||1024,1024)}"></label><label>Пачка (0 = подобрать) <input id="tr-b" type="number" value="0"></label>
 <label>Видеокарт/процессов <input id="tr-g" type="number" value="${gpus}" min="1"></label><label>Распределение <select id="tr-st"><option value="ddp">DDP (модель целиком на каждой карте)</option><option value="fsdp">FSDP (модель делится между картами)</option></select></label>
 <label><input type="checkbox" id="tr-off"> Выгружать в ОЗУ (CPU offload, для огромных моделей)</label><label><input type="checkbox" id="tr-a8"> 8-битный Adam (меньше памяти)</label><label><input type="checkbox" id="tr-ck"> Контрольные точки градиента (меньше памяти, медленнее)</label>
 <label>Затравка для примеров <input id="tr-pr" value="Человек: Привет!\\nИИ:"></label></div>
 <button class="big" onclick="trainGo(${i})">▶ Начать</button> <span id="tr-stt"></span></div>`;$('o-panel').scrollIntoView()}
async function trainGo(i){const m=MODELS[i];const ds=[...$('tr-ds').selectedOptions].map(o=>o.value);if(!ds.length)return say('Выберите набор данных');const mode=$('tr-mode').value;
 if(mode==='lora'){ST.model=m.path;localStorage.model=m.path;ST.method=HW?.gpus?.length?'lora':'cpu';show('train');$('t-model').value=m.path;$('t-ds').value=ds[0];return}
 try{$('tr-stt').textContent='Готовлю данные (токенизация в .bin)…';const b=await runTask('tokenize',{model:m.path,datasets:ds},'tr-stt');
 const cfg={model:m.path,data:b.path,max_steps:+$('tr-steps').value,lr:+$('tr-lr').value*(mode==='pretrain'?0.5:1),max_len:+$('tr-len').value,batch:+$('tr-b').value,gpus:+$('tr-g').value,strategy:$('tr-st').value,
  offload:$('tr-off').checked,adam8:$('tr-a8').checked,grad_ckpt:$('tr-ck').checked,save_steps:200,sample_prompt:$('tr-pr').value.replace(/\\n/g,'\n'),warmup:mode==='pretrain'?0.01:0.02,method:'full'};
 const j=await api('pretrain',cfg);ST.trainJob=localStorage.trainJob=j.id;ST.run=localStorage.run=j.run;show('train');pollTrain()}catch(e){say(e.message)}}
async function exportGguf(i){const m=MODELS[i];const q=prompt('Сжатие: Q4_K_M, Q5_K_M, Q8_0 или F16','Q8_0');if(!q)return;const j=await api('gguf',{model_dir:m.path,quant:q,name:m.name.replace(/\W+/g,'-')});watchJob(j.id)}
async function exportIvan(i){const m=MODELS[i];try{const r=await runTask('ivan_export',{model_dir:m.path,name:m.name.replace(/\W+/g,'-')});say('Сохранено: '+r.path+(r.standard_size?'':'\n\nРазмер нестандартный: веб-движок Ivan GPT читает его из config.json, а generate.py из репозитория ivan-gpt рассчитан на исходный размер.'));U?.reveal(r.path)}catch(e){say(e.message)}}

function renderNew(){const A=CAT.archs;const moe=!!A[NEW.arch].moe;
 $('o-new').innerHTML=`<h3>Шаг 1. Архитектура</h3><div class="row" style="flex-wrap:wrap">${Object.entries(A).map(([k,a])=>`<div class="model ${NEW.arch===k?'fit':''}" style="display:block;max-width:300px;cursor:pointer" onclick="NEW.arch='${k}';renderNew()"><div class="n">${a.name} ${a.rec_new?'<span class="ok">· рекомендуется для новой</span>':''}${a.rec_grow?'<span class="ok">· рекомендуется для роста Ivan GPT</span>':''}</div><small>${esc(a.desc)}</small></div>`).join('')}</div>
 <h3>Шаг 2. Размер</h3><div id="n-sizes">считаю…</div>${advFields('n',{layers:12,dim:512,hidden:1408,heads:8,ctx:1024},moe)}
 <h3>Шаг 3. Токенизатор</h3><label><select id="n-tok"><option value="ivan">Готовый токенизатор Ivan GPT (8000, русский) — нужна скачанная Ivan GPT</option><option value="train">Обучить свой на моих данных (SentencePiece, минуты)</option><option value="qwen">Готовый токенизатор Qwen (151 тыс., много языков и код)</option></select></label>
 <label>Размер словаря (если обучать) <input id="n-voc" type="number" value="16000"></label>
 <h3>Шаг 4. Данные и предобучение</h3><select id="n-ds" multiple size="4"></select><label>Имя модели <input id="n-name" value="moya-${NEW.arch}"></label>
 <label><input type="checkbox" id="n-ok"> Я понимаю: если размер больше возможностей компьютера, обучение может не начаться или идти очень долго</label>
 <button class="big" onclick="newGo()">Создать и начать предобучение</button> <span id="n-st"></span>`;
 api('datasets').then(d=>$('n-ds').innerHTML=d.map(x=>`<option value="${esc(x.path)}">${esc(x.name)}</option>`).join(''));
 api('plan2',{vocab:8000}).then(p=>{PLAN=p;$('n-sizes').innerHTML=sizeCards(p,NEW.size,'newPick')})}
function newPick(i){NEW.size=i;const s=PLAN.sizes[i];for(const[k,v]of[['L',s.layers],['D',s.dim],['F',s.hidden],['H',s.heads]])$('n-'+k).value=v;document.querySelectorAll('#n-sizes .model').forEach((e,j)=>e.style.outline=j===i?'2px solid #2f7d4f':'')}
async function newGo(){const ds=[...$('n-ds').selectedOptions].map(o=>o.value);if(!ds.length)return say('Выберите данные (вкладка «Данные» или источники ниже)');const c=readAdv('n');
 if(CAT.archs[NEW.arch].moe){c.hidden=Math.max(64,Math.round(c.hidden/4/64)*64)}   // у MoE эксперты меньше обычного FFN
 const r=await checkCustom('n');if(!r.custom.fits&&!$('n-ok').checked)return say('Размер больше возможностей компьютера. Отметьте «Я понимаю», чтобы всё равно начать.');
 try{const x=await runTask('new_model',{name:$('n-name').value,arch:NEW.arch,size:c,ctx:c.ctx,tokenizer:$('n-tok').value,vocab:+$('n-voc').value,datasets:ds,experts:c.experts||8,topk:c.topk||2},'n-st');
 $('n-st').textContent='Токенизирую данные…';const b=await runTask('tokenize',{model:x.path,datasets:ds},'n-st');
 const j=await api('pretrain',{model:x.path,data:b.path,max_steps:0,lr:6e-4,max_len:Math.min(c.ctx,1024),batch:0,gpus:HW?.gpus?.length||1,save_steps:200,sample_prompt:'',method:'full'});
 ST.trainJob=localStorage.trainJob=j.id;ST.run=localStorage.run=j.run;loadModels();show('train');pollTrain()}catch(e){say(e.message)}}

function renderSources(){$('o-src').innerHTML=CAT.sources.map((s,i)=>`<div class="card" style="margin:6px 0"><b>${esc(s.name)}</b>${s.gated?' <small class="opt">(нужен HF-токен)</small>':''}<br><small>Лицензия: ${esc(s.license)}${s.url?` · <a href="#" onclick="U?.openExternal('${s.url}')||window.open('${s.url}');return false">подробнее</a>`:''}</small>${s.note?`<br><small>${esc(s.note)}</small>`:''}
 <div class="row" style="flex-wrap:wrap">${Object.entries(s.params).map(([k,v])=>`<input id="s${i}-${k}" value="${esc(v)}" placeholder="${k}" title="${k}" style="width:${k==='url'||k==='repo'?'320px':'120px'}">`).join('')}
 <input id="s${i}-lim" type="number" value="500" title="лимит, МБ текста" style="width:90px"><small>МБ</small><select id="s${i}-lang"><option value="any">любой язык</option><option value="ru">русский</option><option value="en">английский</option></select>
 <select id="s${i}-tok"><option value="">без токенизации</option>${MODELS.map(m=>`<option value="${esc(m.path)}">токенизировать для ${esc(m.name)}</option>`).join('')}</select><button onclick="srcGo(${i})">Скачать</button> <span id="s${i}-st"></span></div></div>`).join('')}
async function srcGo(i){const s=CAT.sources[i];const p={source:s.id,preset:s.preset,limit_mb:+$(`s${i}-lim`).value,lang_filter:$(`s${i}-lang`).value,tokenize_with:$(`s${i}-tok`).value||null};
 for(const k in s.params)p[k]=$(`s${i}-${k}`).value;p.name=(s.preset||s.id)+'_'+(p.lang||p.config||p.split||'')+'_'+Date.now().toString(36);
 try{const r=await runTask('collect',p,`s${i}-st`);$(`s${i}-st`).textContent=`✓ текстов ${r.stats.texts}${r.tokens?`, ${(r.tokens.train_tokens/1e6).toFixed(1)} млн токенов`:''}`}catch(e){$(`s${i}-st`).textContent='';say(e.message)}}
