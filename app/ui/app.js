const $=id=>document.getElementById(id);
const _q=new URLSearchParams(location.search);
const api=async(p,b)=>{const r=await fetch('/api/'+p,b?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}:{});const j=await r.json();if(j&&j.error&&!j.id)throw new Error(j.error);return j};
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const say=m=>alert(m);
let HW=null, ST={model:localStorage.model||'Qwen/Qwen2.5-0.5B-Instruct', method:localStorage.method||'', trainJob:_q.get('job')||localStorage.trainJob||'', run:localStorage.run||''};
const METH={full:'Полное обучение',lora:'LoRA',qlora:'QLoRA 4 бита',cpu:'Процессор (LoRA)'};
document.querySelectorAll('nav a').forEach(a=>a.onclick=()=>show(a.dataset.p));
function show(p){document.querySelectorAll('nav a').forEach(x=>x.classList.toggle('on',x.dataset.p===p));document.querySelectorAll('.page').forEach(x=>x.classList.toggle('on',x.id===p));({env:loadEnv,data:loadDs,train:loadTrain,result:loadRuns,chat:loadChat,jobs:loadJobs,model:renderModels,own:()=>loadOwn()})[p]?.()}

async function loadHw(){HW=await api('hw');const g=HW.gpus;
 $('hw-box').innerHTML=`<div class="kv"><b>Система</b><span>${esc(HW.os)}</span><b>Процессор</b><span>${esc(HW.cpu)}, ядер: ${HW.cores}</span>
 <b>Оперативная память</b><span>${HW.ram_gb} ГБ (свободно ${HW.ram_free_gb} ГБ)</span>
 <b>Видеокарта</b><span>${g.length?g.map(x=>`${esc(x.name)} — ${x.vram_gb} ГБ видеопамяти, драйвер ${esc(x.driver)}`).join('<br>'):'<span class="bad">NVIDIA не найдена</span> — обучать можно только маленькие модели на процессоре'}</span>
 <b>CUDA (драйвер)</b><span>${HW.cuda_driver||'—'}</span><b>Свободно на диске</b><span>${HW.disk_free_gb} ГБ в ${esc(HW.home)}</span></div>
 <h2>Что подойдёт</h2>${advice()}<p><button class="big" onclick="show('model')">Выбрать модель →</button></p>`;
 $('side-info').innerHTML=`${g.length?esc(g[0].name)+'<br>'+g[0].vram_gb+' ГБ VRAM':'без видеокарты'}<br>ОЗУ ${HW.ram_gb} ГБ<br>Диск ${HW.disk_free_gb} ГБ`;}
function advice(){const ms=HW.recommend.models.filter(m=>m.method);if(!ms.length)return '<p class="bad">Памяти мало даже для самых маленьких моделей.</p>';
 const best=ms.reduce((a,b)=>b.params_b>a.params_b?b:a);return `<p>Самая большая модель для этого компьютера: <b>${best.id}</b> (${best.params_b} млрд) способом <b>${best.method_ru}</b>.<br>
 Для проб начните с <b>Qwen/Qwen2.5-0.5B-Instruct</b> — она учится за минуты. Модели крупнее можно учить на более мощном компьютере: та же программа подберёт способ сама.</p>`}
function renderModels(){if(!HW)return;$('model-id').value=ST.model;
 $('model-list').innerHTML=HW.recommend.models.map(m=>`<div class="model ${m.method?'fit':'nofit'}"><div><div class="n">${esc(m.id)}</div><small>${m.params_b} млрд параметров · ${esc(m.desc)}</small><br>
 <small>Память: полное ${m.need.full} ГБ · LoRA ${m.need.lora} ГБ · QLoRA ${m.need.qlora} ГБ · процессор ${m.need.cpu} ГБ ОЗУ</small></div>
 <div class="m">${m.method?`<span class="ok">✓ ${esc(m.method_ru)}</span>`:`<span class="opt">${esc(m.method_ru)}</span>`}<br><button onclick="pickModel('${m.id}',${m.params_b})">${ST.model===m.id?'✓ Выбрана':'Выбрать'}</button></div></div>`).join('')}
function guessB(id){const m=/(\d+(?:\.\d+)?)\s*[bB]\b/.exec(id)||/(\d+(?:\.\d+)?)[bB]/.exec(id);if(m)return +m[1];const k=/(\d+)M/.exec(id);return k?k[1]/1000:7}
function pickModel(id,b){id=id.trim();if(!id)return;b=b||guessB(id);const vr=HW.recommend.vram_total_gb,ram=HW.ram_gb;
 const need={full:b*16+2,lora:b*2.2+2,qlora:b*.7+2,cpu:b*5+2};let m=null;if(vr)for(const k of['full','lora','qlora'])if(need[k]<=vr*.92){m=k;break}
 if(!m&&b<=1.6&&need.cpu<=ram*.7)m='cpu';ST.model=id;ST.method=m||'qlora';localStorage.model=id;localStorage.method=ST.method;
 if(!m)say(`Модели ${id} (~${b} млрд) не хватит памяти на этом компьютере (нужно ≈${need.qlora.toFixed(0)} ГБ видеопамяти даже для QLoRA). Можно подготовить всё здесь, а обучать на более мощном.`);
 renderModels();$('t-model').value=id;$('t-method').value=ST.method}

async function loadEnv(){const e=await api('env');const i=await api('info');$('env-home').textContent=i.home;
 $('env-table').innerHTML='<tr><th></th><th>Компонент</th><th>Подробности</th><th></th></tr>'+e.items.map(x=>`<tr><td class="${x.ok?'ok':x.required?'bad':'opt'}">${x.ok?'✓':x.required?'✗':'—'}</td><td>${esc(x.name)}${x.required?'':' <small>(необязательно)</small>'}</td><td><small>${esc(x.hint||'')}</small></td>
 <td>${!x.ok&&x.install?`<button onclick="install('${x.install}')">${x.install==='driver'?'Открыть сайт':'Установить'}</button>`:''}</td></tr>`).join('')+
 `<tr><td colspan=4>${e.missing.length?`<b class="bad">Не хватает: ${e.missing.length}</b>`:'<b class="ok">Всё готово к обучению</b>'}</td></tr>`}
async function install(w){const r=await api('install',{what:w});if(r.open){window.uchitel?.openExternal(r.open)||window.open(r.open);return}if(r.msg)return say(r.msg);watchJob(r.id,'env')}
async function installAll(){const r=await api('install',{what:'all'});if(r.msg)return say(r.msg);watchJob(r.id,'env')}
function watchJob(id,page){show('jobs');openJob(id)}

async function download(){const url=$('dl-url').value.trim();if(!url)return say('Вставьте ссылку');const j=await api('download',{url,sha256:$('dl-sha').value.trim()});jobToSrc(j.id)}
async function hfdata(){const repo=$('hf-repo').value.trim();if(!repo)return say('Впишите имя набора');const j=await api('hfdata',{repo,split:$('hf-split').value,limit:+$('hf-limit').value});jobToSrc(j.id)}
function jobToSrc(id){const t=setInterval(async()=>{const j=await api('job?id='+id);$('ds-preview').textContent=j.title+'\n'+(j.text||'')+'\n'+j.logs.slice(-6).join('\n');
 if(j.state!=='running'){clearInterval(t);if(j.result?.path){$('src').value=($('src').value+'\n'+j.result.path).trim()}else if(j.error)say('Ошибка: '+j.error)}},1000)}
async function pick(kind){if(!window.uchitel)return say('Выбор файлов работает в программе. Впишите путь вручную.');const ps=await window.uchitel.pick(kind);if(ps?.length)$('src').value=($('src').value+'\n'+ps.join('\n')).trim()}
async function prepare(){const sources=$('src').value.split('\n').filter(Boolean);if(!sources.length)return say('Добавьте файлы или скачайте данные');
 const j=await api('prepare',{sources,name:$('ds-name').value,mode:$('ds-mode').value,chunk:+$('ds-chunk').value,overlap:+$('ds-overlap').value});
 const t=setInterval(async()=>{const s=await api('job?id='+j.id);$('ds-preview').textContent=s.logs.slice(-8).join('\n');
 if(s.state!=='running'){clearInterval(t);if(s.result){$('ds-preview').textContent=s.logs.slice(-3).join('\n')+'\n\nПервые примеры:\n'+s.result.preview.map(x=>JSON.stringify(x,null,1)).join('\n—\n');loadDs()}}},800)}
async function loadDs(){const d=await api('datasets');$('ds-list').innerHTML=d.length?d.map(x=>`<div class="job" onclick="prev('${esc(x.path.replace(/\\/g,'\\\\'))}')"><b>${esc(x.name)}</b> — текстов ${x.info.stats?.texts??'?'}, диалогов ${x.info.stats?.chats??'?'}, убрано повторов ${x.info.stats?.dupes??'?'}</div>`).join(''):'<p class="hint">Пока нет.</p>';return d}
async function prev(p){const r=await api('preview?n=6&path='+encodeURIComponent(p));$('ds-preview').textContent=r.map(x=>JSON.stringify(x,null,1)).join('\n—\n')}

const PRESETS={quick:{epochs:1,steps:30,lr:'2e-4',len:256,batch:1,acc:2,r:8,alpha:16,save:10},normal:{epochs:2,steps:-1,lr:'2e-4',len:512,batch:2,acc:4,r:16,alpha:32,save:50},deep:{epochs:3,steps:-1,lr:'1e-4',len:1024,batch:2,acc:8,r:32,alpha:64,save:100}};
function preset(){const p=PRESETS[$('t-preset').value];for(const k in p)$('t-'+({epochs:'epochs',steps:'steps',lr:'lr',len:'len',batch:'batch',acc:'acc',r:'r',alpha:'alpha',save:'save'}[k])).value=p[k];if($('t-method').value==='full')$('t-lr').value='2e-5'}
async function loadTrain(){$('t-model').value=ST.model;if(!ST.method&&HW){const r=HW.recommend.models.find(m=>m.id===ST.model);ST.method=r?.method||(HW.gpus.length?'qlora':'cpu')}if(ST.method)$('t-method').value=ST.method;const d=await api('datasets');
 $('t-ds').innerHTML=d.map(x=>`<option value="${esc(x.path)}">${esc(x.name)}</option>`).join('')||'<option value="">— сначала подготовьте данные —</option>';if(!$('t-epochs').value)preset();if(ST.trainJob)pollTrain()}
async function startTrain(){if(!$('t-ds').value)return say('Сначала подготовьте набор данных на вкладке «Данные»');
 const cfg={model:$('t-model').value.trim(),dataset:$('t-ds').value,method:$('t-method').value,epochs:+$('t-epochs').value,max_steps:+$('t-steps').value,lr:+$('t-lr').value,max_len:+$('t-len').value,batch:+$('t-batch').value,grad_accum:+$('t-acc').value,lora_r:+$('t-r').value,lora_alpha:+$('t-alpha').value,save_steps:+$('t-save').value,gpus:+$('t-gpus').value};
 const j=await api('train',cfg);ST.trainJob=localStorage.trainJob=j.id;ST.run=localStorage.run=j.run;pollTrain()}
async function pauseTrain(){if(ST.trainJob)await api('pause',{id:ST.trainJob}).catch(e=>say(e.message))}
async function stopTrain(){if(ST.trainJob&&confirm('Остановить обучение? Несохранённые шаги после последней контрольной точки пропадут.'))await api('stop',{id:ST.trainJob}).catch(e=>say(e.message))}
async function resumeTrain(){if(!ST.run)return say('Нет обучения для продолжения');const j=await api('resume',{run:ST.run});ST.trainJob=localStorage.trainJob=j.id;pollTrain()}
let pt=null;function pollTrain(){clearInterval(pt);const f=async()=>{let j;try{j=await api('job?id='+ST.trainJob)}catch{clearInterval(pt);return}
 const L=j.last||{};const eta=L.eta?` · осталось ≈ ${Math.floor(L.eta/60)} мин ${L.eta%60} с`:'';
 $('t-status').textContent=({running:'Идёт обучение',done:'✓ Обучение завершено',paused:'⏸ На паузе',stopped:'Остановлено',error:'✗ Ошибка: '+(j.error||'')}[j.state])+(L.step?` · шаг ${L.step}/${L.max_steps} · эпоха ${L.epoch}`:'')+(L.loss!=null?` · loss ${(+L.loss).toFixed(3)}`:'')+(L.tok_s?` · ${L.tok_s.toLocaleString('ru')} ток/с · съедено ${(100*L.eaten).toFixed(1)}% данных`:'')+eta;
 $('t-bar').style.width=(100*(j.progress||0))+'%';$('t-log').textContent=j.logs.slice(-60).join('\n');$('t-log').scrollTop=1e9;chart(j.points);if(j.state!=='running')clearInterval(pt)};f();pt=setInterval(f,1500)}
function chart(pts){const c=$('loss'),x=c.getContext('2d'),W=c.width,H=c.height;x.clearRect(0,0,W,H);x.font='12px Segoe UI';x.fillStyle='#5b6b60';
 const tr=pts.filter(p=>p.loss!=null),ev=pts.filter(p=>p.eval_loss!=null);if(!tr.length){x.fillText('График ошибки (loss) появится после первых шагов',20,30);return}
 const all=[...tr.map(p=>p.loss),...ev.map(p=>p.eval_loss)],mx=Math.max(...all),mn=Math.min(...all),ms=Math.max(...pts.map(p=>p.step||0),1);
 const X=s=>40+(W-60)*s/ms,Y=v=>H-25-(H-45)*(v-mn)/((mx-mn)||1);x.strokeStyle='#dfe7e1';x.strokeRect(40,20,W-60,H-45);
 x.fillText(mx.toFixed(2),4,24);x.fillText(mn.toFixed(2),4,H-25);x.fillText('шаг '+ms,W-70,H-8);
 for(const[arr,k,col]of[[tr,'loss','#2f7d4f'],[ev,'eval_loss','#c26b00']]){if(!arr.length)continue;x.strokeStyle=col;x.lineWidth=2;x.beginPath();arr.forEach((p,i)=>i?x.lineTo(X(p.step),Y(p[k])):x.moveTo(X(p.step),Y(p[k])));x.stroke()}
 x.fillStyle='#2f7d4f';x.fillText('— обучение',W-200,14);x.fillStyle='#c26b00';x.fillText('— проверка',W-110,14)}

async function loadRuns(){const r=await api('runs');$('runs').innerHTML=r.length?r.map(x=>{const P=esc(x.run.replace(/\\/g,'\\\\'));return `<div class="card" style="margin:8px 0"><b>${esc(x.model)}</b> · ${METH[x.method]||x.method}<br><small>${esc(x.run)}</small><br>
 ${x.adapter?'✓ LoRA-насадка':x.full?'✓ полные веса':x.checkpoints?`есть ${x.checkpoints} контр. точ. — <button onclick="ST.run='${P}';localStorage.run=ST.run;show('train');resumeTrain()">Продолжить обучение</button>`:'не завершено'} ${x.merged?' · ✓ объединено':''}
 <div class="row">${x.adapter?`<button onclick="merge('${P}')">1. Влить LoRA в веса</button>`:''}
 ${x.merged||x.full?`<select id="q-${btoa(x.run).replace(/=/g,'')}"><option>Q4_K_M</option><option>Q5_K_M</option><option>Q8_0</option><option>F16</option></select><button class="big" onclick="gguf('${P}','${x.merged?'merged':'full'}','q-${btoa(x.run).replace(/=/g,'')}')">2. Сохранить в GGUF для телефона</button>`:''}</div></div>`}).join(''):'<p class="hint">Пока нет обученных моделей.</p>';
 const e=await api('exports');$('exports').innerHTML=e.map(f=>`<div class="job">📱 ${esc(f.path)} — ${(f.size/1e6).toFixed(0)} МБ ${window.uchitel?`<button onclick="uchitel.reveal('${esc(f.path.replace(/\\/g,'\\\\'))}')">Показать в папке</button>`:''}</div>`).join('')||'<p class="hint">Нет.</p>'}
async function merge(run){const j=await api('merge',{run});watchJob(j.id)}
async function gguf(run,sub,sel){const sep=run.includes('\\')?'\\':'/';const j=await api('gguf',{model_dir:run+sep+sub,quant:$(sel).value,name:run.split(/[\\/]/).pop()});watchJob(j.id)}

async function loadChat(){$('c-a').value=$('c-a').value||ST.model;const r=await api('runs');const sep=s=>s.includes('\\')?'\\':'/';
 $('c-b').innerHTML=r.flatMap(x=>[x.merged&&[x.run+sep(x.run)+'merged','объединённая: '+x.run],x.adapter&&[x.run+sep(x.run)+'adapter','LoRA: '+x.run],x.full&&[x.run+sep(x.run)+'full','полная: '+x.run]].filter(Boolean)).map(([v,t])=>`<option value="${esc(v)}">${esc(t)}</option>`).join('')}
async function ask(){const q=$('c-q').value.trim();if(!q)return;const msgs=[{role:'user',content:q}];$('c-out-a').textContent='думаю… (первый раз модель загружается)';$('c-out-b').textContent='ждёт…';
 try{$('c-out-a').textContent=(await api('chat',{model:$('c-a').value,messages:msgs,temp:0.3})).text}catch(e){$('c-out-a').textContent='Ошибка: '+e.message}
 if(!$('c-b').value){$('c-out-b').textContent='Нет обученной модели';return}$('c-out-b').textContent='думаю…';
 try{$('c-out-b').textContent=(await api('chat',{model:$('c-b').value,messages:msgs,temp:0.3})).text}catch(e){$('c-out-b').textContent='Ошибка: '+e.message}}

let jt=null;async function loadJobs(){const j=await api('jobs');$('jobs-list').innerHTML=j.reverse().map(x=>`<div class="job" onclick="openJob('${x.id}')"><b>${esc(x.title)}</b> — ${({running:'идёт',done:'✓ готово',error:'✗ ошибка',paused:'пауза',stopped:'остановлено'})[x.state]} ${x.text?'· '+esc(x.text):''}</div>`).join('')||'<p class="hint">Задач пока не было.</p>'}
function openJob(id){clearInterval(jt);const f=async()=>{const j=await api('job?id='+id);$('job-log').textContent=`${j.title}\n${j.state==='running'?Math.round(100*j.progress)+'% '+(j.text||''):''}\n\n`+j.logs.join('\n');$('job-log').scrollTop=1e9;loadJobs();if(j.state!=='running'){clearInterval(jt);if(document.querySelector('#env.on'))loadEnv()}};f();jt=setInterval(f,1200)}

loadHw().catch(e=>$('hw-box').textContent='Ошибка: '+e.message);
if(location.hash.length>1)setTimeout(()=>show(location.hash.slice(1)),300);
