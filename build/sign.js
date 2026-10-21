// Подпись EXE самоподписанным сертификатом Учитель ИИ (ключ — вне репозитория: UCH_SIGN_PFX, UCH_SIGN_PASS_FILE)
const cp=require('child_process'), fs=require('fs');
exports.default=async function(c){ const pfx=process.env.UCH_SIGN_PFX, pf=process.env.UCH_SIGN_PASS_FILE; if(!pfx||!fs.existsSync(pfx)){ console.log('  • подпись пропущена: нет UCH_SIGN_PFX'); return }
 const pass=fs.readFileSync(pf,'utf8').trim(); const out=c.path+'.signed';
 const base=['sign','-pkcs12',pfx,'-pass',pass,'-h','sha256','-n','Учитель ИИ','-i','https://github.com/ivanivorontsov-sudo/uchitel-ii','-in',c.path,'-out',out];
 let r=cp.spawnSync('osslsigncode',[...base.slice(0,-4),'-ts','http://timestamp.digicert.com',...base.slice(-4)],{encoding:'utf8',timeout:25000});
 if(r.status!==0){ try{fs.rmSync(out,{force:true})}catch{} r=cp.spawnSync('osslsigncode',base,{encoding:'utf8'}) }
 if(r.status!==0) throw new Error('osslsigncode: '+String(r.stderr||r.stdout).replace(pass,'***').slice(0,400)); fs.renameSync(out,c.path) };
