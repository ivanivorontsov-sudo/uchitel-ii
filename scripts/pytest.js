// Кроссплатформенный запуск проверок Python: python3 на Linux/macOS, python на Windows.
const {spawnSync}=require('child_process');
const cands=process.platform==='win32'?['python','py','python3']:['python3','python'];
const py=cands.find(c=>{const r=spawnSync(c,['--version']);return r.status===0});
if(!py){console.error('Python не найден');process.exit(1)}
for(const a of [['-m','compileall','-q','backend'],['-m','unittest','discover','-s','tests','-v']]){
  const r=spawnSync(py,a,{stdio:'inherit'}); if(r.status!==0) process.exit(r.status||1)}
