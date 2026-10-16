import json,subprocess,os,datetime,traceback,sys
from pathlib import Path
P=Path(__file__).resolve().parents[1];plan=json.loads((P/'plan.json').read_text());out=P.parents[1]/'new_profiles/udd';out.mkdir(parents=True,exist_ok=True);tmp=out/'temp';tmp.mkdir(exist_ok=True)
status={'state':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'models':[]}
def save():
 t=out/'status.tmp';t.write_text(json.dumps(status,indent=2));t.replace(out/'status.json')
save()
try:
 for row in plan['models']:
  name=row['id'];assert not (out/(name+'.json')).exists(),'Do not silently retry or overwrite profile'
  status['current_model']=name;save()
  with (out/(name+'.log')).open('w') as log:
   r=subprocess.run([sys.executable,'-m','experiment_runtime.profile',name],cwd=P,stdout=log,stderr=subprocess.STDOUT,timeout=600,env=os.environ|{'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','TMPDIR':str(tmp)})
  assert r.returncode==0,(name,r.returncode)
  report=json.loads((out/(name+'.json')).read_text());status['models'].append(name);save();print(name,'profile done',flush=True)
 status.update(state='completed',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
except BaseException as e:
 status.update(state='failed',error=repr(e),traceback=traceback.format_exc());save();raise
