"""Manual entry with portable paths and the archived training recipe.

The organizer does not run this script. A compatible training environment is required.
"""
from pathlib import Path
import argparse, json, os, runpy, sys

ROOT=Path(__file__).resolve().parent
MODELS=['unet','deeplabv3plus','hrnet','pspnet','bisenet','pidnet_m',
        'pidnet_m_ca','pidnet_m_ca_dsc','pidnet_m_ca_dcu','pidnet_m_ca_dsc_dcu']
def main():
    p=argparse.ArgumentParser(description='Manually start ONE new run; never overwrite archived results.')
    p.add_argument('--dataset',choices=['caihua','udd'],required=True)
    p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--seed',type=int,choices=[304,34,3004],required=True)
    p.add_argument('--fold',type=int,choices=range(1,6))
    p.add_argument('--gpu',type=int,default=0,help='Physical CUDA device index for this single-GPU run (default: 0).')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--print-only',action='store_true',help='Show the resolved paths without importing torch or starting training.')
    a=p.parse_args();out=a.output.expanduser().resolve()
    if a.gpu<0:p.error('--gpu must be a non-negative device index.')
    # Archived material is immutable. Only new output directories are accepted.
    for name in ['datasets','code','configs','pretrained','environment','inference_runtime']:
        protected=(ROOT/name).resolve()
        if out==protected or protected in out.parents:p.error('Output must not be inside archived material: '+str(protected))
    if out.exists():p.error('Choose a new, non-existing output directory.')
    if a.dataset=='caihua':
        if a.seed not in [304,34,3004] or a.fold is None:p.error('Caihua requires seed 304/34/3004 and --fold 1..5.')
        model='hrnet_os8' if a.model=='hrnet' else a.model
        code=ROOT/'code/caihua'
        cfg=ROOT/'configs/caihua'/model/('rs%d.yaml'%a.fold)
        command=[sys.executable,str(code/'tools/train.py'),'--cfg',str(cfg),'--seed',str(a.seed),
                 'OUTPUT_DIR',str(out),'LOG_DIR',str(out.parent/(out.name+'_logs'))]
        description={'dataset':a.dataset,'split':str(ROOT/'datasets/caihua'/('rs%d'%a.fold)),'pretrained':str(ROOT/'pretrained/caihua'),'command':command,'training_recipe_unchanged':True}
    else:
        if a.fold is not None:p.error('UDD uses one fixed split, seed 304/34/3004, no --fold.')
        code=ROOT/'code/udd'
        current=ROOT/'datasets';frozen=current/'udd'
        weights=ROOT/'pretrained/udd'
        command=[sys.executable,str(code/'tools/train.py'),'--model',a.model,'--seed',str(a.seed),'--out',str(out)]
        description={'dataset':a.dataset,'entry':str(code/'tools/train.py'),'model':a.model,'seed':a.seed,
                     'current':str(current),'frozen':str(frozen),'pretrained':str(weights),
                     'output':str(out),'path_binding':'project-relative dataset and pretrained directories','training_recipe_unchanged':True}
    if not (code/'tools/train.py').is_file():p.error('Missing training entry.')
    if a.dataset=='caihua' and not cfg.is_file():p.error('Missing training config.')
    split=ROOT/'datasets/caihua'/('rs%d'%a.fold) if a.dataset=='caihua' else frozen
    if not (split/'manifest.json').is_file():p.error('Missing frozen split.')
    description['gpu']={'physical_index':a.gpu,'visible_devices':str(a.gpu),'logical_index':0,'device_count':1}
    if a.print_only:
        print(json.dumps(description,ensure_ascii=False,indent=2));return
    if sys.version_info<(3,9):p.error('Original source requires Python >=3.9; historical training used Python 3.12.3. Current Python can still browse/verify the project.')
    # The frozen Linux code uses UTF-8 JSON and some implicit text opens. Select UTF-8
    # at interpreter startup on Windows without changing the frozen source or system locale.
    if not sys.flags.utf8_mode:
        os.execv(sys.executable,[sys.executable,'-X','utf8',str(Path(__file__).resolve())]+sys.argv[1:])
    os.environ.setdefault('PYTHONUTF8','1')
    # Select one physical device before either dataset implementation imports torch.
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    if a.dataset=='caihua':
        sys.argv=command[1:]
        runpy.run_path(str(code/'tools/train.py'),run_name='__main__')
    else:
        sys.path.insert(0,str(code));os.chdir(str(code))
        from experiment_runtime import support
        support.CURRENT=current;support.FROZEN=frozen;support.WEIGHTS=weights
        # Import after assigning bindings so campaign_train's wildcard import sees the relocated paths.
        from experiment_runtime import train
        sys.argv=command[1:]
        train.main()

if __name__=='__main__':main()
