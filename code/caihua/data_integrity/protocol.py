"""Fail-closed data protocol checks shared by both active training projects."""
from pathlib import Path
from itertools import combinations
import json, math
ROOT = Path(__file__).resolve().parents[3] / "datasets"
ROLES = ('train', 'val', 'test')
class IntegrityError(RuntimeError): pass

def require(condition, message):
    if not condition: raise IntegrityError(message)

def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def resolve_source(path):
    # Manifests keep their original source paths; only the historical storage
    # prefix is translated to the public layout.
    parts=Path(path).parts
    require(bool(parts) and parts[0]=='datasets_clean', 'Unrecognized source path: '+str(path))
    parts=list(parts[1:])
    if parts and parts[0]=='udd_global_preserved_v3': parts[0]='udd'
    p=ROOT.joinpath(*parts).resolve()
    require(ROOT in p.parents, 'Source path escapes datasets: '+str(p))
    return p

def rectangle_gap(a,b):
    dx=max(a['x']-b['x']-b['w'], b['x']-a['x']-a['w'], 0)
    dy=max(a['y']-b['y']-b['h'], b['y']-a['y']-a['h'], 0)
    overlap=(a['x']<b['x']+b['w'] and b['x']<a['x']+a['w']
             and a['y']<b['y']+b['h'] and b['y']<a['y']+a['h'])
    return overlap, math.hypot(dx,dy)

def audit_membership(m):
    require(m.get('schema')==1 and m.get('protocol')=='spatial_disjoint_v1', 'Unapproved data protocol')
    require(m['dataset'] in ('caihua','udd'), 'Unknown dataset')
    rows=m['samples']; by={role:[r for r in rows if r['split']==role] for role in ROLES}
    require(sum(len(v) for v in by.values())==len(rows),'Invalid role')
    require(len({r['id'] for r in rows})==len(rows),'Duplicate sample ID')
    for role,rs in by.items(): require(bool(rs),'Empty '+role)
    result={}
    for a,b in combinations(ROLES,2):
        ra,rb=by[a],by[b]
        for key in ('parent','group','image'):
            require(not ({x[key] for x in ra}&{x[key] for x in rb}), 'Cross-split '+key+' collision: '+a+'/'+b)
        if m['dataset']=='caihua':
            minimum=float('inf')
            for x in ra:
                for y in rb:
                    overlap,gap=rectangle_gap(x['rect'],y['rect'])
                    require(not overlap and gap>=m['minimum_buffer_pixels'], 'Spatial overlap/buffer violation: '+x['id']+' / '+y['id'])
                    minimum=min(minimum,gap)
            result[a+'__'+b]={'minimum_gap_pixels':minimum,'overlaps':0}
        else: result[a+'__'+b]={'parent_collisions':0,'cohort_component_collisions':0}
    stats=m['training_statistics']
    require(stats['source_role']=='train', 'Statistics must use train only')
    require(stats['sample_ids']==sorted(r['id'] for r in by['train']), 'Statistics membership differs from train')
    return result

def audit(root):
    root=Path(root).resolve(); mp=root/'manifest.json'
    require(mp.is_file(), 'Missing clean manifest. Legacy/random splits are disabled: '+str(root))
    m=read_json(mp); result=audit_membership(m)
    # The source registry binds group/geometry and files to the builder's evidence.
    registry=read_json(resolve_source(m['registry']))
    originals={r['id']:r for r in registry['samples']}
    for row in m['samples']:
        require(row['id'] in originals,'Unknown sample')
        original=originals[row['id']]
        require({k:v for k,v in row.items() if k!='split'}==original,'Source provenance changed: '+row['id'])
    for role in ROLES:
        spec=m['lists'][role]; lp=(root/spec['path']).resolve()
        require(lp.parent==root, 'List must be inside split root')
        require(lp.is_file(), 'Missing '+role+' list')
        expected=[]
        for row in m['samples']:
            if row['split']==role:
                import os
                expected.append(os.path.relpath(resolve_source(row['image']),root).replace('\\','/')+' '+os.path.relpath(resolve_source(row['label']),root).replace('\\','/'))
        require(lp.read_text(encoding='utf-8').splitlines()==expected,'List membership mismatch: '+role)
    for row in m['samples']:
        for kind in ('image','label'):
            p=resolve_source(row[kind])
            require(p.is_file(),'Missing source: '+str(p))
    return m,result

def open_split(root, list_path, dataset, augment=False):
    # No bypass switch: all three partitions are checked even when opening one.
    m,_=audit(root)
    require(m['dataset']==dataset,'Dataset/manifest mismatch')
    p=(Path(root)/list_path).resolve()
    roles=[r for r,s in m['lists'].items() if (Path(root)/s['path']).resolve()==p]
    require(len(roles)==1,'List is not a frozen train/val/test partition')
    role=roles[0]
    require(not augment or role=='train','Augmentation is restricted to training data')
    return m,role

def training_preflight(cfg):
    m,role=open_split(cfg.DATASET.ROOT,cfg.DATASET.TRAIN_SET,cfg.DATASET.DATASET.lower())
    require(role=='train','Training must use train partition')
    val=(Path(cfg.DATASET.ROOT)/cfg.DATASET.TEST_SET).resolve()
    require(val==(Path(cfg.DATASET.ROOT)/m['lists']['val']['path']).resolve(),
            'Model selection must use val; test is reserved for final evaluation')
    require(cfg.DATASET.NUM_CLASSES==m['num_classes'],'Class count mismatch')
    require(cfg.TRAIN.IGNORE_LABEL==255,'Clean labels require ignore_label=255')
    require(not getattr(cfg,'AUTO_RESUME',False),'Automatic legacy checkpoint resume is disabled')
    pre=str(cfg.MODEL.PRETRAINED).replace('\\','/').lower()
    require(not any(t in pre for t in ('best.pt','checkpoint','final_state','spatial_kfold','dataset_split_cut')),
            'A dataset-finetuned legacy model cannot initialize a clean experiment')
    return {'protocol':m['protocol'],'dataset':m['dataset'],'split_id':m['split_id'],
            'selection_role':'val','test_role':'held_out'}

def write_run_stamp(output, stamp):
    p=Path(output)/'data_protocol.json';p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists(): require(read_json(p)==stamp,'Output directory belongs to a different split')
    else:
        require(not any((p.parent/n).exists() for n in ('checkpoint.pth.tar','best.pt','final_state.pt')),
                'Unstamped legacy weights in output directory; use a new directory')
        p.write_text(json.dumps(stamp,indent=2,sort_keys=True)+'\n',encoding='utf-8')

def validate_resume(checkpoint, stamp):
    require(checkpoint.get('data_protocol')==stamp,'Checkpoint has no matching clean split provenance; restart from external pretraining')

def validate_evaluation(weight, root):
    m,_=audit(root)
    p=Path(weight).parent/'data_protocol.json'
    require(p.is_file(),'Unstamped legacy checkpoint cannot be evaluated as a clean result')
    stamp=read_json(p)
    require(stamp.get('dataset')==m['dataset'] and stamp.get('split_id')==m['split_id'],'Checkpoint was trained on another split')
    return stamp
