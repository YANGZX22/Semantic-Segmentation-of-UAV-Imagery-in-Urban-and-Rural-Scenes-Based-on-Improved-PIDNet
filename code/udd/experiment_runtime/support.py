"""Explicit routing, frozen inputs and observation for the authorized UDD campaign."""
from pathlib import Path
import importlib,json,logging,os,random
import numpy as np
import torch
import torch.utils.model_zoo as modelzoo
P=Path(__file__).resolve().parents[1]
BASE=P.parents[1];CURRENT=BASE/'datasets';FROZEN=CURRENT/'udd';WEIGHTS=BASE/'pretrained/udd'
PLAN=json.loads((P/'plan.json').read_text());MODELS={m['id']:m for m in PLAN['models']}
PRETRAINED={'unet':'','deeplabv3plus':'mobilenet_v2-b0353104.pth','pspnet':'mobilenet_v2-b0353104.pth','bisenet':'resnet18-5c106cde.pth','hrnet':'hrnet_w18_small_model_v2.pth'}
def save_json(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def config_for(name):
 assert name in MODELS,'Model is outside the paper scope'
 from configs import config
 from configs.default import _C
 global DEFAULT_CONFIG
 if 'DEFAULT_CONFIG' not in globals():DEFAULT_CONFIG=_C.clone()
 cfg=config;cfg.defrost();cfg.clear();cfg.update(DEFAULT_CONFIG.clone())
 cfg.merge_from_file(str(P/'configs/udd'/('pidnet_medium_udd.yaml' if name.startswith('pidnet_') else name+'_udd.yaml')))
 cfg.MODEL.NAME=name;cfg.DATASET.ROOT=str(FROZEN);cfg.DATASET.TRAIN_SET='train.lst';cfg.DATASET.TEST_SET='val.lst';cfg.DATASET.DATASET='udd'
 pre='imagenet/PIDNet_M_ImageNet.pth.tar' if name.startswith('pidnet_') else PRETRAINED[name]
 cfg.MODEL.PRETRAINED=str(WEIGHTS/pre) if pre else ''
 cfg.LOSS.SB_WEIGHTS=0.0
 assert cfg.TRAIN.END_EPOCH==80 and cfg.TRAIN.BATCH_SIZE_PER_GPU==8 and not cfg.TRAIN.RESUME
 assert cfg.TEST.OUTPUT_INDEX==(1 if name.startswith('pidnet_') else 0)
 cfg.freeze();return cfg

def build(cfg):
 name=cfg.MODEL.NAME;pid=name.startswith('pidnet_');modname=MODELS[name]['module'] if pid else {'hrnet':'seg_hrnet'}.get(name,name)
 module=importlib.import_module('models.'+modname)
 original_load=torch.load;original_url=modelzoo.load_url;loads=[]
 def load(path,*args,**kwargs):
  kwargs.setdefault('weights_only',False)
  p=Path(path).resolve();assert p.is_relative_to(WEIGHTS.resolve()),str(p)
  loads.append(dict(path=str(p)));return original_load(path,*args,**kwargs)
 def url(url,*args,**kwargs):
  options={'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth':'mobilenet_v2-b0353104.pth','https://download.pytorch.org/models/resnet18-5c106cde.pth':'resnet18-5c106cde.pth'}
  assert url in options,url;return load(WEIGHTS/options[url],map_location='cpu')
 torch.load=load;modelzoo.load_url=url
 try:
  if pid:
   model=module.PIDNet(m=2,n=3,num_classes=5,planes=64,ppm_planes=96,head_planes=128,augment=True)
   pretrained=load(cfg.MODEL.PRETRAINED,map_location='cpu')['state_dict'];state=model.state_dict();matched={k:v for k,v in pretrained.items() if k in state and state[k].shape==v.shape};assert len(matched)==302
   state.update(matched);model.load_state_dict(state,strict=True);logging.info('Loaded %d parameters!',len(matched));assert model.conv1[0].out_channels==64
  else:model=module.get_seg_model(cfg,imgnet_pretrained=bool(cfg.MODEL.PRETRAINED))
 finally:torch.load=original_load;modelzoo.load_url=original_url
 if name=='hrnet':
  from models.seg_hrnet_OS8 import HighResolutionNet
  model.__class__=HighResolutionNet
 loaded={}
 if cfg.MODEL.PRETRAINED:
  source=original_load(cfg.MODEL.PRETRAINED,map_location='cpu',weights_only=False);source=source.get('state_dict',source)
  if 'model' in source:source=source['model']
  if pid or name=='hrnet':target=model
  elif name=='bisenet':
   target=model.cp.resnet;source={k.replace('module.','').replace('resnet.',''):v for k,v in source.items() if 'fc' not in k}
  else:
   target=model.backbone
  state=target.state_dict();matched={k:v for k,v in source.items() if k in state and state[k].shape==v.shape}
  assert matched and all(torch.equal(v,state[k]) for k,v in matched.items()),'Pretrained tensors not loaded'
  required=[k for k in state if k.startswith(('conv1.','layer1.','layer2.'))] if pid else [k for k in state if not k.startswith('last_layer.')]
  missing=[k for k in required if k not in matched and not k.endswith('num_batches_tracked')];assert not missing,missing
  loaded=dict(matched_keys=len(matched),matched_tensors_equal_source=True,required_missing=missing)
 else:assert name=='unet' and not loads
 result=dict(model=name,parameters=sum(v.numel() for v in model.parameters()),state_bytes=sum(v.numel()*v.element_size() for v in model.state_dict().values()),pretrained=loaded,load_calls=loads,configuration=cfg.dump())
 return model,result

def guard_data(full=True):
 m=json.loads((FROZEN/'manifest.json').read_text());counts={}
 for role,row in m['lists'].items():
  p=FROZEN/row['path'];counts[role]=len(p.read_text().splitlines())
 assert counts==PLAN['split_sizes']
 if full:
  for s in m['samples']:
   for k in ('image','label'):
    parts=Path(s[k]).parts;assert parts and parts[0]=='datasets_clean'
    source=CURRENT.joinpath(*parts[1:]).resolve();assert CURRENT.resolve() in source.parents
    assert source.is_file(),'Missing UDD tile: '+str(source)
 return dict(counts=counts,files_checked=18432 if full else 0,test_evaluated=False)

def observe_forward(model,name,folder):
 shapes={};handles=[]
 def tap(k):
  def hook(module,inputs,out):shapes[k]=list(out.shape)
  return hook
 keys=['layer2','seghead_p.conv2','final_layer.conv2','seghead_d.conv2'] if name.startswith('pidnet_') else {'hrnet':['last_layer']}.get(name,[])
 for k in keys:handles.append(model.get_submodule(k).register_forward_hook(tap(k)))
 def complete(module,inputs,outs):
  shapes['input']=list(inputs[0].shape);outputs=list(outs) if isinstance(outs,(list,tuple)) else [outs];shapes['outputs']=[list(x.shape) for x in outputs]
  assert module.training and shapes['input'][-2:]==[512,512]
  for k in keys:assert shapes[k][-2:]==[64,64],(k,shapes[k])
  if name.startswith('pidnet_'):assert len(outputs)==3 and outputs[-1].shape[1]==1
  save_json(folder/'formal_first_forward.json',dict(model=name,shapes=shapes,training=True,extra_forward=False))
  for h in handles:h.remove()
 handles.append(model.register_forward_hook(complete))

def metric(cm):
 a=np.array(cm,dtype=np.int64);assert a.shape==(5,5) and int(a.sum())==1170*512*512
 pos=a.sum(1);pred=a.sum(0);tp=a.diagonal();iou=tp/np.maximum(1,pos+pred-tp)
 return dict(mIoU=float(iou.mean()),OA=float(tp.sum()/a.sum()),class_IoU=iou.tolist(),class_names=['Building','Vegetation','Road','Vehicle','Background'],confusion_matrix=a.tolist(),valid_pixels=int(a.sum()))
