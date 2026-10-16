"""Formal training only. Original implementations remain separate."""
import argparse,datetime,importlib,json,logging,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
from tensorboardX import SummaryWriter
from experiment_runtime.support import *

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic_torch(value,path):
 tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);os.replace(tmp,path)
def main():
 a=argparse.ArgumentParser();a.add_argument('--model',choices=list(MODELS),required=True);a.add_argument('--seed',type=int,choices=PLAN['seed_order'],required=True);a.add_argument('--out',type=Path,required=True);args=a.parse_args()
 out=args.out;out.mkdir(parents=True,exist_ok=False)
 logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'train.log'),logging.StreamHandler()])
 data_guard=guard_data();save_json(out/'data_guard.json',data_guard)
 cfg=config_for(args.model);(out/'effective_config.yaml').write_text(cfg.dump());seed(args.seed)
 writer={'writer':SummaryWriter(str(out/'tensorboard')),'train_global_steps':0,'valid_global_steps':0}
 torch.backends.cudnn.benchmark=cfg.CUDNN.BENCHMARK;torch.backends.cudnn.deterministic=cfg.CUDNN.DETERMINISTIC;torch.backends.cudnn.enabled=cfg.CUDNN.ENABLED
 model,initial=build(cfg);initial.update(seed=args.seed,utc=utc());save_json(out/'formal_initialization.json',initial)
 ispid=args.model.startswith('pidnet_');isarchive=ispid or args.model=='unet'
 UDD=importlib.import_module('datasets.udd_pidnet_trainstats' if ispid else 'datasets.udd').UDD
 common=dict(root=cfg.DATASET.ROOT,num_classes=5,ignore_label=cfg.TRAIN.IGNORE_LABEL)
 trainset=UDD(**common,list_path=cfg.DATASET.TRAIN_SET,multi_scale=cfg.TRAIN.MULTI_SCALE,flip=cfg.TRAIN.FLIP,base_size=cfg.TRAIN.BASE_SIZE,crop_size=(512,512),scale_factor=cfg.TRAIN.SCALE_FACTOR)
 trainloader=torch.utils.data.DataLoader(trainset,batch_size=8,shuffle=cfg.TRAIN.SHUFFLE,num_workers=cfg.WORKERS,pin_memory=isarchive,drop_last=True)
 valset=UDD(**common,list_path=cfg.DATASET.TEST_SET,multi_scale=False,flip=False,base_size=cfg.TEST.BASE_SIZE,crop_size=(512,512))
 valloader=torch.utils.data.DataLoader(valset,batch_size=cfg.TEST.BATCH_SIZE_PER_GPU,shuffle=False,num_workers=cfg.WORKERS,pin_memory=isarchive)
 assert len(trainset)==6912 and len(valset)==1170 and len(trainloader)==864
 cr=importlib.import_module('utils.criterion' if isarchive else 'utils.comparator_criterion')
 cargs=dict(ignore_label=cfg.TRAIN.IGNORE_LABEL,weight=trainset.class_weights)
 if cfg.LOSS.USE_OHEM:cargs.update(thres=cfg.LOSS.OHEMTHRES,min_kept=cfg.LOSS.OHEMKEEP)
 criterion=(cr.OhemCrossEntropy if cfg.LOSS.USE_OHEM else cr.CrossEntropy)(**cargs)
 wrapper=importlib.import_module('utils.utils' if ispid else 'utils.comparator_utils').FullModel
 full=wrapper(model,criterion,cr.BondaryLoss())
 stats=dict(class_weights=trainset.class_weights.detach().cpu().tolist(),mean=list(trainset.mean),std=list(trainset.std),SB_weight=cfg.LOSS.SB_WEIGHTS,semantic_balance=list(cfg.LOSS.BALANCE_WEIGHTS),validation_main_index=cfg.TEST.OUTPUT_INDEX)
 save_json(out/'formal_data_and_loss.json',stats)
 if ispid:
  assert stats['semantic_balance']==[.4,1.] and stats['SB_weight']==0 and stats['validation_main_index']==1
 observe_forward(model,args.model,out)
 observed=[False]
 def loss_observer(module,inputs,result):
  assert bool(torch.isfinite(result[0]).all()),'Nonfinite loss'
  if not observed[0] and module.training:
   total=float(result[0].detach().mean());sem=float(result[3][0].detach().mean());bd=float(result[3][1].detach().mean());res=total-sem-bd
   if ispid:assert abs(res)<1e-4,'SB=0 residual'
   save_json(out/'formal_first_loss.json',dict(total=total,semantic=sem,boundary=bd,residual=res,SB_config=cfg.LOSS.SB_WEIGHTS,formal_training=True));observed[0]=True
 full.register_forward_hook(loss_observer)
 full=torch.nn.DataParallel(full,device_ids=list(cfg.GPUS)).cuda()
 if cfg.TRAIN.OPTIMIZER=='sgd':
  optimizer=torch.optim.SGD([{'params':list(dict(full.named_parameters()).values()),'lr':cfg.TRAIN.LR}],lr=cfg.TRAIN.LR,momentum=cfg.TRAIN.MOMENTUM,weight_decay=cfg.TRAIN.WD,nesterov=cfg.TRAIN.NESTEROV)
 elif args.model=='unet':optimizer=torch.optim.AdamW(full.parameters(),lr=cfg.TRAIN.LR,weight_decay=cfg.TRAIN.WD)
 else:raise ValueError('Unsupported optimizer for paper model: '+args.model)
 save_json(out/'optimizer.json',dict(type=type(optimizer).__name__,groups=[{k:v for k,v in g.items() if k!='params'}|{'parameter_tensors':len(g['params'])} for g in optimizer.param_groups],lr_schedule='poly power 0.9 after optimizer step',total_updates=69120))
 func=importlib.import_module('utils.function' if isarchive else 'utils.comparator_function')
 valfunc=func if ispid else importlib.import_module('utils.comparator_function')
 save_json(out/'started.json',dict(utc=utc(),pid=os.getpid(),model=args.model,seed=args.seed,torch=torch.__version__,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],text=True).strip(),test_evaluated=False))
 best=0.;began=time.monotonic();memory_epochs=[]
 for ep in range(80):
  t=time.monotonic();torch.cuda.reset_peak_memory_stats();func.train(cfg,ep,80,864,cfg.TRAIN.LR,69120,trainloader,optimizer,full,writer)
  memory_epochs.append(dict(epoch=ep+1,training_peak_allocated_bytes=torch.cuda.max_memory_allocated(),training_peak_reserved_bytes=torch.cuda.max_memory_reserved()))
  save_json(out/'training_memory.json',dict(batch_size=8,input_size=[512,512],precision='float32',peak_allocated_bytes=max(x['training_peak_allocated_bytes'] for x in memory_epochs),peak_reserved_bytes=max(x['training_peak_reserved_bytes'] for x in memory_epochs),epochs=memory_epochs,scope='training forward+loss+backward+optimizer step; PyTorch CUDA allocator; not total process GPU memory'))
  if ispid:
   cms=np.zeros((2,5,5),dtype=np.int64);calls=[0];original=func.get_confusion_matrix
   def capture(*a,**kw):
    cm=original(*a,**kw);cms[calls[0]%2]+=cm.astype(np.int64);calls[0]+=1;return cm
   func.get_confusion_matrix=capture
   try:valid=func.validate(cfg,valloader,full,writer)
   finally:func.get_confusion_matrix=original
   assert calls[0]==2*len(valloader);cm=cms[1]
  else:
   valid=valfunc.validate(cfg,valloader,full,writer);cm=writer['last_confusion_matrices'][cfg.TEST.OUTPUT_INDEX]
  met=metric(cm);assert abs(met['mIoU']-float(valid[1]))<1e-12;assert np.allclose(met['class_IoU'],valid[2],rtol=0,atol=1e-12)
  record=dict(epoch=ep+1,utc=utc(),elapsed_epoch_seconds=time.monotonic()-t,elapsed_total_seconds=time.monotonic()-began,valid_loss=float(valid[0]),**met)
  if met['mIoU']>best:
   best=met['mIoU'];atomic_torch(full.module.model.state_dict(),out/'best.pt');save_json(out/'best_validation.json',record)
  atomic_torch(dict(epoch=ep+1,best_mIoU=best,state_dict=full.module.state_dict(),optimizer=optimizer.state_dict()),out/'checkpoint.pth.tar')
  record['best_mIoU']=best;save_json(out/'epochs'/f'{ep+1:03d}.json',record);save_json(out/'status.json',dict(state='running',**record))
  logging.info('Epoch %d/80 main mIoU %.8f OA %.8f best %.8f',ep+1,100*met['mIoU'],100*met['OA'],100*best)
 atomic_torch(full.module.model.state_dict(),out/'final_state.pt');writer['writer'].close();guard_data(False)
 files={n:dict(bytes=(out/n).stat().st_size) for n in ['best.pt','checkpoint.pth.tar','final_state.pt']}
 save_json(out/'completion.json',dict(state='completed',model=args.model,seed=args.seed,epochs=80,utc=utc(),elapsed_seconds=time.monotonic()-began,weights=files,best=json.loads((out/'best_validation.json').read_text()),test_evaluated=False))
 save_json(out/'status.json',dict(state='completed',epoch=80,utc=utc(),best_mIoU=best))
if __name__=='__main__':main()
