"""User-requested complexity and inference measurements; never reads a dataset."""
import argparse,collections,datetime,gc,json,resource,statistics,subprocess,time
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.flop_counter import FlopCounterMode,flop_registry
from torch.utils._python_dispatch import TorchDispatchMode
from experiment_runtime.support import *
class OpAudit(TorchDispatchMode):
 def __init__(self):super().__init__();self.counts=collections.Counter()
 def __torch_dispatch__(self,func,types,args=(),kwargs=None):
  self.counts[str(func.overloadpacket)]+=1
  return func(*args,**(kwargs or {}))
class Predict(torch.nn.Module):
 def __init__(self,model):super().__init__();self.model=model
 def forward(self,x):
  y=self.model(x)
  if isinstance(y,(tuple,list)):y=y[0]
  if y.shape[-2:]!=x.shape[-2:]:y=F.interpolate(y,size=x.shape[-2:],mode='bilinear',align_corners=True)
  return y

def main():
 a=argparse.ArgumentParser();a.add_argument('model',choices=list(MODELS));args=a.parse_args();name=args.model
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU must be exclusively idle before profile process creates CUDA context'
 cfg=config_for(name);seed(304);model,_=build(cfg)
 total=sum(p.numel() for p in model.parameters());removed={}
 if name.startswith('pidnet_'):
  model.augment=False
  names=['seghead_p','seghead_d']
 elif name=='pspnet':model.augment=False;model.aux_branch=False;names=['auxiliary_branch']
 elif name=='bisenet':model.augment=False;names=['conv_out16']
 else:names=[]
 for n in names:
  removed[n]=sum(p.numel() for p in getattr(model,n).parameters());delattr(model,n)
 inference_params=sum(p.numel() for p in model.parameters());assert inference_params+sum(removed.values())==total
 model=Predict(model).eval().cuda();torch.backends.cudnn.benchmark=True;torch.backends.cudnn.deterministic=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 x=torch.randn(1,3,512,512,device='cuda',dtype=torch.float32);op=OpAudit()
 with torch.inference_mode():
  with FlopCounterMode(display=False) as counter,op:y=model(x)
  assert list(y.shape)==[1,5,512,512] and bool(torch.isfinite(y).all());flops=counter.get_total_flops();assert flops>0
  details={str(k):int(v) for k,v in counter.flop_counts['Global'].items()};assert sum(details.values())==flops
  del y,counter;gc.collect()
  for _ in range(100):y=model(x)
  torch.cuda.synchronize();del y;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
  start_alloc=torch.cuda.memory_allocated();rounds=[]
  for rnd in range(3):
   starts=[torch.cuda.Event(enable_timing=True) for _ in range(100)];ends=[torch.cuda.Event(enable_timing=True) for _ in range(100)]
   torch.cuda.synchronize();t=time.perf_counter()
   for i in range(100):
    starts[i].record();y=model(x);ends[i].record();del y
   torch.cuda.synchronize();wall=time.perf_counter()-t
   times=[float(a.elapsed_time(b)) for a,b in zip(starts,ends)]
   rounds.append(dict(round=rnd+1,latencies_ms=times,mean_ms=statistics.mean(times),median_ms=statistics.median(times),synchronized_wall_seconds=wall,wall_fps=100/wall))
  peak_alloc=torch.cuda.max_memory_allocated();peak_reserved=torch.cuda.max_memory_reserved()
 latency=[x for r in rounds for x in r['latencies_ms']];lat_sorted=sorted(latency);mean=statistics.mean(latency)
 report=dict(model=name,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),total_training_parameters=total,inference_resident_parameters=inference_params,removed_training_only_modules=removed,GFLOPs=flops/1e9,counted_FLOPs=flops,GMACs_equivalent=flops/2e9,FLOPs_by_operator=details,uncounted_operators={k:v for k,v in op.counts.items() if k not in {str(x) for x in flop_registry}},latency_mean_ms=mean,latency_median_ms=statistics.median(latency),latency_p95_ms=lat_sorted[int(.95*(len(lat_sorted)-1))],FPS_from_mean_cuda_latency=1000/mean,wall_FPS=len(latency)/sum(r['synchronized_wall_seconds'] for r in rounds),peak_allocated_bytes=peak_alloc,peak_reserved_bytes=peak_reserved,baseline_allocated_bytes=start_alloc,peak_increment_above_model_input_bytes=peak_alloc-start_alloc,process_peak_cpu_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,rounds=rounds,protocol=PLAN['efficiency_profiling'],torch_version=torch.__version__,cuda_version=torch.version.cuda,device=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],text=True).strip(),precision=dict(parameter_dtype='torch.float32',autocast=False,TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32),data_samples_used=0)
 out=P.parents[1]/'new_profiles/udd';out.mkdir(parents=True,exist_ok=True);save_json(out/(name+'.json'),report)
 print(json.dumps({k:report[k] for k in ['model','total_training_parameters','inference_resident_parameters','GFLOPs','latency_mean_ms','FPS_from_mean_cuda_latency','peak_allocated_bytes']}),flush=True)
if __name__=='__main__':main()
