"""Measure physical CUDA microbatches without changing checkpoint weights."""
import gc
import argparse
import json
import math
import time
import torch
import ensemble_repair as er
import offline_ensemble as oe
import supervised_ensemble as se
from train_ensemble_repair import OUT,corpus,write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--target-batch',type=int,choices=[16,32,64,128]);args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(972001)
    _,groups,_,_=corpus()
    model,adapters,_,planner,conditions=er.load(OUT/'progress.pt')
    base_hash=adapters.base_hash();adapters.activate('organ',training=True);conditions.train()
    candidates=sorted([w for ws in groups['train']['organ'].values() for w in ws],key=lambda w:len(w['notes']),reverse=True)
    examples=[]
    required=max(32,args.target_batch or 32)
    for w in candidates[:32]:
        for beat in range(24,32):
            with oe.limits():ex=se.teacher_example(model,'organ',{**w,'beat':beat,'sampler_type':oe.Sampler})
            if ex is not None:examples.append(ex)
            if len(examples)>=required:break
        if len(examples)>=required:break
    if len(examples)<required:raise ValueError(f'Need {required} stress examples')
    # Include mixed context lengths for the right-padding equivalence check.
    test_examples=[examples[0],examples[7],examples[16],examples[31]]
    delta=torch.zeros(1024,device=model.device)
    def clear():
        model.zero_grad(set_to_none=True);conditions.zero_grad(set_to_none=True)
    clear();serial=[]
    for ex in test_examples:
        losses=er.category_losses(oe.ConditionedModel(model,delta),ex,training=True)
        serial.append({k:float(v.detach()) for k,v in losses.items()})
        (sum(losses.values())/len(losses)/len(test_examples)).backward()
    serial_grad=torch.cat([p.grad.float().flatten() for p in adapters.params('organ') if p.grad is not None]).detach().clone()
    clear();batched=er.batch_category_losses(model,test_examples,[delta]*4,training=True)
    error=max(abs(float(v.detach())-serial[i][k]) for i,row in enumerate(batched) for k,v in row.items())
    sum(sum(row.values())/len(row)/4 for row in batched).backward()
    batch_grad=torch.cat([p.grad.float().flatten() for p in adapters.params('organ') if p.grad is not None])
    gradient_relative=float((batch_grad-serial_grad).norm()/serial_grad.norm().clamp_min(1e-9))
    if error>.1 or gradient_relative>.15:raise ValueError(f'Batched replay differs: loss={error}, gradient={gradient_relative}')
    del batched,batch_grad,serial_grad,losses;clear();gc.collect();torch.cuda.empty_cache()
    free,total=torch.cuda.mem_get_info();other=total-free-torch.cuda.memory_reserved()
    safe_peak=total-other-768*1024**2
    results=[]
    for size in ((16,32,64,128) if args.target_batch==128 else (1,4,8,16,32)):
        try:
            durations=[];peaks=[]
            for repeat in range(3):
                clear();gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                started=time.perf_counter()
                deltas=[conditions(torch.zeros(48,device=model.device),torch.zeros(1,8,device=model.device),'organ') for _ in range(size)]
                losses=er.batch_category_losses(model,examples[:size],deltas,training=True)
                loss=sum(sum(row.values())/len(row)/size for row in losses);loss.backward()
                torch.cuda.synchronize();durations.append(time.perf_counter()-started);peaks.append(torch.cuda.max_memory_reserved())
                del deltas,losses,loss
            seconds=sum(durations[1:])/2;peak=max(peaks)
            results.append(dict(microbatch=size,seconds=seconds,examples_per_second=size/seconds,
                                peak_reserved_mib=peak/1024**2,safe=peak<=safe_peak))
        except torch.cuda.OutOfMemoryError:
            results.append(dict(microbatch=size,oom=True,safe=False));clear();gc.collect();torch.cuda.empty_cache()
            break
        print(results[-1],flush=True)
        write_json(OUT/'batch_benchmark_partial.json',results)
    eligible=[r for r in results if r['safe']]
    if not eligible:raise ValueError('No safe batch found')
    speed=max(r['examples_per_second'] for r in eligible)
    chosen=max((r for r in eligible if r['examples_per_second']>=.95*speed),key=lambda r:r['microbatch'])
    if args.target_batch:
        chosen=max((r for r in eligible if r['microbatch']<=args.target_batch),key=lambda r:r['microbatch'])
    if adapters.base_hash()!=base_hash:raise ValueError('Backbone changed during benchmark')
    result=dict(results=results,selected_microbatch=chosen['microbatch'],phrases_per_role=max(2,math.ceil(chosen['microbatch']/8)),
                max_loss_difference=error,gradient_relative_difference=gradient_relative,backbone_unchanged=True,
                minimum_headroom_mib=768,max_context_events=max(len(e['x']) for e in examples))
    filename=f'batch_benchmark_{args.target_batch}.json' if args.target_batch else 'batch_benchmark.json'
    write_json(OUT/filename,result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
