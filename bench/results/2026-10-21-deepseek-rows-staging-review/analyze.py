"""Offline GGUF-header and existing public benchmark audit. No model execution, no weights read."""
import json
import math
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[3]
HOME=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tools'))
from gguf_reader import GGUFFile, BLOCK_GEOMETRY


def main():
    cfg=json.loads((ROOT/'strata-dsv4-q3_k_m.json').read_text())
    args=cfg['args']
    model=Path(args[args.index('--model')+1])
    gg=GGUFFile(model) # Header reader closes the file before returning; tensor payload is never read.
    arch=gg.metadata['general.architecture']
    nlayer=gg.metadata[arch+'.block_count'];ne=gg.metadata[arch+'.expert_count']
    dim=gg.metadata[arch+'.embedding_length'];streams=gg.metadata[arch+'.hyper_connection.count']
    stage=int(args[args.index('--stage-mib')+1])*2**20
    batch=int(args[args.index('--prompt-batch')+1])
    tensors={t.name:t for t in gg.tensors}
    layers=[]
    for il in range(nlayer):
        ts=[tensors[f'blk.{il}.ffn_{kind}_exps.weight'] for kind in ['gate','up','down']]
        one=[t.expected_bytes()//ne for t in ts]
        assert all(t.expected_bytes()%ne==0 for t in ts)
        raw=sum(one)
        unit=math.lcm(BLOCK_GEOMETRY[ts[0].type_name][1],BLOCK_GEOMETRY[ts[2].type_name][1],8)
        slot=(raw+256+unit-1)//unit*unit # no_disk.cpp: raw + kTail, rounded to quantization unit
        layers.append(dict(layer=il,types=[t.type_name for t in ts],raw_bytes=raw,main_slot_bytes=slot,
            full_layer_bytes=raw*ne,prefill_capacity=stage//raw,full_layer_fits=raw*ne<=stage))
    assert ne==256 and nlayer==43 and all(x['full_layer_fits'] for x in layers)
    reference=ROOT/'bench/results/2026-10-21-deepseek-bat-prefill-specials/bat-no-disk'
    summary=json.loads((reference.parent/'summary.json').read_text())['bat-no-disk']
    allocation=summary['allocations_mib'][0]
    log=(reference/'engine.stderr').read_text()
    count=int(re.search(r'no-disk plan:.*?(\d+) permanent in VRAM',log).group(1))
    sizes=sorted({(x['main_slot_bytes'],x['raw_bytes']) for x in layers})
    assert len(sizes)==2
    small,large=sizes
    candidates=[]
    for big in range(count+1):
        gpu=(count-big)*small[0]+big*large[0]
        if round(gpu/2**20,2)==allocation['experts']:
            gpu_raw=(count-big)*small[1]+big*large[1]
            candidates.append(dict(small_count=count-big,large_count=big,gpu_allocated_bytes=gpu,
                                   gpu_raw_bytes=gpu_raw))
    assert len(candidates)==1
    gpu=candidates[0]
    total_raw=sum(x['full_layer_bytes'] for x in layers)
    host_raw=total_raw-gpu['gpu_raw_bytes']
    prompt=50000
    schedules=[]
    for requested in [16384,32768,50000,53248,65536]:
        capacity=max(batch,requested//batch*batch)
        reads=[]
        remaining=prompt-1
        while remaining:
            n=min(capacity,remaining)
            calls=[min(batch,n-at) for at in range(0,n,batch)]
            reads.append(dict(rows=n,calls=calls,layer_major=len(calls)>1))
            remaining-=n
        extra=(capacity-32768)*streams*dim*4
        schedules.append(dict(requested_chunk=requested,actual_read_rows=capacity,
            rows_mib=capacity*streams*dim*4/2**20,extra_rows_mib=extra/2**20,
            reads=reads,microbatches=sum(len(r['calls']) for r in reads),
            all_reads_layer_major=all(r['layer_major'] for r in reads),
            # Estimate: same device budget, the extra row bytes come from main GPU experts.
            estimated_weight_H2D_GiB=len(reads)*(host_raw+extra)/2**30 if all(r['layer_major'] for r in reads) else None))
    pp=[]
    group=[]
    for line in (reference/'protocol.txt').read_text().splitlines():
        if line.startswith('RESUME '):group=[]
        if line.startswith('PP '):group.append(line)
        if line.startswith('DONE '):
            boundary=next(x for x in group if x.startswith('PP 32768 '))
            first_ms=float(boundary.split()[3]);full_ms=float(line.split()[3])
            pp.append(dict(first_read_rows=32768,first_read_ms=first_ms,
                remaining_read_rows=17231,remaining_plus_last_eval_ms=full_ms-first_ms,
                first_read_tps=32768*1000/first_ms,full_prompt_ms=full_ms))
    result=dict(method='Offline header arithmetic and existing public results; no GPU timings measured',
        model=str(model),architecture=arch,layers=nlayer,experts_per_layer=ne,total_experts=nlayer*ne,
        used_experts=gg.metadata[arch+'.expert_used_count'],embedding=dim,streams=streams,
        stage_mib=stage/2**20,layer_geometry=layers,main_gpu_reconstruction=gpu,
        all_raw_experts_GiB=total_raw/2**30,host_raw_bytes=host_raw,
        host_raw_GiB=host_raw/2**30,current_two_read_expert_H2D_GiB=2*host_raw/2**30,
        schedules=schedules,PP_chunks=pp)
    with (HOME/'analysis.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ['layer_geometry','main_gpu_reconstruction','model']},indent=2))


if __name__=='__main__':main()
