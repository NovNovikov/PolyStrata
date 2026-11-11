"""Recalculate saved numeric diagnostics only. No builds, model loads or private data."""
import json
import statistics
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = HERE.parent/'2026-10-22-poly-dma-coalesced-idle'
OLD = HERE.parent/'2026-10-20-poly-cpu-bottleneck'
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
results = read(BASE/'combined/results.json')
audit = read(BASE/'transfer-analysis.json')
records = [r for r in audit['records'] if r.get('phase')=='decode']
source = subprocess.check_output(['git','show','7a2cf777:src/poly/engine.cpp'],cwd=ROOT).decode('utf-8').splitlines()
assert 'if (want_logits)' in source[3966-1] and 'PA_WAIT_STREAM' in source[3966-1]
assert 'PA_WAIT_STREAM' in source[4270-1]
assert 'PA_WAIT_EVENT' in source[2711-1]

summary = {'new_model_runs':0,'new_builds':0,'source_run':str(BASE),'requests':[]}
for q in results['requests']:
    n=q['number']; c=q['decode_costs']
    rr=[r for r in records if r['request']==n]
    outputs=sum(r['host_ms'] for r in rr if r['file']=='engine.cpp' and r['line'] in (3966,4270) and r['op']=='wait-stream')
    preparation=sum(r['host_ms'] for r in rr if r['op']=='dma-prepare-total')
    basic=c['wall_ms']-c['cpu_expert_ms']-c['route_wait_ms']
    summary['requests'].append(dict(request=n,generated=q['generated'],accounted_intervals=c['accounted_tokens'],
        wall_ms=c['wall_ms'],cpu_expert_main_thread_span_ms=c['cpu_expert_ms'],
        router_ready_main_thread_wait_ms=c['route_wait_ms'],output_main_thread_wait_ms=outputs,
        decode_dma_prepare_ms=preparation,residual_before_output_dma_ms=basic,
        residual_including_cpu_route_ms=basic-outputs-preparation))
keys=[k for k in summary['requests'][0] if k.endswith('_ms')]
summary['totals']={k:sum(r[k] for r in summary['requests']) for k in keys}
t=summary['totals'];t['residual_percent']=100*t['residual_including_cpu_route_ms']/t['wall_ms']
intervals=sum(r['accounted_intervals'] for r in summary['requests'])
t['accounted_intervals']=intervals
t['cpu_expert_ms_per_output_interval']=t['cpu_expert_main_thread_span_ms']/intervals
waits=[r for r in records if r['file']=='engine.cpp' and r['line']==2711 and r['op']=='wait-event']
summary['raw_router_wait']={k:sum(r.get(k,0) for r in waits) for k in ['calls','host_ms']}
unused=[r for r in records if r['op']=='router-host-x-unused-payload']
summary['unused_host_x']={'calls':sum(r['calls'] for r in unused),'bytes':sum(r['bytes'] for r in unused)}
copies=[r for r in records if r['op']=='kernel-D2H-copy2-graph-replay']
summary['executed_graph_D2H_copy2_bytes']=sum(r['bytes'] for r in copies)
summary['captured_gpu_head_timing_coverage']={op:sum(r['gpu_samples'] for r in records if r['op']==op)
    for op in ['attention-head-interval','copy2-kernel-interval']}

phases=[]
for line in (OLD/'phase-probe/engine/engine.stderr').read_text(encoding='utf-8').splitlines():
    if line.startswith('GPU_PHASE '):
        r={k:float(v) for k,v in (v.split('=') for v in line.split()[1:])}
        if r['pos']>=22445:
            assert r['layers']==43
            phases.append(r)
keys=[k for k in phases[0] if k.endswith('_ms')]
summary['older_gpu_probe']={'source':str(OLD/'phase-probe/engine/engine.stderr'),'scalar_samples':len(phases),
    'median_ms':{k:statistics.median(r[k] for r in phases) for k in keys},
    'total_head_median_ms':statistics.median(sum(r[k] for k in keys) for r in phases),
    'different_geometry':True,'may_subtract_from_new_run':False}
summary['limits']=[
    'Residual subtracts disjoint caller-thread spans in this single-GPU small-batch path. It is not a sum of CPU and GPU execution durations.',
    'CPU expert span includes worker waits and overlapping GPU-submission callback. Their GPU time is not added again.',
    'Residual is an upper budget containing CPU route plus other host costs, not an isolated measurement of route().',
    'Moving routing to GPU can alter overlap/dependencies; the residual is not a bound on every possible scheduling optimization.',
    'Latest graph replays account transfer bytes but do not retain per-invocation GPU head/copy completion timestamps.',
    'Older external-event phase probe establishes real GPU work but cannot be numerically subtracted from the newer workload.']
with (HERE/'summary.json').open('x',encoding='utf-8') as f:
    json.dump(summary,f,indent=2)
print(json.dumps({'totals':summary['totals'],'older_gpu_probe':summary['older_gpu_probe'],
                  'unused_host_x':summary['unused_host_x']},indent=2))
