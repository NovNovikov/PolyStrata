"""Extract only numeric data from completed public synthetic tests; no model runs or private prompts."""
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[3]
HOME=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    data=[]
    sources={}
    for name in ['2026-10-21-deepseek-bat-prefill-specials',
                 '2026-10-21-deepseek-bat-prefill-specials-reverse',
                 '2026-10-21-deepseek-cpu-share']:
        p=ROOT/'bench/results'/name
        summary=json.loads((p/'summary.json').read_text())
        sources[name]=dict(summary_sha256=sha(p/'summary.json'),
            input_sha256=sha(p/'public-inputs.json'),seed_sha256=sha(p/'initial-expert-profile.txt'))
        for mode,d in summary.items():
            if 'requests' not in d:continue
            cmd=json.loads((p/mode/'command.json').read_text())
            log=(p/mode/'engine.stderr').read_text()
            item=dict(source=name,mode=mode,binary_sha256=cmd['exe_sha256'],exit=d['exit'],
                load_s=d['load_s'],command=cmd['command'],allocations=d.get('allocations_mib',[]),requests=[],
                exchanges=[x for x in log.splitlines() if 'strata-poly: exchange:' in x],
                parking=[x for x in log.splitlines() if 'conversation cache: parked ' in x])
            for r in d['requests']:
                q={k:r.get(k) for k in ['number','read','resume','generated','finish','prompt_ms','decode_ms',
                    'decode_tps','decode_hit_rate','drafts_accepted','drafts_offered','decode_costs']}
                for phase in ['prefill','decode']:
                    hw=r[phase+'_samples']
                    q[phase]=dict(samples=hw.get('samples'),system_disk_read_GiB=hw['system_disk_read_bytes_delta']/2**30,
                        process_page_faults=hw['page_fault_delta'],
                        RSS_median_GiB=hw['rss']['median']/2**30,
                        GPU_Local_median_GiB=hw['process_gpu_used']['median']/2**30,
                        GPU_NonLocal_median_GiB=hw['process_gpu_shared']['median']/2**30,
                        sampled_RX=hw['gpu_pcie_rx_mb'],sampled_TX=hw['gpu_pcie_tx_mb'])
                item['requests'].append(q)
            data.append(item)
    assert len({x['binary_sha256'] for x in data})==1
    assert len({x['input_sha256'] for x in sources.values()})==1
    assert len({x['seed_sha256'] for x in sources.values()})==1
    assert all(r['read']==50000 and r['resume']==0 and r['generated']==256 for x in data for r in x['requests'])
    out=dict(method='Numeric extraction from 18 completed requests; no new GPU measurements',
        sources=sources,runs=data,checks=dict(same_binary=True,same_public_inputs=True,same_initial_profile=True,
        all_18_full_reads_and_256_outputs=True),
        source_files={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/poly/engine.cpp',
            ROOT/'src/poly/deepseek4.cpp',ROOT/'src/poly/resident_dma.hpp',ROOT/'src/poly/exchange.cpp',
            ROOT/'src/poly/conversations.hpp',ROOT/'src/poly/session_file.hpp']})
    with (HOME/'evidence.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)
    print('18 requests checked: identical executable, public inputs and initial statistics; all read 50000 and generated 256.')

if __name__=='__main__':main()
