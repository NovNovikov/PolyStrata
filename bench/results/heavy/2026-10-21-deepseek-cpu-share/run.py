"""Prepare native inputs through the actual server encoder; launch only when explicitly requested."""
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

HOME = Path(__file__).resolve().parent
ROOT = HOME.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
spec = importlib.util.spec_from_file_location('previous_bench', HOME/'helpers/driver.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
previous.HOME = HOME
previous.runner.HOME = HOME


def prepare(config):
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate
    from serve.server import Service
    tok = Tokenizer.from_dir(config['tokenizer'])
    template = ChatTemplate(Path(config['tokenizer'])/'chat_template.jinja')
    # Only the real server's prompt preparation, with a dummy engine: no server, model, logs or chats opened.
    svc = Service.__new__(Service)
    svc.engine = SimpleNamespace(info={}, batch=0)
    svc.tok, svc.template = tok, template
    svc.preserve_thinking = config.get('preserve_thinking') is True
    svc.effort_end = config.get('effort_position','start') == 'end'
    svc._read_tokenizer(tok)
    sources = [ROOT/x for x in ['docs/DETAILS.md','docs/GLM.md','docs/BACKENDS.md','README.md']]
    corpus = '\n\n'.join(p.read_text(encoding='utf-8') for p in sources)
    raw = tok.encode(corpus) # Plain source text; template controls are handled by Service.encode_prompt below.
    while len(raw)<52000:
        corpus += '\n\n'+corpus
        raw = tok.encode(corpus)
    prompts, checks, messages_out = [], [], []
    for tag in ['A','B']:
        n=49900
        for attempt in range(12):
            text=(f'Independent engineering task {tag}. Read this public documentation.\n\n'+tok.decode(raw[:n])+
                  '\n\nExplain how the engine places experts in RAM and VRAM and handles their cache. '
                  'Give a detailed technical answer with concrete examples.')
            messages=[dict(role='user',content=text)]
            ids=svc.encode_prompt(messages,None,{},store=False) # Match an ordinary request's default thinking.
            if len(ids)==50000:break
            n += 50000-len(ids)
        assert len(ids)==50000
        # Cross-check cached server encoding against its full encode implementation.
        cached=svc.prompts
        svc.prompts=None
        uncached=svc.encode_prompt(messages,None,{},store=False)
        svc.prompts=cached
        assert ids==uncached
        bos=tok.special_ids['tokenizer.ggml.bos_token_id']
        user=tok.special_tokens['<\uff5cUser\uff5c>']
        assistant=tok.special_tokens['<\uff5cAssistant\uff5c>']
        assert ids[0]==bos and ids[1]==user
        assert assistant in ids[-5:] and ids[-1]==tok.special_tokens['<think>']
        assert ids.count(user)==1 and ids.count(assistant)==1
        prompts.append(ids);messages_out.append(messages)
        checks.append(dict(label=tag,tokens=len(ids),server_cached_equals_full=True,bos=bos,user=user,
                           assistant=assistant,first_ids=ids[:12],last_ids=ids[-12:],thinking='server default'))
    common=next(i for i,(a,b) in enumerate(zip(*prompts)) if a!=b)
    assert common<32
    previous.runner.save(HOME/'public-inputs.json',dict(ids=prompts,messages=messages_out,tokens=[50000,50000],
        common_prefix=common,sources={str(p.relative_to(ROOT)):previous.runner.sha(p) for p in sources},
        tokenizer_template_sha256=previous.runner.sha(Path(config['tokenizer'])/'chat_template.jinja'),
        config_sha256=previous.runner.sha(ROOT/'strata-dsv4-q3_k_m.json'),
        server_encoder_sha256=previous.runner.sha(ROOT/'serve/server.py')))
    previous.runner.save(HOME/'input-validation.json',dict(checks=checks,loads_model=False,
        method='Actual Service.encode_prompt including template BOS and protected literal control text'))
    del svc,tok,template,raw,corpus
    gc.collect()
    print(json.dumps(checks),flush=True)
    return prompts


def prepared(config):
    data=json.loads((HOME/'public-inputs.json').read_text())
    assert data['config_sha256']==previous.runner.sha(ROOT/'strata-dsv4-q3_k_m.json')
    assert data['server_encoder_sha256']==previous.runner.sha(ROOT/'serve/server.py')
    assert data['tokenizer_template_sha256']==previous.runner.sha(Path(config['tokenizer'])/'chat_template.jinja')
    return data['ids']


def validate(config):
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate
    from serve.server import Service
    data=json.loads((HOME/'public-inputs.json').read_text())
    prompts=prepared(config)
    tok=Tokenizer.from_dir(config['tokenizer'])
    svc=Service.__new__(Service)
    svc.engine=SimpleNamespace(info={},batch=0)
    svc.tok=tok;svc.template=ChatTemplate(Path(config['tokenizer'])/'chat_template.jinja')
    svc.preserve_thinking=config.get('preserve_thinking') is True
    svc.effort_end=config.get('effort_position','start')=='end'
    svc._read_tokenizer(tok)
    for i,messages in enumerate(data['messages']):
        assert svc.encode_prompt(messages,None,{},store=False)==prompts[i]
        assert len(prompts[i])==50000
    print('OFFLINE VALIDATION OK: both 50000-token inputs equal actual server encoding; no model loaded',flush=True)


if __name__=='__main__':
    config=json.loads((ROOT/'strata-dsv4-q3_k_m.json').read_text(encoding='utf-8-sig'))
    if sys.argv[1:]==['--prepare-only']:
        prepare(config)
    elif sys.argv[1:]==['--validate-prepared']:
        validate(config)
    elif sys.argv[1:] and sys.argv[1]=='--run-prepared':
        import argparse
        parser=argparse.ArgumentParser()
        parser.add_argument('--run-prepared',action='store_true')
        parser.add_argument('--modes',nargs='+',choices=['bat-no-disk','bat-exchange','bat-mmap','bat-cpu-0','bat-cpu-50','bat-cpu-75'],
                            default=['bat-no-disk','bat-exchange','bat-mmap'])
        args=parser.parse_args()
        validate(config)
        previous.RUN_ORDER=args.modes
        previous.inputs=prepared
        previous.main()
    else:
        raise SystemExit('Use --prepare-only, --validate-prepared (offline), or --run-prepared [--modes ...]')
