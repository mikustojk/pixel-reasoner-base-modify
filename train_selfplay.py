"""One persistent answer judge; bootstrap evidence once, then online shared P2."""
import argparse,hashlib,json,os,signal,socket,subprocess,sys,time
from contextlib import nullcontext
from pathlib import Path
from eav_selfplay.io import write_json,digest,load_questions

HERE=Path(__file__).resolve().parent
POLICY_ARCHITECTURE='shared_policy_roles_v1'
TRAINING_PROTOCOL='online_shared_p2_v1'

def validate(cfg):
    if 'p1_model' in cfg or 'p2_model' in cfg:
        raise ValueError('Independent p1_model/p2_model configuration is unsupported; use one policy_model.')
    if 'bootstrap_judge' in cfg:raise ValueError('Legacy snapshot protocol: use judge_model with online shared P2.')
    required=['train_data','policy_model','judge_model','output','train_devices','judge_devices']
    for key in required:
        if not cfg.get(key):raise ValueError('Configure '+key+' before launching training.')
    for key in ['train_data','output','policy_model','judge_model']:
        if not Path(cfg[key]).is_absolute():raise ValueError('Use an absolute path for '+key)
    if any(type(i) is not int or i<0 for i in cfg['train_devices']+cfg['judge_devices']):raise ValueError('GPU IDs must be nonnegative indices within the current allocation.')
    if set(cfg['train_devices']) & set(cfg['judge_devices']):raise ValueError('Training and fixed judge must use disjoint GPUs.')
    if len(set(cfg['train_devices']))!=len(cfg['train_devices']) or len(set(cfg['judge_devices']))!=len(cfg['judge_devices']):raise ValueError('Duplicate GPU IDs.')
    for key in ['policy_model','judge_model']:
        if not (Path(cfg[key])/'config.json').is_file():raise ValueError('Missing model '+key)
    if cfg['cycles']<1 or cfg['group_size']<2 or cfg['max_rounds']<1:raise ValueError('Invalid self-play/group/round counts.')
    if type(cfg['groups_per_phase']) is not int or cfg['groups_per_phase']<0:raise ValueError('groups_per_phase must be 0 (full data) or a positive local limit.')
    if cfg['temperature']<=0 or cfg['max_new_tokens']<1 or cfg['max_context']<2:raise ValueError('Invalid sampling limits.')
    if cfg['backend'] not in ('native','deepspeed'):raise ValueError('Unsupported training backend.')
    if cfg['answer_mode'] not in ('exact','semantic'):raise ValueError('Unsupported answer verifier.')
    if not 0<=cfg['min_label_confidence']<=1:raise ValueError('Invalid pseudo-label threshold.')
    if cfg['max_rounds']*(cfg['area_cost']+cfg['observation_cost'])+cfg['evidence_bonus']>=1:raise ValueError('Secondary costs must not dominate task accuracy.')
    for key in ['area_cost','observation_cost','evidence_bonus','quality_cost','confidence_cost']:
        if cfg[key]<0:raise ValueError('Negative reward weight '+key)
    return load_questions(cfg['train_data'])

def device_list(indices):
    allocation=os.environ.get('CUDA_VISIBLE_DEVICES')
    if allocation is None:return ','.join(map(str,indices))
    devices=allocation.split(',')
    if any(i>=len(devices) for i in indices):raise ValueError('Configured GPU index exceeds the inherited allocation.')
    return ','.join(devices[i] for i in indices)

class JudgeServer:
    def __init__(self,model,cfg,log):self.model=model;self.cfg=cfg;self.log=log
    def __enter__(self):
        import urllib.request
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        self.url=f'http://127.0.0.1:{port}/v1';self.handle=Path(self.log).open('a')
        command=[self.cfg.get('judge_python',sys.executable),'-u','-m','vllm.entrypoints.openai.api_server','--model',str(self.model),
                 '--served-model-name','eav_judge','--host','127.0.0.1','--port',str(port),
                 '--tensor-parallel-size',str(len(self.cfg['judge_devices'])),
                 '--max-model-len',str(self.cfg['judge_context']), '--gpu-memory-utilization',str(self.cfg.get('judge_gpu_memory_utilization',.85)),
                 '--max-num-seqs',str(self.cfg.get('judge_max_num_seqs',1)),
                 '--default-chat-template-kwargs',json.dumps({'enable_thinking':False}),
                 '--dtype','bfloat16','--limit-mm-per-prompt',json.dumps({'image':self.cfg['max_rounds'],'video':0}),
                 '--mm-processor-kwargs',json.dumps({'max_pixels':self.cfg['max_pixels']})]
        if self.cfg.get('judge_enforce_eager',True):command.append('--enforce-eager')
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=device_list(self.cfg['judge_devices'])
        self.process=subprocess.Popen(command,env=env,stdout=self.handle,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            deadline=time.monotonic()+self.cfg['server_start_timeout']
            while time.monotonic()<deadline:
                if self.process.poll() is not None:raise RuntimeError('Fixed judge exited; inspect '+str(self.log))
                try:
                    with urllib.request.urlopen(self.url+'/models',timeout=2) as response:
                        if response.status==200:return self
                except OSError:pass
                time.sleep(2)
            raise TimeoutError('Frozen judge startup timed out.')
        except BaseException:self.__exit__(None,None,None);raise
    def __exit__(self,*args):
        if self.process.poll() is None:
            os.killpg(self.process.pid,signal.SIGTERM)
            try:self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(self.process.pid,signal.SIGKILL);self.process.wait()
        self.handle.close()


def launch(cfg,config_file,role,model,data,output,judge,version,cycle):
    command=[sys.executable,'-m','torch.distributed.run','--standalone',
             '--nproc-per-node',str(len(cfg['train_devices'])),'-m','eav_selfplay.train_phase',
             '--config',str(config_file),'--role',role,'--model',str(model),'--data',str(data),
             '--output',str(output),'--judge-url',judge.url,'--judge-version',version,'--cycle',str(cycle)]
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=device_list(cfg['train_devices'])
    subprocess.run(command,cwd=HERE,env=env,check=True)


def completed_checkpoint(phase):
    """Validate the shared weight chain before reusing or advancing a phase."""
    done=phase['output']/'phase_done.json'
    if not done.is_file():
        raise ValueError('Missing completed phase: '+str(done))
    record=json.loads(done.read_text())
    expected={'policy_architecture':POLICY_ARCHITECTURE,'training_protocol':TRAINING_PROTOCOL,'role':phase['role'],
              'cycle':phase['cycle'],'model_input':str(phase['model']),
              'checkpoint':str(phase['checkpoint']),'judge_version':phase['judge_version'],
              'data_sha256':digest(phase['data'])}
    for key,value in expected.items():
        if record.get(key)!=value:
            raise ValueError(f'Completed phase has an incompatible shared-policy chain: {done} ({key}).')
    if not (phase['checkpoint']/'config.json').is_file():
        raise ValueError('Completed phase checkpoint is missing: '+str(phase['checkpoint']))
    return phase['checkpoint']


def phase_plan(cfg):
    """Plan one sequential parameter chain and reject out-of-order/partial resumes."""
    shared=Path(cfg['policy_model']).resolve();out=Path(cfg['output']).resolve()
    plan=[];unfinished=False
    for cycle in range(1,cfg['cycles']+1):
        judge_model=Path(cfg['judge_model']).resolve()
        version=f'fixed_answer_judge:{judge_model}'
        directory=out/f'cycle_{cycle:03d}'
        for role,name in [('p1','process1'),('p2','process2')]:
            output=directory/name
            phase={'cycle':cycle,'role':role,'model':shared,'output':output,
                   'data':Path(cfg['train_data']).resolve() if role=='p1' else directory/'process1/evidence_bank.jsonl',
                   'checkpoint':output/'hf','judge_model':judge_model,'judge_version':version,
                   'done':(output/'phase_done.json').is_file()}
            if phase['done']:
                if unfinished:raise ValueError('Completed training phases are out of order; shared policy chain cannot resume.')
                completed_checkpoint(phase)
            else:
                unfinished=True
                if any(output.glob('progress_rank*.json')):
                    raise RuntimeError('An interrupted partial phase exists. Preserve this run and use a fresh output; --resume only skips completed phases.')
            plan.append(phase);shared=phase['checkpoint']
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=HERE/'configs/selfplay.json')
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Resume COMPLETED phases; an interrupted partial phase requires a fresh run.')
    parser.add_argument('--output',type=Path,help='Absolute fresh run path; overrides the config output.')
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    if args.output is not None:cfg['output']=str(args.output)
    data=validate(cfg)
    if args.validate_only:
        print(json.dumps({'questions':len(data),'policy_architecture':POLICY_ARCHITECTURE,
                          'training_protocol':TRAINING_PROTOCOL,
                          'policy_model':cfg['policy_model'],'judge':cfg['judge_model'],
                          'groups_per_phase':cfg['groups_per_phase'],'full_data':cfg['groups_per_phase']==0,
                          'train_gpus':cfg['train_devices'],'judge_gpus':cfg['judge_devices']},ensure_ascii=False));return
    out=Path(cfg['output']).resolve()
    # Bind data, config and executable source before any model update.
    fp={'policy_architecture':POLICY_ARCHITECTURE,'training_protocol':TRAINING_PROTOCOL,'config':cfg,'data_sha256':digest(cfg['train_data']),
        'source_hashes':{str(p.relative_to(HERE)):digest(p) for p in sorted((HERE/'eav_selfplay').glob('*.py'))},
        'entry_sha256':digest(__file__)}
    if out.exists():
        if not args.resume or not (out/'run.json').exists():raise ValueError('Output exists; choose a new output or explicitly resume completed phases.')
        if json.loads((out/'run.json').read_text())!=fp:raise ValueError('Resume shared-policy architecture/config/data/source mismatch.')
    plan=phase_plan(cfg)
    if any(not phase['done'] for phase in plan):
        import torch,importlib.util
        if cfg['backend']=='deepspeed' and importlib.util.find_spec('deepspeed') is None:
            raise RuntimeError('Install the separate DeepSpeed training environment before launching real training.')
        judge_python=cfg.get('judge_python',sys.executable)
        if judge_python==sys.executable:
            if importlib.util.find_spec('vllm') is None:raise RuntimeError('The fixed judge server requires vLLM.')
        elif not Path(judge_python).is_file():raise RuntimeError('Missing judge Python: '+judge_python)
        if not torch.cuda.is_available() or max(cfg['train_devices']+cfg['judge_devices'])>=torch.cuda.device_count():
            raise RuntimeError('The GPU allocation does not provide all configured training/judge devices.')
    if not out.exists():out.mkdir(parents=True);write_json(out/'run.json',fp)
    effective=out/'effective_config.json';write_json(effective,cfg)
    shared=Path(cfg['policy_model']).resolve()
    # One external model stays on GPU 0 for the whole run. Later P2 evidence
    # requests execute on the existing trainable model, not a frozen replica.
    server=(JudgeServer(Path(cfg['judge_model']),cfg,out/'judge.log')
            if any(not p['done'] for p in plan) else nullcontext(None))
    with server as judge:
        for cycle in range(1,cfg['cycles']+1):
            directory=out/f'cycle_{cycle:03d}';directory.mkdir(exist_ok=True)
            phases=[phase for phase in plan if phase['cycle']==cycle]
            cycle_input=shared;version=phases[0]['judge_version']
            print(f'Cycle {cycle}: answer judge={cfg["judge_model"]}; evidence={"bootstrap" if cycle==1 else "current shared P2"}',flush=True)
            for phase in phases:
                if shared!=phase['model']:raise RuntimeError('Shared policy input deviated from the planned checkpoint chain.')
                if not phase['done']:
                    launch(cfg,effective,phase['role'],shared,phase['data'],phase['output'],judge,version,cycle)
                shared=completed_checkpoint(phase)
            write_json(directory/'cycle_done.json',{'policy_architecture':POLICY_ARCHITECTURE,
                       'training_protocol':TRAINING_PROTOCOL,
                       'model_input':str(cycle_input),'process1_checkpoint':str(phases[0]['checkpoint']),
                       'policy_checkpoint':str(shared),'judge':version,
                       'evidence_evaluator':'bootstrap_judge' if cycle==1 else 'current_shared_p2'})
    write_json(out/'complete.json',{'cycles':cfg['cycles'],'policy_architecture':POLICY_ARCHITECTURE,'training_protocol':TRAINING_PROTOCOL,
               'policy_checkpoint':str(shared),'real_training_performed':True})
    print('Self-play training finished:',out,flush=True)

if __name__=='__main__':
    def terminate(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate)
    main()
