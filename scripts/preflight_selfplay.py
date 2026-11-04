"""GPU-node dependency/resource checks before any judge or training is started."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from train_selfplay import validate,device_list,TRAINING_PROTOCOL
from eav_selfplay.io import write_json,digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text());data=validate(cfg)
    import torch
    # Import, rather than merely detect, to catch missing/broken training deps.
    import deepspeed
    import transformers
    if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable in the training environment.')
    devices=cfg['train_devices']+cfg['judge_devices']
    if max(devices)>=torch.cuda.device_count():raise RuntimeError('The allocation does not expose all configured GPUs.')
    if len(cfg['train_devices'])!=6 or len(cfg['judge_devices'])!=1:
        raise ValueError('This s92 entry expects six training GPUs and one judge GPU.')
    train_path=Path(cfg['train_data'])
    train_sha256=digest(train_path)
    metadata_path=train_path.parent/'manifest.json'
    dataset=json.loads(metadata_path.read_text())
    if 'outputs' in dataset:
        if train_sha256!=dataset['outputs']['train.jsonl']['sha256']:
            raise ValueError('Training input differs from the prepared dataset manifest.')
    else:
        summary_path=train_path.parent/'summary.json'
        if train_path.name!='train_verified.jsonl' or not summary_path.is_file():
            raise ValueError('Filtered training input needs a completed quality audit.')
        summary=json.loads(summary_path.read_text())
        if not summary.get('completed') or train_sha256!=summary.get('verified_train_sha256'):
            raise ValueError('Filtered training input differs from the completed quality audit.')
        if summary.get('input_sha256')!=dataset.get('input_sha256') or summary.get('counts',{}).get('correct')!=len(data):
            raise ValueError('Filtered audit metadata or selected row count differs from the manifest.')
        if digest(dataset['input'])!=dataset['input_sha256']:
            raise ValueError('Filtered audit source differs from the prepared training data.')
    judge_python=cfg.get('judge_python',sys.executable)
    probe='''import importlib.metadata,json,torch
print(json.dumps({"python":__import__("sys").version,"vllm":importlib.metadata.version("vllm"),"torch":torch.__version__,"cuda":torch.cuda.is_available(),"devices":torch.cuda.device_count()}))'''
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=device_list(cfg['judge_devices'])
    judge=json.loads(subprocess.check_output([judge_python,'-c',probe],env=env,text=True,timeout=120))
    if not judge['cuda'] or judge['devices']!=1:raise RuntimeError('Judge Python must see exactly one allocated CUDA GPU.')
    version=tuple(int(x) for x in judge['vllm'].split('.')[:2])
    judge_architecture=json.loads((Path(cfg['judge_model'])/'config.json').read_text()).get('model_type')
    minimum=(0,18) if judge_architecture in {'qwen3_5','qwen3_5_moe'} else (0,8)
    if version<minimum:
        raise RuntimeError(f'{judge_architecture} judge needs vLLM {minimum[0]}.{minimum[1]}+; found {judge["vllm"]}.')
    report={'training_protocol':TRAINING_PROTOCOL,'training_python':sys.executable,
            'torch':torch.__version__,'transformers':transformers.__version__,
            'deepspeed':importlib.metadata.version('deepspeed'),'judge':judge,
            'judge_architecture':judge_architecture,
            'allocation':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'train_devices':device_list(cfg['train_devices']),'judge_devices':device_list(cfg['judge_devices']),
            'gpus':[{'index':i,'name':torch.cuda.get_device_name(i),
                     'memory_gib':torch.cuda.get_device_properties(i).total_memory/2**30} for i in devices],
            'train_records':len(data),'full_data':cfg['groups_per_phase']==0,
            'train_sha256':train_sha256,
            'cpu_meminfo':Path('/proc/meminfo').read_text().splitlines()[:5]}
    write_json(args.output,report);print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
