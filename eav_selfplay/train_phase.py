"""Shared-policy GRPO: bootstrap evidence teacher, then current-policy P2 inference."""
import argparse,json,os,random,traceback
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import torch
import torch.distributed as dist
from .core import *
from .engine import Engine
from .io import load_questions,rows,write_json,digest
from .judge import FrozenJudge

TRAINING_PROTOCOL='online_shared_p2_v1'


def response_error(event,error,raw=None,**context):
    """Private audit only: never forward diagnostics or teacher text to a role."""
    return {'event':event,'error_type':type(error).__name__,'reason':str(error),
            'raw_response':getattr(error,'raw_response',raw),
            'response_kind':getattr(error,'response_kind',event),
            'finish_reason':getattr(error,'finish_reason',None),
            'traceback':traceback.format_exc(),**context}


def format_diagnostics(traces):
    events=Counter(r.get('event') for r in traces)
    reasons=Counter(r['reason'] for r in traces if r.get('reason'))
    return {'p1_active_outputs':events['crop_accepted']+events['invalid_crop'],
            'valid_crops':events['crop_accepted'],'invalid_crops':events['invalid_crop'],
            'invalid_p2_decisions':events['invalid_decision'],
            'evidence_errors':events['evidence_error'],'answer_errors':events['answer_error'],
            'probe_errors':events['probe_error'],'probe_crop_errors':events['probe_crop_error'],
            'error_reasons':dict(reasons)}


def indices(n,rank,world,seed):
    order=list(range(n));random.Random(seed).shuffle(order)
    order+=(order*((world-1)//n+1))[:(-n)%world]
    return order[rank::world]


def acquisition_group(engine,q,judge,cfg,crop_dir,version,group_id,online_p2=False):
    single_crop=cfg.get('single_crop_only',False)
    if single_crop and (cfg['max_rounds']!=1 or cfg.get('probe_crops_per_group',0)!=0):
        raise ValueError('Single-crop training requires one round and no probe crops.')
    groups=[];rewards=[];eligible=[];bank=[];traces=[]
    evidence_source='current_shared_p2' if online_p2 else 'bootstrap_judge'

    def observation(images):
        if not images:
            # Inactive ranks still enter P2 generation collectives. Never show
            # the original image as a padding crop or put this into the bank.
            from PIL import Image
            padding=Path(crop_dir)/('_padding_'+group_id+'.png')
            padding.parent.mkdir(parents=True,exist_ok=True)
            if not padding.exists():Image.new('RGB',(28,28),'gray').save(padding)
            images=[str(padding)]
        return verification_messages(q.question,q.options,Evidence(tuple(images)),single_crop)

    for sample_index in range(cfg['group_size']):
        own=[];evidence=[];area=0.;ended=False;invalid=False;ok=True;decision=None;correct=False
        samples=[]
        for turn in range(cfg['max_rounds']):
            messages=acquisition_messages(q.question,q.image,q.options,own,Feedback(True),cfg.get('crop_coordinate_mode','normalized'),single_crop)
            sample=engine.sample(messages,'crop',not ended);samples.append(sample)
            active=not ended
            if active:
                own.append(sample.text)
                try:
                    box=parse_crop_for_image(sample.text,q.image,cfg.get('crop_coordinate_mode','normalized'))
                    image,fraction=crop_pixels(q.image,box,crop_dir,cfg['min_crop_side'])
                    evidence.append(image);area+=fraction
                    traces.append({'event':'crop_accepted','candidate':sample_index,'turn':turn})
                except (ValueError,TypeError,json.JSONDecodeError) as error:
                    traces.append(response_error('invalid_crop',error,sample.text,
                                                 candidate=sample_index,turn=turn))
                    invalid=True;ended=True;active=False
            # This call MUST occur on every rank, even after an episode ends or
            # a crop is invalid. It is generation-only, excluded from P1 loss.
            raw=engine.infer(observation(evidence),'decision') if online_p2 else None
            if not active:continue
            error_event='evidence_error'
            try:
                if online_p2:decision=parse_decision(raw,single_crop)
                else:decision,raw=judge.judge(q.question,q.options,Evidence(tuple(evidence)))
                bank.append({'qid':q.qid,'question':q.question,'options':q.options,'reference':q.reference,
                             'evidence':list(evidence),'label':decision.evidence_label,'quality':decision.quality,
                             'confidence':decision.confidence,'judge_version':version,'evidence_source':evidence_source,
                             'group_id':group_id,'candidate':sample_index,'turn':turn})
                traces.append({'candidate':sample_index,'turn':turn,'p2_private_judgment':raw,
                               'evidence_source':evidence_source,'evidence_version':version})
                # Labels are never inferred from QA success/failure.
                if decision.action=='answer':
                    error_event='answer_error'
                    raw=None  # An API failure must not be attributed to the previous evidence reply.
                    correct=judge.answer_correct(common_text(q.question,q.options),q.reference,decision.answer,cfg['answer_mode'])
                    ended=True
                elif single_crop:
                    ended=True
            except Exception as error:
                ok=False;ended=True
                traces.append(response_error(error_event,error,raw,candidate=sample_index,
                                             turn=turn,judge_error=str(error)))
        groups.append(samples);eligible.append(ok)
        rewards.append(p1_reward(correct,decision,area,len(evidence),cfg,invalid))
        traces.append({'candidate':sample_index,'p1_private_outputs':own,'total_area_fraction':area,
                       'correct':correct,'reward':rewards[-1],'eligible':ok})
    # Extra crops are judged by the same evidence evaluator as active episodes.
    # A random crop is NEVER automatically called irrelevant.
    from PIL import Image
    with Image.open(q.image) as im:width,height=im.size
    for probe in range(cfg.get('probe_crops_per_group',0)):
        probe_images=[];probe_error=None
        try:
            w=max(min(1.,cfg['min_crop_side']/width),random.uniform(.1,.6))
            h=max(min(1.,cfg['min_crop_side']/height),random.uniform(.1,.6))
            x=random.uniform(0,1-w);y=random.uniform(0,1-h)
            image,_=crop_pixels(q.image,(x,y,x+w,y+h),crop_dir,cfg['min_crop_side'])
            probe_images=[image]
        except (ValueError,TypeError) as error:probe_error=str(error)
        raw=engine.infer(observation(probe_images),'decision') if online_p2 else None
        if probe_error:
            traces.append({'event':'probe_crop_error','probe':probe,'reason':probe_error,
                           'judge_or_crop_error':probe_error});continue
        try:
            if online_p2:d=parse_decision(raw)
            else:d,raw=judge.judge(q.question,q.options,Evidence(tuple(probe_images)))
            bank.append({'qid':q.qid,'question':q.question,'options':q.options,'reference':q.reference,
                         'evidence':[image],'label':d.evidence_label,'quality':d.quality,
                         'confidence':d.confidence,'judge_version':version,'evidence_source':evidence_source,'group_id':group_id,
                         'candidate':'probe_'+str(probe),'turn':0,'source':'judged_probe'})
            traces.append({'probe':probe,'p2_private_judgment':raw,'evidence_source':evidence_source})
        except Exception as error:
            traces.append(response_error('probe_error',error,raw,probe=probe,
                                         judge_or_crop_error=str(error)))
    return groups,rewards,eligible,bank,traces


def verification_group(engine,record,judge,cfg):
    groups=[];rewards=[];eligible=[];traces=[]
    single_crop=cfg.get('single_crop_only',False)
    messages=verification_messages(record['question'],record['options'],Evidence(tuple(record['evidence'])),single_crop)
    for _ in range(cfg['group_size']):
        sample=engine.sample(messages,'decision');decision=None;ok=True;correct=False
        try:decision=parse_decision(sample.text,single_crop)
        except (ValueError,TypeError,json.JSONDecodeError) as error:
            traces.append(response_error('invalid_decision',error,sample.text))
        if decision is not None and decision.action=='answer' and record['label']=='sufficient':
            try:correct=judge.answer_correct(common_text(record['question'],record['options']),record['reference'],decision.answer,cfg['answer_mode'])
            except Exception as error:
                ok=False;traces.append(response_error('answer_error',error))
        reward=p2_reward(decision,record['label'],record['quality'],record['confidence'],correct,cfg,single_crop)
        groups.append([sample]);rewards.append(reward);eligible.append(ok)
        traces.append({'p2_private_output':sample.text,'reward':reward,'eligible':ok})
    return groups,rewards,eligible,[],traces


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True,type=Path);parser.add_argument('--role',choices=['p1','p2'],required=True)
    parser.add_argument('--model',required=True);parser.add_argument('--data',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--judge-url',required=True)
    parser.add_argument('--judge-version',required=True);parser.add_argument('--cycle',required=True,type=int)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    single_crop=cfg.get('single_crop_only',False)
    if single_crop and (cfg['max_rounds']!=1 or cfg.get('probe_crops_per_group',0)!=0):
        raise ValueError('Single-crop training requires one round and no probe crops.')
    prompt_protocol=(PROMPT_PROTOCOL_PIXEL_SINGLE if cfg.get('crop_coordinate_mode','normalized')=='pixel' else PROMPT_PROTOCOL_SINGLE) if single_crop else (PROMPT_PROTOCOL_PIXEL if cfg.get('crop_coordinate_mode','normalized')=='pixel' else PROMPT_PROTOCOL)
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'))
    if world>1:dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo')
    if torch.cuda.is_available():torch.cuda.set_device(int(os.environ.get('LOCAL_RANK','0')))
    torch.manual_seed(cfg['seed']+rank+1000*args.cycle);random.seed(cfg['seed']+rank+1000*args.cycle)
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'phase_done.json').exists():raise ValueError('Completed phase already exists; use the orchestrator resume option.')
    progress=args.output/f'progress_rank{rank}.json'
    # Incomplete phases intentionally require a fresh output: never silently repeat and overwrite updates.
    if progress.exists():raise RuntimeError('Partial training phase exists. Preserve it and choose a fresh run; completed phases can resume.')
    if args.role=='p1':data=load_questions(args.data)
    else:
        data=[r for r in rows(args.data) if r['label'] in {'sufficient','insufficient'} and r['confidence']>=cfg['min_label_confidence']]
        if cfg.get('require_both_labels',True) and {r['label'] for r in data}!={'sufficient','insufficient'}:
            raise ValueError('The confident label bank must contain both sufficient and insufficient evidence; inspect/augment it before updating P2.')
        if not data:raise ValueError('No confident evidence labels for Process2; inspect the evidence evaluator before training.')
    order=indices(len(data),rank,world,cfg['seed']+args.cycle)
    if cfg['groups_per_phase']>0:order=order[:cfg['groups_per_phase']]
    steps=cfg['group_size']*(cfg['max_rounds'] if args.role=='p1' else 1)
    engine=Engine(cfg,args.model,steps)
    judge=FrozenJudge(args.judge_url,'eav_judge',args.judge_version,timeout=cfg['judge_timeout'],single_crop=single_crop)
    online_p2=args.cycle>1
    print(json.dumps({'role':args.role,'cycle':args.cycle,'world_size':world,'rank':rank,
                     'dataset_records':len(data),'local_groups':len(order),
                     'global_prompt_groups_per_step':world,'rollouts_per_prompt':cfg['group_size'],
                     'deepspeed_sequence_batch':steps*world,
                     'evidence_evaluator':'current_shared_p2' if online_p2 else 'bootstrap_judge',
                     'prompt_protocol':prompt_protocol,
                     'training_protocol':TRAINING_PROTOCOL}),flush=True)
    updates=0;metrics=[];diagnostic_totals=Counter();reason_totals=Counter()
    for step,index in enumerate(order):
        key=f'rank{rank:04d}_group{step:08d}'
        evidence_version=(f'online_shared:cycle={args.cycle}:group={step}:prior_updates={updates}'
                          if online_p2 else args.judge_version)
        if args.role=='p1':result=acquisition_group(engine,data[index],judge,cfg,args.output/'crops',evidence_version,key,online_p2=online_p2)
        else:result=verification_group(engine,data[index],judge,cfg)
        groups,rewards,eligible,bank,traces=result
        # Save raw/error traces BEFORE backward, so a first-update failure does
        # not discard the completions which led to that update.
        write_json(args.output/'bank_groups'/(key+'.json'),bank)
        write_json(args.output/'private_traces'/(key+'.json'),traces)
        diagnostics=format_diagnostics(traces)
        reason_totals.update(diagnostics['error_reasons'])
        diagnostic_totals.update({k:v for k,v in diagnostics.items() if k!='error_reasons'})
        update=engine.update(groups,rewards,eligible);updates+=int(update['updated'])
        metric={'group':key,'role':args.role,'rewards':rewards,'eligible':eligible,
                'prompt_protocol':prompt_protocol,'format_diagnostics':diagnostics,**update}
        metrics.append(metric);write_json(progress,{'groups_completed':step+1,'updates':updates})
        print(json.dumps(metric,ensure_ascii=False),flush=True)
    write_json(args.output/f'format_summary_rank{rank}.json',
               {'rank':rank,'role':args.role,'prompt_protocol':prompt_protocol,
                'groups_completed':len(metrics),'optimizer_updates':updates,
                **dict(diagnostic_totals),'error_reasons':dict(reason_totals)})
    count=torch.tensor([updates],device=engine.device)
    if world>1:dist.all_reduce(count)
    if not count.item():raise RuntimeError('Zero optimizer updates: every reward group was flat or unscorable. Inspect format_summary_rank*.json and private_traces/. This is NOT a successful training phase.')
    engine.save(args.output)
    if rank==0:
        bank=[]
        for f in sorted((args.output/'bank_groups').glob('*.json')):bank.extend(json.loads(f.read_text()))
        with (args.output/'evidence_bank.jsonl').open('w',encoding='utf-8') as f:
            for record in bank:f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
        write_json(args.output/'phase_done.json',{'role':args.role,'cycle':args.cycle,'model_input':args.model,
                   'policy_architecture':'shared_policy_roles_v1',
                   'training_protocol':TRAINING_PROTOCOL,
                   'prompt_protocol':prompt_protocol,
                   'checkpoint':str((args.output/'hf').resolve()),'data_sha256':digest(args.data),
                   'judge_version':args.judge_version,'world_size':world,'optimizer_updates_per_rank':updates,
                   'bank_records':len(bank),'metrics':metrics,'optimizer_state_policy':'Saved for audit; each new alternating phase uses a fresh optimizer.'})
    if world>1:dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
