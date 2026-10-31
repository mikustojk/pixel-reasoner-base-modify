"""Active-role training only: native/DDP debugging or DeepSpeed ZeRO-2 production.

Generated token IDs are retained verbatim. Prompt/vision/opponent tokens never
contribute to the GRPO loss. Every rank makes the same number of generation and
training calls, including inactive padding calls, to keep collectives aligned.
"""
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import json,random
import numpy as np
import torch
import torch.distributed as dist

@dataclass
class Sample:
    inputs: dict
    tokens: list
    text: str
    old_logp: torch.Tensor
    active: bool = True


def relative_advantages(rewards, eligible, eps=1e-6):
    values=torch.tensor(rewards,dtype=torch.float32)
    mask=torch.tensor(eligible,dtype=torch.bool)
    if not torch.isfinite(values[mask]).all():raise ValueError('Nonfinite eligible reward.')
    out=torch.zeros_like(values)
    if int(mask.sum())<2:return out
    std=values[mask].std(unbiased=False)
    if std<=eps:return out
    out[mask]=(values[mask]-values[mask].mean())/(std+eps)
    return out


def clipped_loss_sum(new,old,adv,clip):
    ratio=torch.exp(new-old)
    return -torch.minimum(ratio*adv,ratio.clamp(1-clip,1+clip)*adv).sum()

class Engine:
    def __init__(self,cfg,model_path,microsteps,model=None,processor=None):
        from transformers import AutoProcessor,AutoTokenizer,Qwen2_5_VLForConditionalGeneration
        self.cfg=cfg;self.microsteps=microsteps
        self.rank=dist.get_rank() if dist.is_initialized() else 0
        self.world=dist.get_world_size() if dist.is_initialized() else 1
        self.device=torch.device('cuda',int(__import__('os').environ.get('LOCAL_RANK','0'))) if torch.cuda.is_available() else torch.device('cpu')
        self.ds=cfg['backend']=='deepspeed'
        if self.ds and self.device.type!='cuda':raise RuntimeError('DeepSpeed training requires CUDA.')
        if processor is None:
            processor=AutoProcessor.from_pretrained(model_path,local_files_only=True,use_fast=False,
                         min_pixels=cfg['min_pixels'],max_pixels=cfg['max_pixels'])
            processor.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,use_fast=True)
        self.processor=processor
        if model is None:
            model=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,local_files_only=True,
                      torch_dtype=torch.bfloat16 if self.ds else torch.float32,attn_implementation='sdpa')
        self.bare=model.to(self.device)
        if cfg['freeze_vision']:
            for p in self.bare.model.visual.parameters():p.requires_grad_(False)
        for module in self.bare.modules():
            if isinstance(module,torch.nn.Dropout):module.p=0.0
        if cfg['gradient_checkpointing']:
            self.bare.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        parameters=[p for p in self.bare.parameters() if p.requires_grad]
        if not parameters:raise ValueError('No trainable parameters.')
        if self.ds:
            import deepspeed
            config={'train_micro_batch_size_per_gpu':1,'gradient_accumulation_steps':microsteps,
                    'train_batch_size':microsteps*self.world,'bf16':{'enabled':True},
                    'zero_optimization':{'stage':2,'offload_optimizer':{'device':'cpu'} if cfg['optimizer_offload'] else {'device':'none'}},
                    'gradient_clipping':cfg['grad_clip'],
                    'optimizer':{'type':'AdamW','params':{'lr':cfg['learning_rate'],'weight_decay':cfg['weight_decay']}},
                    'steps_per_print':1000000}
            self.model,self.optimizer,_,_=deepspeed.initialize(model=self.bare,model_parameters=parameters,config=config)
        else:
            if self.world>1:
                self.model=torch.nn.parallel.DistributedDataParallel(self.bare,
                    device_ids=[self.device.index] if self.device.type=='cuda' else None)
            else:self.model=self.bare
            self.optimizer=torch.optim.AdamW(parameters,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])

    def encode(self,messages):
        from PIL import Image
        images=[]
        for m in messages:
            if isinstance(m['content'],list):
                for p in m['content']:
                    if p['type']=='image':
                        with Image.open(p['image']) as im:images.append(im.convert('RGB'))
        prompt=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        result=self.processor(text=[prompt],images=images,padding=False,truncation=False,
                              add_special_tokens=False,return_tensors='pt')
        if result['input_ids'].shape[1]>=self.cfg['max_context']:raise ValueError('Prompt exceeds context; truncation is forbidden.')
        return dict(result)

    def logp(self,sample,gradient):
        inputs=dict(sample.inputs)
        suffix=torch.tensor([sample.tokens],dtype=inputs['input_ids'].dtype)
        inputs['input_ids']=torch.cat([inputs['input_ids'],suffix[:,:-1]],dim=1)
        inputs['attention_mask']=torch.ones_like(inputs['input_ids'])
        positions,_=self.bare.model.get_rope_index(inputs['input_ids'],image_grid_thw=inputs.get('image_grid_thw'),attention_mask=inputs['attention_mask'])
        inputs={k:v.to(self.device) for k,v in inputs.items()}
        inputs['position_ids']=positions.to(self.device)
        with torch.enable_grad() if gradient else torch.no_grad():
            logits=self.model(**inputs,use_cache=False,logits_to_keep=len(sample.tokens),return_dict=True).logits[0]
            logp=torch.log_softmax(logits.float()/self.cfg['temperature'],dim=-1)
            result=logp.gather(-1,suffix[0].to(self.device).unsqueeze(-1)).squeeze(-1)
        return result

    def generate_text(self,messages,tag,do_sample):
        from transformers import StoppingCriteria,StoppingCriteriaList
        inputs=self.encode(messages);length=inputs['input_ids'].shape[1]
        budget=min(self.cfg['max_new_tokens'],self.cfg['max_context']-length)
        tokenizer=self.processor.tokenizer
        class Stop(StoppingCriteria):
            def __call__(self,ids,scores,**kwargs):
                text=tokenizer.decode(ids[0,length:].tolist(),skip_special_tokens=False)
                return torch.tensor(['</'+tag+'>' in text],device=ids.device,dtype=torch.bool)
        self.bare.eval()
        with torch.inference_mode():
            sampling={'temperature':self.cfg['temperature'],'top_p':1.0,'top_k':0} if do_sample else {}
            output=self.bare.generate(**{k:v.to(self.device) for k,v in inputs.items()},
                   do_sample=do_sample,**sampling,
                   repetition_penalty=1.0,max_new_tokens=budget,use_cache=True,
                   synced_gpus=self.world>1,stopping_criteria=StoppingCriteriaList([Stop()]))
            tokens=output[0,length:].tolist()
        if not tokens:raise RuntimeError('Generation returned zero tokens.')
        text=tokenizer.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        return inputs,tokens,text

    def infer(self,messages,tag):
        """Greedy P2 inference on the CURRENT shared policy; no loss/logp or copy.

        Every distributed rank must call this the same number of times. The
        acquisition loop supplies inactive padding observations where necessary.
        """
        _,_,text=self.generate_text(messages,tag,False)
        return text

    def sample(self,messages,tag,active=True):
        inputs,tokens,text=self.generate_text(messages,tag,True)
        item=Sample(inputs,tokens,text,None,active)
        item.old_logp=self.logp(item,False).detach().cpu().clone()
        return item

    def update(self,groups,rewards,eligible):
        advantages=relative_advantages(rewards,eligible)
        effective=torch.tensor([int(bool(advantages.abs().sum()))],device=self.device,dtype=torch.long)
        if self.world>1:dist.all_reduce(effective)
        if not effective.item():return {'updated':False,'flat_group':True,'loss':0.0}
        samples=[(s,float(advantages[i]),eligible[i]) for i,g in enumerate(groups) for s in g]
        if len(samples)!=self.microsteps:raise ValueError('Ranks must use the configured fixed microstep count.')
        total=torch.tensor([sum(len(s.tokens) for s,adv,ok in samples if s.active and ok)],device=self.device,dtype=torch.float64)
        if self.world>1:dist.all_reduce(total)
        if total.item()<=0:raise ValueError('No active response tokens.')
        self.bare.train();self.optimizer.zero_grad();loss_total=0.
        for i,(s,adv,ok) in enumerate(samples):
            sync=nullcontext() if self.ds or self.world==1 or i==len(samples)-1 else self.model.no_sync()
            with sync:
                new=self.logp(s,True);old=s.old_logp.to(self.device)
                if not torch.isfinite(new).all() or not torch.isfinite(old).all():raise RuntimeError('Nonfinite policy log-probabilities.')
                a=adv if s.active and ok else 0.
                loss=clipped_loss_sum(new,old,a,self.cfg['clip_epsilon'])
                loss=loss*(self.world/total.item())*(self.microsteps if self.ds else 1.)
                loss_total+=float(loss.detach())
                if self.ds:self.model.backward(loss)
                else:loss.backward()
            if self.ds:self.model.step()
        if not self.ds:
            torch.nn.utils.clip_grad_norm_([p for p in self.bare.parameters() if p.requires_grad],self.cfg['grad_clip'])
            self.optimizer.step()
        self.bare.eval()
        return {'updated':True,'flat_group':False,'loss':loss_total,'advantages':advantages.tolist()}

    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        if self.ds:
            self.model.save_checkpoint(str(path/'optimizer'),tag='final')
            self.model.save_16bit_model(str(path/'hf'))
            if self.rank==0:
                self.bare.config.save_pretrained(path/'hf')
                self.bare.generation_config.save_pretrained(path/'hf')
        elif self.rank==0:
            self.bare.save_pretrained(path/'hf',safe_serialization=True)
            torch.save(self.optimizer.state_dict(),path/'optimizer.pt')
        if self.rank==0:self.processor.save_pretrained(path/'hf')
        if self.world>1:dist.barrier()
