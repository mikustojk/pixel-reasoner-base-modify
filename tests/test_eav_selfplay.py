import copy,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eav_selfplay.core import *
from eav_selfplay.judge import FrozenJudge,JudgeResponseError
from eav_selfplay.engine import relative_advantages,clipped_loss_sum
from eav_selfplay.train_phase import acquisition_group,verification_group,indices,format_diagnostics
from train_selfplay import device_list

CFG=json.loads((Path(__file__).resolve().parents[1]/'configs/selfplay.json').read_text())

def decision(label,answer='',quality=.8,confidence=.9):
 return '<decision>'+json.dumps({'action':'answer' if label=='sufficient' else 'request_more','evidence_label':label,'quality':quality,'confidence':confidence,'answer':answer})+'</decision>'

class ProtocolTests(unittest.TestCase):
 def test_prompt_examples_are_valid_and_observed_invalid_outputs_remain_invalid(self):
  import re
  examples=re.findall(r'<crop>(\{[^\n]*?\})</crop>',P1_SYSTEM)
  self.assertGreaterEqual(len(examples),3)
  for payload in examples:parse_crop('<crop>'+payload+'</crop>')
  for text in ['<crop>{"bbox_2d":[13,4,389,197]}</crop>',
               '<crop>{"bbox":[13,4,389,197]}</crop>',
               '<crop bbox="[2,1,194,193]"/></crop>',
               '<crop>{"bbox_2d":[0,0,140,212]割去多余的区域}</crop>',
               '<crop>{"bbox":[0.1,0.2,0.8,0.9]}}</crop>']:
   with self.subTest(text=text),self.assertRaises(ValueError):parse_crop(text)
  for payload in re.findall(r'<decision>(\{[^\n]*?\})</decision>',P2_SYSTEM):
   parse_decision('<decision>'+payload+'</decision>')
  with self.assertRaises(ValueError):parse_decision('Based on the image, '+decision('insufficient'))

 def test_invalid_crop_diagnostics_keep_raw_output_and_reward_unchanged(self):
  raw='<crop>{"bbox_2d":[13,4,389,197]}</crop>'
  class Actor:
   def sample(self,*args,**kwargs):return SimpleNamespace(text=raw,active=True)
  class Judge:
   def judge(self,*args):raise AssertionError('Invalid crop must not reach evidence judge')
  c=dict(CFG,group_size=2,max_rounds=1,probe_crops_per_group=0)
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp);image=p/'original.png';Image.new('RGB',(56,56),'red').save(image)
   _,rewards,eligible,bank,traces=acquisition_group(Actor(),Question('q','Q',str(image),'GT',{}),
                                                 Judge(),c,p/'crops','v','g')
   self.assertEqual(rewards,[-1.,-1.]);self.assertEqual(eligible,[True,True]);self.assertFalse(bank)
   report=format_diagnostics(traces)
   self.assertEqual(report['invalid_crops'],2);self.assertEqual(report['valid_crops'],0)
   self.assertEqual(report['p1_active_outputs'],2)
   bad=[r for r in traces if r.get('event')=='invalid_crop']
   self.assertEqual([r['raw_response'] for r in bad],[raw,raw])
   self.assertTrue(all('Traceback' in r['traceback'] for r in bad))
   self.assertEqual(relative_advantages(rewards,eligible).abs().sum().item(),0)

 def test_judge_parse_errors_preserve_complete_raw_response_and_finish_reason(self):
  raw='Outside-tag preamble. '+decision('insufficient')
  def create(**kwargs):
   return SimpleNamespace(choices=[SimpleNamespace(finish_reason='length',message=SimpleNamespace(content=raw))])
  client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
  class Actor:
   def sample(self,*args,**kwargs):return SimpleNamespace(text='<crop>{"bbox":[0,0,1,1]}</crop>',active=True)
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp);image=p/'original.png';Image.new('RGB',(56,56),'red').save(image)
   judge=FrozenJudge('unused','model','version',client=client)
   with self.assertRaises(JudgeResponseError) as caught:judge.judge('Q',{},Evidence((str(image),)))
   self.assertEqual(caught.exception.raw_response,raw);self.assertEqual(caught.exception.finish_reason,'length')
   c=dict(CFG,group_size=2,max_rounds=1,probe_crops_per_group=0)
   _,_,eligible,bank,traces=acquisition_group(Actor(),Question('q','Q',str(image),'GT',{}),judge,
                                           c,p/'crops','v','g')
   self.assertEqual(eligible,[False,False]);self.assertFalse(bank)
   bad=[r for r in traces if r.get('event')=='evidence_error']
   self.assertEqual(len(bad),2);self.assertEqual(bad[0]['raw_response'],raw)
   self.assertEqual(bad[0]['response_kind'],'evidence');self.assertEqual(bad[0]['finish_reason'],'length')
   self.assertIn('Only private thinking',bad[0]['traceback'])
   self.assertEqual(format_diagnostics(traces)['evidence_errors'],2)
   with self.assertRaises(JudgeResponseError) as answer_error:judge.answer_correct('Q','red','blue')
   self.assertEqual(answer_error.exception.raw_response,raw)
   self.assertEqual(answer_error.exception.response_kind,'answer')

 def test_invalid_online_p2_decision_is_audited_without_repair(self):
  raw='I think it is red. '+decision('sufficient','red')
  class Actor:
   def sample(self,*args,**kwargs):return SimpleNamespace(text=raw,active=True)
  class Judge:
   def answer_correct(self,*args):raise AssertionError('Invalid decision must not be scored as an answer')
  c=dict(CFG,group_size=2)
  row={'question':'Q','options':{},'reference':'red','evidence':['crop.png'],
       'label':'sufficient','quality':.9,'confidence':.9}
  _,rewards,eligible,_,traces=verification_group(Actor(),row,Judge(),c)
  self.assertEqual(format_diagnostics(traces)['invalid_p2_decisions'],2)
  self.assertTrue(all(x<0 for x in rewards));self.assertEqual(eligible,[True,True])
  self.assertEqual([r['raw_response'] for r in traces if r.get('event')=='invalid_decision'],[raw,raw])

 def test_masks_and_strict_interfaces(self):
  p1=acquisition_messages('Q','original.png',{},['<think>SECRET_P1</think><crop>{"bbox":[0,0,1,1]}</crop>'],Feedback(True))
  p2=verification_messages('Q',{},Evidence(('crop.png',)))
  self.assertIn('SECRET_P1',json.dumps(p1));self.assertNotIn('SECRET_P1',json.dumps(p2));self.assertNotIn('original.png',json.dumps(p2))
  with self.assertRaises(TypeError):acquisition_messages('Q','x',{},[],parse_decision(decision('sufficient','red')))
  with self.assertRaises(ValueError):parse_crop('<crop>{"bbox":[0,0,1,1],"caption":"secret"}</crop>')
  with self.assertRaises(ValueError):parse_crop('<answer>A</answer><crop>{"bbox":[0,0,1,1]}</crop>')
  with self.assertRaises(ValueError):parse_decision(decision('insufficient','guessed answer'))
  with self.assertRaises(ValueError):parse_crop('<crop>{"bbox":[0,0,NaN,1]}</crop>')
 def test_pixels_area_and_reward_gating(self):
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp);im=Image.new('RGB',(100,100),'red');im.save(p/'original.png')
   path,area=crop_pixels(p/'original.png',(.1,.2,.6,.8),p/'crops')
   with Image.open(path) as crop:self.assertEqual(crop.size,(50,60));self.assertEqual(crop.tobytes(),im.crop((10,20,60,80)).tobytes())
   self.assertAlmostEqual(area,.3)
   with self.assertRaises(ValueError):crop_pixels(p/'original.png',(0,0,.01,.01),p/'crops')
  d=parse_decision(decision('sufficient','red'))
  self.assertEqual(p1_reward(False,d,0.01,1,CFG),p1_reward(False,d,1.,3,CFG))
  self.assertGreater(p1_reward(True,d,1.,3,CFG),p1_reward(False,d,0.,0,CFG))
  self.assertLess(p2_reward(d,'insufficient',.1,.9,True,CFG),0) # lucky right still wrong action
  request=parse_decision(decision('insufficient'))
  self.assertGreater(p2_reward(request,'insufficient',.1,.9,False,CFG),0)
  with self.assertRaises(ValueError):p2_reward(request,'uncertain',.5,.5,False,CFG)
 def test_group_normalization_and_loss(self):
  import torch
  a=relative_advantages([1,3,999],[True,True,False]);self.assertTrue(torch.allclose(a,torch.tensor([-1.,1.,0.]),atol=1e-5))
  self.assertEqual(relative_advantages([1,1],[True,True]).abs().sum().item(),0)
  new=torch.tensor([0.,0.],requires_grad=True);loss=clipped_loss_sum(new,torch.zeros(2),1.,.2)
  self.assertAlmostEqual(loss.item(),-2.);loss.backward();self.assertTrue(torch.allclose(new.grad,torch.tensor([-1.,-1.])))
  self.assertEqual(len(indices(3,0,4,1)),1);self.assertEqual(len(indices(3,3,4,1)),1)
 def test_allocation_respected(self):
  with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-a,GPU-b,GPU-c'}):
   self.assertEqual(device_list([1,2]),'GPU-b,GPU-c')
   with self.assertRaises(ValueError):device_list([3])
 def test_frozen_judge_never_gets_reference_during_image_judgment(self):
  requests=[]
  def create(**kwargs):
   requests.append(kwargs)
   return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=decision('insufficient')))])
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp)/'crop.png';Image.new('RGB',(32,32),'red').save(p)
   client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
   judge=FrozenJudge('unused','model','version',client=client)
   d,_=judge.judge('Q',{},Evidence((str(p),)))
   self.assertEqual(d.evidence_label,'insufficient');self.assertEqual(len(requests),1)
   self.assertNotIn('\"reference\":',json.dumps(requests[0]));self.assertNotIn('GT_SECRET',json.dumps(requests[0]))
 def test_rollout_uses_frozen_labels_not_answer_luck(self):
  class Engine:
   def __init__(self):self.inputs=[]
   def sample(self,messages,tag,active=True):
    self.inputs.append(messages)
    return SimpleNamespace(text='<think>SECRET_P1</think><crop>{"bbox":[0,0,1,1]}</crop>',active=active)
  class Judge:
   def __init__(self):self.n=0
   def judge(self,q,options,evidence):
    self.n+=1
    d=parse_decision(decision('insufficient') if self.n%2 else decision('sufficient','red'))
    return d,'SECRET_P2'
   def answer_correct(self,*args):return True
  c=dict(CFG,group_size=2,max_rounds=2,probe_crops_per_group=0)
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp);Image.new('RGB',(56,56),'red').save(p/'original.png')
   actor=Engine();q=Question('id','Q',str(p/'original.png'),'GT_SECRET',{})
   groups,rewards,eligible,bank,traces=acquisition_group(actor,q,Judge(),c,p/'crops','FROZEN_V1','group')
   self.assertEqual(len(actor.inputs),4);self.assertEqual([r['label'] for r in bank],['insufficient','sufficient']*2)
   self.assertNotIn('SECRET_P1',json.dumps(bank));self.assertNotIn('SECRET_P2',json.dumps(bank))
   self.assertNotIn('GT_SECRET',json.dumps(actor.inputs));self.assertNotIn('SECRET_P2',json.dumps(actor.inputs))
   for row in bank:
    verifier=verification_messages(row['question'],row['options'],Evidence(tuple(row['evidence'])))
    self.assertNotIn('GT_SECRET',json.dumps(verifier));self.assertNotIn('label',verifier[1]['content'][0]['text'])

 def test_online_p2_uses_current_actor_without_remote_image_judge_and_keeps_collectives_aligned(self):
  class Actor:
   def __init__(self,invalid=False):self.calls=[];self.samples=0;self.invalid=invalid
   def sample(self,messages,tag,active=True):
    self.calls.append(('sample',tag,active,copy.deepcopy(messages)))
    self.samples+=1
    text='invalid crop' if self.invalid else '<think>P1_SECRET</think><crop>{"bbox":[0,0,1,1]}</crop>'
    return SimpleNamespace(text=text,active=active)
   def infer(self,messages,tag):
    self.calls.append(('infer',tag,False,copy.deepcopy(messages)))
    return '<think>P2_SECRET</think>'+decision('sufficient','red')
  class Judge:
   def __init__(self):self.answers=[]
   def judge(self,*args):raise AssertionError('Online P2 must not call external image judgment')
   def answer_correct(self,*args):self.answers.append(args);return True
  c=dict(CFG,group_size=2,max_rounds=3,probe_crops_per_group=1)
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp);Image.new('RGB',(56,56),'red').save(p/'original.png')
   q=Question('id','Q',str(p/'original.png'),'GT_SECRET',{})
   valid,bad=Actor(),Actor(invalid=True);judge=Judge()
   groups,rewards,eligible,bank,traces=acquisition_group(valid,q,judge,c,p/'a','ONLINE_0','rank0_group0',online_p2=True)
   badresult=acquisition_group(bad,q,Judge(),c,p/'b','ONLINE_0','rank1_group0',online_p2=True)
   # Valid episodes finish after the first crop, invalid ones before any crop.
   # Both still perform 6 P1 + 7 P2 generations, in exactly the same order.
   self.assertEqual([x[:2] for x in valid.calls],[x[:2] for x in bad.calls])
   self.assertEqual(sum(x[0]=='sample' for x in valid.calls),6)
   self.assertEqual(sum(x[0]=='infer' for x in valid.calls),7)
   self.assertEqual([len(x) for x in groups],[3,3])
   self.assertEqual(len(bank),3);self.assertEqual(len(badresult[3]),1) # probe only
   self.assertEqual(len(judge.answers),2)
   for actor in [valid,bad]:
    for kind,tag,active,messages in actor.calls:
     self.assertNotIn('GT_SECRET',json.dumps(messages))
     if kind=='infer':
      self.assertEqual(messages[0]['content'],P2_SYSTEM)
      self.assertNotIn('P1_SECRET',json.dumps(messages));self.assertNotIn('P2_SECRET',json.dumps(messages))
      self.assertNotIn('original.png',json.dumps(messages))
   self.assertNotIn('P1_SECRET',json.dumps(bank));self.assertNotIn('P2_SECRET',json.dumps(bank))
   self.assertEqual({r['evidence_source'] for r in bank},{'current_shared_p2'})

 def test_seven_ranks_cover_all_training_rows_with_only_one_padding_row(self):
  order=[indices(44673,r,7,42) for r in range(7)]
  self.assertEqual({len(x) for x in order},{6382})
  self.assertEqual(set(i for x in order for i in x),set(range(44673)))
  self.assertEqual(sum(map(len,order)),44674)
  self.assertEqual(CFG['groups_per_phase'],0)

class PipelineTests(unittest.TestCase):
 def test_zero_update_exit_keeps_format_summary_and_original_responses(self):
  import torch
  import eav_selfplay.train_phase as phase
  class Actor:
   device=torch.device('cpu')
   def __init__(self,*args):pass
   def sample(self,messages,tag,active=True):return SimpleNamespace(text='<crop>{"bbox_2d":[1,2,3,4]}</crop>',active=active)
   def update(self,*args):return {'updated':False,'flat_group':True,'loss':0.}
   def save(self,*args):raise AssertionError('Zero-update run must not save a successful model')
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp);image=p/'original.png';Image.new('RGB',(56,56),'red').save(image)
   data=p/'data.jsonl';data.write_text(json.dumps({'qid':'q','question':'Q','image':str(image),'reference':'red','options':{}})+'\n')
   cfg=dict(CFG,backend='native',group_size=2,max_rounds=1,probe_crops_per_group=0)
   config=p/'config.json';config.write_text(json.dumps(cfg));out=p/'phase'
   argv=['phase','--config',str(config),'--role','p1','--model','unused','--data',str(data),
         '--output',str(out),'--judge-url','unused','--judge-version','v','--cycle','1']
   with patch.object(sys,'argv',argv),patch.object(phase,'Engine',Actor),patch.object(phase,'FrozenJudge'),patch.object(torch.cuda,'is_available',return_value=False),patch.dict(os.environ,{'WORLD_SIZE':'1','RANK':'0'}):
    with self.assertRaisesRegex(RuntimeError,'Zero optimizer updates'):phase.main()
   report=json.loads((out/'format_summary_rank0.json').read_text())
   self.assertEqual(report['invalid_crops'],2);self.assertEqual(report['optimizer_updates'],0)
   self.assertEqual(report['prompt_protocol'],PROMPT_PROTOCOL)
   traces=json.loads(next((out/'private_traces').glob('*.json')).read_text())
   self.assertEqual(sum(r.get('event')=='invalid_crop' for r in traces),2)
   self.assertFalse((out/'phase_done.json').exists())

 def test_cluster_entry_preserves_traceback_and_stops_before_full_train_on_smoke_failure(self):
  import subprocess
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp);fake=p/'python'
   fake.write_text('#!/usr/bin/env bash\nfor arg in "$@"; do\n'
                   '  if [[ "$arg" == "scripts/preflight_selfplay.py" ]]; then exit 0; fi\n'
                   'done\necho "Traceback: simulated smoke training failure" >&2\nexit 23\n')
   fake.chmod(0o755)
   script=Path(__file__).resolve().parents[1]/'scripts/run_selfplay_8xh100.sh'
   run=subprocess.run(['bash',str(script),'--phase','all','--train-python',str(fake),
                       '--output-root',str(p/'outputs')],capture_output=True,text=True)
   self.assertEqual(run.returncode,23)
   output=next((p/'outputs').iterdir())
   self.assertIn('Traceback:',(output/'logs/smoke.log').read_text())
   self.assertFalse((output/'logs/train.log').exists())
   self.assertIn('exit_code=23',(output/'logs/launcher.log').read_text())

 def training_fixture(self,p):
  from eav_selfplay.io import write_json
  image=p/'original.png';Image.new('RGB',(56,56),'red').save(image)
  data=p/'data.jsonl';data.write_text(json.dumps({'qid':'q','question':'Q','image':str(image),'reference':'red','options':{}})+'\n')
  cfg=copy.deepcopy(CFG);cfg.update(train_data=str(data),output=str(p/'run'),backend='native',judge_python=sys.executable)
  for field in ['policy_model','judge_model']:
   model=p/field;model.mkdir();write_json(model/'config.json',{'update_index':0});cfg[field]=str(model)
  config=p/'config.json';write_json(config,cfg)
  return cfg,config

 def fake_phase(self,phases,interrupt=None):
  from eav_selfplay.io import write_json,digest
  def launch(cfg,effective,role,model,data,out,judge,version,cycle):
   phases.append((cycle,role,str(model),version))
   if interrupt is not None and interrupt(cycle,role):raise RuntimeError('Interrupted before phase initialization')
   out.mkdir(parents=True);hf=out/'hf';hf.mkdir()
   previous=json.loads((Path(model)/'config.json').read_text())['update_index']
   write_json(hf/'config.json',{'update_index':previous+1})
   (out/'evidence_bank.jsonl').write_text('')
   write_json(out/'phase_done.json',{'checkpoint':str(hf),'model_input':str(model),
      'policy_architecture':'shared_policy_roles_v1','training_protocol':'online_shared_p2_v1','role':role,'cycle':cycle,
      'judge_version':version,'data_sha256':digest(data)})
  return launch

 def test_data_conversion_groups_images_and_preserves_option_variants(self):
  import pyarrow as pa,pyarrow.parquet as pq,io
  from eav_selfplay.prepare_inputs import main
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp);buffer=io.BytesIO();Image.new('RGB',(56,56),'red').save(buffer,format='PNG')
   a={'images':[{'bytes':buffer.getvalue(),'path':None}], 'extra_info':{'question':'Color?','answer':'A'},
      'reward_model':{'ground_truth':'A'},'options':{'A':'red','B':'blue'}}
   b=copy.deepcopy(a);b['options']={'A':'blue','B':'red'};b['reward_model']['ground_truth']='B';b['extra_info']['answer']='B'
   parquet=p/'source.parquet';pq.write_table(pa.Table.from_pylist([a,a,b]),parquet)
   with patch.object(sys,'argv',['prepare_data','--parquet',str(parquet),'--output',str(p/'out')]):main()
   train=[json.loads(l) for l in (p/'out/train.jsonl').read_text().splitlines()]
   held=[json.loads(l) for l in (p/'out/heldout.jsonl').read_text().splitlines()]
   self.assertEqual(len(train)+len(held),2);self.assertTrue(not train or not held)
   self.assertEqual(len({r['qid'] for r in train+held}),2)
 def test_one_policy_chain_keeps_one_answer_judge_and_resumes_completed_phases(self):
  import train_selfplay as entry
  import torch
  from eav_selfplay.io import write_json
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp);c,config=self.training_fixture(p);judges=[];phases=[]
   class Server:
    url='http://unused/v1'
    def __init__(self,model,*args):judges.append(str(model))
    def __enter__(self):return self
    def __exit__(self,*args):pass
   with patch.object(entry,'JudgeServer',Server),patch.object(entry,'launch',side_effect=self.fake_phase(phases)),patch.object(torch.cuda,'is_available',return_value=True),patch.object(torch.cuda,'device_count',return_value=8):
    with patch.object(sys,'argv',['train_selfplay','--config',str(config)]):entry.main()
    self.assertEqual(len(phases),4);self.assertEqual(judges[0],c['judge_model'])
    expected=str(p/'run/cycle_001/process2/hf')
    self.assertEqual(judges,[c['judge_model']])
    self.assertEqual([phase[2] for phase in phases],[c['policy_model'],
       str(p/'run/cycle_001/process1/hf'),expected,str(p/'run/cycle_002/process1/hf')])
    complete=json.loads((p/'run/complete.json').read_text())
    self.assertEqual(complete['policy_checkpoint'],str(p/'run/cycle_002/process2/hf'))
    self.assertEqual(json.loads((Path(complete['policy_checkpoint'])/'config.json').read_text())['update_index'],4)
    self.assertEqual(json.loads((Path(c['policy_model'])/'config.json').read_text())['update_index'],0)
    self.assertEqual(json.loads((Path(judges[0])/'config.json').read_text())['update_index'],0)
    with patch.object(sys,'argv',['train_selfplay','--config',str(config),'--resume']):entry.main()
    self.assertEqual(len(phases),4);self.assertEqual(len(judges),1)
    (p/'run/cycle_002/process2/phase_done.json').unlink()
    write_json(p/'run/cycle_002/process2/progress_rank0.json',{'groups_completed':1})
    with patch.object(sys,'argv',['train_selfplay','--config',str(config),'--resume']):
     with self.assertRaises(RuntimeError):entry.main()

 def test_resume_after_p1_uses_its_checkpoint_and_keeps_answer_judge_fixed(self):
  import train_selfplay as entry
  import torch
  from eav_selfplay.io import write_json
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp);c,config=self.training_fixture(p);phases=[];judges=[]
   class Server:
    url='http://unused/v1'
    def __init__(self,model,*args):judges.append(str(model))
    def __enter__(self):return self
    def __exit__(self,*args):pass
   with patch.object(entry,'JudgeServer',Server),patch.object(torch.cuda,'is_available',return_value=True),patch.object(torch.cuda,'device_count',return_value=8):
    with patch.object(entry,'launch',side_effect=self.fake_phase(phases,lambda cycle,role:cycle==2 and role=='p2')):
     with patch.object(sys,'argv',['train_selfplay','--config',str(config)]):
      with self.assertRaisesRegex(RuntimeError,'Interrupted'):entry.main()
    with patch.object(entry,'launch',side_effect=self.fake_phase(phases)):
     with patch.object(sys,'argv',['train_selfplay','--config',str(config),'--resume']):entry.main()
   previous=str(p/'run/cycle_001/process2/hf')
   self.assertEqual(judges,[c['judge_model'],c['judge_model']])
   self.assertEqual(phases[-1][2],str(p/'run/cycle_002/process1/hf'))
   self.assertEqual(sum(role=='p1' for _,role,_,_ in phases),2)
   # A completed chain can be audited/resumed without starting a GPU service.
   with patch.object(entry,'JudgeServer') as server,patch.object(entry,'launch') as launch,patch.object(torch.cuda,'is_available',return_value=False):
    with patch.object(sys,'argv',['train_selfplay','--config',str(config),'--resume']):entry.main()
    server.assert_not_called();launch.assert_not_called()
   done=p/'run/cycle_002/process2/phase_done.json';record=json.loads(done.read_text())
   record['model_input']=c['policy_model'];write_json(done,record)
   with patch.object(entry,'JudgeServer') as server,patch.object(entry,'launch') as launch:
    with patch.object(sys,'argv',['train_selfplay','--config',str(config),'--resume']):
     with self.assertRaisesRegex(ValueError,'shared-policy chain'):entry.main()
    server.assert_not_called();launch.assert_not_called()

 def test_shared_configuration_validation_and_cpu_preflight(self):
  import io
  import train_selfplay as entry
  self.assertIn('policy_model',CFG);self.assertNotIn('p1_model',CFG);self.assertNotIn('p2_model',CFG)
  with self.assertRaisesRegex(ValueError,'one policy_model'):entry.validate(dict(CFG,p1_model='old'))
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp);cfg,config=self.training_fixture(p)
   output=io.StringIO()
   with patch.object(sys,'argv',['train_selfplay','--config',str(config),'--validate-only']),patch('sys.stdout',output):entry.main()
   result=json.loads(output.getvalue());self.assertEqual(result['policy_model'],cfg['policy_model'])
   self.assertEqual(result['policy_architecture'],'shared_policy_roles_v1');self.assertFalse((p/'run').exists())

 def test_inference_uses_one_client_and_model_with_isolated_role_contexts(self):
  import infer_selfplay as inference
  import io,base64
  requests=[]
  def create(**kwargs):
   requests.append(copy.deepcopy(kwargs));n=len(requests)
   text={1:'<think>SECRET_P1_ONE</think><crop>{"bbox":[0,0,0.5,0.5]}</crop>',
         2:'<think>SECRET_P2_ONE</think>'+decision('insufficient'),
         3:'<think>SECRET_P1_TWO</think><crop>{"bbox":[0.5,0.5,1,1]}</crop>',
         4:decision('sufficient','red')}[n]
   return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
  client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp);image=p/'original.png';Image.new('RGB',(56,56),'red').save(image)
   argv=['infer_selfplay','--url','http://shared/v1','--model','shared_test',
         '--image',str(image),'--question','Q','--output',str(p/'out')]
   with patch.object(inference,'OpenAI',return_value=client) as factory,patch.object(sys,'argv',argv):inference.main()
   factory.assert_called_once();self.assertEqual(len(requests),4)
   self.assertEqual({r['model'] for r in requests},{'shared_test'})
   self.assertIn('SECRET_P1_ONE',json.dumps(requests[2]))
   self.assertNotIn('SECRET_P2_ONE',json.dumps(requests[2]))
   for index in [1,3]:
    request=requests[index];self.assertEqual(request['messages'][0]['content'],P2_SYSTEM)
    self.assertNotIn('SECRET_P1',json.dumps(request));self.assertNotIn('SECRET_P2_ONE',json.dumps(request))
    images=[part for part in request['messages'][1]['content'] if part['type']=='image_url']
    self.assertEqual(len(images),1 if index==1 else 2)
    for part in images:
     with Image.open(io.BytesIO(base64.b64decode(part['image_url']['url'].split(',',1)[1]))) as im:self.assertEqual(im.size,(28,28))
   result=json.loads((p/'out/result.json').read_text())
   self.assertEqual(result['status'],'answered');self.assertEqual(result['model'],'shared_test')
   self.assertEqual(result['policy_architecture'],'shared_policy_roles_v1')

class TinyTrainingTests(unittest.TestCase):
 @unittest.skipUnless(os.environ.get('EAV_TEST_PROCESSOR'),'Set EAV_TEST_PROCESSOR to a local Qwen2.5-VL processor for real optimizer tests.')
 def test_real_multimodal_role_updates_share_weights_and_preserve_frozen_snapshot(self):
  import torch
  from transformers import AutoProcessor,AutoTokenizer,Qwen2_5_VLConfig,Qwen2_5_VLForConditionalGeneration
  from eav_selfplay.engine import Engine,Sample
  torch.set_num_threads(2)
  source=os.environ['EAV_TEST_PROCESSOR']
  proc=AutoProcessor.from_pretrained(source,local_files_only=True,use_fast=False)
  proc.tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True,use_fast=True)
  cfg=Qwen2_5_VLConfig(vocab_size=len(proc.tokenizer),hidden_size=16,intermediate_size=32,num_hidden_layers=2,
    num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=4096,
    rope_scaling={'type':'mrope','mrope_section':[1,1,2]},
    vision_config={'depth':1,'hidden_size':16,'intermediate_size':32,'out_hidden_size':16,'num_heads':2,
                   'patch_size':14,'temporal_patch_size':2,'spatial_merge_size':2,'window_size':112,'fullatt_block_indexes':[0]},
    image_token_id=proc.tokenizer.convert_tokens_to_ids('<|image_pad|>'),video_token_id=proc.tokenizer.convert_tokens_to_ids('<|video_pad|>'),
    vision_start_token_id=proc.tokenizer.convert_tokens_to_ids('<|vision_start|>'),vision_end_token_id=proc.tokenizer.convert_tokens_to_ids('<|vision_end|>'),
    eos_token_id=proc.tokenizer.eos_token_id,pad_token_id=proc.tokenizer.pad_token_id)
  cfg._attn_implementation='sdpa'
  a=Qwen2_5_VLForConditionalGeneration(cfg);b=copy.deepcopy(a).eval()
  for parameter in b.parameters():parameter.requires_grad_(False)
  settings=dict(CFG,backend='native',gradient_checkpointing=False,learning_rate=.01,max_context=4096)
  engine=Engine(settings,source,2,model=a,processor=proc)
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'image.png';Image.new('RGB',(56,56),'red').save(path)
   inputs=engine.encode(acquisition_messages('Q',str(path),{},[],Feedback(True)))
   # Current-policy P2 inference neither computes training logp nor changes weights.
   previous=a.lm_head.weight.detach().clone()
   with patch.object(engine,'logp',side_effect=AssertionError('P2 inference must not compute a training logp')):
    settings['max_new_tokens']=2
    text=engine.infer(verification_messages('Q',{},Evidence((str(path),))),'decision')
    self.assertIsInstance(text,str)
   self.assertTrue(torch.equal(previous,a.lm_head.weight))
   samples=[]
   for label in ['A','B']:
    tokens=proc.tokenizer(label,add_special_tokens=False)['input_ids'];item=Sample(inputs,tokens,label,None)
    item.old_logp=engine.logp(item,False).detach().cpu().clone();samples.append([item])
   before=a.lm_head.weight.detach().clone();other=b.lm_head.weight.detach().clone()
   result=engine.update(samples,[1.,-1.],[True,True])
   self.assertTrue(result['updated']);self.assertFalse(torch.equal(before,a.lm_head.weight));self.assertTrue(torch.equal(other,b.lm_head.weight))
   before=a.lm_head.weight.detach().clone();result=engine.update(samples,[1.,1.],[True,True])
   self.assertFalse(result['updated']);self.assertTrue(torch.equal(before,a.lm_head.weight))
   # The P2 phase starts with the SAME updated policy; only the judge snapshot is separate.
   after_p1=a.lm_head.weight.detach().clone()
   second=Engine(settings,source,2,model=a,processor=proc)
   self.assertIs(second.bare,engine.bare);self.assertTrue(torch.equal(second.bare.lm_head.weight,after_p1))
   p2inputs=second.encode(verification_messages('Q',{},Evidence((str(path),))))
   p2samples=[]
   for label in ['A','B']:
    item=Sample(p2inputs,proc.tokenizer(label,add_special_tokens=False)['input_ids'],label,None)
    item.old_logp=second.logp(item,False).detach().cpu().clone();p2samples.append([item])
   a_before=a.lm_head.weight.detach().clone();b_before=b.lm_head.weight.detach().clone()
   self.assertTrue(second.update(p2samples,[1.,-1.],[True,True])['updated'])
   self.assertFalse(torch.equal(a_before,a.lm_head.weight));self.assertTrue(torch.equal(b_before,b.lm_head.weight))
   second.save(Path(temp)/'saved');self.assertTrue((Path(temp)/'saved/hf/config.json').exists())
   restored=Qwen2_5_VLForConditionalGeneration.from_pretrained(Path(temp)/'saved/hf',local_files_only=True)
   self.assertTrue(torch.equal(restored.lm_head.weight,a.lm_head.weight))

if __name__=='__main__':unittest.main()
