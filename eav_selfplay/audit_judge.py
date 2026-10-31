"""Compare frozen visual judgments against a small HUMAN-reviewed evidence set."""
import argparse,json
from collections import Counter
from pathlib import Path
from .core import Evidence
from .judge import FrozenJudge
from .io import rows,write_json,digest

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--url',required=True);p.add_argument('--model',default='eav_judge')
 p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
 args=p.parse_args()
 if args.output.exists():raise ValueError('Use a new audit output directory.')
 judge=FrozenJudge(args.url,args.model,'bootstrap-audit');result=[];matrix=Counter();failures=0
 for r in rows(args.input):
  if r['expected_label'] not in {'sufficient','insufficient','uncertain'}:raise ValueError('Invalid human label.')
  try:
   decision,raw=judge.judge(r['question'],r.get('options',{}),Evidence(tuple(r['evidence'])))
   matrix[(r['expected_label'],decision.evidence_label)]+=1
   result.append({'id':r['id'],'expected':r['expected_label'],'predicted':decision.evidence_label,
                  'confidence':decision.confidence,'quality':decision.quality,'raw':raw})
  except Exception as error:failures+=1;result.append({'id':r['id'],'error':str(error)})
 success=len(result)-failures;correct=sum(a==b and count or 0 for (a,b),count in matrix.items())
 args.output.mkdir(parents=True)
 write_json(args.output/'records.json',result)
 write_json(args.output/'summary.json',{'input_sha256':digest(args.input),'total':len(result),'parsed':success,'failed':failures,
             'label_agreement':correct/success if success else None,
             'confusion':[{'human':a,'model':b,'count':n} for (a,b),n in matrix.items()],
             'warning':'Human audit agreement is not a guarantee for the remaining pseudo-labels; self-reported confidence is uncalibrated.'})
 if failures:raise RuntimeError('Some frozen-judge outputs failed parsing; inspect the audit.')

if __name__=='__main__':main()
