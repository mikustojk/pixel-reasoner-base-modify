"""One model service, two isolated role contexts; no cross-role reasoning text."""
import argparse,base64,json
from pathlib import Path
from openai import OpenAI
from eav_selfplay.core import *
from eav_selfplay.io import write_json
from eav_selfplay.judge import FrozenJudge


def api_messages(messages):
    result=[]
    for message in messages:
        if isinstance(message['content'],str):result.append(message);continue
        parts=[]
        for p in message['content']:
            if p['type']=='text':parts.append(p)
            else:
                from PIL import Image
                import io
                buffer=io.BytesIO()
                with Image.open(p['image']) as im:im.convert('RGB').save(buffer,format='PNG')
                encoded=base64.b64encode(buffer.getvalue()).decode()
                parts.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+encoded}})
        result.append({'role':message['role'],'content':parts})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',required=True,help='Single shared policy service for both roles.')
    p.add_argument('--model',default='eav_shared',help='Same served-model alias for P1 and P2.')
    p.add_argument('--image',required=True,type=Path);p.add_argument('--question',required=True)
    p.add_argument('--options',default='{}',help='JSON label-to-content dictionary')
    p.add_argument('--max-rounds',type=int,default=1);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--crop-coordinate-mode',choices=['normalized','pixel'],default='normalized')
    args=p.parse_args()
    if args.output.exists():raise ValueError('Use a new output directory.')
    if args.max_rounds<1:raise ValueError('Invalid observation budget.')
    options=json.loads(args.options);args.output.mkdir(parents=True)
    client=OpenAI(base_url=args.url,api_key='EMPTY',timeout=120,max_retries=2)
    single_crop=args.max_rounds==1
    verifier=FrozenJudge(args.url,args.model,'inference_shared_policy',client=client,single_crop=single_crop)
    own=[];evidence=[];trace=[];status='insufficient_evidence' if single_crop else 'observation_budget_exhausted';answer=None
    for turn in range(args.max_rounds):
        messages=acquisition_messages(args.question,args.image,options,own,Feedback(True),args.crop_coordinate_mode,single_crop)
        response=client.chat.completions.create(model=args.model,messages=api_messages(messages),temperature=0.,max_tokens=512)
        text=response.choices[0].message.content or '';own.append(text)
        try:
            box=parse_crop_for_image(text,args.image,args.crop_coordinate_mode);image,area=crop_pixels(args.image,box,args.output/'crops')
            evidence.append(image)
            decision,raw=verifier.judge(args.question,options,Evidence(tuple(evidence)))
        except Exception as error:
            trace.append({'turn':turn,'p1_private':text,'error':str(error)});status='invalid_or_failed_turn';break
        trace.append({'turn':turn,'p1_private':text,'bbox_executor_only':box,'area':area,
                      'p2_private':raw,'decision_action':decision.action})
        if decision.action=='answer':answer=decision.answer;status='answered';break
        if single_crop:status='insufficient_evidence';break
    write_json(args.output/'private_trace.json',trace)
    write_json(args.output/'result.json',{'status':status,'answer':answer,'evidence':evidence,
               'policy_architecture':'shared_policy_roles_v1','crop_coordinate_mode':args.crop_coordinate_mode,
               'model':args.model,'url':args.url,
               'policy':'Only cropped pixels cross from P1 to P2; traces are audit-only.',
               'single_crop_only':single_crop})
    print(json.dumps({'status':status,'answer':answer},ensure_ascii=False))

if __name__=='__main__':main()
