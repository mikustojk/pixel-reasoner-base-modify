"""Frozen multimodal judge. Crop judgments are independent of QA correctness."""
import base64,json,re,time
from pathlib import Path
from .core import parse_decision,verification_messages,Evidence

class JudgeResponseError(ValueError):
    """Keep the unmodified completion when a judge response cannot be parsed."""
    def __init__(self,kind,reason,raw_response,finish_reason=None):
        super().__init__(f'{kind} response parse failed: {reason}')
        self.response_kind=kind;self.raw_response=raw_response;self.finish_reason=finish_reason

class FrozenJudge:
    def __init__(self,url,model,version,timeout=120,retries=2,client=None):
        if client is None:
            from openai import OpenAI
            client=OpenAI(base_url=url,api_key='EMPTY',timeout=timeout,max_retries=retries)
        self.client=client;self.model=model;self.version=version

    def judge(self,question,options,evidence):
        messages=verification_messages(question,options,evidence)
        content=[]
        for p in messages[1]['content']:
            if p['type']=='text':content.append(p)
            else:
                encoded=base64.b64encode(Path(p['image']).read_bytes()).decode()
                content.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+encoded}})
        request=[messages[0],{'role':'user','content':content}]
        result=self.client.chat.completions.create(model=self.model,messages=request,temperature=0.,max_tokens=1024)
        raw=result.choices[0].message.content or ''
        try:return parse_decision(raw),raw
        except (ValueError,TypeError) as error:
            raise JudgeResponseError('evidence',str(error),raw,
                                     getattr(result.choices[0],'finish_reason',None)) from error

    def answer_correct(self,question,reference,prediction,mode='semantic'):
        def norm(x):return str(x).strip().rstrip('.。').strip().casefold()
        if norm(reference)==norm(prediction):return True
        if mode=='exact':return False
        # A separate reward-side request; references never enter either policy input.
        prompt={'question':question,'reference':reference,'prediction':prediction}
        response=self.client.chat.completions.create(model=self.model,temperature=0.,max_tokens=128,
            messages=[{'role':'system','content':'Compare the prediction with the reference answer by meaning. Do not judge image sufficiency. Your entire response must be exactly {"correct": true} or {"correct": false}. No reasoning, preamble, markdown or additional fields.'},
                      {'role':'user','content':json.dumps(prompt,ensure_ascii=False)}])
        text=response.choices[0].message.content or ''
        try:
            match=re.search(r'\{[^{}]*\}',text)
            if not match:raise ValueError('Answer judge returned no JSON.')
            obj=json.loads(match[0])
            if set(obj)!={'correct'} or type(obj['correct']) is not bool:raise ValueError('Invalid answer score.')
            return obj['correct']
        except (ValueError,TypeError) as error:
            raise JudgeResponseError('answer',str(error),text,
                                     getattr(response.choices[0],'finish_reason',None)) from error
