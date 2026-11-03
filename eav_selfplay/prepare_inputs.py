"""Convert local DeepEyes-style parquet to answer-only EAV inputs, grouped by image."""
import argparse,hashlib,io,json,os,tempfile
from pathlib import Path
from PIL import Image
from .io import write_json,digest

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parquet',nargs='+',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--eval-fraction',type=float,default=.05)
    parser.add_argument('--skip-invalid',action='store_true')
    args=parser.parse_args()
    if not 0<=args.eval_fraction<1:parser.error('Invalid holdout fraction.')
    if args.output.exists():raise ValueError('Use a new preparation output directory.')
    import pyarrow.parquet as pq
    args.output.mkdir(parents=True);images=args.output/'images';images.mkdir()
    train=[];held=[];errors=[];seen={}
    for path in args.parquet:
        file=pq.ParquetFile(path)
        for batch in file.iter_batches(batch_size=16):
            for row in batch.to_pylist():
                try:
                    info=row.get('extra_info') or {};reward=row.get('reward_model') or {}
                    question=info.get('question') or row.get('question')
                    answer=reward.get('ground_truth') or info.get('answer') or row.get('answer')
                    image=row.get('images',row.get('image'))
                    if isinstance(image,list):
                        if len(image)!=1:raise ValueError('This image-crop protocol requires exactly one original image; curate multi-image/video samples separately.')
                        image=image[0]
                    if isinstance(image,dict):
                        raw=image.get('bytes')
                        if raw is None:
                            q=Path(image['path']);q=q if q.is_absolute() else path.parent/q;raw=q.read_bytes()
                    elif isinstance(image,str):
                        q=Path(image);q=q if q.is_absolute() else path.parent/q;raw=q.read_bytes()
                    else:raise ValueError('Missing image payload.')
                    if not isinstance(question,str) or not isinstance(answer,str) or not question.strip() or not answer.strip():raise ValueError('Need question and verifiable answer strings.')
                    with Image.open(io.BytesIO(raw)) as src:
                        rgb=src.convert('RGB');identity=hashlib.sha256(rgb.tobytes()+str(rgb.size).encode()).hexdigest()
                        target=images/(identity+'.png')
                        if not target.exists():rgb.save(target)
                    qid=hashlib.sha256((identity+'\n'+question+'\n'+json.dumps(row.get('options') or {},sort_keys=True)).encode()).hexdigest()
                    if qid in seen:
                        if seen[qid]!=answer:raise ValueError('Conflicting references for identical image/question/options.')
                        continue
                    seen[qid]=answer
                    r={'qid':qid,'question':question,'reference':answer,'image':str(target.resolve()),'options':row.get('options') or {}}
                    if not isinstance(r['options'],dict):raise ValueError('Options must be a label-to-content dictionary.')
                    destination=held if int(identity[:16],16)/2**64<args.eval_fraction else train
                    destination.append(r)
                except Exception as error:
                    errors.append({'file':str(path),'error':str(error)})
                    if not args.skip_invalid:
                        write_json(args.output/'errors.json',errors);raise
    for name,data in [('train',train),('heldout',held)]:
        with (args.output/(name+'.jsonl')).open('w',encoding='utf-8') as f:
            for r in data:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    write_json(args.output/'manifest.json',{'sources':{str(p):digest(p) for p in args.parquet},'train':len(train),'heldout':len(held),'invalid':len(errors),
                'policy':'Image-grouped deterministic split; no rationale/tool trajectory copied. Existing benchmark evaluations are not automatically added.'})
    write_json(args.output/'errors.json',errors)
    print(json.dumps({'train':len(train),'heldout':len(held),'invalid':len(errors)}))

if __name__=='__main__':main()
