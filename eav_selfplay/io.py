import hashlib,json,os,tempfile
from pathlib import Path
from .core import Question


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.tmp')
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
        os.replace(tmp,path)
    finally:Path(tmp).unlink(missing_ok=True)


def rows(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():yield json.loads(line)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def load_questions(path):
    result=[];seen=set()
    for r in rows(path):
        if set(r)!={'qid','question','image','reference','options'}:
            raise ValueError('Use prepared question/image/reference/options JSONL; do not pass old chat trajectories.')
        q=Question(**r)
        if q.qid in seen or not q.question or not q.reference or not Path(q.image).is_file():
            raise ValueError('Invalid/duplicate training question or image.')
        seen.add(q.qid);result.append(q)
    if not result:raise ValueError('Empty training data.')
    return result
