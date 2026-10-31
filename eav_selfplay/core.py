"""Strict role interfaces, physical crops, and rewards independent of answer luck."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from PIL import Image

LABELS = {'sufficient', 'insufficient', 'uncertain'}
PROMPT_PROTOCOL='explicit_action_examples_v2'
P1_SYSTEM = '''You acquire visual evidence, not the final answer. Inspect the original image and select ONE rectangular region containing the evidence needed for the question. Include every necessary object/reference for spatial questions. Do not solve or answer the question in your response.
OUTPUT CONTRACT:
- Return exactly one <crop>...</crop> block containing valid JSON with exactly one key: "bbox".
- "bbox" is an array of FOUR NUMBERS [left, top, right, bottom], ALL between 0 and 1. These are fractions of the ORIGINAL image width/height, never pixel coordinates or a 0-to-1000 coordinate scale. For a point (x,y), use x/original_width and y/original_height.
- left < right and top < bottom. Select the actual region; examples below demonstrate FORMAT ONLY.
- Do not use bbox_2d, XML attributes, a self-closing tag, markdown, comments, captions, extra JSON keys, or text inside the numeric array.
- Prefer the crop block directly. If you need a private visual observation first, put it entirely inside ONE <think>...</think> block and keep it brief. No prose is allowed outside these blocks. End immediately after </crop>.
VALID FORMAT EXAMPLE, a central region:
<crop>{"bbox":[0.25,0.25,0.75,0.75]}</crop>
VALID FORMAT EXAMPLE, a different region:
<think>The relevant visual details are near the upper left.</think><crop>{"bbox":[0.05,0.1,0.45,0.6]}</crop>
The entire image, ONLY when needed, is <crop>{"bbox":[0,0,1,1]}</crop>.
Your thoughts and coordinates are NOT sent to the verifier. Only the cropped pixels are sent. Choose your own evidence region; do not blindly repeat an example.'''
P1_OUTPUT_REMINDER='Return one <crop> block with valid JSON and the sole key "bbox". Use four numeric fractions in [0,1], not pixels. End after </crop>.'
P2_SYSTEM = '''You independently verify visual evidence. You receive the question and cropped evidence only, not the original image, crop coordinates, or the acquirer's thoughts. Judge VISIBLE evidence, not whether you can guess an answer. Related but blurry, illegible, incomplete, or missing-reference evidence is insufficient.
OUTPUT CONTRACT: Prefer ONE <decision>...</decision> block directly. If you need private reasoning first, enclose ALL of it in ONE <think>...</think> block. Your response must start with <decision> or <think>, never ordinary prose, a heading, or markdown. Do not put explanations outside those blocks. End immediately after </decision>.
Inside <decision>, output exactly ONE valid JSON object with precisely these fields: action, evidence_label, quality, confidence, answer. Use double-quoted JSON strings and numeric quality/confidence, not comments or extra fields.
A sufficient-evidence FORMAT example is <decision>{"action":"answer","evidence_label":"sufficient","quality":0.9,"confidence":0.9,"answer":"your actual answer"}</decision>.
An insufficient-evidence FORMAT example is <decision>{"action":"request_more","evidence_label":"insufficient","quality":0.1,"confidence":0.9,"answer":""}</decision>.
An uncertain-evidence FORMAT example is <decision>{"action":"request_more","evidence_label":"uncertain","quality":0.4,"confidence":0.4,"answer":""}</decision>.
When unsure, use action=request_more, evidence_label=uncertain, and answer="". Estimate quality and confidence for the actual crop; do NOT copy example numbers. Both are numbers in [0,1]. quality rates visible task-relevant information; confidence rates your evidence judgment. Answer only when sufficient, otherwise request more. Do not call a tool yourself.'''
P2_OUTPUT_REMINDER='Return exactly one <decision>JSON</decision> block. Any optional reasoning must be inside <think>...</think>. No other prose or markdown; end after </decision>.'

@dataclass(frozen=True)
class Question:
    qid: str
    question: str
    image: str
    reference: str
    options: dict

@dataclass(frozen=True)
class Evidence:
    """The ONLY acquisition payload visible to P2. No bbox/caption/private text."""
    images: tuple

@dataclass(frozen=True)
class Feedback:
    """The ONLY verifier payload visible to P1."""
    request_more: bool

@dataclass(frozen=True)
class Decision:
    action: str
    evidence_label: str
    quality: float
    confidence: float
    answer: str


def tagged_json(text, tag):
    matches = list(re.finditer(r'<' + tag + r'>(.*?)</' + tag + r'>', text, re.S))
    if len(matches) != 1:
        raise ValueError('Expected exactly one complete ' + tag + ' output.')
    tail = text[matches[0].end():].replace('<|im_end|>', '').replace('<|endoftext|>', '').strip()
    if tail:
        raise ValueError('Unexpected content following action.')
    prefix = text[:matches[0].start()].strip()
    if prefix and not re.fullmatch(r'<think>.*?</think>', prefix, re.S):
        raise ValueError('Only private thinking may precede the action.')
    value = json.loads(matches[0][1])
    if not isinstance(value, dict):
        raise ValueError('Action must be a JSON object.')
    return value


def parse_crop(text):
    value = tagged_json(text, 'crop')
    if set(value) != {'bbox'}:
        raise ValueError('Crop payload permits coordinates only.')
    box = value['bbox']
    if not isinstance(box, list) or len(box) != 4 or any(type(x) not in (int,float) or not math.isfinite(x) or not 0 <= x <= 1 for x in box):
        raise ValueError('Invalid normalized bbox.')
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError('Empty bbox.')
    return tuple(box)


def parse_decision(text):
    value = tagged_json(text, 'decision')
    if set(value) != {'action','evidence_label','quality','confidence','answer'}:
        raise ValueError('Unexpected verifier fields.')
    if value['action'] not in {'answer','request_more'} or value['evidence_label'] not in LABELS:
        raise ValueError('Invalid decision.')
    for key in ['quality','confidence']:
        if type(value[key]) not in (int,float) or not math.isfinite(value[key]) or not 0 <= value[key] <= 1:
            raise ValueError('Invalid ' + key)
    if not isinstance(value['answer'], str):
        raise ValueError('Answer must be text.')
    if value['action'] == 'answer':
        if value['evidence_label'] != 'sufficient' or not value['answer'].strip():
            raise ValueError('Answer requires sufficient evidence and nonempty content.')
    elif value['evidence_label'] == 'sufficient' or value['answer'].strip():
        raise ValueError('A reobservation request must not contain an answer.')
    return Decision(**value)


def common_text(question, options):
    # References and teacher labels have no path into this formatter.
    text = 'Question: ' + question
    if options:
        text += '\nOptions:\n' + '\n'.join(str(k)+'. '+str(v) for k,v in options.items())
    return text


def acquisition_messages(question, original_image, options, own_history, feedback):
    if not isinstance(feedback, Feedback):
        raise TypeError('P1 requires binary Feedback, never the verifier response.')
    messages = [{'role':'system','content':P1_SYSTEM}, {'role':'user','content':[
        {'type':'image','image':str(original_image)},
        {'type':'text','text':common_text(question,options)}]}]
    # History contains only this policy's own previous generations.
    for response in own_history:
        messages += [{'role':'assistant','content':response},
                     {'role':'user','content':'Additional visual evidence is requested. Select another useful region.'}]
    if feedback.request_more and not own_history:
        messages[-1]['content'].append({'type':'text','text':'Acquire visual evidence now.'})
    if own_history:messages[-1]['content']+=' '+P1_OUTPUT_REMINDER
    else:messages[-1]['content'].append({'type':'text','text':P1_OUTPUT_REMINDER})
    return messages


def verification_messages(question, options, evidence):
    if not isinstance(evidence, Evidence) or not evidence.images:
        raise TypeError('P2 requires cropped pixel evidence only.')
    # Fresh verification context each round; no prior guesses from either role.
    return [{'role':'system','content':P2_SYSTEM}, {'role':'user','content':[
        {'type':'text','text':common_text(question,options)},
        *[{'type':'image','image':str(path)} for path in evidence.images],
        {'type':'text','text':P2_OUTPUT_REMINDER}]}]


def crop_pixels(original, box, output_dir, min_side=28):
    with Image.open(original) as source:
        image = source.convert('RGB'); width,height = image.size
        x1,y1,x2,y2 = box
        actual = (math.floor(x1*width), math.floor(y1*height), math.ceil(x2*width), math.ceil(y2*height))
        if actual[2]-actual[0] < min_side or actual[3]-actual[1] < min_side:
            raise ValueError('Crop is smaller than minimum pixel size.')
        crop = image.crop(actual)
        digest = hashlib.sha256(crop.tobytes()+str(crop.size).encode()).hexdigest()
        path = Path(output_dir)/(digest+'.png');path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            import os,tempfile
            fd,tmp = tempfile.mkstemp(dir=path.parent,suffix='.png');os.close(fd)
            try:
                crop.save(tmp);os.replace(tmp,path)
            finally:Path(tmp).unlink(missing_ok=True)
    area = (actual[2]-actual[0])*(actual[3]-actual[1])/(width*height)
    return str(path), area


def p1_reward(correct, decision, total_area, calls, cfg, invalid=False):
    if invalid:
        return -float(cfg['invalid_penalty'])
    # Accuracy dominates; shrinking an incorrect crop never earns a bonus.
    utility = float(bool(correct))
    if decision is not None and decision.evidence_label == 'sufficient':
        utility += cfg['evidence_bonus']*decision.quality
    return utility - float(bool(correct))*(cfg['area_cost']*total_area + cfg['observation_cost']*calls)


def p2_reward(decision, label, target_quality, target_confidence, correct, cfg):
    if label not in {'sufficient','insufficient'}:
        raise ValueError('Uncertain pseudo-labels cannot train the binary evidence cases.')
    if decision is None:
        return -float(cfg['invalid_penalty'])
    if label == 'insufficient':
        # A lucky correct answer on inadequate evidence is still penalized.
        reward = 1.0 if decision.action == 'request_more' else -1.0
    else:
        reward = 1.0 if decision.action == 'answer' and correct else -1.0
    reward += cfg['label_bonus']*(1 if decision.evidence_label == label else -1)
    reward -= cfg['quality_cost']*(decision.quality-target_quality)**2
    reward -= cfg['confidence_cost']*(decision.confidence-target_confidence)**2
    return reward
