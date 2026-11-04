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
PROMPT_PROTOCOL_PIXEL='pixel_bbox_v2'
PROMPT_PROTOCOL_SINGLE='single_crop_normalized_v1'
PROMPT_PROTOCOL_PIXEL_SINGLE='single_crop_pixel_v1'
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
P1_SYSTEM_PIXEL = '''You acquire visual evidence, not the final answer. Inspect the original image and select ONE rectangular region containing the evidence needed for the question. Include every necessary object/reference for spatial questions. Do not solve or answer the question in your response.
OUTPUT CONTRACT:
- Return exactly one <crop>...</crop> block containing valid JSON with exactly one key: "bbox".
- "bbox" is an array of FOUR NUMBERS [left, top, right, bottom] in PIXELS of the ORIGINAL image, measured from its upper-left corner. The next message gives the original width and height. Use 0 <= left < right <= width and 0 <= top < bottom <= height.
- Do not use bbox_2d, XML attributes such as <crop bbox="...">, a self-closing tag, markdown, comments, captions, extra JSON keys, or text inside the numeric array.
- Prefer the crop block directly. If you need a private visual observation first, put it entirely inside ONE <think>...</think> block and keep it brief. No prose is allowed outside these blocks. End immediately after </crop>.
VALID FORMAT EXAMPLE for an image 640 pixels wide and 428 pixels high:
<crop>{"bbox":[160,100,480,320]}</crop>
VALID FORMAT EXAMPLE for a different region of the same example image:
<think>The relevant visual details are near the upper left.</think><crop>{"bbox":[10,20,250,220]}</crop>
The entire example image, ONLY when needed, is <crop>{"bbox":[0,0,640,428]}</crop>.
Use the ACTUAL dimensions from the next message, not the example dimensions. Your thoughts and coordinates are NOT sent to the verifier. Only the cropped pixels are sent. Choose your own evidence region; do not blindly repeat an example.'''
P1_OUTPUT_REMINDER_PIXEL='Return exactly one <crop>{"bbox":[left,top,right,bottom]}</crop> JSON block with four pixel coordinates within the stated original width and height. Never put bbox in a crop tag attribute. End after </crop>.'
P1_SINGLE_CROP_RULE = '''SINGLE-CROP TASK: This is your only crop. Choose ONE rectangle that contains ALL visual evidence needed to answer the question. For spatial questions include every compared object; for charts include the relevant marks, axes, labels, and legend. If evidence is spread out, use a larger rectangle or the entire original image. The verifier will see only these pixels. You cannot make another crop or revise this choice. Do not answer the question.'''
P2_SYSTEM = '''You independently verify visual evidence. You receive the question and cropped evidence only, not the original image, crop coordinates, or the acquirer's thoughts. Judge VISIBLE evidence, not whether you can guess an answer. Related but blurry, illegible, incomplete, or missing-reference evidence is insufficient.
OUTPUT CONTRACT: Prefer ONE <decision>...</decision> block directly. If you need private reasoning first, enclose ALL of it in ONE <think>...</think> block. Your response must start with <decision> or <think>, never ordinary prose, a heading, or markdown. Do not put explanations outside those blocks. End immediately after </decision>.
Inside <decision>, output exactly ONE valid JSON object with precisely these fields: action, evidence_label, quality, confidence, answer. Use double-quoted JSON strings and numeric quality/confidence, not comments or extra fields.
A sufficient-evidence FORMAT example is <decision>{"action":"answer","evidence_label":"sufficient","quality":0.9,"confidence":0.9,"answer":"your actual answer"}</decision>.
An insufficient-evidence FORMAT example is <decision>{"action":"request_more","evidence_label":"insufficient","quality":0.1,"confidence":0.9,"answer":""}</decision>.
An uncertain-evidence FORMAT example is <decision>{"action":"request_more","evidence_label":"uncertain","quality":0.4,"confidence":0.4,"answer":""}</decision>.
When unsure, use action=request_more, evidence_label=uncertain, and answer="". Estimate quality and confidence for the actual crop; do NOT copy example numbers. Both are numbers in [0,1]. quality rates visible task-relevant information; confidence rates your evidence judgment. Answer only when sufficient, otherwise request more. Do not call a tool yourself.'''
P2_OUTPUT_REMINDER='Return exactly one <decision>JSON</decision> block. Any optional reasoning must be inside <think>...</think>. No other prose or markdown; end after </decision>.'
P2_SYSTEM_SINGLE = '''You verify ONE final crop. You receive the question and cropped pixels only, not the original image, crop coordinates, acquirer's thoughts, or reference answer. No additional crop can be requested. Judge whether the visible pixels contain ALL evidence needed for the question. Related but blurry, illegible, incomplete, or missing-reference evidence is insufficient.
OUTPUT CONTRACT: Return exactly ONE <decision>...</decision> block with a valid JSON object containing precisely action, evidence_label, quality, confidence, answer. Optional private reasoning may appear only in ONE <think>...</think> block before the decision. No ordinary prose, heading, markdown, or text after </decision>.
If evidence is sufficient, use action="answer", evidence_label="sufficient", and a nonempty answer based only on the crop. If evidence is insufficient or uncertain, use action="abstain", evidence_label="insufficient" or "uncertain", and answer="". Abstain is FINAL; never ask for more evidence. quality and confidence must each be a number in [0,1].
Sufficient FORMAT example: <decision>{"action":"answer","evidence_label":"sufficient","quality":0.9,"confidence":0.9,"answer":"your actual answer"}</decision>.
Insufficient FORMAT example: <decision>{"action":"abstain","evidence_label":"insufficient","quality":0.1,"confidence":0.9,"answer":""}</decision>.
Uncertain FORMAT example: <decision>{"action":"abstain","evidence_label":"uncertain","quality":0.4,"confidence":0.4,"answer":""}</decision>.
quality measures how much useful visual evidence is present. confidence measures how certain you are that your evidence_label and action are correct; an obviously missing clue can have LOW quality and HIGH confidence. Do not lower confidence merely because you abstain. Estimate both for the actual crop; do not copy example numbers. End immediately after </decision>.'''
P2_OUTPUT_REMINDER_SINGLE='Return one final <decision>JSON</decision>: answer from sufficient evidence, or abstain with an empty answer. No request for another crop.'

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


def parse_crop_for_image(text, original_image, mode='normalized'):
    if mode == 'normalized':
        return parse_crop(text)
    if mode != 'pixel':
        raise ValueError('Unsupported crop coordinate mode.')
    try:
        value = tagged_json(text, 'crop')
    except ValueError as error:
        # Qwen2.5-VL sometimes emits its native crop tool shape. Accept only a
        # single bbox attribute with no extra fields or visible prose.
        match = re.fullmatch(
            r'\s*(?:<think>.*?</think>\s*)?<crop bbox="(\[[^\]\r\n]+\])">\s*'
            r'(?:<think>.*?</think>\s*)?</crop>\s*'
            r'(?:<\|im_end\|>|<\|endoftext\|>)?\s*', text, re.S)
        if match is None:
            raise error
        value = {'bbox': json.loads(match.group(1))}
    if set(value) != {'bbox'}:
        raise ValueError('Crop payload permits coordinates only.')
    box = value['bbox']
    if not isinstance(box, list) or len(box) != 4 or any(type(x) not in (int,float) or not math.isfinite(x) or x < 0 for x in box):
        raise ValueError('Invalid pixel bbox.')
    # The actor can still follow the normalized examples from its base model.
    if all(x <= 1 for x in box):
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError('Empty bbox.')
        return tuple(box)
    with Image.open(original_image) as image:
        width,height = image.size
    limits=(width,height,width,height)
    if any(x > limit*1.05 for x,limit in zip(box,limits)):
        raise ValueError('Pixel bbox exceeds original image bounds by more than 5%.')
    x1,y1,x2,y2=(min(x,limit) for x,limit in zip(box,limits))
    if x2 <= x1 or y2 <= y1:
        raise ValueError('Empty pixel bbox.')
    return (x1/width,y1/height,x2/width,y2/height)


def parse_decision(text, single_crop=False):
    value = tagged_json(text, 'decision')
    if set(value) != {'action','evidence_label','quality','confidence','answer'}:
        raise ValueError('Unexpected verifier fields.')
    allowed_actions = {'answer','abstain'} if single_crop else {'answer','request_more'}
    if value['action'] not in allowed_actions or value['evidence_label'] not in LABELS:
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
        raise ValueError('An insufficient-evidence decision must not contain an answer.')
    return Decision(**value)


def common_text(question, options):
    # References and teacher labels have no path into this formatter.
    text = 'Question: ' + question
    if options:
        text += '\nOptions:\n' + '\n'.join(str(k)+'. '+str(v) for k,v in options.items())
    return text


def acquisition_messages(question, original_image, options, own_history, feedback, crop_coordinate_mode='normalized', single_crop=False):
    if not isinstance(feedback, Feedback):
        raise TypeError('P1 requires binary Feedback, never the verifier response.')
    if crop_coordinate_mode not in {'normalized','pixel'}:
        raise ValueError('Unsupported crop coordinate mode.')
    system=P1_SYSTEM_PIXEL if crop_coordinate_mode=='pixel' else P1_SYSTEM
    reminder=P1_OUTPUT_REMINDER_PIXEL if crop_coordinate_mode=='pixel' else P1_OUTPUT_REMINDER
    if single_crop:
        if own_history:raise ValueError('Single-crop acquisition cannot have a prior crop.')
        system += '\n' + P1_SINGLE_CROP_RULE
        reminder += ' This is the only crop; include all needed evidence in its one rectangle.'
    messages = [{'role':'system','content':system}, {'role':'user','content':[
        {'type':'image','image':str(original_image)},
        {'type':'text','text':common_text(question,options)}]}]
    if crop_coordinate_mode=='pixel':
        with Image.open(original_image) as image:
            width,height=image.size
        messages[-1]['content'].append({'type':'text','text':f'Original image width={width} pixels; height={height} pixels.'})
    # History contains only this policy's own previous generations.
    for response in own_history:
        messages += [{'role':'assistant','content':response},
                     {'role':'user','content':'Additional visual evidence is requested. Select another useful region.'}]
    if feedback.request_more and not own_history:
        messages[-1]['content'].append({'type':'text','text':'Acquire visual evidence now.'})
    if own_history:messages[-1]['content']+=' '+reminder
    else:messages[-1]['content'].append({'type':'text','text':reminder})
    return messages


def verification_messages(question, options, evidence, single_crop=False):
    if not isinstance(evidence, Evidence) or not evidence.images:
        raise TypeError('P2 requires cropped pixel evidence only.')
    if single_crop and len(evidence.images) != 1:
        raise ValueError('Single-crop verification requires exactly one cropped image.')
    # Fresh verification context each round; no prior guesses from either role.
    return [{'role':'system','content':P2_SYSTEM_SINGLE if single_crop else P2_SYSTEM}, {'role':'user','content':[
        {'type':'text','text':common_text(question,options)},
        *[{'type':'image','image':str(path)} for path in evidence.images],
        {'type':'text','text':P2_OUTPUT_REMINDER_SINGLE if single_crop else P2_OUTPUT_REMINDER}]}]


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


def p1_reward(correct, decision, total_area, calls, cfg, invalid=False, all_correct=False, all_wrong=False):
    if invalid:
        return -float(cfg['invalid_penalty'])
    utility = float(bool(correct))
    if correct and decision is not None and decision.evidence_label == 'sufficient':
        utility += cfg['evidence_bonus']*decision.quality
    reward = utility - float(bool(correct))*cfg['observation_cost']*calls
    if all_correct and correct:
        reward -= cfg['area_cost']*total_area
    elif all_wrong and not correct:
        reward += cfg.get('all_wrong_area_bonus',0.0)*total_area
    return reward


def p2_reward(decision, label, target_quality, target_confidence, correct, cfg, single_crop=False):
    if label not in {'sufficient','insufficient'}:
        raise ValueError('Uncertain pseudo-labels cannot train the binary evidence cases.')
    if decision is None:
        return -float(cfg['invalid_penalty'])
    if label == 'insufficient':
        # A lucky correct answer on inadequate evidence is still penalized.
        reward = 1.0 if decision.action == ('abstain' if single_crop else 'request_more') else -1.0
    else:
        reward = 1.0 if decision.action == 'answer' and correct else -1.0
    reward += cfg['label_bonus']*(1 if decision.evidence_label == label else -1)
    reward -= cfg['quality_cost']*(decision.quality-target_quality)**2
    reward -= cfg['confidence_cost']*(decision.confidence-target_confidence)**2
    return reward
