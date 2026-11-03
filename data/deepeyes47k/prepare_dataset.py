"""Prepare local DeepEyes 47k inputs; preserve sources and quarantine defects.

Run from any directory with `python -u /absolute/path/to/prepare_dataset.py`.
Only this dataset directory is written. Existing completed outputs are refused.
"""
import base64
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import unicodedata

from PIL import Image
import pyarrow.parquet as pq

ROOT = Path('/inspire/hdd/global_user/lichengqi-253108120152')
HERE = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get('EAV_DATA_SOURCE', ROOT / 'DeepEyes/data/benchmarks/DeepEyes-Datasets-47k'))
VSTAR = Path(os.environ.get('EAV_VSTAR_ROOT', ROOT / 'DeepEyes/data/benchmarks/vstar_bench'))
HRBENCH = Path(os.environ.get('EAV_HRBENCH_ROOT', ROOT / 'DeepEyes/data/benchmarks/hrbench'))
OUT = HERE / 'prepared_v1'
OPTION_PATTERNS = (
    re.compile(r'(?<![A-Za-z0-9])([A-F])[.．、]\s*'),
    re.compile(r'[（(]([A-F])[）)]\s*'),
)


def log(message):
    print(f'[{datetime.now(timezone.utc).isoformat(timespec="seconds")}] {message}', flush=True)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def jsonl(path, values):
    with path.open('w') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def norm(value):
    return ' '.join(unicodedata.normalize('NFKC', value).split())


def pixel_hash(image):
    return hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()


def parse_options(question, declared):
    if declared:
        if not isinstance(declared, dict) or not all(
            re.fullmatch('[A-F]', str(k)) and isinstance(v, str) and v.strip()
            for k, v in declared.items()
        ):
            raise ValueError('Malformed explicit options dictionary')
        return {str(k): v.strip() for k, v in declared.items()}, 'declared'
    # A complete A/B/... sequence distinguishes options from variables and units.
    for pattern in OPTION_PATTERNS:
        marks = list(pattern.finditer(question))
        if len(marks) < 2:
            continue
        labels = [m[1] for m in marks]
        if labels != list('ABCDEF'[:len(labels)]):
            continue
        values = {m[1]: question[m.end():marks[i+1].start() if i+1 < len(marks)
                               else len(question)].strip()
                  for i, m in enumerate(marks)}
        if all(values.values()):
            return values, 'question_sequence'
    return {}, 'not_detected'


def build_eval_index():
    """Exact RGB identity; re-encoded/resized/near-duplicate images are not certified."""
    by_pixel = defaultdict(list)
    encoded_cache = {}
    files = {}
    for split in ['direct_attributes', 'relative_position']:
        paths = sorted((VSTAR / split).glob('*.jpg'))
        for path in paths:
            with Image.open(path) as src:
                rgb = src.convert('RGB')
                identity = pixel_hash(rgb)
            by_pixel[identity].append({'benchmark': 'vstar', 'split': split, 'image': str(path)})
        files[f'vstar/{split}'] = {'images': len(paths)}
        log(f'Evaluation index: V* {split}, {len(paths)} images')
    csv.field_size_limit(100_000_000)
    for split in ['4k', '8k']:
        path = HRBENCH / f'hr_bench_{split}.tsv'
        count = 0
        with path.open() as stream:
            for row in csv.DictReader(stream, delimiter='\t'):
                payload = row['image']
                compressed = hashlib.sha256(payload.encode()).hexdigest()
                if compressed not in encoded_cache:
                    raw = base64.b64decode(payload)
                    with Image.open(io.BytesIO(raw)) as src:
                        encoded_cache[compressed] = pixel_hash(src.convert('RGB'))
                identity = encoded_cache[compressed]
                by_pixel[identity].append({'benchmark': 'hrbench', 'split': split, 'index': row['index']})
                count += 1
                if count % 200 == 0:
                    log(f'Evaluation index: HRBench {split} {count}/800')
        files[f'hrbench/{split}'] = {'rows': count, 'path': str(path), 'sha256': sha_file(path)}
    assert files['vstar/direct_attributes']['images'] == 115
    assert files['vstar/relative_position']['images'] == 76
    assert files['hrbench/4k']['rows'] == files['hrbench/8k']['rows'] == 800
    dump(OUT / 'evaluation_image_index.json', dict(by_pixel))
    return by_pixel, files


def extract(args):
    """Decode one unique source payload in a bounded worker pool."""
    encoded, raw = args
    with Image.open(io.BytesIO(raw)) as src:
        image = src.convert('RGB')
        identity = pixel_hash(image)
        path = OUT / 'images' / (identity + '.png')
        # Distinct encodings may decode to the same image; PNG writes are atomic.
        if not path.exists():
            temporary = path.with_name(identity + '.' + encoded + '.tmp')
            image.save(temporary, format='PNG', compress_level=1)
            temporary.replace(path)
        return encoded, {'identity': identity, 'path': str(path),
                         'width': image.width, 'height': image.height}


def main():
    if OUT.exists():
        if os.environ.get('EAV_RESUME_PREP') != '1' or not (OUT / 'images').is_dir():
            raise ValueError(f'Output already exists; preserve it and select a new version: {OUT}')
        log(f'Reusing atomically written images in {OUT}; rebuilding all manifests and rows')
    else:
        OUT.mkdir(parents=True)
        (OUT / 'images').mkdir()
    sources = {}
    for path in sorted(SOURCE.glob('*.parquet')):
        sources[path.name] = {'path': str(path), 'bytes': path.stat().st_size,
                              'sha256': sha_file(path),
                              'rows': pq.ParquetFile(path).metadata.num_rows}
    assert len(sources) == 3 and sum(v['rows'] for v in sources.values()) == 47052
    log('Source fingerprints recorded; building evaluation image index')
    evaluation, evaluation_sources = build_eval_index()
    audit = []
    candidates = defaultdict(list)
    image_cache = {}
    observed_sources = Counter()
    workers = ThreadPoolExecutor(max_workers=6)
    try:
        for name, source in sources.items():
            path = Path(source['path'])
            parquet = pq.ParquetFile(path)
            columns = [c for c in parquet.schema_arrow.names if c not in ('rationale', 'prompt')]
            offset = 0
            for batch in parquet.iter_batches(batch_size=24, columns=columns):
                rows = batch.to_pylist()
                pending = {}
                row_payloads = []
                for row in rows:
                    try:
                        images = row.get('images')
                        if not isinstance(images, list) or len(images) != 1:
                            raise ValueError('Exactly one original image required')
                        image = images[0]
                        raw = image.get('bytes')
                        if raw is None:
                            image_path = Path(image['path'])
                            raw = (image_path if image_path.is_absolute()
                                   else path.parent / image_path).read_bytes()
                        if not isinstance(raw, bytes) or not raw:
                            raise ValueError('Empty/nonbinary image payload')
                        encoded = hashlib.sha256(raw).hexdigest()
                        row_payloads.append((encoded, None))
                        if encoded not in image_cache:
                            pending[encoded] = raw
                    except Exception as error:
                        row_payloads.append((None, str(error)))
                futures = {encoded: workers.submit(extract, (encoded, raw))
                           for encoded, raw in pending.items()}
                for encoded, future in futures.items():
                    try:
                        _, image_cache[encoded] = future.result()
                    except Exception as error:
                        image_cache[encoded] = {'error': str(error)}
                for local, (row, (encoded, payload_error)) in enumerate(zip(rows, row_payloads), 1):
                    info = row.get('extra_info') or {}
                    reward = row.get('reward_model') or {}
                    record = {'source_file': name, 'source_row_1based': offset + local,
                              'source_index': info.get('index'), 'data_source': row.get('data_source'),
                              'ability': row.get('ability'), 'source_split': info.get('split')}
                    observed_sources[record['data_source']] += 1
                    audit.append(record)
                    try:
                        if payload_error:
                            raise ValueError(payload_error)
                        image = image_cache[encoded]
                        if 'error' in image:
                            raise ValueError(image['error'])
                        question = info.get('question') or row.get('question')
                        reference = reward.get('ground_truth') or info.get('answer') or row.get('answer')
                        if not isinstance(question, str) or not question.strip():
                            raise ValueError('Missing question')
                        if not isinstance(reference, str) or not reference.strip():
                            raise ValueError('Missing answer')
                        question, reference = question.strip(), reference.strip()
                        options, option_method = parse_options(question, row.get('options'))
                        raw_reference = reference
                        letter = re.fullmatch(r'\s*[（(]?([A-F])[）).．]?\s*', reference)
                        if letter:
                            if letter[1] not in options:
                                raise ValueError('Letter-only reference without unambiguously parsed options')
                            reference = options[letter[1]]
                        identity = image['identity']
                        record.update(image_sha256=identity, encoded_sha256=encoded,
                                      image_width=image['width'], image_height=image['height'],
                                      question=question, original_reference=raw_reference,
                                      reference=reference, options=options, option_method=option_method)
                        key = '\n'.join([identity, norm(question),
                                         json.dumps(options, sort_keys=True, ensure_ascii=False)])
                        qid = hashlib.sha256(key.encode()).hexdigest()
                        record['qid'] = qid
                        record['item'] = {'qid': qid, 'question': question, 'image': image['path'],
                                          'reference': reference, 'options': options}
                        if identity in evaluation:
                            record.update(status='excluded_evaluation_overlap',
                                          evaluation_matches=evaluation[identity])
                        else:
                            record['status'] = 'candidate'
                            candidates[qid].append(record)
                    except Exception as error:
                        record.update(status='quarantined_invalid', reason=str(error))
                offset += len(rows)
                if offset % 1200 == 0 or offset == source['rows']:
                    log(f'{name}: {offset}/{source["rows"]}; unique decoded encodings={len(image_cache)}')
    finally:
        workers.shutdown(wait=True)
    train, held = [], []
    for qid, records in candidates.items():
        references = {norm(r['reference']) for r in records}
        if len(references) != 1:
            for record in records:
                record.update(status='quarantined_conflicting_answers',
                              conflicting_references=sorted(references))
            continue
        first = records[0]
        heldout = int(first['image_sha256'][:16], 16) / 2**64 < .05
        split = 'heldout' if heldout else 'train'
        first['status'] = split
        (held if heldout else train).append(first['item'])
        for record in records[1:]:
            record.update(status='excluded_duplicate', kept_qid=qid,
                          kept_source_file=first['source_file'],
                          kept_source_row_1based=first['source_row_1based'])
    jsonl(OUT / 'train.jsonl', train)
    jsonl(OUT / 'heldout.jsonl', held)
    # Training rows obey the existing strict five-field loader. Lineage is separate.
    for record in audit:
        record.pop('item', None)
    jsonl(OUT / 'row_audit.jsonl', audit)
    jsonl(OUT / 'quarantine.jsonl', [r for r in audit if r['status'].startswith('quarantined')])
    jsonl(OUT / 'evaluation_overlaps.jsonl',
          [r for r in audit if r['status'] == 'excluded_evaluation_overlap'])
    train_images = {Path(r['image']).stem for r in train}
    held_images = {Path(r['image']).stem for r in held}
    statuses = Counter(r['status'] for r in audit)
    assert len(audit) == 47052 and sum(statuses.values()) == 47052
    assert not train_images & held_images
    assert not (train_images | held_images) & set(evaluation)
    assert len({r['qid'] for r in train + held}) == len(train) + len(held)
    sys.path.insert(0, str(HERE.parents[1]))
    from eav_selfplay.io import load_questions
    assert len(load_questions(OUT / 'train.jsonl')) == len(train)
    assert len(load_questions(OUT / 'heldout.jsonl')) == len(held)
    source_unchanged = {name: sha_file(Path(meta['path'])) == meta['sha256']
                        for name, meta in sources.items()}
    assert all(source_unchanged.values())
    by_source = defaultdict(Counter)
    for record in audit:
        by_source[record['data_source']][record['status']] += 1
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'raw_rows': len(audit), 'source_counts': dict(observed_sources),
        'status_counts': dict(statuses), 'by_source': {k: dict(v) for k, v in by_source.items()},
        'train_rows': len(train), 'heldout_rows': len(held),
        'train_images': len(train_images), 'heldout_images': len(held_images),
        'decoded_unique_encodings': len(image_cache),
        'evaluation_unique_pixel_images': len(evaluation),
        'quarantine_reasons': dict(Counter(r.get('reason', r['status']) for r in audit
                                            if r['status'].startswith('quarantined'))),
        'sources': sources, 'evaluation_sources': evaluation_sources,
        'source_hashes_unchanged': source_unchanged,
        'split_policy': 'RGB SHA256 first 64 bits / 2**64 < 0.05 -> heldout; entire images grouped.',
        'reference_policy': 'Preserve answer text; unambiguous option-letter references mapped to option text. Original references retained in row_audit.',
        'overlap_policy': 'Exclude exact decoded RGB matches to V* 191 and HRBench 4K/8K 1600 records. No semantic, resized or near-duplicate clearance claimed.',
        'duplicate_policy': 'Keep first identical image/question/options with matching whitespace-normalized answer; quarantine all members with conflicting answers.',
        'image_policy': 'Original decoded RGB pixels, lossless PNG; no resize, crop or upsampling.',
        'validation': {'strict_training_loader': True, 'image_disjoint_splits': True,
                       'exact_evaluation_overlap_remaining': 0, 'unique_qids': True,
                       'reconciled_raw_rows': True},
        'outputs': {name: {'path': str(OUT / name), 'sha256': sha_file(OUT / name)}
                    for name in ['train.jsonl', 'heldout.jsonl', 'row_audit.jsonl',
                                 'quarantine.jsonl', 'evaluation_overlaps.jsonl']},
        'preparation_script_sha256': sha_file(Path(__file__)),
    }
    dump(OUT / 'manifest.json', report)
    lines = ['# DeepEyes 47k 数据准备记录', '',
             f'原始记录：{len(audit):,}。原始 Parquet 未修改，未下载数据。', '',
             '| 来源 | 原始 | 训练 | 验证 | 重复 | 冲突 | 无效 | 评测重叠 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for source, counts in by_source.items():
        names = ['train', 'heldout', 'excluded_duplicate', 'quarantined_conflicting_answers',
                 'quarantined_invalid', 'excluded_evaluation_overlap']
        lines.append(f'| {source} | {sum(counts.values())} | ' +
                     ' | '.join(str(counts[n]) for n in names) + ' |')
    lines += ['', f'训练 {len(train):,} 条，涉及 {len(train_images):,} 张原图；'
              f'验证 {len(held):,} 条，涉及 {len(held_images):,} 张原图。', '',
              '按原图 RGB 像素哈希固定划分，目标约 95%/5%，同图不跨训练、验证。',
              '图片保存为无损 PNG，不缩放、不裁剪。训练行仅含 qid、question、image、reference、options；'
              '不继承原系统提示、思维链或工具轨迹。',
              '选项字母能够明确映射时转换为答案内容；原始答案和来源行号保留在 row_audit.jsonl。',
              '所有无效、答案冲突样本均隔离并记录，不将冲突中的任一答案擅自视为正确。', '',
              '## 评测重叠和验证', '',
              '已检查 V* 191 张评测图和 HRBench 4K/8K 共 1600 条评测记录的解码 RGB 精确重叠，'
              '匹配的训练来源记录全部排除。该检查不证明没有缩放、重新压缩或语义近似图像重叠。',
              '训练与验证无同图交叉、无重复 qid；均通过当前训练代码的严格数据加载器。'
              '源 Parquet 的处理前后 SHA-256 一致。完整检查结果见 manifest.json。', '',
              '## 使用', '', '训练输入：`prepared_v1/train.jsonl`。独立验证：`prepared_v1/heldout.jsonl`。',
              '这里只完成数据整理；训练配置尚未填写，裁判质量和真实 GPU 训练尚未验证。', '']
    (HERE / '数据准备说明.md').write_text('\n'.join(lines))
    log(json.dumps({k: report[k] for k in ['raw_rows', 'status_counts', 'train_rows',
                                         'heldout_rows', 'train_images', 'heldout_images']},
                   ensure_ascii=False))


if __name__ == '__main__':
    main()
