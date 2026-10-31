"""Finalize option extraction, reusing the first pass's exact decoded images.

Run after prepare_dataset.py completes. prepared_v1 is an audit/cache pass;
prepared_v2 is the final training input. No image files or raw Parquets are changed.
"""
import json
from pathlib import Path
import re
import prepare_dataset as prep


def parse_options(question, declared):
    if declared:
        return original_options(question, declared)
    # Prefer actual line-level choices. Do not interpret matrix A. / feature F.
    # inside an option as another choice boundary. The first A may adjoin '?'.
    for pattern in prep.OPTION_PATTERNS:
        all_marks = list(pattern.finditer(question))
        first_a = next((m for m in all_marks if m[1] == 'A'), None)
        marks = ([first_a] if first_a else []) + [
            m for m in all_marks if m[1] != 'A'
            and not question[:m.start()].rsplit('\n', 1)[-1].strip()
        ]
        if len(marks) < 2 or [m[1] for m in marks] != list('ABCDEF'[:len(marks)]):
            continue
        if any(a.start() >= b.start() for a, b in zip(marks, marks[1:])):
            continue
        values = {m[1]: question[m.end():marks[i+1].start() if i+1 < len(marks)
                               else len(question)].strip()
                  for i, m in enumerate(marks)}
        if all(values.values()):
            return values, 'line_aware_question_sequence'
    return original_options(question, declared)


original_options = prep.parse_options


def main():
    first = prep.HERE / 'prepared_v1'
    first_manifest = json.loads((first / 'manifest.json').read_text())
    assert first_manifest['validation']['reconciled_raw_rows']
    cached = {}
    for line in (first / 'row_audit.jsonl').open():
        r = json.loads(line)
        if 'image_sha256' in r:
            cached[r['encoded_sha256']] = {
                'identity': r['image_sha256'],
                'path': str(first / 'images' / (r['image_sha256'] + '.png')),
                'width': r['image_width'], 'height': r['image_height'],
            }
    original_extract = prep.extract
    original_main = prep.main
    prep.OUT = prep.HERE / 'prepared_v2'
    prep.parse_options = parse_options

    def extract(args):
        encoded, _ = args
        if encoded in cached:
            return encoded, cached[encoded]
        return original_extract(args)

    def evaluation_index():
        index = json.loads((first / 'evaluation_image_index.json').read_text())
        prep.dump(prep.OUT / 'evaluation_image_index.json', index)
        prep.log('Reused verified evaluation RGB index and lossless image cache from pass 1')
        return index, first_manifest['evaluation_sources']

    prep.extract = extract
    prep.build_eval_index = evaluation_index
    original_main()
    # Put authoritative image assets outside the versioned audit passes.
    common_images = prep.HERE / 'images'
    (first / 'images').rename(common_images)
    (first / 'images').symlink_to(common_images, target_is_directory=True)
    for image in (prep.OUT / 'images').glob('*.png'):
        if not (common_images / image.name).exists():
            image.rename(common_images / image.name)
    for name in ['train.jsonl', 'heldout.jsonl']:
        path = prep.OUT / name
        rows = [json.loads(line) for line in path.open()]
        for row in rows:
            row['image'] = str(common_images / Path(row['image']).name)
            assert Path(row['image']).is_file()
        prep.jsonl(path, rows)
    report_path = prep.OUT / 'manifest.json'
    report = json.loads(report_path.read_text())
    report['finalizer_script_sha256'] = prep.sha_file(Path(__file__))
    report['cache_pass_manifest'] = {
        'path': str(first / 'manifest.json'),
        'sha256': prep.sha_file(first / 'manifest.json'),
    }
    report['final_input_version'] = 'prepared_v2'
    report['image_cache_dependency'] = str(common_images)
    for name, receipt in report['outputs'].items():
        receipt['sha256'] = prep.sha_file(prep.OUT / name)
    report['option_policy'] = 'Line-aware sequential A/B/... choices; option-internal variable names are not boundaries. All source question text is preserved.'
    prep.dump(report_path, report)
    description = prep.HERE / '数据准备说明.md'
    text = description.read_text().replace('prepared_v1/train.jsonl', 'prepared_v2/train.jsonl').replace(
        'prepared_v1/heldout.jsonl', 'prepared_v2/heldout.jsonl')
    text += ('\n最终训练输入使用 prepared_v2，图片统一位于同级 images/。prepared_v1 仅保留首轮审计；'
             '其 images/ 是指向统一图片目录的兼容链接。第二轮修正了选项正文中'
             '“矩阵 A.”、“特征 F.”被误当作选项边界的问题，重新进行了全量去重、冲突、划分检查。\n')
    description.write_text(text)


if __name__ == '__main__':
    main()
