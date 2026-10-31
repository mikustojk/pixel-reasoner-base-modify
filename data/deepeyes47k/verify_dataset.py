"""Independent final-path, lineage, split and saved-image verification (CPU)."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE / 'prepared_v2'
sys.path.insert(0, str(HERE.parents[1]))
from eav_selfplay.io import digest, load_questions, rows


def main():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    train = load_questions(OUT / 'train.jsonl')
    held = load_questions(OUT / 'heldout.jsonl')
    audit = list(rows(OUT / 'row_audit.jsonl'))
    assert len(train) == 44673 and len(held) == 2340
    counts = Counter(r['status'] for r in audit)
    assert counts == Counter(train=44673, heldout=2340, excluded_duplicate=39)
    assert len(audit) == sum(v['rows'] for v in manifest['sources'].values()) == 47052
    tq, hq = {q.qid for q in train}, {q.qid for q in held}
    ti, hi = {q.image for q in train}, {q.image for q in held}
    assert not tq & hq and not ti & hi
    assert len(ti) == 38704 and len(hi) == 2030
    eval_index = json.loads((OUT / 'evaluation_image_index.json').read_text())
    assert not {Path(p).stem for p in ti | hi} & set(eval_index)
    canonical = {r['qid']: r for r in audit if r['status'] in ('train', 'heldout')}
    for split, questions in [('train', train), ('heldout', held)]:
        for q in questions:
            source = canonical[q.qid]
            assert source['status'] == split
            assert q.question == source['question'] and q.reference == source['reference']
            assert q.options == source['options']
            assert Path(q.image).parent == HERE / 'images'
            assert Path(q.image).stem == source['image_sha256']
            expected_held = int(source['image_sha256'][:16], 16) / 2**64 < .05
            assert expected_held == (split == 'heldout')
    rng = random.Random(20261030)
    image_checks = []
    for origin in sorted(manifest['source_counts']):
        pool = [q for q in train + held if canonical[q.qid]['data_source'] == origin]
        for q in rng.sample(pool, 16):
            source = canonical[q.qid]
            with Image.open(q.image) as image:
                assert image.format == 'PNG' and image.mode == 'RGB'
                image.load()
                assert image.size == (source['image_width'], source['image_height'])
                actual = hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()
                assert actual == source['image_sha256']
            image_checks.append({'qid': q.qid, 'source': origin, 'image': q.image,
                                 'rgb_sha256': actual})
    for name, receipt in manifest['outputs'].items():
        assert digest(OUT / name) == receipt['sha256']
    assert all(manifest['source_hashes_unchanged'].values())
    report = {'passed': True, 'train_rows': len(train), 'heldout_rows': len(held),
              'raw_rows_reconciled': len(audit), 'duplicate_rows': counts['excluded_duplicate'],
              'train_heldout_shared_images': len(ti & hi),
              'strict_loader_all_rows': True, 'all_row_lineage_checked': True,
              'all_output_hashes_checked': True, 'common_image_paths_checked': True,
              'sample_seed': 20261030, 'saved_png_pixel_checks': image_checks,
              'scope': '48 saved PNGs sampled (16 per source); original sources decoded in full during preparation. No resized/near-duplicate clearance claimed.',
              'verification_script_sha256': digest(Path(__file__))}
    (OUT / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'saved_png_pixel_checks'},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
