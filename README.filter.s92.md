# s92 训练集 27B 图像质量审计

输入是 `data/deepeyes47k/prepared_v2/train.jsonl` 的 44,673 条训练样本；验证集不参与。`scripts/run_filter_s92.sh` 在 8 张 H200 上各启动一个本机 `/mnt/source/models/Qwen/Qwen3.8-27B` vLLM 服务，再由 `scripts/filter_train_s92.py` 逐条检查原图、问题、选项和参考答案。它不修改原训练集。

每条经过两次独立判定，最终只有以下三类达到置信度阈值时才自动归档：

| 结果 | 含义 | 判定门槛 |
| --- | --- | --- |
| `correct` | 图像足以回答，GT 与图像一致 | 两次同意且各自置信度 ≥ 0.85 |
| `gt_error` | 图像足以回答，但 GT 错误 | 两次同意、修正答案一致且各自置信度 ≥ 0.90 |
| `unanswerable` | 问题含糊、缺少必要视觉信息或无法可靠读出 | 两次同意且各自置信度 ≥ 0.90 |

任何低置信度、格式错误或两次判定不一致的样本进入 `needs_review`，避免直接删掉有争议的训练数据。`visual_answer` 只是模型提出的候选，不会自动写回 GT。尤其图表题可能出现图例误读，应人工复核 `gt_error`。两次相同的模型判定也不等于人工真值。

```bash
cd /mnt/source/chengqili/pixel-reasoner-base-modify
bash scripts/run_filter_s92.sh
```

默认输出在 `data/deepeyes47k/quality_27b_v1/`：

- `audit.jsonl`：可恢复的逐条原始响应、理由、置信度和最终分类。
- `correct.jsonl`、`gt_error.jsonl`、`unanswerable.jsonl`、`needs_review.jsonl`：原记录及审计信息。
- `train_verified.jsonl`：仅含 `correct` 原始记录，可供训练脚本直接读取。
- `summary.json`：完成标记、分类数量及输出哈希。只有全部 44,673 条被核对后才生成。

中断后使用同一命令和输出目录续跑；已有结果不会重复请求。运行日志在 `outputs/data_filter_27b/<UTC时间>-<PID>/`。若 GPU 数量不同，可设置 `FILTER_GPUS=0,1,...`；每卡并发可由 `FILTER_CONCURRENCY_PER_SERVER` 和 `FILTER_MAX_NUM_SEQS` 调整。正式训练前应先查看 `summary.json` 和各类抽样，再使用 `configs/selfplay_s92_filtered.json`：

```bash
bash scripts/run_selfplay_s92.sh --phase train --train-config configs/selfplay_s92_filtered.json
```

筛选后的 s92 训练配置启用单次裁剪：每个候选只选一个包含全部证据的矩形，P2 一次性回答或放弃，不再请求补充证据；随机探针裁剪也关闭。本过滤命令不会自动启动训练。


## 已提交到本分支的正确样本记录

`data/deepeyes47k/quality_27b_v1/` 中以下文件已纳入 Git：

| 文件 | 内容 |
| --- | --- |
| `train_verified.jsonl` | 25,073 条训练记录：`qid`、`question`、`image`、`reference`、`options`。 |
| `correct.jsonl` | 同一批记录，额外包含 `quality_audit`：原始索引、逐条哈希、标签、理由、首次判定和确认判定等。 |
| `correct_qids.txt` | 每行一个正确样本的 `qid`，顺序与训练清单一致。 |
| `summary.json` | 完成标记、分类数量、训练清单 SHA-256。 |
| `manifest.json` | 输入哈希、裁判模型、提示词版本、随机种子和置信度门槛。 |
| `records_manifest.json` | 上述文件的 SHA-256、文件大小和记录数量。 |

本次审计处理 44,673 条输入，分为正确 25,073 条、GT 有问题 2,140 条、不可答 2,247 条、待复核 15,213 条。`train_verified.jsonl` 和 `correct.jsonl` 按 `qid` 一一对应。训练清单 SHA-256 为 `2f5184573f19ae0584627e3a84078dd346d42d78895e50820ae002eb5c702f75`。

图片以及其他分类的逐条记录保存在 s92，未纳入本次提交。JSONL 的 `image` 字段指向 s92 上的绝对路径；克隆分支后在其他机器训练，需要先准备对应图片并映射路径。

在仓库根目录校验文件完整性：

```bash
python3 - <<'PYVERIFY'
import hashlib, json
from pathlib import Path
base = Path('data/deepeyes47k/quality_27b_v1')
manifest = json.loads((base / 'records_manifest.json').read_text())
for name, info in manifest['files'].items():
    path = base / name
    assert path.stat().st_size == info['bytes'], name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == info['sha256'], name
print('OK:', manifest['records'], 'correct samples')
PYVERIFY
```
