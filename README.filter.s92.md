# s92 训练集 27B 图像质量审计

本文件同时保留首轮审计和后续盲测复核记录。最新可用数据见下方“2026-11-05：不提供 GT 的复核结果”；首轮 25,073 条通过记录继续保留用于溯源。

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


## 2026-11-05：不提供 GT 的复核结果

首轮审计会把参考答案与图片一起交给模型，可能产生答案提示偏差。因此，这次仅对首轮通过的 25,073 条训练题再次复核，不包含 heldout 或评测数据。正式复核已完成，完成时间为北京时间 2026-11-05 17:24:42。

作答使用 s92 的 `/mnt/source/models/Qwen/Qwen3.8-27B`，8 张卡各一个 TP=1 副本（DP=8）。作答请求通过字段白名单只发送原图、题目和选项，允许选择 `insufficient_evidence`，不发送 GT、旧审计结论或上一轮作答结果。流程为：

1. 先关闭 thinking，最多输出 1,024 token。
2. 若未答对或认为证据不足，发起全新的 thinking 请求，`reasoning_effort=medium`，最多输出 12,288 token。
3. 任意一次答对即保留，首轮通过则跳过 thinking。两次都正常完成但未通过则记为 `rejected`；解析、截断或比较异常且没有任何成功结果则记为 `needs_review`。

GT 只进入独立、无图片、无 thinking 的文本答案比较请求，不进入作答上下文，也不反馈给重试请求；相同答案可直接归一化匹配，其余按语义比较。作答和比较都使用 JSON 输出约束。两种作答模式温度均为 0。协议版本为 `blind_nothink_then_think_v2`，输入和提示词指纹见相应 `manifest.json`。

| 复核结果 | 条数 |
| --- | ---: |
| 输入：首轮通过集 | 25,073 |
| 最终通过 `correct` | 21,809 |
| 两次均未通过 `rejected` | 3,255 |
| 异常待复核 `needs_review` | 9 |
| 通过来源：no-think 首轮 | 12,519 |
| 通过来源：thinking 重试 | 9,290 |
| 实际触发 thinking 重试 | 12,554 |

`rejected` 不代表已证明 GT 错误或题目不可答，也可能是模型未能正确作答。通过结果仍是模型验证结果，不能替代人工真值。GT 未被自动修改。

### 本分支新增的完整通过集记录

目录为 [`data/deepeyes47k/quality_27b_blind_v2/`](data/deepeyes47k/quality_27b_blind_v2/)。

| 文件 | 内容 |
| --- | --- |
| `train_verified.jsonl` | 21,809 条原始五字段训练记录，顺序沿用输入。 |
| `correct.jsonl` | 同一批记录及 `quality_audit`，包含两种模式的回答、thinking 内容、独立评分和通过来源。 |
| `correct_qids.txt` | 按训练清单顺序排列的 21,809 个 QID。 |
| `audit_index.jsonl` | 全部 25,073 条的原始索引、QID、输入行哈希、去留标签和通过模式；覆盖排除及待复核记录。 |
| `manifest.json`、`summary.json` | s92 产出的原始协议清单与完成统计。 |
| `records_manifest.json` | 本分支所收录文件的 SHA-256、字节数及记录数。 |

新训练清单 SHA-256：`94b945ffef92bcf59bac1c58667aa667596b0ebcd6a2e8dbd7e16288424ce4c2`。
其输入 SHA-256：`2f5184573f19ae0584627e3a84078dd346d42d78895e50820ae002eb5c702f75`，与本分支首轮通过训练清单一致。

完整原始审计和排除记录仍保存在 s92 的 `/mnt/source/chengqili/pixel-reasoner-base-modify-next/data/deepeyes47k/quality_27b_blind_v2/`。本分支使用紧凑的全量索引记录排除与待复核结果，不重复收录这些记录的完整响应。

### 细粒度感知子集

从上述 21,809 条通过记录中，用 QID 关联原始 `prepared_v2/row_audit.jsonl` 中 `status=train` 的来源记录，选择 `data_source=vstar`；不通过问题关键词猜测任务类别。这里的 `vstar` 是原始训练来源标签，不是加入 V* 评测集。

| 通过集的原始来源 | 条数 |
| --- | ---: |
| `vstar`：细粒度感知 | 11,221 |
| `thinklite_eureka` | 5,643 |
| `chart` | 4,945 |
| 合计 | 21,809 |

细粒度子集覆盖 10,143 张图，其中 no-think 通过 6,932 条、thinking 重试通过 4,289 条。已核对它与原始训练记录、盲测通过记录的一致性，并检查与 heldout 图像及已知评测图像的解码 RGB 精确身份重叠为零；该检查不保证不存在近重复图。

子集目录为 [`data/deepeyes47k/quality_27b_blind_v2_finegrained_v1/`](data/deepeyes47k/quality_27b_blind_v2_finegrained_v1/)，收录 `train_verified.jsonl`、`correct_qids.txt`、逐题来源及通过模式的 `selection.jsonl`、原始 `manifest.json`、`summary.json` 和文件指纹 `records_manifest.json`。

子集训练清单 SHA-256：`ef2bdd6b28cac88e647007072f6c566a4654b5c1bf88ccc66fe989207bd52160`。s92 上对应目录为 `/mnt/source/chengqili/pixel-reasoner-base-modify-next/data/deepeyes47k/quality_27b_blind_v2_finegrained_v1/`。

图片未纳入 Git；所有训练清单的 `image` 仍指向 s92 上的原图绝对路径。此记录更新保留首轮文件和现有训练配置；使用新完整集或细粒度子集时，应显式将训练配置的 `train_data` 指向对应新目录中的 `train_verified.jsonl`。

在仓库根目录校验两份新记录的文件完整性：

```bash
python3 - <<'PYVERIFY'
import hashlib, json
from pathlib import Path
for name in ['quality_27b_blind_v2', 'quality_27b_blind_v2_finegrained_v1']:
    base = Path('data/deepeyes47k') / name
    manifest = json.loads((base / 'records_manifest.json').read_text())
    for filename, info in manifest['files'].items():
        payload = (base / filename).read_bytes()
        assert len(payload) == info['bytes'], filename
        assert hashlib.sha256(payload).hexdigest() == info['sha256'], filename
        if 'records' in info:
            assert len(payload.splitlines()) == info['records'], filename
    print('OK:', name, manifest['records'])
PYVERIFY
```
