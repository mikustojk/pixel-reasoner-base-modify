# s92（8×H200）适配

训练集 27B 质量审计的类别、阈值和续跑方式见 [README.filter.s92.md](README.filter.s92.md)。

本机训练模型为 `/mnt/source/models/Qwen/Qwen2.5-VL-7B-Instruct`，固定多模态裁判为 `/mnt/source/models/Qwen/Qwen3.8-27B`。GPU 0 运行裁判，GPU 1～7 训练共享的 7B 策略。27B 的配置架构为 `qwen3_5`，带原生视觉编码器。它与旧集群的 Qwen3.6-27B 权重不同；证据标签的质量仍需通过 smoke 运行和人工审计确认。

## 环境

现有 `/mnt/source/chengqili/env/verl` 含 PyTorch 2.6.0+cu124。`scripts/setup_s92_env.sh` 建立独立训练环境 `/mnt/source/chengqili/env/eav-selfplay-s92`，复用现有 CUDA/PyTorch 包，并仅在新环境安装 DeepSpeed 0.16.9 与 Transformers 4.57.1。裁判使用机器已有的 `/mnt/source/chengqili/miniconda3/envs/swift`（vLLM 0.27.1、Transformers 5.15.1），不与训练环境混装。安装命令使用机器已经配置的 pip 镜像。

本机只有 CUDA 13.0 编译工具链，和训练 PyTorch 的 CUDA 12.4 不匹配。s92 配置将 `optimizer_offload` 设为 `false`，在 DeepSpeed ZeRO-2 下使用 PyTorch AdamW，避免 JIT 编译 DeepSpeed 的 CPU Adam 扩展。H200 单卡约 140 GiB，实际显存是否足够仍由 smoke 验证。旧集群配置保持原来的 CPU 优化器 offload 设置。

本机 7B 基座在 smoke 的审计输出中持续返回原图像素坐标，例如 640×428 图片上的 `[178,131,386,309]`，即使旧提示要求 0～1 坐标。s92 配置因此使用 `crop_coordinate_mode=pixel`：P1 提示明确给出原图尺寸，执行器把像素框换算为原图归一化坐标后裁剪；只容许最多 5% 的边界超出并裁到图片边缘，明显越界仍报错。该模式还接受单一 `bbox` 的 `<crop bbox="[...]">` 形式，拒绝额外属性或普通文字；原始响应留在私有审计中。旧配置默认严格 0～1 JSON 模式。推理 s92 权重时也需传入 `infer_selfplay.py --crop-coordinate-mode pixel`。

```bash
cd /mnt/source/chengqili/pixel-reasoner-base-modify
bash scripts/setup_s92_env.sh
```

## 数据

原始 DeepEyes 47k Parquet 位于 `/mnt/source/chengqili/Efficient-tool-call/data/DeepEyes-Datasets-47k`；V* 和 HRBench 的本机副本位于相邻目录。准备脚本仍按原流程生成 `prepared_v1` 审计缓存和最终 `prepared_v2`，并检查训练图片与评测图片的解码 RGB 精确重叠。输出图片较大，约需数十 GiB。

```bash
cd /mnt/source/chengqili/pixel-reasoner-base-modify
export EAV_DATA_SOURCE=/mnt/source/chengqili/Efficient-tool-call/data/DeepEyes-Datasets-47k
export EAV_VSTAR_ROOT=/mnt/source/chengqili/Efficient-tool-call/data/vstar_bench
export EAV_HRBENCH_ROOT=/mnt/source/chengqili/Efficient-tool-call/data/HR-Bench
/mnt/source/chengqili/env/verl/bin/python -u data/deepeyes47k/prepare_dataset.py
/mnt/source/chengqili/env/verl/bin/python -u data/deepeyes47k/finalize_dataset.py
/mnt/source/chengqili/env/verl/bin/python -u data/deepeyes47k/verify_dataset.py
```

准备脚本不会覆盖已有 `prepared_v1`。若中断且只留下部分原子写入的 PNG，可设置 `EAV_RESUME_PREP=1` 重建索引和清单，同时复用已有 PNG；要先确认没有另一个准备进程在运行。最终训练输入为 `data/deepeyes47k/prepared_v2/train.jsonl`，验证集独立保存在 `heldout.jsonl`。

## 预检与训练

本机配置分别是 `configs/selfplay_s92.json`（原始训练集全量）、`configs/selfplay_s92_filtered.json`（27B 筛选后的 25,073 条全量）和 `configs/selfplay_s92_smoke.json`（筛选集、每卡每阶段 2 个问题组）。入口默认只运行两轮 smoke。预检验证数据、模型、训练依赖、CUDA 和裁判环境；日志与输出留在 `outputs/cluster/<UTC时间>-<PID>/`。

当前 s92 配置启用 `single_crop_only=true`：`max_rounds=1`、`probe_crops_per_group=0`。每个 P1 候选只生成一次裁剪，提示词要求一个矩形包含回答所需的全部对象、图表标注和其他证据；证据分散时允许框住整图。P2 对这张裁剪只做一次最终判断：证据充分则回答，否则 `abstain`，不再要求更多证据。P2 的 `confidence` 表示对这一判断的把握，不等同于证据充分程度。证据入池门槛维持 0.8；P2 阶段仍要求充分和不足两类都存在，否则拒绝训练。推理入口的默认裁剪预算也为 1。旧运行目录里的多轮轨迹不能续接到新协议。

```bash
cd /mnt/source/chengqili/pixel-reasoner-base-modify
bash scripts/run_selfplay_s92.sh --phase check --train-config configs/selfplay_s92_filtered.json
bash scripts/run_selfplay_s92.sh --phase smoke
# 审计 smoke 的裁剪格式、证据标签和真实梯度更新后：
bash scripts/run_selfplay_s92.sh --phase train --train-config configs/selfplay_s92_filtered.json
```

`--phase all` 会先跑 smoke，成功后直接从原始 7B 基座启动全量训练。`--train-python` 可覆盖训练环境路径。入口尊重继承的 `CUDA_VISIBLE_DEVICES`；若调度器只分配部分 GPU，配置中的 0～7 必须在该分配内。

ZeRO-2 的优化器检查点照常保存；供下一阶段加载的模型权重保存为 4GB 分片的 `safetensors`。单个约 16GB 的 `pytorch_model.bin` 在本机多进程加载时会被反复读取，耗时异常。旧运行产生的 `.bin` 不建议直接用于 s92 的后续阶段。

## 本机验证

2026-11-03 的单次裁剪两轮 GPU smoke 位于 `outputs/cluster/20261103T031629Z-69867/smoke`，四个阶段的 7 个 rank 各完成 2 次更新，`complete.json` 记录了真实训练。两个 P1 阶段的证据池分别有 46 条和 42 条记录，每条恰好一张裁剪图；第 2 轮的 30 条充分、12 条不足标签都达到 0.8 的入池门槛。有效 P2 判断中的不足证据使用 `abstain`，没有 `request_more`。

2026-11-02 的两轮 GPU smoke 位于 `outputs/cluster/20261102T052025Z-19989/smoke`，`complete.json` 记录了 `real_training_performed=true`。这次 smoke 使用旧的多轮裁剪提示和配置，仅验证基础环境与模型链。四个训练阶段（每轮 P1、P2）都完成了每卡 2 次优化器更新，最终模型位于 `cycle_002/process2/hf`。两轮之间实际使用了训练后的共享 P2；裁判始终为本机 Qwen3.8-27B。P2 加载分片权重约一分钟内进入训练；此前单文件 `.bin` 在 7 个进程中反复读取，无法在合理时间内完成载入。

`python -m unittest discover -s tests -p 'test_*.py'` 通过 20 项测试，其中 1 项跳过。smoke 中仍有少量基座模型输出的无效裁剪框和 P2 决策，审计记录了具体原因；后续正式训练应关注这些比例和证据标签质量。

原 `README.md` 中 `/inspire/...` 路径、Qwen3.6 裁判和 H100 历史运行记录适用于旧集群，不描述本机运行结果。
