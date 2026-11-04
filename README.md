# Evidence Acquisition–Verification Self-Play

> s92（8×H200）的本机路径、环境和运行入口见 [README.s92.md](README.s92.md)。下文保留原八卡集群的历史配置与联调记录。

本分支以 Pixel-Reasoner 为基础，实现证据获取 P1／证据验证 P2 的交替 GRPO。P1/P2 是同一个 Qwen2.5-VL-7B-Instruct、同一套可训练权重，通过角色提示和独立上下文切换。原项目说明见 [README.upstream.md](README.upstream.md)。

## 已提交的 27B 筛选数据记录

本分支包含 27B 筛选后 **25,073 条正确样本**的 [训练清单](data/deepeyes47k/quality_27b_v1/train_verified.jsonl)、[逐条审计记录](data/deepeyes47k/quality_27b_v1/correct.jsonl) 和 [正确样本 qid 清单](data/deepeyes47k/quality_27b_v1/correct_qids.txt)。判定标准为两次独立判定一致且置信度均至少 0.85；“正确”是模型审计结果。

[筛选摘要](data/deepeyes47k/quality_27b_v1/summary.json) 记录全部 44,673 条输入的分类数量，[文件清单](data/deepeyes47k/quality_27b_v1/records_manifest.json) 提供 SHA-256 和大小。图片保存在 s92，JSONL 中的 `image` 为该机器的绝对路径；在其他机器使用时需要准备图片并映射路径。筛选流程、字段和校验命令见 [README.filter.s92.md](README.filter.s92.md)。

## 当前资源与模型

| 设备 | 用途 |
|---|---|
| GPU 0 | 一个常驻 Qwen3.6-27B vLLM 裁判服务 |
| GPU 1～7 | 七卡训练同一个共享 Qwen2.5-VL-7B-Instruct |

裁判路径为 `/inspire/hdd/global_user/lichengqi-253108120152/models/Qwen3.6-27B`。这是本地目录名称；配置的实际架构为 Qwen3.5，使用此前 GPU 评测所用的 vLLM 0.18+ 环境。训练模型仍是公共目录中的 Qwen2.5-VL-7B-Instruct，没有换成 27B。

运行协议为 `online_shared_p2_v1`。旧版 `bootstrap_judge` 配置和没有此协议标记的阶段检查点不能用于本协议续跑。

## 一轮怎样训练

```text
初始共享权重 θ0 → P1 更新 θ1 → P2 从 θ1 继续更新 θ2
下一轮：θ2 → P1 更新 θ3 → P2 继续更新 θ4
```

1. 首轮 P1 从原图选择裁剪区域，由外部 27B 裁判以 P2 提示判断证据是否充分、回答或请求继续观察。它判断证据时看不到标准答案。
2. 首轮之后，P1 的裁剪结果交给当前正在训练的同一个 7B 模型，以 P2 提示、greedy、无梯度方式判断；不加载上一轮冻结快照，不另启 7B 服务。一个问题组的 rollout 期间不更新权重；组完成后更新，下一组 P2 评判使用更新后的当前权重。
3. 所有轮次的最终答案正确性均由固定 27B 服务核对，允许与标准答案比较。这是单独的文本判分请求，不把标准答案传给 P1 或 P2 的观察输入。
4. P1 同题采样多个 episode，用最终正确性、证据质量和观察成本计算 GRPO 奖励；面积和调用成本只在答对时扣除。没有奖励差异的组不更新；整个阶段零更新会报错。
5. P1 阶段生成带来源、采样组和模型版本标记的证据标签池。P2 从本轮 P1 更新后的共享权重继续训练，使用该标签池中充分／不足且达到置信阈值的记录。充分证据应答对，不足证据应请求更多；不把“碰巧答对”当作证据充分。
6. 后续轮次的标签由当前 P2 自己生成，会随共享权重变化，是伪标签，不能保证没有噪声或自我强化。随机 crop 探针也由同一评判方判断，不预先假定不足；uncertain、低置信度样本不进入 P2 二分类奖励，默认要求池中有正负两类。

“首轮”指 `cycle=1` 整个 P1 阶段，不是只给第一道题或第一次裁剪调用外部裁判。27B 在后续仍驻留 GPU 0，用于答案判分。

## 角色隔离与多卡对齐

P1 接收原图、问题／选项、自己的历史和布尔继续信号；只将裁剪像素交给 P2，不传坐标、label、caption 或私有思维链。P2 每次只接收问题／选项和累计裁剪图，不继承 P1 文字、原图或上一轮 P2 猜想。标准答案和标签只进入奖励控制器。

裁剪使用原图归一化坐标，直接裁剪原始像素；没有额外 padding。图片模型的标准视觉预处理仍存在。面积按实际像素区域计算，多次与重复观察都累计计费。

后续 P1 阶段中，每个 rank 固定执行 `group_size × max_rounds` 次 P1 生成和同样次数的 P2 推理，另有固定数量的探针 P2 推理。提前结束或无效裁剪仍执行对齐调用；没有证据时只用灰色占位图，结果不进入标签池或奖励。P2 推理不计算训练 log-prob、不加入 P1 的 loss。

这保证各 rank 进入生成同步的次数一致；真实七卡 NCCL／DeepSpeed 仍需在 GPU 节点联调，CPU 测试不构成生产验证。

## 已准备的数据与训练规模

训练来源仅为官方 DeepEyes 47k，最终去重后：

- 训练：44,673 条，`data/deepeyes47k/prepared_v2/train.jsonl`。
- 验证：2,340 条，保持独立，没有加入训练。
- 方法、逐行来源、指纹和复查命令见 [数据准备说明](data/deepeyes47k/数据准备说明.md)。

[configs/selfplay.json](configs/selfplay.json) 是全量配置：`groups_per_phase=0` 表示不截断每个 rank 的数据。P1 每轮覆盖全部训练记录；七卡为对齐补一条重复记录，共执行 44,674 个问题组。P2 覆盖本轮证据池中符合条件的记录，为 rank 对齐也可能补少量重复记录。

默认每张卡每步一个问题组、每题采样 4 条轨迹。因此全局每步 7 个问题组、28 条轨迹；P1 最多 3 轮时 DeepSpeed 累积序列 batch 为 84（含对齐填充），P2 为 28，均为 7 的倍数。它们不是每次同时在显存中堆积 84 条序列。

[configs/selfplay_8xh100_smoke.json](configs/selfplay_8xh100_smoke.json) 每卡每阶段只跑 2 个问题组，但保留两轮，覆盖首轮外部裁判和后续当前 P2 两条路径。全量配置默认两轮；它是初始训练安排，不保证是最优论文配方。

## 训练环境与开训前检查

训练和裁判默认都使用现有 `vllm_env/bin/python`；其 GPU 环境目录已有 DeepSpeed 0.16.9、PyTorch 2.10.0、Transformers 4.57.1、vLLM 0.18.0 的安装记录。本次没有安装或替换其中的包。安装记录不是兼容性验证，必须通过实际 GPU 预检和小规模训练。

如果实际 GPU 检查发现不兼容，再用单独环境安装 [requirements-selfplay.txt](requirements-selfplay.txt)，通过 `--train-python /实际环境/bin/python` 或 `SELFPLAY_TRAIN_PYTHON` 指定；裁判 Python 可在配置的 `judge_python` 中单独指定。CUDA、PyTorch、DeepSpeed、CPU 优化器扩展和多卡通信必须实际可用。

```bash
python3 -B train_selfplay.py --config configs/selfplay.json --validate-only
```

上面是 CPU 数据／配置检查，不加载 7B 或裁判，不等于环境检查或训练成功。实际集群入口会先运行 `scripts/preflight_selfplay.py`：检查训练依赖、CUDA 分配、裁判 Python／vLLM、数据文件指纹，保存 GPU 和软件信息，失败会留下 traceback 并退出。

裁判并发默认 1、eager 模式，给 80GB 单卡留出推理空间；上下文与图片预算仍需真实节点验证。首轮裁判标签应先用独立人工样例检查；小规模训练成功不等于裁判标签正确。

## 集群入口：前台运行与日志

已有集群启动命令中的旧 benchmark 脚本，替换为本项目脚本即可。以下使用现有 GPU 环境，先检查再运行：

```bash
cd /inspire/hdd/global_user/lichengqi-253108120152/
unset CUDA_VISIBLE_DEVICES
bash guohyx/Pixel-Reasoner/scripts/run_selfplay_8xh100.sh \
  --phase all
```

父命令的 `unset` 是用户提供的八卡集群启动约定；项目脚本自身尊重继承的 GPU 分配，不主动扩展分配范围。

- `--phase all`：环境检查 → 小规模两轮训练 → 成功后全量两轮训练。
- `--phase smoke`：只检查环境并跑小规模两轮。
- `--phase train`：只检查环境并跑全量两轮。
- `--phase check`：只检查环境和资源。

如另有已准备好的训练环境，追加 `--train-python /实际环境/bin/python`。本脚本不自动安装依赖，也不使用 remote_runner、nohup、disown 或将整个训练任务放到后台。

主脚本全程前台等待。Python 启动器只后台启动一个受管理的 27B 子服务，等待就绪，再阻塞等待 torchrun 的七卡训练。所有周期完成或发生异常时清理裁判；最终把退出码交给集群。退出后实例自动关闭的情况下，日志和权重仍在共享磁盘。

每次运行先打印持久目录：`outputs/cluster/<UTC时间>-<PID>/`，相对项目根目录。目录中：

```text
logs/launcher.log         启动过程与最终退出码
logs/preflight.log        环境检查输出与失败 traceback
logs/environment.json    检查通过后的依赖／GPU 信息
logs/smoke.log            小规模训练 stdout + stderr
logs/train.log            全量训练 stdout + stderr
smoke/judge.log           小规模裁判服务 stdout + stderr
train/judge.log           全量裁判服务 stdout + stderr
smoke/                   小规模模型及审计数据
train/                   正式模型及审计数据
```

所有 Python 使用非缓冲输出并启用 faulthandler，训练命令使用 `> 日志 2>&1`。小规模失败立即退出，不继续全量训练；全量运行从原始共享基座重新开始，不继承小规模测试权重。GPU 无法启动、服务加载失败、训练异常都会保留非零退出状态；系统强制杀进程不一定产生 Python traceback。

## 输出、恢复和限制

### 首次 GPU 联调与格式修正

首次八卡运行目录为 `outputs/cluster/20261030T145345Z-20/`。八卡识别、DeepSpeed 初始化、7B 采样和 27B 裁判服务启动已执行，没有因为显存不足而退出；但 **真实 7B 反向传播尚未验证**。P1 的 14 个问题组、56 条轨迹全部裁剪无效，奖励均为 −1，无组内差异，触发零更新保护，全量阶段没有开始。

实际失败输出中，27 条字段不符合要求（常见为 `bbox_2d`），18 条裁剪块数量或标签格式不符合要求，9 条 JSON 损坏，2 条虽使用 `bbox` 但坐标不是合法的 0～1 范围。27B 的 14 次探针评判有 9 次因前缀格式不符合要求而未通过解析；旧日志未保存这 9 条响应的完整原文，不能从原因字符串还原其具体文字。

本次修正使用 `explicit_action_examples_v2` 提示版本：P1 加入明确数字示例、归一化定义、唯一字段名和结束标签要求；P2／裁判明确回复开头、JSON 和标签外文字规则；每次角色请求末尾再提醒格式。只改善提示，不使用约束解码、不自动纠正 JSON、不把像素坐标猜测性换算成归一化坐标，也不改奖励、采样温度、训练规模或零更新保护。

日志新增：

- 每个问题组的训练日志包含 `format_diagnostics`，统计有效／无效裁剪、P2 格式错误、裁判错误及原因。
- `private_traces/<group>.json` 中错误事件包含完整 `raw_response`、异常类型、原因和 traceback；裁判解析错误还保留 `response_kind` 与 `finish_reason`。信息只进审计文件，不反馈给 P1/P2。
- 原始输出与标签池在反向传播之前保存，便于第一次梯度更新失败时仍能追溯。
- 阶段目录的 `format_summary_rank0.json` 至 `format_summary_rank6.json` 汇总各 rank 格式统计和更新数，在阶段零更新退出前写入。

提示修正是否提高真实模型合规率，必须重新做 GPU 小规模验证，CPU 测试不能证明这一点。下一次集群启动先用：

```bash
cd /inspire/hdd/global_user/lichengqi-253108120152/
unset CUDA_VISIBLE_DEVICES
bash guohyx/Pixel-Reasoner/scripts/run_selfplay_8xh100.sh \
  --phase smoke
```

这次只跑两轮小规模测试，不自动进入全量。新运行会创建新目录；旧失败记录保留，代码指纹改变后不对该失败目录直接续跑。

### 阶段文件与恢复

每轮 `cycle_001/`、`cycle_002/` 中：

- `process1/hf/`、`process2/hf/` 是同一模型依次更新的阶段版本。
- `process1/evidence_bank.jsonl` 是本轮证据标签池，不包含 P1 思维链或 bbox。
- `private_traces/` 保存角色私有输出和评判错误，仅供审计。
- `phase_done.json` 记录协议、输入／输出检查点、数据哈希和真实更新。

根目录 `run.json` 绑定代码、配置和数据；`effective_config.json` 记录本次实际输出路径；`complete.json` 只给出一个最终 `policy_checkpoint`。两个阶段的权重连续传递，但每个新阶段使用新优化器。

`--resume` 仅跳过完整完成的阶段，核对参数链与协议，不支持中断阶段内部微步恢复。裁判服务的 resume 日志追加保存。需要继续同一完整阶段链时：

```bash
python -u train_selfplay.py \
  --config configs/selfplay.json \
  --output /之前持久目录/train \
  --resume
```

有未完成阶段的进度记录时会拒绝覆盖，应保留该运行并用新输出目录。当前为单机七训练卡实现，没有验证跨节点训练。ZeRO-2、CPU 优化器 offload、冻结视觉编码器和梯度检查点均已有实现，第一次七卡尝试尚未产生真实梯度更新。

## 双角色推理

只部署最终共享 `policy_checkpoint` 的一个服务，P1/P2 通过独立请求切换角色：

```bash
vllm serve /path/to/final/shared/checkpoint \
  --served-model-name eav_shared --port 8001

python infer_selfplay.py \
  --url http://127.0.0.1:8001/v1 --model eav_shared \
  --image /path/to/original.png --question 'Your question' \
  --output /path/to/new_inference
```

## CPU 检查

```bash
CUDA_VISIBLE_DEVICES='' \
EAV_TEST_PROCESSOR=/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-VL-7B-Instruct \
python3 -m unittest discover -s tests -v
```

包含共享检查点链、唯一外部裁判、协议／续跑校验、后续 P2 无梯度推理、提前结束／无效裁剪的调用对齐、上下文和 GT 隔离、七 rank 全量覆盖、集群入口失败退出与日志，以及小型真实多模态模型的反向传播和保存／重载。CPU 检查不能代替真实七卡联调。
