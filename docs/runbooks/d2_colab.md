# D2 Colab 运行手册：B0/B1 基线 + logprob 一致性

单卡规则：Colab 承担 B0/B1 与评测；A100 上的轮询任务互不影响。两项都只做推理，Colab A100 或 G4 均可（4B bf16 约 8GB）。

## 1. 环境（Colab 单元格）
```bash
!git clone https://github.com/QFAN04/stateful-search-rl.git && cd stateful-search-rl && git submodule update --init third_party/Agent-R1
!pip install -q vllm faiss-cpu FlagEmbedding modelscope pandas pyarrow
```

## 2. 数据
```bash
# 固定划分（make_splits.py 产物；data/processed 不进 git）
!mkdir -p data/processed/hotpotqa_splits
# 将本地 make_splits.py 生成的 dev_select.parquet 上传到上述目录。
# 检索语料与索引（约 2.3GB）
!modelscope download --dataset Melmaphother/Agent-R1-data --include "hotpotqa_corpus/hpqa_corpus.jsonl" "hotpotqa_corpus/index.bin" --local_dir /content/agent-r1-data
```
核对 `dev_select.parquet` 的题目 ID 与 `data/manifests/hotpotqa_dev_select.json` 一致（ids_sha256）。

## 3. 运行
```bash
# 先用 --limit 20 冒烟，确认输出格式与 EM 计算正常，再全量（500 题）
!python -I scripts/baselines_b0b1.py --split data/processed/hotpotqa_splits/dev_select.parquet \
    --corpus-dir /content/agent-r1-data/hotpotqa_corpus --out runs/d2_b0b1_dev_q3-4b-2507_<日期>
!PYTHONPATH=src python -m ssrl.checks.logprob_consistency --split data/processed/hotpotqa_splits/dev_select.parquet \
    --corpus-dir /content/agent-r1-data/hotpotqa_corpus --out runs/d2_logprob_q3-4b-2507_<日期> --n 64
```
跑完：下载 `runs/<run_id>/` 后再销毁 Colab 会话。

## 4. 注意
- B0/B1 只在 dev_select 上跑；封存 test 到 D9 与 E0–E3 一起只跑一次。
- B1 的检索与 recipe 首次自动检索完全相同（同一检索器类、top-5、问题原文作 query）。
- 一致性检查只覆盖"推理引擎 vs 训练前向"的数值差异（Agent-R1 直接使用 vLLM 返回的 token ids，不存在重新分词问题，见接口笔记 §1.3）。**2026-11-03 起改为在 A100 训练镜像里跑**：Colab 自动装的是 vLLM 0.33 / torch 2.14 / transformers 5.20，与训练镜像（vLLM 0.12 / torch 2.9 / transformers 4.57）不同，测出的差值不代表训练栈。
- Colab CLI（Windows）注意：`colab exec -f` 需设 `PYTHONUTF8=1`（否则按 GBK 读含中文的脚本）；Git Bash 下远端路径需 `MSYS_NO_PATHCONV=1`；长任务用 `scripts/colab/launch_bg.py` 在内核里后台启动，避免 CLI 等待超时。
- vLLM 与检索器同进程时必须用 spawn 起引擎进程（脚本已设 `VLLM_WORKER_MULTIPROC_METHOD=spawn`），否则报 CUDA initialization error。
