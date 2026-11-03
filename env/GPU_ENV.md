# GPU 运行环境（D1 确定，之后不随意变更）

| 项 | 值 |
|---|---|
| 基础镜像 | `verlai/verl@sha256:9576682f85ca36f4ef719efccc5a5deb4d0b6f66f06fc14f43fdfed0749fbf5d`（tag `vllm012.latest`，本地 tag `ssrl/verl:vllm012`，约 41 GB） |
| 镜像内另装 | `verl==0.7.0`（`--no-deps`）、`faiss-cpu`、`FlagEmbedding`、`modelscope`，以镜像原有 torch / vllm / transformers / flash-attn / ray / tensordict / numpy 版本为 constraints |
| 版本（容器内实测，2026-11-03） | verl 0.7.0、vLLM 0.12.0、torch 2.9.0+cu129、transformers 4.57.3、ray 2.52.1、flash-attn 2.8.1；完整列表：VM 上 `~/work/pip-list-gpu.txt`（随 D1 run 同步到 GCS） |
| 上游 SHA | Agent-R1 `b124aa46534cbf2fb8bc8af11405774984c42ac7`（见 third_party/README.md） |
| 宿主机 | GCP Deep Learning VM `common-cu129-ubuntu-2204-nvidia-580`，NVIDIA 驱动 580.178.04；Docker CE + nvidia-container-toolkit（`scripts/gcp/setup_vm.sh`） |
| 初始化 | `scripts/gcp/bootstrap_d1.sh`（可重复执行，按步骤打标记） |
| 机器 | D1：us-central1-a 按需 g2-standard-16（1×L4 24GB）`ssrl-l4`；正式训练：us-central1 Spot A100 80GB `ssrl-a100`（-c，已建，11-03 两次被抢占）/ Colab（B0/B1、评测、副项目） |
| 数据 | ModelScope `Melmaphother/Agent-R1-data`：`hotpotqa/`（train / validation / 2wiki / musique parquet）、`hotpotqa_corpus/{hpqa_corpus.jsonl 271MB, index.bin 2.09GB}` |
| 模型 | D1 有效 run 实际用 **Qwen/Qwen3-0.6B**（运行时由 HF 自动下载，snapshot 未记录，D2 起在 meta 中记录）；Qwen/Qwen3-1.7B snapshot `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` 已下载但在 L4 上 OOM |

规则：镜像或依赖变更必须记录在本文件，并在实验卡中注明。
