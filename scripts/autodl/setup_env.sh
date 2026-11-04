#!/usr/bin/env bash
# AutoDL 实例上搭训练环境（在实例内运行，可重复执行；每步完成后写标记）。
# 目标：复刻训练镜像 verlai/verl vllm012（env/GPU_ENV.md）的版本组合，不用 Docker。
#   conda 环境（默认 base，要求 Python 3.12；装在系统盘，保存镜像时会带上）
#   torch 2.9.0+cu129 / vLLM 0.12.0 / verl 0.7.0 / flash-attn 2.8.1，其余包按 env/autodl/constraints.txt 锁版本
#   数据和模型放数据盘 /root/autodl-tmp（不进镜像）
# 用法（仓库已上传到 /root/autodl-tmp/stateful-search-rl；用有卡模式开机，无卡模式 CPU 太少）：
#   nohup bash scripts/autodl/setup_env.sh > /root/autodl-tmp/setup_env.log 2>&1 &
set -euxo pipefail
REPO=/root/autodl-tmp/stateful-search-rl
DATA=/root/autodl-tmp/data
MARK=/root/.ssrl_setup_marks
ENV_NAME="${ENV_NAME:-base}"   # 镜像自带 Python 3.12 时直接装进 base，省一个环境的磁盘
mkdir -p "$MARK" "$DATA"
done_step() { touch "$MARK/$1"; }
is_done() { [[ -f "$MARK/$1" ]]; }
# 数据盘上的步骤把标记放在数据盘：系统盘会被存进镜像，新实例的数据盘是空的，不能沿用系统盘上的标记
DMARK="$DATA/.ssrl_setup_marks"; mkdir -p "$DMARK"
data_done_step() { touch "$DMARK/$1"; }
data_is_done() { [[ -f "$DMARK/$1" ]]; }

source /root/miniconda3/etc/profile.d/conda.sh
# 学术加速：GitHub / PyTorch 官方源 / HF 需要；ModelScope 和 pip 国内镜像不需要
turbo_on() { [[ -f /etc/network_turbo ]] && source /etc/network_turbo || true; }
turbo_off() { unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY || true; }

# 1. conda 环境
if ! is_done conda; then
  [[ "$ENV_NAME" == base ]] || conda env list | grep -q "^$ENV_NAME " || conda create -y -n "$ENV_NAME" python=3.12
  done_step conda
fi
conda activate "$ENV_NAME"
python -V
# pip 源：腾讯云 HTTPS 镜像（11-04 实测大文件 9.4MB/s；阿里云 HTTP 0.7MB/s）
# 注意：无卡模式只有 0.5 核 CPU，pip 会被 CPU 卡在约 1MB/s，装环境请用有卡模式开机
pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple

# 2. torch（与训练镜像相同的 cu129 构建）
if ! is_done torch; then
  # 国内镜像有同一个 cu129 构建（走官方源 + 学术加速时 1 小时没下完）；nvidia-* 依赖从默认的阿里云 PyPI 镜像拿
  # 大文件先用 curl 断点续传下到数据盘（pip 下载 1.2GB 的 torch 时连接会卡死不动），再从本地装
  W=/root/autodl-tmp/wheels; mkdir -p "$W"
  for f in torch-2.9.0%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl            torchvision-0.24.0%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl            torchaudio-2.9.0%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl; do
    out="$W/${f//%2B/+}"
    # 已完整下载（zip 目录可读）就跳过；对完整文件续传会得到 HTTP 416
    if [[ -s "$out" ]] && python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).close()" "$out" 2>/dev/null; then continue; fi
    for i in 1 2 3 4 5 6 7 8; do
      curl -fL --retry 5 --speed-limit 102400 --speed-time 30 -C - -o "$out"         "https://mirrors.aliyun.com/pytorch-wheels/cu129/$f" && break
      echo "curl retry $i for $f"; sleep 5
    done
  done
  pip install --timeout 60 --retries 10 "$W"/torch-2.9.0+cu129-*.whl "$W"/torchvision-0.24.0+cu129-*.whl "$W"/torchaudio-2.9.0+cu129-*.whl
  done_step torch
fi

# 3. vLLM + verl + recipe 依赖，全部按训练镜像的 pip list 锁版本
if ! is_done pip; then
  pip install --timeout 60 --retries 10 -c "$REPO/env/autodl/constraints.txt" vllm==0.12.0 transformers==4.57.3 "ray[default]==2.52.1" \
    tensordict==0.10.0 peft==0.18.0 accelerate==1.12.0 datasets==4.4.2 hydra-core==1.3.2 codetiming torchdata \
    pylatexenc liger_kernel wandb faiss-cpu FlagEmbedding modelscope pandas pyarrow pytest
  pip install --no-deps verl==0.7.0
  done_step pip
fi

# 4. flash-attn 2.8.1 预编译 wheel（cu12 / torch2.9 / cxx11abiTRUE / cp312）
if ! is_done flash; then
  turbo_on
  pip install --no-deps "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
  turbo_off
  done_step flash
fi

python - <<'EOF'
import torch, vllm, verl, transformers, ray, flash_attn
print("versions", "torch", torch.__version__, "vllm", vllm.__version__, "verl", verl.__version__,
      "transformers", transformers.__version__, "ray", ray.__version__, "flash_attn", flash_attn.__version__)
print("cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
EOF
pip list > "$DATA/pip-list-autodl.txt"

# 5. 数据：HotpotQA parquet + 检索语料与索引（ModelScope，境内直连）
if ! data_is_done data; then
  modelscope download --dataset Melmaphother/Agent-R1-data \
    --include "hotpotqa/*" "hotpotqa_corpus/hpqa_corpus.jsonl" "hotpotqa_corpus/index.bin" \
    --local_dir "$DATA/agent-r1-data"
  ls -la "$DATA/agent-r1-data/hotpotqa" "$DATA/agent-r1-data/hotpotqa_corpus"
  data_done_step data
fi
mkdir -p "$REPO/third_party/Agent-R1/data/corpus"
ln -sfn "$DATA/agent-r1-data/hotpotqa" "$REPO/third_party/Agent-R1/data/corpus/hotpotqa"

# 6. 模型（ModelScope 下载到数据盘；训练时用本地路径）
if ! data_is_done model06; then
  modelscope download --model Qwen/Qwen3-0.6B --local_dir "$DATA/models/Qwen3-0.6B"
  data_done_step model06
fi
# 检索编码器（recipe 默认 BAAI/bge-large-en-v1.5；运行时用 HOTPOTQA_EMBEDDING_MODEL 指向本地目录）
if ! data_is_done bge; then
  modelscope download --model BAAI/bge-large-en-v1.5 --local_dir "$DATA/models/bge-large-en-v1.5"
  data_done_step bge
fi

# pip 缓存（几 GB）在系统盘，会被存进镜像，装完就清掉
pip cache purge || true
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
df -h / /root/autodl-tmp
echo SETUP_DONE
