#!/usr/bin/env bash
# D1 VM 初始化（宿主机上运行，可重复执行；每步完成后写标记，Spot 抢占后重跑会跳过已完成步骤）。
# 前提：~/repo.bundle 已由本地 scp 上来（私有仓库，不在 VM 上放 GitHub 凭据）。
# 用法：nohup bash bootstrap_d1.sh > ~/bootstrap.log 2>&1 &
set -euxo pipefail
IMAGE_DIGEST="verlai/verl@sha256:9576682f85ca36f4ef719efccc5a5deb4d0b6f66f06fc14f43fdfed0749fbf5d"
IMAGE_TAG="ssrl/verl:vllm012"
AGENT_R1_SHA="b124aa46534cbf2fb8bc8af11405774984c42ac7"
REPO=~/work/stateful-search-rl
MARK=~/.bootstrap_marks
mkdir -p "$MARK" ~/work ~/data ~/hf
done_step() { touch "$MARK/$1"; }
is_done() { [[ -f "$MARK/$1" ]]; }

# 1. Docker + NVIDIA container toolkit
is_done docker2 || { bash ~/setup_vm.sh; done_step docker2; }
DOCKER="sudo docker"

# 2. 仓库：从 bundle 克隆；子模块（公开）从 GitHub 拉并固定 SHA
if ! is_done repo; then
  [[ -d "$REPO/.git" ]] || git clone -b main ~/repo.bundle "$REPO"
  git -C "$REPO" submodule update --init third_party/Agent-R1
  git -C "$REPO/third_party/Agent-R1" checkout "$AGENT_R1_SHA"
  done_step repo
fi

# 3. 镜像（按 digest 拉，记录 digest）
if ! is_done image; then
  $DOCKER pull "$IMAGE_DIGEST"
  $DOCKER tag "$IMAGE_DIGEST" "$IMAGE_TAG"
  $DOCKER image inspect "$IMAGE_TAG" --format '{{json .RepoDigests}} {{.Size}}' > ~/image_digest.txt
  done_step image
fi

# 4. 容器（常驻）
if ! $DOCKER ps -a --format '{{.Names}}' | grep -qx ssrl; then
  $DOCKER run -d --name ssrl --runtime=nvidia --gpus all --net=host --shm-size=32g \
    -v ~/work:/work -v ~/data:/data -v ~/hf:/root/.cache/huggingface \
    "$IMAGE_TAG" sleep infinity
fi
$DOCKER start ssrl >/dev/null

# 5. 容器内依赖：verl 0.7.0 + recipe 依赖，用 constraints 锁住镜像原有版本
if ! is_done pip; then
  $DOCKER exec ssrl bash -c '
    set -ex
    pip freeze | grep -iE "^(torch|vllm|transformers|flash.attn|ray|tensordict|numpy)==" > /tmp/pins.txt
    pip install --no-deps verl==0.7.0
    pip install -c /tmp/pins.txt "faiss-cpu>=1.7.4" "FlagEmbedding>=1.2.0" modelscope
    python -c "import verl, vllm, torch, transformers, ray, flash_attn; print(\"versions\", verl.__version__, vllm.__version__, torch.__version__, transformers.__version__, ray.__version__, flash_attn.__version__)"
    pip list > /work/pip-list-gpu.txt'
  done_step pip
fi

# 6. 数据：HotpotQA parquet + 检索语料与索引（ModelScope）
if ! is_done data; then
  $DOCKER exec ssrl bash -c '
    set -ex
    modelscope download --dataset Melmaphother/Agent-R1-data \
      --include "hotpotqa/*" "hotpotqa_corpus/hpqa_corpus.jsonl" "hotpotqa_corpus/index.bin" \
      --local_dir /data/agent-r1-data
    ls -la /data/agent-r1-data/hotpotqa /data/agent-r1-data/hotpotqa_corpus'
  done_step data
fi
mkdir -p "$REPO/third_party/Agent-R1/data/corpus"
ln -sfn /data/agent-r1-data/hotpotqa "$REPO/third_party/Agent-R1/data/corpus/hotpotqa"

# 7. 模型
if ! is_done model; then
  $DOCKER exec ssrl bash -c 'huggingface-cli download Qwen/Qwen3-1.7B && ls /root/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/'
  done_step model
fi

# 8. 固定划分（make_splits.py 产物，已上传 GCS）
mkdir -p ~/data/hotpotqa_splits
gcloud storage rsync gs://ssrl-runs-gqf-2026/data/hotpotqa_splits ~/data/hotpotqa_splits

nvidia-smi --query-gpu=name,driver_version --format=csv
echo BOOTSTRAP_DONE
