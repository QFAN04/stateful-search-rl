# @title D2 Colab setup：仓库、依赖、检索语料
# 在 Colab 会话中执行（colab exec -s <session> -f scripts/colab/setup_d2.py）。可重复执行。
# 前提：/content/repo.bundle 与 /content/dev_select.parquet 已用 colab upload 传上来。
import os
import subprocess


def sh(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


REPO = "/content/stateful-search-rl"
if not os.path.isdir(f"{REPO}/.git"):
    sh(f"git clone -q -b main /content/repo.bundle {REPO}")
else:
    sh(f"git -C {REPO} fetch -q /content/repo.bundle main && git -C {REPO} reset -q --hard FETCH_HEAD")
sh(f"git -C {REPO} submodule update --init -q third_party/Agent-R1")
sh(f"git -C {REPO} log --oneline -1 && git -C {REPO}/third_party/Agent-R1 log --oneline -1")

os.makedirs(f"{REPO}/data/processed/hotpotqa_splits", exist_ok=True)
sh(f"cp /content/dev_select.parquet {REPO}/data/processed/hotpotqa_splits/dev_select.parquet")
sh(f"sha256sum {REPO}/data/processed/hotpotqa_splits/dev_select.parquet")

sh("pip install -q vllm faiss-cpu FlagEmbedding modelscope 2>&1 | tail -3")
sh("python -c \"import vllm, torch, transformers; print('versions', vllm.__version__, torch.__version__, transformers.__version__)\"")

if not os.path.exists("/content/agent-r1-data/hotpotqa_corpus/index.bin"):
    sh("modelscope download --dataset Melmaphother/Agent-R1-data --include 'hotpotqa_corpus/hpqa_corpus.jsonl' "
       "'hotpotqa_corpus/index.bin' --local_dir /content/agent-r1-data > /content/modelscope.log 2>&1")
sh("ls -la /content/agent-r1-data/hotpotqa_corpus")
print("SETUP_DONE")
