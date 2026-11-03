# @title 在 Colab 内核里后台启动一条命令（不阻塞 colab exec，避免 CLI 等待超时）
# 用法：先改下面的 CMD / LOG，再 colab exec -s <session> -f scripts/colab/launch_bg.py
import subprocess
import time

CMD = (
    "cd /content/stateful-search-rl && HOTPOTQA_EMBEDDING_DEVICE=cpu python -I scripts/baselines_b0b1.py "
    "--split data/processed/hotpotqa_splits/dev_select.parquet --corpus-dir /content/agent-r1-data/hotpotqa_corpus "
    "--out runs/d2_b0b1_dev_q3-4b-2507_20261103"
)
LOG = "/content/b0b1_dev.log"

p = subprocess.Popen(f"nohup bash -c '{CMD}' > {LOG} 2>&1 &", shell=True)
time.sleep(5)
print(subprocess.run(f"ls -la {LOG}; pgrep -af baselines_b0b1 | head -2", shell=True, capture_output=True, text=True).stdout)
