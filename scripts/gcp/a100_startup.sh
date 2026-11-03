#!/bin/bash
# ssrl-a100 的 startup-script（metadata），每次开机以 root 运行：在后台以普通用户启动 job_entry.sh。
# 任务完全在 VM 内后台运行，不依赖 SSH 或本地电脑。
U=GQF
su - "$U" -c 'gcloud storage cp gs://ssrl-runs-gqf-2026/code/job_entry.sh ~/job_entry.sh && setsid nohup bash ~/job_entry.sh >> ~/job.log 2>&1 < /dev/null &'
