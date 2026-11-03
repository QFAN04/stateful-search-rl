#!/bin/bash
# 轮询机的 startup-script：从自身 metadata 取 poller-script，安装 cron（每 3 分钟）。
curl -sf -H 'Metadata-Flavor: Google' \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/poller-script" > /usr/local/bin/ssrl-poller.sh
chmod +x /usr/local/bin/ssrl-poller.sh
echo '*/3 * * * * root /usr/local/bin/ssrl-poller.sh' > /etc/cron.d/ssrl-poller
chmod 644 /etc/cron.d/ssrl-poller
