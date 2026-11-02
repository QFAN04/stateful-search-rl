# GPU 运行环境（D1 确定后填写，之后不随意变更）

| 项 | 值 |
|---|---|
| 基础镜像 | 待定：veRL 0.7.x 官方镜像（D1 固定到具体 tag / digest） |
| CUDA / torch / vLLM | 待定（以镜像内 `pip list` 为准，导出到 env/pip-freeze-gpu.txt） |
| 上游 SHA | 见 third_party/README.md |
| 机器 | GCP A100 80GB（Spot 优先）/ Colab A100、G4 / TC2 A40（候补） |

规则：镜像或依赖变更必须记录在本文件，并在实验卡中注明。
