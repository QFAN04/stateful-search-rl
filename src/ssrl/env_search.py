"""我们自己的 HotpotQA agent flow（在 recipe 的 HotpotQAAgentFlow 之上，不改上游代码）。

当前只加了一个参数：检索 top-k（上游 `search_tool.py` 写死为 5）。D3 将在这里实现 C1–C3 的上下文策略。
通过 flow 配置 yaml 注册，例如 `configs/flows/hotpotqa_k3.yaml`：
  - name: ssrl_hotpotqa
    _target_: ssrl.env_search.SSRLHotpotQAFlow
    top_k: 3
    （其余键与 recipes/hotpotqa/base.yaml 相同）
"""
from __future__ import annotations

import logging
from typing import Any

from recipes.hotpotqa.env.search_tool import HotpotQASearchToolLegacy
from recipes.hotpotqa.hotpotqa_agent_flow import HotpotQAAgentFlow

logger = logging.getLogger(__name__)


class TopKSearchTool(HotpotQASearchToolLegacy):
    """与上游 batch_execute 相同，只把写死的 top-5 换成 self.top_k。"""

    top_k: int = 5

    def batch_execute(self, args_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not args_list:
            return []
        try:
            queries = [str(x["query"]) for x in args_list]
            embeddings = self._encode_queries(queries)
            assert self._index is not None
            _, ids = self._index.search(embeddings, self.top_k)
            return [{"content": self._format_results(ids[i]), "success": True} for i in range(len(ids))]
        except Exception as e:  # noqa: BLE001  与上游一致：失败时逐行返回错误，而不是让整条轨迹崩溃
            logger.warning("TopKSearchTool.batch_execute failed (%s queries): %s", len(args_list), e)
            return [{"content": str(e), "success": False} for _ in args_list]


class SSRLHotpotQAFlow(HotpotQAAgentFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 复用父类已加载好的模型和索引，只替换检索方法
        self.search_tool.__class__ = TopKSearchTool
        self.search_tool.top_k = int(kwargs.get("top_k", 5))
