"""我们自己的 HotpotQA agent flow（在 recipe 的 HotpotQAAgentFlow 之上，不改上游代码）。

- `SSRLHotpotQAFlow`：recipe 原版流程，只把检索 top-k 做成参数（上游 `search_tool.py` 写死为 5）。D2 环境 probe 用。
- `SSRLSearchFlow`：E0–E3 用，`context_policy: c0|c1|c2|c3`，渲染逻辑在 `ssrl.context_policies`。
通过 flow 配置 yaml 注册，见 `configs/flows/ssrl_c{0,1,2,3}.yaml`（其余键与 recipes/hotpotqa/base.yaml 相同）。
"""
from __future__ import annotations

import logging
from typing import Any

from uuid import uuid4

from agent_r1.agent_flow.agent_flow import AgentFlowOutput, AgentFlowStep
from recipes.hotpotqa.env.search_tool import HotpotQASearchToolLegacy, parse_legacy_tool_result
from recipes.hotpotqa.hotpotqa_agent_flow import (
    _RETRIEVAL_TOOL_NAMES,
    HotpotQAAgentFlow,
    _decode_tool_arguments,
    _recover_tool_calls_from_text,
)
from recipes.hotpotqa.prompts import HOTPOTQA_SYSTEM_PROMPT
from verl.utils.profiler import simple_timer

from ssrl.note_bonus import load_sf_titles, note_bonus
from ssrl.context_policies import (
    NOTES_PER_STEP,
    PolicyConfig,
    TrajState,
    parse_note,
    render,
    visible_passage_ids,
    tool_schemas,
)

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
    """D2 环境配置 probe 用：recipe 原版流程，只改 top-k。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 复用父类已加载好的模型和索引，只替换检索方法
        self.search_tool.__class__ = TopKSearchTool
        self.search_tool.top_k = int(kwargs.get("top_k", 5))


class SSRLSearchFlow(SSRLHotpotQAFlow):
    """E0–E3 用的 flow：循环结构与 recipe `HotpotQAAgentFlow.run` 相同，只把状态换成 `TrajState`，
    把每步 prompt 的渲染换成 `context_policies.render`（C0 与 recipe 逐字一致，见 tests/test_context_policies.py）。

    与 recipe 的差别只有 C2：多一个 `note` 工具；一步里检索最多 max_parallel_calls 个、笔记最多 NOTES_PER_STEP 个，
    分开计数（C0/C1/C3 沿用 recipe 的 `tool_calls[:max_parallel_calls]`）。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_cfg = PolicyConfig(
            policy=str(kwargs.get("context_policy", "c0")),
            window_k=int(kwargs.get("window_k", 1)),
            c3_max_sentences=int(kwargs.get("c3_max_sentences", 6)),
        )
        self.tool_schemas = tool_schemas(self.policy_cfg.policy)
        # E4：笔记过程奖励（λ=0 时完全不生效，E0–E3 / D9 行为不变）
        self.note_bonus_lambda = float(kwargs.get("note_bonus_lambda", 0.0) or 0.0)
        self.sf_titles: dict[str, list[str]] = {}
        if self.note_bonus_lambda > 0:
            assert self.policy_cfg.policy == "c2", "笔记奖励只用于 C2"
            path = kwargs.get("supporting_facts_path")
            assert path, "note_bonus_lambda > 0 需要 supporting_facts_path"
            self.sf_titles = load_sf_titles(str(path))

    # ---- 检索：与 recipe _do_search / _do_search_batch 相同的容错，但记录成"一轮" ----
    def _search_round(self, st: TrajState, queries: list[str]) -> None:
        texts: list[list[str]] = [[] for _ in queries]
        try:
            results = self.search_tool.batch_execute([{"query": q} for q in queries])
            for i, item in enumerate(results[: len(queries)]):
                if item.get("success", False):
                    texts[i] = [p.text for p in parse_legacy_tool_result(str(item.get("content", "")))]
        except Exception as e:  # noqa: BLE001  与 recipe 一致：检索失败只记 query，不中断轨迹
            logger.warning("[ssrl_search] search failed for %d queries: %s", len(queries), e)
        st.add_round(queries, texts)

    async def _prompt_ids(self, st: TrajState, fb_text: str, num_step: int) -> tuple[list[int], int]:
        """与 recipe _prompt_ids_within_budget 相同：只收缩文档段，绝不截掉开头的工具定义。
        同时返回最终使用的文档字符预算，用来确定本步真正展示了哪些文档（C2 笔记只能引用这些）。"""
        max_chars, min_chars = self.prompt_length * 3, 400
        while True:
            messages = [
                {"role": "system", "content": HOTPOTQA_SYSTEM_PROMPT},
                {"role": "user", "content": render(self.policy_cfg, st, fb_text, max_chars)},
            ]
            ids = await self.apply_chat_template(messages, tools=self.tool_schemas)
            if len(ids) <= self.prompt_length or max_chars <= min_chars:
                break
            max_chars = max(min_chars, int(max_chars * 0.72))
        if len(ids) > self.prompt_length:
            logger.warning("[ssrl_search][step=%d] prompt still too long (%d > %d)", num_step, len(ids), self.prompt_length)
        return ids[: self.prompt_length], max_chars

    def _extra(self, st: TrajState, anchor_obs: str, n_prompt: int, acc: float = 0.0, bonus: float = 0.0) -> dict[str, Any]:
        info = {
            "num_tool_steps": len(st.history_actions),
            "acc": acc,
            "prompt_tokens": n_prompt,
            "num_rounds": len(st.round_passages),
            "num_notes": len(st.notes),
        }
        if self.note_bonus_lambda > 0:  # 同一个 run 里所有步都有这个键（Agent-R1 按键拼成数组，键必须一致）
            info["note_bonus"] = bonus
        return {"anchor_obs": anchor_obs, "ssrl_state": st.snapshot(), "reward_extra_info": info}

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        question = list(kwargs["raw_prompt"])[0]["content"]
        cfg = self.policy_cfg
        metrics: dict[str, Any] = {}
        steps: list[AgentFlowStep] = []
        st = TrajState(question=question)
        if self.force_first_search:
            self._search_round(st, [question])
        fb_lines: list[str] = []

        num_steps = 0
        while num_steps < self.max_steps:
            num_steps += 1
            fb_text = "\n".join(fb_lines[-3:]) if (fb_lines and self.enable_tool_parse_feedback) else ""
            anchor_obs = render(cfg, st, fb_text, self.prompt_length * 3)
            prompt_ids, used_chars = await self._prompt_ids(st, fb_text, num_steps)
            allowed_sources = {i + 1 for i in visible_passage_ids(cfg, st, used_chars)}

            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=uuid4().hex, prompt_ids=prompt_ids, sampling_params=sampling_params
                )
            response_ids = output.token_ids[: self.response_length]
            logprobs = output.log_probs[: self.response_length] if output.log_probs else None
            _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            if not tool_calls:
                recovered = _recover_tool_calls_from_text(response_text)
                if recovered:
                    tool_calls = recovered
                elif self.enable_tool_parse_feedback and "<tool_call>" in response_text and "</tool_call>" in response_text:
                    fb_lines.append(
                        "Previous assistant message contained <tool_call>...</tool_call> but the JSON "
                        "could not be parsed. Use strict JSON with double-quoted keys and string values, "
                        "comma between fields: name must be search and arguments must include query."
                    )

            if not tool_calls:  # 终局步：reward_score=None → EM
                step = AgentFlowStep(
                    prompt_ids=prompt_ids, response_ids=response_ids, response_logprobs=logprobs,
                    reward_score=None, extra_fields=self._extra(st, anchor_obs, len(prompt_ids)),
                )
                step = await self._postprocess(step, **kwargs)
                acc = step.extra_fields.get("reward_extra_info", {}).get("acc", 0.0)
                bonus = 0.0
                if self.note_bonus_lambda > 0:  # E4：EM 之外的笔记过程奖励；acc 仍记 EM，dev 选 checkpoint 只看 acc
                    bonus = note_bonus(acc, len(st.round_passages), [n.source for n in st.notes],
                                       [t for _, t in st.passages], self.sf_titles.get(question.strip(), []))
                    step.reward_score = float(step.reward_score) + self.note_bonus_lambda * bonus
                step.extra_fields["reward_extra_info"] = self._extra(st, anchor_obs, len(prompt_ids), acc, bonus)["reward_extra_info"]
                steps.append(step)
                break

            if cfg.policy == "c2":
                search_calls = [tc for tc in tool_calls if tc.name in _RETRIEVAL_TOOL_NAMES][: self.max_parallel_calls]
                note_calls = [tc for tc in tool_calls if tc.name == "note"][:NOTES_PER_STEP]
            else:
                search_calls, note_calls = tool_calls[: self.max_parallel_calls], []

            queries: list[str] = []
            for tc in search_calls:
                if tc.name not in _RETRIEVAL_TOOL_NAMES:
                    continue
                args = _decode_tool_arguments(tc.arguments)
                if args and args.get("query"):
                    queries.append(str(args["query"]))

            n_notes = 0
            for tc in note_calls:  # 笔记引用的是本步 prompt 里展示的文档，所以在检索之前处理
                note, err = parse_note(tc.arguments, allowed_sources)
                if note is None:
                    fb_lines.append(f"Previous note was rejected: {err}.")
                else:
                    st.notes.append(note)
                    n_notes += 1

            if queries:
                with simple_timer("tool_calls", metrics):
                    self._search_round(st, queries)
            elif n_notes == 0 and self.enable_tool_parse_feedback:
                fb_lines.append(
                    "Previous tool call was recognized but arguments were invalid. "
                    "For search, arguments must be a JSON object with a string field query."
                )

            step = AgentFlowStep(
                prompt_ids=prompt_ids, response_ids=response_ids, response_logprobs=logprobs,
                reward_score=0.0, extra_fields=self._extra(st, anchor_obs, len(prompt_ids)),
            )
            step = await self._postprocess(step, **kwargs)
            steps.append(step)

        return AgentFlowOutput(steps=steps, metrics=metrics)
