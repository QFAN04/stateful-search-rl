"""RQ2：C2 笔记质量（训练前后），实验卡 docs/experiment_cards/d10_rq2_note_quality.md。

步骤：
  extract  从 E2 的 dev 输出、训练 rollouts、D9 test 输出里抽取每次 note 调用 + 引用段落，做自动指标
  judge    --judge gemini|deepseek，判 faithful / relevant（可断点续跑，结果追加写入 judged_<judge>.jsonl）
           Gemini 项目 2026-11-08 被 Google 限制（630/751 已判）；DeepSeek 全部 751 条重判、与 Gemini 交叉对比
  report   汇总 + Wilson 区间，写 summary.json
用法（仓库根目录）：python -I scripts/rq2_note_quality.py extract|judge|report [--out runs/d10_rq2_note_quality_20261108]
judge 需要 gcloud 用户登录（gcloud auth print-access-token），项目 llm-rl-lab-2026。
"""
import argparse
import glob
import json
import math
import pathlib
import random
import re
import string
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
E2 = ROOT / "runs" / "e2_c2_q3-4b-2507_lora_pro6000_20261106"
D9 = ROOT / "runs" / "d9_test_c2_q3-4b-2507_lora_pro6000_20261108"
HF = ROOT / "data" / "raw" / "hotpot_qa_hf"
MODEL = "gemini-3.8-flash"
URL = f"https://aiplatform.googleapis.com/v1/projects/llm-rl-lab-2026/locations/global/publishers/google/models/{MODEL}:generateContent"
SEED, CAP = 20261108, 400
TOOL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def norm(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def query_of(text):
    return text.split("### User Query")[1].split("###")[0].strip() if "### User Query" in text else ""


def passages(prompt):
    """该步 prompt 的 Retrieved Passages：{编号: 去掉 (query: ...) 前缀后的段落文本}。"""
    if "### Retrieved Passages" not in prompt:
        return {}
    block = prompt.split("### Retrieved Passages")[1].split("\n###")[0]
    out = {}
    for m in re.finditer(r"^\[(\d+)\]\s*(?:\(query:[^)]*\)\s*)?(.*?)(?=^\[\d+\]|\Z)", block, re.S | re.M):
        out[int(m.group(1))] = m.group(2).strip()
    return out


def gold_map():
    import pandas as pd
    m = {}
    for f in sorted(HF.glob("distractor_*.parquet")):
        df = pd.read_parquet(f, columns=["question", "answer", "supporting_facts"])
        for q, a, sf in zip(df.question, df.answer, df.supporting_facts):
            m[q.strip()] = (a, sorted(set(str(t) for t in sf["title"])))
    return m


def notes_in(traj, group, gold):
    q = query_of(traj["steps"][0]["input"] if traj.get("steps") else traj["input"])
    ans, sf = gold.get(q, (None, []))
    out = []
    for si, st in enumerate(traj.get("steps") or []):
        ps = passages(st["input"])
        for m in TOOL.finditer(st.get("output", "")):
            try:
                call = json.loads(m.group(1))
            except Exception:
                continue
            if call.get("name") != "note":
                continue
            args = call.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            fact = str(args.get("fact", "")).strip() if isinstance(args, dict) else ""
            src_m = re.search(r"\d+", str(args.get("source", ""))) if isinstance(args, dict) else None
            src = int(src_m.group()) if src_m else None
            psg = ps.get(src) if src is not None else None
            rec = {"group": group, "question": q, "answer": ans, "step_index": si, "fact": fact, "source": src,
                   "passage": psg, "valid": bool(fact and psg), "traj_score": float(traj["score"])}
            if rec["valid"]:
                rec["cites_supporting_fact"] = any(psg.startswith(t) for t in sf) if sf else None
                rec["fact_contains_answer"] = bool(ans) and norm(ans) in norm(fact) and norm(ans) not in ("yes", "no")
            out.append(rec)
    return out


def load_jsonl(paths):
    for p in paths:
        for line in open(p, encoding="utf-8"):
            yield json.loads(line)


def extract(out):
    gold = gold_map()
    sets = {"dev_step0": [E2 / "val" / "0.jsonl"], "dev_step150": [E2 / "val" / "150.jsonl"],
            "train_1-10": [E2 / "rollouts" / f"{i}.jsonl" for i in range(1, 11)],
            "train_141-150": [E2 / "rollouts" / f"{i}.jsonl" for i in range(141, 151)],
            "d9_test": glob.glob(str(D9 / "val" / "*.jsonl"))}
    counts = {}
    for s in (20, 40, 60, 80, 100, 120, 140):
        n = sum(len(notes_in(t, "x", gold)) for t in load_jsonl([E2 / "val" / f"{s}.jsonl"]))
        counts[f"dev_step{s}"] = n
    allrecs = []
    for g, paths in sets.items():
        trajs = list(load_jsonl(paths))
        recs = [r for t in trajs for r in notes_in(t, g, gold)]
        counts[g] = len(recs)
        counts[g + "_trajs"] = len(trajs)
        allrecs += recs
    unmatched = sum(1 for r in allrecs if r["answer"] is None)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "notes.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(allrecs):
            f.write(json.dumps({"id": i, **r}, ensure_ascii=False) + "\n")
    json.dump({"counts": counts, "gold_unmatched": unmatched}, open(out / "extract_meta.json", "w"), indent=1)
    print(json.dumps(counts, indent=1), "gold_unmatched", unmatched)


PROMPT = """You are auditing short evidence notes written by a search agent while answering a multi-hop question.

Question: {q}
Gold answer: {a}
Cited passage: {p}
Note written by the agent: {n}

Judge the note only against the cited passage.
- faithful: "yes" if every claim in the note is stated in or directly entailed by the cited passage; "partial" if some claims are supported but at least one is not; "no" if the main claim is unsupported or contradicts the passage.
- relevant: "yes" if the note records information that is needed (as a bridge entity, an attribute to compare, or the answer itself) to answer the question; otherwise "no".
Return JSON: {{"faithful": "yes|partial|no", "relevant": "yes|no", "reason": "<one short sentence>"}}"""

_tok = {"t": None, "at": 0}
_lock = threading.Lock()


def token():
    with _lock:
        if time.time() - _tok["at"] > 1800:
            _tok["t"] = subprocess.run("gcloud auth print-access-token", shell=True, capture_output=True, text=True).stdout.strip()
            _tok["at"] = time.time()
        return _tok["t"]


def call(rec):
    body = {"contents": [{"role": "user", "parts": [{"text": PROMPT.format(q=rec["question"], a=rec["answer"], p=rec["passage"][:2000], n=rec["fact"])}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "thinkingConfig": {"thinkingLevel": "MEDIUM"}}}
    for attempt in range(5):
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Authorization": f"Bearer {token()}", "Content-Type": "application/json"})
            d = json.loads(urllib.request.urlopen(req, timeout=120).read())
            txt = d["candidates"][0]["content"]["parts"][-1]["text"]
            j = json.loads(txt)
            return {"id": rec["id"], "faithful": j.get("faithful"), "relevant": j.get("relevant"), "reason": j.get("reason"), "model": MODEL}
        except Exception as e:  # noqa: BLE001
            err = str(e)
            if "401" in err:  # gcloud 返回的缓存 token 过期：强制下次重新取
                with _lock:
                    _tok["at"] = 0
            time.sleep(2 ** attempt)
    return {"id": rec["id"], "error": err}


DS_MODEL = "deepseek-flash"


def call_deepseek(rec):
    """同一 prompt，DeepSeek（OpenAI 兼容接口，JSON 模式）。2026-11-08 Gemini 项目被限制后全部 751 条重判、与 Gemini 交叉对比。"""
    import os
    body = {"model": DS_MODEL, "temperature": 0, "reasoning_effort": "high", "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": PROMPT.format(q=rec["question"], a=rec["answer"], p=rec["passage"][:2000], n=rec["fact"])}]}
    err = ""
    for attempt in range(5):
        try:
            req = urllib.request.Request("https://api.deepseek.com/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}", "Content-Type": "application/json"})
            d = json.loads(urllib.request.urlopen(req, timeout=180).read())
            j = json.loads(d["choices"][0]["message"]["content"])
            return {"id": rec["id"], "faithful": j.get("faithful"), "relevant": j.get("relevant"), "reason": j.get("reason"),
                    "model": d.get("model", DS_MODEL), "usage": d.get("usage", {}).get("total_tokens")}
        except Exception as e:  # noqa: BLE001
            err = str(e)
            if "403" in err or "402" in err:  # 被拒或余额不足：不重试
                break
            time.sleep(2 ** attempt)
    return {"id": rec["id"], "error": err}


JUDGES = {"gemini": call, "deepseek": call_deepseek}


def judge(out, which="gemini"):
    recs = [json.loads(line) for line in open(out / "notes.jsonl", encoding="utf-8")]
    rng = random.Random(SEED)
    todo = []
    for g in sorted({r["group"] for r in recs}):
        v = [r for r in recs if r["group"] == g and r["valid"] and r["answer"] is not None]
        todo += rng.sample(v, CAP) if len(v) > CAP else v
    rng.shuffle(todo)  # 训练前后打乱，judge 看不到组别
    done = set()
    jf = out / f"judged_{which}.jsonl"
    if jf.exists():
        done = {json.loads(line)["id"] for line in open(jf, encoding="utf-8") if "error" not in line}
    todo = [r for r in todo if r["id"] not in done]
    print("to judge:", len(todo))
    with open(jf, "a", encoding="utf-8") as f, ThreadPoolExecutor(8) as ex:
        for i, res in enumerate(ex.map(JUDGES[which], todo)):
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0:
                print(i, res.get("faithful"), res.get("error", "")[:80], flush=True)


def wilson(k, n):
    if n == 0:
        return (None, None, None)
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(p, 3), round(c - h, 3), round(c + h, 3))


def load_judged(path):
    jud = {}
    if path.exists():
        for line in open(path, encoding="utf-8"):
            j = json.loads(line)
            if "error" not in j:
                jud[j["id"]] = j
    return jud


def kappa(pairs):
    """Cohen's kappa 与原始一致率。"""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(a == b for a, b in pairs) / n
    labels = {x for p in pairs for x in p}
    pe = sum((sum(a == lab for a, _ in pairs) / n) * (sum(b == lab for _, b in pairs) / n) for lab in labels)
    return {"n": n, "agreement": round(po, 3), "kappa": round((po - pe) / (1 - pe), 3) if pe < 1 else None}


def report(out):
    recs = {r["id"]: r for r in map(json.loads, open(out / "notes.jsonl", encoding="utf-8"))}
    meta = json.load(open(out / "extract_meta.json"))
    J = {w: load_judged(out / f"judged_{w}.jsonl") for w in JUDGES}
    both = sorted(set(J["gemini"]) & set(J["deepseek"]))
    summ = {"extract": meta, "agreement_gemini_vs_deepseek": {
        k: kappa([(J["gemini"][i][k], J["deepseek"][i][k]) for i in both]) for k in ("faithful", "relevant")}, "groups": {}}
    for w in JUDGES:
        summ["groups"][w] = _groups(recs, J[w])
    json.dump(summ, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summ, indent=1, ensure_ascii=False))


def _groups(recs, jud):
    res = {}
    for g in ["dev_step0", "dev_step150", "train_1-10", "train_141-150", "d9_test"]:
        allg = [r for r in recs.values() if r["group"] == g]
        val = [r for r in allg if r["valid"]]
        jg = [jud[r["id"]] for r in val if r["id"] in jud]
        sfk = [r for r in val if r["cites_supporting_fact"] is not None]
        res[g] = {
            "notes": len(allg), "valid": wilson(len(val), len(allg)),
            "cites_supporting_fact": wilson(sum(r["cites_supporting_fact"] for r in sfk), len(sfk)),
            "fact_contains_answer": wilson(sum(r["fact_contains_answer"] for r in val), len(val)),
            "judged": len(jg),
            "faithful_yes": wilson(sum(j["faithful"] == "yes" for j in jg), len(jg)),
            "faithful_no": wilson(sum(j["faithful"] == "no" for j in jg), len(jg)),
            "relevant_yes": wilson(sum(j["relevant"] == "yes" for j in jg), len(jg)),
            "traj_em_with_notes": round(sum(r["traj_score"] for r in allg) / max(len(allg), 1), 3),
        }
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "judge", "report"])
    ap.add_argument("--judge", choices=list(JUDGES), default="deepseek")
    ap.add_argument("--out", default=str(ROOT / "runs" / "d10_rq2_note_quality_20261108"))
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    if a.stage == "judge":
        judge(out, a.judge)
    else:
        {"extract": extract, "report": report}[a.stage](out)
