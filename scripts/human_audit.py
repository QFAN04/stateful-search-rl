"""RQ2 笔记质量的人工复核：导出待标注表 → 人工填写 → 计算与 DeepSeek / Gemini / Claude 的一致率。

用法（仓库根目录）：
  python -I scripts/human_audit.py export   # 生成 runs/d10_rq2_note_quality_20261108/human_audit.csv（与 Claude 盲审同一批 30 条）
  （用 Excel / WPS 打开，只填 faithful、relevant 两列，保存为 CSV UTF-8）
  python -I scripts/human_audit.py score    # 读回填好的表，打印一致率和每条分歧

标注标准（与 judge prompt 相同）：
  faithful：笔记里每个说法都能在「引用段落」里找到或直接推出 → yes；部分能、至少一处不能 → partial；主要说法找不到或矛盾 → no
  relevant：笔记记下了答题需要的信息（桥接实体、要比较的属性、或答案本身）→ yes；否则 → no
导出的表故意不含任何 judge / Claude 的判定，填完再看 score 的结果。
"""
import csv
import json
import pathlib
import sys

OUT = pathlib.Path(__file__).resolve().parents[1] / "runs" / "d10_rq2_note_quality_20261108"
CSV = OUT / "human_audit.csv"


def export():
    recs = {r["id"]: r for r in map(json.loads, open(OUT / "notes.jsonl", encoding="utf-8"))}
    ids = json.load(open(OUT / "audit_ids.json"))
    with open(CSV, "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig：Excel 打开中文 / 特殊字符不乱码
        w = csv.writer(f)
        w.writerow(["id", "question", "gold_answer", "note", "cited_passage", "faithful (yes/partial/no)", "relevant (yes/no)", "comment"])
        for i in ids:
            r = recs[i]
            w.writerow([i, r["question"], r["answer"], r["fact"], r["passage"], "", "", ""])
    print("wrote", CSV, f"({len(ids)} rows)")


def score():
    rows = list(csv.reader(open(CSV, encoding="utf-8-sig")))[1:]
    human = {int(r[0]): (r[5].strip().lower(), r[6].strip().lower()) for r in rows if r and r[5].strip() and r[6].strip()}
    if not human:
        sys.exit("还没有填写 faithful / relevant 列")
    others = {"deepseek": {}, "gemini": {}}
    for w in others:
        for line in open(OUT / f"judged_{w}.jsonl", encoding="utf-8"):
            j = json.loads(line)
            if "error" not in j:
                others[w][j["id"]] = (j["faithful"], j["relevant"])
    others["claude"] = {a["id"]: (a["faithful"], a["relevant"]) for a in json.load(open(OUT / "audit_claude.json", encoding="utf-8"))}
    print(f"已标注 {len(human)} 条")
    for w, lab in others.items():
        ids = [i for i in human if i in lab]
        for k, name in ((0, "faithful"), (1, "relevant")):
            agree = sum(human[i][k] == lab[i][k] for i in ids)
            print(f"  人工 vs {w:<8} {name:<8} 一致 {agree}/{len(ids)} = {agree / len(ids):.2f}")
    print("\n分歧明细（人工 / deepseek / gemini / claude）：")
    for i, (f_, r_) in sorted(human.items()):
        labs = [others[w].get(i, ("-", "-")) for w in ("deepseek", "gemini", "claude")]
        if any(l != (f_, r_) for l in labs):
            print(f"  #{i}: faithful {f_} / {' / '.join(l[0] for l in labs)}   relevant {r_} / {' / '.join(l[1] for l in labs)}")
    json.dump({str(k): v for k, v in human.items()}, open(OUT / "audit_human.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    {"export": export, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "export"]()
