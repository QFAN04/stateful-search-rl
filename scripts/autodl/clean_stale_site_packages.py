"""删除 site-packages 里不属于任何已安装包的 .py / .so 残留文件（AutoDL 私有镜像专用，可重复执行）。

原因（2026-11-05 在 RTX PRO 6000 实例上发现）：环境是在基础镜像（torch 2.8、numpy 2.x 等）上升级/降级安装的，
在原实例里旧文件已被 pip 删除，但保存成私有镜像后，这些删除没有被保留，旧文件又出现了。
例如 torch/_inductor/kernel/flex_attention.py（2.8 的旧位置）与 2.9 的 kernel/flex/flex_attention.py
重复注册同一个模板，导致 `import vllm` 报 "duplicate template name"。

判断依据：文件位于某个已安装包的顶层目录下，但不在任何 *.dist-info/RECORD 中。只处理 .py / .so，
不碰 bin 目录（RECORD 里以 ../ 开头的条目）。
用法：python scripts/autodl/clean_stale_site_packages.py [--dry-run]
"""
import collections
import csv
import glob
import os
import site
import sys


def main() -> None:
    dry = "--dry-run" in sys.argv
    sp = site.getsitepackages()[0]
    os.chdir(sp)
    recorded: set[str] = set()
    for rec in glob.glob("*.dist-info/RECORD"):
        with open(rec, encoding="utf-8", errors="replace") as f:
            for row in csv.reader(f):
                if row and not row[0].startswith(".."):
                    recorded.add(os.path.normpath(row[0]))
    tops = {p.split(os.sep)[0] for p in recorded}
    stale = []
    for top in sorted(tops):
        if not os.path.isdir(top) or top.endswith(".dist-info"):
            continue
        for root, _, files in os.walk(top):
            if "__pycache__" in root.split(os.sep):
                continue
            for name in files:
                p = os.path.normpath(os.path.join(root, name))
                if name.endswith((".py", ".so")) and p not in recorded:
                    stale.append(p)
    print(f"site-packages={sp} stale={len(stale)}",
          dict(collections.Counter(p.split(os.sep)[0] for p in stale).most_common()))
    if not dry:
        for p in stale:
            os.remove(p)
        print("removed")


if __name__ == "__main__":
    main()
