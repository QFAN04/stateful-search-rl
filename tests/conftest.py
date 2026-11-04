import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
# recipes.hotpotqa.prompts 只是字符串常量，本地无 torch 也能 import
for p in (ROOT / "src", ROOT / "third_party" / "Agent-R1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
