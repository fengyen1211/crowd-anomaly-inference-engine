"""
scripts/evaluate_against_labels.py
=====================================
拿人工標註（scripts/make_label_sheet.py 產生的頁面下載的 JSON）對照 sweep
結果，算驗證判斷與警報等級的準確率。

對照方式：
- 驗證判斷：標註「有發生」應判 support；「沒有發生」判 contradict 或
  uncertain 都算對（兩者都不會發出警報）；「看不出來」不計入。
- 警報等級：標註的通報程度 none／attention／danger 對應 info／warning／
  critical；低於標註算漏報、高於標註算誤報。

驗證判斷只記在 log（VisualVerifier 那一行），所以要一起給 sweep 的 .out。

使用方式（在 project/ 目錄下執行）：
    python scripts/evaluate_against_labels.py data/labels_video7.json \
        reports/sweep_566003 logs/model-sweep_566003.out
"""

import argparse
import json
import re
from pathlib import Path

_ARMS = ["baseline", "vlm_7b", "llm_27b", "both_mid"]
_LEVEL_RANK = {"info": 0, "warning": 1, "critical": 2}
_DANGER_TO_LEVEL = {"none": "info", "attention": "warning", "danger": "critical"}
_VERDICT_RE = re.compile(r"VisualVerifier record_id=(\S+) verdict=(\w+)")


def main() -> None:
    parser = argparse.ArgumentParser(description="用人工標註評估 sweep 結果")
    parser.add_argument("labels", help="人工標註 JSON")
    parser.add_argument("sweep_dir", help="reports/sweep_<jobid>")
    parser.add_argument("log", nargs="?", help="logs/model-sweep_<jobid>.out（沒有就只評估警報等級）")
    args = parser.parse_args()

    labels = {x["record_id"]: x for x in json.loads(Path(args.labels).read_text(encoding="utf-8"))["labels"]}

    # log 依組別順序執行，每組每筆事件一行驗證結果，依序配對
    verdicts = _VERDICT_RE.findall(Path(args.log).read_text(encoding="utf-8", errors="replace")) if args.log else []
    vi = 0

    print(f"{'組別':<10}{'驗證判對':>8}{'等級正確':>8}{'漏報':>6}{'誤報':>6}   錯誤明細")
    for arm in _ARMS:
        path = Path(args.sweep_dir) / f"{arm}.json"
        if not path.exists():
            continue
        results = json.loads(path.read_text(encoding="utf-8"))["results"]
        v_ok = v_n = lvl_ok = miss = false_alarm = n = 0
        notes = []
        for r in results:
            rid = r["record_id"]
            verdict = None
            if vi < len(verdicts) and verdicts[vi][0] == rid:
                verdict = verdicts[vi][1]
                vi += 1
            label = labels.get(rid)
            if r["status"] != "ok" or not label:
                continue
            short = rid.replace("tinyformer7_", "")

            occurred = label.get("occurred")
            if verdict and occurred in ("yes", "no"):
                v_n += 1
                correct = (verdict == "support") == (occurred == "yes")
                v_ok += correct
                if not correct:
                    notes.append(f"{short} 判 {verdict}（實際{'有' if occurred == 'yes' else '沒有'}）")

            truth = _DANGER_TO_LEVEL.get(label.get("danger"))
            if truth:
                n += 1
                got = r["output"]["alert_level"]
                diff = _LEVEL_RANK[got] - _LEVEL_RANK[truth]
                lvl_ok += diff == 0
                miss += diff < 0
                false_alarm += diff > 0
                if diff:
                    notes.append(f"{short} {got}（應為 {truth}）")

        v_text = f"{v_ok}/{v_n}" if v_n else "-"
        print(f"{arm:<10}{v_text:>8}{f'{lvl_ok}/{n}':>8}{miss:>6}{false_alarm:>6}   {'；'.join(notes)}")


if __name__ == "__main__":
    main()
