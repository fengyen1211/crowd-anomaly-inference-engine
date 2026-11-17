"""
scripts/evaluate_against_labels.py
=====================================
拿人工標註（scripts/make_label_sheet.py 產生的頁面下載的 JSON）對照 sweep
結果，算驗證判斷與警報等級的準確率。

對照方式：
- 驗證判斷：依 --target 選標註的哪一題——box（峰值時框內的人，檢驗影像端
  定位）或 scene（整個畫面，檢驗該不該發警報，預設）。第一版標註只有一題
  occurred（問的是框內），兩種 target 都退回用它。標註「有發生」應判 support；「沒有發生」判 contradict 或
  uncertain 都算對（兩者都不會發出警報）；「看不出來」不計入。
- 通報：標註 attention 或 danger 應該通報（warning、critical 都算有通報），
  none 不該通報；應通報卻是 info 算漏報，不該通報卻發出算誤報。
- 等級：只看「應該通報、也有通報」的事件，danger 應為 critical、attention
  應為 warning；danger 只拿到 warning 算低估，attention 拿到 critical 算高估。
  舊版把 danger 直接對應 critical、低於標註就算漏報，但警報規則
  （services/alert_service.py）只有 panic 類與場景模型的爆炸／縱火／槍擊會發
  critical，標成 danger 的 fight 拿到 warning 其實有通報，卻被算成漏報。
- 通知：有通報的事件中，沒有併入同一支攝影機較早警報（merged_into，見
  services/alert_throttle.py）、實際另外通知的則數。

驗證判斷只記在 log（VisualVerifier 那一行），所以要一起給 sweep 的 .out。

使用方式（在 project/ 目錄下執行）：
    python scripts/evaluate_against_labels.py data/labels_video7.json \
        reports/sweep_566003 logs/model-sweep_566003.out
    python scripts/evaluate_against_labels.py data/labels_ucf2_gt.json reports/sweep_793568 --details
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
    parser.add_argument("--target", choices=["box", "scene"], default="scene", help="驗證判斷對照框內或整體畫面")
    parser.add_argument("--details", action="store_true", help="逐筆列出判錯、漏報、誤報與等級不符的事件")
    args = parser.parse_args()

    labels = {x["record_id"]: x for x in json.loads(Path(args.labels).read_text(encoding="utf-8"))["labels"]}

    # log 依組別順序執行，每組每筆事件一行驗證結果，依序配對
    verdicts = _VERDICT_RE.findall(Path(args.log).read_text(encoding="utf-8", errors="replace")) if args.log else []
    vi = 0

    print(f"{'組別':<10}{'驗證判對':>8}{'漏報':>8}{'誤報':>8}{'等級一致':>8}{'低估':>6}{'高估':>6}{'通知':>10}")
    for arm in _ARMS:
        path = Path(args.sweep_dir) / f"{arm}.json"
        if not path.exists():
            continue
        results = json.loads(path.read_text(encoding="utf-8"))["results"]
        v_ok = v_n = 0
        should = shouldnt = miss = false_alarm = 0
        both = lvl_ok = under = over = 0
        alerted = notified = 0
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

            occurred = label.get(f"occurred_{args.target}", label.get("occurred"))
            if verdict and occurred in ("yes", "no"):
                v_n += 1
                correct = (verdict == "support") == (occurred == "yes")
                v_ok += correct
                if not correct:
                    notes.append(f"{short} 判 {verdict}（實際{'有' if occurred == 'yes' else '沒有'}）")

            got = r["output"]["alert_level"]
            if got != "info":
                alerted += 1
                notified += not r["output"].get("merged_into")

            truth = _DANGER_TO_LEVEL.get(label.get("danger"))
            if truth == "info":
                shouldnt += 1
                if got != "info":
                    false_alarm += 1
                    notes.append(f"{short} 誤報 {got}")
            elif truth:
                should += 1
                if got == "info":
                    miss += 1
                    notes.append(f"{short} 漏報（應為 {truth}）")
                    continue
                both += 1
                diff = _LEVEL_RANK[got] - _LEVEL_RANK[truth]
                lvl_ok += diff == 0
                under += diff < 0
                over += diff > 0
                if diff:
                    notes.append(f"{short} {'低估' if diff < 0 else '高估'} {got}（應為 {truth}）")

        v_text = f"{v_ok}/{v_n}" if v_n else "-"
        print(
            f"{arm:<10}{v_text:>8}{f'{miss}/{should}':>8}{f'{false_alarm}/{shouldnt}':>8}"
            f"{f'{lvl_ok}/{both}':>8}{under:>6}{over:>6}{f'{notified}/{alerted}':>10}"
        )
        if args.details:
            for note in notes:
                print(f"    {note}")


if __name__ == "__main__":
    main()
