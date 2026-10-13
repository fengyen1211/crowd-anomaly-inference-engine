"""
scripts/compare_model_reports.py
===================================
彙整 scripts/nano4/run_model_sweep.sbatch 產生的多份批次報告，
輸出一份 Markdown 對照表，方便直接貼進進度報告。

比較內容：
- 總覽：每一組用的 VLM / LLM、成功筆數、總耗時、平均每筆耗時
- 逐筆事件：各組判定的 alert_level 與耗時
- 逐筆事件：各組產生的 summary 全文（品質要靠人工比對）

使用方式（在 project/ 目錄下執行）：
    python scripts/compare_model_reports.py reports/sweep_123456
    python scripts/compare_model_reports.py reports/sweep_123456 --output sweep_123456.md
"""

import argparse
import json
from pathlib import Path


def load_reports(report_dir: Path) -> dict:
    """讀取目錄下所有 .json 報告，key 是組名（檔名去掉副檔名）。"""
    reports = {}
    for path in sorted(report_dir.glob("*.json")):
        reports[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return reports


def build_markdown(reports: dict) -> str:
    arms = list(reports)
    lines = ["# 大小模型對照實驗結果", ""]

    # ---------- 總覽 ----------
    lines += [
        "## 總覽",
        "",
        "| 組別 | VLM | LLM | 成功筆數 | 總耗時 (s) | 平均每筆 (s) |",
        "|---|---|---|---|---|---|",
    ]
    for arm in arms:
        report = reports[arm]
        models = report.get("models", {})
        ok = [r for r in report["results"] if r["status"] == "ok"]
        avg = sum(r["elapsed_sec"] for r in ok) / len(ok) if ok else 0
        lines.append(
            f"| {arm} | {models.get('vlm', '?')} | {models.get('llm', '?')} "
            f"| {len(ok)}/{report['total_records']} | {report['total_elapsed_sec']} | {avg:.1f} |"
        )

    # 逐筆比較以第一份報告的事件順序為準
    record_ids = [r["record_id"] for r in reports[arms[0]]["results"]]
    by_arm = {arm: {r["record_id"]: r for r in reports[arm]["results"]} for arm in arms}

    # ---------- 警報等級與耗時 ----------
    lines += [
        "",
        "## 逐筆事件：警報等級／耗時",
        "",
        "| 事件 | " + " | ".join(arms) + " |",
        "|---|" + "---|" * len(arms),
    ]
    for record_id in record_ids:
        cells = []
        for arm in arms:
            result = by_arm[arm].get(record_id)
            if result is None:
                cells.append("—")
            elif result["status"] != "ok":
                cells.append(f"失敗（{result.get('error_type', '?')}）")
            else:
                cells.append(f"{result['output']['alert_level']} / {result['elapsed_sec']}s")
        lines.append(f"| {record_id} | " + " | ".join(cells) + " |")

    # ---------- 摘要全文 ----------
    lines += ["", "## 逐筆事件：摘要內容", ""]
    for record_id in record_ids:
        lines += [f"### {record_id}", ""]
        for arm in arms:
            result = by_arm[arm].get(record_id)
            if result is None or result["status"] != "ok":
                text = "（無結果）"
            else:
                text = result["output"]["summary"]
            lines += [f"**{arm}**：{text}", ""]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="彙整大小模型對照實驗的批次報告")
    parser.add_argument("report_dir", help="run_model_sweep.sbatch 的輸出目錄（reports/sweep_<jobid>）")
    parser.add_argument("--output", help="Markdown 輸出路徑，預設印到主控台")
    args = parser.parse_args()

    reports = load_reports(Path(args.report_dir))
    if not reports:
        raise SystemExit(f"{args.report_dir} 底下沒有任何 .json 報告")

    markdown = build_markdown(reports)
    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
        print(f"對照表已寫入：{args.output}")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
